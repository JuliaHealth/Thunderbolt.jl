# emRKC against the production reaction-diffusion split, on the first EP tutorial's monodomain, host
# and device. What it costs to buy a simulated millisecond with an explicit stabilized scheme, next
# to `LieTrotterGodunov(BackwardEulerSolver + KrylovJL_CG, ForwardEulerCellSolver)`.
#
# MATCHED ACCURACY, NOT MATCHED STEP SIZE. Comparing the two at one step size would measure nothing:
# the step size each method can afford is a property of the method, and that is the whole claim under
# test. So each picks its own, and they are compared where they deliver the same error.
#
#  * Reference: each method against ITSELF at `DTREF`, in Float64 on the host. Deliberately not a
#    shared reference -- emRKC lumps the mass matrix and the backward Euler stage does not, so the two
#    integrate different semidiscretizations of the same monodomain problem. That gap is printed
#    below; it is of the order of the accuracy band itself, and a shared reference would lay it under
#    both error curves as a floor that no step size could get beneath. Each reference's own error is
#    estimated by its distance to the same method at `2 DTREF`, printed beside it. The backward Euler
#    reference tightens its conjugate gradient well past the tutorial's setting, because otherwise the
#    linear solve's own error -- which accumulates with the step count -- is what the reference
#    measures. The timed arms keep the tutorial's tolerances, so that floor shows up where it belongs:
#    in the error curve of the configuration a user would actually run.
#  * Step size: for each method, the largest in `SWEEP` whose final-time φₘ relative error stays
#    inside `BAND`. The sweep runs in the precision and on the hardware the host arm is timed in, so
#    the calibration *is* the host arm's validation.
#  * Validation: every timed arm is checked against the reference at its own step size before it is
#    timed -- the device arms repeat the check in Float32 on the device. A silently wrong kernel must
#    not produce a headline number.
#  * Timing: minimum over `NPASS` passes of `NSTEPS` steps. Device arms first step for
#    `WARMUP_SECONDS` without interruption, because this card idles at 300 MHz and a cold arm measures
#    the clock ramp instead of the kernel; the clocks are read back per arm and printed, so the reader
#    can see the card was hot. Timing starts from a developed spiral wave, not from the initial
#    condition.
#  * Composition: for the splitting arms the per step time is split over the two children through the
#    same calls `OrdinaryDiffEqOperatorSplitting`'s own `_perform_step!` makes, which separates the
#    linear solve from the pointwise reaction. emRKC has no linear solve to separate out; what is
#    reported in its place is the stage structure `(s, m)` that stands in for one.
#
# This is a benchmark, not a CI gate: the numbers are reported as measured, whichever way they fall.
#
# CUDA is a weak dependency, so this runs in the GPU test environment rather than the package one:
# `julia --project=test/gpu benchmarks/benchmark-emrkc.jl`.

using Thunderbolt
using CUDA
using LinearSolve
using OrdinaryDiffEqOperatorSplitting
using LinearAlgebra
using Printf

import Thunderbolt: SciMLBase, ThreadedSparseMatrixCSR
import OrdinaryDiffEqOperatorSplitting:
    advance_solution_by!, forward_sync_subintegrator!, backward_sync_subintegrator!

const N              = 512          # 263169 dofs, the size at which the solve dominates
const TEND           = 25.0         # ms, the first EP tutorial's own visualization window
const DTREF          = 0.025        # ms, reference step size
const BAND           = 1.0e-2       # final-time φₘ relative error a step size has to stay inside
const SWEEP          = (0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2)
const NSTEPS         = 25
const NPASS          = 3
const WARMUP_SECONDS = 1.5
const CuCSR          = CUDA.CUSPARSE.CuSparseMatrixCSR{Float32, Int32}

####################################
## Problem
####################################

function ep01_form(::Type{T}) where {T}
    mesh = generate_mesh(Quadrilateral, (N, N), Vec{2}((0.0, 0.0)), Vec{2}((2.5, 2.5)))
    model = MonodomainModel(
        ConstantCoefficient(1.0),
        ConstantCoefficient(1.0),
        ConstantCoefficient(SymmetricTensor{2, 2, Float64}((4.5e-5, 0.0, 2.0e-5))),
        NoStimulationProtocol(),
        Thunderbolt.ParametrizedFHNModel{T}(),
        CartesianCoordinateSystem(mesh),
        :φₘ, :s,
    )
    return semidiscretize(
        ReactionDiffusionSplit(model),
        FiniteElementDiscretization(Dict(:φₘ => LagrangeCollection{1}())),
        mesh,
    )
end

function ep01_u0(form, ::Type{T}) where {T}
    u₀ = create_initial_condition(form, T)
    setvariable!(u₀, form, :φₘ) do x
        (x[1] ≤ 1.25 && x[2] ≤ 1.25) ? one(T) : zero(T)
    end
    setvariable!(u₀, form, :s) do x
        x[2] ≥ 1.25 ? T(0.1) : zero(T)
    end
    return u₀
end

# The two algorithms under test, as a downstream user would spell them. The linear solver tolerances
# are the first EP tutorial's.
emrkc(::Type{VT}, ::Type{MT}) where {VT, MT} =
    EMRKC(solution_vector_type = VT, system_matrix_type = MT)

function splitting(::Type{VT}, ::Type{MT}; atol = 1.0e-6, rtol = 1.0e-5) where {VT, MT}
    T = eltype(VT)
    return LieTrotterGodunov((
        BackwardEulerSolver(
            solution_vector_type = VT,
            system_matrix_type   = MT,
            inner_solver         = KrylovJL_CG(atol = T(atol), rtol = T(rtol)),
        ),
        ForwardEulerCellSolver(solution_vector_type = VT),
    ))
end

# `init` takes the initial condition as the integrator's own state, so every arm gets a copy: the
# host and device initial conditions are shared across arms and would otherwise be consumed by the
# first one to run.
build(form, u0, alg, Δt, tend) =
    init(OperatorSplittingProblem(form, copy(u0), (zero(Δt), tend)), alg; dt = Δt, verbose = false)

function solve_to_end(form, u0, alg, Δt)
    integrator = build(form, u0, alg, Δt, oftype(Δt, TEND))
    solve!(integrator)
    return integrator
end

sync(::Vector) = nothing
sync(::CuVector) = CUDA.synchronize()

relerr(a, b) = norm(Vector(a) .- Vector(b)) / norm(Vector(b))

####################################
## Reference and step size selection
####################################

"""
The Float64 host reference of one method, and the estimate of its own error: the distance to the same
method at twice the step size, which for a first order scheme bounds the reference's error up to the
factor two between them.
"""
function reference(form, u0, alg, φₘ)
    coarse = getvariable(solve_to_end(form, u0, alg, 2DTREF).u, φₘ)
    fine   = getvariable(solve_to_end(form, u0, alg, DTREF).u, φₘ)
    return copy(fine), relerr(coarse, fine)
end

"""
The largest step size in `SWEEP` whose final state stays inside `BAND` of `φ_ref`, together with the
error it lands at. Errors are printed for the whole sweep: where the band is crossed is as much of
the result as which step size wins.
"""
function select_dt(form, u0, alg, φₘ, φ_ref, label)
    best = nothing
    for Δt in SWEEP
        # In the solution vector's precision: `(t, Δt)` reach the reaction kernel, and a Float64 step
        # size against Float32 storage runs the hot loop in double precision.
        integrator = solve_to_end(form, u0, alg, eltype(u0)(Δt))
        φ = getvariable(integrator.u, φₘ)
        ok = integrator.sol.retcode == SciMLBase.ReturnCode.Success && all(isfinite, φ)
        err = ok ? relerr(φ, φ_ref) : NaN
        @printf("    %-8s Δt = %5.2f   rel err = %-10.4g %s\n", label, Δt, err,
                ok ? (err ≤ BAND ? "in band" : "") : "FAILED")
        ok && err ≤ BAND && (best = (Δt, err))
    end
    best === nothing && error("$label: no step size in SWEEP reaches the $BAND band.")
    return best
end

####################################
## Timing
####################################

function gpu_clocks()
    out = read(
        `nvidia-smi --query-gpu=clocks.sm,clocks.mem --format=csv,noheader,nounits`, String,
    )
    sm, mem = parse.(Int, strip.(split(first(split(strip(out), '\n')), ',')))
    return sm, mem
end

"""
Step without interruption for `WARMUP_SECONDS` and read the clocks back *while still stepping*. On a
card that idles at 300 MHz a short warmup measures the ramp rather than the kernel; the returned
clocks are what says this one did not. They are sampled mid-flight and not after the loop, because
the card starts dropping back within tens of milliseconds of going idle -- which is less than one
`nvidia-smi` query takes.
"""
function prewarm!(integrator, on_device)
    t0 = time_ns()
    clocks = (0, 0)
    sampled = false
    while (time_ns() - t0) / 1.0e9 < WARMUP_SECONDS
        step!(integrator)
        if on_device && !sampled && (time_ns() - t0) / 1.0e9 > WARMUP_SECONDS / 2
            clocks = gpu_clocks()
            sampled = true
        end
    end
    sync(integrator.u)
    return clocks
end

"Minimum seconds per step over `NPASS` passes of `NSTEPS` steps."
function measure!(integrator)
    best = Inf
    for _ = 1:NPASS
        t0 = time_ns()
        for _ = 1:NSTEPS
            step!(integrator)
        end
        sync(integrator.u)
        best = min(best, (time_ns() - t0) / 1.0e9 / NSTEPS)
    end
    return best
end

"""
Seconds per step spent in each child of a splitting step, through the same three calls
`OrdinaryDiffEqOperatorSplitting`'s `_perform_step!` makes for it. Replicating the loop rather than
instrumenting it keeps this honest about what the parent step actually does; the integrator is
advanced by real steps and is left usable.
"""
function child_composition!(integrator, Δt)
    times = zeros(length(integrator.child_subintegrators))
    for _ = 1:NSTEPS
        for (i, child) in enumerate(integrator.child_subintegrators)
            idxs   = integrator.child_solution_indices[i]
            syncer = integrator.child_synchronizers[i]
            forward_sync_subintegrator!(integrator, child, idxs, syncer)
            t0 = time_ns()
            advance_solution_by!(integrator, child, Δt)
            sync(integrator.u)
            times[i] += (time_ns() - t0) / 1.0e9
            backward_sync_subintegrator!(integrator, child, idxs, syncer)
        end
    end
    return times ./ NSTEPS
end

####################################
## Arms
####################################

struct Arm
    label::String
    method::String
    Δt::Float64
    seconds_per_step::Float64
    validation::Float64
    solve_fraction::Float64   # share of the step in the linear solve; NaN where there is none
    stages::String
    clocks::Tuple{Int, Int}
end

function run_arm(label, method, form, u0, alg, Δt, φₘ, φ_ref, on_device, has_solve)
    # Validate before timing: the trajectory this arm produces, at the step size it was given.
    checked = solve_to_end(form, u0, alg, Δt)
    φ = getvariable(Vector(checked.u), φₘ)
    err = relerr(φ, φ_ref)
    stages = if has_solve
        ""
    else
        s, _, m = Thunderbolt._emrkc_step_sizing(checked.alg, Δt, checked.cache.ρS, checked.cache.ρF)
        "s=$s m=$m"
    end

    # Time from a developed spiral wave rather than from the initial condition: `tend` here only has
    # to outlast the warmup and the passes.
    timed = build(form, copy(u0), alg, Δt, oftype(Δt, 1.0e5))
    clocks = prewarm!(timed, on_device)
    seconds = measure!(timed)

    fraction = NaN
    if has_solve
        parts = child_composition!(timed, Δt)
        fraction = parts[1] / sum(parts)
    end
    arm = Arm(label, method, Δt, seconds, err, fraction, stages, clocks)
    GC.gc()
    on_device && CUDA.reclaim()
    return arm
end

function report(arms)
    println("\n", "="^118)
    @printf("%-22s %-7s %9s %9s %11s %10s %9s %9s\n",
            "arm", "Δt/ms", "s/step", "steps/s", "s / sim ms", "lin solve", "stages", "clocks")
    println("-"^118)
    for a in arms
        @printf("%-22s %-7.2f %9.5f %9.1f %11.5f %10s %9s %9s\n",
                a.label, a.Δt, a.seconds_per_step, 1 / a.seconds_per_step,
                a.seconds_per_step / a.Δt,
                isnan(a.solve_fraction) ? "none" : @sprintf("%.0f%%", 100a.solve_fraction),
                isempty(a.stages) ? "-" : a.stages,
                a.clocks == (0, 0) ? "host" : @sprintf("%d/%d", a.clocks[1], a.clocks[2]))
    end
    println("-"^118)
    for (host, device) in (("host emRKC", "device emRKC"), ("host splitting", "device splitting"))
        h = findfirst(a -> a.label == host, arms)
        d = findfirst(a -> a.label == device, arms)
        h === nothing || d === nothing || @printf(
            "  %-18s host -> device: %.2fx per simulated ms\n", arms[h].method,
            (arms[h].seconds_per_step / arms[h].Δt) / (arms[d].seconds_per_step / arms[d].Δt),
        )
    end
    for (a, b) in (("host emRKC", "host splitting"), ("device emRKC", "device splitting"))
        i = findfirst(x -> x.label == a, arms)
        j = findfirst(x -> x.label == b, arms)
        i === nothing || j === nothing || @printf(
            "  %-18s emRKC vs splitting: %.2fx per simulated ms\n",
            split(a)[1], (arms[j].seconds_per_step / arms[j].Δt) / (arms[i].seconds_per_step / arms[i].Δt),
        )
    end
    println("\nvalidation (final-time φₘ relative error against each method's own reference):")
    for a in arms
        @printf("  %-22s %.6g %s\n", a.label, a.validation, a.validation ≤ 2BAND ? "" : "  <-- OUT OF BAND")
    end
end

####################################

function main()
    CUDA.functional() || error("This benchmark needs a functional CUDA device.")

    form64 = ep01_form(Float64)
    u64    = ep01_u0(form64, Float64)
    form32 = ep01_form(Float32)
    u32    = ep01_u0(form32, Float32)
    φₘ64   = solution_variable(form64, :φₘ)
    φₘ32   = solution_variable(form32, :φₘ)

    println("ep01 monodomain, ", N, " x ", N, ", ", Thunderbolt.solution_size(form64),
            " states, t ∈ [0, ", TEND, "] ms")
    @printf("reference Δt = %.4g ms, accuracy band = %.3g\n", DTREF, BAND)

    cpu64_emrkc = emrkc(Vector{Float64}, ThreadedSparseMatrixCSR{Float64, Int64})
    # The reference's conjugate gradient runs far tighter than the tutorial's. At the tutorial's
    # tolerances the solve's own error accumulates with the step count, so the backward Euler error
    # curve turns around below Δt ≈ 0.2 ms and a reference stepped there is *less* accurate than the
    # step sizes it is meant to certify. That floor belongs in the arms, which are configured the way
    # a user would configure them; it does not belong in the reference.
    cpu64_split = splitting(
        Vector{Float64}, ThreadedSparseMatrixCSR{Float64, Int64}; atol = 1.0e-12, rtol = 1.0e-10,
    )
    φ_ref_emrkc, err_emrkc = reference(form64, u64, cpu64_emrkc, φₘ64)
    φ_ref_split, err_split = reference(form64, u64, cpu64_split, φₘ64)
    @printf("  emRKC reference error ≈ %.3g ; splitting reference error ≈ %.3g\n", err_emrkc, err_split)
    @printf("  the two references differ by %.4g -- the mass lumping, not a step size\n",
            relerr(φ_ref_emrkc, φ_ref_split))

    println("\nstep size selection (Float32, host):")
    cpu32_emrkc = emrkc(Vector{Float32}, ThreadedSparseMatrixCSR{Float32, Int32})
    cpu32_split = splitting(Vector{Float32}, ThreadedSparseMatrixCSR{Float32, Int32})
    Δt_emrkc, _ = select_dt(form32, u32, cpu32_emrkc, φₘ32, φ_ref_emrkc, "emRKC")
    Δt_split, _ = select_dt(form32, u32, cpu32_split, φₘ32, φ_ref_split, "splitting")

    gpu_emrkc = emrkc(CuVector{Float32}, CuCSR)
    gpu_split = splitting(CuVector{Float32}, CuCSR)
    ugpu = CuVector(u32)

    arms = Arm[]
    push!(arms, run_arm("host emRKC", "emRKC", form32, u32, cpu32_emrkc, Float32(Δt_emrkc), φₘ32, φ_ref_emrkc, false, false))
    push!(arms, run_arm("host splitting", "splitting", form32, u32, cpu32_split, Float32(Δt_split), φₘ32, φ_ref_split, false, true))
    push!(arms, run_arm("device emRKC", "emRKC", form32, ugpu, gpu_emrkc, Float32(Δt_emrkc), φₘ32, φ_ref_emrkc, true, false))
    push!(arms, run_arm("device splitting", "splitting", form32, ugpu, gpu_split, Float32(Δt_split), φₘ32, φ_ref_split, true, true))

    report(arms)
    return nothing
end

main()
