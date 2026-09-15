# emRKC against the production reaction-diffusion split, on an ep01-shaped monodomain, host and
# device, for two cell models. What it costs to buy a simulated millisecond with an explicit
# stabilized scheme, next to `LieTrotterGodunov(BackwardEulerSolver + KrylovJL_CG,
# ForwardEulerCellSolver)`.
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
#    timed -- the device arms repeat the check in their own precision (Float32, or Float64 where noted
#    below) on the device. A silently wrong kernel must not produce a headline number.
#  * Timing: minimum over `NPASS` passes of `NSTEPS` steps. Device arms first step for
#    `WARMUP_SECONDS` without interruption, because this card idles at 300 MHz and a cold arm measures
#    the clock ramp instead of the kernel; the clocks are read back per arm and printed, so the reader
#    can see the card was hot. Timing starts from the initial condition having had `WARMUP_SECONDS`
#    worth of real steps to develop -- for FHN that is a spiral wave, for PCG2019 see below.
#  * Composition: for the splitting arms the per step time is split over the two children through the
#    same calls `OrdinaryDiffEqOperatorSplitting`'s own `_perform_step!` makes, which separates the
#    linear solve from the pointwise reaction. emRKC has no linear solve to separate out; what is
#    reported in its place is the stage structure `(s, m)` that stands in for one.
#  * Knife edge: roughly half of the reported speedup is the step size each method lands on, and that
#    half sits right at the accuracy band's edge, not in its middle. In one measured FHN run (2.38x
#    host, 2.68x device), the splitting arm's step size missed the band by 5%; run one step smaller
#    instead, the gap closes to 1.76x (host) / 1.40x (device). The other half is per step and is not a
#    knife edge: emRKC takes no linear solve, which is the bulk of a splitting step (54-76% above).
#
# MODELS: `ENV["EMRKC_MODELS"]`, comma-separated, defaults to `"FHN,PCG2019"` -- both run in one
# invocation (one Julia startup, one CUDA warmup) back to back. `N`, `TEND`, `DTREF`, `BAND`, `SWEEP`,
# `NSTEPS`, `NPASS`, `WARMUP_SECONDS` are shared, unchanged discipline across both models; each gets
# its own mesh extent, monodomain coefficients and initial condition, set out in `MODEL_CONFIGS`
# below. A model whose arms error out is caught and reported, and does not stop the other model.
#  * FHN: 2.5mm x 2.5mm, the ep01 tutorial's own dimensionless diffusion tensor, an excite/refractory
#    box initial condition that develops into a sustained spiral.
#  * PCG2019: 10mm x 10mm, κ/(Cₘχ) = 0.4 mm²/ms -- the physiological value already validated for this
#    model in `test/integration/test_emrkc_electrophysiology.jl` -- and that same test's planar S1
#    front (left 12% of the domain excited to 20mV, the rest at the model's own resting default). That
#    test reports the front taking ~15-20ms to cross this 10mm domain (mesh-resolution independent, so
#    unchanged at this benchmark's N=512), so at TEND=25ms, and through most of the timed passes'
#    warmup, the tissue is genuinely mixed -- upstroke, plateau and still-resting side by side -- which
#    is what both the accuracy check and the per-step cost need to see; once it has crossed the domain
#    settles into a shared plateau (a real AP's plateau lasts ~200-300ms, past this benchmark's own
#    horizon either way). `EMRKC(gates = :all)` is spelled out explicitly: PCG2019's six gates
#    (`h,m,f,s,xs,xr`) are exactly what the exponential treatment is for, and `gates = ()` would
#    silently run past them.
#
# THREADING: host arms run 2 threads -- the capped profile this file is meant to be run under is
# `systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 -p CPUWeight=25 env
# JULIA_NUM_THREADS=2 julia -t2 --heap-size-hint=3G --project=test/gpu benchmarks/benchmark-emrkc.jl`.
# `ThreadedSparseMatrixCSR`'s SpMV is threaded across those 2 host threads; the pointwise reaction/gate
# sweeps are not threaded (single-threaded on host, one CUDA thread per dof on device). Uncapped/-t8
# arms are not measured here. This applies to both models.
#
# PCG2019 DEVICE PRECISION: `ParametrizedPCG2019Model{T}` is genuinely `T`-parametrized --
# `ParametrizedPCG2019Model{Float32}()` does carry real `Float32` fields, the same as
# `ParametrizedFHNModel{Float32}`. Its gate kernel does not stay `Float32`, though:
# `_pcg2019_sigmoid` and the `h`-gate's `τ_h` (`src/modeling/cells/pcg2019.jl`) hardcode bare
# `1.0`/`2.0`/`-1.0` literals, so every sigmoid and every `gate_coefficients` call computes internally
# in `Float64` regardless of the struct's `T`, truncating back to `Float32` only at the return
# boundary -- confirmed by `@code_typed`, which shows `fpext`-to-`Float64` throughout
# `cell_rhs_fast!`, `cell_rhs_slow!` and `gate_coefficients`. On host this costs a few wasted cycles;
# on a GPU whose `Float32` throughput is the point, it would mean a "Float32" device arm silently
# running its hottest loop in `Float64`. The fix is mechanical (typed constants in place of the bare
# literals) but lives in `src/modeling/cells/pcg2019.jl`, out of scope for this benchmark-file-only
# change -- so PCG2019's device arms run in genuine `Float64`
# (`CuVector{Float64}`/`CuSparseMatrixCSR{Float64,Int32}`) rather than a `Float32` number that silently
# wasn't one. FHN's device arms are unaffected (`ParametrizedFHNModel`'s own literals are already
# `T`-wrapped) and stay `Float32`.
#
# EXPECTATION (PCG2019, stated before measuring): its reaction is expensive relative to FHN's -- 6
# exponential gates and stiff kinetics (`τ_m = 0.12ms`, λ ≈ -8.3/ms) against FHN's one linear gate --
# which dilutes the relative weight of "no linear solve", emRKC's whole advantage over the split. Both
# arms' step sizes are expected reaction-accuracy-limited to ~0.2-0.4ms per prior 0D single-cell
# probes, similar in scale to each other, which further narrows what the step size itself buys (the
# "knife edge" bullet above). FHN's measured 2.38x host / 2.68x device ratios are a ceiling here, not
# an expectation -- PCG2019 is measured below, not assumed.
#
# MEASURED OUTCOME vs. that expectation: the step sizes landed where expected (0.2ms/0.1ms, inside the
# 0.2-0.4ms band), but the ratio itself did not come in lower -- 3.46x host is level with FHN's 3.38x,
# and 8.45x device is well above FHN's 3.04x. The reaction dilution is real (its own share of the step
# drops from FHN's 24%/host to PCG2019's 13%/host), but PCG2019's physiological κ = 0.4mm²/ms makes the
# backward Euler system far worse conditioned than FHN's dimensionless one, so the linear solve gets
# *more* expensive in absolute terms, not less -- splitting's host step is 11x FHN's, not merely
# reaction-heavier. On device that same linear solve runs in Float64 (see above) on hardware built for
# Float32, which costs the CG-heavy splitting arm far more than the SpMV-heavy emRKC one: hence the
# device ratio exceeding the host one, the opposite of FHN's own pattern (3.04x device < 3.38x host).
#
# This is a benchmark, not a CI gate: the numbers are reported as measured, whichever way they fall.
#
# CUDA is a weak dependency, so this runs in the GPU test environment rather than the package one:
# `julia --project=test/gpu benchmarks/benchmark-emrkc.jl`.
#
# Requires a memory-capped cgroup (enforced by `_assert_memory_capped()` below; `BENCHMARK_UNCAPPED=1`
# overrides it) -- an uncapped run's GC sizes itself against the whole machine, not the 8G it is meant
# to run in. Canonical invocation:
# `systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 env JULIA_NUM_THREADS=2 julia -t2
# --heap-size-hint=3G --project=test/gpu benchmarks/benchmark-emrkc.jl`.
#
# RESULTS (most recent run, capped 2-thread host profile, RTX 2080; N=512, 512x512 mesh both models):
#
# == FHN (526338 states, device Float32) ==
# arm                    Δt/ms      s/step   steps/s  s / sim ms  lin solve  stages    clocks
# host emRKC             0.40      0.02187      45.7     0.05466       none  s=1 m=2      host
# host splitting         0.20      0.03694      27.1     0.18472        76%       -      host
# device emRKC           0.40      0.00098    1023.6     0.00244       none  s=1 m=2 1905/6800
# device splitting       0.20      0.00149     672.5     0.00744        76%       - 1905/6800
# host->device: emRKC 22.38x, splitting 24.84x  |  emRKC vs splitting: 3.38x host, 3.04x device
# validation (rel err, band=0.01): host emRKC 0.00968, host splitting 0.00562,
#   device emRKC 0.00968, device splitting 0.00562 -- all in band
#
# == PCG2019 (1842183 states, device Float64 -- see PCG2019 DEVICE PRECISION above) ==
# arm                    Δt/ms      s/step   steps/s  s / sim ms  lin solve  stages     clocks
# host emRKC             0.20      0.23364       4.3     1.16818       none  s=1 m=23      host
# host splitting         0.10      0.40404       2.5     4.04045        87%        -      host
# device emRKC           0.20      0.00732     136.7     0.03659       none  s=1 m=23 1905/6800
# device splitting       0.10      0.03091      32.3     0.30913        95%        - 1890/6800
# host->device: emRKC 31.93x, splitting 13.07x  |  emRKC vs splitting: 3.46x host, 8.45x device
# validation (rel err, band=0.01): host emRKC 0.00157, host splitting 0.00337,
#   device emRKC 0.00157, device splitting 0.00296 -- all in band
# splitting sweep: Δt=0.05 err=0.00105, Δt=0.10 err=0.00337 (selected), Δt≥0.20 NaN (blew up --
#   the stiff gate's FE substep limit, as expected; the selected Δt is safely below it)
# emRKC sweep: Δt=0.05/0.10/0.20 in band (err 0.00099/0.00139/0.00157), Δt=0.40 err=0.020 (out);
#   0.20ms selected, inside the 0.2-0.4ms window the 0D probes anticipated

using Thunderbolt
using CUDA
using LinearSolve
using OrdinaryDiffEqOperatorSplitting
using LinearAlgebra
using Printf

import Thunderbolt: SciMLBase, ThreadedSparseMatrixCSR
import OrdinaryDiffEqOperatorSplitting:
    advance_solution_by!, forward_sync_subintegrator!, backward_sync_subintegrator!

const N              = 512          # dofs/node-count shared across models, the size at which the
                                     # solve dominates -- independent of the physical domain size L
const TEND           = 25.0         # ms, the first EP tutorial's own visualization window
const DTREF          = 0.025        # ms, reference step size
const BAND           = 1.0e-2       # final-time φₘ relative error a step size has to stay inside
const SWEEP          = (0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2)
const NSTEPS         = 25
const NPASS          = 3
const WARMUP_SECONDS = 1.5
const CuCSR          = CUDA.CUSPARSE.CuSparseMatrixCSR{Float32, Int32}
const CuCSR64        = CUDA.CUSPARSE.CuSparseMatrixCSR{Float64, Int32}

####################################
## Problem
####################################

"""
One model's mesh extent, monodomain coefficients, ionic model constructor and initial condition.
`device_eltype` is `Float32` unless the model's own kernel is known to leak to `Float64` regardless
(see the PCG2019 header note above), in which case the device arms run in genuine `Float64`.
"""
struct ModelConfig
    name::String
    ion::Function                    # (::Type{T}) -> ionic model instance
    Cₘ::Float64
    χ::Float64
    κ::SymmetricTensor{2, 2, Float64}
    L::Float64                       # domain side
    u0!::Function                    # (u₀, form, ::Type{T}) -> u₀
    device_eltype::DataType
end

function fhn_u0!(u₀, form, ::Type{T}) where {T}
    setvariable!(u₀, form, :φₘ) do x
        (x[1] ≤ 1.25 && x[2] ≤ 1.25) ? one(T) : zero(T)
    end
    setvariable!(u₀, form, :s) do x
        x[2] ≥ 1.25 ? T(0.1) : zero(T)
    end
    return u₀
end

const PCG2019_L = 10.0

"""
A planar S1 activation front: the left 12% of the domain driven to a real excited potential, the rest
at the model's own resting default (`create_initial_condition` already wrote it) -- the same protocol
`test/integration/test_emrkc_electrophysiology.jl` validates for this model, scaled to this
benchmark's domain. A first attempt used a cross-field excite/refractory-block IC (bottom-left
quadrant excited, top half's `h` gate forced to 0) to chase a sustained spiral the way FHN's own IC
does; that discontinuous `h` jump kept emRKC's self-referenced reference error (`DTREF` vs `2DTREF`)
above `BAND` regardless of step size -- not a step-size-selection failure but an unconverged
reference, an artifact of the synthetic discontinuity rather than a property of emRKC or PCG2019
worth reporting. The planar front has no such discontinuity and converges cleanly.
"""
function pcg2019_u0!(u₀, form, ::Type{T}) where {T}
    setvariable!(u₀, form, :φₘ) do x
        x[1] ≤ T(0.12PCG2019_L) ? T(20.0) : T(-85.0)
    end
    return u₀
end

const FHN_CONFIG = ModelConfig(
    "FHN", T -> Thunderbolt.ParametrizedFHNModel{T}(), 1.0, 1.0,
    SymmetricTensor{2, 2, Float64}((4.5e-5, 0.0, 2.0e-5)), 2.5, fhn_u0!, Float32,
)
const PCG2019_CONFIG = ModelConfig(
    "PCG2019", T -> Thunderbolt.ParametrizedPCG2019Model{T}(), 1.0, 1.0,
    SymmetricTensor{2, 2, Float64}((0.4, 0.0, 0.4)), PCG2019_L, pcg2019_u0!, Float64,
)
const AVAILABLE_MODELS = Dict("FHN" => FHN_CONFIG, "PCG2019" => PCG2019_CONFIG)
const MODEL_CONFIGS    = [AVAILABLE_MODELS[m] for m in split(get(ENV, "EMRKC_MODELS", "FHN,PCG2019"), ",")]

function ep01_form(::Type{T}, cfg::ModelConfig) where {T}
    mesh = generate_mesh(Quadrilateral, (N, N), Vec{2}((0.0, 0.0)), Vec{2}((cfg.L, cfg.L)))
    model = MonodomainModel(
        ConstantCoefficient(cfg.Cₘ),
        ConstantCoefficient(cfg.χ),
        ConstantCoefficient(cfg.κ),
        NoStimulationProtocol(),
        cfg.ion(T),
        CartesianCoordinateSystem(mesh),
        :φₘ, :s,
    )
    return semidiscretize(
        ReactionDiffusionSplit(model),
        FiniteElementDiscretization(Dict(:φₘ => LagrangeCollection{1}())),
        mesh,
    )
end

function ep01_u0(form, ::Type{T}, cfg::ModelConfig) where {T}
    u₀ = create_initial_condition(form, T)
    cfg.u0!(u₀, form, T)
    return u₀
end

# The two algorithms under test, as a downstream user would spell them. The linear solver tolerances
# are the first EP tutorial's. `gates = :all` is EMRKC's own default; spelled out here so the choice is
# visible in the file rather than implicit (see the PCG2019 header note).
emrkc(::Type{VT}, ::Type{MT}) where {VT, MT} =
    EMRKC(solution_vector_type = VT, system_matrix_type = MT, gates = :all)

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

    # Time from `WARMUP_SECONDS` of real steps past the initial condition, not from the initial
    # condition itself: `tend` here only has to outlast the warmup and the passes.
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
## Machine discipline
####################################

"""
The effective `memory.max` (bytes) of this process's own cgroup v2 leaf, read by walking
`/proc/self/cgroup`'s `0::<path>` up through `/sys/fs/cgroup<path>` until a `memory.max` file is
found. `nothing` for "max" (unset) or if no such file is found at all.
"""
function _cgroup_memory_limit()
    lines = try
        readlines("/proc/self/cgroup")
    catch
        return nothing
    end
    idx = findfirst(l -> startswith(l, "0::"), lines)
    idx === nothing && return nothing
    dir = "/sys/fs/cgroup" * split(lines[idx], "0::")[2]
    while true
        f = joinpath(dir, "memory.max")
        if isfile(f)
            v = strip(read(f, String))
            return v == "max" ? nothing : parse(Int, v)
        end
        parent = dirname(dir)
        parent == dir && return nothing
        dir = parent
    end
end

"""
Refuses to run outside a memory-capped cgroup: an uncapped run's GC sizes its heap against the whole
machine rather than the 8G this benchmark is meant to run in, which has frozen this box before.
`BENCHMARK_UNCAPPED=1` overrides for a deliberate uncapped run.
"""
function _assert_memory_capped()
    get(ENV, "BENCHMARK_UNCAPPED", "0") == "1" && return nothing
    limit = _cgroup_memory_limit()
    capped = limit !== nothing && limit ≤ 12 * 1024^3
    capped || error(
        "No memory-capped cgroup detected (effective memory.max = $(limit === nothing ? "unset" : limit) " *
        "bytes). An uncapped run's GC sizes itself against the whole machine. Run:\n" *
        "  systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 env JULIA_NUM_THREADS=2 " *
        "julia -t2 --heap-size-hint=3G --project=test/gpu benchmarks/benchmark-emrkc.jl\n" *
        "or set BENCHMARK_UNCAPPED=1 to run uncapped deliberately.",
    )
    Base.JLOptions().heap_size_hint == 0 &&
        println("WARNING: no --heap-size-hint given -- GC growth is unbounded even inside the cgroup.")
    return nothing
end

####################################

function run_model(cfg::ModelConfig)
    println("\n", "#"^118)
    println("# ", cfg.name, "  (", N, " x ", N, ", L = ", cfg.L, ", device = ", cfg.device_eltype, ")")
    println("#"^118)

    form64 = ep01_form(Float64, cfg)
    u64    = ep01_u0(form64, Float64, cfg)
    φₘ64   = solution_variable(form64, :φₘ)

    println(cfg.name, ": ", Thunderbolt.solution_size(form64), " states, t ∈ [0, ", TEND, "] ms")
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
    form32 = ep01_form(Float32, cfg)
    u32    = ep01_u0(form32, Float32, cfg)
    φₘ32   = solution_variable(form32, :φₘ)
    cpu32_emrkc = emrkc(Vector{Float32}, ThreadedSparseMatrixCSR{Float32, Int32})
    cpu32_split = splitting(Vector{Float32}, ThreadedSparseMatrixCSR{Float32, Int32})
    Δt_emrkc, _ = select_dt(form32, u32, cpu32_emrkc, φₘ32, φ_ref_emrkc, "emRKC")
    Δt_split, _ = select_dt(form32, u32, cpu32_split, φₘ32, φ_ref_split, "splitting")

    # Device precision: Float32 for every model except where the header notes a kernel that leaks to
    # Float64 regardless of the struct's own T (PCG2019) -- there the device arm reuses the Float64
    # host reference form/IC instead, rather than reporting a Float32 number that was never really one.
    DT = cfg.device_eltype
    form_dev, u_dev, φₘ_dev = DT === Float32 ? (form32, u32, φₘ32) : (form64, u64, φₘ64)
    CuMatType = DT === Float32 ? CuCSR : CuCSR64
    gpu_emrkc = emrkc(CuVector{DT}, CuMatType)
    gpu_split = splitting(CuVector{DT}, CuMatType)
    ugpu = CuVector(u_dev)

    arms = Arm[]
    push!(arms, run_arm("host emRKC", "emRKC", form32, u32, cpu32_emrkc, Float32(Δt_emrkc), φₘ32, φ_ref_emrkc, false, false))
    push!(arms, run_arm("host splitting", "splitting", form32, u32, cpu32_split, Float32(Δt_split), φₘ32, φ_ref_split, false, true))
    push!(arms, run_arm("device emRKC", "emRKC", form_dev, ugpu, gpu_emrkc, DT(Δt_emrkc), φₘ_dev, φ_ref_emrkc, true, false))
    push!(arms, run_arm("device splitting", "splitting", form_dev, ugpu, gpu_split, DT(Δt_split), φₘ_dev, φ_ref_split, true, true))

    report(arms)
    return arms
end

function main()
    _assert_memory_capped()
    CUDA.functional() || error("This benchmark needs a functional CUDA device.")
    println("models: ", join(getfield.(MODEL_CONFIGS, :name), ", "),
            "  (host: 2 threads capped, ThreadedSparseMatrixCSR SpMV threaded, pointwise sweeps single-threaded)")
    for cfg in MODEL_CONFIGS
        try
            run_model(cfg)
        catch err
            println("\n", cfg.name, " FAILED: ", sprint(showerror, err))
        end
        GC.gc()
        CUDA.reclaim()
    end
    return nothing
end

main()
