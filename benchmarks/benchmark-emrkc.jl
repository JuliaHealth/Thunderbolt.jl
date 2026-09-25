# emRKC against the production reaction-diffusion split, host and device: an ep01-shaped monodomain
# sheet for two cell models, and an ideal left ventricle of ~1e6 hexahedra carrying an orthotropic
# fiber architecture. The baseline is `LieTrotterGodunov(BackwardEulerSolver + Jacobi-preconditioned
# KrylovJL_CG, AdaptiveForwardEulerSubstepper)`.
#
# MATCHED ACCURACY, NOT MATCHED STEP SIZE: the step size a method can afford is a property of the
# method and is the claim under test, so each arm picks its own and they are compared where they
# deliver the same error. Each method is referenced against ITSELF at `DTREF` in Float64 on the host
# -- not a shared reference, since emRKC lumps the mass matrix and the backward Euler stage does not,
# and a shared one would lay that O(h²) gap under both error curves as an unreachable floor. A
# reference's own error is its distance to the same method at `2 DTREF`, and the reference tightens
# its conjugate gradient past the tutorial's setting so that it does not measure the linear solve's
# own accumulated error; the timed arms keep the tutorial's tolerances. The selected step size is the
# largest in `SWEEP` staying inside `BAND`, and every timed arm is validated against its reference at
# its own step size before it is timed. Timing is the minimum over `NPASS` passes of `NSTEPS` steps,
# after `WARMUP_SECONDS` of uninterrupted stepping -- this card idles at 300 MHz, so the clocks are
# read back mid-flight per arm and printed. For splitting arms the step is decomposed over the two
# children through the same calls `OrdinaryDiffEqOperatorSplitting`'s `_perform_step!` makes; emRKC
# has no linear solve to separate out and reports its stage structure `(s, m)` in that column.
#
# MODELS: `ENV["EMRKC_MODELS"]`, comma-separated, default `"FHN,PCG2019,LV"`; they run back to back
# in one invocation and a model whose arms error out is reported without stopping the others. `BAND`,
# `NSTEPS`, `NPASS`, `WARMUP_SECONDS` are shared; `N`, `TEND`, `DTREF`, `SWEEP` apply to the two sheet
# `ModelConfig`s, which select their own step size, and the LV `LVConfig` overrides them.
#  * `ENV["EMRKC_LV_STAGES"]`, comma-separated, default `"certify,time"`: which parts of the LV run
#    to do -- `certify` (coarse-mesh step sizes), `families` (the RKC1/RKL1/RKG1 comparison, which
#    certifies on the coarse mesh itself), `time` (the ~1e6-element timed arms).
#  * `ENV["EMRKC_LV_BASE"]` / `ENV["EMRKC_LV_COARSE_BASE"]`, `"circumferential,transmural,longitudinal"`
#    before hexahedralization, default `"191,20,33"` / `"96,10,17"`: the timed and the coarse LV mesh
#    resolutions, fitted to the 8 GiB host cgroup and the 8 GiB card.
#  * FHN: 2.5mm x 2.5mm, the ep01 tutorial's dimensionless diffusion tensor, an excite/refractory box
#    initial condition developing into a sustained spiral.
#  * PCG2019: 10mm x 10mm, κ/(Cₘχ) = 0.4 mm²/ms and the planar S1 front of
#    `test/integration/test_emrkc_electrophysiology.jl`, which take ~15-20ms to cross the domain. At
#    TEND=25ms the tissue is mixed -- upstroke, plateau and resting side by side -- which is what both
#    the accuracy check and the per-step cost need to see. `gates = :all` is spelled out because
#    PCG2019's six gates are exactly what the exponential treatment is for.
#  * LV: PCG2019 on an ideal left ventricle with a Streeter helix (+60° endo to -60° epi); geometry,
#    conductivities, protocol and step sizes below.
#
# PRECISION AND THREADS: host arms run 2 threads -- `ThreadedSparseMatrixCSR`'s SpMV is threaded
# across them, the pointwise reaction/gate sweeps are not (one CUDA thread per dof on device).
# Uncapped/-t8 arms are not measured. Every device arm is Float32 throughout, `cell_rhs!` and
# `gate_coefficients` included. Float32 on the HOST is PARTIAL: element/quadrature evaluation follows
# `T` because `ep01_form`/`lv_form` pass `qrcs` explicitly, but the global M/K `SparseMatrixCSC`
# storage takes its value type from the assembly strategy and stays Float64 regardless of `T`.
# Making M/K storage follow `T` is a FerriteOperators.jl-side seam, not reachable from this file.
# The LV's microstructure is also Float64 and shared across arms, so its `SpectralTensorCoefficient`
# promotes the diffusion tensor's own assembly back to Float64 even at `T = Float32`.
#
# PRECONDITIONING: the splitting arms' conjugate gradient carries a Jacobi (diagonal) preconditioner
# on host and device through `KrylovJL_CG`'s `precs` seam (see `JacobiPrecon`). The unpreconditioned
# iteration count is measured once per configuration and printed beside the preconditioned one,
# because the linear solve is 64-93% of a splitting step. emRKC has no linear solve.
#
# RUN: CUDA is a weak dependency, so this runs in the GPU test environment. A memory-capped cgroup is
# required (`_assert_memory_capped`, in `benchmarks/common.jl`; `BENCHMARK_UNCAPPED=1` overrides) --
# an uncapped run's GC sizes itself against the whole machine rather than the 8G this is meant to run
# in. Canonical invocation:
# `systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 env JULIA_NUM_THREADS=2 julia -t2
# --heap-size-hint=3G --project=test/gpu benchmarks/benchmark-emrkc.jl`.
#
# This is a benchmark, not a CI gate: the numbers are reported as measured, whichever way they fall.
#
# RESULTS (capped 2-thread host profile, RTX 2080; N=512 sheet meshes; device arms Float32; splitting
# arms Jacobi-preconditioned):
#
# == FHN (526338 states) ==
# arm                    Δt/ms      s/step   steps/s  s / sim ms  lin solve   cg its    stages    clocks
# host emRKC             0.40      0.02260      44.2     0.05651       none        -   s=1 m=2      host
# host splitting         0.40      0.03330      30.0     0.08325        79%      5.0         -      host
# device emRKC           0.40      0.00099    1011.9     0.00247       none        -   s=1 m=2 1905/6800
# device splitting       0.40      0.00184     542.9     0.00461        81%      5.0         - 1905/6800
# host->device: emRKC 22.87x, splitting 18.08x  |  emRKC vs splitting: 1.47x host, 1.86x device
# CG iterations/step at Δt=0.40, host: 8.0 unpreconditioned, 5.0 with Jacobi
# validation (rel err, band=0.01): host emRKC 0.00968, host splitting 0.00983,
#   device emRKC 0.00968, device splitting 0.00983 -- all in band
# KNIFE EDGE: both arms land on Δt=0.40 and splitting sits 1.7% inside the band (0.00983 against
#   0.01). A 2% shift in that one error drops splitting back to 0.20 and roughly doubles the ratio,
#   so FHN's 1.47x/1.86x has to be quoted as a knife edge. PCG2019's does not: its splitting arm NaNs
#   at Δt ≥ 0.20, so 0.10 against emRKC's 0.20 is a hard factor of two.
#
# == PCG2019 (1842183 states) ==
# arm                    Δt/ms      s/step   steps/s  s / sim ms  lin solve   cg its    stages    clocks
# host emRKC             0.20      0.23544       4.2     1.17718       none        -  s=1 m=23      host
# host splitting         0.10      0.34435       2.9     3.44351        81%     59.4         -      host
# device emRKC           0.20      0.00405     247.2     0.02023       none        -  s=1 m=23 1905/6800
# device splitting       0.10      0.01111      90.0     0.11109        93%     45.2         - 1890/6800
# host->device: emRKC 58.20x, splitting 31.00x  |  emRKC vs splitting: 2.93x host, 5.49x device
# CG iterations/step at Δt=0.10, host: 114.8 unpreconditioned, 59.4 with Jacobi
# validation (rel err, band=0.01): host emRKC 0.00157, host splitting 0.00342,
#   device emRKC 0.00158, device splitting 0.00337 -- all in band
# splitting sweep: Δt=0.05 err=0.00108, Δt=0.10 err=0.00342 (selected), Δt≥0.20 NaN (the stiff gate's
#   FE substep limit; the selected Δt is safely below it)
# emRKC sweep: Δt=0.05/0.10/0.20 in band (err 0.00099/0.00139/0.00157), Δt=0.40 err=0.020 (out)
#
# == LV-PCG2019 (1031400 hexahedra, 1065057 nodes, 7455399 states) ==
# arm                    Δt/ms      s/step   steps/s  s / sim ms  lin solve   cg its    stages    clocks
# host emRKC             0.0500    2.65296       0.4    53.05925       none        -  s=1 m=28      host
# host splitting         0.0068    1.69589       0.6   249.39491        64%     47.0         -      host
# device emRKC           0.0500    0.03048      32.8     0.60956       none        -  s=1 m=28 1905/6800
# device splitting       0.0068    0.05269      19.0     7.74867        88%     46.6         - 1890/6800
# host->device: emRKC 87.04x, splitting 32.19x  |  emRKC vs splitting: 4.70x host, 12.71x device
# CG iterations/step at Δt=0.0068, host: 83.5 unpreconditioned, 47.0 with Jacobi -- the conditioning
#   comes from the diffusion operator, not the reaction solver
# host-vs-device agreement over 5 steps, both Float32: emRKC 3.30e-6, splitting 1.08e-6
# memory: host peak 3.80 GiB of the 8 GiB cgroup (52% headroom), device 0.94 GiB of 7.60 (88%)
# emRKC's Δt=0.05 is CERTIFIED directly on the coarse mesh: the sweep 0.20/0.10/0.05/0.025 gives rel
#   err 0.1294/0.03402/0.003255/0.0005165, so 0.05 is the largest in-band value.
# splitting's Δt=0.0068 is UNCERTIFIED. Its reference does not converge within the budget spent on it
#   (own error 0.0348 at the finest level tried, ~3.5x above BAND), and none of 0.05/0.025/0.0125
#   lands in BAND against it (0.197/0.099/0.037). The Richardson fallback off the finest pair
#   (apparent order 0.91) gives Δt ≲ 0.0016 -- an extrapolation off an unconverged reference, not a
#   certification. The carried 0.0068 is what is timed, pending a deeper ladder.
#
# STS FAMILY COMPARISON (RKC1 vs RKL1 vs RKG1, outer=inner=family, Δt=0.05, measured via
# `EMRKC_LV_STAGES=families`; `lv_family_comparison`): all three in band at Δt=0.05, no ladder fallback.
#   family   s   m   host s/sim-ms   device s/sim-ms   rel err (own Float64 ref, coarse mesh)
#   RKC1     1  28        53.11232           0.57388                             0.003255
#   RKL1     1  39        72.32883           0.75003                             0.002809
#   RKG1     1  52        95.19491           0.98767                             0.004172
# RKC1 cheapest on both host (1.36x/1.79x behind RKL1/RKG1) and device (1.31x/1.72x); no accuracy
#   trade favors the extra cost. Default (RKC1) stands.
#
# LV GEOMETRY: `inner_radius`/`outer_radius` fix the wall at exactly 6 mm (40 transmural elements at
# h = 150 µm) and `apex_inner`/`apex_outer` set the long axis independently, breaking the generator's
# default proportions on purpose: keeping both h and the transmural count at ~1e6 elements is only
# reachable by shrinking the other dimensions. Endocardial equatorial radius 3.112 mm, epicardial
# 9.112 mm, apex-base length 8.73 mm, median true edge 126.0 µm, wall volume 1.501 mL. The chamber is
# a thick-walled, small, non-physiological ventricle by design -- the accepted trade.
#
# DIFFUSION is genuinely orthotropic: `SpectralTensorCoefficient` over an
# `OrthotropicMicrostructureModel` from `create_simple_microstructure_model` on the
# `compute_lv_coordinate_system` frame. σ = (0.13342, 0.02674, 0.00859) mS/mm in the (fiber,
# sheetlet, normal) frame with Cₘ = 0.01 µF/mm² and χ = 140 /mm, so D = (0.0953, 0.0191, 0.0061)
# mm²/ms. The fiber value is the harmonic mean of Clerc's measured intra- and extracellular
# longitudinal conductivities -- the monodomain reduction the Niederer et al. N-version benchmark
# uses, and the number `ep04_geselowitz-ecg.jl` already carries; the cross-fiber pair splits that
# benchmark's single transverse value (0.0176) by the squared ratios of the orthotropic conduction
# velocities Caldwell et al. (2009) measured in ventricular tissue (0.67 : 0.30 : 0.17 m/s), and
# brackets it. Measured tissue data throughout, not a fit. That the microstructure reaches the
# assembly is checked on the coarse mesh against a trace-matched isotropic tensor: the two disagree
# by 0.169 after 20 steps, where one that never arrived would agree to round-off.
#
# DEVICE PATH for the LV is host-assembled and mirrored, not device-assembled: the field-backed
# `OrthotropicMicrostructureModel` stores its f/s/n vectors in host `ElementwiseData` with no adapt
# rule, so it cannot cross the `KernelAbstractionsDevice` assembly seam. The host strategy assembles
# and `MirroredBilinearOperator` uploads the nonzeros -- what the sheet device arms use too. Assembly
# is setup-only at fixed Δt, so this costs the timed arms nothing, and the device arms solve with
# exactly the host's orthotropic matrix; the host-vs-device agreement above is what says so.
#
# PROTOCOL: an apex S1 written as an initial condition -- the apical 12% of the long axis raised to
# 20 mV, the rest at PCG2019's resting default -- over a 15 ms window, no full beat. At 15 ms the
# tissue is mixed: 57.1% above -40 mV with φₘ ∈ [-85, 21] mV under emRKC, 77.1% and [-88, 23] mV
# under splitting.

using Thunderbolt
using CUDA
using LinearSolve
using OrdinaryDiffEqOperatorSplitting
using LinearAlgebra
using StaticArrays
using Printf

import Ferrite
import Thunderbolt: SciMLBase, ThreadedSparseMatrixCSR, create_simple_microstructure_model
import OrdinaryDiffEqOperatorSplitting:
    advance_solution_by!, forward_sync_subintegrator!, backward_sync_subintegrator!

# The Jacobi preconditioner, the cgroup guard and the timing harness, shared with the other
# benchmarks in this directory.
include(joinpath(@__DIR__, "common.jl"))

const N              = 512          # elements per side; the size at which the solve dominates
const TEND           = 25.0         # ms, the first EP tutorial's own visualization window
const DTREF          = 0.025        # ms, reference step size
const BAND           = 1.0e-2       # final-time φₘ relative error a step size has to stay inside
const SWEEP          = (0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2)
const NSTEPS         = 25
const NPASS          = 3
const WARMUP_SECONDS = 1.5
const CuCSR          = CUDA.CUSPARSE.CuSparseMatrixCSR{Float32, Int32}

####################################
## Conjugate gradient bookkeeping
####################################

"Krylov iterations of the most recent backward Euler solve; `stats` is reset per solve, so this has
to be read out per step."
cg_iters(child) = child.cache.stage.linear_solver.cacheval.stats.niter

"Mean conjugate gradient iterations per step over `n` real steps. The integrator is advanced."
function mean_cg_iters!(integrator, n = NSTEPS)
    total = 0
    for _ = 1:n
        step!(integrator)
        total += cg_iters(integrator.child_subintegrators[1])
    end
    return total / n
end

####################################
## Problem
####################################

"One sheet model's mesh extent, monodomain coefficients, ionic model constructor and initial condition."
struct ModelConfig
    name::String
    ion::Function                    # (::Type{T}) -> ionic model instance
    Cₘ::Float64
    χ::Float64
    κ::SymmetricTensor{2, 2, Float64}
    L::Float64                       # domain side
    u0!::Function                    # (u₀, form, ::Type{T}) -> u₀
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
A planar S1 activation front: the left 12% of the domain driven to an excited potential, the rest at
the model's resting default -- the protocol `test/integration/test_emrkc_electrophysiology.jl`
validates for this model, scaled to this benchmark's domain. It must stay continuous: a spiral-chasing
IC with a discontinuous `h` jump keeps the self-referenced reference error above `BAND` at every step
size, which is an unconverged reference rather than a step-size failure.
"""
function pcg2019_u0!(u₀, form, ::Type{T}) where {T}
    setvariable!(u₀, form, :φₘ) do x
        x[1] ≤ T(0.12PCG2019_L) ? T(20.0) : T(-85.0)
    end
    return u₀
end

const FHN_CONFIG = ModelConfig(
    "FHN", T -> Thunderbolt.ParametrizedFHNModel{T}(), 1.0, 1.0,
    SymmetricTensor{2, 2, Float64}((4.5e-5, 0.0, 2.0e-5)), 2.5, fhn_u0!,
)
const PCG2019_CONFIG = ModelConfig(
    "PCG2019", T -> Thunderbolt.ParametrizedPCG2019Model{T}(), 1.0, 1.0,
    SymmetricTensor{2, 2, Float64}((0.4, 0.0, 0.4)), PCG2019_L, pcg2019_u0!,
)
const MODEL_CONFIGS = String[strip(m) for m in split(get(ENV, "EMRKC_MODELS", "FHN,PCG2019,LV"), ",")]

# `mass` is the discretization's treatment: the emRKC arms take the lumped form, the implicit
# splitting arms the consistent one, on the same mesh and interpolation (shared dof layout and u₀).
function ep01_form(::Type{T}, cfg::ModelConfig; mass = LumpedMass()) where {T}
    mesh = generate_mesh(Quadrilateral, (N, N), Vec{2}((0.0, 0.0)), Vec{2}((cfg.L, cfg.L)))
    model = MonodomainModel(
        ConstantCoefficient(T(cfg.Cₘ)),
        ConstantCoefficient(T(cfg.χ)),
        ConstantCoefficient(convert(SymmetricTensor{2, 2, T}, cfg.κ)),
        NoStimulationProtocol(),
        cfg.ion(T),
        CartesianCoordinateSystem(mesh),
        :φₘ, :s,
    )
    # `qrcs` makes the quadrature (hence element-evaluation) precision follow `T`; left at the
    # `FiniteElementDiscretization` default it is `Float64` regardless of `T`.
    return semidiscretize(
        ReactionDiffusionSplit(model),
        FiniteElementDiscretization(
            Dict(:φₘ => LagrangeCollection{1}());
            qrcs = Dict(:φₘ => QuadratureRuleCollection(T, 2)),
            mass,
        ),
        mesh,
    )
end

function ep01_u0(form, ::Type{T}, cfg::ModelConfig) where {T}
    u₀ = create_initial_condition(form, T)
    cfg.u0!(u₀, form, T)
    return u₀
end

# The two algorithms under test, as a downstream user would spell them; linear solver tolerances are
# the first EP tutorial's. `gates = :all` is EMRKC's own default, spelled out so the choice is visible.
# `outer`/`inner` default to `RKC1(0.05)`, `EMRKC`'s own default for both -- so every call site that
# does not pass them is byte-identical to before this kwarg existed. They are independently settable
# because `EMRKC` already exposes them that way (`outer`, `inner` fields of
# `ExponentialMultirateSTSAlgorithm`); see `lv_family_comparison` for the STS-family comparison arms.
emrkc(::Type{VT}, ::Type{MT}; outer = RKC1(0.05), inner = RKC1(0.05)) where {VT, MT} =
    EMRKC(solution_vector_type = VT, system_matrix_type = MT, gates = :all, outer = outer, inner = inner)

"""
`reaction = :substepper` (the default) is the production shape: `AdaptiveForwardEulerSubstepper` with
`reaction_threshold = 0.1` and `substeps = 10`, as the ep01 tutorial and `benchmark-gpu-split.jl`
carry it, which decouples reaction stability and accuracy from the outer step.
`reaction = :plain` is a plain `ForwardEulerCellSolver` -- one Euler update at the *outer* Δt, no
internal error control -- for a labeled secondary comparison only.
"""
function splitting(
    ::Type{VT}, ::Type{MT}; atol = 1.0e-6, rtol = 1.0e-5, jacobi = true,
    reaction = :substepper, reaction_threshold = 0.1, substeps = 10,
) where {VT, MT}
    T = eltype(VT)
    cg = jacobi ?
        KrylovJL_CG(atol = T(atol), rtol = T(rtol), precs = jacobi_precs) :
        KrylovJL_CG(atol = T(atol), rtol = T(rtol))
    cell_solver = reaction === :substepper ?
        AdaptiveForwardEulerSubstepper(
            solution_vector_type = VT, reaction_threshold = T(reaction_threshold), substeps = substeps,
        ) :
        reaction === :plain ? ForwardEulerCellSolver(solution_vector_type = VT) :
        error("splitting: unknown reaction = $reaction (:substepper or :plain)")
    return LieTrotterGodunov((
        BackwardEulerSolver(
            solution_vector_type = VT,
            system_matrix_type   = MT,
            inner_solver         = cg,
        ),
        cell_solver,
    ))
end

function solve_to_end(form, u0, alg, Δt, tend = TEND)
    integrator = build(form, u0, alg, Δt, oftype(Δt, tend))
    solve!(integrator)
    return integrator
end

"`n` real steps from the initial condition. The LV arms validate this way rather than over the whole
window: at a million dofs a full-window host solve per arm costs more than the timing it guards, and
that the device reproduces the host is visible after a few steps."
function solve_n_steps(form, u0, alg, Δt, n)
    integrator = build(form, u0, alg, Δt, oftype(Δt, 1.0e5))
    for _ = 1:n
        step!(integrator)
    end
    return integrator
end

relerr(a, b) = norm(Vector(a) .- Vector(b)) / norm(Vector(b))

####################################
## Reference and step size selection
####################################

"""
The Float64 host reference of one method and the estimate of its own error: the distance to the same
method at twice the step size, which for a first order scheme bounds that error up to a factor two.
"""
function reference(form, u0, alg, φₘ, tend = TEND)
    coarse = getvariable(solve_to_end(form, u0, alg, 2DTREF, tend).u, φₘ)
    fine   = getvariable(solve_to_end(form, u0, alg, DTREF, tend).u, φₘ)
    return copy(fine), relerr(coarse, fine)
end

"""
The largest step size in `SWEEP` whose final state stays inside `BAND` of `φ_ref`, and the error it
lands at. Errors are printed for the whole sweep: where the band is crossed is as much of the result
as which step size wins.
"""
function select_dt(form, u0, alg, φₘ, φ_ref, label)
    best = nothing
    for Δt in SWEEP
        # In the solution vector's precision: `(t, Δt)` reach the reaction kernel, so a Float64 step
        # size against Float32 storage would run the hot loop in double precision.
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

"""
Seconds per step spent in each child of a splitting step, through the same three calls
`OrdinaryDiffEqOperatorSplitting`'s `_perform_step!` makes. The integrator is advanced by real steps
and left usable.

Also returns the first child's mean conjugate gradient iterations, read out of the same steps rather
than a second pass: every extra step advances the Float32 clock these arms run on, and the splitting
integrator's parent/child time synchronization is what pays for that drift.
"""
function child_composition!(integrator, Δt)
    times = zeros(length(integrator.child_subintegrators))
    iters = 0
    for _ = 1:NSTEPS
        for (i, child) in enumerate(integrator.child_subintegrators)
            idxs   = integrator.child_solution_indices[i]
            syncer = integrator.child_synchronizers[i]
            forward_sync_subintegrator!(integrator, child, idxs, syncer)
            t0 = time_ns()
            advance_solution_by!(integrator, child, Δt)
            sync(integrator.u)
            times[i] += (time_ns() - t0) / 1.0e9
            i == 1 && (iters += cg_iters(child))
            backward_sync_subintegrator!(integrator, child, idxs, syncer)
        end
    end
    return times ./ NSTEPS, iters / NSTEPS
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
    cg_iters::Float64         # mean conjugate gradient iterations per step; NaN where there is none
    stages::String
    clocks::Tuple{Int, Int}
end

"""
Validate the arm, then time it. Returns the arm and the validated final `φₘ` on the host, so a host
arm can serve as the reference its device counterpart is checked against.

`steps` validates over that many steps from the initial condition instead of over the whole window;
`φ_ref === nothing` validates finiteness only, for an arm whose step size is certified elsewhere.
"""
function run_arm(
    label, method, form, u0, alg, Δt, φₘ, φ_ref, on_device, has_solve;
    tend = TEND, steps = nothing,
)
    checked = steps === nothing ? solve_to_end(form, u0, alg, Δt, tend) :
              solve_n_steps(form, u0, alg, Δt, steps)
    φ = getvariable(Vector(checked.u), φₘ)
    all(isfinite, φ) || error("$label: validation produced a non-finite solution")
    err = φ_ref === nothing ? NaN : relerr(φ, φ_ref)
    stages = if has_solve
        ""
    else
        s, _, m = Thunderbolt._emrkc_step_sizing(checked.alg, Δt, checked.cache.ρS, checked.cache.ρF)
        "s=$s m=$m"
    end

    # Timed from `WARMUP_SECONDS` of real steps past the initial condition, so `tend` here only has
    # to outlast the warmup and the passes.
    timed = build(form, copy(u0), alg, Δt, oftype(Δt, 1.0e5))
    clocks = prewarm!(timed, on_device, WARMUP_SECONDS)
    seconds = measure!(timed, NSTEPS, NPASS)

    fraction, iters = NaN, NaN
    if has_solve
        parts, iters = child_composition!(timed, Δt)
        fraction = parts[1] / sum(parts)
    end
    arm = Arm(label, method, Δt, seconds, err, fraction, iters, stages, clocks)
    GC.gc()
    on_device && CUDA.reclaim()
    return arm, φ
end

function report(arms)
    println("\n", "="^128)
    @printf("%-22s %-7s %9s %9s %11s %10s %9s %9s %9s\n",
            "arm", "Δt/ms", "s/step", "steps/s", "s / sim ms", "lin solve", "cg its",
            "stages", "clocks")
    println("-"^128)
    for a in arms
        @printf("%-22s %-7.4f %9.5f %9.1f %11.5f %10s %9s %9s %9s\n",
                a.label, a.Δt, a.seconds_per_step, 1 / a.seconds_per_step,
                a.seconds_per_step / a.Δt,
                isnan(a.solve_fraction) ? "none" : @sprintf("%.0f%%", 100a.solve_fraction),
                isnan(a.cg_iters) ? "-" : @sprintf("%.1f", a.cg_iters),
                isempty(a.stages) ? "-" : a.stages,
                a.clocks == (0, 0) ? "host" : @sprintf("%d/%d", a.clocks[1], a.clocks[2]))
    end
    println("-"^128)
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
    println("\nvalidation (φₘ relative error against this arm's reference):")
    for a in arms
        if isnan(a.validation)
            @printf("  %-22s finite; step size set on the coarse mesh, see the header block\n", a.label)
        else
            @printf("  %-22s %.6g %s\n", a.label, a.validation,
                    a.validation ≤ 2BAND ? "" : "  <-- OUT OF BAND")
        end
    end
end

####################################
## Ideal left ventricle
####################################

"""
The LV arm's geometry and step sizes. Unlike the sheet configs this one does not select its own step
size: it carries fixed ones and certifies them on `coarse_base` (see `run_model(::LVConfig)`).
"""
struct LVConfig
    name::String
    base::NTuple{3, Int}          # circumferential, transmural, longitudinal, BEFORE hexahedralization
    coarse_base::NTuple{3, Int}
    Δt_emrkc::Float64
    Δt_split::Float64
end

# `generate_ideal_lv_mesh` emits a wedge fan over the apex, so the mesh is built at 2h and
# `hexahedralize`d: that halves h and makes every cell a hexahedron while keeping the fan's apex.
# `generate_ideal_lv_mesh_hex`'s all-hex O-grid cap was measured as an alternative at matched
# resolution and rejected -- its minimum edge is smaller than the fan's (1.41 vs 3.18 µm) and its
# median/min spread more than double, the defect sitting inside the O-grid core's mapping, so the
# alternative relocates a same-order sliver rather than fixing the apex.
const LV_INNER_RADIUS = 3.1122   # mm, endocardial equatorial radius -- shrunk, non-physiological
const LV_OUTER_RADIUS = LV_INNER_RADIUS + 6.0   # mm, wall fixed at 6 mm (40 elements transmural)
const LV_APEX_INNER   = 5.7798   # mm
const LV_APEX_OUTER   = 6.6690   # mm, sets the long axis: apex-base length = LV_APEX_OUTER*(1-cospi(0.6))
const LV_Z_APEX  = LV_APEX_OUTER
const LV_Z_BASE  = LV_APEX_OUTER * cospi(0.6)
const LV_STIM_Z  = LV_Z_APEX - 0.12(LV_Z_APEX - LV_Z_BASE)
const LV_TEND    = 15.0          # ms; the front transits the apical wall and starts apicobasal
const LV_STEPS   = 5             # host-vs-device agreement steps on the timed mesh

# Monodomain conductivities, mS/mm, in the (fiber, sheetlet, normal) frame; sources in the header.
const LV_σ  = SVector(0.13342, 0.02674, 0.00859)
const LV_Cₘ = 0.01               # µF/mm²
const LV_χ  = 140.0              # 1/mm ; D = σ/(Cₘχ) = (0.0953, 0.0191, 0.0061) mm²/ms

host_rss_gib() = parse(Int, split(read("/proc/self/statm", String))[2]) * 4096 / 1024^3
gpu_used_gib() = (CUDA.total_memory() - CUDA.free_memory()) / 1024^3

function lv_geometry(base)
    mesh = hexahedralize(generate_ideal_lv_mesh(
        base...;
        inner_radius = LV_INNER_RADIUS, outer_radius = LV_OUTER_RADIUS,
        apex_inner = LV_APEX_INNER, apex_outer = LV_APEX_OUTER, longitudinal_upper = 0.2,
    ))
    cs = compute_lv_coordinate_system(mesh)
    microstructure = create_simple_microstructure_model(
        cs, LagrangeCollection{1}()^3;
        endo_helix_angle = deg2rad(60.0), epi_helix_angle = deg2rad(-60.0),
    )
    return mesh, microstructure
end

"""
`MonodomainModel`'s first two positional arguments are `χ, Cₘ` by its own field order, which every
other call site in the repository spells the other way round. They only ever enter as the product
`Cₘχ`, so the disagreement is invisible until they differ -- as they do here. Struct order it is.
"""
function lv_form(::Type{T}, mesh, microstructure; κ = nothing, mass = LumpedMass()) where {T}
    # σ follows `T`; the microstructure's f/s/n fields do not -- it is built once in Float64 and
    # SHARED across every arm, so one Float64 factor per quadrature point promotes the diffusion
    # tensor's own assembly back to Float64 even at `T = Float32`. Mass, reaction and σ's own scalars
    # are unaffected. Making the microstructure precision-parametric would mean holding it twice.
    model = MonodomainModel(
        ConstantCoefficient(T(LV_χ)),
        ConstantCoefficient(T(LV_Cₘ)),
        κ === nothing ? SpectralTensorCoefficient(microstructure, ConstantCoefficient(T.(LV_σ))) : κ,
        NoStimulationProtocol(),
        Thunderbolt.ParametrizedPCG2019Model{T}(),
        CartesianCoordinateSystem(mesh),
        :φₘ, :s,
    )
    return semidiscretize(
        ReactionDiffusionSplit(model),
        FiniteElementDiscretization(
            Dict(:φₘ => LagrangeCollection{1}());
            qrcs = Dict(:φₘ => QuadratureRuleCollection(T, 2)),
            mass,
        ),
        mesh,
    )
end

"An apex S1 stimulus written as an initial condition -- the apical 12% of the long axis raised above
threshold, the rest at the cell model's resting default. Same shape as the sheet PCG2019 config, and
for the same reason: no applied-current amplitude to tune against excitability."
function lv_u0(form, ::Type{T}) where {T}
    u₀ = create_initial_condition(form, T)
    setvariable!(u₀, form, :φₘ) do x
        x[3] ≥ LV_STIM_Z ? T(20.0) : T(-85.0)
    end
    return u₀
end

"Fraction of the tissue above -40 mV, and the range of φₘ: what says the timed state is a front
mid-flight rather than uniformly resting or uniformly plateaued."
function activation_state(φ)
    v = Vector(φ)
    return count(>(-40.0f0), v) / length(v), minimum(v), maximum(v)
end

# Which parts of the LV run to do, as one comma-separated set (mirroring `DV_STAGES` in
# `benchmark-discretization-variants.jl`):
#   certify  -- the coarse-mesh step size certification, a property of the physics and not of the
#               timed mesh, so it can be dropped when re-timing already-certified step sizes
#   families -- the RKC1/RKL1/RKG1 comparison (`lv_family_comparison`), which certifies each family
#               on the coarse mesh itself and then times it
#   time     -- the ~1e6-element setup and the timed arms
const LV_STAGES = Set(strip(s) for s in split(get(ENV, "EMRKC_LV_STAGES", "certify,time"), ","))

function run_model(cfg::LVConfig)
    println("\n", "#"^128)
    println("# ", cfg.name, "  (ideal LV, wall ", LV_OUTER_RADIUS - LV_INNER_RADIUS,
            " mm, PCG2019, gates = :all, device Float32; stages ",
            join(sort(collect(LV_STAGES)), ","), ")")
    println("#"^128)

    "certify" in LV_STAGES && lv_certify_step_sizes(cfg)
    "families" in LV_STAGES && lv_family_comparison(cfg)
    "time" in LV_STAGES && lv_time_arms(cfg)
    return nothing
end

function lv_certify_step_sizes(cfg::LVConfig)
    println("\n-- coarse spot-check mesh --")
    t0 = time()
    cmesh, cms = lv_geometry(cfg.coarse_base)
    @printf("  %d hexahedra, %d nodes, setup %.0f s, host RSS %.2f GiB\n",
            Ferrite.getncells(cmesh.grid), Ferrite.getnnodes(cmesh.grid), time() - t0, host_rss_gib())

    cform64 = lv_form(Float64, cmesh, cms)
    cu64    = lv_u0(cform64, Float64)
    cform32 = lv_form(Float32, cmesh, cms)
    cu32    = lv_u0(cform32, Float32)
    φc64, φc32 = solution_variable(cform64, :φₘ), solution_variable(cform32, :φₘ)
    cform64c = lv_form(Float64, cmesh, cms; mass = ConsistentMass())
    cform32c = lv_form(Float32, cmesh, cms; mass = ConsistentMass())

    # Orthotropy is live, not merely configured: a trace-matched isotropic tensor has to produce a
    # different solution, where a microstructure that never reached the assembly agrees to round-off.
    σ_iso = sum(LV_σ) / 3
    ciso = lv_form(Float32, cmesh, cms; κ = ConstantCoefficient(
        SymmetricTensor{2, 3, Float64}((σ_iso, 0.0, 0.0, σ_iso, 0.0, σ_iso))))
    alg32 = emrkc(Vector{Float32}, ThreadedSparseMatrixCSR{Float32, Int32})
    φ_ortho = getvariable(solve_n_steps(cform32, cu32, alg32, Float32(cfg.Δt_emrkc), 20).u, φc32)
    φ_iso   = getvariable(solve_n_steps(ciso, lv_u0(ciso, Float32), alg32, Float32(cfg.Δt_emrkc), 20).u,
                          solution_variable(ciso, :φₘ))
    anisotropy = relerr(φ_ortho, φ_iso)
    @printf("  orthotropic vs trace-matched isotropic after 20 steps: rel diff %.4g %s\n",
            anisotropy, anisotropy > 1.0e-3 ? "(microstructure is live)" : "<-- SUSPECT")
    ciso = φ_iso = φ_ortho = nothing
    GC.gc()

    println("\n  self-referenced convergence at the carried step sizes (reference Float64, arms Float32):")
    cpu64_emrkc = emrkc(Vector{Float64}, ThreadedSparseMatrixCSR{Float64, Int64})
    φ_ref, ref_err = reference(cform64, cu64, cpu64_emrkc, φc64, LV_TEND)
    got = solve_to_end(cform32, cu32, alg32, Float32(cfg.Δt_emrkc), Float32(LV_TEND))
    φ = getvariable(got.u, φc32)
    ok = got.sol.retcode == SciMLBase.ReturnCode.Success && all(isfinite, φ)
    err = ok ? relerr(φ, φ_ref) : NaN
    frac, lo, hi = ok ? activation_state(φ) : (NaN, NaN, NaN)
    @printf("    %-10s Δt = %5.3f  rel err = %-9.4g %-8s\n",
            "emRKC", cfg.Δt_emrkc, err, ok ? (err ≤ BAND ? "in band" : "OUT OF BAND") : "FAILED")
    if ref_err ≥ BAND
        @printf("               reference UNCONVERGED (own error %.3g ≥ band): the number above is\n",
                ref_err)
        @printf("               not a verdict. Richardson: E(Δt) ≈ %.3g·Δt, so Δt ≲ %.4f for the band.\n",
                ref_err / DTREF, BAND * DTREF / ref_err)
    else
        @printf("               reference's own error %.3g\n", ref_err)
    end
    @printf("               t = %.0f ms: %.1f%% of tissue above -40 mV, φₘ ∈ [%.1f, %.1f] mV\n",
            LV_TEND, 100frac, lo, hi)
    φ_ref = φ = nothing

    lv_certify_splitting(cform64c, cu64, cform32c, cu32, φc64, φc32, cfg.Δt_split)

    cmesh = cms = cform64 = cform32 = cform64c = cform32c = cu64 = cu32 = nothing
    GC.gc()
    @printf("  host RSS after releasing the coarse mesh: %.2f GiB\n", host_rss_gib())
    return nothing
end

####################################
## Splitting certification: mechanism check + Δt-refinement ladder
####################################

"Fraction of tissue above -40 mV, the same threshold `activation_state` uses."
active_fraction(φ) = count(>(-40.0), Vector(φ)) / length(φ)

"""
Solve to `tend`, tracking the active-tissue fraction every real step. Returns the linearly
interpolated time it first reaches `target` (`nothing` if it never does) and the final φₘ, so the
mechanism check and the certification reference share one pass rather than doubling the runs.
"""
function solve_with_arrival(form, u0, alg, Δt, φₘ, tend; target = 0.5)
    integrator = build(form, copy(u0), alg, Δt, oftype(Δt, tend))
    t_prev = integrator.t
    f_prev = active_fraction(getvariable(integrator.u, φₘ))
    arrival = f_prev ≥ target ? t_prev : nothing
    while integrator.t < tend
        step!(integrator)
        if arrival === nothing
            f_now = active_fraction(getvariable(integrator.u, φₘ))
            f_now ≥ target && (arrival = t_prev + (target - f_prev) / (f_now - f_prev) * (integrator.t - t_prev))
            t_prev, f_prev = integrator.t, f_now
        end
    end
    return arrival, copy(getvariable(Vector(integrator.u), φₘ))
end

"log2 of the ratio between two successive Δt-halving errors; NaN where that is not meaningful."
apparent_order(e1, e2) = (e1 === nothing || e2 === nothing || e1 ≤ 0 || e2 ≤ 0) ? NaN : log2(e1 / e2)

"""
Whether a sub-linear apparent convergence order is the integrator's or the metric's: runs a Δt-halving
ladder and measures the apparent order of two error metrics on the SAME solves -- the final-time L2
relative error, and the shift in arrival time (first crossing of 50% active tissue). For a traveling
front, final-time L2 error is dominated by the front's phase offset once that exceeds the front width,
and a first-order phase error then shows as an L2 order of ~0.5 while arrival order stays ~1. That
combination means the metric sits in that regime; anything else is reported as measured.

Runs the production-shaped splitting arm (`reaction = :substepper`) in Float64 with tight CG
tolerance -- the same runs the certification reference is built from.
"""
function lv_mechanism_check(cform64, cu64, φc64, alg, Δts, tend)
    println("\n  mechanism check for splitting's apparent order (Float64, reaction = :substepper):")
    arrivals, φs = Float64[], Any[]
    for Δt in Δts
        arrival, φ = solve_with_arrival(cform64, cu64, alg, Δt, φc64, tend)
        push!(arrivals, something(arrival, NaN))
        push!(φs, φ)
        @printf("    Δt = %6.4f  arrival(50%% active) = %s ms\n", Δt,
                arrival === nothing ? "never" : @sprintf("%.4f", arrival))
    end
    l2_errs    = [relerr(φs[i], φs[i + 1]) for i = 1:(length(φs) - 1)]
    arr_shifts = [abs(arrivals[i] - arrivals[i + 1]) for i = 1:(length(arrivals) - 1)]
    for i in eachindex(l2_errs)
        @printf("    Δt %.4f -> %.4f : L2 rel err %.4g, arrival shift %.4g ms\n",
                Δts[i], Δts[i + 1], l2_errs[i], arr_shifts[i])
    end
    l2_order  = length(l2_errs)    >= 2 ? apparent_order(l2_errs[1], l2_errs[2])       : NaN
    arr_order = length(arr_shifts) >= 2 ? apparent_order(arr_shifts[1], arr_shifts[2]) : NaN
    @printf("    apparent order: L2 = %.3g, arrival-time = %.3g  %s\n", l2_order, arr_order,
            (isfinite(l2_order) && isfinite(arr_order) && l2_order < 0.7 && arr_order > 0.8) ?
            "-- consistent with the sqrt-of-phase-error regime" : "")
    return (; Δts, l2_errs, arr_shifts, l2_order, arr_order, φs)
end

"""
Certifies the production-shaped (`reaction = :substepper`) splitting arm's Δt on the coarse mesh: runs
the mechanism-check ladder to build a Float64 reference, extends it by up to `max_extra` further
halvings while the fitted L2 order says that is worth it, then tests `candidates` in the Float32
production configuration against the best available reference and reports the largest one inside
`BAND`. The Richardson bound off the ladder's own data is always also reported, as a stated fallback
for a reference that does not converge within budget.
"""
function lv_certify_splitting(
    cform64, cu64, cform32, cu32, φc64, φc32, Δt_carried;
    ladder = (0.05, 0.025, 0.0125), candidates = (0.05, 0.025, 0.0125), max_extra = 1,
    target_margin = 5.0,
)
    alg64 = splitting(Vector{Float64}, ThreadedSparseMatrixCSR{Float64, Int64};
                       atol = 1.0e-12, rtol = 1.0e-10)
    mech = lv_mechanism_check(cform64, cu64, φc64, alg64, ladder, LV_TEND)
    Δts, φs, l2_errs = collect(mech.Δts), mech.φs, mech.l2_errs

    # A non-positive or non-finite order means the ladder is not visibly converging, so extending it
    # is not justified; stop and fall back to Richardson instead.
    extra = 0
    while extra < max_extra && isfinite(mech.l2_order) && mech.l2_order > 0.05 &&
        l2_errs[end] > BAND / target_margin
        Δt_next = last(Δts) / 2
        arrival, φ = solve_with_arrival(cform64, cu64, alg64, Δt_next, φc64, LV_TEND)
        push!(Δts, Δt_next)
        push!(φs, φ)
        push!(l2_errs, relerr(φs[end - 1], φ))
        @printf("    extended: Δt %.4f -> %.4f : L2 rel err %.4g\n", Δts[end - 1], Δts[end], l2_errs[end])
        extra += 1
    end
    ref_err = l2_errs[end]
    φ_ref   = φs[end]
    converged = ref_err < BAND / target_margin
    @printf("  reference after %d level(s) (finest Δt = %.5f): own error %.4g -- %s\n",
            length(Δts), Δts[end], ref_err, converged ? "converged" : "NOT converged within budget")

    println("\n  certifying splitting (Float32, production config, reaction = :substepper):")
    best = nothing
    for Δt in candidates
        arm = splitting(Vector{Float32}, ThreadedSparseMatrixCSR{Float32, Int32})
        got = solve_to_end(cform32, cu32, arm, Float32(Δt), Float32(LV_TEND))
        φ = getvariable(got.u, φc32)
        ok = got.sol.retcode == SciMLBase.ReturnCode.Success && all(isfinite, φ)
        err = ok ? relerr(φ, φ_ref) : NaN
        @printf("    Δt = %6.4f  rel err vs reference = %-10.4g %s\n", Δt, err,
                ok ? (err ≤ BAND ? "in band" : "") : "FAILED")
        ok && err ≤ BAND && (best === nothing || Δt > best[1]) && (best = (Δt, err))
    end
    if converged
        if best === nothing
            println("  NO candidate Δt in ", candidates, " lands inside BAND against the converged reference.")
        else
            @printf("  CERTIFIED: Δt = %.4f, err = %.4g against a converged reference (own error %.4g)\n",
                    best[1], best[2], ref_err)
        end
    else
        println("  Reference did not converge within budget -- the candidate table above is informative,",
                " not a certification.")
    end

    # The order is refit from the finest available pair, more local than `mech.l2_order`, which is
    # fixed to the first two ladder levels. Under E(Δt) ≈ C·Δt^p with `l2_errs[end] ≈ E(Δts[end])`,
    # E(Δt) = BAND at Δt = Δts[end]·(BAND/E)^(1/p).
    p = length(l2_errs) >= 2 ? apparent_order(l2_errs[end - 1], l2_errs[end]) : NaN
    if isfinite(p) && p > 0
        richardson_dt = Δts[end] * (BAND / l2_errs[end])^(1 / p)
        @printf("  Richardson-off-the-substepper-arm fallback: apparent order %.3g (finest pair)\n", p)
        @printf("               E(Δt) ≈ C·Δt^%.3g with E(%.4f) = %.4g => Δt ≲ %.4f\n",
                p, Δts[end], l2_errs[end], richardson_dt)
    else
        @printf("  Richardson fallback unavailable: apparent order %.3g is not usable for extrapolation.\n", p)
    end
    @printf("  (carried Δt = %.4f for comparison)\n", Δt_carried)
    return nothing
end

####################################
## STS family comparison (RKC1 vs RKL1 vs RKG1)
####################################

# Fallback Δt ladder for a family whose error at `cfg.Δt_emrkc` misses `BAND`: the same points
# `lv_certify_step_sizes`'s header block reports for RKC1 (0.20/0.10/0.05/0.025). Error decreases
# monotonically with Δt on that ladder for RKC1; checked, not assumed, for RKL1/RKG1 too.
const LV_FAMILY_LADDER = (0.2, 0.1, 0.05, 0.025)

"""
Self-referenced coarse-LV band check for one STS family (`outer = inner = fam`, matching how a
downstream user would pick a family) at `Δt_default`, the LV's carried `Δt_emrkc`. Falls back through
`LV_FAMILY_LADDER` for the largest in-band Δt if `Δt_default` itself misses `BAND`. Returns
`(Δt, err, ref_err)`; errors if no ladder point lands in band.
"""
function lv_family_band_check(cform64, cu64, cform32, cu32, φc64, φc32, fam, label, Δt_default)
    alg64 = emrkc(Vector{Float64}, ThreadedSparseMatrixCSR{Float64, Int64}; outer = fam, inner = fam)
    alg32 = emrkc(Vector{Float32}, ThreadedSparseMatrixCSR{Float32, Int32}; outer = fam, inner = fam)
    φ_ref, ref_err = reference(cform64, cu64, alg64, φc64, LV_TEND)

    function trial(Δt)
        got = solve_to_end(cform32, cu32, alg32, Float32(Δt), Float32(LV_TEND))
        φ = getvariable(got.u, φc32)
        ok = got.sol.retcode == SciMLBase.ReturnCode.Success && all(isfinite, φ)
        return ok ? relerr(φ, φ_ref) : NaN
    end

    err = trial(Δt_default)
    in_band = isfinite(err) && err ≤ BAND
    @printf("    %-6s Δt = %5.3f  rel err = %-9.4g %s  (reference own error %.4g)\n", label, Δt_default,
            err, in_band ? "in band" : "OUT OF BAND", ref_err)
    in_band && return (Δt_default, err, ref_err)

    println("    ", label, ": Δt = ", Δt_default, " out of band -- falling back through ",
            LV_FAMILY_LADDER)
    best = nothing
    for Δt in LV_FAMILY_LADDER
        Δt == Δt_default && continue
        e = trial(Δt)
        e_in_band = isfinite(e) && e ≤ BAND
        @printf("    %-6s Δt = %5.3f  rel err = %-9.4g %s\n", label, Δt, e, e_in_band ? "in band" : "")
        e_in_band && (best === nothing || Δt > best[1]) && (best = (Δt, e))
    end
    best === nothing && error(
        "$label: no point in $LV_FAMILY_LADDER lands inside BAND = $BAND against its own reference " *
        "(reference own error $ref_err).",
    )
    return (best[1], best[2], ref_err)
end

"""
RKC1 (the shipped default) vs RKL1 vs RKG1 on the LV emRKC arm, both outer and inner set to the same
family. Certifies each family's Δt on the coarse mesh (`lv_family_band_check`), then times host and
device arms at that Δt on the timed mesh. Opt-in via `EMRKC_LV_STAGES=families`.
"""
function lv_family_comparison(cfg::LVConfig)
    println("\n-- STS family comparison: coarse-mesh band check --")
    cmesh, cms = lv_geometry(cfg.coarse_base)
    cform64 = lv_form(Float64, cmesh, cms)
    cu64    = lv_u0(cform64, Float64)
    cform32 = lv_form(Float32, cmesh, cms)
    cu32    = lv_u0(cform32, Float32)
    φc64, φc32 = solution_variable(cform64, :φₘ), solution_variable(cform32, :φₘ)

    families = (("RKC1", RKC1(0.05)), ("RKL1", RKL1()), ("RKG1", RKG1()))
    certified = NamedTuple[]
    for (label, fam) in families
        Δt, err, ref_err =
            lv_family_band_check(cform64, cu64, cform32, cu32, φc64, φc32, fam, label, cfg.Δt_emrkc)
        push!(certified, (; label, fam, Δt, err, ref_err))
    end
    cmesh = cms = cform64 = cform32 = cu64 = cu32 = nothing
    GC.gc()

    println("\n-- STS family comparison: timed mesh --")
    t0 = time()
    mesh, ms = lv_geometry(cfg.base)
    ncells, nnodes = Ferrite.getncells(mesh.grid), Ferrite.getnnodes(mesh.grid)
    form = lv_form(Float32, mesh, ms)
    u32  = lv_u0(form, Float32)
    ugpu = CuVector(u32)
    φₘ32 = solution_variable(form, :φₘ)
    @printf("  %d hexahedra, %d nodes, %d states, setup %.0f s\n",
            ncells, nnodes, Thunderbolt.solution_size(form), time() - t0)

    arms = Arm[]
    for (; label, fam, Δt) in certified
        alg_h = emrkc(Vector{Float32}, ThreadedSparseMatrixCSR{Float32, Int32}; outer = fam, inner = fam)
        ah, φh = run_arm("host emRKC $label", "emRKC $label", form, u32, alg_h, Float32(Δt), φₘ32,
                          nothing, false, false; steps = LV_STEPS)
        push!(arms, ah)
        @printf("  host RSS after %s host arm: %.2f GiB\n", label, host_rss_gib())

        alg_d = emrkc(CuVector{Float32}, CuCSR; outer = fam, inner = fam)
        ad, _ = run_arm("device emRKC $label", "emRKC $label", form, ugpu, alg_d, Float32(Δt), φₘ32,
                         φh, true, false; steps = LV_STEPS)
        push!(arms, ad)
        @printf("  device memory after %s device arm: %.2f GiB of %.2f GiB\n",
                label, gpu_used_gib(), CUDA.total_memory() / 1024^3)
    end

    report(arms)
    println("\nband errors (own Float64 reference per family, coarse mesh):")
    for (; label, Δt, err, ref_err) in certified
        @printf("  %-6s Δt = %.4f  rel err = %.4g  (reference own error %.4g)\n", label, Δt, err, ref_err)
    end
    return arms, certified
end

function lv_time_arms(cfg::LVConfig)
    println("\n-- timed mesh --")
    t0 = time()
    mesh, ms = lv_geometry(cfg.base)
    ncells, nnodes = Ferrite.getncells(mesh.grid), Ferrite.getnnodes(mesh.grid)
    form  = lv_form(Float32, mesh, ms)
    formc = lv_form(Float32, mesh, ms; mass = ConsistentMass())
    u32  = lv_u0(form, Float32)
    φₘ32 = solution_variable(form, :φₘ)
    @printf("  %d hexahedra, %d nodes, %d states, setup %.0f s\n",
            ncells, nnodes, Thunderbolt.solution_size(form), time() - t0)

    cpu_emrkc = emrkc(Vector{Float32}, ThreadedSparseMatrixCSR{Float32, Int32})
    cpu_split = splitting(Vector{Float32}, ThreadedSparseMatrixCSR{Float32, Int32})
    gpu_emrkc = emrkc(CuVector{Float32}, CuCSR)
    gpu_split = splitting(CuVector{Float32}, CuCSR)
    ugpu = CuVector(u32)

    println("\nCG iterations per step, host, Δt = ", cfg.Δt_split, " ms:")
    plain = build(formc, u32,
                  splitting(Vector{Float32}, ThreadedSparseMatrixCSR{Float32, Int32}; jacobi = false),
                  Float32(cfg.Δt_split), 1.0f5)
    # Past the assembly and the zero initial guess, but not `prewarm!`: this probe counts iterations
    # rather than timing them, and a wall-clock warmup's thousands of steps drift the Float32 time far
    # enough apart that the splitting integrator's parent/child synchronization gives up.
    for _ = 1:5
        step!(plain)
    end
    @printf("  unpreconditioned %.1f\n", mean_cg_iters!(plain, NSTEPS))
    plain = nothing
    GC.gc()

    arms = Arm[]
    # The host arms are only checked for finiteness here -- their step sizes are certified on the
    # coarse mesh -- and each device arm against its own host counterpart at the same step size.
    ahe, φ_host_emrkc = run_arm("host emRKC", "emRKC", form, u32, cpu_emrkc, Float32(cfg.Δt_emrkc),
                                φₘ32, nothing, false, false; steps = LV_STEPS)
    push!(arms, ahe)
    ahs, φ_host_split = run_arm("host splitting", "splitting", formc, u32, cpu_split, Float32(cfg.Δt_split),
                                φₘ32, nothing, false, true; steps = LV_STEPS)
    push!(arms, ahs)
    @printf("\nhost peak RSS %.2f GiB of the 8 GiB cap (%.0f%% headroom)\n",
            host_rss_gib(), 100(1 - host_rss_gib() / 8))
    for (label, method, f, alg, Δt, φ_ref, has_solve) in (
        ("device emRKC", "emRKC", form, gpu_emrkc, Float32(cfg.Δt_emrkc), φ_host_emrkc, false),
        ("device splitting", "splitting", formc, gpu_split, Float32(cfg.Δt_split), φ_host_split, true),
    )
        arm, _ = run_arm(label, method, f, ugpu, alg, Δt, φₘ32, φ_ref, true, has_solve; steps = LV_STEPS)
        push!(arms, arm)
        @printf("%-18s device memory in use %.2f GiB of %.2f GiB (%.0f%% headroom)\n",
                label, gpu_used_gib(), CUDA.total_memory() / 1024^3,
                100(1 - gpu_used_gib() / (CUDA.total_memory() / 1024^3)))
    end

    report(arms)
    println("\nhost-vs-device agreement is over ", LV_STEPS,
            " steps in Float32 on both sides. emRKC's step size is certified by the coarse sweep; ",
            "splitting's is a Richardson\nestimate, because its own reference does not converge at ",
            "DTREF -- see the LV-PCG2019 header block.")
    return arms
end

####################################

function run_model(cfg::ModelConfig)
    println("\n", "#"^118)
    println("# ", cfg.name, "  (", N, " x ", N, ", L = ", cfg.L, " mm, device Float32)")
    println("#"^118)

    form64  = ep01_form(Float64, cfg)
    form64c = ep01_form(Float64, cfg; mass = ConsistentMass())
    u64     = ep01_u0(form64, Float64, cfg)
    φₘ64    = solution_variable(form64, :φₘ)

    println(cfg.name, ": ", Thunderbolt.solution_size(form64), " states, t ∈ [0, ", TEND, "] ms")
    @printf("reference Δt = %.4g ms, accuracy band = %.3g\n", DTREF, BAND)

    cpu64_emrkc = emrkc(Vector{Float64}, ThreadedSparseMatrixCSR{Float64, Int64})
    # The reference's conjugate gradient runs far tighter than the tutorial's: at the tutorial's
    # tolerances the solve's own error accumulates with the step count, so the backward Euler error
    # curve turns around below Δt ≈ 0.2 ms and a reference stepped there would be *less* accurate than
    # the step sizes it certifies. That floor belongs in the arms, not in the reference.
    cpu64_split = splitting(
        Vector{Float64}, ThreadedSparseMatrixCSR{Float64, Int64}; atol = 1.0e-12, rtol = 1.0e-10,
    )
    φ_ref_emrkc, err_emrkc = reference(form64, u64, cpu64_emrkc, φₘ64)
    φ_ref_split, err_split = reference(form64c, u64, cpu64_split, φₘ64)
    @printf("  emRKC reference error ≈ %.3g ; splitting reference error ≈ %.3g\n", err_emrkc, err_split)
    @printf("  the two references differ by %.4g -- the mass lumping, not a step size\n",
            relerr(φ_ref_emrkc, φ_ref_split))

    println("\nstep size selection (Float32, host):")
    form32  = ep01_form(Float32, cfg)
    form32c = ep01_form(Float32, cfg; mass = ConsistentMass())
    u32     = ep01_u0(form32, Float32, cfg)
    φₘ32    = solution_variable(form32, :φₘ)
    cpu32_emrkc = emrkc(Vector{Float32}, ThreadedSparseMatrixCSR{Float32, Int32})
    cpu32_split = splitting(Vector{Float32}, ThreadedSparseMatrixCSR{Float32, Int32})
    Δt_emrkc, _ = select_dt(form32, u32, cpu32_emrkc, φₘ32, φ_ref_emrkc, "emRKC")
    Δt_split, _ = select_dt(form32c, u32, cpu32_split, φₘ32, φ_ref_split, "splitting")

    DT = Float32
    form_dev, u_dev, φₘ_dev = form32, u32, φₘ32
    gpu_emrkc = emrkc(CuVector{DT}, CuCSR)
    gpu_split = splitting(CuVector{DT}, CuCSR)
    ugpu = CuVector(u_dev)

    # What the preconditioner is worth: the same splitting arm at the same step size with `precs` left
    # at its default identity.
    println("\nCG iterations per step, host, Δt = ", Δt_split, " ms:")
    plain = build(form32c, u32, splitting(Vector{Float32}, ThreadedSparseMatrixCSR{Float32, Int32}; jacobi = false),
                  Float32(Δt_split), 1.0f5)
    # Past the assembly and the zero initial guess, but not `prewarm!`: this probe counts iterations
    # rather than timing them, and a wall-clock warmup's thousands of steps drift the Float32 time far
    # enough apart that the splitting integrator's parent/child synchronization gives up.
    for _ = 1:5
        step!(plain)
    end
    @printf("  unpreconditioned %.1f\n", mean_cg_iters!(plain, NSTEPS))
    plain = nothing
    GC.gc()

    arms = Arm[]
    for (label, method, form, u0, alg, Δt, φₘ, φ_ref, dev, has_solve) in (
        ("host emRKC", "emRKC", form32, u32, cpu32_emrkc, Float32(Δt_emrkc), φₘ32, φ_ref_emrkc, false, false),
        ("host splitting", "splitting", form32c, u32, cpu32_split, Float32(Δt_split), φₘ32, φ_ref_split, false, true),
        ("device emRKC", "emRKC", form_dev, ugpu, gpu_emrkc, DT(Δt_emrkc), φₘ_dev, φ_ref_emrkc, true, false),
        ("device splitting", "splitting", form32c, ugpu, gpu_split, DT(Δt_split), φₘ_dev, φ_ref_split, true, true),
    )
        arm, _ = run_arm(label, method, form, u0, alg, Δt, φₘ, φ_ref, dev, has_solve)
        push!(arms, arm)
    end

    report(arms)
    return arms
end

# The element counts are fitted to the 8 GiB host cgroup and the 8 GiB card, and are what a smaller
# card would have to turn down.
_lv_base(key, default) = Tuple(parse.(Int, split(get(ENV, key, default), ",")))

# Every model this file can run, by the name `EMRKC_MODELS` selects them with. Defined here rather
# than beside the two sheet configs because the LV entry needs the whole LV section above it, down to
# its own step sizes -- measured on the coarse mesh, and the sheet's do not transfer.
const AVAILABLE_MODELS = Dict{String, Any}(
    "FHN" => FHN_CONFIG,
    "PCG2019" => PCG2019_CONFIG,
    "LV" => LVConfig(
        "LV-PCG2019",
        _lv_base("EMRKC_LV_BASE", "191,20,33"),
        _lv_base("EMRKC_LV_COARSE_BASE", "96,10,17"),
        0.05, 0.0068,
    ),
)

function main()
    _assert_memory_capped("benchmarks/benchmark-emrkc.jl")
    CUDA.functional() || error("This benchmark needs a functional CUDA device.")
    println("models: ", join(MODEL_CONFIGS, ", "),
            "  (host: 2 threads capped, ThreadedSparseMatrixCSR SpMV threaded, pointwise sweeps single-threaded)")
    for name in MODEL_CONFIGS
        try
            run_model(AVAILABLE_MODELS[name])
        catch err
            println("\n", name, " FAILED: ", sprint(showerror, err))
        end
        GC.gc()
        CUDA.reclaim()
    end
    return nothing
end

# Only when this file is what was run: `include`ing it from a REPL session loads the definitions
# without starting an hour of measurement.
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
