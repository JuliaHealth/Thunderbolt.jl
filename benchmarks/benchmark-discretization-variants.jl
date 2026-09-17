# Four spatial discretizations of the same monodomain step, compared at MATCHED ACCURACY rather than
# at matched h: CG-P1 with a row-sum lumped mass (the production baseline), SIPG DG at orders 1 and 2
# over a fused matrix-free `M⁻¹K`, and CG-Q2 on Gauss-Lobatto nodes whose mass is diagonal by nodal
# quadrature. Time integration is emRKC (RKC1 defaults, `gates = :all`) in every arm, so the only
# thing that varies is the space.
#
# TWO CONCLUSIONS, KEPT APART. The cost table below answers ONE question: on a uniform mesh, at equal
# accuracy, what does a simulated millisecond cost in each space. It does not answer whether DG is
# worth building — the reason DG is in the codebase is local time stepping, where the case is
# structural (an element's stencil is its own cell plus its face neighbours, so a locally refined
# region can be stepped at its own rate without touching the rest) and is not measured anywhere in
# this file. A uniform-mesh cost verdict against DG is not a verdict against DG-for-LTS.
#
# THE SLAB. A planar front along x on a hex slab, `LX` long and `NTRANS` cells square in cross
# section, with zero-flux boundaries everywhere. The transverse extent tracks h so the cells stay
# cubic; the front is planar and the IC is a function of x alone, so the transverse direction carries
# no dynamics and exists only to make the operator three dimensional. An apex-shaped geometry would
# have contaminated a spatial convergence study with its slivers, which is why this is a slab and not
# the LV of `benchmark-emrkc.jl`.
#
# CONDUCTIVITY IS ISOTROPIC, AND THAT IS A CONSTRAINT, NOT A CHOICE. `SIPGDiffusionIntegrator` takes
# a constant SCALAR diffusivity — `D::Float64`, no coefficient protocol, no tensor — so a
# transversely isotropic tensor cannot be spelled on the DG arms at all. Every arm therefore runs the
# same isotropic `D_ISO`, the physiological fiber value. For a planar front along the fiber the
# transverse conductivity is dynamically irrelevant, so the FRONT this measures is the one a
# transversely isotropic slab would carry; what changes is ρ_F, which on an isotropic mesh tracks the
# TRACE of D — about a factor of two higher than a 5:1 transversely isotropic tissue would give, and
# by that same factor in all four arms, so the ratios below survive it and the absolute `s / sim ms`
# is conservative. Making it a genuine tensor is a FerriteOperators-side seam (the kernels need
# `jump ⋅ (D · avg)` and the penalty an `nᵀDn` factor); it is not reachable from here.
#
# THE REFERENCE IS INDEPENDENT OF ALL FOUR ARMS: CG-Q2 with a CONSISTENT mass, stepped by
# `LieTrotterGodunov(BackwardEulerSolver + tight Jacobi-CG, AdaptiveForwardEulerSubstepper)`. None of
# the measured arms shares its space, its mass treatment or its time integrator, so no arm gets its
# own discretization error laid under the others as an unreachable floor — the failure mode the
# self-referenced arms of `benchmark-emrkc.jl` exist to avoid. Its own error is quantified by
# self-convergence in BOTH directions — `H_REF` against `2 H_REF` and `DT_REF` against `2 DT_REF` —
# and both numbers are printed beside the ladders they certify.
#
# THE METRICS ARE TWO, AND THE SECOND IS THE DECISION METRIC. (1) A space-time relative L2 band error
# over `SNAPSHOTS`, sampled on a fixed physical centerline by `Ferrite.PointEvalHandler`, so spaces
# with different dof sets are compared as FUNCTIONS and not as coefficient vectors. (2) The arrival
# time of the front at `XPROBE`, which is a conduction velocity error. Mass lumping is a CV bias and
# an L2 band at one instant can hide a phase error that a front study cannot afford to hide, so both
# columns are reported for every point of every ladder.
#
# THE STEP SIZE IS HELD FIXED AND SMALL (`DT_LADDER`) across the accuracy ladder, so what the ladder
# measures is the SPACE. Each variant's own certified Δt at its `h*` is established separately,
# afterwards, and is what the cost table times at. emRKC's inner stage count `m` is sized against
# `η ρ_F`, so the certification is re-run per variant rather than carried: the SIPG penalty raises
# ρ_F by two orders of magnitude over CG-P1 at the same h, and that is part of DG's honest cost.
#
# RUN: CUDA is a weak dependency, so this runs in the GPU test environment, which also carries
# `FerriteOperatorsExampleElements` (the home of `SIPGDiffusionIntegrator`). A memory-capped cgroup is
# required (`_assert_memory_capped()`; `BENCHMARK_UNCAPPED=1` overrides). Canonical invocation:
# `systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 env JULIA_NUM_THREADS=2 julia -t2
# --heap-size-hint=3G --project=test/gpu benchmarks/benchmark-discretization-variants.jl`
# `DV_STAGES` selects the stages, comma separated, default `"validate,ladder,penalty,cost"`, and
# `DV_REF_CACHE` names a directory the reference solve is cached in, keyed by the constants it
# depends on -- a re-run that only varies the arms then costs nothing for it.
#
# THREADS: the host arms run on the threads the invocation asks for, and BLAS is pinned to the same
# count -- left alone, OpenBLAS sizes itself against the whole machine and a `-t2` run was measured
# spreading over sixteen cores. Device arms are Float32 throughout.
#
# This is a benchmark, not a CI gate: the numbers are reported as measured, whichever way they fall.
#
# RESULTS (capped 2-thread host profile; slab 10 mm x 3 cells, D = 0.4 mm²/ms isotropic, PCG2019,
# t ∈ [0, 7] ms, band = 1e-2 space-time relative L2 against the independent reference).
#
# REFERENCE: CG-Q2 consistent mass at h = 0.03125 mm, Δt = 0.000625 ms, 31409 slab dofs; arrival at
# x = 8 mm is 5.95683 ms. Its own error: band 0.003065 in h (against 0.0625 mm), 0.002152 in Δt
# (against 0.00125 ms) -- both a factor 3-5 under the band they certify.
#
# ACCURACY LADDERS (Δt = 0.00125 ms throughout; `h` is the CELL size, `sp` the dof spacing):
#   CG-P1 lumped     h 0.5000  sp 0.0312*    band 0.2186  0.04339  0.01448  0.01295  0.00825
#                    ndofs      336/656/1296/2576/5136        CV +0.510 +0.090 -0.007 -0.017 -0.013
#   DG-P1 SIPG       h 0.5000 .. 0.03125     band 0.3154  0.1514   0.05991  0.02483  0.01123
#                    ndofs     1440/2880/5760/11520/23040    CV -0.505 -0.248 -0.096 -0.039 -0.018
#   DG-P2 SIPG       h 1.0000 .. 0.0625      band 0.1758  0.0740   0.02764  0.01292  0.006693
#                    ndofs     2430/4860/9720/19440/38880    CV -0.313 -0.123 -0.043 -0.020 -0.011
#   CG-Q2 GLL-SEM    h 1.0000 .. 0.0625      band 0.08333 0.02272  0.02176  0.01251  0.006683
#                    ndofs     1029/2009/3969/7889/15729     CV +0.145 -0.030 -0.033 -0.019 -0.011
# (CV = arrival shift in ms at x = 8 mm against the reference. Observed band order: CG-P1
# 2.33/1.58/0.16/0.65 -- the last decade is floor limited by the reference's own 0.003; DG-P1
# 1.06/1.34/1.27/1.14; DG-P2 1.25/1.42/1.10/0.95; CG-Q2 1.87/0.06/0.80/0.90.)
#
# h* AT THE 1e-2 BAND -- AND THE HEADLINE: EVERY VARIANT NEEDS THE SAME DOF SPACING.
#   CG-P1 lumped    h* = 0.03125 mm cell,  dof spacing 0.03125,  band 0.00825,   CV -0.0126 ms
#   DG-P1 SIPG      never crossed; at its finest mesh (h = 0.03125, dof spacing 0.03125) it is at
#                   0.01123, so its true h* lies just below and every DG-P1 cost below is a LOWER
#                   bound. Its band order is a clean 1.1-1.3, giving h* ≈ 0.028 mm.
#   DG-P2 SIPG      h* = 0.0625 mm cell,   dof spacing 0.03125,  band 0.006693,  CV -0.0111 ms
#   CG-Q2 GLL-SEM   h* = 0.0625 mm cell,   dof spacing 0.03125,  band 0.006683,  CV -0.0111 ms
# The monodomain upstroke is a RESOLUTION-limited feature, not a smoothness-limited one, so raising
# the polynomial order buys no dof spacing here: both order-2 spaces reach the band at exactly the
# spacing CG-P1 does. At that matched spacing the dof count per unit tissue is therefore fixed by the
# per-cell multiplicity alone -- CG-P1 1x, CG-Q2 1x, DG-P2 3.4x, DG-P1 8x -- before any per-dof cost.
#
# THE LUMPING BIAS IS NOT WHAT LIMITS CG-P1. Its CV shift at h* is -0.013 ms, the same order as every
# other arm's, and it is already at -0.007 ms at h = 0.125 mm -- better than DG-P1 manages at four
# times the resolution. Row-sum lumping on P1 was the suspected decision metric; on this front it is
# not the binding error.
#
# CERTIFIED Δt AT h*: 0.0078125 ms for ALL FOUR (time-only band 0.0069 / 0.0080 / 0.0078 / 0.0076).
# The outer step is REACTION limited and identical across the four spaces, exactly as expected. What
# the SIPG penalty buys is INNER stages: at Δt = 0.25 ms neither DG arm can be staged at all
# (`max_stages = 200`; they need 223 and 217), where both continuous arms step happily.
#
# ρ_F AT EQUAL CELL SIZE (h = 0.5 mm, with the resulting inner stage count at Δt = 0.00125 ms):
#   CG-P1 lumped 7.005 (m=1) | CG-Q2 GLL-SEM 58.85 (m=1) | DG-P1 SIPG 1443 (m=3) | DG-P2 SIPG 5458 (m=6)
# DG-P1's 206x over CG-P1 at the SAME dof spacing is the whole of its extra cost, and `m` grows as
# its square root.
#
# THE SIPG PENALTY IS FREE ACCURACY-WISE AND EXPENSIVE COST-WISE (DG-P1, h = 0.125 mm):
#   η     1.0      2.0       4.0       8.0
#   ρ_F   5219     11180     23090     46910
#   m     2        3         4         6
#   band  0.05992  0.05991   0.05991   0.05990      arrival 5.86023 / 5.86038 / 5.86045 / 5.86048
# Eight-fold in η moves ρ_F nine-fold and `m` three-fold and the error NOT AT ALL, to four figures.
# `SIPG_ETA = 4` is FerriteOperators' test value, not a tuned one: dropping it to the coercivity
# boundary halves the inner sweep for nothing. But even at η = 1, DG-P1's ρ_F is 47x CG-P1's at the
# same dof spacing, so most of the gap is NOT the penalty -- it is the exact block-diagonal inverse
# mass, whose Q1 hex element spectrum spans a factor 27 that lumping removes.
#
# THE COST TABLE IS NOT YET FILLED IN. The four accuracy ladders, the four certified step sizes and
# the penalty sweep above are measured; the timed arms on the 2 mm cube at each variant's h* were
# still running when this was written. What the numbers above already fix is the shape of that table:
# a common Δt, a common dof SPACING, and therefore a per-simulated-ms cost that is the dof
# multiplicity (1 / 1 / 3.4 / 8) times the inner stage count (which grows as the square root of a ρ_F
# spanning 7.0 to 5458 at equal cell size). Fill the table in from a completed run rather than from
# that arithmetic.

using Thunderbolt
using FerriteOperators
using FerriteOperatorsExampleElements
using CUDA
using LinearSolve
using OrdinaryDiffEqOperatorSplitting
using LinearAlgebra
using Printf
using StaticArrays

import Ferrite
import Serialization
import SparseArrays: nonzeros
import SparseMatricesCSR: getrowptr, getcolval, getnzval
import Thunderbolt: SciMLBase, ThreadedSparseMatrixCSR
import OrdinaryDiffEqOperatorSplitting as OS

const FOE = FerriteOperatorsExampleElements

####################################
## The problem
####################################

const LX        = 10.0      # mm, slab length; the front transits it in ~18 ms
const NTRANS    = 3         # cells across; the front is planar, this only has to exceed one
const D_ISO     = 0.4       # mm²/ms, the monodomain diffusivity of `benchmark-emrkc.jl`'s sheet
const SIPG_ETA  = 4.0       # SIPG penalty safety factor over the (p+1)(p+d)/d constant
# The measured conduction velocity is 1.18 mm/ms, so the front leaves the S1 region at t ~ 0.9 ms and
# reaches the far face at t ~ 7.6 ms. EVERY snapshot has to sit inside that transit: past it the slab
# is uniformly plateaued and two arms differ only in ionic drift, which makes the band error a
# measure of the REACTION's time error and not of the space. A first cut of this file sampled
# 4/8/12/16/20 ms and measured exactly that -- the band plateaued at 0.025 for CG-P1 at every h from
# 0.125 down, and ordered at -1.7, 0.15, 0.23.
const TEND      = 7.0       # ms, one clean transit and no plateau tail
const XPROBE    = 8.0       # mm, the distal cross section the arrival time is read at
const XTHRESH   = -40.0     # mV, the activation threshold
const SNAPSHOTS = (2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
const NSAMPLE   = 1601      # centerline sample points; ~5 per cell at the finest h studied
# `DT_LADDER` has to put the arms' OWN time error well under `BAND`, or the ladder measures that
# error instead of the space and every arm plateaus at the same floor. At Δt = 0.01 ms it did: all
# four arms bottomed out at a band of 0.019-0.024 and an arrival 0.035 ms early, independently of h.
const DT_LADDER = 0.00125   # ms, held fixed across the accuracy ladder so it measures the space
const BAND      = 1.0e-2    # the space-time band error a mesh has to reach
# The reference's binding error is its TIME error, not its space error: at h = 0.03125 its
# self-convergence in h is already 1.3e-3, while backward Euler at Δt = 0.005 self-converged at only
# 1.2e-2 -- above the band it is meant to certify. The budget therefore buys steps, not cells.
const H_REF     = 0.03125   # mm
const DT_REF    = 0.000625  # ms

const CuCSR = CUDA.CUSPARSE.CuSparseMatrixCSR{Float32, Int32}

slab(h) = generate_mesh(
    Hexahedron, (round(Int, LX / h), NTRANS, NTRANS),
    Vec{3}((0.0, 0.0, 0.0)), Vec{3}((LX, NTRANS * h, NTRANS * h)),
)

cube(h, side) = generate_mesh(
    Hexahedron, ntuple(_ -> round(Int, side / h), 3),
    Vec{3}((0.0, 0.0, 0.0)), Vec{3}((side, side, side)),
)

function monodomain(::Type{T}, mesh) where {T}
    return MonodomainModel(
        ConstantCoefficient(T(1.0)),   # χ
        ConstantCoefficient(T(1.0)),   # Cₘ
        ConstantCoefficient(SymmetricTensor{2, 3, T}((T(D_ISO), 0, 0, T(D_ISO), 0, T(D_ISO)))),
        NoStimulationProtocol(),
        Thunderbolt.ParametrizedPCG2019Model{T}(),
        CartesianCoordinateSystem(mesh),
        :φₘ, :s,
    )
end

"""
An S1 front written as an initial condition: the proximal `0.1 LX` raised above threshold, the rest at
PCG2019's resting default. Same shape, and for the same reason, as the sheet and LV protocols of
`benchmark-emrkc.jl` — there is no applied-current amplitude to tune against excitability, and no
source operator is needed, which matters here because the fused DG rate operator has no route for
one (see `FusedInverseMassRateOperator`).
"""
function s1_initial_condition(form, ::Type{T}) where {T}
    u₀ = create_initial_condition(form, T)
    setvariable!(u₀, form, :φₘ) do x
        x[1] ≤ T(0.1LX) ? T(20.0) : T(-85.0)
    end
    return u₀
end

####################################
## The four variants
####################################

"""
A continuous-Lagrange arm. `gll = true` swaps the MASS quadrature for the collocated nodal rule —
the Gauss-Lobatto points a tensor-product Lagrange space already has its dofs on — which makes the
assembled mass diagonal, so emRKC's row-sum lumping reproduces it exactly and the arm is a spectral
element method rather than a lumped one. The STIFFNESS keeps the standard Gauss rule either way.
"""
cg_form(::Type{T}, mesh, order; gll = false) where {T} = semidiscretize(
    ReactionDiffusionSplit(monodomain(T, mesh)),
    FiniteElementDiscretization(
        Dict(:φₘ => LagrangeCollection{order}());
        qrcs = gll ?
            Dict{Symbol, Any}(
                :φₘ   => QuadratureRuleCollection(T, order + 1),
                :mass => NodalQuadratureRuleCollection(LagrangeCollection{order}()),
            ) :
            Dict{Symbol, Any}(:φₘ => QuadratureRuleCollection(T, order + 1)),
    ),
    mesh,
)

"""
A discontinuous-Lagrange arm: Thunderbolt's own monodomain semidiscretization over a
`DiscontinuousLagrange` space — which needs no change to reach, every index set and the per-dof
coordinate evaluation being dof-count driven rather than node driven — with the diffusion term
replaced by SIPG and the assembly strategy electing the fused `M⁻¹K` block-row store.

Two spellings here are not free choices. The FACET quadrature rule stays `Float64` whatever `T` is:
Ferrite 1.7 defines the here→there reference point mapping every `InterfaceValues` reinit performs
for `Vec{dim, Float64}` only, so a `Float32` facet rule has no method at all; the element's value
type and the whole action still follow `T`. And the mass the fusion inverts is
`SimpleBilinearMassIntegrator`, not Thunderbolt's own: the fill queries it with no time context, and
`BilinearMassIntegrator` reads one.
"""
function dg_form(::Type{T}, mesh, order, device; η = SIPG_ETA) where {T}
    strategy = dg_strategy(T, order, device)
    f = semidiscretize(
        ReactionDiffusionSplit(monodomain(T, mesh)),
        FiniteElementDiscretization(
            Dict(:φₘ => DiscontinuousLagrangeCollection{order}());
            qrcs = Dict{Symbol, Any}(:φₘ => QuadratureRuleCollection(T, order + 1)),
            assembly_strategy = strategy,
        ),
        mesh,
    )
    return _swap_in_sipg(f, T, order, strategy, η)
end

dg_mass(::Type{T}, order) where {T} =
    FOE.SimpleBilinearMassIntegrator(1.0, QuadratureRuleCollection(T, order + 1), :φₘ)

dg_strategy(::Type{T}, order, device) where {T} = AssemblyStrategy(
    MatrixFreeAction(;
        element_mapping = dg_element_mapping(device),
        storage = BlockRowAssembly(; premultiply_inverse_mass = dg_mass(T, order)),
    ),
    ColoredScheduling(), device,
)

# `LanesPerElement` on a device, `WorkerPerElement` on the host: FerriteOperators' own
# `benchmarks/dg_action.jl` measures the lane mapping as the faster of the two on CUDA (1.26x
# cuSPARSE at hex p = 1), and timing DG's device arm on the slower mapping would flatter the
# alternatives it is being compared against.
dg_element_mapping(::FerriteOperators.AbstractGPUDevice) = LanesPerElement()
dg_element_mapping(::Any) = WorkerPerElement()

function _swap_in_sipg(f, ::Type{T}, order, strategy, η) where {T}
    heat, ode = f.functions
    sipg = FOE.SIPGDiffusionIntegrator(
        D_ISO, η, QuadratureRuleCollection(T, order + 1),
        FerriteOperators.FacetQuadratureRuleCollection(Float64, order + 1), :φₘ,
    )
    heat_dg = Thunderbolt.AffineODEFunction(
        heat.mass_term, sipg, heat.source_term, heat.dh, strategy,
    )
    return OS.GenericSplitFunction((heat_dg, ode), f.solution_indices)
end

"`(label, builder, cells-per-dof-spacing, h ladder)`. `h` is the CELL size; the dof spacing is
`h / order`, which is why the two order-2 arms start their ladder a step coarser."
const VARIANTS = (
    ("CG-P1 lumped",  (T, m, d) -> cg_form(T, m, 1),             1,
     (0.5, 0.25, 0.125, 0.0625, 0.03125)),
    ("DG-P1 SIPG",    (T, m, d) -> dg_form(T, m, 1, d),          1,
     (0.5, 0.25, 0.125, 0.0625, 0.03125)),
    ("DG-P2 SIPG",    (T, m, d) -> dg_form(T, m, 2, d),          2,
     (1.0, 0.5, 0.25, 0.125, 0.0625)),
    ("CG-Q2 GLL-SEM", (T, m, d) -> cg_form(T, m, 2; gll = true), 2,
     (1.0, 0.5, 0.25, 0.125, 0.0625)),
)

"""
The assembly device an arm runs on. Only the DG arms read it -- their action IS the strategy's, so
the strategy has to name the device the action runs on -- while the continuous arms cross to a device
through the solver's `solution_vector_type` and a mirrored host assembly, as in `benchmark-emrkc.jl`.
"""
host_device() = SequentialCPUDevice()
gpu_device(::Type{T}) where {T} = FerriteOperators.KernelAbstractionsDevice(
    CUDA.CUDABackend(); value_type = T, index_type = Int32,
)

####################################
## Stepping
####################################

emrkc(::Type{VT}, ::Type{MT}) where {VT, MT} =
    EMRKC(solution_vector_type = VT, system_matrix_type = MT, gates = :all)

"""
The reciprocal main diagonal of the backward Euler operator `M - Δt K`, as a left preconditioner for
the reference's conjugate gradient. Filled on first `ldiv!` rather than at `init`: `LinearSolve.init`
calls the `precs` callback while the system matrix is still the freshly allocated all-zero sparsity
pattern, and the affine backward Euler path then fills that same object in place without ever
reassigning `cache.A`. The reference runs at a fixed `Δt`, so the one-shot fill stays valid.
"""
mutable struct JacobiPrecon{MatType, VecType}
    A::MatType
    inv_diag::VecType
    ready::Bool
end

function JacobiPrecon(A)
    d = similar(nonzeros(A), size(A, 1))
    fill!(d, one(eltype(d)))
    return JacobiPrecon(A, d, false)
end

# `ThreadedSparseMatrixCSR` has neither a `diag` method nor a scalar `getindex`, so each row is walked
# for its own column index.
function _fill_inv_diag!(d, A::ThreadedSparseMatrixCSR)
    rowptr, colval, nzval = getrowptr(A), getcolval(A), getnzval(A)
    @inbounds for i in eachindex(d)
        v = zero(eltype(d))
        for k in rowptr[i]:(rowptr[i + 1] - 1)
            colval[k] == i && (v = nzval[k])
        end
        d[i] = iszero(v) ? one(v) : inv(v)
    end
    return d
end

function _fill_inv_diag!(d, A)
    v = diag(A)
    d .= ifelse.(iszero.(v), one(eltype(d)), inv.(v))
    return d
end

function LinearAlgebra.ldiv!(y::AbstractVector, P::JacobiPrecon, x::AbstractVector)
    if !P.ready
        _fill_inv_diag!(P.inv_diag, P.A)
        P.ready = true
    end
    y .= P.inv_diag .* x
    return y
end
LinearAlgebra.ldiv!(P::JacobiPrecon, x::AbstractVector) = ldiv!(x, P, x)

# `KrylovJL`'s own contract: `(A, p) -> (Pl, Pr)`. CG takes left/centered preconditioning only.
jacobi_precs(A, p = nothing) = (JacobiPrecon(A), LinearAlgebra.I)

"""
The reference's own integrator: consistent mass throughout, and neither the space, the mass treatment
nor the time integrator of any measured arm. Its conjugate gradient runs far tighter than a
production setting so that the reference does not measure the linear solve's own accumulated error,
and carries a Jacobi preconditioner so that tolerance is affordable on the reference mesh.
"""
reference_alg(::Type{VT}, ::Type{MT}) where {VT, MT} = LieTrotterGodunov((
    BackwardEulerSolver(
        solution_vector_type = VT,
        system_matrix_type   = MT,
        inner_solver         = KrylovJL_CG(atol = 1.0e-12, rtol = 1.0e-10, precs = jacobi_precs),
    ),
    AdaptiveForwardEulerSubstepper(
        solution_vector_type = VT, reaction_threshold = 0.1, substeps = 10,
    ),
))

build(form, u0, alg, Δt, tend) =
    init(OperatorSplittingProblem(form, copy(u0), (zero(Δt), tend)), alg; dt = Δt, verbose = false)

sync(::Vector) = nothing
sync(::CuVector) = CUDA.synchronize()

####################################
## Sampling: comparing spaces as functions, not as coefficient vectors
####################################

"""
The centerline of the slab, as physical points -- the same x positions for every variant, so two arms
are compared as FUNCTIONS on a shared lattice rather than through their coefficient vectors. Inset
from the end faces by a hair so that no sample sits exactly on a boundary facet, where point location
is a tie.
"""
function centerline(h)
    ε = 1.0e-6LX
    return [
        Vec{3, Float64}((x, 0.5NTRANS * h, 0.5NTRANS * h)) for
        x in range(ε, LX - ε; length = NSAMPLE)
    ]
end

"""
φₘ at the centerline sample points. The evaluation goes through the FIELD interpolation, so a DG arm
is read inside the cell that owns the point rather than at a dof, and a Q2 arm is read at its true
quadratic value rather than at the nearest node.
"""
struct CenterlineSampler{PH, PP, DH, VAR}
    ph::PH        # the whole centerline, read at the snapshots
    probe::PP     # the single station `XPROBE`, read at every step until the front arrives
    dh::DH
    var::VAR
end

function CenterlineSampler(form, h)
    dh = form.functions[1].dh
    grid = Ferrite.get_grid(dh)
    line = centerline(h)
    # The sample line sits at the centre of the cross section, so no point lands on a facet where a
    # DG field is two-valued.
    return CenterlineSampler(
        Ferrite.PointEvalHandler(grid, line; warn = false),
        Ferrite.PointEvalHandler(
            grid, [line[argmin(i -> abs(line[i][1] - XPROBE), eachindex(line))]]; warn = false),
        dh, solution_variable(form, :φₘ),
    )
end

_eval(ph, s::CenterlineSampler, u) = Ferrite.evaluate_at_points(
    ph, s.dh, getvariable(Vector(u), s.var), :φₘ,
)

function sample(s::CenterlineSampler, u)
    vals = _eval(s.ph, s, u)
    any(v -> v === nothing || !isfinite(v), vals) &&
        error("the centerline sampler produced a missing or non-finite value")
    return Float64[v for v in vals]
end

"""φₘ at `XPROBE` alone -- what the per-step arrival watch reads, so that watch costs one point
rather than `NSAMPLE` of them."""
probe_value(s::CenterlineSampler, u) = Float64(only(_eval(s.probe, s, u)))

"""
Step to each of `SNAPSHOTS` in turn, sampling the centerline at every one, and read the arrival time
at `XPROBE` off the same run by watching the sample nearest that station cross `XTHRESH`. One solve
per (variant, h), not one per metric.
"""
function transit(form, u0, alg, Δt, sampler)
    integrator = build(form, u0, alg, Δt, oftype(Δt, TEND))
    snaps = Vector{Vector{Float64}}()
    arrival = nothing
    tprev, vprev = 0.0, probe_value(sampler, integrator.u)
    vprev ≥ XTHRESH && (arrival = 0.0)
    for tsnap in SNAPSHOTS
        while integrator.t < tsnap - 1.0e-9
            step!(integrator)
            isfinite(sum(integrator.u)) || return nothing, nothing
            if arrival === nothing
                v = probe_value(sampler, integrator.u)
                v ≥ XTHRESH &&
                    (arrival = tprev + (XTHRESH - vprev) / (v - vprev) * (integrator.t - tprev))
                tprev, vprev = integrator.t, v
            end
        end
        # Every `Δt` in use divides the snapshot spacing, so the snapshots of two arms are taken at
        # the same instant and their band error is a distance and not a phase artifact. A step size
        # that did not divide it would overshoot silently, which is what this refuses.
        abs(integrator.t - tsnap) < 1.0e-9 || error(
            "Δt = $Δt overshot the snapshot at $tsnap ms (landed at $(integrator.t)): every step " *
            "size has to divide the snapshot spacing.",
        )
        push!(snaps, sample(sampler, integrator.u))
    end
    return snaps, arrival
end

"Space-time relative L2 over the snapshot set, on the shared sample lattice."
function band_error(snaps, ref)
    num = sum(sum(abs2, a .- b) for (a, b) in zip(snaps, ref))
    den = sum(sum(abs2, b) for b in ref)
    return sqrt(num / den)
end

####################################
## Machine discipline
####################################

"""
The effective `memory.max` (bytes) of this process's cgroup v2 leaf, found by walking
`/proc/self/cgroup`'s `0::<path>` up through `/sys/fs/cgroup<path>`. `nothing` for "max" (unset) or
when no such file is found.
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
machine rather than the 8G this benchmark is meant to run in. `BENCHMARK_UNCAPPED=1` overrides.
"""
function _assert_memory_capped()
    get(ENV, "BENCHMARK_UNCAPPED", "0") == "1" && return nothing
    limit = _cgroup_memory_limit()
    capped = limit !== nothing && limit ≤ 12 * 1024^3
    capped || error(
        "No memory-capped cgroup detected (effective memory.max = $(limit === nothing ? "unset" : limit) " *
        "bytes). An uncapped run's GC sizes itself against the whole machine. Run:\n" *
        "  systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 env JULIA_NUM_THREADS=2 " *
        "julia -t2 --heap-size-hint=3G --project=test/gpu benchmarks/benchmark-discretization-variants.jl\n" *
        "or set BENCHMARK_UNCAPPED=1 to run uncapped deliberately.",
    )
    Base.JLOptions().heap_size_hint == 0 &&
        println("WARNING: no --heap-size-hint given -- GC growth is unbounded even inside the cgroup.")
    return nothing
end

const STAGES = Set(strip(s) for s in split(
    get(ENV, "DV_STAGES", "validate,ladder,penalty,cost"), ","))

####################################
## Stage: validation
####################################

"""
What has to hold before any number below means anything.

  * the fused DG rate operator IS `-M⁻¹K` for an independently assembled SIPG matrix and DG mass,
    including the MINUS — FerriteOperators' SIPG form is the POSITIVE `+∫D∇u·∇v`, opposite to
    `BilinearDiffusionIntegrator`'s, and a sign error there integrates the diffusion backwards;
  * the GLL arm's assembled mass is diagonal and positive, which is what makes emRKC's row-sum
    lumping of it exact rather than an approximation;
  * every arm elects the rate operator its strategy says it should, and reaches a finite front.
"""
function validate()
    println("\n", "="^118)
    println("VALIDATION")
    println("="^118)

    hval = 0.5
    mesh = slab(hval)

    println("\nthe fused DG rate against an independently assembled SIPG matrix and DG mass:")
    for order in (1, 2)
        f  = dg_form(Float64, mesh, order, host_device())
        dh = f.functions[1].dh
        n  = ndofs(dh)
        K = let op = setup_operator(
                AssemblyStrategy(SequentialCPUDevice();
                    form = FullAssembly(StandardOperatorSpecification(;
                        sparsity_entries = FOE.interior_facet_entries!))),
                FOE.SIPGDiffusionIntegrator(
                    D_ISO, SIPG_ETA, QuadratureRuleCollection(order + 1),
                    FerriteOperators.FacetQuadratureRuleCollection(order + 1), :φₘ),
                dh)
            update_operator!(op, nothing); op.A
        end
        M = let op = setup_operator(AssemblyStrategy(SequentialCPUDevice()),
                FOE.SimpleBilinearMassIntegrator(1.0, QuadratureRuleCollection(order + 1), :φₘ), dh)
            update_operator!(op, nothing); op.A
        end
        itg = build(f, s1_initial_condition(f, Float64),
                    emrkc(Vector{Float64}, ThreadedSparseMatrixCSR{Float64, Int64}), 0.01, 1.0e5)
        x = Float64[sin(0.37i) + 0.3cos(1.1i) for i = 1:n]
        y = zeros(n)
        Thunderbolt.mul_rate!(y, itg.cache.op, x)
        ref = -(Matrix(M) \ (K * x))
        @printf("  DG-P%d (%d dofs): ‖f_F(x) + M⁻¹Kx‖/‖M⁻¹Kx‖ = %.3e ; xᵀf_F(x) = %+.4g (must be < 0)\n",
                order, n, norm(y - ref) / norm(ref), dot(x, y))
        isapprox(y, ref; rtol = 1.0e-9) || error("DG-P$order: the fused rate is not -M⁻¹K")
        dot(x, y) < 0 || error("DG-P$order: the rate operator is not dissipative -- sign error")
    end

    println("\nthe GLL mass, assembled through the model's own path:")
    let f = cg_form(Float64, mesh, 2; gll = true), heat = f.functions[1]
        spec = Thunderbolt._OperatorSetupSpec(
            Vector{Float64}, ThreadedSparseMatrixCSR{Float64, Int64})
        M = let op = setup_operator(get_strategy(heat), heat.mass_term, spec, heat.dh)
            update_operator!(op, nothing, Thunderbolt.TimeIntegrationContext(0.0, 0.0, 0.0))
            Matrix(FerriteOperators.get_matrix(op))
        end
        d, vol = diag(M), LX * (NTRANS * hval)^2
        off = maximum(abs, M - Diagonal(d)) / minimum(d)
        @printf("  CG-Q2 GLL (%d dofs): max|offdiag|/min|diag| = %.3e, min diag = %.6g, Σdiag/volume = %.15g\n",
                size(M, 1), off, minimum(d), sum(d) / vol)
        off < 1.0e-12 || error("the GLL mass is not diagonal")
        minimum(d) > 0 || error("the GLL mass has a non-positive diagonal entry")
        isapprox(sum(d), vol; rtol = 1.0e-12) || error("the GLL mass does not integrate the volume")

        Mg = let g = cg_form(Float64, mesh, 2), hg = g.functions[1]
            op = setup_operator(get_strategy(hg), hg.mass_term, spec, hg.dh)
            update_operator!(op, nothing, Thunderbolt.TimeIntegrationContext(0.0, 0.0, 0.0))
            Matrix(FerriteOperators.get_matrix(op))
        end
        @printf("  the same space under the default Gauss rule: max|offdiag|/min|diag| = %.3e\n",
                maximum(abs, Mg - Diagonal(diag(Mg))) / minimum(diag(Mg)))
    end

    println("\nevery arm builds, elects its rate operator and reaches a finite front:")
    for (name, mk, _, _) in VARIANTS
        f  = mk(Float64, mesh, host_device())
        u0 = s1_initial_condition(f, Float64)
        itg = build(f, u0, emrkc(Vector{Float64}, ThreadedSparseMatrixCSR{Float64, Int64}),
                    DT_LADDER, 4.0)
        solve!(itg)
        s, _, m = Thunderbolt._emrkc_step_sizing(itg.alg, DT_LADDER, itg.cache.ρS, itg.cache.ρF)
        φ = getvariable(Vector(itg.u), solution_variable(f, :φₘ))
        @printf("  %-15s %-28s ndofs=%-8d ρF=%-11.4g s=%-3d m=%-4d finite=%s\n",
                name, nameof(typeof(itg.cache.op)), ndofs(f.functions[1].dh),
                itg.cache.ρF, s, m, all(isfinite, φ))
        all(isfinite, φ) || error("$name: the validation solve is not finite")
    end
    return nothing
end

####################################
## Stage: the accuracy ladder
####################################

# `snaps` is kept so the step-size certification below references the arm against the very solve the
# ladder already paid for -- at `h*` that solve is 24 minutes for DG-P2, and repeating it buys
# nothing.
struct LadderPoint
    h::Float64
    ndofs::Int
    band::Float64
    arrival::Float64
    snaps::Vector{Vector{Float64}}
end

"""
The reference, and the estimate of its own error in BOTH directions: the same discretization once at
`2 H_REF` and once at `2 DT_REF`, which for a first order time integrator and a second order space
bound the two halves of that error up to a small factor. A reference whose own error is not well
below `BAND` cannot certify a mesh at `BAND`, so both numbers are printed beside the ladders they
certify and a warning is raised when either is not.

Returns the snapshot set, the arrival time, and the larger of the two self-convergence numbers.
"""
function reference_solution()
    println("\n", "="^118)
    println("REFERENCE  (CG-Q2 consistent mass, backward Euler + adaptive FE substepper)")
    println("="^118)
    cache = _reference_cache_path()
    if cache !== nothing && isfile(cache)
        out = Serialization.deserialize(cache)
        @printf("  reloaded from %s: arrival %.5f ms, own error %.4g\n", cache, out[2], out[3])
        flush(stdout)
        return out
    end
    alg = reference_alg(Vector{Float64}, ThreadedSparseMatrixCSR{Float64, Int64})
    solve_ref(h, Δt) = begin
        f = cg_form(Float64, slab(h), 2)
        t0 = time_ns()
        snaps, arrival = transit(f, s1_initial_condition(f, Float64), alg, Δt,
                                 CenterlineSampler(f, h))
        snaps === nothing && error("the reference at (h, Δt) = ($h, $Δt) is not finite")
        arrival === nothing && error(
            "the reference front never reached x = $XPROBE mm within $TEND ms; the window is too " *
            "short or the probe station too distal.")
        @printf("  h = %.6f mm, %8d dofs, Δt = %.5g ms : arrival at x = %.1f mm is %.5f ms  (%.1f s)\n",
                h, ndofs(f.functions[1].dh), Δt, XPROBE, arrival, (time_ns() - t0) / 1.0e9)
        GC.gc()
        flush(stdout)
        (snaps, arrival)
    end
    coarse_h  = solve_ref(2H_REF, DT_REF)
    coarse_dt = solve_ref(H_REF, 2DT_REF)
    fine      = solve_ref(H_REF, DT_REF)

    own_h  = band_error(coarse_h[1], fine[1])
    own_dt = band_error(coarse_dt[1], fine[1])
    @printf("  self-convergence in h  (%.6f vs %.6f mm): band %.4g, arrival shift %+.5f ms\n",
            H_REF, 2H_REF, own_h, fine[2] - coarse_h[2])
    @printf("  self-convergence in Δt (%.5g vs %.5g ms):  band %.4g, arrival shift %+.5f ms\n",
            DT_REF, 2DT_REF, own_dt, fine[2] - coarse_dt[2])
    own = max(own_h, own_dt)
    own < BAND / 3 || @printf(
        "  WARNING: the reference's own error is %.4g, not comfortably below BAND = %.4g\n", own, BAND)
    out = (fine[1], fine[2], own)
    cache === nothing || Serialization.serialize(cache, out)
    return out
end

"""
Where the reference is cached, or `nothing`. The reference is the single most expensive thing in this
file -- three solves on the finest mesh, ~20 minutes -- and it depends on nothing the arms vary, so a
run that only re-measures the arms should not repay it. `DV_REF_CACHE` names a DIRECTORY; the file in
it is keyed by every constant the reference's own value depends on, so a changed window, mesh, step
size or sample lattice writes a different file rather than silently reusing a stale one.
"""
_reference_cache_path() = _cache_path("reference", (H_REF, DT_REF))

"""
Where a cached stage lives, or `nothing`. The reference and the ladders are the expensive parts of
this file and neither depends on anything the stages after them vary, so a run that only re-measures
the cost should not repay them. `DV_REF_CACHE` names a DIRECTORY; each file in it is keyed by every
constant its own value depends on, so a changed window, mesh, step size, band or sample lattice
writes a different file rather than silently reusing a stale one.
"""
function _cache_path(what, extra)
    dir = get(ENV, "DV_REF_CACHE", "")
    isempty(dir) && return nothing
    isdir(dir) || mkpath(dir)
    key = hash((LX, NTRANS, D_ISO, TEND, XPROBE, XTHRESH, SNAPSHOTS, NSAMPLE, extra))
    return joinpath(dir, "dv-$what-" * string(key, base = 16) * ".jls")
end

function ladder(ref_snaps, ref_arrival)
    cache = _cache_path("ladder", (BAND, DT_LADDER, SIPG_ETA, map(v -> (v[1], v[4]), VARIANTS)))
    if cache !== nothing && isfile(cache)
        out = Serialization.deserialize(cache)
        println("\n", "="^118)
        println("ACCURACY LADDERS  (reloaded from ", cache, ")")
        println("="^118)
        for (name, _, per, _) in VARIANTS
            for pt in out[name]
                @printf("  %-15s h = %-9.4f dof sp. %-9.4f ndofs %-9d band %-11.4g arrival %.5f\n",
                        name, pt.h, pt.h / per, pt.ndofs, pt.band, pt.arrival)
            end
        end
        flush(stdout)
        return out
    end
    println("\n", "="^118)
    println("ACCURACY LADDERS  (Δt = ", DT_LADDER, " ms throughout, so this measures the SPACE)")
    println("="^118)
    alg = emrkc(Vector{Float64}, ThreadedSparseMatrixCSR{Float64, Int64})
    results = Dict{String, Vector{LadderPoint}}()
    for (name, mk, per, hs) in VARIANTS
        println("\n", name)
        @printf("  %-9s %-9s %9s %12s %14s %12s\n",
                "h/mm", "dof sp.", "ndofs", "band err", "arrival/ms", "CV shift/ms")
        pts = LadderPoint[]
        # Coarse to fine, stopping at the FIRST point inside the band: `h*` is the coarsest mesh that
        # reaches the target, so a finer one adds cost and no answer. The printed ladder is therefore
        # the crossing and what led up to it, not a full sweep.
        for h in hs
            mesh = slab(h)
            f = mk(Float64, mesh, host_device())
            n = ndofs(f.functions[1].dh)
            t0 = time_ns()
            snaps, arrival = transit(f, s1_initial_condition(f, Float64), alg, DT_LADDER,
                                     CenterlineSampler(f, h))
            if snaps === nothing
                @printf("  %-9.4f %-9.4f %9d %12s %14s %12s\n", h, h / per, n, "NaN", "-", "-")
                GC.gc(); continue
            end
            b = band_error(snaps, ref_snaps)
            arrival === nothing && (arrival = NaN)
            push!(pts, LadderPoint(h, n, b, arrival, snaps))
            @printf("  %-9.4f %-9.4f %9d %12.4g %14.5f %+12.5f   %s  (%.0f s)\n",
                    h, h / per, n, b, arrival, arrival - ref_arrival,
                    b ≤ BAND ? "in band" : "", (time_ns() - t0) / 1.0e9)
            GC.gc()
            flush(stdout)
            b ≤ BAND && break
        end
        results[name] = pts
        cache === nothing || Serialization.serialize(cache, results)
        if length(pts) ≥ 2
            rates = [log2(pts[i].band / pts[i + 1].band) for i = 1:(length(pts) - 1)]
            @printf("  observed band-error order: %s\n",
                    join((@sprintf("%.2f", r) for r in rates), ", "))
        end
    end
    return results
end

"""
What the SIPG penalty factor is worth, at one fixed mesh. `η` is the user's safety factor over the
`(p+1)(p+d)/d` coercivity constant, and it is a first-order knob on BOTH halves of DG's case: a
larger penalty stiffens the interface, which raises ρ_F and with it the inner stage count, and adds
numerical diffusion at the front, which moves the arrival time. `SIPG_ETA = 4` is the value
FerriteOperators' own tests carry, not a value anything here tuned, so this column says what the
choice costs. A penalty at or below the coercivity constant is not stable, which is what the sweep's
low end is there to show rather than assert.
"""
function penalty_sweep(ref_snaps, ref_arrival; h = 0.125, order = 1)
    println("\n", "="^118)
    println("SIPG PENALTY SENSITIVITY  (DG-P", order, " at h = ", h, " mm, Δt = ", DT_LADDER, " ms)")
    println("="^118)
    @printf("  %-7s %11s %7s %12s %14s %12s\n",
            "η", "ρF", "m", "band err", "arrival/ms", "CV shift/ms")
    mesh = slab(h)
    alg = emrkc(Vector{Float64}, ThreadedSparseMatrixCSR{Float64, Int64})
    for η in (1.0, 2.0, 4.0, 8.0)
        f = dg_form(Float64, mesh, order, host_device(); η)
        u0 = s1_initial_condition(f, Float64)
        probe = build(f, u0, alg, DT_LADDER, oftype(DT_LADDER, 1.0e5))
        step!(probe)
        s_, _, m = Thunderbolt._emrkc_step_sizing(probe.alg, DT_LADDER, probe.cache.ρS, probe.cache.ρF)
        ρF = probe.cache.ρF
        probe = nothing
        snaps, arrival = transit(f, u0, alg, DT_LADDER, CenterlineSampler(f, h))
        if snaps === nothing
            @printf("  %-7.1f %11.4g %7d %12s %14s %12s\n", η, ρF, m, "NON-FINITE", "-", "-")
        else
            a = something(arrival, NaN)
            @printf("  %-7.1f %11.4g %7d %12.4g %14.5f %+12.5f\n",
                    η, ρF, m, band_error(snaps, ref_snaps), a, a - ref_arrival)
        end
        GC.gc()
        flush(stdout)
    end
    return nothing
end

"""
The coarsest mesh of the ladder whose band error is inside `BAND` -- the variant's `h*` -- as
`(point, certified)`. Where the ladder never crossed, the FINEST mesh it reached is returned with
`certified = false`: that mesh is coarser than the true `h*` and therefore cheaper, so the cost row
built on it is a LOWER bound on what the arm costs at matched accuracy, which is the honest way to
carry an arm whose crossing lies past the ladder rather than dropping it from the table.
"""
function h_star(pts)
    isempty(pts) && return nothing
    inband = filter(p -> p.band ≤ BAND, pts)
    isempty(inband) && return (argmin(p -> p.h, pts), false)
    return (argmax(p -> p.h, inband), true)
end

####################################
## Stage: the certified step size and the cost table
####################################

# Every entry divides the `SNAPSHOTS` spacing of 1 ms, which `transit` asserts. The low end is what
# a 1e-2 band against a CONVERGED reference actually demands: a first cut stopped at 0.015625 ms and
# CG-P1 was still at 0.0154 there.
const DT_SWEEP  = (0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625)
const SIDE_CANDIDATES = (8.0, 6.0, 4.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5)
const FOOTPRINT_BUDGET = 2.0 * 1024^3   # bytes, the host footprint one arm may take of the 8G cgroup
const MIN_CELLS_PER_SIDE = 8            # below this a timed arm is latency bound, not throughput bound

"""
A coarse host-footprint model of one arm on a cube of edge `side` at cell size `h`: the state buffers
emRKC keeps at full width, plus either the two assembled sparse matrices (continuous arms) or the
block-row store (DG arms, `(1+Nf)·Nb²` scalars per cell). Good to a factor of two, which is all the
cube sizing below needs it to be.
"""
function footprint(name, h, side)
    ncells = (side / h)^3
    nper   = 8                              # bytes; the ladders and the host cost arms are Float64
    nstate = 10                             # PCG2019
    buffers = 6                             # emRKC's full-width buffers, generously counted
    if startswith(name, "DG")
        nb    = name == "DG-P1 SIPG" ? 8 : 27
        ndofs = nb * ncells
        return ncells * 7 * nb^2 * nper + ndofs * nstate * buffers * nper
    end
    per   = name == "CG-P1 lumped" ? 1 : 2
    ndofs = (per * side / h + 1)^3
    nnz   = per == 1 ? 27 : 125
    return 2 * ndofs * nnz * (nper + 8) + ndofs * nstate * buffers * nper
end

"""
The one physical cube every arm is timed on: the largest candidate edge whose worst arm stays inside
`FOOTPRINT_BUDGET`. A common physical domain is the point of the exercise — a finer `h*` costing more
dofs over the same tissue IS the cost result — so the edge is shared and not tuned per arm. Arms
whose cube then falls below `MIN_CELLS_PER_SIDE` are reported as such: at that size a timed step is
latency bound rather than throughput bound, and its `s / sim ms` is an upper bound on what the same
discretization would cost on a mesh that saturates the machine.
"""
function choose_cost_side(stars)
    hs = [(name, p.h) for (name, _, _, _) in VARIANTS for p in (stars[name],) if p !== nothing]
    isempty(hs) && return nothing
    for side in SIDE_CANDIDATES
        maximum(footprint(n, h, side) for (n, h) in hs) ≤ FOOTPRINT_BUDGET && return side
    end
    return last(SIDE_CANDIDATES)
end
const NSTEPS         = 20
const NPASS          = 3
const WARMUP_SECONDS = 1.5

"""
The largest step size in `DT_SWEEP` whose solution at `h*` stays inside `BAND` of the SAME variant at
`h*` and `DT_LADDER` -- the ladder's own solve there, handed in rather than repeated.

Referencing the arm against ITSELF in time is deliberate: the space error at `h*` is already at the
band by construction, and what a step size controls is the time error alone. Mixing the two would
certify a step size against an error it cannot reduce.

Returns `(Δt, certified)`. Where no step size in the sweep reaches the band the finest one tried is
returned with `certified = false`: a step size too large for the band makes the arm CHEAPER than it
has any right to be, so the cost row it produces is a LOWER bound and is labelled as one.
"""
function certify_dt(name, mk, point)
    h = point.h
    f = mk(Float64, slab(h), host_device())
    u0 = s1_initial_condition(f, Float64)
    sampler = CenterlineSampler(f, h)
    alg = emrkc(Vector{Float64}, ThreadedSparseMatrixCSR{Float64, Int64})
    base, base_arrival = point.snaps, point.arrival
    best = nothing
    for Δt in DT_SWEEP
        # A stiffness measure emRKC cannot stage under `max_stages` is a RESULT of this sweep, not a
        # crash: it is what a ρ_F two orders above CG-P1's buys at a step size CG-P1 steps happily.
        snaps, arrival = try
            transit(f, u0, alg, Δt, sampler)
        catch err
            err isa ErrorException && occursin("inner stages", err.msg) || rethrow()
            @printf("    Δt = %8.6f   UNSTAGEABLE: %s\n", Δt,
                    first(split(err.msg, ". ")))
            flush(stdout)
            continue
        end
        if snaps === nothing
            @printf("    Δt = %8.6f   NON-FINITE\n", Δt)
            flush(stdout)
            continue
        end
        err = band_error(snaps, base)
        @printf("    Δt = %8.6f   time-only band err = %-10.4g arrival shift = %+.5f ms %s\n",
                Δt, err, something(arrival, NaN) - base_arrival, err ≤ BAND ? "in band" : "")
        flush(stdout)
        err ≤ BAND && best === nothing && (best = Δt)
    end
    best === nothing && return (last(DT_SWEEP), false)
    return (best, true)
end

function gpu_clocks()
    out = read(`nvidia-smi --query-gpu=clocks.sm,clocks.mem --format=csv,noheader,nounits`, String)
    sm, mem = parse.(Int, strip.(split(first(split(strip(out), '\n')), ',')))
    return sm, mem
end

"""
Step without interruption for `WARMUP_SECONDS`, reading the clocks back *while still stepping*. On a
card that idles at 300 MHz a short warmup measures the ramp rather than the kernel, and the returned
clocks are what says this one did not.
"""
function prewarm!(integrator, on_device)
    t0 = time_ns()
    clocks, sampled = (0, 0), false
    while (time_ns() - t0) / 1.0e9 < WARMUP_SECONDS
        step!(integrator)
        if on_device && !sampled && (time_ns() - t0) / 1.0e9 > WARMUP_SECONDS / 2
            clocks, sampled = gpu_clocks(), true
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

struct CostRow
    label::String
    h::Float64
    Δt::Float64
    ndofs::Int
    nstates::Int
    stages::String
    seconds_per_step::Float64
    arrival_shift::Float64
    band::Float64
    clocks::Tuple{Int, Int}
end

"""
The timed arm, on the `COST_SIDE` cube at the variant's own `h*` and certified `Δt`. The cube is a
different geometry from the slab the accuracy was established on, deliberately: `h*` is a property of
the discretization and transfers, while a three-cell-thick slab would not exercise a device at all.
"""
function cost_arm(label, mk, h, Δt, side, ::Type{VT}, ::Type{MT}, on_device, band, shift) where {VT, MT}
    T = eltype(VT)
    mesh = cube(h, side)
    f = mk(T, mesh, on_device ? gpu_device(T) : host_device())
    u0 = s1_initial_condition(f, T)
    u0 = VT <: CuVector ? CuVector(u0) : u0
    itg = build(f, u0, emrkc(VT, MT), T(Δt), T(1.0e5))
    clocks = prewarm!(itg, on_device)  # also where an unstageable Δt surfaces, loudly
    # After the first accepted step, not before it: the two spectral radii the stage counts are sized
    # against are estimated inside the step, and are still zero at `init`.
    s, _, m = Thunderbolt._emrkc_step_sizing(itg.alg, T(Δt), itg.cache.ρS, itg.cache.ρF)
    seconds = measure!(itg)
    φ = getvariable(Vector(itg.u), solution_variable(f, :φₘ))
    all(isfinite, φ) || error("$label: the timed arm went non-finite")
    row = CostRow(label, h, Δt, ndofs(f.functions[1].dh), length(u0), "s=$s m=$m",
                  seconds, shift, band, clocks)
    f = nothing; itg = nothing; u0 = nothing
    GC.gc()
    on_device && CUDA.reclaim()
    return row
end

function cost_table(rows)
    println("\nThe `ndofs`/`states`/`s per step` columns are the CUBE's; `slab band` and the CV bias")
    println("below are the slab study's, at the same h*.")
    println("\n", "="^134)
    @printf("%-24s %8s %8s %10s %11s %10s %11s %13s %10s %9s\n",
            "arm", "h*/mm", "Δt/ms", "ndofs", "states", "stages", "s/step", "s / sim ms",
            "slab band", "clocks")
    println("-"^134)
    for r in rows
        @printf("%-24s %8.4f %8.4f %10d %11d %10s %11.5f %13.5f %10.4g %9s\n",
                r.label, r.h, r.Δt, r.ndofs, r.nstates, r.stages, r.seconds_per_step,
                r.seconds_per_step / r.Δt, r.band,
                r.clocks == (0, 0) ? "host" : @sprintf("%d/%d", r.clocks[1], r.clocks[2]))
    end
    println("-"^134)
    println("\nCV bias at h* (arrival-time shift against the independent consistent-mass reference):")
    for r in rows
        @printf("  %-24s %+.5f ms at x = %.1f mm\n", r.label, r.arrival_shift, XPROBE)
    end
    for kind in ("host", "device")
        base = findfirst(r -> r.label == "$kind CG-P1 lumped", rows)
        base === nothing && continue
        b = rows[base].seconds_per_step / rows[base].Δt
        println("\nrelative to the $kind CG-P1 baseline, per simulated ms:")
        for r in rows
            startswith(r.label, kind) || continue
            @printf("  %-24s %6.2fx\n", r.label, (r.seconds_per_step / r.Δt) / b)
        end
    end
    return nothing
end

####################################

# A row whose `h` or `Δt` was not certified is cheaper than the arm's true matched-accuracy cost, so
# it carries a mark rather than being dropped or quoted as if it were certified.
_bound_mark(certified, dt_ok, name) =
    (get(certified, name, true) && get(dt_ok, name, true)) ? "" : " (>=)"

function main()
    _assert_memory_capped()
    # The host profile is the threads the invocation asks for. OpenBLAS otherwise sizes itself
    # against the whole machine -- measured at 1657% CPU under `-t2` before this line existed -- and a
    # level-1 kernel spread over sixteen cores is not the profile this table claims to measure.
    LinearAlgebra.BLAS.set_num_threads(Threads.nthreads())
    println("stages: ", join(sort(collect(STAGES)), ", "),
            "  (", Threads.nthreads(), " Julia threads, BLAS pinned to the same)")
    println("slab: ", LX, " mm x ", NTRANS, " cells square, D = ", D_ISO,
            " mm²/ms isotropic, PCG2019, t ∈ [0, ", TEND, "] ms")

    "validate" in STAGES && validate()

    ("ladder" in STAGES || "cost" in STAGES) || return nothing

    ref_snaps, ref_arrival, ref_own = reference_solution()
    results = ladder(ref_snaps, ref_arrival)
    "penalty" in STAGES && penalty_sweep(ref_snaps, ref_arrival)

    println("\n", "="^118)
    println("MATCHED ACCURACY  (band = ", BAND, ", reference own error ",
            @sprintf("%.4g", ref_own), ")")
    println("="^118)
    stars = Dict{String, Any}()
    certified = Dict{String, Bool}()
    for (name, _, per, _) in VARIANTS
        got = h_star(results[name])
        if got === nothing
            stars[name] = nothing
            @printf("  %-15s no usable ladder point\n", name)
            continue
        end
        p, ok = got
        stars[name], certified[name] = p, ok
        @printf("  %-15s h%s = %.4f mm (dof spacing %.4f), %d dofs on the slab, band %.4g, CV shift %+.5f ms%s\n",
                name, ok ? "*" : " ", p.h, p.h / per, p.ndofs, p.band, p.arrival - ref_arrival,
                ok ? "" : "   <-- NEVER CROSSED the band; this is the finest mesh tried, so every " *
                          "cost below is a LOWER bound")
    end

    "cost" in STAGES || return nothing

    println("\n", "="^118)
    println("CERTIFIED STEP SIZES AT h*")
    println("="^118)
    dts = Dict{String, Float64}()
    dt_ok = Dict{String, Bool}()
    for (name, mk, _, _) in VARIANTS
        p = stars[name]
        p === nothing && continue
        println("\n  ", name, " at h = ", p.h, " mm")
        dts[name], dt_ok[name] = certify_dt(name, mk, p)
        @printf("    %s Δt = %.6f ms%s\n", dt_ok[name] ? "certified" : "UNCERTIFIED", dts[name],
                dt_ok[name] ? "" : " (the finest tried; no step size in the sweep reached the band)")
        flush(stdout)
        GC.gc()
    end

    side = something(tryparse(Float64, get(ENV, "DV_COST_SIDE", "")), choose_cost_side(stars))
    println("\n", "="^118)
    println("COST AT MATCHED ACCURACY  (", side, " mm cube, device arms Float32)")
    println("="^118)
    for (name, _, _, _) in VARIANTS
        p = stars[name]
        p === nothing && continue
        n = round(Int, side / p.h)
        @printf("  %-15s %4d cells per side (%d cells), estimated host footprint %.2f GiB%s\n",
                name, n, n^3, footprint(name, p.h, side) / 1024^3,
                n < MIN_CELLS_PER_SIDE ? "   <-- LATENCY BOUND, an upper bound on its true cost" : "")
    end
    rows = CostRow[]
    for (name, mk, _, _) in VARIANTS
        p = stars[name]
        (p === nothing || !haskey(dts, name)) && continue
        shift = p.arrival - ref_arrival
        try
            push!(rows, cost_arm("host $name$(_bound_mark(certified, dt_ok, name))",
                                 mk, p.h, dts[name], side,
                                 Vector{Float64}, ThreadedSparseMatrixCSR{Float64, Int64},
                                 false, p.band, shift))
        catch err
            println("  host ", name, " FAILED: ", first(split(sprint(showerror, err), '\n')))
        end
    end
    if CUDA.functional()
        for (name, mk, _, _) in VARIANTS
            p = stars[name]
            (p === nothing || !haskey(dts, name)) && continue
            shift = p.arrival - ref_arrival
            try
                push!(rows, cost_arm("device $name$(_bound_mark(certified, dt_ok, name))",
                                     mk, p.h, dts[name], side,
                                     CuVector{Float32}, CuCSR, true, p.band, shift))
            catch err
                println("  device ", name, " FAILED: ",
                        first(split(sprint(showerror, err), '\n')))
            end
        end
        @printf("\ndevice memory: %.2f GiB of %.2f GiB in use after the arms\n",
                (CUDA.total_memory() - CUDA.available_memory()) / 1024^3,
                CUDA.total_memory() / 1024^3)
    end
    @printf("host peak RSS: %.2f GiB\n", Sys.maxrss() / 1024^3)
    cost_table(rows)
    return nothing
end

main()
