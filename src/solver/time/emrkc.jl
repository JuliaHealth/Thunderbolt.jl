#####################################################################
#  emRKC: exponential multirate super-time-stepping                 #
#####################################################################

"""
    EMRKCDivergence(msg)

A stiffness measure [`EMRKC`](@ref) cannot size a sweep against: a runaway spectral radius estimate,
or a stage count past `max_stages` or past `Int` itself.

Its own exception type rather than a plain `error`, because the step has to tell the two apart: a
state that diverged mid-run is a step failure the integrator may retry from, while every other error
raised on the same path -- an unusable option, a bug in an estimate -- must stay loud. The three
sites that raise it are `_rho_runaway_error`, `_sts_safe_ceil` and `_emrkc_stage_count_error`.
"""
struct EMRKCDivergence <: Exception
    msg::String
end

Base.showerror(io::IO, e::EMRKCDivergence) = print(io, e.msg)

"""
    PassiveChildSolver()

A leaf timestepper that advances its clock and leaves its state untouched.

[`EMRKC`](@ref) integrates both operators of a reaction-diffusion split in one monolithic sweep on
the parent solution vector, but still runs the Lie-Trotter-Godunov child loop afterwards so that the
operator splitting invariants keep holding: child clocks equal to the parent's (asserted by
`OrdinaryDiffEqOperatorSplitting`'s `validate_time_point` on every accepted step), child state
synchronized from the parent, rollback anchors intact.
"""
struct PassiveChildSolver <: AbstractSolver end

SciMLBase.isadaptive(::PassiveChildSolver) = false
OrdinaryDiffEqCore.default_controller(QT, ::PassiveChildSolver) =
    OrdinaryDiffEqCore.DummyController()

# The two buffers every `ThunderboltTimeIntegrator` reads off its cache, and nothing else.
struct PassiveChildCache{SolutionType, PrevSolutionType} <: AbstractTimeSolverCache
    uₙ::SolutionType
    uₙ₋₁::PrevSolutionType
end

function setup_solver_cache(f, solver::PassiveChildSolver, t₀; u = nothing, uprev = nothing)
    (u === nothing || uprev === nothing) && error(
        "A `PassiveChildSolver` owns no state: it can only be built as the child of a splitting " *
        "algorithm that hands it the parent's solution slice, not through a standalone `init`.",
    )
    return PassiveChildCache(u, uprev)
end

perform_step!(f, cache::PassiveChildCache, t, Δt) = true

# The `setup_operator(strategy, integrator, solver, dh)` family dispatches on `::AbstractSolver` and
# reads exactly these two fields (`src/solver/interface.jl`). An operator splitting *algorithm* --
# not an `AbstractSolver`, owning no solver cache -- reaches that assembly path, device and mirrored
# branches included, by handing one of these over in a solver's place.
# TODO(OperatorStorageRequest): what the family should dispatch on is a storage request naming the
# two types, not a solver; that refactor deletes this type. Internal to this file until then.
struct _OperatorSetupSpec{SolutionVectorType, SystemMatrixType} <: AbstractSolver
    solution_vector_type::Type{SolutionVectorType}
    system_matrix_type::Type{SystemMatrixType}
end

"""
    EMRKC(; outer, inner, gates, rho_recompute, rho_safety, ...)

Exponential multirate super-time-stepping for a reaction-diffusion split, following Algorithm 3 of
[Rosilho de Souza, Grote, Pezzuto & Krause, arXiv:2401.01745]. First order, explicit, fixed step.

One step evaluates an *averaged force* through a stabilized inner sweep and drives an outer
[`AbstractSTSFamily`](@ref) sweep with it, so the stiff diffusion never dictates the outer step
size. States declared by `gating_symbols` are integrated exponentially and removed from the slow
force, so the outer stage count follows the *remaining* reaction stiffness instead of the gates'.

Per step, with `Δt` the outer step, `ρ_S` the slow (reaction) and `ρ_F` the fast (diffusion)
spectral radius: `s` stages of `outer` resolve `Δt ρ_S`, the averaging window is
`η = 2Δt / ℓ_outer(s)` ([`sts_stability_boundary`](@ref)), and `m` stages of `inner` resolve `η ρ_F`.

The fast force is the rate form `M⁻¹K` of the split's mass and diffusion terms
(`FerriteOperators.RateFormIntegrator`), so the mass must be invertible cell by cell: on a
continuous space that is a [`LumpedMass`](@ref) or [`CollocatedMass`](@ref) discretization, on a
discontinuous space the consistent mass serves. The sign of `K` is the diffusion term's
[`rate_sign`](@ref).

# Fields
- `outer`, `inner`: the STS families of the two sweeps ([`RKC1`](@ref) by default).
- `gates`: `:all`, or a `Tuple` of `gating_symbols` to integrate exponentially. `()` is the
  degenerate mRKC-without-exponential mode, which runs *any* cell model, including one that declares
  no gates at all.
- `solution_vector_type`, `system_matrix_type`: what the mass, diffusion and source operators are
  assembled into, as for [`BackwardEulerSolver`](@ref).
- `rho_recompute`: `:once` (the default -- see Limitations), `n::Int` for every `n` steps, or a
  callable of the number of steps since the last estimate. A step failure and the first step after
  an `init`/`reinit!` always force a re-estimate.
- `rho_safety`: multiplies *every* `ρ` this algorithm uses, estimated or overridden. The default
  `1.1` is the hedge against the complex part of the spectrum, which the real-axis stability
  boundaries of the STS families do not cover.
- `rho_S_estimate`: `:fd` (Jacobian-free directional finite differences over the slow force) or a
  number to use verbatim.
- `rho_F_estimate`: `:power` (power iteration over the rate operator), `:gershgorin` (a host-only
  upper bound, unavailable under device assembly), or a number to use verbatim.
- `max_stages`: refuse rather than silently truncate a stage count this large.
- `batch_size_hint`: `Polyester` batch size of the pointwise stages.
- `inner_algs`: the splitting children, both [`PassiveChildSolver`](@ref)s -- the step is monolithic
  and the children exist for the splitting bookkeeping only.

# Limitations
- Single domain only: the split has to be exactly one `AffineODEFunction` over one
  [`PointwiseODEFunction`](@ref), which is what `semidiscretize(ReactionDiffusionSplit(...))`
  produces for a [`MonodomainModel`](@ref).
- The mass matrix is row-sum lumped (hence P1) and the ionic current enters pointwise at the dofs.
  That is a different semidiscretization from the consistent-mass one [`BackwardEulerSolver`](@ref)
  steps, and on a propagating front the O(h²) between the two is what dominates the difference
  between the schemes -- at every step size, not only at coarse ones.
- Fixed step size: the stage counts are stability control, not a local error estimate.
- The stage counts resolve the STS families' real-axis stability boundaries. The complex part of the
  spectrum is a gap in the underlying theory, and `rho_safety` is the only knob against it.
- `rho_recompute = :once` estimates ρ_S/ρ_F once, at `t = 0`, and trusts that estimate for the whole
  run, with tightened estimator tolerances (`reltol = 1.0e-4`, `maxiters = 200`) to keep it honest.
  The reference implementation has no such mode: it re-estimates both radii every 5 steps by
  default. A frozen `t = 0` estimate under-stages a run whose true ρ_S grows -- measured on a
  PCG2019 propagating front, ρ_S moved up to ~4.4x over the run.
"""
Base.@kwdef struct ExponentialMultirateSTSAlgorithm{
    OuterFamilyType <: AbstractSTSFamily,
    InnerFamilyType <: AbstractSTSFamily,
    GateSelectionType,
    SolutionVectorType,
    SystemMatrixType,
    RhoPolicyType,
    RhoSlowType,
    RhoFastType,
    T <: Real,
} <: OS.AbstractOperatorSplittingAlgorithm
    outer::OuterFamilyType = RKC1(0.05)
    inner::InnerFamilyType = RKC1(0.05)
    gates::GateSelectionType = :all
    solution_vector_type::Type{SolutionVectorType} = Vector{Float64}
    system_matrix_type::Type{SystemMatrixType} = ThreadedSparseMatrixCSR{Float64, Int64}
    rho_recompute::RhoPolicyType = :once
    rho_safety::T = 1.1
    rho_S_estimate::RhoSlowType = :fd
    rho_F_estimate::RhoFastType = :power
    max_stages::Int = 200
    batch_size_hint::Int = 32
    inner_algs::Tuple{PassiveChildSolver, PassiveChildSolver} =
        (PassiveChildSolver(), PassiveChildSolver())
end

@doc (@doc ExponentialMultirateSTSAlgorithm)
const EMRKC = ExponentialMultirateSTSAlgorithm

# The stage counts are a stability device, not a local error estimate.
@inline SciMLBase.isadaptive(::ExponentialMultirateSTSAlgorithm) = false

function Base.show(io::IO, alg::ExponentialMultirateSTSAlgorithm)
    print(io, "EMRKC(outer = ")
    Base.show(io, alg.outer)
    print(io, ", inner = ")
    Base.show(io, alg.inner)
    print(io, ", gates = ", alg.gates, ")")
    return nothing
end

#####################################################################
#  Pointwise stages                                                 #
#####################################################################

# The exponential gate half-step of one outer stage, in place: the caller fills `ymat` with the
# outer stage value `Y`, and every *selected* gate row is overwritten by its exact solution over the
# averaging window `η`, which reaches the kernel through the `Δt` slot of
# `_pointwise_step_inner_kernel!`. This stage transforms STATE rather than producing a rate, which
# is why its buffer is not named after a derivative. `mask` selects which of
# `gidx = gating_indices(model)` the algorithm's `gates` option asked for; an empty selection is
# lowered to empty tuples, so a model that declares no gates -- or is run with `gates = ()` -- never
# reaches `gate_coefficients` at all.
struct EMRKCGateStageCache{ymType, xType, NG} <: AbstractPointwiseSolverCache
    ymat::ymType
    gidx::NTuple{NG, Int}
    mask::NTuple{NG, Bool}
    batch_size_hint::Int
    xs::xType
end
Adapt.@adapt_structure EMRKCGateStageCache

@inline function _pointwise_step_inner_kernel!(
    cell_model::F,
    i::I,
    t::T,
    η::T,
    cache::C,
) where {F, C <: EMRKCGateStageCache, T <: Real, I <: Integer}
    _emrkc_gate_step!(
        (@view cache.ymat[i, :]),
        cell_model,
        cache.gidx,
        cache.mask,
        getcoordinate(cache, i),
        t,
        η,
    )
    return true
end

@inline _emrkc_gate_step!(y, cell_model, ::Tuple{}, ::Tuple{}, x, t, η) = nothing

@inline function _emrkc_gate_step!(
    y,
    cell_model,
    gidx::NTuple{NG, Int},
    mask::NTuple{NG, Bool},
    x,
    t,
    η,
) where {NG}
    # Read before write, so `gate_coefficients` sees the unmodified stage value and the update may
    # run in place even where the transmembrane potential is itself a declared gate.
    φ = y[transmembranepotential_index(cell_model)]
    λ, y∞ = gate_coefficients(cell_model, φ, y, t)
    @inbounds for k = 1:NG
        if mask[k]
            j = gidx[k]
            y[j] = exponential_gate_step(y[j], λ[k], y∞[k], η)
        end
    end
    return nothing
end

# The slow force `f_S` of one outer stage: `cell_rhs!` at the gate-advanced state `uₙmat`, written
# into `dumat`, with every row the exponential stage already integrated zeroed out so it is not
# integrated twice. That zeroing is what makes `f_S` non-stiff, and what the outer stage count is
# sized against.
struct EMRKCReactionStageCache{umType, dumType, xType, NG} <: AbstractPointwiseSolverCache
    uₙmat::umType
    dumat::dumType
    gidx::NTuple{NG, Int}
    mask::NTuple{NG, Bool}
    batch_size_hint::Int
    xs::xType
end
Adapt.@adapt_structure EMRKCReactionStageCache

@inline function _pointwise_step_inner_kernel!(
    cell_model::F,
    i::I,
    t::T,
    Δt::T,
    cache::C,
) where {F, C <: EMRKCReactionStageCache, T <: Real, I <: Integer}
    u_local  = @view cache.uₙmat[i, :]
    du_local = @view cache.dumat[i, :]

    cell_rhs!(du_local, u_local, getcoordinate(cache, i), t, cell_model)

    @inbounds for k = 1:length(cache.gidx)
        cache.mask[k] && (du_local[cache.gidx[k]] = zero(eltype(du_local)))
    end
    return true
end

#####################################################################
#  Cache                                                            #
#####################################################################

# `u`/`uprev` are the parent's own buffers, held to satisfy the splitting cache interface. The step
# reads `u` as the read-only anchor `Y₀` of the outer sweep and writes it exactly once, at the end;
# `uprev` is never touched, since it is the rollback anchor of the surrounding integrator.
mutable struct EMRKCCache{
    uType,
    uprevType,
    ODEFunctionType,
    RateOperatorType,
    SourceOperatorType,
    VecType,
    SubVecType,
    ViewType,
    GateStageType,
    ReactionStageType,
    SlowWorkspaceType,
    FastWorkspaceType,
    T,
    RangeType,
} <: OS.AbstractOperatorSplittingCache
    u::uType
    uprev::uprevType
    # The pointwise function the two stage kernels are launched over.
    odefun::ODEFunctionType
    # f_F: x ↦ Mₗ⁻¹Kx, plus the lumped inverse mass the source is scaled by.
    op::RateOperatorType
    source_op::SourceOperatorType
    # Full width: the two rotating outer stage buffers, the averaged force, the gate-advanced
    # state and the slow force.
    Ya::VecType
    Yb::VecType
    fbar::VecType
    yE::VecType
    fS::VecType
    # Transmembrane width: the two rotating inner stage buffers and the inner rate.
    Ua::SubVecType
    Ub::SubVecType
    du_inner::SubVecType
    # The transmembrane rows of the full width buffers, as views taken once.
    yEV::ViewType
    fSV::ViewType
    gate_stage::GateStageType
    reaction_stage::ReactionStageType
    ws_S::SlowWorkspaceType
    ws_F::FastWorkspaceType
    ρS::T
    ρF::T
    steps_since_estimate::Int
    Vrange::RangeType
end

#####################################################################
#  Setup                                                            #
#####################################################################

function OS.init_cache(
    f::GenericSplitFunction,
    alg::ExponentialMultirateSTSAlgorithm;
    uprev::AbstractArray,
    u::AbstractVector,
)
    length(f.functions) == 2 || _emrkc_function_tree_error(f)
    fheat, fode = f.functions[1], f.functions[2]
    (fheat isa AffineODEFunction && fode isa PointwiseODEFunction) || _emrkc_function_tree_error(f)
    fode.layout isa StateBlockedLayout || _emrkc_layout_error(fode)

    Vrange = f.solution_indices[1]
    Vrange isa AbstractUnitRange || _emrkc_heat_dofrange_error(Vrange)
    f.solution_indices[2] == 1:length(u) || _emrkc_multidomain_error(f, u)

    T = eltype(u)
    ion = fode.ode
    nstates = num_states(ion)
    npoints = length(fode.associated_states) ÷ nstates
    nV = length(Vrange)
    nV == npoints || error(
        "The transmembrane index set has $(nV) entries but the cell model covers $(npoints) " *
        "points. `EMRKC` expects one transmembrane dof per pointwise state point.",
    )

    # Assembly as the affine backward Euler stage's setup does it, with `spec` standing in for the
    # solver whose two type knobs `setup_operator` reads. There is no `t₀` at this point of the
    # operator splitting init path, so the stationary parts are assembled at zero; a time dependent
    # conductivity would need a re-assembly hook the splitting cache has no place for yet. The
    # *source* is refreshed at the real stage time on every outer stage below.
    spec            = _OperatorSetupSpec(alg.solution_vector_type, alg.system_matrix_type)
    dh              = fheat.dh
    strategy        = get_strategy(fheat)
    ctx₀            = TimeIntegrationContext(zero(T), zero(T), zero(T))
    rate_form       = FerriteOperators.RateFormIntegrator(fheat.bilinear_term, fheat.mass_term)
    rate_operator   = setup_operator(strategy, rate_form, spec, dh; initial_context = ctx₀)
    source_operator = setup_operator(strategy, fheat.source_term, spec, dh)
    @timeit_debug "initial assembly" begin
        update_operator!(rate_operator, nothing, ctx₀)
        update_operator!(source_operator, nothing, ctx₀)
    end
    op = RateOperator(
        rate_operator,
        _rate_source_inverse_mass(rate_operator, source_operator, strategy, fheat.mass_term, dh, ctx₀),
        rate_sign(fheat.bilinear_term),
    )


    # Zeroed, not merely allocated: a cell model whose `cell_rhs!` leaves a state untouched would
    # otherwise read whatever was in the slow force buffer.
    zeroed(n...) = fill!(similar(u, n...), zero(T))
    Ya, Yb, fbar, yE, fS = zeroed(), zeroed(), zeroed(), zeroed(), zeroed()
    Ua, Ub, du_inner = zeroed(nV), zeroed(nV), zeroed(nV)

    gidx, mask = _emrkc_gate_selection(alg.gates, ion)
    xs = fode.x === nothing ? nothing : adapt_vector_type(alg.solution_vector_type, fode.x)
    gate_stage =
        EMRKCGateStageCache(reshape(yE, (npoints, nstates)), gidx, mask, alg.batch_size_hint, xs)
    reaction_stage = EMRKCReactionStageCache(
        reshape(yE, (npoints, nstates)),
        reshape(fS, (npoints, nstates)),
        gidx,
        mask,
        alg.batch_size_hint,
        xs,
    )

    return EMRKCCache(
        u,
        uprev,
        fode,
        op,
        source_operator,
        Ya,
        Yb,
        fbar,
        yE,
        fS,
        Ua,
        Ub,
        du_inner,
        view(yE, Vrange),
        view(fS, Vrange),
        gate_stage,
        reaction_stage,
        SpectralRadiusWorkspace(u),
        SpectralRadiusWorkspace(Ua),
        zero(real(T)),
        zero(real(T)),
        -1, # never estimated
        Vrange,
    )
end

# What a source is scaled by. Nothing where there is none; the rate form's own `M⁻¹` where it keeps
# one (the diagonal's reciprocal, or the per-cell inverse operator); and where the block-row store
# fused `M⁻¹` away, a per-cell inverse operator of its own on the same device.
_rate_source_inverse_mass(rate_operator, ::LinearNullOperator, strategy, mass, dh, ctx₀) = nothing
function _rate_source_inverse_mass(rate_operator, source_operator, strategy, mass, dh, ctx₀)
    minv = FerriteOperators.rate_form_inverse_mass(rate_operator)
    minv === nothing || return _inverse_mass_apply(minv)
    return setup_operator(
        AssemblyStrategy(
            MatrixFreeAction(; storage = FerriteOperators.ElementAssembly()),
            strategy.scheduling,
            strategy.device,
        ),
        FerriteOperators.ElementInverse(mass),
        dh;
        initial_context = ctx₀,
    )
end
_inverse_mass_apply(minv::Diagonal) = minv.diag
_inverse_mass_apply(minv) = minv

# An empty selection collapses to empty tuples rather than an all-false mask, which is what makes
# `gates = ()` run a model that implements no `gate_coefficients` at all.
function _emrkc_gate_selection(gates, ion)
    gsyms = gating_symbols(ion)
    mask  = _emrkc_gate_mask(gates, gsyms, ion)
    any(mask) || return ((), ())
    return (gating_indices(ion), mask)
end

_emrkc_gate_mask(gates::Symbol, gsyms, ion) =
    gates === :all ? ntuple(i -> true, length(gsyms)) : _emrkc_gates_option_error(gates)

function _emrkc_gate_mask(gates::Tuple, gsyms, ion)
    for g in gates
        g ∈ gsyms || _emrkc_gate_not_declared(g, gsyms, ion)
    end
    return ntuple(i -> gsyms[i] ∈ gates, length(gsyms))
end

_emrkc_gate_mask(gates, gsyms, ion) = _emrkc_gates_option_error(gates)

@noinline _emrkc_gates_option_error(gates) = error(
    "`EMRKC(gates = $(repr(gates)))` is not a partition: pass `:all`, or a `Tuple` of the " *
    "symbols `gating_symbols` declares (`()` for the degenerate non-exponential mode).",
)

@noinline _emrkc_gate_not_declared(g, gsyms, ion) = error(
    "`EMRKC(gates = ...)` asks for $(repr(g)), which $(nameof(typeof(ion))) does not declare as " *
    "a gate: `gating_symbols` returns $(gsyms). A state may only be integrated exponentially " *
    "once the model declares it follows the gate normal form.",
)

@noinline _emrkc_function_tree_error(f) = error(
    "`EMRKC` steps a reaction-diffusion split: exactly one `AffineODEFunction` over one " *
    "`PointwiseODEFunction`, as `semidiscretize(ReactionDiffusionSplit(model), ...)` produces " *
    "for a `MonodomainModel`. Got $(length(f.functions)) operators of types " *
    "$(map(typeof, f.functions)).",
)

@noinline _emrkc_layout_error(fode) = error(
    "`EMRKC` reads the pointwise state as a state-blocked matrix, but this " *
    "`PointwiseODEFunction` carries a $(nameof(typeof(fode.layout))).",
)

@noinline _emrkc_heat_dofrange_error(Vrange) = error(
    "`EMRKC` addresses the transmembrane rows as a contiguous slice of the solution vector, but " *
    "the split's first index set is a $(typeof(Vrange)). That is the multi-domain case, which " *
    "this algorithm does not serve yet -- the seam is the `Vrange` field of `EMRKCCache` and the " *
    "views taken from it.",
)

@noinline _emrkc_multidomain_error(f, u) = error(
    "`EMRKC` expects the pointwise operator to cover the whole solution vector (a single " *
    "domain), but its index set is $(f.solution_indices[2]) of $(length(u)) unknowns.",
)

#####################################################################
#  Spectral radii                                                   #
#####################################################################

# `estimate_rho!` takes the operator as an `apply!`; these two are that callable, as structs rather
# than closures so the iteration stays allocation free.
struct _EMRKCRateApply{OperatorType}
    op::OperatorType
end
(a::_EMRKCRateApply)(w, v) = mul_rate!(w, a.op, v)

# `v ↦ J_S v` as the directional finite difference `(f_S(u + (δ/‖v‖)v) - f_S(u)) / (δ/‖v‖)`, with
# `f_S(u)` precomputed into `cache.fbar`. Homogeneous of degree one in `v` by construction -- the
# perturbation scales the direction to a fixed *length* `δ` and divides by that same step -- which is
# what lets a linear power iteration drive it. Same structure as `OrdinaryDiffEqStabilizedRK`'s
# `maxeig!`, which is where the `‖u‖√eps` perturbation comes from.
struct _EMRKCSlowJacobianApply{CacheType, VecType, TimeType, T}
    cache::CacheType
    u::VecType
    t::TimeType
    δ::T
end

function (a::_EMRKCSlowJacobianApply)(w, v)
    (; cache, u, t, δ) = a
    ε = δ / norm(v)
    cache.yE .= u .+ ε .* v
    _emrkc_slow_force!(cache, t)
    w .= (cache.fS .- cache.fbar) ./ ε
    return w
end

# f_S at whatever `cache.yE` currently holds, into `cache.fS`. The source is deliberately *not* added
# here: it does not depend on the state, so it cancels out of the finite difference, and the averaged
# force adds it separately on the transmembrane rows only.
function _emrkc_slow_force!(cache::EMRKCCache, t)
    _pointwise_step_outer_kernel!(cache.odefun, t, zero(t), cache.reaction_stage, cache.fS)
    return cache.fS
end

# The perturbation *length* of the directional difference: `‖u‖√eps` balances truncation against
# cancellation, falling back to `√eps` at a zero state, where there is no scale to read.
function _emrkc_fd_perturbation(u)
    T  = real(eltype(u))
    nu = norm(u)
    return nu > 0 ? T(nu * sqrt(eps(T))) : sqrt(eps(T))
end

# Whether ρ needs re-estimating this step, for the `rho_recompute` policies `EMRKC` documents. A
# NEGATIVE `steps_since` means none has run yet and, like a step failure, always forces one.
function _should_reestimate(policy, steps_since, stepfail::Bool)
    (stepfail || steps_since < 0) && return true
    return _reestimate_due(policy, steps_since)
end

_reestimate_due(policy::Symbol, steps_since) =
    policy === :once ? false :
    error("Unknown rho_recompute policy :$policy -- expected :once, an Int, or a callable.")
_reestimate_due(n::Integer, steps_since) = steps_since ≥ n - 1
_reestimate_due(f, steps_since) = f(steps_since)

function _emrkc_refresh_rho!(cache::EMRKCCache, alg, parent, t)
    # A reinit! resets `iter`, so the first attempted step of every run re-estimates regardless of
    # the policy -- `:once` means once per run, and ρ_S depends on the state the run starts from.
    forced = parent.iter ≤ 1 || parent.last_step_failed
    if _should_reestimate(alg.rho_recompute, cache.steps_since_estimate, forced)
        cache.ρF = _emrkc_rho_F!(alg.rho_F_estimate, cache, alg)
        cache.ρS = _emrkc_rho_S!(alg.rho_S_estimate, cache, alg, parent.u, t)
        cache.steps_since_estimate = 0
    else
        cache.steps_since_estimate += 1
    end
    return nothing
end

# `:once` pays for the estimate once per run and then trusts it, so it gets a tighter stopping rule
# than the repeated policies: the loose default reltol = 1e-2 can plateau well below the true radius
# on a near-degenerate spectrum, eating through `rho_safety` in the destabilizing direction.
_emrkc_estimator_options(policy) =
    policy === :once ? (maxiters = 200, reltol = 1.0e-4) : (maxiters = 50, reltol = 1.0e-2)

_emrkc_rho_F!(ρ::Real, cache::EMRKCCache, alg) = _emrkc_rho_type(cache)(alg.rho_safety * ρ)

function _emrkc_rho_F!(mode::Symbol, cache::EMRKCCache, alg)
    if mode === :power
        return estimate_rho!(
            cache.ws_F,
            _EMRKCRateApply(cache.op);
            safety = alg.rho_safety,
            _emrkc_estimator_options(alg.rho_recompute)...,
        )
    elseif mode === :gershgorin
        return _emrkc_rho_type(cache)(alg.rho_safety * _gershgorin_bound(cache.op))
    end
    return error(
        "Unknown `rho_F_estimate` $(repr(mode)) -- expected `:power`, `:gershgorin`, or a number.",
    )
end

_emrkc_rho_S!(ρ::Real, cache::EMRKCCache, alg, u, t) = _emrkc_rho_type(cache)(alg.rho_safety * ρ)

function _emrkc_rho_S!(mode::Symbol, cache::EMRKCCache, alg, u, t)
    mode === :fd ||
        return error("Unknown `rho_S_estimate` $(repr(mode)) -- expected `:fd` or a number.")
    cache.yE .= u
    _emrkc_slow_force!(cache, t)
    cache.fbar .= cache.fS # the finite difference baseline f_S(u)
    return estimate_rho!(
        cache.ws_S,
        _EMRKCSlowJacobianApply(cache, u, t, _emrkc_fd_perturbation(u));
        safety = alg.rho_safety,
        describe = () -> _emrkc_rho_S_runaway_context(u, alg),
        _emrkc_estimator_options(alg.rho_recompute)...,
    )
end

# What `estimate_rho!`'s generic runaway guard cannot name: the state a Jacobian-free difference was
# taken at, and `EMRKC`'s own knobs.
_emrkc_rho_S_runaway_context(u, alg) = " The state norm is ‖u‖ = $(norm(u)); the likely cause is " *
    "state drift or a non-smooth right-hand side at this evaluation point. Consider a different " *
    "`rho_recompute` policy (currently $(repr(alg.rho_recompute))) or bypassing the estimator " *
    "with a raw-number `rho_S_estimate` override."

_emrkc_rho_type(cache::EMRKCCache) = typeof(cache.ρS)

# The Gershgorin bound reads rows of the assembled diffusion matrix, which only a host operator
# exposes; a matrix on a device has no method here and says so through `_gershgorin_bound`.
_emrkc_host_matrix(op) = FerriteOperators.get_matrix(op)
_emrkc_host_matrix(op::MirroredBilinearOperator) = FerriteOperators.get_matrix(op.host_operator)

# The bound needs the assembled diffusion matrix and a diagonal inverse mass, both on the host.
_gershgorin_bound(r::RateOperator) = _gershgorin_bound(
    _emrkc_host_matrix(FerriteOperators.rate_form_rhs(r.op)),
    _host_inverse_mass_diagonal(FerriteOperators.rate_form_inverse_mass(r.op)),
)
_host_inverse_mass_diagonal(d::Diagonal) = collect(d.diag)
@noinline _host_inverse_mass_diagonal(::Any) = error(
    "`rho_F_estimate = :gershgorin` reads the rows of the assembled rate matrix, which needs a " *
    "diagonal mass; this rate operator applies its inverse mass per cell or matrix free. Use " *
    "`:power`, or a number.",
)

#####################################################################
#  The step                                                         #
#####################################################################

# The inner sweep's right hand side on the transmembrane rows: `v ↦ Mₗ⁻¹Kv + f_S`, with `f_S` frozen
# at the value the enclosing outer stage computed.
struct _EMRKCInnerRHS{OperatorType, VecType}
    op::OperatorType
    frozen::VecType
end

function (r::_EMRKCInnerRHS)(du, v, t)
    mul_rate!(du, r.op, v)
    du .+= r.frozen
    return nothing
end

# The averaged force `f̄(t, Y) = (u_η - Y)/η` of Algorithm 3, as the right hand side the outer sweep
# calls once per stage. `u_η` is the state at `η` of
#
#     v' = f_F(v) + f_S(y_E),    v(0) = y_E,
#
# where `y_E` is `Y` with every selected gate advanced exactly over `η` and `f_S` is the slow force
# at `y_E`, both frozen for the whole inner sweep.
#
# Three placements decide whether this is emRKC or a scheme that merely resembles it, and none of
# them is visible in a convergence test: the exponential is sized by `η` and taken once per outer
# stage (not by the outer step), the inner sweep starts from `y_E` (not `Y`), and the difference
# quotient is taken against `Y` (not `y_E`).
#
# Only the transmembrane rows are swept: `f_F` acts on those alone, so everywhere else the inner
# right hand side is the constant `f_S`, integrated exactly as `u_η = y_E + η f_S`.
struct _EMRKCAveragedForce{CacheType, AlgType, T}
    cache::CacheType
    alg::AlgType
    η::T
    m::Int
end

function (force::_EMRKCAveragedForce)(fbar, Y, t)
    (; cache, alg, η, m) = force
    V = cache.Vrange

    # (1) y_E: the outer stage value with the selected gates integrated exactly over η.
    cache.yE .= Y
    _pointwise_step_outer_kernel!(cache.odefun, t, η, cache.gate_stage, cache.yE)

    # (2) the slow force there, with the exponentially integrated rows removed, plus the source.
    _emrkc_slow_force!(cache, t)
    refresh_source_operator!(cache.source_op, t)
    _emrkc_add_source_rate!(cache.fSV, cache.op, cache.source_op)

    # (3) the inner sweep over the transmembrane rows.
    U = sts_sweep!(
        _EMRKCInnerRHS(cache.op, cache.fSV),
        cache.Ua,
        cache.Ub,
        cache.du_inner,
        cache.yEV,
        t,
        η,
        m,
        alg.inner,
    )

    # (4) the difference quotient against Y: the analytic finish of the rows the sweep skipped, then
    # an overwrite of the rows it did not. The transmembrane view is taken from the `fbar` the sweep
    # handed in, so nothing here depends on which buffer that is.
    @.. fbar = (cache.yE - Y) / η + cache.fS
    YV = @view Y[V]
    (@view fbar[V]) .= (U .- YV) ./ η
    return nothing
end

_emrkc_add_source_rate!(y, rate_op, ::LinearNullOperator) = y
_emrkc_add_source_rate!(y, rate_op, source_op) =
    add_source_rate!(y, rate_op, FerriteOperators.operator_payload(source_op))

function OS._perform_step!(parent, children::Tuple, cache::EMRKCCache, dt)
    alg = parent.alg
    t   = parent.t

    # An `EMRKCDivergence` caught here (`estimate_rho!`'s runaway guard, or a stage count over
    # `max_stages` or `Int` itself) is a step failure like the NaN check below -- but only once
    # there is a step to fail: on the first attempt of a run nothing has been accepted yet, so it
    # stays loud instead of being swallowed into a silent, immediate `ReturnCode.Failure`. Every
    # other error on this path is a bug or an unusable option, and is rethrown whatever the step
    # count.
    s, η, m = try
        @timeit_debug "spectral radii" _emrkc_refresh_rho!(cache, alg, parent, t)
        _emrkc_step_sizing(alg, dt, cache.ρS, cache.ρF)
    catch e
        (e isa EMRKCDivergence && parent.iter > 1) || rethrow()
        parent.force_stepfail = true
        @warn "EMRKC failed the step at t = $t: $(e.msg)" _group = :timeintegration
        return
    end

    @timeit_debug "outer sweep" Ys = sts_sweep!(
        _EMRKCAveragedForce(cache, alg, η, m),
        cache.Ya,
        cache.Yb,
        cache.fbar,
        parent.u, # read-only anchor Y₀
        t,
        dt,
        s,
        alg.outer,
    )

    # Every stage carries its predecessor forward with a nonzero coefficient, per row, so a NaN
    # anywhere in the sweep reaches the final stage: one check here catches a mid-sweep blow-up and
    # leaves `parent.u` -- and with it the rollback anchor `parent.uprev` -- untouched for the retry.
    if !all(isfinite, Ys)
        parent.force_stepfail = true
        return
    end
    parent.u .= Ys

    _emrkc_advance_children!(parent, children, dt)
    return
end

# The three numbers Algorithm 3 derives from a step size and the two spectral radii (both of which
# already carry `rho_safety`): the outer stage count `s` resolving `Δt ρ_S`, the averaging window
# `η = 2Δt / ℓ_outer(s)`, and the inner stage count `m` resolving `η ρ_F`.
#
# The window is what couples the two: an `s`-stage outer sweep is stable up to `ℓ_outer(s)`, the
# averaged force's own spectral radius is at most `2/η`, and equating the two gives the largest
# window -- hence the cheapest inner sweep -- the outer sweep can still carry.
#
# `η` is converted to `typeof(Δt)`: `sts_stability_boundary` always computes in `Float64`, but `η`
# feeds the per-point inner sweep once per outer stage and must not promote its broadcasts.
function _emrkc_step_sizing(alg, Δt, ρS, ρF)
    T = typeof(Δt)
    s = _emrkc_stage_count(alg.outer, Δt * ρS, alg, :outer)
    η = 2Δt / T(sts_stability_boundary(alg.outer, s))
    m = _emrkc_stage_count(alg.inner, η * ρF, alg, :inner)
    return s, η, m
end

function _emrkc_stage_count(fam, z, alg, which::Symbol)
    isfinite(z) && z ≥ 0 || _emrkc_stage_count_error(which, z, alg)
    s = sts_stage_count(fam, z)
    s ≤ alg.max_stages || _emrkc_stage_count_error(which, z, alg, s)
    return s
end

@noinline function _emrkc_stage_count_error(which, z, alg, s = nothing)
    what = which === :outer ? "outer (Δt·ρ_S)" : "inner (η·ρ_F)"
    s === nothing && throw(EMRKCDivergence(
        "`EMRKC` cannot size its $(what) sweep: the stiffness measure came out as $z. The state " *
        "has most likely diverged, or a `rho_*_estimate` override is not a usable spectral radius.",
    ))
    throw(EMRKCDivergence(
        "`EMRKC` needs $(s) $(which) stages for a stiffness measure of $(what) = $z, above " *
        "`max_stages = $(alg.max_stages)`. Truncating the count would silently drop the " *
        "stability this stage count buys, so reduce `dt` or raise `max_stages`.",
    ))
end

# The Lie-Trotter-Godunov child sequence, run *after* the monolithic step has written `parent.u`.
# The children are passive, so the forward sync distributes the new state into them, the advance
# moves only their clocks, and the backward sync writes back what it just read.
#
# `OrdinaryDiffEqOperatorSplitting` owns that sequence -- and with it the sync protocol -- so this
# calls into it rather than repeating it. `LieTrotterGodunovCache` is an immutable pair of the two
# buffers the sequence needs, so building one per step costs nothing.
_emrkc_advance_children!(parent, children::Tuple, dt) =
    OS._perform_step!(parent, children, OS.LieTrotterGodunovCache(parent.u, parent.uprev), dt)
