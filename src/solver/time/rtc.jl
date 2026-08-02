"""
    ReactionTangentController{AlgTupleType <: Tuple, T <: Real} <: OS.AbstractOperatorSplittingAlgorithm

An adaptive [`LieTrotterGodunov`](@ref) [Lie:1880:tti,Tro:1959:psg,God:1959:dmn](@cite)
operator splitting algorithm whose timestep length is controlled by the reaction tangent
as proposed in [OgiBalPer:2023:seats](@cite).
The next timestep length is calculated as
```math
\\sigma\\left(R_{\\max }\\right):=\\left(1.0-\\frac{1}{1+\\exp \\left(\\left(\\sigma_{\\mathrm{c}}-R_{\\max }\\right) \\cdot \\sigma_{\\mathrm{s}}\\right)}\\right) \\cdot\\left(\\Delta t_{\\max }-\\Delta t_{\\min }\\right)+\\Delta t_{\\min }
```
Note that this is a pure heuristic: the controller has no error estimator and never
rejects a step on accuracy grounds.

# Fields
- `inner_algs::AlgTupleType`: the timesteppers for the inner problems, as in [`LieTrotterGodunov`](@ref)
- `σ_s::T`: steepness
- `σ_c::T`: offset in R axis
- `Δt_bounds::NTuple{2,T}`: lower and upper timestep length bounds

A `LieTrotterGodunov` algorithm may be passed in place of `inner_algs`; its inner
timesteppers are then unwrapped.
"""
struct ReactionTangentController{AlgTupleType <: Tuple, T <: Real} <:
       OS.AbstractOperatorSplittingAlgorithm
    inner_algs::AlgTupleType
    σ_s::T
    σ_c::T
    Δt_bounds::NTuple{2, T}
end

ReactionTangentController(ltg::OS.LieTrotterGodunov, σ_s, σ_c, Δt_bounds) =
    ReactionTangentController(ltg.inner_algs, σ_s, σ_c, Δt_bounds)

@inline SciMLBase.isadaptive(::ReactionTangentController) = true

# Required by the adaptive-algorithm interface. RTC has no error estimator (see below),
# so this order never enters a step size law; 1 matches the underlying first-order
# Lie-Trotter-Godunov sequence.
OS.alg_adaptive_order(::ReactionTangentController) = 1

# RTC steps exactly like LieTrotterGodunov -- only the step size selection differs, and
# that lives in the controller cache below. Reusing the LTG cache also reuses its
# `_perform_step!`, which dispatches on the cache type.
OS.init_cache(f::GenericSplitFunction, alg::ReactionTangentController; uprev, u) =
    OS.init_cache(f, OS.LieTrotterGodunov(alg.inner_algs); uprev, u)

"""
    get_reaction_tangent(integrator::OS.AnySplitIntegrator)
Returns the maximal reaction magnitude using the [`PointwiseODEFunction`](@ref) of an operator splitting integrator that uses [`LieTrotterGodunov`](@ref) [Lie:1880:tti,Tro:1959:psg,God:1959:dmn](@cite).
It is assumed that the problem containing the reaction tangent is a [`PointwiseODEFunction`](@ref).
"""
@inline function get_reaction_tangent(integrator::OS.AnySplitIntegrator)
    R, _ = _get_reaction_tangent(integrator.child_subintegrators)
    return R
end

@inline @unroll function _get_reaction_tangent(subintegrators, n_reaction_tangents::Int = 0)
    R = 0.0
    @unroll for subintegrator in subintegrators
        if subintegrator isa Tuple || subintegrator isa OS.SplitSubIntegrator
            children = subintegrator isa Tuple ? subintegrator : subintegrator.child_subintegrators
            Rinner, n_reaction_tangents = _get_reaction_tangent(children, n_reaction_tangents)
            R = max(R, Rinner)
        elseif subintegrator.f isa PointwiseODEFunction
            n_reaction_tangents += 1
            φₘidx = transmembranepotential_index(subintegrator.f.ode)
            R = max(R, maximum(@view subintegrator.cache.dumat[:, φₘidx]))
        elseif subintegrator.f isa PointwiseMultiODEFunction
            n_reaction_tangents += 1
            for (i, f) in enumerate(subintegrator.f.functions)
                φₘidx = transmembranepotential_index(f.ode)
                R = max(R, maximum(@view subintegrator.cache.dumat[i][:, φₘidx]))
            end
        end
    end
    @assert n_reaction_tangents == 1 "No or multiple integrators using PointwiseODEFunction found"
    return (R, n_reaction_tangents)
end

# The controller half of the algorithm. The controller object itself is stateless (all
# parameters live on the algorithm); the cache carries the reaction tangent of the last
# attempted step.
struct ReactionTangentStepsizeController <: OrdinaryDiffEqCore.AbstractController end

mutable struct ReactionTangentControllerCache{T <: Real} <:
               OrdinaryDiffEqCore.AbstractControllerCache
    R::T
end

# An adaptive splitting node without an explicit `controller` runs this controller.
OS.default_controller(::ReactionTangentController, ::NamedTuple) =
    ReactionTangentStepsizeController()

OrdinaryDiffEqCore.setup_controller_cache(
    alg::ReactionTangentController,
    cache,
    ::ReactionTangentStepsizeController,
    ::Type{EEstT},
    disco_probs,
) where {EEstT} = ReactionTangentControllerCache(zero(EEstT))

# Controller protocol. RTC ignores the error estimate (`integrator.EEst`) entirely -- it
# is a heuristic dt = σ(R) map, so do not look for an error estimator here.
@inline function OS.stepsize_controller!(
    integrator::OS.AnySplitIntegrator,
    controller_cache::ReactionTangentControllerCache,
    alg::ReactionTangentController,
)
    controller_cache.R = get_reaction_tangent(integrator)
    # There is no dt ratio q; σ(R) below maps R directly to the next dt.
    return nothing
end

@inline function OS.step_accept_controller!(
    integrator::OS.AnySplitIntegrator,
    controller_cache::ReactionTangentControllerCache,
    alg::ReactionTangentController,
    q,
)
    @unpack σ_s, σ_c, Δt_bounds = alg
    @unpack R = controller_cache
    dtnew = if isinf(σ_s)
        R > σ_c ? Δt_bounds[1] : Δt_bounds[2]
    else
        (1 - 1 / (1 + exp((σ_c - R) * σ_s))) * (Δt_bounds[2] - Δt_bounds[1]) + Δt_bounds[1]
    end
    # The caller assigns integrator.dt and integrator.dtcache from the returned value.
    return integrator.tdir * dtnew
end

@inline function OS.step_reject_controller!(
    integrator::OS.AnySplitIntegrator,
    controller_cache::ReactionTangentControllerCache,
    alg::ReactionTangentController,
)
    # Unreachable through the error-estimate path, since `accept_step_controller` below
    # never rejects; kept because it is part of the controller protocol. Unlike the
    # accept hook, this one sets dt itself.
    if abs(integrator.dt) ≤ alg.Δt_bounds[1] # Check for "≤" to also handle the boundary cases
        error("RTC cannot recover from step rejection below Δt min") # Force failure
    else
        integrator.dt = integrator.tdir * alg.Δt_bounds[1]
    end
    return nothing
end

# No error estimator: every step whose inner solves succeeded is accepted.
@inline OrdinaryDiffEqCore.accept_step_controller(
    integrator,
    ::ReactionTangentControllerCache,
    alg,
) = true

# The force_stepfail retry path divides dt by this factor. RTC has no controller knobs to
# store it in, so fall back to the algorithm default.
@inline OrdinaryDiffEqCore.get_failfactor(integrator, ::ReactionTangentControllerCache) =
    OrdinaryDiffEqCore.failfactor_default(integrator.alg)

function OrdinaryDiffEqCore.reinit_controller!(
    integrator::SciMLBase.DEIntegrator,
    cache::ReactionTangentControllerCache,
)
    cache.R = zero(cache.R)
    return nothing
end
