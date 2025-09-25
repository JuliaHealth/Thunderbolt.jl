"""
    ReactionTangentController{LTG <: OS.LieTrotterGodunov, T <: Real} <: OS.AbstractOperatorSplittingAlgorithm
A timestep length controller for [`LieTrotterGodunov`](@ref) [Lie:1880:tti,Tro:1959:psg,God:1959:dmn](@cite)
operator splitting using the reaction tangent as proposed in [OgiBalPer:2023:seats](@cite)
The next timestep length is calculated as
```math
\\sigma\\left(R_{\\max }\\right):=\\left(1.0-\\frac{1}{1+\\exp \\left(\\left(\\sigma_{\\mathrm{c}}-R_{\\max }\\right) \\cdot \\sigma_{\\mathrm{s}}\\right)}\\right) \\cdot\\left(\\Delta t_{\\max }-\\Delta t_{\\min }\\right)+\\Delta t_{\\min }
```
# Fields
- `ltg`::`LTG`: `LieTrotterGodunov` algorithm
- `σ_s::T`: steepness
- `σ_c::T`: offset in R axis
- `Δt_bounds::NTuple{2,T}`: lower and upper timestep length bounds
"""
struct ReactionTangentController{LTG <: OS.LieTrotterGodunov, T <: Real} <: OS.AbstractOperatorSplittingAlgorithm
    ltg::LTG
    σ_s::T
    σ_c::T
    Δt_bounds::NTuple{2,T}
end

mutable struct ReactionTangentControllerCache{T <: Real, LTGCache <: OS.LieTrotterGodunovCache, uType} <: OS.AbstractOperatorSplittingCache
    const ltg_cache::LTGCache
    u::uType
    uprev::uType # True previous solution
    R::T
    function ReactionTangentControllerCache(ltg_cache::LTGCache, R::T) where {T, LTGCache <: OS.LieTrotterGodunovCache}
        uType = typeof(ltg_cache.u)
        return new{T, LTGCache, uType}(ltg_cache, ltg_cache.u, ltg_cache.uprev, R)
    end
end

@inline DiffEqBase.get_tmp_cache(integrator::OS.OperatorSplittingIntegrator, alg::OS.AbstractOperatorSplittingAlgorithm, cache::ReactionTangentControllerCache) = DiffEqBase.get_tmp_cache(integrator, alg, cache.ltg_cache)

@inline function OS.advance_solution_to!(outer_integrator::OS.OperatorSplittingIntegrator, subintegrators::Tuple, solution_indices::Tuple, synchronizers::Tuple, cache::ReactionTangentControllerCache, tnext)
    OS.advance_solution_to!(outer_integrator, subintegrators, solution_indices, synchronizers, cache.ltg_cache, tnext)
end

@inline DiffEqBase.isadaptive(::ReactionTangentController) = true

"""
    get_reaction_tangent(integrator::OS.OperatorSplittingIntegrator)
Returns the maximal reaction magnitude using the [`PointwiseODEFunction`](@ref) of an operator splitting integrator that uses [`LieTrotterGodunov`](@ref) [Lie:1880:tti,Tro:1959:psg,God:1959:dmn](@cite).
It is assumed that the problem containing the reaction tangent is a [`PointwiseODEFunction`](@ref).
"""
@inline function get_reaction_tangent(integrator::OS.OperatorSplittingIntegrator)
    R, _ = _get_reaction_tangent(integrator.subintegrator_tree)
    return R
end

@inline @unroll function _get_reaction_tangent(subintegrators, n_reaction_tangents::Int = 0)
    R = 0.0
    @unroll for subintegrator in subintegrators
        if subintegrator isa Tuple
            R, n_reaction_tangents = _get_reaction_tangent(subintegrator, n_reaction_tangents)
        elseif subintegrator.f isa PointwiseODEFunction
            n_reaction_tangents += 1
            φₘidx = transmembranepotential_index(subintegrator.f.ode)
            R = max(R, maximum(@view subintegrator.cache.dumat[:, φₘidx]))
        end
    end
    @assert n_reaction_tangents == 1 "No or multiple integrators using PointwiseODEFunction found"
    return (R, n_reaction_tangents)
end

@inline function OS.stepsize_controller!(integrator::OS.OperatorSplittingIntegrator, alg::ReactionTangentController)
    integrator.cache.R = get_reaction_tangent(integrator)
    return nothing
end

@inline function OS.step_accept_controller!(integrator::OS.OperatorSplittingIntegrator, alg::ReactionTangentController, q)
    @unpack R = integrator.cache
    @unpack σ_s, σ_c, Δt_bounds = alg

    if isinf(σ_s)
        integrator.dt = R > σ_c ? Δt_bounds[1] : Δt_bounds[2]
    else
        integrator.dt = (1 - 1/(1+exp((σ_c - R)*σ_s)))*(Δt_bounds[2] - Δt_bounds[1]) + Δt_bounds[1]
    end
    return nothing
end

@inline function OS.step_reject_controller!(integrator::OS.OperatorSplittingIntegrator, alg::ReactionTangentController, q)
    if integrator.dt ≤ Δt_bounds[1] # Check for "≤" to also handle the boundary cases
        error("RTC cannot recover from step rejection below Δt min") # Force failure
    else
        integrator.dt = Δt_bounds[1]
    end
    return nothing # Do nothing
end

function OS.build_subintegrator_tree_with_cache(
    prob::OS.OperatorSplittingProblem, alg::ReactionTangentController,
    uprevouter::AbstractVector, uouter::AbstractVector,
    solution_indices,
    t0, dt, tf,
    tstops, saveat, d_discontinuities, callback,
    adaptive, verbose,
)
    subintegrators, inner_cache = OS.build_subintegrator_tree_with_cache(
        prob, alg.ltg, uprevouter, uouter, solution_indices,
        t0, dt, tf,
        tstops, saveat, d_discontinuities, callback,
        adaptive, verbose,
    )

    return subintegrators, ReactionTangentControllerCache(
        inner_cache,
        zero(eltype(uouter)),
    )
end
