"""
    EisenstatWalkerForcing

Eisenstat-Walker Algorithm 2 from Eisenstat & Walker (1996) for adaptive inner solver
tolerances in Newton's method. On each Newton step k, the relative tolerance for the
linear solve is set to:

    ηₖ = γ · (‖rₖ‖ / ‖rₖ₋₁‖)^α

with an optional safeguard to prevent ηₖ from dropping too fast:

    ηₖ = max(ηₖ, γ·ηₖ₋₁^α)  if  γ·ηₖ₋₁^α > safeguard_threshold

Only active when the linear solver is iterative (a Krylov subspace method).
"""
struct EisenstatWalkerForcing{T <: AbstractFloat}
    η₀::T
    ηₘₐₓ::T
    γ::T
    α::T
    safeguard::Bool
    safeguard_threshold::T
end

function EisenstatWalkerForcing(;
    η₀                  = 0.5,
    ηₘₐₓ                = 0.9,
    γ                   = 0.9,
    α                   = 2.0,
    safeguard           = true,
    safeguard_threshold = 0.1,
)
    T = promote_type(typeof(η₀), typeof(ηₘₐₓ), typeof(γ), typeof(α), typeof(safeguard_threshold))
    return EisenstatWalkerForcing{T}(η₀, ηₘₐₓ, γ, α, safeguard, safeguard_threshold)
end

mutable struct EisenstatWalkerForcingCache{T}
    η::T
    rnorm::T            # residual norm of the previous Newton step
    const p::EisenstatWalkerForcing{T}
end

"""
    NewtonRaphsonSolver{T}

Classical Newton-Raphson solver to solve nonlinear problems of the form `F(u) = 0`.

It solves an [`AbstractStageFunction`](@ref), so what it needs from a problem is that stage's
[`update_stage_linearization!`](@ref) and [`evaluate_stage_residual!`](@ref), not a method on the
semidiscrete function itself.

If `simplified_newton = true`, the Jacobian (and preconditioner) assembled at the first
Newton iteration is reused for all subsequent iterations. Only the residual is recomputed
via [`evaluate_stage_residual!`](@ref) each step. This saves Jacobian assembly and factorization
cost per step at the expense of slower outer convergence.

Either way the Jacobian is assembled only for a step that is taken, so after a solve it belongs to the
iterate before the converged one.
"""
Base.@kwdef struct NewtonRaphsonSolver{T} <: AbstractNonlinearSolver
    # Convergence tolerance
    tol::T = 1e-4
    # Maximum number of iterations
    max_iter::Int = 100
    # Read once, by `setup_solver_cache`. Untyped because the field takes both a `LinearSolve`
    # algorithm and a `KrylovMGSolver` description, which share no supertype -- and because the whole
    # solver stack above it (this solver, its cache, the stage, the time scheme, the integrator)
    # would otherwise be respecialized per linear solver choice for a value read at setup.
    inner_solver::Any = LinearSolve.KrylovJL_GMRES()
    # Called once per Newton iteration, so dynamic dispatch here costs nothing measurable. Untyped
    # because the monitor protocol is a pair of methods (`nonlinear_step_monitor`,
    # `nonlinear_finalize_monitor`), not a subtype relation, so a user monitor is any type at all.
    monitor::Any = DefaultProgressMonitor()
    enforce_monotonic_convergence::Bool = true
    # Adaptive linear solver tolerance (Eisenstat-Walker); only active for iterative solvers.
    forcing::Union{Nothing, EisenstatWalkerForcing} = nothing
    # When true, reuse the Jacobian and preconditioner from the first Newton iteration.
    simplified_newton::Bool = false
end

# The operator is deliberately absent: it belongs to the `AbstractStageFunction` being solved, which
# is what knows which nonlinear problem this is. One owner, so a scheme cannot hand the solver one
# operator and the cache another.
mutable struct NewtonRaphsonSolverCache{ResidualType, T} <: AbstractNonlinearSolverCache
    # Cache for the right hand side f(u)
    residual::ResidualType
    #
    const parameters::NewtonRaphsonSolver{T}
    # LinearSolve's cache is legitimately specialized on its algorithm; nothing above it is. Held
    # untyped so this cache -- and every stage and scheme cache that holds one -- has the same type
    # for every linear solver. `_newton_increment_step!` is the function barrier that recovers
    # specialization for the vector sized work: one dispatch per Newton iteration.
    linear_solver_cache::Any
    forcing_cache::Union{Nothing, EisenstatWalkerForcingCache{T}}
    Θks::Vector{T} # TODO modularize this
    #
    iter::Int
end

"""
    global_newton_cache(cache)

The Newton cache carrying the global convergence history (`Θks`) and the Newton
parameters. A plain Newton cache is its own; a wrapping cache (multi-level) returns the
cache it wraps. The Deuflhard continuation controllers go through this instead of reaching
for `cache.Θks` directly, which is a layout only one of the two cache types has.
"""
global_newton_cache(cache) = cache

function Base.show(io::IO, cache::NewtonRaphsonSolverCache)
    println(io, "NewtonRaphsonSolverCache:")
    Base.show(io, cache.parameters)
end

"""
    setup_solver_cache(sf::AbstractStageFunction, solver::NewtonRaphsonSolver)

Allocate the Newton work buffers for the stage `sf`.

The residual is sized by the *linear system*, i.e. by [`uncondensed_range`](@ref), not by the stage's
unknowns. The two coincide unless the function condenses internal variables at quadrature point
level, in which case the condensed tail lives in the solution vector but has no equation in the
global system -- and `apply_zero!` would refuse a right hand side longer than the matrix.
"""
function setup_solver_cache(sf::AbstractStageFunction, solver::NewtonRaphsonSolver{T}) where {T}
    @unpack inner_solver = solver
    f = getfunction(sf)
    op = getoperator(sf)
    J = getJ(op)
    nlinear = length(uncondensed_range(sf))
    residual = Vector{T}(undef, nlinear)
    Δu = Vector{T}(undef, nlinear)

    # Connect both solver caches
    inner_prob  = LinearSolve.LinearProblem(J, residual; u0 = Δu)
    maxiters    = _linear_maxiters(inner_solver)
    init_kw     = maxiters === nothing ? (;) : (; maxiters = maxiters)
    inner_cache = init(inner_prob, _materialize_inner_solver(f, inner_solver); alias = LinearAliasSpecifier(alias_A = true, alias_b = true), init_kw...)
    @assert inner_cache.b === residual
    @assert inner_cache.A === J

    NewtonRaphsonSolverCache(
        residual,
        solver,
        inner_cache,
        _build_forcing_cache(solver.forcing, inner_cache, T),
        T[],
        0,
    )
end

# Build the Eisenstat-Walker forcing cache only when the linear solver is a Krylov
# (iterative) method — direct factorizations have no tolerance to adapt.
_build_forcing_cache(::Nothing, inner_cache, ::Type) = nothing
function _build_forcing_cache(f::EisenstatWalkerForcing, inner_cache, ::Type{T}) where {T}
    if !(inner_cache.alg isa LinearSolve.AbstractKrylovSubspaceMethod)
        @warn "EisenstatWalkerForcing requires a Krylov linear solver; adaptive tolerance disabled." maxlog=1
        return nothing
    end
    return EisenstatWalkerForcingCache(
        T(f.η₀),
        typemax(T),
        EisenstatWalkerForcing{T}(f.η₀, f.ηₘₐₓ, f.γ, f.α, f.safeguard, f.safeguard_threshold),
    )
end

# No-op for direct solvers or when forcing is disabled.
_ew_prestep!(::Nothing, linear_solver_cache, residualnorm, iter) = nothing
function _ew_prestep!(fc::EisenstatWalkerForcingCache, linear_solver_cache, residualnorm, iter)
    p = fc.p
    if iter == 0
        fc.η = min(p.η₀, p.ηₘₐₓ)
    else
        ηprev = fc.η
        η = p.γ * (residualnorm / fc.rnorm)^p.α
        if p.safeguard
            ηsg = p.γ * ηprev^p.α
            if ηsg > p.safeguard_threshold && ηsg > η
                η = ηsg
            end
        end
        fc.η = clamp(η, zero(η), p.ηₘₐₓ)
    end
    fc.rnorm = residualnorm
    LinearSolve.update_tolerances!(linear_solver_cache; reltol = fc.η)
    @debug "Eisenstat-Walker η=$(fc.η) at iter=$iter" _group=:nlsolve
    return nothing
end

"""
    _newton_increment_step!(u, sf, cache, linear_solver_cache, t, f, reset_increment) -> (ok, ‖Δu‖)

Solve `J Δu = r` for the current Newton increment and subtract it from `u`.

The linear cache is an argument rather than a field read of `cache`. [`NewtonRaphsonSolverCache`](@ref)
holds it untyped so that the Newton stack is not specialized on the linear algorithm; this call is
the one place per Newton iteration where that costs a dispatch, and every vector sized operation
happens on the specialized side of it.

`reset_increment` zeroes `Δu` before the solve. The Eisenstat-Walker criterion is relative to ‖r₀‖,
which a warm started increment satisfies trivially — Krylov.jl would then return after 0–1 steps.

‖Δu‖ is returned in the residual's precision, which is the one the convergence history `Θks` is kept
in. That also keeps the return type inferable at the call sites, where only `linear_solver_cache` is
of unknown type.

Shared by the plain and the multilevel Newton, whose linear steps are the same.
"""
function _newton_increment_step!(u, sf, cache, linear_solver_cache, t, f, reset_increment::Bool)
    T = eltype(cache.residual)
    Δu = linear_solver_cache.u
    reset_increment && fill!(Δu, zero(eltype(Δu)))
    @timeit_debug "solve" sol = LinearSolve.solve!(linear_solver_cache)
    nonlinear_step_monitor(cache, t, f, u, cache.parameters.monitor)
    solve_succeeded =
        LinearSolve.SciMLBase.successful_retcode(sol) ||
        sol.retcode == LinearSolve.ReturnCode.Default # The latter seems off...
    solve_succeeded || return (false, zero(T))

    eliminate_constraints_from_increment!(Δu, sf, cache)
    # Only the entries the linear system solves for; the condensed tail is written by the assembly,
    # not by the increment.
    @inbounds @views u[uncondensed_range(sf)] .-= Δu
    return (true, convert(T, norm(Δu)))
end

"""
    _update_newton_linearization!(cache, sf, u) -> Bool

Assemble the residual and the Jacobian at `u`, eliminate the constraints from both and notify the
linear solver that its matrix and preconditioner are out of date. `false` if the stage failed.
"""
function _update_newton_linearization!(cache::NewtonRaphsonSolverCache, sf, u)
    (; residual, linear_solver_cache) = cache
    fill!(residual, 0.0)
    @timeit_debug "update operator" update_stage_linearization!(sf, residual, u) || return false
    @timeit_debug "elimination" eliminate_constraints_from_linearization!(cache, sf)
    linear_solver_cache.isfresh = true
    linear_solver_cache.precsisfresh = true
    return true
end

"""
    nlsolve!(z, sf::AbstractStageFunction, cache, t)

Solve the stage `sf` for its unknowns `z`.

`t` is the time, used for monitoring only. Everything the operator needs travels in
`stage_parameters(sf)`, a [`StageEvaluation`](@ref): the time the step is solved at is its `ctx`,
the previous solution and reconstructed rates are its `slots`, and its `p` is the parameter bag
`FerriteOperators` hands the elements through `query_cell_parameters`.

`p` carries the **parameters being optimized**, so that a solve can be differentiated with respect
to them. Note "being optimized", not "the model's parameters": a material with ten parameters of
which nine are known from experiment contributes exactly one entry to `p`, so calibrating it is a 1D
problem rather than a 10D one. The known nine stay in the model struct. Time and history are `ctx`
and `slots` and never belong here.

A stage with nothing to optimize leaves `p` at `nothing`, which is the common case;
`RSAFDQ20223DFunction` is the exception, passing the solver-supplied chamber reference volumes.
"""
function nlsolve!(
    u::AbstractVector{T},
    sf::AbstractStageFunction,
    cache::NewtonRaphsonSolverCache,
    t,
) where {T}
    f = getfunction(sf)
    op = getoperator(sf)
    @unpack residual, linear_solver_cache, Θks = cache
    monitor = cache.parameters.monitor
    simplified = cache.parameters.simplified_newton
    cache.iter = -1
    residualnormprev = 0.0
    incrementnormprev = 0.0
    resize!(Θks, 0)
    while true
        cache.iter += 1
        # Iteration 0 always takes a step. After that the residual alone decides whether another one
        # follows, so the converged iterate costs no Jacobian assembly and factorization.
        if cache.iter == 0
            _update_newton_linearization!(cache, sf, u) || return false
        else
            fill!(residual, 0.0)
            @timeit_debug "update residual" evaluate_stage_residual!(sf, residual, u) ||
                                            return false
            @timeit_debug "elimination" eliminate_constraints_from_residual!(cache, sf)
        end

        residualnorm = residual_norm(cache, sf)
        if residualnorm < cache.parameters.tol && cache.iter > 0
            push!(Θks, 0.0)
            break
        elseif cache.iter > cache.parameters.max_iter
            push!(Θks, Inf)
            @debug "Reached maximum Newton iterations. Aborting. ||r|| = $residualnorm" _group=:nlsolve
            return false
        elseif any(isnan.(residualnorm))
            push!(Θks, Inf)
            @debug "Newton-Raphson diverged. Aborting. ||r|| = $residualnorm" _group=:nlsolve
            return false
        end

        # Simplified Newton: reuse Jacobian and preconditioner from iter 0.
        if cache.iter > 0 && !simplified
            _update_newton_linearization!(cache, sf, u) || return false
        end

        _ew_prestep!(cache.forcing_cache, linear_solver_cache, residualnorm, cache.iter)
        solve_succeeded, incrementnorm = _newton_increment_step!(
            u,
            sf,
            cache,
            linear_solver_cache,
            t,
            f,
            cache.forcing_cache !== nothing,
        )
        solve_succeeded || return false

        if cache.iter > 0
            Θk = min(residualnorm/residualnormprev, incrementnorm/incrementnormprev)
            if residualnormprev ≈ 0.0 || incrementnormprev ≈ 0.0
                push!(Θks, 0.0)
            else
                push!(Θks, Θk)
            end
            # Try to prevent oversolving when we really just wanted to force the solve to happen once.
            if residualnorm < eps(T) || incrementnorm < eps(T)
                break
            end
            if cache.parameters.enforce_monotonic_convergence && Θk ≥ 1.0
                @debug "Newton-Raphson diverged. Aborting. ||r|| = $residualnorm" _group=:nlsolve
                return false
            end
        end

        # if incrementnorm < cache.parameters.tol
        #     break
        # end

        residualnormprev  = residualnorm
        incrementnormprev = incrementnorm
    end
    nonlinear_finalize_monitor(cache, t, f, monitor)
    return true
end
