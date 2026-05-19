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
    η₀               = 0.5,
    ηₘₐₓ             = 0.9,
    γ                = 0.9,
    α                = 2.0,
    safeguard        = true,
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
To use the Newton-Raphson solver you have to dispatch on
* [update_linearization!](@ref)

If `simplified_newton = true`, the Jacobian (and preconditioner) assembled at the first
Newton iteration is reused for all subsequent iterations. Only the residual is recomputed
via [`residual!`](@ref) each step. This saves Jacobian assembly and factorization cost per
step at the expense of slower outer convergence.
"""
Base.@kwdef struct NewtonRaphsonSolver{T, solverType, MonitorType, ForcingType} <: AbstractNonlinearSolver
    # Convergence tolerance
    tol::T = 1e-4
    # Maximum number of iterations
    max_iter::Int = 100
    inner_solver::solverType = LinearSolve.KrylovJL_GMRES()
    monitor::MonitorType = DefaultProgressMonitor()
    enforce_monotonic_convergence::Bool = true
    # Adaptive linear solver tolerance (Eisenstat-Walker); only active for iterative solvers.
    forcing::ForcingType = nothing
    # When true, reuse the Jacobian and preconditioner from the first Newton iteration.
    simplified_newton::Bool = false
end

mutable struct NewtonRaphsonSolverCache{
    OpType,
    ResidualType,
    T,
    NewtonType <: NewtonRaphsonSolver{T},
    InnerSolverCacheType,
    ForcingCacheType,
} <: AbstractNonlinearSolverCache
    # The nonlinear operator
    op::OpType
    # Cache for the right hand side f(u)
    residual::ResidualType
    #
    const parameters::NewtonType
    linear_solver_cache::InnerSolverCacheType
    forcing_cache::ForcingCacheType
    Θks::Vector{T} # TODO modularize this
    #
    iter::Int
end

function Base.show(io::IO, cache::NewtonRaphsonSolverCache)
    println(io, "NewtonRaphsonSolverCache:")
    Base.show(io, cache.parameters)
    Base.show(io, cache.op)
end

function setup_solver_cache(
    f::AbstractSemidiscreteFunction,
    solver::NewtonRaphsonSolver{T},
) where {T}
    @unpack inner_solver = solver
    op = setup_operator(f, solver)
    residual = Vector{T}(undef, solution_size(f))
    Δu = Vector{T}(undef, solution_size(f))

    # Connect both solver caches
    inner_prob = LinearSolve.LinearProblem(op.J, residual; u0 = Δu)
    maxiters    = _linear_maxiters(inner_solver)
    init_kw    = maxiters === nothing ? (;) : (; maxiters = maxiters)
    inner_cache =
        init(inner_prob, _materialize_inner_solver(f, inner_solver);
             alias = LinearAliasSpecifier(alias_A = true, alias_b = true), init_kw...)
    @assert inner_cache.b === residual
    @assert inner_cache.A === op.J

    NewtonRaphsonSolverCache(op, residual, solver, inner_cache, _build_forcing_cache(solver.forcing, inner_cache, T), T[], 0)
end

function setup_solver_cache(
    f::AbstractSemidiscreteBlockedFunction,
    solver::NewtonRaphsonSolver{T},
) where {T}
    @unpack inner_solver = solver
    op = setup_operator(f, solver)
    sizeu = solution_size(f)
    residual = Vector{T}(undef, sizeu)
    Δu = Vector{T}(undef, sizeu)
    # Connect both solver caches
    inner_prob = LinearSolve.LinearProblem(op.J, residual; u0 = Δu)
    maxiters    = _linear_maxiters(inner_solver)
    init_kw    = maxiters === nothing ? (;) : (; maxiters = maxiters)
    inner_cache =
        init(inner_prob, _materialize_inner_solver(f, inner_solver);
             alias = LinearAliasSpecifier(alias_A = true, alias_b = true), init_kw...)
    @assert inner_cache.alg == inner_solver
    @assert inner_cache.b === residual
    @assert inner_cache.A === op.J

    NewtonRaphsonSolverCache(op, residual, solver, inner_cache, _build_forcing_cache(solver.forcing, inner_cache, T), T[], 0)
end

# Build the Eisenstat-Walker forcing cache only when the linear solver is a Krylov
# (iterative) method — direct factorizations have no tolerance to adapt.
_build_forcing_cache(::Nothing, inner_cache, ::Type) = nothing
function _build_forcing_cache(f::EisenstatWalkerForcing, inner_cache, ::Type{T}) where T
    if !(inner_cache.alg isa LinearSolve.AbstractKrylovSubspaceMethod)
        @warn "EisenstatWalkerForcing requires a Krylov linear solver; adaptive tolerance disabled." maxlog=1
        return nothing
    end
    return EisenstatWalkerForcingCache(T(f.η₀), typemax(T), EisenstatWalkerForcing{T}(f.η₀, f.ηₘₐₓ, f.γ, f.α, f.safeguard, f.safeguard_threshold))
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

function nlsolve!(
    u::AbstractVector{T},
    f::AbstractSemidiscreteFunction,
    cache::NewtonRaphsonSolverCache,
    t,
) where {T}
    @unpack op, residual, linear_solver_cache, Θks = cache
    monitor = cache.parameters.monitor
    simplified = cache.parameters.simplified_newton
    cache.iter = -1
    Δu = linear_solver_cache.u
    residualnormprev = 0.0
    incrementnormprev = 0.0
    resize!(Θks, 0)
    while true
        cache.iter += 1
        fill!(residual, 0.0)
        if simplified && cache.iter > 0
            # Simplified Newton: reuse Jacobian and preconditioner from iter 0.
            @timeit_debug "update residual" residual!(op, residual, u, t)
            @timeit_debug "elimination" eliminate_constraints_from_residual!(cache, f)
            # Leave isfresh / precsisfresh false → reuse existing factorization.
        else
            @timeit_debug "update operator" update_linearization!(op, residual, u, t)
            @timeit_debug "elimination" eliminate_constraints_from_linearization!(cache, f)
            linear_solver_cache.isfresh = true        # Notify linear solver that both the matrix and the preconditioner need to be updated.
            linear_solver_cache.precsisfresh = true
        end

        residualnorm = residual_norm(cache, f)
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

        _ew_prestep!(cache.forcing_cache, linear_solver_cache, residualnorm, cache.iter)
        # The Eisenstat-Walker analysis assumes a fresh start (x₀ = 0) so that the
        # initial residual equals the RHS and the tolerance η is meaningful relative
        # to ‖b‖.  With a warm-started Δu, Krylov.jl's criterion ‖r‖ ≤ η·‖r₀‖ is
        # trivially met in 0–1 steps because ‖r₀‖ << ‖b‖.
        cache.forcing_cache !== nothing && fill!(Δu, zero(T))
        @timeit_debug "solve" sol = LinearSolve.solve!(linear_solver_cache)
        nonlinear_step_monitor(cache, t, f, u, cache.parameters.monitor)
        solve_succeeded =
            LinearSolve.SciMLBase.successful_retcode(sol) ||
            sol.retcode == LinearSolve.ReturnCode.Default # The latter seems off...
        solve_succeeded || return false

        eliminate_constraints_from_increment!(Δu, f, cache)

        u .-= Δu # Current guess
        incrementnorm = norm(Δu)

        if cache.iter > 0
            Θk = min(residualnorm/residualnormprev, incrementnorm/incrementnormprev)
            if residualnormprev ≈ 0.0 || incrementnormprev ≈ 0.0
                push!(Θks, 0.0)
            else
                push!(Θks, Θk)
            end
            # Try to prevent oversolving when we really just wanted to force the solve to happen once.
            if cache.iter == 1 &&
               residualnormprev < eps(T) &&
               residualnorm < eps(T) &&
               incrementnorm < eps(T) &&
               incrementnormprev < eps(T)
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
