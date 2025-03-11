"""
    NewtonRaphsonSolver{T}

Classical Newton-Raphson solver to solve nonlinear problems of the form `F(u) = 0`.
To use the Newton-Raphson solver you have to dispatch on
* [update_linearization!](@ref)
"""
Base.@kwdef struct NewtonRaphsonSolver{T, solverType, MonitorType} <: AbstractNonlinearSolver
    # Convergence tolerance
    tol::T = 1e-4
    # Maximum number of iterations
    max_iter::Int = 100
    inner_solver::solverType = LinearSolve.KrylovJL_GMRES()
    monitor::MonitorType = DefaultProgressMonitor()
    enforce_monotonic_convergence::Bool = false
end

mutable struct NewtonRaphsonSolverCache{OpType, ResidualType, T, NewtonType <: NewtonRaphsonSolver{T}, InnerSolverCacheType} <: AbstractNonlinearSolverCache
    # The nonlinear operator
    op::OpType
    # Cache for the right hand side f(u)
    residual::ResidualType
    #
    const parameters::NewtonType
    linear_solver_cache::InnerSolverCacheType
    Θks::Vector{T} # TODO modularize this
    #
    iter::Int
end

function setup_solver_cache(f::AbstractSemidiscreteFunction, solver::NewtonRaphsonSolver{T}) where {T}
    @unpack inner_solver = solver
    op = setup_operator(f, solver)
    residual = Vector{T}(undef, solution_size(f))
    Δu = Vector{T}(undef, solution_size(f))

    # Connect both solver caches
    inner_prob = LinearSolve.LinearProblem(
        getJ(op), residual; u0=Δu
    )
    inner_cache = init(inner_prob, inner_solver; alias_A=true, alias_b=true)
    @assert inner_cache.b === residual
    @assert inner_cache.A === getJ(op)

    NewtonRaphsonSolverCache(op, residual, solver, inner_cache, T[], 0)
end

function setup_solver_cache(f::AbstractSemidiscreteBlockedFunction, solver::NewtonRaphsonSolver{T}) where {T}
    @unpack inner_solver = solver
    op = setup_operator(f, solver)
    sizeu = solution_size(f)
    residual = Vector{T}(undef, sizeu)
    Δu = Vector{T}(undef, sizeu)
    # Connect both solver caches
    inner_prob = LinearSolve.LinearProblem(
        getJ(op), residual; u0=Δu
    )
    inner_cache = init(inner_prob, inner_solver; alias_A=true, alias_b=true)
    @assert inner_cache.b === residual
    @assert inner_cache.A === getJ(op)

    NewtonRaphsonSolverCache(op, residual, solver, inner_cache, T[], 0)
end

function nlsolve!(u::AbstractVector, f::AbstractSemidiscreteFunction, cache::NewtonRaphsonSolverCache, t)
    @unpack op, residual, linear_solver_cache, Θks = cache
    monitor = cache.parameters.monitor
    cache.iter = -1
    Δu = linear_solver_cache.u
    residualnormprev = 0.0
    incrementnormprev = 0.0
    resize!(Θks, 0)
    while true
        cache.iter += 1
        fill!(residual, 0.0)
        @timeit_debug "update operator" update_linearization!(op, residual, u, t)
        @timeit_debug "elimination" eliminate_constraints_from_linearization!(cache, f)
        linear_solver_cache.isfresh = true # Notify linear solver that we touched the system matrix

        residualnorm = residual_norm(cache, f)
        if residualnorm < cache.parameters.tol
            if cache.iter == 1
                push!(Θks, 0.0)
            end
            break
        elseif cache.iter > cache.parameters.max_iter
            push!(Θks,Inf)
            @debug "Reached maximum Newton iterations. Aborting. ||r|| = $residualnorm" _group=:nlsolve
            return false
        elseif any(isnan.(residualnorm))
            push!(Θks,Inf)
            @debug "Newton-Raphson diverged. Aborting. ||r|| = $residualnorm" _group=:nlsolve
            return false
        end

        @timeit_debug "solve" sol = LinearSolve.solve!(linear_solver_cache)
        nonlinear_step_monitor(cache, t, f, u, cache.parameters.monitor)
        solve_succeeded = LinearSolve.SciMLBase.successful_retcode(sol) || sol.retcode == LinearSolve.ReturnCode.Default # The latter seems off...
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
            if cache.parameters.enforce_monotonic_convergence && Θk ≥ 1.0
                @debug "Newton-Raphson diverged. Aborting. ||r|| = $residualnorm" _group=:nlsolve
                return false
            end

            # Late out on second iteration
            if residualnorm < cache.parameters.tol
                break
            end
        end

        residualnormprev  = residualnorm
        incrementnormprev = incrementnorm
    end
    nonlinear_finalize_monitor(cache, t, f, monitor)
    return true
end
