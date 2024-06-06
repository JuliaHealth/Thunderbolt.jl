"""
    NewtonRaphsonSolver{T}

Classical Newton-Raphson solver to solve nonlinear problems of the form `F(u) = 0`.
To use the Newton-Raphson solver you have to dispatch on
* [update_linearization!](@ref)
"""
Base.@kwdef struct NewtonRaphsonSolver{T, solverType} <: AbstractNonlinearSolver
    # Convergence tolerance
    tol::T = 1e-4
    # Maximum number of iterations
    max_iter::Int = 100
    inner_solver::solverType = LinearSolve.KrylovJL_GMRES()
end

mutable struct NewtonRaphsonSolverCache{OpType, ResidualType, T, InnerSolverCacheType} <: AbstractNonlinearSolverCache
    # The nonlinear operator
    op::OpType
    # Cache for the right hand side f(u)
    residual::ResidualType
    #
    const parameters::NewtonRaphsonSolver{T}
    linear_solver_cache::InnerSolverCacheType
end

function setup_solver_cache(f::AbstractSemidiscreteFunction, solver::NewtonRaphsonSolver{T}) where {T}
    @unpack inner_solver = solver
    op = setup_operator(f, solver)
    residual = Vector{T}(undef, solution_size(f))
    Δu = Vector{T}(undef, solution_size(f))

    # Connect both solver caches
    inner_prob = LinearSolve.LinearProblem(
        getJ(op), residual; Δu
    )
    inner_cache = init(inner_prob, inner_solver)
    
    NewtonRaphsonSolverCache(op, residual, solver, inner_cache)
end

function setup_solver_cache(f::AbstractSemidiscreteBlockedFunction, solver::NewtonRaphsonSolver{T}) where {T}
    residual_buffer = mortar([
        Vector{T}(undef, bsize) for bsize ∈ blocksizes(f)
    ])
    NewtonRaphsonSolverCache(setup_operator(f, solver), residual_buffer, solver)
end

function nlsolve!(u::AbstractVector, f::AbstractSemidiscreteFunction, cache::NewtonRaphsonSolverCache, t)
    @unpack op, residual, linear_solver_cache = cache
    newton_itr = -1
    Δu = linear_solver_cache.u
    while true
        newton_itr += 1

        residual .= 0.0
        @timeit_debug "update operator" update_linearization!(op, u, residual, t)
        @timeit_debug "elimination" eliminate_constraints_from_linearization!(cache, f)
        # vtk_grid("newton-debug-$newton_itr", problem.structural_problem.dh) do vtk
        #     vtk_point_data(vtk, f.structural_problem.dh, u[Block(1)])
        #     vtk_point_data(vtk, f.structural_problem.dh, residual[Block(1)], :residual)
        # end
        residualnorm = residual_norm(cache, f)
        @info "Newton itr $newton_itr: ||r||=$residualnorm"
        if residualnorm < cache.parameters.tol #|| (newton_itr > 0 && norm(Δu) < cache.parameters.tol)
            break
        elseif newton_itr > cache.parameters.max_iter
            @warn "Reached maximum Newton iterations. Aborting. ||r|| = $residualnorm"
            return false
        elseif any(isnan.(residualnorm))
            @warn "Newton-Raphson diverged. Aborting. ||r|| = $residualnorm"
            return false
        end

        @timeit_debug "solve" sol = LinearSolve.solve!(linear_solver_cache)
        @info sol.stats
        solve_succeeded = LinearSolve.SciMLBase.successful_retcode(sol.retcode) || sol.retcode == LinearSolve.ReturnCode.Default # The latter seems off...
        solve_succeeded || return false

        eliminate_constraints_from_increment!(Δu, f, cache)

        u .-= Δu # Current guess
    end
    return true
end
