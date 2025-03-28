Base.@kwdef struct GenericLocalNonlinearSolver <: AbstractNonlinearSolver
    max_iters::Int = 10
    tol::Float64 = 1e-4
end

Base.@kwdef mutable struct GenericLocalNonlinearSolverCache{JacobianType, ResidualType, CorrectorRhsType}
    const params::GenericLocalNonlinearSolver
    const J::JacobianType
    const residual::ResidualType
    const rhs_corrector::CorrectorRhsType
    outer_tol::Float64 = Inf
    retcode::SciMLBase.ReturnCode.T = SciMLBase.ReturnCode.Default
end

"""
    MultilevelNewtonRaphsonSolver{T}

Multilevel Newton-Raphson solver [RabSanHsu:1979:mna](@ref) for nonlinear problems of the form `F(u,v) = 0; G(u,v) = 0`.
To use the Multilevel solver you have to dispatch on
* [update_linearization!](@ref)
"""
Base.@kwdef struct MultiLevelNewtonRaphsonSolver{gSolverType <: NewtonRaphsonSolver, lSolverType} <: AbstractNonlinearSolver
    newton::gSolverType = NewtonRaphsonSolver()
    local_solver::lSolverType = GenericLocalNonlinearSolver()
end

struct MultiLevelNewtonRaphsonSolverCache{gCacheType, lCacheType} <: AbstractNonlinearSolverCache
    global_solver_cache::gCacheType
    local_solver_cache::lCacheType
end

function nlsolve!(u::AbstractVector, f::AbstractSemidiscreteFunction, mlcache::MultiLevelNewtonRaphsonSolverCache, t)
    cache = mlcache.global_solver_cache

    @unpack op, residual, linear_solver_cache, Θks = cache
    monitor = cache.parameters.monitor
    cache.iter = -1
    Δu = linear_solver_cache.u
    residualnormprev = 0.0
    Θ1prev = length(Θks) > 0 ? first(Θks) : 0.0
    resize!(Θks, 0)
    mlcache.local_solver_cache.outer_tol = Inf
    while true
        cache.iter += 1
        residual .= 0.0
        @timeit_debug "update operator" update_linearization!(op, residual, u, t)
        # Check if local solve failed
        if mlcache.local_solver_cache.retcode ∉ (SciMLBase.ReturnCode.Default, SciMLBase.ReturnCode.Success)
            @debug "Some local newton did not converge. Aborting. ||r|| = $residualnorm" _group=:nlsolve
            return false
        end
        @timeit_debug "elimination" eliminate_constraints_from_linearization!(cache, f)
        linear_solver_cache.isfresh = true # Notify linear solver that we touched the system matrix

        residualnorm = residual_norm(cache, f)
        mlcache.local_solver_cache.outer_tol = residualnorm
        if residualnorm < cache.parameters.tol && cache.iter > 1 # Do at least two iterations to get a sane convergence estimate
            break
        elseif cache.iter > cache.parameters.max_iter
            @debug "Reached maximum Newton iterations. Aborting. ||r|| = $residualnorm" _group=:nlsolve
            return false
        elseif any(isnan.(residualnorm))
            @debug "Newton-Raphson diverged. Aborting. ||r|| = $residualnorm" _group=:nlsolve
            return false
        end

        @timeit_debug "solve" sol = LinearSolve.solve!(linear_solver_cache)
        nonlinear_step_monitor(cache, t, f, cache.parameters.monitor)
        solve_succeeded = LinearSolve.SciMLBase.successful_retcode(sol) || sol.retcode == LinearSolve.ReturnCode.Default # The latter seems off...
        solve_succeeded || return false

        eliminate_constraints_from_increment!(Δu, f, cache)

        u[1:ndofs(op.dh)] .-= Δu # Current guess

        if cache.iter > 0
            # In this case we might be unablet to estimate the convergence rate, because we are too close to the solution
            if residualnormprev < cache.parameters.tol && residualnorm < cache.parameters.tol
                push!(Θks, Θ1prev^2)
                break
            end
            Θk = residualnorm/residualnormprev
            push!(Θks, isnan(Θk) ? 0.0 : Θk)
            if Θk ≥ 1.0
                @debug "Newton-Raphson diverged. Aborting. ||r|| = $residualnorm" _group=:nlsolve
                return false
            end

            # Late out on second iteration
            if residualnorm < cache.parameters.tol
                break
            end
        end

        residualnormprev = residualnorm
    end
    nonlinear_finalize_monitor(cache, t, f, monitor)
    return true
end
