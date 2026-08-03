Base.@kwdef struct GenericLocalNonlinearSolver <: AbstractNonlinearSolver
    max_iters::Int = 10
    tol::Float64 = 1e-4
end

"""
    LocalSolveReport

Outcome of the local Newton at one quadrature point.
"""
struct LocalSolveReport
    retcode::SciMLBase.ReturnCode.T
    residual::Float64
end
LocalSolveReport() = LocalSolveReport(SciMLBase.ReturnCode.Default, 0.0)

_local_solve_failed(report::LocalSolveReport) =
    report.retcode ∉ (SciMLBase.ReturnCode.Default, SciMLBase.ReturnCode.Success)

"""
    setup_local_solve_reports(dh, lvh, ndofs_per_quadrature_point, cellset)

One report slot per quadrature point, in a single contiguous vector addressed per cell.

Each local solve writes exactly its own slot, so the store needs no counter and never grows -- which
is what makes it safe under threaded assembly and adaptable to a device. `duplicate_for_device`
shares it rather than copying it, so failures recorded by a worker remain visible to the global
solver.

The per-cell quadrature point count is recovered from the [`InternalVariableHandler`](@ref), which
lays out `nqp * ndofs_per_quadrature_point` condensed unknowns per cell. `cellset` restricts the
store to the subdomain this solver serves; cells outside it get an empty range.
"""
function setup_local_solve_reports(dh, lvh, ndofs_per_quadrature_point::Int, cellset)
    ndofs_per_quadrature_point == 0 && return nothing
    ncells     = getncells(get_grid(dh))
    ndofs_last = ndofs(dh) + ndofs(lvh)
    offsets    = Vector{Int}(undef, ncells+1)
    offsets[1] = 1
    for cellid = 1:ncells
        cell_end =
            cellid < ncells ? FerriteOperators.internal_variable_offset(lvh, cellid+1) : ndofs_last
        ndofs_cell = cell_end - FerriteOperators.internal_variable_offset(lvh, cellid)
        nqp =
            (cellset === nothing || cellid ∈ cellset) ? ndofs_cell ÷ ndofs_per_quadrature_point : 0
        offsets[cellid+1] = offsets[cellid] + nqp
    end
    return DenseDataRange(fill(LocalSolveReport(), offsets[end]-1), offsets)
end

"""
    GenericLocalNonlinearSolverCache

Immutable, so that it can be passed to a device kernel by value. Everything that changes during a
solve lives in the arrays it holds: the outer solver's current tolerance in `outer_tol`, and the
per-quadrature-point outcomes in `reports`.
"""
Base.@kwdef struct GenericLocalNonlinearSolverCache{
    JacobianType,
    ResidualType,
    CorrectorRhsType,
    ReportsType,
    TolType,
}
    params::GenericLocalNonlinearSolver
    J::JacobianType
    residual::ResidualType
    rhs_corrector::CorrectorRhsType
    reports::ReportsType = nothing
    outer_tol::TolType = [0.0]
end

function duplicate_for_device(device, cache::GenericLocalNonlinearSolverCache)
    GenericLocalNonlinearSolverCache(;
        params        = cache.params,
        J             = duplicate_for_device(device, cache.J),
        residual      = duplicate_for_device(device, cache.residual),
        rhs_corrector = duplicate_for_device(device, cache.rhs_corrector),
        # Both are deliberately shared rather than copied: workers write disjoint report slots and a
        # failure must survive the worker, and the tolerance is written once per outer iteration and
        # has to reach every worker.
        reports   = cache.reports,
        outer_tol = cache.outer_tol,
    )
end

"""
    record_local_solve!(local_solver_cache, cellid, qpi, retcode, residualnorm)

Record the outcome of one local solve in the shared per-quadrature-point store.
"""
@inline function record_local_solve!(
    local_solver_cache::GenericLocalNonlinearSolverCache,
    cellid,
    qpi,
    retcode,
    residualnorm,
)
    reports = local_solver_cache.reports
    reports === nothing && return nothing
    get_data_for_index(reports, cellid)[qpi] = LocalSolveReport(retcode, residualnorm)
    return nothing
end

"""
    local_solve_report(local_solver_cache, cellid, qpi)

The outcome recorded for one quadrature point of the current assembly pass.
"""
@inline function local_solve_report(
    local_solver_cache::GenericLocalNonlinearSolverCache,
    cellid,
    qpi,
)
    reports = local_solver_cache.reports
    reports === nothing && return LocalSolveReport()
    return get_data_for_index(reports, cellid)[qpi]
end

function check_local_solve_covergence(local_solver_cache::GenericLocalNonlinearSolverCache)
    reports = local_solver_cache.reports
    reports === nothing && return false
    return any(_local_solve_failed, reports.data)
end
function check_local_solve_covergence(local_solver_cache::Tuple)
    return any(check_local_solve_covergence.(local_solver_cache))
end
function check_local_solve_covergence(local_solver_cache::AbstractVector)
    return any(check_local_solve_covergence.(local_solver_cache))
end

"""
    describe_local_solve_failures(local_solver_cache)

Every failing quadrature point of the last assembly pass, as `cell`/`qp` pairs with their local
residual norm and return code. Empty when nothing failed.
"""
function describe_local_solve_failures(local_solver_cache::GenericLocalNonlinearSolverCache)
    reports = local_solver_cache.reports
    reports === nothing && return ""
    io = IOBuffer()
    for cellid = 1:(length(reports.offsets)-1)
        for (qpi, report) in enumerate(get_data_for_index(reports, cellid))
            _local_solve_failed(report) || continue
            println(io, "  cell $cellid qp $qpi: $(report.retcode), ||r|| = $(report.residual)")
        end
    end
    return String(take!(io))
end
function describe_local_solve_failures(local_solver_cache::Union{Tuple, AbstractVector})
    return join(describe_local_solve_failures.(local_solver_cache))
end

"""
    reset_local_solve_status!(local_solver_cache)

Clear the recorded outcomes before an assembly pass, so `check_local_solve_covergence` reports on
*that* pass alone.

Without this the failures latch, and the first one would poison every later assembly — including any
retry of the step, which is the only way a local failure can ever be recovered from.

Note that recovering additionally requires the time integrator to actually retry at a shorter `dt`.
`BackwardEulerSolver` does not: it is not adaptive, so a local failure currently ends the solve with
`ConvergenceFailure`.
"""
function reset_local_solve_status!(local_solver_cache::GenericLocalNonlinearSolverCache)
    reports = local_solver_cache.reports
    reports === nothing || fill!(reports.data, LocalSolveReport())
    return nothing
end
function reset_local_solve_status!(local_solver_cache::Tuple)
    foreach(reset_local_solve_status!, local_solver_cache)
end
function reset_local_solve_status!(local_solver_cache::AbstractVector)
    foreach(reset_local_solve_status!, local_solver_cache)
end

function set_local_solver_tol(local_solver_cache::GenericLocalNonlinearSolverCache, tol)
    local_solver_cache.outer_tol[1] = tol
end
function set_local_solver_tol(local_solver_cache::Tuple, tol)
    set_local_solver_tol.(local_solver_cache, tol)
end
function set_local_solver_tol(local_solver_cache::AbstractVector, tol)
    set_local_solver_tol.(local_solver_cache, tol)
end

"""
    MultilevelNewtonRaphsonSolver{T}

Multilevel Newton-Raphson solver [RabSanHsu:1979:mna](@ref) for nonlinear problems of the form `F(u,v) = 0; G(u,v) = 0`.
To use the Multilevel solver you have to dispatch on
* [update_linearization!](@ref)
* [`residual!`](@ref), if the global Newton runs with `simplified_newton = true`

The global Newton's `simplified_newton` and `forcing` settings apply here as they do to the plain
[`NewtonRaphsonSolver`](@ref). Note what a simplified step does *not* skip: the local problems are
re-solved on every residual evaluation, because the residual is a function of the condensed state.
What is reused is the global Jacobian, so the local sensitivities are the part that is saved.
"""
Base.@kwdef struct MultiLevelNewtonRaphsonSolver{gSolverType <: NewtonRaphsonSolver, lSolverType} <:
                   AbstractNonlinearSolver
    newton::gSolverType = NewtonRaphsonSolver()
    local_solver::lSolverType = GenericLocalNonlinearSolver()
end

struct MultiLevelNewtonRaphsonSolverCache{gCacheType, lCacheType} <: AbstractNonlinearSolverCache
    global_solver_cache::gCacheType
    local_solver_cache::lCacheType
end

# `Θks` and the Newton parameters live on the global cache this one wraps.
global_newton_cache(cache::MultiLevelNewtonRaphsonSolverCache) = cache.global_solver_cache

function Base.show(io::IO, cache::MultiLevelNewtonRaphsonSolverCache)
    println(io, "MultiLevelNewtonRaphsonSolverCache:")
    Base.show(io, cache.global_solver_cache)
    if cache.local_solver_cache isa Tuple
        for local_solver_cache in cache.local_solver_cache
            Base.show(io, local_solver_cache)
        end
    else
        Base.show(io, cache.local_solver_cache)
    end
end

function nlsolve!(
    u::AbstractVector,
    f::AbstractSemidiscreteFunction,
    mlcache::MultiLevelNewtonRaphsonSolverCache,
    t,
    p = t,
)
    cache = mlcache.global_solver_cache

    @unpack op, residual, linear_solver_cache, Θks = cache
    monitor = cache.parameters.monitor
    simplified = cache.parameters.simplified_newton
    cache.iter = -1
    Δu = linear_solver_cache.u
    residualnormprev = 0.0
    Θ1prev = length(Θks) > 0 ? first(Θks) : 0.0
    resize!(Θks, 0)
    set_local_solver_tol(mlcache.local_solver_cache, 0.0)
    while true
        cache.iter += 1
        residual .= 0.0
        reset_local_solve_status!(mlcache.local_solver_cache)
        if simplified && cache.iter > 0
            # Simplified Newton: reuse the Jacobian and preconditioner from iteration 0. The local
            # problems are still solved -- the condensed state is what the residual is a function of
            # -- only their sensitivities are not, since no tangent is requested.
            @timeit_debug "update residual" residual!(op, residual, u, p)
        else
            @timeit_debug "update operator" update_linearization!(op, residual, u, p)
        end
        # Check if local solve failed. The global residual is reported alongside, because a local
        # failure at a small global residual points somewhere very different than one far from the
        # solution.
        if check_local_solve_covergence(mlcache.local_solver_cache)
            @debug "Some local newton did not converge. Aborting. ||r|| = $(residual_norm(cache, f))\n$(describe_local_solve_failures(mlcache.local_solver_cache))" _group =
                :nlsolve
            return false
        end
        if simplified && cache.iter > 0
            @timeit_debug "elimination" eliminate_constraints_from_residual!(cache, f)
            # Leave isfresh / precsisfresh false → reuse the existing factorization.
        else
            @timeit_debug "elimination" eliminate_constraints_from_linearization!(cache, f)
            linear_solver_cache.isfresh = true # Notify linear solver that we touched the system matrix
        end

        residualnorm = residual_norm(cache, f)
        set_local_solver_tol(mlcache.local_solver_cache, residualnorm^2)
        if residualnorm < cache.parameters.tol && cache.iter > 1 # Do at least two iterations to get a sane convergence estimate
            break
        elseif cache.iter > cache.parameters.max_iter
            @debug "Reached maximum Newton iterations. Aborting. ||r|| = $residualnorm" _group=:nlsolve
            return false
        elseif any(isnan.(residualnorm))
            @debug "Newton-Raphson diverged. Aborting. ||r|| = $residualnorm" _group=:nlsolve
            return false
        end

        _ew_prestep!(cache.forcing_cache, linear_solver_cache, residualnorm, cache.iter)
        # See the note in the plain Newton: the Eisenstat-Walker criterion is relative to ‖r₀‖, so a
        # warm-started increment would satisfy it trivially.
        cache.forcing_cache !== nothing && fill!(Δu, zero(eltype(Δu)))
        @timeit_debug "solve" sol = LinearSolve.solve!(linear_solver_cache)
        nonlinear_step_monitor(cache, t, f, u, cache.parameters.monitor)
        solve_succeeded =
            LinearSolve.SciMLBase.successful_retcode(sol) ||
            sol.retcode == LinearSolve.ReturnCode.Default # The latter seems off...
        solve_succeeded || return false

        eliminate_constraints_from_increment!(Δu, f, cache)

        u[1:length(Δu)] .-= Δu # Current guess

        if cache.iter > 0
            # In this case we might be unablet to estimate the convergence rate, because we are too close to the solution
            if residualnormprev < cache.parameters.tol && residualnorm < cache.parameters.tol
                push!(Θks, Θ1prev^2)
                break
            end
            Θk = residualnorm/residualnormprev
            push!(Θks, isnan(Θk) ? 0.0 : Θk)
            # A simplified Newton converges linearly, so a rate close to one is expected rather than
            # a symptom -- hence the same opt-out the plain Newton has.
            if cache.parameters.enforce_monotonic_convergence && Θk ≥ 1.0
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
