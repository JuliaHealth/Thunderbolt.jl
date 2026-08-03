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

Base.@kwdef mutable struct GenericLocalNonlinearSolverCache{
    JacobianType,
    ResidualType,
    CorrectorRhsType,
    ReportsType,
}
    const params::GenericLocalNonlinearSolver
    const J::JacobianType
    const residual::ResidualType
    const rhs_corrector::CorrectorRhsType
    const reports::ReportsType = nothing
    outer_tol::Float64 = 0.0
    retcode::SciMLBase.ReturnCode.T = SciMLBase.ReturnCode.Default
end

function duplicate_for_device(device, cache::GenericLocalNonlinearSolverCache)
    GenericLocalNonlinearSolverCache(;
        params        = cache.params,
        J             = duplicate_for_device(device, cache.J),
        residual      = duplicate_for_device(device, cache.residual),
        rhs_corrector = duplicate_for_device(device, cache.rhs_corrector),
        # Deliberately shared: workers write disjoint slots, and a failure must survive the worker.
        reports   = cache.reports,
        outer_tol = cache.outer_tol,
        retcode   = cache.retcode,
    )
end

"""
    record_local_solve!(local_solver_cache, cellid, qpi, retcode, residualnorm)

Record the outcome of one local solve, both on the cache -- where the material routine reads it back
to skip the sensitivity solves of a failed point -- and in the shared per-quadrature-point store.
"""
@inline function record_local_solve!(
    local_solver_cache::GenericLocalNonlinearSolverCache,
    cellid,
    qpi,
    retcode,
    residualnorm,
)
    local_solver_cache.retcode = retcode
    reports = local_solver_cache.reports
    reports === nothing && return nothing
    get_data_for_index(reports, cellid)[qpi] = LocalSolveReport(retcode, residualnorm)
    return nothing
end

# The scalar `retcode` lives on the cache, which `duplicate_for_device` copies per worker, so it only
# ever reports what *this* worker last did. The store is shared, hence authoritative.
function check_local_solve_covergence(local_solver_cache::GenericLocalNonlinearSolverCache)
    reports = local_solver_cache.reports
    reports === nothing && return local_solver_cache.retcode ∉
           (SciMLBase.ReturnCode.Default, SciMLBase.ReturnCode.Success)
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

The failing quadrature points of the last assembly pass, as `cell`/`qp` pairs with their local
residual norm and return code. Empty when nothing failed.

Reports the first failure *per assembly worker*, not every failing point: the condensation entry
points stop solving once a worker has seen a failure, since the pass is discarded either way.
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

Clear the local solvers' failure flag and reports before an assembly pass, so
`check_local_solve_covergence` reports on *that* pass alone.

Without this the flag latches: a failed local solve is only ever written by the local Newton, and the
condensation entry points skip that Newton entirely once the flag is set. The first local failure
would therefore poison every later assembly — including the retry after the time integrator shortens
`dt`, which is exactly the mechanism meant to recover from it.
"""
function reset_local_solve_status!(local_solver_cache::GenericLocalNonlinearSolverCache)
    local_solver_cache.retcode = SciMLBase.ReturnCode.Default
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
    local_solver_cache.outer_tol = tol
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
        @timeit_debug "update operator" update_linearization!(op, residual, u, p)
        # Check if local solve failed. The global residual is reported alongside, because a local
        # failure at a small global residual points somewhere very different than one far from the
        # solution.
        if check_local_solve_covergence(mlcache.local_solver_cache)
            @debug "Some local newton did not converge. Aborting. ||r|| = $(residual_norm(cache, f))\n$(describe_local_solve_failures(mlcache.local_solver_cache))" _group =
                :nlsolve
            return false
        end
        @timeit_debug "elimination" eliminate_constraints_from_linearization!(cache, f)
        linear_solver_cache.isfresh = true # Notify linear solver that we touched the system matrix

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
