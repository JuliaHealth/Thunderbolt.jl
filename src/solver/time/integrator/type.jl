mutable struct IntegratorStats
    naccept::Int64
    nreject::Int64
    # TODO inner solver stats
end

IntegratorStats() = IntegratorStats(0, 0)

Base.@kwdef mutable struct IntegratorOptions{tType,fType,F1,F2,F3,F4,F5,progressMonitorType,SType,tstopsType,saveatType,discType,tcache,savecache,disccache}
    force_dtmin::Bool = false
    dtmin::tType = eps(tType)
    dtmax::tType = Inf
    failfactor::fType = 4.0
    verbose::Bool = false
    adaptive::Bool = false # Redundant with the dispatch on SciMLBase.isadaptive below (alg adaptive + controller not nothing)
    maxiters::Int = 1000000
    # Internal norms to measure matrix and vector sizes (in the sense of normed vector spaces)
    internalnorm::F1 = DiffEqBase.ODE_DEFAULT_NORM
    internalopnorm::F2 = LinearAlgebra.opnorm
    # Function to check whether the solution is still inside the domain it is defined on
    isoutofdomain::F3 = DiffEqBase.ODE_DEFAULT_ISOUTOFDOMAIN
    # Function to check whether the solution is unstable
    unstable_check::F4 = DiffEqBase.ODE_DEFAULT_UNSTABLE_CHECK
    # This is mostly OrdinaryDiffEqCore compat
    progress::Bool = true
    progress_steps::Int = 1
    progress_monitor::progressMonitorType = DefaultProgressMonitor()
    save_idxs::SType = nothing
    save_end::Bool = true
    dense::Bool = false
    save_on::Bool = false
    # TODO vvv factor these into some event management data type vvv
    tstops::tstopsType = nothing
    saveat::saveatType = nothing
    d_discontinuities::discType = nothing
    tstops_cache::tcache = ()
    saveat_cache::savecache = ()
    d_discontinuities_cache::disccache = ()
    # TODO ^^^ factor these into some event management data type ^^^
    # Callbacks are inconsistent with the remaining event management above, as the
    # associated cache is stored in the integrator instead of the options data type
    callback::F5
end

"""
Internal helper to integrate a single inner operator
over some time interval.
"""
mutable struct ThunderboltTimeIntegrator{
    algType,
    fType,
    uType,
    uprevType,
    tType,
    pType,
    cacheType,
    callbackcacheType,
    solType,
    controllerType,
} <: SciMLBase.DEIntegrator{algType,true,uType,tType}
    alg::algType
    const f::fType # Right hand side
    u::uType # Current local solution
    uprev::uprevType
    p::pType
    t::tType
    tprev::tType
    dt::tType
    tdir::tType
    # cache of the time integration algorithm
    cache::cacheType
    # We need this one explicitly, because there is no API to access this variable
    # E.g. https://github.com/SciML/DiffEqBase.jl/blob/ceec4c0ae42a5cf50b78ec876ed650993d7a55b5/src/callbacks.jl#L112
    callback_cache::callbackcacheType
    sol::solType
    const dtchangeable::Bool
    controller::controllerType
    stats::IntegratorStats
    const opts::IntegratorOptions{tType}
    # OrdinaryDiffEqCore compat
    force_stepfail::Bool # This is a flag for the inner solver to communicate that it failed
    iter::Int64 # TODO move into stats
    u_modified::Bool
    isout::Bool
    reeval_fsal::Bool
    last_step_failed::Bool
    saveiter::Int
    saveiter_dense::Int
    just_hit_tstop::Bool
end

function init_cache(prob, alg; dt, kwargs...)
    return setup_solver_cache(prob.f, alg, prob.tspan[1]; kwargs...)
end

# Interpolation
function (integrator::ThunderboltTimeIntegrator)(tmp, t)
    OS.linear_interpolation!(tmp, t, integrator.uprev, integrator.u, integrator.t - integrator.dt, integrator.t)
end

# CommonSolve interface
@inline function SciMLBase.step!(integrator::ThunderboltTimeIntegrator)
    @inbounds while true # Emulate do-while
        step_header!(integrator)
        if SciMLBase.check_error!(integrator) != SciMLBase.ReturnCode.Success
            return
        end
        OrdinaryDiffEqCore.perform_step!(integrator, integrator.cache)
        step_footer!(integrator)
        # Exit condition
        should_accept_step(integrator) && break
    end
    @inbounds OrdinaryDiffEqCore.handle_tstop!(integrator)
end

function SciMLBase.__init(
    prob::AbstractSemidiscreteProblem,
    alg::AbstractSolver,
    args...;
    dt,
    saveat=(),
    tstops=(),
    d_discontinuities=(),
    ts_init=(),
    ks_init=(),
    save_end=nothing,
    save_everystep=false,
    save_idxs=nothing,
    callback=nothing,
    advance_to_tstop=false,
    adaptive=SciMLBase.isadaptive(alg),
    verbose=false,
    alias_u0=true,
    # alias_du0 = false,
    controller=nothing,
    maxiters=1000000,
    dense=save_everystep &&
              !(alg isa DAEAlgorithm) && !(prob isa DiscreteProblem),
    dtmin=nothing,
    dtmax=nothing,
    kwargs...,
)
    (; f, u0, p) = prob
    t0, tf = prob.tspan

    dt > zero(dt) || error("dt must be positive")
    _dt = dt
    tdir = tf > t0 ? 1.0 : -1.0
    tType = typeof(dt)

    dtchangeable = SciMLBase.isadaptive(alg)

    dtmin = dtmin === nothing ? tType(0.0) : tType(dtmin)
    dtmax = dtmax === nothing ? tType(tf - t0) : tType(dtmax)

    if tstops isa AbstractArray || tstops isa Tuple || tstops isa Number
        _tstops = nothing
    else
        _tstops = tstops
        tstops = ()
    end

    # Setup tstop logic
    tstops_internal = OrdinaryDiffEqCore.initialize_tstops(tType, tstops, d_discontinuities, prob.tspan)
    saveat_internal = OrdinaryDiffEqCore.initialize_saveat(tType, saveat, prob.tspan)
    d_discontinuities_internal = OrdinaryDiffEqCore.initialize_d_discontinuities(tType, d_discontinuities, prob.tspan)

    save_end = save_end === nothing ? save_everystep || isempty(saveat) || saveat isa Number || tf in saveat : save_end

    # Setup solution buffers
    u = setup_u(prob, alg, alias_u0)
    uType = typeof(u)
    uBottomEltype = OrdinaryDiffEqCore.recursive_bottom_eltype(u)
    uBottomEltypeNoUnits = OrdinaryDiffEqCore.recursive_unitless_bottom_eltype(u)

    # Setup callbacks
    callbacks_internal = SciMLBase.CallbackSet(callback)
    max_len_cb = DiffEqBase.max_vector_callback_length_int(callbacks_internal)
    if max_len_cb !== nothing
        uBottomEltypeReal = real(uBottomEltype)
        if SciMLBase.isinplace(prob)
            callback_cache = SciMLBase.CallbackCache(u, max_len_cb, uBottomEltypeReal,
                uBottomEltypeReal)
        else
            callback_cache = SciMLBase.CallbackCache(max_len_cb, uBottomEltypeReal,
                uBottomEltypeReal)
        end
    else
        callback_cache = nothing
    end

    # Setup solution
    save_idxs, saved_subsystem = SciMLBase.get_save_idxs_and_saved_subsystem(prob, save_idxs)

    rate_prototype = compute_rate_prototype(prob)
    rateType = typeof(rate_prototype)
    if save_idxs === nothing
        ksEltype = Vector{rateType}
    else
        ks_prototype = rate_prototype[save_idxs]
        ksEltype = Vector{typeof(ks_prototype)}
    end

    ts = ts_init === () ? tType[] : convert(Vector{tType}, ts_init)
    ks = ks_init === () ? ksEltype[] : convert(Vector{ksEltype}, ks_init)

    sol = SciMLBase.build_solution(
        prob, alg, ts, uType[],
        dense=dense, k=ks, saved_subsystem=saved_subsystem,
        calculate_error=false
    )

    # Setup algorithm cache
    cache = init_cache(prob, alg; dt=dt, u=u)

    # Setup controller
    if controller === nothing && adaptive
        controller = default_controller(alg, cache)
    end

    # Setup the actual integrator object
    integrator = ThunderboltTimeIntegrator(
        alg,
        f,
        cache.uₙ,
        cache.uₙ₋₁,
        p,
        t0,
        t0,
        dt,
        tdir,
        cache,
        callback_cache,
        sol,
        dtchangeable,
        adaptive ? controller : nothing,
        IntegratorStats(),
        IntegratorOptions(
            dtmin=dtmin,
            dtmax=dtmax,
            verbose=verbose,
            adaptive=adaptive,
            maxiters=maxiters,
            callback=callbacks_internal,
            save_end=save_end,
            tstops=tstops_internal,
            saveat=saveat_internal,
            d_discontinuities=d_discontinuities_internal,
            tstops_cache=tstops,
            saveat_cache=saveat,
            d_discontinuities_cache=d_discontinuities,
        ),
        false,
        0,
        false,
        false,
        false,
        false,
        0,
        0,
        false,
    )
    OrdinaryDiffEqCore.initialize_callbacks!(integrator)
    DiffEqBase.initialize!(integrator, integrator.cache)

    if _tstops !== nothing
        tstops = _tstops(parameter_values(integrator), prob.tspan)
        for tstop in tstops
            add_tstop!(integrator, tstop)
        end
    end

    OrdinaryDiffEqCore.handle_dt!(integrator)

    return integrator
end

DiffEqBase.initialize!(integrator::ThunderboltTimeIntegrator, cache::AbstractTimeSolverCache) = nothing

function SciMLBase.solve!(integrator::ThunderboltTimeIntegrator)
    @inbounds while SciMLBase.has_tstop(integrator)
        while integrator.tdir * integrator.t < SciMLBase.SciMLBase.first_tstop(integrator)
            step_header!(integrator)
            if SciMLBase.check_error!(integrator) != SciMLBase.ReturnCode.Success
                return integrator.sol
            end
            OrdinaryDiffEqCore.perform_step!(integrator, integrator.cache)
            step_footer!(integrator)
            if !SciMLBase.has_tstop(integrator)
                break
            end
        end
        OrdinaryDiffEqCore.handle_tstop!(integrator)
    end
    SciMLBase.postamble!(integrator)
    if integrator.sol.retcode != SciMLBase.ReturnCode.Default
        return integrator.sol
    end
    return integrator.sol = SciMLBase.solution_new_retcode(integrator.sol, SciMLBase.ReturnCode.Success)
end

# Utils
function setup_u(prob::AbstractSemidiscreteProblem, solver, alias_u0)
    if alias_u0
        return prob.u0
    else
        return SciMLBase.recursivecopy(prob.u0)
    end
end

# Controller interface
function reject_step!(integrator::ThunderboltTimeIntegrator)
    OrdinaryDiffEqCore.increment_reject!(integrator.stats)
    reject_step!(integrator, integrator.cache, integrator.controller)
end
function reject_step!(integrator::ThunderboltTimeIntegrator, cache, controller)
    integrator.u .= integrator.uprev
end
function reject_step!(integrator::ThunderboltTimeIntegrator, cache, ::Nothing)
    if length(integrator.uprev) == 0
        error("Cannot roll back integrator. Aborting time integration step at $(integrator.t).")
    end
end

adapt_dt!(integrator::ThunderboltTimeIntegrator) = adapt_dt!(integrator, integrator.cache, integrator.controller)
function adapt_dt!(integrator::ThunderboltTimeIntegrator, cache, controller)
    error("Step size control not implemented for $(alg).")
end
adapt_dt!(integrator::ThunderboltTimeIntegrator, cache, ::Nothing) = nothing


include("diffeq-interface.jl")
include("operatorsplitting-interface.jl")
