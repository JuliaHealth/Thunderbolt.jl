mutable struct IntegratorStats
    naccept::Int64
    nreject::Int64
    # TODO inner solver stats
end

IntegratorStats() = IntegratorStats(0, 0)

Base.@kwdef mutable struct IntegratorOptions{
    tType,
    fType,
    F1,
    F2,
    F3,
    F4,
    F5,
    progressMonitorType,
    SType,
    tstopsType,
    saveatType,
    discType,
    tcache,
    savecache,
    disccache,
    VT,
}
    force_dtmin::Bool = false
    # Error tolerances of the step size control, in the SciML sense: a step is accepted when the
    # scheme's local error estimate satisfies `‖e‖ ≤ abstol + reltol·‖u‖` componentwise. Unused by the
    # non-adaptive schemes.
    reltol::tType = 1.0e-3
    abstol::tType = 1.0e-6
    dtmin::tType = eps(tType)
    dtmax::tType = Inf
    failfactor::fType = 4.0
    verbose::VT = Standard()
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
    # `advance_to_tstop` is read by `step!` below, `stop_at_next_tstop` by
    # `SciMLBase.done`. Both are read after `__init` returns, so they live here.
    advance_to_tstop::Bool = false
    stop_at_next_tstop::Bool = false
    save_end::Bool = true
    # The *raw* user-supplied `save_end`, before the `nothing` default is resolved into
    # `save_end`. `solution_endpoint_match_cur_integrator!` distinguishes "the user asked
    # for the endpoint" from "we defaulted to saving it", so it needs both.
    save_end_user::Union{Nothing, Bool} = nothing
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
} <: SciMLBase.DEIntegrator{algType, true, uType, tType}
    alg::algType
    const f::fType # Right hand side
    u::uType # Current local solution
    uprev::uprevType
    p::pType
    t::tType
    tprev::tType
    dt::tType
    dtcache::tType
    dtpropose::tType
    tdir::tType
    # cache of the time integration algorithm
    cache::cacheType
    # We need this one explicitly, because there is no API to access this variable
    # E.g. https://github.com/SciML/DiffEqBase.jl/blob/ceec4c0ae42a5cf50b78ec876ed650993d7a55b5/src/callbacks.jl#L112
    callback_cache::callbackcacheType
    sol::solType
    const dtchangeable::Bool
    controller_cache::controllerType
    stats::IntegratorStats
    const opts::IntegratorOptions{tType}
    # OrdinaryDiffEqCore compat
    force_stepfail::Bool # This is a flag for the inner solver to communicate that it failed
    iter::Int64 # TODO move into stats
    derivative_discontinuity::Bool
    isout::Bool
    reeval_fsal::Bool
    last_step_failed::Bool
    saveiter::Int
    saveiter_dense::Int
    just_hit_tstop::Bool
    # Tstop bookkeeping for `OrdinaryDiffEqCore.modify_dt_for_tstops!` /
    # `fixed_t_for_tstop_error!`; see the accessors in `diffeq-interface.jl`.
    next_step_tstop::Bool
    tstop_target::tType
end

function init_cache(prob, alg; dt, kwargs...)
    return setup_solver_cache(prob.f, alg, prob.tspan[1]; kwargs...)
end

# The initial velocity is state of the *problem*, but only the function reaches `setup_solver_cache`
# through the generic path above, so it is forwarded here.
function init_cache(prob::ElastodynamicsProblem, alg; dt, kwargs...)
    return setup_solver_cache(prob.f, alg, prob.tspan[1]; v0 = prob.v0, kwargs...)
end

# `@SciMLMessage` dispatches on `Bool` (what the operator splitting parent hands its children) and on
# `AbstractVerbositySpecifier` (what `DiffEqBase.init` passes), but not on the bare `SciMLLogging`
# presets, which are what our own default and a `verbose = Standard()` from user code are. Lower
# those to a `DEVerbosity` so every entry point ends up with something `@SciMLMessage` understands.
normalize_verbosity(verbose) = verbose
normalize_verbosity(verbose::AbstractVerbosityPreset) = DiffEqBase.DEVerbosity(verbose)

# Interpolation
#
# Dispatching on the *cache* is what lets a scheme supply an interpolant of its own accuracy. The
# default is linear, which is first order and correct for any scheme, but below the order of every
# scheme above first. A scheme that carries a derivative of the solution can do better -- see
# `interpolate_solution!` for `NewmarkSolverCache`.
(integrator::ThunderboltTimeIntegrator)(tmp, t) =
    interpolate_solution!(tmp, integrator, integrator.cache, t)

"""
    interpolate_solution!(out, integrator, cache, t)

Write the solution at `t` into `out`, interpolating between the step endpoints the integrator
currently holds.

The fallback is linear in `(uprev, u)`. Override for a solver cache that can do better; a scheme with
a solution derivative at both ends has a Hermite interpolant available at no extra assembly.
"""
function interpolate_solution!(out, integrator::ThunderboltTimeIntegrator, cache, t)
    OS.linear_interpolation!(out, t, integrator.uprev, integrator.u, integrator.tprev, integrator.t)
    return out
end

# CommonSolve interface
@inline function SciMLBase.step!(integrator::ThunderboltTimeIntegrator)
    # The target is captured before the loop: `handle_tstop!` pops the tstop on arrival,
    # so re-reading the heap would carry on to the next one.
    if integrator.opts.advance_to_tstop && SciMLBase.has_tstop(integrator)
        tdir_tstop = SciMLBase.first_tstop(integrator)
        @inbounds while integrator.tdir * integrator.t < tdir_tstop
            _step_once!(integrator) || return
        end
        return
    end
    @inbounds _step_once!(integrator)
    return
end

# One accepted step, retrying on rejection. Returns whether the step succeeded, so the
# `advance_to_tstop` loop above stops on an error instead of spinning.
@inline function _step_once!(integrator::ThunderboltTimeIntegrator)
    @inbounds while true # Emulate do-while
        step_header!(integrator)
        if SciMLBase.check_error!(integrator) != SciMLBase.ReturnCode.Success
            return false
        end
        OrdinaryDiffEqCore.perform_step!(integrator, integrator.cache)
        step_footer!(integrator)
        # Exit condition
        should_accept_step(integrator) && break
    end
    @inbounds OrdinaryDiffEqCore.handle_tstop!(integrator)
    return true
end

_is_empty_saveat(::Nothing) = true
_is_empty_saveat(saveat::Number) = false
_is_empty_saveat(saveat) = isempty(saveat)

# Dense output and intermediate saving are not implemented -- only the endpoint reaches
# `sol` -- so refuse the keywords rather than ignore them. Whoever implements saving
# deletes this and wires up `save_on`/`save_everystep`.
function _reject_unsupported_saving(saveat, save_everystep, dense, save_idxs)
    requested = String[]
    _is_empty_saveat(saveat) || push!(requested, "saveat")
    save_everystep && push!(requested, "save_everystep")
    dense && push!(requested, "dense")
    save_idxs === nothing || push!(requested, "save_idxs")
    isempty(requested) && return nothing
    return error(
        "ThunderboltTimeIntegrator does not implement intermediate saving or dense " *
        "output, but was given $(join(requested, ", ")). Only the solution endpoint is " *
        "stored. Drop the keyword(s), or read the state from the integrator directly.",
    )
end

# Callbacks are not implemented: the solution and integrator lack the stats and event
# state they need. Refuse at construction rather than fail mid-step.
function _reject_callbacks(callback)
    callback === nothing && return nothing
    cbs = SciMLBase.CallbackSet(callback)
    (isempty(cbs.continuous_callbacks) && isempty(cbs.discrete_callbacks)) && return nothing
    return error(
        "ThunderboltTimeIntegrator does not implement callbacks. Pass `callback = nothing` " *
        "and drive the integrator with `step!` if you need to act between steps.",
    )
end

function SciMLBase.__init(
    prob::AbstractSemidiscreteProblem,
    alg::AbstractSolver,
    args...;
    dt,
    saveat = (),
    tstops = (),
    d_discontinuities = (),
    ts_init = (),
    ks_init = (),
    save_end = nothing,
    save_everystep = false,
    save_idxs = nothing,
    callback = nothing,
    advance_to_tstop = false,
    stop_at_next_tstop = false,
    adaptive = SciMLBase.isadaptive(alg),
    verbose = Standard(),
    alias_u0 = true,
    # alias_du0 = false,
    controller = nothing,
    maxiters = 1000000,
    dense = false,
    dtmin = nothing,
    dtmax = nothing,
    reltol = 1.0e-3,
    abstol = 1.0e-6,
    internalnorm = OrdinaryDiffEqCore.ODE_DEFAULT_NORM,
    kwargs...,
)
    (; f, u0, p, tspan) = prob
    t0, tf = tspan

    tType = eltype(tspan)
    tTypeNoUnits = typeof(one(tType))

    _reject_unsupported_saving(saveat, save_everystep, dense, save_idxs)
    _reject_callbacks(callback)

    dt > zero(dt) || error("dt must be positive")
    # `dt` is a magnitude here and the direction lives in `tdir`, but the upstream
    # helpers we call assume a *signed* dt.
    tf ≥ t0 || error(
        "Backward integration is not supported: got tspan = ($t0, $tf). " *
        "Integrate forward and reverse the solution instead.",
    )
    _dt = dt
    tdir = one(tType)

    dtchangeable = OrdinaryDiffEqCore.isdtchangeable(alg)

    # `prob2dtmin` is the SciML integrator default; zero would make `DtLessThanMin`
    # unreachable.
    dtmin = dtmin === nothing ? tType(DiffEqBase.prob2dtmin(prob)) : tType(dtmin)
    dtmax = dtmax === nothing ? tType(tf - t0) : tType(dtmax)

    if tstops isa AbstractArray || tstops isa Tuple || tstops isa Number
        _tstops = nothing
    else
        _tstops = tstops
        tstops = ()
    end

    # Setup tstop logic
    tstops_internal = OrdinaryDiffEqCore.initialize_tstops(tType, tstops, d_discontinuities, tspan)
    saveat_internal = OrdinaryDiffEqCore.initialize_saveat(tType, saveat, tspan)
    d_discontinuities_internal =
        OrdinaryDiffEqCore.initialize_d_discontinuities(tType, d_discontinuities, tspan)

    save_end_user = save_end
    save_end =
        save_end === nothing ?
        save_everystep || isempty(saveat) || saveat isa Number || tf in saveat : save_end

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
            callback_cache =
                SciMLBase.CallbackCache(u, max_len_cb, uBottomEltypeReal, uBottomEltypeReal)
        else
            callback_cache =
                SciMLBase.CallbackCache(max_len_cb, uBottomEltypeReal, uBottomEltypeReal)
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
        prob,
        alg,
        ts,
        uType[],
        dense = dense,
        k = ks,
        saved_subsystem = saved_subsystem,
        calculate_error = false,
    )

    # Setup algorithm cache
    cache = init_cache(prob, alg; dt = dt, u = u)

    QT = OrdinaryDiffEqCore.determine_controller_datatype(u, internalnorm, tspan)

    if controller === nothing && adaptive
        controller = OrdinaryDiffEqCore.default_controller(QT, alg)
    end

    EEstT = if tTypeNoUnits <: Integer
        QT
    elseif prob isa SciMLBase.AbstractDiscreteProblem
        constvalue(tTypeNoUnits)
    else
        typeof(internalnorm(u, t0))
    end

    controller_cache = if controller !== nothing
        # `disco_probs` is the discontinuity detection problem cache of the upstream controllers.
        # Detection is off by default, so an empty vector of the expected element type is what the
        # resolved controller options want.
        OrdinaryDiffEqCore.setup_controller_cache(
            alg,
            cache,
            controller,
            EEstT,
            SciMLBase.IntervalNonlinearProblem[],
        )
    else
        nothing
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
        dt,
        dt,
        tdir,
        cache,
        callback_cache,
        sol,
        dtchangeable,
        controller_cache,
        IntegratorStats(),
        IntegratorOptions(
            dtmin = dtmin,
            dtmax = dtmax,
            reltol = tType(reltol),
            abstol = tType(abstol),
            verbose = normalize_verbosity(verbose),
            adaptive = adaptive,
            maxiters = maxiters,
            callback = callbacks_internal,
            advance_to_tstop = advance_to_tstop,
            stop_at_next_tstop = stop_at_next_tstop,
            save_end = save_end,
            save_end_user = save_end_user,
            tstops = tstops_internal,
            saveat = saveat_internal,
            d_discontinuities = d_discontinuities_internal,
            tstops_cache = tstops,
            saveat_cache = saveat,
            d_discontinuities_cache = d_discontinuities,
            internalnorm = internalnorm,
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
        false,
        tType(t0),
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

DiffEqBase.initialize!(integrator::ThunderboltTimeIntegrator, cache::AbstractTimeSolverCache) =
    nothing

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
    return integrator.sol =
        SciMLBase.solution_new_retcode(integrator.sol, SciMLBase.ReturnCode.Success)
end

# Utils
function setup_u(prob::AbstractSemidiscreteProblem, solver, alias_u0)
    if alias_u0
        return prob.u0
    else
        return recursivecopy(prob.u0)
    end
end

# Controller interface
function reject_step!(integrator::ThunderboltTimeIntegrator)
    OrdinaryDiffEqCore.increment_reject!(integrator.stats)
    rollback_state!(integrator, integrator.cache)
    reject_step!(integrator, integrator.cache, integrator.controller_cache)
end

"""
    rollback_state!(integrator, cache)

Restore the state a rejected step advanced. Separate from [`reject_step!`](@ref), which proposes the
step size for the retry: restoring state is the *scheme's* business and proposing a step size is the
*controller's*, so expressing both on one generic would make the method count their product.

The fallback restores the solution vector. A scheme whose state is not fully contained in it -- a
second order scheme carries a velocity and an acceleration -- extends this.
"""
function rollback_state!(integrator::ThunderboltTimeIntegrator, cache)
    if length(integrator.uprev) == 0
        error("Cannot roll back integrator. Aborting time integration step at $(integrator.t).")
    end
    integrator.u .= integrator.uprev
    return nothing
end

reject_step!(integrator::ThunderboltTimeIntegrator, cache, controller) = nothing

adapt_dt!(integrator::ThunderboltTimeIntegrator) =
    adapt_dt!(integrator, integrator.cache, integrator.controller_cache)
function adapt_dt!(integrator::ThunderboltTimeIntegrator, cache, controller)
    error("Step size control not implemented for $(integrator.alg).")
end
adapt_dt!(integrator::ThunderboltTimeIntegrator, cache, ::Union{Nothing, DummyControllerCache}) =
    nothing


include("diffeq-interface.jl")
include("operatorsplitting-interface.jl")
