struct DummyODESolution <: SciMLBase.AbstractODESolution{Float64, 2, Vector{Float64}}
    retcode::SciMLBase.ReturnCode.T
    prob::Any
end
DummyODESolution(prob) = DummyODESolution(SciMLBase.ReturnCode.Default, prob)
function SciMLBase.solution_new_retcode(sol::DummyODESolution, retcode)
    return DiffEqBase.@set sol.retcode = retcode
end
fix_solution_buffer_sizes!(integrator, sol::DummyODESolution) = nothing

# After an outer rollback the child has not "just failed" anymore. Clearing the branded
# retcode alone is not enough: our `check_error` would re-derive ConvergenceFailure from
# the sticky `last_step_failed` flag on the very next attempt and turn the whole retry
# into an abort.
#
# This method is NOT redundant with the upstream generic for `DEIntegrator`, which reaches
# the flag through `hasfield(typeof(child), :last_stepfail)` -- `OrdinaryDiffEqCore`'s
# spelling. Ours is `last_step_failed`, so the generic would silently do nothing for this
# integrator. Do not delete this method on the assumption that upstream covers it.
function OS._reset_child_failure!(child::ThunderboltTimeIntegrator)
    if child.sol.retcode ∉ (SciMLBase.ReturnCode.Default, SciMLBase.ReturnCode.Success)
        child.sol = SciMLBase.solution_new_retcode(child.sol, SciMLBase.ReturnCode.Default)
    end
    child.last_step_failed = false
    child.force_stepfail = false
    return nothing
end

function OS._build_child(
    prob::OS.OperatorSplittingProblem,
    alg::AbstractSolver,
    f::F,
    p::P,
    uprevouter::S,
    uouter::S,
    u_master::S,
    solution_indices,
    t0::T,
    tf::T,
    tstops,
    saveat,
    d_discontinuities,
    callback,
    config::OS.ConfigTree,
) where {S, T, P, F}

    (; tspan) = prob

    # This node's own settings live in `config.values`; everything else the caller passed
    # to `init` travels to the leaves untouched via `inner_values`.
    tdir = tf > t0 ? one(T) : -one(T)
    # `signed_dt_tree` gave the configured dt the integration direction; this integrator
    # keeps dt as a magnitude and carries the direction in `tdir`.
    dt = abs(config.values.dt)
    # Since we do not have control over the adaptivity setting yet, this toggles adaptivity off for inner algs.
    adaptive = config.values.adaptive & SciMLBase.isadaptive(alg)
    verbose = normalize_verbosity(config.values.verbose)
    controller = config.values.controller
    inner = OS.inner_values(config.values)
    internalnorm = get(inner, :internalnorm, OrdinaryDiffEqCore.ODE_DEFAULT_NORM)
    save_end = get(inner, :save_end, false)
    save_everystep = get(inner, :save_everystep, false)
    maxiters = get(inner, :maxiters, 1000000)

    # The child's solution ALIASES the parent's solution vector -- no copies on the hot
    # path; this shared-solution wiring is the point of this integrator. Its rollback
    # buffer must NOT: the integrator loop refreshes `uprev` from `u` on every internal
    # step (`update_uprev!` in `step_header!`), and through a view that write would land
    # in the parent's rollback anchor, which has to survive the whole outer step
    # untouched. `forward_sync_internal!` refreshes this buffer at the start of every
    # sub-solve. (Upstream's leaf builder copies `u` as well; we deviate only for `u`.)
    uprev = uprevouter[solution_indices]
    u = @view uouter[solution_indices]

    dt > zero(dt) || error("dt must be positive")
    _dt = dt
    tType = typeof(dt)
    tTypeNoUnits = typeof(one(tType))

    dtchangeable = OrdinaryDiffEqCore.isdtchangeable(alg)

    if tstops isa AbstractArray || tstops isa Tuple || tstops isa Number
        _tstops = nothing
    else
        _tstops = tstops
        tstops = ()
    end

    # Setup tstop logic
    tstops_internal =
        OrdinaryDiffEqCore.initialize_tstops(tType, tstops, d_discontinuities, (t0, tf))
    saveat_internal = OrdinaryDiffEqCore.initialize_saveat(tType, saveat, (t0, tf))
    d_discontinuities_internal =
        OrdinaryDiffEqCore.initialize_d_discontinuities(tType, d_discontinuities, (t0, tf))

    cache = setup_solver_cache(f, alg, t0; uprev = uprev, u = u)

    # Setup solution buffers
    uType = typeof(u)
    uBottomEltype = OrdinaryDiffEqCore.recursive_bottom_eltype(u)
    uBottomEltypeNoUnits = OrdinaryDiffEqCore.recursive_unitless_bottom_eltype(u)

    # Setup callbacks
    callbacks_internal = SciMLBase.CallbackSet(callback)
    max_len_cb = DiffEqBase.max_vector_callback_length_int(callbacks_internal)
    if max_len_cb !== nothing
        uBottomEltypeReal = real(uBottomEltype)
        # if SciMLBase.isinplace(prob)
        #     callback_cache = SciMLBase.CallbackCache(u, max_len_cb, uBottomEltypeReal,
        #         uBottomEltypeReal)
        # else
        callback_cache = SciMLBase.CallbackCache(max_len_cb, uBottomEltypeReal, uBottomEltypeReal)
        # end
    else
        callback_cache = nothing
    end

    # # Setup solution
    # save_idxs, saved_subsystem = SciMLBase.get_save_idxs_and_saved_subsystem(prob, save_idxs)

    # rate_prototype = compute_rate_prototype(prob)
    # rateType = typeof(rate_prototype)
    # if save_idxs === nothing
    #     ksEltype = Vector{rateType}
    # else
    #     ks_prototype = rate_prototype[save_idxs]
    #     ksEltype = Vector{typeof(ks_prototype)}
    # end

    ts = tType[]
    ks = uType[]

    # sol = SciMLBase.build_solution(
    #     prob, alg, ts, uType[],
    #     # dense = dense, k = ks, saved_subsystem = saved_subsystem,
    #     calculate_error = false
    # )
    sol = DummyODESolution(
        OS.OperatorSplittingProblem(prob.f, view(prob.u0, solution_indices), tspan),
    )

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
        OrdinaryDiffEqCore.setup_controller_cache(alg, cache, controller, EEstT)
    else
        nothing
    end

    save_end =
        save_end === nothing ?
        save_everystep || isempty(saveat) || saveat isa Number || tf in saveat : save_end

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
            dtmin = tType(get(inner, :dtmin, zero(tType))),
            dtmax = tType(get(inner, :dtmax, tf - t0)),
            verbose = verbose,
            adaptive = adaptive,
            maxiters = maxiters,
            callback = callbacks_internal,
            save_end = save_end,
            tstops = tstops_internal,
            saveat = saveat_internal,
            d_discontinuities = d_discontinuities_internal,
            tstops_cache = tstops,
            saveat_cache = saveat,
            d_discontinuities_cache = d_discontinuities,
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

    if _tstops !== nothing
        tstops = _tstops(parameter_values(integrator), prob.tspan)
        for tstop in tstops
            add_tstop!(integrator, tstop)
        end
    end

    OrdinaryDiffEqCore.handle_dt!(integrator)

    return integrator
end
