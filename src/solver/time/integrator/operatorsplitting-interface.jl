struct DummyODESolution <: SciMLBase.AbstractODESolution{Float64,2,Vector{Float64}}
    retcode::SciMLBase.ReturnCode.T
end
DummyODESolution() = DummyODESolution(SciMLBase.ReturnCode.Default)
function SciMLBase.solution_new_retcode(sol::DummyODESolution, retcode)
    return DiffEqBase.@set sol.retcode = retcode
end
fix_solution_buffer_sizes!(integrator, sol::DummyODESolution) = nothing

function OS.build_subintegrator_tree_with_cache(
    prob::OS.OperatorSplittingProblem,
    alg::AbstractSolver, f, p,
    uprevouter::AbstractVector, uouter::AbstractVector,
    solution_indices,
    t0, dt, tf,
    tstops, saveat, d_discontinuities, callback,
    adaptive, verbose,
    save_end=false,
    controller=nothing,
)
    uprev = @view uprevouter[solution_indices]
    u = @view uouter[solution_indices]

    dt > zero(dt) || error("dt must be positive")
    _dt = dt
    tdir = tf > t0 ? one(dt) : -one(dt)
    tType = typeof(dt)

    if tstops isa AbstractArray || tstops isa Tuple || tstops isa Number
        _tstops = nothing
    else
        _tstops = tstops
        tstops = ()
    end

    # Setup tstop logic
    tstops_internal = OrdinaryDiffEqCore.initialize_tstops(tType, tstops, d_discontinuities, (t0, tf))
    saveat_internal = OrdinaryDiffEqCore.initialize_saveat(tType, saveat, (t0, tf))
    d_discontinuities_internal = OrdinaryDiffEqCore.initialize_d_discontinuities(tType, d_discontinuities, (t0, tf))

    cache = setup_solver_cache(f, alg, t0; uprev=uprev, u=u)

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
        callback_cache = SciMLBase.CallbackCache(max_len_cb, uBottomEltypeReal,
            uBottomEltypeReal)
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
    sol = DummyODESolution()

    if controller === nothing && adaptive && SciMLBase.isadaptive(alg)
        controller = default_controller(alg, cache)
    end

    save_end = save_end === nothing ? save_everystep || isempty(saveat) || saveat isa Number || tf in saveat : save_end

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
        true,
        adaptive ? controller : nothing,
        IntegratorStats(),
        IntegratorOptions(
            dtmin=zero(tType),
            dtmax=tType(tf - t0),
            verbose=verbose,
            adaptive=adaptive,
            # maxiters = maxiters,
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

    if _tstops !== nothing
        tstops = _tstops(parameter_values(integrator), prob.tspan)
        for tstop in tstops
            add_tstop!(integrator, tstop)
        end
    end

    OrdinaryDiffEqCore.handle_dt!(integrator)

    return integrator, integrator.cache
end
