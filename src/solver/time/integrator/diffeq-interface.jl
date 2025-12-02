# ----------------------------------- SciMLBase.jl Integrator Interface ------------------------------------
SciMLBase.has_stats(::ThunderboltTimeIntegrator) = true

SciMLBase.has_tstop(integrator::ThunderboltTimeIntegrator) = !isempty(integrator.opts.tstops)
SciMLBase.first_tstop(integrator::ThunderboltTimeIntegrator) = first(integrator.opts.tstops)
SciMLBase.pop_tstop!(integrator::ThunderboltTimeIntegrator) = pop!(integrator.opts.tstops)

function SciMLBase.add_tstop!(integrator::ThunderboltTimeIntegrator, t)
    integrator.tdir * (t - integrator.t) < zero(integrator.t) &&
    error("Tried to add a tstop that is behind the current time. This is strictly forbidden")
    push!(integrator.opts.tstops, integrator.tdir * t)
end

@inline function SciMLBase.get_tmp_cache(integrator::ThunderboltTimeIntegrator)
    return (integrator.cache.tmp,)
end
@inline function SciMLBase.get_tmp_cache(integrator::ThunderboltTimeIntegrator, alg::AbstractSolver, cache::AbstractTimeSolverCache)
    return (cache.tmp,)
end

function SciMLBase.terminate!(integrator::ThunderboltTimeIntegrator, retcode = ReturnCode.Terminated)
    integrator.sol = SciMLBase.solution_new_retcode(integrator.sol, retcode)
    integrator.opts.tstops.valtree = typeof(integrator.opts.tstops.valtree)()
end

# @inline function SciMLBase.get_du(integrator::ThunderboltTimeIntegrator)
# end

@inline SciMLBase.get_proposed_dt(integrator::ThunderboltTimeIntegrator) = integrator.dt

@inline function SciMLBase.u_modified!(integrator::ThunderboltTimeIntegrator, bool::Bool)
    integrator.u_modified = bool
end

SciMLBase.get_sol(integrator::ThunderboltTimeIntegrator) = integrator.sol

function SciMLBase.set_proposed_dt!(integrator::ThunderboltTimeIntegrator, dt)
    if integrator.dtchangeable == true
        integrator.dt = dt
    elseif integrator.dt != dt
        error("Trying to change dt on constant time step integrator.")
    end
end

function SciMLBase.isadaptive(integrator::ThunderboltTimeIntegrator)
    integrator.controller === nothing && return false
    if !SciMLBase.isadaptive(integrator.alg)
        error("Algorithm $(integrator.alg) is not adaptive, but the integrator is trying to adapt. Aborting.")
    end
    return true
end

function SciMLBase.last_step_failed(integrator::ThunderboltTimeIntegrator)
    integrator.last_step_failed
end

SciMLBase.postamble!(integrator::ThunderboltTimeIntegrator) = _postamble!(integrator)

function SciMLBase.savevalues!(integrator::ThunderboltTimeIntegrator, force_save = false, reduce_size = true)
    OrdinaryDiffEqCore._savevalues!(integrator, force_save, reduce_size)
end

# ---------------------------------- DiffEqBase.jl Interface ------------------------------------
DiffEqBase.get_tstops(integ::ThunderboltTimeIntegrator) = integ.opts.tstops
DiffEqBase.get_tstops_array(integ::ThunderboltTimeIntegrator) = get_tstops(integ).valtree
DiffEqBase.get_tstops_max(integ::ThunderboltTimeIntegrator) = maximum(get_tstops_array(integ))


DiffEqBase.has_reinit(integrator::ThunderboltTimeIntegrator) = true
function DiffEqBase.reinit!(
    integrator::ThunderboltTimeIntegrator,
    u0 = integrator.sol.prob.u0;
    t0 = integrator.sol.prob.tspan[1],
    tf = integrator.sol.prob.tspan[2],
    dt0 = tf-t0,
    erase_sol = false,
    tstops = integrator.opts.tstops_cache,
    saveat =  integrator.opts.saveat_cache,
    d_discontinuities = integrator.opts.d_discontinuities_cache,
    reinit_callbacks = true,
    reinit_retcode = true,
    reinit_cache = true,
)
    SciMLBase.recursivecopy!(integrator.u, u0)
    SciMLBase.recursivecopy!(integrator.uprev, integrator.u)
    integrator.t = t0
    integrator.tprev = t0

    integrator.iter = 0
    integrator.u_modified = false

    integrator.stats.naccept = 0
    integrator.stats.nreject = 0

    if erase_sol
        resize!(integrator.sol.t, 0)
        resize!(integrator.sol.u, 0)
    end
    if reinit_callbacks
        DiffEqBase.initialize!(integrator.opts.callback, u0, t0, integrator)
    else # always reinit the saving callback so that t0 can be saved if needed
        saving_callback = integrator.opts.callback.discrete_callbacks[end]
        DiffEqBase.initialize!(saving_callback, u0, t0, integrator)
    end
    if reinit_retcode
        integrator.sol = DiffEqBase.solution_new_retcode(integrator.sol, SciMLBase.ReturnCode.Default)
    end

    tType = typeof(integrator.t)
    tspan = (tType(t0), tType(tf))
    integrator.opts.tstops = OrdinaryDiffEqCore.initialize_tstops(tType, tstops, d_discontinuities, tspan)
    integrator.opts.saveat = OrdinaryDiffEqCore.initialize_saveat(tType, saveat, tspan)
    integrator.opts.d_discontinuities = OrdinaryDiffEqCore.initialize_d_discontinuities(tType,
        d_discontinuities,
        tspan)

    if reinit_cache
        DiffEqBase.initialize!(integrator, integrator.cache)
    end
end


# ----------------------------------- OrdinaryDiffEqCore compat ----------------------------------
OrdinaryDiffEqCore.has_discontinuity(integrator::ThunderboltTimeIntegrator) = !isempty(integrator.opts.d_discontinuities)
OrdinaryDiffEqCore.first_discontinuity(integrator::ThunderboltTimeIntegrator) = first(integrator.opts.d_discontinuities)
OrdinaryDiffEqCore.pop_discontinuity!(integrator::ThunderboltTimeIntegrator) = pop!(integrator.opts.d_discontinuities)

function _postamble!(integrator)
    DiffEqBase.finalize!(integrator.opts.callback, integrator.u, integrator.t, integrator)
    OrdinaryDiffEqCore.solution_endpoint_match_cur_integrator!(integrator)
    fix_solution_buffer_sizes!(integrator, integrator.sol)
    finalize_integration_monitor(integrator)
end

OrdinaryDiffEqCore.alg_extrapolates(alg::AbstractSolver) = false

OrdinaryDiffEqCore.choose_algorithm!(integrator, cache::AbstractTimeSolverCache) = nothing

function OrdinaryDiffEqCore.perform_step!(integ::ThunderboltTimeIntegrator, cache::AbstractTimeSolverCache)
    if !perform_step!(integ.f, cache, integ.t, integ.dt)
        integ.force_stepfail = true
    end
    return nothing
end

# --------------------------- New Interface Stuff (to be upstreamed) ---------------------------------

# Solution looping interface
function should_accept_step(integrator::ThunderboltTimeIntegrator)
    if integrator.force_stepfail || integrator.isout
        return false
    end
    return should_accept_step(integrator, integrator.cache, integrator.controller)
end
function should_accept_step(integrator::ThunderboltTimeIntegrator, cache, ::Nothing)
    return !(integrator.force_stepfail)
end

function accept_step!(integrator::ThunderboltTimeIntegrator)
    OrdinaryDiffEqCore.increment_accept!(integrator.stats)
    accept_step!(integrator, integrator.cache, integrator.controller)
end
function accept_step!(integrator::ThunderboltTimeIntegrator, cache, controller)
    store_previous_info!(integrator)
end

function store_previous_info!(integrator::ThunderboltTimeIntegrator)
    if length(integrator.uprev) > 0 # Integrator can rollback
        update_uprev!(integrator)
    end
end

function step_header!(integrator::ThunderboltTimeIntegrator)
    # Accept or reject the step
    if !is_first_iteration(integrator)
        if should_accept_step(integrator)
            accept_step!(integrator)
        else # Step should be rejected and hence repeated
            reject_step!(integrator)
        end
    elseif integrator.u_modified # && integrator.iter == 0
        update_uprev!(integrator)
    end

    # Before stepping we might need to adjust the dt
    increment_iteration(integrator)
    OrdinaryDiffEqCore.choose_algorithm!(integrator, integrator.cache)
    OrdinaryDiffEqCore.fix_dt_at_bounds!(integrator)
    OrdinaryDiffEqCore.modify_dt_for_tstops!(integrator)
    integrator.force_stepfail = false
end

function update_uprev!(integrator::ThunderboltTimeIntegrator)
    # # OrdinaryDiffEqCore.update_uprev!(integrator) # FIXME recover
    # if alg_extrapolates(integrator.alg)
    #     if isinplace(integrator.sol.prob)
    #         SciMLBase.recursivecopy!(integrator.uprev2, integrator.uprev)
    #     else
    #         integrator.uprev2 = integrator.uprev
    #     end
    # end
    # if isinplace(integrator.sol.prob) # This should be dispatched in the integrator directly
        SciMLBase.recursivecopy!(integrator.uprev, integrator.u)
        if integrator.alg isa OrdinaryDiffEqCore.DAEAlgorithm
            SciMLBase.recursivecopy!(integrator.duprev, integrator.du)
        end
    # else
    #     integrator.uprev = integrator.u
    #     if integrator.alg isa DAEAlgorithm
    #         integrator.duprev = integrator.du
    #     end
    # end
    nothing
end

function controller_message_on_dtmin_error(integrator::SciMLBase.DEIntegrator)
    if isdefined(integrator, :EEst)
       return ", and step error estimate = $(integrator.EEst)"
    else
        return ""
    end
end

function SciMLBase.check_error(integrator::ThunderboltTimeIntegrator)
    if integrator.sol.retcode ∉ (SciMLBase.ReturnCode.Success, SciMLBase.ReturnCode.Default)
        return integrator.sol.retcode
    end
    opts = integrator.opts
    verbose = opts.verbose
    # This implementation is intended to be used for ODEIntegrator and
    # SDEIntegrator.
    if isnan(integrator.dt)
        if verbose
            @warn("NaN dt detected. Likely a NaN value in the state, parameters, or derivative value caused this outcome.")
        end
        return SciMLBase.ReturnCode.DtNaN
    end
    if hasproperty(integrator, :iter) && hasproperty(opts, :maxiters) && integrator.iter > opts.maxiters
        if verbose
            @warn("Interrupted. Larger maxiters is needed. If you are using an integrator for non-stiff ODEs or an automatic switching algorithm (the default), you may want to consider using a method for stiff equations. See the solver pages for more details (e.g. https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/#Stiff-Problems).")
        end
        return SciMLBase.ReturnCode.MaxIters
    end

    # The last part:
    # Bail out if we take a step with dt less than the minimum value (which may be time dependent)
    # except if we are successfully taking such a small timestep is to hit a tstop exactly
    # We also exit if the ODE is unstable according to a user chosen callback
    # but only if we accepted the step to prevent from bailing out as unstable
    # when we just took way too big a step)
    step_accepted = should_accept_step(integrator)
    step_rejected = !step_accepted
    force_dtmin   = hasproperty(integrator, :force_dtmin) && integrator.force_dtmin
    if !force_dtmin && SciMLBase.isadaptive(integrator)
        dt_below_min      = abs(integrator.dt) ≤ abs(opts.dtmin)
        before_next_tstop = SciMLBase.has_tstop(integrator) ? integrator.t + integrator.dt < integrator.tdir * SciMLBase.first_tstop(integrator) : true
        if dt_below_min && (step_rejected || before_next_tstop)
            if verbose
                controller_string = controller_message_on_dtmin_error(integrator)
                @warn("dt($(integrator.dt)) <= dtmin($(opts.dtmin)) at t=$(integrator.t)$(controller_string). Aborting. There is either an error in your model specification or the true solution is unstable.")
            end
            return SciMLBase.ReturnCode.DtLessThanMin
        elseif step_rejected && integrator.t isa AbstractFloat &&
               abs(integrator.dt) <= abs(eps(integrator.t)) # = DiffEqBase.timedepentdtmin(integrator)
            if verbose
                controller_string = controller_message_on_dtmin_error(integrator)
                @warn("At t=$(integrator.t), dt was forced below floating point epsilon $(integrator.dt)$(controller_string). Aborting. There is either an error in your model specification or the true solution is unstable (or the true solution can not be represented in the precision of $(eltype(integrator.u))).")
            end
            return SciMLBase.ReturnCode.Unstable
        end
    end
    if step_accepted && (hasproperty(opts, :unstable_check) && 
       opts.unstable_check(integrator.dt, integrator.u, integrator.p, integrator.t))
        if verbose
            @warn("Instability detected. Aborting")
        end
        return SciMLBase.ReturnCode.Unstable
    end
    if SciMLBase.last_step_failed(integrator) && !SciMLBase.isadaptive(integrator)
        if verbose
            @warn("Newton steps could not converge and algorithm is not adaptive. Use a lower dt.")
        end
        return SciMLBase.ReturnCode.ConvergenceFailure
    end
    return SciMLBase.ReturnCode.Success
end

function footer_reset_flags!(integrator)
    integrator.u_modified = false
end

function fix_solution_buffer_sizes!(integrator, sol)
    resize!(integrator.sol.t, integrator.saveiter)
    resize!(integrator.sol.u, integrator.saveiter)
    if !(integrator.sol isa SciMLBase.DAESolution)
        resize!(integrator.sol.k, integrator.saveiter_dense)
    end
end

function setup_validity_flags!(integrator, t_next)
    integrator.isout = integrator.opts.isoutofdomain(integrator.u, integrator.p, t_next)
end

function step_footer!(integrator::ThunderboltTimeIntegrator)
    ttmp = integrator.t + integrator.tdir * integrator.dt

    footer_reset_flags!(integrator)
    setup_validity_flags!(integrator, ttmp)

    if should_accept_step(integrator)
        integrator.last_step_failed = false
        integrator.tprev = integrator.t
        integrator.t = OrdinaryDiffEqCore.fixed_t_for_floatingpoint_error!(integrator, ttmp)
        OrdinaryDiffEqCore.handle_callbacks!(integrator)
        adapt_dt!(integrator) # Noop for non-adaptive algorithms
    elseif integrator.force_stepfail
        if SciMLBase.isadaptive(integrator)
            OrdinaryDiffEqCore.post_newton_controller!(integrator, integrator.alg)
        # elseif integrator.dtchangeable # Non-adaptive but can change dt
        #     integrator.dt *= integrator.opts.failfactor
        elseif integrator.last_step_failed
            return
        end
        integrator.last_step_failed = true
    end

    integration_monitor_step(integrator)

    return nothing
end

notify_integrator_hit_tstop!(integrator) = integrator.just_hit_tstop = true

is_first_iteration(integrator) = integrator.iter == 0
increment_iteration(integrator) = integrator.iter += 1

function integration_monitor_step(integrator)
    if integrator.opts.progress && integrator.iter % integrator.opts.progress_steps == 0
        integration_step_monitor(integrator, integrator.opts.progress_monitor)
    end
end

function finalize_integration_monitor(integrator)
    if integrator.opts.progress
        integration_finalize_monitor(integrator, integrator.opts.progress_monitor)
    end
end

notify_integrator_hit_tstop!(integrator::ThunderboltTimeIntegrator) = nothing

# TODO upstream into OrdinaryDiffEqCore
function compute_rate_prototype(prob)
    u = prob.u0

    tType = eltype(prob.tspan)
    tTypeNoUnits = typeof(one(tType))

    isdae = (
        prob.f.mass_matrix != I &&
        !(prob.f.mass_matrix isa Tuple) &&
        ArrayInterface.issingular(prob.f.mass_matrix)
    )
    if !isdae && isinplace(prob) && u isa AbstractArray && eltype(u) <: Number &&
        uBottomEltypeNoUnits == uBottomEltype && tType == tTypeNoUnits # Could this be more efficient for other arrays?
        return SciMLBase.recursivecopy(u)
    else
        _compute_rate_prototype_mass_matrix_form(prob)
    end
end
function compute_rate_prototype(prob::SciMLBase.DiscreteProblem)
    _compute_rate_prototype_mass_matrix_form(prob)
end
function _compute_rate_prototype_mass_matrix_form(prob)
    u = prob.u0

    tType = eltype(prob.tspan)
    tTypeNoUnits = typeof(one(tType))
    uBottomEltype        = OrdinaryDiffEqCore.recursive_bottom_eltype(u)
    uBottomEltypeNoUnits = OrdinaryDiffEqCore.recursive_unitless_bottom_eltype(u)
    if (uBottomEltypeNoUnits == uBottomEltype && tType == tTypeNoUnits) || eltype(u) <: Enum
        return u
    else # has units!
        return u / oneunit(tType)
    end
end
function compute_rate_prototype(prob::SciMLBase.DAEProblem)
    return prob.du0
end

function compute_rate_prototype(prob::AbstractSemidiscreteProblem)
    _compute_rate_prototype_mass_matrix_form(prob)
end

# Slightly modified functions are here
function OrdinaryDiffEqCore.modify_dt_for_tstops!(integrator::ThunderboltTimeIntegrator)
    if SciMLBase.has_tstop(integrator)
        tdir_t = integrator.tdir * integrator.t
        tdir_tstop = SciMLBase.first_tstop(integrator)
        if integrator.opts.adaptive
            integrator.dt = integrator.tdir *
                            min(abs(integrator.dt), abs(tdir_tstop - tdir_t)) # step! to the end
        elseif integrator.dtchangeable
            dtpropose = SciMLBase.get_proposed_dt(integrator)
            if iszero(dtpropose)
                integrator.dt = integrator.tdir * abs(tdir_tstop - tdir_t)
            elseif !integrator.force_stepfail
                # always try to step! with dtcache, but lower if a tstop
                # however, if force_stepfail then don't set to dtcache, and no tstop worry
                integrator.dt = integrator.tdir *
                                min(abs(dtpropose), abs(tdir_tstop - tdir_t)) # step! to the end
            end
        end
    end
end

function OrdinaryDiffEqCore.handle_tstop!(integrator::ThunderboltTimeIntegrator)
    if SciMLBase.has_tstop(integrator)
        tdir_t = integrator.tdir * integrator.t
        tdir_tstop = SciMLBase.first_tstop(integrator)
        if tdir_t == tdir_tstop
            while tdir_t == tdir_tstop #remove all redundant copies
                res = SciMLBase.pop_tstop!(integrator)
                SciMLBase.has_tstop(integrator) ? (tdir_tstop = SciMLBase.first_tstop(integrator)) : break
            end
            notify_integrator_hit_tstop!(integrator)
        elseif tdir_t > tdir_tstop
            if !integrator.dtchangeable
                SciMLBase.change_t_via_interpolation!(integrator,
                    integrator.tdir *
                    SciMLBase.pop_tstop!(integrator), Val{true})
                    notify_integrator_hit_tstop!(integrator)
            else
                error("Something went wrong. Integrator stepped past tstops but the algorithm was dtchangeable. Please report this error.")
            end
        end
    end
    return nothing
end
