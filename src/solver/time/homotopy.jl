"""
    HomotopyPathSolver{IS, T, PFUN}

Solve the nonlinear problem `F(u,t)=0` with given time increments `Δt`on some interval `[t_begin, t_end]`
where `t` is some pseudo-time parameter.
"""
struct HomotopyPathSolver{IS} <: AbstractSolver
    inner_solver::IS
end

mutable struct HomotopyPathSolverCache{SFT, ISC, T, VT <: AbstractVector{T}, VTprev} <:
               AbstractTimeSolverCache
    # Continuation condenses nothing, so the stage unknowns are the function's.
    stage_function::SFT
    inner_solver_cache::ISC
    uₙ::VT
    uₙ₋₁::VTprev
    tmp::VT
end

"""
    check_internal_variables_are_rate_free(f)

Reject a model whose internal variable carries its own evolution law, which continuation cannot
integrate along a load path.

`HomotopyPathSolver` is load stepping, not a time scheme: it has neither a previous solution nor a
timestep, so `dₜQ = L(F, Q)` has nothing to be discretized against. The combination is rejected here,
during setup, so that it is reported once with a name and a remedy instead of surfacing per element
from the assembly loop.
"""
check_internal_variables_are_rate_free(f) = nothing
check_internal_variables_are_rate_free(f::AbstractSemidiscreteBlockedFunction) =
    foreach(check_internal_variables_are_rate_free, blocks(f))
check_internal_variables_are_rate_free(f::QuasiStaticFunction) =
    foreach(_check_model_is_rate_free, _volume_models(get_volume_integrator(f)))

# Unknown integrator types deliberately have no fallback here: silently skipping the check would be
# worse than the `MethodError`.
_volume_models(integrator::NonlinearIntegrator) = (integrator.volume_model,)
_volume_models(integrator::NonlinearMultiDomainIntegrator2) =
    (subintegrator.volume_model for subintegrator in values(integrator.subintegrators))

function _check_model_is_rate_free(model)
    evolution = internal_variable_evolution(model.material_model)
    is_rate_free(evolution) && return nothing
    error(
        "$(typeof(model.material_model).name.name) carries an internal variable with a time " *
        "derivative ($(typeof(evolution).name.name)), which `HomotopyPathSolver` cannot integrate: " *
        "continuation supplies neither a previous solution nor a timestep. Use a time integrator " *
        "instead. A material whose internal variable is genuinely steady state — an algebraic " *
        "`0 = L(F, Q)`, as in growth and remodelling — declares `SteadyStateEvolution()` and is " *
        "accepted here.\n" *
        "Note that `AsRateIndependent` does *not* help: it drops the velocity dependence, leaving " *
        "`dₜQ = L(F, Q)`, which still needs a timestep.",
    )
end

# Continuation poses the internal forces alone: no previous solution, no timestep, no inertia. The
# handler is the function's own, because for these functions the solution vector and the weak form
# live on the same one.
setup_stage_operator(
    f::AbstractSemidiscreteFunction,
    solver::HomotopyPathSolver,
    local_solver_cache,
    t₀,
) = setup_operator(get_strategy(f), get_volume_integrator(f), f.dh)

# A `NullFunction` matches both the null method (any solver) and the continuation method (any
# function), and neither signature dominates. The answer is the null operator either way.
setup_stage_operator(f::NullFunction, solver::HomotopyPathSolver, local_solver_cache, t₀) =
    NullOperator{Float64, solution_size(f), solution_size(f)}()

# An elastodynamics function's solution vector carries a velocity field that the internal forces have
# no equation for, so there is no one operator that answers for it. Refusing is the honest answer:
# pairing the displacement's integrator with the state handler would assemble a residual into a
# handler twice its size.
setup_stage_operator(
    f::ElastodynamicsFunction,
    solver::HomotopyPathSolver,
    local_solver_cache,
    t₀,
) = error(
    "An elastodynamics function has no single operator: the inertia belongs to the time scheme, not " *
    "to the function. Pose the continuation on `f.structural` to solve for the static equilibrium.",
)

function setup_solver_cache(
    f::AbstractSemidiscreteFunction,
    solver::HomotopyPathSolver,
    t₀;
    uprev       = nothing,
    u           = nothing,
    alias_uprev = true,
    alias_u     = false,
)
    check_internal_variables_are_rate_free(f)
    # The stage carries the operator, so it is built before the solver cache that works on it. A
    # continuation offers neither a previous solution nor a timestep, so its parameters are the bare
    # pseudo-time.
    stage_function = FullStateStage(f, setup_stage_operator(f, solver, nothing, t₀), t₀)
    inner_solver_cache = setup_solver_cache(stage_function, solver.inner_solver)

    vtype = Vector{Float64}

    if u === nothing
        _u = vtype(undef, solution_size(f))
        @warn "Cannot initialize u for $(typeof(solver))."
    else
        _u = alias_u ? u : recursivecopy(u)
    end

    if uprev === nothing
        _uprev = vtype(undef, solution_size(f))
        _uprev .= u
    else
        _uprev = alias_uprev ? uprev : recursivecopy(uprev)
    end

    solver_cache = HomotopyPathSolverCache(
        stage_function,
        inner_solver_cache,
        _u,
        _uprev,
        vtype(undef, solution_size(f)),
    )

    # Make sure the initial state is consistent
    perform_step!(f, solver_cache, t₀, 0.0) ||
        error("Initial guess is not consistent with the model or the problem is not well-posed!")

    return solver_cache
end

function setup_solver_cache(
    f::AbstractSemidiscreteBlockedFunction,
    solver::HomotopyPathSolver,
    t₀;
    uprev       = nothing,
    u           = nothing,
    alias_uprev = true,
    alias_u     = false,
)
    check_internal_variables_are_rate_free(f)
    stage_function = FullStateStage(f, setup_stage_operator(f, solver, nothing, t₀), t₀)
    inner_solver_cache = setup_solver_cache(stage_function, solver.inner_solver)

    vtype = Vector{Float64}
    if u === nothing
        _u = mortar([vtype(undef, solution_size(fi)) for fi ∈ blocks(f)])
        @warn "Cannot initialize u for $(typeof(solver))."
    else
        if alias_u
            _u = u
        else
            _u = mortar([vtype(undef, solution_size(fi)) for fi ∈ blocks(f)])
            _u .= u
        end
    end

    if uprev === nothing
        _uprev = mortar([vtype(undef, solution_size(fi)) for fi ∈ blocks(f)])
        _uprev .= u
    else
        if alias_uprev
            _uprev = uprev
        else
            _uprev = mortar([vtype(undef, solution_size(fi)) for fi ∈ blocks(f)])
            _uprev .= uprev
        end
    end

    solver_cache = HomotopyPathSolverCache(
        stage_function,
        inner_solver_cache,
        _u,
        _uprev,
        mortar([vtype(undef, solution_size(fi)) for fi ∈ blocks(f)]),
    )

    # Make sure the initial state is consistent
    perform_step!(f, solver_cache, t₀, 0.0) ||
        error("Initial guess is not consistent with the model or the problem is not well-posed!")

    return solver_cache
end

function perform_step!(
    f::AbstractSemidiscreteFunction,
    solver_cache::HomotopyPathSolverCache,
    t,
    Δt,
)
    update_constraints!(f, solver_cache, t + Δt)
    sf = solver_cache.stage_function
    set_stage_parameters!(sf, t + Δt)
    if !nlsolve!(solver_cache.uₙ, sf, solver_cache.inner_solver_cache, t + Δt)
        return false
    end

    return true
end

@doc raw"""
    Deuflhard2004DiscreteContinuationController(Θbar, p)

Θbar ($\overbar{\Theta}$) is the target convergence rate.

Θk ($\Theta_0$) is the estimated convergence rate for the nonlinear solve iteration k.

Predictor time step length: $\Delta t^0_n = \sqrt[p]{\frac{g(\overbar{\Theta})}{2\Theta_0}} \Delta t^{\textrm{last}}_{n-1}$ [Deu:2004:nmn; p. 248](@cite)

Predictor time step length: $\Delta t^i_n = \sqrt[p]{\frac{\overbar{\Theta}}{\Theta}_k} \Delta t^{i-1}_{n-1}$ [Deu:2004:nmn; Eq. 5.24, p. 248](@cite)

Here $g(x) = \sqrt{1+4\Theta}-1$ and $\Theta_0 \geq \Theta_{\textrm{min}}$

The retry criterion for the time step is $\Theta}_k > \frac{1}{2}$.
"""
Base.@kwdef struct Deuflhard2004DiscreteContinuationController
    Θmin::Float64
    p::Int64
    Θreject::Float64 = 0.95
    Θbar::Float64 = 0.5
    γ::Float64 = 0.95
    qmin::Float64 = 1/5
    qmax::Float64 = 5.0
end

function should_accept_step(
    integrator::ThunderboltTimeIntegrator,
    cache::HomotopyPathSolverCache,
    controller::Deuflhard2004DiscreteContinuationController,
)
    (; Θks) = global_newton_cache(cache.inner_solver_cache)
    (; Θreject) = controller
    if global_newton_cache(cache.inner_solver_cache).parameters.enforce_monotonic_convergence
        result = all(Θks .≤ Θreject)
        return result
    else
        return all(isfinite.(Θks))
    end
end
function reject_step!(
    integrator::ThunderboltTimeIntegrator,
    cache::HomotopyPathSolverCache,
    controller::Deuflhard2004DiscreteContinuationController,
)
    # `dt` shrinks once per failed attempt: the step footer's `post_newton_controller!` owns the
    # solve-failure case, this hook owns the convergence-rate case. The state restore is
    # `rollback_state!`'s.
    integrator.force_stepfail && return nothing

    @inline g(x) = √(1+4x) - 1

    # Shorten dt according to (Eq. 5.24)
    (; Θks) = global_newton_cache(cache.inner_solver_cache)
    (; Θbar, Θreject, γ, Θmin, qmin, qmax, p) = controller
    for Θk in Θks
        if Θk > Θreject
            q = clamp(γ * (g(Θbar)/g(Θk))^(1/p), qmin, qmax)
            integrator.dt = q * integrator.dt
            return
        end
    end
end

function adapt_dt!(
    integrator::ThunderboltTimeIntegrator,
    cache::HomotopyPathSolverCache,
    controller::Deuflhard2004DiscreteContinuationController,
)
    @inline g(x) = √(1+4x) - 1

    # Adapt dt with a priori estimate (Eq. 5.24)
    (; Θks) = global_newton_cache(cache.inner_solver_cache)
    (; Θbar, γ, Θmin, qmin, qmax, p) = controller

    Θ₀ = length(Θks) > 0 ? max(first(Θks), Θmin) : Θmin
    q = clamp(γ * (g(Θbar)/(2Θ₀))^(1/p), qmin, qmax)
    integrator.dt = q * integrator.dt
end

Base.@kwdef struct Deuflhard2004_B_DiscreteContinuationControllerVariant
    Θmin::Float64
    p::Int64
    Θreject::Float64 = 0.95
    Θbar::Float64 = 0.5
    γ::Float64 = 0.95
    qmin::Float64 = 1/5
    qmax::Float64 = 5.0
end

function should_accept_step(
    integrator::ThunderboltTimeIntegrator,
    cache::HomotopyPathSolverCache,
    controller::Deuflhard2004_B_DiscreteContinuationControllerVariant,
)
    (; Θks) = global_newton_cache(cache.inner_solver_cache)
    (; Θreject) = controller
    if global_newton_cache(cache.inner_solver_cache).parameters.enforce_monotonic_convergence
        result = all(Θks .≤ Θreject)
        return result
    else
        return all(isfinite.(Θks))
    end
end
function reject_step!(
    integrator::ThunderboltTimeIntegrator,
    cache::HomotopyPathSolverCache,
    controller::Deuflhard2004_B_DiscreteContinuationControllerVariant,
)
    integrator.force_stepfail && return nothing

    @inline g(x) = √(1+4x) - 1

    # Shorten dt according to (Eq. 5.24)
    (; Θks) = global_newton_cache(cache.inner_solver_cache)
    (; Θbar, Θreject, γ, Θmin, qmin, qmax, p) = controller
    for Θk in Θks
        if Θk > Θreject
            q = clamp(γ * (g(Θbar)/g(Θk))^(1/p), qmin, qmax)
            integrator.dt = q * integrator.dt
            return
        end
    end
end

function adapt_dt!(
    integrator::ThunderboltTimeIntegrator,
    cache::HomotopyPathSolverCache,
    controller::Deuflhard2004_B_DiscreteContinuationControllerVariant,
)
    @inline g(x) = √(1+4x) - 1

    # Adapt dt with a priori estimate (Eq. 5.24)
    (; Θks) = global_newton_cache(cache.inner_solver_cache)
    (; Θbar, γ, Θmin, qmin, qmax, p) = controller

    Θ₀ = length(Θks) > 0 ? max(first(Θks), Θmin) : Θmin
    q = clamp(γ * (g(Θbar)/(g(Θ₀)))^(1/p), qmin, qmax)
    integrator.dt = q * integrator.dt
end

@doc raw"""
    ExperimentalDiscreteContinuationController(Θbar, p)
"""
Base.@kwdef struct ExperimentalDiscreteContinuationController
    Θmin::Float64
    p::Int64
    Θreject::Float64 = 0.9
    Θbar::Float64 = 0.75
    γ::Float64 = 0.95
    qmin::Float64 = 1/5
    qmax::Float64 = 5.0
end

function should_accept_step(
    integrator::ThunderboltTimeIntegrator,
    cache::HomotopyPathSolverCache,
    controller::ExperimentalDiscreteContinuationController,
)
    (; Θks) = global_newton_cache(cache.inner_solver_cache)
    (; Θreject) = controller
    if global_newton_cache(cache.inner_solver_cache).parameters.enforce_monotonic_convergence
        result = all(Θks .≤ Θreject)
        return result
    else
        return all(isfinite.(Θks))
    end
end
function reject_step!(
    integrator::ThunderboltTimeIntegrator,
    cache::HomotopyPathSolverCache,
    controller::ExperimentalDiscreteContinuationController,
)
    integrator.force_stepfail && return nothing

    @inline g(x) = √(1+4x) - 1

    # Shorten dt according to (Eq. 5.24)
    (; Θks) = global_newton_cache(cache.inner_solver_cache)
    (; Θbar, γ, Θmin, qmin, qmax, p) = controller
    Θk = maximum(Θks)
    q = clamp(γ * (g(Θbar)/g(Θk))^(1/p), qmin, qmax)
    integrator.dt = q * integrator.dt
end

function adapt_dt!(
    integrator::ThunderboltTimeIntegrator,
    cache::HomotopyPathSolverCache,
    controller::ExperimentalDiscreteContinuationController,
)
    @inline g(x) = √(1+4x) - 1

    # Adapt dt with a priori estimate (Eq. 5.24)
    (; Θks) = global_newton_cache(cache.inner_solver_cache)
    (; Θbar, γ, Θmin, qmin, qmax, p) = controller
    Θ₀ = length(Θks) > 0 ? max(mean(Θks), Θmin) : Θmin
    q = clamp(γ * (g(Θbar)/(2Θ₀))^(1/p), qmin, qmax)
    integrator.dt = q * integrator.dt
end



# OrdinaryDiffEqCore.default_controller(QT, ::HomotopyPathSolver) = ExperimentalDiscreteContinuationController(; Θmin=1/8, p=1)
OrdinaryDiffEqCore.default_controller(QT, ::HomotopyPathSolver) =
    Deuflhard2004_B_DiscreteContinuationControllerVariant(; Θmin = QT(1/8), p = 1)
SciMLBase.isadaptive(::HomotopyPathSolver) = true

OrdinaryDiffEqCore.setup_controller_cache(
    _alg,
    cache,
    controller::Union{
        Deuflhard2004DiscreteContinuationController,
        Deuflhard2004_B_DiscreteContinuationControllerVariant,
        ExperimentalDiscreteContinuationController,
    },
    EEstT,
    disco_probs,
) = controller
