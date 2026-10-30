"""
    HomotopyPathSolver(inner_solver)

Solve the nonlinear problem `F(u,t)=0` with given time increments `Δt`on some interval `[t_begin, t_end]`
where `t` is some pseudo-time parameter.
"""
struct HomotopyPathSolver{LS} <: AbstractSolver
    # Read once, at setup, to build the nonlinear solver cache.
    inner_solver::AbstractNonlinearSolver
    # Solves `0 = L(F, Q)` per quadrature point for a `SteadyStateEvolution` material.
    local_solver::LS
end

HomotopyPathSolver(inner_solver::AbstractNonlinearSolver) =
    HomotopyPathSolver(inner_solver, GenericLocalNonlinearSolver())

mutable struct HomotopyPathSolverCache{SFT, T, VT <: AbstractVector{T}, VTprev} <:
               AbstractTimeSolverCache
    # Continuation condenses nothing, so the stage unknowns are the function's.
    stage_function::SFT
    # Entered once per load step, through `nlsolve!`.
    inner_solver_cache::AbstractNonlinearSolverCache
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
check_internal_variables_are_rate_free(f::QuasiStaticFunction) =
    foreach(_check_model_is_rate_free, _volume_models(get_volume_integrator(f)))

# Unknown integrator types deliberately have no fallback here: silently skipping the check would be
# worse than the `MethodError`.
_volume_models(integrator::NonlinearIntegrator) = (integrator.volume_model,)
_volume_models(integrator::NonlinearMultiDomainIntegrator2) =
    (subintegrator.volume_model for subintegrator in values(integrator.subintegrators))

"""
    check_weak_boundary_conditions_are_rate_free(f)

Reject a dashpot boundary condition, which continuation cannot form a velocity for.

The surface counterpart of [`check_internal_variables_are_rate_free`](@ref), and rejected for the same
reason: `HomotopyPathSolver` is load stepping, so it offers neither a previous solution nor a timestep.
Without this the combination fails inside the assembly loop, where the message is a missing field on a
`Float64` rather than a named boundary condition.
"""
check_weak_boundary_conditions_are_rate_free(f) = nothing
check_weak_boundary_conditions_are_rate_free(f::QuasiStaticFunction) =
    foreach(_check_facet_model_is_rate_free, _facet_models(get_volume_integrator(f)))

_facet_models(integrator::NonlinearIntegrator) = (integrator.facet_model,)
_facet_models(integrator::NonlinearMultiDomainIntegrator2) =
    (subintegrator.facet_model for subintegrator in values(integrator.subintegrators))

# A facet model is either one boundary condition or a tuple of them. Anything else -- a wrapper this
# check does not know -- is passed over rather than guessed at; the assembly still refuses it, only
# less legibly.
_check_facet_model_is_rate_free(facet_model::Tuple) =
    foreach(_check_facet_model_is_rate_free, facet_model)
_check_facet_model_is_rate_free(bc::ConsistencyCheckWeakBoundaryCondition) =
    _check_facet_model_is_rate_free(bc.bc)
_check_facet_model_is_rate_free(bc) = nothing
_check_facet_model_is_rate_free(bc::AbstractViscousWeakBoundaryCondition) = error(
    "$(typeof(bc).name.name) on boundary \"$(bc.boundary_name)\" resists the velocity, which " *
    "`HomotopyPathSolver` cannot supply: continuation is load stepping, so it has neither a " *
    "previous solution nor a timestep. Use a time integrator instead — `BackwardEulerSolver` for a " *
    "quasi-static problem, or `NewmarkSolver` when inertia matters. The corresponding spring " *
    "(`RobinBC`, `NormalSpringBC`) resists the displacement and is accepted here.",
)

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
setup_local_solver_cache(f::QuasiStaticFunction, solver::HomotopyPathSolver) =
    _setup_local_solver_cache(solver.local_solver, f.integrator, f.dh, f.lvh)

# A function that is not a quasi-static one poses no quadrature point local problem the continuation
# could solve -- a `NullFunction` has no elements at all.
setup_local_solver_cache(f, ::HomotopyPathSolver) = nothing

setup_stage_operator(
    f::AbstractSemidiscreteFunction,
    solver::HomotopyPathSolver,
    local_solver_cache,
    t₀,
) = setup_operator(
    get_strategy(f),
    _annotate_with_local_solver_cache(get_volume_integrator(f), local_solver_cache),
    f.dh;
    slots = THUNDERBOLT_STAGE_SLOTS,
)

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

# Continuation is load stepping: there is no timestep and no rate to reconstruct, so the context
# carries the pseudo-time alone and the scheme matrix is the plain `∂F/∂u`. `f` selects the
# parameter bag: most functions need none, but e.g. `RSAFDQ20223DFunction` overrides this to carry
# the solver-supplied chamber reference volumes (see `rsafdq2022.jl`).
_homotopy_stage_evaluation(f, t) = StageEvaluation(;
    ctx       = TimeIntegrationContext(t, zero(t), zero(t)),
    condensed = _homotopy_condenses(f),
)

# Only a solid mechanics function can carry a condensed tail, and only that one answers the query.
_homotopy_condenses(f) = false
_homotopy_condenses(f::AbstractSolidMechanicsFunction) = has_internal_variables(f)

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
    check_weak_boundary_conditions_are_rate_free(f)
    # The stage carries the operator, so it is built before the solver cache that works on it. A
    # continuation offers neither a previous solution nor a timestep, so its context is the bare
    # pseudo-time; `_homotopy_stage_evaluation` decides what else `f` needs in `p`.
    local_solver_cache = setup_local_solver_cache(f, solver)
    stage_function = FullStateStage(
        f,
        setup_stage_operator(f, solver, local_solver_cache, t₀),
        _homotopy_stage_evaluation(f, t₀),
    )
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

function perform_step!(
    f::AbstractSemidiscreteFunction,
    solver_cache::HomotopyPathSolverCache,
    t,
    Δt,
)
    update_constraints!(f, solver_cache, t + Δt)
    sf = solver_cache.stage_function
    set_stage_parameters!(sf, _homotopy_stage_evaluation(f, t + Δt))
    if !nlsolve!(solver_cache.uₙ, sf, solver_cache.inner_solver_cache, t + Δt)
        return false
    end

    return true
end

contraction_rate_cache(cache::HomotopyPathSolverCache) =
    global_newton_cache(cache.inner_solver_cache)

# A rejected load step discards the trial `q` along with `u`, so the correctors computed for it have
# to go too -- the generic `restore_state!` only copies.
restore_state!(u::AbstractVector, uprev::AbstractVector, cache::HomotopyPathSolverCache) =
    rollback_stage!(u, uprev, cache.stage_function)

# --- convergence driven step size control --------------------------------------------------------
#
# The controllers below are Deuflhard's, and they read only how well Newton contracted — never a local
# error estimate. They therefore dispatch on the *controller*, not on the solver cache, and ask for the
# Newton cache through [`contraction_rate_cache`](@ref). Any scheme that answers that query can use
# them; [`BackwardEulerSolver`](@ref) does.
#
# That combination is the point rather than an accident. When a rate term is present only to
# regularize — a dashpot added so a quasi-static solve has something to contract against — the
# temporal error is not a quantity anyone wants to control, and asking the step size to track the
# Newton convergence instead is exactly right. Backward Euler brings no error estimate, so a
# `PIDController` has nothing to work with there; these have everything they need.

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
    cache,
    controller::Deuflhard2004DiscreteContinuationController,
)
    (; Θks) = contraction_rate_cache(cache)
    (; Θreject) = controller
    if contraction_rate_cache(cache).parameters.enforce_monotonic_convergence
        result = all(Θks .≤ Θreject)
        return result
    else
        return all(isfinite.(Θks))
    end
end
function reject_step!(
    integrator::ThunderboltTimeIntegrator,
    cache,
    controller::Deuflhard2004DiscreteContinuationController,
)
    # `dt` shrinks once per failed attempt: the step footer's `post_newton_controller!` owns the
    # solve-failure case, this hook owns the convergence-rate case. The state restore is
    # `rollback_state!`'s.
    integrator.force_stepfail && return nothing

    @inline g(x) = √(1+4x) - 1

    # Shorten dt according to (Eq. 5.24)
    (; Θks) = contraction_rate_cache(cache)
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
    cache,
    controller::Deuflhard2004DiscreteContinuationController,
)
    @inline g(x) = √(1+4x) - 1

    # Adapt dt with a priori estimate (Eq. 5.24)
    (; Θks) = contraction_rate_cache(cache)
    (; Θbar, γ, Θmin, qmin, qmax, p) = controller

    Θ₀ = length(Θks) > 0 ? max(first(Θks), Θmin) : Θmin
    q = clamp(γ * (g(Θbar)/(2Θ₀))^(1/p), qmin, qmax)
    integrator.dt = min(q * integrator.dt, integrator.opts.dtmax)
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
    cache,
    controller::Deuflhard2004_B_DiscreteContinuationControllerVariant,
)
    (; Θks) = contraction_rate_cache(cache)
    (; Θreject) = controller
    if contraction_rate_cache(cache).parameters.enforce_monotonic_convergence
        result = all(Θks .≤ Θreject)
        return result
    else
        return all(isfinite.(Θks))
    end
end
function reject_step!(
    integrator::ThunderboltTimeIntegrator,
    cache,
    controller::Deuflhard2004_B_DiscreteContinuationControllerVariant,
)
    integrator.force_stepfail && return nothing

    @inline g(x) = √(1+4x) - 1

    # Shorten dt according to (Eq. 5.24)
    (; Θks) = contraction_rate_cache(cache)
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
    cache,
    controller::Deuflhard2004_B_DiscreteContinuationControllerVariant,
)
    @inline g(x) = √(1+4x) - 1

    # Adapt dt with a priori estimate (Eq. 5.24)
    (; Θks) = contraction_rate_cache(cache)
    (; Θbar, γ, Θmin, qmin, qmax, p) = controller

    Θ₀ = length(Θks) > 0 ? max(first(Θks), Θmin) : Θmin
    q = clamp(γ * (g(Θbar)/(g(Θ₀)))^(1/p), qmin, qmax)
    integrator.dt = min(q * integrator.dt, integrator.opts.dtmax)
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
    cache,
    controller::ExperimentalDiscreteContinuationController,
)
    (; Θks) = contraction_rate_cache(cache)
    (; Θreject) = controller
    if contraction_rate_cache(cache).parameters.enforce_monotonic_convergence
        result = all(Θks .≤ Θreject)
        return result
    else
        return all(isfinite.(Θks))
    end
end
function reject_step!(
    integrator::ThunderboltTimeIntegrator,
    cache,
    controller::ExperimentalDiscreteContinuationController,
)
    integrator.force_stepfail && return nothing

    @inline g(x) = √(1+4x) - 1

    # Shorten dt according to (Eq. 5.24)
    (; Θks) = contraction_rate_cache(cache)
    (; Θbar, γ, Θmin, qmin, qmax, p) = controller
    Θk = maximum(Θks)
    q = clamp(γ * (g(Θbar)/g(Θk))^(1/p), qmin, qmax)
    integrator.dt = min(q * integrator.dt, integrator.opts.dtmax)
end

function adapt_dt!(
    integrator::ThunderboltTimeIntegrator,
    cache,
    controller::ExperimentalDiscreteContinuationController,
)
    @inline g(x) = √(1+4x) - 1

    # Adapt dt with a priori estimate (Eq. 5.24)
    (; Θks) = contraction_rate_cache(cache)
    (; Θbar, γ, Θmin, qmin, qmax, p) = controller
    Θ₀ = length(Θks) > 0 ? max(mean(Θks), Θmin) : Θmin
    q = clamp(γ * (g(Θbar)/(2Θ₀))^(1/p), qmin, qmax)
    integrator.dt = min(q * integrator.dt, integrator.opts.dtmax)
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

# The controller is its own cache and holds no history; the contraction rates live in the solver cache.
OrdinaryDiffEqCore.reinit_controller!(
    ::SciMLBase.DEIntegrator,
    ::Union{
        Deuflhard2004DiscreteContinuationController,
        Deuflhard2004_B_DiscreteContinuationControllerVariant,
        ExperimentalDiscreteContinuationController,
    },
) = nothing
