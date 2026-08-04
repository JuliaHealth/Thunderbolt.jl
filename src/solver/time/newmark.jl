#####################################################################
#              Newmark-β time integration for elastodynamics        #
#####################################################################
@doc raw"""
    NewmarkSolver(; β, γ, inner_solver, system_matrix_type, monitor)

Newmark-β integration of the second order system ``M \ddot{u} + f_\mathrm{int}(u) = f_\mathrm{ext}``.

Given ``(u_n, v_n, a_n)`` the scheme forms the predictors
```math
\tilde{u} = u_n + \Delta t\, v_n + (\tfrac12 - \beta)\Delta t^2 a_n , \qquad
\tilde{v} = v_n + (1 - \gamma)\Delta t\, a_n
```
and solves the balance of momentum at ``t_{n+1}`` for the **displacement**, with
```math
a(u) = \frac{u - \tilde{u}}{\beta \Delta t^2} , \qquad v(u) = \tilde{v} + \gamma \Delta t\, a(u) .
```
The nonlinear solver therefore sees the residual and tangent
```math
r(u) = M a(u) + f_\mathrm{int}(u) - f_\mathrm{ext} , \qquad
J(u) = K(u) + \frac{1}{\beta \Delta t^2} M .
```

!!! note "Displacement form, not acceleration form"
    Textbooks usually solve the *acceleration* form, whose effective mass matrix
    ``M + \beta \Delta t^2 K`` is constant and can be factorized once. That relies on `K` being
    constant, which is true for linear elasticity and false for every material in this package. With
    a nonlinear internal force the unknown has to be the quantity the material is a function of. For
    a linear material the two forms are the same system scaled by ``\beta\Delta t^2``.

The defaults ``\beta = 1/4``, ``\gamma = 1/2`` are the average acceleration rule: unconditionally
stable, second order, and energy conserving. ``\gamma > 1/2`` adds numerical dissipation and drops
the scheme to first order.

Damping is not supported yet; the model is `M ü + f_int(u) = f_ext`.
"""
Base.@kwdef struct NewmarkSolver{T, SolverType, SystemMatrixType, MonitorType} <: AbstractSolver
    β::T                                       = 1 / 4
    γ::T                                       = 1 / 2
    inner_solver::SolverType                   = MultiLevelNewtonRaphsonSolver()
    system_matrix_type::Type{SystemMatrixType} = ThreadedSparseMatrixCSR{Float64, Int64}
    # Tolerances of the local error estimate, used only when the integrator runs adaptively
    reltol::T = 1.0e-3
    abstol::T = 1.0e-6
    # DO NOT USE THIS (will be replaced by proper logging system)
    monitor::MonitorType = DefaultProgressMonitor()
end

SciMLBase.isadaptive(::NewmarkSolver) = true

"""
    NewmarkStageOperator

The nonlinear operator of one Newmark stage: the internal force operator plus the inertia
contribution the time scheme adds on top of it.

This is the only place that knows the scheme's coefficients. It delegates to the wrapped operator for
`f_int`/`K` and then adds `M a(u)` to the residual and `M/(βΔt²)` to the linearization, so the
nonlinear solver keeps seeing a plain operator.

`M` and `K` are assembled separately and recombined rather than fused into one element loop. That is
the same choice [`BackwardEulerAffineODEStage`](@ref) makes, and it is what keeps a change of `Δt`
cheap — the mass matrix is constant, only its scalar weight moves.
"""
mutable struct NewmarkStageOperator{OpType, MassOpType, VectorType, T}
    const op::OpType
    const M::MassOpType
    # Displacement predictor ũ of the current step, written once per step
    const ũ::VectorType
    # Scratch for a(u)
    const aₜₘₚ::VectorType
    # βΔt² of the current step
    βΔt²::T
end

getJ(op::NewmarkStageOperator) = getJ(op.op)

# a(u) = (u - ũ)/(βΔt²), on the displacement dofs. `u` carries the condensed internal variables in its
# tail, which have no acceleration, so the view is not optional.
function _newmark_acceleration!(a, sop::NewmarkStageOperator, u::AbstractVector)
    ndofs = length(a)
    @inbounds @views @.. a = (u[1:ndofs] - sop.ũ) / sop.βΔt²
    return a
end

function _add_inertia_residual!(residual, sop::NewmarkStageOperator, u::AbstractVector)
    a = _newmark_acceleration!(sop.aₜₘₚ, sop, u)
    mul!(residual, sop.M, a, true, true)
    return nothing
end

# J ← K + M/(βΔt²). Both matrices come from the same `DofHandler` and therefore share a sparsity
# pattern, which is what makes the nonzero-wise update valid.
function _add_inertia_linearization!(sop::NewmarkStageOperator)
    Jnz = nonzeros(getJ(sop))
    Mnz = nonzeros(sop.M.A)
    @inbounds @.. Jnz = Jnz + Mnz / sop.βΔt²
    return nothing
end

function FerriteOperators.update_linearization!(
    sop::NewmarkStageOperator,
    residual::AbstractVector,
    u::AbstractVector,
    p,
)
    update_linearization!(sop.op, residual, u, p)
    _add_inertia_residual!(residual, sop, u)
    _add_inertia_linearization!(sop)
    return nothing
end

function FerriteOperators.update_linearization!(sop::NewmarkStageOperator, u::AbstractVector, p)
    update_linearization!(sop.op, u, p)
    _add_inertia_linearization!(sop)
    return nothing
end

function FerriteOperators.residual!(
    sop::NewmarkStageOperator,
    residual::AbstractVector,
    u::AbstractVector,
    p,
)
    residual!(sop.op, residual, u, p)
    _add_inertia_residual!(residual, sop, u)
    return nothing
end

struct NewmarkStageCache{SolverType, StageOpType, T}
    nlsolver::SolverType
    stage_op::StageOpType
    β::T
    γ::T
    # Error tolerances of the step size control. Kept here rather than on the controller so that
    # `perform_step!`, which never sees the integrator, can scale its estimate.
    reltol::T
    abstol::T
    # The estimator straddles a step, so it is not computable on the first one.
    first_step::Base.RefValue{Bool}
end

mutable struct NewmarkSolverCache{
    T,
    SolutionType <: AbstractVector{T},
    PrevSolutionType <: AbstractVector{T},
    TmpType <: AbstractVector{T},
    VelocityType <: AbstractVector{T},
    VelocityReferenceType <: AbstractVector{T},
    StageType,
    MonitorType,
} <: AbstractTimeSolverCache
    # Current solution buffer
    uₙ::SolutionType
    # Last solution buffer
    uₙ₋₁::PrevSolutionType
    # Temporary buffer for interpolations and stuff
    tmp::TmpType
    # Velocity and acceleration of the displacement dofs. Not part of the solution vector: the global
    # unknown is the displacement alone, see `ElastodynamicsModel`.
    vₙ::VelocityType
    aₙ::VelocityType
    # Rollback buffers, the `uprev` of the velocity and the acceleration. The integrator owns one for
    # the solution vector and restores it on a rejected step; `v` and `a` are state of the same ODE and
    # have to be rolled back with it, so they need their own -- see `reject_step!` below.
    vₙ₋₁::VelocityType
    aₙ₋₁::VelocityType
    # Scratch for the velocity predictor
    ṽ::VelocityType
    # The `uᵥ` of this step's `AffineVelocity`, held at full solution length so that the element query
    # can slice a cell out of it exactly as it does for the previous solution.
    uᵥ::VelocityReferenceType
    # Scaled local error estimate of the last attempted step, see `_newmark_error_estimate!`.
    # `Inf` until the second step, when the estimator becomes computable.
    EEst::T
    stage::StageType
    # DO NOT USE THIS (will be replaced by proper logging system)
    monitor::MonitorType
end

"""
    velocity(cache)
    acceleration(cache)

The velocity and acceleration the Newmark scheme reconstructed for the current solution. They are not
part of the solution vector, so this is how a consumer gets at them.
"""
velocity(cache::NewmarkSolverCache) = cache.vₙ
acceleration(cache::NewmarkSolverCache) = cache.aₙ
velocity(integrator::ThunderboltTimeIntegrator) = velocity(integrator.cache)
acceleration(integrator::ThunderboltTimeIntegrator) = acceleration(integrator.cache)

function setup_solver_cache(
    f::ElastodynamicsFunction,
    solver::NewmarkSolver,
    t₀;
    uprev       = nothing,
    u           = nothing,
    v0          = nothing,
    alias_uprev = true,
    alias_u     = false,
)
    vtype = Vector{Float64}
    nfe   = ndofs(f.dh)

    if u === nothing
        _u = vtype(undef, solution_size(f))
        @warn "Cannot initialize u for $(typeof(solver))."
    else
        _u = alias_u ? u : recursivecopy(u)
    end

    if uprev === nothing
        _uprev = vtype(undef, solution_size(f))
        _uprev .= _u
    else
        _uprev = alias_uprev ? uprev : recursivecopy(uprev)
    end

    (; integrator, dh) = f
    (; newton, local_solver) = solver.inner_solver

    local_solver_cache = _setup_local_solver_cache(local_solver, integrator, dh, f.lvh)
    op = setup_operator(
        f.assembly_strategy,
        _annotate_with_local_solver_cache(integrator, local_solver_cache),
        dh,
    )
    mass_operator = setup_operator(get_strategy(f), f.mass_term, solver, dh)
    @timeit_debug "mass assembly" update_operator!(mass_operator, t₀)

    stage_op = NewmarkStageOperator(
        op,
        mass_operator,
        zeros(nfe),
        zeros(nfe),
        one(Float64), # overwritten by the first step
    )
    nlsolver = _setup_multilevel_newton_cache(f, stage_op, local_solver_cache, newton, nfe)

    vₙ = v0 === nothing ? zeros(nfe) : recursivecopy(v0)
    aₙ = _consistent_initial_acceleration(f, stage_op, _u, vₙ, t₀)

    return NewmarkSolverCache(
        _u,
        _uprev,
        copy(_u),
        vₙ,
        aₙ,
        copy(vₙ),
        copy(aₙ),
        zeros(nfe),
        zeros(solution_size(f)),
        Inf,
        NewmarkStageCache(
            nlsolver,
            stage_op,
            solver.β,
            solver.γ,
            solver.reltol,
            solver.abstol,
            Ref(true),
        ),
        solver.monitor,
    )
end

@doc raw"""
    _consistent_initial_acceleration(f, stage_op, u₀, t₀)

Solve ``M a_0 = f_\mathrm{ext}(t_0) - f_\mathrm{int}(u_0)`` for the initial acceleration.

Newmark needs an acceleration to start from, and it is not free to choose: it is what the balance of
momentum says at `t₀`. Starting from `a₀ = 0` instead is only correct when the initial state is an
equilibrium, and is silently wrong otherwise — which is exactly the interesting case, a structure
released from a deflected state.
"""
function _consistent_initial_acceleration(f::ElastodynamicsFunction, stage_op, u₀, v₀, t₀)
    nfe = ndofs(f.dh)
    r = zeros(nfe)

    # Two things have to be right about this evaluation, and the Newmark parameter object is what
    # makes both expressible:
    #
    #  * the internal state must not advance -- the forces are wanted at `Q₀`, not one step past it.
    #    A vanishing timestep does that: `(Q - Qprev)/Δt = L(F, Q)` forces `Q → Qprev` as `Δt → 0`, so
    #    the local solve returns the state it was handed.
    #  * the deformation rate must be `∇v₀`, not zero. A rate dependent material's stress reads it, and
    #    at `t₀` the body is moving at `v₀`. Taking the slope `1` puts the velocity reference at
    #    `u₀ - v₀`, for which `∂v∂u (u₀ - uᵥ) = v₀` exactly.
    #
    # `u₀` is copied because writing the condensed tail back is what the assembly does with it.
    uᵥ = copy(u₀)
    @inbounds @views @.. uᵥ[1:nfe] = u₀[1:nfe] - v₀
    p = NewmarkTimeParameters(
        nothing,
        t₀,
        eps(Float64),
        AffineVelocity(one(eltype(u₀)), uᵥ),
        copy(u₀),
    )
    residual!(stage_op.op, r, copy(u₀), p)
    r .= .-r

    # On a copy of the mass matrix: `apply_zero!` rewrites the constrained rows and columns, and the
    # stage operator keeps using `M` for every step afterwards.
    M = copy(SparseMatrixCSC(stage_op.M.A))
    apply_zero!(M, r, getch(f))
    a₀ = M \ r
    apply_zero!(a₀, getch(f))
    return a₀
end

@doc raw"""
    _newmark_affine_velocity!(uᵥ, ũ, ṽ, ∂v∂u)

Write the displacement at which the reconstructed velocity vanishes, i.e. the `uᵥ` of the
[`AffineVelocity`](@ref) this step hands the element.

Inserting the corrector ``v = \tilde{v} + \gamma\Delta t\,(u - \tilde{u})/(\beta\Delta t^2)`` into
``v(u) = \partial_u v\,(u - u_v)`` and collecting terms gives ``u_v = \tilde{u} - \tilde{v}/\partial_u v``.
The element then forms the rate from the *same* two ingredients backward Euler uses, and needs to know
nothing about Newmark.
"""
function _newmark_affine_velocity!(uᵥ, ũ, ṽ, ∂v∂u)
    @inbounds @views @.. uᵥ[1:length(ũ)] = ũ - ṽ / ∂v∂u
    return uᵥ
end

function perform_step!(f::ElastodynamicsFunction, cache::NewmarkSolverCache, t, Δt)
    (; uₙ, uₙ₋₁, vₙ, aₙ, ṽ, uᵥ, stage) = cache
    (; nlsolver, stage_op, β, γ) = stage
    nfe = length(vₙ)

    update_constraints!(f, cache, t + Δt)

    # Predictors, in the same shape as the Ferrite reference implementation.
    @inbounds @views @.. stage_op.ũ = uₙ₋₁[1:nfe] + Δt * vₙ + (1 / 2 - β) * Δt^2 * aₙ
    @inbounds @.. ṽ = vₙ + (1 - γ) * Δt * aₙ
    stage_op.βΔt² = β * Δt^2

    # The two time quantities backward Euler conflates. `Δt` is what the *internal variable* integrates
    # over: `dₜQ = L(F, Q)` stays first order whatever the global scheme does with `u`, so its local
    # problem is unchanged. The `AffineVelocity` is how the deformation rate is formed and linearized.
    ∂v∂u = γ / (β * Δt)
    _newmark_affine_velocity!(uᵥ, stage_op.ũ, ṽ, ∂v∂u)
    p = NewmarkTimeParameters(nothing, t + Δt, Δt, AffineVelocity(∂v∂u, uᵥ), uₙ₋₁)
    if !nlsolve!(uₙ, f, nlsolver, t + Δt, p)
        # A failed solve leaves no error estimate. Marking it as such is what keeps the controller from
        # reading the *previous* step's estimate -- which is finite and small, and would grow `dt` on a
        # step that just failed.
        cache.EEst = Inf
        return false
    end

    # Correctors
    a = _newmark_acceleration!(stage_op.aₜₘₚ, stage_op, uₙ)
    @inbounds @.. aₙ = a
    @inbounds @.. vₙ = ṽ + γ * Δt * aₙ

    _newmark_error_estimate!(cache, f, Δt, β)
    return true
end

@doc raw"""
    _newmark_error_estimate!(cache, f, Δt, β)

Local error estimate of Zienkiewicz and Xie [ZieXie:1991:sae](@cite),

```math
e_{n+1} = \Delta t^2 \left( \beta - \tfrac{1}{6} \right) \left( a_{n+1} - a_n \right) ,
```

scaled into the usual `EEst ≤ 1` convention against `abstol + reltol·max(|u_{n+1}|, |u_n|)`.

It compares the Newmark update against the third order accurate one obtained with ``\beta = 1/6``, so
the difference of the two accelerations across the step is the whole estimate. Since
``a_{n+1} - a_n = O(\Delta t)``, the estimate is ``O(\Delta t^3)`` — the local error of a second order
scheme, which is what makes the exponent `1/3` in the controller correct.

!!! note "Why no second right hand side evaluation"
    In the first order `(v, u)` formulation the acceleration is not part of the state, so an
    implementation there has to evaluate the right hand side again at the new state to recover it. The
    displacement form does not: the converged Newton *is* the statement `M a_{n+1} = f_ext - f_int(u_{n+1})`,
    so `aₙ` already holds that acceleration to solver tolerance and the estimator is free.

    The corollary is worth recording, because it is a trap: in this form the difference between the
    solved acceleration and a fresh right hand side evaluation at the same state is **zero**, so an
    estimator built from those two quantities measures nothing but the Newton tolerance. The estimate
    has to straddle the step.

`β = 1/6` makes the estimator vanish identically, which is correct — that is the member of the family
the estimate is taken against — but it leaves the scheme with no error estimate, so it is refused.
"""
function _newmark_error_estimate!(cache::NewmarkSolverCache, f, Δt, β)
    (; uₙ, uₙ₋₁, aₙ, aₙ₋₁, stage) = cache
    nfe = length(aₙ)
    # The first step has no previous acceleration to compare against -- `aₙ₋₁` still holds the initial
    # acceleration, which belongs to the same step. `Inf` forces the controller to leave `dt` alone.
    if stage.first_step[]
        stage.first_step[] = false
        cache.EEst = Inf
        return nothing
    end
    scale = stage.reltol
    atol = stage.abstol
    err = zero(eltype(uₙ))
    @inbounds for i = 1:nfe
        eᵢ = Δt^2 * (β - 1 / 6) * (aₙ[i] - aₙ₋₁[i])
        tolᵢ = atol + scale * max(abs(uₙ[i]), abs(uₙ₋₁[i]))
        err += (eᵢ / tolᵢ)^2
    end
    cache.EEst = sqrt(err / nfe)
    return nothing
end


@doc raw"""
A second order scheme carries state that the solution vector does not: the balance of momentum is
`M ü + f_int(u) = f_ext`, so the state of the ODE is the pair `(u, v)` and the scheme additionally
remembers `a`. The integrator owns a rollback buffer for the solution vector alone, so `v` and `a`
need theirs here, kept in lockstep with `uprev`.

Without this, a rejected step restores `u` and leaves `v` and `a` at the values the *rejected* step
produced. The retry then builds its predictors from them and converges to a wrong answer without any
symptom. The path is reachable today -- a failed Newton already increments `nreject` -- and is only
harmless because such a solve subsequently aborts rather than retrying. It stops being harmless the
moment this scheme gets an error estimate and a real controller.
"""
function _newmark_store_previous!(cache::NewmarkSolverCache)
    cache.vₙ₋₁ .= cache.vₙ
    cache.aₙ₋₁ .= cache.aₙ
    return nothing
end

function _newmark_rollback!(integrator, cache::NewmarkSolverCache)
    if length(integrator.uprev) == 0
        error("Cannot roll back integrator. Aborting time integration step at $(integrator.t).")
    end
    integrator.u .= integrator.uprev
    cache.vₙ .= cache.vₙ₋₁
    cache.aₙ .= cache.aₙ₋₁
    return nothing
end

function accept_step!(integrator::ThunderboltTimeIntegrator, cache::NewmarkSolverCache, controller)
    _newmark_store_previous!(cache)
    return store_previous_info!(integrator)
end

# Spelled out against both controller signatures of the generic fallback, which are equally specific
# in the cache slot and would otherwise be ambiguous.
function reject_step!(integrator::ThunderboltTimeIntegrator, cache::NewmarkSolverCache, controller)
    return _newmark_rollback!(integrator, cache)
end

function reject_step!(
    integrator::ThunderboltTimeIntegrator,
    cache::NewmarkSolverCache,
    ::Union{Nothing, DummyControllerCache},
)
    return _newmark_rollback!(integrator, cache)
end


@doc raw"""
    NewmarkErrorController(; γ = 0.9, qmin = 0.2, qmax = 5.0)

Step size control from the local error estimate of [`_newmark_error_estimate!`](@ref).

A step is accepted when the scaled estimate satisfies `EEst ≤ 1`, and the next step is proposed with
the textbook elementary controller

```math
q = \mathrm{clamp}\left( \frac{1}{\gamma}\,\mathrm{EEst}^{1/3},\; q_{\min}^{-1}, q_{\max}^{-1} \right) ,
\qquad \Delta t_{\mathrm{new}} = \Delta t / q .
```

The exponent is `1/(order+1) = 1/3`: Newmark is second order, so its local error is `O(Δt³)`.

`γ < 1` is the usual safety factor, and `qmin`/`qmax` bound how far a single step may move `dt`. The
tolerances themselves live on [`NewmarkSolver`](@ref), because the estimate is scaled where it is
computed.
"""
Base.@kwdef struct NewmarkErrorController{T}
    γ::T = 0.9
    qmin::T = 0.2
    qmax::T = 5.0
end

OrdinaryDiffEqCore.default_controller(QT, ::NewmarkSolver) = NewmarkErrorController{QT}()

OrdinaryDiffEqCore.setup_controller_cache(_alg, cache, controller::NewmarkErrorController, EEstT) =
    controller

# `Inf` on the first step, where the estimator is not yet computable, so it is accepted unconditionally
# and `dt` is left alone.
function should_accept_step(
    integrator::ThunderboltTimeIntegrator,
    cache::NewmarkSolverCache,
    controller::NewmarkErrorController,
)
    return !isfinite(cache.EEst) || cache.EEst ≤ one(cache.EEst)
end

function _newmark_proposed_dt(integrator, cache::NewmarkSolverCache, controller)
    (; γ, qmin, qmax) = controller
    isfinite(cache.EEst) || return integrator.dt
    EEst = max(cache.EEst, eps(typeof(cache.EEst)))
    q = clamp(EEst^(1 / 3) / γ, inv(qmax), inv(qmin))
    return min(integrator.dt / q, integrator.opts.dtmax)
end

function adapt_dt!(
    integrator::ThunderboltTimeIntegrator,
    cache::NewmarkSolverCache,
    controller::NewmarkErrorController,
)
    integrator.dt = _newmark_proposed_dt(integrator, cache, controller)
    return nothing
end

function reject_step!(
    integrator::ThunderboltTimeIntegrator,
    cache::NewmarkSolverCache,
    controller::NewmarkErrorController,
)
    _newmark_rollback!(integrator, cache)
    # A rejected step has `EEst > 1`, so the controller shrinks; a rejected *solve* has no estimate at
    # all and falls back to the integrator's failure factor.
    integrator.dt = if isfinite(cache.EEst)
        _newmark_proposed_dt(integrator, cache, controller)
    else
        integrator.dt / integrator.opts.failfactor
    end
    return nothing
end
