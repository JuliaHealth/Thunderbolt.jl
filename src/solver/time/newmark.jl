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

Error tolerances are `init` keywords (`reltol`, `abstol`) as everywhere else in SciML, not fields
here.

Damping is not supported yet; the model is `M ü + f_int(u) = f_ext`.
"""
Base.@kwdef struct NewmarkSolver{T, SolverType, SystemMatrixType, MonitorType} <: AbstractSolver
    β::T                                       = 1 / 4
    γ::T                                       = 1 / 2
    inner_solver::SolverType                   = MultiLevelNewtonRaphsonSolver()
    system_matrix_type::Type{SystemMatrixType} = ThreadedSparseMatrixCSR{Float64, Int64}
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
mutable struct NewmarkStageOperator{OpType, MassOpType, VectorType, T} <:
               FerriteOperators.AbstractNonlinearOperator
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
Base.eltype(sop::NewmarkStageOperator) = eltype(getJ(sop))
Base.size(sop::NewmarkStageOperator, args...) = size(getJ(sop), args...)

# The inertia is part of the operator, so it is part of its action too. Forwarding to `sop.op` alone
# would return the internal force product and silently drop `M/(βΔt²)`.
function LinearAlgebra.mul!(out::AbstractVector, sop::NewmarkStageOperator, x::AbstractVector)
    mul!(out, sop.op, x)
    mul!(out, sop.M, x, inv(sop.βΔt²), true)
    return out
end

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

struct NewmarkStageCache{StageType, SolverType, StageOpType, T}
    # Newmark condenses the velocity out, but the velocity is not part of the solution vector yet, so
    # the stage's unknowns are still the function's. It becomes a stage of its own once the velocity
    # is a genuine field.
    stage_function::StageType
    nlsolver::SolverType
    stage_op::StageOpType
    β::T
    γ::T
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
    stage::StageType
    # DO NOT USE THIS (will be replaced by proper logging system)
    monitor::MonitorType
end

"""
    velocity(integrator)      -> v at `integrator.t`
    acceleration(integrator)  -> a at `integrator.t`
    velocity(integrator, t)   -> v interpolated to `t`
    acceleration(integrator, t)

The velocity and acceleration the scheme reconstructed. They are not part of the solution vector, so
this is how a consumer reaches them.

**Pass the time whenever the solution is read at a chosen time rather than at a step boundary.**
`TimeChoiceIterator` and `intervals` interpolate `u` to the requested `t` but leave the integrator
sitting at the end of the step that bracketed it, so the no-argument form returns a velocity from a
*different* time than the `u` handed to the loop body:

```julia
for (u, t) in TimeChoiceIterator(integrator, 0.0:dtvis:tend)
    v = velocity(integrator, t)      # matches `u`
    v_wrong = velocity(integrator)   # the end of the bracketing step
end
```

!!! note "Accuracy"
    All three fields are interpolated linearly between step endpoints, matching the interpolation the
    integrator already applies to `u`. That is first order, below the scheme's own second order. A
    Hermite interpolant built from `(u, v)` at both ends would be consistent with the update formulas
    and second order; it is worth having and is not what this does.
"""
velocity(cache::NewmarkSolverCache) = cache.vₙ
acceleration(cache::NewmarkSolverCache) = cache.aₙ
velocity(integrator::ThunderboltTimeIntegrator) = velocity(integrator.cache)
acceleration(integrator::ThunderboltTimeIntegrator) = acceleration(integrator.cache)

# `vₙ₋₁`/`aₙ₋₁` are written by `accept_step!`, which runs in the header of the *following* step, so
# after a completed step they hold the previous step's values and pair with `integrator.tprev`.
velocity(integrator::ThunderboltTimeIntegrator, t) =
    _newmark_hermite(integrator, integrator.cache, t, Val(1))
acceleration(integrator::ThunderboltTimeIntegrator, t) =
    _newmark_hermite(integrator, integrator.cache, t, Val(2))

# The scheme has `u` and `v` at both ends of the step, so the unique cubic matching all four is
# available at no extra assembly, and it is second order rather than the first order a linear
# interpolant gives. Its first and second derivatives are the velocity and the acceleration, which is
# what keeps the three fields mutually consistent -- a linear interpolation of each separately does
# not satisfy `v = dₜu`.
interpolate_solution!(out, integrator::ThunderboltTimeIntegrator, cache::NewmarkSolverCache, t) =
    _newmark_hermite!(out, integrator, cache, t, Val(0))

@doc raw"""
    _newmark_hermite!(out, integrator, cache, t, ::Val{D})

`D`-th time derivative at `t` of the cubic Hermite interpolant through `(uₙ₋₁, vₙ₋₁)` and `(uₙ, vₙ)`.

With ``	heta = (t - t_{n-1})/\Delta t`` the interpolant is
```math
u(	heta) = h_{00}u_{n-1} + \Delta t\, h_{10} v_{n-1} + h_{01} u_n + \Delta t\, h_{11} v_n
```
with the standard Hermite basis. `D = 0` gives the displacement, `D = 1` the velocity, `D = 2` the
acceleration. The velocity is *exact* at both endpoints by construction; the acceleration is the
interpolant's, which is linear in `θ` and therefore only an approximation of the scheme's own.
"""
function _newmark_hermite!(out, integrator::ThunderboltTimeIntegrator, cache, t, ::Val{D}) where {D}
    fe = fe_dof_range(integrator.f)
    iv = internal_variable_range(integrator.f)
    Δt = integrator.t - integrator.tprev
    # Before the first step there is no interval to interpolate over. The current state answers any
    # `t` that can be asked at that point, and a zero-width interval would divide by zero.
    if Δt == zero(Δt)
        _newmark_endpoint!(out, cache, Val(D))
        return out
    end
    θ = (t - integrator.tprev) / Δt

    uprev = @view integrator.uprev[fe]
    u = @view integrator.u[fe]
    vprev, v = cache.vₙ₋₁, cache.vₙ

    c₀, c₁, c₂, c₃ = _hermite_weights(θ, Δt, Val(D))
    @inbounds @views @.. out[fe] = c₀ * uprev + c₁ * vprev + c₂ * u + c₃ * v
    # The condensed internal variables have no derivative here, so they stay linear. `out` is only
    # required to hold the finite element block -- the out-of-place entry points allocate it that
    # long -- so the tail is filled only when the caller supplied room for it.
    if D == 0 && !isempty(iv) && lastindex(out) ≥ last(iv)
        OS.linear_interpolation!(
            @view(out[iv]),
            t,
            @view(integrator.uprev[iv]),
            @view(integrator.u[iv]),
            integrator.tprev,
            integrator.t,
        )
    end
    return out
end

function _newmark_hermite(integrator::ThunderboltTimeIntegrator, cache, t, ::Val{D}) where {D}
    out = similar(cache.vₙ)
    _newmark_hermite!(out, integrator, cache, t, Val(D))
    return out
end

_newmark_endpoint!(out, cache, ::Val{0}) = (out .= 0; nothing)
_newmark_endpoint!(out, cache, ::Val{1}) = (out .= cache.vₙ; nothing)
_newmark_endpoint!(out, cache, ::Val{2}) = (out .= cache.aₙ; nothing)

# Hermite basis and its first two derivatives with respect to `t`, as the four weights of
# `(uprev, vprev, u, v)`.
@inline function _hermite_weights(θ, Δt, ::Val{0})
    θ² = θ * θ
    θ³ = θ² * θ
    return (2θ³ - 3θ² + 1, Δt * (θ³ - 2θ² + θ), -2θ³ + 3θ², Δt * (θ³ - θ²))
end
@inline function _hermite_weights(θ, Δt, ::Val{1})
    θ² = θ * θ
    return ((6θ² - 6θ) / Δt, 3θ² - 4θ + 1, (-6θ² + 6θ) / Δt, 3θ² - 2θ)
end
@inline _hermite_weights(θ, Δt, ::Val{2}) =
    ((12θ - 6) / Δt^2, (6θ - 4) / Δt, (-12θ + 6) / Δt^2, (6θ - 2) / Δt)

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
    stage_function = FullStateStage(
        f,
        stage_op,
        NewmarkTimeParameters(
            nothing,
            t₀,
            zero(t₀),
            AffineVelocity(one(Float64), zeros(solution_size(f))),
            _uprev,
        ),
    )
    nlsolver = _setup_multilevel_newton_cache(stage_function, local_solver_cache, newton, nfe)

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
        NewmarkStageCache(stage_function, nlsolver, stage_op, solver.β, solver.γ, Ref(true)),
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
    fe = fe_dof_range(f)
    r = zeros(length(fe))

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
    @inbounds @views @.. uᵥ[fe] = u₀[fe] - v₀
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

# The scheme reports its error estimate to the *controller* via `set_error_estimate!`,
# so this dispatches on the integrator rather than on `(f, cache, t, Δt)`. Nothing about the step size
# control is stored on the solver cache.
function OrdinaryDiffEqCore.perform_step!(
    integrator::ThunderboltTimeIntegrator,
    cache::NewmarkSolverCache,
)
    if !perform_step!(integrator.f, cache, integrator.t, integrator.dt)
        integrator.force_stepfail = true
        return nothing
    end
    _newmark_report_error!(integrator, cache, integrator.dt, cache.stage.β)
    return nothing
end

function perform_step!(f::ElastodynamicsFunction, cache::NewmarkSolverCache, t, Δt)
    (; uₙ, uₙ₋₁, vₙ, aₙ, ṽ, uᵥ, stage) = cache
    (; stage_function, nlsolver, stage_op, β, γ) = stage
    fe = fe_dof_range(f)

    update_constraints!(f, cache, t + Δt)
    # Predictors, in the same shape as the Ferrite reference implementation.
    @inbounds @views @.. stage_op.ũ = uₙ₋₁[fe] + Δt * vₙ + (1 / 2 - β) * Δt^2 * aₙ
    @inbounds @.. ṽ = vₙ + (1 - γ) * Δt * aₙ
    stage_op.βΔt² = β * Δt^2

    # The two time quantities backward Euler conflates. `Δt` is what the *internal variable* integrates
    # over: `dₜQ = L(F, Q)` stays first order whatever the global scheme does with `u`, so its local
    # problem is unchanged. The `AffineVelocity` is how the deformation rate is formed and linearized.
    ∂v∂u = γ / (β * Δt)
    _newmark_affine_velocity!(uᵥ, stage_op.ũ, ṽ, ∂v∂u)
    set_stage_parameters!(
        stage_function,
        NewmarkTimeParameters(nothing, t + Δt, Δt, AffineVelocity(∂v∂u, uᵥ), uₙ₋₁),
    )
    if !nlsolve!(uₙ, stage_function, nlsolver, t + Δt)
        return false
    end

    # Correctors
    a = _newmark_acceleration!(stage_op.aₜₘₚ, stage_op, uₙ)
    @inbounds @.. aₙ = a
    @inbounds @.. vₙ = ṽ + γ * Δt * aₙ

    return true
end

@doc raw"""
    _newmark_report_error!(integrator, cache, Δt, β)

Local error estimate of Zienkiewicz and Xie [ZieXie:1991:sae](@cite),

```math
e_{n+1} = \Delta t^2 \left( \beta - \tfrac{1}{6} \right) \left( a_{n+1} - a_n \right) ,
```

scaled into the usual `EEst ≤ 1` convention against `abstol + reltol·max(|u_{n+1}|, |u_n|)` and handed
to the controller with [`set_error_estimate!`](@ref).

It compares the Newmark update against the third order accurate one obtained with ``\beta = 1/6``, so
the difference of the two accelerations across the step is the whole estimate. Since
``a_{n+1} - a_n = O(\Delta t)``, the estimate is ``O(\Delta t^3)`` — the local error of a second order
scheme, which is what makes `alg_adaptive_order = 2` correct.

!!! note "Why no second right hand side evaluation"
    In the first order `(v, u)` formulation the acceleration is not part of the state, so an
    implementation there has to evaluate the right hand side again at the new state to recover it. The
    displacement form does not: the converged Newton *is* the statement
    `M a_{n+1} = f_ext - f_int(u_{n+1})`, so `aₙ` already holds that acceleration to solver tolerance
    and the estimator is free.

    An estimator formed from the solved acceleration and a fresh right hand side evaluation at the
    same state is identically zero, so the estimate has to straddle the step.

`β = 1/6` makes the estimator vanish identically, which is correct — that is the member of the family
the estimate is taken against — but it leaves the scheme with no error estimate.
"""
function _newmark_report_error!(integrator, cache::NewmarkSolverCache, Δt, β)
    controller_cache = integrator.controller_cache
    controller_cache === nothing && return nothing # fixed step size, nothing to report to

    (; uₙ, uₙ₋₁, aₙ, aₙ₋₁, stage) = cache
    # The first step has no previous acceleration to compare against -- `aₙ₋₁` still holds the initial
    # acceleration, which belongs to the same step. An estimate of zero leaves `dt` to the controller's
    # own first-step growth bound.
    if stage.first_step[]
        stage.first_step[] = false
        set_error_estimate!(controller_cache, zero(eltype(uₙ)))
        return nothing
    end

    reltol = integrator.opts.reltol
    abstol = integrator.opts.abstol
    fe = fe_dof_range(integrator.f)
    err = zero(eltype(uₙ))
    # The acceleration carries the scheme's own numbering while `uₙ` carries the solution vector's.
    # They coincide today; walking them as a pair rather than with one shared index keeps the estimate
    # correct once a stage solves against a handler of its own.
    @inbounds for (k, i) in enumerate(fe)
        eᵢ = Δt^2 * (β - 1 / 6) * (aₙ[k] - aₙ₋₁[k])
        tolᵢ = abstol + reltol * max(abs(uₙ[i]), abs(uₙ₋₁[i]))
        err += (eᵢ / tolᵢ)^2
    end
    set_error_estimate!(controller_cache, sqrt(err / length(fe)))
    return nothing
end

function _newmark_store_previous!(cache::NewmarkSolverCache)
    cache.vₙ₋₁ .= cache.vₙ
    cache.aₙ₋₁ .= cache.aₙ
    return nothing
end

function accept_step!(integrator::ThunderboltTimeIntegrator, cache::NewmarkSolverCache, controller)
    _newmark_store_previous!(cache)
    return store_previous_info!(integrator)
end

"""
Step size control for [`NewmarkSolver`](@ref) uses Thunderbolt's own [`PIDController`](@ref).

The only thing the scheme owes it is the scaled error estimate, reported with
[`set_error_estimate!`](@ref) at the end of a step. Everything derived from it -- the proposed step
size, whether the step is accepted, the error history -- lives on the controller cache.
"""
adaptive_order(::NewmarkSolver) = 2

# Söderlind's coefficients. `PIDController` scales them by the order, so they are stated once here
# rather than per scheme.
OrdinaryDiffEqCore.default_controller(QT, ::NewmarkSolver) =
    PIDController(QT(3 // 5), QT(-1 // 5), QT(0))

# The velocity and the acceleration are state of the same second order ODE as the displacement but
# are not in the solution vector, so the generic rollback does not cover them.
function rollback_state!(integrator::ThunderboltTimeIntegrator, cache::NewmarkSolverCache)
    @invoke rollback_state!(integrator, cache::Any)
    cache.vₙ .= cache.vₙ₋₁
    cache.aₙ .= cache.aₙ₋₁
    return nothing
end
