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

"""
    NewmarkStage(f, op, mapping, velocity_dofs, p)

The nonlinear problem one Newmark step poses.

Newmark condenses the velocity out: the step solves for the displacement and the condensed internal
variables against the *structural* problem's dof handler, and the velocity follows from the converged
displacement. So the stage's unknowns are a strict subset of the state's, wired to it by a
[`SolutionVectorMapping`](@ref) rather than by an assumed layout.

`update_state!` reconstructs the velocity from the same [`AffineVelocity`](@ref) the element is handed
to form the deformation rate, so the global corrector and the element's rate cannot drift apart.
"""
mutable struct NewmarkStage{FType, OpType, MapType, PType} <: AbstractStageFunction
    # The structural sub-problem: what the operator assembles against, and whose constraints and
    # residual norm the Newton uses.
    const f::FType
    const op::OpType
    const mapping::MapType
    # `velocity_dofs[i]` is the state dof carrying the velocity at structural dof `i`.
    const velocity_dofs::Vector{Int}
    p::PType
end

getoperator(sf::NewmarkStage) = sf.op
getfunction(sf::NewmarkStage) = sf.f
stage_mapping(sf::NewmarkStage) = sf.mapping
stage_parameters(sf::NewmarkStage) = sf.p
set_stage_parameters!(sf::NewmarkStage, p) = (sf.p = p; sf)

function update_state!(u::AbstractVector, sf::NewmarkStage, z::AbstractVector)
    scatter!(u, z, stage_mapping(sf))
    # v = ∂v∂u (u - uᵥ), in the structural numbering, written into the state's velocity block. This is
    # the same relation the element uses to form the deformation rate -- one statement, not two.
    (; ∂v∂u, uᵥ) = sf.p.velocity
    @inbounds for (i, d) in enumerate(sf.velocity_dofs)
        u[d] = ∂v∂u * (z[i] - uᵥ[i])
    end
    return u
end

struct NewmarkStageCache{StageType, SolverType, T}
    # The nonlinear problem one Newmark step poses: solve for the displacement and the condensed
    # internal variables, then reconstruct the velocity. The operator lives on it -- one owner, as for
    # `NewtonRaphsonSolverCache`.
    stage_function::StageType
    nlsolver::SolverType
    # The stage unknowns and the previous state gathered into the stage numbering.
    z::Vector{Float64}
    zprev::Vector{Float64}
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
    VelocityViewType <: AbstractVector{T},
    AccelerationType <: AbstractVector{T},
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
    # The velocity block of the solution vector. A view rather than a buffer: the velocity is state of
    # this second order system, so the integrator's own `u .= uprev` rollback covers it and no bespoke
    # restore is needed. The previous step's velocity is read straight out of `uprev` where it is
    # wanted, so it needs no field here.
    vₙ::VelocityViewType
    # The acceleration is *not* state -- it is determined by `(u, v)` through the balance of momentum
    # -- so it stays scheme workspace, kept only to avoid re-solving `M a = f_ext - f_int` each step.
    #
    # Caching it is what forces the rollback buffer. A step whose *solve* fails leaves `aₙ` untouched,
    # because `perform_step!` returns before the corrector. A step the error controller rejects does
    # not: the solve succeeded, `aₙ` already holds the rejected attempt's value, and the retry needs
    # the last accepted one to form its predictors. `aₙ₋₁` is that value.
    aₙ::AccelerationType
    aₙ₋₁::AccelerationType
    # Scratch for the velocity predictor
    ṽ::AccelerationType
    # The `uᵥ` of this step's `AffineVelocity`, held at the structural problem's length so that the
    # element query can slice a cell out of it exactly as it does for the previous solution.
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

The velocity and acceleration the scheme reconstructed.

The velocity is a field of the solution vector and could also be read out of it directly; these
accessors exist because the *interpolated* velocity at an interior `t` is not something a consumer can
assemble by hand. The acceleration has no block of its own -- it is determined by the state rather
than part of it -- so this is the only way to reach it.

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
    The displacement and the velocity come from the cubic Hermite interpolant through `(u, v)` at both
    step ends, which is second order and consistent with the update formulas -- in particular the
    velocity returned is the derivative of the displacement returned. The acceleration is that
    interpolant's second derivative, which is linear in the step and therefore only an approximation
    of the scheme's own. The condensed internal variables stay linear; they have no derivative here.
"""
velocity(cache::NewmarkSolverCache) = cache.vₙ
acceleration(cache::NewmarkSolverCache) = cache.aₙ
velocity(integrator::ThunderboltTimeIntegrator) = velocity(integrator.cache)
acceleration(integrator::ThunderboltTimeIntegrator) = acceleration(integrator.cache)

# `aₙ₋₁` is written by `accept_step!`, which runs in the header of the *following* step, so after a
# completed step it holds the previous step's value and pairs with `integrator.tprev`. The previous
# velocity needs no such care: it is the velocity block of `integrator.uprev`.
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

`D`-th time derivative at `t` of the cubic Hermite interpolant through the displacement and
velocity blocks of `uprev` and `u`.

With ``\theta = (t - t_{n-1})/\Delta t`` the interpolant is
```math
u(\theta) = h_{00}u_{n-1} + \Delta t\, h_{10} v_{n-1} + h_{01} u_n + \Delta t\, h_{11} v_n
```
with the standard Hermite basis. `D = 0` gives the displacement, `D = 1` the velocity, `D = 2` the
acceleration. The velocity is *exact* at both endpoints by construction; the acceleration is the
interpolant's, which is linear in `θ` and therefore only an approximation of the scheme's own.
"""
function _newmark_hermite!(out, integrator::ThunderboltTimeIntegrator, cache, t, ::Val{D}) where {D}
    f = integrator.f
    udofs = f.state_mapping.dofs
    vdofs = f.velocity_dofs
    iv = internal_variable_range(f)
    Δt = integrator.t - integrator.tprev
    # Before the first step there is no interval to interpolate over. The current state answers any
    # `t` that can be asked at that point, and a zero-width interval would divide by zero.
    if Δt == zero(Δt)
        _newmark_endpoint!(out, integrator, cache, Val(D))
        return out
    end
    θ = (t - integrator.tprev) / Δt

    uprev, u = integrator.uprev, integrator.u
    c₀, c₁, c₂, c₃ = _hermite_weights(θ, Δt, Val(D))
    # The displacement block and the velocity block of the *same* interpolant: the velocity written
    # out is the derivative of the displacement written out, so a consumer reading both at one `t`
    # sees a consistent pair rather than two independently interpolated fields.
    d₀, d₁, d₂, d₃ = _hermite_weights(θ, Δt, Val(D + 1))
    @inbounds for i ∈ eachindex(udofs)
        du, dv = udofs[i], vdofs[i]
        out[du] = c₀ * uprev[du] + c₁ * uprev[dv] + c₂ * u[du] + c₃ * u[dv]
        out[dv] = d₀ * uprev[du] + d₁ * uprev[dv] + d₂ * u[du] + d₃ * u[dv]
    end
    # The condensed internal variables have no derivative here, so they stay linear.
    if D == 0 && !isempty(iv)
        OS.linear_interpolation!(
            @view(out[iv]),
            t,
            @view(uprev[iv]),
            @view(u[iv]),
            integrator.tprev,
            integrator.t,
        )
    end
    return out
end

function _newmark_hermite(integrator::ThunderboltTimeIntegrator, cache, t, ::Val{D}) where {D}
    out = zeros(eltype(integrator.u), length(integrator.u))
    _newmark_hermite!(out, integrator, cache, t, Val(D))
    # The `D`-th derivative lives in the *displacement* block whatever `D` is -- the velocity block
    # carries the next one, so that a state vector filled at `D = 0` is internally consistent. Reading
    # it here would return the derivative one order too high.
    return out[displacement_dofs(integrator.f)]
end

function _newmark_endpoint!(out, integrator, cache, ::Val{D}) where {D}
    f = integrator.f
    if D == 0
        copyto!(out, integrator.u)
    else
        fill!(out, zero(eltype(out)))
        src = D == 1 ? cache.vₙ : cache.aₙ
        @inbounds for (i, d) ∈ enumerate(f.velocity_dofs)
            out[d] = src[i]
        end
    end
    return nothing
end

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
# The cubic's third derivative is constant, and is what the velocity block of an acceleration query
# carries.
@inline _hermite_weights(θ, Δt, ::Val{3}) = (12 / Δt^3, 6 / Δt^2, -12 / Δt^3, 6 / Δt^2)

function setup_solver_cache(
    f::ElastodynamicsFunction,
    solver::NewmarkSolver,
    t₀;
    uprev       = nothing,
    u           = nothing,
    alias_uprev = true,
    alias_u     = false,
)
    vtype = Vector{Float64}
    # The stage assembles against the structural problem, so its sizes -- not the state's -- are what
    # the operator, the residual and the acceleration are built from.
    structural = f.structural
    nfe = ndofs(structural.dh)

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

    (; integrator, dh) = structural
    (; newton, local_solver) = solver.inner_solver

    local_solver_cache = _setup_local_solver_cache(local_solver, integrator, dh, structural.lvh)
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
    stage_function = NewmarkStage(
        structural,
        stage_op,
        f.state_mapping,
        f.velocity_dofs,
        NewmarkTimeParameters(
            nothing,
            t₀,
            zero(t₀),
            AffineVelocity(one(Float64), zeros(solution_size(structural))),
            zeros(solution_size(structural)),
        ),
    )
    nlsolver = _setup_multilevel_newton_cache(stage_function, local_solver_cache, newton, nfe)

    vₙ = view(_u, f.velocity_dofs)
    aₙ = _consistent_initial_acceleration(f, stage_op, _u, vₙ, t₀)

    return NewmarkSolverCache(
        _u,
        _uprev,
        copy(_u),
        vₙ,
        aₙ,
        copy(aₙ),
        zeros(nfe),
        zeros(solution_size(structural)),
        NewmarkStageCache(
            stage_function,
            nlsolver,
            zeros(solution_size(structural)),
            zeros(solution_size(structural)),
            solver.β,
            solver.γ,
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
    # Everything here is in the *structural* numbering: that is what the operator assembles against
    # and what the mass matrix is sized by.
    structural = f.structural
    z = zeros(solution_size(structural))
    gather!(z, u₀, f.state_mapping)
    fe = fe_dof_range(structural)
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
    uᵥ = copy(z)
    @inbounds @views @.. uᵥ[fe] = z[fe] - v₀
    p = NewmarkTimeParameters(
        nothing,
        t₀,
        eps(Float64),
        AffineVelocity(one(eltype(z)), uᵥ),
        copy(z),
    )
    residual!(stage_op.op, r, copy(z), p)
    r .= .-r

    # On a copy of the mass matrix: `apply_zero!` rewrites the constrained rows and columns, and the
    # stage operator keeps using `M` for every step afterwards.
    M = copy(SparseMatrixCSC(stage_op.M.A))
    apply_zero!(M, r, getch(structural))
    a₀ = M \ r
    apply_zero!(a₀, getch(structural))
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
    (; stage_function, nlsolver, β, γ) = stage
    stage_op = getoperator(stage_function)
    # The predictors and the stage unknowns live in the structural numbering.
    z = stage.z
    fe = fe_dof_range(f.structural)

    update_constraints!(f, cache, t + Δt)
    init_stage!(z, stage_function, uₙ)
    zprev = stage.zprev
    init_stage!(zprev, stage_function, uₙ₋₁)

    # Predictors, in the same shape as the Ferrite reference implementation.
    @inbounds @views @.. stage_op.ũ = zprev[fe] + Δt * vₙ + (1 / 2 - β) * Δt^2 * aₙ
    @inbounds @.. ṽ = vₙ + (1 - γ) * Δt * aₙ
    stage_op.βΔt² = β * Δt^2

    # The two time quantities backward Euler conflates. `Δt` is what the *internal variable* integrates
    # over: `dₜQ = L(F, Q)` stays first order whatever the global scheme does with `u`, so its local
    # problem is unchanged. The `AffineVelocity` is how the deformation rate is formed and linearized.
    ∂v∂u = γ / (β * Δt)
    _newmark_affine_velocity!(uᵥ, stage_op.ũ, ṽ, ∂v∂u)
    set_stage_parameters!(
        stage_function,
        NewmarkTimeParameters(nothing, t + Δt, Δt, AffineVelocity(∂v∂u, uᵥ), zprev),
    )
    if !nlsolve!(z, stage_function, nlsolver, t + Δt)
        return false
    end

    # The acceleration is scheme workspace, so it is corrected here; the velocity is state and is
    # reconstructed by `update_state!` as the stage writes itself back.
    a = _newmark_acceleration!(stage_op.aₜₘₚ, stage_op, z)
    @inbounds @.. aₙ = a
    update_state!(uₙ, stage_function, z)

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
    udofs = integrator.f.state_mapping.dofs
    err = zero(eltype(uₙ))
    # `aₙ` is in the structural numbering, `uₙ` in the state's: `udofs[k]` is the state dof of
    # structural dof `k`.
    @inbounds for (k, i) in enumerate(udofs)
        eᵢ = Δt^2 * (β - 1 / 6) * (aₙ[k] - aₙ₋₁[k])
        tolᵢ = abstol + reltol * max(abs(uₙ[i]), abs(uₙ₋₁[i]))
        err += (eᵢ / tolᵢ)^2
    end
    set_error_estimate!(controller_cache, sqrt(err / length(udofs)))
    return nothing
end

# Only the acceleration: the velocity is part of the solution vector, so the integrator's own
# `uprev` bookkeeping carries it.
function _newmark_store_previous!(cache::NewmarkSolverCache)
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

# The velocity is state and rides along in the solution vector, so the generic rollback restores it.
# The acceleration is not state -- it is determined by `(u, v)` -- and is kept only as workspace, so it
# is the one quantity that still needs a buffer of its own.
function rollback_state!(integrator::ThunderboltTimeIntegrator, cache::NewmarkSolverCache)
    @invoke rollback_state!(integrator, cache::Any)
    cache.aₙ .= cache.aₙ₋₁
    return nothing
end
