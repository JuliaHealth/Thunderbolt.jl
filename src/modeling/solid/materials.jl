# TODO (FILE) I think we should change the design here. Instea of dispatching on Ψ we should make the material callable or equip it with a function.

abstract type AbstractMaterialModel end

default_initial_state!(uq, ::AbstractMaterialModel) =
    error("Initial condition setup not implemented yet.")

"""
    RateDependence

Whether a material's stress depends on the deformation *rate* `Ḟ` in addition to `F`.

A [`RateDependent`](@ref) material implements the five-argument
`stress_and_tangent(model, F, Ḟ, coefficients, state)`, returning `(P, ∂P∂F, ∂P∂Ḟ)`. The element
assembles

    dP/du = ∂P/∂F ⋅ ∂F/∂u + ∂P/∂Ḟ ⋅ ∂Ḟ/∂u

where `∂Ḟ/∂u` comes from the *time scheme* (`1/Δt` for backward Euler, `γ/(βΔt)` for Newmark), so a
material never learns how the rate was formed. A material must not difference `F` against a previous
value itself: that bakes one time discretization into the constitutive law.

A [`RateIndependent`](@ref) material implements only the four-argument form; the uniform entry point
answers for it with a zero rate tangent.
"""
abstract type RateDependence end

"""
    RateIndependent <: RateDependence

`P = P(F, Q)`. The default for every material.
"""
struct RateIndependent <: RateDependence end

"""
    RateDependent <: RateDependence

`P = P(F, Ḟ, Q)`. Implements the five-argument `stress_and_tangent`.
"""
struct RateDependent <: RateDependence end

"""
    rate_dependence(model::AbstractMaterialModel)

The [`RateDependence`](@ref) of `model`. Defaults to [`RateIndependent`](@ref); override for a
material whose stress reads `Ḟ`.
"""
rate_dependence(::AbstractMaterialModel) = RateIndependent()

"""
    AbstractKinematics

What the time scheme offers a material at a quadrature point. The scheme builds it -- backward Euler
from `(F - Fprev)/Δt`, Newmark from `γ/(βΔt)`, an energy-momentum scheme from two configurations --
so a material never learns how the rate was formed.

This is the argument `material_routine` takes in place of a bare `F`. Below that seam the material
functions (`stress_and_tangent`, `stress_function`, `Ψ`) keep taking bare tensors, so the automatic
differentiation closures still capture leaves rather than a container.

Offering more than a material reads is fine: a [`RateIndependent`](@ref) material accepts
[`DeformationGradientWithRate`](@ref) and ignores the rate. Offering less is a `MethodError`, which is the
point of having two types rather than one with an optional field.
"""
abstract type AbstractKinematics end

"""
    DeformationGradient(F)

Deformation gradient only. What a rate-free scheme offers.
"""
struct DeformationGradient{TF <: Tensor{2}} <: AbstractKinematics
    F::TF
end

"""
    DeformationGradientWithRate(F, Ḟ)

Deformation gradient and its rate. What a first-order-in-time scheme offers.
"""
struct DeformationGradientWithRate{TF <: Tensor{2}, TḞ <: Tensor{2}} <: AbstractKinematics
    F::TF
    Ḟ::TḞ
end

@inline deformation_gradient(kinematics::AbstractKinematics) = kinematics.F

@inline deformation_rate(kinematics::DeformationGradientWithRate) = kinematics.Ḟ
@inline deformation_rate(::DeformationGradient) = error(
    "A rate dependent material was assembled by a time scheme that offers no deformation rate. " *
    "Either wrap its internal model in `AsRateIndependent`, or use a scheme that supplies `Ḟ`.",
)

"""
    AbstractKinematicSensitivities

What `material_routine` returns alongside the stress: one sensitivity per kinematic quantity the
scheme offered. The type is the conjugate of the [`AbstractKinematics`](@ref) that went in, so
`material_routine` answers in the same currency it was asked in and the element cannot silently
receive a sensitivity it has no `∂·/∂u` for.

The element turns these into the single tangent modulus its assembly loop contracts with
[`consistent_tangent`](@ref).
"""
abstract type AbstractKinematicSensitivities end

"""
    KinematicSensitivities(∂P∂F)

Conjugate to [`DeformationGradient`](@ref).
"""
struct KinematicSensitivities{T∂P∂F} <: AbstractKinematicSensitivities
    ∂P∂F::T∂P∂F
end

"""
    KinematicSensitivitiesWithRate(∂P∂F, ∂P∂Ḟ)

Conjugate to [`DeformationGradientWithRate`](@ref). A material that does not read the rate still
answers in this currency, with a zero `∂P∂Ḟ`.
"""
struct KinematicSensitivitiesWithRate{T∂P∂F, T∂P∂Ḟ} <: AbstractKinematicSensitivities
    ∂P∂F::T∂P∂F
    ∂P∂Ḟ::T∂P∂Ḟ
end

@doc raw"""
    KinematicLinearization(∂F∂u, ∂Ḟ∂u)
    KinematicLinearization(∂F∂u)

How a variation of the quantity the *global solver* iterates on reaches each kinematic slot. One
factor per slot, supplied by the time scheme:

| scheme | unknown | `∂F∂u` | `∂Ḟ∂u` |
| :----- | :------ | :----- | :----- |
| backward Euler | ``u_{n+1}`` | ``1`` | ``1/\Delta t`` |
| Newmark | ``u_{n+1}`` | ``1`` | ``\gamma/(\beta \Delta t)`` |
| BDF-``k`` | ``u_{n+1}`` | ``1`` | ``\alpha_0/\Delta t`` |
| SDIRK stage ``i``, rate form | ``k_i`` | ``\Delta t\, a_{ii}`` | ``1`` |

The SDIRK row is why both factors are carried rather than just the rate one: solving for the stage
*derivative* moves the timestep onto `∂F∂u` and leaves `∂Ḟ∂u` at unity. A scheme that offers no rate
uses the single-argument form.
"""
struct KinematicLinearization{T∂F∂u, T∂Ḟ∂u}
    ∂F∂u::T∂F∂u
    ∂Ḟ∂u::T∂Ḟ∂u
end

KinematicLinearization(∂F∂u) = KinematicLinearization(∂F∂u, nothing)

@doc raw"""
    consistent_tangent(sensitivities, linearization)

Fold the material's sensitivities into the tangent modulus the element's assembly loop contracts,
```math
\mathrm{d}P/\mathrm{d}u_j = \left(\frac{\partial P}{\partial F}\frac{\partial F}{\partial u} + \frac{\partial P}{\partial \dot{F}}\frac{\partial \dot{F}}{\partial u}\right) : \nabla \delta u_j
```
Both factors come from [`KinematicLinearization`](@ref), so the material never learns how the rate
was formed. They are scalars, so this collapses to one tensor add per quadrature point, outside the
test function loops.

Folding here is right for a single-stage scheme, which needs one matrix. A fully implicit
Runge-Kutta scheme must **not** fold: it recombines the same two contributions with ``s^2`` different
weights ``\Delta t\, a_{ij}``, so it consumes the [`AbstractKinematicSensitivities`](@ref) directly
and assembles the two parts separately. That is why `material_routine` returns them unfolded.
"""
@inline consistent_tangent(
    sensitivities::KinematicSensitivities,
    linearization::KinematicLinearization,
) = sensitivities.∂P∂F * linearization.∂F∂u

@inline consistent_tangent(
    sensitivities::KinematicSensitivitiesWithRate,
    linearization::KinematicLinearization,
) = sensitivities.∂P∂F * linearization.∂F∂u + sensitivities.∂P∂Ḟ * linearization.∂Ḟ∂u

# A rate-free scheme has no `∂Ḟ/∂u` to offer, so it may only consume rate-free sensitivities. The
# missing `KinematicSensitivitiesWithRate` method is the counterpart of `deformation_rate` erroring on
# a `DeformationGradient`: neither direction of the seam degrades silently.
@inline consistent_tangent(sensitivities::KinematicSensitivities) = sensitivities.∂P∂F

@doc raw"""
    RateTypeCondensationMaterialStateCache

Every condensation cache whose local problem carries a time derivative, i.e. `dₜQ = L(F, Q)` or
`dₜQ = L(F, dₜF, Q)`. Both need a timestep and a known state, which is what separates them from
the rate-free caches used by continuation solvers.

## The `(Qknownflat, Δt)` contract

Every local solve below poses the same problem,
```math
(Q - Q_{\textrm{known}}) / \Delta t = L(F, Q)
```
and **both arguments are effective quantities the caller normalizes**. `Qknownflat` is deliberately
not called `Qprev`: for anything past backward Euler it is a linear combination of history, not a
previous value.

| scheme | ``Q_{\textrm{known}}`` | ``\Delta t`` |
| :----- | :--------------------- | :----------- |
| backward Euler | ``Q_n`` | ``\Delta t`` |
| BDF-``k`` | ``-\frac{1}{\alpha_0}\sum_{j\geq 1}\alpha_j Q_{n+1-j}`` | ``\Delta t/\alpha_0`` |
| DIRK stage ``i`` | ``Q_n + \Delta t \sum_{j<i} a_{ij} L_j`` | ``a_{ii}\Delta t`` |
| Newmark | ``Q_n`` | ``\Delta t`` |

So one local solver serves all four: the scheme's order and stage structure live entirely in how the
caller computes these two values, exactly as the kinematics seam keeps the rate's provenance out of
the material. A fully implicit Runge-Kutta scheme is the exception — its stages are genuinely
coupled, so it needs a local solver over all ``s`` stages at once rather than this one.
"""
const RateTypeCondensationMaterialStateCache = Union{
    RateIndependentCondensationMaterialStateCache,
    RateDependentCondensationMaterialStateCache,
}

# Uniform five-argument entry point. A rate-independent material answers through its existing
# four-argument method with a zero rate tangent, so the element assembles one expression either way.
@inline stress_and_tangent(
    model::AbstractMaterialModel,
    F::Tensor{2},
    Ḟ::Tensor{2},
    coefficients,
    state,
) = _stress_and_tangent_rate(rate_dependence(model), model, F, Ḟ, coefficients, state)

@inline function _stress_and_tangent_rate(
    ::RateIndependent,
    model,
    F::Tensor{2},
    Ḟ::Tensor{2},
    coefficients,
    state,
)
    P, ∂P∂F = stress_and_tangent(model, F, coefficients, state)
    return P, ∂P∂F, zero(∂P∂F)
end

@inline _stress_and_tangent_rate(
    ::RateDependent,
    model,
    F::Tensor{2},
    Ḟ::Tensor{2},
    coefficients,
    state,
) = error(
    "$(typeof(model)) declares `rate_dependence(...) = RateDependent()` but does not implement " *
    "`stress_and_tangent(model, F, Ḟ, coefficients, state) -> (P, ∂P∂F, ∂P∂Ḟ)`.",
)

# The kinematics seam. A material that does not read the deformation rate is served by unpacking `F`
# and calling its existing method, so introducing kinematics costs the rate-independent path nothing.
# A rate dependent material overrides these on its own cache type and reaches for `deformation_rate`.
@inline function material_routine(
    material_model::AbstractMaterialModel,
    kinematics::DeformationGradient,
    coefficient_cache,
    state_cache,
    geometry_cache::Ferrite.CellCache,
    qp::QuadraturePoint,
    time,
)
    P, ∂P∂F = material_routine(
        material_model,
        deformation_gradient(kinematics),
        coefficient_cache,
        state_cache,
        geometry_cache,
        qp,
        time,
    )
    return P, KinematicSensitivities(∂P∂F)
end

@inline function material_routine(
    material_model::AbstractMaterialModel,
    kinematics::DeformationGradientWithRate,
    coefficient_cache,
    state_cache,
    geometry_cache::Ferrite.CellCache,
    qp::QuadraturePoint,
    time,
)
    P, ∂P∂F = material_routine(
        material_model,
        deformation_gradient(kinematics),
        coefficient_cache,
        state_cache,
        geometry_cache,
        qp,
        time,
    )
    return P, KinematicSensitivitiesWithRate(∂P∂F, zero(∂P∂F))
end

@inline function material_routine(
    material_model::AbstractMaterialModel,
    kinematics::DeformationGradient,
    coefficient_cache,
    state_cache,
    geometry_cache::Ferrite.CellCache,
    qp::QuadraturePoint,
    time,
    Qflat,
    Qknownflat,
    Δt,
)
    P, ∂P∂F = material_routine(
        material_model,
        deformation_gradient(kinematics),
        coefficient_cache,
        state_cache,
        geometry_cache,
        qp,
        time,
        Qflat,
        Qknownflat,
        Δt,
    )
    return P, KinematicSensitivities(∂P∂F)
end

# A material that does not read the rate answers the rate-carrying question with a zero rate
# sensitivity — offering more than a material reads must stay free.
@inline function material_routine(
    material_model::AbstractMaterialModel,
    kinematics::DeformationGradientWithRate,
    coefficient_cache,
    state_cache,
    geometry_cache::Ferrite.CellCache,
    qp::QuadraturePoint,
    time,
    Qflat,
    Qknownflat,
    Δt,
)
    P, ∂P∂F = material_routine(
        material_model,
        deformation_gradient(kinematics),
        coefficient_cache,
        state_cache,
        geometry_cache,
        qp,
        time,
        Qflat,
        Qknownflat,
        Δt,
    )
    return P, KinematicSensitivitiesWithRate(∂P∂F, zero(∂P∂F))
end

@inline reduced_material_routine(
    material_model::AbstractMaterialModel,
    kinematics::AbstractKinematics,
    coefficient_cache,
    state_cache,
    geometry_cache::Ferrite.CellCache,
    qp::QuadraturePoint,
    time,
) = reduced_material_routine(
    material_model,
    deformation_gradient(kinematics),
    coefficient_cache,
    state_cache,
    geometry_cache,
    qp,
    time,
)

@inline reduced_material_routine(
    material_model::AbstractMaterialModel,
    kinematics::AbstractKinematics,
    coefficient_cache,
    state_cache,
    geometry_cache::Ferrite.CellCache,
    qp::QuadraturePoint,
    time,
    Qflat,
    Qknownflat,
    Δt,
) = reduced_material_routine(
    material_model,
    deformation_gradient(kinematics),
    coefficient_cache,
    state_cache,
    geometry_cache,
    qp,
    time,
    Qflat,
    Qknownflat,
    Δt,
)

function material_routine(
    material_model::AbstractMaterialModel,
    F::Tensor{2},
    coefficient_cache,
    ::EmptyInternalCache,
    geometry_cache::Ferrite.CellCache,
    qp::QuadraturePoint,
    time,
)
    coefficients = evaluate_coefficient(coefficient_cache, geometry_cache, qp, time)
    return stress_and_tangent(material_model, F, coefficients, EmptyInternalModel())
end

function material_routine(
    material_model::AbstractMaterialModel,
    F::Tensor{2},
    coefficient_cache,
    state_cache::TrivialCondensationMaterialStateCache,
    geometry_cache::Ferrite.CellCache,
    qp::QuadraturePoint,
    time,
)
    coefficients = evaluate_coefficient(coefficient_cache, geometry_cache, qp, time)
    Q = state(state_cache, geometry_cache, qp, time)
    return stress_and_tangent(material_model, F, coefficients, Q)
end

# `gto1` form: the caller supplies the internal variable and the timestep, so no time data is read
# from the cache. This is the form the element assembly uses. The shim below derives the same data
# from the cache and disappears once the backward Euler stage wrapper is retired.
function material_routine(
    material_model::AbstractMaterialModel,
    F::Tensor{2},
    coefficient_cache,
    state_cache::RateTypeCondensationMaterialStateCache,
    geometry_cache::Ferrite.CellCache,
    qp::QuadraturePoint,
    time,
    Qflat,
    Qknownflat,
    Δt,
)
    coefficients = evaluate_coefficient(coefficient_cache, geometry_cache, qp, time)
    Q, ∂P∂QdQdF = solve_local_constraint(
        F,
        coefficients,
        material_model,
        state_cache,
        geometry_cache,
        qp,
        time,
        Qflat,
        Qknownflat,
        Δt,
    )
    P, ∂P∂F = stress_and_tangent(material_model, F, coefficients, Q)
    return P, ∂P∂F + ∂P∂QdQdF
end

# Rate dependent condensation. The stress itself has no explicit rate dependence — `P = P(F, Q)` — so
# `∂P/∂Ḟ` is entirely mediated by the internal variable, `∂P/∂Q · ∂Q/∂Ḟ`, and comes out of the local
# solve rather than out of `stress_and_tangent`.
function material_routine(
    material_model::AbstractMaterialModel,
    kinematics::DeformationGradientWithRate,
    coefficient_cache,
    state_cache::RateDependentCondensationMaterialStateCache,
    geometry_cache::Ferrite.CellCache,
    qp::QuadraturePoint,
    time,
    Qflat,
    Qknownflat,
    Δt,
)
    F = deformation_gradient(kinematics)
    coefficients = evaluate_coefficient(coefficient_cache, geometry_cache, qp, time)
    Q, ∂P∂QdQdF, ∂P∂QdQdḞ = solve_local_constraint(
        F,
        deformation_rate(kinematics),
        coefficients,
        material_model,
        state_cache,
        geometry_cache,
        qp,
        time,
        Qflat,
        Qknownflat,
        Δt,
    )
    P, ∂P∂F = stress_and_tangent(material_model, F, coefficients, Q)
    return P, KinematicSensitivitiesWithRate(∂P∂F + ∂P∂QdQdF, ∂P∂QdQdḞ)
end

# A rate dependent material cannot be served rate-free kinematics. `deformation_rate` is what says so;
# this method exists only so the message arrives from the call the element actually made.
material_routine(
    material_model::AbstractMaterialModel,
    kinematics::DeformationGradient,
    coefficient_cache,
    state_cache::RateDependentCondensationMaterialStateCache,
    geometry_cache::Ferrite.CellCache,
    qp::QuadraturePoint,
    time,
    Qflat,
    Qknownflat,
    Δt,
) = deformation_rate(kinematics)

# A condensation material reached the bare-`time` assembly path, which is used by
# `HomotopyPathSolver` — a load-stepping continuation with no timestep and no previous solution. Such
# a material must therefore be *rate free*: its local problem has to be the algebraic constraint
# `L(F, Q) = 0` rather than a time-discretized `(Q - Qprev)/Δt = L(F, Q)`. No rate-free condensation
# material exists yet, so the local solver for it is deliberately not written; implement a
# `material_routine` for that cache type taking the current `Q` only.
material_routine(
    material_model::AbstractMaterialModel,
    F::Tensor{2},
    coefficient_cache,
    state_cache::RateTypeCondensationMaterialStateCache,
    geometry_cache::Ferrite.CellCache,
    qp::QuadraturePoint,
    time,
) = error(
    "$(typeof(material_model)) carries a rate-type internal variable, so it needs a timestep and a " *
    "previous state. It cannot be assembled on the rate-free path used by e.g. HomotopyPathSolver.",
)

# Materials without condensed state ignore the `gto1` payload. Deliberately restricted to the two
# stateless cache types: any *other* cache reaching this arity without its own method is a
# `MethodError` rather than a silent fallback to cache-held state.
material_routine(
    material_model::AbstractMaterialModel,
    F::Tensor{2},
    coefficient_cache,
    state_cache::Union{EmptyInternalCache, TrivialCondensationMaterialStateCache},
    geometry_cache::Ferrite.CellCache,
    qp::QuadraturePoint,
    time,
    Qflat,
    Qknownflat,
    Δt,
) = material_routine(material_model, F, coefficient_cache, state_cache, geometry_cache, qp, time)

function reduced_material_routine(
    material_model::AbstractMaterialModel,
    F::Tensor{2},
    coefficient_cache,
    ::EmptyInternalCache,
    geometry_cache::Ferrite.CellCache,
    qp::QuadraturePoint,
    time,
)
    coefficients = evaluate_coefficient(coefficient_cache, geometry_cache, qp, time)
    return stress_function(material_model, F, coefficients, EmptyInternalModel())
end

function reduced_material_routine(
    material_model::AbstractMaterialModel,
    F::Tensor{2},
    coefficient_cache,
    state_cache::TrivialCondensationMaterialStateCache,
    geometry_cache::Ferrite.CellCache,
    qp::QuadraturePoint,
    time,
)
    coefficients = evaluate_coefficient(coefficient_cache, geometry_cache, qp, time)
    Q = state(state_cache, geometry_cache, qp, time)
    return stress_function(material_model, F, coefficients, Q)
end

# See the `material_routine` counterpart above.
reduced_material_routine(
    material_model::AbstractMaterialModel,
    F::Tensor{2},
    coefficient_cache,
    state_cache::RateTypeCondensationMaterialStateCache,
    geometry_cache::Ferrite.CellCache,
    qp::QuadraturePoint,
    time,
) = error(
    "$(typeof(material_model)) carries a rate-type internal variable, so it needs a timestep and a " *
    "previous state. It cannot be assembled on the rate-free path used by e.g. HomotopyPathSolver.",
)

# `gto1` form, see the `material_routine` counterpart above.
function reduced_material_routine(
    material_model::AbstractMaterialModel,
    F::Tensor{2},
    coefficient_cache,
    state_cache::RateTypeCondensationMaterialStateCache,
    geometry_cache::Ferrite.CellCache,
    qp::QuadraturePoint,
    time,
    Qflat,
    Qknownflat,
    Δt,
)
    coefficients = evaluate_coefficient(coefficient_cache, geometry_cache, qp, time)
    Q = solve_local_constraint_state_only(
        F,
        coefficients,
        material_model,
        state_cache,
        geometry_cache,
        qp,
        time,
        Qflat,
        Qknownflat,
        Δt,
    )
    # Residual-only variant: no tangent is requested here, matching the other
    # `reduced_material_routine` methods and the single-value call site in solid/elements.jl.
    return stress_function(material_model, F, coefficients, Q)
end

# Rate-coupled residual. This method is what keeps the residual and the tangent posing the same local
# problem: without it the generic kinematics forwarding would unpack `F` and drop `Ḟ`, so the
# residual-only assembly would freeze the sarcomere while `material_routine` linearizes a moving one.
function reduced_material_routine(
    material_model::AbstractMaterialModel,
    kinematics::DeformationGradientWithRate,
    coefficient_cache,
    state_cache::RateDependentCondensationMaterialStateCache,
    geometry_cache::Ferrite.CellCache,
    qp::QuadraturePoint,
    time,
    Qflat,
    Qknownflat,
    Δt,
)
    F = deformation_gradient(kinematics)
    coefficients = evaluate_coefficient(coefficient_cache, geometry_cache, qp, time)
    Q = solve_local_constraint_state_only(
        F,
        deformation_rate(kinematics),
        coefficients,
        material_model,
        state_cache,
        geometry_cache,
        qp,
        time,
        Qflat,
        Qknownflat,
        Δt,
    )
    return stress_function(material_model, F, coefficients, Q)
end

# See the `material_routine` counterpart: a rate dependent material may not be served rate-free
# kinematics, and the message should name the call the element actually made.
reduced_material_routine(
    material_model::AbstractMaterialModel,
    kinematics::DeformationGradient,
    coefficient_cache,
    state_cache::RateDependentCondensationMaterialStateCache,
    geometry_cache::Ferrite.CellCache,
    qp::QuadraturePoint,
    time,
    Qflat,
    Qknownflat,
    Δt,
) = deformation_rate(kinematics)

reduced_material_routine(
    material_model::AbstractMaterialModel,
    F::Tensor{2},
    coefficient_cache,
    state_cache::Union{EmptyInternalCache, TrivialCondensationMaterialStateCache},
    geometry_cache::Ferrite.CellCache,
    qp::QuadraturePoint,
    time,
    Qflat,
    Qknownflat,
    Δt,
) = reduced_material_routine(
    material_model,
    F,
    coefficient_cache,
    state_cache,
    geometry_cache,
    qp,
    time,
)

@doc raw"""
    PrestressedMechanicalModel(inner_model, prestress_field)

Models the stress formulated in the 1st Piola-Kirchhoff stress tensor based on a multiplicative split
of the deformation gradient $$F = F_{\textrm{e}} F_{0}$$ where we compute $$P(F_{\textrm{e}}) = P(F F^{-1}_{0})$$.

Please note that it is assumed that $$F^{-1}_{0}$$ is the quantity computed by `prestress_field`.
"""
struct PrestressedMechanicalModel{MM, FF} <: AbstractMaterialModel
    inner_model::MM
    prestress_field::FF
end

struct PrestressedMechanicalModelCoefficientCache{T1, T2}
    inner_cache::T1
    prestress_cache::T2
end

default_initial_state!(uq, model::PrestressedMechanicalModel) =
    default_initial_state!(uq, model.inner_model)

function setup_coefficient_cache(
    m::PrestressedMechanicalModel,
    qr::QuadratureRule,
    sdh::SubDofHandler,
)
    return PrestressedMechanicalModelCoefficientCache(
        setup_coefficient_cache(m.inner_model, qr, sdh),
        setup_coefficient_cache(m.prestress_field, qr, sdh),
    )
end
function duplicate_for_device(device, cache::PrestressedMechanicalModelCoefficientCache)
    return PrestressedMechanicalModelCoefficientCache(
        duplicate_for_device(device, cache.inner_cache),
        duplicate_for_device(device, cache.prestress_cache),
    )
end

material_routine(
    material_model::PrestressedMechanicalModel,
    F::Tensor{2},
    coefficient_cache,
    state_cache::EmptyInternalCache,
    geometry_cache::Ferrite.CellCache,
    qp::QuadraturePoint,
    time,
) = prestressed_material_routine(
    material_model,
    F,
    coefficient_cache,
    state_cache,
    geometry_cache,
    qp,
    time,
)
material_routine(
    material_model::PrestressedMechanicalModel,
    F::Tensor{2},
    coefficient_cache,
    state_cache::TrivialCondensationMaterialStateCache,
    geometry_cache::Ferrite.CellCache,
    qp::QuadraturePoint,
    time,
) = prestressed_material_routine(
    material_model,
    F,
    coefficient_cache,
    state_cache,
    geometry_cache,
    qp,
    time,
)
material_routine(
    material_model::PrestressedMechanicalModel,
    F::Tensor{2},
    coefficient_cache,
    state_cache::RateTypeCondensationMaterialStateCache,
    geometry_cache::Ferrite.CellCache,
    qp::QuadraturePoint,
    time,
) = prestressed_material_routine(
    material_model,
    F,
    coefficient_cache,
    state_cache,
    geometry_cache,
    qp,
    time,
)
function prestressed_material_routine(
    material_model::PrestressedMechanicalModel,
    F::Tensor{2},
    coefficient_cache,
    state_cache,
    geometry_cache::Ferrite.CellCache,
    qp::QuadraturePoint,
    time,
)
    F₀inv = evaluate_coefficient(coefficient_cache.prestress_cache, geometry_cache, qp, time)
    Fᵉ = F ⋅ F₀inv
    ∂Ψᵉ∂Fᵉ, ∂²Ψᵉ∂Fᵉ² = material_routine(
        material_model.inner_model,
        Fᵉ,
        coefficient_cache.inner_cache,
        state_cache,
        geometry_cache,
        qp,
        time,
    )
    Pᵉ = ∂Ψᵉ∂Fᵉ # Elastic PK1
    P = Pᵉ ⋅ transpose(F₀inv) # Obtained by Coleman-Noll procedure
    Aᵉ = ∂²Ψᵉ∂Fᵉ² # Elastic mixed modulus
    # TODO condense these steps into a single operation "A_imkn F_jm F_ln"
    # Pull elastic modulus from intermediate to reference configuration
    ∂Pᵉ∂F = Aᵉ ⋅ transpose(F₀inv)
    ∂P∂F = dot_2_1t(∂Pᵉ∂F, F₀inv)
    return P, ∂P∂F
end

reduced_material_routine(
    material_model::PrestressedMechanicalModel,
    F::Tensor{2},
    coefficient_cache,
    state_cache::EmptyInternalCache,
    geometry_cache::Ferrite.CellCache,
    qp::QuadraturePoint,
    time,
) = reduced_prestressed_material_routine(
    material_model,
    F,
    coefficient_cache,
    state_cache,
    geometry_cache,
    qp,
    time,
)
reduced_material_routine(
    material_model::PrestressedMechanicalModel,
    F::Tensor{2},
    coefficient_cache,
    state_cache::TrivialCondensationMaterialStateCache,
    geometry_cache::Ferrite.CellCache,
    qp::QuadraturePoint,
    time,
) = reduced_prestressed_material_routine(
    material_model,
    F,
    coefficient_cache,
    state_cache,
    geometry_cache,
    qp,
    time,
)
reduced_material_routine(
    material_model::PrestressedMechanicalModel,
    F::Tensor{2},
    coefficient_cache,
    state_cache::RateTypeCondensationMaterialStateCache,
    geometry_cache::Ferrite.CellCache,
    qp::QuadraturePoint,
    time,
) = reduced_prestressed_material_routine(
    material_model,
    F,
    coefficient_cache,
    state_cache,
    geometry_cache,
    qp,
    time,
)
function reduced_prestressed_material_routine(
    material_model::PrestressedMechanicalModel,
    F::Tensor{2},
    coefficient_cache,
    state_cache,
    geometry_cache::Ferrite.CellCache,
    qp::QuadraturePoint,
    time,
)
    F₀inv = evaluate_coefficient(coefficient_cache.prestress_cache, geometry_cache, qp, time)
    Fᵉ = F ⋅ F₀inv
    ∂Ψᵉ∂Fᵉ = reduced_material_routine(
        material_model.inner_model,
        Fᵉ,
        coefficient_cache.inner_cache,
        state_cache,
        geometry_cache,
        qp,
        time,
    )
    Pᵉ = ∂Ψᵉ∂Fᵉ # Elastic PK1
    P = Pᵉ ⋅ transpose(F₀inv) # Obtained by Coleman-Noll procedure
    return P
end
setup_internal_cache(
    material_model::PrestressedMechanicalModel,
    qr::QuadratureRule,
    sdh::SubDofHandler,
) = setup_internal_cache(material_model.inner_model, qr, sdh)
internal_variable_evolution(material_model::PrestressedMechanicalModel) =
    internal_variable_evolution(material_model.inner_model)

@doc raw"""
    PK1Model(material, coefficient_field)
    PK1Model(material, internal_model, coefficient_field)

Models the stress formulated in the 1st Piola-Kirchhoff stress tensor. If the material is energy-based,
then the term is formulated as follows:
$$\int_{\Omega_0} P(u,s) \cdot \delta F dV = \int_{\Omega_0} \partial_{F} \psi(u,s) \cdot \delta \nabla u $$
"""
struct PK1Model{PMat, IMod, CFType} <: AbstractMaterialModel
    material::PMat
    internal_model::IMod
    coefficient_field::CFType
end

PK1Model(material, coefficient_field) = PK1Model(material, EmptyInternalModel(), coefficient_field)

function setup_coefficient_cache(m::PK1Model, qr::QuadratureRule, sdh::SubDofHandler)
    return setup_coefficient_cache(m.coefficient_field, qr, sdh)
end

default_initial_state!(uq, model::PK1Model) = default_initial_state!(uq, model.internal_model)

setup_internal_cache(material_model::PK1Model, qr::QuadratureRule, sdh::SubDofHandler) =
    setup_internal_cache(material_model.internal_model, qr, sdh)
internal_variable_evolution(material_model::PK1Model) =
    internal_variable_evolution(material_model.internal_model)

# The AD helpers below take the *energy model* rather than the material model, and that is the whole
# point: a Julia closure captures whole variables, not the fields its body reads. Writing
# `F_ad -> Ψ(F_ad, coefficients, model.material)` therefore captures all of `model`, so
# `Tensors.hessian` — which specializes on the closure type — is recompiled for every distinct
# `PK1Model{PMat, IMod, CFType}`, even though `IMod` and `CFType` are never read inside. That is
# ~1.6 s of byte-identical AD codegen per redundant combination.
#
# Capturing only the leaves keys the specialization on `(typeof(material), typeof(coefficients))`.
# `coefficients` here is the *evaluated* coefficient value, so different coefficient *fields*
# (`ConstantCoefficient`, `FieldCoefficient`, …) collapse onto one instance while everything the
# arithmetic depends on stays concrete. Identical code reaches LLVM; only how often changes.
@inline _pk1_stress(material, F, coefficients) =
    Tensors.gradient(F_ad -> Ψ(F_ad, coefficients, material), F)
@inline _pk1_stress_and_tangent(material, F, coefficients) =
    Tensors.hessian(F_ad -> Ψ(F_ad, coefficients, material), F, :all)

function stress_function(model::PK1Model, F::Tensor{2}, coefficients, ::EmptyInternalModel)
    ∂Ψ∂F = _pk1_stress(model.material, F, coefficients)

    return ∂Ψ∂F
end

function stress_and_tangent(model::PK1Model, F::Tensor{2}, coefficients, ::EmptyInternalModel)
    ∂²Ψ∂F², ∂Ψ∂F = _pk1_stress_and_tangent(model.material, F, coefficients)

    return ∂Ψ∂F, ∂²Ψ∂F²
end

@doc raw"""
    GeneralizedHillModel(passive_spring_model, active_spring_model, active_deformation_gradient_model,contraction_model, microstructure_model)

The generalized Hill framework as proposed by [GokMenKuh:2014:ghm](@citet).

In this framework the model is formulated as an energy minimization problem with the following additively split energy:

$W(\mathbf{F}, \mathbf{F}^{\rm{a}}) = W_{\rm{passive}}(\mathbf{F}) + W_{\rm{active}}(\mathbf{F}\mathbf{F}^{-\rm{a}})$

Where $W_{\rm{passive}}$ is the passive material response and $W_{\rm{active}}$ the active response
respectvely.
"""
struct GeneralizedHillModel{PMat, AMat, ADGMod, CMod, MS} <: AbstractMaterialModel
    passive_spring::PMat
    active_spring::AMat
    active_deformation_gradient_model::ADGMod
    contraction_model::CMod
    microstructure_model::MS
end

function setup_coefficient_cache(m::GeneralizedHillModel, qr::QuadratureRule, sdh::SubDofHandler)
    return setup_coefficient_cache(m.microstructure_model, qr, sdh)
end

# Capture the two springs, not the model — see the note above `_pk1_stress`.
@inline _generalized_hill_stress(passive_spring, active_spring, Fᵃ, F, coefficients) =
    Tensors.gradient(
        F_ad -> Ψ(F_ad, coefficients, passive_spring) + Ψ(F_ad, Fᵃ, coefficients, active_spring),
        F,
    )
@inline _generalized_hill_stress_and_tangent(passive_spring, active_spring, Fᵃ, F, coefficients) =
    Tensors.hessian(
        F_ad -> Ψ(F_ad, coefficients, passive_spring) + Ψ(F_ad, Fᵃ, coefficients, active_spring),
        F,
        :all,
    )

function stress_function(model::GeneralizedHillModel, F::Tensor{2}, coefficients, state)
    # TODO what is a good abstraction here?
    Fᵃ = compute_Fᵃ(
        state,
        coefficients,
        model.contraction_model,
        model.active_deformation_gradient_model,
    )

    ∂Ψ∂F = _generalized_hill_stress(model.passive_spring, model.active_spring, Fᵃ, F, coefficients)

    return ∂Ψ∂F
end

function stress_and_tangent(model::GeneralizedHillModel, F::Tensor{2}, coefficients, state)
    # TODO what is a good abstraction here?
    Fᵃ = compute_Fᵃ(
        state,
        coefficients,
        model.contraction_model,
        model.active_deformation_gradient_model,
    )

    ∂²Ψ∂F², ∂Ψ∂F = _generalized_hill_stress_and_tangent(
        model.passive_spring,
        model.active_spring,
        Fᵃ,
        F,
        coefficients,
    )

    return ∂Ψ∂F, ∂²Ψ∂F²
end


@doc raw"""
    ExtendedHillModel(passive_spring_model, active_spring_model, active_deformation_gradient_model,contraction_model, microstructure_model)

The extended (generalized) Hill model as proposed by [OgiBalPer:2023:aeg](@citet). The original formulation dates back to [StaKlaHol:2008:smc](@citet) for smooth muscle tissues.

In this framework the model is formulated as an energy minimization problem with the following additively split energy:

$W(\mathbf{F}, \mathbf{F}^{\rm{a}}) = W_{\rm{passive}}(\mathbf{F}) + \mathcal{N}(\bm{\alpha})W_{\rm{active}}(\mathbf{F}\mathbf{F}^{-\rm{a}})$

Where $W_{\rm{passive}}$ is the passive material response and $W_{\rm{active}}$ the active response
respectvely. $\mathcal{N}$ is the amount of formed crossbridges. We refer to the original paper [OgiBalPer:2023:aeg](@cite) for more details.
"""
struct ExtendedHillModel{PMat, AMat, ADGMod, CMod, MS} <: AbstractMaterialModel
    passive_spring::PMat
    active_spring::AMat
    active_deformation_gradient_model::ADGMod
    contraction_model::CMod
    microstructure_model::MS
end

function setup_coefficient_cache(m::ExtendedHillModel, qr::QuadratureRule, sdh::SubDofHandler)
    return setup_coefficient_cache(m.microstructure_model, qr, sdh)
end

# Capture the two springs and the scalar `N`, not the model — see the note above `_pk1_stress`.
@inline _extended_hill_stress(passive_spring, active_spring, Fᵃ, N, F, coefficients) =
    Tensors.gradient(
        F_ad -> Ψ(F_ad, coefficients, passive_spring) + N*Ψ(F_ad, Fᵃ, coefficients, active_spring),
        F,
    )
@inline _extended_hill_stress_and_tangent(passive_spring, active_spring, Fᵃ, N, F, coefficients) =
    Tensors.hessian(
        F_ad -> Ψ(F_ad, coefficients, passive_spring) + N*Ψ(F_ad, Fᵃ, coefficients, active_spring),
        F,
        :all,
    )

function stress_function(model::ExtendedHillModel, F::Tensor{2}, coefficients, cell_state)
    # TODO what is a good abstraction here?
    Fᵃ = compute_Fᵃ(
        cell_state,
        coefficients,
        model.contraction_model,
        model.active_deformation_gradient_model,
    )
    N = 𝓝(cell_state, F, coefficients, model.contraction_model)

    ∂Ψ∂F = _extended_hill_stress(model.passive_spring, model.active_spring, Fᵃ, N, F, coefficients)

    return ∂Ψ∂F
end

function stress_and_tangent(model::ExtendedHillModel, F::Tensor{2}, coefficients, cell_state)
    # TODO what is a good abstraction here?
    Fᵃ = compute_Fᵃ(
        cell_state,
        coefficients,
        model.contraction_model,
        model.active_deformation_gradient_model,
    )
    N = 𝓝(cell_state, F, coefficients, model.contraction_model)

    ∂²Ψ∂F², ∂Ψ∂F = _extended_hill_stress_and_tangent(
        model.passive_spring,
        model.active_spring,
        Fᵃ,
        N,
        F,
        coefficients,
    )

    return ∂Ψ∂F, ∂²Ψ∂F²
end


@doc raw"""
    ActiveStressModel(material_model, active_stress_model, contraction_model, microstructure_model)

The active stress model as originally proposed by [GucWalMcC:1993:mac](@citet).

In this framework the model is formulated via balance of linear momentum in the first Piola Kirchhoff $\mathbf{P}$:

$\mathbf{P}(\mathbf{F},T^{\rm{a}}) := \partial_{\mathbf{F}} W_{\rm{passive}}(\mathbf{F}) + \mathbf{P}^{\rm{a}}(\mathbf{F}, T^{\rm{a}})$

where the passive material response can be described by an energy $W_{\rm{passive}$ and $T^{\rm{a}}$ the active tension generated by the contraction model.
"""
struct ActiveStressModel{Mat, ASMod, CMod, MS} <: AbstractMaterialModel
    material_model::Mat
    active_stress_model::ASMod
    contraction_model::CMod
    microstructure_model::MS
end

# An active stress model is rate dependent exactly when its sarcomere is, so the material's
# capability reads straight off the internal variable model rather than being declared twice.
rate_dependence(model::ActiveStressModel) =
    _rate_dependence(internal_variable_evolution(model.contraction_model))
_rate_dependence(::RateCoupledEvolution) = RateDependent()
_rate_dependence(::InternalVariableEvolution) = RateIndependent()

default_initial_state!(
    uq,
    model::Union{GeneralizedHillModel, ExtendedHillModel, ActiveStressModel},
) = default_initial_state!(uq, model.contraction_model)

function setup_coefficient_cache(m::ActiveStressModel, qr::QuadratureRule, sdh::SubDofHandler)
    return setup_coefficient_cache(m.microstructure_model, qr, sdh)
end

# Two independent AD closures here, so two pairs of leaves to capture — see the note above
# `_pk1_stress`. Neither reads `model.microstructure_model`, which is why `MS` is dead weight in the
# specialization key.
@inline _active_stress_passive(material_model, F, coefficients) =
    Tensors.gradient(F_ad -> Ψ(F_ad, coefficients, material_model), F)
@inline _active_stress_passive_and_tangent(material_model, F, coefficients) =
    Tensors.hessian(F_ad -> Ψ(F_ad, coefficients, material_model), F, :all)
@inline _active_stress_active_and_tangent(
    contraction_model,
    active_stress_model,
    cell_state,
    F,
    coefficients,
) = Tensors.gradient(
    F_ad ->
        𝓝(cell_state, F_ad, coefficients, contraction_model) *
        active_stress(active_stress_model, F_ad, coefficients),
    F,
    :all,
)

function stress_function(model::ActiveStressModel, F::Tensor{2}, coefficients, cell_state)
    ∂Ψ∂F = _active_stress_passive(model.material_model, F, coefficients)

    P2 =
        𝓝(cell_state, F, coefficients, model.contraction_model) *
        active_stress(model.active_stress_model, F, coefficients)
    return ∂Ψ∂F + P2
end
function stress_and_tangent(model::ActiveStressModel, F::Tensor{2}, coefficients, cell_state)
    ∂²Ψ∂F², ∂Ψ∂F = _active_stress_passive_and_tangent(model.material_model, F, coefficients)

    ∂2, P2 = _active_stress_active_and_tangent(
        model.contraction_model,
        model.active_stress_model,
        cell_state,
        F,
        coefficients,
    )
    return ∂Ψ∂F + P2, ∂²Ψ∂F² + ∂2
end

function gather_internal_variable_infos(model::ActiveStressModel)
    return gather_internal_variable_infos(model.contraction_model)
end

setup_internal_cache(
    material_model::Union{<:ActiveStressModel, <:ExtendedHillModel, <:GeneralizedHillModel},
    qr::QuadratureRule,
    sdh::SubDofHandler,
) = setup_contraction_model_cache(material_model.contraction_model, qr, sdh)
setup_internal_cache(
    material_model::Union{
        <:ElastodynamicsModel{<:ActiveStressModel},
        <:ElastodynamicsModel{<:ExtendedHillModel},
        <:ElastodynamicsModel{<:GeneralizedHillModel},
    },
    qr::QuadratureRule,
    sdh::SubDofHandler,
) = setup_contraction_model_cache(material_model.rhs.contraction_model, qr, sdh)
internal_variable_evolution(
    material_model::Union{<:ActiveStressModel, <:ExtendedHillModel, <:GeneralizedHillModel},
) = internal_variable_evolution(material_model.contraction_model)
internal_variable_evolution(
    material_model::Union{
        <:ElastodynamicsModel{<:ActiveStressModel},
        <:ElastodynamicsModel{<:ExtendedHillModel},
        <:ElastodynamicsModel{<:GeneralizedHillModel},
    },
) = internal_variable_evolution(material_model.rhs.contraction_model)

# TODO this actually belongs to the multi-level newton file :)
# Dual (global cache and element-level cache) use for now to make it non-allocating.
# Immutable by construction: everything time dependent (`Q`, `Qprev`, `Δt`) is now supplied per call
# by `gto1`, and the `localQ`/`localQprev` scratch existed only to copy the state out of the global
# vector. What remains is genuinely per-subdomain setup data.
#
# This is where the GPU story improves: the cache no longer captures a global solution vector, so
# nothing here has to be re-pointed to move an element loop onto a device.
struct GenericFirstOrderRateIndependentCondensationMaterialStateCache{
    LocalModelType,
    LocalModelCacheType,
    LocalSolverType,
} <: RateIndependentCondensationMaterialStateCache
    # The actual model
    model::LocalModelType
    model_cache::LocalModelCacheType
    local_solver_cache::LocalSolverType
end

function duplicate_for_device(
    device,
    cache::GenericFirstOrderRateIndependentCondensationMaterialStateCache,
)
    return GenericFirstOrderRateIndependentCondensationMaterialStateCache(
        cache.model,
        duplicate_for_device(device, cache.model_cache),
        duplicate_for_device(device, cache.local_solver_cache),
    )
end

"""
    GenericFirstOrderRateDependentCondensationMaterialStateCache

The `RateCoupledEvolution` counterpart of
[`GenericFirstOrderRateIndependentCondensationMaterialStateCache`](@ref): same data, but its local
problem is `dₜQ = L(F, dₜF, Q)`, so it needs the deformation rate on top of `F`.

The two carry identical fields and differ only in their supertype. That is the point: the supertype
is what tells `solve_local_constraint` which local problem to pose, and Julia's single inheritance
gives a struct exactly one. Wrapping a rate dependent sarcomere in `AsRateIndependent` selects the
other cache, and with it the other local problem.
"""
struct GenericFirstOrderRateDependentCondensationMaterialStateCache{
    LocalModelType,
    LocalModelCacheType,
    LocalSolverType,
} <: RateDependentCondensationMaterialStateCache
    model::LocalModelType
    model_cache::LocalModelCacheType
    local_solver_cache::LocalSolverType
end

function duplicate_for_device(
    device,
    cache::GenericFirstOrderRateDependentCondensationMaterialStateCache,
)
    return GenericFirstOrderRateDependentCondensationMaterialStateCache(
        cache.model,
        duplicate_for_device(device, cache.model_cache),
        duplicate_for_device(device, cache.local_solver_cache),
    )
end

"""
    GenericFirstOrderCondensationMaterialStateCache

Either generic condensation cache. Used by the parts of the local solve that are genuinely shared —
the Newton loop, the state-only solve — so that only the methods where the deformation rate actually
enters have to be written twice.
"""
const GenericFirstOrderCondensationMaterialStateCache = Union{
    GenericFirstOrderRateIndependentCondensationMaterialStateCache,
    GenericFirstOrderRateDependentCondensationMaterialStateCache,
}

function _solve_local_sarcomere_dQdF(
    dQdλ,
    dλdF,
    λ,
    F,
    coefficients,
    active_term_model,
    wrapper::Union{CaDrivenInternalSarcomereModel, AsRateIndependent},
)
    return _solve_local_sarcomere_dQdF(
        dQdλ,
        dλdF,
        λ,
        F,
        coefficients,
        active_term_model,
        wrapper.model,
    )
end

# Contribution of an internal-variable chain to the stress tangent,
# `∂P/∂Q ⊗ ∂Q/∂X ⊗ ∂X/∂Y`, for the active part `P = (Q₁₈ + Q₂₀)·fso(λ)·active_stress(F)`. `dQdX` comes
# from a corrector solve and is already `+∂Q/∂X` by the implicit function theorem, so the result is
# added to `∂P∂F` as-is.
function _solve_local_sarcomere_dQdF(
    dQdλ,
    dλdF,
    λ,
    F,
    coefficients,
    active_term_model,
    sacromere_model::RDQ20MFModel,
)
    dfgdQ = active_stress(active_term_model, F, coefficients) * fraction_single_overlap(sacromere_model, λ)
    dQdF  = (dQdλ[18] + dQdλ[20]) * dfgdQ ⊗ dλdF
    return dQdF
end

# Local solve
#
# `t` and `Δt` are passed in rather than read from `state_cache`. Under `gto1` the time
# discretization is supplied per call (`GenericFirstOrderTimeParameters`), so the local problem must
# not depend on time data baked into the cache at setup.
function solve_internal_timestep(
    material_model::ActiveStressModel,
    state_cache::GenericFirstOrderCondensationMaterialStateCache,
    geometry_cache,
    qp,
    λ,
    dλdt,
    Q,
    Qprev,
    Ca,
    t,
    Δt,
)
    #     dsdt = sarcomere_rhs(s,λ,t)
    # <=> (sₜ₁ - sₜ₀) / Δt = sarcomere_rhs(sₜ₁,λₜ₁,t1)

    function local_residual!(R, Q, λ, dλdt)
        dQ = zeros(eltype(Q), length(Q)) # TODO preallocate during setup
        sarcomere_rhs!(dQ, Q, λ, dλdt, Ca, t, material_model.contraction_model)
        @.. R = (Q - Qprev) / Δt - dQ
        return nothing
    end

    function local_residual_jac_wrap!(R, Q)
        return local_residual!(R, Q, λ, dλdt)
    end

    lcache = state_cache.local_solver_cache
    cid = cellid(geometry_cache)
    R = lcache.residual
    J = lcache.J
    # Inexact Newton: the outer solve writes the square of its own residual norm into `outer_tol`, so
    # the local problems are solved loosely while the global iterate is far from the solution and at
    # full accuracy once it is close. `params.tol` is the floor -- the local residual is an absolute
    # quantity on a scale unrelated to the global one, so demanding more than `params.tol` is neither
    # useful nor reachable within `max_iters`. `outer_tol = 0` means "no relaxation".
    rtol = max(lcache.params.tol, lcache.outer_tol[1])
    for newton_iter = 1:lcache.params.max_iters
        ForwardDiff.jacobian!(J, local_residual_jac_wrap!, R, Q)
        local_residual!(R, Q, λ, dλdt)
        residualnorm = norm(R)
        # A singular local Jacobian is a failure the time integrator can act on by shortening the
        # step, so it must not escape as an exception -- `J \ R` would throw.
        Jfac = lu(J; check = false)
        if !issuccess(Jfac)
            record_local_solve!(
                lcache,
                cid,
                qp.i,
                SciMLBase.ReturnCode.InternalLinearSolveFailed,
                residualnorm,
            )
            @debug "Local Newton hit a singular Jacobian at cell $cid qp $(qp.i). ||r|| = $residualnorm" _group =
                :nlsolve
            return Q, J
        end
        Q .-= Jfac \ R
        if residualnorm < rtol
            break
        elseif newton_iter == lcache.params.max_iters
            record_local_solve!(lcache, cid, qp.i, SciMLBase.ReturnCode.MaxIters, residualnorm)
            @debug "Local Newton hit max iterations at cell $cid qp $(qp.i). ||r|| = $residualnorm (rtol = $rtol)" _group =
                :nlsolve
            return Q, J
        elseif isnan(residualnorm)
            record_local_solve!(
                lcache,
                cid,
                qp.i,
                SciMLBase.ReturnCode.ConvergenceFailure,
                residualnorm,
            )
            @debug "Local Newton diverged at cell $cid qp $(qp.i). ||r|| = $residualnorm" _group =
                :nlsolve
            return Q, J
        end
    end
    ForwardDiff.jacobian!(J, local_residual_jac_wrap!, R, Q)
    residualnorm = norm(R)
    # A converged but inadmissible state is still a failure, and one the time integrator can act on:
    # the usual cause is a step too long for the internal variable's own dynamics.
    if !internal_state_in_bounds(material_model.contraction_model, Q)
        record_local_solve!(lcache, cid, qp.i, SciMLBase.ReturnCode.Infeasible, residualnorm)
        @debug "Local Newton converged to an inadmissible state at cell $cid qp $(qp.i). ||r|| = $residualnorm" _group =
            :nlsolve
        return Q, J
    end
    record_local_solve!(lcache, cid, qp.i, SciMLBase.ReturnCode.Success, residualnorm)
    return Q, J
end

# Fiber stretch `λ = |F ⋅ f₀|`. Takes the fiber direction rather than the whole coefficient bundle so
# that the AD closures below capture a leaf -- see the closure-specialization note in CLAUDE.md.
@inline function _fiber_stretch(F, f₀)
    f = F ⋅ f₀
    return √(f ⋅ f)
end

"""
    _solve_local_sarcomere(model, state_cache, geometry_cache, qp, time, λ, dλdt, Qflat, Qknownflat, Δt)

Newton solve of the sarcomere's local problem, shared by the rate-free and rate-coupled entry points.

The stretch and its rate are computed by the *caller*, because the two paths need different
derivatives of `λ`: the rate-free one only its gradient, the rate-coupled one also its Hessian.
"""
function _solve_local_sarcomere(
    material_model::ActiveStressModel,
    state_cache::GenericFirstOrderCondensationMaterialStateCache,
    geometry_cache,
    qp,
    time,
    λ,
    dλdt,
    Qflat,
    Qknownflat,
    Δt,
)
    Ca = evaluate_coefficient(state_cache.model_cache.calcium_cache, geometry_cache, qp, time)
    Q, J = solve_internal_timestep(
        material_model,
        state_cache,
        geometry_cache,
        qp,
        λ,
        dλdt,
        Qflat,
        Qknownflat,
        Ca,
        time,
        Δt,
    )
    return Q, J, Ca
end

# One corrector solve `∂Q/∂x` for a frozen scalar `x`, given the converged state and its Jacobian.
# `rhs_corrector` is scratch that the linear solve consumes immediately, so successive corrector
# solves may reuse it.
@inline function _sarcomere_corrector(state_cache, J, local_residual_rhs_wrap!, x)
    R     = state_cache.local_solver_cache.residual
    ∂fₗ∂x = state_cache.local_solver_cache.rhs_corrector
    ForwardDiff.derivative!(∂fₗ∂x, local_residual_rhs_wrap!, R, x)
    return J \ -∂fₗ∂x
end

# Whether the local solve at *this* quadrature point succeeded. The sensitivity solves below it
# operate on the Jacobian it left behind, so they must not run on a point that failed.
@inline _local_solve_ok(state_cache, geometry_cache, qp) =
    !_local_solve_failed(
        local_solve_report(state_cache.local_solver_cache, cellid(geometry_cache), qp.i),
    )

function solve_local_constraint(
    F::Tensor{2, dim},
    coefficients,
    material_model::ActiveStressModel,
    state_cache::GenericFirstOrderCondensationMaterialStateCache,
    geometry_cache,
    qp,
    time,
    Qflat,
    Qknownflat,
    Δt,
) where {dim}
    f₀ = coefficients.f
    dλdF, λ = Tensors.gradient(F -> _fiber_stretch(F, f₀), F, :all)
    # A zero rate poses the rate-free local problem, `dₜQ = L(F, Q)`.
    dλdt = zero(λ)

    Q, J, Ca = _solve_local_sarcomere(
        material_model,
        state_cache,
        geometry_cache,
        qp,
        time,
        λ,
        dλdt,
        Qflat,
        Qknownflat,
        Δt,
    )
    # Abort if local solve failed
    _local_solve_ok(state_cache, geometry_cache, qp) ||
        return Qflat, zero(Tensor{4, dim, Float64, 4^dim})

    # Reached outside the closures so they capture the leaf models rather than all of
    # `material_model` -- see the closure-specialization note in CLAUDE.md.
    contraction_model   = material_model.contraction_model
    active_stress_model = material_model.active_stress_model

    function local_residual_rhs_wrap!(R, λ)
        dQ = zeros(eltype(λ), length(Q)) # TODO preallocate during setup
        sarcomere_rhs!(dQ, Q, λ, dλdt, Ca, time, contraction_model)
        @.. R = (Q - Qknownflat) / Δt - dQ
        return nothing
    end
    dQdλ = _sarcomere_corrector(state_cache, J, local_residual_rhs_wrap!, λ)

    return Q,
    _solve_local_sarcomere_dQdF(
        dQdλ,
        dλdF,
        λ,
        F,
        coefficients,
        active_stress_model,
        contraction_model,
    )
end

@doc raw"""
Rate-coupled form of the sarcomere local solve, `dₜQ = L(F, dₜF, Q)`.

The internal variable now responds to the stretch *and* to its rate, so there are two corrector
solves, ``\partial Q/\partial\lambda`` and ``\partial Q/\partial\dot\lambda``. They combine into the
two sensitivities the element asks for by the chain rule through
``\dot\lambda = (\partial\lambda/\partial F) : \dot F``:
```math
\frac{\partial Q}{\partial \dot F} = \frac{\partial Q}{\partial \dot\lambda}\otimes\frac{\partial \lambda}{\partial F}
```
```math
\frac{\partial Q}{\partial F} =
  \frac{\partial Q}{\partial \lambda}\otimes\frac{\partial \lambda}{\partial F}
+ \frac{\partial Q}{\partial \dot\lambda}\otimes\left(\frac{\partial^2 \lambda}{\partial F^2} : \dot F\right)
```
`∂Q/∂Ḟ` is exact because ``\lambda`` is a function of ``F`` alone, so ``\dot\lambda`` is *linear* in
``\dot F`` with coefficient ``\partial\lambda/\partial F``.

**The second term of `∂Q/∂F` is not optional.** ``\dot\lambda`` varies with ``F`` too, through the
curvature of ``\lambda``, and that term is the same order as the first. Dropping it leaves a
descent direction rather than a Newton direction: measured on the contracting cuboid, the global
residual then falls by a factor of only ~0.6 per iteration and stalls six orders short of the
tolerance. It is why this method takes the Hessian of ``\lambda`` where the rate-free one takes only
its gradient.
"""
function solve_local_constraint(
    F::Tensor{2, dim},
    Ḟ::Tensor{2, dim},
    coefficients,
    material_model::ActiveStressModel,
    state_cache::GenericFirstOrderRateDependentCondensationMaterialStateCache,
    geometry_cache,
    qp,
    time,
    Qflat,
    Qknownflat,
    Δt,
) where {dim}
    Z = zero(Tensor{4, dim, Float64, 4^dim})

    f₀ = coefficients.f
    ∂²λ∂F², dλdF, λ = Tensors.hessian(F -> _fiber_stretch(F, f₀), F, :all)
    dλdt = dλdF ⊡ Ḟ
    ∂dλdt∂F = ∂²λ∂F² ⊡ Ḟ

    Q, J, Ca = _solve_local_sarcomere(
        material_model,
        state_cache,
        geometry_cache,
        qp,
        time,
        λ,
        dλdt,
        Qflat,
        Qknownflat,
        Δt,
    )
    _local_solve_ok(state_cache, geometry_cache, qp) || return Qflat, Z, Z

    contraction_model   = material_model.contraction_model
    active_stress_model = material_model.active_stress_model

    function local_residual_rhs_wrap!(R, λ)
        dQ = zeros(eltype(λ), length(Q)) # TODO preallocate during setup
        sarcomere_rhs!(dQ, Q, λ, dλdt, Ca, time, contraction_model)
        @.. R = (Q - Qknownflat) / Δt - dQ
        return nothing
    end
    dQdλ = _sarcomere_corrector(state_cache, J, local_residual_rhs_wrap!, λ)

    function local_residual_rate_wrap!(R, dλdt)
        dQ = zeros(eltype(dλdt), length(Q)) # TODO preallocate during setup
        sarcomere_rhs!(dQ, Q, λ, dλdt, Ca, time, contraction_model)
        @.. R = (Q - Qknownflat) / Δt - dQ
        return nothing
    end
    dQddλdt = _sarcomere_corrector(state_cache, J, local_residual_rate_wrap!, dλdt)

    # `_solve_local_sarcomere_dQdF(dQdX, dXdY, …)` is the contribution of the `X` chain to `∂P/∂Y`,
    # so the three chain-rule terms are three calls differing only in their first two arguments, all
    # entering with the same sign.
    ∂P∂QdQdF =
        _solve_local_sarcomere_dQdF(
            dQdλ,
            dλdF,
            λ,
            F,
            coefficients,
            active_stress_model,
            contraction_model,
        ) + _solve_local_sarcomere_dQdF(
            dQddλdt,
            ∂dλdt∂F,
            λ,
            F,
            coefficients,
            active_stress_model,
            contraction_model,
        )
    ∂P∂QdQdḞ = _solve_local_sarcomere_dQdF(
        dQddλdt,
        dλdF,
        λ,
        F,
        coefficients,
        active_stress_model,
        contraction_model,
    )
    return Q, ∂P∂QdQdF, ∂P∂QdQdḞ
end

# Residual-only counterpart: the same local problem, without the corrector solves.
#
# It must pose the *identical* problem to `solve_local_constraint`, rate included. A residual that
# freezes `dλdt` while the tangent linearizes a moving one is not a slower Newton, it is a Newton on
# two different problems.
function solve_local_constraint_state_only(
    F::Tensor{2, dim},
    coefficients,
    material_model::ActiveStressModel,
    state_cache::GenericFirstOrderCondensationMaterialStateCache,
    geometry_cache,
    qp,
    time,
    Qflat,
    Qknownflat,
    Δt,
) where {dim}
    return solve_local_constraint_state_only(
        F,
        zero(F),
        coefficients,
        material_model,
        state_cache,
        geometry_cache,
        qp,
        time,
        Qflat,
        Qknownflat,
        Δt,
    )
end

function solve_local_constraint_state_only(
    F::Tensor{2, dim},
    Ḟ::Tensor{2, dim},
    coefficients,
    material_model::ActiveStressModel,
    state_cache::GenericFirstOrderCondensationMaterialStateCache,
    geometry_cache,
    qp,
    time,
    Qflat,
    Qknownflat,
    Δt,
) where {dim}
    # Only the gradient is needed here: no tangent is requested, so the curvature term that the
    # rate-coupled `solve_local_constraint` needs never arises.
    f₀ = coefficients.f
    dλdF, λ = Tensors.gradient(F -> _fiber_stretch(F, f₀), F, :all)

    Q, _, _ = _solve_local_sarcomere(
        material_model,
        state_cache,
        geometry_cache,
        qp,
        time,
        λ,
        dλdF ⊡ Ḟ,
        Qflat,
        Qknownflat,
        Δt,
    )
    # Abort if local solve failed
    _local_solve_ok(state_cache, geometry_cache, qp) || return Qflat

    return Q
end

# Some debug materials
Base.@kwdef struct LinearMaxwellMaterial{T, sdim} <: AbstractMaterialModel
    E₀::T
    E₁::T
    μ::T
    η₁::T
    ν::T
end
LinearMaxwellMaterial(E₀::T, Eₗ::T, μ::T, η₁::T, ν::T) where {T} =
    LinearMaxwellMaterial{T, 3}(E₀, Eₗ, μ, η₁, ν)

internal_variable_size(model::QuasiStaticModel, cid, qp) =
    internal_variable_size(model.material_model, cid, qp)
function internal_variable_size(model::AbstractMaterialModel, cid, qp)
    return _compute_internal_variable_size(0, gather_internal_variable_infos(model))
end

function _compute_internal_variable_size(total, lvis::Base.AbstractVecOrTuple)
    for lvi in lvis
        total += _compute_internal_variable_size(total, lvi)
    end
    return total
end

function _compute_internal_variable_size(total, lvi::InternalVariableInfo)
    return lvi.size
end

function solve_internal_timestep(
    material::LinearMaxwellMaterial,
    state_cache::GenericFirstOrderCondensationMaterialStateCache,
    ε::SymmetricTensor{2, dim},
    εᵛflat,
    εᵛprevflat,
    Δt,
) where {dim}
    εᵛ₁ = SymmetricTensor{2, dim}(εᵛflat)
    εᵛ₀ = SymmetricTensor{2, dim}(εᵛprevflat)
    #     dεᵛdt = E₁/η₁ c : (ε - εᵛ)
    # <=> (εᵛ₁ - εᵛ₀) / Δt = E₁/η₁ c : (ε - εᵛ₁) = E₁/η₁ c : ε - E₁/η₁ c : εᵛ₁
    # <=> εᵛ₁ / Δt + E₁/η₁ c : εᵛ₁ = εᵛ₀/Δt + E₁/η₁ c : ε
    # <=> (𝐈 / Δt + E₁/η₁ c) : εᵛ₁ = εᵛ₀/Δt + E₁/η₁ c : ε

    (; E₀, E₁, μ, η₁, ν) = material
    I = one(ε)
    c₁ = ν / ((ν + 1)*(1-2ν)) * I ⊗ I
    c₂ = 1 / (1+ν) * one(c₁)
    ℂ = c₁ + c₂

    # FIXME non-allocating version by using state_cache nlsolver
    A = tomandel(SMatrix, one(ℂ)/Δt + E₁/η₁ * ℂ)
    b = tomandel(SVector, εᵛ₀/Δt + E₁/η₁ * ℂ ⊡ ε)
    return frommandel(typeof(ε), A \ b)
end

function solve_local_constraint(
    F::Tensor{2, dim},
    coefficients,
    material_model::LinearMaxwellMaterial,
    state_cache::GenericFirstOrderCondensationMaterialStateCache,
    geometry_cache,
    qp,
    time,
    Qflat,
    Qknownflat,
    Δt,
) where {dim}
    ε = symmetric(F - one(F))
    Q = solve_internal_timestep(material_model, state_cache, ε, Qflat, Qknownflat, Δt)
    Qflat .= Q.data

    # Corrector
    function solve_internal_timestep_corrector(
        material::LinearMaxwellMaterial,
        state_cache::GenericFirstOrderCondensationMaterialStateCache,
        ε,
        εᵛflat,
        εᵛprevflat,
        coefficients,
        Δt,
    )
        εᵛ₁ = SymmetricTensor{2, dim}(εᵛflat)
        εᵛ₀ = SymmetricTensor{2, dim}(εᵛprevflat)
        # Local problem: (𝐈 / Δt + E₁/η₁ c) : εᵛ₁ = εᵛ₀/Δt + E₁/η₁ c : ε
        # =>  dLdQ = 𝐈 / Δt + E₁/η₁ c   := A
        # => -dLdF = E₁/η₁ c            := B

        (; E₀, E₁, μ, η₁, ν) = material
        I = one(ε)
        c₁ = ν / ((ν + 1)*(1-2ν)) * I ⊗ I
        c₂ = 1 / (1+ν) * one(c₁)
        ℂ = c₁ + c₂

        # FIXME non-allocating version by using state_cache nlsolver
        A = tomandel(SMatrix, one(ℂ)/Δt + E₁/η₁ * ℂ)
        B = tomandel(SMatrix, E₁/η₁ * ℂ)
        return frommandel(typeof(ℂ), A \ B)
    end
    dQdF = solve_internal_timestep_corrector(
        material_model,
        state_cache,
        ε,
        Qflat,
        Qknownflat,
        coefficients,
        Δt,
    )
    ∂P∂Q = Tensors.gradient(εᵛ->stress_function(material_model, ε, coefficients, εᵛ), Q)

    return Q, ∂P∂Q ⊡ dQdF
end

function solve_local_constraint_state_only(
    F::Tensor{2, dim},
    coefficients,
    material_model::LinearMaxwellMaterial,
    state_cache::GenericFirstOrderCondensationMaterialStateCache,
    geometry_cache,
    qp,
    time,
    Qflat,
    Qknownflat,
    Δt,
) where {dim}
    ε = symmetric(F - one(F))
    Q = solve_internal_timestep(material_model, state_cache, ε, Qflat, Qknownflat, Δt)
    Qflat .= Q.data

    return Q
end

function stress_function(material::LinearMaxwellMaterial, ε, coefficients, εᵛ)
    (; E₀, E₁, μ, η₁, ν) = material
    I = one(ε)
    c₁ = ν / ((ν + 1)*(1-2ν)) * I ⊗ I
    c₂ = 1 / (1+ν) * one(c₁)
    ℂ = c₁ + c₂
    return E₀ * ℂ ⊡ ε + E₁ * ℂ ⊡ (ε - εᵛ)
end

function stress_and_tangent(
    material_model::LinearMaxwellMaterial,
    F::Tensor{2},
    coefficients,
    εᵛ::SymmetricTensor{2},
)
    ε = symmetric(F - one(F))
    ∂σ∂ε, σ = Tensors.gradient(ε->stress_function(material_model, ε, coefficients, εᵛ), ε, :all)
    return σ, ∂σ∂ε
end

function setup_coefficient_cache(m::LinearMaxwellMaterial, qr::QuadratureRule, sdh::SubDofHandler)
    return NoMicrostructureModel() # FIXME what should we do here? :)
end

function setup_internal_cache(
    material_model::LinearMaxwellMaterial,
    qr::QuadratureRule,
    sdh::SubDofHandler,
)
    return EmptyRateIndependentCondensationMaterialStateCache()
end

# η₁ dₜεᵛ = E₁ (ε - εᵛ), i.e. first order in the internal variable and independent of dₜε.
internal_variable_evolution(::LinearMaxwellMaterial) = FirstOrderEvolution()

function gather_internal_variable_infos(model::LinearMaxwellMaterial{T, sdim}) where {T, sdim}
    if sdim == 3
        return (InternalVariableInfo(:εᵛ, 6),)
    else
        return (InternalVariableInfo(:εᵛ, 4),)
    end
end
