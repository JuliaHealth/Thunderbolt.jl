@doc raw"""
Any boundary condition stated in the weak form
```math
\int f(u, n_0), \delta u \mathrm{d} \partial \Omega_0
```
"""
abstract type AbstractWeakBoundaryCondition end

"""
    getboundaryname(bc::AbstractWeakBoundaryCondition)

The facetset `bc` acts on. That set *is* the term's traversal, so it is also its
`FerriteOperators.facet_items` declaration.
"""
@inline getboundaryname(bc::AbstractWeakBoundaryCondition) = bc.boundary_name

# The declared set restricted to the subdomain that owns the cells: a facet item's local system is
# its owning cell's, so each `SubDofHandler` declares the part of the surface it owns. Nothing here
# couples to a dof outside that cell, so `facet_item_global_dofs` keeps the framework default `()`
# and the item's local system stays cell-shaped.
FerriteOperators.facet_items(bc::AbstractWeakBoundaryCondition, sdh::SubDofHandler) =
    filter(facet -> facet[1] ∈ sdh.cellset, getfacetset(get_grid(sdh.dh), getboundaryname(bc)))

function field_name_of_weak_boundary_condition(
    bc::AbstractWeakBoundaryCondition,
    sdh::SubDofHandler,
)
    return if hasfield(typeof(bc), :field_name) && bc.field_name != :auto
        bc.field_name
    else
        @assert length(sdh.field_names) == 1 "For problems with multiple fields please pass the field name into your weak boundary conditions via `field_name`"
        first(sdh.field_names)
    end
end

@doc raw"""
    RobinBC(α, boundary_name::String [, field_name::Symbol])

```math
\bm{P}(\bm{u}) \cdot \bm{n}_0 = - \alpha \bm{u} \quad \textbf{x} \in \partial \Omega_0,
```

See [`ViscousRobinBC`](@ref) for the dashpot which resists the *rate* instead of the displacement.
"""
struct RobinBC <: AbstractWeakBoundaryCondition
    α::Float64
    boundary_name::String
    field_name::Symbol
end
RobinBC(a, b) = RobinBC(a, b, :auto)

@doc raw"""
    NormalSpringBC(kₛ, boundary_name::String [, field_name::Symbol])

```math
\bm{P}(\bm{u}) \cdot \bm{n}_0 = - k_s (\bm{u} \cdot \bm{n}_0) \bm{n}_0 \quad \textbf{x} \in \partial \Omega_0,
```

See [`ViscousNormalSpringBC`](@ref) for the dashpot which resists the *rate* instead of the
displacement.
"""
struct NormalSpringBC <: AbstractWeakBoundaryCondition
    kₛ::Float64
    boundary_name::String
    field_name::Symbol
end
NormalSpringBC(a, b) = NormalSpringBC(a, b, :auto)

@doc raw"""
    BendingSpringBC(kᵇ, boundary_name::String [, field_name::Symbol])

```math
\bm{P}(\bm{u}) \cdot \bm{n}_0 = - \partial_F \frac{1}{2} k_b \left (cof(F) n_0 - n_0 \right) \quad \textbf{x} \in \partial \Omega_0,
```
"""
struct BendingSpringBC <: AbstractWeakBoundaryCondition
    kᵇ::Float64
    boundary_name::String
    field_name::Symbol
end
BendingSpringBC(a, b) = BendingSpringBC(a, b, :auto)

@doc raw"""
    ConstantPressureBC(p::Real, boundary_name::String [, field_name::Symbol])

```math
\bm{P}(\bm{u}) \cdot \bm{n}_0 = - p n_0 \quad \textbf{x} \in \partial \Omega_0,
```
"""
struct ConstantPressureBC <: AbstractWeakBoundaryCondition
    p::Float64
    boundary_name::String
    field_name::Symbol
end
ConstantPressureBC(a, b) = ConstantPressureBC(a, b, :auto)

@doc raw"""
    PressureFieldBC(pressure_field, boundary_name::String [, field_name::Symbol])

```math
\bm{P}(\bm{u}) \cdot \bm{n}_0 = - p(\bm{x}, t) J \bm{F}^{-\mathrm{T}} \bm{n}_0 \quad \textbf{x} \in \partial \Omega_0,
```
"""
struct PressureFieldBC{C} <: AbstractWeakBoundaryCondition
    pc::C
    boundary_name::String
    field_name::Symbol
end
PressureFieldBC(a, b) = PressureFieldBC(a, b, :auto)

@doc raw"""
Supertype of the weak boundary conditions which resist the *rate* of the displacement rather than the
displacement itself, i.e. the dashpots of the Robin family.

```math
\bm{P} \cdot \bm{n}_0 = - \bm{D}(\bm{n}_0) \cdot \bm{v} \quad \textbf{x} \in \partial \Omega_0,
```

The whole family is linear in the facet velocity ``\bm{v}``, so a subtype is fully described by its
positive semi-definite damping tensor [`damping_tensor`](@ref); it needs no assembly code of its own.

Unlike the springs these read the reconstructed velocity slot `:v`, which only a time scheme can
supply. They are therefore rejected by `HomotopyPathSolver`, which is load stepping and has no rate to
offer. See [`ViscousRobinBC`](@ref) and [`ViscousNormalSpringBC`](@ref).
"""
abstract type AbstractViscousWeakBoundaryCondition <: AbstractWeakBoundaryCondition end

@doc raw"""
    ViscousRobinBC(η, boundary_name::String [, field_name::Symbol])

Dashpot resisting the full velocity, the rate analogue of [`RobinBC`](@ref).

```math
\bm{P} \cdot \bm{n}_0 = - \eta \bm{v} \quad \textbf{x} \in \partial \Omega_0,
```

``\eta`` is a viscosity per unit reference area. The velocity is the one the time scheme reconstructs
from the unknown displacement, so under backward Euler this is
``\bm{v} = (\bm{u}_n - \bm{u}_{n-1}) / \Delta t_n``, and under Newmark the corresponding
``\gamma / (\beta \Delta t)`` reconstruction.
"""
struct ViscousRobinBC <: AbstractViscousWeakBoundaryCondition
    η::Float64
    boundary_name::String
    field_name::Symbol
end
ViscousRobinBC(a, b) = ViscousRobinBC(a, b, :auto)

@doc raw"""
    ViscousNormalSpringBC(cₛ, boundary_name::String [, field_name::Symbol])

Dashpot resisting the normal velocity only, the rate analogue of [`NormalSpringBC`](@ref).

```math
\bm{P} \cdot \bm{n}_0 = - c_s (\bm{v} \cdot \bm{n}_0) \bm{n}_0 \quad \textbf{x} \in \partial \Omega_0,
```

Tangential sliding is left free, which is what makes this the usual companion of
[`NormalSpringBC`](@ref) when modelling the pericardium: the sac resists the chamber pushing against
it, not the chamber sliding within it.
"""
struct ViscousNormalSpringBC <: AbstractViscousWeakBoundaryCondition
    cₛ::Float64
    boundary_name::String
    field_name::Symbol
end
ViscousNormalSpringBC(a, b) = ViscousNormalSpringBC(a, b, :auto)

@doc raw"""
    damping_tensor(bc::AbstractViscousWeakBoundaryCondition, n₀::Vec)

The damping tensor ``\bm{D}`` of a dashpot boundary condition, evaluated at a quadrature point with
reference normal ``n_0``.

This is the entire constitutive content of [`AbstractViscousWeakBoundaryCondition`](@ref): the
traction is ``-\bm{D} \cdot \bm{v}`` and, because that is linear in the velocity, ``\bm{D}`` is also
the sensitivity the tangent needs. A new dashpot is one method here.
"""
function damping_tensor end

@inline damping_tensor(bc::ViscousRobinBC, ::Vec{dim, T}) where {dim, T} =
    bc.η * one(SymmetricTensor{2, dim, T})
@inline damping_tensor(bc::ViscousNormalSpringBC, n₀::Vec) = bc.cₛ * symmetric(n₀ ⊗ n₀)

"""
Standard cache for surface integrals.
"""
struct SimpleFacetCache{MP, FV} <: AbstractSurfaceElementCache
    mp::MP
    fv::FV
    dof_range::UnitRange{Int}
end
function duplicate_for_device(device, cache::SimpleFacetCache)
    return SimpleFacetCache(cache.mp, duplicate_for_device(device, cache.fv), cache.dof_range)
end

FerriteOperators.provides_analytic(
    ::Type{<:SimpleFacetCache},
    ::FerriteOperators.JacobianKind{:u},
) = true
FerriteOperators.provides_analytic(
    ::Type{<:SimpleFacetCache},
    ::FerriteOperators.JacobianResidualKind,
) = true
# Every boundary condition served by this cache is a function of `(u, t)` alone and never reads the
# rate slot `:v` -- that is where the dashpots of `ViscousFacetCache` act -- so the `:u` weight is the
# only one of a weighted Jacobian which acts here.
FerriteOperators.provides_analytic(
    ::Type{<:SimpleFacetCache},
    ::FerriteOperators.WeightedJacobianKind,
) = true

# A weighted sweep which does not name `:u` weights no displacement sensitivity at all, so a
# displacement-only boundary condition contributes nothing to it.
@inline _u_weight(weights::NamedTuple) = haskey(weights, :u) ? weights.u : false

# `global_dof_range` is the slice of the facet items' global-dof tail this term owns, which for a weak
# boundary condition is empty -- it is a parameter of the hook, not of this family. The same holds for
# the two `setup_facet_item_cache` methods below.
function FerriteOperators.setup_facet_item_cache(
    facet_model::AbstractWeakBoundaryCondition,
    qr::FacetQuadratureRule,
    sdh::SubDofHandler,
    global_dof_range = (),
)
    field_name = field_name_of_weak_boundary_condition(facet_model, sdh)
    ip         = Ferrite.getfieldinterpolation(sdh, field_name)
    ip_geo     = geometric_subdomain_interpolation(sdh)
    dof_range  = Ferrite.dof_range(sdh, field_name)
    return SimpleFacetCache(facet_model, FacetValues(qr, ip, ip_geo), dof_range)
end

# The kernels below visit every shape function of the cell, but on a facet those of the other nodes
# vanish. Their rows and columns are exactly zero, so they are skipped: `iszero(δuᵢ) && continue`.

function FerriteOperators.assemble_facet!(
    req::FerriteOperators.JacobianResidualRequest,
    cache::SimpleFacetCache{<:RobinBC},
    args,
    lfi::Int,
)
    @unpack mp, fv = cache
    @unpack α = mp
    uₑ = args.states.u

    reinit!(fv, args.cell, lfi)

    ndofs_facet = getnbasefunctions(fv)
    for qp in QuadratureIterator(fv)
        dΓ = getdetJdV(fv, qp)

        u_q = function_value(fv, qp, @view uₑ[cache.dof_range])
        ∂²Ψ∂u², ∂Ψ∂u = Tensors.hessian(u -> α*u⋅u, u_q, :all)

        # Add contribution to the residual from this test function
        for i = 1:ndofs_facet
            δuᵢ = shape_value(fv, qp, i)
            iszero(δuᵢ) && continue
            req.r[cache.dof_range[i]] += δuᵢ ⋅ ∂Ψ∂u * dΓ

            δuᵢ∂²Ψ∂u² = δuᵢ ⋅ ∂²Ψ∂u² # Hoisted computation
            for j = 1:ndofs_facet
                δuⱼ = shape_value(fv, qp, j)
                iszero(δuⱼ) && continue
                # Add contribution to the tangent
                req.K[cache.dof_range[i], cache.dof_range[j]] += (δuᵢ∂²Ψ∂u² ⋅ δuⱼ) * dΓ
            end
        end
    end
end

FerriteOperators.assemble_facet!(
    req::FerriteOperators.JacobianRequest{:u},
    cache::SimpleFacetCache{<:RobinBC},
    args,
    lfi::Int,
) = _assemble_robin_jacobian!(req, cache, args, lfi, true)

FerriteOperators.assemble_facet!(
    req::FerriteOperators.WeightedJacobianRequest,
    cache::SimpleFacetCache{<:RobinBC},
    args,
    lfi::Int,
) = _assemble_robin_jacobian!(req, cache, args, lfi, _u_weight(req.weights))

function _assemble_robin_jacobian!(req, cache::SimpleFacetCache{<:RobinBC}, args, lfi::Int, w)
    @unpack mp, fv = cache
    @unpack α = mp
    uₑ = args.states.u

    reinit!(fv, args.cell, lfi)

    ndofs_facet = getnbasefunctions(fv)
    for qp in QuadratureIterator(fv)
        dΓ = getdetJdV(fv, qp)

        u_q = function_value(fv, qp, @view uₑ[cache.dof_range])
        ∂²Ψ∂u², ∂Ψ∂u = Tensors.hessian(u -> α*u⋅u, u_q, :all)

        for i = 1:ndofs_facet
            δuᵢ = shape_value(fv, qp, i)
            iszero(δuᵢ) && continue

            δuᵢ∂²Ψ∂u² = δuᵢ ⋅ ∂²Ψ∂u² # Hoisted computation
            for j = 1:ndofs_facet
                δuⱼ = shape_value(fv, qp, j)
                iszero(δuⱼ) && continue
                # Add contribution to the tangent
                req.K[cache.dof_range[i], cache.dof_range[j]] += w * (δuᵢ∂²Ψ∂u² ⋅ δuⱼ) * dΓ
            end
        end
    end
end

function FerriteOperators.assemble_facet!(
    req::FerriteOperators.ResidualRequest,
    cache::SimpleFacetCache{<:RobinBC},
    args,
    lfi::Int,
)
    @unpack mp, fv = cache
    @unpack α = mp
    uₑ = args.states.u

    reinit!(fv, args.cell, lfi)

    ndofs_facet = getnbasefunctions(fv)
    for qp in QuadratureIterator(fv)
        dΓ = getdetJdV(fv, qp)

        u_q = function_value(fv, qp, @view uₑ[cache.dof_range])
        ∂Ψ∂u = Tensors.gradient(u -> α*u⋅u, u_q)

        # Add contribution to the residual from this test function
        for i = 1:ndofs_facet
            δuᵢ = shape_value(fv, qp, i)
            iszero(δuᵢ) && continue
            req.r[cache.dof_range[i]] += δuᵢ ⋅ ∂Ψ∂u * dΓ
        end
    end
end



function FerriteOperators.assemble_facet!(
    req::FerriteOperators.JacobianResidualRequest,
    cache::SimpleFacetCache{<:NormalSpringBC},
    args,
    lfi::Int,
)
    @unpack mp, fv = cache
    @unpack kₛ = mp
    uₑ = args.states.u

    reinit!(fv, args.cell, lfi)

    ndofs_facet = getnbasefunctions(fv)
    for qp in QuadratureIterator(fv)
        dΓ = getdetJdV(fv, qp)
        N = getnormal(fv, qp)

        u_q = function_value(fv, qp, @view uₑ[cache.dof_range])
        ∂²Ψ∂u², ∂Ψ∂u = Tensors.hessian(u -> 0.5*kₛ*(u⋅N)^2, u_q, :all)

        # Add contribution to the residual from this test function
        for i = 1:ndofs_facet
            δuᵢ = shape_value(fv, qp, i)
            iszero(δuᵢ) && continue
            req.r[cache.dof_range[i]] += δuᵢ ⋅ ∂Ψ∂u * dΓ

            δuᵢ∂²Ψ∂u² = δuᵢ ⋅ ∂²Ψ∂u² # Hoisted computation
            for j = 1:ndofs_facet
                δuⱼ = shape_value(fv, qp, j)
                iszero(δuⱼ) && continue
                # Add contribution to the tangent
                req.K[cache.dof_range[i], cache.dof_range[j]] += (δuᵢ∂²Ψ∂u² ⋅ δuⱼ) * dΓ
            end
        end
    end
end

FerriteOperators.assemble_facet!(
    req::FerriteOperators.JacobianRequest{:u},
    cache::SimpleFacetCache{<:NormalSpringBC},
    args,
    lfi::Int,
) = _assemble_normal_spring_jacobian!(req, cache, args, lfi, true)

FerriteOperators.assemble_facet!(
    req::FerriteOperators.WeightedJacobianRequest,
    cache::SimpleFacetCache{<:NormalSpringBC},
    args,
    lfi::Int,
) = _assemble_normal_spring_jacobian!(req, cache, args, lfi, _u_weight(req.weights))

function _assemble_normal_spring_jacobian!(
    req,
    cache::SimpleFacetCache{<:NormalSpringBC},
    args,
    lfi::Int,
    w,
)
    @unpack mp, fv = cache
    @unpack kₛ = mp
    uₑ = args.states.u

    reinit!(fv, args.cell, lfi)

    ndofs_facet = getnbasefunctions(fv)
    for qp in QuadratureIterator(fv)
        dΓ = getdetJdV(fv, qp)
        N = getnormal(fv, qp)

        u_q = function_value(fv, qp, @view uₑ[cache.dof_range])
        ∂²Ψ∂u², ∂Ψ∂u = Tensors.hessian(u -> 0.5*kₛ*(u⋅N)^2, u_q, :all)

        for i = 1:ndofs_facet
            δuᵢ = shape_value(fv, qp, i)
            iszero(δuᵢ) && continue

            δuᵢ∂²Ψ∂u² = δuᵢ ⋅ ∂²Ψ∂u² # Hoisted computation
            for j = 1:ndofs_facet
                δuⱼ = shape_value(fv, qp, j)
                iszero(δuⱼ) && continue
                # Add contribution to the tangent
                req.K[cache.dof_range[i], cache.dof_range[j]] += w * (δuᵢ∂²Ψ∂u² ⋅ δuⱼ) * dΓ
            end
        end
    end
end

function FerriteOperators.assemble_facet!(
    req::FerriteOperators.ResidualRequest,
    cache::SimpleFacetCache{<:NormalSpringBC},
    args,
    lfi::Int,
)
    @unpack mp, fv = cache
    @unpack kₛ = mp
    uₑ = args.states.u

    reinit!(fv, args.cell, lfi)

    ndofs_facet = getnbasefunctions(fv)
    for qp in QuadratureIterator(fv)
        dΓ = getdetJdV(fv, qp)
        N = getnormal(fv, qp)

        u_q = function_value(fv, qp, @view uₑ[cache.dof_range])
        ∂Ψ∂u = Tensors.gradient(u -> 0.5*kₛ*(u⋅N)^2, u_q)

        # Add contribution to the residual from this test function
        for i = 1:ndofs_facet
            δuᵢ = shape_value(fv, qp, i)
            iszero(δuᵢ) && continue
            req.r[cache.dof_range[i]] += δuᵢ ⋅ ∂Ψ∂u * dΓ
        end
    end
end



function FerriteOperators.assemble_facet!(
    req::FerriteOperators.JacobianResidualRequest,
    cache::SimpleFacetCache{<:BendingSpringBC},
    args,
    lfi::Int,
)
    @unpack mp, fv = cache
    @unpack kᵇ = mp
    uₑ = args.states.u

    reinit!(fv, args.cell, lfi)

    ndofs_facet = getnbasefunctions(fv)
    for qp in QuadratureIterator(fv)
        dΓ = getdetJdV(fv, qp)
        N = getnormal(fv, qp)

        ∇u = function_gradient(fv, qp, @view uₑ[cache.dof_range])
        F = one(∇u) + ∇u

        ∂²Ψ∂F², ∂Ψ∂F = Tensors.hessian(
            F_ -> 0.5*kᵇ*(transpose(inv(F_))⋅N - N)⋅(transpose(inv(F_))⋅N - N),
            F,
            :all,
        )

        # Add contribution to the residual from this test function
        for i = 1:ndofs_facet
            ∇δui = shape_gradient(fv, qp, i)
            req.r[cache.dof_range[i]] += ∇δui ⊡ ∂Ψ∂F * dΓ

            ∇δui∂P∂F = ∇δui ⊡ ∂²Ψ∂F² # Hoisted computation
            for j = 1:ndofs_facet
                ∇δuj = shape_gradient(fv, qp, j)
                # Add contribution to the tangent
                req.K[cache.dof_range[i], cache.dof_range[j]] += (∇δui∂P∂F ⊡ ∇δuj) * dΓ
            end
        end
    end
end

FerriteOperators.assemble_facet!(
    req::FerriteOperators.JacobianRequest{:u},
    cache::SimpleFacetCache{<:BendingSpringBC},
    args,
    lfi::Int,
) = _assemble_bending_spring_jacobian!(req, cache, args, lfi, true)

FerriteOperators.assemble_facet!(
    req::FerriteOperators.WeightedJacobianRequest,
    cache::SimpleFacetCache{<:BendingSpringBC},
    args,
    lfi::Int,
) = _assemble_bending_spring_jacobian!(req, cache, args, lfi, _u_weight(req.weights))

function _assemble_bending_spring_jacobian!(
    req,
    cache::SimpleFacetCache{<:BendingSpringBC},
    args,
    lfi::Int,
    w,
)
    @unpack mp, fv = cache
    @unpack kᵇ = mp
    uₑ = args.states.u

    reinit!(fv, args.cell, lfi)

    ndofs_facet = getnbasefunctions(fv)
    for qp in QuadratureIterator(fv)
        dΓ = getdetJdV(fv, qp)
        N = getnormal(fv, qp)

        ∇u = function_gradient(fv, qp, @view uₑ[cache.dof_range])
        F = one(∇u) + ∇u

        ∂²Ψ∂F², ∂Ψ∂F = Tensors.hessian(
            F_ -> 0.5*kᵇ*(transpose(inv(F_))⋅N - N)⋅(transpose(inv(F_))⋅N - N),
            F,
            :all,
        )

        for i = 1:ndofs_facet
            ∇δui = shape_gradient(fv, qp, i)

            ∇δui∂P∂F = ∇δui ⊡ ∂²Ψ∂F² # Hoisted computation
            for j = 1:ndofs_facet
                ∇δuj = shape_gradient(fv, qp, j)
                # Add contribution to the tangent
                req.K[cache.dof_range[i], cache.dof_range[j]] += w * (∇δui∂P∂F ⊡ ∇δuj) * dΓ
            end
        end
    end
end

function FerriteOperators.assemble_facet!(
    req::FerriteOperators.ResidualRequest,
    cache::SimpleFacetCache{<:BendingSpringBC},
    args,
    lfi::Int,
)
    @unpack mp, fv = cache
    @unpack kᵇ = mp
    uₑ = args.states.u

    reinit!(fv, args.cell, lfi)

    ndofs_facet = getnbasefunctions(fv)
    for qp in QuadratureIterator(fv)
        dΓ = getdetJdV(fv, qp)
        N = getnormal(fv, qp)

        ∇u = function_gradient(fv, qp, @view uₑ[cache.dof_range])
        F = one(∇u) + ∇u

        ∂Ψ∂F =
            Tensors.gradient(F_ -> 0.5*kᵇ*(transpose(inv(F_))⋅N - N)⋅(transpose(inv(F_))⋅N - N), F)

        # Add contribution to the residual from this test function
        for i = 1:ndofs_facet
            ∇δui = shape_gradient(fv, qp, i)
            req.r[cache.dof_range[i]] += ∇δui ⊡ ∂Ψ∂F * dΓ
        end
    end
end



function assemble_facet_pressure_qp!(
    req::FerriteOperators.JacobianResidualRequest,
    uₑ::AbstractVector,
    p,
    qp,
    fv::FacetValues,
    dof_range,
)
    ndofs_facet = getnbasefunctions(fv)

    dΓ = getdetJdV(fv, qp)
    n₀ = getnormal(fv, qp)

    ∇u = function_gradient(fv, qp, @view uₑ[dof_range])
    F = one(∇u) + ∇u

    invF = inv(F)
    cofF = transpose(invF)
    J = det(F)
    # @info qp, J, cofF ⋅ n₀
    neumann_term = p * J * cofF ⋅ n₀
    # neumann_term = p * n₀
    for i = 1:ndofs_facet
        δuᵢ = shape_value(fv, qp, i)
        iszero(δuᵢ) && continue
        req.r[dof_range[i]] += neumann_term ⋅ δuᵢ * dΓ
    end
    _assemble_facet_pressure_tangent_qp!(req, invF, cofF, J, n₀, p * dΓ, qp, fv, dof_range)
end

# The term depends on the trial function alone, up to the test function it is multiplied with last.
function _assemble_facet_pressure_tangent_qp!(req, invF, cofF, J, n₀, pdΓ, qp, fv, dof_range)
    ndofs_facet = getnbasefunctions(fv)
    for j = 1:ndofs_facet
        ∇δuⱼ = shape_gradient(fv, qp, j)
        # Add contribution to the tangent
        #   δF^-1 = -F^-1 δF F^-1
        #   δJ = J tr(δF F^-1)
        # Product rule
        δcofF = -transpose(invF ⋅ ∇δuⱼ ⋅ invF)
        δJ = J * tr(∇δuⱼ ⋅ invF)
        δJcofF = δJ * cofF + J * δcofF
        tangent_term = pdΓ * (δJcofF ⋅ n₀) # Hoisted computation
        for i = 1:ndofs_facet
            δuᵢ = shape_value(fv, qp, i)
            iszero(δuᵢ) && continue
            req.K[dof_range[i], dof_range[j]] += tangent_term ⋅ δuᵢ
        end
    end
end

function assemble_facet_pressure_jacobian_qp!(
    req,
    uₑ::AbstractVector,
    p,
    qp,
    fv::FacetValues,
    dof_range,
    w,
)
    dΓ = getdetJdV(fv, qp)
    n₀ = getnormal(fv, qp)

    ∇u = function_gradient(fv, qp, @view uₑ[dof_range])
    F = one(∇u) + ∇u

    invF = inv(F)
    cofF = transpose(invF)
    J = det(F)
    _assemble_facet_pressure_tangent_qp!(req, invF, cofF, J, n₀, w * p * dΓ, qp, fv, dof_range)
end

function assemble_facet_pressure_qp!(
    req::FerriteOperators.ResidualRequest,
    uₑ::AbstractVector,
    p,
    qp,
    fv::FacetValues,
    dof_range,
)
    ndofs_facet = getnbasefunctions(fv)

    dΓ = getdetJdV(fv, qp)
    n₀ = getnormal(fv, qp)

    ∇u = function_gradient(fv, qp, @view uₑ[dof_range])
    F = one(∇u) + ∇u

    invF = inv(F)
    cofF = transpose(invF)
    J = det(F)
    neumann_term = p * J * cofF ⋅ n₀
    # neumann_term = p * n₀
    for i = 1:ndofs_facet
        δuᵢ = shape_value(fv, qp, i)
        iszero(δuᵢ) && continue
        req.r[dof_range[i]] += neumann_term ⋅ δuᵢ * dΓ
    end
end


function FerriteOperators.assemble_facet!(
    req::FerriteOperators.JacobianResidualRequest,
    cache::SimpleFacetCache{<:PressureFieldBC},
    args,
    lfi::Int,
)
    @unpack mp, fv = cache
    @unpack pc = mp
    t = FerriteOperators.evaluation_time(args.ctx)

    reinit!(fv, args.cell, lfi)

    for qp in QuadratureIterator(fv)
        pressure = evaluate_coefficient(pc, args.cell, qp, t)
        assemble_facet_pressure_qp!(req, args.states.u, pressure, qp, fv, cache.dof_range)
    end
end

FerriteOperators.assemble_facet!(
    req::FerriteOperators.JacobianRequest{:u},
    cache::SimpleFacetCache{<:PressureFieldBC},
    args,
    lfi::Int,
) = _assemble_pressure_field_jacobian!(req, cache, args, lfi, true)

FerriteOperators.assemble_facet!(
    req::FerriteOperators.WeightedJacobianRequest,
    cache::SimpleFacetCache{<:PressureFieldBC},
    args,
    lfi::Int,
) = _assemble_pressure_field_jacobian!(req, cache, args, lfi, _u_weight(req.weights))

function _assemble_pressure_field_jacobian!(
    req,
    cache::SimpleFacetCache{<:PressureFieldBC},
    args,
    lfi::Int,
    w,
)
    @unpack mp, fv = cache
    @unpack pc = mp
    t = FerriteOperators.evaluation_time(args.ctx)

    reinit!(fv, args.cell, lfi)

    for qp in QuadratureIterator(fv)
        pressure = evaluate_coefficient(pc, args.cell, qp, t)
        assemble_facet_pressure_jacobian_qp!(
            req,
            args.states.u,
            pressure,
            qp,
            fv,
            cache.dof_range,
            w,
        )
    end
end

function FerriteOperators.assemble_facet!(
    req::FerriteOperators.ResidualRequest,
    cache::SimpleFacetCache{<:PressureFieldBC},
    args,
    lfi::Int,
)
    @unpack mp, fv = cache
    @unpack pc = mp
    t = FerriteOperators.evaluation_time(args.ctx)

    reinit!(fv, args.cell, lfi)

    for qp in QuadratureIterator(fv)
        pressure = evaluate_coefficient(pc, args.cell, qp, t)
        assemble_facet_pressure_qp!(req, args.states.u, pressure, qp, fv, cache.dof_range)
    end
end



function FerriteOperators.assemble_facet!(
    req::FerriteOperators.JacobianResidualRequest,
    cache::SimpleFacetCache{<:ConstantPressureBC},
    args,
    lfi::Int,
)
    @unpack mp, fv = cache
    pressure = mp.p

    reinit!(fv, args.cell, lfi)

    for qp in QuadratureIterator(fv)
        assemble_facet_pressure_qp!(req, args.states.u, pressure, qp, fv, cache.dof_range)
    end
end

FerriteOperators.assemble_facet!(
    req::FerriteOperators.JacobianRequest{:u},
    cache::SimpleFacetCache{<:ConstantPressureBC},
    args,
    lfi::Int,
) = _assemble_constant_pressure_jacobian!(req, cache, args, lfi, true)

FerriteOperators.assemble_facet!(
    req::FerriteOperators.WeightedJacobianRequest,
    cache::SimpleFacetCache{<:ConstantPressureBC},
    args,
    lfi::Int,
) = _assemble_constant_pressure_jacobian!(req, cache, args, lfi, _u_weight(req.weights))

function _assemble_constant_pressure_jacobian!(
    req,
    cache::SimpleFacetCache{<:ConstantPressureBC},
    args,
    lfi::Int,
    w,
)
    @unpack mp, fv = cache
    pressure = mp.p

    reinit!(fv, args.cell, lfi)

    for qp in QuadratureIterator(fv)
        assemble_facet_pressure_jacobian_qp!(
            req,
            args.states.u,
            pressure,
            qp,
            fv,
            cache.dof_range,
            w,
        )
    end
end

function FerriteOperators.assemble_facet!(
    req::FerriteOperators.ResidualRequest,
    cache::SimpleFacetCache{<:ConstantPressureBC},
    args,
    lfi::Int,
)
    @unpack mp, fv = cache
    pressure = mp.p

    reinit!(fv, args.cell, lfi)

    for qp in QuadratureIterator(fv)
        assemble_facet_pressure_qp!(req, args.states.u, pressure, qp, fv, cache.dof_range)
    end
end


# --- viscous (dashpot) boundary conditions ------------------------------------------------------
#
# Same data as `SimpleFacetCache`, different type -- and the type is the whole point, since it is what
# selects the kernel set. A dashpot is a function of the velocity rather than of `(u, t)`: its residual
# reads the reconstructed `:v` slot, which only a time scheme supplies. The same split, for the same
# reason, distinguishes `QuasiStaticElementCache` from the condensed caches in `solid/elements.jl`.
#
# The whole family is linear in the velocity, so `damping_tensor` is both the traction and the tangent
# sensitivity and no automatic differentiation is involved. The tangent is independent of `uₑ`.

struct ViscousFacetCache{MP, FV} <: AbstractSurfaceElementCache
    mp::MP
    fv::FV
    dof_range::UnitRange{Int}
end
function duplicate_for_device(device, cache::ViscousFacetCache)
    return ViscousFacetCache(cache.mp, duplicate_for_device(device, cache.fv), cache.dof_range)
end

FerriteOperators.provides_analytic(
    ::Type{<:ViscousFacetCache},
    ::FerriteOperators.JacobianKind{:u},
) = true
FerriteOperators.provides_analytic(
    ::Type{<:ViscousFacetCache},
    ::FerriteOperators.JacobianResidualKind,
) = true
FerriteOperators.provides_analytic(
    ::Type{<:ViscousFacetCache},
    ::FerriteOperators.WeightedJacobianKind,
) = true

function FerriteOperators.setup_facet_item_cache(
    facet_model::AbstractViscousWeakBoundaryCondition,
    qr::FacetQuadratureRule,
    sdh::SubDofHandler,
    global_dof_range = (),
)
    field_name = field_name_of_weak_boundary_condition(facet_model, sdh)
    ip         = Ferrite.getfieldinterpolation(sdh, field_name)
    ip_geo     = geometric_subdomain_interpolation(sdh)
    dof_range  = Ferrite.dof_range(sdh, field_name)
    return ViscousFacetCache(facet_model, FacetValues(qr, ip, ip_geo), dof_range)
end

function FerriteOperators.assemble_facet!(
    req::FerriteOperators.ResidualRequest,
    cache::ViscousFacetCache,
    args,
    lfi::Int,
)
    @unpack mp, fv = cache
    vₑ = args.states.v

    reinit!(fv, args.cell, lfi)

    ndofs_facet = getnbasefunctions(fv)
    for qp in QuadratureIterator(fv)
        dΓ = getdetJdV(fv, qp)
        D  = damping_tensor(mp, getnormal(fv, qp))

        traction = D ⋅ function_value(fv, qp, @view vₑ[cache.dof_range])

        for i = 1:ndofs_facet
            δuᵢ = shape_value(fv, qp, i)
            iszero(δuᵢ) && continue
            req.r[cache.dof_range[i]] += δuᵢ ⋅ traction * dΓ
        end
    end
end

# `:v` is a reconstructed slot and is frozen under a `:u` Jacobian, so a dashpot's `∂F/∂u` is zero;
# the damping block belongs to the weighted kernel below.
FerriteOperators.assemble_facet!(
    req::FerriteOperators.JacobianRequest{:u},
    cache::ViscousFacetCache,
    args,
    lfi::Int,
) = nothing

FerriteOperators.assemble_facet!(
    req::FerriteOperators.JacobianResidualRequest,
    cache::ViscousFacetCache,
    args,
    lfi::Int,
) = FerriteOperators.assemble_facet!(FerriteOperators.ResidualRequest(req.r), cache, args, lfi)

function FerriteOperators.assemble_facet!(
    req::FerriteOperators.WeightedJacobianRequest,
    cache::ViscousFacetCache,
    args,
    lfi::Int,
)
    @unpack mp, fv = cache
    haskey(req.weights, :v) || throw(
        ArgumentError(
            "A dashpot boundary condition contributes `∂v∂u ⋅ D` to the `:v` slot, but the weighted " *
            "Jacobian was requested with weights $(keys(req.weights)) and no `:v` entry.",
        ),
    )
    wv = req.weights.v

    reinit!(fv, args.cell, lfi)

    ndofs_facet = getnbasefunctions(fv)
    for qp in QuadratureIterator(fv)
        dΓ = getdetJdV(fv, qp)
        D  = damping_tensor(mp, getnormal(fv, qp))

        for i = 1:ndofs_facet
            δuᵢ = shape_value(fv, qp, i)
            iszero(δuᵢ) && continue
            δuᵢD = δuᵢ ⋅ D # Hoisted computation
            for j = 1:ndofs_facet
                δuⱼ = shape_value(fv, qp, j)
                iszero(δuⱼ) && continue
                req.K[cache.dof_range[i], cache.dof_range[j]] += wv * (δuᵢD ⋅ δuⱼ) * dΓ
            end
        end
    end
end


# We can use this to debug weak BCs for their consistency
struct ConsistencyCheckWeakBoundaryCondition{BC} <: AbstractWeakBoundaryCondition
    bc::BC
    Δ::Float64
end

"""
Finite-difference referee for the inner cache's tangent: assembles the inner cache's Jacobian and
compares it against the difference quotient of the inner cache's residual.

Which slots are perturbed, and with which chain-rule scalar, is the request's own combination: a
fused `∂F/∂u` differences the `:u` slot alone, a weighted `Σₛ wₛ ∂F/∂s` differences every slot its
weights name. Every other slot is held at the value the sweep gathered, so a fused check on a
`ViscousFacetCache` compares its `:u` Jacobian against zero while the weighted check sees the
dashpot's rate term.
"""
struct ConsistencyCheckWeakBoundaryConditionCache{IC} <: AbstractSurfaceElementCache
    inner_cache::IC
    Kₑfd::Matrix{Float64}
    uₑfd::Vector{Float64}
    residualₑfd::Vector{Float64}
    residualₑref::Vector{Float64}
    Δ::Float64
end
function duplicate_for_device(device, cache::ConsistencyCheckWeakBoundaryConditionCache)
    return ConsistencyCheckWeakBoundaryConditionCache(
        duplicate_for_device(device, cache.inner_cache),
        duplicate_for_device(device, cache.Kₑfd),
        duplicate_for_device(device, cache.uₑfd),
        duplicate_for_device(device, cache.residualₑfd),
        duplicate_for_device(device, cache.residualₑref),
        cache.Δ,
    )
end
@inline getboundaryname(check::ConsistencyCheckWeakBoundaryCondition) = getboundaryname(check.bc)

FerriteOperators.provides_analytic(
    ::Type{<:ConsistencyCheckWeakBoundaryConditionCache},
    ::FerriteOperators.JacobianResidualKind,
) = true
# The referee pulls its own residual sweeps, so a request carrying no residual is checkable too.
FerriteOperators.provides_analytic(
    ::Type{<:ConsistencyCheckWeakBoundaryConditionCache},
    ::FerriteOperators.WeightedJacobianKind,
) = true

function FerriteOperators.setup_facet_item_cache(
    ccc::ConsistencyCheckWeakBoundaryCondition,
    qr::FacetQuadratureRule,
    sdh::SubDofHandler,
    global_dof_range = (),
)
    # The scratch is the local system the referee differences over, which for a weak boundary
    # condition is its owning cell's -- the family's global-dof tail is empty for everything this
    # referee wraps.
    N = ndofs_per_cell(sdh)
    return ConsistencyCheckWeakBoundaryConditionCache(
        FerriteOperators.setup_facet_item_cache(ccc.bc, qr, sdh, global_dof_range),
        zeros(N, N),
        zeros(N),
        zeros(N),
        zeros(N),
        ccc.Δ,
    )
end

# A residual sweep carries no tangent to check, so it is the inner cache's alone.
FerriteOperators.assemble_facet!(
    req::FerriteOperators.ResidualRequest,
    cache::ConsistencyCheckWeakBoundaryConditionCache,
    args,
    lfi::Int,
) = FerriteOperators.assemble_facet!(req, cache.inner_cache, args, lfi)

# The two tangent-carrying requests differ only in which combination they claim to assemble, so they
# hand that combination to the same referee. The fused route's is `∂F/∂u`, i.e. the `:u` slot alone at
# unit weight; the weighted route's is the request's own `weights`, which is the only route that
# carries a reconstructed slot into the matrix and therefore the only one that can check it.
FerriteOperators.assemble_facet!(
    req::FerriteOperators.JacobianResidualRequest,
    cache::ConsistencyCheckWeakBoundaryConditionCache,
    args,
    lfi::Int,
) = _check_weak_bc_tangent!(req, cache, args, lfi, (u = true,))

FerriteOperators.assemble_facet!(
    req::FerriteOperators.WeightedJacobianRequest,
    cache::ConsistencyCheckWeakBoundaryConditionCache,
    args,
    lfi::Int,
) = _check_weak_bc_tangent!(req, cache, args, lfi, req.weights)

function _check_weak_bc_tangent!(
    req,
    cache::ConsistencyCheckWeakBoundaryConditionCache,
    args,
    lfi::Int,
    weights::NamedTuple,
)
    (; Δ, inner_cache, Kₑfd, uₑfd, residualₑfd, residualₑref) = cache

    # The incoming element matrix might be non-empty, so we need to start by storing the offset.
    Kₑfd .= req.K

    # The actual assembly is happening here
    FerriteOperators.assemble_facet!(req, inner_cache, args, lfi)

    # Now we get a fresh reference state to pull the differences
    fill!(residualₑref, 0.0)
    FerriteOperators.assemble_facet!(
        FerriteOperators.ResidualRequest(residualₑref),
        inner_cache,
        args,
        lfi,
    )
    # Here we actually compute the finite difference, one weighted slot at a time
    for (slot, w) in pairs(weights)
        sₑ = args.states[slot]
        for i = 1:length(uₑfd)
            fill!(residualₑfd, 0.0)
            uₑfd    .= sₑ
            uₑfd[i] += Δ
            FerriteOperators.assemble_facet!(
                FerriteOperators.ResidualRequest(residualₑfd),
                inner_cache,
                with_states(args, merge(args.states, NamedTuple{(slot,)}((uₑfd,)))),
                lfi,
            )
            residualₑfd .-= residualₑref
            residualₑfd ./= Δ
            Kₑfd[:, i] .+= w .* residualₑfd
        end
    end

    # Finally we check for consistency
    if maximum(abs.(Kₑfd .- req.K)) > Δ
        @warn "Inconsistent element $(cellid(args.cell)) facet $(lfi)! Jacobian difference: $(maximum(abs.(Kₑfd .- req.K)))"
        @info args.states.u
    end
end


# No weak boundary condition contributes to a facet reduction. `nothing` is the fold's "no
# contribution", and stating it is what lets one share a facet item with a term that *does*
# contribute -- a spring supported on a surface the chamber tying integrates over.
FerriteOperators.evaluate_facet_functional(
    kind,
    ::Union{SimpleFacetCache, ViscousFacetCache, ConsistencyCheckWeakBoundaryConditionCache},
    args,
    lfi::Int,
) = nothing
