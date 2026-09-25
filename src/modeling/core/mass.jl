@doc raw"""
    BilinearMassIntegrator{MT, CV}

Represents the integrand of the bilinearform ``a(u,v) = \int \rho(x) v(x) u(x) dx`` for ``u,v`` from the same function space with some given density field $\rho(x)$.

`qrc` is any collection answering [`getquadraturerule`](@ref), not only a
[`QuadratureRuleCollection`](@ref): a mass is the one term whose *rule* is a modelling choice rather
than an accuracy knob, which is why `FiniteElementDiscretization` lets a `:mass` entry override the
field's rule. A `NodalQuadratureRuleCollection` over a tensor-product Lagrange space is what
makes the assembled mass diagonal -- the spectral-element mass an explicit integrator can lump for
free.
"""
struct BilinearMassIntegrator{CoefficientType, QRC} <: AbstractBilinearIntegrator
    ρ::CoefficientType
    qrc::QRC
    sym::Symbol
end

"""
The cache associated with [`BilinearMassIntegrator`](@ref) to assemble element mass matrices.
"""
struct BilinearMassElementCache{IT, CV} <: AbstractVolumetricElementCache
    ρcache::IT
    cellvalues::CV
end

function duplicate_for_device(device, cache::BilinearMassElementCache)
    return BilinearMassElementCache(
        duplicate_for_device(device, cache.ρcache),
        duplicate_for_device(device, cache.cellvalues),
    )
end

Ferrite.getnquadpoints(element_cache::BilinearMassElementCache) =
    getnquadpoints(element_cache.cellvalues)
FerriteOperators.reinit_values!(element_cache::BilinearMassElementCache, cell) =
    reinit!(element_cache.cellvalues, cell)

FerriteOperators.provides_analytic(
    ::Type{<:BilinearMassElementCache},
    ::FerriteOperators.JacobianKind{:u},
) = true

function FerriteOperators.assemble_cell!(
    req::FerriteOperators.JacobianRequest{:u},
    element_cache::BilinearMassElementCache,
    args::FerriteOperators.CellArgs,
)
    @unpack ρcache, cellvalues = element_cache
    Mₑ = req.K
    cell = args.cell
    time = FerriteOperators.evaluation_time(args.ctx)
    n_basefuncs = getnbasefunctions(cellvalues)
    for qp in QuadratureIterator(cellvalues)
        ρ = evaluate_coefficient(ρcache, cell, qp, time)
        dΩ = getdetJdV(cellvalues, qp)
        for i = 1:n_basefuncs
            Nᵢ = shape_value(cellvalues, qp, i)
            for j = 1:n_basefuncs
                Nⱼ = shape_value(cellvalues, qp, j)
                Mₑ[i, j] += ρ * (Nᵢ ⋅ Nⱼ) * dΩ
            end
        end
    end
end

# The bilinear form induces a linear operator, so its residual is the element mass matrix acting on
# the element vector -- mandatory, so the element composes into nonlinear operators and AD-based
# sensitivities.
function FerriteOperators.assemble_cell!(
    req::FerriteOperators.ResidualRequest,
    element_cache::BilinearMassElementCache,
    args::FerriteOperators.CellArgs,
)
    @unpack ρcache, cellvalues = element_cache
    cell = args.cell
    uₑ = args.states.u
    time = FerriteOperators.evaluation_time(args.ctx)
    n_basefuncs = getnbasefunctions(cellvalues)
    for qp in QuadratureIterator(cellvalues)
        ρ = evaluate_coefficient(ρcache, cell, qp, time)
        dΩ = getdetJdV(cellvalues, qp)
        u = function_value(cellvalues, qp, uₑ)
        for i = 1:n_basefuncs
            Nᵢ = shape_value(cellvalues, qp, i)
            req.r[i] += ρ * (Nᵢ ⋅ u) * dΩ
        end
    end
end

FerriteOperators.element_value_type(c::BilinearMassElementCache) =
    FerriteOperators.element_value_type(c.cellvalues)

function setup_element_cache(element_model::BilinearMassIntegrator, sdh)
    @assert length(sdh.dh.field_names) == 1 "Support for multiple fields not yet implemented."
    qr = getquadraturerule(element_model.qrc, sdh)
    field_name = first(sdh.dh.field_names)
    ip = Ferrite.getfieldinterpolation(sdh, field_name)
    ip_geo = geometric_subdomain_interpolation(sdh)
    return BilinearMassElementCache(
        setup_coefficient_cache(element_model.ρ, qr, sdh),
        CellValues(FerriteOperators.element_value_type(qr), qr, ip, ip_geo),
    )
end

@doc raw"""
    CollocatedMassIntegrator(ρ, ipc, sym)

The spectral-element mass: ``a(u,v) = \int \rho\, u\, v\, dx`` integrated over the nodes of the
interpolation collection `ipc` itself (`NodalQuadratureRuleCollection`), where every basis
function but one vanishes. The element matrix is diagonal by construction,
``M_{ii} = \rho(\xi_i)\, w_i \det J(\xi_i)``, and is declared and written as such
(`FerriteOperators.DiagonalElementMatrix`): `O(N_b)` per cell, never the square. Exact collocated
weights exist on hypercubes carrying a tensor-product Lagrange space; any other shape is refused at
setup. Built by [`FiniteElementDiscretization`](@ref) under [`CollocatedMass`](@ref).
"""
struct CollocatedMassIntegrator{CoefficientType, IPC <: InterpolationCollection} <:
       AbstractBilinearIntegrator
    ρ::CoefficientType
    ipc::IPC
    sym::Symbol
end

struct CollocatedMassElementCache{IT, CV} <: AbstractVolumetricElementCache
    ρcache::IT
    cellvalues::CV
end

FerriteOperators.element_matrix_structure(::CollocatedMassElementCache) =
    FerriteOperators.DiagonalElementMatrix()
FerriteOperators.element_value_type(c::CollocatedMassElementCache) =
    FerriteOperators.element_value_type(c.cellvalues)
duplicate_for_device(device, cache::CollocatedMassElementCache) = CollocatedMassElementCache(
    duplicate_for_device(device, cache.ρcache),
    duplicate_for_device(device, cache.cellvalues),
)
Ferrite.getnquadpoints(c::CollocatedMassElementCache) = getnquadpoints(c.cellvalues)
FerriteOperators.reinit_values!(c::CollocatedMassElementCache, cell) = reinit!(c.cellvalues, cell)
FerriteOperators.provides_analytic(
    ::Type{<:CollocatedMassElementCache},
    ::FerriteOperators.JacobianKind{:u},
) = true

function FerriteOperators.assemble_cell!(
    req::FerriteOperators.JacobianRequest{:u},
    c::CollocatedMassElementCache,
    args::FerriteOperators.CellArgs,
)
    (; ρcache, cellvalues) = c
    time = FerriteOperators.evaluation_time(args.ctx)
    for qp in QuadratureIterator(cellvalues)
        req.K[qp.i] += evaluate_coefficient(ρcache, args.cell, qp, time) * getdetJdV(cellvalues, qp)
    end
end

function FerriteOperators.assemble_cell!(
    req::FerriteOperators.ResidualRequest,
    c::CollocatedMassElementCache,
    args::FerriteOperators.CellArgs,
)
    (; ρcache, cellvalues) = c
    uₑ = args.states.u
    time = FerriteOperators.evaluation_time(args.ctx)
    for qp in QuadratureIterator(cellvalues)
        req.r[qp.i] +=
            evaluate_coefficient(ρcache, args.cell, qp, time) * getdetJdV(cellvalues, qp) * uₑ[qp.i]
    end
end

function setup_element_cache(element_model::CollocatedMassIntegrator, sdh)
    qr = getquadraturerule(NodalQuadratureRuleCollection(element_model.ipc), sdh)
    any(isnan, Ferrite.getweights(qr)) && error(
        "`CollocatedMassIntegrator` needs the collocated weights of a tensor-product Lagrange space " *
        "on a hypercube; the nodal rule of this subdomain carries none.",
    )
    ip = Ferrite.getfieldinterpolation(sdh, element_model.sym)
    ip_geo = geometric_subdomain_interpolation(sdh)
    cv = CellValues(FerriteOperators.element_value_type(qr), qr, ip, ip_geo)
    getnquadpoints(cv) == getnbasefunctions(cv) || error(
        "`CollocatedMassIntegrator`: the nodal rule has $(getnquadpoints(cv)) points for " *
        "$(getnbasefunctions(cv)) basis functions, so it is not the field's own interpolation.",
    )
    return CollocatedMassElementCache(setup_coefficient_cache(element_model.ρ, qr, sdh), cv)
end
