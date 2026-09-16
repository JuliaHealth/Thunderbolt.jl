@doc raw"""
    BilinearMassIntegrator{MT, CV}

Represents the integrand of the bilinearform ``a(u,v) = \int \rho(x) v(x) u(x) dx`` for ``u,v`` from the same function space with some given density field $\rho(x)$.

`qrc` is any collection answering [`getquadraturerule`](@ref), not only a
[`QuadratureRuleCollection`](@ref): a mass is the one term whose *rule* is a modelling choice rather
than an accuracy knob, which is why `FiniteElementDiscretization` lets a `:mass` entry override the
field's rule. A [`NodalQuadratureRuleCollection`](@ref) over a tensor-product Lagrange space is what
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

FerriteOperators.allocate_element_matrix(c::BilinearMassElementCache, sdh) =
    element_matrix_buffer(c.cellvalues, sdh)
FerriteOperators.allocate_element_unknown_vector(c::BilinearMassElementCache, sdh) =
    element_vector_buffer(c.cellvalues, sdh)
FerriteOperators.allocate_element_residual_vector(c::BilinearMassElementCache, sdh) =
    element_vector_buffer(c.cellvalues, sdh)

function setup_element_cache(element_model::BilinearMassIntegrator, sdh)
    @assert length(sdh.dh.field_names) == 1 "Support for multiple fields not yet implemented."
    qr = getquadraturerule(element_model.qrc, sdh)
    field_name = first(sdh.dh.field_names)
    ip = Ferrite.getfieldinterpolation(sdh, field_name)
    ip_geo = geometric_subdomain_interpolation(sdh)
    return BilinearMassElementCache(
        setup_coefficient_cache(element_model.ρ, qr, sdh),
        CellValues(element_precision(qr), qr, ip, ip_geo),
    )
end
