@doc raw"""
    BilinearMassIntegrator{MT, CV}

Represents the integrand of the bilinearform ``a(u,v) = \int \rho(x) v(x) u(x) dx`` for ``u,v`` from the same function space with some given density field $\rho(x)$.
"""
struct BilinearMassIntegrator{CoefficientType, QRC <: QuadratureRuleCollection} <: AbstractBilinearIntegrator
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

function assemble_element!(Mₑ::AbstractMatrix, cell, element_cache::BilinearMassElementCache, time)
    @unpack ρcache, cellvalues = element_cache
    reinit!(element_cache.cellvalues, cell)
    n_basefuncs = getnbasefunctions(cellvalues)
    for qp in QuadratureIterator(cellvalues)
        ρ = evaluate_coefficient(ρcache, cell, qp, time)
        dΩ = getdetJdV(cellvalues, qp)
        for i in 1:n_basefuncs
            Nᵢ = shape_value(cellvalues, qp, i)
            for j in 1:n_basefuncs
                Nⱼ = shape_value(cellvalues, qp, j)
                Mₑ[i,j] += ρ * Nᵢ * Nⱼ * dΩ 
            end
        end
    end
end

function setup_element_cache(element_model::BilinearMassIntegrator, sdh)
    @assert length(sdh.dh.field_names) == 1 "Support for multiple fields not yet implemented."
    qr = getquadraturerule(element_model.qrc, sdh)
    field_name = first(sdh.dh.field_names)
    ip          = Ferrite.getfieldinterpolation(sdh, field_name)
    ip_geo = geometric_subdomain_interpolation(sdh)
    return BilinearMassElementCache(setup_coefficient_cache(element_model.ρ, qr, sdh), CellValues(qr, ip, ip_geo))
end
