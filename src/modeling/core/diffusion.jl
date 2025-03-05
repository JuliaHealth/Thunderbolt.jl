@doc raw"""
    BilinearDiffusionIntegrator{CoefficientType}

Represents the integrand of the bilinear form ``a(u,v) = -\int \nabla v(x) \cdot D(x) \nabla u(x) dx`` for a given diffusion tensor ``D(x)`` and ``u,v`` from the same function space.
"""
struct BilinearDiffusionIntegrator{CoefficientType, QRC <: QuadratureRuleCollection} <: AbstractBilinearIntegrator
    D::CoefficientType
    qrc::QRC
    sym::Symbol
end

"""
The cache associated with [`BilinearDiffusionIntegrator`](@ref) to assemble element diffusion matrices.
"""
struct BilinearDiffusionElementCache{CoefficientCacheType, CV} <: AbstractVolumetricElementCache
    Dcache::CoefficientCacheType
    cellvalues::CV
end

function assemble_element!(Kₑ::AbstractMatrix, cell, element_cache::BilinearDiffusionElementCache, time)
    @unpack cellvalues, Dcache = element_cache
    n_basefuncs = getnbasefunctions(cellvalues)

    reinit!(cellvalues, cell)

    for qp in QuadratureIterator(cellvalues)
        D_loc = evaluate_coefficient(Dcache, cell, qp, time)
        dΩ = getdetJdV(cellvalues, qp)
        for i in 1:n_basefuncs
            ∇Nᵢ = shape_gradient(cellvalues, qp, i)
            for j in 1:n_basefuncs
                ∇Nⱼ = shape_gradient(cellvalues, qp, j)
                Kₑ[i,j] -= _inner_product_helper(∇Nⱼ, D_loc, ∇Nᵢ) * dΩ
            end
        end
    end
end

function setup_element_cache(element_model::BilinearDiffusionIntegrator, sdh::SubDofHandler)
    @assert length(sdh.dh.field_names) == 1 "Support for multiple fields not yet implemented."
    qr = getquadraturerule(element_model.qrc, sdh)
    field_name = first(sdh.dh.field_names)
    ip          = Ferrite.getfieldinterpolation(sdh, field_name)
    ip_geo = geometric_subdomain_interpolation(sdh)
    BilinearDiffusionElementCache(setup_coefficient_cache(element_model.D, qr, sdh), CellValues(qr, ip, ip_geo))
end

@doc raw"""
    TransientDiffusionModel(conductivity_coefficient, source_term, solution_variable_symbol)

Model formulated as ``\partial_t u = \nabla \cdot \kappa(x) \nabla u + f``
"""
struct TransientDiffusionModel{ConductivityCoefficientType, SourceType <: AbstractSourceTerm}
    κ::ConductivityCoefficientType
    source::SourceType
    solution_variable_symbol::Symbol
end

@doc raw"""
    SteadyDiffusionModel(conductivity_coefficient, source_term, solution_variable_symbol)

Model formulated as ``\nabla \cdot \kappa(x) \nabla u = f``
"""
struct SteadyDiffusionModel{ConductivityCoefficientType, SourceType <: AbstractSourceTerm}
    κ::ConductivityCoefficientType
    source::SourceType
    solution_variable_symbol::Symbol
end
