@doc raw"""
    BilinearDiffusionIntegrator{CoefficientType}

Represents the integrand of the bilinear form ``a(u,v) = -\int \nabla v(x) \cdot D(x) \nabla u(x) dx`` for a given diffusion tensor ``D(x)`` and ``u,v`` from the same function space.
"""
struct BilinearDiffusionIntegrator{CoefficientType, QRC <: QuadratureRuleCollection} <:
       AbstractBilinearIntegrator
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

function duplicate_for_device(device, cache::BilinearDiffusionElementCache)
    return BilinearDiffusionElementCache(
        duplicate_for_device(device, cache.Dcache),
        duplicate_for_device(device, cache.cellvalues),
    )
end

function assemble_element!(
    Kₑ::AbstractMatrix,
    cell,
    element_cache::BilinearDiffusionElementCache,
    time,
)
    @unpack cellvalues, Dcache = element_cache
    n_basefuncs = getnbasefunctions(cellvalues)

    reinit!(cellvalues, cell)

    for qp in QuadratureIterator(cellvalues)
        D_loc = evaluate_coefficient(Dcache, cell, qp, time)
        dΩ = getdetJdV(cellvalues, qp)
        for i = 1:n_basefuncs
            ∇Nᵢ = shape_gradient(cellvalues, qp, i)
            for j = 1:n_basefuncs
                ∇Nⱼ = shape_gradient(cellvalues, qp, j)
                Kₑ[i, j] -= _inner_product_helper(∇Nⱼ, D_loc, ∇Nᵢ) * dΩ
            end
        end
    end
end

function setup_element_cache(element_model::BilinearDiffusionIntegrator, sdh::SubDofHandler)
    qr         = getquadraturerule(element_model.qrc, sdh)
    ip         = Ferrite.getfieldinterpolation(sdh, element_model.sym)
    ip_geo     = geometric_subdomain_interpolation(sdh)
    BilinearDiffusionElementCache(
        setup_coefficient_cache(element_model.D, qr, sdh),
        CellValues(qr, ip, ip_geo),
    )
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

get_volumetric_weak_form_names(model::TransientDiffusionModel) = [model.solution_variable_symbol] # FIXME

@doc raw"""
    BilinearDiffusionIntegrator{CoefficientType}

Represents the integrand of the bilinear form ``a(u,v) = -\int \nabla v(x) \cdot D(x) \nabla u(x) dx`` for a given diffusion tensor ``D(x)`` and ``u,v`` from the same function space.
"""
struct BilinearInterfaceDiffusionIntegrator{CoefficientType, QRC <: QuadratureRuleCollection} <:
       AbstractBilinearIntegrator
    D::CoefficientType
    qrc::QRC
    sym1::Symbol
    sym2::Symbol
end

"""
The cache associated with [`BilinearDiffusionIntegrator`](@ref) to assemble element diffusion matrices.
"""
struct BilinearInterfaceDiffusionElementCache{CoefficientCacheType, CV} <: AbstractVolumetricElementCache
    Dcache::CoefficientCacheType
    cellvalues::CV
end

function duplicate_for_device(device, cache::BilinearInterfaceDiffusionElementCache)
    return BilinearInterfaceDiffusionElementCache(
        duplicate_for_device(device, cache.Dcache),
        duplicate_for_device(device, cache.cellvalues),
    )
end

function assemble_element!(
    Kₑ::AbstractMatrix,
    cell,
    element_cache::BilinearInterfaceDiffusionElementCache,
    time,
)
    (; cellvalues, Dcache) = element_cache
    n_basefuncs = getnbasefunctions(cellvalues)

    reinit!(cellvalues, cell)

    for qp in 1:getnquadpoints(cellvalues)
        D_loc = evaluate_coefficient(Dcache, cell, qp, time)
        dΩ = getdetJdV_average(cellvalues, qp)
        for i in 1:getnbasefunctions(cellvalues)
            jump_δu = shape_value_jump(cellvalues, qp, i)
            for j in 1:getnbasefunctions(cellvalues)
                jump_u = shape_value_jump(cellvalues, qp, j)
                Kₑ[i, j] += (jump_δu * D_loc * jump_u) * dΩ
            end
        end
    end
end

function setup_element_cache(element_model::BilinearInterfaceDiffusionIntegrator, sdh::SubDofHandler)
    qr = getquadraturerule(element_model.qrc, sdh)
    ip = Ferrite.getfieldinterpolation(sdh, element_model.sym1)
    cv = InterfaceCellValues(qr, ip)
    return BilinearInterfaceDiffusionElementCache(
        setup_coefficient_cache(element_model.D, qr, sdh),
        cv,
    )
end


@doc raw"""
    TransientDiffusionModel(conductivity_coefficient, source_term, solution_variable_symbol)

Model formulated as ``\int_{\Gamma^{\text{P}/\text{M}}} [\![ \delta u ]\!] G [\![ u ]\!] \mathrm{d}\Gamma``.
"""
@concrete struct InterfaceDiffusionModel
    G
    solution_variable_symbol::Symbol
    interface_interpolation_symbol::Symbol
end

get_volumetric_weak_form_names(model::InterfaceDiffusionModel) = [model.solution_variable_symbol] # FIXME

@doc raw"""
    SteadyDiffusionModel(conductivity_coefficient, source_term, solution_variable_symbol)

Model formulated as ``\nabla \cdot \kappa(x) \nabla u = f``
"""
struct SteadyDiffusionModel{ConductivityCoefficientType, SourceType <: AbstractSourceTerm}
    κ::ConductivityCoefficientType
    source::SourceType
    solution_variable_symbol::Symbol
end

get_volumetric_weak_form_names(model::SteadyDiffusionModel) = [model.solution_variable_symbol] # FIXME
