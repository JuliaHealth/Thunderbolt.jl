"""
    QuasiStaticElementCache

A generic cache to assemble elements coming from a [StructuralModel](@ref).

Right now the model has to be formulated in the first Piola Kirchhoff stress tensor and F.
"""
struct QuasiStaticElementCache{M, CCache, CMCache, CV} <: AbstractVolumetricElementCache
    # This one determines the exact material
    constitutive_model::M
    # This one is a helper to evaluate coefficients in a type stable way without allocations
    coefficient_cache::CCache
    # This one is a helper to condense local variables
    internal_cache::CMCache
    # FEValue scratch for the ansatz space
    cv::CV
end

# TODO how to control dispatch on required input for the material routin?
# TODO finer granularity on the dispatch here. depending on the evolution law of the internal variable this routine looks slightly different.
function assemble_element!(Kₑ::AbstractMatrix, residualₑ::AbstractVector, uₑ::AbstractVector, geometry_cache::CellCache, element_cache::QuasiStaticElementCache, time)
    @unpack constitutive_model, internal_cache, cv, coefficient_cache = element_cache
    ndofs = getnbasefunctions(cv)

    reinit!(cv, geometry_cache)

    @inbounds for qp ∈ QuadratureIterator(cv)
        dΩ = getdetJdV(cv, qp)

        # Compute deformation gradient F
        ∇u = function_gradient(cv, qp, uₑ)
        F = one(∇u) + ∇u

        # Compute stress and tangent
        P, ∂P∂F = material_routine(constitutive_model, F, coefficient_cache, internal_cache, geometry_cache, qp, time)

        # Loop over test functions
        for i in 1:ndofs
            ∇δui = shape_gradient(cv, qp, i)

            # Add contribution to the residual from this test function
            residualₑ[i] += ∇δui ⊡ P * dΩ

            ∇δui∂P∂F = ∇δui ⊡ ∂P∂F # Hoisted computation
            for j in 1:ndofs
                ∇δuj = shape_gradient(cv, qp, j)
                # Add contribution to the tangent
                Kₑ[i, j] += ( ∇δui∂P∂F ⊡ ∇δuj ) * dΩ
            end
        end
    end
end

function assemble_element!(Kₑ::AbstractMatrix, uₑ::AbstractVector, geometry_cache::CellCache, element_cache::QuasiStaticElementCache, time)
    @unpack constitutive_model, internal_cache, cv, coefficient_cache = element_cache
    ndofs = getnbasefunctions(cv)

    reinit!(cv, geometry_cache)

    @inbounds for qp ∈ QuadratureIterator(cv)
        dΩ = getdetJdV(cv, qp)

        # Compute deformation gradient F
        ∇u = function_gradient(cv, qp, uₑ)
        F = one(∇u) + ∇u

        # Compute stress and tangent
        P, ∂P∂F = material_routine(constitutive_model, F, coefficient_cache, internal_cache, geometry_cache, qp, time)

        # Loop over test functions
        for i in 1:ndofs
            ∇δui = shape_gradient(cv, qp, i)

            # Add contribution to the residual from this test function
            # residualₑ[i] += ∇δui ⊡ P * dΩ

            ∇δui∂P∂F = ∇δui ⊡ ∂P∂F # Hoisted computation
            for j in 1:ndofs
                ∇δuj = shape_gradient(cv, qp, j)
                # Add contribution to the tangent
                Kₑ[i, j] += ( ∇δui∂P∂F ⊡ ∇δuj ) * dΩ
            end
        end
    end
end

function assemble_element!(residualₑ::AbstractVector, uₑ::AbstractVector, geometry_cache::CellCache, element_cache::QuasiStaticElementCache, time)
    @unpack constitutive_model, internal_cache, cv, coefficient_cache = element_cache
    ndofs = getnbasefunctions(cv)

    reinit!(cv, geometry_cache)

    @inbounds for qp ∈ QuadratureIterator(cv)
        dΩ = getdetJdV(cv, qp)

        # Compute deformation gradient F
        ∇u = function_gradient(cv, qp, uₑ)
        F = one(∇u) + ∇u

        # Compute stress and tangent
        P, ∂P∂F = material_routine(constitutive_model, F, coefficient_cache, internal_cache, geometry_cache, qp, time)

        # Loop over test functions
        for i in 1:ndofs
            ∇δui = shape_gradient(cv, qp, i)

            # Add contribution to the residual from this test function
            residualₑ[i] += ∇δui ⊡ P * dΩ

            # ∇δui∂P∂F = ∇δui ⊡ ∂P∂F # Hoisted computation
            # for j in 1:ndofs
            #     ∇δuj = shape_gradient(cv, qp, j)
            #     # Add contribution to the tangent
            #     Kₑ[i, j] += ( ∇δui∂P∂F ⊡ ∇δuj ) * dΩ
            # end
        end
    end
end

function setup_element_cache(model::QuasiStaticModel, qr::QuadratureRule, sdh::SubDofHandler)
    @assert length(sdh.dh.field_names) == 1 "Support for multiple fields not yet implemented."
    field_name = first(sdh.dh.field_names)
    ip          = Ferrite.getfieldinterpolation(sdh, field_name)
    ip_geo = geometric_subdomain_interpolation(sdh)
    cv = CellValues(qr, ip, ip_geo)
    return QuasiStaticElementCache(
        model.material_model,
        setup_coefficient_cache(model.material_model, qr, sdh),
        setup_internal_cache(model.material_model, qr, sdh),
        cv
    )
end
