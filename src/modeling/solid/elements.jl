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

        # Compute "tangent only"
        _, ∂P∂F = material_routine(constitutive_model, F, coefficient_cache, internal_cache, geometry_cache, qp, time)

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

        # Compute stress only
        P = reduced_material_routine(constitutive_model, F, coefficient_cache, internal_cache, geometry_cache, qp, time)

        # Loop over test functions
        for i in 1:ndofs
            ∇δui = shape_gradient(cv, qp, i)

            # Add contribution to the residual from this test function
            residualₑ[i] += ∇δui ⊡ P * dΩ
        end
    end
end

struct MultiMaterialModel{MaterialTuple} <: AbstractMaterialModel
    materials::MaterialTuple
    domains::Vector{OrderedSet{Int}} # These must match the subdofhandler sets and hence be disjoint
    domain_names::Vector{String}     # These must match the subdofhandler sets and hence be disjoint
    function MultiMaterialModel(materials::MaterialTuple, subdomain_names::Vector{String}, mesh::AbstractGrid) where MaterialTuple
        return new{MaterialTuple}(materials, [getcellset(mesh, subdomain_name) for subdomain_name in subdomain_names], subdomain_names)
    end
end

function setup_quasistatic_element_cache(material_model::MultiMaterialModel, qr::QuadratureRule, sdh::SubDofHandler, cv::CellValues)
    return setup_quasistatic_element_cache_multi(material_model.materials, material_model.domains, qr, sdh, cv)
end
@unroll function setup_quasistatic_element_cache_multi(materials::Tuple, domains::Vector, qr::QuadratureRule, sdh::SubDofHandler, cv::CellValues)
    idx = 1
    @unroll for material ∈ materials
        if first(domains[idx]) ∈ sdh.cellset
            return QuasiStaticElementCache(
                material,
                setup_coefficient_cache(material, qr, sdh),
                setup_internal_cache(material, qr, sdh),
                cv
            )
        end
        idx += 1
    end
    error("MultiDomainIntegrator is broken: Requested to construct an element cache for a SubDofHandler which is not associated with the integrator.")
end
function setup_quasistatic_element_cache(material_model::AbstractMaterialModel, qr::QuadratureRule, sdh::SubDofHandler, cv::CellValues)
    return QuasiStaticElementCache(
        material_model,
        setup_coefficient_cache(material_model, qr, sdh),
        setup_internal_cache(material_model, qr, sdh),
        cv
    )
end
function setup_element_cache(model::QuasiStaticModel, qr::QuadratureRule, sdh::SubDofHandler)
    @assert length(sdh.dh.field_names) == 1 "Support for multiple fields not yet implemented."
    field_name = first(sdh.dh.field_names)
    ip         = Ferrite.getfieldinterpolation(sdh, field_name)
    ip_geo     = geometric_subdomain_interpolation(sdh)
    cv         = CellValues(qr, ip, ip_geo)
    return setup_quasistatic_element_cache(model.material_model, qr, sdh, cv)
end

@unroll function setup_internal_cache_multi(materials::Tuple, domains::Vector, qr::QuadratureRule, sdh::SubDofHandler)
    idx = 1
    @unroll for material ∈ materials
        if first(domains[idx]) ∈ sdh.cellset
            return setup_internal_cache(material, qr, sdh)
        end
        idx += 1
    end
    error("MultiDomainIntegrator is broken: Requested to construct an internal cache for a SubDofHandler which is not associated with the integrator.")
end
function setup_internal_cache(model::MultiMaterialModel, qr::QuadratureRule, sdh::SubDofHandler)
    return setup_internal_cache_multi(model.materials, model.domains, qr, sdh)
end
