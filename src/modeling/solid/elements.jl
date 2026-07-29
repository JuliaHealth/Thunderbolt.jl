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

"""
    get_number_of_internal_dofs_per_element(integrator, element_cache, sdh)

Number of condensed unknowns each cell of `sdh` carries, as an iterable of `length(sdh.cellset)`.
Used by `FerriteOperators` to lay out the [`InternalVariableHandler`](@ref). Dispatches on the
element cache, since that is what determines how many condensed unknowns a cell carries.
"""
function FerriteOperators.get_number_of_internal_dofs_per_element(
    integrator,
    element_cache::QuasiStaticElementCache,
    sdh::SubDofHandler,
)
    nqp          = getnquadpoints(element_cache.cv)
    ndofs_per_qp = internal_variable_size(element_cache.constitutive_model, nothing, nothing)
    return Iterators.repeated(ndofs_per_qp*nqp, length(sdh.cellset))
end

# The condensed unknowns of a cell are appended after its finite element dofs, so the element
# unknown vector is laid out as `[fe_dofs | internal_variables]`. Sizes are computed from the cache
# rather than stored, which keeps `QuasiStaticElementCache` unchanged.
_qs_nbase(e::QuasiStaticElementCache) = getnbasefunctions(e.cv)
_qs_ninternal(e::QuasiStaticElementCache) =
    internal_variable_size(e.constitutive_model, nothing, nothing)*getnquadpoints(e.cv)

FerriteOperators.allocate_element_unknown_vector(e::QuasiStaticElementCache, _) =
    zeros(_qs_nbase(e) + _qs_ninternal(e))

function FerriteOperators.load_element_unknowns!(uₑ, u, cell, ivh, e::QuasiStaticElementCache)
    n, ni = _qs_nbase(e), _qs_ninternal(e)
    @views uₑ[1:n] .= u[celldofs(cell)]
    if ni > 0
        o = internal_variable_offset(ivh, cellid(cell))
        @views uₑ[(n+1):(n+ni)] .= u[(o+1):(o+ni)]
    end
    return nothing
end

# TODO how to control dispatch on required input for the material routin?
# TODO finer granularity on the dispatch here. depending on the evolution law of the internal variable this routine looks slightly different.
function assemble_element!(
    Kₑ::AbstractMatrix,
    residualₑ::AbstractVector,
    uₑ::AbstractVector,
    geometry_cache::CellCache,
    element_cache::QuasiStaticElementCache,
    time,
)
    @unpack constitutive_model, internal_cache, cv, coefficient_cache = element_cache
    ndofs = getnbasefunctions(cv)
    dₑ = @view uₑ[1:ndofs]

    reinit!(cv, geometry_cache)

    @inbounds for qp ∈ QuadratureIterator(cv)
        dΩ = getdetJdV(cv, qp)

        # Compute deformation gradient F
        ∇u = function_gradient(cv, qp, dₑ)
        F = one(∇u) + ∇u

        # Compute stress and tangent
        P, ∂P∂F = material_routine(
            constitutive_model,
            F,
            coefficient_cache,
            internal_cache,
            geometry_cache,
            qp,
            time,
        )

        # Loop over test functions
        for i = 1:ndofs
            ∇δui = shape_gradient(cv, qp, i)

            # Add contribution to the residual from this test function
            residualₑ[i] += ∇δui ⊡ P * dΩ

            ∇δui∂P∂F = ∇δui ⊡ ∂P∂F # Hoisted computation
            for j = 1:ndofs
                ∇δuj = shape_gradient(cv, qp, j)
                # Add contribution to the tangent
                Kₑ[i, j] += (∇δui∂P∂F ⊡ ∇δuj) * dΩ
            end
        end
    end
end

function assemble_element!(
    Kₑ::AbstractMatrix,
    uₑ::AbstractVector,
    geometry_cache::CellCache,
    element_cache::QuasiStaticElementCache,
    time,
)
    @unpack constitutive_model, internal_cache, cv, coefficient_cache = element_cache
    ndofs = getnbasefunctions(cv)
    dₑ = @view uₑ[1:ndofs]

    reinit!(cv, geometry_cache)

    @inbounds for qp ∈ QuadratureIterator(cv)
        dΩ = getdetJdV(cv, qp)

        # Compute deformation gradient F
        ∇u = function_gradient(cv, qp, dₑ)
        F = one(∇u) + ∇u

        # Compute "tangent only"
        _, ∂P∂F = material_routine(
            constitutive_model,
            F,
            coefficient_cache,
            internal_cache,
            geometry_cache,
            qp,
            time,
        )

        # Loop over test functions
        for i = 1:ndofs
            ∇δui = shape_gradient(cv, qp, i)

            # Add contribution to the residual from this test function
            # residualₑ[i] += ∇δui ⊡ P * dΩ

            ∇δui∂P∂F = ∇δui ⊡ ∂P∂F # Hoisted computation
            for j = 1:ndofs
                ∇δuj = shape_gradient(cv, qp, j)
                # Add contribution to the tangent
                Kₑ[i, j] += (∇δui∂P∂F ⊡ ∇δuj) * dΩ
            end
        end
    end
end

function assemble_element!(
    residualₑ::AbstractVector,
    uₑ::AbstractVector,
    geometry_cache::CellCache,
    element_cache::QuasiStaticElementCache,
    time,
)
    @unpack constitutive_model, internal_cache, cv, coefficient_cache = element_cache
    ndofs = getnbasefunctions(cv)
    dₑ = @view uₑ[1:ndofs]

    reinit!(cv, geometry_cache)

    @inbounds for qp ∈ QuadratureIterator(cv)
        dΩ = getdetJdV(cv, qp)

        # Compute deformation gradient F
        ∇u = function_gradient(cv, qp, dₑ)
        F = one(∇u) + ∇u

        # Compute stress only
        P = reduced_material_routine(
            constitutive_model,
            F,
            coefficient_cache,
            internal_cache,
            geometry_cache,
            qp,
            time,
        )

        # Loop over test functions
        for i = 1:ndofs
            ∇δui = shape_gradient(cv, qp, i)

            # Add contribution to the residual from this test function
            residualₑ[i] += ∇δui ⊡ P * dΩ
        end
    end
end

function setup_quasistatic_element_cache(
    material_model::AbstractMaterialModel,
    qr::QuadratureRule,
    sdh::SubDofHandler,
    cv::CellValues,
)
    return QuasiStaticElementCache(
        material_model,
        setup_coefficient_cache(material_model, qr, sdh),
        setup_internal_cache(material_model, qr, sdh),
        cv,
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

duplicate_for_device(device, model::AbstractMaterialModel) = model
function duplicate_for_device(device, cache::QuasiStaticElementCache)
    return QuasiStaticElementCache(
        duplicate_for_device(device, cache.constitutive_model),
        duplicate_for_device(device, cache.coefficient_cache),
        duplicate_for_device(device, cache.internal_cache),
        duplicate_for_device(device, cache.cv),
    )
end
