"""
    FieldCoefficient(data, interpolation)

A constant in time data field, interpolated per element with a given interpolation.
"""
struct FieldCoefficient{T, TA <: AbstractArray{T, 2}, IPC <: InterpolationCollection}
    # TODO use DenseDataRange
    elementwise_data::TA #2d ragged array (element_idx, base_fun_idx)
    ip_collection::IPC
end

struct FieldCoefficientCache{T, TA <: AbstractArray{T, 2}, CV}
    elementwise_data::TA
    cv::CV
end

duplicate_for_device(device, cache::FieldCoefficientCache) = cache

@inline function setup_coefficient_cache(
    coefficient::FieldCoefficient,
    qr::QuadratureRule,
    sdh::SubDofHandler,
)
    return _create_field_coefficient_cache(coefficient, coefficient.ip_collection, qr, sdh)
end

function _create_field_coefficient_cache(
    coefficient::FieldCoefficient{T},
    ipc::ScalarInterpolationCollection,
    qr::QuadratureRule,
    sdh::SubDofHandler,
) where {T}
    cell = get_first_cell(sdh)
    ip = getinterpolation(coefficient.ip_collection, cell)
    fv = Ferrite.FunctionValues{0}(T, ip, qr, ip^3)
    Nξs = size(fv.Nξ)
    return FieldCoefficientCache(
        coefficient.elementwise_data,
        FerriteUtils.StaticInterpolationValues(fv.ip, SMatrix{Nξs[1], Nξs[2]}(fv.Nξ), nothing),
    )
end

function _create_field_coefficient_cache(
    coefficient::FieldCoefficient{<:Vec{<:Any, T}},
    ipc::VectorizedInterpolationCollection,
    qr::QuadratureRule,
    sdh::SubDofHandler,
) where {T}
    cell = get_first_cell(sdh)
    ip = getinterpolation(coefficient.ip_collection, cell)
    fv = Ferrite.FunctionValues{0}(T, ip.ip, qr, ip)
    Nξs = size(fv.Nξ)
    return FieldCoefficientCache(
        coefficient.elementwise_data,
        FerriteUtils.StaticInterpolationValues(fv.ip, SMatrix{Nξs[1], Nξs[2]}(fv.Nξ), nothing),
    )
end

function evaluate_coefficient(
    cache::FieldCoefficientCache{T},
    geometry_cache::CellCache,
    qp::QuadraturePoint,
    t,
) where {T}
    @unpack elementwise_data, cv = cache
    val = zero(T)
    cellidx = cellid(geometry_cache)

    @inbounds for i = 1:getnbasefunctions(cv)
        val += shape_value(cv, qp, i) * elementwise_data[i, cellidx]
    end
    return val
end

"""
    ConstantCoefficient(value)

Evaluates to the same value in space and time everywhere.
"""
struct ConstantCoefficient{T}
    val::T
end

duplicate_for_device(device, cache::ConstantCoefficient) = cache

function setup_coefficient_cache(
    coefficient::ConstantCoefficient,
    qr::QuadratureRule,
    sdh::SubDofHandler,
)
    return coefficient
end

evaluate_coefficient(coeff::ConstantCoefficient, ::CellCache, qp, t) = coeff.val


"""
    ConductivityToDiffusivityCoefficient(conductivity_tensor_coefficient, capacitance_coefficient, χ_coefficient)

Internal helper for ep problems.
"""
struct ConductivityToDiffusivityCoefficient{DTC, CC, STVC}
    conductivity_tensor_coefficient::DTC
    capacitance_coefficient::CC
    χ_coefficient::STVC
end

struct ConductivityToDiffusivityCoefficientCache{DTC, CC, STVC}
    conductivity_tensor_cache::DTC
    capacitance_cache::CC
    χ_cache::STVC
end

function setup_coefficient_cache(
    coefficient::ConductivityToDiffusivityCoefficient,
    qr::QuadratureRule,
    sdh::SubDofHandler,
)
    return ConductivityToDiffusivityCoefficientCache(
        setup_coefficient_cache(coefficient.conductivity_tensor_coefficient, qr, sdh),
        setup_coefficient_cache(coefficient.capacitance_coefficient, qr, sdh),
        setup_coefficient_cache(coefficient.χ_coefficient, qr, sdh),
    )
end

function evaluate_coefficient(
    coeff::ConductivityToDiffusivityCoefficientCache,
    cell_cache::CellCache,
    qp::QuadraturePoint,
    t,
)
    κ = evaluate_coefficient(coeff.conductivity_tensor_cache, cell_cache, qp, t)
    Cₘ = evaluate_coefficient(coeff.capacitance_cache, cell_cache, qp, t)
    χ = evaluate_coefficient(coeff.χ_cache, cell_cache, qp, t)
    return κ/(Cₘ*χ)
end

function duplicate_for_device(device, cache::ConductivityToDiffusivityCoefficientCache)
    ConductivityToDiffusivityCoefficientCache(
        duplicate_for_device(device, cache.conductivity_tensor_cache),
        duplicate_for_device(device, cache.capacitance_cache),
        duplicate_for_device(device, cache.χ_cache),
    )
end

"""
    evaluate_coefficient_at_dof_locations(coefficient, dh, field_name; cellset = nothing)
    evaluate_coefficient_at_dof_locations!(a, coefficient, dh, field_name; cellset = nothing)

Evaluate `coefficient` at the spatial locations `field_name`'s degrees of freedom sit at, returning (or
filling) a vector indexed by dof of `dh`.

**These are the nodes of the ansatz space, not the nodes of the mesh.** The locations come from the
*field* interpolation's reference coordinates mapped through the geometric interpolation, so a quadratic
ansatz on a linear mesh also evaluates at edge midpoints and cell centres — 25 locations on a 9-node
`Quadrilateral` patch, not 9. The two coincide only for a first-order Lagrange field on a matching
geometry, which is why the name says "dof locations" rather than "nodal".

Only meaningful for interpolations with the delta property — Lagrange and friends, where a dof's value
*is* the function value at its location. That is the same restriction `Ferrite.apply_analytical!`
documents, and it is why the quadrature rule below can use the reference coordinates as its points.

Works for any coefficient, not just coordinate systems: the evaluation goes through the ordinary
`setup_coefficient_cache` / `evaluate_coefficient` protocol, and only the choice of quadrature points is
special. The allocating form additionally needs `value_type(coefficient)` to size its output, which today
only the coordinate systems implement — pass your own vector to the in-place form otherwise.

`cellset` restricts the evaluation to the `SubDofHandler`s living on those cells. That matters on a mixed
grid: an interface `SubDofHandler` may carry the same field while its interpolation has no reference
coordinates, so evaluating there is neither meaningful nor possible. Entries outside the set are left
untouched.
"""
evaluate_coefficient_at_dof_locations(
    coefficient,
    dh::DofHandler,
    field_name::Symbol;
    cellset = nothing,
) = evaluate_coefficient_at_dof_locations!(
    Vector{value_type(coefficient)}(UndefInitializer(), ndofs(dh)),
    coefficient,
    dh,
    field_name;
    cellset,
)

@doc (@doc evaluate_coefficient_at_dof_locations)
function evaluate_coefficient_at_dof_locations!(
    a::AbstractVector,
    coefficient,
    dh::DofHandler,
    field_name::Symbol;
    cellset = nothing,
)
    for sdh in dh.subdofhandlers
        field_name ∈ sdh.field_names || continue
        cellset === nothing || first(sdh.cellset) ∈ cellset || continue
        ip = Ferrite.getfieldinterpolation(sdh, field_name)
        rdim = Ferrite.getrefdim(ip)
        # The positions live in reference space, so their element type comes from the interpolation --
        # not from whatever the coefficient evaluates to. Tying the two together breaks for any
        # coefficient whose values are not floats, e.g. a cell index.
        positions = Vec{rdim}.(Ferrite.reference_coordinates(ip))
        T = eltype(eltype(positions))
        #! format: off
        # This little trick uses the delta property of interpolations
        qr = QuadratureRule{Ferrite.getrefshape(ip)}([T(1.0) for _ = 1:length(positions)], positions)
        #! format: on
        cc = setup_coefficient_cache(coefficient, qr, sdh)
        # A field need not occupy the first dofs of a cell, so address it through its own dof range.
        drange = Ferrite.dof_range(sdh, field_name)
        for cell in CellIterator(sdh)
            dofs = @view celldofs(cell)[drange]
            for qp in QuadratureIterator(qr)
                a[dofs[qp.i]] = evaluate_coefficient(cc, cell, qp, NaN)
            end
        end
    end
    return a
end

struct CellIndexCoordinateSystemCache end

duplicate_for_device(device, cache::CellIndexCoordinateSystemCache) = cache

setup_coefficient_cache(::CellIndexCoordinateSystem, ::QuadratureRule, ::SubDofHandler) =
    CellIndexCoordinateSystemCache()

evaluate_coefficient(
    ::CellIndexCoordinateSystemCache,
    geometry_cache::CellCache,
    ::QuadraturePoint,
    t,
) = cellid(geometry_cache)

struct CartesianCoordinateSystemCache{CS, CV}
    cs::CS
    cv::CV
end

duplicate_for_device(device, cache::CartesianCoordinateSystemCache) = cache

function setup_coefficient_cache(
    cs::CartesianCoordinateSystem,
    qr::QuadratureRule{<:Any, <:AbstractArray{T}},
    sdh::SubDofHandler,
) where {T}
    cell = get_first_cell(sdh)
    ip = getcoordinateinterpolation(cs, cell)
    fv = Ferrite.FunctionValues{0}(T, ip.ip, qr, ip) # We scalarize the interpolation again as an optimization step
    Nξs = size(fv.Nξ)
    return CartesianCoordinateSystemCache(
        cs,
        FerriteUtils.StaticInterpolationValues(fv.ip, SMatrix{Nξs[1], Nξs[2]}(fv.Nξ), nothing),
    )
end

function evaluate_coefficient(
    coeff::CartesianCoordinateSystemCache{<:CartesianCoordinateSystem{sdim}},
    geometry_cache::CellCache,
    qp::QuadraturePoint{<:Any, T},
    t,
) where {sdim, T}
    @unpack cv = coeff
    x          = zero(Vec{sdim, T})
    coords     = getcoordinates(geometry_cache)
    for i = 1:getnbasefunctions(cv)
        x += shape_value(cv, qp, i) * coords[i]
    end
    return x
end

struct LVCoordinateSystemCache{CS <: LVCoordinateSystem, CV, CVR}
    cs::CS
    cv::CV
    cv_rotational::CVR
end

duplicate_for_device(device, cache::LVCoordinateSystemCache) = cache

function setup_coefficient_cache(
    cs::LVCoordinateSystem,
    qr::QuadratureRule{<:Any, <:AbstractArray{T}},
    sdh::SubDofHandler,
) where {T}
    cell   = get_first_cell(sdh)
    ip = Thunderbolt.getcoordinateinterpolation(cs, cell)
    ip_geo = getcoordinateinterpolation(cs, cell)^3
    Nξs    = size(fv.Nξ)
    return Thunderbolt.LVCoordinateSystemCache(cs, CellValues(qr, ip, ip_geo), CellValues(qr, ip, ip_geo))
end

function evaluate_coefficient(
    coeff::LVCoordinateSystemCache,
    geometry_cache::CellCache,
    qp::QuadraturePoint{ref_shape, T},
    t,
) where {ref_shape, T}
    @unpack cv, cv_rotational, cs = coeff
    x1             = zero(T)
    x2             = zero(T)
    x3             = zero(T)
    dofs           = celldofsview(cs.dh, cellid(geometry_cache))
    @inbounds for i = 1:getnbasefunctions(cv)
        val = shape_value(cv, qp, i)::T
        x1 += val * cs.u_transmural[dofs[i]]
        x2 += val * cs.u_apicobasal[dofs[i]]
    end
    # The rotational coordinate lives on its own discontinuous dofs, which is what keeps the
    # interpolant affine across the seam; wrapping brings the result back into [0, 1).
    dofs_rotational = celldofsview(cs.dh_rotational, cellid(geometry_cache))
    @inbounds for i = 1:getnbasefunctions(cv_rotational)
        x3 += shape_value(cv_rotational, qp, i)::T * cs.u_rotational[dofs_rotational[i]]
    end
    return LVCoordinate(x1, x2, wrap_rotational(x3))
end

struct BiVCoordinateSystemCache{CS <: BiVCoordinateSystem, CV}
    cs::CS
    cv::CV
end

duplicate_for_device(device, cache::BiVCoordinateSystemCache) = cache

function setup_coefficient_cache(
    cs::BiVCoordinateSystem,
    qr::QuadratureRule{<:Any, <:AbstractArray{T}},
    sdh::SubDofHandler,
) where {T}
    cell   = get_first_cell(sdh)
    ip     = getcoordinateinterpolation(cs, cell)
    ip_geo = ip^3
    fv     = Ferrite.FunctionValues{0}(T, ip, qr, ip_geo)
    Nξs    = size(fv.Nξ)
    return BiVCoordinateSystemCache(
        cs,
        FerriteUtils.StaticInterpolationValues(fv.ip, SMatrix{Nξs[1], Nξs[2]}(fv.Nξ), nothing),
    )
end

function evaluate_coefficient(
    cc::BiVCoordinateSystemCache,
    cell_cache::CellCache,
    qp::QuadraturePoint{<:Any, T},
    t,
) where {T}
    @unpack cv, cs = cc
    @unpack dh     = cs
    dofs           = celldofsview(dh, cellid(cell_cache))
    x1             = zero(T)
    x2             = zero(T)
    x3             = zero(T)
    x4             = zero(T)
    @inbounds for i = 1:getnbasefunctions(cv)
        val = shape_value(cv, qp, i)::T
        x1 += val * cs.u_transmural[dofs[i]]
        x2 += val * cs.u_apicobasal[dofs[i]]
        x3 += val * cs.u_rotational[dofs[i]]
        x4 += val * cs.u_transventricular[dofs[i]]
    end
    return BiVCoordinate(x1, x2, x3, x4)
end

"""
    SpectralTensorCoefficient(eigenvector_coefficient, eigenvalue_coefficient)

Represent a tensor A via spectral decomposition ∑ᵢ λᵢ vᵢ ⊗ vᵢ.
"""
struct SpectralTensorCoefficient{MSC, TC}
    eigenvectors::MSC
    eigenvalues::TC
end

struct SpectralTensorCoefficientCache{C1, C2}
    eigenvector_cache::C1
    eigenvalue_cache::C2
end

function duplicate_for_device(device, cache::SpectralTensorCoefficientCache)
    SpectralTensorCoefficientCache(
        duplicate_for_device(device, cache.eigenvector_cache),
        duplicate_for_device(device, cache.eigenvalue_cache),
    )
end

function setup_coefficient_cache(
    coefficient::SpectralTensorCoefficient,
    qr::QuadratureRule,
    sdh::SubDofHandler,
)
    return SpectralTensorCoefficientCache(
        setup_coefficient_cache(coefficient.eigenvectors, qr, sdh),
        setup_coefficient_cache(coefficient.eigenvalues, qr, sdh),
    )
end

function evaluate_coefficient(
    coeff::SpectralTensorCoefficientCache,
    cell_cache::CellCache,
    qp::QuadraturePoint,
    t,
)
    M = evaluate_coefficient(coeff.eigenvector_cache, cell_cache, qp, t)
    λ = evaluate_coefficient(coeff.eigenvalue_cache, cell_cache, qp, t)
    return _eval_st_coefficient(M, λ) # Dispatches can be found e.g. in modeling/microstructure.jl
end

@inline _eval_st_coefficient(M, λ) = error(
    "Spectral tensor evaluation not implemented for M=$(typeof(M)) and λ=$(typeof(λ)). Please provide a dispatch for _eval_st_coefficient(M, λ).",
)

"""
    SpatiallyHomogeneousDataField(timings::Vector, data::Vector)

A data field which is constant in space and piecewise constant in time.

The value during the time interval [tᵢ,tᵢ₊₁] is dataᵢ, where t₀ is negative infinity and the last time point+1 is positive infinity.
"""
struct SpatiallyHomogeneousDataField{T, TD <: AbstractVector{T}, TV <: AbstractVector}
    timings::TV
    data::TD
end

duplicate_for_device(device, cache::SpatiallyHomogeneousDataField) = cache

function setup_coefficient_cache(
    coefficient::SpatiallyHomogeneousDataField,
    qr::QuadratureRule,
    sdh::SubDofHandler,
)
    return coefficient
end

evaluate_coefficient(coeff::SpatiallyHomogeneousDataField, ::CellCache, ::QuadraturePoint, t) =
    _evaluate_coefficient(coeff, t)

function _evaluate_coefficient(coeff::SpatiallyHomogeneousDataField, t)
    @unpack timings, data = coeff
    i = 1
    tᵢ = timings[1]
    while tᵢ < t
        i+=1
        if i > length(timings)
            return data[end]
        end
        tᵢ = timings[i]
    end
    return data[i] # TODO interpolation
end
