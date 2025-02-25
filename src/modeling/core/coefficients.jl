"""
    FieldCoefficient(data, interpolation)

A constant in time data field, interpolated per element with a given interpolation.
"""
struct FieldCoefficient{T,TA<:AbstractArray{T,2},IPC<:InterpolationCollection}
    # TODO use DenseDataRange
    elementwise_data::TA #2d ragged array (element_idx, base_fun_idx)
    ip_collection::IPC
end

struct FieldCoefficientCache{T, TA <: AbstractArray{T, 2}, CV}
    elementwise_data::TA
    cv::CV
end

duplicate_for_parallel(cache::FieldCoefficientCache) = cache

@inline function setup_coefficient_cache(coefficient::FieldCoefficient, qr::QuadratureRule, sdh::SubDofHandler)
    return _create_field_coefficient_cache(coefficient, coefficient.ip_collection, qr, sdh)
end

function _create_field_coefficient_cache(coefficient::FieldCoefficient{T}, ipc::ScalarInterpolationCollection, qr::QuadratureRule, sdh::SubDofHandler) where T
    cell = get_first_cell(sdh)
    ip     = getinterpolation(coefficient.ip_collection, cell)
    fv     = Ferrite.FunctionValues{0}(T, ip, qr, ip^3)
    Nξs    = size(fv.Nξ)
    return FieldCoefficientCache(coefficient.elementwise_data, FerriteUtils.StaticInterpolationValues(fv.ip, SMatrix{Nξs[1], Nξs[2]}(fv.Nξ), nothing))
end

function _create_field_coefficient_cache(coefficient::FieldCoefficient{<:Vec{<:Any, T}}, ipc::VectorizedInterpolationCollection, qr::QuadratureRule, sdh::SubDofHandler) where T
    cell = get_first_cell(sdh)
    ip     = getinterpolation(coefficient.ip_collection, cell)
    fv     = Ferrite.FunctionValues{0}(T, ip.ip, qr, ip)
    Nξs    = size(fv.Nξ)
    return FieldCoefficientCache(coefficient.elementwise_data, FerriteUtils.StaticInterpolationValues(fv.ip, SMatrix{Nξs[1], Nξs[2]}(fv.Nξ), nothing))
end

function evaluate_coefficient(cache::FieldCoefficientCache{T}, geometry_cache::FerriteUtils.AnyCellCache, qp::QuadraturePoint, t) where T
    @unpack elementwise_data, cv = cache
    val = zero(T)
    cellidx = cellid(geometry_cache)

    @inbounds for i in 1:getnbasefunctions(cv)
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

duplicate_for_parallel(cache::ConstantCoefficient) = cache

function setup_coefficient_cache(coefficient::ConstantCoefficient, qr::QuadratureRule, sdh::SubDofHandler)
    return coefficient
end

evaluate_coefficient(coeff::ConstantCoefficient, ::FerriteUtils.AnyCellCache, qp, t) = coeff.val


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

function setup_coefficient_cache(coefficient::ConductivityToDiffusivityCoefficient, qr::QuadratureRule, sdh::SubDofHandler)
    return ConductivityToDiffusivityCoefficientCache(
        setup_coefficient_cache(coefficient.conductivity_tensor_coefficient, qr, sdh),
        setup_coefficient_cache(coefficient.capacitance_coefficient, qr, sdh),
        setup_coefficient_cache(coefficient.χ_coefficient, qr, sdh),
    )
end

function evaluate_coefficient(coeff::ConductivityToDiffusivityCoefficientCache, cell_cache::FerriteUtils.AnyCellCache, qp::QuadraturePoint, t)
    κ  = evaluate_coefficient(coeff.conductivity_tensor_cache, cell_cache, qp, t)
    Cₘ = evaluate_coefficient(coeff.capacitance_cache, cell_cache, qp, t)
    χ  = evaluate_coefficient(coeff.χ_cache, cell_cache, qp, t)
    return κ/(Cₘ*χ)
end

function duplicate_for_parallel(cache::ConductivityToDiffusivityCoefficientCache)
    ConductivityToDiffusivityCoefficientCache(
        duplicate_for_parallel(cache.conductivity_tensor_cache),
        duplicate_for_parallel(cache.capacitance_cache),
        duplicate_for_parallel(cache.χ_cache),
    )
end

"""
    CoordinateSystemCoefficient(coordinate_system)

Helper to obtain the location in some possibly problem-specific coordinate system, e.g. for analytical coefficients (see [`AnalyticalCoefficient`](@ref)).
"""
struct CoordinateSystemCoefficient{CS}
    cs::CS
end

function compute_nodal_values(csc::CoordinateSystemCoefficient, dh::DofHandler, field_name::Symbol)
    Tv = value_type(csc.cs)
    nodal_values = Vector{Tv}(UndefInitializer(), ndofs(dh))
    T = eltype(Tv)
    for sdh in dh.subdofhandlers
        field_name ∈ sdh.field_names || continue
        ip   = Ferrite.getfieldinterpolation(sdh, field_name)
        rdim = Ferrite.getrefdim(ip)
        positions = Vec{rdim,T}.(Ferrite.reference_coordinates(ip))
        # This little trick uses the delta property of interpolations
        qr = QuadratureRule{Ferrite.getrefshape(ip)}([T(1.0) for _ in 1:length(positions)], positions)
        cc = setup_coefficient_cache(csc, qr, sdh)
        _compute_nodal_values!(nodal_values, qr, cc, sdh)
    end
    return nodal_values
end

function _compute_nodal_values!(nodal_values, qr, cc, sdh)
    for cell in CellIterator(sdh)
        dofs = celldofs(cell)
        for qp in QuadratureIterator(qr)
            nodal_values[dofs[qp.i]] = evaluate_coefficient(cc, cell, qp, NaN)
        end
    end
end

struct CartesianCoordinateSystemCache{CS, CV}
    cs::CS
    cv::CV
end

duplicate_for_parallel(cache::CartesianCoordinateSystemCache) = cache

function setup_coefficient_cache(coefficient::CoordinateSystemCoefficient{<:CartesianCoordinateSystem}, qr::QuadratureRule{<:Any, <:AbstractArray{T}}, sdh::SubDofHandler) where T
    cell = get_first_cell(sdh)
    ip = getcoordinateinterpolation(coefficient.cs, cell)
    fv = Ferrite.FunctionValues{0}(T, ip.ip, qr, ip) # We scalarize the interpolation again as an optimization step
    Nξs    = size(fv.Nξ)
    return CartesianCoordinateSystemCache(coefficient.cs, FerriteUtils.StaticInterpolationValues(fv.ip, SMatrix{Nξs[1], Nξs[2]}(fv.Nξ), nothing))
end

function evaluate_coefficient(coeff::CartesianCoordinateSystemCache{<:CartesianCoordinateSystem{sdim}}, geometry_cache::FerriteUtils.AnyCellCache, qp::QuadraturePoint{<:Any,T}, t) where {sdim, T}
    @unpack cv = coeff
    x          = zero(Vec{sdim, T})
    coords     = getcoordinates(geometry_cache) 
    for i in 1:getnbasefunctions(cv)
        x += shape_value(cv, qp, i) * coords[i]
    end
    return x
end

struct LVCoordinateSystemCache{CS <: LVCoordinateSystem, CV}
    cs::CS
    cv::CV
end

duplicate_for_parallel(cache::LVCoordinateSystemCache) = cache

function setup_coefficient_cache(coefficient::CoordinateSystemCoefficient{<:LVCoordinateSystem}, qr::QuadratureRule{<:Any, <:AbstractArray{T}}, sdh::SubDofHandler) where T
    cell = get_first_cell(sdh)
    ip = getcoordinateinterpolation(coefficient.cs, cell)
    ip_geo = ip^3
    fv     = Ferrite.FunctionValues{0}(T, ip, qr, ip_geo)
    Nξs    = size(fv.Nξ)
    return LVCoordinateSystemCache(coefficient.cs, FerriteUtils.StaticInterpolationValues(fv.ip, SMatrix{Nξs[1], Nξs[2]}(fv.Nξ), nothing))
end

function evaluate_coefficient(coeff::LVCoordinateSystemCache, geometry_cache::FerriteUtils.AnyCellCache, qp::QuadraturePoint{ref_shape,T}, t) where {ref_shape,T}
    @unpack cv, cs = coeff
    @unpack dh     = cs
    x1 = zero(T)
    x2 = zero(T)
    x3 = zero(T)
    dofs = celldofsview(dh, cellid(geometry_cache))
    @inbounds for i in 1:getnbasefunctions(cv)
        val = shape_value(cv, qp, i)::T
        x1 += val * cs.u_transmural[dofs[i]]
        x2 += val * cs.u_apicobasal[dofs[i]]
        x3 += val * cs.u_rotational[dofs[i]]
    end
    return LVCoordinate(x1, x2, x3)
end

struct BiVCoordinateSystemCache{CS <: BiVCoordinateSystem, CV}
    cs::CS
    cv::CV
end

duplicate_for_parallel(cache::BiVCoordinateSystemCache) = cache

function setup_coefficient_cache(coefficient::CoordinateSystemCoefficient{<:BiVCoordinateSystem}, qr::QuadratureRule{<:Any,<:AbstractArray{T}}, sdh::SubDofHandler) where T
    cell = get_first_cell(sdh)
    ip = getcoordinateinterpolation(coefficient.cs, cell)
    ip_geo = ip^3
    fv     = Ferrite.FunctionValues{0}(T, ip, qr, ip_geo)
    Nξs    = size(fv.Nξ)
    return BiVCoordinateSystemCache(coefficient.cs, FerriteUtils.StaticInterpolationValues(fv.ip, SMatrix{Nξs[1], Nξs[2]}(fv.Nξ), nothing))
end

function evaluate_coefficient(cc::BiVCoordinateSystemCache, cell_cache::FerriteUtils.AnyCellCache, qp::QuadraturePoint{<:Any,T}, t) where {T}
    @unpack cv, cs = cc
    @unpack dh     = cs
    dofs = celldofsview(dh, cellid(cell_cache))
    x1 = zero(T)
    x2 = zero(T)
    x3 = zero(T)
    x4 = zero(T)
    @inbounds for i in 1:getnbasefunctions(cv)
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

function duplicate_for_parallel(cache::SpectralTensorCoefficientCache)
    SpectralTensorCoefficientCache(
        duplicate_for_parallel(cache.eigenvectors),
        duplicate_for_parallel(cache.eigenvalues),
    )
end

function setup_coefficient_cache(coefficient::SpectralTensorCoefficient, qr::QuadratureRule, sdh::SubDofHandler)
    return SpectralTensorCoefficientCache(
        setup_coefficient_cache(coefficient.eigenvectors, qr, sdh),
        setup_coefficient_cache(coefficient.eigenvalues, qr, sdh),
    )
end

function evaluate_coefficient(coeff::SpectralTensorCoefficientCache, cell_cache::FerriteUtils.AnyCellCache, qp::QuadraturePoint, t)
    M = evaluate_coefficient(coeff.eigenvector_cache, cell_cache, qp, t)
    λ = evaluate_coefficient(coeff.eigenvalue_cache, cell_cache, qp, t)
    return _eval_st_coefficient(M, λ) # Dispatches can be found e.g. in modeling/microstructure.jl
end

@inline _eval_st_coefficient(M, λ) = error("Spectral tensor evaluation not implemented for M=$(typeof(M)) and λ=$(typeof(λ)). Please provide a dispatch for _eval_st_coefficient(M, λ).")

"""
    SpatiallyHomogeneousDataField(timings::Vector, data::Vector)

A data field which is constant in space and piecewise constant in time.

The value during the time interval [tᵢ,tᵢ₊₁] is dataᵢ, where t₀ is negative infinity and the last time point+1 is positive infinity.
"""
struct SpatiallyHomogeneousDataField{T, TD <: AbstractVector{T}, TV <: AbstractVector}
    timings::TV
    data::TD
end

duplicate_for_parallel(cache::SpatiallyHomogeneousDataField) = cache

function setup_coefficient_cache(coefficient::SpatiallyHomogeneousDataField, qr::QuadratureRule, sdh::SubDofHandler)
    return coefficient
end

Thunderbolt.evaluate_coefficient(coeff::SpatiallyHomogeneousDataField, ::FerriteUtils.AnyCellCache, ::QuadraturePoint, t) = _evaluate_coefficient(coeff, t)
  
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
