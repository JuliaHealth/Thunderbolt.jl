"""
    FieldCoefficient(data, interpolation)

A constant in time data field, interpolated per element with a given interpolation.
"""
struct FieldCoefficient{T,T2,TA<:Array{T,2},IPC<:InterpolationCollection}
    # TODO data structure for this
    elementwise_data::TA #2d ragged array (element_idx, base_fun_idx)
    ip_collection::IPC
    # TODO investigate usage of CellValueCollection
    qbuf::Vector{T2}
end

FieldCoefficient(data::Array{T,2}, ipc::ScalarInterpolationCollection) where T = FieldCoefficient(data, ipc, T[])
FieldCoefficient(data::Array{T,2}, ipc::VectorizedInterpolationCollection) where T = FieldCoefficient(data, ipc, eltype(T)[])

function evaluate_coefficient(coeff::FieldCoefficient{<:Any,<:Any,<:Any,<:ScalarInterpolationCollection}, cell_cache, qp::QuadraturePoint{<:Any, T}, t) where T
    @unpack elementwise_data, ip_collection = coeff
    ip = getinterpolation(ip_collection, getcells(cell_cache.grid, cellid(cell_cache)))
    n_base_funcs = Ferrite.getnbasefunctions(ip)
    val = zero(T)

    resize!(coeff.qbuf, n_base_funcs)
    Ferrite.shape_values!(coeff.qbuf, ip, qp.ξ)

    @inbounds for i in 1:n_base_funcs
        val += coeff.qbuf[i] * elementwise_data[i, cellid(cell_cache)]
    end
    return val
end

function evaluate_coefficient(coeff::FieldCoefficient{<:Any,<:Any,<:Any,<:VectorizedInterpolationCollection{vdim}}, cell_cache, qp::QuadraturePoint{<:Any, T}, t) where {vdim,T}
    @unpack elementwise_data, ip_collection = coeff
    ip = getinterpolation(ip_collection, getcells(cell_cache.grid, cellid(cell_cache)))
    n_base_funcs = Ferrite.getnbasefunctions(ip.ip)
    val = zero(Vec{vdim, T})

    resize!(coeff.qbuf, n_base_funcs)
    Ferrite.shape_values!(coeff.qbuf, ip.ip, qp.ξ)

    @inbounds for i in 1:n_base_funcs
        val += coeff.qbuf[i] * elementwise_data[i, cellid(cell_cache)]
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

evaluate_coefficient(coeff::ConstantCoefficient, cell_cache, qp, t) = coeff.val


"""
    ConductivityToDiffusivityCoefficient(conductivity_tensor_coefficient, capacitance_coefficient, χ_coefficient)

Internal helper for ep problems.
"""
struct ConductivityToDiffusivityCoefficient{DTC, CC, STVC}
    conductivity_tensor_coefficient::DTC
    capacitance_coefficient::CC
    χ_coefficient::STVC
end

function evaluate_coefficient(coeff::ConductivityToDiffusivityCoefficient, cell_cache, qp, t)
    κ  = evaluate_coefficient(coeff.conductivity_tensor_coefficient, cell_cache, qp, t)
    Cₘ = evaluate_coefficient(coeff.capacitance_coefficient, cell_cache, qp, t)
    χ  = evaluate_coefficient(coeff.χ_coefficient, cell_cache, qp, t)
    return κ/(Cₘ*χ)
end


"""
    CoordinateSystemCoefficient(coordinate_system)

Helper to obtain the location in some possibly problem-specific coordinate system, e.g. for analytical coefficients (see [`AnalyticalCoefficient`](@ref)).
"""
struct CoordinateSystemCoefficient{CS}
    cs::CS
end

function evaluate_coefficient(coeff::CoordinateSystemCoefficient{<:CartesianCoordinateSystem{sdim}}, cell_cache, qp::QuadraturePoint{<:Any,T}, t) where {sdim, T}
    x = zero(Vec{sdim, T})
    ip = getcoordinateinterpolation(coeff.cs, getcells(cell_cache.grid, cellid(cell_cache)))
    for i in 1:getnbasefunctions(ip.ip)
        x += Ferrite.shape_value(ip.ip, qp.ξ, i) * cell_cache.coords[i]
    end
    return x
end

function evaluate_coefficient(coeff::CoordinateSystemCoefficient{<:LVCoordinateSystem}, cell_cache, qp::QuadraturePoint{ref_shape,T}, t) where {ref_shape,T}
    ip = getcoordinateinterpolation(coeff.cs, getcells(cell_cache.grid, cellid(cell_cache)))
    dofs = celldofsview(coeff.cs.dh, cellid(cell_cache))
    return _evaluate_coefficient(coeff, ip, dofs, qp, t)
end

function _evaluate_coefficient(coeff::CoordinateSystemCoefficient{<:LVCoordinateSystem}, ip::ScalarInterpolation, dofs, qp::QuadraturePoint{ref_shape,T}, t) where {ref_shape,T}
    x = @MVector zeros(T, 3)
    @inbounds for i in 1:getnbasefunctions(ip)
        val = Ferrite.shape_value(ip, qp.ξ, i)::Float64
        x[1] += val * coeff.cs.u_transmural[dofs[i]]
        x[2] += val * coeff.cs.u_apicobasal[dofs[i]]
        x[3] += val * coeff.cs.u_circumferential[dofs[i]]
    end
    return LVCoordinate(x[1], x[2], x[3])
end

function evaluate_coefficient(coeff::CoordinateSystemCoefficient{<:BiVCoordinateSystem}, cell_cache, qp::QuadraturePoint{ref_shape,T}, t) where {ref_shape,T}
    ip = getcoordinateinterpolation(coeff.cs, getcells(cell_cache.grid, cellid(cell_cache)))
    dofs = celldofsview(coeff.cs.dh, cellid(cell_cache))
    return _evaluate_coefficient(coeff, ip, dofs, qp, t)
end

function _evaluate_coefficient(coeff::CoordinateSystemCoefficient{<:BiVCoordinateSystem}, ip::ScalarInterpolation, dofs, qp::QuadraturePoint{ref_shape,T}, t) where {ref_shape,T}
    x = @MVector zeros(T, 4)
    @inbounds for i in 1:getnbasefunctions(ip)
        val = Ferrite.shape_value(ip, qp.ξ, i)
        x[1] += val * coeff.cs.u_transmural[dofs[i]]
        x[2] += val * coeff.cs.u_apicobasal[dofs[i]]
        x[3] += val * coeff.cs.u_rotational[dofs[i]]
        x[4] += val * coeff.cs.u_transventricular[dofs[i]]
    end
    return BiVCoordinate(x[1], x[2], x[3], x[4])
end

"""
    AnalyticalCoefficient(f::Function, cs::CoordinateSystemCoefficient)

A coefficient given as the analytical function f(x,t) in the specified coordiante system.
"""
struct AnalyticalCoefficient{F<:Function, CSYS<:CoordinateSystemCoefficient}
    f::F
    coordinate_system_coefficient::CSYS
end

function evaluate_coefficient(coeff::AnalyticalCoefficient{F}, cell_cache, qp::QuadraturePoint{<:Any,T}, t) where {F, T}
    x = evaluate_coefficient(coeff.coordinate_system_coefficient, cell_cache, qp, t)
    return coeff.f(x, t)
end


"""
    SpectralTensorCoefficient(eigenvector_coefficient, eigenvalue_coefficient)

Represent a tensor A via spectral decomposition ∑ᵢ λᵢ vᵢ ⊗ vᵢ.
"""
struct SpectralTensorCoefficient{MSC, TC}
    eigenvectors::MSC
    eigenvalues::TC
end

@inline _eval_sdt_coefficient(M::SVector{MS}, λ::SVector{λS}) where {MS, λS} = error("Incompatible dimensions! dim(M)=$MS dim(λ)=$λS")
@inline _eval_sdt_coefficient(M::SVector{rdim,<:Vec{sdim}}, λ::SVector{rdim}) where {rdim,sdim} = sum(i->λ[i] * M[i] ⊗ M[i], 1:rdim; init=zero(Tensor{2,sdim}))

function evaluate_coefficient(coeff::SpectralTensorCoefficient, cell_cache, qp::QuadraturePoint, t)
    M = evaluate_coefficient(coeff.eigenvectors, cell_cache, qp, t)
    λ = evaluate_coefficient(coeff.eigenvalues, cell_cache, qp, t)
    return _eval_sdt_coefficient(M, λ)
end


"""
    SpatiallyHomogeneousDataField(timings::Vector, data::Vector)

A data field which is constant in space and piecewise constant in time.

The value during the time interval [tᵢ,tᵢ₊₁] is dataᵢ, where t₀ is negative infinity and the last time point+1 is positive infinity.
"""
struct SpatiallyHomogeneousDataField{T}
    timings::Vector{Float64}
    data::Vector{T}
end

function Thunderbolt.evaluate_coefficient(coeff::SpatiallyHomogeneousDataField, cell_cache, qp::QuadraturePoint, t)
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
