using Ferrite
using StaticArrays
import Base: @propagate_inbounds

# QuadratureValuesIterator
struct QuadratureValuesIterator{VT,XT}
    v::VT
    cell_coords::XT # Union{AbstractArray{<:Vec}, Nothing}
    function QuadratureValuesIterator(v::V) where V
        return new{V, Nothing}(v, nothing)
    end
    function QuadratureValuesIterator(v::V, cell_coords::VT) where {V, VT <: AbstractArray}
        reinit!(v, cell_coords)
        return new{V, VT}(v, cell_coords)
    end
end

function Base.iterate(iterator::QuadratureValuesIterator{<:Any, Nothing}, q_point=1)
    checkbounds(Bool, 1:getnquadpoints(iterator.v), q_point) || return nothing
    qp_v = @inbounds quadrature_point_values(iterator.v, q_point)
    return (qp_v, q_point+1)
end
function Base.iterate(iterator::QuadratureValuesIterator{<:Any, <:AbstractVector}, q_point=1)
    checkbounds(Bool, 1:getnquadpoints(iterator.v), q_point) || return nothing
    qp_v = @inbounds quadrature_point_values(iterator.v, q_point, iterator.cell_coords)
    return (qp_v, q_point+1)
end
Base.IteratorEltype(::Type{<:QuadratureValuesIterator}) = Base.EltypeUnknown()
Base.length(iterator::QuadratureValuesIterator) = getnquadpoints(iterator.v)

# Define `keys` and `getindex` for compatibility with `pairs`
Base.keys(it::QuadratureValuesIterator) = 1:length(it)
Base.getindex(it::QuadratureValuesIterator, i) = iterate(it, i)

# AbstractQuadratureValues
abstract type AbstractQuadratureValues end

function Ferrite.function_value(qp_v::AbstractQuadratureValues, u::AbstractVector, dof_range = eachindex(u))
    n_base_funcs = getnbasefunctions(qp_v)
    length(dof_range) == n_base_funcs || throw_incompatible_dof_length(length(dof_range), n_base_funcs)
    @boundscheck checkbounds(u, dof_range)
    val = function_value_init(qp_v, u)
    @inbounds for (i, j) in pairs(dof_range)
        val += shape_value(qp_v, i) * u[j]
    end
    return val
end

function Ferrite.function_gradient(qp_v::AbstractQuadratureValues, u::AbstractVector, dof_range = eachindex(u))
    n_base_funcs = getnbasefunctions(qp_v)
    length(dof_range) == n_base_funcs || throw_incompatible_dof_length(length(dof_range), n_base_funcs)
    @boundscheck checkbounds(u, dof_range)
    grad = function_gradient_init(qp_v, u)
    @inbounds for (i, j) in pairs(dof_range)
        grad += shape_gradient(qp_v, i) * u[j]
    end
    return grad
end

function Ferrite.function_symmetric_gradient(qp_v::AbstractQuadratureValues, u::AbstractVector, dof_range)
    grad = function_gradient(qp_v, u, dof_range)
    return symmetric(grad)
end

function Ferrite.function_symmetric_gradient(qp_v::AbstractQuadratureValues, u::AbstractVector)
    grad = function_gradient(qp_v, u)
    return symmetric(grad)
end

function Ferrite.function_divergence(qp_v::AbstractQuadratureValues, u::AbstractVector, dof_range = eachindex(u))
    return divergence_from_gradient(function_gradient(qp_v, u, dof_range))
end

function Ferrite.function_curl(qp_v::AbstractQuadratureValues, u::AbstractVector, dof_range = eachindex(u))
    return curl_from_gradient(function_gradient(qp_v, u, dof_range))
end

function Ferrite.spatial_coordinate(qp_v::AbstractQuadratureValues, x::AbstractVector{<:Vec})
    n_base_funcs = getngeobasefunctions(qp_v)
    length(x) == n_base_funcs || throw_incompatible_coord_length(length(x), n_base_funcs)
    vec = zero(eltype(x))
    @inbounds for i in 1:n_base_funcs
        vec += geometric_value(qp_v, i) * x[i]
    end
    return vec
end


# Specific design for QuadratureValues <: AbstractQuadratureValues
# which contains standard AbstractValues
struct QuadratureValues{VT<:Ferrite.AbstractValues} <: AbstractQuadratureValues
    v::VT
    q_point::Int
    Base.@propagate_inbounds function QuadratureValues(v::Ferrite.AbstractValues, q_point::Int)
        @boundscheck checkbounds(1:getnbasefunctions(v), q_point)
        return new{typeof(v)}(v, q_point)
    end
end

@inline quadrature_point_values(fe_v::Ferrite.AbstractValues, q_point, args...) = QuadratureValues(fe_v, q_point)

@propagate_inbounds Ferrite.getngeobasefunctions(qv::QuadratureValues) = getngeobasefunctions(qv.v)
@propagate_inbounds Ferrite.geometric_value(qv::QuadratureValues, i) = geometric_value(qv.v, qv.q_point, i)
Ferrite.geometric_interpolation(qv::QuadratureValues) = geometric_interpolation(qv.v)

Ferrite.getdetJdV(qv::QuadratureValues) = @inbounds getdetJdV(qv.v, qv.q_point)

# Accessors for function values 
Ferrite.getnbasefunctions(qv::QuadratureValues) = getnbasefunctions(qv.v)
Ferrite.function_interpolation(qv::QuadratureValues) = function_interpolation(qv.v)
Ferrite.function_difforder(qv::QuadratureValues) = function_difforder(qv.v)
Ferrite.shape_value_type(qv::QuadratureValues) = shape_value_type(qv.v)
Ferrite.shape_gradient_type(qv::QuadratureValues) = shape_gradient_type(qv.v)

@propagate_inbounds Ferrite.shape_value(qv::QuadratureValues, i::Int) = shape_value(qv.v, qv.q_point, i)
@propagate_inbounds Ferrite.shape_gradient(qv::QuadratureValues, i::Int) = shape_gradient(qv.v, qv.q_point, i)
@propagate_inbounds Ferrite.shape_symmetric_gradient(qv::QuadratureValues, i::Int) = shape_symmetric_gradient(qv.v, qv.q_point, i)



#= Proposed syntax, for heatflow in general 
function assemble_element!(Ke::Matrix, fe::Vector, cellvalues)
    n_basefuncs = getnbasefunctions(cellvalues)
    for qv in Ferrite.QuadratureValuesIterator(cellvalues)
        dΩ = getdetJdV(qv)
        for i in 1:n_basefuncs
            δu  = shape_value(qv, i)
            ∇δu = shape_gradient(qv, i)
            fe[i] += δu * dΩ
            for j in 1:n_basefuncs
                ∇u = shape_gradient(qv, j)
                Ke[i, j] += (∇δu ⋅ ∇u) * dΩ
            end
        end
    end
    return Ke, fe
end

Where the default for a QuadratureValuesIterator would be to return a 
`QuadratureValues` as above, but custom `AbstractValues` can be created where 
for example the element type would be a static QuadPointValue type which doesn't 
use heap allocated buffers, e.g. by only saving the cell and coordinates during reinit, 
and then calculating all values for each element in the iterator. 

References: 
https://github.com/termi-official/Thunderbolt.jl/pull/53/files#diff-2b486be5a947c02ef2a38ff3f82af3141193af0b6f01ed9d5129b914ed1d84f6
https://github.com/Ferrite-FEM/Ferrite.jl/compare/master...kam/StaticValues2
=#

struct StaticQuadratureValues{T, N_t, dNdx_t, M_t, NumN, NumM,dim ,Ti<:Integer} <: AbstractQuadratureValues
    detJdV::T
    N::SVector{NumN, N_t}
    dNdx::SVector{NumN, dNdx_t}
    M::SVector{NumM, M_t}
    weight::T
    ξ::Vec{dim,T}
    idx::Ti
end

@propagate_inbounds Ferrite.getngeobasefunctions(qv::StaticQuadratureValues) = length(qv.M)
@propagate_inbounds Ferrite.geometric_value(qv::StaticQuadratureValues, i) = qv.M[i]
# geometric_interpolation(qv::StaticQuadratureValues) = geometric_interpolation(qv.v) # Not included

Ferrite.getdetJdV(qv::StaticQuadratureValues) = qv.detJdV

# Accessors for function values 
Ferrite.getnbasefunctions(qv::StaticQuadratureValues) = length(qv.N)
# function_interpolation(qv::StaticQuadratureValues) = function_interpolation(qv.v) # Not included
Ferrite.shape_value_type(::StaticQuadratureValues{<:Any, N_t}) where N_t = N_t
Ferrite.shape_gradient_type(::StaticQuadratureValues{<:Any, <:Any, dNdx_t}) where dNdx_t = dNdx_t

@propagate_inbounds Ferrite.shape_value(qv::StaticQuadratureValues, i::Int) = qv.N[i]
@propagate_inbounds Ferrite.shape_gradient(qv::StaticQuadratureValues, i::Int) = qv.dNdx[i]
@propagate_inbounds Ferrite.shape_symmetric_gradient(qv::StaticQuadratureValues, i::Int) = symmetric(qv.dNdx[i])

@propagate_inbounds Ferrite.geometric_value(qv::StaticQuadratureValues, i::Int) = qv.M[i]

# StaticInterpolationValues: interpolation and precalculated values for all quadrature points
# Can be both for function and geometric shape functions. 
# DiffOrder parameter?
# TODO: Could perhaps denote this just InterpolationValues and replace GeometryMapping
# Just need to make Nξ::AbstractMatrix instead as in GeometryMapping to make it equivalent (except fieldnames)
struct StaticInterpolationValues{IP, N, Nqp, N_et, dNdξ_t, Nall}
    ip::IP
    Nξ::SMatrix{N, Nqp, N_et, Nall}
    dNdξ::dNdξ_t        # Union{SMatrix{N, Nqp}, Nothing}
    #dN2dξ2::dN2dξ2_t   # Union{SMatrix{N, Nqp}, Nothing}
end
function StaticInterpolationValues(fv::Ferrite.FunctionValues)
    N = getnbasefunctions(fv.ip)
    Nq = size(fv.Nξ, 2)
    Nξ = SMatrix{N, Nq}(fv.Nξ)
    dNdξ = SMatrix{N, Nq}(fv.dNdξ)
    return StaticInterpolationValues(fv.ip, Nξ, dNdξ)
end
function StaticInterpolationValues(gm::Ferrite.GeometryMapping)
    N = getnbasefunctions(gm.ip)
    Nq = size(gm.M, 2)
    M = SMatrix{N, Nq}(gm.M)
    dMdξ = SMatrix{N, Nq}(gm.dMdξ)
    return StaticInterpolationValues(gm.ip, M, dMdξ)
end

Ferrite.shape_value(siv::StaticInterpolationValues, qp::Int, i::Int) = siv.Nξ[i, qp]
Ferrite.getnbasefunctions(siv::StaticInterpolationValues) = getnbasefunctions(siv.ip)


# Dispatch on DiffOrder parameter? 
# Reuse functions for GeometryMapping - same signature but need access functions
# Or merge GeometryMapping and StaticInterpolationValues => InterpolationValues
@propagate_inbounds @inline function Ferrite.calculate_mapping(ip_values::StaticInterpolationValues{<:Any, N}, q_point, x) where N
    fecv_J = zero(Ferrite.otimes_returntype(eltype(x), eltype(ip_values.dNdξ)))
    @inbounds for j in 1:N
        #fecv_J += x[j] ⊗ geo_mapping.dMdξ[j, q_point]
        fecv_J += Ferrite.otimes_helper(x[j], ip_values.dNdξ[j, q_point])
    end
    return Ferrite.MappingValues(fecv_J, nothing)
end

@propagate_inbounds @inline function calculate_mapped_values(funvals::StaticInterpolationValues, q_point, mapping_values, args...)
    return calculate_mapped_values(funvals, Ferrite.mapping_type(funvals.ip), q_point, mapping_values, args...)
end

@propagate_inbounds @inline function calculate_mapped_values(funvals::StaticInterpolationValues, ::Ferrite.IdentityMapping, q_point, mapping_values, args...)
    Jinv = Ferrite.calculate_Jinv(Ferrite.getjacobian(mapping_values))
    Nx = funvals.Nξ[:, q_point]
    dNdx = map(dNdξ -> Ferrite.dothelper(dNdξ, Jinv), funvals.dNdξ[:, q_point])
    return Nx, dNdx
end

struct StaticCellValues{FV, GM, Nqp, T,dim}
    fv::FV # StaticInterpolationValues
    gm::GM # StaticInterpolationValues
    weights::NTuple{Nqp, T}
    ξs::NTuple{Nqp,Vec{dim,T}} # quadrature points
    function StaticCellValues{FV, GM, Nqp, T, dim}(fv::FV, gm::GM, weights::NTuple{Nqp, T}, ξs::NTuple{Nqp, Vec{dim, T}}) where {FV, GM, Nqp, T, dim}
        return new{FV, GM, Nqp, T, dim}(fv, gm, weights, ξs)
    end
end


function StaticCellValues(cv::CellValues) 
    fv = StaticInterpolationValues(cv.fun_values)
    gm = StaticInterpolationValues(cv.geo_mapping)
    weights = ntuple(i -> getweights(cv.qr)[i], getnquadpoints(cv))
    #positions = ntuple(i ->  getpoints(cv.qr)[i].data, getnquadpoints(qr))
    ξs = ntuple(i ->  getpoints(cv.qr)[i], getnquadpoints(qr))
    return StaticCellValues(fv, gm,weights, ξs)
end

# function StaticCellValues(cv::CellValues, ::Val{SaveCoords}=Val(true)) where SaveCoords
#     fv = StaticInterpolationValues(cv.fun_values)
#     gm = StaticInterpolationValues(cv.geo_mapping)
#     sdim = sdim_from_gradtype(shape_gradient_type(cv))
#     x = SaveCoords ? fill(zero(Vec{sdim}), getngeobasefunctions(cv)) : nothing
#     weights = ntuple(i -> getweights(cv.qr)[i], getnquadpoints(cv))
#     return StaticCellValues(fv, gm, x, weights)
# end

Ferrite.getnquadpoints(cv::StaticCellValues)  = length(cv.weights)
Ferrite.getnbasefunctions(cv::StaticCellValues) = getnbasefunctions(cv.fv)
Ferrite.getngeobasefunctions(cv::StaticCellValues) = getnbasefunctions(cv.gm)

@inline function Ferrite.reinit!(::StaticCellValues, ::AbstractVector)
    nothing # Nothing to do on reinit if x is not saved.
end

@inline function quadrature_point_values(fe_v::StaticCellValues, q_point::Int)
    return _quadrature_point_values(fe_v, q_point, fe_v.x, detJ -> throw_detJ_not_pos(detJ))
end

@inline function quadrature_point_values(fe_v::StaticCellValues, q_point::Int, cell_coords::AbstractVector)
    return _quadrature_point_values(fe_v, q_point, cell_coords, detJ -> throw_detJ_not_pos(detJ))
end

@inline function quadrature_point_values(fe_v::StaticCellValues, q_point::Int, cell_coords::StaticVector)
    return _quadrature_point_values(fe_v, q_point, cell_coords, detJ -> -1)
end

function _quadrature_point_values(fe_v::StaticCellValues, q_point::Int, cell_coords::AbstractVector, neg_detJ_err_fun::Function)
    #q_point bounds checked, ok to use @inbounds
    @inbounds begin
            mapping = Ferrite.calculate_mapping(fe_v.gm, q_point, cell_coords)
        
            detJ = Ferrite.calculate_detJ(Ferrite.getjacobian(mapping))
            detJ > 0.0f0 || neg_detJ_err_fun(detJ) # Cannot throw error on GPU, TODO: return error code instead
            weight = fe_v.weights[q_point]
            detJdV = detJ * weight
        
            Nx, dNdx = calculate_mapped_values(fe_v.fv, q_point, mapping)
            M = fe_v.gm.Nξ[:, q_point]
            ξ = fe_v.ξs[q_point]
        end
        return StaticQuadratureValues(detJdV, Nx, dNdx, M, weight, ξ, q_point)
end
