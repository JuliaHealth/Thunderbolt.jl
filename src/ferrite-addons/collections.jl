"""
    InterpolationCollection

A collection of compatible interpolations over some (possilby different) cells.
"""
abstract type InterpolationCollection end

"""
    ScalarInterpolationCollection

A collection of compatible scalar-valued interpolations over some (possilby different) cells.
"""
abstract type ScalarInterpolationCollection <: InterpolationCollection end

"""
    VectorInterpolationCollection

A collection of compatible vector-valued interpolations over some (possilby different) cells.
"""
abstract type VectorInterpolationCollection <: InterpolationCollection end

getinterpolation(qrc::InterpolationCollection, sdh::SubDofHandler) = getfacetvalues(qrc, get_first_cell(sdh))

"""
    LagrangeCollection{order} <: InterpolationCollection

A collection of fixed-order Lagrange interpolations across different cell types.
"""
struct LagrangeCollection{order} <: ScalarInterpolationCollection
end

getorder(::LagrangeCollection{order}) where order = order
getinterpolation(lc::LagrangeCollection{order}, cell::AbstractCell{ref_shape}) where {order, ref_shape <: Ferrite.AbstractRefShape} = Lagrange{ref_shape, order}()
getinterpolation(lc::LagrangeCollection{order}, ::Type{ref_shape}) where {order, ref_shape <: Ferrite.AbstractRefShape} = Lagrange{ref_shape, order}()

"""
    DiscontinuousLagrangeCollection{order} <: InterpolationCollection

A collection of fixed-order Lagrange interpolations across different cell types.
"""
struct DiscontinuousLagrangeCollection{order} <: ScalarInterpolationCollection
end

getorder(::DiscontinuousLagrangeCollection{order}) where order = order
getinterpolation(lc::DiscontinuousLagrangeCollection{order}, cell::AbstractCell{ref_shape}) where {order, ref_shape <: Ferrite.AbstractRefShape} = DiscontinuousLagrange{ref_shape, order}()
getinterpolation(lc::DiscontinuousLagrangeCollection{order}, ::Type{ref_shape}) where {order, ref_shape <: Ferrite.AbstractRefShape} = DiscontinuousLagrange{ref_shape, order}()


"""
    VectorizedInterpolationCollection{order} <: InterpolationCollection

A collection of fixed-order vectorized Lagrange interpolations across different cell types.
"""
struct VectorizedInterpolationCollection{vdim, IPC <: ScalarInterpolationCollection} <: VectorInterpolationCollection
    base::IPC
    function VectorizedInterpolationCollection{vdim}(ip::SIPC) where {vdim, SIPC <: ScalarInterpolationCollection}
        return new{vdim, SIPC}(ip)
    end
end

Base.:(^)(ip::ScalarInterpolationCollection, vdim::Int) = VectorizedInterpolationCollection{vdim}(ip)

getorder(ipc::VectorizedInterpolationCollection) = getorder(ipc.base)
getinterpolation(ipc::VectorizedInterpolationCollection{vdim, IPC}, cell::AbstractCell{ref_shape}) where {vdim, IPC, ref_shape <: Ferrite.AbstractRefShape} = getinterpolation(ipc.base, cell)^vdim
getinterpolation(ipc::VectorizedInterpolationCollection{vdim, IPC}, type::Type{ref_shape}) where {vdim, IPC, ref_shape <: Ferrite.AbstractRefShape} = getinterpolation(ipc.base, type)^vdim


"""
    QuadratureRuleCollection(order::Int)

A collection of quadrature rules across different cell types.
"""
struct QuadratureRuleCollection{order}
end

QuadratureRuleCollection(order::Int) = QuadratureRuleCollection{order}()

getquadraturerule(qrc::QuadratureRuleCollection{order}, cell::AbstractCell{ref_shape}) where {order,ref_shape} = QuadratureRule{ref_shape}(order)
getquadraturerule(qrc::QuadratureRuleCollection, sdh::SubDofHandler) = getquadraturerule(qrc, get_first_cell(sdh))


"""
    NodalQuadratureRuleCollection(::InterpolationCollection)

A collection of nodal quadrature rules across different cell types.

!!! warning
    The computation for the weights is not implemented yet and hence they default to NaN.
"""
struct NodalQuadratureRuleCollection{IPC <: InterpolationCollection}
    ipc::IPC
end

function getquadraturerule(nqr::NodalQuadratureRuleCollection, cell::AbstractCell{ref_shape}) where {ref_shape}
    ip = getinterpolation(nqr.ipc, cell)
    positions = Ferrite.reference_coordinates(ip)
    return QuadratureRule{ref_shape}([NaN for _ in 1:length(positions)], positions)
end
getquadraturerule(qrc::NodalQuadratureRuleCollection, sdh::SubDofHandler) = getquadraturerule(qrc, get_first_cell(sdh))


"""
    FacetQuadratureRuleCollection(order::Int)

A collection of quadrature rules across different cell types.
"""
struct FacetQuadratureRuleCollection{order}
end

FacetQuadratureRuleCollection(order::Int) = FacetQuadratureRuleCollection{order}()

getquadraturerule(qrc::FacetQuadratureRuleCollection{order}, cell::AbstractCell{ref_shape}) where {order,ref_shape} = FacetQuadratureRule{ref_shape}(order)
getquadraturerule(qrc::FacetQuadratureRuleCollection, sdh::SubDofHandler) = getquadraturerule(qrc, get_first_cell(sdh))


"""
    CellValueCollection(::QuadratureRuleCollection, ::InterpolationCollection)

Helper to construct and query the correct cell values on mixed grids.
"""
struct CellValueCollection{QRC <: Union{QuadratureRuleCollection, NodalQuadratureRuleCollection}, IPC <: InterpolationCollection}
    qrc::QRC
    ipc::IPC
end

getcellvalues(cv::CellValueCollection, cell::CellType) where {CellType <: AbstractCell} = CellValues(
    getquadraturerule(cv.qrc, cell),
    getinterpolation(cv.ipc, cell),
    Ferrite.geometric_interpolation(CellType)
)
getcellvalues(qrc::CellValueCollection, sdh::SubDofHandler) = getcellvalues(qrc, get_first_cell(sdh))


"""
    FacetValueCollection(::QuadratureRuleCollection, ::InterpolationCollection)

Helper to construct and query the correct facet values on mixed grids.
"""
struct FacetValueCollection{QRC <: FacetQuadratureRuleCollection, IPC <: InterpolationCollection}
    qrc::QRC
    ipc::IPC
end

getfacetvalues(fv::FacetValueCollection, cell::CellType) where {CellType <: AbstractCell} = FacetValues(
    getquadraturerule(fv.qrc, cell),
    getinterpolation(fv.ipc, cell),
    Ferrite.geometric_interpolation(CellType)
)
getfacetvalues(qrc::FacetValueCollection, sdh::SubDofHandler) = getfacetvalues(qrc, get_first_cell(sdh))


"""
    ElementwiseData(data, offsets)

Container to handle manage quadrature data and friends on mixed grids.
"""
struct ElementwiseData{DataType, StorageType <: AbstractVector{DataType}, IndexStorageType <: AbstractVector{<:Int}} <: AbstractMatrix{DataType}
    data::StorageType
    offsets::IndexStorageType
    # FIXME the cells must be ordered ascending for this to work, which is not true in general. Insert the unit ranges here instead.
end

Base.getindex(data::ElementwiseData, i::Int) = data.data[i]
Base.length(data::ElementwiseData) = length(data.data)
Base.size(data::ElementwiseData) = (0, length(data.offsets))
function Base.show(io::IO, ::MIME"text/plain", data::ElementwiseData{DataType, StorageType, IndexStorageType}) where {DataType, StorageType, IndexStorageType}
    print(io, "ElementwiseData{DataType=$DataType, StorageType=$StorageType, IndexStorageType=$IndexStorageType} with $(length(data.data)) entries and outer dimension $(length(data.offsets)).")
end

function Base.setindex!(data::ElementwiseData{T}, v::T, i::Int) where T
    data.data[i] = v
end

function Base.getindex(data::ElementwiseData, j::Int, i::Int)
    os = data.offsets[i]:(data.offsets[i+1]-1)
    dv = @view data.data[os]
    return dv[j]
end

function Base.setindex!(data::ElementwiseData{T}, v::T, j::Int, i::Int) where T
    os = data.offsets[i]:(data.offsets[i+1]-1)
    dv = @view data.data[os]
    dv[j] = v
end
