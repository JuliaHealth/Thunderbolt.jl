function FerriteOperators.getquadraturerule(
    qrc::FerriteOperators.QuadratureRuleCollection,
    cell::InterfaceCell,
)
    return getquadraturerule(qrc, cell.here)
end

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

struct InterfaceCollection{IPC} <: InterpolationCollection
    ipc::IPC
end

getorder(ic::InterfaceCollection) = getorder(ic.ipc)

function getinterpolation(ic::InterfaceCollection, cell::InterfaceCell)
    return InterfaceCellInterpolation(getinterpolation(ic.ipc, cell.here))
end

# Wildcard
"""
    getinterpolation(ipc::InterpolationCollection, cell::AbstractCell)
    getinterpolation(ipc::InterpolationCollection, ::Type{<:AbstractRefShape})
    getinterpolation(ipc::InterpolationCollection, sdh::SubDofHandler)

The collection's interpolation for a reference shape: the cell's own, or -- the form the
discretization uses -- the shape of the subdomain's first cell, shared by every cell of a
`SubDofHandler`.
"""
getinterpolation(ipc::InterpolationCollection, sdh::SubDofHandler) =
    getinterpolation(ipc, get_first_cell(sdh))

"""
    LagrangeCollection{order} <: InterpolationCollection

A collection of fixed-order Lagrange interpolations across different cell types.
"""
struct LagrangeCollection{order} <: ScalarInterpolationCollection end

getorder(::LagrangeCollection{order}) where {order} = order
getinterpolation(
    lc::LagrangeCollection{order},
    cell::AbstractCell{ref_shape},
) where {order, ref_shape <: Ferrite.AbstractRefShape} = Lagrange{ref_shape, order}()
getinterpolation(
    lc::LagrangeCollection{order},
    ::Type{ref_shape},
) where {order, ref_shape <: Ferrite.AbstractRefShape} = Lagrange{ref_shape, order}()

"""
    DiscontinuousLagrangeCollection{order} <: InterpolationCollection

A collection of fixed-order Lagrange interpolations across different cell types.
"""
struct DiscontinuousLagrangeCollection{order} <: ScalarInterpolationCollection end

getorder(::DiscontinuousLagrangeCollection{order}) where {order} = order
getinterpolation(
    lc::DiscontinuousLagrangeCollection{order},
    cell::AbstractCell{ref_shape},
) where {order, ref_shape <: Ferrite.AbstractRefShape} = DiscontinuousLagrange{ref_shape, order}()
getinterpolation(
    lc::DiscontinuousLagrangeCollection{order},
    ::Type{ref_shape},
) where {order, ref_shape <: Ferrite.AbstractRefShape} = DiscontinuousLagrange{ref_shape, order}()


"""
    VectorizedInterpolationCollection{order} <: InterpolationCollection

A collection of fixed-order vectorized Lagrange interpolations across different cell types.
"""
struct VectorizedInterpolationCollection{vdim, IPC <: ScalarInterpolationCollection} <:
       VectorInterpolationCollection
    base::IPC
    function VectorizedInterpolationCollection{vdim}(
        ip::SIPC,
    ) where {vdim, SIPC <: ScalarInterpolationCollection}
        return new{vdim, SIPC}(ip)
    end
end

Base.:(^)(ip::ScalarInterpolationCollection, vdim::Int) =
    VectorizedInterpolationCollection{vdim}(ip)

getorder(ipc::VectorizedInterpolationCollection) = getorder(ipc.base)
getinterpolation(
    ipc::VectorizedInterpolationCollection{vdim, IPC},
    cell::AbstractCell{ref_shape},
) where {vdim, IPC, ref_shape <: Ferrite.AbstractRefShape} = getinterpolation(ipc.base, cell)^vdim
getinterpolation(
    ipc::VectorizedInterpolationCollection{vdim, IPC},
    type::Type{ref_shape},
) where {vdim, IPC, ref_shape <: Ferrite.AbstractRefShape} = getinterpolation(ipc.base, type)^vdim

"""
    NodalQuadratureRuleCollection(::InterpolationCollection)

A collection of nodal (collocated) quadrature rules across different cell types: the rule whose
points are the interpolation's own dof locations.

On a **hypercube** carrying a `Lagrange` or `DiscontinuousLagrange` interpolation of order `p` the
nodes are the tensor product of the `p+1` Gauss-Lobatto points, so this rule is Ferrite's own
`:lobatto` rule reordered into the interpolation's node order, and its weights are exact. That is
what makes the mass matrix assembled through it *diagonal* — the spectral-element mass — while the
same space under a Gauss rule is not.

!!! warning
    On any other reference shape the collocated weights are not implemented and default to `NaN`.
    Such a rule still positions correctly, which is all the field-evaluation callers need, but it
    cannot integrate.
"""
struct NodalQuadratureRuleCollection{IPC <: InterpolationCollection}
    ipc::IPC
end

function getquadraturerule(
    nqr::NodalQuadratureRuleCollection,
    cell::AbstractCell{ref_shape},
) where {ref_shape}
    ip = getinterpolation(nqr.ipc, cell)
    positions = Ferrite.reference_coordinates(ip)
    return QuadratureRule{ref_shape}(_nodal_quadrature_weights(ip, positions), positions)
end
getquadraturerule(qrc::NodalQuadratureRuleCollection, sdh::SubDofHandler) =
    getquadraturerule(qrc, get_first_cell(sdh))

_nodal_quadrature_weights(ip, positions) = [NaN for _ = 1:length(positions)]

const _TensorProductLagrange{dim, order} = Union{
    Lagrange{Ferrite.RefHypercube{dim}, order},
    DiscontinuousLagrange{Ferrite.RefHypercube{dim}, order},
}

# Ferrite orders hypercube Lagrange nodes by entity (vertices, then edges, ...) and its `:lobatto`
# rule lexicographically, so the weights are matched by POSITION rather than by index. The matching
# is asserted to be a bijection: a node the rule does not carry would otherwise silently take a
# neighbour's weight, and the mass matrix would come out diagonal and wrong.
function _nodal_quadrature_weights(
    ::_TensorProductLagrange{dim, order},
    positions,
) where {dim, order}
    qr = QuadratureRule{Ferrite.RefHypercube{dim}}(Float64, :lobatto, order + 1)
    points, weights = Ferrite.getpoints(qr), Ferrite.getweights(qr)
    length(points) == length(positions) || error(
        "The collocated Gauss-Lobatto rule of order $order on RefHypercube{$dim} has " *
        "$(length(points)) points but the interpolation has $(length(positions)) nodes.",
    )
    taken = falses(length(points))
    out = Vector{Float64}(undef, length(positions))
    for (i, x) in pairs(positions)
        j = argmin(k -> maximum(abs, points[k] - x), eachindex(points))
        (maximum(abs, points[j] - x) < 1.0e-10 && !taken[j]) || error(
            "Node $i of the interpolation sits at $x, which is not an unmatched point of the " *
            "collocated Gauss-Lobatto rule. The two are meant to be the same point set.",
        )
        taken[j] = true
        out[i] = weights[j]
    end
    return out
end


"""
    FacetQuadratureRuleCollection(order::Int)

A collection of quadrature rules across different cell types.
"""
struct FacetQuadratureRuleCollection{order} end

FacetQuadratureRuleCollection(order::Int) = FacetQuadratureRuleCollection{order}()

getquadraturerule(
    qrc::FacetQuadratureRuleCollection{order},
    cell::AbstractCell{ref_shape},
) where {order, ref_shape} = FacetQuadratureRule{ref_shape}(order)
getquadraturerule(
    qrc::FacetQuadratureRuleCollection{order},
    ::Type{ref_shape},
) where {order, ref_shape <: Ferrite.AbstractRefShape} = FacetQuadratureRule{ref_shape}(order)
getquadraturerule(qrc::FacetQuadratureRuleCollection, sdh::SubDofHandler) =
    getquadraturerule(qrc, get_first_cell(sdh))


"""
    CellValueCollection(::QuadratureRuleCollection, ::InterpolationCollection)

Helper to construct and query the correct cell values on mixed grids.
"""
struct CellValueCollection{
    QRC <: Union{QuadratureRuleCollection, NodalQuadratureRuleCollection},
    IPC <: InterpolationCollection,
}
    qrc::QRC
    ipc::IPC
end

getcellvalues(cv::CellValueCollection, cell::CellType) where {CellType <: AbstractCell} =
    CellValues(
        getquadraturerule(cv.qrc, cell),
        getinterpolation(cv.ipc, cell),
        Ferrite.geometric_interpolation(CellType),
    )
getcellvalues(qrc::CellValueCollection, sdh::SubDofHandler) =
    getcellvalues(qrc, get_first_cell(sdh))


"""
    FacetValueCollection(::QuadratureRuleCollection, ::InterpolationCollection)

Helper to construct and query the correct facet values on mixed grids.
"""
struct FacetValueCollection{QRC <: FacetQuadratureRuleCollection, IPC <: InterpolationCollection}
    qrc::QRC
    ipc::IPC
end

getfacetvalues(fv::FacetValueCollection, cell::CellType) where {CellType <: AbstractCell} =
    FacetValues(
        getquadraturerule(fv.qrc, cell),
        getinterpolation(fv.ipc, cell),
        Ferrite.geometric_interpolation(CellType),
    )
getfacetvalues(qrc::FacetValueCollection, sdh::SubDofHandler) =
    getfacetvalues(qrc, get_first_cell(sdh))


"""
    ElementwiseData(data, offsets)

Container to handle manage quadrature data and friends on mixed grids.
"""
struct ElementwiseData{
    DataType,
    StorageType <: AbstractVector{DataType},
    IndexStorageType <: AbstractVector{<:Int},
} <: AbstractMatrix{DataType}
    data::StorageType
    offsets::IndexStorageType
    sizes::IndexStorageType
end

Base.getindex(data::ElementwiseData, i::Int) = data.data[i]
Base.length(data::ElementwiseData) = length(data.data)
Base.size(data::ElementwiseData) = (0, length(data.offsets))
function Base.show(
    io::IO,
    ::MIME"text/plain",
    data::ElementwiseData{DataType, StorageType, IndexStorageType},
) where {DataType, StorageType, IndexStorageType}
    print(
        io,
        "ElementwiseData{DataType=$DataType, StorageType=$StorageType, IndexStorageType=$IndexStorageType} with $(length(data.data)) entries and outer dimension $(length(data.offsets)).",
    )
end

function Base.setindex!(data::ElementwiseData{T}, v::T, i::Int) where {T}
    data.data[i] = v
end

function Base.getindex(data::ElementwiseData, j::Int, i::Int)
    os = data.offsets[i]:(data.offsets[i]+data.sizes[i]-1)
    dv = @view data.data[os]
    return dv[j]
end

function Base.setindex!(data::ElementwiseData{T}, v::T, j::Int, i::Int) where {T}
    os = data.offsets[i]:(data.offsets[i]+data.sizes[i]-1)
    dv = @view data.data[os]
    dv[j] = v
end


"""
    ApproximationDescriptor(symbol, interpolation_collection)
"""
struct ApproximationDescriptor
    sym::Symbol
    ipc::InterpolationCollection
end

"""
    add_subdomain!(dh, name::String, approximations::Vector{ApproximationDescriptor})
    add_subdomain!(dh, name::String, sym => interpolation_collection)
    add_subdomain!(dh, approximations)

Add the fields described by `approximations` to `dh` on the mesh's volumetric subdomain `name`, one
`SubDofHandler` per cell type occurring there. Errors if the mesh has no subdomain of that name.

The form without a name applies to the mesh's only subdomain and asserts that there is exactly one.
"""
function add_subdomain!(
    dh::DofHandler{<:Any, <:SimpleMesh},
    name::String,
    approxmations::Vector{ApproximationDescriptor},
)
    mesh = dh.grid
    cells = mesh.grid.cells
    haskey(mesh.volumetric_subdomains, name) || error(
        "Volumetric Subdomain $name not found on mesh. Available subdomains: $(keys(mesh.volumetric_subdomains))",
    )
    for (celltype, cellset) in mesh.volumetric_subdomains[name].data
        # @info name, length(cellset)
        sdh = SubDofHandler(dh, OrderedSet{Int}([idx.idx for idx in cellset]))
        for ad in approxmations
            add!(sdh, ad.sym, getinterpolation(ad.ipc, cells[first(sdh.cellset)]))
        end
    end
end
add_subdomain!(dh, domain_name, descriptor::Pair) =
    add_subdomain!(dh, domain_name, [ApproximationDescriptor(descriptor[1], descriptor[2])])
function add_subdomain!(dh, descriptor)
    vsubdomain = get_grid(dh).volumetric_subdomains
    @assert length(vsubdomain) == 1 "Mesh has multiple subdomains. Please specify the subdomain on which the approximation is defined."
    add_subdomain!(dh, first(keys(vsubdomain)), descriptor)
end

# function add_surface_subdomain!(dh::DofHandler{<:Any, <:SimpleMesh}, name::String, approxmations::Vector{ApproximationDescriptor})
#     mesh = dh.grid
#     haskey(mesh.surface_subdomains, name) || error("Surface Subdomain $name not found on mesh. Available subdomains: $(keys(mesh.surface_subdomains))")
#     for (celltype, cellset) in mesh.surface_subdomains[name].data
#         dh_solid_quad = SubDofHandler(dh, cellset)
#         for ad in approxmations
#             add!(dh_solid_quad, ad.sym, getinterpolation(ipc, celltype))
#         end
#     end
# end

# function add_interface_subdomain!(dh::DofHandler{<:Any, <:SimpleMesh}, name::String, approxmations::Vector{ApproximationDescriptor})
#     mesh = dh.grid
#     haskey(mesh.interface_subdomains, name) || error("Interface Subdomain $name not found on mesh. Available subdomains: $(keys(mesh.interface_subdomains))")
#     for (celltype, cellset) in mesh.interface_subdomains[name].data
#         dh_solid_quad = SubDofHandler(dh, cellset)
#         for ad in approxmations
#             add!(dh_solid_quad, ad.sym, getinterpolation(ipc, celltype))
#         end
#     end
# end
