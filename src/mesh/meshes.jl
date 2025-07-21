
# abstract type AbstractSurface end
# abstract type AbstractPath end
# struct ConformingFacet3D <: AbstractSurface
#     cellidx_a::Int
#     cellidx_b::Int
#     defining_nodes::NTuple{3,Int}
# end

# struct SimpleEdge3D <: AbstractPath
#     defining_nodes::NTuple{3,Int}
# end

# TODO we might want to add this to Ferrite (and especially FerriteViz) in one or another way. Maybe traits are better, because they allow more extensibility.
const LinearCellGeometry = Union{Hexahedron, Tetrahedron, Pyramid, Wedge, Triangle, Quadrilateral, Line}

elementtypes(grid::Grid{3,Hexahedron}) = @SVector [Hexahedron]
elementtypes(grid::Grid{3,QuadraticHexahedron}) = @SVector [QuadraticHexahedron]
elementtypes(grid::Grid{3,Tetrahedron}) = @SVector [Tetrahedron]
elementtypes(grid::Grid{3,QuadraticTetrahedron}) = @SVector [QuadraticTetrahedron]

include("simple_meshes.jl")
include("tools.jl")
include("generators.jl")

Ferrite.PointEvalHandler(mesh::SimpleMesh{sdim}, points::AbstractVector{Vec{sdim, T}}) where {sdim, T} = Ferrite.PointEvalHandler(mesh.grid, points)


# TODO where to put this?
"""
    ApproximationDescriptor(symbol, interpolation_collection)
"""
struct ApproximationDescriptor
    sym::Symbol
    ipc::InterpolationCollection
end

function add_subdomain!(dh::DofHandler{<:Any, <:SimpleMesh}, name::String, approxmations::Vector{ApproximationDescriptor})
    mesh = dh.grid
    cells = mesh.grid.cells
    haskey(mesh.volumetric_subdomains, name) || error("Volumetric Subdomain $name not found on mesh. Available subdomains: $(keys(mesh.volumetric_subdomains))")
    for (celltype, cellset) in mesh.volumetric_subdomains[name].data
        # @info name, length(cellset)
        sdh = SubDofHandler(dh, OrderedSet{Int}([idx.idx for idx in cellset]))
        for ad in approxmations
            add!(sdh, ad.sym, getinterpolation(ad.ipc, cells[first(sdh.cellset)]))
        end
    end
end

function _compatible_cellset(dh::DofHandler, firstcell::Int)
    for sdh in dh.subdofhandlers
        if firstcell ∈ sdh.cellset
            return sdh.cellset
        end
    end
    error("Cell $firstcell not found.")
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
