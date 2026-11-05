
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
const LinearCellGeometry =
    Union{Hexahedron, Tetrahedron, Pyramid, Wedge, Triangle, Quadrilateral, Line}

"""
    elementtypes(mesh) -> SVector{<:Type}

The concrete cell types occurring in `mesh`, as a statically sized vector.

Defined for grids and meshes of a single element type. A mixed mesh carries its per-type split in
its subdomain descriptors instead, so it is not answered here.
"""
elementtypes(grid::Grid{3, Hexahedron}) = @SVector [Hexahedron]
elementtypes(grid::Grid{3, QuadraticHexahedron}) = @SVector [QuadraticHexahedron]
elementtypes(grid::Grid{3, Tetrahedron}) = @SVector [Tetrahedron]
elementtypes(grid::Grid{3, QuadraticTetrahedron}) = @SVector [QuadraticTetrahedron]

include("simple_meshes.jl")
include("tools.jl")
include("generators.jl")
include("long_axis.jl")
include("reorder.jl")

Ferrite.PointEvalHandler(
    mesh::SimpleMesh{sdim},
    points::AbstractVector{Vec{sdim, T}},
) where {sdim, T} = Ferrite.PointEvalHandler(mesh.grid, points)
