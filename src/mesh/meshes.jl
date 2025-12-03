
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
