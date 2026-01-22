struct ActivationCoordinate{CoordT<:Union{LVCoordinate, BiVCoordinate}, T}
    coord::CoordT
    time_offset::T
end

@kwdef struct SimplicialEikonalDiscretization{ActivationCoordinateT <: ActivationCoordinate}
    order::Int
    activation_points::Vector{ActivationCoordinateT}
    subdomains::Vector{String} = [""]
end

function semidiscretize(
    model::EikonalModel,
    discretization::SimplicialEikonalDiscretization,
    mesh::SimpleMesh,
)
    activation_points = discretization.activation_points

    vertices = getproperty.(mesh.grid.nodes, :x)
    cells = getproperty.(mesh.grid.cells, :nodes)
    nnodes = getnnodes(mesh.grid)

    max_vertices, max_edges, max_faces = Ferrite._max_nentities_per_cell(getcells(mesh.grid))
    vertex_to_cell = (Ferrite.build_vertex_to_cell(getcells(mesh.grid); max_vertices, nnodes))
    return EikonalFunction(
        vertices,
        cells,
        vertex_to_cell,
        activation_points
    )
end
