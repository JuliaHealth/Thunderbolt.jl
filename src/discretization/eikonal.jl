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
    # first_node_type = typeof(first(mesh.grid.nodes).x)
    # nvertices_per_cell = 4 #strictly for Tetrahedra
    # nodes_non_zeros_indices = Int[]
    # nodes_non_zeros_values = first_node_type[]
    # cells_non_zeros_indices = Int[]
    # cells_non_zeros_values = NTuple{nvertices_per_cell, Int}[]
    activation_points = discretization.activation_points
    # for subdomain in discretization.subdomains
    #     descriptor = mesh.volumetric_subdomains[subdomain]
    #     for (type, set) in descriptor.data 
    #         type != Tetrahedron && throw(error("Only Tetrahedral meshes are allowed"))
    #         sizehint!(cells_non_zeros_indices, length(cells_non_zeros_indices) + length(set))
    #         sizehint!(cells_non_zeros_values, length(cells_non_zeros_values) + length(set))
    #         for cell in set
    #             push!(cells_non_zeros_indices, cell.idx)
    #             push!(cells_non_zeros_values, mesh.grid.cells[cell.idx].nodes)
    #             for node_id in mesh.grid.cells[cell.idx].nodes
    #                 coords = mesh.grid.nodes[node_id].x
    #                 ff = findfirst(isequal(node_id), nodes_non_zeros_indices)
    #                 if isnothing(ff)
    #                     push!(nodes_non_zeros_indices, node_id)
    #                     push!(nodes_non_zeros_values, coords)
    #                 end
    #             end
    #         end
    #     end
    # end
    # max_vertices, max_edges, max_faces = Ferrite._max_nentities_per_cell(getcells(mesh.grid))
    # vertex_to_cell = (Ferrite.build_vertex_to_cell(getcells(mesh.grid); max_vertices, nnodes=getnnodes(mesh.grid)))
    
    # vertices = SparseVector(length(mesh.grid.nodes), nodes_non_zeros_indices, nodes_non_zeros_values)
    # cells = SparseVector(length(mesh.grid.cells), cells_non_zeros_indices, cells_non_zeros_values)


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
