"""
Coordinates used for defining activation points when solving the eikonal equation.
"""
struct ActivationCoordinate{CoordT<:Union{LVCoordinate, BiVCoordinate}, T}
    coord::CoordT
    time_offset::T
end

abstract type AbstractEikonalActivationProtocol end

"""
Activation protocol for activating specific points with time offsets to mimic the behavior of a Purkinje network.
"""
struct NodalEikonalActivationProtocol{VectorT<:AbstractVector{<:ActivationCoordinate}} <:AbstractEikonalActivationProtocol
    nodes::VectorT
end

"""
Activation protocol for uniformally activating the endocardium with zero wave arrival time.
"""
struct UniformEndocardialEikonalActivationProtocol <:AbstractEikonalActivationProtocol
end

"""
Descriptor for a tetrahedral element discretization used for solving the Eikonal equation explicitly.
"""
@kwdef struct SimplicialEikonalDiscretization{ActivationProtocolT <: AbstractEikonalActivationProtocol}
    order::Int = 1
    activation_protocol::ActivationProtocolT
    subdomains::Vector{String} = [""]
end

"""
Returns the permutation used to reverser [reorder_nodal!](@ref)
"""
function get_nodes_to_vertex_permutaion(dh::DofHandler)
    res = Vector{Int}(undef, dh.ndofs)
    grid = Ferrite.get_grid(dh)
    for i in 1:getncells(grid)
        res[SVector(getcells(grid, i).nodes)] .= (@view dh.cell_dofs[dh.cell_dofs_offset[i]:(dh.cell_dofs_offset[i] + Ferrite.ndofs_per_cell(dh, i) - 1)])
    end
    return sortperm(res)
end

"""
Returns the indicies of the nodes belonging to the endocardium, defaulting to 5% of the
transmural coordinate range as endocardium.
"""
function get_nodes(::UniformEndocardialEikonalActivationProtocol, mesh, cs)
    perm = get_nodes_to_vertex_permutaion(cs.dh)|>sortperm
    nodes = Int[]
    for node in eachindex(mesh.grid.nodes)
        cs.u_transmural[perm[node]] <= 0.05 || continue
        push!(nodes, node)
    end
    return nodes
end

function semidiscretize(
    ::EikonalModel,
    ::SimplicialEikonalDiscretization,
    activation_points,
    mesh::SimpleMesh,
)

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
