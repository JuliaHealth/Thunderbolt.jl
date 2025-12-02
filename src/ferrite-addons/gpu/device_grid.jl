# Utility which holds partial information for assembly.
struct DeviceGrid{sdim, C<:Ferrite.AbstractCell, T<:Real, CellDataType <: AbstractVector{C}, NodeDataType <: AbstractVector} <: Ferrite.AbstractGrid{sdim}
    cells::CellDataType
    nodes::NodeDataType
    #TODO subdomain info
end

function DeviceGrid(
    cells::CellDataType,
    nodes::NodeDataType
) where {C<:Ferrite.AbstractCell,CellDataType<:AbstractArray{C,1},NodeDataType<:AbstractArray{Node{dim,T}}} where {dim,T}
    return DeviceGrid{dim,C,T,CellDataType,NodeDataType}(cells, nodes)
end

Ferrite.get_coordinate_type(::DeviceGrid{sdim, <:Any, T,<:Any,<:Any}) where {sdim, T} = Vec{sdim, T} # Node is baked into the mesh type.

@inline Ferrite.getcells(grid::DeviceGrid, v::Ti) where {Ti <: Integer} = grid.cells[v]
@inline Ferrite.getcells(grid::DeviceGrid, v::Int) = grid.cells[v] # to pass ambiguity test
@inline Ferrite.getnodes(grid::DeviceGrid, v::Integer) = grid.nodes[v]
@inline Ferrite.getnodes(grid::DeviceGrid, v::Int) = grid.nodes[v]


function _getcoordinates(grid::DeviceGrid, e::Ti) where {Ti<: Integer}
    # e is the element index.
    CT = Ferrite.get_coordinate_type(grid)
    cell = getcells(grid, e)
    N = nnodes(cell)
    x = MVector{N, CT}(undef) # local array to store the coordinates of the nodes of the cell.
    node_ids = get_node_ids(cell)
    for i in 1:length(x)
        x[i] = get_node_coordinate(grid, node_ids[i])
    end

    return SVector(x...)
end

Ferrite.getcoordinates(grid::DeviceGrid, e::Integer) = _getcoordinates(grid, e)
Ferrite.getcoordinates(grid::DeviceGrid, e::Int) = _getcoordinates(grid, e) # to pass ambiguity test
