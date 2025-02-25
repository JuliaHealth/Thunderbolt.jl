
# This file contains the cuda implementation for `device_iterator.jl` in the `ferrite-addons` module.

# iterator with no memory allocation (for testing purposes)
function _build_cell_iterator(sdh::DeviceSubDofHandler,n_cells::Integer)
    bd = blockDim().x
    local_thread_id = threadIdx().x
    global_thread_id = (blockIdx().x - Int32(1)) * bd + local_thread_id
    global_thread_id <= n_cells || return DeviceOutOfBoundCellIterator()
    cell_mem = NoCellMem()
    return  DeviceCellIterator(sdh, n_cells, cell_mem)
end

# iterator with global memory allocation
function _build_cell_iterator(sdh::DeviceSubDofHandler,n_cells::Integer, global_mem::AbstractDeviceGlobalMem)
    bd = blockDim().x
    local_thread_id = threadIdx().x
    global_thread_id = (blockIdx().x - Int32(1)) * bd + local_thread_id 
    global_thread_id <= n_cells || return DeviceOutOfBoundCellIterator()
    cell_mem = cellmem(global_mem, global_thread_id)
    return DeviceCellIterator(sdh, n_cells, cell_mem)
end

# iterator with shared memory allocation
function _build_cell_iterator(sdh::DeviceSubDofHandler,n_cells::Integer, buffer_alloc::AbstractDeviceSharedMem)
    local_thread_id = threadIdx().x
    cell_mem = cellmem(buffer_alloc, local_thread_id)
    return DeviceCellIterator(sdh, n_cells, cell_mem)
end

# Global memory allocation
function Ferrite.CellIterator(sdh::DeviceSubDofHandler{<:Ti}, buffer_alloc::AbstractDeviceGlobalMem) where {Ti <: Integer}
    ## iterate over all cells in the subdomain
    n_cells = sdh.cellset |> length |> (x -> convert(Ti, x)) 
    return _build_cell_iterator(sdh,n_cells,buffer_alloc)
end
# No memory allocation (for testing purposes)
function Ferrite.CellIterator(sdh::DeviceSubDofHandler{<:Ti}) where {Ti <: Integer}
    ## iterate over all cells in the subdomain
    # check if the subdomain index is valid
    n_cells =sdh.cellset |> length |> (x -> convert(Ti, x)) 
    return _build_cell_iterator(sdh,n_cells)
end

# Shared memory allocation
function Ferrite.CellIterator(sdh::DeviceSubDofHandler{Ti}, buffer_alloc::AbstractDeviceSharedMem) where {Ti <: Integer}
    ## iterate over all cells in the subdomain
    # check if the subdomain index is valid
    n_cells = sdh.cellset |> length |> (x -> convert(Ti, x)) 
    return _build_cell_iterator(sdh, n_cells, buffer_alloc)
end
   
function Base.iterate(iterator::DeviceCellIterator)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    i <= ncells(iterator)  || return nothing
    return (_makecache(iterator, i), i)
end

function Base.iterate(iterator::DeviceCellIterator, state)
    # state is the global thread index
    stride = blockDim().x * gridDim().x
    i = state + stride
    i <= ncells(iterator) || return nothing
    return (_makecache(iterator, i), i)
end


Base.iterate(::DeviceOutOfBoundCellIterator) = nothing

Base.iterate(::DeviceOutOfBoundCellIterator, state) = nothing # I believe this is not necessary


# cuda cell cache #
function _makecache(iterator::AbstractDeviceCellIterator, e::Ti) where {Ti <: Integer}
    # NOTE: e is the global thread index, so in case of iterating over the whole cells in the grid, e is the cell index
    # whereas in the case of we are only iterating over a subdomain, e is just the iterator index from 1:n_cells in subdomain
    sdh = iterator.sdh
    grid = get_grid(sdh)
    cellid = sdh.cellset[e]
    cell = getcells(grid, cellid)

    #Extract the node IDs of the cell.
    nodes = SVector(convert.(Ti, get_node_ids(cell))...)

    # Extract the degrees of freedom for the cell.
    dofs = celldofsview(sdh, cellid)

    N = nnodes(cell)
    coords = SVector((get_node_coordinate(grid, nodes[i]) for i in 1:N)...)
    
    # Return the DeviceCellCache containing the cell's data.
    return  DeviceCellCache(coords, dofs, cellid, nodes, iterator.cell_mem)
end

@inline function _cellke(cell_mem::AbstractCellMem)
    ke =  cell_mem.ke
    FT = eltype(ke)
    return CUDA.fill!(ke, zero(FT))
end

_cellke(cell_mem::FeCellMem) = error("$(typeof(cell_mem)) does not have ke field.")

Thunderbolt.FerriteUtils.cellke(cc::DeviceCellCache) = _cellke(cc.cell_mem)

@inline function _cellfe(cell_mem::AbstractCellMem)
    fe =  cell_mem.fe
    FT = eltype(fe)
    return CUDA.fill!(fe, zero(FT))
end

_cellfe(cell_mem::KeCellMem) = error("$(typeof(cell_mem)) does not have fe field.")

Thunderbolt.FerriteUtils.cellfe(cc::DeviceCellCache) = _cellfe(cc.cell_mem)
