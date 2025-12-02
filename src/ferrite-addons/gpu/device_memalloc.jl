###########################
# Memory Allocation Shape #
###########################

# some type definitions to represent shape of memory to be allocated.
abstract type AbstractMemShape{Tv <: Real} end
struct FeMemShape{Tv<:Real} <:AbstractMemShape{Tv}  end
struct KeMemShape{Tv<:Real} <:AbstractMemShape{Tv}  end
struct KeFeMemShape{Tv<:Real} <:AbstractMemShape{Tv}  end


###########################
# Group Memory Allocation #
###########################

# some abstract types to represent group memory allocation on the different devices 
# two types of memory allocation are considered: shared memory and global memory
# group memory: these are the memory allocation for the entire group (e.g. shared memory -> one per block , global memory -> one per grid)
# those memory are gonna further reduced to cell memory (i.e. memory per cell) in the device iterator
# each of which can allocate different shapes of memory (e.g. Ke, Fe, KeFe)
abstract type AbstractDeviceGroupMem  end
abstract type AbstractDeviceSharedMem <: AbstractDeviceGroupMem end
abstract type AbstractDeviceGlobalMem <: AbstractDeviceGroupMem end

# interfaces
mem_size(shared_mem::AbstractDeviceSharedMem) = shared_mem.tot_mem_size

cellmem(shared_mem::AbstractDeviceSharedMem, ::Integer) = error("please provide concrete implementation for $(typeof(shared_mem)).")
cellmem(shared_mem::AbstractDeviceGlobalMem, ::Integer) = error("please provide concrete implementation for $(typeof(shared_mem)).")


function allocate_device_mem(mem_shape::Type{AbstractMemShape{Tv}}, n_cells::Ti, n_basefuncs::Ti) where {Ti <: Integer, Tv <: Real}
    error("please provide concrete implementation for $(typeof(mem_shape)).")
end
   
##########################
# Cell Memory Allocation #
##########################

# cell memory: these are the memory allocation for each cell (i.e. memory per cell)
# property in cell cell cache that encompass the cell memory (ke, fe, etc.)
abstract type AbstractCellMem end

# concrete types (independent of the device)
struct FeCellMem{VectorType} <: AbstractCellMem 
    fe::VectorType
end

struct KeCellMem{MatrixType} <: AbstractCellMem 
    ke::MatrixType
end

struct KeFeCellMem{MatrixType, VectorType} <: AbstractCellMem 
    ke::MatrixType
    fe::VectorType
end

struct NoCellMem <: AbstractCellMem end # mainly for testing purposes
