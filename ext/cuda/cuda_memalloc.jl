# This file contains the cuda implementation for `device_memalloc.jl` in the `ferrite-addons` module.

#############################
# Shared Memory Allocation  #
#############################

# since shared memory allocation has to be done at the kernel level, 
#we need to define a function that will be called at the kernel level
struct DynamicSharedMemFunction{N, Tv <: Real, Ti <: Integer}
    mem_size::NTuple{N, Ti} # e.g. (3rd order tensor, 2nd order tensor)
    offset::Ti # nonzero when memory shape is of type KeFe 
end


function (dsf::DynamicSharedMemFunction{N, Tv, Ti})() where {N, Tv, Ti}
    (; mem_size, offset) = dsf
    return CUDA.@cuDynamicSharedMem(Tv, mem_size, offset)
end

struct KeFeSharedMem{N, M, Tv <: Real, Ti <: Integer} <: AbstractDeviceSharedMem
    Ke::DynamicSharedMemFunction{N, Tv, Ti} ## block level allocation (i.e. each block will execute this function)
    fe::DynamicSharedMemFunction{M, Tv, Ti} ## block level allocation (i.e. each block will execute this function)
    tot_mem_size::Ti
end

struct FeSharedMem{N, Tv <: Real, Ti <: Integer} <: AbstractDeviceSharedMem
    fe::DynamicSharedMemFunction{N, Tv, Ti} ## block level allocation (i.e. each block will execute this function)
    tot_mem_size::Ti
end

struct KeSharedMem{N, Tv <: Real, Ti <: Integer} <: AbstractDeviceSharedMem
    Ke::DynamicSharedMemFunction{N, Tv, Ti} ## block level allocation (i.e. each block will execute this function)
    tot_mem_size::Ti
end


function _can_use_dynshmem(required_shmem::Integer)
    dev = device()
    MAX_DYN_SHMEM = CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)
    return required_shmem < MAX_DYN_SHMEM
end


function _try_allocate_shared_mem(::Type{KeFeMemShape{Tv}}, block_dim::Ti, n_basefuncs::Ti) where {Ti <: Integer, Tv <: Real}
    shared_mem = convert(Ti, sizeof(Tv) * (block_dim) * (n_basefuncs) * n_basefuncs + sizeof(Tv) * (block_dim) * n_basefuncs)
    _can_use_dynshmem(shared_mem) || return nothing
    Ke = DynamicSharedMemFunction{3, Tv, Ti}((block_dim, n_basefuncs, n_basefuncs), convert(Ti, 0))
    fe = DynamicSharedMemFunction{2, Tv, Ti}((block_dim, n_basefuncs), convert(Ti, sizeof(Tv) * block_dim * n_basefuncs * n_basefuncs))
    return KeFeSharedMem(Ke, fe, shared_mem)
end


function _try_allocate_shared_mem(::Type{FeMemShape{Tv}}, block_dim::Ti, n_basefuncs::Ti) where {Ti <: Integer, Tv <: Real}
    shared_mem = convert(Ti, sizeof(Tv) * (block_dim) * n_basefuncs)
    _can_use_dynshmem(shared_mem) || return nothing
    fe = DynamicSharedMemFunction{2, Tv, Ti}((block_dim, n_basefuncs), convert(Ti, 0))
    return FeSharedMem( fe, shared_mem)
end

function _try_allocate_shared_mem(::Type{KeMemShape{Tv}}, block_dim::Ti, n_basefuncs::Ti) where {Ti <: Integer, Tv <: Real}
    shared_mem = convert(Ti, sizeof(Tv) * (block_dim) * (n_basefuncs) * n_basefuncs)
    _can_use_dynshmem(shared_mem) || return nothing
    Ke = DynamicSharedMemFunction{3, Tv, Ti}((block_dim, n_basefuncs, n_basefuncs), convert(Ti, 0))
    return KeSharedMem(Ke, shared_mem)
end

#############################
# Global Memory Allocation  #
#############################

struct KeFeGlobalMem{MatricesType, VectorsType} <: AbstractDeviceGlobalMem
    Kes::MatricesType ## global level allocation (i.e. memory for all blocks -> 3rd order tensor)
    fes::VectorsType  ## global level allocation (i.e. memory for all blocks -> 2nd order tensor)
end

struct FeGlobalMem{VectorsType} <: AbstractDeviceGlobalMem
    fes::VectorsType  ## global level allocation (i.e. memory for all blocks -> 2nd order tensor)
end

struct KeGlobalMem{MatricesType} <: AbstractDeviceGlobalMem
    Kes::MatricesType ## global level allocation (i.e. memory for all blocks -> 3rd order tensor)
end


function _allocate_global_mem(::Type{KeFeMemShape{Tv}}, nactive_cells::Ti, n_basefuncs::Ti) where {Ti <: Integer, Tv <: Real}
    # allocate memory for the active cells only (i.e. nblocks * threads)
    Kes = CUDA.zeros(Tv, nactive_cells, n_basefuncs, n_basefuncs) 
    fes = CUDA.zeros(Tv, nactive_cells, n_basefuncs)
    return KeFeGlobalMem(Kes, fes)
end

function _allocate_global_mem(::Type{FeMemShape{Tv}}, nactive_cells::Ti, n_basefuncs::Ti) where {Ti <: Integer, Tv <: Real}
    fes = CUDA.zeros(Tv, nactive_cells, n_basefuncs)
    return FeGlobalMem( fes)
end

function _allocate_global_mem(::Type{KeMemShape{Tv}}, nactive_cells::Ti, n_basefuncs::Ti) where {Ti <: Integer, Tv <: Real}
    Kes = CUDA.zeros(Tv, nactive_cells, n_basefuncs, n_basefuncs) 
    return KeGlobalMem(Kes)
end

function Thunderbolt.FerriteUtils.allocate_device_mem(::Type{MemShape}, threads::Ti,blocks::Ti, n_basefuncs::Ti) where {Ti <: Integer, Tv <: Real,MemShape<:AbstractMemShape{Tv}}
    shared_mem_alloc = _try_allocate_shared_mem(MemShape, threads, n_basefuncs)
    shared_mem_alloc isa Nothing && return _allocate_global_mem(MemShape, threads*blocks, n_basefuncs)
    return shared_mem_alloc
end


##########################
# Cell Memory Allocation #
##########################

function Thunderbolt.FerriteUtils.cellmem(global_mem::KeFeGlobalMem, i::Integer)
    ke = @view global_mem.Kes[i, :, :]
    fe = @view global_mem.fes[i, :]
    return KeFeCellMem(ke, fe)
end
function Thunderbolt.FerriteUtils.cellmem(global_mem::FeGlobalMem, i::Integer)
    fe = @view global_mem.fes[i, :]
    return FeCellMem(fe)
end
function Thunderbolt.FerriteUtils.cellmem(global_mem::KeGlobalMem, i::Integer)
    ke = @view global_mem.Kes[i, :, :]
    return KeCellMem(ke)
end


function Thunderbolt.FerriteUtils.cellmem(shared_mem::KeFeSharedMem, i::Integer)
    block_ke = shared_mem.Ke()
    block_fe = shared_mem.fe()
    ke = @view block_ke[i, :, :] 
    fe = @view block_fe[i, :]
    return KeFeCellMem(ke, fe)
end
function Thunderbolt.FerriteUtils.cellmem(shared_mem::FeSharedMem, i::Integer)
    block_fe = shared_mem.fe()
    fe = @view block_fe[i, :]
    return FeCellMem(fe)
end
function Thunderbolt.FerriteUtils.cellmem(shared_mem::KeSharedMem, i::Integer)
    block_ke = shared_mem.Ke()
    ke = @view block_ke[i, :, :] 
    return KeCellMem(ke)
end
