module CuThunderboltExt

using Thunderbolt
using LinearSolve
using KernelAbstractions
using SparseMatricesCSR

import SparseArrays:SparseMatrixCSC,AbstractSparseMatrix

import CUDA:
    CUDA, CuArray, CuVector, CUSPARSE,blockDim,blockIdx,gridDim,threadIdx,
    threadIdx, blockIdx, blockDim, @cuda, @cushow,
    CUDABackend, launch_configuration, cu,cudaconvert

import Thunderbolt:
    UnPack.@unpack,
    SimpleMesh,
    SparseMatrixCSR, SparseMatrixCSC,
    AbstractSolver, AbstractSemidiscreteFunction, AbstractPointwiseFunction, solution_size,
    AbstractPointwiseSolverCache,assemble_element!,
    LinearIntegrator,LinearOperator,QuadratureRuleCollection,
    setup_element_cache,update_operator!,FieldCoefficientCache, CudaDevice, ElementAssemblyStrategy, value_type, index_type, 
    convert_vec_to_concrete

import Thunderbolt.FerriteUtils:
    StaticInterpolationValues,StaticCellValues, allocate_device_mem,
    CellIterator, mem_size, cellmem,ncells,celldofsview,
    DeviceDofHandlerData, DeviceSubDofHandler, DeviceDofHandler, DeviceGrid,
    cellfe, AbstractDeviceGlobalMem, AbstractDeviceSharedMem,AbstractDeviceCellIterator,AbstractCellMem,
    FeMemShape, KeMemShape, KeFeMemShape, DeviceCellIterator,DeviceOutOfBoundCellIterator,DeviceCellCache,
    FeCellMem, KeCellMem, KeFeCellMem,NoCellMem,AbstractMemShape

import Thunderbolt.Preconditioners:
    sparsemat_format_type, CSCFormat, CSRFormat


import Ferrite:
    AbstractDofHandler,get_grid,CellIterator,get_node_coordinate,getcoordinates,get_coordinate_eltype,getcells,
    get_node_ids,get_coordinate_type,nnodes

import StaticArrays:
    SVector,MVector
    
import Adapt:
    Adapt, adapt_structure, adapt, @adapt_structure


# ---------------------- Generic part ------------------------

# Pointwise cuda solver wrapper
function _gpu_pointwise_step_inner_kernel_wrapper!(f, t, Δt, cache::AbstractPointwiseSolverCache)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    i > size(cache.dumat, 1) && return nothing
    Thunderbolt._pointwise_step_inner_kernel!(f, i, t, Δt, cache)
    return nothing
end

# This controls the outer loop over the ODEs
function Thunderbolt._pointwise_step_outer_kernel!(f::AbstractPointwiseFunction, t::Real, Δt::Real, cache::AbstractPointwiseSolverCache, ::Union{<:CuVector, SubArray{<:Any,1,<:CuVector}})
    kernel = @cuda launch=false _gpu_pointwise_step_inner_kernel_wrapper!(f.ode, t, Δt, cache) # || return false
    config = launch_configuration(kernel.fun)
    threads = min(f.npoints, config.threads)
    blocks =  cld(f.npoints, threads)
    kernel(f.ode, t, Δt, cache;  threads, blocks)
    return true
end

_allocate_matrix(dh::DeviceDofHandler, A::SparseMatrixCSR, ::CuVector) = CuSparseMatrixCSR(A)
_allocate_matrix(dh::DeviceDofHandler, A::SparseMatrixCSC, ::CuVector) = CuSparseMatrixCSC(A)

Thunderbolt.create_system_vector(::Type{<:CuVector{T}}, f::AbstractSemidiscreteFunction) where T = CUDA.zeros(T, solution_size(f))
Thunderbolt.create_system_vector(::Type{<:CuVector{T}}, dh::DofHandler) where T                  = CUDA.zeros(T, ndofs(dh))

function Thunderbolt.create_system_matrix(SpMatType::Type{<:Union{CUSPARSE.CuSparseMatrixCSC, CUSPARSE.CuSparseMatrixCSR}}, dh::AbstractDofHandler)
    # FIXME in general the pattern is not symmetric
    Acpu      = allocate_matrix(dh)
    colptrgpu = CuArray(Acpu.colptr)
    rowvalgpu = CuArray(Acpu.rowval)
    nzvalgpu  = CuArray(Acpu.nzval)
    return SpMatType(colptrgpu, rowvalgpu, nzvalgpu, (Acpu.m, Acpu.n))
end

Thunderbolt.__add_to_vector!(b::Vector, a::CuVector) = b .+= Vector(a)
Thunderbolt.__add_to_vector!(b::CuVector, a::Vector) = b .+= CuVector(a)

function Thunderbolt.adapt_vector_type(::Type{<:CuVector}, v::VT) where {VT <: Vector}
    return CuVector(v)
end

include("cuda/cuda_operator.jl")
include("cuda/cuda_memalloc.jl")
include("cuda/cuda_adapt.jl")
include("cuda/cuda_iterator.jl")
include("cuda/cuda_preconditioner.jl") #NOTE: min version for CUDA is v"5.7.3"

end
