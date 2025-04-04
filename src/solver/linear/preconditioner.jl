####################################
## L1 Gauss Seidel Preconditioner ##
####################################

## Structs ##
abstract type AbstractPartitioning end

struct GpuPartitioning{Ti,Backend} <: AbstractPartitioning
    partsize::Ti # number of diagonals per partition
    nparts::Ti # number of partitions
    size_A::Ti
    backend::Backend # GPU backend 
end

# since we use Polyester.jl for CPU multithreading, which is not supported by KA,
# therefore we need to define a separate partitioning struct for CPU
struct CpuPartitioning{Ti} <: AbstractPartitioning
    partsize::Ti # number of diagonals per partition (i.e. n_threads in CUDA)
    nparts::Ti # number of partitions (i.e. n_blocks in CUDA)
    size_A::Ti
end

struct L1Preconditioner{Partitioning,MatrixType}
    partitioning::Partitioning
    A::MatrixType
end

abstract type AbstractL1PrecBuilder end

struct GpuL1PrecBuilder <: AbstractL1PrecBuilder
    backend::GPU
    function GpuL1PrecBuilder(backend::GPU)
        if functional(backend)
            return new(backend)
        else
            error(" $backend is not functional, please check your GPU driver")
        end
    end
end
struct CpuL1PrecBuilder <: AbstractL1PrecBuilder end

(builder::AbstractL1PrecBuilder)(A::AbstractMatrix,partsize::Ti,nparts::Ti) where {Ti} = 
    build_l1prec(builder, A,partsize, nparts)

# CSR and CSC are exact the same in symmetric matrices,so we need to hold symmetry info
# in order to be exploited in CSC case (i.e. treat CSC as CSR for symmetric matrices).
abstract type AbstractMatrixSymmetry end
struct SymmetricMatrix <: AbstractMatrixSymmetry end 
struct NonSymmetricMatrix <: AbstractMatrixSymmetry end
    
abstract type AbstractDiagonalIterator end
## Naming convention here: 'Device' is usedfor objects that are usedin GPU conext.
## 'Gpu' is used for the objects that are used in CPU context but hold some data for GPU.  
struct DeviceDiagonalIterator{MatrixFormat,MatrixSymmetry  <: AbstractMatrixSymmetry,MatrixType,Ti} <: AbstractDiagonalIterator
    A::MatrixType
    partsize::Ti
    nparts::Ti
    local_idx::Ti # local index never changes
    initial_partition_idx::Ti # why initial ? because we are using stride to loop over the diagonals. 
    initial_global_idx::Ti 
    function DeviceDiagonalIterator(::Type{symT}, A::MatrixType,partsize::Ti,nparts::Ti,local_idx::Ti,initial_partition_idx::Ti) where {symT<:AbstractMatrixSymmetry,MatrixType,Ti}
        format_type = sparsemat_format_type(A) 
        initial_global_idx = (initial_partition_idx - convert(Ti,1)) * partsize + local_idx
        new{format_type,symT,MatrixType,Ti}(A,partsize,nparts,local_idx,initial_partition_idx,initial_global_idx)
    end
end

struct DiagonalIterator{MatrixType,MatrixSymmetry<:AbstractMatrixSymmetry,Ti<:Integer} <: AbstractDiagonalIterator
    A::MatrixType
    k::Ti # partition index
    partsize::Ti # partition size
end

DiagonalIterator(::Type{SymT}, k::Ti, partsize::Ti, A::MatrixType) where {SymT<:AbstractMatrixSymmetry,MatrixType,Ti<:Integer} =
    DiagonalIterator{MatrixType,SymT,Ti}(A, k, partsize)

struct DiagonalCache{Ti,Tv}
    k::Ti # partition index
    idx::Ti # diagonal index
    b::Tv # partition diagonal value
    d::Tv # off-partition absolute sum
end

abstract type AbstractMatrixFormat end
struct CSR <: AbstractMatrixFormat end
struct CSC <: AbstractMatrixFormat end

function _diag_offpart_csr(rowPtr, colVal, nzVal, idx::Integer, part_start::Integer, part_end::Integer)
    Tv = eltype(nzVal)
    b = zero(Tv)
    d = zero(Tv)

    row_start = rowPtr[idx]
    row_end = rowPtr[idx+1] - 1

    for i in row_start:row_end
        col = colVal[i]
        v = nzVal[i]

        if col == idx
            b = v
        elseif col < part_start || col > part_end
            d += abs(v)
        end
    end

    return b, d
end

function _diag_offpart_csc(colPtr, rowVal, nzVal, idx::Integer, part_start::Integer, part_end::Integer)
    Tv = eltype(nzVal)
    b = zero(Tv)
    d = zero(Tv)

    ncols = length(colPtr) - 1

    for col in 1:ncols
        col_start = colPtr[col]
        col_end = colPtr[col+1] - 1

        for i in col_start:col_end
            row = rowVal[i]
            v = nzVal[i]

            if row == idx
                if col == idx
                    b = v
                elseif col < part_start || col > part_end
                    d += abs(v)
                end
            end
        end
    end

    return b, d
end

## KA - GPU ##
function build_l1prec(builder::GpuL1PrecBuilder, A::AbstractSparseMatrix,partsize::Ti, nparts::Ti) where {Ti<:Integer}
    @unpack backend = builder
    partsize == 0 && error("partsize must be greater than 0")
    nparts == 0 && error("nparts must be greater than 0")
    _build_l1prec(backend, A, partsize, nparts)
end

function _build_l1prec(backend::GPU, A::AbstractSparseMatrix,partsize::Ti, nparts::Ti) where {Ti<:Integer}
    # No assumptions on A, i.e. A here might be in either CPU or GPU format, 
    # so we have to convert it to GPU format, if it is not already.
    gpu_A = convert_to_backend(A, backend)
    partitioning = GpuPartitioning(partsize, nparts, size(A, 1), backend)
    L1Preconditioner(partitioning, gpu_A)
end

function convert_to_backend(::AbstractSparseMatrix, ::GPU)
    # this functions needs to be extended to support all the sparse matrix formats in different backends.
    # e.g. `convert_to_backend(A::SparseMatrixCSC, ::CUDABackend)` should return `A |> cu` object.
    # `convert_to_backend(A::CuSparseMatrixCSC, ::CUDABackend)` should return `A` object.
    error("Not implemented")
end

function convert_to_backend(::AbstractVector, ::GPU)
    # this functions needs to be extended to support all vector types in different backends.
    # e.g. `convert_to_backend(x::Vector, ::CUDABackend)` should return `x |> cu` object.
    # `convert_to_backend(x::CuVector, ::CUDABackend)` should return `x` object.
    error("Not implemented")
end

function LinearSolve.ldiv!(y::VectorType, P::L1Preconditioner{GpuPartitioning{Ti,Backend}}, x::VectorType) where {VectorType <: AbstractVector, Ti, Backend}
    # x: residual
    # y: preconditioned residual
    y .= x #works either way, whether x is GpuVectorType (e.g. CuArray) or Vector
    _ldiv!(y, P)
end

function _ldiv!(y::VectorType , P::L1Preconditioner{GpuPartitioning{Ti,Backend}})  where {VectorType <: AbstractVector, Ti, Backend}
    @unpack partitioning, A = P
    @unpack partsize, nparts, size_A, backend = partitioning
    gpu_y = convert_to_backend(y, backend)
    issym = isapprox(A, A',rtol=1e-12)
    # workgroupsize ⇔ n_threads
    # ndrange ⇔ total number of threads
    # ndrange = partsize * nparts
    # Notes:
    #  1. In GPU implementation: n_threads = partsize, n_blocks = nparts 
    #  2. ndrange doesn't need to be equal `size_A`, because for kernel performance reasons one might 
    #     launch kernel with specific n_threads, n_blocks.
    ndrange = partsize * nparts
    kernel = _l1prec_kernel!(backend,partsize,ndrange)
    kernel(gpu_y, A, issym,partsize,nparts; ndrange=ndrange)
    copyto!(y, gpu_y)
    return nothing
end

function sparsemat_format_type(::AbstractSparseMatrix)
    error("Not implemented")
end

function Base.iterate(iterator::DeviceDiagonalIterator)
    @unpack A, initial_partition_idx, initial_global_idx = iterator
    idx = initial_global_idx # diagonal index
    idx <= size(A, 1) || return nothing
    k = initial_partition_idx # partition index
    return (_makecache(iterator,idx,k), (idx,k))
end

function Base.iterate(iterator::DeviceDiagonalIterator, state)
    @unpack A, partsize, nparts = iterator
    total_work_items = partsize * nparts  # stride = n_threads * n_blocks
    idx,k = state
    k += nparts # partition index
    idx = idx + total_work_items # diagonal index
    idx <= size(A, 1) || return nothing
    return (_makecache(iterator,idx,k), (idx,k))
end


_makecache(iterator::DeviceDiagonalIterator{CSC,NonSymmetricMatrix}, idx,k) = 
    _makecache(iterator,idx,k,_diag_offpart_csc)


 _makecache(iterator::DeviceDiagonalIterator{CSC,SymmetricMatrix}, idx,k) = 
    _makecache(iterator,idx,k,_diag_offpart_csr)


function _makecache(iterator::DeviceDiagonalIterator{CSC}, idx::Ti,k::Ti, diag_offpart_func) where {Ti<:Integer} 
    #Ωⁱ := {j ∈ Ωₖ : i ∈ Ωₖ}
    #Ωⁱₒ := {j ∉ Ωₖ : i ∈ Ωₖ} off-partition column values
    # bₖᵢ := Aᵢᵢ
    # dₖᵢ := ∑_{j ∈ Ωⁱₒ} |Aᵢⱼ|
    @unpack A,partsize = iterator
    part_start_idx = (k - convert(Ti,1)) * partsize + convert(Ti,1)
    part_end_idx = min(part_start_idx + partsize - convert(Ti,1), size(A, 2))
    
    b,d = diag_offpart_func(A.colPtr, A.rowVal, A.nzVal, idx, part_start_idx, part_end_idx)

    return DiagonalCache(k, idx,b, d)
end

function _makecache(iterator::DeviceDiagonalIterator{CSR}, idx::Ti,k::Ti) where {Ti<:Integer}
    #Ωⁱ := {j ∈ Ωₖ : i ∈ Ωₖ}
    #Ωⁱₒ := {j ∉ Ωₖ : i ∈ Ωₖ} off-partition column values
    # bₖᵢ := Aᵢᵢ
    # dₖᵢ := ∑_{j ∈ Ωⁱₒ} |Aᵢⱼ|
    @unpack A,partsize = iterator  # A is in CSR format
    part_start_idx = (k - convert(Ti,1)) * partsize + convert(Ti,1)
    part_end_idx = min(part_start_idx + partsize - convert(Ti,1), size(A, 2)) 

    b,d = _diag_offpart_csr(A.rowPtr, A.colVal, A.nzVal, idx, part_start_idx, part_end_idx)

    return DiagonalCache(k, idx, b, d)
end

@kernel function _l1prec_kernel!(y, A,issym,partsize::Ti,nparts::Ti) where {Ti<:Integer}
    # this kernel will loop over the corresponding diagonals in strided fashion.
    # e.g. if partsize = 4, nparts = 2, A is (100 x 100), and current global thread id = 5, 
    # then the kernel will loop over the diagonals with stride:
    # k (partition index) = k + nparts (i.e. 2, 4, 6, 8, 10, 12, 14)
    # idx (diagonal index) = idx + nparts * partsize (i.e. 5, 13, 21, 29, 37, 45, 53)
    symT = issym ? SymmetricMatrix : NonSymmetricMatrix
    local_idx = convert(Ti,@index(Local))
    initial_partition_idx = convert(Ti,@index(Group)) 
    for diagonal in DeviceDiagonalIterator(symT,A,partsize,nparts,local_idx,initial_partition_idx)
        @unpack k, idx, b, d = diagonal
        y[idx] = y[idx]/ (b + d)  
    end
end


## Polyester - CPU ##
function build_l1prec(::CpuL1PrecBuilder, A::AbstractSparseMatrix; partsize::Union{Integer,Nothing}=nothing)
    partsize = isnothing(partsize) ? 1 : partsize
    nparts = ceil(Int, size(A, 1) / partsize)
    partitioning = CpuPartitioning(partsize, nparts, size(A, 1))
    return L1Preconditioner(partitioning, A)
end

(builder::CpuL1PrecBuilder)(A::AbstractSparseMatrix; partsize::Union{Integer,Nothing}=nothing) = build_l1prec(builder, A; partsize=partsize)

function LinearSolve.ldiv!(y::VectorType, P::L1Preconditioner{CpuPartitioning{Ti}}, x::VectorType) where {Ti,VectorType<:AbstractVector}
    # x: residual
    # y: preconditioned residual
    y .= x #works either way, whether x is CuArray or Vector
    _l1prec!(y, P, isapprox(P.A, P.A', rtol=1e-12))
end

function Base.iterate(iterator::DiagonalIterator, state=1)
    @unpack A, k, partsize = iterator
    idx = (k - 1) * partsize + state
    (idx <= size(A, 1) && state <= partsize) || return nothing
    return (_makecache(iterator, idx), state + 1)
end


_makecache(iterator::DiagonalIterator{SparseMatrixCSC{Tv,Ti},NonSymmetricMatrix}, idx) where {Tv,Ti} =
    _makecache(iterator, idx, _diag_offpart_csc)


_makecache(iterator::DiagonalIterator{SparseMatrixCSC{Tv,Ti},SymmetricMatrix}, idx) where {Tv,Ti} =
    _makecache(iterator, idx, _diag_offpart_csr)


function _makecache(iterator::DiagonalIterator{SparseMatrixCSC{Tv,Ti}}, idx, diag_offpart_func) where {Tv,Ti}
    #Ωⁱ := {j ∈ Ωₖ : i ∈ Ωₖ}
    #Ωⁱₒ := {j ∉ Ωₖ : i ∈ Ωₖ} off-partition column values
    # bₖᵢ := Aᵢᵢ
    # dₖᵢ := ∑_{j ∈ Ωⁱₒ} |Aᵢⱼ|
    @unpack A, k, partsize = iterator
    part_start_idx = (k - 1) * partsize + 1
    part_end_idx = min(part_start_idx + partsize - 1, size(A, 2))

    # since matrix is symmetric, then both CSC and CSR are the same.
    b, d = diag_offpart_func(A.colptr, A.rowval, A.nzval, idx, part_start_idx, part_end_idx)

    return DiagonalCache(k, idx, b, d)
end

function _makecache(iterator::DiagonalIterator{SparseMatrixCSR{1,Tv,Ti}}, idx) where {Tv,Ti}
    #Ωⁱ := {j ∈ Ωₖ : i ∈ Ωₖ}
    #Ωⁱₒ := {j ∉ Ωₖ : i ∈ Ωₖ} off-partition column values
    # bₖᵢ := Aᵢᵢ
    # dₖᵢ := ∑_{j ∈ Ωⁱₒ} |Aᵢⱼ|
    @unpack A, k, partsize = iterator
    part_start_idx = (k - 1) * partsize + 1
    part_end_idx = min(part_start_idx + partsize - 1, size(A, 2))

    b, d = _diag_offpart_csr(A.rowptr, A.colval, A.nzval, idx, part_start_idx, part_end_idx)

    return DiagonalCache(k, idx, b, d)
end

function _l1prec!(y, P, issym)
    @unpack partitioning, A = P
    @unpack partsize, nparts = partitioning
    symT = issym ? SymmetricMatrix : NonSymmetricMatrix
    @batch for part in 1:nparts
        # diagonl iterator here is different than the one in GPU implementation.
        # here, there are no strides, and the iterator is just looping over the diagonals in the corresponding partition. 
        for diagonal in DiagonalIterator(symT, part, partsize, A)
            @unpack k, idx, b, d = diagonal
            y[idx] = y[idx] / (b + d)
        end
    end
    return nothing
end
