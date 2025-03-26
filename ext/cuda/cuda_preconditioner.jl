#########################################
## CUDA L1 Gauss Seidel Preconditioner ##
#########################################

struct CudaPartitioning{Ti} <: AbstractPartitioning 
    threads::Ti # number of diagonals per partition
    blocks::Ti # number of partitions
    size_A::Ti
end

function Thunderbolt.build_l1prec(::CudaL1PrecBuilder, A::AbstractSparseMatrix;
    n_threads::Union{Integer,Nothing}=nothing, n_blocks::Union{Integer,Nothing}=nothing)
    if CUDA.functional()
        # Raise error if invalid thread or block count is provided
        if !isnothing(n_threads) && n_threads == 0
            error("n_threads must be greater than zero")
        end
        if !isnothing(n_blocks) && n_blocks == 0
            error("n_blocks must be greater than zero")
        end
        return _build_cuda_l1prec(A, n_threads, n_blocks)
    else
        error("CUDA is not functional, please check your GPU driver and CUDA installation")
    end
end

(builder::CudaL1PrecBuilder)(A::AbstractSparseMatrix;
    n_threads::Union{Integer,Nothing}=nothing, n_blocks::Union{Integer,Nothing}=nothing) = build_l1prec(builder, A; n_threads=n_threads, n_blocks=n_blocks)

function _build_cuda_l1prec(A::AbstractSparseMatrix, n_threads::Union{Integer,Nothing}, n_blocks::Union{Integer,Nothing})
    # Determine threads and blocks if not provided
    # blocks -> number of partitions
    # threads -> number of diagonals per partition
    size_A = convert(Int32,size(A, 1))
    threads = isnothing(n_threads) ? convert(Int32, min(size_A, 256)) : convert(Int32, n_threads)
    blocks = isnothing(n_blocks) ? _calculate_nblocks(threads, size_A) : convert(Int32, n_blocks)
    partitioning = CudaPartitioning(threads, blocks, size_A)
    cuda_A = _cuda_A(A)
    return L1Preconditioner(partitioning, cuda_A)
end

_cuda_A(A::SparseMatrixCSC) = CUSPARSE.CuSparseMatrixCSC(A)
_cuda_A(A::SparseMatrixCSR) = CUSPARSE.CuSparseMatrixCSR(A)
_cuda_A(A::CUSPARSE.CuSparseMatrixCSC) = A
_cuda_A(A::CUSPARSE.CuSparseMatrixCSR) = A

# TODO: should x & b be CuArrays? or leave them as AbstractVector?
function LinearSolve.ldiv!(y::VectorType, P::L1Preconditioner{CudaPartitioning{Ti}}, x::VectorType) where {Ti, VectorType <: AbstractVector}
    # x: residual
    # y: preconditioned residual
    y .= x #works either way, whether x is CuArray or Vector
    _ldiv!(y, P)
end

function _ldiv!(y::CuVector , P::L1Preconditioner{CudaPartitioning{Ti}}) where {Ti}
    @unpack partitioning, A = P
    @unpack threads, blocks, size_A = partitioning
    issym = isapprox(A, A',rtol=1e-12)
    CUDA.@sync CUDA.@cuda threads=threads blocks=blocks _l1prec_kernel!(y, A,issym)
    return nothing
end

function _ldiv!(y::Vector , P::L1Preconditioner{CudaPartitioning{Ti}}) where {Ti}
    @unpack partitioning, A = P
    @unpack threads, blocks, size_A = partitioning
    cy  = y |> cu
    issym = isapprox(A, A',rtol=1e-12)
    CUDA.@sync CUDA.@cuda threads=threads blocks=blocks _l1prec_kernel!(cy, A,issym)
    copyto!(y, cy)
    return nothing
end

abstract type AbstractMatrixSymmetry end

struct SymmetricMatrix <: AbstractMatrixSymmetry end # important for the case of CSC format
struct NonSymmetricMatrix <: AbstractMatrixSymmetry end

# TODO: consider creating unified iterator for both CPU and GPU.
struct DeviceDiagonalIterator{MatrixType, MatrixSymmetry  <: AbstractMatrixSymmetry}
    A::MatrixType
end

struct DeviceDiagonalCache{Ti,Tv}
    k::Ti # partition index
    idx::Ti # diagonal index
    b::Tv # partition diagonal value
    d::Tv # off-partition absolute sum
end

DiagonalIterator(::Type{SymT}, A::MatrixType) where {SymT <: AbstractMatrixSymmetry, MatrixType} =
    DeviceDiagonalIterator{MatrixType, SymT}(A)

function Base.iterate(iterator::DeviceDiagonalIterator)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x # diagonal index
    idx <= size(iterator.A, 1) || return nothing
    k = blockIdx().x # partition index
    return (_makecache(iterator,idx,k), (idx,k))
end

function Base.iterate(iterator::DeviceDiagonalIterator, state)
    n_blocks = gridDim().x
    n_threads = blockDim().x
    idx,k = state
    k += n_blocks # partition index
    stride = n_blocks * n_threads
    idx = idx + stride # diagonal index
    idx <= size(iterator.A, 1) || return nothing
    return (_makecache(iterator,idx,k), (idx,k))
end

function _makecache(iterator::DeviceDiagonalIterator{CUSPARSE.CuSparseDeviceMatrixCSC{Tv,Ti,1},NonSymmetricMatrix}, idx,k) where {Tv,Ti}
    #Ωⁱ := {j ∈ Ωₖ : i ∈ Ωₖ}
    #Ωⁱₒ := {j ∉ Ωₖ : i ∈ Ωₖ} off-partition column values
    # bₖᵢ := Aᵢᵢ
    # dₖᵢ := ∑_{j ∈ Ωⁱₒ} |Aᵢⱼ|
    n_threads = blockDim().x
    @unpack A = iterator
    part_start_idx = (k - Int32(1)) * n_threads + Int32(1)
    part_end_idx = min(part_start_idx + n_threads - Int32(1), size(A, 2))
    
    b = zero(eltype(A))
    d = zero(eltype(A))

    # specific to CSC format
    for col in 1:size(A, 2)
        col_start = A.colPtr[col]
        col_end = A.colPtr[col+1] - 1

        for i in col_start:col_end
            row = A.rowVal[i]
            if row == idx
                v = A.nzVal[i]

                if part_start_idx > col || col > part_end_idx
                    d += abs(v)
                end

                if col == idx
                    b = v
                end
            end
        end
    end

    return DeviceDiagonalCache(k, idx,b, d)
end

function _makecache(iterator::DeviceDiagonalIterator{CUSPARSE.CuSparseDeviceMatrixCSC{Tv,Ti,1},SymmetricMatrix}, idx,k) where {Tv,Ti}
    #Ωⁱ := {j ∈ Ωₖ : i ∈ Ωₖ}
    #Ωⁱₒ := {j ∉ Ωₖ : i ∈ Ωₖ} off-partition column values
    # bₖᵢ := Aᵢᵢ
    # dₖᵢ := ∑_{j ∈ Ωⁱₒ} |Aᵢⱼ|
    n_threads = blockDim().x
    @unpack A = iterator
    part_start_idx = (k - Int32(1)) * n_threads + Int32(1)
    part_end_idx = min(part_start_idx + n_threads - Int32(1), size(A, 2))
    
    b = zero(eltype(A))
    d = zero(eltype(A))

    # since matrix is symmetric, then both CSC and CSR are the same.
    b,d = _diag_offpart_csr(A.colPtr, A.rowVal, A.nzVal, idx, part_start_idx, part_end_idx)

    return DeviceDiagonalCache(k, idx,b, d)
end

function _makecache(iterator::DeviceDiagonalIterator{CUSPARSE.CuSparseDeviceMatrixCSR{Tv,Ti,1}}, idx,k) where {Tv,Ti}
    #Ωⁱ := {j ∈ Ωₖ : i ∈ Ωₖ}
    #Ωⁱₒ := {j ∉ Ωₖ : i ∈ Ωₖ} off-partition column values
    # bₖᵢ := Aᵢᵢ
    # dₖᵢ := ∑_{j ∈ Ωⁱₒ} |Aᵢⱼ|
    n_threads = blockDim().x
    @unpack A = iterator  # A is in CSR format
    part_start_idx = (k - Int32(1)) * n_threads + Int32(1)
    part_end_idx = min(part_start_idx + n_threads - Int32(1), size(A, 2)) 

    b,d = _diag_offpart_csr(A.rowPtr, A.colVal, A.nzVal, idx, part_start_idx, part_end_idx)

    return DeviceDiagonalCache(k, idx, b, d)
end


function _diag_offpart_csr(rowPtr, colVal, nzVal, idx::Integer, part_start::Integer, part_end::Integer)
    Tv = eltype(nzVal)
    b = zero(Tv)
    d = zero(Tv)

    row_start = rowPtr[idx]
    row_end = rowPtr[idx + 1] - 1

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


function _l1prec_kernel!(y, A,issym)
    # this kernel will loop over the corresponding diagonals in strided fashion.
    # e.g. if n_threads = 4, n_blocks = 2, A is (100 x 100), and current global thread id = 5, 
    # then the kernel will loop over the diagonals with stride:
    # k (partition index) = k + n_blocks (i.e. 2, 4, 6, 8, 10, 12, 14)
    # idx (diagonal index) = idx + n_blocks * n_threads (i.e. 5, 13, 21, 29, 37, 45, 53)
    symT = issym ? SymmetricMatrix : NonSymmetricMatrix
    for diagonal in DiagonalIterator(symT,A)
        @unpack k, idx, b, d = diagonal
        @cushow k,d #TODO: remove this line
        y[idx] = y[idx]/ (b + d)  
    end
    return nothing
end
