####################################
## L1 Gauss Seidel Preconditioner ##
####################################

## Interfaces for the L1 Gauss Seidel preconditioner
abstract type AbstractPartitioning end

struct L1Preconditioner{Partitioning,MatrixType}
    partitioning::Partitioning
    A::MatrixType
end


LinearSolve.ldiv!(::VectorType, ::L1Preconditioner{Partitioning}, ::VectorType) where {VectorType<:AbstractVector,Partitioning} =
    error("Not implemented")

abstract type AbstractL1PrecBuilder end
struct CudaL1PrecBuilder <: AbstractL1PrecBuilder end
struct CpuL1PrecBuilder <: AbstractL1PrecBuilder end

function build_l1prec(::AbstractL1PrecBuilder, ::AbstractMatrix)
    error("Not implemented")
end

(builder::AbstractL1PrecBuilder)(A::AbstractMatrix) = build_l1prec(builder, A)


## Shared code for cpu and gpu implementations
abstract type AbstractDiagonalIterator end
abstract type AbstractMatrixSymmetry end

struct SymmetricMatrix <: AbstractMatrixSymmetry end # important for the case of CSC format
struct NonSymmetricMatrix <: AbstractMatrixSymmetry end

struct DiagonalCache{Ti,Tv}
    k::Ti # partition index
    idx::Ti # diagonal index
    b::Tv # partition diagonal value
    d::Tv # off-partition absolute sum
end

function diag_offpart_csr(rowPtr, colVal, nzVal, idx::Integer, part_start::Integer, part_end::Integer)
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

function diag_offpart_csc(colPtr, rowVal, nzVal, idx::Integer, part_start::Integer, part_end::Integer)
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


## CPU Multithreaded implementation for the L1 Gauss Seidel preconditioner

struct CpuPartitioning{Ti} <: AbstractPartitioning
    partsize::Ti # number of diagonals per partition
    nparts::Ti # number of partitions
    size_A::Ti
end

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

struct DiagonalIterator{MatrixType,MatrixSymmetry<:AbstractMatrixSymmetry,Ti<:Integer} <: AbstractDiagonalIterator
    A::MatrixType
    k::Ti # partition index
    partsize::Ti # partition size
end

DiagonalIterator(::Type{SymT}, k::Ti, partsize::Ti, A::MatrixType) where {SymT<:AbstractMatrixSymmetry,MatrixType,Ti<:Integer} =
    DiagonalIterator{MatrixType,SymT,Ti}(A, k, partsize)

function Base.iterate(iterator::DiagonalIterator, state=1)
    @unpack A, k, partsize = iterator
    idx = (k - 1) * partsize + state
    (idx <= size(A, 1) && state <= partsize) || return nothing
    return (_makecache(iterator, idx), state+1)
end


_makecache(iterator::DiagonalIterator{SparseMatrixCSC,NonSymmetricMatrix}, idx) =
    _makecache(iterator, idx, diag_offpart_csc)


_makecache(iterator::DiagonalIterator{SparseMatrixCSC{Tv,Ti},SymmetricMatrix}, idx)  where {Tv,Ti}=
    _makecache(iterator, idx, diag_offpart_csr)


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

function _makecache(iterator::DiagonalIterator{SparseMatrixCSR}, idx) 
    #Ωⁱ := {j ∈ Ωₖ : i ∈ Ωₖ}
    #Ωⁱₒ := {j ∉ Ωₖ : i ∈ Ωₖ} off-partition column values
    # bₖᵢ := Aᵢᵢ
    # dₖᵢ := ∑_{j ∈ Ωⁱₒ} |Aᵢⱼ|
    @unpack A, k, partsize = iterator
    part_start_idx = (k - 1) * partsize + 1
    part_end_idx = min(part_start_idx + partsize - 1, size(A, 2))

    b, d = diag_offpart_csr(A.rowptr, A.colval, A.nzval, idx, part_start_idx, part_end_idx)

    return DiagonalCache(k, idx, b, d)
end


function _l1prec!(y, P, issym)
    @unpack partitioning, A = P
    @unpack partsize, nparts = partitioning
    symT = issym ? SymmetricMatrix : NonSymmetricMatrix
    @batch for part in 1:nparts
        for diagonal in DiagonalIterator(symT, part, partsize, A)
            @unpack k, idx, b, d = diagonal
            y[idx] = y[idx] / (b + d)
        end
    end
    return nothing
end
