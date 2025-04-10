####################################
## L1 Gauss Seidel Preconditioner ##
####################################

## Structs & Constructors ##
struct BlockPartitioning{Ti,Backend}
    partsize::Ti # number of diagonals per partition
    nparts::Ti # number of partitions
    backend::Backend 
end

struct L1GSPreconditioner{Partitioning,VectorType}
    partitioning::Partitioning
    B::VectorType # Diagonal values
    D::VectorType # Off-diagonal absolute sum
end

struct L1GSPrecBuilder
    backend::Backend
    function L1GSPrecBuilder(backend::Backend)
        if functional(backend)
            return new(backend)
        else
            error(" $backend is not functional, please check your backend.")
        end
    end
end

(builder::L1GSPrecBuilder)(A::AbstractMatrix,partsize::Ti,nparts::Ti) where {Ti} = 
    build_l1prec(builder, A,partsize, nparts)

  
struct DiagonalIterator{MatrixFormat,MatrixSymmetry <: AbstractMatrixSymmetry,MatrixType,VectorType,Ti} 
    A::MatrixType
    B::VectorType # Diagonal values
    D::VectorType # Off-diagonal absolute sum
    partsize::Ti
    nparts::Ti
    local_idx::Ti # local index never changes
    initial_partition_idx::Ti # why initial ? because we are using stride to loop over the diagonals. 
    initial_global_idx::Ti 
    function DiagonalIterator(::Type{symT}, A::MatrixType,B::VectorType,D::VectorType,partsize::Ti,nparts::Ti,local_idx::Ti,initial_partition_idx::Ti) where {symT<:AbstractMatrixSymmetry,MatrixType,VectorType,Ti}
        format_type = sparsemat_format_type(A) 
        initial_global_idx = (initial_partition_idx - convert(Ti,1)) * partsize + local_idx
        new{format_type,symT,MatrixType,VectorType,Ti}(A,B,D,partsize,nparts,local_idx,initial_partition_idx,initial_global_idx)
    end
end

DiagonalIterator(::Type{SymT}, k::Ti, partsize::Ti, A::MatrixType) where {SymT<:AbstractMatrixSymmetry,MatrixType,Ti<:Integer} =
    DiagonalIterator{MatrixType,SymT,Ti}(A, k, partsize)

struct DiagonalCache{Ti,Tv}
    k::Ti # partition index
    idx::Ti # diagonal index
    b::Tv # partition diagonal value
    d::Tv # off-partition absolute sum
end

## Preconditioner builder ##
function build_l1prec(builder::L1GSPrecBuilder, A::AbstractSparseMatrix,partsize::Ti, nparts::Ti) where {Ti<:Integer}
    @unpack backend = builder
    partsize == 0 && error("partsize must be greater than 0")
    nparts == 0 && error("nparts must be greater than 0")
    _build_l1prec(backend, A, partsize, nparts)
end

function _build_l1prec(backend::Backend, _A::AbstractSparseMatrix,partsize::Ti, nparts::Ti) where {Ti<:Integer}
    # No assumptions on A, i.e. A here might be in either backend compatible format or not. 
    # So we have to convert it to backend compatible format, if it is not already.
    partitioning = BlockPartitioning(partsize, nparts, backend)
    B,D = _precompute_blocks(_A, partitioning)
    L1GSPreconditioner(partitioning, B,D)
end

function LinearSolve.ldiv!(y::VectorType, P::L1GSPreconditioner{BlockPartitioning{Ti,Backend}}, x::VectorType) where {VectorType <: AbstractVector, Ti<:Integer, Backend}
    # x: residual
    # y: preconditioned residual
    y .= x #works either way, whether x is GpuVectorType (e.g. CuArray) or Vector
    @unpack partitioning,B,D = P
    @unpack backend = partitioning
    # The following code is required because there is no assumption on the compatibality of x with the backend.
    _y = adapt(backend, y)
    _y .= _y ./ (B + D)
    copyto!(y, _y)
    return nothing
end

function LinearSolve.ldiv!(y::Vector, P::L1GSPreconditioner{BlockPartitioning{Ti,CPU}}, x::Vector) where { Ti <: Integer}
    # x: residual
    # y: preconditioned residual
    y .= x 
    @unpack partitioning,B,D = P
    @unpack backend = partitioning
    y .= y ./ (B + D)
    return nothing
end

function (\)(P::L1GSPreconditioner{BlockPartitioning{Ti,Backend}}, x::VectorType) where {VectorType <: AbstractVector, Ti, Backend}
    # P is a preconditioner
    # x is a vector
    y = similar(x)
    LinearSolve.ldiv!(y, P, x)
    return y
end

## L1 GS internal functionalty ##

function Base.iterate(iterator::DiagonalIterator)
    @unpack A, initial_partition_idx, initial_global_idx = iterator
    idx = initial_global_idx # diagonal index
    idx <= size(A, 1) || return nothing
    k = initial_partition_idx # partition index
    return (_makecache(iterator,idx,k), (idx,k))
end

function Base.iterate(iterator::DiagonalIterator, state)
    @unpack A, partsize, nparts = iterator
    total_work_items = partsize * nparts  # stride = n_threads * n_blocks
    idx,k = state
    k += nparts # partition index
    idx = idx + total_work_items # diagonal index
    idx <= size(A, 1) || return nothing
    return (_makecache(iterator,idx,k), (idx,k))
end


_makecache(iterator::DiagonalIterator{CSCFormat,NonSymmetricMatrix}, idx,k) = 
    _makecache(iterator,idx,k,_diag_offpart_csc)


 _makecache(iterator::DiagonalIterator{CSCFormat,SymmetricMatrix}, idx,k) = 
    _makecache(iterator,idx,k,_diag_offpart_csr)


function _makecache(iterator::DiagonalIterator{CSCFormat}, idx::Ti,k::Ti, diag_offpart_func) where {Ti<:Integer} 
    #Ωⁱ := {j ∈ Ωₖ : i ∈ Ωₖ}
    #Ωⁱₒ := {j ∉ Ωₖ : i ∈ Ωₖ} off-partition column values
    # bₖᵢ := Aᵢᵢ
    # dₖᵢ := ∑_{j ∈ Ωⁱₒ} |Aᵢⱼ|
    @unpack A,partsize = iterator
    part_start_idx = (k - convert(Ti,1)) * partsize + convert(Ti,1)
    part_end_idx = min(part_start_idx + partsize - convert(Ti,1), size(A, 2))
    
    b,d = diag_offpart_func(getcolptr(A), rowvals(A), getnzval(A), idx, part_start_idx, part_end_idx)

    return DiagonalCache(k, idx,b, d)
end

function _makecache(iterator::DiagonalIterator{CSRFormat}, idx::Ti,k::Ti) where {Ti<:Integer}
    #Ωⁱ := {j ∈ Ωₖ : i ∈ Ωₖ}
    #Ωⁱₒ := {j ∉ Ωₖ : i ∈ Ωₖ} off-partition column values
    # bₖᵢ := Aᵢᵢ
    # dₖᵢ := ∑_{j ∈ Ωⁱₒ} |Aᵢⱼ|
    @unpack A,partsize = iterator  # A is in CSR format
    part_start_idx = (k - convert(Ti,1)) * partsize + convert(Ti,1)
    part_end_idx = min(part_start_idx + partsize - convert(Ti,1), size(A, 2)) 

    b,d = _diag_offpart_csr(getrowptr(A), colvals(A), getnzval(A), idx, part_start_idx, part_end_idx)

    return DiagonalCache(k, idx, b, d)
end


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

function _precompute_blocks(_A::AbstractSparseMatrix,partitioning::BlockPartitioning)
    @unpack partsize, nparts, backend = partitioning
    A = adapt(backend, _A)
    N = size(A, 1)
    B = adapt(backend,zeros(eltype(A), N))
    D = adapt(backend,zeros(eltype(A), N))
    symT = isapprox(A, A',rtol=1e-12) ? SymmetricMatrix : NonSymmetricMatrix
    ndrange = partsize * nparts
    kernel = _precompute_blocks_kernel!(backend,partsize,ndrange)
    kernel(B,D,A,symT,partsize,nparts; ndrange=ndrange)
    synchronize(backend)
    return B, D
end


@kernel function _precompute_blocks_kernel!(B,D,A,symT,partsize::Ti,nparts::Ti) where {Ti<:Integer}
    # this kernel will loop over the corresponding diagonals in strided fashion.
    # e.g. if partsize = 4, nparts = 2, A is (100 x 100), and current global thread id = 5, 
    # then the kernel will loop over the diagonals with stride:
    # k (partition index) = k + nparts (i.e. 2, 4, 6, 8, 10, 12, 14)
    # idx (diagonal index) = idx + nparts * partsize (i.e. 5, 13, 21, 29, 37, 45, 53)
    local_idx = @index(Local)
    initial_partition_idx = @index(Group)
    for diagonal in DiagonalIterator(symT,A,B,D,partsize,nparts,convert(Ti,local_idx),convert(Ti,initial_partition_idx))
        @unpack k, idx, b, d = diagonal
        B[idx] = b
        D[idx] = d
    end
end
