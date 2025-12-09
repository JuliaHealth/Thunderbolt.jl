####################################
## L1 Gauss Seidel Preconditioner ##
####################################

## Structs & Constructors ##
"""
    BlockPartitioning{Ti<:Integer, Backend}

Struct that encapsulates the diagonal partitioning configuration which is then used to distribute the work across multiple cores.

# Fields
- `partsize::Ti`: Size of each partition (diagonal block).
- `nparts::Ti`: Number of partitions (i.e. size(A,1)/partsize).
- `nchunks::Ti`: Number of workgroups (e.g. CPU cores or GPU blocks).
- `chunksize::Ti`: Number of partitions assigned to each workgroup.
- `backend::Backend`: Execution backend that determines where and how the preconditioner is applied, such as `CPU()` or `CUDABackend()`.

!!! note
    `chuncksize * nchunks` doesn't have to be equal to `nparts` (can be less than or greater than). The diagonal partition iterator will take care of that throught strided iteration.
    The reason for this is obvious in GPUBackend, in which `nblocks (i.e. nchunks)` and `nthreads (i.e. chunksize)` are chosen to maximize occupancy, which may not be equal to `nparts`.
"""
struct BlockPartitioning{Ti <: Integer, Backend}
    partsize::Ti # dimension of each partition
    nparts::Ti # total number of partitions
    nchunks::Ti # no. CPU cores or GPU blocks
    chunksize::Ti # nthreads in GPU backend
    backend::Backend
end

@doc raw"""
    L1GSPreconditioner{Partitioning, VectorType}

The ℓ₁ Gauss–Seidel preconditioner is a robust and parallel-friendly preconditioner for sparse matrices.

# Algorithm

The L1-GS preconditioner is constructed by dividing the matrix into diagonal blocks `nparts`:
- Let Ωₖ denote the block with index `k`.
- For each Ωₖ, we define the following sets:
    - $ Ωⁱ := \{j ∈ Ωₖ : i ∈ Ωₖ\} $ → the set of columns in the diagonal block for row i
    - $ Ωⁱₒ := \{j ∉ Ωₖ : i ∈ Ωₖ\} $ →  the remaining “off-diagonal” columns in row i

The preconditioner matrix $M_{ℓ_1}$  is defined as:
```math
M_{ℓ_1GS} = M_{HGS} + D^{ℓ_1} \\
```
Where $D^{ℓ_1}$ is a diagonal matrix with entries: $d_{ii}^{ℓ_1} = \sum_{j ∈ Ωⁱₒ} |a_{ij}|$, and $M_{HGS}$ is obtained when the diagonal partitions are chosen to be the Gauss–Seidel sweeps on $ A_{kk} $

# Fields
- `partitioning`: Encapsulates partitioning data (e.g. nparts, partsize, backend).
- `D_Dl1`: $D+D^{ℓ_1}$.
- `SLbuffer`: Strictly lower triangular part of all diagonal blocks stacked in a vector.


# Reference
[Baker, A. H., Falgout, R. D., Kolev, T. V., & Yang, U. M. (2011).
*Multigrid Smoothers for Ultraparallel Computing*,
SIAM J. Sci. Comput., 33(5), 2864–2887.](@cite BakFalKolYan:2011:MSU)

!!! note
    For now $M_{HGS}$ applies only a **forward** sweep of the Gauss–Seidel method, which is a lower triangular matrix.
    The interface will be extended in future versions to allow for backward and symmetric sweeps.

# Example
```julia
builder = L1GSPrecBuilder(PolyesterDevice(4))
N = 128*16
A = spdiagm(0 => 2 * ones(N), -1 => -ones(N-1), 1 => -ones(N-1))
partsize = 16
prec = builder(A, partsize)
```
"""
struct L1GSPreconditioner{Partitioning, VectorType}
    partitioning::Partitioning
    D_Dl1::VectorType # D + Dˡ
    SLbuffer::VectorType # strictly lower triangular part of all diagonal blocks. (length = (partsize*(partsize-1)* nparts)/2)
end

"""
    L1GSPrecBuilder(device::AbstractDevice)
A builder for the L1 Gauss-Seidel preconditioner. This struct encapsulates the backend and provides a method to build the preconditioner.
# Fields
- `device::AbstractDevice`: The backend used for the preconditioner. More info [AbstractDevice](@ref).
"""
struct L1GSPrecBuilder{DeviceType <: AbstractDevice}
    device::DeviceType
    function L1GSPrecBuilder(device::AbstractDevice)
        backend = default_backend(device)
        if functional(backend)
            return new{typeof(device)}(device)
        else
            error(" $backend is not functional, please check your backend.")
        end
    end
end

(builder::L1GSPrecBuilder)(A::AbstractMatrix, partsize::Ti) where {Ti <: Integer} =
    build_l1prec(builder, A, partsize)


struct DiagonalPartsIterator{Ti}
    size_A::Ti
    partsize::Ti
    nparts::Ti
    nchunks::Ti # number of CPU cores or GPU blocks
    chunksize::Ti # number of threads per block
    initial_partition_idx::Ti # initial partition index
end

struct DiagonalPartCache{Ti}
    k::Ti # partition index
    partsize::Ti # partition size
    start_idx::Ti # start index of the partition
    end_idx::Ti # end index of the partition
end

## Preconditioner builder ##
function build_l1prec(
    builder::L1GSPrecBuilder,
    A::MatrixType,
    partsize::Ti,
) where {Ti <: Integer, MatrixType}
    partsize == 0 && error("partsize must be greater than 0")
    _build_l1prec(builder, A, partsize)
end

function _build_l1prec(
    builder::L1GSPrecBuilder,
    _A::MatrixType,
    partsize::Ti,
) where {Ti <: Integer, MatrixType}
    # `nchunks` is either CPU cores or GPU blocks.
    # Each chunk will be assigned `nparts`, each of size `partsize`.
    # In GPU backend, `nchunks` is the number of blocks and `partsize` is the number of threads per block.
    A = get_data(_A) # for symmetric case
    partitioning = _blockpartitioning(builder, A, partsize)
    D_Dl1, SLbuffer = _precompute_blocks(A, partitioning)
    L1GSPreconditioner(partitioning, D_Dl1, SLbuffer)
end



function _blockpartitioning(
    builder::L1GSPrecBuilder{<:AbstractCPUDevice},
    A::AbstractSparseMatrix,
    partsize::Ti,
) where {Ti <: Integer}
    (; device) = builder
    (; chunksize) = device
    nparts = convert(Ti, size(A, 1) / partsize |> ceil) #total number of partitions
    nchunks = chunksize * nparts
    return BlockPartitioning(partsize, nparts, nchunks, chunksize, default_backend(device))
end

function _blockpartitioning(
    builder::L1GSPrecBuilder{<:AbstractGPUDevice},
    A::AbstractSparseMatrix,
    partsize::Ti,
) where {Ti <: Integer}
    (; device) = builder
    (; blocks, threads) = device
    (threads == 0 || threads === nothing) && error("`threads` must be set greater than 0")
    (blocks == 0 || blocks === nothing) && error("`blocks`` must be set greater than 0")
    nchunks = blocks # number of GPU blocks
    nparts = convert(Ti, size(A, 1) / partsize |> ceil) #total number of partitions
    chunksize = convert(Ti, (nparts / nchunks) |> ceil) # number of partitions per chunk
    chunksize = chunksize <= threads ? chunksize : threads # number of threads per block
    return BlockPartitioning(partsize, nparts, nchunks, chunksize, default_backend(device))
end

function LinearSolve.ldiv!(
    y::VectorType,
    P::L1GSPreconditioner{BlockPartitioning{Ti, Backend}},
    x::VectorType,
) where {VectorType <: AbstractVector, Ti <: Integer, Backend}
    # x: residual
    # y: preconditioned residual
    y .= x #works either way, whether x is GpuVectorType (e.g. CuArray) or Vector
    @unpack partitioning, D_Dl1 = P
    @unpack backend = partitioning
    # The following code is required because there is no assumption on the compatibality of x with the backend.
    _y = adapt(backend, y)
    _forward_sweep!(_y, P)
    copyto!(y, _y)
    return nothing
end

function LinearSolve.ldiv!(
    y::Vector,
    P::L1GSPreconditioner{BlockPartitioning{Ti, CPU}},
    x::Vector,
) where {Ti <: Integer}
    # x: residual
    # y: preconditioned residual
    y .= x
    _forward_sweep!(y, P)
    return nothing
end

function (\)(
    P::L1GSPreconditioner{BlockPartitioning{Ti, Backend}},
    x::VectorType,
) where {VectorType <: AbstractVector, Ti, Backend}
    # P is a preconditioner
    # x is a vector
    y = similar(x)
    LinearSolve.ldiv!(y, P, x)
    return y
end

## L1 GS internal functionalty ##
get_data(A::AbstractSparseMatrix) = A
get_data(A::Symmetric{Ti, TA}) where {Ti, TA} = TA(A.data) # restore the full matrix, why ? https://discourse.julialang.org/t/is-there-a-symmetric-sparse-matrix-implementation-in-julia/91333/2

function Base.iterate(iterator::DiagonalPartsIterator)
    @unpack initial_partition_idx, nparts = iterator
    initial_partition_idx <= nparts || return nothing
    k = initial_partition_idx
    return (_makecache(iterator, k), k)
end

function Base.iterate(iterator::DiagonalPartsIterator, state)
    @unpack nparts, nchunks, chunksize = iterator
    stride = nchunks * chunksize  # stride = n_threads * n_blocks
    k = state
    k += stride # partition index
    k <= nparts || return nothing
    return (_makecache(iterator, k), k)
end

function _makecache(iterator::DiagonalPartsIterator, k::Ti) where {Ti <: Integer}
    #Ωⁱ := {j ∈ Ωₖ : i ∈ Ωₖ}
    #Ωⁱₒ := {j ∉ Ωₖ : i ∈ Ωₖ} off-partition column values
    # bₖᵢ := Aᵢᵢ
    # dₖᵢ := ∑_{j ∈ Ωⁱₒ} |Aᵢⱼ|
    @unpack size_A, partsize = iterator
    part_start_idx = (k - convert(Ti, 1)) * partsize + convert(Ti, 1)
    part_end_idx = min(part_start_idx + partsize - convert(Ti, 1), size_A)

    #b,d = diag_offpart_func(getcolptr(A), rowvals(A), getnzval(A), idx, part_start_idx, part_end_idx)
    actual_partsize = part_end_idx - part_start_idx + convert(Ti, 1)
    return DiagonalPartCache(k, actual_partsize, part_start_idx, part_end_idx)
end


function _diag_offpart_csr(
    rowPtr,
    colVal,
    nzVal,
    idx::Ti,
    part_start::Ti,
    part_end::Ti,
) where {Ti <: Integer}
    Tv = eltype(nzVal)
    b = zero(Tv)
    d = zero(Tv)

    row_start = rowPtr[idx]
    row_end = rowPtr[idx+1] - 1

    for i = row_start:row_end
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

function _diag_offpart_csc(
    colPtr,
    rowVal,
    nzVal,
    idx::Ti,
    part_start::Ti,
    part_end::Ti,
) where {Ti <: Integer}
    Tv = eltype(nzVal)
    b = zero(Tv)
    d = zero(Tv)

    ncols = length(colPtr) - 1

    for col = 1:ncols
        col_start = colPtr[col]
        col_end = colPtr[col+1] - 1

        for i = col_start:col_end
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

_diag_offpart(
    ::NonSymmetricMatrix,
    ::CSCFormat,
    A,
    idx::Ti,
    part_start::Ti,
    part_end::Ti,
) where {Ti <: Integer} =
    _diag_offpart_csc(getcolptr(A), rowvals(A), getnzval(A), idx, part_start, part_end)

_diag_offpart(
    ::SymmetricMatrix,
    ::CSCFormat,
    A,
    idx::Ti,
    part_start::Ti,
    part_end::Ti,
) where {Ti <: Integer} =
    _diag_offpart_csr(getcolptr(A), rowvals(A), getnzval(A), idx, part_start, part_end)

_diag_offpart(
    ::AbstractMatrixSymmetry,
    ::CSRFormat,
    A,
    idx::Ti,
    part_start::Ti,
    part_end::Ti,
) where {Ti <: Integer} =
    _diag_offpart_csr(getrowptr(A), colvals(A), getnzval(A), idx, part_start, part_end)

function _pack_strict_lower_csr!(
    SLbuffer,
    rowPtr,
    colVal,
    nzVal,
    start_idx::Ti,
    end_idx::Ti,
    partsize::Ti,
    k::Ti,
) where {Ti <: Integer}
    block_stride = (partsize * (partsize - 1)) ÷ 2 # no. off-diagonal elements in a block
    block_offset = (k - 1) * block_stride

    for i = start_idx:end_idx
        local_i = i - start_idx + 1
        # no. of off-diagonal elements in that row
        row_offset = ((local_i-1) * (local_i - 2)) ÷ 2

        # scan the CSR row
        for p = rowPtr[i]:(rowPtr[i+1]-1)
            j = colVal[p]
            if j >= start_idx && j < i
                local_j = j - start_idx + 1
                # off-diagonal index consists of three parts: block offset, row offset, and column index
                off_idx = block_offset + row_offset + local_j
                SLbuffer[off_idx] = nzVal[p]
            end
        end
    end
    return nothing
end

function _pack_strict_lower_csc!(
    SLbuffer,
    colPtr,
    rowVal,
    nzVal,
    start_idx::Ti,
    end_idx::Ti,
    partsize::Ti,
    k::Ti,
) where {Ti <: Integer}
    block_stride = (partsize * (partsize - 1)) ÷ 2 # no. off-diagonal elements in a block
    block_offset = (k - 1) * block_stride
    for col = start_idx:(end_idx-1)
        local_j = col - start_idx + 1
        for p = colPtr[col]:(colPtr[col+1]-1)
            i = rowVal[p]
            if i > col && i <= end_idx
                local_i = i - start_idx + 1
                row_offset = ((local_i-1) * (local_i - 2)) ÷ 2
                off_idx = block_offset + row_offset + local_j
                SLbuffer[off_idx] = nzVal[p]
            end
        end
    end
    return nothing
end

_pack_strict_lower!(
    ::NonSymmetricMatrix,
    ::CSCFormat,
    SLbuffer,
    A,
    start_idx::Ti,
    end_idx::Ti,
    partsize::Ti,
    k::Ti,
) where {Ti <: Integer} = _pack_strict_lower_csc!(
    SLbuffer,
    getcolptr(A),
    rowvals(A),
    getnzval(A),
    start_idx,
    end_idx,
    partsize,
    k,
)

_pack_strict_lower!(
    ::SymmetricMatrix,
    ::CSCFormat,
    SLbuffer,
    A,
    start_idx::Ti,
    end_idx::Ti,
    partsize::Ti,
    k::Ti,
) where {Ti <: Integer} = _pack_strict_lower_csr!(
    SLbuffer,
    getcolptr(A),
    rowvals(A),
    getnzval(A),
    start_idx,
    end_idx,
    partsize,
    k,
)

_pack_strict_lower!(
    ::AbstractMatrixSymmetry,
    ::CSRFormat,
    SLbuffer,
    A,
    start_idx::Ti,
    end_idx::Ti,
    partsize::Ti,
    k::Ti,
) where {Ti <: Integer} = _pack_strict_lower_csr!(
    SLbuffer,
    getrowptr(A),
    colvals(A),
    getnzval(A),
    start_idx,
    end_idx,
    partsize,
    k,
)


function _precompute_blocks(_A::AbstractSparseMatrix, partitioning::BlockPartitioning)
    # No assumptions on A, i.e. A here might be in either backend compatible format or not.
    # So we have to convert it to backend compatible format, if it is not already.
    # `partsize` is the size of each partition, `nparts` is the total number of partitions.
    # `nchunks` is the number of CPU cores or GPU blocks.
    # `chunksize` is the number of threads per block in GPU backend.
    @unpack partsize, nparts, nchunks, chunksize, backend = partitioning
    A = adapt(backend, _A)
    N = size(A, 1)
    D_Dl1 = adapt(backend, zeros(eltype(A), N)) # D + Dˡ
    last_partsize = N - (nparts - 1) * partsize # size of the last partition
    SLbuffer_size =
        (partsize * (partsize - 1) * (nparts-1)) ÷ 2 + last_partsize * (last_partsize - 1) ÷ 2
    SLbuffer = adapt(backend, zeros(eltype(A), SLbuffer_size)) # strictly lower triangular part of all diagonal blocks stored in a 1D array
    symA = isapprox(A, A', rtol = 1e-12) ? SymmetricMatrix() : NonSymmetricMatrix()
    ndrange = nchunks * chunksize
    kernel = _precompute_blocks_kernel!(backend, chunksize, ndrange)
    kernel(D_Dl1, SLbuffer, A, symA, partsize, nparts, nchunks, chunksize; ndrange = ndrange)
    synchronize(backend)
    return D_Dl1, SLbuffer
end


@kernel function _precompute_blocks_kernel!(
    D_Dl1,
    SLbuffer,
    A,
    symA,
    partsize::Ti,
    nparts::Ti,
    nchunks::Ti,
    chunksize::Ti,
) where {Ti <: Integer}
    initial_partition_idx = @index(Global)
    size_A = convert(Ti, size(A, 1))
    format_A = sparsemat_format_type(A)
    # NOTE: `DiagonalPartsIterator` is logic agnostic. It essentially encapsulates the strided iterations over the diagonal blocks.
    for part in DiagonalPartsIterator(
        size_A,
        partsize,
        nparts,
        nchunks,
        chunksize,
        convert(Ti, initial_partition_idx),
    )
        @unpack k, partsize, start_idx, end_idx = part # NOTE: `partsize` here is the actual size of the partition
        # From start_idx to end_idx, extract the diagonal and off-diagonal values
        for i = start_idx:end_idx
            b, d = _diag_offpart(symA, format_A, A, i, start_idx, end_idx)
            #@show i,b,d
            # Update the diagonal and off-diagonal values
            D_Dl1[i] = b + d
            _pack_strict_lower!(symA, format_A, SLbuffer, A, start_idx, end_idx, partsize, k)
        end
    end
end

function _forward_sweep!(y, P)
    @unpack partitioning, D_Dl1, SLbuffer = P
    @unpack partsize, nparts, nchunks, chunksize, backend = partitioning
    ndrange = nchunks * chunksize
    kernel = _forward_sweep_kernel!(backend, chunksize, ndrange)
    size_A = convert(typeof(nparts), length(y))
    kernel(y, D_Dl1, SLbuffer, size_A, partsize, nparts, nchunks, chunksize; ndrange = ndrange)
    synchronize(backend)
    return nothing
end

@kernel function _forward_sweep_kernel!(
    y,
    D_Dl1,
    SLbuffer,
    size_A::Ti,
    partsize::Ti,
    nparts::Ti,
    nchunks::Ti,
    chunksize::Ti,
) where {Ti <: Integer}
    initial_partition_idx = @index(Global)
    # NOTE: `DiagonalPartsIterator` is logic agnostic. It essentially encapsulates the strided iterations over the diagonal blocks.
    for part in DiagonalPartsIterator(
        size_A,
        partsize,
        nparts,
        nchunks,
        chunksize,
        convert(Ti, initial_partition_idx),
    )
        @unpack k, partsize, start_idx, end_idx = part # NOTE: `partsize` here is the actual size of the partition
        block_stride = (partsize * (partsize - 1)) ÷ 2 # no. off-diagonal elements in a block
        block_offset = (k - 1) * block_stride

        # forward‐solve (Ax=b): (1/a_{ii})[bᵢ - ∑_{j<i} a_ij * x_j]
        for i = start_idx:end_idx
            local_i = i - start_idx + 1

            row_offset = ((local_i-1) * (local_i - 2)) ÷ 2

            # accumulate strictly‐lower * y
            acc = zero(eltype(y))
            @inbounds for local_j = 1:(local_i-1) # iterate over the off-diagonal columns in row local_i
                # j’s global index:
                gj = start_idx + (local_j - 1)
                off_idx = block_offset + row_offset + (local_j)
                acc += SLbuffer[off_idx] * y[gj]
            end

            y[i] = (y[i] - acc) / D_Dl1[i]
        end
    end
end
