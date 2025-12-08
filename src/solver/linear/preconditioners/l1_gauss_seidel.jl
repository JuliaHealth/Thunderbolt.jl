####################################
## L1 Gauss Seidel Preconditioner ##
####################################

## User defined types for sweep direction and storage strategy ##
abstract type Sweep end
struct ForwardSweep <: Sweep end # forward sweep -> lower triangular
struct BackwardSweep <: Sweep end # backward sweep -> upper triangular
struct SymmetricSweep <: Sweep end # symmetric sweep -> both lower and upper triangular

abstract type StorageStrategy end
struct OriginalMatrix <: StorageStrategy end # store original matrix
struct PackedBuffer <: StorageStrategy end # store lower/upper triangular parts in a vector buffer -> efficient for small partitions
struct SparseTriangular <: StorageStrategy end # store sparse lower/upper triangular matrices


## Internal types to encapsulate sweep and storage ##
abstract type SweepStorage end

# Forward sweep variants
abstract type ForwardSweepStorage <: SweepStorage end
struct ForwardSweepOriginalMatrix{MatrixType} <: ForwardSweepStorage
    A::MatrixType  # Original matrix
end
struct ForwardSweepPackedBuffer{VectorType} <: ForwardSweepStorage
    SLbuffer::VectorType  # Packed strictly lower triangular elements
end
struct ForwardSweepSparseTriangular{MatrixType} <: ForwardSweepStorage
    L::MatrixType  # Sparse  strictly lower triangular matrix
end

# Backward sweep variants
abstract type BackwardSweepStorage <: SweepStorage end
struct BackwardSweepOriginalMatrix{MatrixType} <: BackwardSweepStorage
    A::MatrixType  # Original matrix
end
struct BackwardSweepPackedBuffer{VectorType} <: BackwardSweepStorage
    SUbuffer::VectorType  # Packed strictly upper triangular elements
end
struct BackwardSweepSparseTriangular{MatrixType} <: BackwardSweepStorage
    U::MatrixType  # Sparse strictly upper triangular matrix
end

# Symmetric sweep variants
abstract type SymmetricSweepStorage <: SweepStorage end
struct SymmetricSweepOriginalMatrix{MatrixType} <: SymmetricSweepStorage
    A::MatrixType  # Original symmetric matrix
end


# TODO: are these two buffers necessary? won't be implemented for now
struct SymmetricSweepPackedBuffer{VectorType} <: SymmetricSweepStorage
    SLbuffer::VectorType  # Packed strictly lower triangular elements
    SUbuffer::VectorType  # Packed strictly upper triangular elements
end
struct SymmetricSweepSparseTriangular{MatrixType} <: SymmetricSweepStorage
    L::MatrixType  # Sparse strictly lower triangular matrix
    U::MatrixType  # Sparse strictly upper triangular matrix
end

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
struct BlockPartitioning{Ti<:Integer,Backend}
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
However, we use another convergant variant, which takes adavantage of the local estimation of θ ( $a_{ii} >= θ * d_{ii}$):
```math
M_{ℓ_1GS*} = M_{HGS} + D^{ℓ_1*}, \quad \text{where} \quad d_{ii}^{ℓ_1*} = \begin{cases} 0, & \text{if } a_{ii} \geq \eta d_{ii}^{ℓ_1}; \\ d_{ii}^{ℓ_1}/2, & \text{otherwise.} \end{cases}
```

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
# NOTES:
# 1. for symmetric A, that's not of type `Symmetric`, or `SparseMatrixCSR` (i.e. in CSR format), then it's recommended to set `isSymA=true` for better performance `prec = builder(A, partsize; isSymA=true)`.
# 2. for any user-defined `η` value, use `prec = builder(A, partsize; η=1.2)`.
```
"""
struct L1GSPreconditioner{Partitioning,VectorType,SweepStorageType<:SweepStorage}
    partitioning::Partitioning
    D_Dl1::VectorType # D + Dˡ
    sweepstorage::SweepStorageType # Encapsulates sweep direction and storage strategy
end

"""
    L1GSPrecBuilder(device::AbstractDevice)
A builder for the L1 Gauss-Seidel preconditioner. This struct encapsulates the backend and provides a method to build the preconditioner.
# Fields
- `device::AbstractDevice`: The backend used for the preconditioner. More info [AbstractDevice](@ref).
"""
struct L1GSPrecBuilder{DeviceType<:AbstractDevice}
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

(builder::L1GSPrecBuilder)(A::AbstractMatrix, partsize::Ti;
    isSymA::Bool = false,
    η = 1.5,
    sweep::Sweep = ForwardSweep(),
    storage::StorageStrategy = _choose_default_device_storage(builder.device)) where {Ti<:Integer} =
    build_l1prec(builder, A, partsize, isSymA, η, sweep, storage)

(builder::L1GSPrecBuilder)(A::Symmetric, partsize::Ti;
    η = 1.5,
    sweep::Sweep = SymmetricSweep(),
    storage::StorageStrategy = OriginalMatrix()) where {Ti<:Integer} =
    build_l1prec(builder, A, partsize, true, η, sweep, storage)

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
function build_l1prec(builder::L1GSPrecBuilder, A::MatrixType, partsize::Ti, isSymA::Bool, η, sweep::Sweep, storage::StorageStrategy) where {Ti<:Integer,MatrixType}
    partsize == 0 && error("partsize must be greater than 0")

    # TODO: do we need this validation (presumably symmetric sweep only works for symmetric matrices)?
    if sweep isa SymmetricSweep && !isSymA && !(A isa Symmetric)
        error("SymmetricSweep requires a symmetric matrix. If `A` is symmetric, then either pass isSymA=true or use Symmetric(A). If not symmetric, consider using `ForwardSweep` or `BackwardSweep` instead.")
    end

    # TODO: warn user if SymmetricSweep is used with non-optimal storage (presumably OriginalMatrix is best, but no clue..)
    if sweep isa SymmetricSweep && !(storage isa OriginalMatrix)
        @warn "SymmetricSweep is most efficient with OriginalMatrix storage. Consider using storage=OriginalMatrix()."
    end

    _build_l1prec(builder, A, partsize, isSymA, η, sweep, storage)
end

function _build_l1prec(builder::L1GSPrecBuilder, _A::MatrixType, partsize::Ti, isSymA::Bool, η, sweep::Sweep, storage::StorageStrategy) where {Ti<:Integer,MatrixType}
    # `nchunks` is either CPU cores or GPU blocks.
    # Each chunk will be assigned `nparts`, each of size `partsize`.
    # In GPU backend, `nchunks` is the number of blocks and `partsize` is the number of threads per block.
    A = get_data(_A) # for symmetric case
    partitioning = _blockpartitioning(builder, A, partsize)

    # Create D_Dl1 and SweepStorage in one step - each combination has its own specialized construction
    D_Dl1, sweepstorage = _create_sweep_storage_with_precompute(sweep, storage, A, partitioning, isSymA, η)

    L1GSPreconditioner(partitioning, D_Dl1, sweepstorage)
end



function _blockpartitioning(builder::L1GSPrecBuilder{<:AbstractCPUDevice}, A::AbstractSparseMatrix, partsize::Ti) where {Ti<:Integer}
    (; device) = builder
    (; chunksize) = device
    nparts = convert(Ti, size(A, 1) / partsize |> ceil) #total number of partitions
    nchunks = chunksize * nparts
    return BlockPartitioning(partsize, nparts, nchunks, chunksize, default_backend(device))
end

function _blockpartitioning(builder::L1GSPrecBuilder{<:AbstractGPUDevice}, A::AbstractSparseMatrix, partsize::Ti) where {Ti<:Integer}
    (; device) = builder
    (; blocks, threads) = device
    (threads == 0 || threads === nothing) && error("`threads` must be set greater than 0")
    (blocks  == 0 || blocks === nothing)  && error("`blocks`` must be set greater than 0")
    nchunks = blocks # number of GPU blocks
    nparts = convert(Ti, size(A, 1) / partsize |> ceil) #total number of partitions
    chunksize = convert(Ti, (nparts / nchunks) |> ceil) # number of partitions per chunk
    chunksize = chunksize <= threads ? chunksize : threads # number of threads per block
    return BlockPartitioning(partsize, nparts, nchunks, chunksize, default_backend(device))
end

function LinearSolve.ldiv!(y::VectorType, P::L1GSPreconditioner, x::VectorType) where {VectorType<:AbstractVector}
    @timeit_debug "ldiv! (generic)" begin
        # x: residual
        # y: preconditioned residual
        @timeit_debug "y .= x" y .= x #works either way, whether x is GpuVectorType (e.g. CuArray) or Vector
        (; partitioning) = P
        (; backend) = partitioning
        # The following code is required because there is no assumption on the compatibality of x with the backend.
        @timeit_debug "adapt(backend, y)" _y = adapt(backend, y)
        @timeit_debug "_apply_sweep!" _apply_sweep!(_y, P)
        @timeit_debug "copyto!(y, _y)" copyto!(y, _y)
    end
    return nothing
end

function LinearSolve.ldiv!(y::Vector, P::L1GSPreconditioner{BlockPartitioning{Ti,CPU}}, x::Vector) where {Ti<:Integer}
    @timeit_debug "ldiv! (CPU)" begin
        @timeit_debug "y .= x" y .= x
        @timeit_debug "_apply_sweep!" _apply_sweep!(y, P)
    end
    return nothing
end

function (\)(P::L1GSPreconditioner{BlockPartitioning{Ti,Backend}}, x::VectorType) where {VectorType<:AbstractVector,Ti,Backend}
    # P is a preconditioner
    # x is a vector
    y = similar(x)
    LinearSolve.ldiv!(y, P, x)
    return y
end

## L1 GS internal functionalty ##
get_data(A::AbstractSparseMatrix) = A
get_data(A::Symmetric{Ti,TA}) where {Ti,TA} = TA(A.data) # restore the full matrix, why ? https://discourse.julialang.org/t/is-there-a-symmetric-sparse-matrix-implementation-in-julia/91333/2


_choose_default_device_storage(::AbstractDevice) = SparseTriangular()
_choose_default_device_storage(::AbstractGPUDevice) = PackedBuffer()


## Specialized construction functions for each (Sweep, Storage) combination ##
# These functions merge allocation, precomputation, and SweepStorage creation into one step

# ForwardSweep + OriginalMatrix: No buffer needed, just reference A
function _create_sweep_storage_with_precompute(::ForwardSweep, ::OriginalMatrix, A, partitioning, isSymA, η)
    D_Dl1 = _compute_D_Dl1(A, partitioning, isSymA, η)
    sweepstorage = ForwardSweepOriginalMatrix(A)
    return D_Dl1, sweepstorage
end

# ForwardSweep + PackedBuffer: Allocate and pack lower triangular
function _create_sweep_storage_with_precompute(::ForwardSweep, ::PackedBuffer, A, partitioning, isSymA, η)
    (;partsize, nparts, nchunks, chunksize, backend) = partitioning
    Tf = eltype(A)
    N = size(A, 1)

    # Allocate buffer for strictly lower triangular
    last_partsize = N - (nparts - 1) * partsize
    buffer_size = (partsize * (partsize - 1) * (nparts-1)) ÷ 2 + last_partsize * (last_partsize - 1) ÷ 2
    SLbuffer = adapt(backend, zeros(Tf, buffer_size))

    # Compute D_Dl1 and pack lower triangular in one kernel
    D_Dl1 = adapt(backend, zeros(Tf, N))
    symA = isSymA ? SymmetricMatrix() : NonSymmetricMatrix()
    η_converted = convert(Tf, η)

    ndrange = nchunks * chunksize
    kernel = _precompute_and_pack_lower_kernel!(backend, chunksize, ndrange)
    kernel(D_Dl1, SLbuffer, A, symA, partsize, nparts, nchunks, chunksize, η_converted; ndrange=ndrange)
    synchronize(backend)

    sweepstorage = ForwardSweepPackedBuffer(SLbuffer)
    return D_Dl1, sweepstorage
end

# BackwardSweep + OriginalMatrix: No buffer needed, just reference A
function _create_sweep_storage_with_precompute(::BackwardSweep, ::OriginalMatrix, A, partitioning, isSymA, η)
    D_Dl1 = _compute_D_Dl1(A, partitioning, isSymA, η)
    sweepstorage = BackwardSweepOriginalMatrix(A)
    return D_Dl1, sweepstorage
end

# BackwardSweep + PackedBuffer: Allocate and pack upper triangular
function _create_sweep_storage_with_precompute(::BackwardSweep, ::PackedBuffer, A, partitioning, isSymA, η)
    (;partsize, nparts, nchunks, chunksize, backend) = partitioning
    Tf = eltype(A)
    N = size(A, 1)

    # Allocate buffer for strictly upper triangular
    last_partsize = N - (nparts - 1) * partsize
    buffer_size = (partsize * (partsize - 1) * (nparts-1)) ÷ 2 + last_partsize * (last_partsize - 1) ÷ 2
    SUbuffer = adapt(backend, zeros(Tf, buffer_size))

    # Compute D_Dl1 and pack upper triangular in one kernel
    D_Dl1 = adapt(backend, zeros(Tf, N))
    symA = isSymA ? SymmetricMatrix() : NonSymmetricMatrix()
    η_converted = convert(Tf, η)

    ndrange = nchunks * chunksize
    kernel = _precompute_and_pack_upper_kernel!(backend, chunksize, ndrange)
    kernel(D_Dl1, SUbuffer, A, symA, partsize, nparts, nchunks, chunksize, η_converted; ndrange=ndrange)
    synchronize(backend)

    sweepstorage = BackwardSweepPackedBuffer(SUbuffer)
    return D_Dl1, sweepstorage
end

# SymmetricSweep + OriginalMatrix: No buffer needed, just reference A
function _create_sweep_storage_with_precompute(::SymmetricSweep, ::OriginalMatrix, A, partitioning, isSymA, η)
    D_Dl1 = _compute_D_Dl1(A, partitioning, isSymA, η)
    sweepstorage = SymmetricSweepOriginalMatrix(A)
    return D_Dl1, sweepstorage
end

# ForwardSweep + SparseTriangular: Extract sparse strictly lower triangular
function _create_sweep_storage_with_precompute(::ForwardSweep, ::SparseTriangular, A, partitioning, isSymA, η)
    (;partsize, nparts, backend) = partitioning
    Tf = eltype(A)
    N = size(A, 1)

    # Compute D_Dl1
    D_Dl1 = _compute_D_Dl1(A, partitioning, isSymA, η)

    # Extract strictly lower triangular part from A (within diagonal blocks only)
    L = _extract_strict_lower_sparse(A, N, partsize, nparts)
    L_adapted = adapt(backend, L)

    sweepstorage = ForwardSweepSparseTriangular(L_adapted)
    return D_Dl1, sweepstorage
end

# BackwardSweep + SparseTriangular: Extract sparse strictly upper triangular
function _create_sweep_storage_with_precompute(::BackwardSweep, ::SparseTriangular, A, partitioning, isSymA, η)
    (;partsize, nparts, backend) = partitioning
    Tf = eltype(A)
    N = size(A, 1)

    # Compute D_Dl1
    D_Dl1 = _compute_D_Dl1(A, partitioning, isSymA, η)

    # Extract strictly upper triangular part from A (within diagonal blocks only)
    U = _extract_strict_upper_sparse(A, N, partsize, nparts)
    U_adapted = adapt(backend, U)

    sweepstorage = BackwardSweepSparseTriangular(U_adapted)
    return D_Dl1, sweepstorage
end

# SymmetricSweep + SparseTriangular: Extract both sparse L and U
function _create_sweep_storage_with_precompute(::SymmetricSweep, ::SparseTriangular, A, partitioning, isSymA, η)
    (;partsize, nparts, backend) = partitioning
    Tf = eltype(A)
    N = size(A, 1)

    # Compute D_Dl1
    D_Dl1 = _compute_D_Dl1(A, partitioning, isSymA, η)

    # Extract both strictly lower and upper triangular parts (within diagonal blocks only)
    L = _extract_strict_lower_sparse(A, N, partsize, nparts)
    U = _extract_strict_upper_sparse(A, N, partsize, nparts)
    L_adapted = adapt(backend, L)
    U_adapted = adapt(backend, U)

    sweepstorage = SymmetricSweepSparseTriangular(L_adapted, U_adapted)
    return D_Dl1, sweepstorage
end

_create_sweep_storage_with_precompute(::SymmetricSweep, ::PackedBuffer, A, partitioning, isSymA, η) =
    error("SymmetricSweep with PackedBuffer not yet implemented")

# Helper function to compute D_Dl1 only (used by OriginalMatrix storage)
function _compute_D_Dl1(A, partitioning, isSymA, η)
    (;partsize, nparts, nchunks, chunksize, backend) = partitioning
    Tf = eltype(A)
    N = size(A, 1)

    A_adapted = adapt(backend, A)
    D_Dl1 = adapt(backend, zeros(Tf, N))
    symA = isSymA ? SymmetricMatrix() : NonSymmetricMatrix()
    η_converted = convert(Tf, η)

    ndrange = nchunks * chunksize
    kernel = _compute_D_Dl1_kernel!(backend, chunksize, ndrange)
    kernel(D_Dl1, A_adapted, symA, partsize, nparts, nchunks, chunksize, η_converted; ndrange=ndrange)
    synchronize(backend)

    return D_Dl1
end

# extract strictly lower triangular matrices within diagonal blocks
function _extract_strict_lower_sparse(A::AbstractSparseMatrix, N, partsize, nparts)
    format_A = sparsemat_format_type(A)
    return _extract_strict_lower_sparse(A, format_A, N, partsize, nparts)
end

# CSC format: iterate columns (efficient for CSC)
function _extract_strict_lower_sparse(A, ::CSCFormat, N, partsize, nparts)
    I_vals = Int[]
    J_vals = Int[]
    V_vals = eltype(A)[]

    rows = rowvals(A)
    vals = nonzeros(A)

    for j = 1:N  # columns
        for idx in nzrange(A, j)
            i = rows[idx]

            # Determine which partition row i and column j belong to
            part_i = div(i - 1, partsize) + 1
            part_j = div(j - 1, partsize) + 1

            # Only include if: (1) same partition, (2) strictly lower (i > j)
            if part_i == part_j && i > j
                push!(I_vals, i)
                push!(J_vals, j)
                push!(V_vals, vals[idx])
            end
        end
    end

    # csr format in gs sweeps is more efficient
    return sparsecsr(I_vals, J_vals, V_vals, N, N)
end

# CSR format: iterate rows (efficient for CSR)
function _extract_strict_lower_sparse(A, ::CSRFormat, N, partsize, nparts)
    I_vals = Int[]
    J_vals = Int[]
    V_vals = eltype(A)[]

    rowPtr = getrowptr(A)
    colVal = colvals(A)
    nzVal = getnzval(A)

    for i = 1:N  # rows
        for idx in rowPtr[i]:(rowPtr[i+1]-1)
            j = colVal[idx]

            # Determine which partition row i and column j belong to
            part_i = div(i - 1, partsize) + 1
            part_j = div(j - 1, partsize) + 1

            # Only include if: (1) same partition, (2) strictly lower (i > j)
            if part_i == part_j && i > j
                push!(I_vals, i)
                push!(J_vals, j)
                push!(V_vals, nzVal[idx])
            end
        end
    end

    return sparsecsr(I_vals, J_vals, V_vals, N, N)
end

# extract strictly upper triangular matrices within diagonal blocks
function _extract_strict_upper_sparse(A::AbstractSparseMatrix, N, partsize, nparts)
    format_A = sparsemat_format_type(A)
    return _extract_strict_upper_sparse(A, format_A, N, partsize, nparts)
end

# CSC format: iterate columns (efficient for CSC)
function _extract_strict_upper_sparse(A, ::CSCFormat, N, partsize, nparts)
    I_vals = Int[]
    J_vals = Int[]
    V_vals = eltype(A)[]

    rows = rowvals(A)
    vals = nonzeros(A)

    for j = 1:N  # columns
        for idx in nzrange(A, j)
            i = rows[idx]

            # Determine which partition row i and column j belong to
            part_i = div(i - 1, partsize) + 1
            part_j = div(j - 1, partsize) + 1

            # Only include if: (1) same partition, (2) strictly upper (i < j)
            if part_i == part_j && i < j
                push!(I_vals, i)
                push!(J_vals, j)
                push!(V_vals, vals[idx])
            end
        end
    end

    # Return in CSR format for efficient row-wise access during sweep
    return sparsecsr(I_vals, J_vals, V_vals, N, N)
end

# CSR format: iterate rows (efficient for CSR)
function _extract_strict_upper_sparse(A, ::CSRFormat, N, partsize, nparts)
    I_vals = Int[]
    J_vals = Int[]
    V_vals = eltype(A)[]

    rowPtr = getrowptr(A)
    colVal = colvals(A)
    nzVal = getnzval(A)

    for i = 1:N  # rows
        for idx in rowPtr[i]:(rowPtr[i+1]-1)
            j = colVal[idx]

            # Determine which partition row i and column j belong to
            part_i = div(i - 1, partsize) + 1
            part_j = div(j - 1, partsize) + 1

            # Only include if: (1) same partition, (2) strictly upper (i < j)
            if part_i == part_j && i < j
                push!(I_vals, i)
                push!(J_vals, j)
                push!(V_vals, nzVal[idx])
            end
        end
    end

    # Return in CSR format for efficient row-wise access during sweep
    return sparsecsr(I_vals, J_vals, V_vals, N, N)
end


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

function _makecache(iterator::DiagonalPartsIterator, k::Ti) where {Ti<:Integer}
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


function _diag_offpart_csr(rowPtr, colVal, nzVal, idx::Ti, part_start::Ti, part_end::Ti) where {Ti<:Integer}
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

function _diag_offpart_csc(colPtr, rowVal, nzVal, idx::Ti, part_start::Ti, part_end::Ti) where {Ti<:Integer}
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

_diag_offpart(::NonSymmetricMatrix, ::CSCFormat, A, idx::Ti, part_start::Ti, part_end::Ti) where {Ti<:Integer} =
    _diag_offpart_csc(getcolptr(A), rowvals(A), getnzval(A), idx, part_start, part_end)

_diag_offpart(::SymmetricMatrix, ::CSCFormat, A, idx::Ti, part_start::Ti, part_end::Ti) where {Ti<:Integer} =
    _diag_offpart_csr(getcolptr(A), rowvals(A), getnzval(A), idx, part_start, part_end)

_diag_offpart(::AbstractMatrixSymmetry, ::CSRFormat, A, idx::Ti, part_start::Ti, part_end::Ti) where {Ti<:Integer} =
    _diag_offpart_csr(getrowptr(A), colvals(A), getnzval(A), idx, part_start, part_end)

function _pack_strict_lower_csr!(SLbuffer, rowPtr, colVal, nzVal, start_idx::Ti, end_idx::Ti, partsize::Ti, k::Ti) where {Ti<:Integer}
    block_stride = (partsize * (partsize - 1)) ÷ 2 # no. off-diagonal elements in a block 
    block_offset = (k - 1) * block_stride

    for i in start_idx:end_idx 
        local_i = i - start_idx + 1  
        # no. of off-diagonal elements in that row
        row_offset = ((local_i-1) * (local_i - 2)) ÷ 2  

        # scan the CSR row
        for p in rowPtr[i]:(rowPtr[i+1]-1)
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

function _pack_strict_lower_csc!(SLbuffer, colPtr, rowVal, nzVal, start_idx::Ti, end_idx::Ti, partsize::Ti, k::Ti) where {Ti<:Integer}
    block_stride = (partsize * (partsize - 1)) ÷ 2 # no. off-diagonal elements in a block
    block_offset = (k - 1) * block_stride
    for col in start_idx:(end_idx-1) 
        local_j = col - start_idx + 1
        for p in colPtr[col]:(colPtr[col+1]-1)
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

_pack_strict_lower!(::NonSymmetricMatrix, ::CSCFormat, SLbuffer, A, start_idx::Ti, end_idx::Ti, partsize::Ti, k::Ti) where {Ti<:Integer} =
    _pack_strict_lower_csc!(SLbuffer, getcolptr(A), rowvals(A), getnzval(A), start_idx, end_idx, partsize, k)

_pack_strict_lower!(::SymmetricMatrix, ::CSCFormat, SLbuffer, A, start_idx::Ti, end_idx::Ti, partsize::Ti, k::Ti) where {Ti<:Integer} =
    _pack_strict_lower_csr!(SLbuffer, getcolptr(A), rowvals(A), getnzval(A), start_idx, end_idx, partsize, k)

_pack_strict_lower!(::AbstractMatrixSymmetry, ::CSRFormat, SLbuffer, A, start_idx::Ti, end_idx::Ti, partsize::Ti, k::Ti) where {Ti<:Integer} =
    _pack_strict_lower_csr!(SLbuffer, getrowptr(A), colvals(A), getnzval(A), start_idx, end_idx, partsize, k)

# Pack upper triangular functions (for backward sweep)
# For upper triangular, we store elements row by row where j > i
# Row i has (partsize - local_i) upper elements
function _pack_strict_upper_csr!(SUbuffer, rowPtr, colVal, nzVal, start_idx::Ti, end_idx::Ti, partsize::Ti, k::Ti) where {Ti<:Integer}
    block_stride = (partsize * (partsize - 1)) ÷ 2
    block_offset = (k - 1) * block_stride

    idx = 1
    for i in start_idx:end_idx
        for p in rowPtr[i]:(rowPtr[i+1]-1)
            j = colVal[p]
            if j > i && j <= end_idx  # strictly upper within partition
                SUbuffer[block_offset + idx] = nzVal[p]
                idx += 1
            end
        end
    end
    return nothing
end

function _pack_strict_upper_csc!(SUbuffer, colPtr, rowVal, nzVal, start_idx::Ti, end_idx::Ti, partsize::Ti, k::Ti) where {Ti<:Integer}
    block_stride = (partsize * (partsize - 1)) ÷ 2
    block_offset = (k - 1) * block_stride

    idx = 1
    # Iterate rows from start to end
    for i in start_idx:end_idx
        # Find all j > i in row i by scanning columns
        for col in (i+1):end_idx
            for p in colPtr[col]:(colPtr[col+1]-1)
                row = rowVal[p]
                if row == i
                    SUbuffer[block_offset + idx] = nzVal[p]
                    idx += 1
                    break
                end
            end
        end
    end
    return nothing
end

_pack_strict_upper!(::NonSymmetricMatrix, ::CSCFormat, SUbuffer, A, start_idx::Ti, end_idx::Ti, partsize::Ti, k::Ti) where {Ti<:Integer} =
    _pack_strict_upper_csc!(SUbuffer, getcolptr(A), rowvals(A), getnzval(A), start_idx, end_idx, partsize, k)

_pack_strict_upper!(::SymmetricMatrix, ::CSCFormat, SUbuffer, A, start_idx::Ti, end_idx::Ti, partsize::Ti, k::Ti) where {Ti<:Integer} =
    _pack_strict_upper_csr!(SUbuffer, getcolptr(A), rowvals(A), getnzval(A), start_idx, end_idx, partsize, k)

_pack_strict_upper!(::AbstractMatrixSymmetry, ::CSRFormat, SUbuffer, A, start_idx::Ti, end_idx::Ti, partsize::Ti, k::Ti) where {Ti<:Integer} =
    _pack_strict_upper_csr!(SUbuffer, getrowptr(A), colvals(A), getnzval(A), start_idx, end_idx, partsize, k)


## Specialized kernels for efficient construction ##
# These kernels merge D_Dl1 computation with buffer packing to avoid wasteful allocations

# Kernel that only computes D_Dl1 (used by OriginalMatrix storage)
@kernel function _compute_D_Dl1_kernel!(D_Dl1, A, symA, partsize::Ti, nparts::Ti, nchunks::Ti, chunksize::Ti, η) where {Ti<:Integer}
    initial_partition_idx = @index(Global)
    size_A = convert(Ti, size(A, 1))
    format_A = sparsemat_format_type(A)

    for part in DiagonalPartsIterator(size_A, partsize, nparts, nchunks, chunksize, convert(Ti, initial_partition_idx))
        (;k, partsize, start_idx, end_idx) = part
        for i in start_idx:end_idx
            a_ii, dl1_ii = _diag_offpart(symA, format_A, A, i, start_idx, end_idx)
            TF = eltype(D_Dl1)
            dl1star_ii = a_ii >= η * dl1_ii ? zero(TF) : dl1_ii / convert(TF, 2.0)
            D_Dl1[i] = a_ii + dl1star_ii
        end
    end
end

# Kernel that computes D_Dl1 AND packs lower triangular (ForwardSweep + PackedBuffer)
@kernel function _precompute_and_pack_lower_kernel!(D_Dl1, SLbuffer, A, symA, partsize::Ti, nparts::Ti, nchunks::Ti, chunksize::Ti, η) where {Ti<:Integer}
    initial_partition_idx = @index(Global)
    size_A = convert(Ti, size(A, 1))
    format_A = sparsemat_format_type(A)

    for part in DiagonalPartsIterator(size_A, partsize, nparts, nchunks, chunksize, convert(Ti, initial_partition_idx))
        (;k, partsize, start_idx, end_idx) = part

        # Compute D_Dl1
        for i in start_idx:end_idx
            a_ii, dl1_ii = _diag_offpart(symA, format_A, A, i, start_idx, end_idx)
            TF = eltype(D_Dl1)
            dl1star_ii = a_ii >= η * dl1_ii ? zero(TF) : dl1_ii / convert(TF, 2.0)
            D_Dl1[i] = a_ii + dl1star_ii
        end

        # Pack strictly lower triangular
        _pack_strict_lower!(symA, format_A, SLbuffer, A, start_idx, end_idx, partsize, k)
    end
end

# Kernel that computes D_Dl1 AND packs upper triangular (BackwardSweep + PackedBuffer)
@kernel function _precompute_and_pack_upper_kernel!(D_Dl1, SUbuffer, A, symA, partsize::Ti, nparts::Ti, nchunks::Ti, chunksize::Ti, η) where {Ti<:Integer}
    initial_partition_idx = @index(Global)
    size_A = convert(Ti, size(A, 1))
    format_A = sparsemat_format_type(A)

    for part in DiagonalPartsIterator(size_A, partsize, nparts, nchunks, chunksize, convert(Ti, initial_partition_idx))
        (;k, partsize, start_idx, end_idx) = part

        # Compute D_Dl1
        for i in start_idx:end_idx
            a_ii, dl1_ii = _diag_offpart(symA, format_A, A, i, start_idx, end_idx)
            TF = eltype(D_Dl1)
            dl1star_ii = a_ii >= η * dl1_ii ? zero(TF) : dl1_ii / convert(TF, 2.0)
            D_Dl1[i] = a_ii + dl1star_ii
        end

        # Pack strictly upper triangular
        _pack_strict_upper!(symA, format_A, SUbuffer, A, start_idx, end_idx, partsize, k)
    end
end


# Main sweep dispatcher - dispatches directly on SweepStorage type
function _apply_sweep!(y, P)
    @timeit_debug "_apply_sweep!" begin
        (; sweepstorage) = P
        _apply_sweep!(y, P, sweepstorage)
    end
    return nothing
end

# Forward sweep with OriginalMatrix
function _apply_sweep!(y, P, ss::ForwardSweepOriginalMatrix)
    @timeit_debug "_apply_sweep! (ForwardSweepOriginalMatrix)" begin
        (; partitioning, D_Dl1) = P
        (;partsize, nparts, nchunks, chunksize, backend) = partitioning
        ndrange = nchunks * chunksize
        @timeit_debug "kernel construction" kernel = _forward_sweep_original_kernel!(backend, chunksize, ndrange)
        size_A = convert(typeof(nparts), length(y))
        @timeit_debug "kernel call" kernel(y, D_Dl1, ss.A, size_A, partsize, nparts, nchunks, chunksize; ndrange=ndrange)
        @timeit_debug "synchronize" synchronize(backend)
    end
    return nothing
end

# Forward sweep with PackedBuffer
function _apply_sweep!(y, P, ss::ForwardSweepPackedBuffer)
    @timeit_debug "_apply_sweep! (ForwardSweepPackedBuffer)" begin
        (; partitioning, D_Dl1) = P
        (;partsize, nparts, nchunks, chunksize, backend) = partitioning
        ndrange = nchunks * chunksize
        @timeit_debug "kernel construction" kernel = _forward_sweep_packed_kernel!(backend, chunksize, ndrange)
        size_A = convert(typeof(nparts), length(y))
        @timeit_debug "kernel call" kernel(y, D_Dl1, ss.SLbuffer, size_A, partsize, nparts, nchunks, chunksize; ndrange=ndrange)
        @timeit_debug "synchronize" synchronize(backend)
    end
    return nothing
end

# Backward sweep with OriginalMatrix
function _apply_sweep!(y, P, ss::BackwardSweepOriginalMatrix)
    @timeit_debug "_apply_sweep! (BackwardSweepOriginalMatrix)" begin
        (; partitioning, D_Dl1) = P
        (;partsize, nparts, nchunks, chunksize, backend) = partitioning
        ndrange = nchunks * chunksize
        @timeit_debug "kernel construction" kernel = _backward_sweep_original_kernel!(backend, chunksize, ndrange)
        size_A = convert(typeof(nparts), length(y))
        @timeit_debug "kernel call" kernel(y, D_Dl1, ss.A, size_A, partsize, nparts, nchunks, chunksize; ndrange=ndrange)
        @timeit_debug "synchronize" synchronize(backend)
    end
    return nothing
end

# Backward sweep with PackedBuffer
function _apply_sweep!(y, P, ss::BackwardSweepPackedBuffer)
    @timeit_debug "_apply_sweep! (BackwardSweepPackedBuffer)" begin
        (; partitioning, D_Dl1) = P
        (;partsize, nparts, nchunks, chunksize, backend) = partitioning
        ndrange = nchunks * chunksize
        @timeit_debug "kernel construction" kernel = _backward_sweep_packed_kernel!(backend, chunksize, ndrange)
        size_A = convert(typeof(nparts), length(y))
        @timeit_debug "kernel call" kernel(y, D_Dl1, ss.SUbuffer, size_A, partsize, nparts, nchunks, chunksize; ndrange=ndrange)
        @timeit_debug "synchronize" synchronize(backend)
    end
    return nothing
end

# Symmetric sweep with OriginalMatrix
function _apply_sweep!(y, P, ss::SymmetricSweepOriginalMatrix)
    @timeit_debug "_apply_sweep! (SymmetricSweepOriginalMatrix)" begin
        (; partitioning, D_Dl1) = P
        (;partsize, nparts, nchunks, chunksize, backend) = partitioning
        ndrange = nchunks * chunksize
        size_A = convert(typeof(nparts), length(y))

        # Forward sweep
        @timeit_debug "forward kernel" begin
            kernel_fwd = _forward_sweep_original_kernel!(backend, chunksize, ndrange)
            kernel_fwd(y, D_Dl1, ss.A, size_A, partsize, nparts, nchunks, chunksize; ndrange=ndrange)
            synchronize(backend)
        end

        # Backward sweep
        @timeit_debug "backward kernel" begin
            kernel_bwd = _backward_sweep_original_kernel!(backend, chunksize, ndrange)
            kernel_bwd(y, D_Dl1, ss.A, size_A, partsize, nparts, nchunks, chunksize; ndrange=ndrange)
            synchronize(backend)
        end
    end
    return nothing
end

# Symmetric sweep with PackedBuffer
function _apply_sweep!(y, P, ss::SymmetricSweepPackedBuffer)
    @timeit_debug "_apply_sweep! (SymmetricSweepPackedBuffer)" begin
        (; partitioning, D_Dl1) = P
        (;partsize, nparts, nchunks, chunksize, backend) = partitioning
        ndrange = nchunks * chunksize
        size_A = convert(typeof(nparts), length(y))

        # Forward sweep
        @timeit_debug "forward kernel" begin
            kernel_fwd = _forward_sweep_packed_kernel!(backend, chunksize, ndrange)
            kernel_fwd(y, D_Dl1, ss.SLbuffer, size_A, partsize, nparts, nchunks, chunksize; ndrange=ndrange)
            synchronize(backend)
        end

        # Backward sweep
        @timeit_debug "backward kernel" begin
            kernel_bwd = _backward_sweep_packed_kernel!(backend, chunksize, ndrange)
            kernel_bwd(y, D_Dl1, ss.SUbuffer, size_A, partsize, nparts, nchunks, chunksize; ndrange=ndrange)
            synchronize(backend)
        end
    end
    return nothing
end

# Forward sweep with SparseTriangular
function _apply_sweep!(y, P, ss::ForwardSweepSparseTriangular)
    @timeit_debug "_apply_sweep! (ForwardSweepSparseTriangular)" begin
        (; partitioning, D_Dl1) = P
        (;partsize, nparts, nchunks, chunksize, backend) = partitioning
        ndrange = nchunks * chunksize
        @timeit_debug "kernel construction" kernel = _forward_sweep_sparse_kernel!(backend, chunksize, ndrange)
        size_A = convert(typeof(nparts), length(y))
        @timeit_debug "kernel call" kernel(y, D_Dl1, ss.L, size_A, partsize, nparts, nchunks, chunksize; ndrange=ndrange)
        @timeit_debug "synchronize" synchronize(backend)
    end
    return nothing
end

# Backward sweep with SparseTriangular
function _apply_sweep!(y, P, ss::BackwardSweepSparseTriangular)
    @timeit_debug "_apply_sweep! (BackwardSweepSparseTriangular)" begin
        (; partitioning, D_Dl1) = P
        (;partsize, nparts, nchunks, chunksize, backend) = partitioning
        ndrange = nchunks * chunksize
        @timeit_debug "kernel construction" kernel = _backward_sweep_sparse_kernel!(backend, chunksize, ndrange)
        size_A = convert(typeof(nparts), length(y))
        @timeit_debug "kernel call" kernel(y, D_Dl1, ss.U, size_A, partsize, nparts, nchunks, chunksize; ndrange=ndrange)
        @timeit_debug "synchronize" synchronize(backend)
    end
    return nothing
end

# Symmetric sweep with SparseTriangular
function _apply_sweep!(y, P, ss::SymmetricSweepSparseTriangular)
    @timeit_debug "_apply_sweep! (SymmetricSweepSparseTriangular)" begin
        (; partitioning, D_Dl1) = P
        (;partsize, nparts, nchunks, chunksize, backend) = partitioning
        ndrange = nchunks * chunksize
        size_A = convert(typeof(nparts), length(y))

        # Forward sweep with L
        @timeit_debug "forward kernel" begin
            kernel_fwd = _forward_sweep_sparse_kernel!(backend, chunksize, ndrange)
            kernel_fwd(y, D_Dl1, ss.L, size_A, partsize, nparts, nchunks, chunksize; ndrange=ndrange)
            synchronize(backend)
        end

        # Backward sweep with U
        @timeit_debug "backward kernel" begin
            kernel_bwd = _backward_sweep_sparse_kernel!(backend, chunksize, ndrange)
            kernel_bwd(y, D_Dl1, ss.U, size_A, partsize, nparts, nchunks, chunksize; ndrange=ndrange)
            synchronize(backend)
        end
    end
    return nothing
end

# Forward sweep kernel with PackedBuffer
@kernel function _forward_sweep_packed_kernel!(y, D_Dl1, buffer, size_A::Ti, partsize::Ti, nparts::Ti, nchunks::Ti, chunksize::Ti) where {Ti<:Integer}
    initial_partition_idx = @index(Global)
    for part in DiagonalPartsIterator(size_A, partsize, nparts, nchunks, chunksize, convert(Ti, initial_partition_idx))
        @unpack k, partsize, start_idx, end_idx = part
        block_stride = (partsize * (partsize - 1)) ÷ 2
        block_offset = (k - 1) * block_stride

        # forward‐solve: (1/a_{ii})[bᵢ - ∑_{j<i} a_ij * x_j]
        for i in start_idx:end_idx
            local_i = i - start_idx + 1
            row_offset = ((local_i-1) * (local_i - 2)) ÷ 2

            acc = zero(eltype(y))
            @inbounds for local_j in 1:(local_i-1)
                gj = start_idx + (local_j - 1)
                off_idx = block_offset + row_offset + (local_j)
                acc += buffer[off_idx] * y[gj]
            end

            y[i] = (y[i] - acc) / D_Dl1[i]
        end
    end
end

# Forward sweep kernel with OriginalMatrix
@kernel function _forward_sweep_original_kernel!(y, D_Dl1, A, size_A::Ti, partsize::Ti, nparts::Ti, nchunks::Ti, chunksize::Ti) where {Ti<:Integer}
    initial_partition_idx = @index(Global)
    format_A = sparsemat_format_type(A)

    for part in DiagonalPartsIterator(size_A, partsize, nparts, nchunks, chunksize, convert(Ti, initial_partition_idx))
        @unpack k, partsize, start_idx, end_idx = part

        # forward‐solve using original matrix A
        for i in start_idx:end_idx
            acc = zero(eltype(y))
            acc = _accumulate_lower_from_matrix(A, format_A, y, i, start_idx, end_idx)
            y[i] = (y[i] - acc) / D_Dl1[i]
        end
    end
end

# Backward sweep kernel with PackedBuffer
# Upper triangular elements are stored row by row: row i contains all j > i
@kernel function _backward_sweep_packed_kernel!(y, D_Dl1, buffer, size_A::Ti, partsize::Ti, nparts::Ti, nchunks::Ti, chunksize::Ti) where {Ti<:Integer}
    initial_partition_idx = @index(Global)
    for part in DiagonalPartsIterator(size_A, partsize, nparts, nchunks, chunksize, convert(Ti, initial_partition_idx))
        @unpack k, partsize, start_idx, end_idx = part
        block_stride = (partsize * (partsize - 1)) ÷ 2
        block_offset = (k - 1) * block_stride

        # backward‐solve: iterate rows in reverse order
        for i in end_idx:-1:start_idx
            num_upper = end_idx - i  # number of upper elements in row i

            acc = zero(eltype(y))
            # We need to find where row i's upper elements are stored
            # Count elements in rows before i
            elements_before = 0
            for ii in start_idx:(i-1)
                elements_before += (end_idx - ii)
            end

            @inbounds for offset in 1:num_upper
                j = i + offset
                buf_idx = block_offset + elements_before + offset
                acc += buffer[buf_idx] * y[j]
            end

            y[i] = (y[i] - acc) / D_Dl1[i]
        end
    end
end

# Backward sweep kernel with OriginalMatrix
@kernel function _backward_sweep_original_kernel!(y, D_Dl1, A, size_A::Ti, partsize::Ti, nparts::Ti, nchunks::Ti, chunksize::Ti) where {Ti<:Integer}
    initial_partition_idx = @index(Global)
    format_A = sparsemat_format_type(A)

    for part in DiagonalPartsIterator(size_A, partsize, nparts, nchunks, chunksize, convert(Ti, initial_partition_idx))
        @unpack k, partsize, start_idx, end_idx = part

        # backward‐solve using original matrix A: iterate rows in reverse
        for i in end_idx:-1:start_idx
            acc = zero(eltype(y))
            acc = _accumulate_upper_from_matrix(A, format_A, y, i, start_idx, end_idx)
            y[i] = (y[i] - acc) / D_Dl1[i]
        end
    end
end

# Helper functions to accumulate from original matrix
function _accumulate_lower_from_matrix(A, ::CSRFormat, y, i, start_idx, end_idx)
    rowPtr = getrowptr(A)
    colVal = colvals(A)
    nzVal = getnzval(A)

    acc = zero(eltype(y))
    for p in rowPtr[i]:(rowPtr[i+1]-1)
        j = colVal[p]
        if j >= start_idx && j < i  # strictly lower within partition
            acc += nzVal[p] * y[j]
        end
    end
    return acc
end

function _accumulate_lower_from_matrix(A, ::CSCFormat, y, i, start_idx, end_idx)
    colPtr = getcolptr(A)
    rowVal = rowvals(A)
    nzVal = getnzval(A)

    acc = zero(eltype(y))
    for col in start_idx:(i-1)  # columns before row i
        for p in colPtr[col]:(colPtr[col+1]-1)
            row = rowVal[p]
            if row == i
                acc += nzVal[p] * y[col]
            end
        end
    end
    return acc
end

function _accumulate_upper_from_matrix(A, ::CSRFormat, y, i, start_idx, end_idx)
    rowPtr = getrowptr(A)
    colVal = colvals(A)
    nzVal = getnzval(A)

    acc = zero(eltype(y))
    for p in rowPtr[i]:(rowPtr[i+1]-1)
        j = colVal[p]
        if j > i && j <= end_idx  # strictly upper within partition
            acc += nzVal[p] * y[j]
        end
    end
    return acc
end

function _accumulate_upper_from_matrix(A, ::CSCFormat, y, i, start_idx, end_idx)
    colPtr = getcolptr(A)
    rowVal = rowvals(A)
    nzVal = getnzval(A)

    acc = zero(eltype(y))
    for col in (i+1):end_idx  # columns after row i
        for p in colPtr[col]:(colPtr[col+1]-1)
            row = rowVal[p]
            if row == i
                acc += nzVal[p] * y[col]
            end
        end
    end
    return acc
end

# Forward sweep kernel with SparseTriangular (using sparse L matrix)
@kernel function _forward_sweep_sparse_kernel!(y, D_Dl1, L, size_A::Ti, partsize::Ti, nparts::Ti, nchunks::Ti, chunksize::Ti) where {Ti<:Integer}
    initial_partition_idx = @index(Global)
    format_L = sparsemat_format_type(L)

    for part in DiagonalPartsIterator(size_A, partsize, nparts, nchunks, chunksize, convert(Ti, initial_partition_idx))
        @unpack k, partsize, start_idx, end_idx = part

        # forward‐solve using sparse L matrix
        for i in start_idx:end_idx
            acc = zero(eltype(y))
            acc = _accumulate_lower_from_matrix(L, format_L, y, i, start_idx, end_idx)
            y[i] = (y[i] - acc) / D_Dl1[i]
        end
    end
end

# Backward sweep kernel with SparseTriangular (using sparse U matrix)
@kernel function _backward_sweep_sparse_kernel!(y, D_Dl1, U, size_A::Ti, partsize::Ti, nparts::Ti, nchunks::Ti, chunksize::Ti) where {Ti<:Integer}
    initial_partition_idx = @index(Global)
    format_U = sparsemat_format_type(U)

    for part in DiagonalPartsIterator(size_A, partsize, nparts, nchunks, chunksize, convert(Ti, initial_partition_idx))
        @unpack k, partsize, start_idx, end_idx = part

        # backward‐solve using sparse U matrix: iterate rows in reverse
        for i in end_idx:-1:start_idx
            acc = zero(eltype(y))
            acc = _accumulate_upper_from_matrix(U, format_U, y, i, start_idx, end_idx)
            y[i] = (y[i] - acc) / D_Dl1[i]
        end
    end
end
