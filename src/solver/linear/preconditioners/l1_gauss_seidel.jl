####################################
## L1 Gauss Seidel Preconditioner ##
####################################

## User layer ##
abstract type Sweep end
struct ForwardSweep <: Sweep end # forward sweep -> lower triangular
struct BackwardSweep <: Sweep end # backward sweep -> upper triangular
struct SymmetricSweep <: Sweep end # symmetric sweep -> both lower and upper triangular

# FIXME: Is there better naming alternatives for this?
abstract type CacheStrategy end
struct PackedBufferCache <: CacheStrategy end # store lower/upper triangular parts in a vector buffer -> efficient for small partitions (e.g. GPU)
struct MatrixViewCache <: CacheStrategy end # store a sparse triangular matrix view -> efficient for large partitions (e.g. CPU)

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

# Forward declaration of sweep plan types (defined later in the file)
abstract type L1GSSweepPlan end

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
struct L1GSPreconditioner{Partitioning, SweepPlanType <: L1GSSweepPlan}
    partitioning::Partitioning
    sweep::SweepPlanType
    # D_Dl1::VectorType # D + Dˡ
    # sweepstorage::SweepStorageType # Encapsulates sweep direction and storage strategy
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

function (builder::L1GSPrecBuilder)(
    A::AbstractMatrix,
    partsize::Ti;
    isSymA::Bool = false,
    η = 1.5,
    sweep::Sweep = ForwardSweep(),
    cache_strategy::CacheStrategy = _choose_default_device_storage(builder.device),
) where {Ti <: Integer}
    build_l1prec(builder, A, partsize, isSymA, η, sweep, cache_strategy)
end

function (builder::L1GSPrecBuilder)(
    A::Symmetric,
    partsize::Ti;
    η = 1.5,
    sweep::Sweep = SymmetricSweep(),
    cache_strategy::CacheStrategy = OriginalMatrix(),
) where {Ti <: Integer}
    build_l1prec(builder, A, partsize, true, η, sweep, cache_strategy)
end


## Preconditioner builder ##
function build_l1prec(
    builder::L1GSPrecBuilder,
    A::MatrixType,
    partsize::Ti,
    isSymA::Bool,
    η,
    sweep::Sweep,
    cache_strategy::CacheStrategy,
) where {Ti <: Integer, MatrixType}
    partsize == 0 && error("partsize must be greater than 0")

    # TODO: do we need this validation (presumably symmetric sweep only works for symmetric matrices)?
    if sweep isa SymmetricSweep && !isSymA && !(A isa Symmetric)
        error(
            "SymmetricSweep requires a symmetric matrix. If `A` is symmetric, then either pass isSymA=true or use Symmetric(A). If not symmetric, consider using `ForwardSweep` or `BackwardSweep` instead.",
        )
    end

    _build_l1prec(builder, A, partsize, isSymA, η, sweep, cache_strategy)
end

function _build_l1prec(
    builder::L1GSPrecBuilder,
    _A::MatrixType,
    partsize::Ti,
    isSymA::Bool,
    η,
    sweep::Sweep,
    cache_strategy::CacheStrategy,
) where {Ti <: Integer, MatrixType}
    # `nchunks` is either CPU cores or GPU blocks.
    # Each chunk will be assigned `nparts`, each of size `partsize`.
    # In GPU backend, `nchunks` is the number of blocks and `partsize` is the number of threads per block.
    A = get_data(_A) # for symmetric case
    partitioning = _blockpartitioning(builder, A, partsize)

    sweep = _make_sweep_plan(sweep, cache_strategy, A, partitioning, isSymA, η)

    L1GSPreconditioner(partitioning, sweep)
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
    P::L1GSPreconditioner,
    x::VectorType,
) where {VectorType <: AbstractVector}
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

function LinearSolve.ldiv!(
    y::Vector,
    P::L1GSPreconditioner{BlockPartitioning{Ti, CPU}},
    x::Vector,
) where {Ti <: Integer}
    @timeit_debug "ldiv! (CPU)" begin
        @timeit_debug "y .= x" y .= x
        @timeit_debug "_apply_sweep!" _apply_sweep!(y, P)
    end
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

####################
## Dispatch layer ##
####################

# types that encapsulate the metadata/structure of A


abstract type AbstractDiagonalIndices end

## DiagonalIndices - for efficient CSC matrix access 
## this code is adapted from iterativesolvers.jl
struct DiagonalIndices{Ti <: Integer} <: AbstractDiagonalIndices
    diag::Vector{Ti}

    function DiagonalIndices{Ti}(A::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
        diag = Vector{Ti}(undef, A.n)

        for col = 1 : A.n
            r1 = Int(A.colptr[col])
            r2 = Int(A.colptr[col + 1] - 1)
            r1 = searchsortedfirst(A.rowval, col, r1, r2, Base.Order.Forward)
            if r1 > r2 || A.rowval[r1] != col || iszero(A.nzval[r1])
                throw(LinearAlgebra.SingularException(col))
            end
            diag[col] = r1
        end

        new(diag)
    end
end

DiagonalIndices(A::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti} = DiagonalIndices{Ti}(A)

@inline Base.getindex(d::DiagonalIndices, i::Int) = d.diag[i]
@inline Base.length(d::DiagonalIndices) = length(d.diag)


struct NoDiagonalIndices <: AbstractDiagonalIndices end


abstract type AbstractCache end
abstract type AbstractLowerCache <: AbstractCache end
abstract type AbstractUpperCache <: AbstractCache end


struct BlockStrictLowerView{MatrixType, SymmetryType<:AbstractMatrixSymmetry, FormatType<:AbstractMatrixFormat, DIType<:AbstractDiagonalIndices} <: AbstractLowerCache
    A::MatrixType
    symA::SymmetryType
    frmt::FormatType
    diag::DIType # imp for csc format (DiagonalIndices for CSC, NoDiagonalIndices for CSR)
end

struct BlockStrictUpperView{MatrixType, SymmetryType<:AbstractMatrixSymmetry, FormatType<:AbstractMatrixFormat, DIType<:AbstractDiagonalIndices} <: AbstractUpperCache
    A::MatrixType
    symA::SymmetryType
    frmt::FormatType
    diag::DIType  # DiagonalIndices for CSC, NoDiagonalIndices for CSR
end

"""
    PackedStrictLower{VectorType}

Cache for strictly lower triangular elements packed into a dense buffer.
Optimal for GPU execution with small partition sizes.
Layout: Elements packed row-by-row, with block_offset = (k-1) * partsize*(partsize-1)/2
"""
struct PackedStrictLower{VectorType} <: AbstractLowerCache
    SLbuffer::VectorType
end

"""
    PackedStrictUpper{VectorType}

Cache for strictly upper triangular elements packed into a dense buffer.
Optimal for GPU execution with small partition sizes.
Layout: Elements packed row-by-row, with block_offset = (k-1) * partsize*(partsize-1)/2
"""
struct PackedStrictUpper{VectorType} <: AbstractUpperCache
    SUbuffer::VectorType
end

abstract type AbstractBlockSolveOperator end

struct BlockLowerSolveOperator{LowerCache <: AbstractLowerCache, VectorType} <: AbstractBlockSolveOperator
    L::LowerCache
    D_DL1::VectorType
end

struct BlockUpperSolveOperator{UpperCache <: AbstractUpperCache, VectorType} <: AbstractBlockSolveOperator
    U::UpperCache
    D_DL1::VectorType
end

struct ForwardL1GSSweep{LowerOp<:BlockLowerSolveOperator} <: L1GSSweepPlan
    op::LowerOp
end

struct BackwardL1GSSweep{UpperOp<:BlockUpperSolveOperator} <: L1GSSweepPlan
    op::UpperOp
end

struct SymmetricL1GSSweep{LowerOp<:BlockLowerSolveOperator, UpperOp<:BlockUpperSolveOperator} <: L1GSSweepPlan
    lop::LowerOp
    uop::UpperOp
end


## L1 GS internal functionalty ##
get_data(A::AbstractSparseMatrix) = A
get_data(A::Symmetric{Ti, TA}) where {Ti, TA} = TA(A.data) # restore the full matrix, why ? https://discourse.julialang.org/t/is-there-a-symmetric-sparse-matrix-implementation-in-julia/91333/2


_choose_default_device_storage(::AbstractDevice) = MatrixViewCache()
_choose_default_device_storage(::AbstractGPUDevice) = PackedBufferCache()


# Helper to create diagonal indices based on format
_create_diag_indices(A::SparseMatrixCSC, ::CSCFormat) = DiagonalIndices(A)
_create_diag_indices(A, ::CSRFormat) = NoDiagonalIndices()

function BlockStrictLowerView(A::SparseMatrixCSC, symA, frmt::CSCFormat)
    diag = DiagonalIndices(A) # precompute diagonal indices for efficient access
    BlockStrictLowerView(A, symA, frmt, diag)
end

function BlockStrictLowerView(A, symA, frmt::CSRFormat)
    # here no need for diagonal indices (already optimal access pattern)
    BlockStrictLowerView(A, symA, frmt, NoDiagonalIndices())
end

function BlockStrictUpperView(A::SparseMatrixCSC, symA, frmt::CSCFormat)
    diag = DiagonalIndices(A)
    BlockStrictUpperView(A, symA, frmt, diag)
end

function BlockStrictUpperView(A, symA, frmt::CSRFormat)
    BlockStrictUpperView(A, symA, frmt, NoDiagonalIndices())
end


## Dispatch layer helper functions ##

function _make_sweep_plan(::ForwardSweep, ::MatrixViewCache, A, partitioning, isSymA, η)
    symA = isSymA ? SymmetricMatrix() : NonSymmetricMatrix()
    frmt = sparsemat_format_type(A)

    # Create diagonal indices (if needed for CSC)
    diag = _create_diag_indices(A, frmt)

    # Compute D_Dl1 using diagonal indices for optimization
    D_Dl1 = _compute_D_Dl1(A, partitioning, symA, frmt, diag, η)

    # Create cache with diagonal indices
    cache = BlockStrictLowerView(A, symA, frmt, diag)
    lop = BlockLowerSolveOperator(cache, D_Dl1)
    return ForwardL1GSSweep(lop)
end

function _make_sweep_plan(::BackwardSweep, ::MatrixViewCache, A, partitioning, isSymA, η)
    symA = isSymA ? SymmetricMatrix() : NonSymmetricMatrix()
    frmt = sparsemat_format_type(A)

    # Create diagonal indices (if needed for CSC)
    diag = _create_diag_indices(A, frmt)

    # Compute D_Dl1 using diagonal indices for optimization
    D_Dl1 = _compute_D_Dl1(A, partitioning, symA, frmt, diag, η)

    # Create cache with diagonal indices
    cache = BlockStrictUpperView(A, symA, frmt, diag)
    uop = BlockUpperSolveOperator(cache, D_Dl1)
    return BackwardL1GSSweep(uop)
end

function _make_sweep_plan(::SymmetricSweep, ::MatrixViewCache, A, partitioning, isSymA, η)
    symA = isSymA ? SymmetricMatrix() : NonSymmetricMatrix()
    frmt = sparsemat_format_type(A)

    # Create diagonal indices (if needed for CSC)
    diag = _create_diag_indices(A, frmt)

    D_Dl1 = _compute_D_Dl1(A, partitioning, symA, frmt, diag, η)

    L_cache = BlockStrictLowerView(A, symA, frmt, diag)
    U_cache = BlockStrictUpperView(A, symA, frmt, diag)

    lop = BlockLowerSolveOperator(L_cache, D_Dl1)
    uop = BlockUpperSolveOperator(U_cache, D_Dl1)

    return SymmetricL1GSSweep(lop, uop)
end

function _make_sweep_plan(::ForwardSweep, ::PackedBufferCache, A, partitioning, isSymA, η)
    (; partsize, nparts, nchunks, chunksize, backend) = partitioning
    symA = isSymA ? SymmetricMatrix() : NonSymmetricMatrix()
    frmt = sparsemat_format_type(A)
    Tf = eltype(A)
    N = size(A, 1)

    # Create diagonal indices (if needed for CSC)
    diag = _create_diag_indices(A, frmt)

    # Allocate buffers
    A_adapted = adapt(backend, A)
    diag_adapted = adapt(backend, diag)
    D_Dl1 = adapt(backend, zeros(Tf, N))

    # Buffer size: partsize*(partsize-1)/2 elements per partition
    buffer_size = nparts * ((partsize * (partsize - 1)) ÷ 2)
    SLbuffer = adapt(backend, zeros(Tf, buffer_size))

    η_converted = convert(Tf, η)

    # Single kernel call to compute D_Dl1 and pack lower triangular
    ndrange = nchunks * chunksize
    kernel = _precompute_and_pack_lower_kernel!(backend, chunksize, ndrange)
    kernel(
        D_Dl1,
        SLbuffer,
        A_adapted,
        symA,
        frmt,
        diag_adapted,
        partsize,
        nparts,
        nchunks,
        chunksize,
        η_converted;
        ndrange = ndrange,
    )
    synchronize(backend)

    cache = PackedStrictLower(SLbuffer)
    lop = BlockLowerSolveOperator(cache, D_Dl1)
    return ForwardL1GSSweep(lop)
end

function _make_sweep_plan(::BackwardSweep, ::PackedBufferCache, A, partitioning, isSymA, η)
    (; partsize, nparts, nchunks, chunksize, backend) = partitioning
    symA = isSymA ? SymmetricMatrix() : NonSymmetricMatrix()
    frmt = sparsemat_format_type(A)
    Tf = eltype(A)
    N = size(A, 1)

    # Create diagonal indices (if needed for CSC)
    diag = _create_diag_indices(A, frmt)

    # Allocate buffers
    A_adapted = adapt(backend, A)
    diag_adapted = adapt(backend, diag)
    D_Dl1 = adapt(backend, zeros(Tf, N))

    # Buffer size: partsize*(partsize-1)/2 elements per partition
    buffer_size = nparts * ((partsize * (partsize - 1)) ÷ 2)
    SUbuffer = adapt(backend, zeros(Tf, buffer_size))

    η_converted = convert(Tf, η)

    # Single kernel call to compute D_Dl1 and pack upper triangular
    ndrange = nchunks * chunksize
    kernel = _precompute_and_pack_upper_kernel!(backend, chunksize, ndrange)
    kernel(
        D_Dl1,
        SUbuffer,
        A_adapted,
        symA,
        frmt,
        diag_adapted,
        partsize,
        nparts,
        nchunks,
        chunksize,
        η_converted;
        ndrange = ndrange,
    )
    synchronize(backend)

    cache = PackedStrictUpper(SUbuffer)
    uop = BlockUpperSolveOperator(cache, D_Dl1)
    return BackwardL1GSSweep(uop)
end

function _make_sweep_plan(::SymmetricSweep, ::PackedBufferCache, A, partitioning, isSymA, η)
    (; partsize, nparts, nchunks, chunksize, backend) = partitioning
    symA = isSymA ? SymmetricMatrix() : NonSymmetricMatrix()
    frmt = sparsemat_format_type(A)
    Tf = eltype(A)
    N = size(A, 1)

    # Create diagonal indices (if needed for CSC)
    diag = _create_diag_indices(A, frmt)

    # Allocate buffers
    A_adapted = adapt(backend, A)
    diag_adapted = adapt(backend, diag)
    D_Dl1 = adapt(backend, zeros(Tf, N))

    # Buffer size: partsize*(partsize-1)/2 elements per partition
    buffer_size = nparts * ((partsize * (partsize - 1)) ÷ 2)
    SLbuffer = adapt(backend, zeros(Tf, buffer_size))
    SUbuffer = adapt(backend, zeros(Tf, buffer_size))

    η_converted = convert(Tf, η)
    ndrange = nchunks * chunksize

    # Pack lower triangular
    kernel_lower = _precompute_and_pack_lower_kernel!(backend, chunksize, ndrange)
    kernel_lower(
        D_Dl1,
        SLbuffer,
        A_adapted,
        symA,
        frmt,
        diag_adapted,
        partsize,
        nparts,
        nchunks,
        chunksize,
        η_converted;
        ndrange = ndrange,
    )
    synchronize(backend)

    # Pack upper triangular (D_Dl1 already computed, but kernel recomputes - optimization opportunity)
    kernel_upper = _precompute_and_pack_upper_kernel!(backend, chunksize, ndrange)
    kernel_upper(
        D_Dl1,
        SUbuffer,
        A_adapted,
        symA,
        frmt,
        diag_adapted,
        partsize,
        nparts,
        nchunks,
        chunksize,
        η_converted;
        ndrange = ndrange,
    )
    synchronize(backend)

    L_cache = PackedStrictLower(SLbuffer)
    U_cache = PackedStrictUpper(SUbuffer)

    lop = BlockLowerSolveOperator(L_cache, D_Dl1)
    uop = BlockUpperSolveOperator(U_cache, D_Dl1)

    return SymmetricL1GSSweep(lop, uop)
end


# Helper function to compute D_Dl1 only (used by MatrixView storage)
function _compute_D_Dl1(A, partitioning, symA, frmt, diag, η)
    (; partsize, nparts, nchunks, chunksize, backend) = partitioning
    Tf = eltype(A)
    N = size(A, 1)

    A_adapted = adapt(backend, A)
    diag_adapted = adapt(backend, diag)
    D_Dl1 = adapt(backend, zeros(Tf, N))
    η_converted = convert(Tf, η)

    ndrange = nchunks * chunksize
    kernel = _compute_D_Dl1_kernel!(backend, chunksize, ndrange)
    kernel(
        D_Dl1,
        A_adapted,
        symA,
        frmt,
        diag_adapted,
        partsize,
        nparts,
        nchunks,
        chunksize,
        η_converted;
        ndrange = ndrange,
    )
    synchronize(backend)

    return D_Dl1
end


## Solver/Core layer ##
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

# Optimized version using diagonal indices for CSC non-symmetric
function _diag_offpart_csc_with_diag(
    colPtr,
    rowVal,
    nzVal,
    diag::DiagonalIndices,
    idx::Ti,
    part_start::Ti,
    part_end::Ti,
) where {Ti <: Integer}
    Tv = eltype(nzVal)
    b = zero(Tv)
    d = zero(Tv)

    ncols = length(colPtr) - 1

    # Process each column with optimized binary search
    for col = 1:ncols
        col_start = colPtr[col]
        col_end = colPtr[col+1] - 1

        # Skip empty columns
        if col_start > col_end
            continue
        end

        # For diagonal column, use O(1) lookup
        if col == idx
            diag_idx = diag.diag[col]
            b = nzVal[diag_idx]
            continue
        end

        # Binary search for row idx in this column
        left, right = col_start, col_end
        found_idx = -1

        while left <= right
            mid = (left + right) >> 1
            row = rowVal[mid]

            if row < idx
                left = mid + 1
            elseif row > idx
                right = mid - 1
            else
                found_idx = mid
                break
            end
        end

        # If we found row idx in this column (off-diagonal element)
        if found_idx != -1
            v = nzVal[found_idx]
            if col < part_start || col > part_end
                d += abs(v)
            end
        end
    end

    return b, d
end

# Optimized version for symmetric CSC: read row i from column i directly
function _diag_offpart_csc_symmetric(
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

    # For symmetric matrix: row i entries = column i entries (transposed)
    # So we can read row i by scanning column i
    col_start = colPtr[idx]
    col_end = colPtr[idx+1] - 1

    for p = col_start:col_end
        row = rowVal[p]
        v = nzVal[p]

        if row == idx
            # Diagonal element
            b = v
        else
            # Off-diagonal: check if outside partition
            # In symmetric matrix, A[idx, row] = A[row, idx] = v
            # So this represents both A[idx, row] and A[row, idx]
            if row < part_start || row > part_end
                d += abs(v)
            end
        end
    end

    return b, d
end

function _diag_offpart(
    ::NonSymmetricMatrix,
    ::CSCFormat,
    A,
    diag::DiagonalIndices,
    idx::Ti,
    part_start::Ti,
    part_end::Ti,
) where {Ti <: Integer}
    _diag_offpart_csc_with_diag(getcolptr(A), rowvals(A), getnzval(A), diag, idx, part_start, part_end)
end

function _diag_offpart(
    ::SymmetricMatrix,
    ::CSCFormat,
    A,
    diag::DiagonalIndices,
    idx::Ti,
    part_start::Ti,
    part_end::Ti,
) where {Ti <: Integer}
    # For symmetric CSC: A[i,j] = A[j,i], so row i = column i
    # This gives us O(nnz_row) access instead of O(n*nnz_col)
    _diag_offpart_csc_symmetric(getcolptr(A), rowvals(A), getnzval(A), idx, part_start, part_end)
end

function _diag_offpart(
    ::AbstractMatrixSymmetry,
    ::CSRFormat,
    A,
    ::NoDiagonalIndices,
    idx::Ti,
    part_start::Ti,
    part_end::Ti,
) where {Ti <: Integer}
    _diag_offpart_csr(getrowptr(A), colvals(A), getnzval(A), idx, part_start, part_end)
end



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

# Optimized CSC lower packing - exploits column-wise storage
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
    block_stride = (partsize * (partsize - 1)) ÷ 2
    block_offset = (k - 1) * block_stride

    # Scan columns in partition (except last column which has no lower elements)
    for col = start_idx:(end_idx-1)
        local_j = col - start_idx + 1
        col_start = colPtr[col]
        col_end = colPtr[col+1] - 1

        # Find first row > col in this column using linear scan
        # (binary search overhead not worth it for small partitions)
        p = col_start
        while p <= col_end && rowVal[p] <= col
            p += 1
        end

        # Pack all strictly lower elements (rows > col) in this column
        while p <= col_end
            i = rowVal[p]
            if i > end_idx
                break  # past partition boundary
            end
            local_i = i - start_idx + 1
            row_offset = ((local_i - 1) * (local_i - 2)) ÷ 2
            off_idx = block_offset + row_offset + local_j
            SLbuffer[off_idx] = nzVal[p]
            p += 1
        end
    end
    return nothing
end

function _pack_strict_lower!(
    ::NonSymmetricMatrix,
    ::CSCFormat,
    SLbuffer,
    A,
    start_idx::Ti,
    end_idx::Ti,
    partsize::Ti,
    k::Ti,
) where {Ti <: Integer}
    _pack_strict_lower_csc!(
        SLbuffer,
        getcolptr(A),
        rowvals(A),
        getnzval(A),
        start_idx,
        end_idx,
        partsize,
        k,
    )
end

function _pack_strict_lower!(
    ::SymmetricMatrix,
    ::CSCFormat,
    SLbuffer,
    A,
    start_idx::Ti,
    end_idx::Ti,
    partsize::Ti,
    k::Ti,
) where {Ti <: Integer}
    _pack_strict_lower_csr!(
        SLbuffer,
        getcolptr(A),
        rowvals(A),
        getnzval(A),
        start_idx,
        end_idx,
        partsize,
        k,
    )
end

function _pack_strict_lower!(
    ::AbstractMatrixSymmetry,
    ::CSRFormat,
    SLbuffer,
    A,
    start_idx::Ti,
    end_idx::Ti,
    partsize::Ti,
    k::Ti,
) where {Ti <: Integer}
    _pack_strict_lower_csr!(
        SLbuffer,
        getrowptr(A),
        colvals(A),
        getnzval(A),
        start_idx,
        end_idx,
        partsize,
        k,
    )
end

# Pack upper triangular functions (for backward sweep)
# For upper triangular, we store elements row by row where j > i
# Row i has (partsize - local_i) upper elements
function _pack_strict_upper_csr!(
    SUbuffer,
    rowPtr,
    colVal,
    nzVal,
    start_idx::Ti,
    end_idx::Ti,
    partsize::Ti,
    k::Ti,
) where {Ti <: Integer}
    block_stride = (partsize * (partsize - 1)) ÷ 2
    block_offset = (k - 1) * block_stride

    idx = 1
    for i = start_idx:end_idx
        for p = rowPtr[i]:(rowPtr[i+1]-1)
            j = colVal[p]
            if j > i && j <= end_idx  # strictly upper within partition
                SUbuffer[block_offset+idx] = nzVal[p]
                idx += 1
            end
        end
    end
    return nothing
end

# Optimized CSC upper packing - processes by column, places in row-wise order
function _pack_strict_upper_csc!(
    SUbuffer,
    colPtr,
    rowVal,
    nzVal,
    start_idx::Ti,
    end_idx::Ti,
    partsize::Ti,
    k::Ti,
) where {Ti <: Integer}
    block_stride = (partsize * (partsize - 1)) ÷ 2
    block_offset = (k - 1) * block_stride

    # Process each column j (columns with j > start_idx have upper elements)
    for col = (start_idx+1):end_idx
        local_j = col - start_idx + 1
        col_start = colPtr[col]
        col_end = colPtr[col+1] - 1

        # Scan elements in this column
        for p = col_start:col_end
            i = rowVal[p]

            # Check if element is strictly upper (i < col) and within partition
            if i < col
                if i < start_idx
                    continue  # skip elements before partition
                end

                # This is element A[i,col] where i < col (strictly upper)
                # We need to place it in row-wise order
                local_i = i - start_idx + 1

                # Calculate position in row-wise packed upper triangular storage
                # Row i starts after: sum of upper elements in rows 1..(i-1)
                # = sum_{r=1}^{i-1} (partsize - r)
                # = (i-1)*partsize - (i-1)*i/2
                # = (i-1)*(partsize - i/2)
                # But we use local indices, so:
                # row_start = sum_{r=1}^{local_i-1} (partsize - r)
                row_start = (local_i - 1) * partsize - ((local_i - 1) * local_i) ÷ 2

                # Within row i, element at column col is at position (local_j - local_i)
                col_offset = local_j - local_i

                off_idx = block_offset + row_start + col_offset
                SUbuffer[off_idx] = nzVal[p]
            elseif i >= col
                break  # rows are sorted, no more upper elements in this column
            end
        end
    end
    return nothing
end

function _pack_strict_upper!(
    ::NonSymmetricMatrix,
    ::CSCFormat,
    SUbuffer,
    A,
    start_idx::Ti,
    end_idx::Ti,
    partsize::Ti,
    k::Ti,
) where {Ti <: Integer}
    _pack_strict_upper_csc!(
        SUbuffer,
        getcolptr(A),
        rowvals(A),
        getnzval(A),
        start_idx,
        end_idx,
        partsize,
        k,
    )
end

function _pack_strict_upper!(
    ::SymmetricMatrix,
    ::CSCFormat,
    SUbuffer,
    A,
    start_idx::Ti,
    end_idx::Ti,
    partsize::Ti,
    k::Ti,
) where {Ti <: Integer}
    _pack_strict_upper_csr!(
        SUbuffer,
        getcolptr(A),
        rowvals(A),
        getnzval(A),
        start_idx,
        end_idx,
        partsize,
        k,
    )
end

function _pack_strict_upper!(
    ::AbstractMatrixSymmetry,
    ::CSRFormat,
    SUbuffer,
    A,
    start_idx::Ti,
    end_idx::Ti,
    partsize::Ti,
    k::Ti,
) where {Ti <: Integer}
    _pack_strict_upper_csr!(
        SUbuffer,
        getrowptr(A),
        colvals(A),
        getnzval(A),
        start_idx,
        end_idx,
        partsize,
        k,
    )
end


## Specialized kernels for efficient construction ##
# These kernels merge D_Dl1 computation with buffer packing to avoid wasteful allocations

# Kernel that only computes D_Dl1 (used by MatrixView storage)
@kernel function _compute_D_Dl1_kernel!(
    D_Dl1,
    A,
    symA,
    format_A,
    diag,
    partsize::Ti,
    nparts::Ti,
    nchunks::Ti,
    chunksize::Ti,
    η,
) where {Ti <: Integer}
    initial_partition_idx = @index(Global)
    size_A = convert(Ti, size(A, 1))

    for part in DiagonalPartsIterator(
        size_A,
        partsize,
        nparts,
        nchunks,
        chunksize,
        convert(Ti, initial_partition_idx),
    )
        (; k, partsize, start_idx, end_idx) = part
        for i = start_idx:end_idx
            a_ii, dl1_ii = _diag_offpart(symA, format_A, A, diag, i, start_idx, end_idx)
            TF = eltype(D_Dl1)
            dl1star_ii = a_ii >= η * dl1_ii ? zero(TF) : dl1_ii / convert(TF, 2.0)
            D_Dl1[i] = a_ii + dl1star_ii
        end
    end
end

# Kernel that computes D_Dl1 AND packs lower triangular (ForwardSweep + PackedBuffer)
@kernel function _precompute_and_pack_lower_kernel!(
    D_Dl1,
    SLbuffer,
    A,
    symA,
    format_A,
    diag,
    partsize::Ti,
    nparts::Ti,
    nchunks::Ti,
    chunksize::Ti,
    η,
) where {Ti <: Integer}
    initial_partition_idx = @index(Global)
    size_A = convert(Ti, size(A, 1))

    for part in DiagonalPartsIterator(
        size_A,
        partsize,
        nparts,
        nchunks,
        chunksize,
        convert(Ti, initial_partition_idx),
    )
        (; k, partsize, start_idx, end_idx) = part

        # Compute D_Dl1
        for i = start_idx:end_idx
            a_ii, dl1_ii = _diag_offpart(symA, format_A, A, diag, i, start_idx, end_idx)
            TF = eltype(D_Dl1)
            dl1star_ii = a_ii >= η * dl1_ii ? zero(TF) : dl1_ii / convert(TF, 2.0)
            D_Dl1[i] = a_ii + dl1star_ii
        end

        # Pack strictly lower triangular
        _pack_strict_lower!(symA, format_A, SLbuffer, A, start_idx, end_idx, partsize, k)
    end
end

# Kernel that computes D_Dl1 AND packs upper triangular (BackwardSweep + PackedBuffer)
@kernel function _precompute_and_pack_upper_kernel!(
    D_Dl1,
    SUbuffer,
    A,
    symA,
    format_A,
    diag,
    partsize::Ti,
    nparts::Ti,
    nchunks::Ti,
    chunksize::Ti,
    η,
) where {Ti <: Integer}
    initial_partition_idx = @index(Global)
    size_A = convert(Ti, size(A, 1))

    for part in DiagonalPartsIterator(
        size_A,
        partsize,
        nparts,
        nchunks,
        chunksize,
        convert(Ti, initial_partition_idx),
    )
        (; k, partsize, start_idx, end_idx) = part

        # Compute D_Dl1
        for i = start_idx:end_idx
            a_ii, dl1_ii = _diag_offpart(symA, format_A, A, diag, i, start_idx, end_idx)
            TF = eltype(D_Dl1)
            dl1star_ii = a_ii >= η * dl1_ii ? zero(TF) : dl1_ii / convert(TF, 2.0)
            D_Dl1[i] = a_ii + dl1star_ii
        end

        # Pack strictly upper triangular
        _pack_strict_upper!(symA, format_A, SUbuffer, A, start_idx, end_idx, partsize, k)
    end
end

function _apply_sweep!(y, P)
    @timeit_debug "_apply_sweep!" begin
        (; partitioning, sweep) = P
        _apply_sweep!(y, partitioning, sweep)
    end
    return nothing
end

# Forward sweep with BlockStrictLowerView (MatrixViewCache)
function _apply_sweep!(y, partitioning, sweep::ForwardL1GSSweep)
    @timeit_debug "_apply_sweep! (ForwardL1GSSweep)" begin
        (; partsize, nparts, nchunks, chunksize, backend) = partitioning
        D_Dl1 = sweep.op.D_DL1
        L = sweep.op.L # lower cache
        ndrange = nchunks * chunksize
        @timeit_debug "kernel construction" kernel =
            _apply_sweep_kernel!(backend, chunksize, ndrange)
        size_A = convert(typeof(nparts), length(y))
        step = convert(typeof(nparts), 1 )
        @timeit_debug "kernel call" kernel(
            y,
            L,
            D_Dl1,
            size_A,
            partsize,
            nparts,
            nchunks,
            chunksize,
            step;
            ndrange = ndrange,
        )
        @timeit_debug "synchronize" synchronize(backend)
    end
    return nothing
end

# Backward sweep with BlockStrictUpperView (MatrixViewCache)
function _apply_sweep!(y, partitioning, sweep::BackwardL1GSSweep)
    @timeit_debug "_apply_sweep! (BackwardL1GSSweep)" begin
        (; partsize, nparts, nchunks, chunksize, backend) = partitioning
        D_Dl1 = sweep.op.D_DL1
        U = sweep.op.U # upper cache
        ndrange = nchunks * chunksize
        @timeit_debug "kernel construction" kernel =
            _apply_sweep_kernel!(backend, chunksize, ndrange)
        size_A = convert(typeof(nparts), length(y))
        step = convert(typeof(nparts), -1 )
        @timeit_debug "kernel call" kernel(
            y,
            U,
            D_Dl1,
            size_A,
            partsize,
            nparts,
            nchunks,
            chunksize,
            step;
            ndrange = ndrange,
        )
        @timeit_debug "synchronize" synchronize(backend)
    end
    return nothing
end

# Symmetric sweep with MatrixView
function _apply_sweep!(y, partitioning, sweep::SymmetricL1GSSweep)
    @timeit_debug "_apply_sweep! (SymmetricL1GSSweep)" begin
        (; partsize, nparts, nchunks, chunksize, backend) = partitioning
        D_Dl1_fwd = sweep.lop.D_DL1
        D_Dl1_bwd = sweep.uop.D_DL1
        L = sweep.lop.L
        U = sweep.uop.U
        ndrange = nchunks * chunksize
        size_A = convert(typeof(nparts), length(y))
        step_fwd = convert(typeof(nparts), 1 )
        step_bwd = convert(typeof(nparts), -1 )

        # Forward sweep
        @timeit_debug "forward kernel" begin
            kernel_fwd = _apply_sweep_kernel!(backend, chunksize, ndrange)
            kernel_fwd(
                y,
                L,
                D_Dl1_fwd,
                size_A,
                partsize,
                nparts,
                nchunks,
                chunksize,
                step_fwd;
                ndrange = ndrange,
            )
            synchronize(backend)
        end

        # Backward sweep
        @timeit_debug "backward kernel" begin
            kernel_bwd = _apply_sweep_kernel!(backend, chunksize, ndrange)
            kernel_bwd(
                y,
                U,
                D_Dl1_bwd,
                U.symA,
                size_A,
                partsize,
                nparts,
                nchunks,
                chunksize,
                step_bwd;
                ndrange = ndrange,
            )
            synchronize(backend)
        end
    end
    return nothing
end

@kernel function _apply_sweep_kernel!(y, cache::AbstractCache, D_Dl1, size_A::Ti, partsize::Ti, nparts::Ti, nchunks::Ti, chunksize::Ti,step)  where {Ti <: Integer}
    initial_partition_idx = @index(Global)

    for part in DiagonalPartsIterator(
        size_A,
        partsize,
        nparts,
        nchunks,
        chunksize,
        convert(Ti, initial_partition_idx),
    )
        @unpack k, partsize, start_idx, end_idx = part
        @inbounds for i = start_idx:step:end_idx
            acc = _accumulate_from_cache(cache, y, i, start_idx, end_idx)
            y[i] = (y[i] - acc) / D_Dl1[i]
        end
    end
end

# # Forward sweep kernel with PackedBuffer
# @kernel function _forward_sweep_packed_kernel!(
#     y,
#     D_Dl1,
#     buffer,
#     size_A::Ti,
#     partsize::Ti,
#     nparts::Ti,
#     nchunks::Ti,
#     chunksize::Ti,
# ) where {Ti <: Integer}
#     initial_partition_idx = @index(Global)
#     for part in DiagonalPartsIterator(
#         size_A,
#         partsize,
#         nparts,
#         nchunks,
#         chunksize,
#         convert(Ti, initial_partition_idx),
#     )
#         @unpack k, partsize, start_idx, end_idx = part
#         block_stride = (partsize * (partsize - 1)) ÷ 2
#         block_offset = (k - 1) * block_stride

#         # forward‐solve: (1/a_{ii})[bᵢ - ∑_{j<i} a_ij * x_j]
#         for i = start_idx:end_idx
#             local_i = i - start_idx + 1
#             row_offset = ((local_i-1) * (local_i - 2)) ÷ 2

#             acc = zero(eltype(y))
#             @inbounds for local_j = 1:(local_i-1)
#                 gj = start_idx + (local_j - 1)
#                 off_idx = block_offset + row_offset + (local_j)
#                 acc += buffer[off_idx] * y[gj]
#             end

#             y[i] = (y[i] - acc) / D_Dl1[i]
#         end
#     end
# end

# Backward sweep kernel with PackedBuffer
# Upper triangular elements are stored row by row: row i contains all j > i
# @kernel function _backward_sweep_packed_kernel!(
#     y,
#     D_Dl1,
#     buffer,
#     size_A::Ti,
#     partsize::Ti,
#     nparts::Ti,
#     nchunks::Ti,
#     chunksize::Ti,
# ) where {Ti <: Integer}
#     initial_partition_idx = @index(Global)
#     for part in DiagonalPartsIterator(
#         size_A,
#         partsize,
#         nparts,
#         nchunks,
#         chunksize,
#         convert(Ti, initial_partition_idx),
#     )
#         @unpack k, partsize, start_idx, end_idx = part
#         block_stride = (partsize * (partsize - 1)) ÷ 2
#         block_offset = (k - 1) * block_stride

#         # backward‐solve: iterate rows in reverse order
#         for i = end_idx:-1:start_idx
#             num_upper = end_idx - i  # number of upper elements in row i

#             acc = zero(eltype(y))
#             # We need to find where row i's upper elements are stored
#             # Count elements in rows before i
#             elements_before = 0
#             for ii = start_idx:(i-1)
#                 elements_before += (end_idx - ii)
#             end

#             @inbounds for offset = 1:num_upper
#                 j = i + offset
#                 buf_idx = block_offset + elements_before + offset
#                 acc += buffer[buf_idx] * y[j]
#             end

#             y[i] = (y[i] - acc) / D_Dl1[i]
#         end
#     end
# end

# # Forward sweep kernel with MatrixView (optimized for both CSR and CSC)
# @kernel function _forward_sweep_matrixview_kernel!(
#     y,
#     D_Dl1,
#     A,
#     diag,
#     symA,
#     size_A::Ti,
#     partsize::Ti,
#     nparts::Ti,
#     nchunks::Ti,
#     chunksize::Ti,
# ) where {Ti <: Integer}
#     initial_partition_idx = @index(Global)
#     format_A = sparsemat_format_type(A)

#     for part in DiagonalPartsIterator(
#         size_A,
#         partsize,
#         nparts,
#         nchunks,
#         chunksize,
#         convert(Ti, initial_partition_idx),
#     )
#         @unpack k, partsize, start_idx, end_idx = part

#         # Forward solve: x[i] = (b[i] - L[i,:]*x) / D[i]
#         # Uses optimized accumulation based on matrix format, diagonal indices, and symmetry
#         @inbounds for i = start_idx:end_idx
#             acc = _accumulate_lower_from_matrix(A, format_A, y, i, start_idx, end_idx, diag, symA)
#             y[i] = (y[i] - acc) / D_Dl1[i]
#         end
#     end
# end

# # Backward sweep kernel with MatrixView (optimized for both CSR and CSC)
# @kernel function _backward_sweep_matrixview_kernel!(
#     y,
#     D_Dl1,
#     A,
#     diag,
#     symA,
#     size_A::Ti,
#     partsize::Ti,
#     nparts::Ti,
#     nchunks::Ti,
#     chunksize::Ti,
# ) where {Ti <: Integer}
#     initial_partition_idx = @index(Global)
#     format_A = sparsemat_format_type(A)

#     for part in DiagonalPartsIterator(
#         size_A,
#         partsize,
#         nparts,
#         nchunks,
#         chunksize,
#         convert(Ti, initial_partition_idx),
#     )
#         @unpack k, partsize, start_idx, end_idx = part

#         # Backward solve: x[i] = (b[i] - U[i,:]*x) / D[i]
#         # Uses optimized accumulation based on matrix format, diagonal indices, and symmetry
#         @inbounds for i = end_idx:-1:start_idx
#             acc = _accumulate_upper_from_matrix(A, format_A, y, i, start_idx, end_idx, diag, symA)
#             y[i] = (y[i] - acc) / D_Dl1[i]
#         end
#     end
# end

# Helper functions to accumulate from original matrix
# Optimized with diagonal indices for CSC and symmetry exploitation

_accumulate_from_cache(cache::BlockStrictLowerView, y, i, start_idx, end_idx) = _accumulate_lower_from_matrix(
    cache.A,
    sparsemat_format_type(cache.A),
    y,
    i,
    start_idx,
    end_idx,
    cache.diag,
    cache.symA,
)

_accumulate_from_cache(cache::BlockStrictUpperView, y, i, start_idx, end_idx) = _accumulate_upper_from_matrix(
    cache.A,
    sparsemat_format_type(cache.A),
    y,
    i,
    start_idx,
    end_idx,
    cache.diag,
    cache.symA,
)

# CSR format: Optimal O(nnz_row) row-wise access, no diagonal indices needed
# here no need for diag indices
function _accumulate_lower_from_matrix(A, ::CSRFormat, y, i, start_idx, end_idx, ::AbstractDiagonalIndices, ::AbstractMatrixSymmetry)
    rowPtr = getrowptr(A)
    colVal = colvals(A)
    nzVal = getnzval(A)

    acc = zero(eltype(y))
    @inbounds for p = rowPtr[i]:(rowPtr[i+1]-1)
        j = colVal[p]
        if j >= start_idx && j < i  # strictly lower within partition
            acc += nzVal[p] * y[j]
        end
    end
    return acc
end

# CSC format (non-symmetric): Binary search with diagonal indices - O(nnz_row * log(nnz_col/2))
function _accumulate_lower_from_matrix(A, ::CSCFormat, y, i, start_idx, end_idx, diag::DiagonalIndices, ::NonSymmetricMatrix)
    colPtr = getcolptr(A)
    rowVal = rowvals(A)
    nzVal = getnzval(A)

    acc = zero(eltype(y))
    # Scan columns before row i (within partition)
    @inbounds for col = start_idx:min(i-1, end_idx)
        diag_idx = diag[col]
        p_lo = diag_idx + 1  # Search ONLY below diagonal
        p_hi = colPtr[col+1] - 1

        # Binary search for row i in reduced range [diag_idx+1, p_hi]
        if p_lo <= p_hi && rowVal[p_lo] <= i && rowVal[p_hi] >= i
            left, right = p_lo, p_hi
            while left <= right
                mid = (left + right) >> 1
                if rowVal[mid] < i
                    left = mid + 1
                elseif rowVal[mid] > i
                    right = mid - 1
                else
                    # Found exact match
                    acc += nzVal[mid] * y[col]
                    break
                end
            end
        end
    end
    return acc
end

function _accumulate_lower_from_matrix(A, ::CSCFormat, y, i, start_idx, end_idx, ::DiagonalIndices, ::SymmetricMatrix)
    colPtr = getcolptr(A); rowVal = rowvals(A); nzVal = getnzval(A)
    acc = zero(eltype(y))
    @inbounds for p = colPtr[i]:(colPtr[i+1]-1)
        j = rowVal[p]                 # this is the "other index"
        if start_idx <= j < i         # j in partition, strictly lower
            acc += nzVal[p] * y[j]    # A[j,i] == A[i,j] by symmetry
        end
    end
    return acc
end

## Upper triangular accumulation ##

# CSR format: Optimal O(nnz_row) row-wise access, no diagonal indices needed
function _accumulate_upper_from_matrix(A, ::CSRFormat, y, i, start_idx, end_idx, ::NoDiagonalIndices, ::AbstractMatrixSymmetry)
    rowPtr = getrowptr(A)
    colVal = colvals(A)
    nzVal = getnzval(A)

    acc = zero(eltype(y))
    @inbounds for p = rowPtr[i]:(rowPtr[i+1]-1)
        j = colVal[p]
        if j > i && j <= end_idx  # strictly upper within partition
            acc += nzVal[p] * y[j]
        end
    end
    return acc
end

# CSC format (non-symmetric): Binary search with diagonal indices - O(nnz_row * log(nnz_col/2))
function _accumulate_upper_from_matrix(A, ::CSCFormat, y, i, start_idx, end_idx, diag::DiagonalIndices, ::NonSymmetricMatrix)
    colPtr = getcolptr(A)
    rowVal = rowvals(A)
    nzVal = getnzval(A)

    acc = zero(eltype(y))
    # Scan columns after row i (within partition)
    @inbounds for col = (i+1):end_idx
        diag_idx = diag[col]
        p_lo = colPtr[col]
        p_hi = diag_idx - 1  # search only above diagonal

        # Binary search for row i in reduced range [p_lo, diag_idx-1]
        if p_lo <= p_hi && rowVal[p_lo] <= i && rowVal[p_hi] >= i
            left, right = p_lo, p_hi
            while left <= right
                mid = (left + right) >> 1
                if rowVal[mid] < i
                    left = mid + 1
                elseif rowVal[mid] > i
                    right = mid - 1
                else
                    acc += nzVal[mid] * y[col]
                    break
                end
            end
        end
    end
    return acc
end

# CSC format (symmetric): Direct O(nnz_row) access via column transposition!
# For symmetric CSC: U[i,j] = L[j,i] = A[i,j] (elements below diagonal in column i)
function _accumulate_upper_from_matrix(A, ::CSCFormat, y, i, start_idx, end_idx, diag::DiagonalIndices, ::SymmetricMatrix)
    colPtr = getcolptr(A)
    rowVal = rowvals(A)
    nzVal = getnzval(A)

    acc = zero(eltype(y))
    diag_idx = diag[i]  # Diagonal position in column i
    p_hi = colPtr[i+1] - 1

    # Elements BELOW diagonal in column i = strictly upper elements of row i (by symmetry)
    @inbounds for p = (diag_idx+1):p_hi
        j = rowVal[p]
        if j > i && j <= end_idx  # within partition and strictly upper
            acc += nzVal[p] * y[j]
        end
    end

    return acc
end


function _accumulate_upper_from_matrix(A, ::CSCFormat, y, i, start_idx, end_idx, ::DiagonalIndices, ::SymmetricMatrix)
    colPtr = getcolptr(A); rowVal = rowvals(A); nzVal = getnzval(A)
    acc = zero(eltype(y))
    @inbounds for p = colPtr[i]:(colPtr[i+1]-1)
        j = rowVal[p]
        if i < j <= end_idx           # j in partition, strictly upper
            acc += nzVal[p] * y[j]
        end
    end
    return acc
end
