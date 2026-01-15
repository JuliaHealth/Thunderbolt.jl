######################################
######################################
### L1 Gauss Seidel Preconditioner ###
######################################
######################################

################
## User layer ##
################

abstract type AbstractSweep end
struct ForwardSweep <: AbstractSweep end # forward sweep -> lower triangular
struct BackwardSweep <: AbstractSweep end # backward sweep -> upper triangular
struct SymmetricSweep <: AbstractSweep end # symmetric sweep -> both lower and upper triangular

# FIXME: Is there better naming alternatives for this?
abstract type AbstractCacheStrategy end
struct PackedBufferCache <: AbstractCacheStrategy end # store lower/upper triangular parts in a vector buffer -> efficient for small partitions (e.g. GPU)
struct MatrixViewCache <: AbstractCacheStrategy end # store a sparse triangular matrix view -> efficient for large partitions (e.g. CPU)

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

abstract type AbstractL1GSSweepPlan end

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
struct L1GSPreconditioner{Partitioning, SweepPlanType <: AbstractL1GSSweepPlan}
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
    sweep::AbstractSweep = ForwardSweep(),
    cache_strategy::AbstractCacheStrategy = _choose_default_device_storage(builder.device),
) where {Ti <: Integer}
    build_l1prec(builder, A, partsize, isSymA, η, sweep, cache_strategy)
end

function (builder::L1GSPrecBuilder)(
    A::Symmetric,
    partsize::Ti;
    η = 1.5,
    sweep::AbstractSweep = SymmetricSweep(),
    cache_strategy::AbstractCacheStrategy = OriginalMatrix(),
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
    sweep::AbstractSweep,
    cache_strategy::AbstractCacheStrategy,
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
    sweep::AbstractSweep,
    cache_strategy::AbstractCacheStrategy,
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

abstract type AbstractDiagonalIndices end

## DiagonalIndices - for efficient CSC matrix access
## this code is adapted from iterativesolvers.jl
struct DiagonalIndices{Ti <: Integer} <: AbstractDiagonalIndices
    diag::Vector{Ti}

    function DiagonalIndices{Ti}(A::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
        diag = Vector{Ti}(undef, A.n)

        for col = 1:A.n
            r1 = Int(A.colptr[col])
            r2 = Int(A.colptr[col+1] - 1)
            r1 = searchsortedfirst(A.rowval, col, r1, r2, Base.Order.Forward)
            if r1 > r2 || A.rowval[r1] != col || iszero(A.nzval[r1])
                throw(LinearAlgebra.SingularException(col))
            end
            diag[col] = r1
        end

        new(diag)
    end
end

DiagonalIndices(A::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti} = DiagonalIndices{Ti}(A)

@inline Base.getindex(d::DiagonalIndices, i::Int) = d.diag[i]
@inline Base.length(d::DiagonalIndices) = length(d.diag)

# for CSR and symmetric CSC, we don't need to store diagonal indices
struct NoDiagonalIndices <: AbstractDiagonalIndices end


abstract type AbstractCache end
abstract type AbstractLowerCache <: AbstractCache end
abstract type AbstractUpperCache <: AbstractCache end


struct BlockStrictLowerView{
    MatrixType,
    SymmetryType <: AbstractMatrixSymmetry,
    FormatType <: AbstractMatrixFormat,
    DIType <: AbstractDiagonalIndices,
} <: AbstractLowerCache
    A::MatrixType
    symA::SymmetryType
    frmt::FormatType
    diag::DIType # imp for csc format (DiagonalIndices for CSC, NoDiagonalIndices for CSR)
end

struct BlockStrictUpperView{
    MatrixType,
    SymmetryType <: AbstractMatrixSymmetry,
    FormatType <: AbstractMatrixFormat,
    DIType <: AbstractDiagonalIndices,
} <: AbstractUpperCache
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

struct BlockLowerSolveOperator{LowerCache <: AbstractLowerCache, VectorType} <:
       AbstractBlockSolveOperator
    L::LowerCache
    D_DL1::VectorType
end

struct BlockUpperSolveOperator{UpperCache <: AbstractUpperCache, VectorType} <:
       AbstractBlockSolveOperator
    U::UpperCache
    D_DL1::VectorType
end

_cache(op::BlockLowerSolveOperator) = op.L
_cache(op::BlockUpperSolveOperator) = op.U



struct ForwardL1GSSweep{LowerOp <: BlockLowerSolveOperator} <: AbstractL1GSSweepPlan
    op::LowerOp
end

struct BackwardL1GSSweep{UpperOp <: BlockUpperSolveOperator} <: AbstractL1GSSweepPlan
    op::UpperOp
end

struct SymmetricL1GSSweep{LowerOp <: BlockLowerSolveOperator, UpperOp <: BlockUpperSolveOperator} <:
       AbstractL1GSSweepPlan
    lop::LowerOp
    uop::UpperOp
end

get_data(A::AbstractSparseMatrix) = A
get_data(A::Symmetric{Ti, TA}) where {Ti, TA} = TA(A.data) # restore the full matrix, why ? https://discourse.julialang.org/t/is-there-a-symmetric-sparse-matrix-implementation-in-julia/91333/2


_choose_default_device_storage(::AbstractDevice) = MatrixViewCache()
_choose_default_device_storage(::AbstractGPUDevice) = PackedBufferCache()


# Helper to create diagonal indices based on format and symmetry
# Symmetric CSC doesn't need diagonal indices - we can read row i from column i directly
_create_diag_indices(A, ::CSCFormat, ::NonSymmetricMatrix) = DiagonalIndices(A)
_create_diag_indices(A, ::CSCFormat, ::SymmetricMatrix) = NoDiagonalIndices()
_create_diag_indices(A, ::CSRFormat, ::AbstractMatrixSymmetry) = NoDiagonalIndices()

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

# first step of dispatching in precompution step is `_make_sweep_plan`,
# which basically can take any permutation of `AbstractCacheStrategy` & `AbstractSweep`
# TODO: adapt for matrix stuff
function _make_sweep_plan(::ForwardSweep, ::MatrixViewCache, A, partitioning, isSymA, η)
    symA = isSymA ? SymmetricMatrix() : NonSymmetricMatrix()
    frmt = sparsemat_format_type(A)

    # Create diagonal indices (only needed for non-symmetric CSC)
    diag = _create_diag_indices(A, frmt, symA)

    A_adapted = adapt(backend, A)
    diag_adapted = adapt(backend, diag)
    D_Dl1 = _compute_D_Dl1(A_adapted, partitioning, symA, frmt, diag_adapted, η)

    # Create cache with diagonal indices
    cache = BlockStrictLowerView(A_adapted, symA, frmt, diag_adapted)
    lop = BlockLowerSolveOperator(cache, D_Dl1)
    return ForwardL1GSSweep(lop)
end

function _make_sweep_plan(::BackwardSweep, ::MatrixViewCache, A, partitioning, isSymA, η)
    symA = isSymA ? SymmetricMatrix() : NonSymmetricMatrix()
    frmt = sparsemat_format_type(A)

    # Create diagonal indices (only needed for non-symmetric CSC)
    diag = _create_diag_indices(A, frmt, symA)

    A_adapted = adapt(backend, A)
    diag_adapted = adapt(backend, diag)
    D_Dl1 = _compute_D_Dl1(A_adapted, partitioning, symA, frmt, diag_adapted, η)

    # Create cache with diagonal indices
    cache = BlockStrictUpperView(A_adapted, symA, frmt, diag_adapted)
    uop = BlockUpperSolveOperator(cache, D_Dl1)
    return BackwardL1GSSweep(uop)
end

function _make_sweep_plan(::SymmetricSweep, ::MatrixViewCache, A, partitioning, isSymA, η)
    symA = isSymA ? SymmetricMatrix() : NonSymmetricMatrix()
    frmt = sparsemat_format_type(A)

    # Create diagonal indices (only needed for non-symmetric CSC)
    diag = _create_diag_indices(A, frmt, symA)

    A_adapted = adapt(backend, A)
    diag_adapted = adapt(backend, diag)
    D_Dl1 = _compute_D_Dl1(A_adapted, partitioning, symA, frmt, diag_adapted, η)

    L_cache = BlockStrictLowerView(A_adapted, symA, frmt, diag_adapted)
    U_cache = BlockStrictUpperView(A_adapted, symA, frmt, diag_adapted)

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

    # Create diagonal indices (only needed for non-symmetric CSC)
    diag = _create_diag_indices(A, frmt, symA)

    # Allocate buffers
    A_adapted = adapt(backend, A)
    diag_adapted = adapt(backend, diag)
    D_Dl1 = adapt(backend, zeros(Tf, N))

    # Buffer size: partsize*(partsize-1)/2 elements per partition
    buffer_size = nparts * ((partsize * (partsize - 1)) ÷ 2)
    SLbuffer = adapt(backend, zeros(Tf, buffer_size))
    cache = PackedStrictLower(SLbuffer)

    η_converted = convert(Tf, η)

    # Single kernel call to compute D_Dl1 and pack triangular
    ndrange = nchunks * chunksize
    kernel = _precompute_and_pack_kernel!(backend, chunksize, ndrange)
    kernel(
        D_Dl1,
        cache,
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

    lop = BlockLowerSolveOperator(cache, D_Dl1)
    return ForwardL1GSSweep(lop)
end

function _make_sweep_plan(::BackwardSweep, ::PackedBufferCache, A, partitioning, isSymA, η)
    (; partsize, nparts, nchunks, chunksize, backend) = partitioning
    symA = isSymA ? SymmetricMatrix() : NonSymmetricMatrix()
    frmt = sparsemat_format_type(A)
    Tf = eltype(A)
    N = size(A, 1)

    # Create diagonal indices (only needed for non-symmetric CSC)
    diag = _create_diag_indices(A, frmt, symA)

    # Allocate buffers
    A_adapted = adapt(backend, A)
    diag_adapted = adapt(backend, diag)
    D_Dl1 = adapt(backend, zeros(Tf, N))

    # Buffer size: partsize*(partsize-1)/2 elements per partition
    buffer_size = nparts * ((partsize * (partsize - 1)) ÷ 2)
    SUbuffer = adapt(backend, zeros(Tf, buffer_size))
    cache = PackedStrictUpper(SUbuffer)

    η_converted = convert(Tf, η)

    # Single kernel call to compute D_Dl1 and pack triangular
    ndrange = nchunks * chunksize
    kernel = _precompute_and_pack_kernel!(backend, chunksize, ndrange)
    kernel(
        D_Dl1,
        cache,
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

    uop = BlockUpperSolveOperator(cache, D_Dl1)
    return BackwardL1GSSweep(uop)
end

function _make_sweep_plan(::SymmetricSweep, ::PackedBufferCache, A, partitioning, isSymA, η)
    (; partsize, nparts, nchunks, chunksize, backend) = partitioning
    symA = isSymA ? SymmetricMatrix() : NonSymmetricMatrix()
    frmt = sparsemat_format_type(A)
    Tf = eltype(A)
    N = size(A, 1)

    # Create diagonal indices (only needed for non-symmetric CSC)
    diag = _create_diag_indices(A, frmt, symA)

    # Allocate buffers
    A_adapted = adapt(backend, A)
    diag_adapted = adapt(backend, diag)
    D_Dl1 = adapt(backend, zeros(Tf, N))

    # Buffer size: partsize*(partsize-1)/2 elements per partition
    buffer_size = nparts * ((partsize * (partsize - 1)) ÷ 2)
    SLbuffer = adapt(backend, zeros(Tf, buffer_size))
    SUbuffer = adapt(backend, zeros(Tf, buffer_size))
    L_cache = PackedStrictLower(SLbuffer)
    U_cache = PackedStrictUpper(SUbuffer)

    η_converted = convert(Tf, η)
    ndrange = nchunks * chunksize

    # Pack lower triangular (also computes D_Dl1)
    kernel = _precompute_and_pack_kernel!(backend, chunksize, ndrange)
    kernel(
        D_Dl1,
        L_cache,
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
    kernel(
        D_Dl1,
        U_cache,
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

    lop = BlockLowerSolveOperator(L_cache, D_Dl1)
    uop = BlockUpperSolveOperator(U_cache, D_Dl1)

    return SymmetricL1GSSweep(lop, uop)
end

# second step of dispatching in application step is `_apply_sweep!`
function _apply_sweep!(y, P::L1GSPreconditioner)
    @timeit_debug "_apply_sweep!" begin
        (; partitioning, sweep) = P
        _apply_sweep!(y, partitioning, sweep)
    end
    return nothing
end

# this will dispatch backward or forward based on sweep plan type
function _apply_sweep!(y, partitioning, sweep_plan::AbstractL1GSSweepPlan)
    @timeit_debug "_apply_sweep!" begin
        (; partsize, nparts, nchunks, chunksize, backend) = partitioning
        D_Dl1 = sweep_plan.op.D_DL1
        cache = _cache(sweep_plan.op)
        ndrange = nchunks * chunksize
        @timeit_debug "kernel construction" kernel =
            _apply_sweep_kernel!(backend, chunksize, ndrange)
        size_A = convert(typeof(nparts), length(y))
        step = convert(typeof(nparts), 1)
        @timeit_debug "kernel call" kernel(
            y,
            cache,
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
    @timeit_debug "_apply_sweep!" begin
        (; partsize, nparts, nchunks, chunksize, backend) = partitioning
        D_Dl1 = sweep.lop.D_DL1
        L = sweep.lop.L
        U = sweep.uop.U
        ndrange = nchunks * chunksize
        size_A = convert(typeof(nparts), length(y))
        step_fwd = convert(typeof(nparts), 1)
        step_bwd = convert(typeof(nparts), -1)

        kernel = _apply_sweep_kernel!(backend, chunksize, ndrange)

        # Forward sweep
        @timeit_debug "forward kernel" begin
            kernel(
                y,
                L,
                D_Dl1,
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
            kernel(
                y,
                U,
                D_Dl1,
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


####################
## Internal layer ##
###################

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


# Helper function to compute D_Dl1 only (used by MatrixView storage)
function _compute_D_Dl1(A, partitioning, symA, frmt, diag, η)
    (; partsize, nparts, nchunks, chunksize, backend) = partitioning
    Tf = eltype(A)
    N = size(A, 1)


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

## precomputation kernels ##
# Kernel that computes D_Dl1 only
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

# Unified kernel that computes D_Dl1 AND packs triangular elements
# Dispatches to lower or upper packing based on cache type (PackedStrictLower or PackedStrictUpper)
@kernel function _precompute_and_pack_kernel!(
    D_Dl1,
    cache,  # PackedStrictLower or PackedStrictUpper
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

        # Pack triangular elements - dispatches based on cache type
        _pack_strict_triangular!(cache, symA, format_A, A, start_idx, end_idx, partsize, k)
    end
end

@kernel function _apply_sweep_kernel!(
    y,
    cache::AbstractCache,
    D_Dl1,
    size_A::Ti,
    partsize::Ti,
    nparts::Ti,
    nchunks::Ti,
    chunksize::Ti,
    step,
) where {Ti <: Integer}
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
            acc = _accumulate_from_cache(cache, y, i, k, partsize, start_idx, end_idx)
            y[i] = (y[i] - acc) / D_Dl1[i]
        end
    end
end

# Row-wise diagonal and off-partition computation
# Works for both:
#   - CSR format: pass (rowPtr, colVal, nzVal)
#   - Symmetric CSC: pass (colPtr, rowVal, nzVal) since row i = column i
function _diag_offpart_rowwise(
    indPtr,   # rowPtr for CSR, colPtr for symmetric CSC
    indices,  # colVal for CSR, rowVal for symmetric CSC
    nzVal,
    idx::Ti,
    part_start::Ti,
    part_end::Ti,
) where {Ti <: Integer}
    Tv = eltype(nzVal)
    b = zero(Tv)
    d = zero(Tv)

    # Scan row i (or column i for symmetric CSC)
    p_start = indPtr[idx]
    p_end = indPtr[idx+1] - 1

    for p = p_start:p_end
        j = indices[p]
        v = nzVal[p]

        if j == idx
            b = v  # diagonal element
        elseif j < part_start || j > part_end
            d += abs(v)  # off-partition element
        end
    end

    return b, d
end

#TODO: not sure if this is optimal or not ?
# Optimized using diagonal indices to reduce search range
# For col < idx: A[idx,col] is below diagonal, search in [diag[col]+1, colPtr[col+1]-1]
# For col > idx: A[idx,col] is above diagonal, search in [colPtr[col], diag[col]-1]
function _diag_offpart(
    ::NonSymmetricMatrix,
    ::CSCFormat,
    A,
    diag::DiagonalIndices,
    idx::Ti,
    part_start::Ti,
    part_end::Ti,
) where {Ti <: Integer}
    colPtr = getcolptr(A)
    rowVal = rowvals(A)
    nzVal = getnzval(A)
    Tv = eltype(nzVal)
    b = zero(Tv)
    d = zero(Tv)

    ncols = length(colPtr) - 1

    @inbounds b = nzVal[diag[idx]] # diagonal element A[idx,idx]

    # Process columns before idx (element A[idx,col] is below diagonal)
    @inbounds for col = 1:(idx-1)
        # Only accumulate off-partition elements
        col < part_start || col > part_end || continue

        diag_idx = diag[col]
        p_lo = diag_idx + 1  # Below diagonal (i.e. has higher row index :D)
        p_hi = colPtr[col+1] - 1

        # Early exit if no elements below diagonal
        p_lo > p_hi && continue

        # Binary search for row idx in reduced range
        if rowVal[p_lo] <= idx && rowVal[p_hi] >= idx
            left, right = p_lo, p_hi
            while left <= right
                mid = (left + right) >> 1
                row = rowVal[mid]
                if row < idx
                    left = mid + 1
                elseif row > idx
                    right = mid - 1
                else
                    d += abs(nzVal[mid])
                    break
                end
            end
        end
    end

    # Process columns after idx (element A[idx,col] is above diagonal)
    @inbounds for col = (idx+1):ncols
        # Only accumulate off-partition elements
        col < part_start || col > part_end || continue

        diag_idx = diag[col]
        p_lo = colPtr[col]
        p_hi = diag_idx - 1  # Above diagonal

        # Early exit if no elements above diagonal
        p_lo > p_hi && continue

        # Binary search for row idx in reduced range
        if rowVal[p_lo] <= idx && rowVal[p_hi] >= idx
            left, right = p_lo, p_hi
            while left <= right
                mid = (left + right) >> 1
                row = rowVal[mid]
                if row < idx
                    left = mid + 1
                elseif row > idx
                    right = mid - 1
                else
                    d += abs(nzVal[mid])
                    break
                end
            end
        end
    end

    return b, d
end

function _diag_offpart(
    ::SymmetricMatrix,
    ::CSCFormat,
    A,
    ::NoDiagonalIndices,
    idx::Ti,
    part_start::Ti,
    part_end::Ti,
) where {Ti <: Integer}
    # Symmetric CSC: row i = column i, so use row-wise access
    _diag_offpart_rowwise(getcolptr(A), rowvals(A), getnzval(A), idx, part_start, part_end)
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
    # CSR: direct row-wise access
    _diag_offpart_rowwise(getrowptr(A), colvals(A), getnzval(A), idx, part_start, part_end)
end

# Row-wise packing for strictly lower triangular elements
# Works for both:
#   - CSR format: pass (rowPtr, colVal, nzVal)
#   - Symmetric CSC: pass (colPtr, rowVal, nzVal) since row i = column i
function _pack_strict_lower_rowwise!(
    SLbuffer,
    indPtr,   # rowPtr for CSR, colPtr for symmetric CSC
    indices,  # colVal for CSR, rowVal for symmetric CSC
    nzVal,
    start_idx::Ti,
    end_idx::Ti,
    partsize::Ti,
    k::Ti,
) where {Ti <: Integer}
    block_stride = (partsize * (partsize - 1)) ÷ 2
    block_offset = (k - 1) * block_stride

    for i = start_idx:end_idx
        local_i = i - start_idx + 1
        row_offset = ((local_i - 1) * (local_i - 2)) ÷ 2

        # Scan row i (or column i for symmetric CSC)
        for p = indPtr[i]:(indPtr[i+1]-1)
            j = indices[p]
            if j >= start_idx && j < i  # strictly lower within partition
                local_j = j - start_idx + 1
                off_idx = block_offset + row_offset + local_j
                SLbuffer[off_idx] = nzVal[p]
            end
        end
    end
    return nothing
end

function _pack_strict_triangular!(
    cache::PackedStrictLower,
    ::NonSymmetricMatrix,
    ::CSCFormat,
    A,
    start_idx::Ti,
    end_idx::Ti,
    partsize::Ti,
    k::Ti,
) where {Ti <: Integer}
    (; SLbuffer) = cache
    colPtr = getcolptr(A)
    rowVal = rowvals(A)
    nzVal = getnzval(A)
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

function _pack_strict_triangular!(
    cache::PackedStrictLower,
    ::SymmetricMatrix,
    ::CSCFormat,
    A,
    start_idx::Ti,
    end_idx::Ti,
    partsize::Ti,
    k::Ti,
) where {Ti <: Integer}
    # Symmetric CSC: row i = column i, so use row-wise access
    _pack_strict_lower_rowwise!(
        cache.SLbuffer,
        getcolptr(A),
        rowvals(A),
        getnzval(A),
        start_idx,
        end_idx,
        partsize,
        k,
    )
end

function _pack_strict_triangular!(
    cache::PackedStrictLower,
    ::AbstractMatrixSymmetry,
    ::CSRFormat,
    A,
    start_idx::Ti,
    end_idx::Ti,
    partsize::Ti,
    k::Ti,
) where {Ti <: Integer}
    # CSR: direct row-wise access
    _pack_strict_lower_rowwise!(
        cache.SLbuffer,
        getrowptr(A),
        colvals(A),
        getnzval(A),
        start_idx,
        end_idx,
        partsize,
        k,
    )
end

# Row-wise packing for strictly upper triangular elements
# Works for both:
#   - CSR format: pass (rowPtr, colVal, nzVal)
#   - Symmetric CSC: pass (colPtr, rowVal, nzVal) since row i = column i
function _pack_strict_upper_rowwise!(
    SUbuffer,
    indPtr,   # rowPtr for CSR, colPtr for symmetric CSC
    indices,  # colVal for CSR, rowVal for symmetric CSC
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
        # Scan row i (or column i for symmetric CSC)
        for p = indPtr[i]:(indPtr[i+1]-1)
            j = indices[p]
            if j > i && j <= end_idx  # strictly upper within partition
                SUbuffer[block_offset+idx] = nzVal[p]
                idx += 1
            end
        end
    end
    return nothing
end

function _pack_strict_triangular!(
    cache::PackedStrictUpper,
    ::NonSymmetricMatrix,
    ::CSCFormat,
    A,
    start_idx::Ti,
    end_idx::Ti,
    partsize::Ti,
    k::Ti,
) where {Ti <: Integer}
    (; SUbuffer) = cache
    colPtr = getcolptr(A)
    rowVal = rowvals(A)
    nzVal = getnzval(A)

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

function _pack_strict_triangular!(
    cache::PackedStrictUpper,
    ::SymmetricMatrix,
    ::CSCFormat,
    A,
    start_idx::Ti,
    end_idx::Ti,
    partsize::Ti,
    k::Ti,
) where {Ti <: Integer}
    # Symmetric CSC: row i = column i, so use row-wise access
    _pack_strict_upper_rowwise!(
        cache.SUbuffer,
        getcolptr(A),
        rowvals(A),
        getnzval(A),
        start_idx,
        end_idx,
        partsize,
        k,
    )
end

function _pack_strict_triangular!(
    cache::PackedStrictUpper,
    ::AbstractMatrixSymmetry,
    ::CSRFormat,
    A,
    start_idx::Ti,
    end_idx::Ti,
    partsize::Ti,
    k::Ti,
) where {Ti <: Integer}
    # CSR: direct row-wise access
    _pack_strict_upper_rowwise!(
        cache.SUbuffer,
        getrowptr(A),
        colvals(A),
        getnzval(A),
        start_idx,
        end_idx,
        partsize,
        k,
    )
end


_accumulate_from_cache(cache::BlockStrictLowerView, y, i, k, partsize, start_idx, end_idx) =
    _accumulate_lower_from_matrix(
        cache.A,
        cache.frmt,
        y,
        i,
        start_idx,
        end_idx,
        cache.diag,
        cache.symA,
    )

_accumulate_from_cache(cache::BlockStrictUpperView, y, i, k, partsize, start_idx, end_idx) =
    _accumulate_upper_from_matrix(
        cache.A,
        cache.frmt,
        y,
        i,
        start_idx,
        end_idx,
        cache.diag,
        cache.symA,
    )

# PackedBuffer caches - use k and partsize to compute buffer offsets
function _accumulate_from_cache(cache::PackedStrictLower, y, i, k, partsize, start_idx, end_idx)
    (; SLbuffer) = cache
    block_stride = (partsize * (partsize - 1)) ÷ 2
    block_offset = (k - 1) * block_stride

    local_i = i - start_idx + 1
    row_offset = ((local_i - 1) * (local_i - 2)) ÷ 2

    acc = zero(eltype(y))
    @inbounds for local_j = 1:(local_i-1)
        gj = start_idx + (local_j - 1)
        off_idx = block_offset + row_offset + local_j
        acc += SLbuffer[off_idx] * y[gj]
    end
    return acc
end

function _accumulate_from_cache(cache::PackedStrictUpper, y, i, k, partsize, start_idx, end_idx)
    (; SUbuffer) = cache
    block_stride = (partsize * (partsize - 1)) ÷ 2
    block_offset = (k - 1) * block_stride

    local_i = i - start_idx + 1
    # Row i starts after: sum of upper elements in rows 1..(local_i-1)
    # = sum_{r=1}^{local_i-1} (partsize - r) = (local_i-1)*partsize - (local_i-1)*local_i/2
    row_start = (local_i - 1) * partsize - ((local_i - 1) * local_i) ÷ 2

    acc = zero(eltype(y))
    # Number of upper elements in row i = partsize - local_i
    num_upper = partsize - local_i
    @inbounds for offset = 1:num_upper
        gj = i + offset
        if gj > end_idx
            break
        end
        off_idx = block_offset + row_start + offset
        acc += SUbuffer[off_idx] * y[gj]
    end
    return acc
end

# Row-wise lower accumulation - works for CSR and symmetric CSC
function _accumulate_lower_rowwise(indPtr, indices, nzVal, y, i, start_idx)
    acc = zero(eltype(y))
    @inbounds for p = indPtr[i]:(indPtr[i+1]-1)
        j = indices[p]
        if start_idx <= j < i  # strictly lower within partition
            acc += nzVal[p] * y[j]
        end
    end
    return acc
end

# CSR format: row-wise access (no diagonal indices needed)
function _accumulate_lower_from_matrix(
    A,
    ::CSRFormat,
    y,
    i,
    start_idx,
    end_idx,
    ::AbstractDiagonalIndices,
    ::AbstractMatrixSymmetry,
)
    _accumulate_lower_rowwise(getrowptr(A), colvals(A), getnzval(A), y, i, start_idx)
end

# CSC format (non-symmetric): Unified binary search with diagonal indices
# col_range: columns to iterate over
# search_range_fn: (colPtr, col, diag_idx) -> (p_lo, p_hi) search bounds
function _accumulate_triangular_csc_binsearch(
    A,
    y,
    i,
    col_range,
    diag::DiagonalIndices,
    search_range_fn::F,
) where {F}
    colPtr = getcolptr(A)
    rowVal = rowvals(A)
    nzVal = getnzval(A)

    acc = zero(eltype(y))
    @inbounds for col in col_range
        diag_idx = diag[col]
        p_lo, p_hi = search_range_fn(colPtr, col, diag_idx)

        # Binary search for row i in reduced range
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

# CSC format (non-symmetric): dispatch to unified binary search
function _accumulate_lower_from_matrix(
    A,
    ::CSCFormat,
    y,
    i,
    start_idx,
    end_idx,
    diag::DiagonalIndices,
    ::NonSymmetricMatrix,
)
    col_range = start_idx:min(i-1, end_idx)
    search_range_fn = (colPtr, col, diag_idx) -> (diag_idx + 1, colPtr[col+1] - 1)
    _accumulate_triangular_csc_binsearch(A, y, i, col_range, diag, search_range_fn)
end


# CSC format (symmetric): row i = column i, so use row-wise access
function _accumulate_lower_from_matrix(
    A,
    ::CSCFormat,
    y,
    i,
    start_idx,
    end_idx,
    ::NoDiagonalIndices,
    ::SymmetricMatrix,
)
    _accumulate_lower_rowwise(getcolptr(A), rowvals(A), getnzval(A), y, i, start_idx)
end

## Upper triangular accumulation ##
function _accumulate_upper_from_matrix(
    A,
    ::CSCFormat,
    y,
    i,
    start_idx,
    end_idx,
    diag::DiagonalIndices,
    ::NonSymmetricMatrix,
)
    col_range = (i+1):end_idx
    search_range_fn = (colPtr, col, diag_idx) -> (colPtr[col], diag_idx - 1)
    _accumulate_triangular_csc_binsearch(A, y, i, col_range, diag, search_range_fn)
end

# Row-wise upper accumulation - works for CSR and symmetric CSC
function _accumulate_upper_rowwise(indPtr, indices, nzVal, y, i, end_idx)
    acc = zero(eltype(y))
    @inbounds for p = indPtr[i]:(indPtr[i+1]-1)
        j = indices[p]
        if i < j <= end_idx  # strictly upper within partition
            acc += nzVal[p] * y[j]
        end
    end
    return acc
end

# CSR format: row-wise access (no diagonal indices needed)
function _accumulate_upper_from_matrix(
    A,
    ::CSRFormat,
    y,
    i,
    start_idx,
    end_idx,
    ::NoDiagonalIndices,
    ::AbstractMatrixSymmetry,
)
    _accumulate_upper_rowwise(getrowptr(A), colvals(A), getnzval(A), y, i, end_idx)
end

# CSC format (symmetric): row i = column i, so use row-wise access
function _accumulate_upper_from_matrix(
    A,
    ::CSCFormat,
    y,
    i,
    start_idx,
    end_idx,
    ::NoDiagonalIndices,
    ::SymmetricMatrix,
)
    _accumulate_upper_rowwise(getcolptr(A), rowvals(A), getnzval(A), y, i, end_idx)
end
