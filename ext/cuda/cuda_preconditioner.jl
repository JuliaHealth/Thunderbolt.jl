#########################################
## CUDA L1 Gauss Seidel Preconditioner ##
#########################################

struct CudaPartitioning{Ti} <: AbstractPartitioning 
    threads::Ti # number of diagonals per partition
    blocks::Ti # number of partitions
    size_A::Ti
end

function Thunderbolt.build_l1prec(::CudaL1PrecBuilder, A::SparseMatrixCSC;
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

(builder::CudaL1PrecBuilder)(A::SparseMatrixCSC;
    n_threads::Union{Integer,Nothing}=nothing, n_blocks::Union{Integer,Nothing}=nothing) = build_l1prec(builder, A; n_threads=n_threads, n_blocks=n_blocks)

function _build_cuda_l1prec(A::SparseMatrixCSC, n_threads::Union{Integer,Nothing}, n_blocks::Union{Integer,Nothing})
    # Determine threads and blocks if not provided
    # blocks -> number of partitions
    # threads -> number of diagonals per partition
    threads = isnothing(n_threads) ? convert(Int32, min(n_cells, 256)) : convert(Int32, n_threads)
    blocks = isnothing(n_blocks) ? _calculate_nblocks(threads, n_cells) : convert(Int32, n_blocks)
    size_A = convert(Int32,size(A, 1))
    partitioning = CudaPartitioning(threads, blocks, size_A)
    cuda_A = CUSPARSE.CuSparseMatrixCSC(A)
    return L1Preconditioner(partitioning, cuda_A)
end

# TODO: should x & b be CuArrays? or leave them as AbstractVector?
function Thunderbolt.ldiv!(y::AbstractVector, P::L1Preconditioner{CudaPartitioning{Ti}}, x::AbstractVector) where {Ti}
    # x: residual
    # y: preconditioned residual
    @show "Hello from cuda"
end
