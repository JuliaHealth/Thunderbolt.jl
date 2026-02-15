# Encapsulates the CUDA backend specific data (e.g. element caches, memory allocation, etc.)
struct CudaElementAssemblyCache{
    Ti <: Integer,
    MemAlloc,
    ElementsCaches,
    DHType <: AbstractDofHandler,
}
    threads::Ti
    blocks::Ti
    mem_alloc::MemAlloc
    eles_caches::ElementsCaches
    dh::DHType
end

# Thunderbolt setup entry point
function Thunderbolt.setup_operator(
    strategy::ElementAssemblyStrategy{<:CudaDevice},
    integrator::LinearIntegrator,
    solver::AbstractSolver,
    dh::AbstractDofHandler,
)
    if CUDA.functional()
        n_threads = strategy.device.threads
        n_blocks  = strategy.device.blocks
        # Raise error if invalid thread or block count is provided
        if !isnothing(n_threads) && n_threads == 0
            error("n_threads must be greater than zero")
        end
        if !isnothing(n_blocks) && n_blocks == 0
            error("n_blocks must be greater than zero")
        end
        return _init_linop_ea_cuda(strategy, integrator, dh)
    else
        error("CUDA is not functional, please check your GPU driver and CUDA installation")
    end
end

function _init_linop_ea_cuda(
    strategy::ElementAssemblyStrategy,
    integrator::LinearIntegrator,
    dh::AbstractDofHandler,
)
    # Multiple domains not yet supported
    Thunderbolt.check_subdomains(dh)

    n_threads = strategy.device.threads
    n_blocks  = strategy.device.blocks

    IT      = index_type(strategy.device)
    FT      = value_type(strategy.device)
    b       = CUDA.zeros(FT, ndofs(dh))
    cu_dh   = Adapt.adapt_structure(strategy.device, dh)
    n_cells = dh |> get_grid |> getncells |> (x -> convert(IT, x))

    # Determine threads and blocks if not provided
    threads = isnothing(n_threads) ? convert(IT, min(n_cells, 256)) : convert(IT, n_threads)
    blocks  = isnothing(n_blocks) ? _calculate_nblocks(threads, n_cells) : convert(IT, n_blocks)

    n_basefuncs      = convert(IT, ndofs_per_cell(dh))
    eles_caches      = _setup_caches(strategy, integrator, dh)
    mem_alloc        = allocate_device_mem(FeMemShape{FT}, threads, blocks, n_basefuncs)
    element_assembly = CudaElementAssemblyCache(threads, blocks, mem_alloc, eles_caches, cu_dh)

    return LinearOperator(b, integrator, dh, element_assembly)
end


function _calculate_nblocks(threads::Ti, n_cells::Ti) where {Ti <: Integer}
    dev             = CUDA.device()
    no_sms          = CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    required_blocks = cld(n_cells, threads)
    required_blocks < 2 * no_sms || return convert(Ti, 2 * no_sms)
    return convert(Ti, required_blocks)
end


function _setup_caches(
    strategy::ElementAssemblyStrategy,
    integrator::LinearIntegrator,
    dh::AbstractDofHandler,
)
    sdh_to_cache =
        sdh -> begin
            # Build evaluation caches
            element_cache =
                Adapt.adapt_structure(strategy.device, setup_element_cache(integrator, sdh))
            return element_cache
        end
    eles_caches = dh.subdofhandlers .|> sdh_to_cache
    return eles_caches
end


function _launch_kernel!(ker, ker_args, threads, blocks, ::AbstractDeviceGlobalMem)
    CUDA.@sync CUDA.@cuda threads = threads blocks = blocks ker(ker_args...)
    return nothing
end

function _launch_kernel!(ker, ker_args, threads, blocks, mem_alloc::AbstractDeviceSharedMem)
    shmem_size = mem_size(mem_alloc)
    CUDA.@sync CUDA.@cuda threads = threads blocks = blocks shmem = shmem_size ker(ker_args...)
    return nothing
end

# Thunderbolt update entry point
function Thunderbolt._update_linear_operator!(
    op::LinearOperator,
    strategy_cache::CudaElementAssemblyCache,
    time,
)
    (; b, strategy_cache) = op
    (; threads, blocks, mem_alloc, eles_caches, dh) = strategy_cache
    fill!(b, zero(eltype(b)))
    for sdh_idx = 1:length(dh.subdofhandlers)
        sdh         = dh.subdofhandlers[sdh_idx]
        ele_cache   = eles_caches[sdh_idx]
        kernel_args = (b, sdh, ele_cache, mem_alloc, time)
        _launch_kernel!(
            _update_linear_operator_cuda_kernel!,
            kernel_args,
            threads,
            blocks,
            mem_alloc,
        )
    end
end

function _update_linear_operator_cuda_kernel!(b, sdh, element_cache, mem_alloc, time)
    for cell in CellIterator(sdh, mem_alloc)
        bₑ = cellfe(cell)
        assemble_element!(bₑ, cell, element_cache, time)
        dofs = celldofs(cell)
        @inbounds for i in eachindex(dofs)
            CUDA.@atomic b[dofs[i]] += bₑ[i]
        end
    end
    return nothing
end
