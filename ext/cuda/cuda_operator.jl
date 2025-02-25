# Encapsulates the CUDA backend specific data (e.g. element caches, memory allocation, etc.)
struct CudaElementAssembly{Ti<:Integer,MemAlloc,ElementsCaches,DHType<:AbstractDofHandler} <: AbstractElementAssembly
    threads::Ti
    blocks::Ti
    mem_alloc::MemAlloc
    eles_caches::ElementsCaches
    strategy::CudaAssemblyStrategy
    dh::DHType
end

function Thunderbolt.init_linear_operator(strategy::CudaAssemblyStrategy, protocol::IntegrandType, qrc::QuadratureRuleCollection, dh::AbstractDofHandler;
    n_threads::Union{Integer,Nothing}=nothing, n_blocks::Union{Integer,Nothing}=nothing) where {IntegrandType}
    if CUDA.functional()
        # Raise error if invalid thread or block count is provided
        if !isnothing(n_threads) && n_threads == 0
            error("n_threads must be greater than zero")
        end
        if !isnothing(n_blocks) && n_blocks == 0
            error("n_blocks must be greater than zero")
        end
        return _init_linop_cuda(strategy, protocol, qrc, dh, n_threads, n_blocks)
    else
        error("CUDA is not functional, please check your GPU driver and CUDA installation")
    end
end

function _init_linop_cuda(strategy::CudaAssemblyStrategy, protocol::IntegrandType, qrc::QuadratureRuleCollection, dh::AbstractDofHandler,
    n_threads::Union{Integer,Nothing}, n_blocks::Union{Integer,Nothing}) where {IntegrandType}
    IT = inttype(strategy)
    FT = floattype(strategy)
    b = CUDA.zeros(FT, ndofs(dh))
    cu_dh = Adapt.adapt_structure(strategy, dh)
    n_cells = dh |> get_grid |> getncells |> (x -> convert(IT, x))

    # Determine threads and blocks if not provided
    threads = isnothing(n_threads) ? convert(IT, min(n_cells, 256)) : convert(IT, n_threads)
    blocks = isnothing(n_blocks) ? _calculate_nblocks(threads, n_cells) : convert(IT, n_blocks)

    n_basefuncs = convert(IT, ndofs_per_cell(dh))
    eles_caches = _setup_caches(strategy, protocol, qrc, dh)
    mem_alloc = allocate_device_mem(FeMemShape{FT}, threads,blocks, n_basefuncs)
    element_assembly = CudaElementAssembly(threads, blocks, mem_alloc, eles_caches, strategy, cu_dh)

    return GeneralLinearOperator(b, element_assembly)
end


function _calculate_nblocks(threads::Ti, n_cells::Ti) where {Ti<:Integer}
    dev = device()
    no_sms = CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    required_blocks = cld(n_cells, threads)
    required_blocks < 2 * no_sms || return convert(Ti, 2 * no_sms)
    return convert(Ti, required_blocks)
end


function _setup_caches(strategy::CudaAssemblyStrategy, integrand::IntegrandType, qrc::QuadratureRuleCollection, dh::AbstractDofHandler) where {IntegrandType}
    sdh_to_cache = sdh ->
        begin
            # Prepare evaluation caches
            ip = Ferrite.getfieldinterpolation(sdh, sdh.field_names[1])
            element_qr = getquadraturerule(qrc, sdh)

            # Build evaluation caches
            element_cache = Adapt.adapt_structure(strategy, setup_element_cache(integrand, element_qr, ip, sdh))
            return element_cache
        end
    eles_caches = dh.subdofhandlers .|> sdh_to_cache
    return eles_caches
end


function _launch_kernel!(ker,ker_args, threads, blocks, ::AbstractDeviceGlobalMem)
    CUDA.@sync CUDA.@cuda threads = threads blocks = blocks ker(ker_args...)
    return nothing
end

function _launch_kernel!(ker,ker_args, threads, blocks, mem_alloc::AbstractDeviceSharedMem)
    shmem_size = mem_size(mem_alloc)
    CUDA.@sync CUDA.@cuda threads = threads blocks = blocks shmem = shmem_size ker(ker_args...)
    return nothing
end

function Thunderbolt.update_operator!(op::GeneralLinearOperator{<:CudaElementAssembly}, time)
    @unpack b, element_assembly = op
    @unpack threads, blocks, mem_alloc, eles_caches, strategy, dh = element_assembly
    fill!(b, zero(eltype(b)))
    for sdh_idx in 1:length(dh.subdofhandlers)
        sdh = dh.subdofhandlers[sdh_idx]
        ele_cache = eles_caches[sdh_idx]
        kernel_args = (b, sdh, ele_cache, mem_alloc, time)
        _launch_kernel!(_update_linear_operator_kernel!,kernel_args, threads, blocks, mem_alloc)
    end
end

function _update_linear_operator_kernel!(b, sdh, element_cache, mem_alloc, time)
    for cell in CellIterator(sdh, mem_alloc)
        bₑ = cellfe(cell)
        assemble_element!(bₑ, cell, element_cache, time)
        dofs = celldofs(cell)
        @inbounds for i in 1:length(dofs)
            CUDA.@atomic b[dofs[i]] += bₑ[i]
        end
    end
    return nothing
end
