# TODO split nonlinear operator and the linearization concepts
# TODO energy based operator?
# TODO maybe a trait system for operators?

"""
    AbstractNonlinearOperator

Models of a nonlinear function F(u)v, where v is a test function.

Interface:
    (op::AbstractNonlinearOperator)(residual::AbstractVector, in::AbstractNonlinearOperator)
    eltype()
    size()

    # linearization
    mul!(out::AbstractVector, op::AbstractNonlinearOperator, in::AbstractVector)
    mul!(out::AbstractVector, op::AbstractNonlinearOperator, in::AbstractVector, α, β)
    update_linearization!(op::AbstractNonlinearOperator, u::AbstractVector, time)
    update_linearization!(op::AbstractNonlinearOperator, residual::AbstractVector, u::AbstractVector, time)
"""
abstract type AbstractNonlinearOperator end

"""
    update_linearization!(op, residual, u, t)

Setup the linearized operator `Jᵤ(u) := dᵤF(u)` in op and its residual `F(u)` in
preparation to solve for the increment `Δu` with the linear problem `J(u) Δu = F(u)`.
"""
update_linearization!(Jᵤ::AbstractNonlinearOperator, residual::AbstractVector, u::AbstractVector, t)

"""
    update_linearization!(op, u, t)

Setup the linearized operator `Jᵤ(u)` in op.
"""
update_linearization!(Jᵤ::AbstractNonlinearOperator, u::AbstractVector, t)

"""
    update_residual!(op, residual, u, problem, t)

Evaluate the residual `F(u)` of the problem.
"""
update_residual!(op::AbstractNonlinearOperator, residual::AbstractVector, u::AbstractVector, t)


abstract type AbstractBlockOperator <: AbstractNonlinearOperator end

getJ(op) = error("J is not explicitly accessible for given operator")

function *(op::AbstractNonlinearOperator, x::AbstractVector)
    y = similar(x)
    mul!(y, op, x)
    return y
end

# TODO constructor which checks for axis compat
struct BlockOperator{OPS <: Tuple, JT} <: AbstractBlockOperator
    # TODO custom "square matrix tuple"
    operators::OPS # stored row by row as in [1 2; 3 4]
    J::JT
end

function BlockOperator(operators::Tuple)
    nblocks = isqrt(length(operators))
    mJs = reshape([getJ(opi) for opi ∈ operators], (nblocks, nblocks))
    block_sizes = [size(op,1) for op in mJs[:,1]]
    total_size = sum(block_sizes)
    # First we define an empty dummy block array
    J = BlockArray(spzeros(total_size,total_size), block_sizes, block_sizes)
    # Then we move the local Js into the dummy to transfer ownership
    for i in 1:nblocks
        for j in 1:nblocks
            J[Block(i,j)] = mJs[i,j]
        end
    end

    return BlockOperator(operators, J)
end

function getJ(op::BlockOperator, i::Block)
    @assert length(i.n) == 2
    return @view op.J[i]
end

getJ(op::BlockOperator) = op.J

function *(op::BlockOperator, x::AbstractVector)
    y = similar(x)
    mul!(y, op, x)
    return y
end

mul!(y, op::BlockOperator, x) = mul!(y, getJ(op), x)

# TODO can we be clever with broadcasting here?
function update_linearization!(op::BlockOperator, u::BlockVector, time)
    for opi ∈ op.operators
        update_linearization!(opi, u, time)
    end
end

# TODO can we be clever with broadcasting here?
function update_linearization!(op::BlockOperator, residual::BlockVector, u::BlockVector, time)
    nops = length(op.operators)
    nrows = isqrt(nops)
    for i ∈ 1:nops
        row, col = divrem(i-1, nrows) .+ 1 # index shift due to 1-based indices
        i1 = Block(row)
        row_residual = @view residual[i1]
        @timeit_debug "update block ($row,$col)" update_linearization!(op.operators[i], row_residual, u, time) # :)
    end
end

# TODO can we be clever with broadcasting here?
function mul!(out::BlockVector, op::BlockOperator, in::BlockVector)
    out .= 0.0
    # 5-arg-mul over 3-ar-gmul because the bocks would overwrite the solution!
    mul!(out, op, in, 1.0, 1.0)
end

# TODO can we be clever with broadcasting here?
function mul!(out::BlockVector, op::BlockOperator, in::BlockVector, α, β)
    nops = length(op.operators)
    nrows = isqrt(nops)
    for i ∈ 1:nops
        i1, i2 = Block.(divrem(i-1, nrows) .+1) # index shift due to 1-based indices
        in_next  = @view in[i1]
        out_next = @view out[i2]
        mul!(out_next, op.operators[i], in_next, α, β)
    end
end

"""
    AssembledNonlinearOperator(J, integrator, dh)
    AssembledNonlinearOperator(integrator, dh)


!!! todo
    signatures

A model for a function with its fully assembled linearization.

Comes with one entry point for each cache type to handle the most common cases:
    assemble_element! -> update jacobian/residual contribution with internal state variables
    assemble_facet! -> update jacobian/residual contribution for boundary

!!! todo
    assemble_interface! -> update jacobian/residual contribution for interface contributions (e.g. DG or FSI)
"""
struct AssembledNonlinearOperator{MatrixType <: AbstractSparseMatrix, IntegratorType, DHType <: AbstractDofHandler, StrategyCacheType} <: AbstractNonlinearOperator
    J::MatrixType
    integrator::IntegratorType
    dh::DHType
    strategy_cache::StrategyCacheType
end

function Base.show(io::IO, cache::AssembledNonlinearOperator)
    println(io, "AssembledNonlinearOperator:")
    Base.show(io, typeof(cache.integrator))
    Base.show(io, MIME"text/plain"(), cache.dh)
end

getJ(op::AssembledNonlinearOperator) = op.J

# Interface
function update_linearization!(op::AssembledNonlinearOperator, u::AbstractVector, time)
    _update_linearization_J!(op, op.strategy_cache, u, time)
end
function update_linearization!(op::AssembledNonlinearOperator, residual::AbstractVector, u::AbstractVector, time)
    _update_linearization_Jr!(op, op.strategy_cache, residual, u, time)
end

"""
    mul!(out::AbstractVector, op::AssembledNonlinearOperator, in::AbstractVector)
    mul!(out::AbstractVector, op::AssembledNonlinearOperator, in::AbstractVector, α, β)

Apply the (scaled) action of the linearization of the contained nonlinear form to the vector `in`.

!!! TODO
    Revisit this decision. Should mul! be the action of the nonlinear operator (i.e. the residual kernel) or of its linearization (i.e. the MV kernel for iterative solvers)?
"""
mul!(out::AbstractVector, op::AssembledNonlinearOperator, in::AbstractVector) = mul!(out, op.J, in)
mul!(out::AbstractVector, op::AssembledNonlinearOperator, in::AbstractVector, α, β) = mul!(out, op.J, in, α, β)

Base.eltype(op::AssembledNonlinearOperator) = eltype(op.J)
Base.size(op::AssembledNonlinearOperator, axis) = size(op.J, axis)

# -------------------------------------------------- Sequential on CPU --------------------------------------------------

# This function is defined to control the dispatch
function _update_linearization_J!(op::AssembledNonlinearOperator, strategy_cache::SequentialAssemblyStrategyCache, u::AbstractVector, time)
    @unpack J, dh, integrator = op

    assembler = start_assemble(J)

    for sdh in dh.subdofhandlers
        # Build evaluation caches
        element_cache  = setup_element_cache(integrator, sdh)
        facet_cache     = setup_boundary_cache(integrator, sdh)

        # Function barrier
        _sequential_update_linearization_on_subdomain_J!(assembler, sdh, element_cache, facet_cache, EmptyTyingCache(), u, time) # TODO remove tying cache
    end

    #finish_assemble(assembler)
end
# This function is defined to make things sufficiently type-stable
function _sequential_update_linearization_on_subdomain_J!(assembler, sdh, element_cache, facet_cache, tying_cache, u, time)
    # Prepare standard values
    ndofs = ndofs_per_cell(sdh)
    Jₑ = zeros(ndofs, ndofs)
    uₑ = zeros(ndofs)
    uₜ = get_tying_dofs(tying_cache, u)
    @inbounds for cell in CellIterator(sdh)
        # Prepare buffers
        fill!(Jₑ, 0)
        uₑ .= @view u[celldofs(cell)]

        # Fill buffers
        @timeit_debug "assemble element" assemble_element!(Jₑ, uₑ, cell, element_cache, time)
        # TODO maybe it makes sense to merge this into the element routine in a modular fasion?
        # TODO benchmark against putting this into the FacetIterator
        @timeit_debug "assemble facets" assemble_element!(Jₑ, uₑ, cell, facet_cache, time)
        @timeit_debug "assemble tying"  assemble_tying!(Jₑ, uₑ, uₜ, cell, tying_cache, time)
        assemble!(assembler, celldofs(cell), Jₑ)
    end
end

# This function is defined to control the dispatch
function _update_linearization_Jr!(op::AssembledNonlinearOperator, strategy_cache::SequentialAssemblyStrategyCache, residual::AbstractVector, u::AbstractVector, time)
    @unpack J, dh, integrator, strategy_cache = op

    assembler = start_assemble(J, residual)

    for sdh in dh.subdofhandlers
        # Build evaluation caches
        element_cache  = setup_element_cache(integrator, sdh)
        facet_cache     = setup_boundary_cache(integrator, sdh)

        # Function barrier
        _sequential_update_linearization_on_subdomain_Jr!(assembler, sdh, element_cache, facet_cache, EmptyTyingCache(), u, time) # TODO remove tying cache
    end

    #finish_assemble(assembler)
end
# This function is defined to make things sufficiently type-stable
function _sequential_update_linearization_on_subdomain_Jr!(assembler, sdh, element_cache, facet_cache, tying_cache, u, time)
    # Prepare standard values
    ndofs = ndofs_per_cell(sdh)
    Jₑ = zeros(ndofs, ndofs)
    uₑ = zeros(ndofs)
    rₑ = zeros(ndofs)
    uₜ = get_tying_dofs(tying_cache, u)
    @inbounds for cell in CellIterator(sdh)
        fill!(Jₑ, 0)
        fill!(rₑ, 0)
        dofs = celldofs(cell)

        uₑ .= @view u[dofs]
        @timeit_debug "assemble element" assemble_element!(Jₑ, rₑ, uₑ, cell, element_cache, time)
        # TODO maybe it makes sense to merge this into the element routine in a modular fasion?
        # TODO benchmark against putting this into the FacetIterator
        @timeit_debug "assemble facets" assemble_element!(Jₑ, rₑ, uₑ, cell, facet_cache, time)
        @timeit_debug "assemble tying"  assemble_tying!(Jₑ, rₑ, uₑ, uₜ, cell, tying_cache, time)
        assemble!(assembler, dofs, Jₑ, rₑ)
    end
end

# -------------------------------------------------- Colored on CPU --------------------------------------------------

# This function is defined to control the dispatch
function _update_linearization_J!(op::AssembledNonlinearOperator, strategy_cache::PerColorAssemblyStrategyCache, u::AbstractVector, time)
    @unpack J, dh, integrator = op

    assembler = start_assemble(J)

    for (sdhidx, sdh) in enumerate(dh.subdofhandlers)
        # Build evaluation caches
        element_cache  = setup_element_cache(integrator, sdh)
        facet_cache     = setup_boundary_cache(integrator, sdh)

        # Function barrier
        _update_colored_linearization_on_subdomain_J!(assembler, strategy_cache.color_cache[sdhidx], sdh, element_cache, facet_cache, EmptyTyingCache(), u, time, strategy_cache.device_cache)
    end

    #finish_assemble(assembler)
end

# This function is defined to make things sufficiently type-stable
function _update_colored_linearization_on_subdomain_J!(assembler, colors, sdh, element_cache, facet_cache, tying_cache, u, time, ::SequentialCPUDevice)
    # Prepare standard values
    ndofs = ndofs_per_cell(sdh)
    Jₑ = zeros(ndofs, ndofs)
    uₑ = zeros(ndofs)
    uₜ = get_tying_dofs(tying_cache, u)
    @inbounds for color in colors
        for cell in CellIterator(sdh.dh, color)
            # Prepare buffers
            fill!(Jₑ, 0)
            uₑ .= @view u[celldofs(cell)]

            # Fill buffers
            @timeit_debug "assemble element" assemble_element!(Jₑ, uₑ, cell, element_cache, time)
            # TODO maybe it makes sense to merge this into the element routine in a modular fasion?
            # TODO benchmark against putting this into the FacetIterator
            @timeit_debug "assemble facets" assemble_element!(Jₑ, uₑ, cell, facet_cache, time)
            @timeit_debug "assemble tying"  assemble_tying!(Jₑ, uₑ, uₜ, cell, tying_cache, time)
            assemble!(assembler, celldofs(cell), Jₑ)
        end
    end
end
function _update_colored_linearization_on_subdomain_J!(assembler, colors, sdh, element_cache, facet_cache, tying_cache, u, time, device::PolyesterDevice)
    (; chunksize) = device
    ncellsmax = maximum(length.(colors))
    nchunksmax = ceil(Int, ncellsmax / chunksize)

    # TODO this should be in the device cache
    ndofs = ndofs_per_cell(sdh)
    Jes  = [zeros(ndofs,ndofs) for tid in 1:nchunksmax]
    ues  = [zeros(ndofs) for tid in 1:nchunksmax]
    tlds = [ChunkLocalAssemblyData(CellCache(sdh), (duplicate_for_device(device, element_cache), duplicate_for_device(device, facet_cache), duplicate_for_device(device, tying_cache))) for tid in 1:nchunksmax]
    assemblers = [duplicate_for_device(device, assembler) for tid in 1:nchunksmax]

    uₜ   = get_tying_dofs(tying_cache, u)

    @inbounds for color in colors
        ncells  = maximum(length(color))
        nchunks = ceil(Int, ncells / chunksize)
        @batch for chunk in 1:nchunks
            chunkbegin = (chunk-1)*chunksize+1
            chunkbound = min(ncells, chunk*chunksize)

            # Unpack chunk scratch
            Jₑ        = Jes[chunk]
            uₑ        = ues[chunk]
            tld       = tlds[chunk]
            assembler = assemblers[chunk]

            for i in chunkbegin:chunkbound
                eid = color[i]
                cell = tld.cc
                reinit!(cell, eid)

                # Prepare buffers
                fill!(Jₑ, 0)
                uₑ .= @view u[celldofs(cell)]

                # Fill buffers
                @timeit_debug "assemble element" assemble_element!(Jₑ, uₑ, cell, tld.ec[1], time)
                # TODO maybe it makes sense to merge this into the element routine in a modular fasion?
                # TODO benchmark against putting this into the FacetIterator
                @timeit_debug "assemble facets" assemble_element!(Jₑ, uₑ, cell, tld.ec[2], time)
                @timeit_debug "assemble tying"  assemble_tying!(Jₑ, uₑ, uₜ, cell, tld.ec[3], time)
                assemble!(assembler, celldofs(cell), Jₑ)
            end
        end
    end
end

# This function is defined to control the dispatch
function _update_linearization_Jr!(op::AssembledNonlinearOperator, strategy_cache::PerColorAssemblyStrategyCache, residual::AbstractVector, u::AbstractVector, time)
    @unpack J, dh, integrator = op

    assembler = start_assemble(J, residual)

    for (sdhidx,sdh) in enumerate(dh.subdofhandlers)
        # Build evaluation caches
        element_cache  = setup_element_cache(integrator, sdh)
        facet_cache     = setup_boundary_cache(integrator, sdh)

        # Function barrier
        _update_colored_linearization_on_subdomain_Jr!(assembler, strategy_cache.color_cache[sdhidx], sdh, element_cache, facet_cache, EmptyTyingCache(), u, time, strategy_cache.device_cache)
    end

    #finish_assemble(assembler)
end

# This function is defined to make things sufficiently type-stable
function _update_colored_linearization_on_subdomain_Jr!(assembler, colors, sdh, element_cache, facet_cache, tying_cache, u, time, ::SequentialCPUDevice)
    # Prepare standard values
    ndofs = ndofs_per_cell(sdh)
    Jₑ = zeros(ndofs, ndofs)
    uₑ = zeros(ndofs)
    rₑ = zeros(ndofs)
    uₜ = get_tying_dofs(tying_cache, u)
    @inbounds for color in colors
        for cell in CellIterator(sdh.dh, color)
            # Prepare buffers
            fill!(Jₑ, 0)
            fill!(rₑ, 0)
            uₑ .= @view u[celldofs(cell)]

            # Fill buffers
            @timeit_debug "assemble element" assemble_element!(Jₑ, rₑ, uₑ, cell, element_cache, time)
            # TODO maybe it makes sense to merge this into the element routine in a modular fasion?
            # TODO benchmark against putting this into the FacetIterator
            @timeit_debug "assemble facets" assemble_element!(Jₑ, rₑ, uₑ, cell, facet_cache, time)
            @timeit_debug "assemble tying"  assemble_tying!(Jₑ, rₑ, uₑ, uₜ, cell, tying_cache, time)
            assemble!(assembler, celldofs(cell), Jₑ, rₑ)
        end
    end
end
function _update_colored_linearization_on_subdomain_Jr!(assembler, colors, sdh, element_cache, facet_cache, tying_cache, u, time, device::PolyesterDevice)
    (; chunksize) = device
    ncellsmax = maximum(length.(colors))
    nchunksmax = ceil(Int, ncellsmax / chunksize)

    # TODO this should be in the device cache
    ndofs = ndofs_per_cell(sdh)
    Jes  = [zeros(ndofs,ndofs) for tid in 1:nchunksmax]
    res  = [zeros(ndofs) for tid in 1:nchunksmax]
    ues  = [zeros(ndofs) for tid in 1:nchunksmax]
    tlds = [ChunkLocalAssemblyData(CellCache(sdh), (duplicate_for_device(device, element_cache), duplicate_for_device(device, facet_cache), duplicate_for_device(device, tying_cache))) for tid in 1:nchunksmax]
    assemblers = [duplicate_for_device(device, assembler) for tid in 1:nchunksmax]

    uₜ   = get_tying_dofs(tying_cache, u)

    @inbounds for color in colors
        ncells  = maximum(length(color))
        nchunks = ceil(Int, ncells / chunksize)
        @batch for chunk in 1:nchunks
            chunkbegin = (chunk-1)*chunksize+1
            chunkbound = min(ncells, chunk*chunksize)

            # Unpack chunk scratch
            Jₑ        = Jes[chunk]
            rₑ        = res[chunk]
            uₑ        = ues[chunk]
            tld       = tlds[chunk]
            assembler = assemblers[chunk]

            for i in chunkbegin:chunkbound
                eid = color[i]
                cell = tld.cc
                reinit!(cell, eid)

                # Prepare buffers
                fill!(Jₑ, 0)
                fill!(rₑ, 0)
                uₑ .= @view u[celldofs(cell)]

                # Fill buffers
                @timeit_debug "assemble element" assemble_element!(Jₑ, rₑ, uₑ, cell, tld.ec[1], time)
                # TODO maybe it makes sense to merge this into the element routine in a modular fasion?
                # TODO benchmark against putting this into the FacetIterator
                @timeit_debug "assemble facets" assemble_element!(Jₑ, rₑ, uₑ, cell, tld.ec[2], time)
                @timeit_debug "assemble tying"  assemble_tying!(Jₑ, rₑ, uₑ, uₜ, cell, tld.ec[3], time)
                assemble!(assembler, celldofs(cell), Jₑ, rₑ)
            end
        end
    end
end

abstract type AbstractBilinearOperator <: AbstractNonlinearOperator end

struct AssembledBilinearOperator{MatrixType, MatrixType2, IntegratorType, DHType <: AbstractDofHandler, StrategyCacheType} <: AbstractBilinearOperator
    A::MatrixType
    A_::MatrixType2 # FIXME we need this if we assemble on a different device type than we solve on (e.g. CPU and GPU)
    integrator::IntegratorType
    dh::DHType
    strategy_cache::StrategyCacheType
end

function update_operator!(op::AssembledBilinearOperator, time)
    _update_bilinaer_operator!(op, op.strategy_cache, time)
end

function _update_bilinaer_operator!(op::AssembledBilinearOperator, strategy_cache::SequentialAssemblyStrategyCache, time)
    @unpack A, A_, integrator, dh  = op

    assembler = start_assemble(A_)

    for sdh in dh.subdofhandlers
        # Build evaluation caches
        element_cache  = setup_element_cache(integrator, sdh)

        # Function barrier
        _update_bilinear_operator_on_subdomain_sequential!(assembler, sdh, element_cache, time)
    end

    #finish_assemble(assembler)

    if A !== A_
        copyto!(nonzeros(A), nonzeros(A_))
    end
end

function _update_bilinear_operator_on_subdomain_sequential!(assembler, sdh, element_cache, time)
    ndofs = ndofs_per_cell(sdh)
    Aₑ = zeros(ndofs, ndofs)

    @inbounds for cell in CellIterator(sdh)
        fill!(Aₑ, 0)
        # TODO instead of "cell" pass object with geometry information only
        @timeit_debug "assemble element" assemble_element!(Aₑ, cell, element_cache, time)
        assemble!(assembler, celldofs(cell), Aₑ)
    end
end

function _update_bilinaer_operator!(op::AssembledBilinearOperator, strategy_cache::PerColorAssemblyStrategyCache{<:AbstractCPUDevice}, time)
    @unpack A, A_, integrator, dh  = op

    assembler = start_assemble(A_)

    for (sdhidx,sdh) in enumerate(dh.subdofhandlers)
        # Build evaluation caches
        element_cache  = setup_element_cache(integrator, sdh)

        # Function barrier
        _update_colored_bilinear_operator_on_subdomain!(assembler, strategy_cache.color_cache[sdhidx], sdh, element_cache, time, strategy_cache.device_cache)
    end

    #finish_assemble(assembler)
    if A !== A_
        copyto!(nonzeros(A), nonzeros(A_))
    end
end

function _update_colored_bilinear_operator_on_subdomain!(assembler, colors, sdh, element_cache, time, device::SequentialCPUDevice)
    # TODO this should be in the device cache
    ndofs     = ndofs_per_cell(sdh)
    Aₑ        = zeros(ndofs,ndofs)

    @timeit_debug "assemble subdomain" for color in colors
        @inbounds for cell in CellIterator(sdh.dh, color)
            fill!(Aₑ, 0)
            @timeit_debug "assemble element" assemble_element!(Aₑ, cell, element_cache, time)
            assemble!(assembler, celldofs(cell), Aₑ)
        end
    end
end

function _update_colored_bilinear_operator_on_subdomain!(assembler, colors, sdh, element_cache, time, device::PolyesterDevice)
    (; chunksize) = device
    ncellsmax = maximum(length.(colors))
    nchunksmax = ceil(Int, ncellsmax / chunksize)
    # TODO this should be in the device cache
    tlds = [ChunkLocalAssemblyData(CellCache(sdh), duplicate_for_device(device, element_cache)) for tid in 1:nchunksmax]

    # TODO this should be in the device cache
    ndofs = ndofs_per_cell(sdh)
    Aes  = [zeros(ndofs,ndofs) for tid in 1:nchunksmax]
    assemblers = [duplicate_for_device(device, assembler) for tid in 1:nchunksmax]

    @timeit_debug "assemble subdomain" for color in colors
        ncells  = maximum(length(color))
        nchunks = ceil(Int, ncells / chunksize)
        @batch for chunk in 1:nchunks
            chunkbegin = (chunk-1)*chunksize+1
            chunkbound = min(ncells, chunk*chunksize)

            # Unpack chunk scratch
            Aₑ        = Aes[chunk]
            tld       = tlds[chunk]
            assembler = assemblers[chunk]

            for i in chunkbegin:chunkbound
                eid = color[i]
                reinit!(tld.cc, eid)

                fill!(Aₑ, 0)
                assemble_element!(Aₑ, tld.cc, tld.ec, time)
                assemble!(assembler, celldofs(tld.cc), Aₑ)
            end
        end
    end
end


update_linearization!(op::AbstractBilinearOperator, residual::AbstractVector, u::AbstractVector, time) = update_operator!(op, time)
update_linearization!(op::AbstractBilinearOperator, u::AbstractVector, time) = update_operator!(op, time)

mul!(out::AbstractVector, op::AssembledBilinearOperator, in::AbstractVector) = mul!(out, op.A, in)
mul!(out::AbstractVector, op::AssembledBilinearOperator, in::AbstractVector, α, β) = mul!(out, op.A, in, α, β)
Base.eltype(op::AssembledBilinearOperator) = eltype(op.A)
Base.size(op::AssembledBilinearOperator, axis) = sisze(op.A, axis)

"""
    DiagonalOperator <: AbstractBilinearOperator

Literally a "diagonal matrix".
"""
struct DiagonalOperator{TV <: AbstractVector} <: AbstractBilinearOperator
    values::TV
end

mul!(out::AbstractVector, op::DiagonalOperator, in::AbstractVector) = out .= op.values .* out
mul!(out::AbstractVector, op::DiagonalOperator, in::AbstractVector, α, β) = out .= α * op.values .* in + β * out
Base.eltype(op::DiagonalOperator) = eltype(op.values)
Base.size(op::DiagonalOperator, axis) = length(op.values)

getJ(op::DiagonalOperator) = spdiagm(op.values)

update_linearization!(::Thunderbolt.DiagonalOperator, ::AbstractVector, ::AbstractVector, t) = nothing

"""
    NullOperator <: AbstractBilinearOperator

Literally a "null matrix".
"""

struct NullOperator{T, SIN, SOUT} <: AbstractBilinearOperator
end

mul!(out::AbstractVector, op::NullOperator, in::AbstractVector) = out .= 0.0
mul!(out::AbstractVector, op::NullOperator, in::AbstractVector, α, β) = out .= β*out
Base.eltype(op::NullOperator{T}) where {T} = T
Base.size(op::NullOperator{T,S1,S2}, axis) where {T,S1,S2} = axis == 1 ? S1 : (axis == 2 ? S2 : error("faulty axis!"))

getJ(op::NullOperator{T, SIN, SOUT}) where {T, SIN, SOUT} = spzeros(T,SIN,SOUT)

update_linearization!(::Thunderbolt.NullOperator, ::AbstractVector, ::AbstractVector, t) = nothing

###############################################################################
"""
    AbstractLinearOperator

Supertype for operators which only depend on the test space.
"""
abstract type AbstractLinearOperator end

"""
    LinearNullOperator <: AbstractLinearOperator

Literally the null vector.
"""
struct LinearNullOperator{T,S} <: AbstractLinearOperator
end
Ferrite.add!(b::AbstractVector, op::LinearNullOperator) = b
Base.eltype(op::LinearNullOperator{T,S}) where {T,S} = T
Base.size(op::LinearNullOperator{T,S}) where {T,S} = S

update_operator!(op::LinearNullOperator, time) = nothing
needs_update(op::LinearNullOperator, t) = false

struct LinearOperator{VectorType, IntegratorType, DHType <: AbstractDofHandler, StrategyCacheType} <: AbstractLinearOperator
    b::VectorType
    integrator::IntegratorType
    dh::DHType
    strategy_cache::StrategyCacheType
end

# Control dispatch for assembly strategy
update_operator!(op::LinearOperator, time) = _update_linear_operator!(op, op.strategy_cache, time)

function _update_linear_operator!(op::LinearOperator, strategy_cache::SequentialAssemblyStrategyCache, time)
    @unpack b, dh, integrator  = op

    fill!(b, 0.0)
    for sdh in dh.subdofhandlers
        # Build evaluation caches
        element_cache = setup_element_cache(integrator, sdh)

        # Function barrier
        _update_linear_operator_on_subdomain_sequential!(b, sdh, element_cache, time)
    end

    #finish_assemble(assembler)
end

function _update_linear_operator_on_subdomain_sequential!(b, sdh, element_cache, time)
    ndofs = ndofs_per_cell(sdh)
    bₑ = zeros(ndofs)
    @inbounds for cell in CellIterator(sdh)
        fill!(bₑ, 0)
        @timeit_debug "assemble element" assemble_element!(bₑ, cell, element_cache, time)
        # assemble!(assembler, celldofs(cell), bₑ)
        b[celldofs(cell)] .+= bₑ
    end
end

# CPU EA dispatch
function _update_linear_operator!(op::AbstractLinearOperator, strategy_cache::ElementAssemblyStrategyCache{<:AbstractCPUDevice}, time)
    @unpack b, dh, integrator  = op

    for sdh in dh.subdofhandlers
        # Build evaluation caches
        element_cache = setup_element_cache(integrator, sdh)

        # Function barrier
        _update_pealinear_operator_on_subdomain!(strategy_cache.ea_data, sdh, element_cache, time, strategy_cache.device_cache)
    end

    fill!(b, 0.0)
    ea_collapse!(b, strategy_cache.ea_data)
end

function _update_pealinear_operator_on_subdomain!(beas::EAVector, sdh, element_cache, time, device::SequentialCPUDevice)
    @timeit_debug "assemble subdomain" @inbounds for cell in CellIterator(sdh)
        bₑ = get_data_for_index(beas, cellid(cell))
        fill!(bₑ, 0)
        @timeit_debug "assemble element" assemble_element!(bₑ, cell, element_cache, time)
    end
end

function _update_pealinear_operator_on_subdomain!(beas::EAVector, sdh, element_cache, time, device::PolyesterDevice)
    (; chunksize) = device
    ncells = length(sdh.cellset)
    nchunks = ceil(Int, ncells / chunksize)
    tlds = [ChunkLocalAssemblyData(CellCache(sdh), duplicate_for_device(device, element_cache)) for tid in 1:nchunks]
    @timeit_debug "assemble subdomain" @batch for chunk in 1:nchunks
        chunkbegin = (chunk-1)*chunksize+1
        chunkbound = min(ncells, chunk*chunksize)
        for i in chunkbegin:chunkbound
            eid = sdh.cellset[i]
            tld = tlds[chunk]
            reinit!(tld.cc, eid)
            bₑ = get_data_for_index(beas, eid)
            fill!(bₑ, 0.0)
            # @timeit_debug "assemble element" assemble_element!(bₑ, tld.cc, tld.ec, time)
            assemble_element!(bₑ, tld.cc, tld.ec, time)
        end
    end
end

function _update_linear_operator!(op::AbstractLinearOperator, strategy_cache::PerColorAssemblyStrategyCache{<:AbstractCPUDevice}, time)
    @unpack b, dh, integrator  = op

    fill!(b, 0.0)

    for (sdhidx, sdh) in enumerate(dh.subdofhandlers)
        # Build evaluation caches
        element_cache = setup_element_cache(integrator, sdh)

        # Function barrier
        _update_colored_linear_operator_on_subdomain!(b, strategy_cache.color_cache[sdhidx], sdh, element_cache, time, strategy_cache.device_cache)
    end
end

function _update_colored_linear_operator_on_subdomain!(b, colors, sdh, element_cache, time, device::SequentialCPUDevice)
    ndofs = ndofs_per_cell(sdh)
    bₑ = zeros(ndofs)
    for color in colors
        @timeit_debug "assemble subdomain" @inbounds for cell in CellIterator(sdh.dh, color)
            fill!(bₑ, 0)
            @timeit_debug "assemble element" assemble_element!(bₑ, cell, element_cache, time)
            b[celldofs(cell)] .+= bₑ
        end
    end
end

function _update_colored_linear_operator_on_subdomain!(b, colors, sdh, element_cache, time, device::PolyesterDevice)
    (; chunksize) = device

    ncellsmax = maximum(length.(colors))
    nchunksmax = ceil(Int, ncellsmax / chunksize)
    # TODO this should be in the device cache
    tlds = [ChunkLocalAssemblyData(CellCache(sdh), duplicate_for_device(device, element_cache)) for tid in 1:nchunksmax]

    # TODO this should be in the device cache
    ndofs = ndofs_per_cell(sdh)
    bes  = [zeros(ndofs) for tid in 1:nchunksmax]

    @timeit_debug "assemble subdomain" for color in colors
        ncells  = maximum(length(color))
        nchunks = ceil(Int, ncells / chunksize)
        @batch for chunk in 1:nchunks
            chunkbegin = (chunk-1)*chunksize+1
            chunkbound = min(ncells, chunk*chunksize)

            # Unpack chunk scratch
            bₑ  = bes[chunk]
            tld = tlds[chunk]

            for i in chunkbegin:chunkbound
                eid = color[i]
                reinit!(tld.cc, eid)

                fill!(bₑ, 0)
                assemble_element!(bₑ, tld.cc, tld.ec, time)
                b[celldofs(tld.cc)] .+= bₑ
            end
        end
    end
end


Ferrite.add!(b::AbstractVector, op::AbstractLinearOperator) = __add_to_vector!(b, op.b)
__add_to_vector!(b::AbstractVector, a::AbstractVector) = b .+= a
Base.eltype(op::AbstractLinearOperator) = eltype(op.b)
Base.size(op::AbstractLinearOperator) = sisze(op.b)

function needs_update(op::LinearOperator, t)
    return _needs_update(op, op.integrator.integrand, t)
end

function _needs_update(op::LinearOperator, protocol::AnalyticalTransmembraneStimulationProtocol, t)
    for nonzero_interval ∈ protocol.nonzero_intervals
        nonzero_interval[1] ≤ t ≤ nonzero_interval[2] && return true
    end
    return false
end

function _needs_update(op::LinearOperator, protocol::NoStimulationProtocol, t)
    return false
end
