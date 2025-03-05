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
    assemble_face! -> update jacobian/residual contribution for boundary

!!! todo
    assemble_interface! -> update jacobian/residual contribution for interface contributions (e.g. DG or FSI)
"""
struct AssembledNonlinearOperator{MatrixType <: AbstractSparseMatrix, IntegratorType, DHType <: AbstractDofHandler} <: AbstractNonlinearOperator
    J::MatrixType
    integrator::IntegratorType
    dh::DHType
end

function AssembledNonlinearOperator(integrator::NonlinearIntegrator, dh::AbstractDofHandler)
    AssembledNonlinearOperator(
        allocate_matrix(dh), # TODO pass matrix type info
        integrator,
        dh,
    )
end

getJ(op::AssembledNonlinearOperator) = op.J

function update_linearization!(op::AssembledNonlinearOperator, u::AbstractVector, time)
    @unpack J, dh, integrator = op

    assembler = start_assemble(J)

    for sdh in dh.subdofhandlers
        # Build evaluation caches
        element_cache  = setup_element_cache(integrator, sdh)
        face_cache     = setup_boundary_cache(integrator, sdh)

        # Function barrier
        _update_linearization_on_subdomain_J!(assembler, sdh, element_cache, face_cache, EmptyTyingCache(), u, time) # TODO remove tying cache
    end

    #finish_assemble(assembler)
end

function _update_linearization_on_subdomain_J!(assembler, sdh, element_cache, face_cache, tying_cache, u, time)
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
        @timeit_debug "assemble faces" for local_face_index ∈ 1:nfacets(cell)
            assemble_face!(Jₑ, uₑ, cell, local_face_index, face_cache, time)
        end
        @timeit_debug "assemble tying"  assemble_tying!(Jₑ, uₑ, uₜ, cell, tying_cache, time)
        assemble!(assembler, celldofs(cell), Jₑ)
    end
end

function update_linearization!(op::AssembledNonlinearOperator, residual::AbstractVector, u::AbstractVector, time)
    @unpack J, dh, integrator = op

    assembler = start_assemble(J, residual)

    for sdh in dh.subdofhandlers
        # Build evaluation caches
        element_cache  = setup_element_cache(integrator, sdh)
        face_cache     = setup_boundary_cache(integrator, sdh)

        # Function barrier
        _update_linearization_on_subdomain_Jr!(assembler, sdh, element_cache, face_cache, EmptyTyingCache(), u, time) # TODO remove tying cache
    end

    #finish_assemble(assembler)
end

function _update_linearization_on_subdomain_Jr!(assembler, sdh, element_cache, face_cache, tying_cache, u, time)
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
        @timeit_debug "assemble faces" for local_face_index ∈ 1:nfacets(cell)
            assemble_face!(Jₑ, rₑ, uₑ, cell, local_face_index, face_cache, time)
        end
        @timeit_debug "assemble tying"  assemble_tying!(Jₑ, rₑ, uₑ, uₜ, cell, tying_cache, time)
        assemble!(assembler, dofs, Jₑ, rₑ)
    end
end

"""
    mul!(out::AbstractVector, op::AssembledNonlinearOperator, in::AbstractVector)
    mul!(out::AbstractVector, op::AssembledNonlinearOperator, in::AbstractVector, α, β)

Apply the (scaled) action of the linearization of the contained nonlinear form to the vector `in`.
"""
mul!(out::AbstractVector, op::AssembledNonlinearOperator, in::AbstractVector) = mul!(out, op.J, in)
mul!(out::AbstractVector, op::AssembledNonlinearOperator, in::AbstractVector, α, β) = mul!(out, op.J, in, α, β)

Base.eltype(op::AssembledNonlinearOperator) = eltype(op.J)
Base.size(op::AssembledNonlinearOperator, axis) = size(op.J, axis)


abstract type AbstractBilinearOperator <: AbstractNonlinearOperator end

struct AssembledBilinearOperator{MatrixType, MatrixType2, IntegratorType, DHType <: AbstractDofHandler} <: AbstractBilinearOperator
    A::MatrixType
    A_::MatrixType2 # FIXME we need this if we assemble on a different device type than we solve on (e.g. CPU and GPU)
    integrator::IntegratorType
    dh::DHType
end

function update_operator!(op::AssembledBilinearOperator, time)
    @unpack A, A_, integrator, dh  = op

    assembler = start_assemble(A_)

    for sdh in dh.subdofhandlers
        # Build evaluation caches
        element_cache  = setup_element_cache(integrator, sdh)

        # Function barrier
        _update_bilinear_operator_on_subdomain!(assembler, sdh, element_cache, time)
    end

    #finish_assemble(assembler)

    copyto!(nonzeros(A), nonzeros(A_))
end

function _update_bilinear_operator_on_subdomain!(assembler, sdh, element_cache, time)
    ndofs = ndofs_per_cell(sdh)
    Aₑ = zeros(ndofs, ndofs)

    @inbounds for cell in CellIterator(sdh)
        fill!(Aₑ, 0)
        # TODO instead of "cell" pass object with geometry information only
        @timeit_debug "assemble element" assemble_element!(Aₑ, cell, element_cache, time)
        assemble!(assembler, celldofs(cell), Aₑ)
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

struct LinearOperator{VectorType, IntegrandType, DHType <: AbstractDofHandler} <: AbstractLinearOperator
    b::VectorType
    integrand::IntegrandType
    qrc::QuadratureRuleCollection
    dh::DHType
end

function update_operator!(op::LinearOperator, time)
    @unpack b, qrc, dh, integrand  = op # TODO qrc into integrand

    fill!(b, 0.0)
    for sdh in dh.subdofhandlers
        # Prepare evaluation caches
        element_qr  = getquadraturerule(qrc, sdh)

        # Build evaluation caches
        element_cache = setup_element_cache(integrand, element_qr, sdh)

        # Function barrier
        _update_linear_operator_on_subdomain!(b, sdh, element_cache, time)
    end

    #finish_assemble(assembler)
end

function _update_linear_operator_on_subdomain!(b, sdh, element_cache, time)
    ndofs = ndofs_per_cell(sdh)
    bₑ = zeros(ndofs)
    @inbounds for cell in CellIterator(sdh)
        fill!(bₑ, 0)
        @timeit_debug "assemble element" assemble_element!(bₑ, cell, element_cache, time)
        # assemble!(assembler, celldofs(cell), bₑ)
        b[celldofs(cell)] .+= bₑ
    end
end

"""
Parallel element assembly linear operator.
"""
struct PEALinearOperator{VectorType, EAType, ProtocolType, DHType <: AbstractDofHandler} <: AbstractLinearOperator
    b::VectorType # [global test function index]
    beas::EAType  # [element in subdomain, local test function index]
                  # global test function index -> element indices
    qrc::QuadratureRuleCollection
    protocol::ProtocolType
    dh::DHType
    chunksize::Int
    function PEALinearOperator(b::AbstractVector, qrc::QuadratureRuleCollection, protocol, dh::AbstractDofHandler; chunksizehint=64)
        beas = EAVector(dh)
        new{typeof(b), typeof(beas), typeof(protocol), typeof(dh)}(b, beas, qrc, protocol, dh, chunksizehint)
    end
end

function update_operator!(op::PEALinearOperator, time)
    _update_operator!(op, op.b, time)
end

# Threaded CPU dispatch
function _update_operator!(op::PEALinearOperator, b::Vector, time)
    @unpack qrc, dh, chunksize, protocol = op

    @timeit_debug "assemble elements" for sdh in dh.subdofhandlers
        # Prepare evaluation caches
        element_qr  = getquadraturerule(qrc, sdh)

        # Build evaluation caches
        element_cache = setup_element_cache(protocol, element_qr, sdh)

        # Function barrier
        _update_pealinear_operator_on_subdomain!(op.beas, sdh, element_cache, time, chunksize)
    end

    fill!(b, 0.0)
    ea_collapse!(b, op.beas)
end

function _update_pealinear_operator_on_subdomain!(beas::EAVector, sdh, element_cache, time, chunksize::Int)
    ncells = length(sdh.cellset)
    nchunks = ceil(Int, ncells / chunksize)
    tlds = [ChunkLocalAssemblyData(CellCache(sdh), duplicate_for_parallel(element_cache)) for tid in 1:nchunks]
    @batch for chunk in 1:nchunks
        chunkbegin = (chunk-1)*chunksize+1
        chunkbound = min(ncells, chunk*chunksize)
        for i in chunkbegin:chunkbound
            eid = sdh.cellset[i]
            tld = tlds[chunk]
            reinit!(tld.cc, eid)
            bₑ = get_data_for_index(beas, eid)
            fill!(bₑ, 0.0)
            assemble_element!(bₑ, tld.cc, tld.ec, time)
        end
    end
end

Ferrite.add!(b::AbstractVector, op::AbstractLinearOperator) = __add_to_vector!(b, op.b)
__add_to_vector!(b::AbstractVector, a::AbstractVector) = b .+= a
Base.eltype(op::AbstractLinearOperator) = eltype(op.b)
Base.size(op::AbstractLinearOperator) = sisze(op.b)

function needs_update(op::Union{LinearOperator, PEALinearOperator}, t)
    return _needs_update(op, op.protocol, t)
end

function _needs_update(op::Union{LinearOperator, PEALinearOperator}, protocol::AnalyticalTransmembraneStimulationProtocol, t)
    for nonzero_interval ∈ protocol.nonzero_intervals
        nonzero_interval[1] ≤ t ≤ nonzero_interval[2] && return true
    end
    return false
end

function _needs_update(op::Union{LinearOperator, PEALinearOperator}, protocol::NoStimulationProtocol, t)
    return false
end


###################################
# GPU dispatch for LinearOperator #
###################################

# This should encapsulate (elements caches, buffers, kernel launch parameters (e.g. threads and blocks in case of GPU))
abstract type AbstractElementAssembly end

# This linear operator is designed to be able to be dispatched to different devices (i.e. GPU and CPU (single threaded and multi-threaded)).
struct GeneralLinearOperator{ElementAssembly,VectorType}
    b::VectorType
    element_assembly::ElementAssembly
end

# interfaces #
# Concrete implementations for the following abstract types can be found in the corresponding extension.
function init_linear_operator(::AbstractAssemblyStrategy,::IntegrandType,::QuadratureRuleCollection,::AbstractDofHandler) where {IntegrandType}
    error("Not implemented")
end

function update_operator!(::GeneralLinearOperator{AbstractElementAssembly}, time)
    error("Not implemented")
end

(op::GeneralLinearOperator)(time) = update_operator!(op, time)
