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
    update_linearization!(op::AbstractNonlinearOperator, u::AbstractVector, residual::AbstractVector, time)
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
function update_linearization!(op::BlockOperator, u::BlockVector, residual::BlockVector, time)
    nops = length(op.operators)
    nrows = isqrt(nops)
    for i ∈ 1:nops
        row, col = divrem(i-1, nrows) .+ 1 # index shift due to 1-based indices
        i1 = Block(row)
        row_residual = @view residual[i1]
        @timeit_debug "update block ($row,$col)" update_linearization!(op.operators[i], u, row_residual, time) # :)
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
    AssembledNonlinearOperator(J, element_cache, face_cache, tying_cache, dh)
    TODO other signatures

A model for a function with its fully assembled linearization.

Comes with one entry point for each cache type to handle the most common cases:
    assemble_element! -> update jacobian/residual contribution with internal state variables
    assemble_face! -> update jacobian/residual contribution for boundary
    assemble_tying! -> update jacobian/residual contribution for non-local/externally coupled unknowns within a block operator

TODO
    assemble_interface! -> update jacobian/residual contribution for interface contributions (e.g. DG or FSI)
"""
struct AssembledNonlinearOperator{MatrixType, ElementCacheType, FacetCacheType, TyingCacheType, DHType <: AbstractDofHandler} <: AbstractNonlinearOperator
    J::MatrixType
    element_cache::ElementCacheType
    face_cache::FacetCacheType
    tying_cache::TyingCacheType
    dh::DHType
    function AssembledNonlinearOperator(J::MatrixType, element_cache::ElementCacheType, face_cache::FacetCacheType, tying_cache::TyingCacheType, dh::DHType) where {MatrixType, ElementCacheType, FacetCacheType, TyingCacheType, DHType <: AbstractDofHandler}
        check_subdomains(dh)
        return new{MatrixType, ElementCacheType, FacetCacheType, TyingCacheType, DHType}(J, element_cache, face_cache, tying_cache, dh)
    end
end

function AssembledNonlinearOperator(dh::AbstractDofHandler, field_name::Symbol, element_model, element_qrc::QuadratureRuleCollection)
    @assert length(dh.subdofhandlers) == 1 "Multiple subdomains not yet supported in the nonlinear opeartor."

    firstcell = getcells(Ferrite.get_grid(dh), first(dh.subdofhandlers[1].cellset))
    ip = Ferrite.getfieldinterpolation(dh.subdofhandlers[1], field_name)
    ip_geo = Ferrite.geometric_interpolation(typeof(firstcell))
    element_qr = getquadraturerule(element_qrc, firstcell)
    boundary_qr = getquadraturerule(boundary_qrc, firstcell)

    element_cache  = setup_element_cache(element_model, element_qr, ip, ip_geo)

    AssembledNonlinearOperator(
        create_sparsity_pattern(dh),
        element_cache,
        EmtpyFacetCache(),
        EmptyTyingCache(),
        dh,
    )
end

#Utility constructor to get the nonlinear operator for a single field problem.
function AssembledNonlinearOperator(dh::AbstractDofHandler, field_name::Symbol, element_model, element_qrc::QuadratureRuleCollection, boundary_model, boundary_qrc::FacetQuadratureRuleCollection)
    @assert length(dh.subdofhandlers) == 1 "Multiple subdomains not yet supported in the nonlinear opeartor."

    firstcell = getcells(Ferrite.get_grid(dh), first(dh.subdofhandlers[1].cellset))
    ip = Ferrite.getfieldinterpolation(dh.subdofhandlers[1], field_name)
    ip_geo = Ferrite.geometric_interpolation(typeof(firstcell))
    element_qr = getquadraturerule(element_qrc, firstcell)
    boundary_qr = getquadraturerule(boundary_qrc, firstcell)

    element_cache  = setup_element_cache(element_model, element_qr, ip, ip_geo)
    boundary_cache = setup_boundary_cache(boundary_model, boundary_qr, ip, ip_geo)

    AssembledNonlinearOperator(
        create_sparsity_pattern(dh),
        element_cache,
        boundary_cache,
        EmptyTyingCache(),
        dh,
    )
end

function AssembledNonlinearOperator(dh::AbstractDofHandler, field_name::Symbol, element_model, element_qrc::QuadratureRuleCollection, boundary_model, boundary_qrc::FacetQuadratureRuleCollection, tying_model, tying_qr)
    @assert length(dh.subdofhandlers) == 1 "Multiple subdomains not yet supported in the nonlinear opeartor."

    firstcell = getcells(Ferrite.get_grid(dh), first(dh.subdofhandlers[1].cellset))
    ip = Ferrite.getfieldinterpolation(dh.subdofhandlers[1], field_name)
    ip_geo = Ferrite.geometric_interpolation(typeof(firstcell))
    element_qr = getquadraturerule(element_qrc, firstcell)
    boundary_qr = getquadraturerule(boundary_qrc, firstcell)

    element_cache  = setup_element_cache(element_model, element_qr, ip, ip_geo)
    boundary_cache = setup_boundary_cache(boundary_model, boundary_qr, ip, ip_geo)
    tying_cache = setup_tying_cache(tying_model, tying_qr, ip, ip_geo)

    AssembledNonlinearOperator(
        create_sparsity_pattern(dh),
        element_cache,
        boundary_cache,
        tying_cache,
        dh,
    )
end

getJ(op::AssembledNonlinearOperator) = op.J

function update_linearization!(op::AssembledNonlinearOperator, u::AbstractVector, time)
    @unpack J, element_cache, face_cache, tying_cache, dh  = op

    assembler = start_assemble(J)

    ndofs = ndofs_per_cell(dh)
    Jₑ = zeros(ndofs, ndofs)
    uₑ = zeros(ndofs)
    uₜ = get_tying_dofs(tying_cache, u)
    @inbounds for cell in CellIterator(dh)
        fill!(Jₑ, 0)
        uₑ .= @view u[celldofs(cell)]
        @timeit_debug "assemble element" assemble_element!(Jₑ, uₑ, cell, element_cache, time)
        # TODO maybe it makes sense to merge this into the element routine in a modular fasion?
        # TODO benchmark against putting this into the FacetIterator
        @timeit_debug "assemble faces" for local_face_index ∈ 1:nfacets(cell)
            assemble_face!(Jₑ, uₑ, cell, local_face_index, face_cache, time)
        end
        @timeit_debug "assemble tying"  assemble_tying!(Jₑ, uₑ, uₜ, cell, tying_cache, time)
        assemble!(assembler, celldofs(cell), Jₑ)
    end

    #finish_assemble(assembler)
end

function update_linearization!(op::AssembledNonlinearOperator, u::AbstractVector, residual::AbstractVector, time)
    @unpack J, element_cache, face_cache, tying_cache, dh  = op

    assembler = start_assemble(J, residual)

    ndofs = ndofs_per_cell(dh)
    Jₑ = zeros(ndofs, ndofs)
    rₑ = zeros(ndofs)
    uₑ = zeros(ndofs)
    uₜ = get_tying_dofs(tying_cache, u)
    @timeit_debug "loop" @inbounds for cell in CellIterator(dh)
        dofs = celldofs(cell)
        fill!(Jₑ, 0)
        fill!(rₑ, 0)
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

    #finish_assemble(assembler)
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

struct AssembledBilinearOperator{MatrixType, MatrixType2, CacheType, DHType <: AbstractDofHandler} <: AbstractBilinearOperator
    A::MatrixType
    A_::MatrixType2 # FIXME we need this if we assemble on a different device type than we solve on (e.g. CPU and GPU)
    element_cache::CacheType
    dh::DHType
    function AssembledBilinearOperator(A::MatrixType, A_::MatrixType2, element_cache::CacheType, dh::DHType) where {MatrixType, MatrixType2, CacheType, DHType <: AbstractDofHandler}
        check_subdomains(dh)
        return new{MatrixType, MatrixType2, CacheType, DHType}(A, A_, element_cache, dh)
    end
end

function update_operator!(op::AssembledBilinearOperator, time)
    @unpack A, A_, element_cache, dh  = op

    assembler = start_assemble(A_)

    ndofs = ndofs_per_cell(dh)
    Aₑ = zeros(ndofs, ndofs)

    @inbounds for cell in CellIterator(dh)
        fill!(Aₑ, 0)
        # TODO instead of "cell" pass object with geometry information only
        @timeit_debug "assemble element" assemble_element!(Aₑ, cell, element_cache, time)
        assemble!(assembler, celldofs(cell), Aₑ)
    end

    #finish_assemble(assembler)

    copyto!(nonzeros(A), nonzeros(A_))
end

update_linearization!(op::AbstractBilinearOperator, u::AbstractVector, residual::AbstractVector, time) = update_operator!(op, time)
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
Ferrite.add!(b::Vector, op::LinearNullOperator) = nothing
needs_update(op::LinearNullOperator, t) = false


struct LinearOperator{VectorType, CacheType, DHType <: AbstractDofHandler} <: AbstractLinearOperator
    b::VectorType
    element_cache::CacheType
    dh::DHType
    function LinearOperator(b::VectorType, element_cache::CacheType, dh::DHType) where {VectorType, CacheType, DHType <: AbstractDofHandler}
        check_subdomains(dh)
        return new{VectorType, CacheType, DHType}(b, element_cache, dh)
    end
end

function update_operator!(op::LinearOperator, time)
    @unpack b, element_cache, dh  = op

    # assembler = start_assemble(b)

    ndofs = ndofs_per_cell(dh)
    bₑ = zeros(ndofs)
    fill!(b, 0.0)
    @inbounds for cell in CellIterator(dh)
        fill!(bₑ, 0)
        @timeit_debug "assemble element" assemble_element!(bₑ, cell, element_cache, time)
        # assemble!(assembler, celldofs(cell), bₑ)
        b[celldofs(cell)] .+= bₑ
    end

    #finish_assemble(assembler)
end

"""
Parallel element assembly linear operator.
"""
struct PEALinearOperator{VectorType, EAType, CacheType, DHType <: AbstractDofHandler} <: AbstractLinearOperator
    b::VectorType # [global test function index]
    beas::EAType  # [element in subdomain, local test function index]
                  # global test function index -> element indices
    element_cache::CacheType # Linear operators do have a static cache only
    dh::DHType
    chunksize::Int
    function PEALinearOperator(b::AbstractVector, element_cache, dh::AbstractDofHandler; chunksizehint=64)
        check_subdomains(dh)
        beas = EAVector(dh)
        new{typeof(b), typeof(beas), typeof(element_cache), typeof(dh)}(b, beas, element_cache, dh, chunksizehint)
    end
end

function update_operator!(op::PEALinearOperator, time)
    _update_operator!(op, op.b, time)
end

# Threaded CPU dispatch
function _update_operator!(op::PEALinearOperator, b::Vector, time)
    @unpack element_cache, dh, chunksize = op

    ndofs = ndofs_per_cell(dh)
    fill!(b, 0.0)

    ncells = getncells(get_grid(dh))
    nchunks = ceil(Int, ncells / chunksize)

    # Allocate scratch per chunk
    tlds = [ChunkLocalAssemblyData(CellCache(dh), duplicate_for_parallel(op.element_cache)) for tid in 1:nchunks]
    @timeit_debug "assemble elements" begin
        @batch for chunk in 1:nchunks
            chunkbegin = (chunk-1)*chunksize+1
            chunkbound = min(size(y, 1), chunk*chunksize)
            for row in chunkbegin:chunkbound
                tld = tlds[chunk]
                reinit!(tld.cc, eid)
                bₑ = get_data_for_index(op.beas, eid)
                fill!(bₑ, 0.0)
                assemble_element!(bₑ, tld.cc, tld.ec, time)
            end
        end
    end

    ea_collapse!(b, op.beas)
end

Ferrite.add!(b::AbstractVector, op::AbstractLinearOperator) = b .+= op.b
Base.eltype(op::AbstractLinearOperator) = eltype(op.b)
Base.size(op::AbstractLinearOperator) = sisze(op.b)

# TODO where to put these?
struct AnalyticalCoefficientElementCache{F <: AnalyticalCoefficient, T, CV}
    f::F
    nonzero_intervals::Vector{SVector{2,T}}
    cv::CV
end
duplicate_for_parallel(ec::AnalyticalCoefficientElementCache) = AnalyticalCoefficientElementCache(ec.f, ec.nonzero_intervals, ec.cv)
function assemble_element!(bₑ, cell, element_cache::AnalyticalCoefficientElementCache, time)
    _assemble_element!(bₑ, getcoordinates(cell), element_cache::AnalyticalCoefficientElementCache, time)
end
# We want this to be as fast as possible, so throw away everything unused
@inline function _assemble_element!(bₑ, coords::AbstractVector{<:Vec{dim,T}}, element_cache::AnalyticalCoefficientElementCache, time) where {dim,T}
    @unpack f, cv = element_cache
    n_geom_basefuncs = Ferrite.getngeobasefunctions(cv)
    @inbounds for (qp, w) in pairs(Ferrite.getweights(cv.qr))
        # Compute dΩ
        mapping = Ferrite.calculate_mapping(cv.geo_mapping, qp, coords)
        dΩ = Ferrite.calculate_detJ(Ferrite.getjacobian(mapping)) * w
        # Compute x
        x = spatial_coordinate(cv, qp, coords)
        # Evaluate f
        fx = f.f(x,time)
        # TODO replace with evaluate_coefficient
        # Evaluate all basis functions
        @inbounds for j ∈ 1:getnbasefunctions(cv)
            δu = shape_value(cv, qp, j)
            bₑ[j] += fx * δu * dΩ
        end
    end
end


function needs_update(op::Union{LinearOperator, PEALinearOperator}, t)
    return _needs_update(op, op.element_cache, t)
end

function _needs_update(op::Union{LinearOperator, PEALinearOperator}, element_cache::AnalyticalCoefficientElementCache, t)
    for nonzero_interval ∈ op.element_cache.nonzero_intervals
        nonzero_interval[1] ≤ t ≤ nonzero_interval[2] && return true
    end
    return false
end

function _needs_update(op::Union{LinearOperator, PEALinearOperator}, element_cache::Any, t)
    return false
end
