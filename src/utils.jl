
const DEBUG = Preferences.@load_preference("use_debug", false)

"""
    Thunderbolt.debug_mode(; enable=true)

Helper to turn on (`enable=true`) or off (`enable=false`) debug expressions in Ferrite.

Debug mode influences `Ferrite.@debug expr`: when debug mode is enabled, `expr` is
evaluated, and when debug mode is disabled `expr` is ignored.
"""
function debug_mode(; enable = true)
    if DEBUG == enable == true
        @info "Debug mode already enabled."
    elseif DEBUG == enable == false
        @info "Debug mode already disabled."
    else
        Preferences.@set_preferences!("use_debug" => enable)
        @info "Debug mode $(enable ? "en" : "dis")abled. Restart the Julia session for this change to take effect!"
    end
    return
end

@static if DEBUG
    @eval begin
        macro debugonly(ex)
            return :($(esc(ex)))
        end
    end
else
    @eval begin
        macro debugonly(ex)
            return nothing
        end
    end
end

# TODO remove these once they are merged
module FerriteUtils
using Ferrite
import GPUArraysCore: AbstractGPUVector, AbstractGPUArray
import Adapt

include("ferrite-addons/gpu/device_dofhandler.jl")
include("ferrite-addons/gpu/device_grid.jl")
include("ferrite-addons/PR883.jl")
include("ferrite-addons/gpu/device_memalloc.jl")
include("ferrite-addons/gpu/device_iterator.jl")
include("ferrite-addons/gpu/adapt.jl")

end

include("ferrite-addons/collections.jl")
include("ferrite-addons/quadrature_iterator.jl")


function celldofsview(dh::Ferrite.AbstractDofHandler, i::Integer) 
    ndofs = ndofs_per_cell(dh, i)
    offset = dh.cell_dofs_offset[i]
    return @views dh.cell_dofs[offset:(offset+ndofs-1)]
end


function calculate_element_volume(cell, cellvalues_u, uₑ)
    reinit!(cellvalues_u, cell)
    evol::Float64=0.0;
    @inbounds for qp in QuadratureIterator(cellvalues_u)
        dΩ = getdetJdV(cellvalues_u, qp)
        ∇u = function_gradient(cellvalues_u, qp, uₑ)
        F = one(∇u) + ∇u
        J = det(F)
        evol += J * dΩ
    end
    return evol
end;

function calculate_volume_deformed_mesh(w, dh::DofHandler, cellvalues_u)
    evol::Float64 = 0.0;
    @inbounds for cell in CellIterator(dh)
        global_dofs = celldofs(cell)
        nu = getnbasefunctions(cellvalues_u)
        global_dofs_u = global_dofs[1:nu]
        uₑ = w[global_dofs_u]
        δevol = calculate_element_volume(cell, cellvalues_u, uₑ)
        evol += δevol;
    end
    return evol
end;

@inline angle(v1::Vec{dim,T}, v2::Vec{dim,T}) where {dim, T} = acos((v1 ⋅ v2)/(norm(v1)*norm(v2)))
@inline angle_deg(v1::Vec{dim,T}, v2::Vec{dim,T}) where {dim, T} =  rad2deg(angle(v1, v2))

"""
    unproject(v::Vec{dim,T}, n::Vec{dim,T}, α::T)::Vec{dim, T}

Unproject the vector `v` from the plane with normal `n` such that the angle between `v` and the
resulting vector is `α` (given in radians).

!!! note It is assumed that the vectors are normalized and orthogonal, i.e. `||v|| = 1`, `||n|| = 1`
         and `v \\cdot n = 0`.
"""
@inline function unproject(v::Vec{dim,T}, n::Vec{dim,T}, α::T)::Vec{dim, T} where {dim, T}
    @debugonly @assert norm(v) ≈ 1.0
    @debugonly @assert norm(n) ≈ 1.0
    @debugonly @assert v ⋅ n ≈ 0.0

    α ≈ π/2.0 && return n # special case to prevent division by 0

    λ = (sqrt(1-cos(α)^2))/cos(α)
    return v + λ * n
end

"""
    rotate_around(v::Vec{dim,T}, a::Vec{dim,T}, θ::T)::Vec{dim,T}

Perform a Rodrigues' rotation of the vector `v` around the axis `a` with `θ` radians.

!!! note It is assumed that the vectors are normalized, i.e. `||v|| = 1` and `||a|| = 1`.
"""
@inline function rotate_around(v::Vec{dim,T}, a::Vec{dim,T}, θ::T)::Vec{dim,T} where {dim, T}
    @debugonly @assert norm(n) ≈ 1.0

    return v * cos(θ) + (a × v) * sin(θ) + a * (a ⋅ v) * (1-cos(θ))
end

"""
    orthogonalize(v₁::Vec{dim,T}, v₂::Vec{dim,T})::Vec{dim,T}

Returns a new `v₁` which is orthogonal to `v₂`.
"""
@inline function orthogonalize(v₁::Vec{dim,T}, v₂::Vec{dim,T})::Vec{dim,T} where {dim, T} 
    return v₁ - (v₁ ⋅ v₂)*v₂
end

"""
    orthogonalize_normal_system(v₁::Vec{dim,T}, v₂::Vec{dim,T})

Returns new vectors which are orthogonal to each other.
"""
@inline function orthogonalize_normal_system(v₁::Vec{2,T}, v₂::Vec{2,T}) where {T}
    w₁ = v₁
    w₂ = v₂ - (w₁ ⋅ v₂)*w₁
    return w₁, w₂
end

orthogonalize_system(v₁::Vec{2}, v₂::Vec{2}) = orthogonalize_normal_system(v₁/norm(v₁), v₂/norm(v₂))

"""
    orthogonalize_normal_system(v₁::Vec{3}, v₂::Vec{3}, v₃::Vec{3})

Returns new vectors which are orthogonal to each other.
"""
@inline function orthogonalize_normal_system(v₁::Vec{3}, v₂::Vec{3}, v₃::Vec{3})
    w₁ = v₁
    w₂ = v₂ - (w₁ ⋅ v₂)*w₁
    w₃ = v₃ - (w₁ ⋅ v₃)*w₁ - (w₂ ⋅ v₃)*w₂
    return w₁, w₂, w₃
end

orthogonalize_system(v₁::Vec{3}, v₂::Vec{3}, v₃::Vec{3}) = orthogonalize_normal_system(v₁/norm(v₁), v₂/norm(v₂), v₃/norm(v₃))

"""
"""
function generate_nodal_quadrature_rule(ip::Interpolation{ref_shape, order}) where {ref_shape, order}
    n_base = Ferrite.getnbasefunctions(ip)
    positions = Ferrite.reference_coordinates(ip)
    return QuadratureRule{ref_shape, Float64}(ones(length(positions)), positions)
end

"""
    ThreadedSparseMatrixCSR
Threaded version of SparseMatrixCSR.

Based on https://github.com/BacAmorim/ThreadedSparseCSR.jl .
"""
struct ThreadedSparseMatrixCSR{Tv,Ti<:Integer} <: AbstractSparseMatrix{Tv,Ti}
    A::SparseMatrixCSR{1,Tv,Ti}
end

function ThreadedSparseMatrixCSR(m::Integer, n::Integer, rowptr::Vector{Ti}, colval::Vector{Ti}, nzval::Vector{Tv}) where {Tv,Ti<:Integer}
    ThreadedSparseMatrixCSR(SparseMatrixCSR{1}(m,n,rowptr,colval,nzval))
end

function ThreadedSparseMatrixCSR(a::Transpose{Tv,<:SparseMatrixCSC} where Tv)
    ThreadedSparseMatrixCSR(SparseMatrixCSR(a))
end

function mul!(y::AbstractVector{<:Number}, A_::ThreadedSparseMatrixCSR, x::AbstractVector{<:Number}, alpha::Number, beta::Number)
    A = A_.A
    A.n == size(x, 1) || throw(DimensionMismatch())
    A.m == size(y, 1) || throw(DimensionMismatch())
    
    @batch minbatch = size(y, 1) ÷ Threads.nthreads() for row in 1:size(y, 1)
        @inbounds begin
            v = zero(eltype(y))
            for nz in nzrange(A, row)
                col = A.colval[nz]
                v += A.nzval[nz]*x[col]
            end
            y[row] = alpha*v + beta*y[row]
        end
    end

    return y
end

function mul!(y::AbstractVector{<:Number}, A_::ThreadedSparseMatrixCSR, x::AbstractVector{<:Number})
    A = A_.A
    A.n == size(x, 1) || throw(DimensionMismatch())
    A.m == size(y, 1) || throw(DimensionMismatch())

    @batch minbatch = max(1, size(y, 1) ÷ Threads.nthreads()) for row in 1:size(y, 1)
        @inbounds begin
            v = zero(eltype(y))
            for nz in nzrange(A, row)
                col = A.colval[nz]
                v += A.nzval[nz]*x[col]
            end
            y[row] = v
        end
    end

    return y
end

function mul(A::ThreadedSparseMatrixCSR, x::AbstractVector)
    y = similar(x, promote_type(eltype(A), eltype(x)), size(A, 1))
    return mul!(y, A, x)
end
*(A::ThreadedSparseMatrixCSR, v::AbstractVector) = mul(A,v)
*(A::ThreadedSparseMatrixCSR, v::BlockArrays.FillArrays.AbstractZeros{<:Any, 1}) = mul(A,v)
*(A::ThreadedSparseMatrixCSR, v::BlockArrays.ArrayLayouts.LayoutVector) = mul(A,v)
*(A::ThreadedSparseMatrixCSR, v::ModelingToolkit.DynamicQuantities.QuantityArray{T, 1, D, Q, V}) where {T, D<:ModelingToolkit.DynamicQuantities.AbstractDimensions, Q<:ModelingToolkit.DynamicQuantities.UnionAbstractQuantity{T, D}, V<:AbstractVector{T}} = mul(A,v)
*(A::ThreadedSparseMatrixCSR, v::ModelingToolkit.DynamicQuantities.QuantityArray{T, 2, D, Q, V}) where {T, D<:ModelingToolkit.DynamicQuantities.AbstractDimensions, Q<:ModelingToolkit.DynamicQuantities.UnionAbstractQuantity{T, D}, V<:AbstractMatrix{T}} = mul(A,v)

Base.eltype(A::ThreadedSparseMatrixCSR)            = Base.eltype(A.A)
Base.size(A::ThreadedSparseMatrixCSR)              = Base.size(A.A)
Base.size(A::ThreadedSparseMatrixCSR,i)            = Base.size(A.A,i)
Base.IndexStyle(::Type{<:ThreadedSparseMatrixCSR}) = IndexCartesian()

SparseMatricesCSR.getrowptr(A::ThreadedSparseMatrixCSR) = SparseMatricesCSR.getrowptr(A.A)
SparseMatricesCSR.getnzval(A::ThreadedSparseMatrixCSR)  = SparseMatricesCSR.getnzval(A.A)
SparseMatricesCSR.getcolval(A::ThreadedSparseMatrixCSR) = SparseMatricesCSR.getcolval(A.A)

SparseArrays.issparse(A::ThreadedSparseMatrixCSR) = issparse(A.A)
SparseArrays.nnz(A::ThreadedSparseMatrixCSR)      = nnz(A.A)
SparseArrays.nonzeros(A::ThreadedSparseMatrixCSR) = nonzeros(A.A)

Base.@propagate_inbounds function SparseArrays.getindex(A::ThreadedSparseMatrixCSR{T}, i0::Integer, i1::Integer) where T
    getindex(A.A,i0,i1)
end
SparseArrays.getindex(A::ThreadedSparseMatrixCSR, ::Colon, ::Colon) = copy(A)
SparseArrays.getindex(A::ThreadedSparseMatrixCSR, i::Int, ::Colon)       = getindex(A.A, i, 1:size(A, 2))
SparseArrays.getindex(A::ThreadedSparseMatrixCSR, ::Colon, i::Int)       = getindex(A.A, 1:size(A, 1), i)

Ferrite.apply_zero!(A::ThreadedSparseMatrixCSR, f::AbstractVector, ch::ConstraintHandler) = apply_zero!(A.A, f, ch)
function Ferrite.apply_zero!(K::SparseMatrixCSR, f::AbstractVector, ch::ConstraintHandler)
    # m = Ferrite.meandiag(K)

    Ferrite.zero_out_columns!(K, ch.dofmapping)
    Ferrite.zero_out_rows!(K, ch.prescribed_dofs)

    @inbounds for i in 1:length(ch.inhomogeneities)
        d = ch.prescribed_dofs[i]
        K[d, d] = #m
        if length(f) != 0
            f[d] = 0.0
        end
    end
end

function Ferrite.zero_out_rows!(K::SparseMatrixCSR, dofs::Vector{Int}) # can be removed in 0.7 with #24711 merged
    @debugonly @assert issorted(dofs)
    for col in dofs
        r = nzrange(K, col)
        K.nzval[r] .= 0.0
    end
end

function Ferrite.zero_out_columns!(K::SparseMatrixCSR, dofmapping::Dict)
    colval = K.colval
    nzval = K.nzval
    @inbounds for i in eachindex(colval, nzval)
        if haskey(dofmapping, colval[i])
            nzval[i] = 0
        end
    end
end

# struct RHSDataCSR{T}
#     m::T
#     constrained_rows::SparseMatrixCSR{T, Int}
# end

# function Ferrite.get_rhs_data(ch::ConstraintHandler, A::ThreadedSparseMatrixCSR)
#     Ferrite.get_rhs_data(ch, A.A)
# end

# function Ferrite.get_rhs_data(ch::ConstraintHandler, A::SparseMatrixCSR)
#     m = Ferrite.meandiag(A)
#     constrained_rows = A[ch.prescribed_dofs, :]
#     return RHSDataCSR(m, constrained_rows)
# end

# function apply_rhs!(data::RHSDataCSR, f::AbstractVector{T}, ch::ConstraintHandler, applyzero::Bool=false) where T
#     K = data.constrained_rows
#     @assert length(f) == size(K, 1)
#     @boundscheck checkbounds(f, ch.prescribed_dofs)
#     m = data.m

#     # TODO: Can the loops be combined or does the order matter?
#     @inbounds for i in 1:length(ch.inhomogeneities)
#         v = ch.inhomogeneities[i]
#         if !applyzero && v != 0
#             # for j in nzrange(K, i)
#             #     f[K.rowval[j]] -= v * K.nzval[j]
#             # end
#             error("Imhomogeneous bcs not implemented for CSR.")
#         end
#     end
#     @inbounds for (i, pdof) in pairs(ch.prescribed_dofs)
#         dofcoef = ch.dofcoefficients[i]
#         b = ch.inhomogeneities[i]
#         if dofcoef !== nothing # if affine constraint
#             # for (d, v) in dofcoef
#             #     f[d] += f[pdof] * v
#             # end
#             error("Affine bcs not implemented for CSR.")
#         end
#         bz = applyzero ? zero(T) : b
#         f[pdof] = bz * m
#     end
# end

# Internal helper to throw uniform error messages on problems with multiple subdomains
@noinline check_subdomains(dh::Ferrite.AbstractDofHandler) = length(dh.subdofhandlers) == 1 || throw(ArgumentError("Using DofHandler with multiple subdomains is not currently supported"))
@noinline check_subdomains(grid::Ferrite.AbstractGrid) = length(elementtypes(grid)) == 1 || throw(ArgumentError("Using mixed grid is not currently supported"))

@inline function default_quadrature_order(f, fieldname)
    @unpack dh = f
    @assert fieldname ∈ dh.field_names "Field $fieldname not found in dof handler. Available fields are: $(dh.field_names)."

    for sdh in dh.subdofhandlers
        idx = findfirst(s->s==fieldname, sdh.field_names)
        idx === nothing && continue
        ip = sdh.field_interpolations[idx]
        return max(2*Ferrite.getorder(ip)-1, 2)
    end
end


mtk_parameter_query_filter(discard_me, sym) = false
mtk_parameter_query_filter(param::ModelingToolkit.BasicSymbolic, sym) = true

function query_mtk_parameter_by_symbol(sys, sym::Symbol)
    symbol_list = ModelingToolkit.parameter_symbols(sys)
    idx = findfirst(param->mtk_parameter_query_filter(param,sym), symbol_list)
    idx === nothing && @error "Symbol $sym not found for system $sys."
    return symbol_list[idx]
end

"""
Examples:

* `DenseDataRange{Vector{Int}, Vector{Int}}` to map dofs (outer index) to elements (inner index)
* `DenseDataRange{Vector{Vec{3,Float64}}, Vector{Int}}` to store fluxes per quadrature point (inner index) per element (outer index)
"""
struct DenseDataRange{DataVectorType, IndexVectorType}
    data::DataVectorType
    offsets::IndexVectorType
end

Base.size(v::DenseDataRange) = size(v.data)
Base.getindex(v::DenseDataRange, i::Int) = getindex(v.data, i)

Base.eltype(data::DenseDataRange) = eltype(data.data)

@inline function get_data_for_index(r::DenseDataRange, i::Integer)
    i1 = r.offsets[i]
    i2 = r.offsets[i+1]-1
    return @view r.data[i1:i2]
end

struct ElementDofPair{IndexType}
    element_index::IndexType
    local_dof_index::IndexType
end

"""
"""
struct EAVector{T, EADataType <: AbstractVector{T}, IndexType <: AbstractVector{<:Int}, DofMapType <: AbstractVector{<:ElementDofPair}} <: AbstractVector{T}
    # Buffer for the per element data
    eadata::DenseDataRange{EADataType, IndexType}
    # Map from global dof index to element index and local dof index
    dof_to_element_map::DenseDataRange{DofMapType, IndexType}
end

Base.size(v::EAVector) = size(v.eadata)
Base.getindex(v::EAVector, i::Int) = getindex(v.eadata, i)

function Base.show(io::IO, mime::MIME"text/plain", data::EAVector{T, EADataType, IndexType}) where {T, EADataType, IndexType}
    println(io, "EAVector{T=", T, ", EADataType=", EADataType, ", IndexType=", IndexType, "} with storate for ", size(data.eadata), " entries." )
end

@inline get_data_for_index(r::EAVector, i::Integer) = get_data_for_index(r.eadata, i)

function EAVector(dh::DofHandler)
    @assert length(dh.field_names) == 1
    map  = create_dof_to_element_map(dh)
    grid = get_grid(dh)

    num_entries = length(dh.cell_dofs)
    eadata      = zeros(num_entries)
    eaoffsets   = Int64[]
    next_offset = 1
    push!(eaoffsets, next_offset)
    for i in 1:getncells(grid)
        next_offset += ndofs_per_cell(dh, i)
        push!(eaoffsets, next_offset)
    end

    return EAVector(
        DenseDataRange(eadata, eaoffsets),
        map,
    )
end

# Transfer the element data into a vector
function ea_collapse!(b::Vector, bes::EAVector)
    ndofs = size(b, 1)
    @batch minbatch=max(1, ndofs÷Threads.nthreads()) for dof ∈ 1:ndofs
        _ea_collapse_kernel!(b, dof, bes)
    end
end

@inline function _ea_collapse_kernel!(b::AbstractVector, dof::Integer, bes::EAVector)
    for edp ∈ get_data_for_index(bes.dof_to_element_map, dof)
        be_range = get_data_for_index(bes.eadata, edp.element_index)
        b[dof] += be_range[edp.local_dof_index]
    end
end

struct ChunkLocalAssemblyData{CellCacheType, ElementCacheType}
    cc::CellCacheType
    ec::ElementCacheType
end

create_dof_to_element_map(dh::DofHandler) = create_dof_to_element_map(Int, dh::DofHandler)

function create_dof_to_element_map(::Type{IndexType}, dh::DofHandler) where IndexType
    # Preallocate storage
    dof_to_element_vs = [Set{ElementDofPair{IndexType}}() for _ in 1:ndofs(dh)]
    # Fill set
    for sdh in dh.subdofhandlers
        for cc in CellIterator(sdh)
            eid = Ferrite.cellid(cc)
            for (ldi,dof) in enumerate(celldofs(cc))
                s = dof_to_element_vs[dof]
                push!(s, ElementDofPair(eid, ldi))
            end
        end
    end
    #
    dof_to_element_vv = ElementDofPair{IndexType}[]
    offset = 1
    offsets = IndexType[]
    for dof in 1:ndofs(dh)
        append!(offsets, offset)
        s = dof_to_element_vs[dof]
        offset += length(s)
        append!(dof_to_element_vv, s)
    end
    append!(offsets, offset)
    #
    return DenseDataRange(
        dof_to_element_vv,
        offsets,
    )
end

# To handle embedded elements in the same code
_inner_product_helper(a::Vec, B::Union{Tensor, SymmetricTensor}, c::Vec) = a ⋅ B ⋅ c
_inner_product_helper(a::SVector, B::Union{Tensor, SymmetricTensor}, c::SVector) = Vec(a.data) ⋅ B ⋅ Vec(c.data)
_inner_product_helper(a::Vec, B::AbstractFloat, c::Vec) = a ⋅ c * B
_inner_product_helper(a::SVector, B::AbstractFloat, c::SVector) = Vec(a.data) ⋅ Vec(c.data) * B


function geometric_subdomain_interpolation(sdh::SubDofHandler)
    grid      = get_grid(sdh.dh)
    sdim      = getspatialdim(grid)
    firstcell = getcells(grid, first(sdh.cellset))
    ip_geo    = Ferrite.geometric_interpolation(typeof(firstcell))^sdim
    return ip_geo
end

function get_first_cell(sdh::SubDofHandler)
    grid = get_grid(sdh.dh)
    return getcells(grid, first(sdh.cellset))
end

function adapt_vector_type(::Type{<:Vector}, v::VT) where VT
    return v
end

function get_closest_vertex(val::Vec, grid::AbstractGrid)
    distance = Inf
    closest_vertex = VertexIndex(1, 1)
    snap_size = 1e-8
    for (cell_idx, cell) in enumerate(getcells(grid))
        for (vertex_idx, vertex) in enumerate(vertices(cell))
            distance2 = norm(val - get_node_coordinate(grid, vertex))
            if distance2 < distance
                distance = distance2
                closest_vertex = VertexIndex(cell_idx, vertex_idx)
                if distance ≤ snap_size
                    return closest_vertex
                end
            end
        end
    end
    return closest_vertex
end

@generated function dot_2_1(S1::FourthOrderTensor{dim}, S2::SecondOrderTensor{dim}) where {dim}
    idxS1(i, j, k, l) = Tensors.compute_index(Tensors.get_base(S1), i, j, k, l)
    idxS2(i, j) = Tensors.compute_index(Tensors.get_base(S2), i, j)
    exps = Expr(:tuple)
    for l in 1:dim, k in 1:dim, j in 1:dim, i in 1:dim
        ex1 = Expr[:(Tensors.get_data(S1)[$(idxS1(i, m, k, l))]) for m in 1:dim]
        ex2 = Expr[:(Tensors.get_data(S2)[$(idxS2(m, j))]) for m in 1:dim]
        push!(exps.args, Tensors.reducer(ex1, ex2))
    end
    quote
        $(Expr(:meta, :inline))
        @inbounds return Tensor{4, dim}($exps)
    end
end

@generated function dot_2_1t(S1::FourthOrderTensor{dim}, S2::SecondOrderTensor{dim}) where {dim}
    idxS1(i, j, k, l) = Tensors.compute_index(Tensors.get_base(S1), i, j, k, l)
    idxS2(i, j) = Tensors.compute_index(Tensors.get_base(S2), i, j)
    exps = Expr(:tuple)
    for l in 1:dim, k in 1:dim, j in 1:dim, i in 1:dim
        ex1 = Expr[:(Tensors.get_data(S1)[$(idxS1(i, m, k, l))]) for m in 1:dim]
        ex2 = Expr[:(Tensors.get_data(S2)[$(idxS2(j, m))]) for m in 1:dim]
        push!(exps.args, Tensors.reducer(ex1, ex2))
    end
    quote
        $(Expr(:meta, :inline))
        @inbounds return Tensor{4, dim}($exps)
    end
end

@generated function dot_3_1(S1::FourthOrderTensor{dim}, S2::SecondOrderTensor{dim}) where {dim}
    idxS1(i, j, k, l) = Tensors.compute_index(Tensors.get_base(S1), i, j, k, l)
    idxS2(i, j) = Tensors.compute_index(Tensors.get_base(S2), i, j)
    exps = Expr(:tuple)
    for l in 1:dim, k in 1:dim, j in 1:dim, i in 1:dim
        ex1 = Expr[:(Tensors.get_data(S1)[$(idxS1(i, j, m, l))]) for m in 1:dim]
        ex2 = Expr[:(Tensors.get_data(S2)[$(idxS2(m, j))]) for m in 1:dim]
        push!(exps.args, Tensors.reducer(ex1, ex2))
    end
    quote
        $(Expr(:meta, :inline))
        @inbounds return Tensor{4, dim}($exps)
    end
end

@generated function dot_3_1t(S1::FourthOrderTensor{dim}, S2::SecondOrderTensor{dim}) where {dim}
    idxS1(i, j, k, l) = Tensors.compute_index(Tensors.get_base(S1), i, j, k, l)
    idxS2(i, j) = Tensors.compute_index(Tensors.get_base(S2), i, j)
    exps = Expr(:tuple)
    for l in 1:dim, k in 1:dim, j in 1:dim, i in 1:dim
        ex1 = Expr[:(Tensors.get_data(S1)[$(idxS1(i, j, m, l))]) for m in 1:dim]
        ex2 = Expr[:(Tensors.get_data(S2)[$(idxS2(j, m))]) for m in 1:dim]
        push!(exps.args, Tensors.reducer(ex1, ex2))
    end
    quote
        $(Expr(:meta, :inline))
        @inbounds return Tensor{4, dim}($exps)
    end
end
