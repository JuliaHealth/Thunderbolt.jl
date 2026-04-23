struct InternalVariableInfo
    name::Symbol
    size::Int
end

# struct SubInternalVariableHandler
#     names::Vector{Symbol} # Contains symbols for all variables in the handler
#     local_ranges::Vector{UnitRange{Int}} # Step at given point
#     subdomain_ranges::Vector{StepRange{Int,Int}} # Full range of indices per subdomain per element
# end

# function add!(slvh::SubInternalVariableHandler, info::InternalVariableInfo, qr::QuadratureRule, nel::Int)
#     @assert info.name ∉ slvh.names "Trying to register local variable $(info.name) twice. Registered variables: $(slvh.names)."

#     push(slvh.names, info.name)
#     nqp = length(qr.points)
#     local_range = if length(slvh.subdomain_ranges) > 0
#         o = last(last(slvh.subdomain_ranges))
#         (o+1):(o+1+info.size)
#     else
#         1:info.size
#     end
#     push!(slvh.local_ranges, local_range)

#     local_range = if length(slvh.subdomain_ranges) > 0
#         o = last(last(slvh.subdomain_ranges))
#         (o+1):(o+1+info.size)
#     else
#         1:info.size
#     end
# end

# """
#     InternalVariableHandler(...)

# Handler for variables without associated field. Also called "internal variable".
# """
# struct InternalVariableHandler{M} #<: AbstractDofHandler
#     mesh::M
#     sublvhandlers::Vector{SubInternalVariableHandler} # Must mirror corresponding SubDofHandler index structure
# end

# SubInternalVariableHandler() = SubInternalVariableHandler(Symbol[], Vector{Vector{Int}}(), Vector{StepRange{Int,Int}}(), Vector{Vector{Int}}())
# InternalVariableHandler(mesh) = InternalVariableHandler(mesh, SubInternalVariableHandler[])
# ndofs(lvh::InternalVariableHandler) = length(lvh.subdomain_ranges) > 0 ? last(last(lvh.subdomain_ranges)) : 0

# function local_dofrange(elementid::Int, sym::Symbol, qp::QuadraturePoint)
#     # Well...
# end

InternalVariableHandler(mesh::SimpleMesh) = InternalVariableHandler(zeros(Int, getncells(mesh)), 0)

_add_ivh_subdomain_recursive!(lvh, sdh, ::Nothing, qr) = nothing

function _add_ivh_subdomain_recursive!(lvh, sdh, ivi::InternalVariableInfo, qr)
    _add_ivh_subdomain_recursive!(lvh, sdh, (ivi,), qr)
    return nothing
end

function _add_ivh_subdomain_recursive!(
    lvh,
    sdh,
    ivis::Base.AbstractVecOrTuple{<:InternalVariableInfo},
    qr,
)
    offset = lvh.ndofs + 1
    ivsize_per_qp = sum([ivi.size for ivi in ivis]; init = 0)
    for cell in sdh.cellset
        @assert lvh.internal_variable_offsets[cell] == 0
        lvh.internal_variable_offsets[cell] = offset
        offset += ivsize_per_qp*getnquadpoints(qr)
    end
    lvh.ndofs = offset - 1
    return nothing
end

function add_subdomain!(
    lvh::InternalVariableHandler,
    name::String,
    ivis#=::Vector{InternalVariableInfo}=#,
    qrc::QuadratureRuleCollection,
    compatible_dh::DofHandler,
)
    mesh  = get_grid(compatible_dh)
    cells = mesh.grid.cells
    haskey(mesh.volumetric_subdomains, name) || error(
        "Volumetric Subdomain $name not found on mesh. Available subdomains: $(keys(mesh.volumetric_subdomains))",
    )
    for (celltype, cellset) in mesh.volumetric_subdomains[name].data
        for sdh in compatible_dh.subdofhandlers
            first(cellset).idx ∈ sdh.cellset || continue
            qr = getquadraturerule(qrc, sdh)
            _add_ivh_subdomain_recursive!(lvh, sdh, ivis, qr)
            return
        end
    end
    error("Subdomain $name not found?")
end

# Function to compute a vector-like object to store information at quadrature points on generic (mixed) meshes.
function construct_qvector(
    ::Type{StorageType},
    ::Type{IndexType},
    mesh::SimpleMesh,
    qrc::QuadratureRuleCollection,
    subdomains::Vector{String} = [""],
) where {StorageType, IndexType}
    num_points = 0
    num_cells  = 0
    for subdomain in subdomains
        for (celltype, cellset) in pairs(mesh.volumetric_subdomains[subdomain].data)
            qr         = getquadraturerule(qrc, getcells(mesh, first(cellset).idx))
            num_points += getnquadpoints(qr)*length(cellset)
            num_cells  += length(cellset)
        end
    end
    data    = zeros(eltype(StorageType), num_points)
    offsets = zeros(num_cells+1)

    offsets[1]        = 1
    next_point_offset = 1
    next_cell         = 1
    for subdomain in subdomains
        for (celltype, cellset) in pairs(mesh.volumetric_subdomains[subdomain].data)
            qr = getquadraturerule(qrc, getcells(mesh, first(cellset).idx))
            for cellidx in cellset
                next_point_offset += getnquadpoints(qr)
                next_cell += 1
                offsets[next_cell] = next_point_offset
            end
        end
    end

    return DenseDataRange(StorageType(data), IndexType(offsets))
end

function _compatible_cellset(dh::DofHandler, firstcell::Int)
    for sdh in dh.subdofhandlers
        if firstcell ∈ sdh.cellset
            return sdh.cellset
        end
    end
    error("Cell $firstcell not found.")
end
