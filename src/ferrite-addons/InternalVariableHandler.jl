struct InternalVariableInfo
    name::Symbol
    size::Int
end

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
    subdomains::Vector{String} = [single_subdomain_or_error(mesh)],
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
