struct InternalVariableInfo
    name::Symbol
    size::Int
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
