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

# This is the easiest solution for now
# TODO optimize.
struct InternalVariableHandler{DH} <: AbstractDofHandler
    dh::DH
end
InternalVariableHandler(mesh::SimpleMesh) = InternalVariableHandler(DofHandler(mesh))
Ferrite.close!(lvh::InternalVariableHandler) = close!(lvh.dh)
Ferrite.ndofs(lvh::InternalVariableHandler) = ndofs(lvh.dh)

# Utils to visualize local variables
struct QuadratureInterpolation{RefShape, QR <: QuadratureRule{RefShape}} <: Ferrite.ScalarInterpolation{RefShape, -1}
    qr::QR
end

Ferrite.getnbasefunctions(ip::QuadratureInterpolation) = getnquadpoints(ip.qr)
Ferrite.n_components(ip::QuadratureInterpolation)   = 1
Ferrite.n_dbc_components(::QuadratureInterpolation) = 0
Ferrite.adjust_dofs_during_distribution(::QuadratureInterpolation) = false
Ferrite.volumedof_interior_indices(ip::QuadratureInterpolation) = ntuple(i->i, getnbasefunctions(ip))
# conformity is only used for VTK export and updating the constraint handler. This is not needed since the internal variables are not constrained.
Ferrite.conformity(::QuadratureInterpolation) = Ferrite.L2Conformity()

function Ferrite.reference_coordinates(ip::QuadratureInterpolation)
    return [qp for i in 1:ip.num_components for qp in getpoints(ip.qr)]
end

function Ferrite.reference_shape_value(ip::QuadratureInterpolation, ::Vec, i::Int)
    throw(ArgumentError("shape function evaluation for interpolation $ip not implemented yet"))
end


_add_ivh_subdomain_recursive!(sdh, ::Nothing, qr) = nothing

function _add_ivh_subdomain_recursive!(sdh, ivi::InternalVariableInfo, qr)
    add!(sdh, ivi.name, QuadratureInterpolation(qr)^ivi.size)
end

function _add_ivh_subdomain_recursive!(sdh, ivis::Base.AbstractVecOrTuple, qr)
    for ivi in ivis
        _add_ivh_subdomain_recursive!(sdh, ivi, qr)
    end
end

function add_subdomain!(lvh::InternalVariableHandler, name::String, ivis #=::Vector{InternalVariableInfo}=#, qrc::QuadratureRuleCollection, compatible_dh::DofHandler)
    (; dh) = lvh
    mesh   = get_grid(dh)
    cells = mesh.grid.cells
    haskey(mesh.volumetric_subdomains, name) || error("Volumetric Subdomain $name not found on mesh. Available subdomains: $(keys(mesh.volumetric_subdomains))")
    for (celltype, cellset) in mesh.volumetric_subdomains[name].data
        sdh = SubDofHandler(dh, _compatible_cellset(compatible_dh, first(cellset).idx))
        qr = getquadraturerule(qrc, sdh)
        _add_ivh_subdomain_recursive!(sdh, ivis, qr)
    end
end

# Function to compute a vector-like object to store information at quadrature points on generic (mixed) meshes.
function construct_qvector(::Type{StorageType}, ::Type{IndexType}, mesh::SimpleMesh, qrc::QuadratureRuleCollection, subdomains::Vector{String} = [""]) where {StorageType, IndexType}
    num_points = 0
    num_cells  = 0
    for subdomain in subdomains
        for (celltype, cellset) in pairs(mesh.volumetric_subdomains[subdomain].data)
            qr = getquadraturerule(qrc, getcells(mesh, first(cellset).idx))
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
