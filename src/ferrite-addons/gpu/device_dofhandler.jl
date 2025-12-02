
# Utility which holds partial information for assembly.
struct DeviceDofHandlerData{sdim, GridType<:Ferrite.AbstractGrid{sdim}, IndexType, IndexVectorType <: AbstractVector{IndexType},Ti<: Integer} <: Ferrite.AbstractDofHandler
    grid::GridType
    cell_dofs::IndexVectorType
    cell_dofs_offset::IndexVectorType
    cell_to_subdofhandler::IndexVectorType
    ndofs::Ti
end

# # Utility which holds partial information for assembly.
struct DeviceSubDofHandler{Ti<:Integer,IPVectorType,IndexType, IndexVectorType <: AbstractVector{IndexType},DHDataType<:DeviceDofHandlerData} <: Ferrite.AbstractDofHandler
    cellset::IndexVectorType
    field_names::IndexVectorType
    field_interpolations::IPVectorType 
    ndofs_per_cell::Ti
    dh_data::DHDataType 
end



struct DeviceDofHandler{DHType <: Ferrite.AbstractDofHandler, SDHTupleType} <: Ferrite.AbstractDofHandler
    dh::DHType
    subdofhandlers::SDHTupleType
end

Ferrite.isclosed(::DeviceDofHandler) = true

function Ferrite.ndofs_per_cell(dh::DeviceDofHandlerData, cell::Ti) where {Ti <: Integer}
    sdhidx = dh.cell_to_subdofhandler[cell]
    sdhidx ∉ 1:length(dh.subdofhandlers) && return 0 # Dof handler is just defined on a subdomain
    return ndofs_per_cell(dh.subdofhandlers[sdhidx])
end
Ferrite.ndofs_per_cell(sdh::DeviceSubDofHandler) = sdh.ndofs_per_cell
Ferrite.get_grid(sdh::DeviceSubDofHandler) = sdh.dh_data.grid
cell_dof_offset(dh::DeviceDofHandlerData, i::Integer) = dh.cell_dofs_offset[i]
Ferrite.get_grid(dh::DeviceDofHandlerData) = dh.grid

function celldofsview(sdh::DeviceSubDofHandler, i::Ti) where {Ti<:Integer}
    offset = cell_dof_offset(sdh.dh_data, i)
    ndofs = ndofs_per_cell(sdh)
    view = @view sdh.dh_data.cell_dofs[offset:(offset + ndofs - convert(Ti, 1))]
    return view
end
