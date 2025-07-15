abstract type AbstractTransferOperator end

function get_subdofhandler_indices_on_subdomains(dh::DofHandler, subdomain_names::Vector{String})
    grid = get_grid(dh)
    sdh_ids = Set{Int}()
    for (sdhidx,sdh) in enumerate(dh.subdofhandlers)
        for subdomain_name in subdomain_names
            if first(sdh.cellset) ∈ getcellset(grid, subdomain_name)
                push!(sdh_ids, sdhidx)
            end
        end
    end
    return sort(collect(sdh_ids))
end

function get_subdofhandler_indices_on_subdomains(dh::DofHandler, subdomain_names::Nothing)
    return collect(1:length(dh.subdofhandlers))
end

function _compute_dof_nodes_barrier!(nodes, sdh, dofrange, gip, dof_to_node_map, ref_coords)
    for cc ∈ CellIterator(sdh)
        # Compute for each dof the spatial coordinate of from the reference coordiante and store.
        # NOTE We assume a continuous coordinate field if the interpolation is continuous.
        dofs = @view celldofs(cc)[dofrange]
        for (dofidx,dof) in enumerate(dofs)
            nodes[dof_to_node_map[dof]] = spatial_coordinate(gip, ref_coords[dofidx], getcoordinates(cc))
        end
    end
end

"""
    NodalIntergridInterpolation(dh_from::DofHandler{sdim}, dh_to::DofHandler{sdim}, field_name_from::Symbol, field_name_to::Symbol; subdomain_from = 1:length(dh_from.subdofhandlers), subdomains_to = 1:length(dh_to.subdofhandlers))
    NodalIntergridInterpolation(dh_from::DofHandler{sdim}, dh_to::DofHandler{sdim}, field_name::Symbol; subdomain_from = 1:length(dh_from.subdofhandlers), subdomains_to = 1:length(dh_to.subdofhandlers))
    NodalIntergridInterpolation(dh_from::DofHandler{sdim}, dh_to::DofHandler{sdim}; subdomain_from = 1:length(dh_from.subdofhandlers), subdomains_to = 1:length(dh_to.subdofhandlers))

Construct a transfer operator to move a field `field_name` from dof handler `dh_from` to another
dof handler `dh_to`, assuming that all spatial coordinates of the dofs for `dh_to` are in the
interior or boundary of the mesh contained within dh_from. This is necessary to have valid
interpolation values, as this operator does not have extrapolation functionality.

!!! note
    We assume a continuous coordinate field, if the interpolation of the named field is continuous.
"""
struct NodalIntergridInterpolation{PH <: PointEvalHandler, DH1 <: AbstractDofHandler, DH2 <: AbstractDofHandler} <: AbstractTransferOperator
    ph::PH
    dh_from::DH1
    dh_to::DH2
    node_to_dof_map::Vector{Int}
    dof_to_node_map::Dict{Int,Int}
    field_name_from::Symbol
    field_name_to::Symbol

    function NodalIntergridInterpolation(dh_from::DofHandler{sdim}, dh_to::DofHandler{sdim}, field_name_from::Symbol, field_name_to::Symbol; subdomains_from = 1:length(dh_from.subdofhandlers), subdomains_to = 1:length(dh_to.subdofhandlers)) where sdim
        @assert field_name_from ∈ Ferrite.getfieldnames(dh_from)
        @assert field_name_to ∈ Ferrite.getfieldnames(dh_to)

        dofset = Set{Int}()
        for sdh in dh_to.subdofhandlers[subdomains_to]
            # Skip subdofhandler if field is not present
            field_name_to ∈ Ferrite.getfieldnames(sdh) || continue
            # Just gather the dofs of the given field in the set
            for cellidx ∈ sdh.cellset
                dofs = celldofs(dh_to, cellidx)
                for dof in dofs[dof_range(sdh, field_name_to)]
                    push!(dofset, dof)
                end
            end
        end
        node_to_dof_map = sort(collect(dofset))

        # Build inverse map
        dof_to_node_map = Dict{Int,Int}()
        next_dof_index = 1
        for dof ∈ node_to_dof_map
            dof_to_node_map[dof] = next_dof_index
            next_dof_index += 1
        end

        # Compute nodes
        grid_to   = Ferrite.get_grid(dh_to)
        grid_from = Ferrite.get_grid(dh_from)
        nodes = Vector{Ferrite.get_coordinate_type(grid_to)}(undef, length(dofset))
        for sdh in dh_to.subdofhandlers[subdomains_to]
            # Skip subdofhandler if field is not present
            field_name_to ∈ Ferrite.getfieldnames(sdh) || continue
            # Grab the reference coordinates of the field to interpolate
            ip = Ferrite.getfieldinterpolation(sdh, field_name_to)
            ref_coords = Ferrite.reference_coordinates(ip)
            # Grab the geometric interpolation
            first_cell = getcells(grid_to, first(sdh.cellset))
            cell_type  = typeof(first_cell)
            gip = Ferrite.geometric_interpolation(cell_type)

            _compute_dof_nodes_barrier!(nodes, sdh, Ferrite.dof_range(sdh, field_name_to), gip, dof_to_node_map, ref_coords)
        end

        ph = PointEvalHandler(Ferrite.get_grid(dh_from), nodes, [sdh.cellset for sdh in dh_from.subdofhandlers[subdomains_from]]; warn=false)

        n_missing = sum(x -> x === nothing, ph.cells)
        n_missing == 0 || @warn "Constructing the interpolation for $field_name_from to $field_name_to failed. $n_missing (out of $(length(ph.cells))) points not found."

        new{typeof(ph), typeof(dh_from), typeof(dh_to)}(
            ph,
            dh_from,
            dh_to,
            node_to_dof_map,
            dof_to_node_map,
            field_name_from,
            field_name_to
        )
    end
end

function NodalIntergridInterpolation(dh_from::DofHandler{sdim}, dh_to::DofHandler{sdim}) where sdim
    @assert length(Ferrite.getfieldnames(dh_from)) == 1 "Multiple fields found in source dof handler. Please specify which field you want to transfer."
    return NodalIntergridInterpolation(dh_from, dh_to, first(Ferrite.getfieldnames(dh_from)))
end

function NodalIntergridInterpolation(dh_from::DofHandler{sdim}, dh_to::DofHandler{sdim}, field_name::Symbol) where sdim
    return NodalIntergridInterpolation(dh_from, dh_to, field_name, field_name)
end

"""
    This is basically a fancy matrix-vector product to transfer the solution from one problem to another one.
"""
function transfer!(u_to::AbstractArray, operator::NodalIntergridInterpolation, u_from::AbstractArray)
    # TODO non-allocating version
    u_to[operator.node_to_dof_map] .= Ferrite.evaluate_at_points(operator.ph, operator.dh_from, u_from, operator.field_name_from)
end


#### Parameter <-> Solution
# TODO generalize these below into something like
# struct ParameterSyncronizer <: AbstractTransferOperator
#     parameter_buffer
#     index_range_global
#     method
# end
# struct IdentityTransferOpeator <: AbstractTransferOperator end
# syncronize_parameters!(integ, f, syncer::SimpleParameterSyncronizer) = transfer!(parameter_buffer.data, syncer.method, @view f.uparent[index_range_global])
# transfer!(y, syncer::IdentityTransferOpeator, x) = y .= x
"""
    Utility function to synchronize the volume in a split [`RSAFDQ2022Function`](@ref)
"""
struct VolumeTransfer0D3D{TP} <: AbstractTransferOperator
    tying::TP
end

function OS.forward_sync_external!(outer_integrator::OS.OperatorSplittingIntegrator, inner_integrator::DiffEqBase.DEIntegrator, sync::VolumeTransfer0D3D)
    # Tying holds a buffer for the 3D problem with some meta information about the 0D problem
    for chamber ∈ sync.tying.chambers
        chamber.V⁰ᴰval = outer_integrator.u[chamber.V⁰ᴰidx_global]
    end
end

"""
    Utility function to synchronize the pressire in a split [`RSAFDQ2022Function`](@ref)
"""
struct PressureTransfer3D0D{TP } <: AbstractTransferOperator
    tying::TP
end

function OS.forward_sync_external!(outer_integrator::OS.OperatorSplittingIntegrator, inner_integrator::DiffEqBase.DEIntegrator, sync::PressureTransfer3D0D)
    f = inner_integrator.f
    # Tying holds a buffer for the 3D problem with some meta information about the 0D problem
    for (chamber_idx,chamber) ∈ enumerate(sync.tying.chambers)
        p = outer_integrator.u[chamber.pressure_dof_index_global]
        # The pressure buffer is constructed in a way that the chamber index and
        # pressure index coincides
        f.p[chamber_idx] = p
    end
end
