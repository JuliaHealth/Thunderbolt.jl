"""
"""
mutable struct ParaViewWriter{PVD}
    const filename::String
    const pvd::PVD
    current_file::Union{VTKGridFile, Nothing}
end

function __paraview_collection(name::String; kwargs...)
    pvd = WriteVTK.paraview_collection(name; kwargs...)
    basename = string(first(split(pvd.path, ".pvd")))
    return WriteVTK.paraview_collection(basename)
end

ParaViewWriter(filename::String; kwargs...) = ParaViewWriter(filename, __paraview_collection("$filename.pvd"; kwargs...), nothing)

function store_timestep!(io::ParaViewWriter, t, grid::AbstractGrid; write_discontinuous = false)
    if io.current_file === nothing
        mkpath(io.filename)
        vtk, cellnodes, node_mapping = Ferrite.create_vtk_grid(io.filename * "/$t.vtu", grid, write_discontinuous)
        io.current_file = VTKGridFile(vtk, cellnodes, node_mapping)
    end
end

function store_timestep!(f::Function, io::ParaViewWriter, t, grid::AbstractGrid)
    store_timestep!(io, t, grid)
    f(io)
    finalize_timestep!(io, t)
end

function store_timestep_field!(io::ParaViewWriter, t, dh::AbstractDofHandler, u::AbstractVector, sym::Symbol, name::String=String(sym))
    @assert io.current_file !== nothing
    fieldnames = Ferrite.getfieldnames(dh)
    idx = findfirst(f->f == sym, fieldnames)
    if idx === nothing
        @warn "Cannot write data for PVD '$(io.filename)'. Field $sym not found in $fieldnames of DofHandler. Skipping."
        return nothing
    end
    data = Ferrite._evaluate_at_grid_nodes(dh, u, sym, #=vtk=# Val(true))
    if Ferrite.write_discontinuous(io.current_file)
        data = Ferrite.evaluate_at_discontinuous_vtkgrid_nodes(dh, u, sym, io.current_file.cellnodes)
    else
        data = Ferrite._evaluate_at_grid_nodes(dh, u, sym, #=vtk=# Val(true))
    end
    Ferrite._vtk_write_node_data(io.current_file.vtk, data, name)
end

function store_timestep_celldata!(io::ParaViewWriter, t, u, coeff_name::String)
    @assert io.current_file !== nothing
    WriteVTK.vtk_cell_data(io.current_file, u, coeff_name)
end

function finalize_timestep!(io::ParaViewWriter, t)
    WriteVTK.vtk_save(io.current_file.vtk)
    io.pvd[t] = io.current_file
    io.current_file = nothing
    # This updates the PVD file
    WriteVTK.save_file(io.pvd.xdoc, io.pvd.path)
end

function finalize!(io::ParaViewWriter)
    close!(io.pvd)
end

# #### Storing coefficients
# # TODO use operator task API
# function store_coefficient!(io::ParaViewWriter, t, coefficient, name)
#     error("Unimplemented")
# end

# function store_coefficient!(io, t, coefficient::AnisotropicPlanarMicrostructureModel, name)
#     store_coefficient!(io, t, coefficient.fiber_coefficient, name*".f", t)
#     store_coefficient!(io, t, coefficient.sheetlet_coefficient, name*".s", t)
# end

# function store_coefficient!(io, t, coefficient::OrthotropicMicrostructureModel, name)
#     store_coefficient!(io, t, coefficient.fiber_coefficient, name*".f", t)
#     store_coefficient!(io, t, coefficient.sheetlet_coefficient, name*".s", t)
#     store_coefficient!(io, t, coefficient.normal_coefficient, name*".n", t)
# end

# function store_coefficient!(io, dh, coefficient::SpectralTensorCoefficient, name, t)
#     store_coefficient!(io, dh, coefficient.eigenvalues, name*".λ", t)
#     store_coefficient!(io, dh, coefficient.eigenvectors, name*".ev", t)
# end

# # TODO split up compute from store
# function store_coefficient!(io::ParaViewWriter, t, coefficient::ConstantCoefficient{T}, name) where {T}
#     data = zeros(T, getncells(grid::AbstractGrid))
#     qrc = QuadratureRuleCollection(1)
#     for cell_cache in CellIterator(dh)
#         qr = getquadraturerule(qr_collection, getcells(get_grid(dh), cellid(cell_cache)))
#         data[cellid(cell_cache)] = evaluate_coefficient(coefficient, cell_cache, first(QuadratureIterator(qr)), t)
#     end
#     WriteVTK.vtk_cell_data(io.current_file, data, name)
# end

# # TODO split up compute from store
# function store_coefficient!(io::ParaViewWriter, dh::DofHandler{sdim}, coefficient::AnalyticalCoefficient{<:Any,<:CoordinateSystemCoefficient}, name, t::TimeType, qr_collection::QuadratureRuleCollection) where {sdim, TimeType}
#     T = Base.return_types(c.f, (Vec{sdim}, TimeType)) # Extract the return type from the function
#     @assert length(T) == 1 "Cannot deduce return type for analytical coefficient! Found: $T"
#     _store_coefficient(T, io, dh, coefficient, name, t, qr_collection)
# end

# function _store_coefficient!(::Union{Type{<:Tuple{T}},Type{<:SVector{T}}}, tlen::Int, io::ParaViewWriter, dh, coefficient::AnalyticalCoefficient, t, qr_collection) where {T}
#     data = zeros(T, getncells(grid::AbstractGrid), tlen)
#     for sdh in dh.subdofhandlers
#         for cell_cache in CellIterator(sdh)
#             qr = getquadraturerule(qr_collection, getcells(get_grid(dh), cellid(cell_cache)))
#             for qp ∈ QuadratureIterator(qr)
#                 tval = evaluate_coefficient(coefficient, cell_cache, qp, t)
#                 for i ∈ 1:tlen
#                     data[cellid(cell_cache), i] += tval[I]
#                 end
#             end
#             data[cellid(cell_cache),:] ./= getnquadpoints(qr)
#         end
#     end
#     for i ∈ 1:tlen
#         WriteVTK.vtk_cell_data(io.current_file, data[:,i], name*".$i") # TODO component names
#     end
# end

# function _store_coefficient!(T::Type, tlen::Int, io::ParaViewWriter, dh, coefficient::AnalyticalCoefficient, t, qr_collection)
#     data = zeros(T, getncells(grid::AbstractGrid))
#     for cell_cache in CellIterator(dh) # TODO subdomain support
#         qr = getquadraturerule(qr_collection, getcells(get_grid(dh), cellid(cell_cache)))
#         for qp ∈ QuadratureIterator(qr)
#             data[cellid(cell_cache)] += evaluate_coefficient(coefficient, cell_cache, qp, t)
#         end
#         data[cellid(cell_cache)] /= getnquadpoints(qr)
#     end
#     WriteVTK.vtk_cell_data(io.current_file, data, name)
# end


"""
"""
mutable struct JLD2Writer{FD}
    const filename::String
    const fd::FD
    grid::Union{Nothing,AbstractGrid}
end

JLD2Writer(filename::String; overwrite::Bool=true, compress::Bool=true) = JLD2Writer(filename, jldopen("$filename.jld2", overwrite ? "w" : "a+"; compress), nothing)

function store_timestep!(io::JLD2Writer, t, grid::AbstractGrid)
    _jld2_maybe_store(io, t, grid)
end

function store_timestep!(f::Function, io::JLD2Writer, t, grid::AbstractGrid)
    store_timestep!(io, t, grid)
    f(io)
    finalize_timestep!(io, t)
end

function _jld2_maybe_store(io::JLD2Writer, t, grid::AbstractGrid)
    if io.grid === nothing
        io.fd["grid"] = grid
    elseif grid !== io.grid
        @info "Grid changed at $t"
        io.fd["grid-$t"] = grid
    end
    io.grid = grid
end

function store_nodal_data!(io::JLD2Writer, t, grid::AbstractGrid, name::String)
    _jld2_maybe_store(io, t, grid)
    io.fd["timesteps/$t/nodal/$name"] = u
end

function store_timestep_field!(io::JLD2Writer, t, dh::AbstractDofHandler, u::AbstractVector, sym::Symbol, name::String=String(sym))
    @assert get_grid(dh) === io.grid
    length(dh.field_names) > 1 && @warn "JLD2Writer cannot handle dof handler with multiple fields yet. Dumping full vector."
    io.fd["timesteps/$t/field/$name"] = u
end

function store_timestep_celldata!(io::JLD2Writer, t, u, name::String)
    @assert lengtu(u) === ncells(io.grid)
    io.fd["timesteps/$t/celldata/$name"] = u
end

finalize_timestep!(io::JLD2Writer, t) = nothing
finalize!(io::JLD2Writer) = nothing

function reorder_nodal!(dh::DofHandler)
    @assert length(dh.field_names) == 1 "Just single field possible."
    grid = Ferrite.get_grid(dh)
    for sdh in dh.subdofhandlers
        firstcellidx = first(sdh.cellset)
        celltype = typeof(getcells(grid,firstcellidx))
        vdim = Ferrite.n_components(sdh.field_interpolations[1])
        for i ∈ 1:getncells(grid)
            dof_offsets = dh.cell_dofs_offset[i]:(dh.cell_dofs_offset[i]+Ferrite.ndofs_per_cell(dh, i)-1)
            for (j,dof_offset_idx) in enumerate(1:vdim:length(dof_offsets))
                for d in 1:vdim
                    dh.cell_dofs[dof_offsets[dof_offset_idx]+d-1] = vdim*(getcells(grid,i).nodes[j]-1) + d
                end
            end
        end
    end
end

function to_ferrite_elements(cells_vtk::Vector{WriteVTK.MeshCell{WriteVTK.VTKCellType, Vector{Int64}}})
    celltype = if cells_vtk[1].ctype == WriteVTK.VTKCellTypes.VTK_TETRA
        Tetrahedron
    elseif cells_vtk[1].ctype == WriteVTK.VTKCellTypes.VTK_HEXAHEDRON
        Hexahedron
    else 
        @error "Unknown cell type" cells_vtk[1].ctype
    end
    cells_ferrite = Vector{celltype}(undef, length(cells_vtk))
    for (i,cell_vtk) in enumerate(cells_vtk)
        cells_ferrite[i] = if cells_vtk[1].ctype == WriteVTK.VTKCellTypes.VTK_TETRA
            Tetrahedron(ntuple(i->cell_vtk.connectivity[i],4))
        elseif cells_vtk[1].ctype == WriteVTK.VTKCellTypes.VTK_HEXAHEDRON
            Hexahedron(ntuple(i->cell_vtk.connectivity[i],8))
        end
    end
    return cells_ferrite
end

function read_vtk_cobivec(filename::String, transmural_id::String, apicobasal_id::String, radial_id::String, transventricular_id::String)
    vtk = ReadVTK.VTKFile(filename)
    points_vtk = ReadVTK.get_points(vtk)
    points_ferrite = [Vec{3}(point) for point in eachcol(points_vtk)]
    cells_vtk = ReadVTK.to_meshcells(ReadVTK.get_cells(vtk))
    cells_ferrite = to_ferrite_elements(cells_vtk)

    grid = Grid(cells_ferrite, Node.(points_ferrite))

    gip =  Ferrite.geometric_interpolation(typeof(grid.cells[1]))

    dh = DofHandler(grid)
    add!(dh, :u, gip)
    close!(dh)
    reorder_nodal!(dh)

    all_data = ReadVTK.get_point_data(vtk)
    u_transmural = ReadVTK.get_data(all_data[transmural_id])
    u_apicobasal = ReadVTK.get_data(all_data[apicobasal_id])
    u_radial = ReadVTK.get_data(all_data[radial_id])
    u_transventricular = ReadVTK.get_data(all_data[transventricular_id])

    epicardium = OrderedSet{FacetIndex}()
    endocardium = OrderedSet{FacetIndex}()
    endocardiumlv = OrderedSet{FacetIndex}()
    endocardiumrv = OrderedSet{FacetIndex}()
    for (cellidx,cell) ∈ enumerate(grid.cells)
        for (facetidx,facetnodes) in enumerate(Ferrite.facets(cell))
            indices = collect(facetnodes)
            if all(u_transmural[indices] .> 1.0-2e-2)
                push!(endocardium, FacetIndex(cellidx, facetidx))
                if all(u_transventricular[indices] .> 0.5) # Right
                    push!(endocardiumrv, FacetIndex(cellidx, facetidx))
                else
                    push!(endocardiumlv, FacetIndex(cellidx, facetidx))
                end
            end
            if all(u_transmural[indices] .< 1e-6)
                push!(epicardium, FacetIndex(cellidx, facetidx))
            end
        end
    end
    Ferrite.addfacetset!(grid, "Epicardium", epicardium)
    Ferrite.addfacetset!(grid, "Endocardium", endocardium)
    Ferrite.addfacetset!(grid, "EndocardiumLV", endocardiumlv)
    Ferrite.addfacetset!(grid, "EndocardiumRV", endocardiumrv)

    return BiVCoordinateSystem(
        dh,
        collect(u_transmural),
        collect(u_apicobasal),
        collect(u_radial),
        collect(u_transventricular)
    )
end
