###################
## adapt Buffers ##
###################

@adapt_structure KeFeGlobalMem
@adapt_structure FeGlobalMem
@adapt_structure KeGlobalMem

#####################################
## Shallow adaption for DofHandler ##
#####################################
function Adapt.adapt_structure(strategy::CudaAssemblyStrategy, dh::DofHandler)
    IT = inttype(strategy)
    grid = _adapt(strategy, dh.grid)
    cell_dofs = dh.cell_dofs .|> (i -> convert(IT,i)) |> cu
    cell_dofs_offset = dh.cell_dofs_offset .|> (i -> convert(IT,i)) |> cu
    cell_to_sdh = dh.cell_to_subdofhandler .|> (i -> convert(IT,i)) |> cu
    dh_data = DeviceDofHandlerData(
        grid,
        cell_dofs,
        cell_dofs_offset,
        cell_to_sdh,
        convert(IT,dh.ndofs))
    subdofhandlers = dh.subdofhandlers .|> (sdh -> _adapt(strategy, sdh,dh_data)) 
    return DeviceDofHandler(dh,subdofhandlers)
end

_symbols_to_int(symbols,IT::Type) = 1:length(symbols) .|> (sym -> convert(IT, sym))


function _adapt(strategy::CudaAssemblyStrategy, sdh::SubDofHandler,dh_data::DeviceDofHandlerData)
    IT = inttype(strategy)
    cellset =  sdh.cellset |> collect .|> (x -> convert(IT, x)) |> cu 
    field_names =  _symbols_to_int(sdh.field_names,IT) |> cu 
    field_interpolations = sdh.field_interpolations |> convert_vec_to_concrete |> cu 
    ndofs_per_cell =  sdh.ndofs_per_cell
    return DeviceSubDofHandler(cellset, field_names, field_interpolations, ndofs_per_cell,dh_data)
end

function _adapt(::CudaAssemblyStrategy, grid::Grid{sdim, cell_type, T}) where {sdim, cell_type, T}
    node_type = typeof(first(grid.nodes))
    cells =  grid.cells |> convert_vec_to_concrete  |> cu
    nodes =  grid.nodes |> cu
    #TODO subdomain info
    return DeviceGrid{sdim, cell_type, T, typeof(cells), typeof(nodes)}(cells, nodes)
end

######################
## adapt Coefficients ##
######################
function Adapt.adapt_structure(::CudaAssemblyStrategy, element_cache::AnalyticalCoefficientElementCache)
    cc = adapt_structure(CuArray, element_cache.cc)
    nz_intervals = adapt(CuArray, element_cache.nonzero_intervals )
    sv = adapt_structure(CuArray, element_cache.cv)
    return AnalyticalCoefficientElementCache(cc, nz_intervals, sv)
end

function Adapt.adapt_structure(::CudaAssemblyStrategy, cysc::FieldCoefficientCache)
    elementwise_data = adapt(CuArray, cysc.elementwise_data)
    cv = adapt_structure(CuArray, cysc.cv)
    return FieldCoefficientCache(elementwise_data, cv)
end
function Adapt.adapt_structure(::CudaAssemblyStrategy, sphdf::SpatiallyHomogeneousDataField)
    timings = adapt(CuArray, sphdf.timings )
    data = adapt(CuArray, sphdf.data )
    return SpatiallyHomogeneousDataField(timings, data)
end
