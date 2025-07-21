struct VolumetricSubdomainDesriptor
    data::OrderedDict{Type, OrderedSet{CellIndex}}
end

function _show_descriptor(io, descriptor)
    for (i, (type, set)) in enumerate(descriptor.data)
        print(io, "$(length(set)) $type")
        if i < length(descriptor.data)-1
            print(io, ", ")
        elseif i < length(descriptor.data)
            print(io, " and ")
        else
            print(io, " ")
        end
    end
    print(io, "cells")
end

function Base.show(io::IO, ::MIME"text/plain", descriptor::VolumetricSubdomainDesriptor)
    _show_descriptor(io, descriptor)
end

struct SurfaceSubdomainDesriptor
    data::OrderedDict{Type, OrderedSet{FacetIndex}}
end

function Base.show(io::IO, ::MIME"text/plain", descriptor::SurfaceSubdomainDesriptor)
    _show_descriptor(io, descriptor)
end

struct InterfaceIndex
    a::FacetIndex
    b::FacetIndex
end

struct InterfaceSubdomainDesriptor
    data::OrderedDict{Type, OrderedSet{InterfaceIndex}}
end

function Base.show(io::IO, ::MIME"text/plain", descriptor::InterfaceSubdomainDesriptor)
    _show_descriptor(io, descriptor)
end

"""
SimpleMesh{sdim, C <: AbstractCell, T <: Real} <: AbstractGrid{sdim}

A grid which also has information abouts its vertices, facets and edges.

It is also a glorified domain manager for mixed grids and actual subdomains.
TODO investigate whetehr we can remove the subdomains without a significant performance hit.
"""
struct SimpleMesh{sdim, C <: AbstractCell, T <: Real} <: AbstractGrid{sdim}
    grid::Grid{sdim, C, T}
    mfaces::OrderedDict{NTuple{3,Int}, Int} # Maps "sortface"-representation to id
    medges::OrderedDict{NTuple{2,Int}, Int} # Maps "sortedge"-representation to id
    mvertices::OrderedDict{Int, Int} # Maps node to id
    number_of_cells_by_type::OrderedDict{DataType, Int}
    volumetric_subdomains::OrderedDict{String, VolumetricSubdomainDesriptor}
    surface_subdomains::OrderedDict{String, SurfaceSubdomainDesriptor}
    interface_subdomains::OrderedDict{String, InterfaceSubdomainDesriptor}
end

function Base.show(io::IO, ::MIME"text/plain", mesh::SimpleMesh)
    print(io, "$(typeof(mesh)) with $(getncells(mesh)) ")
    if isconcretetype(eltype(mesh.grid.cells))
        typestrs = [repr(eltype(mesh.grid.cells))]
    else
        typestrs = sort!(repr.(OrderedSet(typeof(x) for x in mesh.grid.cells)))
    end
    join(io, typestrs, '/')
    println(io, " cells and $(getnnodes(mesh.grid)) nodes")
    if length(mesh.volumetric_subdomains) > 1 && keys(mesh.volumetric_subdomains)[1] != ""
        println(io, "  Volumetric subdomains:")
        for (name, descriptor) in mesh.volumetric_subdomains
            print(io, "    $name ")
            _show_descriptor(io, descriptor)
            println(io, "")
        end
    end
    if length(mesh.surface_subdomains) > 1
        println(io, "  Surface subdomains:")
        for (name, descriptor) in mesh.surface_subdomains
            print(io, "    $name ")
            _show_descriptor(io, descriptor)
            println(io, "")
        end
    end
    if length(mesh.interface_subdomains) > 1
        println(io, "  Interface subdomains:")
        for (name, descriptor) in mesh.interface_subdomains
            print(io, "    $name ")
            _show_descriptor(io, descriptor)
            println(io, "")
        end
    end
end

global_edges(mgrid::SimpleMesh, cell) = [mgrid.medges[sedge] for sedge ∈ first.(sortedge.(edges(cell)))]
global_faces(mgrid::SimpleMesh, cell) = [mgrid.mfaces[sface] for sface ∈ first.(sortface.(faces(cell)))]
global_vertices(mgrid::SimpleMesh, cell) = [mgrid.mvertices[v] for v ∈ vertices(cell)]

get_faceid_from_nodes(mgrid::SimpleMesh, nodes) = mgrid.mfaces[sortface.(nodes)]
get_faceid(mgrid::SimpleMesh, nodes::NTuple{3,Int}) = mgrid.mfaces[nodes]

num_nodes(mgrid::SimpleMesh) = length(mgrid.grid.nodes)
num_faces(mgrid::SimpleMesh) = length(mgrid.mfaces)
num_edges(mgrid::SimpleMesh) = length(mgrid.medges)
num_vertices(mgrid::SimpleMesh) = length(mgrid.mvertices)

elementtypes(::SimpleMesh{sdim,Triangle}) where sdim = @SVector [Triangle]
elementtypes(::SimpleMesh{sdim,Quadrilateral}) where sdim = @SVector [Quadrilateral]
elementtypes(::SimpleMesh{3,Tetrahedron}) = @SVector [Tetrahedron]
elementtypes(::SimpleMesh{3,Hexahedron}) = @SVector [Hexahedron]

subdomain_names(mesh::SimpleMesh) = collect(keys(mesh.volumetric_subdomains))

function to_mesh(grid::Grid)
    mfaces = OrderedDict{NTuple{3,Int}, Int}()
    medges = OrderedDict{NTuple{2,Int}, Int}()
    mvertices = OrderedDict{Int, Int}()
    next_face_idx = 1
    next_edge_idx = 1
    next_vertex_idx = 1
    number_of_cells_by_type = OrderedDict{DataType, Int}()
    for cell ∈ getcells(grid)
        cell_type = typeof(cell)
        if haskey(number_of_cells_by_type, cell_type)
            number_of_cells_by_type[cell_type] += 1
        else
            number_of_cells_by_type[cell_type] = 1
        end

        for v ∈ vertices(cell)
            if !haskey(mvertices, v)
                mvertices[v] = next_vertex_idx
                next_vertex_idx += 1
            end
        end
        for e ∈ first.(sortedge.(edges(cell)))
            if !haskey(medges, e)
                medges[e] = next_edge_idx
                next_edge_idx += 1
            end
        end
        for f ∈ first.(sortface.(faces(cell)))
            if !haskey(mfaces, f)
                mfaces[f] = next_face_idx
                next_face_idx += 1
            end
        end
    end

    # We split the subdomains by element type to support mixed grids
    volumetric_subdomains = OrderedDict{String, VolumetricSubdomainDesriptor}()
    for (name,cellset) in grid.cellsets
        next_split_set = OrderedDict{Type, OrderedSet{CellIndex}}()
        for cellidx in cellset
            celltype = typeof(grid.cells[cellidx])
            if !haskey(next_split_set, celltype)
                next_split_set[celltype] = OrderedSet{CellIndex}()
            end
            union!(next_split_set[celltype], [CellIndex(cellidx)])
        end
        volumetric_subdomains[name] = VolumetricSubdomainDesriptor(next_split_set)
    end

    # If we have no subdomains defined on the mesh, then we still split the mesh up so we can handle mixed grids properly
    if length(volumetric_subdomains) == 0
        next_split_set = OrderedDict{Type, OrderedSet{CellIndex}}()
        for (cellidx,celltype) in enumerate(typeof.(grid.cells))
            if !haskey(next_split_set, celltype)
                next_split_set[celltype] = OrderedSet{CellIndex}()
            end
            union!(next_split_set[celltype], [CellIndex(cellidx)])
        end
        volumetric_subdomains[""] = VolumetricSubdomainDesriptor(next_split_set)
    end

    surface_subdomains = OrderedDict{String, SurfaceSubdomainDesriptor}()
    for (name,facetset) in grid.facetsets
        next_split_set = OrderedDict{Type, OrderedSet{FacetIndex}}()
        for facetidx in facetset
            celltype = typeof(grid.cells[facetidx.idx[1]])
            if !haskey(next_split_set, celltype)
                next_split_set[celltype] = OrderedSet{FacetIndex}()
            end
            union!(next_split_set[celltype], [facetidx])
        end
        surface_subdomains[name] = SurfaceSubdomainDesriptor(next_split_set)
    end

    interface_subdomains = OrderedDict{String, InterfaceSubdomainDesriptor}()

    return SimpleMesh(
        grid,
        mfaces,
        medges,
        mvertices,
        number_of_cells_by_type,
        volumetric_subdomains,
        surface_subdomains,
        interface_subdomains,
    )
end

# Ferrite compat layer for the mesh
@inline Ferrite.getncells(mesh::SimpleMesh) = Ferrite.getncells(mesh.grid)
@inline Ferrite.getcells(mesh::SimpleMesh) = Ferrite.getcells(mesh.grid)
@inline Ferrite.getcells(mesh::SimpleMesh, v::Union{Int, Vector{Int}}) = Ferrite.getcells(mesh.grid, v)
@inline Ferrite.getcells(mesh::SimpleMesh, setname::String) = Ferrite.getcells(mesh.grid, setname)
@inline Ferrite.getcelltype(mesh::SimpleMesh) = Ferrite.getcelltype(mesh.grid)
@inline Ferrite.getcelltype(mesh::SimpleMesh, i::Int) = Ferrite.getcelltype(mesh.grid, i)
@inline Ferrite.getcellset(mesh::SimpleMesh, name::String) = Ferrite.getcellset(mesh.grid, name)
@inline Ferrite.getfacetset(mesh::SimpleMesh, name::String) = Ferrite.getfacetset(mesh.grid, name)
@inline Ferrite.getnodeset(mesh::SimpleMesh, name::String) = Ferrite.getnodeset(mesh.grid, name)

@inline Ferrite.getnnodes(mesh::SimpleMesh) = Ferrite.getnnodes(mesh.grid)
@inline Ferrite.nnodes_per_cell(mesh::SimpleMesh, i::Int=1) = Ferrite.nnodes_per_cell(mesh.grid, i)
@inline Ferrite.getnodes(mesh::SimpleMesh) = Ferrite.getnodes(mesh.grid)
@inline Ferrite.getnodes(mesh::SimpleMesh, v::Union{Int, Vector{Int}}) = Ferrite.getnodes(mesh.grid, v)
@inline Ferrite.getnodes(mesh::SimpleMesh, setname::String) = Ferrite.getnodes(mesh.grid, setname)

@inline Ferrite.VTKFile(filename::String, mesh::SimpleMesh; kwargs...) = VTKFile(filename, mesh.grid, kwargs...)

@inline Ferrite.get_coordinate_type(::SimpleMesh{sdim, <:Any, T}) where {sdim,T} = Vec{sdim,T} 

@inline Ferrite.CellIterator(mesh::SimpleMesh) = CellIterator(mesh.grid)
@inline Ferrite.CellIterator(mesh::SimpleMesh, set::Union{Nothing, AbstractSet{<:Integer}, AbstractVector{<:Integer}}, flags::UpdateFlags) = CellIterator(mesh.grid, set, flags)

# https://github.com/Ferrite-FEM/Ferrite.jl/pull/987
Ferrite.nfacets(cc::CellCache{<:Any, <:SimpleMesh}) = nfacets(cc.grid.grid.cells[cc.cellid[]])
