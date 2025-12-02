function hexahedralize(grid::Grid{3, Hexahedron})
    return grid
end

function hexahedralize(mesh::SimpleMesh{3, Hexahedron})
    return mesh
end

# TODO nonlinear version
function create_center_node(grid::AbstractGrid{dim}, cell::LinearCellGeometry) where {dim}
    center = zero(Vec{dim})
    vs = vertices(cell)
    for v ∈ vs
        node = getnodes(grid, v)
        center += node.x / length(vs)
    end
    return Node(center)
end

function create_edge_center_node(grid::AbstractGrid{dim}, cell::LinearCellGeometry, edge_idx::Int) where {dim}
    center = zero(Vec{dim})
    es = edges(cell)
    for v ∈ es[edge_idx]
        node = getnodes(grid, v)
        center += node.x / length(es[edge_idx])
    end
    return Node(center)
end

function create_face_center_node(grid::AbstractGrid{dim}, cell::LinearCellGeometry, face_idx::Int) where {dim}
    center = zero(Vec{dim})
    fs = faces(cell)
    for v ∈ fs[face_idx]
        node = getnodes(grid, v)
        center += node.x / length(fs[face_idx])
    end
    return Node(center)
end

function refine_element_uniform(mgrid::SimpleMesh, cell::Hexahedron, cell_idx::Int, global_edge_indices, global_face_indices)
    # Compute offsets
    new_edge_offset = num_nodes(mgrid)
    new_face_offset = num_edges(mgrid) + new_edge_offset
    # Compute indices
    vnids = vertices(cell)
    enids = new_edge_offset .+ global_edge_indices
    fnids = new_face_offset .+ global_face_indices
    cnid  = new_face_offset  + num_faces(mgrid) + cell_idx
    # Construct 8 finer cells
    return [
        Hexahedron((
            vnids[1], enids[1], fnids[1], enids[4],
            enids[9], fnids[2], cnid    , fnids[5] ,
        )),
        Hexahedron((
            enids[1], vnids[2] , enids[2], fnids[1],
            fnids[2], enids[10], fnids[3], cnid    ,
        )),
        Hexahedron((
            enids[4], fnids[1], enids[3], vnids[4],
            fnids[5], cnid    , fnids[4], enids[12],
        )),
        Hexahedron((
            fnids[1], enids[2], vnids[3] , enids[3],
            cnid    , fnids[3], enids[11], fnids[4],
        )),
        Hexahedron((
            enids[9], fnids[2], cnid    , fnids[5],
            vnids[5], enids[5], fnids[6], enids[8],
        )),
        Hexahedron((
            fnids[2], enids[10], fnids[3], cnid   ,
            enids[5], vnids[6] , enids[6], fnids[6],
        )),
        Hexahedron((
            fnids[5], cnid    , fnids[4], enids[12],
            enids[8], fnids[6], enids[7], vnids[8] ,
        )),
        Hexahedron((
            cnid    , fnids[3], enids[11], fnids[4],
            fnids[6], enids[6], vnids[7] , enids[7] 
        )),
    ]
end


function hexahedralize_local_face_transfer(cell::Hexahedron, offset::Int, faceid::Int)
    # TODO extract the topology table for this one, because we also need it for AMR
    if faceid == 1
        return OrderedSet([
            FacetIndex(offset+1,1),
            FacetIndex(offset+2,1),
            FacetIndex(offset+3,1),
            FacetIndex(offset+4,1),
        ])
    elseif faceid == 2
        return OrderedSet([
            FacetIndex(offset+1,2),
            FacetIndex(offset+2,2),
            FacetIndex(offset+5,2),
            FacetIndex(offset+6,2),
        ])
    elseif faceid == 3
        return OrderedSet([
            FacetIndex(offset+2,3),
            FacetIndex(offset+4,3),
            FacetIndex(offset+6,3),
            FacetIndex(offset+8,3),
        ])
    elseif faceid == 4
        return OrderedSet([
            FacetIndex(offset+3,4),
            FacetIndex(offset+4,4),
            FacetIndex(offset+7,4),
            FacetIndex(offset+8,4),
        ])
    elseif faceid == 5
        return OrderedSet([
            FacetIndex(offset+1,5),
            FacetIndex(offset+3,5),
            FacetIndex(offset+5,5),
            FacetIndex(offset+7,5),
        ])
    elseif faceid == 6
        return OrderedSet([
            FacetIndex(offset+5,6),
            FacetIndex(offset+6,6),
            FacetIndex(offset+7,6),
            FacetIndex(offset+8,6),
        ])
    else
        error("Invalid face $faceid for Hexahedron")
    end
end


# Hex into 8 hexahedra
hexahedralize_cell(mgrid::SimpleMesh, cell::Hexahedron, cell_idx::Int, global_edge_indices, global_face_indices) = refine_element_uniform(mgrid, cell, cell_idx, global_edge_indices, global_face_indices)

function hexahedralize_cell(mgrid::SimpleMesh, cell::Wedge, cell_idx::Int, global_edge_indices, global_face_indices)
    # Compute offsets
    new_edge_offset = num_nodes(mgrid)
    new_face_offset = new_edge_offset+num_edges(mgrid)
    # Compute indices
    vnids = vertices(cell)
    enids = new_edge_offset .+ global_edge_indices
    fnids = new_face_offset .+ global_face_indices
    cnid  = new_face_offset  + num_faces(mgrid) + cell_idx
    return [
        # Bottom 3
        Hexahedron((
            vnids[1], enids[1], fnids[1], enids[2],
            enids[3], fnids[2], cnid    , fnids[3],
        )),
        Hexahedron((
            enids[1], vnids[2], enids[4], fnids[1],
            fnids[2], enids[5], fnids[4], cnid    ,
        )),
        Hexahedron((
            fnids[1], enids[4], vnids[3], enids[2],
            cnid    , fnids[4], enids[6], fnids[3],
        )),
        # Top 3
        Hexahedron((
            enids[3], fnids[2], cnid    , fnids[3],
            vnids[4], enids[7], fnids[5], enids[8],
        )),
        Hexahedron((
            fnids[2], enids[5], fnids[4], cnid    ,
            enids[7], vnids[5], enids[9], fnids[5],
        )),
        Hexahedron((
            cnid    , fnids[4], enids[6], fnids[3],
            fnids[5], enids[9], vnids[6], enids[8],
        ))
    ]
end

function hexahedralize_local_face_transfer(cell::Wedge, offset::Int, faceid::Int)
    # TODO extract the topology table for this one, because we also need it for AMR
    if faceid == 1
        return OrderedSet([
            FacetIndex(offset+1,1),
            FacetIndex(offset+2,1),
            FacetIndex(offset+3,1),
        ])
    elseif faceid == 2
        return OrderedSet([
            FacetIndex(offset+1,2),
            FacetIndex(offset+2,2),
            FacetIndex(offset+4,2),
            FacetIndex(offset+5,2),
        ])
    elseif faceid == 3
        return OrderedSet([
            FacetIndex(offset+1,5),
            FacetIndex(offset+3,4),
            FacetIndex(offset+4,5),
            FacetIndex(offset+6,4),
        ])
    elseif faceid == 4
        return OrderedSet([
            FacetIndex(offset+2,3),
            FacetIndex(offset+3,3),
            FacetIndex(offset+5,3),
            FacetIndex(offset+6,3),
        ])
    elseif faceid == 5
        return OrderedSet([
            FacetIndex(offset+4,6),
            FacetIndex(offset+5,6),
            FacetIndex(offset+6,6),
        ])
    else
        error("Invalid face $faceid for Wedge")
    end
end


function hexahedralize_local_face_transfer(cell::Tetrahedron, offset::Int, faceid::Int)
    if faceid == 1
        return OrderedSet([
            FacetIndex(offset+1,1),
            FacetIndex(offset+2,1),
            FacetIndex(offset+3,1),
        ])
    elseif faceid == 2
        return OrderedSet([
            FacetIndex(offset+1,2),
            FacetIndex(offset+2,2),
            FacetIndex(offset+4,6),
        ])
    elseif faceid == 3
        return OrderedSet([
            FacetIndex(offset+2,3),
            FacetIndex(offset+3,3),
            FacetIndex(offset+4,3),
        ])
    elseif faceid == 4
        return OrderedSet([
            FacetIndex(offset+1,5),
            FacetIndex(offset+3,4),
            FacetIndex(offset+4,4),
        ])
    else
        error("Invalid face $faceid for Tetrahedron")
    end
end

function hexahedralize_cell(mgrid::SimpleMesh, cell::Tetrahedron, cell_idx::Int, global_edge_indices, global_face_indices)
    # Compute offsets
    new_edge_offset = num_nodes(mgrid)
    new_face_offset = new_edge_offset+num_edges(mgrid)
    # Compute indices
    vnids = vertices(cell)
    enids = new_edge_offset .+ global_edge_indices
    fnids = new_face_offset .+ global_face_indices
    cnid  = new_face_offset  + num_faces(mgrid) + cell_idx
    return [
        Hexahedron((
            vnids[1], enids[1], fnids[1], enids[3],
            enids[4], fnids[2], cnid    , fnids[4],
        )),
        Hexahedron((
            enids[1], vnids[2], enids[2], fnids[1],
            fnids[2], enids[5], fnids[3], cnid,
        )),
        Hexahedron((
            fnids[1], enids[2], vnids[3], enids[3],
            cnid    , fnids[3], enids[6], fnids[4],
        )),
        Hexahedron((
            cnid    , fnids[3], enids[6], fnids[4],
            fnids[2], enids[5], vnids[4], enids[4],
        ))
    ]
end

function uniform_refinement(grid::Grid{3})
    return _uniform_refinement(to_mesh(grid))
end

function uniform_refinement(mesh::SimpleMesh)
    return to_mesh(_uniform_refinement(mesh))
end

function _uniform_refinement(mgrid::SimpleMesh{3,C,T}) where {C,T}
    materialize_all_entities!(mgrid)
    grid = mgrid.grid

    cells = getcells(grid)

    nfacenods = length(mgrid.mfaces)
    new_face_nodes = Array{Node{3,T}}(undef, nfacenods) # We have to add 1 center node per face
    nedgenodes = length(mgrid.medges)
    new_edge_nodes = Array{Node{3,T}}(undef, nedgenodes) # We have to add 1 center node per edge
    ncellnodes = length(cells)
    new_cell_nodes = Array{Node{3,T}}(undef, ncellnodes) # We have to add 1 center node per volume

    new_cells = AbstractCell[]

    for (cellidx,cell) ∈ enumerate(cells)
        # Cell center node
        new_cell_nodes[cellidx] = create_center_node(grid, cell)
        global_edge_indices = global_edges(mgrid, cell)
        # Edge center nodes
        for (edgeidx,gei) ∈ enumerate(global_edge_indices)
            new_edge_nodes[gei] = create_edge_center_node(grid, cell, edgeidx)
        end
        # Facet center nodes
        global_face_indices = global_faces(mgrid, cell)
        for (faceidx,gfi) ∈ enumerate(global_face_indices)
            new_face_nodes[gfi] = create_face_center_node(grid, cell, faceidx)
        end
        append!(new_cells, refine_element_uniform(mgrid, cell, cellidx, global_edge_indices, global_face_indices))
    end
    # TODO boundary sets
    return Grid(new_cells, [grid.nodes; new_edge_nodes; new_face_nodes; new_cell_nodes])
end

function hexahedralize(grid::Grid{3})
    return _hexahedralize(to_mesh(grid))
end

function hexahedralize(mesh::SimpleMesh{3})
    grid = _hexahedralize(mesh)
    return to_mesh(grid)
end

function _hexahedralize(mgrid::SimpleMesh{3,<:Any,T}) where {T}
    materialize_edges!(mgrid)
    materialize_faces!(mgrid)
    (; grid) = mgrid

    cells = getcells(grid)

    nfacenods = length(mgrid.mfaces)
    new_face_nodes = Array{Node{3,T}}(undef, nfacenods) # We have to add 1 center node per face
    nedgenodes = length(mgrid.medges)
    new_edge_nodes = Array{Node{3,T}}(undef, nedgenodes) # We have to add 1 center node per edge
    ncellnodes = length(cells)
    new_cell_nodes = Array{Node{3,T}}(undef, ncellnodes) # We have to add 1 center node per volume

    new_cells = Hexahedron[]

    cell_offsets = Int[]
    for (cellidx,cell) ∈ enumerate(cells)
        # Cell center node
        new_cell_nodes[cellidx] = create_center_node(grid, cell)
        global_edge_indices = global_edges(mgrid, cell)
        # Edge center nodes
        for (edgeidx,gei) ∈ enumerate(global_edge_indices)
            new_edge_nodes[gei] = create_edge_center_node(grid, cell, edgeidx)
        end
        # Face center nodes
        global_face_indices = global_faces(mgrid, cell)
        for (faceidx,gfi) ∈ enumerate(global_face_indices)
            new_face_nodes[gfi] = create_face_center_node(grid, cell, faceidx)
        end
        append!(cell_offsets, length(new_cells))
        append!(new_cells, hexahedralize_cell(mgrid, cell, cellidx, global_edge_indices, global_face_indices))
    end

    # TODO boundary sets
    !isempty(grid.vertexsets) && warn("Vertexsets are not transfered to new mesh!")
    # !isempty(grid.edgesets) && warn("Edgesets are not transfered to new mesh!")

    new_cellsets = Dict{String, OrderedSet{Int}}()
    sizehint!(new_cellsets, length(grid.cellsets))
    for (setname, cellset) ∈ grid.cellsets
        new_cellsets[setname] = OrderedSet{Int}()
        n_new_cells = sum((cellidx == length(cell_offsets) ? length(new_cells) : cell_offsets[cellidx+1]) - cell_offsets[cellidx] for cellidx ∈ cellset)
        sizehint!(new_cellsets[setname], n_new_cells)
        for cellidx ∈ cellset
            new_cells_range = cell_offsets[cellidx] + 1 : (cellidx == length(cell_offsets) ? length(new_cells) : cell_offsets[cellidx+1])
            for new_cell in new_cells_range
                push!(new_cellsets[setname], new_cell)
            end
        end
    end

    new_facetsets = Dict{String, OrderedSet{FacetIndex}}()
    for (setname,facetset) ∈ grid.facetsets
        new_facetsets[setname] = OrderedSet{FacetIndex}()
        for (cellidx,lfi) ∈ facetset
            for f ∈ hexahedralize_local_face_transfer(grid.cells[cellidx], cell_offsets[cellidx], lfi)
                push!(new_facetsets[setname], f)
            end
        end
    end
    return Grid(new_cells, [grid.nodes; new_edge_nodes; new_face_nodes; new_cell_nodes]; cellsets = new_cellsets, facetsets=new_facetsets, nodesets=deepcopy(grid.nodesets))
end

function compute_minΔx(grid::Grid{dim, CT, DT}) where {dim, CT, DT}
    Δx = DT[DT(Inf) for _ ∈ 1:getncells(grid)]
    for (cell_idx,cell) ∈ enumerate(getcells(grid)) # todo cell iterator
        for (node_idx,node1) ∈ enumerate(cell.nodes) # todo node accessor
            for node2 ∈ cell.nodes[node_idx+1:end] # nodo node accessor
                Δx[cell_idx] = min(Δx[cell_idx], norm(grid.nodes[node1].x - grid.nodes[node2].x))
            end
        end
    end
    return Δx
end

function compute_maxΔx(grid::Grid{dim, CT, DT}) where {dim, CT, DT}
    Δx = DT[DT(0.0) for _ ∈ 1:getncells(grid)]
    for (cell_idx,cell) ∈ enumerate(getcells(grid)) # todo cell iterator
        for (node1, node2) ∈ edges(cell)
            Δx[cell_idx] = max(Δx[cell_idx], norm(grid.nodes[node1].x - grid.nodes[node2].x))
        end
    end
    return Δx
end

function compute_degeneracy(grid::Grid{dim, CT, DT}) where {dim, CT, DT}
    ratio = DT[DT(0.0) for _ ∈ 1:getncells(grid)]
    for (cell_idx,cell) ∈ enumerate(getcells(grid)) # todo cell iterator
        Δxmin = DT(Inf)
        Δxmax = zero(DT)
        for (node_idx,node1) ∈ enumerate(cell.nodes) # todo node accessor
            for node2 ∈ cell.nodes[node_idx+1:end] # nodo node accessor
                Δ = norm(grid.nodes[node1].x - grid.nodes[node2].x)
                Δxmin = min(Δxmin, Δ)
                Δxmax = max(Δxmax, Δ)
            end
        end
        ratio[cell_idx] = max(ratio[cell_idx], Δxmin/Δxmax)
    end
    return ratio
end

function load_voom2_elements(filename)
    elements = Vector{Ferrite.AbstractCell}()
    open(filename, "r") do file
        # First line has format number of elements as Int and 2 more integers
        line = strip(readline(file))
        ne = parse(Int64,split(line)[1])
        resize!(elements, ne)

        while !eof(file)
            line = parse.(Int64,split(strip(readline(file))))
            ei = line[1]
            etype = line[2]
            if etype == 8
                elements[ei] = Hexahedron(ntuple(i->line[i+2],8))
            elseif etype == 2
                elements[ei] = Line(ntuple(i->line[i+2],2))
            else
                @warn  "Unknown element type $etype. Skipping." maxlog=1
            end
        end
    end
    return elements
end

function load_voom2_nodes(filename)
    nodes = Vector{Ferrite.Node{3,Float64}}()
    open(filename, "r") do file
        # First line has format number of nodes as Int and 2 more integers
        line = strip(readline(file))
        nn = parse(Int64,split(line)[1])
        resize!(nodes, nn)

        while !eof(file)
            line = split(strip(readline(file)))
            ni = parse(Int64, line[1])
            nodes[ni] = Node(Vec(ntuple(i->parse(Float64,line[i+1]),3)))
        end
    end
    return nodes
end

function load_voom2_fsn(filename)
    # Big table
    f = Vector{Ferrite.Vec{3,Float64}}()
    s = Vector{Ferrite.Vec{3,Float64}}()
    n = Vector{Ferrite.Vec{3,Float64}}()
    open(filename, "r") do file
        while !eof(file)
            line = parse.(Float64,split(strip(readline(file))))
            push!(f, Vec((line[1], line[2], line[3])))
            push!(s, Vec((line[4], line[5], line[6])))
            push!(n, Vec((line[7], line[8], line[9])))
        end
    end
    return f,s,n
end

"""
    load_voom2_grid(filename)

Loader for the [voom2](https://github.com/luigiemp/voom2) legacy format.
"""
function load_voom2_grid(filename)
    nodes = load_voom2_nodes("$filename.nodes")
    elements = load_voom2_elements("$filename.ele")
    return Grid(elements, nodes)
end

"""
    load_mfem_grid(filename)

Loader for straight mfem meshes supporting v1.0.
"""
function load_mfem_grid(filename)
    @info "loading mfem mesh $filename"

    open(filename, "r") do file
        format = strip(readline(file))
        format != "MFEM mesh v1.0" && @error "Unsupported mesh format '$format'"

        # Parse spatial dimension
        while strip(readline(file)) != "dimension" && !eof(file)
        end
        eof(file) && @error "Missing 'dimension' specification"

        sdim = parse(Int64, strip(readline(file)))
        @info "sdim=$sdim"

        # Parse elements
        while strip(readline(file)) != "elements" && !eof(file)
        end
        eof(file) && @error "Missing 'elements' specification"

        ne = parse(Int64, strip(readline(file)))
        @info "number of elements=$ne"
        elements = Vector{Ferrite.AbstractCell}(undef, ne)
        domains = Dict{String,OrderedSet{Int}}()
        for ei in 1:ne
            line = parse.(Int64,split(strip(readline(file))))
            etype = line[2]

            line[3:end] .+= 1 # 0-based to 1-based index

            if etype == 1
                elements[ei] = Line(ntuple(i->line[i+2],2))
            elseif etype == 2
                elements[ei] = Triangle((line[4], line[5], line[3]))
            elseif etype == 3
                elements[ei] = Quadrilateral(ntuple(i->line[i+2],4))
            elseif etype == 4
                elements[ei] = Tetrahedron(ntuple(i->line[i+2],4))
            elseif etype == 5
                elements[ei] = Hexahedron(ntuple(i->line[i+2],8))
            elseif etype == 6
                elements[ei] = Wedge(ntuple(i->line[i+2],6))
            elseif etype == 7
                elements[ei] = Pyramid((line[3], line[4], line[6], line[5], line[7]))
            else
                @warn  "Unknown element type $etype. Skipping." maxlog=1
            end

            attr = line[1]
            if !haskey(domains, "$attr")
                domains["$attr"] = OrderedSet{Int}()
            end
            push!(domains["$attr"], ei)
        end

        # while strip(readline(file)) != "boundary" && !eof(file)
        # end
        # eof(file) && @error "Missing 'boundary' specification"
        @warn "Skipping parsing of boundary sets"

        while strip(readline(file)) != "vertices" && !eof(file)
        end
        eof(file) && @error "Missing 'vertices' specification"
        nv = parse(Int64, strip(readline(file)))
        @info "number of vertices=$nv"
        @assert sdim == parse(Int64, strip(readline(file))) # redundant space dim
        nodes = Vector{Ferrite.Node{sdim,Float64}}(undef, nv)

        for vi in 1:nv
            line = parse.(Float64,split(strip(readline(file))))
            nodes[vi] = Node(Vec(ntuple(i->line[i], sdim)))
        end

        return Grid(elements, nodes; cellsets=domains)
    end
end

function load_carp_elements(filename)
    elements = Vector{Ferrite.AbstractCell}()
    domains = Dict{String,OrderedSet{Int}}()

    open(filename, "r") do file
        # First line has format number of elements as Int and 2 more integers
        line = strip(readline(file))
        ne = parse(Int64,split(line)[1])
        resize!(elements, ne)

        for ei in 1:ne
            eof(file) && error("Premature end of input file")
            line = split(strip(readline(file)))

            etype = line[1]
            attr::Union{Nothing,Int64} = nothing
            if etype == "Ln"
                elements[ei] = Line(ntuple(i->parse(Int64,line[i+1])+1,2))
                if length(line) == 4
                    attr = parse(Int64,line[end])
                end
            elseif etype == "Tr"
                elements[ei] = Triangle(ntuple(i->parse(Int64,line[i+1])+1,3))
                if length(line) == 5
                    attr = parse(Int64,line[end])
                end
            elseif etype == "Qd"
                elements[ei] = Quadrilateral(ntuple(i->parse(Int64,line[i+1])+1,4))
                if length(line) == 6
                    attr = parse(Int64,line[end])
                end
            elseif etype == "Tt"
                elements[ei] = Tetrahedron(ntuple(i->parse(Int64,line[i+1])+1,4))
                if length(line) == 6 
                    attr = parse(Int64,line[end])
                end
            elseif etype == "Pr"
                elements[ei] = Wedge(ntuple(i->parse(Int64,line[i+1])+1,6))
                if length(line) == 8
                    attr = parse(Int64,line[end])
                end
            elseif etype == "Hx"
                elements[ei] = Hexahedron(ntuple(i->parse(Int64,line[i+1])+1,8))
                if length(line) == 10
                    attr = parse(Int64,line[end])
                end
            else
                @warn  "Unknown element type $etype. Skipping." maxlog=1
            end

            attr === nothing && continue # no attribute available
            if !haskey(domains, "$attr")
                domains["$attr"] = OrderedSet{Int}()
            end
            push!(domains["$attr"], ei)
        end
    end
    return elements, domains
end

function load_carp_nodes(filename)
    nodes = Vector{Ferrite.Node{3,Float64}}()
    open(filename, "r") do file
        line = strip(readline(file))
        nv = parse(Int64,split(line)[1])
        resize!(nodes, nv)

        for ni in 1:nv
            eof(file) && error("Premature end of input file")
            line = split(strip(readline(file)))
            nodes[ni] = Node(Vec(ntuple(i->parse(Float64,line[i]),3)))
        end
    end
    return nodes
end

"""
    load_carp_grid(filename)

Mesh format taken from https://carp.medunigraz.at/file_formats.html .
"""
function load_carp_grid(filename)
    elements, domains = load_carp_elements(filename * ".elem")
    nodes = load_carp_nodes(filename * ".pts")
    
    return Grid(elements, nodes; cellsets=domains)
end

function generate_element_for_face(::SimpleMesh, cell::Hexahedron, local_face)
    return Quadrilateral(Ferrite.faces(cell)[local_face])
end

function generate_element_for_face(::SimpleMesh, cell::Tetrahedron, local_face)
    return Triangle(Ferrite.faces(cell)[local_face])
end

function generate_element_for_face(::SimpleMesh, cell::Wedge, local_face)
    facenodes = Ferrite.faces(cell)[local_face]
    if length(facenodes) == 3
        return Triangle(facenodes)
    else
        return Quadrilateral(facenodes)
    end
end

function generate_reverse_index_map(d::Dict{Int,Int})
    inverse_indices = zeros(Int, length(d))
    generate_reverse_index_map!(inverse_indices, d)
    return inverse_indices
end

function generate_reverse_index_map!(inverse_indices, d::Dict{Int,Int})
    for i in keys(d)
        inverse_indices[d[i]] = i
    end
    return nothing
end

remove_unattached_nodes!(mesh::SimpleMesh) = remove_unattached_nodes!(mesh.grid)
function remove_unattached_nodes!(grid::Grid)
    next_nodeid = 1
    nodemap = Dict{Int,Int}()
    # Generate nodemap
    for cell in grid.cells
        for node in cell.nodes
            haskey(nodemap, node) && continue
            nodemap[node] = next_nodeid
            next_nodeid += 1
        end
    end
    # Regenerate nodes
    inverse_indices = generate_reverse_index_map(nodemap)
    grid.nodes = [grid.nodes[i] for i in inverse_indices]
    # Regenerate cells
    for cellid in 1:getncells(grid)
        cell     = grid.cells[cellid]
        celltype = typeof(cell)
        grid.cells[cellid] = celltype(ntuple(i->nodemap[cell.nodes[i]], length(cell.nodes)))
    end
    return nothing
end

function extract_boundary_faces(mesh::SimpleMesh{3}; subdomains = nothing)
    actual_subdomains = subdomains === nothing ? Dict("" => 1:getncells(mesh)) : subdomains
    materialize_faces!(mesh)
    # Cache for the cellid, local faceid pairs.
    # These are 0 if not assigned and -1 if assigned more than once.
    face_elements  = zeros(Int, num_faces(mesh))
    face_localfid  = zeros(Int, num_faces(mesh))
    for (_, subdomain) ∈ actual_subdomains
        for cellid ∈ subdomain
            cell = getcells(mesh, cellid)
            for (j,global_faceid) ∈ enumerate(global_faces(mesh, cell))
                if face_elements[global_faceid] == 0
                    face_elements[global_faceid] = cellid
                    face_localfid[global_faceid] = j
                else
                    face_elements[global_faceid] = -1
                    face_localfid[global_faceid] = -1
                end
            end
        end
    end

    return face_elements, face_localfid
end
function extract_outer_surface_mesh(mesh::SimpleMesh{3}; subdomains = nothing)
    materialize_faces!(mesh)
    actual_subdomains = subdomains === nothing ? Dict("" => 1:getncells(mesh)) : subdomains
    # Cache for the cellid, local faceid pairs.
    # These are 0 if not assigned and -1 if assigned more than once.
    face_elements, face_localfid = extract_boundary_faces(mesh; subdomains)

    # Sparse index map from face_elements to the index of the cells in the new surface grid
    next_cellid = 1
    face_element_index_map = Dict{Tuple{Int,Int},Int}()
    for (i,lfi) in enumerate(zip(face_elements,face_localfid))
        if lfi[1] > 0
            face_element_index_map[lfi] = next_cellid
            next_cellid += 1
        end
    end

    # facetsets -> cellsets
    cellsets = Dict{String, OrderedSet{Int}}()
    for (name,facetset) in mesh.grid.facetsets
        cellset = OrderedSet{Int}()
        for lfi in zip(face_elements, face_localfid)
            if lfi in facetset
                push!(cellset, face_element_index_map[lfi])
            end
        end
        if length(cellset) > 0
            cellsets[name] = cellset
        end
    end

    surface_grid = Grid(
        #if face_elements[i] > 0
        [generate_element_for_face(mesh, getcells(mesh, face_elements[i]), face_localfid[i]) for i in 1:num_faces(mesh) if face_elements[i] > 0],
        mesh.grid.nodes;
        cellsets 
    )

    remove_unattached_nodes!(surface_grid)

    return surface_grid
end

# TODO do this via the operator post-processing interface
function calculate_element_volume(cc, cellvalues)
    reinit!(cellvalues, cc)
    evol =0.0
    @inbounds for qp in QuadratureIterator(cellvalues)
        evol += 1 * getdetJdV(cellvalues, qp)
    end
    return evol
end
# function calculate_element_volume(cell, cellvalues_u, uₑ)
#     reinit!(cellvalues_u, cell)
#     evol = 0.0
#     @inbounds for qp in QuadratureIterator(cellvalues_u)
#         dΩ = getdetJdV(cellvalues_u, qp)
#         ∇u = function_gradient(cellvalues_u, qp, uₑ)
#         F = one(∇u) + ∇u
#         J = det(F)
#         evol += J * dΩ
#     end
#     return evol
# end
# function calculate_volume_deformed_mesh(w, dh::AbstractDofHandler, cellvalues_u)
#     evol::Float64 = 0.0;
#     @inbounds for cell in CellIterator(dh)
#         global_dofs = celldofs(cell)
#         nu = getnbasefunctions(cellvalues_u)
#         global_dofs_u = global_dofs[1:nu]
#         uₑ = w[global_dofs_u]
#         δevol = calculate_element_volume(cell, cellvalues_u, uₑ)
#         evol += δevol;
#     end
#     return evol
# end
function compute_center_of_mass(mesh::SimpleMesh{sdim}; domain_name = first(mesh.volumetric_subdomains.keys)) where sdim
    ∫x = zero(Vec{sdim,Float64})
    ∫1 = 0.0

    order = Ferrite.getorder(Ferrite.geometric_interpolation(getcells(mesh, 1)))
    ipc = LagrangeCollection{order}()
    dh = DofHandler(mesh)
    add_subdomain!(dh, domain_name, :u => ipc)
    close!(dh)

    #
    qrc = QuadratureRuleCollection(max(2Ferrite.getorder(ip)-1,2))
    for sdh in dh.subdofhandlers
        gip = geometric_interpolation(get_first_cell(sdh))
        ip = getinterpolation(ipc, sdh)
        qr = getquadraturerule(qrc, sdh)
        cv = CellValues(qr, ip, gip)
        for cell in CellIterator(sdh)
            reinit!(cv, cell)
            coords = getcoordinates(cell)
            for qp in QuadratureIterator(cv)
                dΩ = getdetJdV(cv, qp)
                x = spatial_coordinate(cv, qp, coords)
                ∫x += x * dΩ
                ∫1 += dΩ
            end
        end
    end

    return ∫x / ∫1
end
function compute_center_of_surface(grid::AbstractGrid{sdim}, name::String) where sdim
    facets = getfacetset(grid, name)
    volumes      = zeros(getncells(grid))
    centerpoints = zeros(Vec{sdim}, getncells(grid))
    ip = Ferrite.geometric_interpolation(getcells(grid, 1))
    qr = FacetQuadratureRule{Ferrite.getrefshape(ip)}(max(2Ferrite.getorder(ip)-1,2))
    cellvalues_vol = FacetValues(qr, ip)
    @inbounds for cc in FacetIterator(grid, facets)
        i = cellid(cc)
        volumes[i] = calculate_element_volume(cc, cellvalues_vol)

        # TODO use spatial_coordinate at reference element center
        nodes = getcells(grid, i).nodes
        for nodeid in nodes
            x = get_node_coordinate(getnodes(grid, nodeid))
            centerpoints[i] += get_node_coordinate(getnodes(grid, nodeid))
        end
        centerpoints[i] /= length(nodes)
    end

    center = sum(centerpoints .* volumes) ./ sum(volumes)
    return Vec((center[1], center[2], center[3]))
end
