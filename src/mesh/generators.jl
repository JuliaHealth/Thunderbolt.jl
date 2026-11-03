"""
    generate_ring_mesh(num_elements_circumferential::Int, num_elements_radial::Int, num_elements_longitudinal::Int; inner_radius::T = Float64(0.75), outer_radius::T = Float64(1.0), longitudinal_lower::T = Float64(-0.2), longitudinal_upper::T = Float64(0.2), apicobasal_tilt::T=Float64(0.0)) where {T}

Generates an idealized full-hexahedral ring with linear ansatz. Geometrically it is the substraction of a small cylinder ``C_i`` of a large cylinder ``C_o``.
The number of elements for the cylindrical system can be controlled by the first three input parameters.
The remaining parameters control the spatial dimensions and the ring shape.

A ring has no right ventricle attached, so it carries no ridges and
[`compute_midmyocardial_section_coordinate_system`](@ref) falls back to the plain azimuth on it. The
internal facetset `RotationalSeam` at ``φ = 0`` says where that azimuth is allowed to jump.

`longitudinal_lower` and `longitudinal_upper` are the **axial extent** of the ring in ``z``, despite
the name they are not angles. They are also not a wall thickness: that is
`outer_radius - inner_radius`. Note this differs from what `longitudinal_upper` means on the
ventricular generators, where it is a basal truncation angle.
"""
function generate_ring_mesh(
    num_elements_circumferential::Int,
    num_elements_radial::Int,
    num_elements_longitudinal::Int;
    inner_radius::T = Float64(0.75),
    outer_radius::T = Float64(1.0),
    longitudinal_lower::T = Float64(-0.2),
    longitudinal_upper::T = Float64(0.2),
    apicobasal_tilt::T = Float64(0.0),
) where {T}
    # Generate a rectangle in cylindrical coordinates and transform coordinates back to carthesian.
    ne_tot = num_elements_circumferential*num_elements_radial*num_elements_longitudinal;
    n_nodes_c = num_elements_circumferential;
    n_nodes_r = num_elements_radial+1;
    n_nodes_l = num_elements_longitudinal+1;
    n_nodes = n_nodes_c * n_nodes_r * n_nodes_l;

    # Generate nodes
    circumferential_angle = range(0.0, stop = 2*π, length = n_nodes_c+1)
    radial_coords = range(inner_radius, stop = outer_radius, length = n_nodes_r)
    longitudinal_coordinate =
        range(longitudinal_upper, stop = longitudinal_lower, length = n_nodes_l)
    nodes = Node{3, T}[]
    for k = 1:n_nodes_l, j = 1:n_nodes_r, i = 1:n_nodes_c
        # cylindrical -> carthesian
        radius =
            radial_coords[j]-apicobasal_tilt*longitudinal_coordinate[k]/maximum(
                abs.(longitudinal_coordinate),
            )
        push!(
            nodes,
            Node((
                radius*cos(circumferential_angle[i]),
                radius*sin(circumferential_angle[i]),
                longitudinal_coordinate[k],
            )),
        )
    end

    # Generate cells
    node_array = reshape(collect(1:n_nodes), (n_nodes_c, n_nodes_r, n_nodes_l))
    cells = Hexahedron[]
    for k = 1:num_elements_longitudinal,
        j = 1:num_elements_radial,
        i = 1:num_elements_circumferential

        i_next = (i == num_elements_circumferential) ? 1 : i + 1
        push!(
            cells,
            Hexahedron((
                node_array[i, j, k],
                node_array[i_next, j, k],
                node_array[i_next, j+1, k],
                node_array[i, j+1, k],
                node_array[i, j, k+1],
                node_array[i_next, j, k+1],
                node_array[i_next, j+1, k+1],
                node_array[i, j+1, k+1],
            )),
        )
    end

    # Cell facets
    cell_array = reshape(
        collect(1:ne_tot),
        (num_elements_circumferential, num_elements_radial, num_elements_longitudinal),
    )
    boundary = FacetIndex[
        [FacetIndex(cl, 1) for cl in cell_array[:, :, 1][:]];
        [FacetIndex(cl, 2) for cl in cell_array[:, 1, :][:]];
        #[FacetIndex(cl, 3) for cl in cell_array[end,:,:][:]];
        [FacetIndex(cl, 4) for cl in cell_array[:, end, :][:]];
        #[FacetIndex(cl, 5) for cl in cell_array[1,:,:][:]];
        [FacetIndex(cl, 6) for cl in cell_array[:, :, end][:]]
    ]

    # Cell facet sets
    offset                   = 0
    facetsets                = Dict{String, OrderedSet{FacetIndex}}()
    facetsets["Myocardium"]  = OrderedSet{FacetIndex}(boundary[(1:length(cell_array[:, :, 1][:])) .+ offset]);
    offset                   += length(cell_array[:, :, 1][:])
    facetsets["Endocardium"] = OrderedSet{FacetIndex}(boundary[(1:length(cell_array[:, 1, :][:])) .+ offset]);
    offset                   += length(cell_array[:, 1, :][:])
    facetsets["Epicardium"]  = OrderedSet{FacetIndex}(boundary[(1:length(cell_array[:, end, :][:])) .+ offset]);
    offset                   += length(cell_array[:, end, :][:])
    facetsets["Base"]        = OrderedSet{FacetIndex}(boundary[(1:length(cell_array[:, :, end][:])) .+ offset]);
    offset                   += length(cell_array[:, :, end][:])
    # The ring closes on itself, so any azimuthal coordinate on it has to jump somewhere. This
    # internal sheet at φ = 0 is where it does -- see [`compute_midmyocardial_section_coordinate_system`](@ref).
    facetsets["RotationalSeam"] =
        OrderedSet{FacetIndex}(FacetIndex(cl, 5) for cl in cell_array[1, :, :][:]);

    nodesets = Dict{String, OrderedSet{Int}}()
    nodesets["MyocardialAnchor1"] = OrderedSet{Int}([node_array[1, 1, 1]])
    nodesets["MyocardialAnchor2"] = OrderedSet{Int}([node_array[1, end, 1]])
    nodesets["MyocardialAnchor3"] = OrderedSet{Int}([node_array[ceil(Int, 1+n_nodes_c/4), 1, 1]])
    nodesets["MyocardialAnchor4"] = OrderedSet{Int}([node_array[ceil(Int, 1+3*n_nodes_c/4), 1, 1]])

    return to_mesh(Grid(cells, nodes, facetsets = facetsets, nodesets = nodesets))
end


"""
    generate_open_ring_mesh(num_elements_circumferential::Int, num_elements_radial::Int, num_elements_longitudinal::Int, opening_angle::Float64; inner_radius::T = Float64(0.75), outer_radius::T = Float64(1.0), longitudinal_lower::T = Float64(-0.2), longitudinal_upper::T = Float64(0.2), apicobasal_tilt::T=Float64(0.0)) where {T}

Generates an idealized full-hexahedral ring with given opening angle and linear ansatz. Geometrically it is the substraction of a small cylinder ``C_i`` of a large cylinder ``C_o``.
The number of elements for the cylindrical system can be controlled by the first three input parameters.
The remaining parameters control the spatial dimensions and the ring shape.
The ring is opened along the Cartesian x-z plane.

`longitudinal_lower` and `longitudinal_upper` are the **axial extent** of the ring in ``z``, despite
the name they are not angles. They are also not a wall thickness: that is
`outer_radius - inner_radius`. Note this differs from what `longitudinal_upper` means on the
ventricular generators, where it is a basal truncation angle.
"""
function generate_open_ring_mesh(
    num_elements_circumferential::Int,
    num_elements_radial::Int,
    num_elements_longitudinal::Int,
    opening_angle::Float64;
    inner_radius::T = Float64(0.75),
    outer_radius::T = Float64(1.0),
    longitudinal_lower::T = Float64(-0.2),
    longitudinal_upper::T = Float64(0.2),
    apicobasal_tilt::T = Float64(0.0),
) where {T}
    # Generate a rectangle in cylindrical coordinates and transform coordinates back to carthesian.
    ne_tot = num_elements_circumferential*num_elements_radial*num_elements_longitudinal;
    n_nodes_c = num_elements_circumferential+1;
    n_nodes_r = num_elements_radial+1;
    n_nodes_l = num_elements_longitudinal+1;
    n_nodes = n_nodes_c * n_nodes_r * n_nodes_l;

    # Generate nodes
    circumferential_angle = range(opening_angle/2, stop = 2*π-opening_angle/2, length = n_nodes_c)
    radial_coords = range(inner_radius, stop = outer_radius, length = n_nodes_r)
    longitudinal_coordinate =
        range(longitudinal_upper, stop = longitudinal_lower, length = n_nodes_l)
    nodes = Node{3, T}[]
    for k = 1:n_nodes_l, j = 1:n_nodes_r, i = 1:n_nodes_c
        # cylindrical -> carthesian
        radius =
            radial_coords[j]-apicobasal_tilt*longitudinal_coordinate[k]/maximum(
                abs.(longitudinal_coordinate),
            )
        push!(
            nodes,
            Node((
                radius*cos(circumferential_angle[i]),
                radius*sin(circumferential_angle[i]),
                longitudinal_coordinate[k],
            )),
        )
    end

    # Generate cells
    node_array = reshape(collect(1:n_nodes), (n_nodes_c, n_nodes_r, n_nodes_l))
    cells = Hexahedron[]
    for k = 1:num_elements_longitudinal,
        j = 1:num_elements_radial,
        i = 1:num_elements_circumferential

        push!(
            cells,
            Hexahedron((
                node_array[i, j, k],
                node_array[i+1, j, k],
                node_array[i+1, j+1, k],
                node_array[i, j+1, k],
                node_array[i, j, k+1],
                node_array[i+1, j, k+1],
                node_array[i+1, j+1, k+1],
                node_array[i, j+1, k+1],
            )),
        )
    end

    # Cell facets
    cell_array = reshape(
        collect(1:ne_tot),
        (num_elements_circumferential, num_elements_radial, num_elements_longitudinal),
    )
    boundary = FacetIndex[
        [FacetIndex(cl, 1) for cl in cell_array[:, :, 1][:]];
        [FacetIndex(cl, 2) for cl in cell_array[:, 1, :][:]];
        [FacetIndex(cl, 3) for cl in cell_array[end, :, :][:]];
        [FacetIndex(cl, 4) for cl in cell_array[:, end, :][:]];
        [FacetIndex(cl, 5) for cl in cell_array[1, :, :][:]];
        [FacetIndex(cl, 6) for cl in cell_array[:, :, end][:]]
    ]

    # Cell facet sets
    offset = 0
    facetsets = Dict{String, OrderedSet{FacetIndex}}()

    facetsets["Myocardium"]  = OrderedSet{FacetIndex}(boundary[(1:length(cell_array[:, :, 1][:])) .+ offset]);
    offset                   += length(cell_array[:, :, 1][:])
    facetsets["Endocardium"] = OrderedSet{FacetIndex}(boundary[(1:length(cell_array[:, 1, :][:])) .+ offset]);
    offset                   += length(cell_array[:, 1, :][:])
    facetsets["Open1"]       = OrderedSet{FacetIndex}(boundary[(1:length(cell_array[end, :, :][:])) .+ offset]);
    offset                   += length(cell_array[end, :, :][:])
    facetsets["Epicardium"]  = OrderedSet{FacetIndex}(boundary[(1:length(cell_array[:, end, :][:])) .+ offset]);
    offset                   += length(cell_array[:, end, :][:])
    facetsets["Open2"]       = OrderedSet{FacetIndex}(boundary[(1:length(cell_array[1, :, :][:])) .+ offset]);
    offset                   += length(cell_array[1, :, :][:])
    facetsets["Base"]        = OrderedSet{FacetIndex}(boundary[(1:length(cell_array[:, :, end][:])) .+ offset]);
    offset                   += length(cell_array[:, :, end][:])

    nodesets = Dict{String, OrderedSet{Int}}()
    nodesets["MyocardialAnchor1"] = OrderedSet{Int}([node_array[1, 1, 1]])
    nodesets["MyocardialAnchor2"] = OrderedSet{Int}([node_array[1, end, 1]])
    nodesets["MyocardialAnchor3"] = OrderedSet{Int}([node_array[ceil(Int, 1+n_nodes_c/4), 1, 1]])
    nodesets["MyocardialAnchor4"] = OrderedSet{Int}([node_array[ceil(Int, 1+3*n_nodes_c/4), 1, 1]])

    return to_mesh(Grid(cells, nodes, facetsets = facetsets, nodesets = nodesets))
end


# const linear_index_to_local_index_table_hex27 = [1,9,2, 12,21,10, 4,11,3,  17,22,18, 25,27,23, 20,24,19, 5,13,6, 16,26,14, 8,15,7]
# const local_index_to_linear_index_table_hex27 = invperm(linear_index_to_local_index_table_hex27)
# const tensorproduct_index_to_local_index_table_hex27 = reshape(raw_index_to_local_index_table_hex27, (3,3,3))

"""
    generate_quadratic_ring_mesh(num_elements_circumferential::Int, num_elements_radial::Int, num_elements_longitudinal::Int; inner_radius::T = Float64(0.75), outer_radius::T = Float64(1.0), longitudinal_lower::T = Float64(-0.2), longitudinal_upper::T = Float64(0.2), apicobasal_tilt::T=Float64(0.0)) where {T}

Generates an idealized full-hexahedral ring with quadratic ansatz. Geometrically it is the substraction of a small cylinder ``C_i`` of a large cylinder ``C_o``.
The number of elements for the cylindrical system can be controlled by the first three input parameters.
The remaining parameters control the spatial dimensions and the ring shape.

`longitudinal_lower` and `longitudinal_upper` are the **axial extent** of the ring in ``z``, despite
the name they are not angles. They are also not a wall thickness: that is
`outer_radius - inner_radius`. Note this differs from what `longitudinal_upper` means on the
ventricular generators, where it is a basal truncation angle.
"""
function generate_quadratic_ring_mesh(
    num_elements_circumferential::Int,
    num_elements_radial::Int,
    num_elements_longitudinal::Int;
    inner_radius::T = Float64(0.75),
    outer_radius::T = Float64(1.0),
    longitudinal_lower::T = Float64(-0.2),
    longitudinal_upper::T = Float64(0.2),
    apicobasal_tilt::T = Float64(0.0),
) where {T}
    # Generate a rectangle in cylindrical coordinates and transform coordinates back to carthesian.
    ne_tot = num_elements_circumferential*num_elements_radial*num_elements_longitudinal;
    n_nodes_c = 2*num_elements_circumferential;
    n_nodes_r = 2*num_elements_radial+1;
    n_nodes_l = 2*num_elements_longitudinal+1;
    n_nodes = n_nodes_c * n_nodes_r * n_nodes_l;

    # Generate nodes
    circumferential_angle = range(0.0, stop = 2*π, length = n_nodes_c+1)
    radial_coords = range(inner_radius, stop = outer_radius, length = n_nodes_r)
    longitudinal_coordinate =
        range(longitudinal_upper, stop = longitudinal_lower, length = n_nodes_l)
    nodes = Node{3, T}[]
    for k = 1:n_nodes_l, j = 1:n_nodes_r, i = 1:n_nodes_c
        # cylindrical -> carthesian
        radius =
            radial_coords[j]-apicobasal_tilt*longitudinal_coordinate[k]/maximum(
                abs.(longitudinal_coordinate),
            )
        push!(
            nodes,
            Node((
                radius*cos(circumferential_angle[i]),
                radius*sin(circumferential_angle[i]),
                longitudinal_coordinate[k],
            )),
        )
    end

    # Generate cells
    node_array = reshape(collect(1:n_nodes), (n_nodes_c, n_nodes_r, n_nodes_l))
    cells = QuadraticHexahedron[]
    for k_ = 1:num_elements_longitudinal,
        j_ = 1:num_elements_radial,
        i_ = 1:num_elements_circumferential

        i_next = (i_ == num_elements_circumferential) ? 1 : 2*i_ + 1
        i = 2*i_-1
        j = 2*j_-1
        k = 2*k_-1
        push!(
            cells,
            QuadraticHexahedron((
                node_array[i+0, j+0, k+0],
                node_array[i_next, j+0, k+0],
                node_array[i_next, j+2, k+0],
                node_array[i+0, j+2, k+0], # Vertex loop back
                node_array[i+0, j+0, k+2],
                node_array[i_next, j+0, k+2],
                node_array[i_next, j+2, k+2],
                node_array[i+0, j+2, k+2],  # Vertex loop front
                node_array[i+1, j+0, k+0],
                node_array[i_next, j+1, k+0],
                node_array[i+1, j+2, k+0],
                node_array[i+0, j+1, k+0], # Edge loop back
                node_array[i+1, j+0, k+2],
                node_array[i_next, j+1, k+2],
                node_array[i+1, j+2, k+2],
                node_array[i+0, j+1, k+2], # Edge loop front
                node_array[i+0, j+0, k+1],
                node_array[i_next, j+0, k+1],
                node_array[i_next, j+2, k+1],
                node_array[i+0, j+2, k+1], # Edge loop center
                node_array[i+1, j+1, k+0],
                node_array[i+1, j+0, k+1],
                node_array[i_next, j+1, k+1],
                node_array[i+1, j+2, k+1],
                node_array[i+0, j+1, k+1],
                node_array[i+1, j+1, k+2], # Facet centers
                node_array[i+1, j+1, k+1],# Center
            )),
        )
    end

    # Cell facets
    cell_array = reshape(
        collect(1:ne_tot),
        (num_elements_circumferential, num_elements_radial, num_elements_longitudinal),
    )
    boundary = FacetIndex[
        [FacetIndex(cl, 1) for cl in cell_array[:, :, 1][:]];
        [FacetIndex(cl, 2) for cl in cell_array[:, 1, :][:]];
        #[FacetIndex(cl, 3) for cl in cell_array[end,:,:][:]];
        [FacetIndex(cl, 4) for cl in cell_array[:, end, :][:]];
        #[FacetIndex(cl, 5) for cl in cell_array[1,:,:][:]];
        [FacetIndex(cl, 6) for cl in cell_array[:, :, end][:]]
    ]

    # Cell facet sets
    offset                   = 0
    facetsets                = Dict{String, OrderedSet{FacetIndex}}()
    facetsets["Myocardium"]  = OrderedSet{FacetIndex}(boundary[(1:length(cell_array[:, :, 1][:])) .+ offset]);
    offset                   += length(cell_array[:, :, 1][:])
    facetsets["Endocardium"] = OrderedSet{FacetIndex}(boundary[(1:length(cell_array[:, 1, :][:])) .+ offset]);
    offset                   += length(cell_array[:, 1, :][:])
    facetsets["Epicardium"]  = OrderedSet{FacetIndex}(boundary[(1:length(cell_array[:, end, :][:])) .+ offset]);
    offset                   += length(cell_array[:, end, :][:])
    facetsets["Base"]        = OrderedSet{FacetIndex}(boundary[(1:length(cell_array[:, :, end][:])) .+ offset]);
    offset                   += length(cell_array[:, :, end][:])

    nodesets = Dict{String, OrderedSet{Int}}()
    nodesets["MyocardialAnchor1"] = OrderedSet{Int}([node_array[1, 1, 1]])
    nodesets["MyocardialAnchor2"] = OrderedSet{Int}([node_array[1, end, 1]])
    nodesets["MyocardialAnchor3"] = OrderedSet{Int}([node_array[ceil(Int, 1+n_nodes_c/4), 1, 1]])
    nodesets["MyocardialAnchor4"] = OrderedSet{Int}([node_array[ceil(Int, 1+3*n_nodes_c/4), 1, 1]])

    return to_mesh(Grid(cells, nodes, facetsets = facetsets, nodesets = nodesets))
end


"""
    generate_quadratic_open_ring_mesh(num_elements_circumferential::Int, num_elements_radial::Int, num_elements_longitudinal::Int, opening_angle::Float64; inner_radius::T = Float64(0.75), outer_radius::T = Float64(1.0), longitudinal_lower::T = Float64(-0.2), longitudinal_upper::T = Float64(0.2), apicobasal_tilt::T=Float64(0.0)) where {T}

Generates an idealized full-hexahedral ring with given opening angle and quadratic ansatz. Geometrically it is the substraction of a small cylinder ``C_i`` of a large cylinder ``C_o``.
The number of elements for the cylindrical system can be controlled by the first three input parameters.
The remaining parameters control the spatial dimensions and the ring shape.
The ring is opened along the Cartesian x-z plane.

`longitudinal_lower` and `longitudinal_upper` are the **axial extent** of the ring in ``z``, despite
the name they are not angles. They are also not a wall thickness: that is
`outer_radius - inner_radius`. Note this differs from what `longitudinal_upper` means on the
ventricular generators, where it is a basal truncation angle.
"""
function generate_quadratic_open_ring_mesh(
    num_elements_circumferential::Int,
    num_elements_radial::Int,
    num_elements_longitudinal::Int,
    opening_angle::Float64;
    inner_radius::T = Float64(0.75),
    outer_radius::T = Float64(1.0),
    longitudinal_lower::T = Float64(-0.2),
    longitudinal_upper::T = Float64(0.2),
    apicobasal_tilt::T = Float64(0.0),
) where {T}
    # Generate a rectangle in cylindrical coordinates and transform coordinates back to carthesian.
    ne_tot = num_elements_circumferential*num_elements_radial*num_elements_longitudinal;
    n_nodes_c = 2*num_elements_circumferential+1;
    n_nodes_r = 2*num_elements_radial+1;
    n_nodes_l = 2*num_elements_longitudinal+1;
    n_nodes = n_nodes_c * n_nodes_r * n_nodes_l;

    # Generate nodes
    circumferential_angle = range(opening_angle/2, stop = 2*π-opening_angle/2, length = n_nodes_c)
    radial_coords = range(inner_radius, stop = outer_radius, length = n_nodes_r)
    longitudinal_coordinate =
        range(longitudinal_upper, stop = longitudinal_lower, length = n_nodes_l)
    nodes = Node{3, T}[]
    for k = 1:n_nodes_l, j = 1:n_nodes_r, i = 1:n_nodes_c
        # cylindrical -> carthesian
        radius =
            radial_coords[j]-apicobasal_tilt*longitudinal_coordinate[k]/maximum(
                abs.(longitudinal_coordinate),
            )
        push!(
            nodes,
            Node((
                radius*cos(circumferential_angle[i]),
                radius*sin(circumferential_angle[i]),
                longitudinal_coordinate[k],
            )),
        )
    end

    # Generate cells
    node_array = reshape(collect(1:n_nodes), (n_nodes_c, n_nodes_r, n_nodes_l))
    cells = QuadraticHexahedron[]
    for k_ = 1:num_elements_longitudinal,
        j_ = 1:num_elements_radial,
        i_ = 1:num_elements_circumferential

        i = 2*i_-1
        j = 2*j_-1
        k = 2*k_-1
        push!(
            cells,
            QuadraticHexahedron((
                node_array[i+0, j+0, k+0],
                node_array[2*i_+1, j+0, k+0],
                node_array[2*i_+1, j+2, k+0],
                node_array[i+0, j+2, k+0], # Vertex loop back
                node_array[i+0, j+0, k+2],
                node_array[2*i_+1, j+0, k+2],
                node_array[2*i_+1, j+2, k+2],
                node_array[i+0, j+2, k+2],  # Vertex loop front
                node_array[i+1, j+0, k+0],
                node_array[2*i_+1, j+1, k+0],
                node_array[i+1, j+2, k+0],
                node_array[i+0, j+1, k+0], # Edge loop back
                node_array[i+1, j+0, k+2],
                node_array[2*i_+1, j+1, k+2],
                node_array[i+1, j+2, k+2],
                node_array[i+0, j+1, k+2], # Edge loop front
                node_array[i+0, j+0, k+1],
                node_array[2*i_+1, j+0, k+1],
                node_array[2*i_+1, j+2, k+1],
                node_array[i+0, j+2, k+1], # Edge loop center
                node_array[i+1, j+1, k+0],
                node_array[i+1, j+0, k+1],
                node_array[2*i_+1, j+1, k+1],
                node_array[i+1, j+2, k+1],
                node_array[i+0, j+1, k+1],
                node_array[i+1, j+1, k+2], # Facet centers
                node_array[i+1, j+1, k+1],# Center
            )),
        )
    end

    # Cell facets
    cell_array = reshape(
        collect(1:ne_tot),
        (num_elements_circumferential, num_elements_radial, num_elements_longitudinal),
    )
    boundary = FacetIndex[
        [FacetIndex(cl, 1) for cl in cell_array[:, :, 1][:]];
        [FacetIndex(cl, 2) for cl in cell_array[:, 1, :][:]];
        #[FacetIndex(cl, 3) for cl in cell_array[end,:,:][:]];
        [FacetIndex(cl, 4) for cl in cell_array[:, end, :][:]];
        #[FacetIndex(cl, 5) for cl in cell_array[1,:,:][:]];
        [FacetIndex(cl, 6) for cl in cell_array[:, :, end][:]]
    ]

    # Cell facet sets
    offset                   = 0
    facetsets                = Dict{String, OrderedSet{FacetIndex}}()
    facetsets["Myocardium"]  = OrderedSet{FacetIndex}(boundary[(1:length(cell_array[:, :, 1][:])) .+ offset]);
    offset                   += length(cell_array[:, :, 1][:])
    facetsets["Endocardium"] = OrderedSet{FacetIndex}(boundary[(1:length(cell_array[:, 1, :][:])) .+ offset]);
    offset                   += length(cell_array[:, 1, :][:])
    facetsets["Epicardium"]  = OrderedSet{FacetIndex}(boundary[(1:length(cell_array[:, end, :][:])) .+ offset]);
    offset                   += length(cell_array[:, end, :][:])
    facetsets["Base"]        = OrderedSet{FacetIndex}(boundary[(1:length(cell_array[:, :, end][:])) .+ offset]);
    offset                   += length(cell_array[:, :, end][:])

    nodesets = Dict{String, OrderedSet{Int}}()
    nodesets["MyocardialAnchor1"] = OrderedSet{Int}([node_array[1, 1, 1]])
    nodesets["MyocardialAnchor2"] = OrderedSet{Int}([node_array[1, end, 1]])
    nodesets["MyocardialAnchor3"] = OrderedSet{Int}([node_array[ceil(Int, 1+n_nodes_c/4), 1, 1]])
    nodesets["MyocardialAnchor4"] = OrderedSet{Int}([node_array[ceil(Int, 1+3*n_nodes_c/4), 1, 1]])

    return to_mesh(Grid(cells, nodes, facetsets = facetsets, nodesets = nodesets))
end

"""
Push the ring nodes of an ellipsoidal shell, in the order its `(circumferential, transmural,
longitudinal)` node array indexes them: circumferential fastest, longitudinal slowest. `point(l, φ,
rp)` places one node from its longitudinal parameter, azimuth and transmural fraction.
"""
function _shell_ring_nodes!(
    nodes,
    point,
    longitudinal_parameters,
    radii_in_percent,
    circumferential_angles,
)
    for l ∈ longitudinal_parameters, radius_percent ∈ radii_in_percent, φ ∈ circumferential_angles

        push!(nodes, Node(point(l, φ, radius_percent)))
    end
    return nodes
end

"""
Push the `nl` hexahedral layers between the rings of `node_array[circumferential, transmural,
longitudinal]`, longitudinal index slowest, so that cell `(i, j, k)` is the
`(k-1)*nr*nc + (j-1)*nc + i`-th of them. The circumferential index wraps.
"""
function _shell_hex_cells!(cells, node_array, nc::Int, nr::Int, nl::Int)
    for k = 1:nl, j = 1:nr, i = 1:nc
        i_next = (i == nc) ? 1 : i + 1
        push!(
            cells,
            Hexahedron((
                node_array[i, j, k],
                node_array[i_next, j, k],
                node_array[i_next, j+1, k],
                node_array[i, j+1, k],
                node_array[i, j, k+1],
                node_array[i_next, j, k+1],
                node_array[i_next, j+1, k+1],
                node_array[i, j+1, k+1],
            )),
        )
    end
    return cells
end

"""
Push the wedge fan closing a shell against the singular edge on its axis, transmural index slowest,
so that cell `(i, j)` is the `(j-1)*nc + i`-th of them.

`ring[i, j]` are the nodes of the ring the fan attaches to and `singular[j]` the `nr+1` nodes of the
edge, innermost first. Facet 1 of the innermost cells and facet 5 of the outermost ones are the two
free surfaces, facets 2 and 3 the sheets at the azimuths of `i` and `i+1`.

`flip` reverses the circumferential orientation, which is what a fan closing the shell beyond the
*end* of the longitudinal index -- rather than before its start -- needs to keep its Jacobian
positive.
"""
function _fan_wedge_cells!(cells, ring, singular, nc::Int, nr::Int, flip::Bool)
    for j = 1:nr, i = 1:nc
        i_next = (i == nc) ? 1 : i + 1
        a, b = flip ? (i_next, i) : (i, i_next)
        push!(
            cells,
            Wedge((singular[j], ring[a, j], ring[b, j], singular[j+1], ring[a, j+1], ring[b, j+1])),
        )
    end
    return cells
end

"""
Push the valvular plate that closes a circular orifice of radius `orifice_radius` in the plane
`z = plane_height`, and return `(cells, ventricular_face, opposite_face)` -- the plate's cell ids and
its two faces as facetsets.

The plate is a slab of thickness `thickness` centered on the plane. It attaches to the endocardial
rim ring `rim` -- the `nc` nodes of the orifice, at the azimuths of `circumferential_angles` --
without duplicating or collapsing nodes: an innermost wedge fan around the plate's center edge,
`np - 2` annular hexahedral layers, and an outermost ring of wedges whose triangular faces are
radial-vertical with the rim node as the third vertex, so the plate tapers to a knife edge exactly at
the orifice.

Both generators using this put the ventricle above the plane, so `ventricular_face` is the plate's
`+z` side.
"""
function _valvular_plate!(
    cells,
    nodes,
    rim,
    circumferential_angles,
    orifice_radius,
    plane_height,
    thickness,
    np::Int,
)
    nc     = length(circumferential_angles)
    top    = plane_height + thickness/2
    bottom = plane_height - thickness/2

    # The plate's interior rings, at the two plate faces, and its center edge. Its outermost ring is
    # the rim ring itself, which is why it is not built here.
    node_offset = length(nodes)
    for j = 1:(np-1), z ∈ (top, bottom), φ ∈ circumferential_angles
        radius = orifice_radius*j/np
        push!(nodes, Node(Vec((radius*cos(φ), radius*sin(φ), z))))
    end
    plate_array = reshape(collect((node_offset+1):length(nodes)), (nc, 2, np-1))
    plate_axis  = (length(nodes)+1):(length(nodes)+2)
    push!(nodes, Node(Vec((zero(top), zero(top), top))))
    push!(nodes, Node(Vec((zero(top), zero(top), bottom))))

    # From the center outwards. The cells stack from the ventricular face to the opposite one, which
    # is what the fan's flipped winding and the transposed node array of the annular layers are for
    # -- both builders stack along their last index.
    offset = length(cells)
    _fan_wedge_cells!(cells, view(plate_array,:,:,1), plate_axis, nc, 1, true)
    fan = collect((offset+1):length(cells))

    offset = length(cells)
    _shell_hex_cells!(cells, permutedims(plate_array, (1, 3, 2)), nc, np-2, 1)
    hex = collect((offset+1):length(cells))

    # The knife edge: each wedge stands on two radial-vertical triangles that run from the outermost
    # plate ring to a single rim node, so the plate ends on the shared ring without a node of its own
    # there and without any collapsed edge.
    offset = length(cells)
    for i = 1:nc
        i_next = (i == nc) ? 1 : i + 1
        push!(
            cells,
            Wedge((
                plate_array[i, 2, np-1],
                plate_array[i, 1, np-1],
                rim[i],
                plate_array[i_next, 2, np-1],
                plate_array[i_next, 1, np-1],
                rim[i_next],
            )),
        )
    end
    taper = collect((offset+1):length(cells))

    ventricular_face = OrderedSet{FacetIndex}(
        [
            [FacetIndex(cl, 1) for cl in fan]
            [FacetIndex(cl, 1) for cl in hex]
            [FacetIndex(cl, 4) for cl in taper]
        ],
    )
    opposite_face = OrderedSet{FacetIndex}(
        [
            [FacetIndex(cl, 5) for cl in fan]
            [FacetIndex(cl, 6) for cl in hex]
            [FacetIndex(cl, 3) for cl in taper]
        ],
    )
    return [fan; hex; taper], ventricular_face, opposite_face
end

"""
    generate_ideal_lv_mesh(num_elements_circumferential::Int, num_elements_radial::Int, num_elements_longitudinal::Int; inner_radius::T = Float64(0.7), outer_radius::T = Float64(1.0), longitudinal_upper::T = Float64(0.2), apex_inner::T = Float64(1.3), apex_outer::T = Float64(1.5), with_valvular_plane = false, valvular_plane_thickness = (outer_radius - inner_radius)/3, num_elements_valvular_plane = 3, septum_fraction = 1//3)

Generate an idealized left ventricle as a truncated ellipsoid.
The number of elements per axis are controlled by the first three parameters.

`longitudinal_upper` truncates the ellipsoid at the base: the polar angle runs from the apex to
`(1 + longitudinal_upper) * π/2`, so `0.0` cuts at the equator and the default `0.2` keeps a fifth of
a quadrant above it. It is an angle here, unlike on the ring generators where the identically named
keyword is an axial extent.

# Sets

Facetsets `"Endocardium"`, `"Epicardium"` and `"Base"`, the annular top face of the wall, plus the
two internal sheets `"SRidgePost"` and `"SRidgeAnt"` that [`compute_lv_coordinate_system`](@ref)
needs. An idealized ventricle has no right ventricle to attach to, so the ridges are placed by
convention: the septum covers `septum_fraction` of the circumference from `φ = 0`, with `SRidgeAnt`
at `φ = 0` and `SRidgePost` at its other end. The Cobiveco chart rises from the posterior ridge
across the free wall to the anterior one, so it increases with `φ`, which with the apex on `+z` is
left-handed about the long axis from the apex to the base -- counterclockwise seen from the apex, as
on an anatomical heart and as the azimuthal fallback runs. The ridges snap to the nearest element
interface, so the split is exact only when `num_elements_circumferential * septum_fraction` is an
integer.

Nodesets `"Apex"`, `"ApexInOut"` and `"MyocardialAnchor1"`-`"MyocardialAnchor4"`, four basal nodes
that pin the rigid body modes of a free-floating ventricle. Cellset `"myocardium"`.

# Valvular plane

`with_valvular_plane` closes the basal orifice with the same plate the two-chamber
[`generate_ideal_lh_mesh`](@ref) puts in the mitral orifice: cellset `"valvular-plane"`, a slab of
thickness `valvular_plane_thickness` centered on the basal plane, `num_elements_valvular_plane`
elements from its center edge to the knife edge it tapers to on the shared endocardial rim ring. The
plate is always *one* element thick; `valvular_plane_thickness` therefore also sets that element's
aspect ratio, and the default keeps it within a factor of about two of the radial element size. A
markedly thinner plate is a single trilinear element in bending and locks.

The plate is meshed rather than imposed, and a 3D-0D coupler puts the chamber pressure on its
ventricular face, so downstream has to give it a material stiff enough to carry that pressure
without bulging -- see the `cm03_3d0d-coupling` tutorial for the contract. A plate soft enough to be
merely "carried along" is the most compliant part of the chamber and absorbs the volume change that
the wall should be doing.

Its two faces are the facetsets `"LVValvularPlane"`, the ventricular one, and
`"ValvularPlaneOuter"`, and `"LVChamberSurface"` is the union of `"Endocardium"` with the former --
the *closed* surface a 3D-0D volume coupler integrates over. The endocardium alone is open at the
orifice, and its divergence-theorem volume is then neither direction-independent nor correct under
deformation, which is what closing it fixes for a base that moves.
"""
function generate_ideal_lv_mesh(
    num_elements_circumferential::Int,
    num_elements_radial::Int,
    num_elements_longitudinal::Int;
    inner_radius::T = Float64(0.7),
    outer_radius::T = Float64(1.0),
    longitudinal_upper::T = Float64(0.2),
    apex_inner::T = Float64(1.3),
    apex_outer::T = Float64(1.5),
    with_valvular_plane::Bool = false,
    valvular_plane_thickness::T = (outer_radius - inner_radius)/3,
    num_elements_valvular_plane::Int = 3,
    septum_fraction = 1//3,
) where {T}
    # Generate a rectangle in cylindrical coordinates and transform coordinates back to carthesian.
    ne_tot = num_elements_circumferential*num_elements_radial*num_elements_longitudinal;
    n_nodes_c = num_elements_circumferential;
    n_nodes_r = num_elements_radial+1;
    n_nodes_l = num_elements_longitudinal+1;
    n_nodes = n_nodes_c * n_nodes_r * n_nodes_l;

    # Generate nodes
    # Take a ring section of the heart and mark its circumferential coordinate by its angle
    circumferential_angle = range(0.0, stop = 2*π, length = n_nodes_c+1)
    # Untransformed radial coordinate of a ring section
    radii_in_percent = range(0.0, stop = 1.0, length = n_nodes_r)
    # z axis expressed as the angle between the apicobasal vector and the current layer from apex (0.0) to base ((1.0+longitudinal_upper)*π/2)
    longitudinal_angle = range(0, stop = (1.0+longitudinal_upper)*π/2, length = n_nodes_l+1)
    # The fan variant is the rotationally symmetric member of the ellipsoid family shared with
    # `generate_ideal_lv_mesh_hex`: no septal flattening, circular cross section.
    point(θ, φ, rp) = _ellipsoid_point(
        θ,
        φ,
        rp;
        inner_radius,
        outer_radius,
        apex_inner,
        apex_outer,
        septum_flatness = 0.0,
        axis_ratio = 1.0,
        eccentricity = 0.0,
    )

    # Rings from the one above the apex up to the base, circumferential index fastest.
    nodes = Node{3, T}[]
    _shell_ring_nodes!(
        nodes,
        point,
        longitudinal_angle[2:end],
        radii_in_percent,
        circumferential_angle[1:(end-1)],
    )

    # Generate all cells but the apex
    node_array = reshape(collect(1:n_nodes), (n_nodes_c, n_nodes_r, n_nodes_l))
    cells = Union{Hexahedron, Wedge}[]
    _shell_hex_cells!(
        cells,
        node_array,
        num_elements_circumferential,
        num_elements_radial,
        num_elements_longitudinal,
    )

    nodesets = Dict{String, OrderedSet{Int}}()
    nodesets["MyocardialAnchor1"] = OrderedSet{Int}([node_array[1, 1, end]])
    nodesets["MyocardialAnchor2"] = OrderedSet{Int}([node_array[1, end, end]])
    nodesets["MyocardialAnchor3"] = OrderedSet{Int}([node_array[ceil(Int, 1+n_nodes_c/4), 1, end]])
    nodesets["MyocardialAnchor4"] =
        OrderedSet{Int}([node_array[ceil(Int, 1+3*n_nodes_c/4), 1, end]])

    # Cell facets
    cell_array = reshape(
        collect(1:ne_tot),
        (num_elements_circumferential, num_elements_radial, num_elements_longitudinal),
    )
    boundary = FacetIndex[
        [FacetIndex(cl, 2) for cl in cell_array[:, 1, :][:]];
        [FacetIndex(cl, 4) for cl in cell_array[:, end, :][:]];
        [FacetIndex(cl, 6) for cl in cell_array[:, :, end][:]]
    ]

    # Cell facet sets
    offset                   = 0
    facetsets                = Dict{String, OrderedSet{FacetIndex}}()
    facetsets["Endocardium"] = OrderedSet{FacetIndex}(boundary[(1:length(cell_array[:, 1, :][:])) .+ offset]);
    offset                   += length(cell_array[:, 1, :][:])
    facetsets["Epicardium"]  = OrderedSet{FacetIndex}(boundary[(1:length(cell_array[:, end, :][:])) .+ offset]);
    offset                   += length(cell_array[:, end, :][:])
    facetsets["Base"]        = OrderedSet{FacetIndex}(boundary[(1:length(cell_array[:, :, end][:])) .+ offset]);
    offset                   += length(cell_array[:, :, end][:])
    # The two internal sheets that stand in for the right ventricular insertions. Both run from the
    # base down to the singular apex edge, where the azimuth stops existing, so together they cut
    # the ventricle into a septum (circumferential index below `i_post`) and a free wall. Each facet
    # is stored on its *septal* cell, which is the orientation the coordinate system reads the two
    # regions off -- hence facet 5, the low-angle side, on the first septal cell and facet 3, the
    # high-angle side, on the last one. The anterior ridge is the low-angle one, so that the chart
    # increases with φ: left-handed about the long axis from the apex at +z to the base.
    i_post                  = clamp(round(Int, num_elements_circumferential*septum_fraction), 1, num_elements_circumferential-1) + 1
    facetsets["SRidgeAnt"]  = OrderedSet{FacetIndex}(FacetIndex(cl, 5) for cl in cell_array[1, :, :][:]);
    facetsets["SRidgePost"] = OrderedSet{FacetIndex}(FacetIndex(cl, 3) for cl in cell_array[i_post-1, :, :][:]);
    nodesets["Apex"]        = OrderedSet{Int}()
    nodesets["ApexInOut"]   = OrderedSet{Int}()

    # Add apex nodes
    apex_nodes = (length(nodes)+1):(length(nodes)+n_nodes_r)
    push!(nodesets["ApexInOut"], first(apex_nodes))
    for radius_percent ∈ radii_in_percent
        push!(nodes, Node(point(0.0, 0.0, radius_percent)))
    end
    push!(nodesets["ApexInOut"], last(apex_nodes))

    # Add apex cells
    apex_offset = length(cells)
    _fan_wedge_cells!(
        cells,
        view(node_array,:,:,1),
        apex_nodes,
        num_elements_circumferential,
        num_elements_radial,
        false,
    )
    for j ∈ 1:num_elements_radial, i ∈ 1:num_elements_circumferential
        cl = apex_offset + (j-1)*num_elements_circumferential + i
        j == 1 && push!(facetsets["Endocardium"], FacetIndex(cl, 1))
        j == num_elements_radial && push!(facetsets["Epicardium"], FacetIndex(cl, 5))
        j == num_elements_radial && push!(nodesets["Apex"], apex_nodes[j+1])
        i == 1 && push!(facetsets["SRidgeAnt"], FacetIndex(cl, 2))
        i == i_post-1 && push!(facetsets["SRidgePost"], FacetIndex(cl, 3))
    end

    cellsets = Dict{String, OrderedSet{Int}}("myocardium" => OrderedSet(1:length(cells)))

    if with_valvular_plane
        num_elements_valvular_plane ≥ 2 || error(
            "`num_elements_valvular_plane` ($(num_elements_valvular_plane)) is below 2: the " *
            "valvular plate needs at least its center fan and its tapering outer ring.",
        )
        valvular_plane_thickness > 0.0 ||
            error("`valvular_plane_thickness` ($(valvular_plane_thickness)) has to be positive.")
        basal_angle = longitudinal_angle[end]
        plate_cells, plate_face, outer_face = _valvular_plate!(
            cells,
            nodes,
            view(node_array, :, 1, n_nodes_l),
            circumferential_angle[1:(end-1)],
            inner_radius*sin(basal_angle),
            apex_outer*cos(basal_angle),
            valvular_plane_thickness,
            num_elements_valvular_plane,
        )
        cellsets["valvular-plane"] = OrderedSet{Int}(plate_cells)
        facetsets["LVValvularPlane"] = plate_face
        facetsets["ValvularPlaneOuter"] = outer_face
        # The closed surface a chamber volume is measured over.
        facetsets["LVChamberSurface"] = union(facetsets["Endocardium"], plate_face)
    end

    return to_mesh(
        Grid(cells, nodes, nodesets = nodesets, facetsets = facetsets, cellsets = cellsets),
    )
end

"""
    generate_ideal_lh_mesh(num_elements_circumferential::Int, num_elements_radial::Int, num_elements_longitudinal::Int, num_elements_longitudinal_la::Int; inner_radius = 0.7, outer_radius = 1.0, longitudinal_upper = 0.2, apex_inner = 1.3, apex_outer = 1.5, la_cavity_radius = outer_radius, la_cavity_depth = inner_radius, la_wall_thickness = (outer_radius - inner_radius)/3, valvular_plane_thickness = la_wall_thickness, num_elements_valvular_plane = 3, septum_fraction = 1//3)

Generate an idealized left heart: the truncated ellipsoid of [`generate_ideal_lv_mesh`](@ref), a
thin-walled left atrium grown from the far side of its basal rim, and a valvular plate closing the
mitral orifice, after the idealized left-heart geometries used in the cardiac FSI literature
(Dedè et al. 2021, Viola et al. 2020).

The first three element counts and the `inner_radius`, `outer_radius`, `longitudinal_upper`,
`apex_inner`, `apex_outer` and `septum_fraction` keywords describe the ventricle exactly as they do
there. `num_elements_longitudinal_la` counts the atrial hexahedral layers between the rim and the
roof fan.

# Atrium

The atrial cavity is given by its equatorial semi-axis `la_cavity_radius` and by `la_cavity_depth`,
how far its roof sits below the annulus plane; `la_wall_thickness` offsets both to reach the
epicardial layer, and defaults to a third of the ventricular wall so the atrium is the
thinner-walled chamber. The roof closes with a wedge fan around a singular edge, like the apex.

The mitral annulus *is* the ventricular rim: the atrial shell reuses its nodes instead of meeting a
second surface there, so the two chambers form one continuous wall and the annulus plane is an
interior facet sheet. With `θ = 0` at the roof pole, the shell at transmural fraction `rp` is the
surface of revolution

    (r sin(θ) cos(φ), r sin(θ) sin(φ), zr + h (cos(θmax) - cos(θ))),   θ ∈ [0, θmax]

with equatorial semi-axis `r = la_cavity_radius + rp*la_wall_thickness`. It has to pass through the
ventricular rim circle of the *same* `rp`, which has radius
`ρ = (inner_radius*(1-rp) + outer_radius*rp)*sin(θb)` and sits at height `zr = apex_outer*cos(θb)`,
where `θb = (1 + longitudinal_upper)*π/2` is the ventricular truncation angle. That fixes the
truncation angle, and the requested pole depth
`d = la_cavity_depth + rp*la_wall_thickness` then fixes the polar semi-axis,

    θmax = π - asin(ρ/r),    h = d/(1 - cos(θmax))

The `π -` branch is what makes the atrium wider than the annulus it stands on, so that both chambers
bulge away from the shared ring and the lumen has a waist there. Deriving `h` from the depth rather
than prescribing it keeps the layers nested for any wall thickness.

Rim matching needs `r ≥ ρ` on every layer, which the generator checks: with a thin wall that forces
both atrial equatorial semi-axes above `outer_radius*sin(θb)`, i.e. the atrial cavity comes out
about as wide as the ventricular *outer* surface. That is the price of taking the whole ventricular
rim as the orifice, and it is also why the wall is thin only away from the annulus: at the annulus
the layers land on the ventricular rim circles, so the atrial wall tapers from the ventricular
thickness there to `la_wall_thickness` at the roof.

# Valvular plate

`"valvular-plane"` is a solid plate of thickness `valvular_plane_thickness`, centered on the annulus
plane, that closes the mitral orifice so that each chamber has a closed surface: without it the
endocardial sets are open at the orifice and the divergence-theorem chamber volume is neither
direction-independent nor correct under deformation. The plate is always *one* element thick, and
its default thickness is the atrial wall's, which keeps that element within a factor of about two of
the radial element size; a markedly thinner plate is a single trilinear element in bending and locks.

It is meshed rather than imposed, and a 3D-0D coupler puts each chamber's pressure on the face
bounding it, so downstream has to give the plate a material stiff enough to carry those pressures
without bulging -- see the `cm03_3d0d-coupling` tutorial for the contract. A plate soft enough to be
merely "carried along" is the most compliant part of the chambers and absorbs the volume change that
the walls should be doing.

It attaches to the *shared* endocardial rim ring, without duplicating or collapsing nodes: an
innermost wedge fan around the plate's center edge, `num_elements_valvular_plane - 2` annular
hexahedral layers, and an outermost ring of wedges whose triangular faces are radial-vertical with
the rim node as the third vertex, so the plate tapers to a knife edge exactly at the annulus.

# Sets

Cellsets `"ventricle"`, `"atrium"` and `"valvular-plane"`. A chamber pressure is declared by the
facet term that reads it, so there is no control cell to carry one.

Facetsets `"LVEndocardium"`, `"LAEndocardium"` and their union `"Endocardium"`, likewise
`"LVEpicardium"`, `"LAEpicardium"` and `"Epicardium"`. The endocardia stay anatomical, i.e. wall
only. `"LVValvularPlane"` and `"LAValvularPlane"` are the two faces of the plate, and the closed
chamber surfaces are their unions with the matching endocardium, `"LVChamberSurface"` and
`"LAChamberSurface"` -- those are what a 3D-0D volume coupler integrates over. Endocardia,
epicardium and the two plate faces are the entire boundary: the annulus plane is interior here, so
unlike the single-chamber generator this mesh carries no `"Base"`.

`"MitralAnnulus"` holds the annulus facets on their ventricular side, which is what
[`compute_lv_coordinate_system`](@ref) takes as `base_name` on this mesh; together with the
ventricle-only ridge sheets `"SRidgePost"`/`"SRidgeAnt"` and `"LVEndocardium"`/`"LVEpicardium"` that
call works on `subdomains = ["ventricle"]`.

Nodesets `"Apex"`, `"ApexInOut"` and `"MyocardialAnchor1"`-`"MyocardialAnchor4"` as on the ventricle
alone, plus `"MitralAnnulusRing"` for the shared rim nodes across the whole wall thickness.

At the defaults the atrial cavity holds ≈0.78 of the ventricular cavity volume — the wide orifice
makes a physiological ≈0.5 reachable only by flattening the dome; lower `la_cavity_depth` if the
ratio matters more than the shape.
"""
function generate_ideal_lh_mesh(
    num_elements_circumferential::Int,
    num_elements_radial::Int,
    num_elements_longitudinal::Int,
    num_elements_longitudinal_la::Int;
    inner_radius::T = Float64(0.7),
    outer_radius::T = Float64(1.0),
    longitudinal_upper::T = Float64(0.2),
    apex_inner::T = Float64(1.3),
    apex_outer::T = Float64(1.5),
    la_cavity_radius::T = outer_radius,
    la_cavity_depth::T = inner_radius,
    la_wall_thickness::T = (outer_radius - inner_radius)/3,
    valvular_plane_thickness::T = la_wall_thickness,
    num_elements_valvular_plane::Int = 3,
    septum_fraction = 1//3,
) where {T}
    nc        = num_elements_circumferential
    nr        = num_elements_radial
    n_nodes_r = num_elements_radial + 1
    n_lv      = num_elements_longitudinal
    n_la      = num_elements_longitudinal_la
    np        = num_elements_valvular_plane

    basal_angle = (1.0 + longitudinal_upper)*π/2
    rim_height = apex_outer*cos(basal_angle)
    rim_radius(rp) = (inner_radius*(1.0-rp) + outer_radius*rp)*sin(basal_angle)

    min(la_cavity_depth, la_wall_thickness, valvular_plane_thickness) > 0.0 || error(
        "`la_cavity_depth` ($(la_cavity_depth)), `la_wall_thickness` ($(la_wall_thickness)) and " *
        "`valvular_plane_thickness` ($(valvular_plane_thickness)) all have to be positive.",
    )
    np ≥ 2 || error(
        "`num_elements_valvular_plane` ($(np)) is below 2: the valvular plate needs at least its " *
        "center fan and its tapering outer ring.",
    )
    la_cavity_radius ≥ rim_radius(0.0) || error(
        "The atrial endocardium cannot reach the ventricular rim: `la_cavity_radius` " *
        "($(la_cavity_radius)) is below the endocardial rim radius $(rim_radius(0.0)).",
    )
    la_cavity_radius + la_wall_thickness ≥ rim_radius(1.0) || error(
        "The atrial epicardium cannot reach the ventricular rim: `la_cavity_radius + " *
        "la_wall_thickness` ($(la_cavity_radius + la_wall_thickness)) is below the epicardial " *
        "rim radius $(rim_radius(1.0)).",
    )

    circumferential_angle = range(0.0, stop = 2*π, length = nc+1)[1:(end-1)]
    radii_in_percent      = range(0.0, stop = 1.0, length = n_nodes_r)
    longitudinal_angle    = range(0.0, stop = basal_angle, length = n_lv+2)

    ventricle_point(θ, φ, rp) = _ellipsoid_point(
        θ,
        φ,
        rp;
        inner_radius,
        outer_radius,
        apex_inner,
        apex_outer,
        septum_flatness = 0.0,
        axis_ratio = 1.0,
        eccentricity = 0.0,
    )

    "Atrial wall at transmural fraction `rp`, `s = 0` at the roof pole and `s = 1` on the rim."
    function atrium_point(s, φ, rp)
        r = la_cavity_radius + rp*la_wall_thickness
        cos_θmax = -sqrt(max(0.0, 1.0 - (rim_radius(rp)/r)^2))
        h = (la_cavity_depth + rp*la_wall_thickness)/(1.0 - cos_θmax)
        θ = s*acos(cos_θmax)
        z = rim_height + h*(cos_θmax - cos(θ))
        return Vec((r*sin(θ)*cos(φ), r*sin(θ)*sin(φ), z))
    end

    # Ventricular rings, from the one above the apex down to the rim, then the atrial rings between
    # the rim and the roof. Ring 1 of the atrial array *is* the ventricular rim, so it is not built
    # twice -- that shared annulus is what joins the two shells.
    nodes = Node{3, T}[]
    _shell_ring_nodes!(
        nodes,
        ventricle_point,
        longitudinal_angle[2:end],
        radii_in_percent,
        circumferential_angle,
    )
    ventricle_array = reshape(collect(1:length(nodes)), (nc, n_nodes_r, n_lv+1))

    atrium_offset = length(nodes)
    _shell_ring_nodes!(
        nodes,
        atrium_point,
        [(n_la+1-m)/(n_la+1) for m = 1:n_la],
        radii_in_percent,
        circumferential_angle,
    )
    atrium_array = Array{Int}(undef, nc, n_nodes_r, n_la+1)
    atrium_array[:, :, 1] .= ventricle_array[:, :, end]
    atrium_array[:, :, 2:end] .=
        reshape(collect((atrium_offset+1):length(nodes)), (nc, n_nodes_r, n_la))

    apex_nodes = (length(nodes)+1):(length(nodes)+n_nodes_r)
    for radius_percent ∈ radii_in_percent
        push!(nodes, Node(ventricle_point(0.0, 0.0, radius_percent)))
    end
    roof_nodes = (length(nodes)+1):(length(nodes)+n_nodes_r)
    for radius_percent ∈ radii_in_percent
        push!(nodes, Node(atrium_point(0.0, 0.0, radius_percent)))
    end

    cells = Union{Hexahedron, Wedge}[]

    _shell_hex_cells!(cells, ventricle_array, nc, nr, n_lv)
    ventricle_hex = reshape(collect(1:length(cells)), (nc, nr, n_lv))

    offset = length(cells)
    _shell_hex_cells!(cells, atrium_array, nc, nr, n_la)
    atrium_hex = reshape(collect((offset+1):length(cells)), (nc, nr, n_la))

    offset = length(cells)
    _fan_wedge_cells!(cells, view(ventricle_array,:,:,1), apex_nodes, nc, nr, false)
    apex_fan = reshape(collect((offset+1):length(cells)), (nc, nr))

    # The roof fan sits beyond the *last* atrial ring, where the apex fan sits before the first
    # ventricular one, so its cells are wound the other way.
    offset = length(cells)
    _fan_wedge_cells!(cells, view(atrium_array,:,:,(n_la+1)), roof_nodes, nc, nr, true)
    roof_fan = reshape(collect((offset+1):length(cells)), (nc, nr))

    plate_cells, lv_plate_face, la_plate_face = _valvular_plate!(
        cells,
        nodes,
        view(ventricle_array, :, 1, size(ventricle_array, 3)),
        circumferential_angle,
        rim_radius(0.0),
        rim_height,
        valvular_plane_thickness,
        np,
    )

    facetsets = Dict{String, OrderedSet{FacetIndex}}()
    facetsets["LVEndocardium"] = OrderedSet{FacetIndex}(
        [
            [FacetIndex(cl, 2) for cl in ventricle_hex[:, 1, :][:]];
            [FacetIndex(cl, 1) for cl in apex_fan[:, 1][:]]
        ],
    )
    facetsets["LVEpicardium"] = OrderedSet{FacetIndex}(
        [
            [FacetIndex(cl, 4) for cl in ventricle_hex[:, end, :][:]];
            [FacetIndex(cl, 5) for cl in apex_fan[:, end][:]]
        ],
    )
    facetsets["LAEndocardium"] = OrderedSet{FacetIndex}(
        [
            [FacetIndex(cl, 2) for cl in atrium_hex[:, 1, :][:]];
            [FacetIndex(cl, 1) for cl in roof_fan[:, 1][:]]
        ],
    )
    facetsets["LAEpicardium"] = OrderedSet{FacetIndex}(
        [
            [FacetIndex(cl, 4) for cl in atrium_hex[:, end, :][:]];
            [FacetIndex(cl, 5) for cl in roof_fan[:, end][:]]
        ],
    )
    facetsets["Endocardium"] = union(facetsets["LVEndocardium"], facetsets["LAEndocardium"])
    facetsets["Epicardium"] = union(facetsets["LVEpicardium"], facetsets["LAEpicardium"])

    facetsets["LVValvularPlane"] = lv_plate_face
    facetsets["LAValvularPlane"] = la_plate_face
    # The closed surfaces a chamber volume is measured over. The endocardia alone are open at the
    # orifice, where the plate closes them.
    facetsets["LVChamberSurface"] = union(facetsets["LVEndocardium"], facetsets["LVValvularPlane"])
    facetsets["LAChamberSurface"] = union(facetsets["LAEndocardium"], facetsets["LAValvularPlane"])

    facetsets["MitralAnnulus"] =
        OrderedSet{FacetIndex}(FacetIndex(cl, 6) for cl in ventricle_hex[:, :, end][:])

    # As on the single ventricle: the ridges are placed by convention, the anterior one at φ = 0 so
    # that the chart increases with φ, they run from the rim down to the singular apex edge, and each
    # facet is stored on its septal cell.
    i_post = clamp(round(Int, nc*septum_fraction), 1, nc-1) + 1
    facetsets["SRidgeAnt"] = OrderedSet{FacetIndex}(
        [
            [FacetIndex(cl, 5) for cl in ventricle_hex[1, :, :][:]];
            [FacetIndex(cl, 2) for cl in apex_fan[1, :][:]]
        ],
    )
    facetsets["SRidgePost"] = OrderedSet{FacetIndex}(
        [
            [FacetIndex(cl, 3) for cl in ventricle_hex[i_post-1, :, :][:]];
            [FacetIndex(cl, 3) for cl in apex_fan[i_post-1, :][:]]
        ],
    )

    nodesets = Dict{String, OrderedSet{Int}}()
    nodesets["MyocardialAnchor1"] = OrderedSet{Int}([ventricle_array[1, 1, end]])
    nodesets["MyocardialAnchor2"] = OrderedSet{Int}([ventricle_array[1, end, end]])
    nodesets["MyocardialAnchor3"] = OrderedSet{Int}([ventricle_array[ceil(Int, 1+nc/4), 1, end]])
    nodesets["MyocardialAnchor4"] = OrderedSet{Int}([ventricle_array[ceil(Int, 1+3*nc/4), 1, end]])
    nodesets["Apex"] = OrderedSet{Int}([last(apex_nodes)])
    nodesets["ApexInOut"] = OrderedSet{Int}([first(apex_nodes), last(apex_nodes)])
    nodesets["MitralAnnulusRing"] = OrderedSet{Int}(ventricle_array[:, :, end][:])

    cellsets = Dict{String, OrderedSet{Int}}(
        "ventricle" => OrderedSet{Int}([ventricle_hex[:]; apex_fan[:]]),
        "atrium" => OrderedSet{Int}([atrium_hex[:]; roof_fan[:]]),
        "valvular-plane" => OrderedSet{Int}(plate_cells),
    )

    return to_mesh(
        Grid(cells, nodes, nodesets = nodesets, facetsets = facetsets, cellsets = cellsets),
    )
end

# Utils for the hex LV mesh
"""
Perimeter of the O-grid core, as `nc` points of the unit disk in which the apex
cap is parametrized. The points sit at the same angles as the nodes of the first
longitudinal ring, so the cells joining core to ring are radial.

The shape interpolates between the inscribed diamond (`roundness = 0`, giving a
perfectly square core but a strongly varying gap to the ring) and the circle
(`roundness = 1`, where the four corners flatten to 180° and the core
degenerates). The corners stay at the four cardinal angles either way.
"""
function _ogrid_perimeter(nc::Int, size, roundness)
    return map(0:(nc-1)) do k
        φ = 2π*k/nc
        ρ = (1 - roundness)/(abs(cos(φ)) + abs(sin(φ))) + roundness
        size*ρ*Vec((cos(φ), sin(φ)))
    end
end

"""
Lattice index `(a, b)` of the core node carrying perimeter position `k`, walking
the four sides of the `(m+1)×(m+1)` core counterclockwise from the corner at
angle 0.
"""
function _ogrid_perimeter_index(k::Int, m::Int)
    k = mod(k, 4m)
    k <= m && return (k+1, 1)
    k <= 2m && return (m+1, k-m+1)
    k <= 3m && return (3m-k+1, m+1)
    return (1, 4m-k+1)
end

"Core of the O-grid: transfinite interpolation of its four perimeter sides."
function _ogrid_core(nc::Int, size, roundness)
    m = nc ÷ 4
    P = _ogrid_perimeter(nc, size, roundness)
    at(k) = P[mod(k, nc)+1]
    lattice = Matrix{eltype(P)}(undef, m+1, m+1)
    for b = 1:(m+1), a = 1:(m+1)
        u = (a-1)/m;
        v = (b-1)/m
        south = at(a-1);
        north = at(3m-(a-1));
        west = at(-(b-1));
        east = at(m+b-1)
        lattice[a, b] =
            (1-v)*south + v*north + (1-u)*west + u*east -
            ((1-u)*(1-v)*at(0) + u*(1-v)*at(m) + (1-u)*v*at(3m) + u*v*at(2m))
    end
    return lattice
end


"""
Point of the idealized ventricular wall at longitudinal angle `θ` (0 at the apex), circumferential
angle `φ` and transmural fraction `rp` (0 endocardial, 1 epicardial). `septum_flatness`, `axis_ratio`
and `eccentricity` deform the truncated ellipsoid towards an anatomical shape; at
`septum_flatness = eccentricity = 0` and `axis_ratio = 1` it is the plain surface of revolution that
[`generate_ideal_lv_mesh`](@ref) uses.
"""
function _ellipsoid_point(
    θ,
    φ,
    rp;
    inner_radius,
    outer_radius,
    apex_inner,
    apex_outer,
    septum_flatness,
    axis_ratio,
    eccentricity,
)
    radius1 = (inner_radius*(1.0-rp) + outer_radius*rp)*axis_ratio
    radius2 = (inner_radius*(1.0-rp) + outer_radius*rp)/axis_ratio
    z = θ < π/2 ? (apex_inner*(1.0-rp) + apex_outer*rp)*cos(θ) : apex_outer*cos(θ)
    x = radius1*(cos(φ)*sin(θ)) + sin(septum_flatness*θ)*inner_radius
    y = radius2*sin(φ)*sin(θ) + eccentricity*x*(1.0-rp)
    x -= septum_flatness*0.125*y^2
    return Vec((x, y, z))
end

"""
Generate an idealized left ventricle as a truncated ellipsoid, all-hexahedral, with an O-grid cap
covering the apex instead of a fan of wedges around a singular edge.

Like [`generate_ideal_lv_mesh`](@ref) it carries the `SRidgePost` and `SRidgeAnt` facetsets, but
they stop at the O-grid core. The core is a regular patch across the apex, so no facet sheet inside
it continues the ridges, and the rotational coordinate of
[`compute_lv_coordinate_system`](@ref) degrades over the core -- roughly the apical eighth of the
ventricle. Use the fan variant where the coordinate has to be accurate right into the apex.
"""
function generate_ideal_lv_mesh_hex(
    num_elements_circumferential::Int,
    num_elements_radial::Int,
    num_elements_longitudinal::Int;
    inner_radius::T = Float64(0.7),
    outer_radius::T = Float64(1.0),
    longitudinal_upper::T = Float64(0.2),
    apex_inner::T = Float64(1.3),
    apex_outer::T = Float64(1.5),
    septum_flatness::T = Float64(0.6),
    axis_ratio::T = Float64(1.2),
    eccentricity::T = Float64(0.0),
    core_size = clamp(1 - 2π/num_elements_circumferential, 0.35, 0.9),
    core_roundness = 0.45,
    septum_fraction = 1//3,
) where {T}
    num_elements_circumferential % 4 == 0 || throw(
        ArgumentError(
            "the O-grid apex needs num_elements_circumferential divisible by 4, got $num_elements_circumferential",
        ),
    )
    m = num_elements_circumferential ÷ 4
    i_post =
        clamp(
            round(Int, num_elements_circumferential*septum_fraction),
            1,
            num_elements_circumferential-1,
        ) + 1

    n_nodes_c = num_elements_circumferential
    n_nodes_r = num_elements_radial + 1
    n_nodes_l = num_elements_longitudinal + 1

    circumferential_angle = range(0.0, stop = 2*π, length = n_nodes_c+1)
    radii_in_percent      = range(0.0, stop = 1.0, length = n_nodes_r)
    longitudinal_angle    = range(0, stop = (1.0+longitudinal_upper)*π/2, length = n_nodes_l+1)

    point(θ, φ, rp) = _ellipsoid_point(
        θ,
        φ,
        rp;
        inner_radius,
        outer_radius,
        apex_inner,
        apex_outer,
        septum_flatness,
        axis_ratio,
        eccentricity,
    )

    # Wall, identical to the fan variant: rings from the one above the apex up to
    # the base, circumferential index fastest.
    nodes = Node{3, T}[]
    for θ ∈ longitudinal_angle[2:end],
        radius_percent ∈ radii_in_percent,
        φ ∈ circumferential_angle[1:(end-1)]

        push!(nodes, Node(point(θ, φ, radius_percent)))
    end
    node_array =
        reshape(collect(1:(n_nodes_c*n_nodes_r*n_nodes_l)), (n_nodes_c, n_nodes_r, n_nodes_l))

    # One copy of the core per transmural shell. The cap map sends the unit disk
    # to the shell between the apex (ρ = 0) and the first longitudinal ring
    # (ρ = 1); it is smooth at the apex, so the core lands on a regular patch
    # there rather than on a singular point.
    θ_cap = longitudinal_angle[2]
    lattice = _ogrid_core(num_elements_circumferential, core_size, core_roundness)
    core_offset = length(nodes)
    for radius_percent ∈ radii_in_percent, b = 1:(m+1), a = 1:(m+1)
        X = lattice[a, b]
        push!(nodes, Node(point(norm(X)*θ_cap, atan(X[2], X[1]), radius_percent)))
    end
    core_array = reshape(collect(1:((m+1)^2*n_nodes_r)) .+ core_offset, (m+1, m+1, n_nodes_r))

    cells = Hexahedron[]
    for k = 1:num_elements_longitudinal,
        j = 1:num_elements_radial,
        i = 1:num_elements_circumferential

        i_next = (i == num_elements_circumferential) ? 1 : i + 1
        push!(
            cells,
            Hexahedron((
                node_array[i, j, k],
                node_array[i_next, j, k],
                node_array[i_next, j+1, k],
                node_array[i, j+1, k],
                node_array[i, j, k+1],
                node_array[i_next, j, k+1],
                node_array[i_next, j+1, k+1],
                node_array[i, j+1, k+1],
            )),
        )
    end

    ne_wall = num_elements_circumferential*num_elements_radial*num_elements_longitudinal
    cell_array = reshape(
        collect(1:ne_wall),
        (num_elements_circumferential, num_elements_radial, num_elements_longitudinal),
    )
    facetsets = Dict{String, OrderedSet{FacetIndex}}(
        "Endocardium" =>
            OrderedSet{FacetIndex}(FacetIndex(cl, 2) for cl in cell_array[:, 1, :][:]),
        "Epicardium" =>
            OrderedSet{FacetIndex}(FacetIndex(cl, 4) for cl in cell_array[:, end, :][:]),
        "Base" => OrderedSet{FacetIndex}(FacetIndex(cl, 6) for cl in cell_array[:, :, end][:]),
        # The two internal sheets standing in for the right ventricular insertions, see
        # [`compute_lv_coordinate_system`](@ref), placed as on the fan variant. Unlike the fan
        # variant they stop at the O-grid core: the core is a regular patch covering the apex, so no
        # facet sheet inside it separates the two sides, and the rotational coordinate is smeared
        # over the core instead.
        "SRidgeAnt" =>
            OrderedSet{FacetIndex}(FacetIndex(cl, 5) for cl in cell_array[1, :, :][:]),
        "SRidgePost" =>
            OrderedSet{FacetIndex}(FacetIndex(cl, 3) for cl in cell_array[i_post-1, :, :][:]),
    )

    # Apex cells are extruded transmurally rather than longitudinally, so their
    # endo- and epicardial facets are the bottom and top ones.
    for j = 1:num_elements_radial, i = 1:num_elements_circumferential
        i_next = (i == num_elements_circumferential) ? 1 : i + 1
        a, b   = _ogrid_perimeter_index(i-1, m)
        an, bn = _ogrid_perimeter_index(i, m)
        push!(
            cells,
            Hexahedron((
                node_array[i, j, 1],
                node_array[i_next, j, 1],
                core_array[an, bn, j],
                core_array[a, b, j],
                node_array[i, j+1, 1],
                node_array[i_next, j+1, 1],
                core_array[an, bn, j+1],
                core_array[a, b, j+1],
            )),
        )
        j == 1 && push!(facetsets["Endocardium"], FacetIndex(length(cells), 1))
        j == num_elements_radial && push!(facetsets["Epicardium"], FacetIndex(length(cells), 6))
        i == 1 && push!(facetsets["SRidgeAnt"], FacetIndex(length(cells), 5))
        i == i_post-1 && push!(facetsets["SRidgePost"], FacetIndex(length(cells), 3))
    end
    for j = 1:num_elements_radial, b = 1:m, a = 1:m
        push!(
            cells,
            Hexahedron((
                core_array[a, b, j],
                core_array[a+1, b, j],
                core_array[a+1, b+1, j],
                core_array[a, b+1, j],
                core_array[a, b, j+1],
                core_array[a+1, b, j+1],
                core_array[a+1, b+1, j+1],
                core_array[a, b+1, j+1],
            )),
        )
        j == 1 && push!(facetsets["Endocardium"], FacetIndex(length(cells), 1))
        j == num_elements_radial && push!(facetsets["Epicardium"], FacetIndex(length(cells), 6))
    end

    ca, cb = Tuple(argmin(norm.(lattice)))
    nodesets = Dict{String, OrderedSet{Int}}(
        "MyocardialAnchor1" => OrderedSet{Int}([node_array[1, 1, end]]),
        "MyocardialAnchor2" => OrderedSet{Int}([node_array[1, end, end]]),
        "MyocardialAnchor3" => OrderedSet{Int}([node_array[ceil(Int, 1+n_nodes_c/4), 1, end]]),
        "MyocardialAnchor4" =>
            OrderedSet{Int}([node_array[ceil(Int, 1+3*n_nodes_c/4), 1, end]]),
        "Apex" => OrderedSet{Int}([core_array[ca, cb, end]]),
        "ApexInOut" => OrderedSet{Int}([core_array[ca, cb, 1], core_array[ca, cb, end]]),
    )

    return to_mesh(Grid(cells, nodes, nodesets = nodesets, facetsets = facetsets))
end

"""
    generate_mesh(args...) -> SimpleMesh

`Ferrite.generate_grid(args...)` passed through [`to_mesh`](@ref). Accepts and forwards whatever
`generate_grid` accepts.
"""
generate_mesh(args...) = to_mesh(generate_grid(args...))

function generate_simple_disc_grid(::Type{Quadrilateral}, n; radius = 1.0)
    nnodes = 2n + 1
    θ = deg2rad(360/2n)

    nodepos = Vec((0.0, radius))
    nodes = [rotate(nodepos, θ*i) for i ∈ 0:(2n-1)]
    push!(nodes, Vec((0.0, 0.0)))

    elements = Quadrilateral[
        Quadrilateral((2i-1==0 ? nnodes-1 : 2i-1, 2i, 2i+1 == nnodes ? 1 : 2i+1, nnodes)) for
        i ∈ 1:n
    ]

    facetsets = Dict(
        "boundary" =>
            OrderedSet([FacetIndex(i, 1) for i ∈ 1:n]) ∪ OrderedSet([FacetIndex(i, 2) for i ∈ 1:n]),
    )

    return Grid(elements, Node.(nodes); facetsets = facetsets)
end

generate_simple_disc_mesh(::Type{Quadrilateral}, n; radius = 1.0) =
    to_mesh(generate_simple_disc_grid(Quadrilateral, n; radius))
