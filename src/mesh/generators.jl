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
    generate_ideal_lv_mesh(num_elements_circumferential::Int, num_elements_radial::Int, num_elements_longitudinal::Int; inner_radius::T = Float64(0.7), outer_radius::T = Float64(1.0), longitudinal_upper::T = Float64(0.2), apex_inner::T = Float64(1.3), apex_outer::T = Float64(1.5), septum_fraction = 1//3)

Generate an idealized left ventricle as a truncated ellipsoid.
The number of elements per axis are controlled by the first three parameters.

`longitudinal_upper` truncates the ellipsoid at the base: the polar angle runs from the apex to
`(1 + longitudinal_upper) * π/2`, so `0.0` cuts at the equator and the default `0.2` keeps a fifth of
a quadrant above it. It is an angle here, unlike on the ring generators where the identically named
keyword is an axial extent.

The mesh carries the two internal facetsets `SRidgePost` and `SRidgeAnt` that
[`compute_lv_coordinate_system`](@ref) needs. An idealized ventricle has no right ventricle to
attach to, so the ridges are placed by convention: `SRidgePost` at `φ = 0` and `SRidgeAnt` such that
the septum between them covers `septum_fraction` of the circumference. They snap to the nearest
element interface, so the split is exact only when `num_elements_circumferential * septum_fraction`
is an integer.
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
    with_control_point::Bool = false,
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
    for θ ∈ longitudinal_angle[2:end],
        radius_percent ∈ radii_in_percent,
        φ ∈ circumferential_angle[1:(end-1)]

        push!(nodes, Node(point(θ, φ, radius_percent)))
    end

    # Generate all cells but the apex
    node_array = reshape(collect(1:n_nodes), (n_nodes_c, n_nodes_r, n_nodes_l))
    cells = with_control_point ? Union{Hexahedron, Wedge, Point}[] : Union{Hexahedron, Wedge}[]
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
    # the ventricle into a septum (circumferential index below `i_ant`) and a free wall. Each facet
    # is stored on its *septal* cell, which is the orientation the coordinate system reads the two
    # regions off -- hence facet 5, the low-angle side, on the first septal cell and facet 3, the
    # high-angle side, on the last one.
    i_ant                   = clamp(round(Int, num_elements_circumferential*septum_fraction), 1, num_elements_circumferential-1) + 1
    facetsets["SRidgePost"] = OrderedSet{FacetIndex}(FacetIndex(cl, 5) for cl in cell_array[1, :, :][:]);
    facetsets["SRidgeAnt"]  = OrderedSet{FacetIndex}(FacetIndex(cl, 3) for cl in cell_array[i_ant-1, :, :][:]);
    nodesets["Apex"]        = OrderedSet{Int}()
    nodesets["ApexInOut"]   = OrderedSet{Int}()

    # Add apex nodes
    push!(nodesets["ApexInOut"], length(nodes)+1)
    for radius_percent ∈ radii_in_percent
        push!(nodes, Node(point(0.0, 0.0, radius_percent)))
    end
    push!(nodesets["ApexInOut"], length(nodes))

    # Add apex cells
    for j ∈ 1:num_elements_radial, i ∈ 1:num_elements_circumferential
        i_next = (i == num_elements_circumferential) ? 1 : i + 1
        singular_index = length(nodes)-num_elements_radial+j-1
        push!(
            cells,
            Wedge((
                singular_index,
                node_array[i, j, 1],
                node_array[i_next, j, 1],
                singular_index+1,
                node_array[i, j+1, 1],
                node_array[i_next, j+1, 1],
            )),
        )
        j == 1 && push!(facetsets["Endocardium"], FacetIndex(length(cells), 1))
        j == num_elements_radial && push!(facetsets["Epicardium"], FacetIndex(length(cells), 5))
        j == num_elements_radial && push!(nodesets["Apex"], singular_index+1)
        i == 1 && push!(facetsets["SRidgePost"], FacetIndex(length(cells), 2))
        i == i_ant-1 && push!(facetsets["SRidgeAnt"], FacetIndex(length(cells), 3))
    end

    if with_control_point
        push!(nodes, Node(Vec((0.0, 0.0, 0.0))))
        push!(cells, Point(length(nodes)))
        cellsets = Dict([
            "myocardium" => OrderedSet(1:(length(cells)-1)),
            "lv-volume-control" => OrderedSet([length(cells)]),
        ])
    else
        cellsets = Dict(["myocardium" => OrderedSet(1:length(cells))])
    end

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
    i_ant =
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
        # [`compute_lv_coordinate_system`](@ref). Unlike the fan variant they stop at the O-grid
        # core: the core is a regular patch covering the apex, so no facet sheet inside it separates
        # the two sides, and the rotational coordinate is smeared over the core instead.
        "SRidgePost" =>
            OrderedSet{FacetIndex}(FacetIndex(cl, 5) for cl in cell_array[1, :, :][:]),
        "SRidgeAnt" =>
            OrderedSet{FacetIndex}(FacetIndex(cl, 3) for cl in cell_array[i_ant-1, :, :][:]),
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
        i == 1 && push!(facetsets["SRidgePost"], FacetIndex(length(cells), 5))
        i == i_ant-1 && push!(facetsets["SRidgeAnt"], FacetIndex(length(cells), 3))
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
