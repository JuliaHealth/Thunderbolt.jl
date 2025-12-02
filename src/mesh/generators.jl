"""
    generate_ring_mesh(num_elements_circumferential::Int, num_elements_radial::Int, num_elements_logintudinal::Int; inner_radius::T = Float64(0.75), outer_radius::T = Float64(1.0), longitudinal_lower::T = Float64(-0.2), longitudinal_upper::T = Float64(0.2), apicobasal_tilt::T=Float64(0.0)) where {T}

Generates an idealized full-hexahedral ring with linear ansatz. Geometrically it is the substraction of a small cylinder ``C_i`` of a large cylinder ``C_o``.
The number of elements for the cylindrical system can be controlled by the first three input parameters.
The remaining parameters control the spatial dimensions and the ring shape.
"""
function generate_ring_mesh(num_elements_circumferential::Int, num_elements_radial::Int, num_elements_logintudinal::Int; inner_radius::T = Float64(0.75), outer_radius::T = Float64(1.0), longitudinal_lower::T = Float64(-0.2), longitudinal_upper::T = Float64(0.2), apicobasal_tilt::T=Float64(0.0)) where {T}
    # Generate a rectangle in cylindrical coordinates and transform coordinates back to carthesian.
    ne_tot = num_elements_circumferential*num_elements_radial*num_elements_logintudinal;
    n_nodes_c = num_elements_circumferential;
    n_nodes_r = num_elements_radial+1;
    n_nodes_l = num_elements_logintudinal+1;
    n_nodes = n_nodes_c * n_nodes_r * n_nodes_l;

    # Generate nodes
    circumferential_angle = range(0.0, stop=2*π, length=n_nodes_c+1)
    radial_coords = range(inner_radius, stop=outer_radius, length=n_nodes_r)
    longitudinal_angle = range(longitudinal_upper, stop=longitudinal_lower, length=n_nodes_l)
    nodes = Node{3,T}[]
    for k in 1:n_nodes_l, j in 1:n_nodes_r, i in 1:n_nodes_c
        # cylindrical -> carthesian
        radius = radial_coords[j]-apicobasal_tilt*longitudinal_angle[k]/maximum(abs.(longitudinal_angle))
        push!(nodes, Node((radius*cos(circumferential_angle[i]), radius*sin(circumferential_angle[i]), longitudinal_angle[k])))
    end

    # Generate cells
    node_array = reshape(collect(1:n_nodes), (n_nodes_c, n_nodes_r, n_nodes_l))
    cells = Hexahedron[]
    for k in 1:num_elements_logintudinal, j in 1:num_elements_radial, i in 1:num_elements_circumferential
        i_next = (i == num_elements_circumferential) ? 1 : i + 1
        push!(cells, Hexahedron((node_array[i,j,k], node_array[i_next,j,k], node_array[i_next,j+1,k], node_array[i,j+1,k],
                                 node_array[i,j,k+1], node_array[i_next,j,k+1], node_array[i_next,j+1,k+1], node_array[i,j+1,k+1])))
    end

    # Cell facets
    cell_array = reshape(collect(1:ne_tot),(num_elements_circumferential, num_elements_radial, num_elements_logintudinal))
    boundary = FacetIndex[[FacetIndex(cl, 1) for cl in cell_array[:,:,1][:]];
                            [FacetIndex(cl, 2) for cl in cell_array[:,1,:][:]];
                            #[FacetIndex(cl, 3) for cl in cell_array[end,:,:][:]];
                            [FacetIndex(cl, 4) for cl in cell_array[:,end,:][:]];
                            #[FacetIndex(cl, 5) for cl in cell_array[1,:,:][:]];
                            [FacetIndex(cl, 6) for cl in cell_array[:,:,end][:]]]

    # Cell facet sets
    offset = 0
    facetsets = Dict{String,OrderedSet{FacetIndex}}()
    facetsets["Myocardium"]  = OrderedSet{FacetIndex}(boundary[(1:length(cell_array[:,:,1][:]))   .+ offset]); offset += length(cell_array[:,:,1][:])
    facetsets["Endocardium"] = OrderedSet{FacetIndex}(boundary[(1:length(cell_array[:,1,:][:]))   .+ offset]); offset += length(cell_array[:,1,:][:])
    facetsets["Epicardium"]  = OrderedSet{FacetIndex}(boundary[(1:length(cell_array[:,end,:][:])) .+ offset]); offset += length(cell_array[:,end,:][:])
    facetsets["Base"]        = OrderedSet{FacetIndex}(boundary[(1:length(cell_array[:,:,end][:])) .+ offset]); offset += length(cell_array[:,:,end][:])

    nodesets = Dict{String,OrderedSet{Int}}()
    nodesets["MyocardialAnchor1"] = OrderedSet{Int}([node_array[1,1,1]])
    nodesets["MyocardialAnchor2"] = OrderedSet{Int}([node_array[1,end,1]])
    nodesets["MyocardialAnchor3"] = OrderedSet{Int}([node_array[ceil(Int,1+n_nodes_c/4),1,1]])
    nodesets["MyocardialAnchor4"] = OrderedSet{Int}([node_array[ceil(Int,1+3*n_nodes_c/4),1,1]])

    return to_mesh(Grid(cells, nodes, facetsets=facetsets, nodesets=nodesets))
end


"""
    generate_open_ring_mesh(num_elements_circumferential::Int, num_elements_radial::Int, num_elements_logintudinal::Int, opening_angle::Float64; inner_radius::T = Float64(0.75), outer_radius::T = Float64(1.0), longitudinal_lower::T = Float64(-0.2), longitudinal_upper::T = Float64(0.2), apicobasal_tilt::T=Float64(0.0)) where {T}

Generates an idealized full-hexahedral ring with given opening angle and linear ansatz. Geometrically it is the substraction of a small cylinder ``C_i`` of a large cylinder ``C_o``.
The number of elements for the cylindrical system can be controlled by the first three input parameters.
The remaining parameters control the spatial dimensions and the ring shape.
The ring is opened along the Cartesian x-z plane.
"""
function generate_open_ring_mesh(num_elements_circumferential::Int, num_elements_radial::Int, num_elements_logintudinal::Int, opening_angle::Float64; inner_radius::T = Float64(0.75), outer_radius::T = Float64(1.0), longitudinal_lower::T = Float64(-0.2), longitudinal_upper::T = Float64(0.2), apicobasal_tilt::T=Float64(0.0)) where {T}
    # Generate a rectangle in cylindrical coordinates and transform coordinates back to carthesian.
    ne_tot = num_elements_circumferential*num_elements_radial*num_elements_logintudinal;
    n_nodes_c = num_elements_circumferential+1;
    n_nodes_r = num_elements_radial+1;
    n_nodes_l = num_elements_logintudinal+1;
    n_nodes = n_nodes_c * n_nodes_r * n_nodes_l;

    # Generate nodes
    circumferential_angle = range(opening_angle/2, stop=2*π-opening_angle/2, length=n_nodes_c)
    radial_coords = range(inner_radius, stop=outer_radius, length=n_nodes_r)
    longitudinal_angle = range(longitudinal_upper, stop=longitudinal_lower, length=n_nodes_l)
    nodes = Node{3,T}[]
    for k in 1:n_nodes_l, j in 1:n_nodes_r, i in 1:n_nodes_c
        # cylindrical -> carthesian
        radius = radial_coords[j]-apicobasal_tilt*longitudinal_angle[k]/maximum(abs.(longitudinal_angle))
        push!(nodes, Node((radius*cos(circumferential_angle[i]), radius*sin(circumferential_angle[i]), longitudinal_angle[k])))
    end

    # Generate cells
    node_array = reshape(collect(1:n_nodes), (n_nodes_c, n_nodes_r, n_nodes_l))
    cells = Hexahedron[]
    for k in 1:num_elements_logintudinal, j in 1:num_elements_radial, i in 1:num_elements_circumferential
        push!(cells, Hexahedron((node_array[i,j,k], node_array[i + 1,j,k], node_array[i + 1,j+1,k], node_array[i,j+1,k],
                                 node_array[i,j,k+1], node_array[i + 1,j,k+1], node_array[i + 1,j+1,k+1], node_array[i,j+1,k+1])))
    end

    # Cell facets
    cell_array = reshape(collect(1:ne_tot),(num_elements_circumferential, num_elements_radial, num_elements_logintudinal))
    boundary = FacetIndex[[FacetIndex(cl, 1) for cl in cell_array[:,:,1][:]];
                            [FacetIndex(cl, 2) for cl in cell_array[:,1,:][:]];
                            [FacetIndex(cl, 3) for cl in cell_array[end,:,:][:]];
                            [FacetIndex(cl, 4) for cl in cell_array[:,end,:][:]];
                            [FacetIndex(cl, 5) for cl in cell_array[1,:,:][:]];
                            [FacetIndex(cl, 6) for cl in cell_array[:,:,end][:]]]

    # Cell facet sets
    offset = 0
    facetsets = Dict{String,OrderedSet{FacetIndex}}()

    facetsets["Myocardium"] = OrderedSet{FacetIndex}(boundary[(1:length(cell_array[:,:,1][:]))   .+ offset]); offset += length(cell_array[:,:,1][:])
    facetsets["Endocardium"]  = OrderedSet{FacetIndex}(boundary[(1:length(cell_array[:,1,:][:]))   .+ offset]); offset += length(cell_array[:,1,:][:])
    facetsets["Open1"]  = OrderedSet{FacetIndex}(boundary[(1:length(cell_array[end,:,:][:])) .+ offset]); offset += length(cell_array[end,:,:][:])
    facetsets["Epicardium"]   = OrderedSet{FacetIndex}(boundary[(1:length(cell_array[:,end,:][:])) .+ offset]); offset += length(cell_array[:,end,:][:])
    facetsets["Open2"]   = OrderedSet{FacetIndex}(boundary[(1:length(cell_array[1,:,:][:]))   .+ offset]); offset += length(cell_array[1,:,:][:])
    facetsets["Base"]    = OrderedSet{FacetIndex}(boundary[(1:length(cell_array[:,:,end][:])) .+ offset]); offset += length(cell_array[:,:,end][:])

    nodesets = Dict{String,OrderedSet{Int}}()
    nodesets["MyocardialAnchor1"] = OrderedSet{Int}([node_array[1,1,1]])
    nodesets["MyocardialAnchor2"] = OrderedSet{Int}([node_array[1,end,1]])
    nodesets["MyocardialAnchor3"] = OrderedSet{Int}([node_array[ceil(Int,1+n_nodes_c/4),1,1]])
    nodesets["MyocardialAnchor4"] = OrderedSet{Int}([node_array[ceil(Int,1+3*n_nodes_c/4),1,1]])

    return to_mesh(Grid(cells, nodes, facetsets=facetsets, nodesets=nodesets))
end


# const linear_index_to_local_index_table_hex27 = [1,9,2, 12,21,10, 4,11,3,  17,22,18, 25,27,23, 20,24,19, 5,13,6, 16,26,14, 8,15,7]
# const local_index_to_linear_index_table_hex27 = invperm(linear_index_to_local_index_table_hex27)
# const tensorproduct_index_to_local_index_table_hex27 = reshape(raw_index_to_local_index_table_hex27, (3,3,3))

"""
    generate_quadratic_ring_mesh(num_elements_circumferential::Int, num_elements_radial::Int, num_elements_logintudinal::Int; inner_radius::T = Float64(0.75), outer_radius::T = Float64(1.0), longitudinal_lower::T = Float64(-0.2), longitudinal_upper::T = Float64(0.2), apicobasal_tilt::T=Float64(0.0)) where {T}

Generates an idealized full-hexahedral ring with quadratic ansatz. Geometrically it is the substraction of a small cylinder ``C_i`` of a large cylinder ``C_o``.
The number of elements for the cylindrical system can be controlled by the first three input parameters.
The remaining parameters control the spatial dimensions and the ring shape.
"""
function generate_quadratic_ring_mesh(num_elements_circumferential::Int, num_elements_radial::Int, num_elements_logintudinal::Int; inner_radius::T = Float64(0.75), outer_radius::T = Float64(1.0), longitudinal_lower::T = Float64(-0.2), longitudinal_upper::T = Float64(0.2), apicobasal_tilt::T=Float64(0.0)) where {T}
    # Generate a rectangle in cylindrical coordinates and transform coordinates back to carthesian.
    ne_tot = num_elements_circumferential*num_elements_radial*num_elements_logintudinal;
    n_nodes_c = 2*num_elements_circumferential;
    n_nodes_r = 2*num_elements_radial+1;
    n_nodes_l = 2*num_elements_logintudinal+1;
    n_nodes = n_nodes_c * n_nodes_r * n_nodes_l;

    # Generate nodes
    circumferential_angle = range(0.0, stop=2*π, length=n_nodes_c+1)
    radial_coords = range(inner_radius, stop=outer_radius, length=n_nodes_r)
    longitudinal_angle = range(longitudinal_upper, stop=longitudinal_lower, length=n_nodes_l)
    nodes = Node{3,T}[]
    for k in 1:n_nodes_l, j in 1:n_nodes_r, i in 1:n_nodes_c
        # cylindrical -> carthesian
        radius = radial_coords[j]-apicobasal_tilt*longitudinal_angle[k]/maximum(abs.(longitudinal_angle))
        push!(nodes, Node((radius*cos(circumferential_angle[i]), radius*sin(circumferential_angle[i]), longitudinal_angle[k])))
    end

    # Generate cells
    node_array = reshape(collect(1:n_nodes), (n_nodes_c, n_nodes_r, n_nodes_l))
    cells = QuadraticHexahedron[]
    for k_ in 1:num_elements_logintudinal, j_ in 1:num_elements_radial, i_ in 1:num_elements_circumferential
        i_next = (i_ == num_elements_circumferential) ? 1 : 2*i_ + 1
        i = 2*i_-1
        j = 2*j_-1
        k = 2*k_-1
        push!(cells, QuadraticHexahedron((
            node_array[i+0,j+0,k+0], node_array[i_next,j+0,k+0], node_array[i_next,j+2,k+0], node_array[i+0,j+2,k+0], # Vertex loop back
            node_array[i+0,j+0,k+2], node_array[i_next,j+0,k+2], node_array[i_next,j+2,k+2], node_array[i+0,j+2,k+2],  # Vertex loop front
            node_array[i+1,j+0,k+0], node_array[i_next,j+1,k+0], node_array[i+1,j+2,k+0],    node_array[i+0,j+1,k+0], # Edge loop back
            node_array[i+1,j+0,k+2], node_array[i_next,j+1,k+2], node_array[i+1,j+2,k+2],    node_array[i+0,j+1,k+2], # Edge loop front
            node_array[i+0,j+0,k+1], node_array[i_next,j+0,k+1], node_array[i_next,j+2,k+1], node_array[i+0,j+2,k+1], # Edge loop center
            node_array[i+1,j+1,k+0], node_array[i+1,j+0,k+1],    node_array[i_next,j+1,k+1], node_array[i+1,j+2,k+1], node_array[i+0,j+1,k+1], node_array[i+1,j+1,k+2], # Facet centers
            node_array[i+1,j+1,k+1]# Center
        )))
    end

    # Cell facets
    cell_array = reshape(collect(1:ne_tot),(num_elements_circumferential, num_elements_radial, num_elements_logintudinal))
    boundary = FacetIndex[[FacetIndex(cl, 1) for cl in cell_array[:,:,1][:]];
                            [FacetIndex(cl, 2) for cl in cell_array[:,1,:][:]];
                            #[FacetIndex(cl, 3) for cl in cell_array[end,:,:][:]];
                            [FacetIndex(cl, 4) for cl in cell_array[:,end,:][:]];
                            #[FacetIndex(cl, 5) for cl in cell_array[1,:,:][:]];
                            [FacetIndex(cl, 6) for cl in cell_array[:,:,end][:]]]

    # Cell facet sets
    offset = 0
    facetsets = Dict{String,OrderedSet{FacetIndex}}()
    facetsets["Myocardium"] = OrderedSet{FacetIndex}(boundary[(1:length(cell_array[:,:,1][:]))   .+ offset]); offset += length(cell_array[:,:,1][:])
    facetsets["Endocardium"]  = OrderedSet{FacetIndex}(boundary[(1:length(cell_array[:,1,:][:]))   .+ offset]); offset += length(cell_array[:,1,:][:])
    facetsets["Epicardium"]   = OrderedSet{FacetIndex}(boundary[(1:length(cell_array[:,end,:][:])) .+ offset]); offset += length(cell_array[:,end,:][:])
    facetsets["Base"]    = OrderedSet{FacetIndex}(boundary[(1:length(cell_array[:,:,end][:])) .+ offset]); offset += length(cell_array[:,:,end][:])

    nodesets = Dict{String,OrderedSet{Int}}()
    nodesets["MyocardialAnchor1"] = OrderedSet{Int}([node_array[1,1,1]])
    nodesets["MyocardialAnchor2"] = OrderedSet{Int}([node_array[1,end,1]])
    nodesets["MyocardialAnchor3"] = OrderedSet{Int}([node_array[ceil(Int,1+n_nodes_c/4),1,1]])
    nodesets["MyocardialAnchor4"] = OrderedSet{Int}([node_array[ceil(Int,1+3*n_nodes_c/4),1,1]])

    return to_mesh(Grid(cells, nodes, facetsets=facetsets, nodesets=nodesets))
end


"""
    generate_quadratic_open_ring_mesh(num_elements_circumferential::Int, num_elements_radial::Int, num_elements_logintudinal::Int, opening_angle::Float64; inner_radius::T = Float64(0.75), outer_radius::T = Float64(1.0), longitudinal_lower::T = Float64(-0.2), longitudinal_upper::T = Float64(0.2), apicobasal_tilt::T=Float64(0.0)) where {T}

Generates an idealized full-hexahedral ring with given opening angle and quadratic ansatz. Geometrically it is the substraction of a small cylinder ``C_i`` of a large cylinder ``C_o``.
The number of elements for the cylindrical system can be controlled by the first three input parameters.
The remaining parameters control the spatial dimensions and the ring shape.
The ring is opened along the Cartesian x-z plane.
"""
function generate_quadratic_open_ring_mesh(num_elements_circumferential::Int, num_elements_radial::Int, num_elements_logintudinal::Int, opening_angle::Float64; inner_radius::T = Float64(0.75), outer_radius::T = Float64(1.0), longitudinal_lower::T = Float64(-0.2), longitudinal_upper::T = Float64(0.2), apicobasal_tilt::T=Float64(0.0)) where {T}
    # Generate a rectangle in cylindrical coordinates and transform coordinates back to carthesian.
    ne_tot = num_elements_circumferential*num_elements_radial*num_elements_logintudinal;
    n_nodes_c = 2*num_elements_circumferential+1;
    n_nodes_r = 2*num_elements_radial+1;
    n_nodes_l = 2*num_elements_logintudinal+1;
    n_nodes = n_nodes_c * n_nodes_r * n_nodes_l;

    # Generate nodes
    circumferential_angle = range(opening_angle/2, stop=2*π-opening_angle/2, length=n_nodes_c)
    radial_coords = range(inner_radius, stop=outer_radius, length=n_nodes_r)
    longitudinal_angle = range(longitudinal_upper, stop=longitudinal_lower, length=n_nodes_l)
    nodes = Node{3,T}[]
    for k in 1:n_nodes_l, j in 1:n_nodes_r, i in 1:n_nodes_c
        # cylindrical -> carthesian
        radius = radial_coords[j]-apicobasal_tilt*longitudinal_angle[k]/maximum(abs.(longitudinal_angle))
        push!(nodes, Node((radius*cos(circumferential_angle[i]), radius*sin(circumferential_angle[i]), longitudinal_angle[k])))
    end

    # Generate cells
    node_array = reshape(collect(1:n_nodes), (n_nodes_c, n_nodes_r, n_nodes_l))
    cells = QuadraticHexahedron[]
    for k_ in 1:num_elements_logintudinal, j_ in 1:num_elements_radial, i_ in 1:num_elements_circumferential
        i = 2*i_-1
        j = 2*j_-1
        k = 2*k_-1
        push!(cells, QuadraticHexahedron((
            node_array[i+0,j+0,k+0], node_array[2*i_ + 1,j+0,k+0], node_array[2*i_ + 1,j+2,k+0], node_array[i+0,j+2,k+0], # Vertex loop back
            node_array[i+0,j+0,k+2], node_array[2*i_ + 1,j+0,k+2], node_array[2*i_ + 1,j+2,k+2], node_array[i+0,j+2,k+2],  # Vertex loop front
            node_array[i+1,j+0,k+0], node_array[2*i_ + 1,j+1,k+0], node_array[i+1,j+2,k+0],    node_array[i+0,j+1,k+0], # Edge loop back
            node_array[i+1,j+0,k+2], node_array[2*i_ + 1,j+1,k+2], node_array[i+1,j+2,k+2],    node_array[i+0,j+1,k+2], # Edge loop front
            node_array[i+0,j+0,k+1], node_array[2*i_ + 1,j+0,k+1], node_array[2*i_ + 1,j+2,k+1], node_array[i+0,j+2,k+1], # Edge loop center
            node_array[i+1,j+1,k+0], node_array[i+1,j+0,k+1],    node_array[2*i_ + 1,j+1,k+1], node_array[i+1,j+2,k+1], node_array[i+0,j+1,k+1], node_array[i+1,j+1,k+2], # Facet centers
            node_array[i+1,j+1,k+1]# Center
        )))
    end

    # Cell facets
    cell_array = reshape(collect(1:ne_tot),(num_elements_circumferential, num_elements_radial, num_elements_logintudinal))
    boundary = FacetIndex[[FacetIndex(cl, 1) for cl in cell_array[:,:,1][:]];
                            [FacetIndex(cl, 2) for cl in cell_array[:,1,:][:]];
                            #[FacetIndex(cl, 3) for cl in cell_array[end,:,:][:]];
                            [FacetIndex(cl, 4) for cl in cell_array[:,end,:][:]];
                            #[FacetIndex(cl, 5) for cl in cell_array[1,:,:][:]];
                            [FacetIndex(cl, 6) for cl in cell_array[:,:,end][:]]]

    # Cell facet sets
    offset = 0
    facetsets = Dict{String,OrderedSet{FacetIndex}}()
    facetsets["Myocardium"] = OrderedSet{FacetIndex}(boundary[(1:length(cell_array[:,:,1][:]))   .+ offset]); offset += length(cell_array[:,:,1][:])
    facetsets["Endocardium"]  = OrderedSet{FacetIndex}(boundary[(1:length(cell_array[:,1,:][:]))   .+ offset]); offset += length(cell_array[:,1,:][:])
    facetsets["Epicardium"]   = OrderedSet{FacetIndex}(boundary[(1:length(cell_array[:,end,:][:])) .+ offset]); offset += length(cell_array[:,end,:][:])
    facetsets["Base"]    = OrderedSet{FacetIndex}(boundary[(1:length(cell_array[:,:,end][:])) .+ offset]); offset += length(cell_array[:,:,end][:])

    nodesets = Dict{String,OrderedSet{Int}}()
    nodesets["MyocardialAnchor1"] = OrderedSet{Int}([node_array[1,1,1]])
    nodesets["MyocardialAnchor2"] = OrderedSet{Int}([node_array[1,end,1]])
    nodesets["MyocardialAnchor3"] = OrderedSet{Int}([node_array[ceil(Int,1+n_nodes_c/4),1,1]])
    nodesets["MyocardialAnchor4"] = OrderedSet{Int}([node_array[ceil(Int,1+3*n_nodes_c/4),1,1]])

    return to_mesh(Grid(cells, nodes, facetsets=facetsets, nodesets=nodesets))
end


"""
    generate_ideal_lv_mesh(num_elements_circumferential::Int, num_elements_radial::Int, num_elements_logintudinally::Int; inner_radius::T = Float64(0.7), outer_radius::T = Float64(1.0), longitudinal_upper::T = Float64(0.2), apex_inner::T = Float64(1.3), apex_outer::T = Float64(1.5))

Generate an idealized left ventricle as a truncated ellipsoid.
The number of elements per axis are controlled by the first three parameters.
"""
function generate_ideal_lv_mesh(num_elements_circumferential::Int, num_elements_radial::Int, num_elements_logintudinal::Int; inner_radius::T = Float64(0.7), outer_radius::T = Float64(1.0), longitudinal_upper::T = Float64(0.2), apex_inner::T = Float64(1.3), apex_outer::T = Float64(1.5)) where {T}
    # Generate a rectangle in cylindrical coordinates and transform coordinates back to carthesian.
    ne_tot = num_elements_circumferential*num_elements_radial*num_elements_logintudinal;
    n_nodes_c = num_elements_circumferential; n_nodes_r = num_elements_radial+1; n_nodes_l = num_elements_logintudinal+1;
    n_nodes = n_nodes_c * n_nodes_r * n_nodes_l;

    # Generate nodes
    # Take a ring section of the heart and mark its circumferential coordinate by its angle
    circumferential_angle = range(0.0, stop=2*π, length=n_nodes_c+1)
    # Untransformed radial coordinate of a ring section
    radii_in_percent = range(0.0, stop=1.0, length=n_nodes_r)
    # z axis expressed as the angle between the apicobasal vector and the current layer from apex (0.0) to base ((1.0+longitudinal_upper)*π/2)
    longitudinal_angle = range(0, stop=(1.0+longitudinal_upper)*π/2, length=n_nodes_l+1)
    nodes = Node{3,T}[]
    # Add nodes up to apex
    for θ ∈ longitudinal_angle[2:(end-1)], radius_percent ∈ radii_in_percent, φ ∈ circumferential_angle[1:(end-1)]
        # thickness_inner = inner_radius*longitudinal_percent + apex_inner*(1.0-longitudinal_percent)
        # thickness_outer = outer_radius*longitudinal_percent + apex_outer*(1.0-longitudinal_percent)
        # radius = radius_percent*thickness_outer + (1.0-radius_percent)*thickness_inner
        radius = inner_radius*(1.0-radius_percent) + outer_radius*(radius_percent)
        # cylindrical -> carthesian
        z = θ < π/2 ? (apex_inner*(1.0-radius_percent) + apex_outer*(radius_percent))*cos(θ) : apex_outer*cos(θ)
        push!(nodes, Node((radius*cos(φ)*sin(θ), radius*sin(φ)*sin(θ), z)))
    end

    # Add flat base
    for θ ∈ longitudinal_angle[end], radius_percent ∈ radii_in_percent, φ ∈ circumferential_angle[1:(end-1)]
        radius = inner_radius*(1.0-radius_percent) + outer_radius*(radius_percent)
        # cylindrical -> carthesian
        push!(nodes, Node((radius*cos(φ)*sin(θ), radius*sin(φ)*sin(θ), apex_outer*cos(θ))))
    end

    # Generate all cells but the apex
    node_array = reshape(collect(1:n_nodes), (n_nodes_c, n_nodes_r, n_nodes_l))
    cells = Union{Hexahedron,Wedge}[]
    for k in 1:num_elements_logintudinal, j in 1:num_elements_radial, i in 1:num_elements_circumferential
        i_next = (i == num_elements_circumferential) ? 1 : i + 1
        push!(cells, Hexahedron((node_array[i,j,k], node_array[i_next,j,k], node_array[i_next,j+1,k], node_array[i,j+1,k],
                                 node_array[i,j,k+1], node_array[i_next,j,k+1], node_array[i_next,j+1,k+1], node_array[i,j+1,k+1])))
    end

    nodesets = Dict{String,OrderedSet{Int}}()
    nodesets["MyocardialAnchor1"] = OrderedSet{Int}([node_array[1,1,end]])
    nodesets["MyocardialAnchor2"] = OrderedSet{Int}([node_array[1,end,end]])
    nodesets["MyocardialAnchor3"] = OrderedSet{Int}([node_array[ceil(Int,1+n_nodes_c/4),1,end]])
    nodesets["MyocardialAnchor4"] = OrderedSet{Int}([node_array[ceil(Int,1+3*n_nodes_c/4),1,end]])

    # Cell facets
    cell_array = reshape(collect(1:ne_tot),(num_elements_circumferential, num_elements_radial, num_elements_logintudinal))
    boundary = FacetIndex[[FacetIndex(cl, 2) for cl in cell_array[:,1,:][:]];
                          [FacetIndex(cl, 4) for cl in cell_array[:,end,:][:]];
                          [FacetIndex(cl, 6) for cl in cell_array[:,:,end][:]]]

    # Cell facet sets
    offset = 0
    facetsets = Dict{String,OrderedSet{FacetIndex}}()
    facetsets["Endocardium"]  = OrderedSet{FacetIndex}(boundary[(1:length(cell_array[:,1,:][:]))   .+ offset]); offset += length(cell_array[:,1,:][:])
    facetsets["Epicardium"]   = OrderedSet{FacetIndex}(boundary[(1:length(cell_array[:,end,:][:])) .+ offset]); offset += length(cell_array[:,end,:][:])
    facetsets["Base"]    = OrderedSet{FacetIndex}(boundary[(1:length(cell_array[:,:,end][:])) .+ offset]); offset += length(cell_array[:,:,end][:])
    nodesets["Apex"] = OrderedSet{Int}()
    nodesets["ApexInOut"] = OrderedSet{Int}()

    # Add apex nodes
    push!(nodesets["ApexInOut"], length(nodes)+1)
    for radius_percent ∈ radii_in_percent
        radius = apex_inner*(1.0-radius_percent) + apex_outer*(radius_percent)
        push!(nodes, Node((0.0, 0.0, radius)))
    end
    push!(nodesets["ApexInOut"], length(nodes))

    # Add apex cells
    for j ∈ 1:num_elements_radial, i ∈ 1:num_elements_circumferential
        i_next = (i == num_elements_circumferential) ? 1 : i + 1
        singular_index = length(nodes)-num_elements_radial+j-1
        push!(cells, Wedge((
            singular_index , node_array[i,j,1], node_array[i_next,j,1],
            singular_index+1, node_array[i,j+1,1], node_array[i_next,j+1,1],
        )))
        j == 1 && push!(facetsets["Endocardium"], FacetIndex(length(cells), 1))
        j == num_elements_radial && push!(facetsets["Epicardium"], FacetIndex(length(cells), 5))
        j == num_elements_radial && push!(nodesets["Apex"], singular_index+1)
    end

    return to_mesh(Grid(cells, nodes, nodesets=nodesets, facetsets=facetsets))
end

generate_mesh(args...) = to_mesh(generate_grid(args...))

function generate_simple_disc_grid(::Type{Quadrilateral}, n; radius= 1.0)
    nnodes = 2n + 1
    θ = deg2rad(360/2n)

    nodepos = Vec((0.0,radius))
    nodes = [rotate(nodepos, θ*i) for i ∈ 0:(2n-1)]
    push!(nodes, Vec((0.0,0.0)))

    elements = Quadrilateral[Quadrilateral((2i-1==0 ? nnodes-1 : 2i-1,2i,2i+1 == nnodes ? 1 : 2i+1,nnodes)) for i ∈ 1:n]

    facetsets = Dict(
        "boundary" => OrderedSet([FacetIndex(i,1) for i ∈ 1:n]) ∪ OrderedSet([FacetIndex(i,2) for i ∈ 1:n]),
    )

    return Grid(elements, Node.(nodes); facetsets=facetsets)
end

generate_simple_disc_mesh(::Type{Quadrilateral}, n; radius= 1.0) = to_mesh(generate_simple_disc_grid(Quadrilateral, n; radius))
