abstract type AbstractTransferOperator end

function get_subdofhandler_indices_on_subdomains(dh::DofHandler, subdomain_names::Vector{String})
    grid = get_grid(dh)
    sdh_ids = Set{Int}()
    for (sdhidx, sdh) in enumerate(dh.subdofhandlers)
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

function _compute_dof_nodes_barrier!(
    nodes,
    sdh,
    dofrange,
    gip,
    dof_to_node_map,
    ref_coords,
    adj = nothing,
)
    _compute_dof_nodes_barrier!(
        nodes,
        sdh,
        dofrange,
        gip,
        dof_to_node_map,
        ref_coords,
        adj,
    )
end

function _compute_dof_nodes_barrier!(
    nodes,
    sdh,
    dofrange,
    gip,
    dof_to_node_map,
    ref_coords,
    adj::AbstractMatrix,
)
    for cc ∈ CellIterator(sdh)
        # Compute for each dof the spatial coordinate of from the reference coordiante and store.
        # NOTE We assume a continuous coordinate field if the interpolation is continuous.
        dofs = @view celldofs(cc)[dofrange]
        for (dofidx, dof) in enumerate(dofs)
            nodes[dof_to_node_map[dof]] =
                spatial_coordinate(gip, ref_coords[dofidx], getcoordinates(cc))
        end
        for dof_i in dofs
            for dof_j in dofs
                adj[dof_i, dof_j] =
                    norm(nodes[dof_to_node_map[dof_i]] - nodes[dof_to_node_map[dof_j]])
            end
        end
    end
end

function _compute_dof_nodes_barrier!(
    nodes,
    sdh,
    dofrange,
    gip,
    dof_to_node_map,
    ref_coords,
    adj::Nothing,
)
    for cc ∈ CellIterator(sdh)
        # Compute for each dof the spatial coordinate of from the reference coordiante and store.
        # NOTE We assume a continuous coordinate field if the interpolation is continuous.
        dofs = @view celldofs(cc)[dofrange]
        for (dofidx, dof) in enumerate(dofs)
            nodes[dof_to_node_map[dof]] =
                spatial_coordinate(gip, ref_coords[dofidx], getcoordinates(cc))
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
struct NodalIntergridInterpolation{
    PH <: PointEvalHandler,
    DH1 <: AbstractDofHandler,
    DH2 <: AbstractDofHandler,
} <: AbstractTransferOperator
    ph::PH
    dh_from::DH1
    dh_to::DH2
    node_to_dof_map::Vector{Int}
    dof_to_node_map::Dict{Int, Int}
    field_name_from::Symbol
    field_name_to::Symbol

    function NodalIntergridInterpolation(
        dh_from::DofHandler{sdim},
        dh_to::DofHandler{sdim},
        field_name_from::Symbol,
        field_name_to::Symbol;
        subdomains_from = 1:length(dh_from.subdofhandlers),
        subdomains_to = 1:length(dh_to.subdofhandlers),
    ) where {sdim}
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
        dof_to_node_map = Dict{Int, Int}()
        next_dof_index = 1
        for dof ∈ node_to_dof_map
            dof_to_node_map[dof] = next_dof_index
            next_dof_index += 1
        end

        # Compute nodes
        grid_to   = Ferrite.get_grid(dh_to)
        grid_from = Ferrite.get_grid(dh_from)
        nodes     = Vector{Ferrite.get_coordinate_type(grid_to)}(undef, length(dofset))
        for sdh in dh_to.subdofhandlers[subdomains_from]
            # Skip subdofhandler if field is not present
            field_name_to ∈ Ferrite.getfieldnames(sdh) || continue
            # Grab the reference coordinates of the field to interpolate
            ip = Ferrite.getfieldinterpolation(sdh, field_name_to)
            ref_coords = Ferrite.reference_coordinates(ip)
            # Grab the geometric interpolation
            first_cell = getcells(grid_to, first(sdh.cellset))
            cell_type  = typeof(first_cell)
            gip        = Ferrite.geometric_interpolation(cell_type)

            _compute_dof_nodes_barrier!(
                nodes,
                sdh,
                Ferrite.dof_range(sdh, field_name_to),
                gip,
                dof_to_node_map,
                ref_coords,
            )
        end

        ph = PointEvalHandler(Ferrite.get_grid(dh_from), nodes; warn = false)

        n_missing = sum(x -> x === nothing, ph.cells)
        n_missing == 0 ||
            @warn "Constructing the interpolation for $field_name_from to $field_name_to failed. $n_missing (out of $(length(ph.cells))) points not found."

        new{typeof(ph), typeof(dh_from), typeof(dh_to)}(
            ph,
            dh_from,
            dh_to,
            node_to_dof_map,
            dof_to_node_map,
            field_name_from,
            field_name_to,
        )
    end
end

function NodalIntergridInterpolation(
    dh_from::DofHandler{sdim},
    dh_to::DofHandler{sdim},
) where {sdim}
    @assert length(Ferrite.getfieldnames(dh_from)) == 1 "Multiple fields found in source dof handler. Please specify which field you want to transfer."
    return NodalIntergridInterpolation(dh_from, dh_to, first(Ferrite.getfieldnames(dh_from)))
end

function NodalIntergridInterpolation(
    dh_from::DofHandler{sdim},
    dh_to::DofHandler{sdim},
    field_name::Symbol,
) where {sdim}
    return NodalIntergridInterpolation(dh_from, dh_to, field_name, field_name)
end

@inline function rbf_value(d, r)
    (max((1 - d / r), zero(r))^4) * (1 + 4d / r)
end

function build_sparse_matrix_kdtree(
    coords_vec,
    rbf_func,
    tree,
    distances,
    distance_func,
    α = 2.0,
)
    N = length(coords_vec)

    rows = Int[]
    cols = Int[]
    vals = Float64[]

    @views for j = 1:N
        radius = distances[j] * α
        # Query all points within radius around point j (including j itself)
        idxs = inrange(tree, coords_vec[j], radius)
        # dists_vec = norm.(coords_vec[idxs] .- Ref(coords_vec[j]))
        dists_vec = distance_func(coords_vec, j, coords_vec, idxs)
        for (i, d) in zip(idxs, dists_vec)
            # No need to check d < radius – guaranteed by inrange
            val = rbf_func(d, radius)
            push!(rows, i)
            push!(cols, j)
            push!(vals, val)
        end
    end

    # Build CSC matrix; duplicate entries (if any) will be summed automatically
    A = sparse(rows, cols, vals)
    return A
end

function construct_RBF_dist_kdtree(
    coords_src,
    distances,
    coords_dist,
    rbf_func,
    distance_func,
    α = 2.0,
)
    N_src = length(coords_src)   # number of columns in A
    N_dst = length(coords_dist)  # number of rows in A

    # Build KD‑tree on destination points (used to query points within radius)
    tree = KDTree(coords_dist)

    rows = Int[]
    cols = Int[]
    vals = Float64[]

    @views for j = 1:N_src
        radius = distances[j] * α
        idxs = inrange(tree, coords_src[j], radius)
        # dists_vec = norm.(coords_dist[idxs] .- Ref(coords_src[j]))
        dists_vec = distance_func(coords_src, j, coords_dist, idxs)
        for (i, d) in zip(idxs, dists_vec)
            val = rbf_func(d, radius)
            push!(rows, i)
            push!(cols, j)
            push!(vals, val)
        end
    end

    # Build CSC matrix of size N_dst × N_src
    A = sparse(rows, cols, vals)
    return A
end
"""
    RadialBasisFunctionTransferOperator(dh_from::DofHandler{sdim}, dh_to::DofHandler{sdim}, field_name_from::Symbol, field_name_to::Symbol; subdomain_from = 1:length(dh_from.subdofhandlers), subdomains_to = 1:length(dh_to.subdofhandlers))
    RadialBasisFunctionTransferOperator(dh_from::DofHandler{sdim}, dh_to::DofHandler{sdim}, field_name::Symbol; subdomain_from = 1:length(dh_from.subdofhandlers), subdomains_to = 1:length(dh_to.subdofhandlers))
    RadialBasisFunctionTransferOperator(dh_from::DofHandler{sdim}, dh_to::DofHandler{sdim}; subdomain_from = 1:length(dh_from.subdofhandlers), subdomains_to = 1:length(dh_to.subdofhandlers))

Construct a transfer operator to move a field `field_name` from dof handler `dh_from` to another
dof handler `dh_to`, assuming that all spatial coordinates of the dofs for `dh_to` are in the
interior or boundary of the mesh contained within dh_from. This is necessary to have valid
interpolation values, as this operator does not have extrapolation functionality.

!!! note
    We assume a continuous coordinate field, if the interpolation of the named field is continuous.
"""
struct RadialBasisFunctionTransferOperator{
    Rescale,
    Geodesic,
    DH1 <: AbstractDofHandler,
    DH2 <: AbstractDofHandler,
    SIMT <: AbstractMatrix,
    DIMT <: AbstractMatrix,
    γT <: Tuple{<: AbstractVector, <: AbstractVector},
    SLinsolveCT,
    TreeT,
} <: AbstractTransferOperator
    dh_from::DH1
    dh_to::DH2
    node_to_dof_map_from::Vector{Int}
    node_to_dof_map_to::Vector{Int}
    dof_to_node_map_from::Dict{Int, Int}
    dof_to_node_map_to::Dict{Int, Int}
    field_name_from::Symbol
    field_name_to::Symbol
    source_influence_matrix::SIMT
    destination_influence_matrix::DIMT
    interpolation_weights::γT
    source_linsolve_cache::SLinsolveCT
    source_kdtree::TreeT
end

function RadialBasisFunctionTransferOperator(
    dh_from::DofHandler,
    dh_to::DofHandler,
    field_name_from::Symbol,
    field_name_to::Symbol;
    subdomains_from = 1:length(dh_from.subdofhandlers),
    subdomains_to = 1:length(dh_to.subdofhandlers),
    rescale = Val(false),
    geodesic = Val(true),
) 
    _RadialBasisFunctionTransferOperator(
    dh_from,
    dh_to,
    field_name_from,
    field_name_to,
    rescale,
    geodesic;
    subdomains_from,
    subdomains_to,
    )

end

function _RadialBasisFunctionTransferOperator(
    dh_from::DofHandler{sdim},
    dh_to::DofHandler{sdim},
    field_name_from::Symbol,
    field_name_to::Symbol,
    rescale,
    geodesic::Val{true};
    subdomains_from = 1:length(dh_from.subdofhandlers),
    subdomains_to = 1:length(dh_to.subdofhandlers),

) where {sdim}
    @assert field_name_from ∈ Ferrite.getfieldnames(dh_from)
    @assert field_name_to ∈ Ferrite.getfieldnames(dh_to)

    dofset_from = Set{Int}()
    dofset_to = Set{Int}()
    distances_source = allocate_matrix(dh_from)
    for sdh in dh_to.subdofhandlers[subdomains_to]
        # Skip subdofhandler if field is not present
        field_name_to ∈ Ferrite.getfieldnames(sdh) || continue
        # Just gather the dofs of the given field in the set
        for cellidx ∈ sdh.cellset
            dofs = celldofs(dh_to, cellidx)
            for dof in dofs[dof_range(sdh, field_name_to)]
                push!(dofset_to, dof)
            end
        end
    end
    for sdh in dh_from.subdofhandlers[subdomains_from]
        # Skip subdofhandler if field is not present
        field_name_from ∈ Ferrite.getfieldnames(sdh) || continue
        # Just gather the dofs of the given field in the set
        for cellidx ∈ sdh.cellset
            dofs = celldofs(dh_from, cellidx)
            for dof in dofs[dof_range(sdh, field_name_from)]
                push!(dofset_from, dof)
            end
        end
    end
    node_to_dof_map_to = sort(collect(dofset_to))
    node_to_dof_map_from = sort(collect(dofset_from))

    # Build inverse map
    dof_to_node_map_to = Dict{Int, Int}()
    dof_to_node_map_from = Dict{Int, Int}()
    next_dof_index = 1
    for dof ∈ node_to_dof_map_to
        dof_to_node_map_to[dof] = next_dof_index
        next_dof_index += 1
    end

    next_dof_index = 1
    for dof ∈ node_to_dof_map_from
        dof_to_node_map_from[dof] = next_dof_index
        next_dof_index += 1
    end

    # Compute nodes
    grid_to = Ferrite.get_grid(dh_to)
    grid_from = Ferrite.get_grid(dh_from)
    nodes_from = Vector{Ferrite.get_coordinate_type(grid_from)}(undef, length(dofset_from))
    nodes_to = Vector{Ferrite.get_coordinate_type(grid_to)}(undef, length(dofset_to))
    for sdh in dh_from.subdofhandlers[subdomains_from]
        # Skip subdofhandler if field is not present
        field_name_from ∈ Ferrite.getfieldnames(sdh) || continue
        # Grab the reference coordinates of the field to interpolate
        ip = Ferrite.getfieldinterpolation(sdh, field_name_from)
        ref_coords = Ferrite.reference_coordinates(ip)
        # Grab the geometric interpolation
        first_cell = getcells(grid_from, first(sdh.cellset))
        cell_type  = typeof(first_cell)
        gip        = Ferrite.geometric_interpolation(cell_type)

        _compute_dof_nodes_barrier!(
            nodes_from,
            sdh,
            Ferrite.dof_range(sdh, field_name_from),
            gip,
            dof_to_node_map_from,
            ref_coords,
            distances_source,
        )
    end
    for sdh in dh_to.subdofhandlers[subdomains_to]
        # Skip subdofhandler if field is not present
        field_name_to ∈ Ferrite.getfieldnames(sdh) || continue
        # Grab the reference coordinates of the field to interpolate
        ip = Ferrite.getfieldinterpolation(sdh, field_name_to)
        ref_coords = Ferrite.reference_coordinates(ip)
        # Grab the geometric interpolation
        first_cell = getcells(grid_to, first(sdh.cellset))
        cell_type  = typeof(first_cell)
        gip        = Ferrite.geometric_interpolation(cell_type)

        _compute_dof_nodes_barrier!(
            nodes_to,
            sdh,
            Ferrite.dof_range(sdh, field_name_to),
            gip,
            dof_to_node_map_to,
            ref_coords,
        )
    end
    γf = zeros(length(node_to_dof_map_from))
    γg = zeros(length(node_to_dof_map_from))
    source_kdtree = KDTree(nodes_from)
    source_graph = SimpleGraph(length(nodes_from))
    for i in eachindex(IndexCartesian(), distances_source)
        add_edge!(source_graph, i[1], i[2]);
        add_edge!(source_graph, i[2], i[1]);
    end
    source_sortest_path = Parallel.dijkstra_shortest_paths(
        source_graph,
        GraphsVertices(source_graph),
        distances_source,
        parallel = :threads,
    ).dists
    M = 15
    α = 2
    support_radii = maximum.(last.(knn.(Ref(source_kdtree), nodes_from, M)))
    h_max = maximum(distances_source)
    β = 2
    distance_func = (x, xi, y, yi) -> begin
            coords_destination = y[yi]
            nearest_node_in_source, distance_to_neighbor =
                nn(source_kdtree, coords_destination)
            norm_distance = norm.(Ref(x[xi]) .- (y[yi]))
            nn_distance = source_sortest_path[nearest_node_in_source, xi]
            ret = [
                nn_distance[i] <= (norm_distance[i] + β*h_max) ? norm_distance[i] :
                (nn_distance[i] < support_radii[i]*α ? nn_distance[i] : Inf) for
                i = 1:length(norm_distance)
            ]
            return ret
        end

    source_influence_matrix =
        build_sparse_matrix_kdtree(nodes_from, rbf_value, source_kdtree, support_radii, distance_func, α)
    destination_influence_matrix =
        construct_RBF_dist_kdtree(nodes_from, support_radii, nodes_to, rbf_value, distance_func, α)
    prob = LinearSolve.LinearProblem(source_influence_matrix, copy(γf))
    linsolve = LinearSolve.init(prob)

    RadialBasisFunctionTransferOperator{
        typeof(rescale),
        typeof(geodesic),
        typeof(dh_from),
        typeof(dh_to),
        typeof(source_influence_matrix),
        typeof(destination_influence_matrix),
        typeof((γf, γg)),
        typeof(linsolve),
        typeof(source_kdtree),
    }(
        dh_from,
        dh_to,
        node_to_dof_map_from,
        node_to_dof_map_to,
        dof_to_node_map_from,
        dof_to_node_map_to,
        field_name_from,
        field_name_to,
        source_influence_matrix,
        destination_influence_matrix,
        (γf, γg),
        linsolve,
        source_kdtree,
    )
end

function _RadialBasisFunctionTransferOperator(
    dh_from::DofHandler{sdim},
    dh_to::DofHandler{sdim},
    field_name_from::Symbol,
    field_name_to::Symbol,
    rescale,
    geodesic::Val{false};
    subdomains_from = 1:length(dh_from.subdofhandlers),
    subdomains_to = 1:length(dh_to.subdofhandlers),

) where {sdim}
    @assert field_name_from ∈ Ferrite.getfieldnames(dh_from)
    @assert field_name_to ∈ Ferrite.getfieldnames(dh_to)

    dofset_from = Set{Int}()
    dofset_to = Set{Int}()
    for sdh in dh_to.subdofhandlers[subdomains_to]
        # Skip subdofhandler if field is not present
        field_name_to ∈ Ferrite.getfieldnames(sdh) || continue
        # Just gather the dofs of the given field in the set
        for cellidx ∈ sdh.cellset
            dofs = celldofs(dh_to, cellidx)
            for dof in dofs[dof_range(sdh, field_name_to)]
                push!(dofset_to, dof)
            end
        end
    end
    for sdh in dh_from.subdofhandlers[subdomains_from]
        # Skip subdofhandler if field is not present
        field_name_from ∈ Ferrite.getfieldnames(sdh) || continue
        # Just gather the dofs of the given field in the set
        for cellidx ∈ sdh.cellset
            dofs = celldofs(dh_from, cellidx)
            for dof in dofs[dof_range(sdh, field_name_from)]
                push!(dofset_from, dof)
            end
        end
    end
    node_to_dof_map_to = sort(collect(dofset_to))
    node_to_dof_map_from = sort(collect(dofset_from))

    # Build inverse map
    dof_to_node_map_to = Dict{Int, Int}()
    dof_to_node_map_from = Dict{Int, Int}()
    next_dof_index = 1
    for dof ∈ node_to_dof_map_to
        dof_to_node_map_to[dof] = next_dof_index
        next_dof_index += 1
    end

    next_dof_index = 1
    for dof ∈ node_to_dof_map_from
        dof_to_node_map_from[dof] = next_dof_index
        next_dof_index += 1
    end

    # Compute nodes
    grid_to = Ferrite.get_grid(dh_to)
    grid_from = Ferrite.get_grid(dh_from)
    nodes_from = Vector{Ferrite.get_coordinate_type(grid_from)}(undef, length(dofset_from))
    nodes_to = Vector{Ferrite.get_coordinate_type(grid_to)}(undef, length(dofset_to))
    for sdh in dh_from.subdofhandlers[subdomains_from]
        # Skip subdofhandler if field is not present
        field_name_from ∈ Ferrite.getfieldnames(sdh) || continue
        # Grab the reference coordinates of the field to interpolate
        ip = Ferrite.getfieldinterpolation(sdh, field_name_from)
        ref_coords = Ferrite.reference_coordinates(ip)
        # Grab the geometric interpolation
        first_cell = getcells(grid_from, first(sdh.cellset))
        cell_type  = typeof(first_cell)
        gip        = Ferrite.geometric_interpolation(cell_type)

        _compute_dof_nodes_barrier!(
            nodes_from,
            sdh,
            Ferrite.dof_range(sdh, field_name_from),
            gip,
            dof_to_node_map_from,
            ref_coords,
            nothing,
        )
    end
    for sdh in dh_to.subdofhandlers[subdomains_to]
        # Skip subdofhandler if field is not present
        field_name_to ∈ Ferrite.getfieldnames(sdh) || continue
        # Grab the reference coordinates of the field to interpolate
        ip = Ferrite.getfieldinterpolation(sdh, field_name_to)
        ref_coords = Ferrite.reference_coordinates(ip)
        # Grab the geometric interpolation
        first_cell = getcells(grid_to, first(sdh.cellset))
        cell_type  = typeof(first_cell)
        gip        = Ferrite.geometric_interpolation(cell_type)

        _compute_dof_nodes_barrier!(
            nodes_to,
            sdh,
            Ferrite.dof_range(sdh, field_name_to),
            gip,
            dof_to_node_map_to,
            ref_coords,
        )
    end
    γf = zeros(length(node_to_dof_map_from))
    γg = zeros(length(node_to_dof_map_from))
    source_kdtree = KDTree(nodes_from)
    M = 15
    α = 2
    support_radii = maximum.(last.(knn.(Ref(source_kdtree), nodes_from, M)))
    distance_func = (x, xi, y, yi) -> norm.(Ref(x[xi]) .- (y[yi]))
    source_influence_matrix =
        build_sparse_matrix_kdtree(nodes_from, rbf_value, source_kdtree, support_radii, distance_func, α)
    destination_influence_matrix =
        construct_RBF_dist_kdtree(nodes_from, support_radii, nodes_to, rbf_value, distance_func, α)
    prob = LinearSolve.LinearProblem(source_influence_matrix, copy(γf))
    linsolve = LinearSolve.init(prob)

    RadialBasisFunctionTransferOperator{
        typeof(rescale),
        typeof(geodesic),
        typeof(dh_from),
        typeof(dh_to),
        typeof(source_influence_matrix),
        typeof(destination_influence_matrix),
        typeof((γf, γg)),
        typeof(linsolve),
        typeof(source_kdtree),
    }(
        dh_from,
        dh_to,
        node_to_dof_map_from,
        node_to_dof_map_to,
        dof_to_node_map_from,
        dof_to_node_map_to,
        field_name_from,
        field_name_to,
        source_influence_matrix,
        destination_influence_matrix,
        (γf, γg),
        linsolve,
        source_kdtree,
    )
end

function RadialBasisFunctionTransferOperator(
    dh_from::DofHandler{sdim},
    dh_to::DofHandler{sdim};
    rescale = Val(true),
    geodesic = Val(false),
) where {sdim}
    @assert length(Ferrite.getfieldnames(dh_from)) == 1 "Multiple fields found in source dof handler. Please specify which field you want to transfer."
    return RadialBasisFunctionTransferOperator(
        dh_from,
        dh_to,
        first(Ferrite.getfieldnames(dh_from));
        rescale = rescale,
        geodesic = geodesic,
    )
end

function RadialBasisFunctionTransferOperator(
    dh_from::DofHandler{sdim},
    dh_to::DofHandler{sdim},
    field_name::Symbol;
    rescale = Val(true),
    geodesic = Val(false),
) where {sdim}
    return RadialBasisFunctionTransferOperator(
        dh_from,
        dh_to,
        field_name,
        field_name;
        rescale = rescale,
        geodesic = geodesic,
    )
end

"""
    This is basically a fancy matrix-vector product to transfer the solution from one problem to another one.
"""
function transfer!(
    u_to::AbstractArray,
    operator::RadialBasisFunctionTransferOperator{Val{true}},
    u_from::AbstractArray,
)
    operator.source_linsolve_cache.b = u_from[operator.node_to_dof_map_from]
    sol = LinearSolve.solve!(operator.source_linsolve_cache)
    operator.interpolation_weights[1] .= sol.u
    operator.source_linsolve_cache.b = ones(length(operator.node_to_dof_map_from)) #TODO Cache this
    sol = LinearSolve.solve!(operator.source_linsolve_cache)
    operator.interpolation_weights[2] .= sol.u
    u_to[operator.node_to_dof_map_to] .=
        (operator.destination_influence_matrix * operator.interpolation_weights[1]) ./
        (operator.destination_influence_matrix * operator.interpolation_weights[2])
end

"""
    This is basically a fancy matrix-vector product to transfer the solution from one problem to another one.
"""
function transfer!(
    u_to::AbstractArray,
    operator::RadialBasisFunctionTransferOperator{Val{false}},
    u_from::AbstractArray,
)
    operator.source_linsolve_cache.b = u_from[operator.node_to_dof_map_from]
    sol = LinearSolve.solve!(operator.source_linsolve_cache)
    operator.interpolation_weights[1] .= sol.u
    u_to[operator.node_to_dof_map_to] .=
        (operator.destination_influence_matrix * operator.interpolation_weights[1])
end

"""
    This is basically a fancy matrix-vector product to transfer the solution from one problem to another one.
"""
function transfer!(
    u_to::AbstractArray,
    operator::NodalIntergridInterpolation,
    u_from::AbstractArray,
)
    # TODO non-allocating version
    u_to[operator.node_to_dof_map] .=
        Ferrite.evaluate_at_points(operator.ph, operator.dh_from, u_from, operator.field_name_from)
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

function OS.forward_sync_external!(
    outer_integrator::OS.OperatorSplittingIntegrator,
    inner_integrator::SciMLBase.DEIntegrator,
    sync::VolumeTransfer0D3D,
)
    # Tying holds a buffer for the 3D problem with some meta information about the 0D problem
    for chamber ∈ sync.tying.chambers
        chamber.V⁰ᴰval = outer_integrator.u[chamber.V⁰ᴰidx_global]
    end
end
function OS.backward_sync_external!(
    outer_integrator::OS.OperatorSplittingIntegrator,
    inner_integrator::SciMLBase.DEIntegrator,
    sync::VolumeTransfer0D3D,
)
    nothing
end

"""
    Utility function to synchronize the pressire in a split [`RSAFDQ2022Function`](@ref)
"""
struct PressureTransfer3D0D{TP} <: AbstractTransferOperator
    tying::TP
end

function OS.forward_sync_external!(
    outer_integrator::OS.OperatorSplittingIntegrator,
    inner_integrator::SciMLBase.DEIntegrator,
    sync::PressureTransfer3D0D,
)
    # Tying holds a buffer for the 3D problem with some meta information about the 0D problem
    for chamber ∈ sync.tying.chambers
        pressure = outer_integrator.u[chamber.pressure_dof_index_global]
        inner_integrator.p[chamber.pressure_parameter_index_local] = pressure
    end
end
function OS.backward_sync_external!(
    outer_integrator::OS.OperatorSplittingIntegrator,
    inner_integrator::SciMLBase.DEIntegrator,
    sync::PressureTransfer3D0D,
)
    nothing
end
