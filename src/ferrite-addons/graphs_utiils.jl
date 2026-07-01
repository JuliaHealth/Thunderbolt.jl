using Graphs: AbstractGraph

"""
    precompute_dijkstra_with_cutoffs(
        source_graph, all_nodes, support_radii, α, edge_lengths_matrix; parallel=:threads
    )

Pre-compute Dijkstra paths from all nodes with individual cutoff distances.

For each node `xi`, compute distances up to `support_radii[xi] * α`.
"""
function precompute_dijkstra_with_cutoffs(
    source_graph::AbstractGraph,
    all_nodes::Vector{Int},
    support_radii::Vector{T},
    α::T,
    edge_lengths_matrix::AbstractMatrix{T};
) where {T <: Number}

    n_nodes = length(all_nodes)
    results = Vector{Vector{T}}(undef, n_nodes)

    Threads.@threads for i = 1:n_nodes
        node = all_nodes[i]
        maxdist = support_radii[i] * α

        state = dijkstra_shortest_paths(source_graph, node, edge_lengths_matrix; maxdist = maxdist)
        results[i] = state.dists
    end

    # Convert to dictionary after all threads complete
    dijkstra_cache = Dict(all_nodes[i] => results[i] for i = 1:n_nodes)
    return dijkstra_cache
end
