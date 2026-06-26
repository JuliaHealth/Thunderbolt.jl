include("graphs_utiils.jl")

abstract type AbstractTransferOperator end

abstract type AbstractDistanceMeasure end

abstract type AbstractDistanceMeasureCache end

abstract type AbstractFieldTransferEvaluation end

abstract type AbstractFieldTransferEvaluationCache end

abstract type AbstractRadialBasisFunctionTransferEvaluation <: AbstractFieldTransferEvaluation end

struct IntergridDofMapping{VecT <: AbstractVector{<:Real}, IntT <: Integer}
    nodes_from::Vector{VecT}
    nodes_to::Vector{VecT}
    node_to_dof_map_from::Vector{IntT}
    node_to_dof_map_to::Vector{IntT}
    dof_to_node_map_from::Dict{IntT, IntT}
    dof_to_node_map_to::Dict{IntT, IntT}
end

struct EuclideanDistanceMeasure <: AbstractDistanceMeasure
    M::Int
    α::Float64
end

struct EuclideanDistanceMeasureCache <: AbstractDistanceMeasureCache
    M::Int
    α::Float64
end

function create_distance_measure_cache(distance_measure::EuclideanDistanceMeasure, argv...)
    return EuclideanDistanceMeasureCache(distance_measure.M, distance_measure.α)
end

function (::EuclideanDistanceMeasureCache)(x, xi, y, yi)
    return norm(x[xi] - y[yi])
end

struct NullDistanceMeasure <: AbstractDistanceMeasure end

struct NullDistanceMeasureCache <: AbstractDistanceMeasureCache end

function create_distance_measure_cache(::NullDistanceMeasure, argv...)
    return NullDistanceMeasureCache()
end

struct GeodesicDistanceMeasure <: AbstractDistanceMeasure
    M::Int
    α::Float64
    β::Float64
end

struct GeodesicDistanceMeasureCache{
    KDTreeT,
    GraphT,
    FloatT <: Real,
    EdgeLenghtMatrixT <: AbstractMatrix,
    DijkstraT,
} <: AbstractDistanceMeasureCache
    source_kdtree::KDTreeT
    source_graph::GraphT
    support_radii::Vector{FloatT}
    edge_lengths_matrix::EdgeLenghtMatrixT
    M::Int
    α::Float64
    β::Float64
    h_max::Float64
    dijkstra_cache::DijkstraT
end

function create_distance_measure_cache(
    source_kdtree,
    source_graph,
    support_radii,
    edge_lengths_matrix,
    M,
    α,
    β,
)
    n_nodes = nv(source_graph)
    h_max = maximum(edge_lengths_matrix)

    # Parallel pre-computation with per-node cutoffs
    dijkstra_cache = precompute_dijkstra_with_cutoffs(
        source_graph,
        collect(1:n_nodes),
        support_radii,
        α,
        edge_lengths_matrix;
        parallel = true,
    )

    return GeodesicDistanceMeasureCache(
        source_kdtree,
        source_graph,
        support_radii,
        edge_lengths_matrix,
        M,
        α,
        β,
        h_max,
        dijkstra_cache,
    )
end

function create_distance_measure_cache(
    distance_measure::GeodesicDistanceMeasure,
    source_kdtree,
    source_graph,
    support_radii,
    mapping,
    dh_from,
)
    edge_lengths_matrix = allocate_matrix(dh_from)
    for i in mapping.node_to_dof_map_from
        for j in mapping.node_to_dof_map_from
            edge_lengths_matrix[i, j] = norm(
                mapping.nodes_from[mapping.dof_to_node_map_from[i]] -
                mapping.nodes_from[mapping.dof_to_node_map_from[j]],
            )
        end
    end
    return create_distance_measure_cache(
        source_kdtree,
        source_graph,
        support_radii,
        edge_lengths_matrix,
        distance_measure.M,
        distance_measure.α,
        distance_measure.β,
    )
end

# Fast O(1) lookup in measure function
function (measure::GeodesicDistanceMeasureCache)(x, xi, y, yi)
    (; source_kdtree, support_radii, h_max, α, β, dijkstra_cache) = measure

    coords_destination = y[yi]
    nearest_node_in_source, distance_to_neighbor = nn(source_kdtree, coords_destination)
    norm_distance = norm(x[xi] - y[yi])
    dijkstra_cutoff_distance = support_radii[xi] * α

    # O(1) cache lookup
    nn_distance = dijkstra_cache[nearest_node_in_source][xi] + distance_to_neighbor

    ret =
        nn_distance <= (norm_distance + β * h_max) ? norm_distance :
        (nn_distance < dijkstra_cutoff_distance ? nn_distance : Inf)
    return ret
end

struct RadialBasisFunctionEvaluation{DistanceMeasureT <: AbstractDistanceMeasure, SLinsolveCT} <:
       AbstractRadialBasisFunctionTransferEvaluation
    distance_measure::DistanceMeasureT
    source_linsolve::SLinsolveCT
end

struct RescaledRadialBasisFunctionEvaluation{
    DistanceMeasureT <: AbstractDistanceMeasure,
    SLinsolveCT,
} <: AbstractRadialBasisFunctionTransferEvaluation
    distance_measure::DistanceMeasureT
    source_linsolve::SLinsolveCT
end

struct RadialBasisFunctionEvaluationCache{
    DistanceMeasureT <: AbstractDistanceMeasureCache,
    SIMT <: AbstractMatrix,
    DIMT <: AbstractMatrix,
    γT <: AbstractVector,
    SLinsolveCT,
} <: AbstractFieldTransferEvaluationCache
    distance_measure::DistanceMeasureT
    source_influence_matrix::SIMT
    destination_influence_matrix::DIMT
    γf::γT
    source_linsolve_cache::SLinsolveCT
end

struct RescaledRadialBasisFunctionEvaluationCache{
    DistanceMeasureT <: AbstractDistanceMeasureCache,
    SIMT <: AbstractMatrix,
    DIMT <: AbstractMatrix,
    γT <: AbstractVector,
    SLinsolveCT,
} <: AbstractFieldTransferEvaluationCache
    distance_measure::DistanceMeasureT
    source_influence_matrix::SIMT
    destination_influence_matrix::DIMT
    γf::γT
    γg::γT
    source_linsolve_cache::SLinsolveCT
end

function create_field_transfer_eval_cache(
    evaluator_type::AbstractRadialBasisFunctionTransferEvaluation,
    mapping::IntergridDofMapping,
    dh_from::DofHandler,
    dh_to::DofHandler,
)
    M = evaluator_type.distance_measure.M
    α = evaluator_type.distance_measure.α
    source_linsolve = evaluator_type.source_linsolve
    distances_source = allocate_matrix(SparseMatrixCSC{Bool, Int}, dh_from)
    distances_source.nzval .= 1.0

    # The "nodal" values projected into the RBF space.
    γf = zeros(length(mapping.node_to_dof_map_from))

    source_kdtree = KDTree(mapping.nodes_from)
    # Build graph from adjacency, we index here because we might get unwanted
    # connectivity if we don't use all sdhs.
    source_graph =
        SimpleGraph(distances_source[mapping.node_to_dof_map_from, mapping.node_to_dof_map_from])
    Nsrc = length(mapping.nodes_from)
    support_radii = zeros(Float64, Nsrc)
    for src = 1:Nsrc
        hop_dists = neighborhood_dists(source_graph, src, M)
        dist = maximum(x->last(x), hop_dists)
        support_radii[src] = dist
    end
    distance_measure_cache = create_distance_measure_cache(
        evaluator_type.distance_measure,
        source_kdtree,
        source_graph,
        support_radii,
        mapping,
        dh_from,
    )
    source_influence_matrix = build_sparse_matrix_kdtree(
        mapping.nodes_from,
        rbf_value,
        source_kdtree,
        support_radii,
        distance_measure_cache,
        α,
    )
    destination_influence_matrix = construct_RBF_dist_kdtree(
        mapping.nodes_from,
        support_radii,
        mapping.nodes_to,
        rbf_value,
        distance_measure_cache,
        α,
    )
    prob = LinearSolve.LinearProblem(source_influence_matrix, copy(γf))
    linsolve = LinearSolve.init(prob, source_linsolve)

    return _create_field_transfer_eval_cache(
        evaluator_type,
        distance_measure_cache,
        source_influence_matrix,
        destination_influence_matrix,
        γf,
        linsolve,
    )
end

function _create_field_transfer_eval_cache(
    ::RescaledRadialBasisFunctionEvaluation,
    distance_measure_cache,
    source_influence_matrix,
    destination_influence_matrix,
    γf,
    linsolve,
)
    γg = zeros(length(γf))
    return RescaledRadialBasisFunctionEvaluationCache(
        distance_measure_cache,
        source_influence_matrix,
        destination_influence_matrix,
        γf,
        γg,
        linsolve,
    )
end


function _create_field_transfer_eval_cache(
    ::RadialBasisFunctionEvaluation,
    distance_measure_cache,
    source_influence_matrix,
    destination_influence_matrix,
    γf,
    linsolve,
)
    return RadialBasisFunctionEvaluationCache(
        distance_measure_cache,
        source_influence_matrix,
        destination_influence_matrix,
        γf,
        linsolve,
    )
end

struct NodalIntergridEvaluation <: AbstractFieldTransferEvaluation end

struct NodalIntergridEvaluationCache{PointEvalHandlerT <: PointEvalHandler} <:
       AbstractFieldTransferEvaluationCache
    ph::PointEvalHandlerT
end

function create_field_transfer_eval_cache(
    ::NodalIntergridEvaluation,
    mapping::IntergridDofMapping,
    dh_from::DofHandler,
    dh_to::DofHandler,
)
    ph = PointEvalHandler(Ferrite.get_grid(dh_from), mapping.nodes_to; warn = false)
    n_missing = sum(x -> x === nothing, ph.cells)
    n_missing == 0 ||
        @warn "Constructing the nodal intergrid interpolation failed. $n_missing (out of $(length(ph.cells))) points not found."
    return NodalIntergridEvaluationCache(ph)
end

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

function _compute_dof_nodes_barrier!(nodes, sdh, dofrange, gip, dof_to_node_map, ref_coords)
    for cc in CellIterator(sdh)
        # Compute for each dof the spatial coordinate of from the reference coordiante and store.
        # NOTE We assume a continuous coordinate field if the interpolation is continuous.
        dofs = @view celldofs(cc)[dofrange]
        for (dofidx, dof) in enumerate(dofs)
            nodes[dof_to_node_map[dof]] =
                spatial_coordinate(gip, ref_coords[dofidx], getcoordinates(cc))
        end
    end
end

function IntergridDofMapping(
    dh_from::DofHandler{sdim},
    dh_to::DofHandler{sdim},
    field_name_from::Symbol,
    field_name_to::Symbol;
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
        for cellidx in sdh.cellset
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
        for cellidx in sdh.cellset
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
    for dof in node_to_dof_map_to
        dof_to_node_map_to[dof] = next_dof_index
        next_dof_index += 1
    end

    next_dof_index = 1
    for dof in node_to_dof_map_from
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
    return IntergridDofMapping(
        nodes_from,
        nodes_to,
        node_to_dof_map_from,
        node_to_dof_map_to,
        dof_to_node_map_from,
        dof_to_node_map_to,
    )
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


function NodalIntergridInterpolation(args...; kwargs...)
    FieldTransferOperator(args..., NodalIntergridEvaluation(); kwargs...)
end

@inline function rbf_value(d, r)
    (max((1 - d / r), zero(r))^4) * (1 + 4d / r)
end

function build_sparse_matrix_kdtree(coords_vec, rbf_func, tree, distances, distance_func, α = 2.0)
    N = length(coords_vec)

    # First pass: estimate number of nonzeros to pre-size arrays
    nnz_est = 0
    for j = 1:N
        radius = distances[j] * α
        idxs = inrange(tree, coords_vec[j], radius)
        nnz_est += length(idxs)
    end

    rows = zeros(Int, nnz_est)
    cols = zeros(Int, nnz_est)
    vals = zeros(Float64, nnz_est)
    i__ = 1
    # Second pass: fill arrays
    @views for j = 1:N
        radius = distances[j] * α
        # Query all points within radius around point j (including j itself)
        idxs = inrange(tree, coords_vec[j], radius)
        for i in idxs
            d = distance_func(coords_vec, j, coords_vec, i)
            # No need to check d < radius – guaranteed by inrange
            val = rbf_func(d, radius)
            rows[i__] = i
            cols[i__] = j
            vals[i__] = val
            i__ += 1
        end
    end
    # resize!(rows, length(rows))
    # resize!(cols, length(cols))
    # resize!(vals, length(vals))
    # Build CSC matrix; duplicate entries (if any) will be summed automatically
    A = SparseArrays.sparse!(rows, cols, vals)
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

    # Build KD‑tree on destination points (used to query points within radius)
    tree = KDTree(coords_dist)

    # First pass: estimate number of nonzeros
    nnz_est = 0
    for j = 1:N_src
        radius = distances[j] * α
        idxs = inrange(tree, coords_src[j], radius)
        nnz_est += length(idxs)
    end

    rows = zeros(Int, nnz_est)
    cols = zeros(Int, nnz_est)
    vals = zeros(Float64, nnz_est)
    i__ = 1
    # Second pass: fill arrays
    @views @inbounds for j = 1:N_src
        radius = distances[j] * α
        idxs = inrange(tree, coords_src[j], radius)
        for i in idxs
            d = distance_func(coords_src, j, coords_dist, i)
            val = rbf_func(d, radius)
            rows[i__] = i
            cols[i__] = j
            vals[i__] = val
            i__ += 1
        end
    end
    # Build CSC matrix of size N_dst × N_src
    A = SparseArrays.sparse!(rows, cols, vals, length(coords_dist), length(coords_src))
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
struct FieldTransferOperator{
    EvaluationCacheT <: AbstractFieldTransferEvaluationCache,
    DH1 <: AbstractDofHandler,
    DH2 <: AbstractDofHandler,
    IntergridDofMappingT <: IntergridDofMapping,
} <: AbstractTransferOperator
    dh_from::DH1
    dh_to::DH2
    mapping::IntergridDofMappingT
    field_name_from::Symbol
    field_name_to::Symbol
    evaluation_cache::EvaluationCacheT
end

function FieldTransferOperator(
    dh_from::DofHandler{sdim},
    dh_to::DofHandler{sdim},
    field_name_from::Symbol,
    field_name_to::Symbol,
    evaluation_cache_type::AbstractFieldTransferEvaluation;
    subdomains_from = 1:length(dh_from.subdofhandlers),
    subdomains_to = 1:length(dh_to.subdofhandlers),
) where {sdim}

    # These are the "nodes" placed at the support points of the Dofs.
    mapping = IntergridDofMapping(
        dh_from,
        dh_to,
        field_name_from,
        field_name_to;
        subdomains_from = subdomains_from,
        subdomains_to = subdomains_to,
    )
    evaluation_cache =
        create_field_transfer_eval_cache(evaluation_cache_type, mapping, dh_from, dh_to)

    FieldTransferOperator{typeof(evaluation_cache), typeof(dh_from), typeof(dh_to), typeof(mapping)}(
        dh_from,
        dh_to,
        mapping,
        field_name_from,
        field_name_to,
        evaluation_cache,
    )
end

function FieldTransferOperator(
    dh_from::DofHandler{sdim},
    dh_to::DofHandler{sdim},
    evaluation_cache_type::AbstractFieldTransferEvaluation,
) where {sdim}
    @assert length(Ferrite.getfieldnames(dh_from)) == 1 "Multiple fields found in source dof handler. Please specify which field you want to transfer."
    return FieldTransferOperator(
        dh_from,
        dh_to,
        first(Ferrite.getfieldnames(dh_from)),
        evaluation_cache_type,
    )
end

function FieldTransferOperator(
    dh_from::DofHandler{sdim},
    dh_to::DofHandler{sdim},
    field_name::Symbol,
    evaluation_cache_type::AbstractFieldTransferEvaluation,
) where {sdim}
    return FieldTransferOperator(dh_from, dh_to, field_name, field_name, evaluation_cache_type)
end

"""
    This is basically a fancy matrix-vector product to transfer the solution from one problem to another one.
"""
function transfer!(
    u_to::AbstractArray,
    operator::FieldTransferOperator{<:NodalIntergridEvaluationCache},
    u_from::AbstractArray,
)
    # TODO non-allocating version
    u_to[operator.mapping.node_to_dof_map_to] .= Ferrite.evaluate_at_points(
        operator.evaluation_cache.ph,
        operator.dh_from,
        u_from,
        operator.field_name_from,
    )
end

"""
    This is basically a fancy matrix-vector product to transfer the solution from one problem to another one.
"""
function transfer!(
    u_to::AbstractArray,
    operator::FieldTransferOperator{<:RescaledRadialBasisFunctionEvaluationCache},
    u_from::AbstractArray,
)
    operator.evaluation_cache.source_linsolve_cache.b .=
        (@view u_from[operator.mapping.node_to_dof_map_from])
    sol = LinearSolve.solve!(operator.evaluation_cache.source_linsolve_cache)
    operator.evaluation_cache.γf .= sol.u
    operator.evaluation_cache.source_linsolve_cache.b .= 1.0
    sol = LinearSolve.solve!(operator.evaluation_cache.source_linsolve_cache)
    operator.evaluation_cache.γg .= sol.u
    u_to[operator.mapping.node_to_dof_map_to] .=
        (operator.evaluation_cache.destination_influence_matrix * operator.evaluation_cache.γf) ./
        (operator.evaluation_cache.destination_influence_matrix * operator.evaluation_cache.γg)
end

"""
    This is basically a fancy matrix-vector product to transfer the solution from one problem to another one.
"""
function transfer!(
    u_to::AbstractArray,
    operator::FieldTransferOperator{<:RadialBasisFunctionEvaluationCache},
    u_from::AbstractArray,
)
    operator.evaluation_cache.source_linsolve_cache.b .=
        (@view u_from[operator.mapping.node_to_dof_map_from])
    sol = LinearSolve.solve!(operator.evaluation_cache.source_linsolve_cache)
    operator.evaluation_cache.γf .= sol.u
    u_to[operator.mapping.node_to_dof_map_to] .=
        (operator.evaluation_cache.destination_influence_matrix * operator.evaluation_cache.γf)
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
    for chamber in sync.tying.chambers
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
    for chamber in sync.tying.chambers
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
