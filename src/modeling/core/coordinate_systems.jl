abstract type CoordinateSystemCoefficient end

"""
    LocalCoordinateAxes(transmural, apicobasal, rotational)

The right-handed orthonormal frame that a ventricular coordinate system induces *at one point* --
as opposed to [`LVAxes`](@ref), which is the anatomical frame of the chamber as a whole.

Each direction is the one in which the corresponding coordinate increases. Query it with
[`evaluate_coordinate_axes`](@ref), which is to the frame what [`evaluate_coefficient`](@ref) is to
the coordinate values themselves.
"""
struct LocalCoordinateAxes{T}
    transmural::Vec{3, T}
    apicobasal::Vec{3, T}
    rotational::Vec{3, T}
end

"""
Build the local frame from the gradients of the transmural and apicobasal coordinates.

The transmural direction is taken as given and the apicobasal one is orthogonalized against it. The
two coordinate fields are not orthogonal on any real anatomy, so without that step the frame would
not be one, and the microstructure rotations built on it would not preserve angles.

The rotational direction is what is left over rather than `∇rotational`: the rotational coordinate
jumps across the seam and does not exist on the long axis, so its gradient is unusable exactly where
the other two are fine.
"""
function _local_axes(∇transmural::Vec{3}, ∇apicobasal::Vec{3})
    transmural = ∇transmural / norm(∇transmural)
    apicobasal = orthogonalize(∇apicobasal / norm(∇apicobasal), transmural)
    apicobasal /= norm(apicobasal)
    rotational = transmural × apicobasal
    return LocalCoordinateAxes(transmural, apicobasal, rotational / norm(rotational))
end

"""
    CartesianCoordinateSystem(mesh)

Standard cartesian coordinate system.
"""
struct CartesianCoordinateSystem{sdim, T} <: CoordinateSystemCoefficient
    function CartesianCoordinateSystem{sdim}() where {sdim}
        return new{sdim, Float32}()
    end
end

value_type(::CartesianCoordinateSystem{sdim, T}) where {sdim, T} = Vec{sdim, T}

CartesianCoordinateSystem(mesh::AbstractGrid{sdim}) where {sdim} = CartesianCoordinateSystem{sdim}()

"""
    getcoordinateinterpolation(cs::CartesianCoordinateSystem, cell::AbstractCell)

Get interpolation function for the cartesian coordinate system.
"""
getcoordinateinterpolation(
    cs::CartesianCoordinateSystem{sdim},
    cell::CellType,
) where {sdim, CellType <: AbstractCell} = Ferrite.geometric_interpolation(CellType)^sdim


"""
    CellIndexCoordinateSystem()

The cell index as a coordinate.

Useful where a cell model varies by element rather than by position -- a 1D Purkinje network whose cells
are distinguished by identity, or heterogeneity tabulated per element. It is a coefficient like any other,
so it composes with [`AnalyticalCoefficient`](@ref) and can be handed to a `MonodomainModel` as its
`cell_coordinates` alongside a subdomain that uses a generalized coordinate instead.
"""
struct CellIndexCoordinateSystem <: CoordinateSystemCoefficient end

value_type(::CellIndexCoordinateSystem) = Int

"""
    LVCoordinateSystem(dh, ip_collection, u_transmural, u_apicobasal, dh_rotational, ip_collection_rotational, u_rotational)

Simplified universal ventricular coordinate on LV only, containing the transmural, apicobasal and
rotational coordinates. See [`compute_lv_coordinate_system`](@ref) to construct it.

The transmural and apicobasal coordinates are continuous scalar fields and live on `dh`. The
rotational coordinate wraps around the ventricle, so somewhere it has to jump from 1 back to 0 --
on the posterior ridge, where the free wall reads 0 and the septum reads 1. A continuous nodal field
cannot carry that jump: the ridge nodes are shared by the elements on both sides of it, so a single
nodal value there forces the coordinate to ramp back through the entire range across one layer of
elements. It is therefore stored on its own discontinuous dof handler `dh_rotational`, where each
element carries its own copy of the ridge nodes and the jump sits exactly on the interface it
belongs on.

Consequently `u_rotational` is *not* dof-for-dof comparable with `u_transmural` and `u_apicobasal`,
and it is only defined modulo 1: under the azimuthal fallback an element straddling the branch cut
holds values slightly above 1 (or below 0) so that its interpolant stays affine.
`evaluate_coefficient` wraps the interpolated value back into `[0, 1)`.
"""
struct LVCoordinateSystem{T, DH <: AbstractDofHandler, IPC, DHR <: AbstractDofHandler, IPCR} <:
       CoordinateSystemCoefficient
    dh::DH
    ip_collection::IPC # TODO special dof handler with type stable interpolation
    u_transmural::Vector{T}
    u_apicobasal::Vector{T}
    dh_rotational::DHR
    ip_collection_rotational::IPCR
    u_rotational::Vector{T}
end


"""
    LVCoordinate{T}

LV only part of the universal ventricular coordinate, containing
    * transmural
    * apicobasal
    * rotational
"""
Base.@kwdef struct LVCoordinate{T}
    transmural::T
    apicobasal::T
    rotational::T
end
Base.zero(::Type{LVCoordinate{T}}) where {T} = LVCoordinate(T(0.0), T(0.0), T(0.0))
Base.eltype(::Type{LVCoordinate{T}}) where {T} = T
Base.eltype(::LVCoordinate{T}) where {T} = T
value_type(::LVCoordinateSystem{T}) where {T} = LVCoordinate{T}


"""
    getcoordinateinterpolation(cs::LVCoordinateSystem, cell::AbstractCell)

Get interpolation function for the LV coordinate system.
"""
getcoordinateinterpolation(cs::LVCoordinateSystem, cell::AbstractCell) =
    getinterpolation(cs.ip_collection, cell)

"""
    getrotationalinterpolation(cs::LVCoordinateSystem, cell::AbstractCell)

Get the interpolation of the rotational coordinate, which is discontinuous across elements -- see
[`LVCoordinateSystem`](@ref).
"""
getrotationalinterpolation(cs::LVCoordinateSystem, cell::AbstractCell) =
    getinterpolation(cs.ip_collection_rotational, cell)

"Wrap an angular coordinate back into `[0, 1)`."
@inline wrap_rotational(r::T) where {T <: Real} = mod(r, one(T))

"""
Assemble the scalar Laplacian on all subdomains of `dh`.
"""
function _assemble_laplacian(dh::DofHandler, ip_collection)
    cv_collection = CellValueCollection(QuadratureRuleCollection(2), ip_collection)
    K = allocate_matrix(dh)
    assembler = start_assemble(K)
    for sdh in dh.subdofhandlers
        cellvalues = getcellvalues(cv_collection, getcells(get_grid(dh), first(sdh.cellset)))
        n_basefuncs = getnbasefunctions(cellvalues)
        Ke = zeros(n_basefuncs, n_basefuncs)
        @inbounds for cell in CellIterator(sdh)
            fill!(Ke, 0)
            reinit!(cellvalues, cell)
            for qp in QuadratureIterator(cellvalues)
                dΩ = getdetJdV(cellvalues, qp)
                for i = 1:n_basefuncs
                    ∇v = shape_gradient(cellvalues, qp, i)
                    for j = 1:n_basefuncs
                        ∇u = shape_gradient(cellvalues, qp, j)
                        Ke[i, j] += (∇v ⋅ ∇u) * dΩ
                    end
                end
            end
            assemble!(assembler, celldofs(cell), Ke)
        end
    end
    return K
end

"""
Solve `Δu = 0` on the single scalar field of `dh` with the given Dirichlet data, each entry a
`(set, value)` pair where the set is either a facetset or a nodeset. `K` is the (unconstrained)
Laplacian and is left untouched.
"""
function _solve_dirichlet_laplace(
    K,
    dh::DofHandler,
    solver,
    constraints;
    field_name::Symbol = first(Ferrite.getfieldnames(dh)),
)
    ch = ConstraintHandler(dh)
    for (set, value) in constraints
        Ferrite.add!(ch, Dirichlet(field_name, set, (x, t) -> value))
    end
    close!(ch)
    update!(ch, 0.0)

    A = copy(K)
    f = zeros(ndofs(dh))
    apply!(A, f, ch)
    u = solve(LinearSolve.LinearProblem(A, f), solver).u
    # An iterative solver only reaches the prescribed values to its own tolerance, which leaves a
    # coordinate that is 1 - 3e-10 on the surface it is pinned to. Write them back exactly: the
    # whole point of a harmonic coordinate is that it attains its endpoints on the annotated
    # surfaces, on every mesh and with every solver.
    apply!(u, ch)
    return u
end

"""
Lumped L² projection of `∇u` onto the dofs of `dh`, together with the lumped mass `mᵢ = ∫ ϕᵢ dΩ`
that weights it. Both are returned in dof order. Writing the mass as a quadrature rather than as a
per-cell volume share is what lets this serve tetrahedral, hexahedral and wedge meshes alike.
"""
function _lumped_gradient(dh::DofHandler, ip_collection, u::AbstractVector{<:Real})
    cv_collection = CellValueCollection(QuadratureRuleCollection(2), ip_collection)
    grad = [zero(Vec{3, Float64}) for _ = 1:ndofs(dh)]
    mass = zeros(Float64, ndofs(dh))
    for sdh in dh.subdofhandlers
        cellvalues = getcellvalues(cv_collection, getcells(get_grid(dh), first(sdh.cellset)))
        ue = zeros(Float64, getnbasefunctions(cellvalues))
        @inbounds for cell in CellIterator(sdh)
            reinit!(cellvalues, cell)
            dofs = celldofs(cell)
            for i in eachindex(dofs)
                ue[i] = u[dofs[i]]
            end
            for qp in QuadratureIterator(cellvalues)
                dΩ = getdetJdV(cellvalues, qp)
                ∇u = function_gradient(cellvalues, qp, ue)
                for (k, d) in enumerate(dofs)
                    w = shape_value(cellvalues, qp, k) * dΩ
                    grad[d] += ∇u * w
                    mass[d] += w
                end
            end
        end
    end
    @inbounds for i in eachindex(grad)
        mass[i] > 0 && (grad[i] /= mass[i])
    end
    return grad, mass
end

"""
    apicobasal_from_laplace(dh, ip_collection, u; nbins = 200)

Recalibrate the apicobasal Laplace field `u` (0 on the apex, 1 on the base, natural boundary
conditions on the endocardium and epicardium) to arc length along its own trajectories.

Along a trajectory of `u` we have `ds = du/‖∇u‖`, so the arc length from level `u` to the base is
`∫_u^1 dū/‖∇u‖`. Averaging `‖∇u‖` over each level set turns that into a one-dimensional quadrature,

    F(u) = ∫_u^1 dū / ⟨‖∇u‖⟩(ū)        ab = 1 − F(u)/F(0)

Three properties follow, and together they are what the coordinate system needs:

- `ab` is a monotone function of `u`, so `∇ab ∥ ∇u`. The chart
  `(transmural, rotational, apicobasal)` can never reverse orientation through the apicobasal
  coordinate.
- `ab` is exactly 1 on the base and exactly 0 on the apex Dirichlet set of `u`, on every mesh, so
  two meshes of different anatomies agree on where the ends of the coordinate are.
- The parametrisation is arc length, not the harmonic field's own bunching. That bunching is severe:
  pinning `u = 0` at the apex is a point Dirichlet condition in 3D, whose solution behaves like a
  point source, so the raw field sits near 1 over most of the ventricle and burns half its range in
  the last few percent of wall next to the apex.
"""
function apicobasal_from_laplace(
    dh::DofHandler,
    ip_collection,
    u_laplace::AbstractVector{<:Real};
    nbins::Int = 200,
)
    u = clamp.(Float64.(u_laplace), 0.0, 1.0)
    grad, weight = _lumped_gradient(dh, ip_collection, u)
    gradnorm = norm.(grad)

    # ⟨‖∇u‖⟩ per level-set bin, volume weighted.
    edges = range(0.0, 1.0; length = nbins + 1)
    du = 1.0 / nbins
    bin(v) = clamp(searchsortedlast(edges, v), 1, nbins)
    num = zeros(nbins)
    den = zeros(nbins)
    for i in eachindex(u)
        b = bin(u[i])
        num[b] += weight[i] * gradnorm[i]
        den[b] += weight[i]
    end
    # Bins with no dofs inherit a populated neighbour so the quadrature never divides by zero;
    # sweeping both ways covers empty bins at either end.
    g = [den[b] > 0 ? num[b] / den[b] : 0.0 for b = 1:nbins]
    for b = 2:nbins
        g[b] == 0 && (g[b] = g[b-1])
    end
    for b = (nbins-1):-1:1
        g[b] == 0 && (g[b] = g[b+1])
    end

    F = zeros(nbins + 1)
    for b = nbins:-1:1
        F[b] = F[b+1] + du / max(g[b], eps())
    end
    total = F[1]
    total > 0 || return fill(0.0, length(u))

    return [
        begin
            b = bin(ui)
            λ = (ui - edges[b]) / du
            clamp(1.0 - (F[b] + λ * (F[b+1] - F[b])) / total, 0.0, 1.0)
        end for ui in u
    ]
end

_has_facetset(grid, name::String) = haskey(Ferrite.getfacetsets(grid), name)

"""
Cylindrical frame around the long axis of a ventricle. `e₁` is the direction of azimuth 0 and
`e₁ × e₂ = axis`, so the azimuth increases in the right-handed sense around the long axis.
"""
struct AzimuthalFrame
    origin::Vec{3, Float64}
    axis::Vec{3, Float64}
    e₁::Vec{3, Float64}
    e₂::Vec{3, Float64}
end

function AzimuthalFrame(
    origin::Vec{3, Float64},
    axis::Vec{3, Float64},
    zero_direction::Vec{3, Float64},
)
    l = axis / norm(axis)
    e₁ = orthogonalize(zero_direction, l)
    norm(e₁) < sqrt(eps(Float64)) && throw(
        ArgumentError("The azimuth reference direction must not be collinear with the long axis."),
    )
    e₁ /= norm(e₁)
    return AzimuthalFrame(origin, l, e₁, l × e₁)
end

"Azimuth of `x` in `[0, 2π)`, or `nothing` if `x` sits on the long axis where it does not exist."
function azimuth(frame::AzimuthalFrame, x::Vec{3, Float64}, tol::Float64 = 0.0)
    d = orthogonalize(x - frame.origin, frame.axis)
    norm(d) ≤ tol && return nothing
    return mod(atan(d ⋅ frame.e₂, d ⋅ frame.e₁), 2π)
end

"""
Mean radial direction of the facets in `name`, or `nothing` when the set is missing, empty or not a
radial sheet. Averaging the unit directions rather than the angles is what makes this immune to the
wrap of the azimuth.
"""
function _sheet_direction(grid::AbstractGrid, name::String, frame::AzimuthalFrame)
    _has_facetset(grid, name) || return nothing
    coords = _node_coordinates(grid)
    direction = zero(Vec{3, Float64})
    for nodeid in _facetset_node_ids(grid, getfacetset(grid, name))
        d = orthogonalize(Vec{3, Float64}(coords[nodeid]) - frame.origin, frame.axis)
        norm(d) < eps() && continue
        direction += d / norm(d)
    end
    norm(direction) < 0.5 && return nothing
    return direction / norm(direction)
end

"Is `θ` inside the shorter of the two arcs delimited by `θa` and `θp`?"
function _in_minor_arc(θ::Float64, θa::Float64, θp::Float64)
    Δ = mod(θp - θa, 2π)
    if Δ ≤ π
        return mod(θ - θa, 2π) ≤ Δ
    else
        return mod(θ - θp, 2π) ≤ 2π - Δ
    end
end

"Volume-average position of a cell."
function _cell_centroid(grid::AbstractGrid, cellid::Int)
    coords = _node_coordinates(grid)
    nodeids = getcells(grid, cellid).nodes
    return Vec{3, Float64}(sum(coords[nodeid] for nodeid in nodeids) / length(nodeids))
end

"""
Split the cells into septum and free wall by the arc they fall into.

The septum is the wall between the two right ventricular insertions -- the shorter of the two arcs,
since it spans roughly a third of the circumference.

This is a fallback. It decides each cell from its centroid alone, so cells sitting within half an
element of a ridge can go either way, and there it matters: the rotational coordinate jumps by a
full turn across the posterior ridge, so a misclassified cell carries `r ≈ 1` where its neighbours
carry `r ≈ 0`. Prefer [`_septal_cells_by_partition`](@ref), which does not guess.
"""
function _septal_cells_by_arc(
    grid::AbstractGrid,
    frame::AzimuthalFrame,
    θ_anterior::Float64,
    θ_posterior::Float64,
)
    septal = falses(getncells(grid))
    for cellid = 1:getncells(grid)
        θ = azimuth(frame, _cell_centroid(grid, cellid))
        θ === nothing && continue
        septal[cellid] = _in_minor_arc(θ, θ_anterior, θ_posterior)
    end
    return septal
end

"Unique key of one facet, identifying it independently of which cell it is seen from."
_facet_key(cell::AbstractCell, local_facet::Int) =
    Ferrite.sortfacet_fast(Ferrite.facets(cell)[local_facet])

"""
Split the cells into septum and free wall by the partition the ridges induce, or `nothing` if the
ridges do not separate the mesh.

The two ridge sheets *are* the interface between the septum and the free wall, so they cut the
myocardium in two and the septum is one of the halves -- no geometry needed, and nothing to get
wrong at the ridges themselves. Which half is fixed by how the sets are stored: a ridge facet is a
facet of the *septal* cell that owns it, which is the orientation the ridge extraction produces when
it walks the interface and keeps the septal side. Flooding the cell adjacency graph out from those
cells, refusing to cross either sheet, then recovers the septum exactly.

Returns `nothing` when the flood escapes into the whole mesh, which means the sheets have a hole
in them and there is no partition to recover.
"""
function _septal_cells_by_partition(
    grid::AbstractGrid,
    ridge_anterior::String,
    ridge_posterior::String,
)
    blocked = Set{NTuple{3, Int}}()
    frontier = Int[]
    for name in (ridge_anterior, ridge_posterior)
        for (cellid, local_facet) in getfacetset(grid, name)
            push!(blocked, _facet_key(getcells(grid, cellid), local_facet))
            push!(frontier, cellid)
        end
    end
    isempty(frontier) && return nothing

    # Facet key -> the (one or two) cells sharing it.
    neighbours = Dict{NTuple{3, Int}, Vector{Int}}()
    for cellid = 1:getncells(grid)
        cell = getcells(grid, cellid)
        for local_facet in 1:Ferrite.nfacets(cell)
            push!(get!(neighbours, _facet_key(cell, local_facet), Int[]), cellid)
        end
    end

    septal = falses(getncells(grid))
    septal[frontier] .= true
    while !isempty(frontier)
        cellid = pop!(frontier)
        cell = getcells(grid, cellid)
        for local_facet in 1:Ferrite.nfacets(cell)
            key = _facet_key(cell, local_facet)
            key in blocked && continue
            for other in neighbours[key]
                if !septal[other]
                    septal[other] = true
                    push!(frontier, other)
                end
            end
        end
    end

    all(septal) && return nothing
    return septal
end

"Node ids of the two ridges, with the nodes they share near the apex removed from both."
function _ridge_nodes(grid::AbstractGrid, anterior::String, posterior::String)
    ant = _facetset_node_ids(grid, getfacetset(grid, anterior))
    post = _facetset_node_ids(grid, getfacetset(grid, posterior))
    # The two ridge sheets share the handful of nodes where they converge towards the apex. They
    # cannot be pinned to 0 and 1 at once, so they are left out of both Dirichlet sets and the
    # Laplace solve interpolates them: the rotational coordinate genuinely has no value there.
    shared = intersect(Set(ant), Set(post))
    return OrderedSet{Int}(n for n in ant if !(n in shared)),
    OrderedSet{Int}(n for n in post if !(n in shared))
end

"""
Rotational coordinate in the sense of Cobiveco: solve `Δv = 0` with `v = 0` on the posterior ridge
and `v = 1` on the anterior ridge, then map the free wall onto `[0, 2/3]` and the septum onto
`[2/3, 1]`.

The two ridges cut the myocardium in two, so `v` is solved independently on each half and rises
monotonically from one ridge to the other on both. The mapping stitches the halves into a single
chart around the ventricle: it is continuous across the anterior ridge, where both branches give
2/3, and jumps by a full turn across the posterior ridge, where the free wall gives 0 and the septum
gives 1. Since the region is a property of the cell, the discontinuous dofs carry the jump exactly.
"""
function _compute_rotational_from_ridges!(
    u::Vector{Float64},
    dh_rotational::DofHandler,
    dh::DofHandler,
    K,
    solver,
    ridge_anterior_nodes::OrderedSet{Int},
    ridge_posterior_nodes::OrderedSet{Int},
    septal::BitVector,
)
    v = _solve_dirichlet_laplace(
        K,
        dh,
        solver,
        [(ridge_posterior_nodes, 0.0), (ridge_anterior_nodes, 1.0)],
    )
    clamp!(v, 0.0, 1.0)

    for sdh in dh.subdofhandlers
        for cell in CellIterator(sdh)
            dofs = celldofs(cell)
            dofs_rotational = celldofsview(dh_rotational, cellid(cell))
            is_septal = septal[cellid(cell)]
            @inbounds for i in eachindex(dofs)
                vi = v[dofs[i]]
                u[dofs_rotational[i]] = is_septal ? 1 - vi / 3 : 2vi / 3
            end
        end
    end
    return u
end

"Unwrap the angles of one element in place so that the branch cut falls outside of it."
function _unwrap_cell_angles!(angles::Vector{Float64}, defined::Vector{Bool})
    i0 = findfirst(defined)
    if i0 === nothing
        # The whole element sits on the long axis, where the angle does not exist.
        fill!(angles, 0.0)
        return angles
    end
    reference = angles[i0]
    total = 0.0
    count = 0
    @inbounds for i in eachindex(angles)
        defined[i] || continue
        angles[i] = reference + rem(angles[i] - reference, 1.0, RoundNearest)
        total += angles[i]
        count += 1
    end
    mean = total / count
    # Only the differences within the element carry information -- the element as a whole is free to
    # sit on any turn. Nodes exactly on the seam land on either side of the branch cut depending on
    # rounding, so pick the turn deterministically instead of letting that decide it.
    turn = floor(mean)
    @inbounds for i in eachindex(angles)
        angles[i] = defined[i] ? angles[i] - turn : mean - turn
    end
    return angles
end

"""
Rotational coordinate as the plain azimuth around the long axis, `r = θ/2π`.

This is the fallback for meshes that carry no ridge annotation. It is *not* the same chart as the
ridge based one: it distributes the coordinate by angle rather than by
the position of the right ventricular insertions, so two meshes only agree under it if they are
aligned the same way in space.

The angles are unwrapped per element, which puts the branch cut on element interfaces instead of
smearing it across a layer of elements. Where the cut lands is decided by `frame.e₁` -- the annotated
seam when there is one.
"""
function _compute_rotational_from_azimuth!(
    u::Vector{Float64},
    dh::DofHandler,
    ip_collection,
    frame::AzimuthalFrame,
)
    grid = get_grid(dh)
    cv_collection =
        CellValueCollection(NodalQuadratureRuleCollection(ip_collection), ip_collection)
    angles = Float64[]
    defined = Bool[]
    for sdh in dh.subdofhandlers
        cellvalues = getcellvalues(cv_collection, getcells(grid, first(sdh.cellset)))
        n_basefuncs = getnbasefunctions(cellvalues)
        resize!(angles, n_basefuncs)
        resize!(defined, n_basefuncs)
        for cell in CellIterator(sdh)
            reinit!(cellvalues, cell)
            coords = getcoordinates(cell)
            dofs = celldofs(cell)
            # Nodes closer to the long axis than this carry no angular information. The threshold
            # scales with the element so that it is a statement about the element, not about the
            # units of the mesh.
            tol = 1.0e-6 * maximum(norm(x - first(coords)) for x in coords)
            for qp in QuadratureIterator(cellvalues)
                θ = azimuth(
                    frame,
                    Vec{3, Float64}(spatial_coordinate(cellvalues, qp, coords)),
                    tol,
                )
                defined[qp.i] = θ !== nothing
                angles[qp.i] = θ === nothing ? 0.0 : θ / (2π)
            end
            _unwrap_cell_angles!(angles, defined)
            @inbounds for i = 1:n_basefuncs
                u[dofs[i]] = angles[i]
            end
        end
    end
    return u
end

"""
Fill the rotational coordinate, preferring the ridge based chart and falling back to the plain
azimuth when the mesh carries no ridges.
"""
function _compute_rotational!(
    u::Vector{Float64},
    dh_rotational::DofHandler,
    ip_collection_rotational,
    dh::DofHandler,
    K,
    solver,
    frame::AzimuthalFrame,
    ridge_anterior::Union{Nothing, String},
    ridge_posterior::Union{Nothing, String},
)
    grid = get_grid(dh)
    asked_for_ridges = ridge_anterior !== nothing && ridge_posterior !== nothing
    if !asked_for_ridges ||
       !_has_facetset(grid, ridge_anterior) ||
       !_has_facetset(grid, ridge_posterior)
        # Passing `nothing` says the mesh is known to have no ridges; missing sets that were asked
        # for are worth complaining about, because the fallback silently answers a different question.
        asked_for_ridges && @warn "No ridge annotation ('$ridge_anterior' and '$ridge_posterior') on the mesh, falling back to the plain azimuth around the long axis. That is a different chart than the ridge based one, so this coordinate is not comparable with one computed on an annotated mesh."
        return _compute_rotational_from_azimuth!(u, dh_rotational, ip_collection_rotational, frame)
    end

    septal = _septal_cells_by_partition(grid, ridge_anterior, ridge_posterior)
    if septal === nothing
        # The ridges have a hole, so they induce no partition to read the septum off. The O-grid
        # apex cap is the case that matters: its core is a regular patch across the apex and no
        # facet sheet inside it continues the ridges.
        direction_anterior = _sheet_direction(grid, ridge_anterior, frame)
        direction_posterior = _sheet_direction(grid, ridge_posterior, frame)
        (direction_anterior === nothing || direction_posterior === nothing) && error(
            "The ridge facetsets '$ridge_anterior' and '$ridge_posterior' neither cut the mesh in two nor describe radial sheets around the long axis, so the septum cannot be identified.",
        )
        septal = _septal_cells_by_arc(
            grid,
            frame,
            azimuth(frame, frame.origin + direction_anterior)::Float64,
            azimuth(frame, frame.origin + direction_posterior)::Float64,
        )
    end
    anterior_nodes, posterior_nodes = _ridge_nodes(grid, ridge_anterior, ridge_posterior)
    return _compute_rotational_from_ridges!(
        u,
        dh_rotational,
        dh,
        K,
        solver,
        anterior_nodes,
        posterior_nodes,
        septal,
    )
end

"""
The pair of dof handlers every LV coordinate system is built on: a continuous one carrying the
transmural and apicobasal fields, and a discontinuous one carrying the rotational field -- see
[`LVCoordinateSystem`](@ref) for why the latter cannot share the former.
"""
function _coordinate_dofhandlers(mesh, subdomains, ip_collection, ip_collection_rotational)
    dh = DofHandler(mesh)
    dh_rotational = DofHandler(mesh)
    for name in subdomains
        add_subdomain!(dh, name, [ApproximationDescriptor(:coordinates, ip_collection)])
        add_subdomain!(
            dh_rotational,
            name,
            [ApproximationDescriptor(:coordinates, ip_collection_rotational)],
        )
    end
    Ferrite.close!(dh)
    Ferrite.close!(dh_rotational)
    return dh, dh_rotational
end

"Harmonic transmural coordinate, 0 on the endocardium and 1 on the epicardium."
function _transmural_coordinate(mesh, K, dh, solver, endocardium_name, epicardium_name)
    transmural = _solve_dirichlet_laplace(
        K,
        dh,
        solver,
        [(getfacetset(mesh, endocardium_name), 0.0), (getfacetset(mesh, epicardium_name), 1.0)],
    )
    # The Krylov solve overshoots the Dirichlet values by a few ulps, which would make the coordinate
    # leave its own range.
    clamp!(transmural, 0.0, 1.0)
    return transmural
end

"""
Rotational coordinate around the long axis through `origin`, on the discontinuous dofs of
`dh_rotational`. See [`_compute_rotational!`](@ref) for the two charts it picks between.
"""
function _rotational_coordinate(
    mesh,
    K,
    dh,
    dh_rotational,
    ip_collection_rotational,
    solver,
    longitudinal::Vec{3, Float64},
    origin::Vec{3, Float64},
    ridge_anterior,
    ridge_posterior,
    rotational_zero_direction,
)
    frame = AzimuthalFrame(
        origin,
        longitudinal,
        _azimuth_zero_direction(
            mesh,
            origin,
            longitudinal,
            _azimuth_seam_names(ridge_posterior),
            rotational_zero_direction,
        ),
    )
    rotational = zeros(ndofs(dh_rotational))
    _compute_rotational!(
        rotational,
        dh_rotational,
        ip_collection_rotational,
        dh,
        K,
        solver,
        frame,
        ridge_anterior,
        ridge_posterior,
    )
    return rotational
end

"Facetsets to look for the branch cut on, most specific first."
_azimuth_seam_names(ridge_posterior::Union{Nothing, String}) =
    ridge_posterior === nothing ? ["RotationalSeam"] : ["RotationalSeam", ridge_posterior]

"""
Resolve the direction of azimuth zero for the fallback chart. An explicit `zero_direction` wins;
otherwise the annotated seam is used, and failing that an arbitrary direction orthogonal to the long
axis -- in which case the branch cut lands wherever the geometry puts it.
"""
function _azimuth_zero_direction(
    grid::AbstractGrid,
    origin::Vec{3, Float64},
    longitudinal::Vec{3, Float64},
    seams::Vector{String},
    zero_direction::Union{Nothing, Vec{3}},
)
    zero_direction !== nothing && return Vec{3, Float64}(zero_direction)
    provisional = AzimuthalFrame(origin, longitudinal, _any_orthogonal(longitudinal))
    for seam in seams
        direction = _sheet_direction(grid, seam, provisional)
        direction !== nothing && return direction
    end
    return _any_orthogonal(longitudinal)
end

"""
    compute_lv_coordinate_system(mesh::SimpleMesh)

Compute the transmural, apicobasal and rotational coordinates of a left ventricle.

Requires a mesh with facetsets
    * Base
    * Epicardium
    * Endocardium
and a nodeset pinning the apical end of the apicobasal coordinate
    * Apex

and the two internal facetsets marking the lines along which the right ventricle attaches
    * SRidgeAnt
    * SRidgePost
which the idealized generators emit and the ridge extraction of a segmented anatomy produces.

Every one of those is a keyword, so a mesh that names its annotations differently needs no renaming.

`apex_nodeset` pins the epicardial apex alone by default. Pinning both ends of the apical wall
instead -- `apex_nodeset = "ApexInOut"` -- holds the coordinate at 0 through the whole thickness
there, and the resulting plateau flattens the gradient that the arc length recalibration integrates
against. How much it flattens depends on the shape and thickness of the apical cap, so two
differently shaped ventricles come out distorted by different amounts: on a pair of idealized
anatomies that moves the coordinate by a median of 0.06 on one of them and 0.004 on the other, and
degrades the rotational agreement of a transfer between them from 0.003 to 0.089.

The coordinates are

  * `transmural`: harmonic, 0 on the endocardium and 1 on the epicardium.
  * `apicobasal`: harmonic between the apex and the base, recalibrated to arc length by
    [`apicobasal_from_laplace`](@ref) -- without that step the raw harmonic field is useless as a
    coordinate, see its docstring.
  * `rotational`: the Cobiveco chart, 0 on the posterior ridge, 2/3 on the anterior ridge and back
    to 1 at the posterior ridge through the septum. It is anchored on the ridges rather than on the
    orientation of the mesh in space, which is what makes it comparable between different hearts.
    Stored discontinuously so that the jump at the posterior ridge is exact, see
    [`LVCoordinateSystem`](@ref).

Meshes without ridges fall back to the plain azimuth around the long axis, which is *not* the same
chart: it distributes the coordinate by angle rather than by the position of the right ventricular
insertions, so two hearts only agree under it if they happen to be aligned the same way in space.
Pass `ridge_anterior = ridge_posterior = nothing` to ask for that fallback deliberately.
"""
function compute_lv_coordinate_system(
    mesh::SimpleMesh{3, <:Any, T};
    subdomains::Vector{String} = [single_subdomain_or_error(mesh)],
    axes::LVAxes = compute_lv_axes(mesh),
    apex_nodeset::String = "Apex",
    base_name::String = "Base",
    epicardium_name::String = "Epicardium",
    endocardium_name::String = "Endocardium",
    ridge_anterior::Union{Nothing, String} = "SRidgeAnt",
    ridge_posterior::Union{Nothing, String} = "SRidgePost",
    rotational_zero_direction::Union{Nothing, Vec{3}} = nothing,
    apicobasal_bins::Int = 200,
    solver = LinearSolve.KrylovJL_CG(), # FIXME add AMG preconditioner
) where {T}
    ip_collection = LagrangeCollection{1}()
    ip_collection_rotational = DiscontinuousLagrangeCollection{1}()
    dh, dh_rotational =
        _coordinate_dofhandlers(mesh, subdomains, ip_collection, ip_collection_rotational)

    # TODO use bilinear operator for performance
    K = _assemble_laplacian(dh, ip_collection)

    transmural = _transmural_coordinate(mesh, K, dh, solver, endocardium_name, epicardium_name)

    apicobasal_laplace = _solve_dirichlet_laplace(
        K,
        dh,
        solver,
        [(getfacetset(mesh, base_name), 1.0), (getnodeset(mesh, apex_nodeset), 0.0)],
    )
    apicobasal =
        apicobasal_from_laplace(dh, ip_collection, apicobasal_laplace; nbins = apicobasal_bins)

    rotational = _rotational_coordinate(
        mesh,
        K,
        dh,
        dh_rotational,
        ip_collection_rotational,
        solver,
        Vec{3, Float64}(axes.longitudinal),
        Vec{3, Float64}(axes.base_center),
        ridge_anterior,
        ridge_posterior,
        rotational_zero_direction,
    )

    return LVCoordinateSystem(
        dh,
        ip_collection,
        T.(transmural),
        T.(apicobasal),
        dh_rotational,
        ip_collection_rotational,
        T.(rotational),
    )
end

"""
    compute_midmyocardial_section_coordinate_system(mesh::SimpleMesh)

Requires a mesh with facetsets
    * Base
    * Epicardium
    * Endocardium
    * Myocardium

Unlike [`compute_lv_coordinate_system`](@ref) this is a ring section with no apex, so the long axis
cannot be derived from the geometry and is given by `up` instead. The apicobasal coordinate is the
height along `up`, rescaled into `[apicobasal_lower, apicobasal_upper]`. The rotational coordinate
is built exactly as in the LV case: from the two ridges when the section carries them, and from the
plain azimuth around `up` otherwise, which is what a plain ring gets. Either way it is stored
discontinuously, so its jump sits on an element interface instead of being smeared across a layer of
elements.
"""
function compute_midmyocardial_section_coordinate_system(
    mesh::SimpleMesh{3, <:Any, T},
    subdomains::Vector{String} = [single_subdomain_or_error(mesh)];
    up = Vec((T(0.0), T(0.0), T(1.0))),
    apicobasal_lower = 0.4,
    apicobasal_upper = 0.6,
    epicardium_name::String = "Epicardium",
    endocardium_name::String = "Endocardium",
    ridge_anterior::Union{Nothing, String} = "SRidgeAnt",
    ridge_posterior::Union{Nothing, String} = "SRidgePost",
    rotational_zero_direction::Union{Nothing, Vec{3}} = nothing,
    solver = LinearSolve.KrylovJL_CG(), # FIXME add AMG preconditioner
) where {T}
    ip_collection = LagrangeCollection{1}()
    ip_collection_rotational = DiscontinuousLagrangeCollection{1}()
    dh, dh_rotational =
        _coordinate_dofhandlers(mesh, subdomains, ip_collection, ip_collection_rotational)

    # TODO use bilinear operator from FerriteOperators to parallelize assembly
    K = _assemble_laplacian(dh, ip_collection)

    transmural = _transmural_coordinate(mesh, K, dh, solver, endocardium_name, epicardium_name)

    # Apicobasal coordinate: the height along `up`, rescaled into the requested range.
    apicobasal = zeros(ndofs(dh))
    apply_analytical!(apicobasal, dh, :coordinates, x -> x ⋅ up)
    apicobasal .-= minimum(apicobasal)
    apicobasal ./= maximum(apicobasal)
    apicobasal .*= (apicobasal_upper - apicobasal_lower)
    apicobasal .+= apicobasal_lower

    longitudinal = Vec{3, Float64}(up)
    longitudinal /= norm(longitudinal)
    rotational = _rotational_coordinate(
        mesh,
        K,
        dh,
        dh_rotational,
        ip_collection_rotational,
        solver,
        longitudinal,
        Vec{3, Float64}(sum(_node_coordinates(mesh)) / getnnodes(mesh)),
        ridge_anterior,
        ridge_posterior,
        rotational_zero_direction,
    )

    return LVCoordinateSystem(
        dh,
        ip_collection,
        T.(transmural),
        T.(apicobasal),
        dh_rotational,
        ip_collection_rotational,
        T.(rotational),
    )
end

"""
    vtk_coordinate_system(vtk, cs::LVCoordinateSystem)

Store the LV coordinate system in a vtk file.

The rotational coordinate jumps across the seam, which a continuous VTK grid cannot represent -- it
has one value per node, and the seam nodes are shared. On such a file the circular embedding
`(cos 2πr, sin 2πr)` is written instead, which is single valued everywhere and carries the same
information. Open the file with `write_discontinuous = true` to get the coordinate itself.
"""
function vtk_coordinate_system(vtk, cs::LVCoordinateSystem)
    Ferrite.write_solution(vtk, cs.dh, cs.u_apicobasal, "apicobasal_")
    Ferrite.write_solution(vtk, cs.dh, cs.u_transmural, "transmural_")
    if Ferrite.write_discontinuous(vtk)
        Ferrite.write_solution(vtk, cs.dh_rotational, cs.u_rotational, "rotational_")
    else
        sine, cosine = _nodal_rotational_embedding(cs)
        Ferrite.write_solution(vtk, cs.dh, sine, "rotational_sin_")
        Ferrite.write_solution(vtk, cs.dh, cosine, "rotational_cos_")
    end
end

"""
The rotational coordinate as `(sin 2πr, cos 2πr)` on the continuous dofs of `cs.dh`. Elements
disagree on `r` only by whole turns, so both are single valued and scattering them nodewise is
lossless.
"""
function _nodal_rotational_embedding(cs::LVCoordinateSystem{T}) where {T}
    sine = zeros(T, ndofs(cs.dh))
    cosine = zeros(T, ndofs(cs.dh))
    for sdh in cs.dh.subdofhandlers
        for cell in CellIterator(sdh)
            dofs = celldofs(cell)
            dofs_rotational = celldofsview(cs.dh_rotational, cellid(cell))
            @inbounds for i in eachindex(dofs)
                r = cs.u_rotational[dofs_rotational[i]]
                sine[dofs[i]] = sinpi(2r)
                cosine[dofs[i]] = cospi(2r)
            end
        end
    end
    return sine, cosine
end

"""
    BiVCoordinateSystem(dh, u_transmural, u_apicobasal, u_rotational, u_transventricular)

Universal ventricular coordinate, containing the transmural, apicobasal, rotational
and transventricular coordinates.
"""
struct BiVCoordinateSystem{T, DH <: Ferrite.AbstractDofHandler} <: CoordinateSystemCoefficient
    dh::DH
    u_transmural::Vector{T}
    u_apicobasal::Vector{T}
    u_rotational::Vector{T}
    u_transventricular::Vector{T}
end


"""
BiVCoordinate{T}

Biventricular universal coordinate, containing
    * transmural
    * apicobasal
    * rotational
    * transventricular
"""
Base.@kwdef struct BiVCoordinate{T}
    transmural::T
    apicobasal::T
    rotational::T
    transventricular::T
end
Base.zero(::Type{BiVCoordinate{T}}) where {T} = BiVCoordinate(T(0.0), T(0.0), T(0.0), T(0.0))
Base.eltype(::Type{BiVCoordinate{T}}) where {T} = T
Base.eltype(::BiVCoordinate{T}) where {T} = T
value_type(::BiVCoordinateSystem) = BiVCoordinate

"""
    getcoordinateinterpolation(cs::BiVCoordinateSystem, cell::AbstractCell)

Get interpolation function for the biventricular coordinate system.

The subdomain carrying `cell` has to be looked up, because on a mixed mesh each element type gets its
own subdofhandler and its own interpolation.
"""
function getcoordinateinterpolation(cs::BiVCoordinateSystem, cell::Ferrite.AbstractCell)
    for (i, sdh) in enumerate(cs.dh.subdofhandlers)
        typeof(get_first_cell(sdh)) === typeof(cell) &&
            return Ferrite.getfieldinterpolation(cs.dh, (i, 1))
    end
    error("The coordinate system has no subdomain holding $(typeof(cell)) cells.")
end

function vtk_coordinate_system(vtk, cs::BiVCoordinateSystem)
    Ferrite.write_solution(vtk, bivcs.dh, bivcs.u_transmural, "_transmural")
    Ferrite.write_solution(vtk, bivcs.dh, bivcs.u_apicobasal, "_apicobasal")
    Ferrite.write_solution(vtk, bivcs.dh, bivcs.u_rotational, "_rotational")
    Ferrite.write_solution(vtk, bivcs.dh, bivcs.u_transventricular, "_transventricular")
end
