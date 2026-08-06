# Named access to the solution vector.
#
# A semidiscrete function answers `solution_variables(f)` with the named quantities its solution vector
# holds. Everything user-facing -- initial conditions, post-processing, IO -- is built on that one method,
# so no consumer has to know how a particular function type lays out its unknowns.
#
# Nothing here is a hot path: descriptors are built once per query and never reach an assembly loop or a
# GPU kernel. They are deliberately not type stable.

# ---------------------------------------------------------------------------------------------------
# Where model-local state lives
# ---------------------------------------------------------------------------------------------------

"""
    StatePointSet

Describes *where* a model's local state lives in the solution vector, and nothing else.

## Interface

    npoints(points)         -> Int
    state_range(points, k)  -> StepRange{Int, Int}   # the slots of point `k`

`state_range` returns a `StepRange` rather than a `UnitRange` because a point's states are contiguous
under a point-blocked layout but strided under a state-blocked one. Consumers index the result
(`state_range(points, k)[c]`) instead of doing arithmetic themselves, which is what lets the same code
serve both layouts, ragged quadrature-point state, and discretizations in which the state points bear no
relation to any field's dofs.
"""
abstract type StatePointSet end

"""
    StateBlockedLayout()
    PointBlockedLayout()

How one block of pointwise state is arranged: `StateBlockedLayout` stores all points of a state
consecutively (structure of arrays), `PointBlockedLayout` stores all states of a point consecutively
(array of structs).
"""
struct StateBlockedLayout end
struct PointBlockedLayout end

"""
    StateBlock(offset, npoints, nstates, layout)

One homogeneous block of pointwise state: `npoints` points carrying `nstates` values each, based at the
0-based `offset` in the solution vector.

A block is the unit over which the layout is constant. Several blocks make up a
[`BlockedStridedStatePoints`](@ref), because different subdomains carry different cell models with
different state counts.
"""
struct StateBlock{L}
    offset::Int
    npoints::Int
    nstates::Int
    layout::L
end

@inline function state_range(b::StateBlock{StateBlockedLayout}, k::Int)
    first_slot = b.offset + k
    return first_slot:(b.npoints):(first_slot+(b.nstates-1)*b.npoints)
end

@inline function state_range(b::StateBlock{PointBlockedLayout}, k::Int)
    first_slot = b.offset + (k - 1) * b.nstates + 1
    return first_slot:1:(first_slot+b.nstates-1)
end

shift(b::StateBlock, by::Int) = StateBlock(b.offset + by, b.npoints, b.nstates, b.layout)

"""
    BlockedStridedStatePoints(blocks)

Pointwise state spread over several blocks, each internally strided with its own point count, state count
and layout.

This is what an operator-split electrophysiology problem produces: subdomains carry different cell models,
so neither the number of states nor the stride is global, and no single `reshape` describes the storage.
"""
struct BlockedStridedStatePoints{B <: StateBlock} <: StatePointSet
    blocks::Vector{B}
    # first_point[b] is the global index of block b's first point, so that a global point index can be
    # resolved to (block, point-within-block) by a single search.
    first_point::Vector{Int}
end

function BlockedStridedStatePoints(blocks::Vector{<:StateBlock})
    first_point = Vector{Int}(undef, length(blocks))
    running = 1
    for (i, b) in enumerate(blocks)
        first_point[i] = running
        running += b.npoints
    end
    return BlockedStridedStatePoints(blocks, first_point)
end

npoints(p::BlockedStridedStatePoints) = isempty(p.blocks) ? 0 : sum(b -> b.npoints, p.blocks)

@inline function state_range(p::BlockedStridedStatePoints, k::Int)
    b = searchsortedlast(p.first_point, k)
    return state_range(p.blocks[b], k - p.first_point[b] + 1)
end

shift(p::BlockedStridedStatePoints, by::Int) =
    BlockedStridedStatePoints([shift(b, by) for b in p.blocks], p.first_point)

"""
    QuadratureStateCell(offset, nqp, size_per_qp)

One cell's worth of quadrature-point-local state, based at the absolute 0-based `offset` that
`FerriteOperators.internal_variable_offset` reports.
"""
struct QuadratureStateCell
    offset::Int
    nqp::Int
    size_per_qp::Int
end

@inline function state_range(c::QuadratureStateCell, q::Int)
    first_slot = c.offset + (q - 1) * c.size_per_qp + 1
    return first_slot:1:(first_slot+c.size_per_qp-1)
end

shift(c::QuadratureStateCell, by::Int) = QuadratureStateCell(c.offset + by, c.nqp, c.size_per_qp)

"""
    QuadratureStatePoints(cells)

Quadrature-point-local state, one entry per quadrature point of every cell that carries any. Ragged by
construction: mixed grids give different cells different quadrature counts and different per-point sizes.
"""
struct QuadratureStatePoints <: StatePointSet
    cells::Vector{QuadratureStateCell}
    first_point::Vector{Int}
end

function QuadratureStatePoints(cells::Vector{QuadratureStateCell})
    first_point = Vector{Int}(undef, length(cells))
    running = 1
    for (i, c) in enumerate(cells)
        first_point[i] = running
        running += c.nqp
    end
    return QuadratureStatePoints(cells, first_point)
end

npoints(p::QuadratureStatePoints) = isempty(p.cells) ? 0 : sum(c -> c.nqp, p.cells)

@inline function state_range(p::QuadratureStatePoints, k::Int)
    c = searchsortedlast(p.first_point, k)
    return state_range(p.cells[c], k - p.first_point[c] + 1)
end

shift(p::QuadratureStatePoints, by::Int) =
    QuadratureStatePoints([shift(c, by) for c in p.cells], p.first_point)

# ---------------------------------------------------------------------------------------------------
# Descriptors
# ---------------------------------------------------------------------------------------------------

"""
    SolutionVariable

A named quantity in a solution vector. Three kinds exist, because three genuinely different mechanisms
produce and initialize them:

  - [`FieldVariable`](@ref)      -- a finite element field, backed by a `DofHandler`
  - [`LocalStateVariable`](@ref) -- model-local state at a set of evaluation points
  - [`GlobalVariable`](@ref)     -- a single scalar unknown

## Interface (for a new kind)

    variable_name(v)        -> the name, usually a `Symbol`
    variable_indices(v)     -> AbstractVector{Int}   # positions in the solution vector
    translate(v, indices)   -> SolutionVariable      # re-express against a parent vector
"""
abstract type SolutionVariable end

variable_name(v::SolutionVariable) = v.name

"""
    FieldVariable(name, dh, dofs)

A finite element field. `dofs[i]` is the position in the solution vector of dof `i` of `dh`, so
`view(u, dofs)` is a vector Ferrite can index by dof number -- which is what `Ferrite.apply_analytical!`
and the VTK writers require.

`dofs` covers the whole `DofHandler`, not just this field: a `DofHandler` may carry several fields, and
`dh`-indexed is `dh`-indexed. Ask [`variable_indices`](@ref) for this field's own positions.
"""
struct FieldVariable{DH, I} <: SolutionVariable
    name::Symbol
    dh::DH
    dofs::I
end

"""
    LocalStateVariable(name, model, points, components, component_names, coordinates)

State owned by a model and attached to a set of evaluation points -- ionic cell state, sarcomere state,
viscoelastic history. These are the same mechanism at different point sets, which is why they share a
descriptor.

`components` selects which slots of a point's local state vector this variable owns. It is not always
all of them and not always contiguous: an electrophysiology cell state excludes the transmembrane
potential, which is a field, and `ParametrizedAlievPanfilovModel` carries that potential at index 2 of 2.

`coordinates` holds one coordinate per point, or `nothing` when the model was built without a coordinate
system; it is what lets a state be initialized from a spatial function.
"""
struct LocalStateVariable{M, P <: StatePointSet, X} <: SolutionVariable
    name::Symbol
    model::M
    points::P
    components::Vector{Int}
    component_names::Vector{Symbol}
    coordinates::X
end

"""
    GlobalVariable(name, index)

A single scalar unknown that is not attached to the mesh -- a chamber pressure, a lumped circuit state.

The name is not constrained to be a `Symbol`: a ModelingToolkit-backed circuit names its states with
`Num`s, and the coupling already carries those names untyped.
"""
struct GlobalVariable{N} <: SolutionVariable
    name::N
    index::Int
end

# --- indices -----------------------------------------------------------------------------------------

"""
    variable_indices(v) -> AbstractVector{Int}

The positions in the solution vector that `v` occupies.

For a [`FieldVariable`](@ref) on a multi-field `DofHandler` this walks the grid, so hoist it out of a loop
rather than calling it per timestep.
"""
function variable_indices(v::FieldVariable)
    fieldnames = Ferrite.getfieldnames(v.dh)
    length(fieldnames) == 1 && return collect(v.dofs)
    dofs = Int[]
    for sdh in v.dh.subdofhandlers
        v.name in sdh.field_names || continue
        drange = Ferrite.dof_range(sdh, v.name)
        for cell in CellIterator(sdh)
            append!(dofs, @view celldofs(cell)[drange])
        end
    end
    sort!(dofs)
    unique!(dofs)
    return v.dofs[dofs]
end

# Point major: all of a point's components, then the next point. Deliberately independent of how the
# points are actually stored, so a caller can reshape the result to `(ncomponents, npoints)` regardless of
# the underlying blocks and layouts.
function variable_indices(v::LocalStateVariable)
    idx = Vector{Int}(undef, npoints(v.points) * length(v.components))
    n = 0
    for k = 1:npoints(v.points)
        r = state_range(v.points, k)
        for c in v.components
            idx[n+=1] = r[c]
        end
    end
    return idx
end

variable_indices(v::GlobalVariable) = v.index:v.index

# --- translation into a parent solution vector -------------------------------------------------------

"""
    translate(v, indices)

Re-express `v`, whose positions are local to some sub-function, against the parent solution vector, where
`indices` maps that sub-function's positions into the parent.

`GenericSplitFunction` solution indices are relative to the parent at every level, so translating once per
level composes correctly through arbitrarily nested splits.
"""
translate(v::FieldVariable, indices) = FieldVariable(v.name, v.dh, indices[v.dofs])
translate(v::GlobalVariable, indices) = GlobalVariable(v.name, indices[v.index])

function translate(v::LocalStateVariable, indices)
    # A point set is described by offsets and strides, so it can only be rebased by a shift. Every
    # split in the package hands its children a contiguous block, so this always holds; a gather would
    # need the point set to carry explicit indices instead.
    indices isa AbstractUnitRange || error(
        "Cannot translate the local state variable $(repr(v.name)) through a non-contiguous index set " *
        "of type $(typeof(indices)). Pointwise state is described by offsets and strides, so the " *
        "enclosing split has to hand it a contiguous block of the solution vector.",
    )
    return LocalStateVariable(
        v.name,
        v.model,
        shift(v.points, first(indices) - 1),
        v.components,
        v.component_names,
        v.coordinates,
    )
end

# ---------------------------------------------------------------------------------------------------
# The one method a function type implements
# ---------------------------------------------------------------------------------------------------

"""
    solution_variables(f) -> Vector{SolutionVariable}

The named quantities `f`'s solution vector holds, with positions **local to `f`**, i.e. indices into
`1:solution_size(f)`. A composite function translates its children's answers ([`translate`](@ref)).

Defaults to empty: a function type that does not implement this is simply anonymous, and every other part
of the package keeps working. That is deliberate -- it is what keeps the interface open to function types
that do not exist yet, without a central registry anything has to be added to.
"""
solution_variables(::Any) = SolutionVariable[]

"""
    solution_variable_names(f)

Every name `f` publishes, in the order [`solution_variables`](@ref) reports them.
"""
solution_variable_names(f) = [variable_name(v) for v in solution_variables(f)]

"""
    solution_variable(f, name) -> SolutionVariable

The descriptor `f` publishes under `name`. Errors listing the available names otherwise.
"""
function solution_variable(f, name)
    vars = solution_variables(f)
    idx = findfirst(v -> variable_name(v) == name, vars)
    idx === nothing && error(
        "No solution variable named $(repr(name)) in a $(nameof(typeof(f))). " * (
            isempty(vars) ?
            "It publishes none -- `Thunderbolt.solution_variables` is not implemented " *
            "for this function type." :
            "Available: $(join(repr.(variable_name.(vars)), ", "))."
        ),
    )
    return vars[idx]
end

has_solution_variable(f, name) = any(v -> variable_name(v) == name, solution_variables(f))

"""
    solution_indices(f, name) -> AbstractVector{Int}

Positions in `f`'s solution vector of the quantity called `name`. Hoist it out of a loop; each call
rebuilds the descriptor tree.
"""
solution_indices(f, name) = variable_indices(solution_variable(f, name))

# ---------------------------------------------------------------------------------------------------
# Reading and writing
# ---------------------------------------------------------------------------------------------------

"""
    getvariable(u, f, name)
    getvariable(u, v::SolutionVariable)

The value of a named quantity: a scalar for a [`GlobalVariable`](@ref), a view otherwise.
"""
getvariable(u::AbstractVector, f, name) = getvariable(u, solution_variable(f, name))
getvariable(u::AbstractVector, v::GlobalVariable) = u[v.index]
getvariable(u::AbstractVector, v::SolutionVariable) = view(u, variable_indices(v))

"""
    field_view(u, v::FieldVariable)

A `DofHandler`-indexed view of `u`, i.e. one where entry `i` is dof `i` of `v.dh`. This is what Ferrite's
own routines expect; it spans every field of the handler, not just `v`.
"""
field_view(u::AbstractVector, v::FieldVariable) = view(u, v.dofs)

"""
    setvariable!(u, f, name, value)
    setvariable!(u, v::SolutionVariable, value)

Write a named quantity. Dispatch is on the pair `(descriptor, value)`, so both a new kind of quantity and
a new kind of value are additive.

`value` may be a number, a per-component collection, a function of position, or a coefficient.

What a function receives depends on the kind. For a [`FieldVariable`](@ref) it is the physical coordinate,
via `Ferrite.apply_analytical!`, so a coefficient has to be passed explicitly to work in a generalized
coordinate system. For a [`LocalStateVariable`](@ref) it is whatever the owning model's coordinate system
yields -- a `Vec`, an `LVCoordinate`, a cell index -- because the state is initialized at the same points,
and against the same coordinate, the model is evaluated at.
"""
setvariable!(u::AbstractVector, f, name, value) = setvariable!(u, solution_variable(f, name), value)

# `do` block form, for the common case of a spatial function.
setvariable!(fun::Function, u::AbstractVector, f, name) = setvariable!(u, f, name, fun)
setvariable!(fun::Function, u::AbstractVector, v::SolutionVariable) = setvariable!(u, v, fun)

function setvariable!(u::AbstractVector, v::GlobalVariable, value::Number)
    u[v.index] = value
    return u
end

function setvariable!(u::AbstractVector, v::FieldVariable, fun::Function)
    Ferrite.apply_analytical!(field_view(u, v), v.dh, v.name, fun)
    return u
end

function setvariable!(u::AbstractVector, v::FieldVariable, value::Number)
    idx = variable_indices(v)
    u[idx] .= value
    return u
end

function setvariable!(u::AbstractVector, v::FieldVariable, coeff)
    a = field_view(u, v)
    evaluate_coefficient_at_dofs!(a, coeff, v.dh, v.name)
    return u
end

function setvariable!(u::AbstractVector, v::LocalStateVariable, value::Number)
    for k = 1:npoints(v.points)
        r = state_range(v.points, k)
        for c in v.components
            u[r[c]] = value
        end
    end
    return u
end

function setvariable!(u::AbstractVector, v::LocalStateVariable, fun)
    coords = local_state_coordinates(v)
    for k = 1:npoints(v.points)
        r = state_range(v.points, k)
        vals = _state_values(fun, coords[k], v)
        for (j, c) in enumerate(v.components)
            u[r[c]] = vals[j]
        end
    end
    return u
end

function local_state_coordinates(v::LocalStateVariable)
    v.coordinates === nothing && error(
        "Cannot set $(repr(v.name)) from a spatial function: the model carries no coordinate system, so " *
        "its state points have no coordinates. Give the model one -- e.g. " *
        "`MonodomainModel(Cₘ, χ, κ, stim, cell_model, CartesianCoordinateSystem(mesh), :φₘ, :s)` -- or " *
        "write the values directly through `getvariable(u, f, $(repr(v.name)))`.",
    )
    return v.coordinates
end

# A single-component state may be given by a scalar-valued function; anything wider has to return one
# value per component.
@inline function _state_values(fun, x, v::LocalStateVariable)
    vals = fun(x)
    if length(v.components) == 1 && !(vals isa Union{Tuple, AbstractVector})
        return (vals,)
    end
    length(vals) == length(v.components) || error(
        "Setting $(repr(v.name)) expects $(length(v.components)) value(s) per point " *
        "($(join(repr.(v.component_names), ", "))), got $(length(vals)).",
    )
    return vals
end

"""
    evaluate_coefficient_at_dofs!(a, coefficient, dh, field_name)

Evaluate `coefficient` at the dof positions of `field_name` and write the results into the
`DofHandler`-indexed vector `a`.

Uses the delta property of nodal interpolations, exactly as `compute_nodal_values` does, so it goes
through the ordinary `setup_coefficient_cache`/`evaluate_coefficient` protocol and works for any
coefficient rather than only coordinate systems.
"""
function evaluate_coefficient_at_dofs!(
    a::AbstractVector,
    coefficient,
    dh::DofHandler,
    field_name::Symbol,
)
    for sdh in dh.subdofhandlers
        field_name ∈ sdh.field_names || continue
        ip = Ferrite.getfieldinterpolation(sdh, field_name)
        rdim = Ferrite.getrefdim(ip)
        positions = Vec{rdim, Float64}.(Ferrite.reference_coordinates(ip))
        qr = QuadratureRule{Ferrite.getrefshape(ip)}([1.0 for _ in positions], positions)
        cc = setup_coefficient_cache(coefficient, qr, sdh)
        drange = Ferrite.dof_range(sdh, field_name)
        for cell in CellIterator(sdh)
            dofs = @view celldofs(cell)[drange]
            for qp in QuadratureIterator(qr)
                a[dofs[qp.i]] = evaluate_coefficient(cc, cell, qp, NaN)
            end
        end
    end
    return a
end

# ---------------------------------------------------------------------------------------------------
# Composition and the uniqueness invariant
# ---------------------------------------------------------------------------------------------------

"""
    merge_and_check_unique(vars) -> Vector{SolutionVariable}

Enforce that a symbol denotes exactly one quantity across the whole function tree.

Descriptors of the same kind that name the same quantity on different subdomains are merged -- this is how
one cell state spread over several subdomains, or one field spread over several `SubDofHandler`s, stays a
single name. Anything else sharing a name is a modelling error and raises.
"""
function merge_and_check_unique(vars::Vector{SolutionVariable})
    out = SolutionVariable[]
    for v in vars
        i = findfirst(w -> variable_name(w) == variable_name(v), out)
        if i === nothing
            push!(out, v)
        else
            out[i] = merge_variables(out[i], v)
        end
    end
    return out
end

merge_variables(a::SolutionVariable, b::SolutionVariable) = error(
    "Two different quantities are both named $(repr(variable_name(a))): a $(nameof(typeof(a))) and a " *
    "$(nameof(typeof(b))). Solution variable names have to be unique across the whole function tree; " *
    "rename one of them in the model that introduces it.",
)

function merge_variables(a::LocalStateVariable, b::LocalStateVariable)
    (typeof(a.model) === typeof(b.model) && a.components == b.components) || error(
        "Two different local states are both named $(repr(a.name)): one on a $(nameof(typeof(a.model))) " *
        "and one on a $(nameof(typeof(b.model))). Give the subdomains distinct state symbols, since a " *
        "name has to denote one quantity across the whole function tree.",
    )
    return LocalStateVariable(
        a.name,
        a.model,
        _merge_points(a.points, b.points),
        a.components,
        a.component_names,
        _merge_coordinates(a.coordinates, b.coordinates),
    )
end

_merge_points(a::BlockedStridedStatePoints, b::BlockedStridedStatePoints) =
    BlockedStridedStatePoints(vcat(a.blocks, b.blocks))
_merge_points(a::QuadratureStatePoints, b::QuadratureStatePoints) =
    QuadratureStatePoints(vcat(a.cells, b.cells))
_merge_points(a::StatePointSet, b::StatePointSet) =
    error("Cannot merge state points of type $(typeof(a)) and $(typeof(b)).")

_merge_coordinates(::Nothing, ::Nothing) = nothing
_merge_coordinates(a, b) = (a === nothing || b === nothing) ? nothing : vcat(a, b)

function merge_variables(a::FieldVariable, b::FieldVariable)
    a.dh === b.dh || error(
        "The field $(repr(a.name)) is published against two different DofHandlers. A field symbol has to " *
        "denote one quantity across the whole function tree.",
    )
    a.dofs == b.dofs ||
        error("The field $(repr(a.name)) is published twice with different dof maps.")
    return a
end

"""
    validate_solution_variables(f)

Check that `f`'s published variables are consistent: names unique across the whole tree, and every claimed
index inside `1:solution_size(f)`. Raises on the first problem. Intended for tests -- it is
`O(solution_size)`.

Overlapping *indices* are not an error: the transmembrane potential is legitimately both a field and one
slot of every point's ionic state vector, which is the whole point of an operator split. Overlapping
*names* are, and [`merge_and_check_unique`](@ref) has already rejected those by the time this runs.
"""
function validate_solution_variables(f)
    n = solution_size(f)
    vars = solution_variables(f)
    names = variable_name.(vars)
    allunique(names) || error("Duplicate solution variable names: $(names).")
    for v in vars
        for i in variable_indices(v)
            (1 ≤ i ≤ n) ||
                error("Variable $(repr(variable_name(v))) claims index $i, outside 1:$n.")
        end
    end
    return true
end
