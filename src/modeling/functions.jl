# For the mapping against the SciML ecosystem, a "Thunderbolt function" is essentially equivalent to a "SciML function" with parameters, which does not have all evaluation information
"""
    AbstractSemidiscreteFunction <: SciMLBase.AbstractDiffEqFunction{iip=true}

Supertype for all functions coming from PDE discretizations.

## Interface

    solution_size(::AbstractSemidiscreteFunction)
    get_strategy(::AbstractSemidiscreteFunction)
"""
abstract type AbstractSemidiscreteFunction <: SciMLBase.AbstractDiffEqFunction{true} end
get_strategy(::AbstractSemidiscreteFunction) = SequentialAssemblyStrategy(SequentialCPUDevice())

abstract type AbstractPointwiseFunction <: AbstractSemidiscreteFunction end

"""
    AbstractSemidiscreteBlockedFunction <: AbstractSemidiscreteFunction

Supertype for all functions coming from PDE discretizations with blocked structure.

## Interface

    BlockArrays.blocksizes(::AbstractSemidiscreteFunction)
    BlockArrays.blocks(::AbstractSemidiscreteFunction) -> Iterable
"""
abstract type AbstractSemidiscreteBlockedFunction <: AbstractSemidiscreteFunction end
solution_size(f::AbstractSemidiscreteBlockedFunction) = sum(blocksizes(f))

"""
    NullFunction(ndofs)

Utility type to describe that Jacobian and residual are zero, but ndofs dofs are present.
"""
struct NullFunction <: AbstractSemidiscreteFunction
    ndofs::Int
end

solution_size(f::NullFunction) = f.ndofs

"""
    PointwiseODEFunction

This acts as as a launch-pad for batches of ODE steps.
"""
struct PointwiseODEFunction{ODEType, xType, IndexVectorType, L} <: AbstractPointwiseFunction
    ode::ODEType
    x::xType
    # Indices of the states associated with this
    associated_states::IndexVectorType
    # The name this block of state is published under, from the model that introduced it.
    name::Symbol
    # How this block is arranged. Recorded rather than inferred, so a function is self-describing: a
    # standalone one is state blocked, one nested in a `PointwiseMultiODEFunction` is point blocked, and
    # nothing has to guess which from the enclosing type. `setup_solver_cache` still hardcodes the
    # matching `reshape`; the two can be tied together once the GPU paths are testable again.
    # Only read at setup and post-processing time -- neither this struct nor these fields are adapted,
    # so they cost nothing on the GPU.
    layout::L
end

PointwiseODEFunction(ode, x, associated_states, name::Symbol) =
    PointwiseODEFunction(ode, x, associated_states, name, StateBlockedLayout())

solution_size(f::PointwiseODEFunction) = length(f.associated_states)

"""
    PointwiseMultiODEFunction

This acts as as a launch-pad for batches of ODE steps.
"""
struct PointwiseMultiODEFunction{xType} <: AbstractPointwiseFunction
    functions::Vector{<:PointwiseODEFunction}
    x::xType
end

solution_size(f::PointwiseMultiODEFunction) = sum(solution_size.(f.functions))

struct AffineODEFunction{MI, BI, ST, DH, AS} <: AbstractSemidiscreteFunction
    mass_term::MI
    bilinear_term::BI
    source_term::ST
    dh::DH
    assembly_strategy::AS
end
get_strategy(f::AffineODEFunction) = f.assembly_strategy

solution_size(f::AffineODEFunction) = ndofs(f.dh)

struct AffineSteadyStateFunction{BI, ST, DH, CH, AS} <: AbstractSemidiscreteFunction
    bilinear_term::BI
    source_term::ST
    dh::DH
    ch::CH
    assembly_strategy::AS
end
get_strategy(f::AffineSteadyStateFunction) = f.assembly_strategy

solution_size(f::AffineSteadyStateFunction) = ndofs(f.dh)

"""
    AbstractSolidMechanicsFunction <: AbstractSemidiscreteFunction

A spatially discrete solid mechanics problem: a displacement field, quadrature point local internal
variables condensed into the tail of the solution vector, and Dirichlet conditions from a
`ConstraintHandler`.

It classifies the *spatial* problem only. Whether the inertial terms are present is a property of the
concrete function ([`QuasiStaticFunction`](@ref) versus [`ElastodynamicsFunction`](@ref)), and the time
scheme is the solver's business.

Not every solid mechanics problem in this package subtypes it: the coupled FSI and 3D-0D functions are
blocked functions and live outside this branch.

## Interface

    dh, ch, lvh                    # fields laying out the solution vector
    get_volume_integrator(f)       # the volumetric weak form
    solution_size(f)               # ndofs(dh) + ndofs(lvh)

`dh` may carry more than one field -- an [`ElastodynamicsFunction`](@ref) holds a velocity next to the
displacement -- so nothing may assume the finite element block is one field's, nor that a field
occupies a contiguous range of it.
"""
abstract type AbstractSolidMechanicsFunction <: AbstractSemidiscreteFunction end

"""
    QuasiStaticFunction{...}

A discrete nonlinear (possibly multi-level) problem with time dependent terms.
Abstractly written we want to solve the problem G(u, q, t) = 0, L(u, q, dₜq, t) = 0 on some time interval [t₁, t₂].
"""
struct QuasiStaticFunction{
    I <: AbstractNonlinearIntegrator,
    DH <: Ferrite.AbstractDofHandler,
    CH <: ConstraintHandler,
    LVH <: InternalVariableHandler,
    AS <: AbstractAssemblyStrategy,
} <: AbstractSolidMechanicsFunction
    dh::DH
    ch::CH
    lvh::LVH
    integrator::I
    assembly_strategy::AS
end
get_strategy(f::QuasiStaticFunction) = f.assembly_strategy

"""
    ElastodynamicsFunction{...}

The spatially discrete counterpart of an [`ElastodynamicsModel`](@ref): a
[`QuasiStaticFunction`](@ref) plus the mass term of the inertia contribution.

The mass term is kept as an *integrator* rather than an assembled matrix, like every other term in
this package -- the solver decides when and how to materialize it.
"""
struct ElastodynamicsFunction{
    QSF <: QuasiStaticFunction,
    MI <: AbstractBilinearIntegrator,
    DH <: Ferrite.AbstractDofHandler,
    CH <: ConstraintHandler,
    LVH <: InternalVariableHandler,
    MAP,
    AS <: AbstractAssemblyStrategy,
} <: AbstractSolidMechanicsFunction
    # The *state*: displacement and velocity, both genuine fields, plus the condensed internal
    # variables. This is what the solution vector holds.
    dh::DH
    ch::CH
    lvh::LVH
    # The internal force problem, posed on the displacement alone. A scheme that reconstructs the
    # velocity rather than solving for it assembles against *this*, so its linear system carries no
    # empty velocity rows.
    structural::QSF
    mass_term::MI
    # Wiring from the structural numbering into the state numbering: `state_mapping` for the block a
    # stage solves for, `velocity_dofs` for the block a scheme reconstructs. `velocity_dofs[i]` is the
    # state dof carrying the velocity at structural displacement dof `i`.
    state_mapping::MAP
    velocity_dofs::Vector{Int}
    assembly_strategy::AS
end
get_strategy(f::ElastodynamicsFunction) = f.assembly_strategy

"""
    get_volume_integrator(f)

The integrator carrying the volumetric weak form. For an [`ElastodynamicsFunction`](@ref) that is
the internal force term, which lives on the structural sub-problem rather than on the state.
"""
get_volume_integrator(f::AbstractSolidMechanicsFunction) = f.integrator
get_volume_integrator(f::ElastodynamicsFunction) = f.structural.integrator

solution_size(f::AbstractSolidMechanicsFunction) = ndofs(f.dh)+ndofs(f.lvh)

"""
    fe_dof_range(f)

The finite element dofs of the solution vector, i.e. everything the `DofHandler` distributes.

The whole block, so a consumer that wants only part of it -- a stage solver typically solves for a
subset of the fields -- must ask for that subset by name instead.
"""
fe_dof_range(f::AbstractSolidMechanicsFunction) = Base.OneTo(ndofs(f.dh))

"""
    internal_variable_range(f)
    internal_variable_range(dh, lvh)

The quadrature point local internal variables of the solution vector, condensed into its tail. The
two-argument form serves call sites that hold the handlers but not the function.

The solution vector is laid out as `[fe_dofs | internal_variables]`. The internal variables are condensed
out at quadrature point level and do not enter the global linear system, so indices into the solution
vector are not indices into the system matrix.

See [`fe_dof_range`](@ref) for why this is a query rather than arithmetic at the call site. Both are
*unnamed*, block-level queries for solvers; ask [`solution_indices`](@ref) for a named quantity.
"""
internal_variable_range(dh::Ferrite.AbstractDofHandler, lvh::InternalVariableHandler) =
    (ndofs(dh)+1):(ndofs(dh)+ndofs(lvh))
internal_variable_range(f::AbstractSolidMechanicsFunction) = internal_variable_range(f.dh, f.lvh)

solution_variables(f::AffineODEFunction) = _field_variables(f.dh)
solution_variables(f::AffineSteadyStateFunction) = _field_variables(f.dh)

_field_variables(dh) = SolutionVariable[
    FieldVariable(name, dh, Base.OneTo(ndofs(dh))) for name in Ferrite.getfieldnames(dh)
]

function solution_variables(f::AbstractSolidMechanicsFunction)
    vars = _field_variables(f.dh)
    ndofs(f.lvh) == 0 && return vars
    for sdh in f.dh.subdofhandlers
        append!(vars, _internal_state_variables(f, get_volume_integrator(f), sdh))
    end
    return merge_and_check_unique(vars)
end

_internal_state_variables(f, integrator::NonlinearIntegrator, sdh) =
    _internal_state_variables(f, integrator, integrator.volume_model, sdh)

function _internal_state_variables(
    f::AbstractSolidMechanicsFunction,
    integrator::NonlinearIntegrator,
    volume_model::QuasiStaticModel,
    sdh,
)
    (; material_model) = volume_model
    infos = gather_internal_variable_infos(material_model)
    isempty(infos) && return SolutionVariable[]

    nqp = getnquadpoints(getquadraturerule(integrator.qrc, sdh))
    cells = StateBlock[]
    for cell in CellIterator(sdh)
        cid = cellid(cell)
        # Take the per-point size from what the handler actually allocated rather than re-deriving it
        # from the model, so a mixed grid with differing quadrature counts stays consistent.
        nlocal = length(FerriteOperators.internal_variable_range(f.lvh, cid))
        nlocal == 0 && continue
        # Splitting the cell's block evenly assumes every quadrature point of the cell carries the same
        # amount of state. That holds for every material in the package, but not in general -- see
        # `internal_variable_size`. Assert rather than truncate, so a model that breaks it fails here
        # instead of silently addressing the wrong slots.
        nlocal % nqp == 0 || error(
            "Cell $cid was allocated $nlocal internal variable dofs, which is not divisible by its " *
            "$nqp quadrature points. Per-quadrature-point state of varying size is not representable " *
            "yet: `InternalVariableHandler` stores one offset per cell, and both the element unknown " *
            "layout and the local solver caches assume a single size per point.",
        )
        push!(cells, QuadratureStateCell(internal_variable_offset(f.lvh, cid), nqp, nlocal ÷ nqp))
    end
    isempty(cells) && return SolutionVariable[]

    points = QuadratureStatePoints(cells)
    vars = SolutionVariable[]
    offset = 0
    for info in infos
        components = collect((offset+1):(offset+info.size))
        push!(vars, LocalStateVariable(info.name, material_model, points, components, nothing))
        offset += info.size
    end
    return vars
end

function _internal_state_variables(
    f::AbstractSolidMechanicsFunction,
    integrator::NonlinearMultiDomainIntegrator2,
    sdh,
)
    vars = SolutionVariable[]
    for (name, set) in sdh.dh.grid.volumetric_subdomains
        # First, check if the subdomain at hand is used by the integrator
        haskey(integrator.subintegrators, name) || continue
        # Then check if the subdofhandler is part of the subdomain
        first(sdh.cellset) ∈ getcellset(sdh.dh.grid, name) || continue
        append!(vars, _internal_state_variables(f, integrator.subintegrators[name], sdh))
    end
    return vars
end

"""
    default_initial_condition!(u, f)
    create_initial_condition(f, [T])

The initial condition a semidiscrete function defaults to, written per named variable.

`default_initial_condition!` **assumes `u` is zeroed** and writes only the entries it owns, so that it
composes under recursion into a split without clobbering siblings. `create_initial_condition` owns the
allocation and the zeroing.
"""
function default_initial_condition!(u::AbstractVector, f)
    for v in solution_variables(f)
        default_initial_condition!(u, v)
    end
    return u
end

function create_initial_condition(f, ::Type{T} = Float64) where {T}
    u = zeros(T, solution_size(f))
    default_initial_condition!(u, f)
    return u
end

# A field has no canonical default beyond zero. Where a field aliases model-local state -- the
# transmembrane potential is one slot of every point's ionic state vector -- the state's default below
# supplies it.
default_initial_condition!(u::AbstractVector, ::FieldVariable) = u
default_initial_condition!(u::AbstractVector, ::GlobalVariable) = u

function default_initial_condition!(u::AbstractVector, v::LocalStateVariable)
    # `default_initial_state!` speaks in whole per-point state vectors, so write the whole block even when
    # this variable only *names* part of it. That is what makes the aliased transmembrane potential come
    # out right without a second code path.
    for k = 1:npoints(v.points)
        default_initial_state!(view(u, state_range(v.points, k)), v.model)
    end
    return u
end

# Pointwise state. The transmembrane potential is excluded from the *name*: it is a field the enclosing
# problem solves for, and in an operator split it aliases one slot of every point's state vector. A
# standalone pointwise problem therefore publishes its recovery states only.
function solution_variables(f::PointwiseODEFunction)
    ode        = f.ode
    nstates    = num_states(ode)
    npoints    = solution_size(f) ÷ nstates
    φidx       = transmembranepotential_index(ode)
    components = [i for i = 1:nstates if i != φidx]
    points     = BlockedStridedStatePoints([StateBlock(0, npoints, nstates, f.layout)])
    return SolutionVariable[LocalStateVariable(f.name, ode, points, components, f.x)]
end

# Like every other composite, this asks its children for their own frame and translates. `associated_states`
# is the child's block of this function's vector, which is exactly the index set `translate` needs.
function solution_variables(f::PointwiseMultiODEFunction)
    vars = SolutionVariable[]
    for g in f.functions
        append!(vars, (translate(v, g.associated_states) for v in solution_variables(g)))
    end
    return merge_and_check_unique(vars)
end

# Ionic models declare their default as a value; materials declare theirs in place. Bridging the two here
# is what finally gives `default_initial_state` a consumer.
default_initial_state!(Q::AbstractVector, model::AbstractIonicModel) =
    (Q .= default_initial_state(model); Q)

"""
    gather_internal_variable_infos(model) -> Tuple{Vararg{InternalVariableInfo}}

Describe the quadrature-point-local state a model carries. Return an **empty tuple** when the model
has none. Implementations must always return a tuple, so that consumers never have to branch on the
shape of the result.
"""
gather_internal_variable_infos(model::QuasiStaticModel) =
    gather_internal_variable_infos(model.material_model)
gather_internal_variable_infos(model::AbstractMaterialModel) = ()
