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
num_blocks(f::AbstractSemidiscreteBlockedFunction) = length(blocksizes(f))

"""
    SolutionBlock(name, indices, placement, time_order)

One block of a semidiscrete solution vector: the field or auxiliary state `name`, occupying
`indices` of the solution vector, laid out according to `placement`.

`placement` is a [`DofPlacement`](@ref) for ordinary field variables and for auxiliary state living
on dofs, and a [`QuadraturePointPlacement`](@ref) for auxiliary state living at quadrature points.

`time_order` is the highest order of time derivative of this block appearing in the semidiscrete
problem: `0` for a purely algebraic block (a quasi-static displacement, an elliptic potential), `1`
for a first order block, `2` for a second order one. It is stated per block rather than per problem
because problems are routinely mixed - see [`SolutionPartition`](@ref).
"""
struct SolutionBlock{I <: AbstractVector{Int}, P <: AbstractStatePlacement}
    name::Symbol
    indices::I
    placement::P
    time_order::Int
end

solution_size(b::SolutionBlock) = length(b.indices)

"This block is algebraic, i.e. no time derivative of it appears in the problem."
is_algebraic(b::SolutionBlock) = b.time_order == 0

"This block is differential, i.e. some time derivative of it appears in the problem."
is_differential(b::SolutionBlock) = b.time_order > 0

"""
    SolutionPartition(blocks)

Describes how a semidiscrete solution vector is partitioned into named blocks.

This exists because the dof bookkeeping of split problems is the painful part `semidiscretize`
automates, and it used to be produced implicitly - each consumer re-derived the index sets from
scratch, under conventions that could and did disagree. Building it once, where the dof layout is
actually known, makes those layouts inspectable and keeps the consumers honest:

  - operator splitting derives its `solution_indices` from a partition rather than owning them,
  - monolithic solvers can use it for field-split or block preconditioning,
  - solvers partitioned by physics can dispatch on the blocks they are handed.

Query with [`solution_partition`](@ref).
"""
struct SolutionPartition{B <: AbstractVector{<:SolutionBlock}}
    blocks::B
end

# NOTE: deliberately not called `blocks` - `BlockArrays.blocks` is in scope package-wide and a
# `SolutionPartition` is not a block array.
solution_blocks(p::SolutionPartition) = p.blocks
num_blocks(p::SolutionPartition) = length(p.blocks)
solution_size(p::SolutionPartition) = sum(solution_size, p.blocks; init = 0)

"""
    max_time_order(partition)

Highest order of time derivative appearing anywhere in the problem: `0` for a steady problem, `1`
for first order, `2` for second order.
"""
max_time_order(p::SolutionPartition) = maximum(b.time_order for b in p.blocks; init = 0)

"""
    has_algebraic_blocks(partition)

Whether some blocks carry a time derivative and others do not - equivalently, whether the mass
matrix of the semidiscrete system is singular.

Derived from the per-block `time_order` rather than declared, so mixed problems classify themselves.
Quasi-static mechanics with internal variables is the canonical example: the displacement block is
algebraic (the weak form contains no `u̇`) while the internal variables evolve. A bidomain
formulation is another, with a parabolic transmembrane potential alongside an elliptic extracellular
potential.

!!! warning "This does not by itself say the problem is a DAE"
    The block structure gives only half the classification. Which kind of problem this is
    additionally depends on whether the internal variable model consumes *rates*:

    | internal variable model | resulting system |
    | :---------------------- | :--------------- |
    | `dₜQ = L(F, Q)` - strains only | ODE in mass matrix form, `M ẏ = f(y,t)` with singular `M` (formally a DAE) |
    | `dₜQ = L(F, dₜF, Q)` - rate dependent | true DAE: `f` depends on `ẏ`, so the system is not in mass matrix form |
    | fractional `dₜQ` | fractional ODE or fractional DAE, again depending on strain rate dependence |

    Distinguishing these needs a rate-dependence trait on the internal variable model, which does
    not exist yet. Until it does, do not infer "DAE" from this predicate alone.
"""
has_algebraic_blocks(p::SolutionPartition) =
    any(is_algebraic, p.blocks) && any(is_differential, p.blocks)

"""
    block_indices(partition) -> Tuple
    block_indices(partition, name::Symbol)

The index sets of all blocks in order - ready to hand to a `GenericSplitFunction` - or the indices
of the single block called `name`.
"""
block_indices(p::SolutionPartition) = Tuple(b.indices for b in p.blocks)
function block_indices(p::SolutionPartition, name::Symbol)
    for b in p.blocks
        b.name === name && return b.indices
    end
    error("No solution block named $name. Available blocks: $([b.name for b in p.blocks]).")
end

"""
    solution_partition(f) -> SolutionPartition

How the solution vector of the semidiscrete function `f` is laid out. See [`SolutionPartition`](@ref).
"""
function solution_partition end


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
struct PointwiseODEFunction{ODEType, xType, IndexVectorType} <: AbstractPointwiseFunction
    ode::ODEType
    x::xType
    # Indices of the states associated with this
    associated_states::IndexVectorType
end

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

abstract type AbstractQuasiStaticFunction <: AbstractSemidiscreteFunction end

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
} <: AbstractQuasiStaticFunction
    dh::DH
    ch::CH
    lvh::LVH
    integrator::I
    assembly_strategy::AS
end
get_strategy(f::QuasiStaticFunction) = f.assembly_strategy

solution_size(f::QuasiStaticFunction) = ndofs(f.dh)+ndofs(f.lvh)

"""
    solution_partition(f::QuasiStaticFunction)

The solution vector is laid out as `[fe_dofs | internal_variables]`.

!!! note
    The internal variables are condensed out at quadrature point level and do not enter the global
    linear system. Indices into the solution vector are therefore not indices into the system
    matrix.
"""
function solution_partition(f::QuasiStaticFunction)
    nfe = ndofs(f.dh)
    niv = ndofs(f.lvh)
    # The quasi-static weak form contains no time derivative of the displacement, so the finite
    # element block is algebraic; the internal variables evolve, so the mass matrix is singular.
    # Whether that makes this a mass-matrix ODE or a true DAE depends on whether the internal
    # variable model is rate dependent - see `has_algebraic_blocks`.
    fe_block = SolutionBlock(:fe_dofs, 1:nfe, DofPlacement(Tuple(f.dh.field_names)), 0)
    niv == 0 && return SolutionPartition([fe_block])
    return SolutionPartition([
        fe_block,
        SolutionBlock(:internal_variables, (nfe+1):(nfe+niv), QuadraturePointPlacement(), 1),
    ])
end

internal_variable_offset(f::QuasiStaticFunction, cid) = internal_variable_offset(f.lvh, cid)
internal_variable_size(f::QuasiStaticFunction, cid, qp) =
    internal_variable_size(get_material_model(f, cid, qp), cid, qp)
function default_initial_condition!(u::AbstractVector, f::QuasiStaticFunction)
    fill!(u, 0.0)
    ndofs(f.lvh) == 0 && return # no internal variable
    for sdh in f.dh.subdofhandlers
        default_initial_condition_quasistatic_subdomain!(u, f, f.integrator, sdh)
    end
end

function default_initial_condition_quasistatic_subdomain!(
    u,
    f::QuasiStaticFunction,
    integrator::NonlinearIntegrator,
    sdh,
)
    default_initial_condition_quasistatic_subdomain!(u, f, integrator, integrator.volume_model, sdh)
end

function default_initial_condition_quasistatic_subdomain!(
    u,
    f::QuasiStaticFunction,
    integrator::NonlinearIntegrator,
    volume_model::QuasiStaticModel,
    sdh,
)
    (; material_model) = volume_model

    qr = getquadraturerule(integrator.qrc, sdh)
    for cell in CellIterator(sdh)
        cid = cellid(cell)
        offset = internal_variable_offset(f.lvh, cid)
        offset == 0 && continue
        for qp in QuadratureIterator(qr)
            ivsize_per_qp = internal_variable_size(material_model, cid, qp)
            ivsize_per_qp == 0 && continue
            # `internal_variable_offset` is an absolute 0-based offset into the solution vector
            # (the FerriteOperators convention), so index `u` directly rather than a view of the
            # internal variable block.
            q = @view u[(offset+1):(offset+ivsize_per_qp)]
            default_initial_state!(q, material_model)
            offset += ivsize_per_qp
        end
    end
end

function default_initial_condition_quasistatic_subdomain!(
    u,
    f::QuasiStaticFunction,
    integrator::NonlinearMultiDomainIntegrator2,
    sdh,
)
    for (name, set) in sdh.dh.grid.volumetric_subdomains
        # First, check if the subdomain at hand is used byt he integrator
        haskey(integrator.subintegrators, name) || continue
        # Then check if the subdofhandler is part of the subdomain
        first(sdh.cellset) ∈ getcellset(sdh.dh.grid, name) || continue
        @debug "Setting default initial condition for subdomain $name"
        subintegrator = integrator.subintegrators[name]
        default_initial_condition_quasistatic_subdomain!(u, f, subintegrator, sdh)
    end
end

"""
    gather_internal_variable_infos(model) -> Tuple{Vararg{InternalVariableInfo}}

Describe the quadrature-point-local state a model carries. Return an **empty tuple** when the model
has none. Implementations must always return a tuple, so that consumers never have to branch on the
shape of the result.
"""
gather_internal_variable_infos(model::QuasiStaticModel) =
    gather_internal_variable_infos(model.material_model)
gather_internal_variable_infos(model::AbstractMaterialModel) = ()

__get_material_model(model::AbstractMaterialModel, cid, qp) = model
get_material_model(f::QuasiStaticFunction, cid, qp) =
    __get_material_model(_volume_model_for_cell(f, f.integrator, cid).material_model, cid, qp)

_volume_model_for_cell(f, integrator::NonlinearIntegrator, cid) = integrator.volume_model
function _volume_model_for_cell(f, integrator::NonlinearMultiDomainIntegrator2, cid)
    grid = get_grid(f.dh)
    for (name, subintegrator) in integrator.subintegrators
        cid ∈ getcellset(grid, name) && return subintegrator.volume_model
    end
    error(
        "Cell $cid is not covered by any subdomain of the integrator. Available subdomains: $(collect(keys(integrator.subintegrators))).",
    )
end
