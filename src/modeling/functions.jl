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

# The solution vector is laid out as `[fe_dofs | internal_variables]`. The internal variables are
# condensed out at quadrature point level and do not enter the global linear system, so indices into
# the solution vector are not indices into the system matrix.
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
