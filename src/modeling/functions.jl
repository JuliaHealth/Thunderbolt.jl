# For the mapping against the SciML ecosystem, a "Thunderbolt function" is essentially equivalent to a "SciML function" with parameters, which does not have all evaluation information
"""
    AbstractSemidiscreteFunction <: DiffEqBase.AbstractDiffEqFunction{iip=true}

Supertype for all functions coming from PDE discretizations.

## Interface

    solution_size(::AbstractSemidiscreteFunction)
    get_strategy(::AbstractSemidiscreteFunction)
"""
abstract type AbstractSemidiscreteFunction <: DiffEqBase.AbstractDiffEqFunction{true} end
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


# TODO replace this with the original
struct ODEFunction{ODET,F,P} <: AbstractSemidiscreteFunction
    ode::ODET
    f::F
    p::P
end

solution_size(f::ODEFunction) = num_states(f.ode)

# See https://github.com/JuliaGPU/Adapt.jl/issues/84 for the reason why hardcoding Int does not work
struct PointwiseODEFunction{IndexType <: Integer, ODEType, xType} <: AbstractPointwiseFunction
    npoints::IndexType
    ode::ODEType
    x::xType
end
Adapt.@adapt_structure PointwiseODEFunction

solution_size(f::PointwiseODEFunction) = f.npoints*num_states(f.ode)

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
struct QuasiStaticFunction{I <: AbstractNonlinearIntegrator, DH <: Ferrite.AbstractDofHandler, CH <: ConstraintHandler, LVH <: InternalVariableHandler, AS <: AbstractAssemblyStrategy} <: AbstractQuasiStaticFunction
    dh::DH
    ch::CH
    lvh::LVH
    integrator::I
    assembly_strategy::AS
end
get_strategy(f::QuasiStaticFunction) = f.assembly_strategy

solution_size(f::QuasiStaticFunction) = ndofs(f.dh)+ndofs(f.lvh)
function local_function_size(f::QuasiStaticFunction)
    error("Local function size of QuasiStaticFunction can vary!")
end
function default_initial_condition!(u::AbstractVector, f::QuasiStaticFunction)
    fill!(u, 0.0)
    ndofs(f.lvh) == 0 && return # no internal variable
    offset = 1
    uq = @view u[(ndofs(f.dh)+1):end]
    for sdh in f.lvh.dh.subdofhandlers
        qr = getquadraturerule(f.integrator.qrc, sdh)
        # ivsize_per_qp = sum(sdh.field_n_components; init=0) # FIXME broken...
        ivsize_per_qp = sum(Ferrite.n_components.(sdh.field_interpolations); init=0)
        ivsize_per_qp == 0 && continue
        material_model = get_material_model(f, sdh)
        for cell in CellIterator(sdh)
            for qp in QuadratureIterator(qr)
                q = @view uq[offset:(offset+ivsize_per_qp-1)]
                default_initial_state!(q, material_model)
                offset += ivsize_per_qp
            end
        end
    end
end

gather_internal_variable_infos(model::QuasiStaticModel) = gather_internal_variable_infos(model.material_model)
gather_internal_variable_infos(model::AbstractMaterialModel) = InternalVariableInfo[]

@unroll function __get_material_model_multi(materials, domains, sdh)
    idx = 1
    @unroll for material ∈ materials
        if first(domains[idx]) ∈ sdh.cellset
            return material
        end
        idx += 1
    end
    error("MultiDomainIntegrator is broken: Requested to construct an internal cache for a SubDofHandler which is not associated with the integrator.")
end
__get_material_model(model::MultiMaterialModel, sdh) = __get_material_model_multi(model.materials, model.domains, sdh)
__get_material_model(model::AbstractMaterialModel, sdh) = model
get_material_model(f::QuasiStaticFunction, sdh) = __get_material_model(f.integrator.volume_model.material_model, sdh)
