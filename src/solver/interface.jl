abstract type AbstractSolver <: SciMLBase.AbstractDEAlgorithm end
abstract type AbstractNonlinearSolver <: AbstractSolver end

abstract type AbstractNonlinearSolverCache end

abstract type AbstractTimeSolverCache end

# TODO remove
getJ(op) = op.J

# Nonlinear
function setup_stage_operator(f::NullFunction, solver::AbstractSolver)
    return NullOperator{Float64, solution_size(f), solution_size(f)}()
end

# Linear
# Unrolled to disambiguate
function setup_operator(
    strategy::SequentialAssemblyStrategy{<:AbstractCPUDevice},
    ::LinearIntegrator{<:NoStimulationProtocol},
    solver::AbstractSolver,
    dh::AbstractDofHandler,
)
    LinearNullOperator{value_type(strategy.device), ndofs(dh)}()
end
function setup_operator(
    strategy::PerColorAssemblyStrategy{<:AbstractCPUDevice},
    ::LinearIntegrator{<:NoStimulationProtocol},
    solver::AbstractSolver,
    dh::AbstractDofHandler,
)
    LinearNullOperator{value_type(strategy.device), ndofs(dh)}()
end
function setup_operator(
    strategy::ElementAssemblyStrategy{<:AbstractCPUDevice},
    ::LinearIntegrator{<:NoStimulationProtocol},
    solver::AbstractSolver,
    dh::AbstractDofHandler,
)
    LinearNullOperator{value_type(strategy.device), ndofs(dh)}()
end
function setup_operator(
    strategy::SequentialAssemblyStrategy{<:AbstractGPUDevice},
    ::LinearIntegrator{<:NoStimulationProtocol},
    solver::AbstractSolver,
    dh::AbstractDofHandler,
)
    LinearNullOperator{value_type(strategy.device), ndofs(dh)}()
end
function setup_operator(
    strategy::PerColorAssemblyStrategy{<:AbstractGPUDevice},
    ::LinearIntegrator{<:NoStimulationProtocol},
    solver::AbstractSolver,
    dh::AbstractDofHandler,
)
    LinearNullOperator{value_type(strategy.device), ndofs(dh)}()
end
function setup_operator(
    strategy::ElementAssemblyStrategy{<:AbstractGPUDevice},
    ::LinearIntegrator{<:NoStimulationProtocol},
    solver::AbstractSolver,
    dh::AbstractDofHandler,
)
    LinearNullOperator{value_type(strategy.device), ndofs(dh)}()
end

function setup_operator(
    strategy,
    integrator::AbstractLinearIntegrator,
    solver::AbstractSolver,
    dh::AbstractDofHandler,
)
    return setup_operator(strategy, integrator, dh)
end

# Bilinear
function setup_operator(
    strategy::Union{
        SequentialAssemblyStrategy{<:AbstractCPUDevice},
        PerColorAssemblyStrategy{<:AbstractCPUDevice},
    },
    integrator::AbstractBilinearIntegrator,
    solver::AbstractSolver,
    dh::AbstractDofHandler,
)
    setup_assembled_operator(strategy, integrator, solver.system_matrix_type, dh)
end
function setup_assembled_operator(
    strategy::SequentialAssemblyStrategy{<:AbstractCPUDevice},
    integrator::AbstractBilinearIntegrator,
    system_matrix_type::Type,
    dh::AbstractDofHandler,
)
    setup_operator(strategy, integrator, dh) # FIXME
end

# Nonlinear
"""
    setup_stage_operator(f, solver)

The nonlinear operator a solver works on for the function `f`.

Distinct from the `setup_operator(strategy, integrator, dh)` family, which materializes one integrator
against one handler and knows nothing about who will solve with it. This one answers "what is the
nonlinear problem posed by `f`", which is what an [`AbstractStageFunction`](@ref) carries -- directly
for a continuation, wrapped in a scheme's own operator for a time step.

The integrator and the dof handler it assembles against are separate queries
([`get_volume_integrator`](@ref), [`get_assembly_dh`](@ref)) because they need not come from the same
object.
"""
function setup_stage_operator(f::AbstractSolidMechanicsFunction, solver::AbstractNonlinearSolver)
    return setup_operator(get_strategy(f), get_volume_integrator(f), get_assembly_dh(f))
end

function update_constraints!(
    f::AbstractSemidiscreteFunction,
    solver_cache::AbstractTimeSolverCache,
    t,
)
    Ferrite.update!(getch(f), t)
    apply!(solver_cache.uₙ, getch(f))
end

update_constraints!(f, solver_cache::AbstractTimeSolverCache, t) = nothing

function update_constraints!(
    f::AbstractSemidiscreteBlockedFunction,
    solver_cache::AbstractTimeSolverCache,
    t,
)
    for (i, pi) ∈ enumerate(blocks(f))
        update_constraints_block!(pi, Block(i), solver_cache, t)
    end
end

function update_constraints_block!(
    f::AbstractSemidiscreteFunction,
    i::Block,
    solver_cache::AbstractTimeSolverCache,
    t,
)
    Ferrite.update!(getch(f), t)
    u = @view solver_cache.uₙ[i]
    apply!(u, getch(f))
end

update_constraints_block!(
    f::SciMLBase.AbstractDiffEqFunction,
    i::Block,
    solver_cache::AbstractTimeSolverCache,
    t,
) = nothing

update_constraints_block!(f::NullFunction, i::Block, solver_cache::AbstractTimeSolverCache, t) =
    nothing


create_system_matrix(T::Type{<:AbstractMatrix}, f::AbstractSemidiscreteFunction) =
    create_system_matrix(T, f.dh)

function create_system_matrix(
    ::Type{<:ThreadedSparseMatrixCSR{Tv, Ti}},
    dh::AbstractDofHandler,
) where {Tv, Ti}
    Acsct = transpose(convert(SparseMatrixCSC{Tv, Ti}, allocate_matrix(dh)))
    return ThreadedSparseMatrixCSR(Acsct)
end

function create_system_matrix(SpMatType::Type{<:SparseMatrixCSC}, dh::AbstractDofHandler)
    A = convert(SpMatType, allocate_matrix(dh))
    return A
end

function create_system_vector(::Type{<:Vector{T}}, f::AbstractSemidiscreteFunction) where {T}
    return zeros(T, solution_size(f))
end

function create_system_vector(::Type{<:Vector{T}}, dh::DofHandler) where {T}
    return zeros(T, ndofs(dh))
end
