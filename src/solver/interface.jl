abstract type AbstractSolver <: DiffEqBase.AbstractDEAlgorithm end
abstract type AbstractNonlinearSolver <: AbstractSolver end

abstract type AbstractNonlinearSolverCache end

abstract type AbstractTimeSolverCache end

# Nonlinear
function setup_operator(f::NullFunction, solver::AbstractSolver)
    return NullOperator{Float64,solution_size(f),solution_size(f)}()
end

# Linear
# Unrolled to disambiguate
function setup_operator(strategy::SequentialAssemblyStrategy{<:AbstractCPUDevice}, ::LinearIntegrator{<:NoStimulationProtocol}, solver::AbstractSolver, dh::AbstractDofHandler)
    LinearNullOperator{value_type(strategy.device), ndofs(dh)}()
end
function setup_operator(strategy::PerColorAssemblyStrategy{<:AbstractCPUDevice}, ::LinearIntegrator{<:NoStimulationProtocol}, solver::AbstractSolver, dh::AbstractDofHandler)
    LinearNullOperator{value_type(strategy.device), ndofs(dh)}()
end
function setup_operator(strategy::ElementAssemblyStrategy{<:AbstractCPUDevice}, ::LinearIntegrator{<:NoStimulationProtocol}, solver::AbstractSolver, dh::AbstractDofHandler)
    LinearNullOperator{value_type(strategy.device), ndofs(dh)}()
end
function setup_operator(strategy::SequentialAssemblyStrategy{<:AbstractGPUDevice}, ::LinearIntegrator{<:NoStimulationProtocol}, solver::AbstractSolver, dh::AbstractDofHandler)
    LinearNullOperator{value_type(strategy.device), ndofs(dh)}()
end
function setup_operator(strategy::PerColorAssemblyStrategy{<:AbstractGPUDevice}, ::LinearIntegrator{<:NoStimulationProtocol}, solver::AbstractSolver, dh::AbstractDofHandler)
    LinearNullOperator{value_type(strategy.device), ndofs(dh)}()
end
function setup_operator(strategy::ElementAssemblyStrategy{<:AbstractGPUDevice}, ::LinearIntegrator{<:NoStimulationProtocol}, solver::AbstractSolver, dh::AbstractDofHandler)
    LinearNullOperator{value_type(strategy.device), ndofs(dh)}()
end
function setup_operator(strategy::SequentialAssemblyStrategy{<:AbstractGPUDevice}, integrator::LinearIntegrator, solver::AbstractSolver, dh::AbstractDofHandler)
    return LinearOperator(
        zeros(value_type(strategy.device), ndofs(dh)),
        integrator,
        dh,
        SequentialAssemblyStrategyCache(strategy.device),
    )
end
function setup_operator(strategy::PerColorAssemblyStrategy{<:AbstractCPUDevice}, integrator::LinearIntegrator, solver::AbstractSolver, dh::AbstractDofHandler)
    return LinearOperator(
        zeros(value_type(strategy.device), ndofs(dh)),
        integrator,
        dh,
        PerColorAssemblyStrategyCache(strategy.device, create_dh_coloring(dh)),
    )
end
function setup_operator(strategy::ElementAssemblyStrategy{<:AbstractCPUDevice},  integrator::LinearIntegrator, solver::AbstractSolver, dh::AbstractDofHandler)
    return LinearOperator(
        zeros(value_type(strategy.device), ndofs(dh)),
        integrator,
        dh,
        ElementAssemblyStrategyCache(strategy.device, EAVector(value_type(strategy.device), index_type(strategy.device), dh)),
    )
end

# Bilinear
function setup_operator(strategy::Union{SequentialAssemblyStrategy{<:AbstractCPUDevice},PerColorAssemblyStrategy{<:AbstractCPUDevice}}, integrator::AbstractBilinearIntegrator, solver::AbstractSolver, dh::AbstractDofHandler)
    setup_assembled_operator(strategy, integrator, solver.system_matrix_type, dh)
end
function setup_assembled_operator(strategy::SequentialAssemblyStrategy{<:AbstractCPUDevice}, integrator::AbstractBilinearIntegrator, system_matrix_type::Type, dh::AbstractDofHandler)
    A  = create_system_matrix(system_matrix_type, dh)
    A_ = if strategy.device isa AbstractCPUDevice && system_matrix_type isa SparseMatrixCSC #if "can assemble with system_matrix_type"
        A
    else
        allocate_matrix(dh)
    end

    return AssembledBilinearOperator(
        A, A_,
        integrator,
        dh,
        SequentialAssemblyStrategyCache(strategy.device),
    )
end
function setup_assembled_operator(strategy::PerColorAssemblyStrategy, integrator::AbstractBilinearIntegrator, system_matrix_type::Type, dh::AbstractDofHandler)
    A  = create_system_matrix(system_matrix_type, dh)
    A_ = if strategy.device isa AbstractCPUDevice && system_matrix_type isa SparseMatrixCSC #if "can assemble with system_matrix_type"
        A
    else
        allocate_matrix(dh)
    end

    return AssembledBilinearOperator(
        A, A_,
        integrator,
        dh,
        PerColorAssemblyStrategyCache(strategy.device, create_dh_coloring(dh)),
    )
end

# Nonlinear
function setup_operator(f::AbstractQuasiStaticFunction, solver::AbstractNonlinearSolver)
    return setup_assembled_nonlinear_operator(get_strategy(f), f, solver)
end
function setup_assembled_nonlinear_operator(strategy::SequentialAssemblyStrategy, f::AbstractQuasiStaticFunction, solver::AbstractNonlinearSolver)
    return AssembledNonlinearOperator(
        allocate_matrix(f.dh),
        f.integrator,
        f.dh,
        SequentialAssemblyStrategyCache(strategy.device),
    )
end
function setup_assembled_nonlinear_operator(strategy::PerColorAssemblyStrategy, f::AbstractQuasiStaticFunction, solver::AbstractNonlinearSolver)
    return AssembledNonlinearOperator(
        allocate_matrix(f.dh),
        f.integrator,
        f.dh,
        PerColorAssemblyStrategyCache(strategy.device, create_dh_coloring(f.dh)),
    )
end

# # TODO correct dispatches
# function setup_coupling_operator(first_problem::DiffEqBase.AbstractDEProblem, second_problem::DiffEqBase.AbstractDEProblem, relevant_couplings, solver::AbstractNonlinearSolver)
#     NullOperator{Float64,solution_size(second_problem),solution_size(first_problem)}()
# end

# # Block-Diagonal entry
# setup_operator(coupled_problem::CoupledProblem, i::Int, solver) = setup_operator(coupled_problem.base_problems[i], coupled_problem.couplings, solver)
# # Offdiagonal entry
# setup_coupling_operator(coupled_problem::CoupledProblem, i::Int, j::Int, solver) = setup_coupling_operator(coupled_problem.base_problems[i], coupled_problem.base_problems[j], coupled_problem.couplings, solver)

function update_constraints!(f::AbstractSemidiscreteFunction, solver_cache::AbstractTimeSolverCache, t)
    Ferrite.update!(getch(f), t)
    apply!(solver_cache.uₙ, getch(f))
end

update_constraints!(f, solver_cache::AbstractTimeSolverCache, t) = nothing

function update_constraints!(f::AbstractSemidiscreteBlockedFunction, solver_cache::AbstractTimeSolverCache, t)
    for (i,pi) ∈ enumerate(blocks(f))
        update_constraints_block!(pi, Block(i), solver_cache, t)
    end
end

function update_constraints_block!(f::AbstractSemidiscreteFunction, i::Block, solver_cache::AbstractTimeSolverCache, t)
    Ferrite.update!(getch(f), t)
    u = @view solver_cache.uₙ[i]
    apply!(u, getch(f))
end

update_constraints_block!(f::DiffEqBase.AbstractDiffEqFunction, i::Block, solver_cache::AbstractTimeSolverCache, t) = nothing

update_constraints_block!(f::NullFunction, i::Block, solver_cache::AbstractTimeSolverCache, t) = nothing


create_system_matrix(T::Type{<:AbstractMatrix}, f::AbstractSemidiscreteFunction) = create_system_matrix(T, f.dh)

function create_system_matrix(::Type{<:ThreadedSparseMatrixCSR{Tv,Ti}}, dh::AbstractDofHandler) where {Tv,Ti}
    Acsct = transpose(convert(SparseMatrixCSC{Tv,Ti}, allocate_matrix(dh)))
    return ThreadedSparseMatrixCSR(Acsct)
end

function create_system_matrix(SpMatType::Type{<:SparseMatrixCSC}, dh::AbstractDofHandler)
    A = convert(SpMatType, allocate_matrix(dh))
    return A
end

function create_system_vector(::Type{<:Vector{T}}, f::AbstractSemidiscreteFunction) where T
    return zeros(T, solution_size(f))
end

function create_system_vector(::Type{<:Vector{T}}, dh::DofHandler) where T
    return zeros(T, ndofs(dh))
end
