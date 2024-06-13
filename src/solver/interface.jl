abstract type AbstractSolver <: DiffEqBase.AbstractDEAlgorithm end
abstract type AbstractNonlinearSolver <: AbstractSolver end

abstract type AbstractNonlinearSolverCache end

abstract type AbstractTimeSolverCache end

function setup_operator(f::NullFunction, solver::AbstractSolver)
    return NullOperator{Float64,solution_size(f),solution_size(f)}()
end

function setup_operator(f::AbstractQuasiStaticFunction, solver::AbstractNonlinearSolver)
    @unpack dh, constitutive_model, face_models = f
    @assert length(dh.subdofhandlers) == 1 "Multiple subdomains not yet supported in the nonlinear solver."
    @assert length(dh.field_names) == 1 "Multiple fields not yet supported in the nonlinear solver."

    displacement_symbol = first(dh.field_names)

    intorder = quadrature_order(f, displacement_symbol)::Int
    qr = QuadratureRuleCollection(intorder)
    qr_face = FacetQuadratureRuleCollection(intorder)

    return AssembledNonlinearOperator(
        dh, displacement_symbol, constitutive_model, qr, face_models, qr_face
    )
end

function setup_operator(::NoStimulationProtocol, solver::AbstractSolver, dh::AbstractDofHandler, field_name::Symbol, qr)
    check_subdomains(dh)
    LinearNullOperator{Float64, ndofs(dh)}()
end

function setup_operator(protocol::AnalyticalTransmembraneStimulationProtocol, solver::AbstractSolver, dh::AbstractDofHandler, field_name::Symbol, qr)
    check_subdomains(dh)
    ip_g = Ferrite.geometric_interpolation(typeof(getcells(grid, 1)))
    qr = QuadratureRule{Ferrite.getrefshape(ip_g)}(Ferrite.getorder(ip_g)+1)
    cv = CellValues(qr, ip, ip_g) # TODO replace with something more lightweight
    return PEALinearOperator(
        zeros(ndofs(dh)),
        AnalyticalCoefficientElementCache(
            protocol.f,
            protocol.nonzero_intervals,
            cv
        ),
        dh,
    )
end

function setup_assembled_operator(integrator::AbstractBilinearIntegrator, system_matrix_type::Type, dh::AbstractDofHandler, field_name::Symbol, qrc)
    @assert length(dh.subdofhandlers) == 1 "Multiple subdomains not yet supported in the bilinear opeartor."

    firstcell = getcells(Ferrite.get_grid(dh), first(dh.subdofhandlers[1].cellset))
    ip = Ferrite.getfieldinterpolation(dh.subdofhandlers[1], field_name)
    ip_geo = Ferrite.geometric_interpolation(typeof(firstcell))
    element_qr = getquadraturerule(qrc, firstcell)

    element_cache = setup_element_cache(integrator, element_qr, ip, ip_geo)

    A  = create_system_matrix(system_matrix_type, dh)
    A_ = create_sparsity_pattern(dh) #  TODO how to query this?
    return AssembledBilinearOperator(
        A, A_,
        element_cache,
        dh,
    )
end

function setup_operator(integrator::AbstractBilinearIntegrator, solver::AbstractSolver, dh::AbstractDofHandler, field_name::Symbol, qrc)
    setup_assembled_operator(integrator, solver.system_matrix_type, dh, field_name, qrc)
end

# function setup_operator(problem::QuasiStaticProblem, relevant_coupler, solver::AbstractNonlinearSolver)
#     @unpack dh, constitutive_model, face_models, displacement_symbol = problem
#     @assert length(dh.subdofhandlers) == 1 "Multiple subdomains not yet supported in the Newton solver."
#     @assert length(dh.field_names) == 1 "Multiple fields not yet supported in the nonlinear solver."

#     intorder = quadrature_order(problem, displacement_symbol)
#     qr = QuadratureRuleCollection(intorder)
#     qr_face = FacetQuadratureRuleCollection(intorder)

#     return AssembledNonlinearOperator(
#         dh, displacement_symbol, constitutive_model, qr, face_models, qr_face, relevant_coupler, ???, <- depending on the coupler either face or element qr
#     )
# end

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
    Acsct = transpose(convert(SparseMatrixCSC{Tv,Ti}, create_sparsity_pattern(dh)))
    return ThreadedSparseMatrixCSR(Acsct)
end

function create_system_matrix(SpMatType::Type{<:SparseMatrixCSC}, dh::AbstractDofHandler)
    A = convert(SpMatType, create_sparsity_pattern(dh))
    return A
end

function create_system_vector(::Type{<:Vector{T}}, f::AbstractSemidiscreteFunction) where T
    return zeros(T, solution_size(f))
end

function create_system_vector(::Type{<:Vector{T}}, dh::DofHandler) where T
    return zeros(T, ndofs(dh))
end

function create_quadrature_rule(f::AbstractSemidiscreteFunction, solver::AbstractSolver, field_name::Symbol)
    intorder = quadrature_order(f, field_name)
    return QuadratureRuleCollection(intorder)
end
