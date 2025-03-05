#########################################################
#  This file contains optimized backward Euler solvers  #
#########################################################
Base.@kwdef struct BackwardEulerSolver{SolverType, SolutionVectorType, SystemMatrixType, MonitorType} <: AbstractSolver
    inner_solver::SolverType                       = LinearSolve.KrylovJL_CG()
    solution_vector_type::Type{SolutionVectorType} = Vector{Float64}
    system_matrix_type::Type{SystemMatrixType}     = ThreadedSparseMatrixCSR{Float64, Int64}
    # mass operator info
    # diffusion opeartor info
    # DO NOT USE THIS (will be replaced by proper logging system)
    monitor::MonitorType = DefaultProgressMonitor()
end

SciMLBase.isadaptive(::BackwardEulerSolver) = false

mutable struct BackwardEulerSolverCache{T, SolutionType <: AbstractVector{T}, StageType, MonitorType} <: AbstractTimeSolverCache
    # Current solution buffer
    uₙ::SolutionType
    # Last solution buffer
    uₙ₋₁::SolutionType
    # # Temporary buffer for interpolations and stuff
    # tmp::SolutionType
    # Utility to decide what kind of stage we solve (i.e. linear problem, full DAE or mass-matrix ODE)
    stage::StageType
    # DO NOT USE THIS (will be replaced by proper logging system)
    monitor::MonitorType
end

# Performs a backward Euler step
function perform_step!(f, cache::BackwardEulerSolverCache, t, Δt)
    perform_backward_euler_step!(f, cache, cache.stage, t, Δt)
end

#########################################################
#                   Affine Problems                     #
#########################################################
# Mutable to change Δt_last
mutable struct BackwardEulerAffineODEStage{T, MassMatrixType, DiffusionMatrixType, SourceTermType, SolverCacheType}
    # Mass matrix
    M::MassMatrixType
    # Diffusion matrix
    K::DiffusionMatrixType
    # Helper for possible source terms
    source_term::SourceTermType
    # Linear solver for (M - Δtₙ₋₁ K) uₙ = M uₙ₋₁  + f
    linear_solver::SolverCacheType
    # Last time step length as a check if we have to update A
    Δt_last::T
end

function perform_backward_euler_step!(f::AffineODEFunction, cache::BackwardEulerSolverCache, stage::BackwardEulerAffineODEStage, t, Δt)
    @unpack uₙ, uₙ₋₁ = cache
    @unpack linear_solver, M, Δt_last = stage

    # Update matrix if time step length has changed
    Δt ≈ Δt_last || implicit_euler_heat_solver_update_system_matrix!(stage, Δt)

    # Prepare right hand side b = M uₙ₋₁
    @timeit_debug "b = M uₙ₋₁" mul!(linear_solver.b, M, uₙ₋₁)

    # Update source term
    @timeit_debug "update source term" begin
        implicit_euler_heat_update_source_term!(stage, t + Δt)
        add!(linear_solver.b, stage.source_term)
    end

    # Solve linear problem, where sol.u === uₙ
    @timeit_debug "inner solve" sol = LinearSolve.solve!(linear_solver)
    solve_failed = !(DiffEqBase.SciMLBase.successful_retcode(sol.retcode) || sol.retcode == DiffEqBase.ReturnCode.Default)
    linear_finalize_monitor(linear_solver, cache.monitor, sol)
    return !solve_failed
end

# Helper to get A into the right form
function implicit_euler_heat_solver_update_system_matrix!(cache::BackwardEulerAffineODEStage, Δt)
    _implicit_euler_heat_solver_update_system_matrix!(cache.linear_solver.A, cache.M, cache.K, Δt)

    cache.Δt_last = Δt
end

function _implicit_euler_heat_solver_update_system_matrix!(A, M, K, Δt)
    # nonzeros(A) .= nonzeros(M.A) .- Δt.*nonzeros(K.A)
    Anz = nonzeros(A)
    Knz = nonzeros(K.A)
    Mnz = nonzeros(M.A)
    @inbounds @.. Anz = Mnz - Δt * Knz
end

function implicit_euler_heat_update_source_term!(cache::BackwardEulerAffineODEStage, t)
    needs_update(cache.source_term, t) && update_operator!(cache.source_term, t)
end

function setup_solver_cache(f::AffineODEFunction, solver::BackwardEulerSolver, t₀; u = nothing, uprev = nothing)
    @unpack dh = f
    @unpack inner_solver = solver
    @assert length(dh.field_names) == 1 # TODO relax this assumption
    field_name = dh.field_names[1]

    A     = create_system_matrix(solver.system_matrix_type  , f)
    b     = create_system_vector(solver.solution_vector_type, f)
    u0    = u === nothing ? create_system_vector(solver.solution_vector_type, f) : u
    uprev = uprev === nothing ? create_system_vector(solver.solution_vector_type, f) : uprev

    T = eltype(u0)

    # Left hand side ∫dₜu δu dV
    mass_operator = setup_operator(
        f.mass_term,
        solver, dh,
    )

    # Affine right hand side, e.g. ∫D grad(u) grad(δu) dV + ...
    bilinear_operator = setup_operator(
        f.bilinear_term,
        solver, dh,
    )
    # ... + ∫f δu dV
    source_operator    = setup_operator(
        f.source_term,
        solver, dh,
        f.bilinear_term.qrc, # source follows linearity of diffusion for now...
    )

    inner_prob  = LinearSolve.LinearProblem(
        A, b; u0
    )
    inner_cache = init(inner_prob, inner_solver)

    cache       = BackwardEulerSolverCache(
        u0, # u
        uprev,
        # tmp,
        BackwardEulerAffineODEStage(
            mass_operator,
            bilinear_operator,
            source_operator,
            inner_cache,
            T(0.0),
        ),
        solver.monitor,
    )

    @timeit_debug "initial assembly" begin
        update_operator!(mass_operator, t₀)
        update_operator!(bilinear_operator, t₀)
        update_operator!(source_operator, t₀)
    end

    return cache
end

#########################################################
#                     DAE Problems                      #
#########################################################

struct BackwardEulerStageCache{SolverType}
    # Nonlinear solver for generic backward Euler discretizations
    nlsolver::SolverType
end

# This is an annotation to setup the operator in the inner nonlinear problem correctly.
struct BackwardEulerStageAnnotation{F,U}
    f::F
    u::U
    uprev::U
end

abstract type AbstractTimeDiscretizationAnnotation{T} end

# This is the wrapper used to communicate solver info into the operator.
# In a nutshell this should contain all information to setup the evaluation of the nonlinear problem to make the backward Euler step.
mutable struct BackwardEulerStageFunctionWrapper{F,U,T,S, LVH} <: AbstractTimeDiscretizationAnnotation{F}
    const f::F
    const u::U
    const uprev::U
    Δt::T
    const inner_solver::S
    const lvh::LVH
end

function extract_global_function(f)
    return f
end

function extract_local_function(f)
    return f
end

# We unpack to dispatch per function class
function setup_solver_cache(wrapper::BackwardEulerStageAnnotation, solver::AbstractNonlinearSolver)
    _setup_solver_cache(wrapper, wrapper.f, solver)
end
@inline function _setup_solver_cache(wrapper::BackwardEulerStageAnnotation, f::QuasiStaticFunction, solver::MultiLevelNewtonRaphsonSolver)
    @unpack integrator, dh = f
    @unpack volume_model, face_model = integrator
    @unpack local_solver, newton = solver

    # Extract condensable parts
    Q     = @view wrapper.u[(ndofs(dh)+1):end]
    Qprev = @view wrapper.uprev[(ndofs(dh)+1):end]
    # Connect nonlinear problem and timestepper
    volume_wrapper = BackwardEulerStageFunctionWrapper(
        volume_model,
        Q, Qprev,
        0.0,
        nothing, # inner_solver_cache.local_solver_cache,
        f.lvh,
    )
    face_wrapper = BackwardEulerStageFunctionWrapper(
        face_model,
        Q, Qprev,
        0.0,
        nothing, # inner model is volume only per construction
        f.lvh,
    )
    # This is copy paste of setup_solver_cache(G, solver.newton)
    op = AssembledNonlinearOperator(
        NonlinearIntegrator(volume_wrapper, face_wrapper, integrator.syms, integrator.qrc, integrator.fqrc), dh,
    )
    # op = setup_operator(f, solver)
    T = Float64
    residual = Vector{T}(undef, ndofs(dh))#solution_size(G))
    Δu = Vector{T}(undef, ndofs(dh))#solution_size(G))

    # Connect both solver caches
    inner_prob = LinearSolve.LinearProblem(
        getJ(op), residual; u0=Δu
    )
    inner_cache = init(inner_prob, newton.inner_solver; alias_A=true, alias_b=true)
    @assert inner_cache.b === residual
    @assert inner_cache.A === getJ(op)

    newton_cache = NewtonRaphsonSolverCache(op, residual, newton, inner_cache, T[], 0)

    return MultiLevelNewtonRaphsonSolverCache(
        # FIXME global_f and local_f :)
        newton_cache, # setup_solver_cache(G, solver.newton),
        nothing, #setup_solver_cache(L, solver.local_newton), # FIXME pass
    )
end

# TODO Refactor the setup into generic parts and use multiple dispatch for the specifics.
function setup_solver_cache(f::AbstractSemidiscreteFunction, solver::BackwardEulerSolver, t₀;
        uprev = nothing,
        u = nothing,
        alias_uprev = true,
        alias_u     = false,
    )
    vtype = Vector{Float64}

    if u === nothing
        _u = vtype(undef, solution_size(f))
        @warn "Cannot initialize u for $(typeof(solver))."
    else
        _u = alias_u ? u : SciMLBase.recursivecopy(u)
    end

    if uprev === nothing
        _uprev = vtype(undef, solution_size(f))
        _uprev .= u
    else
        _uprev = alias_uprev ? uprev : SciMLBase.recursivecopy(uprev)
    end

    cache       = BackwardEulerSolverCache(
        _u,
        _uprev,
        # tmp,
        BackwardEulerStageCache(
            setup_solver_cache(BackwardEulerStageAnnotation(f, _u, _uprev), solver.inner_solver)
        ),
        solver.monitor,
    )

    return cache
end

# The idea is simple. QuasiStaticModels always have the form
#    0 = G(u,v)
#    0 = L(u,v,dₜu,dₜv)     (or simpler dₜv = L(u,v))
# so we pass the stage information into the interior.
function setup_element_cache(wrapper::AbstractTimeDiscretizationAnnotation{<:QuasiStaticModel}, qr::QuadratureRule, sdh::SubDofHandler)
    @assert length(sdh.dh.field_names) == 1 "Support for multiple fields not yet implemented."
    field_name = first(sdh.dh.field_names)
    ip          = Ferrite.getfieldinterpolation(sdh, field_name)
    ip_geo = geometric_subdomain_interpolation(sdh)
    cv = CellValues(qr, ip, ip_geo)
    return QuasiStaticElementCache(
        wrapper.f.material_model,
        setup_coefficient_cache(wrapper.f.material_model, qr, sdh),
        setup_internal_cache(wrapper, qr, sdh),
        cv
    )
end

# update_stage!(stage::BackwardEulerStageCache, kΔt) = update_stage!(stage, stage.nlsolver.local_solver_cache.op, kΔt)
update_stage!(stage::BackwardEulerStageCache, kΔt) = update_stage!(stage, stage.nlsolver.global_solver_cache.op, kΔt)
function update_stage!(stage::BackwardEulerStageCache, op::AssembledNonlinearOperator, kΔt)
    op.integrator.volume_model.Δt = kΔt
    op.integrator.face_model.Δt   = kΔt
end

function perform_backward_euler_step!(f::QuasiStaticFunction, cache::BackwardEulerSolverCache, stage_info::BackwardEulerStageCache, t, Δt)
    update_constraints!(f, cache, t + Δt)
    update_stage!(stage_info, Δt)
    if !nlsolve!(cache.uₙ, f, stage_info.nlsolver, t + Δt)
        return false
    end
    return true
end

function setup_internal_cache(wrapper::BackwardEulerStageFunctionWrapper{<:QuasiStaticModel}, qr::QuadratureRule, sdh::SubDofHandler)
    n_ivs_per_qp = local_function_size(wrapper.f.material_model)
    return GenericFirstOrderRateIndependentMaterialStateCache(
        wrapper.f,
        wrapper.u,
        wrapper.uprev,
        wrapper.Δt,
        wrapper.lvh,
        zeros(n_ivs_per_qp),
        zeros(n_ivs_per_qp),
    )
end

function setup_boundary_cache(wrapper::BackwardEulerStageFunctionWrapper, fqr, sdh)
    # TODO this technically unlocks differential boundary conditions, if done correctly.
    setup_boundary_cache(wrapper.f, fqr, sdh)
end

#########################################################
#                     ODE Problems                      #
#########################################################

# Multi-rate version
Base.@kwdef struct ForwardEulerSolver{SolutionVectorType} <: AbstractSolver
    rate::Int
    solution_vector_type::Type{SolutionVectorType} = Vector{Float64}
end

mutable struct ForwardEulerSolverCache{VT,VTrate,VTprev,F} <: AbstractTimeSolverCache
    rate::Int
    du::VTrate
    uₙ::VT
    uₙ₋₁::VTprev
    rhs!::F
end

function perform_step!(f::ODEFunction, solver_cache::ForwardEulerSolverCache, t::Float64, Δt::Float64)
    @unpack rate, du, uₙ, rhs! = solver_cache
    Δtsub = Δt/rate
    for i ∈ 1:rate
        @inbounds rhs!(du, uₙ, t, f.p)
        @inbounds @.. uₙ = uₙ + Δtsub * du
        t += Δtsub
    end

    return !any(isnan.(uₙ))
end

function setup_solver_cache(f::ODEFunction, solver::ForwardEulerSolver, t₀; u = nothing, uprev = nothing)
    du = create_system_vector(solver.solution_vector_type, f)
    u = u === nothing ? create_system_vector(solver.solution_vector_type, f) : u
    return ForwardEulerSolverCache(
        solver.rate,
        du,
        u,
        u,
        f.f
    )
end
