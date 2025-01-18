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
    # Last time step length as a check if we have to update K
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
    # nonzeros(A) .= nonzeros(M.A) - Δt*nonzeros(K.A)
    nonzeros(A) .= nonzeros(K.A)
    nonzeros(A) .*= -Δt
    nonzeros(A) .+= nonzeros(M.A)
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

    qr = create_quadrature_rule(f, solver, field_name)

    # Left hand side ∫dₜu δu dV
    mass_operator = setup_operator(
        f.mass_term,
        solver, dh, field_name, qr
    )

    # Affine right hand side, e.g. ∫D grad(u) grad(δu) dV + ...
    bilinear_operator = setup_operator(
        f.bilinear_term,
        solver, dh, field_name, qr
    )
    # ... + ∫f δu dV
    source_operator    = setup_operator(
        f.source_term,
        solver, dh, field_name, qr
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

struct BackwardEulerDAEStage{SolverType}
    nlsolver::SolverType
end

function perform_backward_euler_step!(f, cache::BackwardEulerSolverCache, stage::BackwardEulerDAEStage, t, Δt)
    # What to do here?
    @warn "Not implemented yet."
    return false
end

function setup_solver_cache(f::MultiLevelFunction, solver::BackwardEulerSolver, t₀; u = nothing, uprev = nothing)
    inner_solver_cache = setup_solver_cache(f, solver.inner_solver)

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

    cache = BackwardEulerSolverCache(
        u0, # u
        uprev,
        # tmp,
        BackwardEulerDAEStage(
            # mass_operator,
            # bilinear_operator,
            # source_operator,
            inner_solver_cache,
            # T(0.0),
        ),
        solver.monitor,
    )

    return cache
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
        @inbounds uₙ .= uₙ .+ Δtsub .* du
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
