#########################################################
########################## TIME #########################
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

# TODO decouple from heat problem via special ODEFunction (AffineODEFunction)
mutable struct BackwardEulerSolverCache{T, SolutionType <: AbstractVector{T}, MassMatrixType, DiffusionMatrixType, SourceTermType, SolverCacheType, MonitorType} <: AbstractTimeSolverCache
    # Current solution buffer
    uₙ::SolutionType
    # Last solution buffer
    uₙ₋₁::SolutionType
    # Mass matrix
    M::MassMatrixType
    # Diffusion matrix
    K::DiffusionMatrixType
    # Helper for possible source terms
    source_term::SourceTermType
    # Linear solver for (M - Δtₙ₋₁ K) uₙ = M uₙ₋₁
    inner_solver::SolverCacheType
    # Last time step length as a check if we have to update K
    Δt_last::T
    # DO NOT USE THIS (will be replaced by proper logging system)
    monitor::MonitorType
end

# Helper to get A into the right form
function implicit_euler_heat_solver_update_system_matrix!(cache::BackwardEulerSolverCache, Δt)
    _implicit_euler_heat_solver_update_system_matrix!(cache.inner_solver.A, cache.M, cache.K, Δt)

    cache.Δt_last = Δt
end

function _implicit_euler_heat_solver_update_system_matrix!(A, M, K, Δt)
    # nonzeros(A) .= nonzeros(M.A) .- Δt.*nonzeros(K.A)
    Anz = nonzeros(A)
    Knz = nonzeros(K.A)
    Mnz = nonzeros(M.A)
    @inbounds @.. Anz = Mnz - Δt * Knz
end

function implicit_euler_heat_update_source_term!(cache::BackwardEulerSolverCache, t)
    needs_update(cache.source_term, t) && update_operator!(cache.source_term, t)
end

# Performs a backward Euler step
function perform_step!(f::TransientDiffusionFunction, cache::BackwardEulerSolverCache, t, Δt)
    @unpack Δt_last, M, uₙ, uₙ₋₁, inner_solver = cache
    # Update matrix if time step length has changed
    Δt ≈ Δt_last || implicit_euler_heat_solver_update_system_matrix!(cache, Δt)
    # Prepare right hand side b = M uₙ₋₁
    @timeit_debug "b = M uₙ₋₁" mul!(inner_solver.b, M, uₙ₋₁)
    # TODO How to remove these two lines here?
    # Update source term
    @timeit_debug "update source term" begin
        implicit_euler_heat_update_source_term!(cache, t + Δt)
        add!(inner_solver.b, cache.source_term)
    end
    # Solve linear problem
    @timeit_debug "inner solve" sol = LinearSolve.solve!(inner_solver)
    solve_failed = !(DiffEqBase.SciMLBase.successful_retcode(sol.retcode) || sol.retcode == DiffEqBase.ReturnCode.Default)
    linear_finalize_monitor(inner_solver, cache.monitor, sol)
    return !solve_failed
end

function setup_solver_cache(f::TransientDiffusionFunction, solver::BackwardEulerSolver, t₀; u = nothing, uprev = nothing)
    @unpack dh = f
    @unpack inner_solver = solver
    @assert length(dh.field_names) == 1 # TODO relax this assumption, maybe.
    field_name = dh.field_names[1]

    A     = create_system_matrix(solver.system_matrix_type  , f)
    b     = create_system_vector(solver.solution_vector_type, f)
    u0    = u === nothing ? create_system_vector(solver.solution_vector_type, f) : u
    uprev = uprev === nothing ? create_system_vector(solver.solution_vector_type, f) : uprev

    T = eltype(u0)

    qr = create_quadrature_rule(f, solver, field_name)

    # Left hand side ∫dₜu δu dV
    mass_operator = setup_operator(
        BilinearMassIntegrator(
            ConstantCoefficient(T(1.0))
        ),
        solver, dh, field_name, qr
    )

    # Affine right hand side ∫D grad(u) grad(δu) dV + ...
    diffusion_operator = setup_operator(
        BilinearDiffusionIntegrator(
            f.diffusion_tensor_field,
        ),
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
        mass_operator,
        diffusion_operator,
        source_operator,
        inner_cache,
        T(0.0),
        solver.monitor,
    )

    @timeit_debug "initial assembly" begin
        update_operator!(mass_operator, t₀)
        update_operator!(diffusion_operator, t₀)
        update_operator!(source_operator, t₀)
    end

    return cache
end

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

function perform_step!(f::ODEFunction, solver_cache::ForwardEulerSolverCache, t, Δt)
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
