#####################################################################
#  This file contains optimized forward and backward Euler solvers  #
#####################################################################
Base.@kwdef struct BackwardEulerSolver{
    SolverType,
    SolutionVectorType,
    SystemMatrixType,
    MonitorType,
} <: AbstractSolver
    inner_solver::SolverType                       = LinearSolve.KrylovJL_CG()
    solution_vector_type::Type{SolutionVectorType} = Vector{Float64}
    system_matrix_type::Type{SystemMatrixType}     = ThreadedSparseMatrixCSR{Float64, Int64}
    # DO NOT USE THIS (will be replaced by proper logging system)
    monitor::MonitorType = DefaultProgressMonitor()
end

SciMLBase.isadaptive(::BackwardEulerSolver) = false
OrdinaryDiffEqCore.default_controller(QT, ::BackwardEulerSolver) =
    OrdinaryDiffEqCore.DummyController()

mutable struct BackwardEulerSolverCache{
    T,
    SolutionType <: AbstractVector{T},
    # On the operator splitting path uₙ views the outer solution vector while uₙ₋₁ is the
    # integrator's own rollback buffer, so the two types differ.
    PrevSolutionType <: AbstractVector{T},
    TmpType <: AbstractVector{T},
    StageType,
    MonitorType,
} <: AbstractTimeSolverCache
    # Current solution buffer
    uₙ::SolutionType
    # Last solution buffer
    uₙ₋₁::PrevSolutionType
    # # Temporary buffer for interpolations and stuff
    tmp::TmpType
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
mutable struct BackwardEulerAffineODEStage{
    T,
    MassMatrixType,
    DiffusionMatrixType,
    SourceTermType,
    SolverCacheType,
}
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

function perform_backward_euler_step!(
    f::AffineODEFunction,
    cache::BackwardEulerSolverCache,
    stage::BackwardEulerAffineODEStage,
    t,
    Δt,
)
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
    solve_failed = !(
        DiffEqBase.SciMLBase.successful_retcode(sol.retcode) ||
        sol.retcode == DiffEqBase.ReturnCode.Default
    )
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

function setup_solver_cache(
    f::AffineODEFunction,
    solver::BackwardEulerSolver,
    t₀;
    u = nothing,
    uprev = nothing,
)
    @unpack dh = f
    @unpack inner_solver = solver
    @assert length(dh.field_names) == 1 # TODO relax this assumption
    field_name = dh.field_names[1]

    A = create_system_matrix(solver.system_matrix_type, f)
    b = create_system_vector(solver.solution_vector_type, f)
    u0 = u === nothing ? create_system_vector(solver.solution_vector_type, f) : u
    uprev = uprev === nothing ? create_system_vector(solver.solution_vector_type, f) : uprev
    uprev .= u0

    T = eltype(u0)

    # Left hand side ∫dₜu δu dV
    mass_operator = setup_operator(get_strategy(f), f.mass_term, solver, dh)

    # Affine right hand side, e.g. ∫D grad(u) grad(δu) dV + ...
    bilinear_operator = setup_operator(get_strategy(f), f.bilinear_term, solver, dh)
    # ... + ∫f δu dV
    source_operator = setup_operator(
        ElementAssemblyStrategy(get_strategy(f).device), #The EA strategy should always outperform other strats for the linear operator
        f.source_term,
        solver,
        dh,
    )

    inner_prob  = LinearSolve.LinearProblem(A, b; u0)
    inner_cache = init(inner_prob, inner_solver)

    cache = BackwardEulerSolverCache(
        u0, # u
        uprev,
        copy(u0),
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
struct BackwardEulerStageAnnotation{F, U}
    f::F
    u::U
    uprev::U
end

# Marks a model tree rewritten to carry solver-side information down to the element caches.
abstract type AbstractModelAnnotation{T} end

# This is the wrapper used to communicate solver info into the operator.
# Carries the *solver-owned* state that has to reach the element caches, and nothing else. Since
# `gto1` supplies the previous solution and the timestep as call parameters, the only thing left to
# inject is the local nonlinear solver cache, which the material routine needs for the per-quadrature
# point Newton (`materials.jl`, `solve_internal_timestep`).
#
# Formerly `BackwardEulerStageFunctionWrapper`, which additionally carried `u`, `uprev` and a mutable
# `Δt` that `update_stage!` wrote into before every step. Both are gone: this no longer encodes any
# time discretization, hence the rename.
struct LocalSolverCacheAnnotation{F, S} <: AbstractModelAnnotation{F}
    f::F
    local_solver_cache::S
end

# We unpack to dispatch per function class
function setup_solver_cache(wrapper::BackwardEulerStageAnnotation, solver::AbstractNonlinearSolver)
    _setup_solver_cache(wrapper, wrapper.f, solver)
end
function _setup_local_solver_cache(
    local_solver::GenericLocalNonlinearSolver,
    material_model::AbstractMaterialModel,
)
    singleQsize = internal_variable_size(material_model, nothing, nothing) # FIXME what to do here?
    @debug "Setting up local nonlinear solver with size(Q)=$(singleQsize) for material $(material_model)" _group=:nlsolve
    return GenericLocalNonlinearSolverCache(
        # Solver parameters
        local_solver,
        # Buffers
        zeros(singleQsize, singleQsize),
        zeros(singleQsize),
        zeros(singleQsize),
        # Globally requested tolerance
        Inf,
        # Local convergence
        SciMLBase.ReturnCode.Default,
    )
end
function _setup_local_solver_cache(
    local_solver::GenericLocalNonlinearSolver,
    model::QuasiStaticModel,
)
    return _setup_local_solver_cache(local_solver, model.material_model)
end
function _setup_local_solver_cache(
    local_solver::GenericLocalNonlinearSolver,
    integrator::NonlinearIntegrator,
)
    return _setup_local_solver_cache(local_solver, integrator.volume_model)
end
function _setup_local_solver_cache(
    local_solver::GenericLocalNonlinearSolver,
    integrator::NonlinearMultiDomainIntegrator2,
)
    return map(
        subintegrator -> _setup_local_solver_cache(local_solver, subintegrator.volume_model),
        values(integrator.subintegrators),
    )
end

function _annotate_with_local_solver_cache(integrator::NonlinearIntegrator, local_solver_cache)
    (; volume_model, facet_model) = integrator
    return NonlinearIntegrator(
        LocalSolverCacheAnnotation(volume_model, local_solver_cache),
        # The inner model is volume only per construction, so facets have no local solve.
        LocalSolverCacheAnnotation(facet_model, nothing),
        integrator.syms,
        integrator.qrc,
        integrator.fqrc,
    )
end

function _annotate_with_local_solver_cache(
    integrator::NonlinearMultiDomainIntegrator2,
    local_solver_cache,
)
    return NonlinearMultiDomainIntegrator2(
        Dict(
            name => _annotate_with_local_solver_cache(subintegrator, local_solver_cache[i]) for
            (i, (name, subintegrator)) in enumerate(integrator.subintegrators)
        ),
    )
end

@inline function _setup_solver_cache(
    wrapper::BackwardEulerStageAnnotation,
    f::QuasiStaticFunction,
    solver::MultiLevelNewtonRaphsonSolver,
)
    (; integrator, dh, lvh) = f
    (; local_solver, newton) = solver

    local_solver_cache = _setup_local_solver_cache(solver.local_solver, integrator)

    # The previous solution and the timestep are no longer threaded in here: `gto1` supplies them per
    # call via `GenericFirstOrderTimeParameters`. Only the local solver cache still has to reach the
    # element.
    op = setup_operator(
        SequentialAssemblyStrategy(SequentialCPUDevice()), # FIXME f.assembly_strategy,
        _annotate_with_local_solver_cache(integrator, local_solver_cache),
        dh,
    )
    T = Float64
    residual = Vector{T}(undef, ndofs(dh))#solution_size(G))
    Δu = Vector{T}(undef, ndofs(dh))#solution_size(G))

    # Connect both solver caches
    inner_prob = LinearSolve.LinearProblem(op.J, residual; u0 = Δu)
    inner_cache = init(
        inner_prob,
        newton.inner_solver;
        alias = LinearAliasSpecifier(alias_A = true, alias_b = true),
    )
    @assert inner_cache.b === residual
    @assert inner_cache.A === op.J

    newton_cache = NewtonRaphsonSolverCache(
        op,
        residual,
        newton,
        inner_cache,
        _build_forcing_cache(newton.forcing, inner_cache, T),
        T[],
        0,
    )

    cache = MultiLevelNewtonRaphsonSolverCache(
        newton_cache, # setup_solver_cache(G, solver.newton),
        local_solver_cache, #setup_solver_cache(L, solver.local_newton), # FIXME pass
    )
    @debug "Setting up Multi-Level Newton-Raphson solver." _group=:nlsolve
    # @debug cache _group=:nlsolve
    return cache
end

# TODO Refactor the setup into generic parts and use multiple dispatch for the specifics.
function setup_solver_cache(
    f::AbstractSemidiscreteFunction,
    solver::BackwardEulerSolver,
    t₀;
    uprev       = nothing,
    u           = nothing,
    alias_uprev = true,
    alias_u     = false,
)
    vtype = Vector{Float64}

    if u === nothing
        _u = vtype(undef, solution_size(f))
        @warn "Cannot initialize u for $(typeof(solver))."
    else
        _u = alias_u ? u : recursivecopy(u)
    end

    if uprev === nothing
        _uprev = vtype(undef, solution_size(f))
        _uprev .= u
    else
        _uprev = alias_uprev ? uprev : recursivecopy(uprev)
    end

    cache = BackwardEulerSolverCache(
        _u,
        _uprev,
        copy(_u),
        BackwardEulerStageCache(
            setup_solver_cache(BackwardEulerStageAnnotation(f, _u, _uprev), solver.inner_solver),
        ),
        solver.monitor,
    )

    return cache
end

# The idea is simple. QuasiStaticModels always have the form
#    0 = G(u,v)
#    0 = L(u,v,dₜu,dₜv)     (or simpler dₜv = L(u,v))
# so we pass the stage information into the interior.
function setup_quasistatic_element_cache(
    wrapper::LocalSolverCacheAnnotation,
    material_model::AbstractMaterialModel,
    qr::QuadratureRule,
    sdh::SubDofHandler,
    cv::CellValues,
)
    internal_cache = setup_internal_cache(wrapper, qr, sdh)
    return quasistatic_element_cache_type(internal_variable_evolution(material_model))(
        material_model,
        setup_coefficient_cache(material_model, qr, sdh),
        internal_cache,
        cv,
    )
end
function setup_element_cache(
    wrapper::AbstractModelAnnotation{<:QuasiStaticModel},
    qr::QuadratureRule,
    sdh::SubDofHandler,
)
    @assert length(sdh.dh.field_names) == 1 "Support for multiple fields not yet implemented."
    field_name = first(sdh.dh.field_names)
    ip         = Ferrite.getfieldinterpolation(sdh, field_name)
    ip_geo     = geometric_subdomain_interpolation(sdh)
    cv         = CellValues(qr, ip, ip_geo)
    return setup_quasistatic_element_cache(wrapper, wrapper.f.material_model, qr, sdh, cv)
end

function perform_backward_euler_step!(
    f::QuasiStaticFunction,
    cache::BackwardEulerSolverCache,
    stage_info::BackwardEulerStageCache,
    t,
    Δt,
)
    update_constraints!(f, cache, t + Δt)
    # `gto1`: hand the nonlinear solver the previous solution and the timestep as *parameters*
    # instead of mutating them into the element caches beforehand. `update_stage!` is what used to do
    # the mutating and is now unnecessary.
    #
    # The leading `nothing` is the inner parameter object, which FerriteOperators forwards to the
    # element via `query_element_parameters(element, cell, ivh, p.p)`. It is the slot reserved for the
    # parameters being *optimized* — not the model's parameters in general, which stay in the model
    # struct. Nothing is optimized here, hence `nothing`. See the `nlsolve!` docstring.
    p = FerriteOperators.GenericFirstOrderTimeParameters(nothing, t + Δt, Δt, cache.uₙ₋₁)
    if !nlsolve!(cache.uₙ, f, stage_info.nlsolver, t + Δt, p)
        return false
    end
    return true
end

# Whether the element needs a local solver cache is the same question as which element cache it gets,
# so it is answered by the same trait rather than by a second classification of the state cache.
function _setup_internal_cache_annotation_unwrap(
    wrapper::LocalSolverCacheAnnotation{<:QuasiStaticModel},
    material_model::AbstractMaterialModel,
    internal_cache,
    ::NoEvolution,
    qr::QuadratureRule,
    sdh::SubDofHandler,
)
    return internal_cache
end
function _setup_internal_cache_annotation_unwrap(
    wrapper::LocalSolverCacheAnnotation{<:QuasiStaticModel},
    material_model::AbstractMaterialModel,
    internal_cache,
    ::Union{FirstOrderEvolution, RateCoupledEvolution},
    qr::QuadratureRule,
    sdh::SubDofHandler,
)
    return GenericFirstOrderRateIndependentCondensationMaterialStateCache(
        # Pass the model
        material_model,
        # And some cache to speed up evaluation of f and associated coefficients
        internal_cache,
        # Local nonlinear solver cache
        wrapper.local_solver_cache,
    )
end
function setup_internal_cache(
    wrapper::LocalSolverCacheAnnotation{<:QuasiStaticModel},
    qr::QuadratureRule,
    sdh::SubDofHandler,
)
    return _setup_internal_cache_annotation_unwrap(
        wrapper,
        wrapper.f.material_model,
        setup_internal_cache(wrapper.f.material_model, qr, sdh),
        internal_variable_evolution(wrapper.f.material_model),
        qr,
        sdh,
    )
end

function setup_boundary_cache(wrapper::LocalSolverCacheAnnotation, fqr, sdh)
    # TODO this technically unlocks differential boundary conditions, if done correctly.
    setup_boundary_cache(wrapper.f, fqr, sdh)
end

OrdinaryDiffEqCore.is_constant_cache(::BackwardEulerSolverCache) = false
