"""
    LoadDrivenSolver{IS, T, PFUN}

Solve the nonlinear problem `F(u,t)=0` with given time increments `Δt`on some interval `[t_begin, t_end]`
where `t` is some pseudo-time parameter.
"""
mutable struct LoadDrivenSolver{IS} <: AbstractSolver
    inner_solver::IS
end

mutable struct LoadDrivenSolverCache{ISC, T, VT <: AbstractVector{T}} <: AbstractTimeSolverCache
    inner_solver_cache::ISC
    uₙ::VT
    uₙ₋₁::VT
    tmp::VT
end

function setup_solver_cache(f::AbstractSemidiscreteFunction, solver::LoadDrivenSolver, t₀)
    inner_solver_cache = setup_solver_cache(f, solver.inner_solver)
    T = Float64 # TODO query
    vtype = Vector{T}
    LoadDrivenSolverCache(
        inner_solver_cache,
        vtype(undef, solution_size(f)),
        vtype(undef, solution_size(f)),
        vtype(undef, solution_size(f)),
    )
end

function setup_solver_cache(f::AbstractSemidiscreteBlockedFunction, solver::LoadDrivenSolver, t₀)
    inner_solver_cache = setup_solver_cache(f, solver.inner_solver)
    T = Float64 # TODO query
    vtype = Vector{T}
    LoadDrivenSolverCache(
        inner_solver_cache,
        mortar([
            vtype(undef, solution_size(fi)) for fi ∈ blocks(f)
        ]),
        mortar([
            vtype(undef, solution_size(fi)) for fi ∈ blocks(f)
        ]),
        mortar([
            vtype(undef, solution_size(fi)) for fi ∈ blocks(f)
        ]),
    )
end

function perform_step!(f::AbstractSemidiscreteFunction, solver_cache::LoadDrivenSolverCache, t, Δt)
    solver_cache.uₙ₋₁ .= solver_cache.uₙ
    @info "Load step from $t to $(t+Δt)."
    update_constraints!(f, solver_cache, t + Δt)
    if !nlsolve!(solver_cache.uₙ, f, solver_cache.inner_solver_cache, t + Δt) # TODO remove ,,t'' here. But how?
        @warn "Inner solver failed on from $t to $(t+Δt)]"
        return false
    end

    return true
end

@doc raw"""
    Deuflhard2004DiscreteContinuationController(Θbar, p)

Θbar ($\overbar{\Theta}$) is the target convergence rate.

Θk ($\Theta_0$) is the estimated convergence rate for the nonlinear solve iteration k.

Predictor time step length: $\Delta t^0_n = \sqrt[p]{\frac{g(\overbar{\Theta})}{2\Theta_0}} \Delta t^{\textrm{last}}_{n-1}$ [Deu:2004:nmn; p. 248](@cite)

Predictor time step length: $\Delta t^i_n = \sqrt[p]{\frac{\overbar{\Theta}}{\Theta}_k} \Delta t^{i-1}_{n-1}$ [Deu:2004:nmn; Eq. 5.24, p. 248](@cite)

Here $g(x) = \sqrt{1+4\Theta}-1$ and $\Theta_0 \geq \Theta_{\textrm{min}}$

The retry criterion for the time step is $\Theta}_k > \frac{1}{2}$.
"""
Base.@kwdef struct Deuflhard2004DiscreteContinuationController
    Θmin::Float64
    p::Int64
    Θreject::Float64 = 0.95
    Θbar::Float64 = 0.5
    γ::Float64    = 0.95
    qmin::Float64 = 1/5
    qmax::Float64 = 5.0
end

function should_accept_step(integrator::ThunderboltTimeIntegrator, cache::LoadDrivenSolverCache, controller::Deuflhard2004DiscreteContinuationController)
    (; Θks) = cache.inner_solver_cache
    (; Θreject) = controller
    result = all(Θks .≤ Θreject)
    return result
end
function reject_step!(integrator::ThunderboltTimeIntegrator, cache::LoadDrivenSolverCache, controller::Deuflhard2004DiscreteContinuationController)
    # Reset solution
    integrator.u .= integrator.uprev

    @inline g(x) = √(1+4x) - 1

    # Shorten dt according to (Eq. 5.24)
    (; Θks) = cache.inner_solver_cache
    (; Θbar, Θreject, γ, Θmin, qmin, qmax, p) = controller
    for Θk in Θks
        if Θk > Θreject
            q = clamp(γ * (g(Θbar)/g(Θk))^(1/p), qmin, qmax)
            integrator.dt = q * integrator.dt
            return
        end
    end
end

function adapt_dt!(integrator::ThunderboltTimeIntegrator, cache::LoadDrivenSolverCache, controller::Deuflhard2004DiscreteContinuationController)
    @inline g(x) = √(1+4x) - 1

    # Adapt dt with a priori estimate (Eq. 5.24)
    (; Θks) = cache.inner_solver_cache
    (; Θbar, γ, Θmin, qmin, qmax, p) = controller

    Θ₀ = length(Θks) > 0 ? max(first(Θks), Θmin) : Θmin
    q = clamp(γ * (g(Θbar)/(2Θ₀))^(1/p), qmin, qmax)
    integrator.dt = q * integrator.dt
end

Base.@kwdef struct Deuflhard2004_B_DiscreteContinuationControllerVariant
    Θmin::Float64
    p::Int64
    Θreject::Float64 = 0.95
    Θbar::Float64 = 0.5
    γ::Float64    = 0.95
    qmin::Float64 = 1/5
    qmax::Float64 = 5.0
end

function should_accept_step(integrator::ThunderboltTimeIntegrator, cache::LoadDrivenSolverCache, controller::Deuflhard2004_B_DiscreteContinuationControllerVariant)
    (; Θks) = cache.inner_solver_cache
    (; Θreject) = controller
    result = all(Θks .≤ Θreject)
    return result
end
function reject_step!(integrator::ThunderboltTimeIntegrator, cache::LoadDrivenSolverCache, controller::Deuflhard2004_B_DiscreteContinuationControllerVariant)
    # Reset solution
    integrator.u .= integrator.uprev

    @inline g(x) = √(1+4x) - 1

    # Shorten dt according to (Eq. 5.24)
    (; Θks) = cache.inner_solver_cache
    (; Θbar, Θreject, γ, Θmin, qmin, qmax, p) = controller
    for Θk in Θks
        if Θk > Θreject
            q = clamp(γ * (g(Θbar)/g(Θk))^(1/p), qmin, qmax)
            integrator.dt = q * integrator.dt
            return
        end
    end
end

function adapt_dt!(integrator::ThunderboltTimeIntegrator, cache::LoadDrivenSolverCache, controller::Deuflhard2004_B_DiscreteContinuationControllerVariant)
    @inline g(x) = √(1+4x) - 1

    # Adapt dt with a priori estimate (Eq. 5.24)
    (; Θks) = cache.inner_solver_cache
    (; Θbar, γ, Θmin, qmin, qmax, p) = controller

    Θ₀ = length(Θks) > 0 ? max(first(Θks), Θmin) : Θmin
    q = clamp(γ * (g(Θbar)/(g(Θ₀)))^(1/p), qmin, qmax)
    integrator.dt = q * integrator.dt
end

@doc raw"""
    ExperimentalDiscreteContinuationController(Θbar, p)
"""
Base.@kwdef struct ExperimentalDiscreteContinuationController
    Θmin::Float64
    p::Int64
    Θreject::Float64 = 0.9
    Θbar::Float64 = 0.75
    γ::Float64    = 0.95
    qmin::Float64 = 1/5
    qmax::Float64 = 5.0
end

function should_accept_step(integrator::ThunderboltTimeIntegrator, cache::LoadDrivenSolverCache, controller::ExperimentalDiscreteContinuationController)
    (; Θks) = cache.inner_solver_cache
    (; Θreject) = controller
    result = all(Θks .≤ Θreject)
    return result
end
function reject_step!(integrator::ThunderboltTimeIntegrator, cache::LoadDrivenSolverCache, controller::ExperimentalDiscreteContinuationController)
    # Reset solution
    integrator.u .= integrator.uprev

    @inline g(x) = √(1+4x) - 1

    # Shorten dt according to (Eq. 5.24)
    (; Θks) = cache.inner_solver_cache
    (; Θbar, γ, Θmin, qmin, qmax, p) = controller
    Θk = maximum(Θks)
    q = clamp(γ *(g(Θbar)/g(Θk))^(1/p), qmin, qmax)
    integrator.dt = q * integrator.dt
end

function adapt_dt!(integrator::ThunderboltTimeIntegrator, cache::LoadDrivenSolverCache, controller::ExperimentalDiscreteContinuationController)
    @inline g(x) = √(1+4x) - 1

    # Adapt dt with a priori estimate (Eq. 5.24)
    (; Θks) = cache.inner_solver_cache
    (; Θbar, γ, Θmin, qmin, qmax, p) = controller
    Θ₀ = length(Θks) > 0 ? max(mean(Θks), Θmin) : Θmin
    q = clamp(γ * (g(Θbar)/(2Θ₀))^(1/p), qmin, qmax)
    integrator.dt = q * integrator.dt
end



# default_controller(::LoadDrivenSolver, cache) = ExperimentalDiscreteContinuationController(; Θmin=1/8, p=1)
default_controller(::LoadDrivenSolver, cache) = Deuflhard2004_B_DiscreteContinuationControllerVariant(; Θmin=1/8, p=1)
DiffEqBase.isadaptive(::LoadDrivenSolver) = true
