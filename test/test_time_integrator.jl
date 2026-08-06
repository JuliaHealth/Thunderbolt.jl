using Thunderbolt
using DiffEqBase, SciMLBase
using LinearAlgebra
using Logging
using Test

import Thunderbolt: solution_size, ThunderboltTimeIntegrator
import SciMLLogging: Standard
import OrdinaryDiffEqCore

# Tests for the standalone `ThunderboltTimeIntegrator`.

# Pure Neumann diffusion without a source: the constant state is a steady state, so
# `u ≡ u₀ ≡ 1` is the exact solution for every t and every Δt.
const DIFFUSION_FUN = semidiscretize(
    TransientDiffusionModel(
        ConstantCoefficient(SymmetricTensor{2, 2, Float64}((1.0, 0.0, 1.0))),
        NoStimulationProtocol(),
        :u,
    ),
    FiniteElementDiscretization(Dict(:u => LagrangeCollection{1}())),
    generate_mesh(Quadrilateral, (4, 4), Vec{2}((0.0, 0.0)), Vec{2}((1.0, 1.0))),
)

# `init` aliases `u0` into the solver cache, so every integrator needs its own copy.
transient_diffusion_problem() =
    Thunderbolt.ODEProblem(DIFFUSION_FUN, ones(Float64, solution_size(DIFFUSION_FUN)), (0.0, 1.0))

@testset "Backward Euler on a steady state" begin
    prob = transient_diffusion_problem()
    u₀ = copy(prob.u0)
    integrator = init(prob, BackwardEulerSolver(), dt = 0.1)

    # A rollback buffer that does not start at u₀ annihilates the solution here while
    # still reporting Success, so assert on values rather than on the retcode.
    SciMLBase.step!(integrator)
    @test integrator.u ≈ u₀ atol = 1.0e-4
    sol = SciMLBase.solve!(integrator)
    @test integrator.u ≈ u₀ atol = 1.0e-4
    @test sol.retcode == SciMLBase.ReturnCode.Success
end

@testset "Failures surface as retcodes, verbose = $verbose" for verbose in (true, false, Standard())
    with_logger(NullLogger()) do
        integrator =
            init(transient_diffusion_problem(), BackwardEulerSolver(), dt = 0.1, verbose = verbose)
        integrator.dt = NaN
        @test SciMLBase.check_error(integrator) == SciMLBase.ReturnCode.DtNaN

        sol = SciMLBase.solve!(
            init(
                transient_diffusion_problem(),
                BackwardEulerSolver(),
                dt = 0.1,
                verbose = verbose,
                maxiters = 2,
            ),
        )
        @test sol.retcode == SciMLBase.ReturnCode.MaxIters
    end
end

# A scalar problem with a closed form solution, for the step size bookkeeping. The FEM
# problem above is the right oracle for the solver but far too coarse an instrument here.
struct ScalarDecay <: Thunderbolt.AbstractSemidiscreteFunction
    λ::Float64
end
Thunderbolt.solution_size(::ScalarDecay) = 1

struct ScalarForwardEuler <: Thunderbolt.AbstractSolver end
DiffEqBase.isadaptive(::ScalarForwardEuler) = false

mutable struct ScalarForwardEulerCache{uType, uprevType} <: Thunderbolt.AbstractTimeSolverCache
    du::Vector{Float64}
    uₙ::uType
    uₙ₋₁::uprevType
    λ::Float64
    fail_at_iter::Int # 0 = never fail
    nfails::Int
end

function Thunderbolt.setup_solver_cache(
    f::ScalarDecay,
    ::ScalarForwardEuler,
    t₀;
    u = nothing,
    uprev = nothing,
)
    n = Thunderbolt.solution_size(f)
    uₙ = u === nothing ? zeros(n) : u
    uₙ₋₁ = uprev === nothing ? copy(uₙ) : uprev
    return ScalarForwardEulerCache(zeros(n), uₙ, uₙ₋₁, f.λ, 0, 0)
end

function Thunderbolt.OrdinaryDiffEqCore.perform_step!(
    integ::ThunderboltTimeIntegrator,
    cache::ScalarForwardEulerCache,
)
    # There is no public way to make one specific step fail transiently. The return value
    # of `perform_step!` is not consulted, so the failure goes through `force_stepfail`.
    if cache.fail_at_iter > 0 && integ.iter == cache.fail_at_iter && cache.nfails == 0
        cache.nfails += 1
        integ.u .= 1.0e6 # a half written attempt, which the rollback must discard
        integ.force_stepfail = true
        return false
    end
    @. cache.du = -cache.λ * integ.u
    @. integ.u += integ.dt * cache.du
    return true
end

scalar_decay_problem(λ = 0.7, tspan = (0.0, 1.0), u0 = 2.0) =
    Thunderbolt.ODEProblem(ScalarDecay(λ), [u0], tspan)

function accepted_times(integrator, tf)
    ts = Float64[]
    while integrator.t < tf
        SciMLBase.step!(integrator)
        push!(ts, integrator.t)
    end
    return ts
end

@testset "Convergence order" begin
    # Halving dt must halve the error of a first order method.
    λ, tf, u0 = 0.7, 1.0, 2.0
    exact = u0 * exp(-λ * tf)
    errors = map((0.01, 0.005)) do dt
        integrator = init(scalar_decay_problem(λ, (0.0, tf), u0), ScalarForwardEuler(), dt = dt)
        SciMLBase.solve!(integrator)
        abs(integrator.u[1] - exact)
    end
    @test errors[1] > 0
    @test errors[1] / errors[2] ≈ 2 rtol = 0.1

    integrator = init(scalar_decay_problem(λ, (0.0, tf), u0), ScalarForwardEuler(), dt = 0.1)
    SciMLBase.solve!(integrator)
    @test integrator.t == tf
    @test integrator.stats.naccept == 10 # no spurious extra step to close the tf tstop
end

@testset "A rejected step discards the attempt" begin
    # A non-adaptive method cannot recover, so ConvergenceFailure is the documented
    # outcome -- but the state must be the last accepted one, not the failed attempt.
    integrator = init(scalar_decay_problem(), ScalarForwardEuler(), dt = 0.1, verbose = false)
    integrator.cache.fail_at_iter = 3
    u_before = with_logger(NullLogger()) do
        SciMLBase.step!(integrator)
        SciMLBase.step!(integrator)
        u = copy(integrator.u)
        SciMLBase.step!(integrator)
        u
    end
    @test integrator.u ≈ u_before
    @test integrator.stats.nreject == 1
    @test SciMLBase.check_error(integrator) == SciMLBase.ReturnCode.ConvergenceFailure
end

@testset "tstops" begin
    tf = 1.0

    @testset "t0, duplicates, interior hit, dt recovery, tf" begin
        integrator = init(
            scalar_decay_problem(0.7, (0.0, tf)),
            ScalarForwardEuler(),
            dt = 0.3,
            tstops = [0.0, 0.5, 0.5, tf],
        )
        ts = accepted_times(integrator, tf)
        @test ts ≈ [0.3, 0.5, 0.8, 1.0] # 0.8 pins that dt recovered after the clipped step
        @test ts[2] == 0.5              # landed exactly, not within a rounding error
        @test integrator.t == tf
        @test minimum(diff([0.0; ts])) > 1.0e-12
    end

    @testset "No micro-steps when dt does not divide the interval" begin
        integrator = init(scalar_decay_problem(0.7, (0.0, 100.0)), ScalarForwardEuler(), dt = 0.1π)
        ts = accepted_times(integrator, 100.0)
        @test integrator.t == 100.0
        @test minimum(diff([0.0; ts])) > 1.0e-12
        @test length(ts) == ceil(Int, 100.0 / (0.1π))
    end

    @testset "add_tstop!" begin
        integrator = init(scalar_decay_problem(0.7, (0.0, tf)), ScalarForwardEuler(), dt = 0.3)
        SciMLBase.step!(integrator)
        SciMLBase.add_tstop!(integrator, 0.45)
        @test count(≈(0.45), accepted_times(integrator, tf)) == 1
        @test integrator.t == tf
        @test_throws ErrorException SciMLBase.add_tstop!(integrator, 0.1)
    end

    @testset "advance_to_tstop" begin
        integrator = init(
            scalar_decay_problem(0.7, (0.0, tf)),
            ScalarForwardEuler(),
            dt = 0.1,
            tstops = [0.5],
            advance_to_tstop = true,
        )
        SciMLBase.step!(integrator)
        @test integrator.t == 0.5 # one `step!`, five steps of work
        SciMLBase.step!(integrator)
        @test integrator.t == tf
    end

    @testset "stop_at_next_tstop ends the iteration at the tstop" begin
        integrator = init(
            scalar_decay_problem(0.7, (0.0, tf)),
            ScalarForwardEuler(),
            dt = 0.1,
            tstops = [0.5],
            stop_at_next_tstop = true,
        )
        for _ in integrator
        end
        @test integrator.t == 0.5
    end
end

@testset "reinit! yields a usable integrator, also after a failed run" begin
    integrator = init(scalar_decay_problem(), ScalarForwardEuler(), dt = 0.1, verbose = false)
    SciMLBase.solve!(integrator)
    u_ref = copy(integrator.u)

    DiffEqBase.reinit!(integrator, [2.0])
    integrator.cache.fail_at_iter = 3
    integrator.cache.nfails = 0
    with_logger(NullLogger()) do
        SciMLBase.solve!(integrator)
    end
    @test integrator.sol.retcode == SciMLBase.ReturnCode.ConvergenceFailure

    # Cause removed, so the reinit must recover fully -- which it cannot if any failure
    # flag or save counter from the aborted run survives.
    integrator.cache.fail_at_iter = 0
    DiffEqBase.reinit!(integrator, [2.0])
    SciMLBase.solve!(integrator)
    @test integrator.sol.retcode == SciMLBase.ReturnCode.Success
    @test integrator.t == 1.0
    @test integrator.u ≈ u_ref
end

@testset "Unsupported options are refused at construction" begin
    for kw in (
        (saveat = 0.0:0.2:1.0,),
        (saveat = 0.2,),
        (save_everystep = true,),
        (dense = true,),
        (save_idxs = [1],),
        (callback = SciMLBase.DiscreteCallback((u, t, i) -> false, i -> nothing),),
    )
        @test_throws ErrorException init(
            scalar_decay_problem(),
            ScalarForwardEuler();
            dt = 0.1,
            kw...,
        )
    end
    @test_throws ErrorException init(
        scalar_decay_problem(0.7, (1.0, 0.0)),
        ScalarForwardEuler(),
        dt = 0.1,
    )
    @test init(scalar_decay_problem(), ScalarForwardEuler(), dt = 0.1) isa ThunderboltTimeIntegrator
    # An empty CallbackSet is what `init` builds internally and must stay accepted.
    @test init(
        scalar_decay_problem(),
        ScalarForwardEuler(),
        dt = 0.1,
        callback = SciMLBase.CallbackSet(),
    ) isa ThunderboltTimeIntegrator
end

# A cell model whose right hand side depends on t, so a substepper that froze the clock
# across the outer Δt would integrate the wrong function.
struct TimeProbeCell <: Thunderbolt.AbstractIonicModel end
Thunderbolt.num_states(::Type{TimeProbeCell}) = 1
Thunderbolt.state_symbols(::Type{TimeProbeCell}) = (:φₘ,)
Thunderbolt.cell_rhs!(du, u, x, t, ::TimeProbeCell) = (du[1] = 1.0 + sin(t); nothing)

@testset "Substepper evaluates each substep at its own time" begin
    t₀, Δt, substeps = 2.0, 0.4, 4
    prob = Thunderbolt.PointwiseODEProblem(
        Thunderbolt.PointwiseODEFunction(TimeProbeCell(), nothing, 1:1, :s),
        zeros(1),
        (t₀, t₀ + Δt),
    )
    integrator = init(prob, AdaptiveForwardEulerSubstepper(substeps = substeps), dt = Δt)
    SciMLBase.step!(integrator)

    Δtₛ = Δt / substeps
    @test integrator.u[1] ≈ sum(Δtₛ * (1.0 + sin(t₀ + (s - 1) * Δtₛ)) for s = 1:substeps)
end

# The Deuflhard controllers are pure functions of the Newton convergence history, so a stub
# cache with prescribed `Θks` pins their step size laws exactly. End to end that history is
# emergent and cannot be prescribed, which is why this stays a unit test.
struct StubNewtonCache
    Θks::Vector{Float64}
    parameters::NamedTuple{(:enforce_monotonic_convergence,), Tuple{Bool}}
end

# The controllers read only the Newton history, so the stage the cache carries is irrelevant here.
stub_homotopy_cache(Θks; enforce = true) = Thunderbolt.HomotopyPathSolverCache(
    nothing,
    StubNewtonCache(Θks, (; enforce_monotonic_convergence = enforce)),
    [0.0],
    [0.0],
    [0.0],
)

g_deuflhard(x) = √(1 + 4x) - 1

@testset "Deuflhard continuation controllers" begin
    dummy(dt) = init(scalar_decay_problem(0.7, (0.0, 10.0)), ScalarForwardEuler(), dt = dt)
    controllers = (
        Thunderbolt.Deuflhard2004DiscreteContinuationController(Θmin = 1 / 8, p = 1),
        Thunderbolt.Deuflhard2004_B_DiscreteContinuationControllerVariant(Θmin = 1 / 8, p = 1),
        Thunderbolt.ExperimentalDiscreteContinuationController(Θmin = 1 / 8, p = 1),
    )

    @testset "Acceptance: $(nameof(typeof(controller)))" for controller in controllers
        Θreject = controller.Θreject
        @test Thunderbolt.should_accept_step(
            dummy(0.1),
            stub_homotopy_cache([0.1, 0.2]),
            controller,
        )
        @test !Thunderbolt.should_accept_step(
            dummy(0.1),
            stub_homotopy_cache([0.1, Θreject + 0.01]),
            controller,
        )
        # Without monotonic convergence enforced only finiteness matters.
        @test Thunderbolt.should_accept_step(
            dummy(0.1),
            stub_homotopy_cache([10.0]; enforce = false),
            controller,
        )
        @test !Thunderbolt.should_accept_step(
            dummy(0.1),
            stub_homotopy_cache([0.1, NaN]; enforce = false),
            controller,
        )
    end

    @testset "Rejection shrinks dt and rolls u back" begin
        controller = controllers[1]
        (; Θbar, γ, qmin, qmax, p) = controller
        Θk = 2.0 # above Θreject
        integrator = dummy(0.4)
        integrator.u .= 5.0
        # The two halves of a rejection have separate owners: `rollback_state!` restores the state,
        # which is the scheme's business, and `reject_step!` proposes the next step size, which is the
        # controller's.
        Thunderbolt.rollback_state!(integrator, integrator.cache)
        Thunderbolt.reject_step!(integrator, stub_homotopy_cache([0.1, Θk]), controller)
        @test integrator.dt ≈
              clamp(γ * (g_deuflhard(Θbar) / g_deuflhard(Θk))^(1 / p), qmin, qmax) * 0.4
        @test integrator.dt < 0.4
        @test integrator.u ≈ integrator.uprev

        # With every Θk below the threshold the loop falls through and dt is untouched --
        # the shape that makes a NaN Θk livelock, since `NaN > Θreject` is false.
        integrator = dummy(0.4)
        Thunderbolt.reject_step!(integrator, stub_homotopy_cache([0.1, 0.2]), controller)
        @test integrator.dt == 0.4
    end

    @testset "Step size laws" begin
        Θks = [0.3, 0.4]
        (; Θbar, γ, Θmin, qmin, qmax, p) = controllers[1]
        Θ₀ = max(first(Θks), Θmin)

        # The three variants differ only in the denominator: 2Θ₀, g(Θ₀), and the mean.
        integrator = dummy(0.4)
        Thunderbolt.adapt_dt!(integrator, stub_homotopy_cache(Θks), controllers[1])
        @test integrator.dt ≈ clamp(γ * (g_deuflhard(Θbar) / (2Θ₀))^(1 / p), qmin, qmax) * 0.4

        integrator = dummy(0.4)
        Thunderbolt.adapt_dt!(integrator, stub_homotopy_cache(Θks), controllers[2])
        @test integrator.dt ≈
              clamp(γ * (g_deuflhard(Θbar) / g_deuflhard(Θ₀))^(1 / p), qmin, qmax) * 0.4

        integrator = dummy(0.4)
        Thunderbolt.adapt_dt!(integrator, stub_homotopy_cache(Θks), controllers[3])
        let (; Θbar, γ, Θmin, qmin, qmax, p) = controllers[3]
            Θ₀exp = max(sum(Θks) / length(Θks), Θmin)
            @test integrator.dt ≈
                  clamp(γ * (g_deuflhard(Θbar) / (2Θ₀exp))^(1 / p), qmin, qmax) * 0.4
        end

        # An empty history falls back to Θmin.
        integrator = dummy(0.4)
        Thunderbolt.adapt_dt!(integrator, stub_homotopy_cache(Float64[]), controllers[1])
        @test integrator.dt ≈ clamp(γ * (g_deuflhard(Θbar) / (2Θmin))^(1 / p), qmin, qmax) * 0.4
    end

    @testset "A multi-level cache reaches the global Newton cache" begin
        # The forwarding that lets these controllers run under MultiLevelNewtonRaphsonSolver.
        mlcache = Thunderbolt.MultiLevelNewtonRaphsonSolverCache(
            StubNewtonCache([0.1, 0.2], (; enforce_monotonic_convergence = true)),
            nothing,
        )
        cache = Thunderbolt.HomotopyPathSolverCache(nothing, mlcache, [0.0], [0.0], [0.0])
        @test Thunderbolt.should_accept_step(dummy(0.1), cache, controllers[1])
    end
end

@testset "PID step size controller" begin
    alg = NewmarkSolver()
    k = Thunderbolt.adaptive_order(alg) + 1
    controller = Thunderbolt.PIDController(3 // 5, -1 // 5, 1 // 10)
    fresh() = OrdinaryDiffEqCore.setup_controller_cache(alg, nothing, controller, Float64, nothing)

    @testset "The history holds three distinct steps" begin
        # `err` must be (current, previous, previous-previous). Shifting it in both the factor
        # computation and the accept hook would leave `err[3]` a duplicate of `err[2]`, which the
        # default `β₃ = 0` hides by raising it to the zeroth power.
        cache = fresh()
        estimates = (0.5, 0.8, 0.3)
        for e in estimates
            Thunderbolt.set_error_estimate!(cache, e)
            Thunderbolt._pid_dt_factor!(cache, alg)
            Thunderbolt._pid_accept!(cache)
        end
        Thunderbolt.set_error_estimate!(cache, 0.9)
        Thunderbolt._pid_dt_factor!(cache, alg)
        @test cache.err == (1 / 0.9, 1 / 0.3, 1 / 0.8)
    end

    @testset "A rejected attempt does not consume history" begin
        cache = fresh()
        Thunderbolt.set_error_estimate!(cache, 0.5)
        Thunderbolt._pid_dt_factor!(cache, alg)
        Thunderbolt._pid_accept!(cache)
        accepted = cache.err

        Thunderbolt.set_error_estimate!(cache, 4.0)   # over tolerance
        Thunderbolt._pid_dt_factor!(cache, alg)       # rejected: no `_pid_accept!`
        Thunderbolt.set_error_estimate!(cache, 0.5)
        Thunderbolt._pid_dt_factor!(cache, alg)
        @test cache.err[2] == accepted[2]
        @test cache.err[3] == accepted[3]
    end

    @testset "The factor is the Söderlind law" begin
        cache = fresh()
        β = controller.β
        for e in (0.5, 0.8, 0.3, 0.9)
            Thunderbolt.set_error_estimate!(cache, e)
            Thunderbolt._pid_dt_factor!(cache, alg)
            Thunderbolt._pid_accept!(cache)
        end
        Thunderbolt.set_error_estimate!(cache, 0.4)
        factor = Thunderbolt._pid_dt_factor!(cache, alg)
        ε = cache.err
        @test factor ≈ controller.limiter(ε[1]^(β[1] / k) * ε[2]^(β[2] / k) * ε[3]^(β[3] / k))
    end

    @testset "The limiter saturates and a vanishing estimate is finite" begin
        @test Thunderbolt.default_dt_factor_limiter(0.0)≈1 - π / 4 atol=0.3
        @test Thunderbolt.default_dt_factor_limiter(1.0e12) < 1 + π / 2
        cache = fresh()
        Thunderbolt.set_error_estimate!(cache, 0.0)
        @test isfinite(Thunderbolt._pid_dt_factor!(cache, alg))
    end

    @testset "Acceptance is on the factor, not the estimate" begin
        # `EEst` slightly above one gives a factor near one, which `accept_safety = 0.81` tolerates.
        # A rule on the estimate itself would discard that step.
        cache = fresh()
        Thunderbolt.set_error_estimate!(cache, 1.05)
        @test Thunderbolt._pid_dt_factor!(cache, alg) ≥ controller.accept_safety
    end
end
