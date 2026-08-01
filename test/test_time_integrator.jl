using Thunderbolt
using DiffEqBase, SciMLBase
using LinearAlgebra
using Logging
using Test

import Thunderbolt: solution_size, ThunderboltTimeIntegrator
import SciMLLogging: Standard

# Unit-level tests for the *standalone* `ThunderboltTimeIntegrator`. Note that `test_integrators.jl`
# covers the outer operator-splitting integrator, which reaches this one only as a child built by
# `_build_child` -- a path that passes `uprev` and a `Bool` verbosity, and therefore misses both of
# the defects below.

# Pure Neumann diffusion without a source term. The constant state is a steady state of this problem,
# so `u ≡ u₀ ≡ 1` is the exact solution for every t and every Δt.
const DIFFUSION_FUN = semidiscretize(
    TransientDiffusionModel(
        ConstantCoefficient(SymmetricTensor{2, 2, Float64}((1.0, 0.0, 1.0))),
        NoStimulationProtocol(),
        :u,
    ),
    FiniteElementDiscretization(Dict(:u => LagrangeCollection{1}())),
    generate_mesh(Quadrilateral, (16, 16), Vec{2}((0.0, 0.0)), Vec{2}((1.0, 1.0))),
)

# `init` aliases `u0` into the solver cache, so every integrator needs its own copy.
steady_state_diffusion_problem(tspan = (0.0, 1.0)) =
    Thunderbolt.ODEProblem(DIFFUSION_FUN, ones(Float64, solution_size(DIFFUSION_FUN)), tspan)

@testset "ThunderboltTimeIntegrator" begin
    @testset "init transports the initial condition into uprev" begin
        prob = steady_state_diffusion_problem()
        u₀   = copy(prob.u0)

        integrator = init(prob, BackwardEulerSolver(), dt = 0.1)
        # The rollback buffer *is* the right hand side of the first step (b = M uₙ₋₁), so a zeroed
        # `uprev` silently annihilates the solution while still reporting `ReturnCode.Success`.
        @test integrator.uprev ≈ u₀

        # Assert on values, not on the retcode -- the defect this guards against returned `Success`.
        SciMLBase.step!(integrator)
        @test integrator.u ≈ u₀ atol = 1.0e-4

        sol = SciMLBase.solve!(integrator)
        @test integrator.u ≈ u₀ atol = 1.0e-4
        @test sol.retcode == SciMLBase.ReturnCode.Success
    end

    @testset "check_error reports failures as retcodes" begin
        # Every failure branch used to hit `if verbose` on a non-`Bool` verbosity object and throw a
        # `TypeError`, so a diverging solve could not report that it diverged.
        with_logger(NullLogger()) do
            integrator = init(steady_state_diffusion_problem(), BackwardEulerSolver(), dt = 0.1)
            integrator.dt = NaN
            @test SciMLBase.check_error(integrator) == SciMLBase.ReturnCode.DtNaN

            # End to end through `solve!`: the retcode has to reach the solution object.
            integrator = init(
                steady_state_diffusion_problem(),
                BackwardEulerSolver(),
                dt = 0.1,
                maxiters = 2,
            )
            sol = SciMLBase.solve!(integrator)
            @test sol.retcode == SciMLBase.ReturnCode.MaxIters

            # `DiffEqBase.init` normalizes its own default, but a user-supplied `SciMLLogging` preset
            # reaches `__init` unwrapped and must be understood just as well.
            integrator = init(
                steady_state_diffusion_problem(),
                BackwardEulerSolver(),
                dt = 0.1,
                verbose = Standard(),
            )
            integrator.dt = NaN
            @test SciMLBase.check_error(integrator) == SciMLBase.ReturnCode.DtNaN
        end
    end
end

# ---------------------------------------------------------------------------
# A scalar semidiscrete problem plus an explicit Euler solver. The FEM problem above is
# the right oracle for the initial-condition and retcode defects, but far too coarse an
# instrument for step-size bookkeeping -- these checks want a problem whose exact solution
# is a closed form and whose step is a single multiply.
# ---------------------------------------------------------------------------
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
    # A one-shot inner failure. The return value of `perform_step!` is not consulted by
    # `step!`, so the failure has to be signalled through `force_stepfail`.
    if cache.fail_at_iter > 0 && integ.iter == cache.fail_at_iter && cache.nfails == 0
        cache.nfails += 1
        integ.force_stepfail = true
        return false
    end
    @. cache.du = -cache.λ * integ.u
    @. integ.u += integ.dt * cache.du
    return true
end

scalar_decay_problem(λ, tspan, u0) = Thunderbolt.ODEProblem(ScalarDecay(λ), [u0], tspan)

@testset "Standalone integrator: stepping" begin
    λ, u0, tf = 0.7, 2.0, 1.0
    exact = u0 * exp(-λ * tf)

    @testset "Convergence order" begin
        # There is no convergence-order test anywhere else in the suite: every existing
        # oracle is a self-consistency check (nested vs flat split, adaptive vs fixed),
        # which is blind to a scheme that is *consistently* wrong. Halving dt must halve
        # the error of a first order method.
        errors = map((0.01, 0.005)) do dt
            integrator = init(scalar_decay_problem(λ, (0.0, tf), u0), ScalarForwardEuler(), dt = dt)
            SciMLBase.solve!(integrator)
            abs(integrator.u[1] - exact)
        end
        @test errors[1] > 0 # guard against a vacuous ratio
        @test errors[1] / errors[2] ≈ 2 rtol = 0.1
    end

    @testset "naccept counts every accepted step" begin
        dt = 0.1
        integrator = init(scalar_decay_problem(λ, (0.0, tf), u0), ScalarForwardEuler(), dt = dt)
        SciMLBase.solve!(integrator)
        @test integrator.t == tf
        @test integrator.stats.nreject == 0
        # With nothing rejected, every attempt was accepted -- this is the invariant F17
        # broke, and it holds independently of how many steps the tstop logic produces.
        @test integrator.stats.naccept == integrator.iter
        @test integrator.stats.naccept == round(Int, tf / dt)
    end

    @testset "A rejected step rolls u back" begin
        # A non-adaptive method cannot recover from a failed step, so the documented
        # outcome is ConvergenceFailure -- but the state must still be the last accepted
        # one, not the half-written attempt.
        dt = 0.1
        integrator = init(
            scalar_decay_problem(λ, (0.0, tf), u0),
            ScalarForwardEuler(),
            dt = dt,
            verbose = false,
        )
        integrator.cache.fail_at_iter = 3
        u_before_failing_step = with_logger(NullLogger()) do
            SciMLBase.step!(integrator) # iter 1
            SciMLBase.step!(integrator) # iter 2
            u = copy(integrator.u)
            SciMLBase.step!(integrator) # iter 3 -- fails
            u
        end
        @test integrator.stats.nreject == 1
        @test integrator.u ≈ u_before_failing_step
        @test SciMLBase.check_error(integrator) == SciMLBase.ReturnCode.ConvergenceFailure
    end
end

@testset "Standalone integrator: tstops" begin
    # The only tstop present anywhere in the suite is `tf` itself; interior tstops,
    # `add_tstop!` and the dt restoration after a clipped step were entirely uncovered.
    dt, tf = 0.3, 1.0

    accepted_times(integrator) = begin
        ts = Float64[]
        while integrator.t < tf
            SciMLBase.step!(integrator)
            push!(ts, integrator.t)
        end
        ts
    end

    @testset "An interior tstop is hit exactly once and dt recovers" begin
        integrator = init(
            scalar_decay_problem(0.7, (0.0, tf), 1.0),
            ScalarForwardEuler(),
            dt = dt,
            tstops = [0.5],
        )
        ts = accepted_times(integrator)
        @test count(≈(0.5), ts) == 1
        @test ts ≈ [0.3, 0.5, 0.8, 1.0]
        @test integrator.t == tf # exactly, not approximately
        # The tstop is landed on bit-exactly, not merely to within a rounding error --
        # which is what stops the leftover gap becoming an extra step.
        @test ts[2] == 0.5
    end

    @testset "No micro-steps when dt does not divide the interval" begin
        # The regression that motivated the tstop-target contract. `t + dt + dt + …`
        # drifts, so the final tstop used to be approached to within ~1e-16 and then
        # closed by a step of that size -- a full solve and two matrix rebuilds for
        # nothing. Every gap must be a real step.
        integrator =
            init(scalar_decay_problem(0.7, (0.0, 100.0), 1.0), ScalarForwardEuler(), dt = 0.1π)
        ts = Float64[]
        while integrator.t < 100.0
            SciMLBase.step!(integrator)
            push!(ts, integrator.t)
        end
        @test integrator.t == 100.0
        @test minimum(diff([0.0; ts])) > 1.0e-12
        @test length(ts) == ceil(Int, 100.0 / (0.1π))
        @test integrator.stats.naccept == length(ts)
    end

    @testset "Duplicated tstops collapse into one step" begin
        integrator = init(
            scalar_decay_problem(0.7, (0.0, tf), 1.0),
            ScalarForwardEuler(),
            dt = dt,
            tstops = [0.5, 0.5, 0.5],
        )
        ts = accepted_times(integrator)
        @test ts ≈ [0.3, 0.5, 0.8, 1.0]
        @test integrator.t == tf
    end

    @testset "A tstop at t0 does not produce a zero-length step" begin
        integrator = init(
            scalar_decay_problem(0.7, (0.0, tf), 1.0),
            ScalarForwardEuler(),
            dt = dt,
            tstops = [0.0],
        )
        ts = accepted_times(integrator)
        @test minimum(diff([0.0; ts])) > 1.0e-12
        @test integrator.t == tf
    end

    @testset "A tstop in the past is rejected" begin
        integrator = init(scalar_decay_problem(0.7, (0.0, tf), 1.0), ScalarForwardEuler(), dt = dt)
        SciMLBase.step!(integrator)
        @test_throws ErrorException SciMLBase.add_tstop!(integrator, 0.1)
    end

    @testset "add_tstop! after the first step" begin
        integrator = init(scalar_decay_problem(0.7, (0.0, tf), 1.0), ScalarForwardEuler(), dt = dt)
        SciMLBase.step!(integrator)
        SciMLBase.add_tstop!(integrator, 0.45)
        ts = accepted_times(integrator)
        @test count(≈(0.45), ts) == 1
        @test integrator.t == tf
    end

    @testset "A tstop coinciding with tf does not add a step" begin
        integrator = init(
            scalar_decay_problem(0.7, (0.0, tf), 1.0),
            ScalarForwardEuler(),
            dt = dt,
            tstops = [tf],
        )
        ts = accepted_times(integrator)
        @test ts ≈ [0.3, 0.6, 0.9, 1.0]
        @test integrator.t == tf
    end
end

struct TimeProbeCell end
Thunderbolt.num_states(::TimeProbeCell) = 1
Thunderbolt.transmembranepotential_index(::TimeProbeCell) = 1

const PROBED_TIMES = Float64[]
function Thunderbolt.cell_rhs!(du, u, x, t, ::TimeProbeCell)
    push!(PROBED_TIMES, t)
    du[1] = 1.0 # above `reaction_threshold`, so the substepping branch is taken
    return nothing
end

@testset "Substepper evaluates each substep at its own time" begin
    substeps, t, Δt = 4, 2.0, 0.4
    du, u = zeros(1), zeros(1)
    cache = Thunderbolt.AdaptiveForwardEulerSubstepperCache(
        du,
        u,
        u,
        reshape(du, (1, 1)),
        reshape(u, (1, 1)),
        substeps,
        0.1,
        1,
        nothing,
    )

    empty!(PROBED_TIMES)
    Thunderbolt._pointwise_step_inner_kernel!(TimeProbeCell(), 1, t, Δt, cache)

    Δtₛ = Δt / substeps
    @test length(PROBED_TIMES) == substeps
    @test PROBED_TIMES ≈ [t + (s - 1) * Δtₛ for s = 1:substeps]
end
