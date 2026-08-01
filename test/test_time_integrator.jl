using Thunderbolt
using DiffEqBase, SciMLBase
using LinearAlgebra
using Logging
using Test

import Thunderbolt: solution_size
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
