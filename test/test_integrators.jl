import Thunderbolt: OS, ThunderboltTimeIntegrator
# using BenchmarkTools
using UnPack

@testset "Operator Splitting API" begin

    ODEFunction = Thunderbolt.DiffEqBase.ODEFunction

    # For testing purposes
    struct DummyForwardEuler <: Thunderbolt.AbstractSolver
    end
    
    Thunderbolt.DiffEqBase.isadaptive(::DummyForwardEuler) = false

    mutable struct DummyForwardEulerCache{duType, uType, duMatType} <: Thunderbolt.AbstractTimeSolverCache
        du::duType
        dumat::duMatType
        uₙ::uType
        uₙ₋₁::uType
    end

    # Dispatch for leaf construction
    Thunderbolt.num_states(::ODEFunction) = 2                   # FIXME
    Thunderbolt.transmembranepotential_index(::ODEFunction) = 1 # FIXME

    function Thunderbolt.setup_solver_cache(f::Any, solver::DummyForwardEuler, t₀; u = nothing, uprev = nothing)
        n = u === nothing ? Thunderbolt.num_states(f) : length(u)
        du      = zeros(n)
        uₙ      = u === nothing ? zeros(n) : u
        uₙ₋₁    = uprev === nothing ? zeros(n) : uprev
        dumat = reshape(du, (:,1))

        return DummyForwardEulerCache(du, dumat, uₙ, uₙ₋₁)
    end

    # Dispatch innermost solve
    function Thunderbolt.OrdinaryDiffEqCore.perform_step!(integ::ThunderboltTimeIntegrator, cache::DummyForwardEulerCache)
        @unpack f, dt, u, p, t = integ
        @unpack du = cache

        f isa Thunderbolt.PointwiseODEFunction ? f.ode(du, u, p, t) : f(du, u, p, t)
        @. u += dt * du
        cache.dumat[:,1] .= du

        return true
    end

    # Operator splitting

    # Reference
    function ode_true(du, u, p, t)
        du .= -0.1u
        du[1] += 0.01u[3]
        du[3] += 0.01u[1]
    end

    # Setup individual functions
    # Diagonal components
    function ode1(du, u, p, t)
        @. du = -0.1u
    end
    # Offdiagonal components
    function ode2(du, u, p, t)
        du[1] = 0.01u[2]
        du[2] = 0.01u[1]
    end

    f1 = ODEFunction(ode1)
    f2 = ODEFunction(ode2)

    # Here we describe index sets f1dofs and f2dofs that map the
    # local indices in f1 and f2 into the global problem. Just put
    # ode_true and ode1/ode2 side by side to see how they connect.
    f1dofs = [1,2,3]
    f2dofs = [1,3]
    fpw = PointwiseODEFunction(
        1,
        f2,
        [0.0]
    )

    fsplit1 = GenericSplitFunction((f1,fpw), (f1dofs, f2dofs))

    # Now the usual setup just with our new problem type.
    # u0 = rand(3)
    u0 = [0.7611944793397108
        0.9059606424982555
        0.5755174199139956]
    tspan = (0.0,100.0)
    prob = OperatorSplittingProblem(fsplit1, u0, tspan)

    # Now some recursive splitting
    function ode3(du, u, p, t)
        du[1] = 0.005u[2]
        du[2] = 0.005u[1]
    end
    f3 = ODEFunction(ode3)
    # The time stepper carries the individual solver information.

    # Note that we define the dof indices w.r.t the parent function.
    # Hence the indices for `fsplit2_inner` are.
    f1dofs = [1,2,3]
    f2dofs = [1,3]
    f3dofs = [1,3]
    fsplit2_inner = GenericSplitFunction((fpw,f3), (f3dofs, f3dofs))
    fsplit2_outer = GenericSplitFunction((f1,fsplit2_inner), (f1dofs, f2dofs))

    prob2 = OperatorSplittingProblem(fsplit2_outer, u0, tspan)

    function ode_NaN(du, u, p, t)
        du[1] = NaN
        du[2] = 0.01u[1]
    end

    f_NaN = ODEFunction(ode_NaN)
    fpw_NaN = PointwiseODEFunction(
        1,
        f_NaN,
        [0.0]
    )
    f_NaN_dofs = f3dofs
    fsplit_NaN = GenericSplitFunction((f1,fpw_NaN), (f1dofs, f_NaN_dofs))
    prob_NaN = OperatorSplittingProblem(fsplit_NaN, u0, tspan)

    fsplit_multiple_pwode_outer = GenericSplitFunction((fpw,fsplit2_inner), (f3dofs, f2dofs))

    prob_multiple_pwode = OperatorSplittingProblem(fsplit_multiple_pwode_outer, u0, tspan)

    function ode2_force_half(du, u, p, t)
        du[1] = 0.5
        du[2] = 0.5
    end

    fpw_force_half = PointwiseODEFunction(
        1,
        ODEFunction(ode2_force_half),
        [0.0]
    )

    fsplit_force_half = GenericSplitFunction((f1,fpw_force_half), (f1dofs, f2dofs))
    prob_force_half = OperatorSplittingProblem(fsplit_force_half, u0, tspan)

    dt = 0.1π
    adaptive_tstep_range = (dt * 1, dt * 2)
    @testset "OperatorSplitting" begin
        for TimeStepperType in (LieTrotterGodunov,)
            timestepper = TimeStepperType(
                (DummyForwardEuler(), DummyForwardEuler())
            )
            timestepper_adaptive = Thunderbolt.ReactionTangentController(timestepper, 0.5, 1.0, adaptive_tstep_range)
            timestepper_inner = TimeStepperType(
                (DummyForwardEuler(), DummyForwardEuler())
            )
            timestepper_inner_adaptive = Thunderbolt.ReactionTangentController(timestepper_inner, 0.5, 1.0, adaptive_tstep_range) #TODO: Copy the controller instead
            timestepper2 = TimeStepperType(
                (DummyForwardEuler(), timestepper_inner)
            )
            timestepper2_adaptive = Thunderbolt.ReactionTangentController(timestepper2, 0.5, 1.0, adaptive_tstep_range)

            for (tstepper1, tstepper_inner, tstepper2) in (
                    (timestepper, timestepper_inner, timestepper2),
                    (timestepper_adaptive, timestepper_inner_adaptive, timestepper2_adaptive)
                    )
                # The remaining code works as usual.
                integrator = DiffEqBase.init(prob, tstepper1, dt=dt, verbose=true, alias_u0=false)
                @test integrator.sol.retcode == DiffEqBase.ReturnCode.Default
                DiffEqBase.solve!(integrator)
                @test integrator.sol.retcode == DiffEqBase.ReturnCode.Success
                ufinal = copy(integrator.u)
                @test ufinal ≉ u0 # Make sure the solve did something

                DiffEqBase.reinit!(integrator)
                @test integrator.sol.retcode == DiffEqBase.ReturnCode.Default
                for (u, t) in DiffEqBase.TimeChoiceIterator(integrator, 0.0:5.0:100.0)
                end
                @test  isapprox(ufinal, integrator.u, atol=1e-8)

                DiffEqBase.reinit!(integrator)
                @test integrator.sol.retcode == DiffEqBase.ReturnCode.Default
                for (uprev, tprev, u, t) in DiffEqBase.intervals(integrator)
                end
                @test  isapprox(ufinal, integrator.u, atol=1e-8)

                DiffEqBase.reinit!(integrator)
                @test integrator.sol.retcode == DiffEqBase.ReturnCode.Default
                DiffEqBase.solve!(integrator)
                @test integrator.sol.retcode == DiffEqBase.ReturnCode.Success

                integrator2 = DiffEqBase.init(prob2, tstepper2, dt=dt, verbose=true, alias_u0=false)
                @test integrator2.sol.retcode == DiffEqBase.ReturnCode.Default
                DiffEqBase.solve!(integrator2)
                @test integrator.sol.retcode == DiffEqBase.ReturnCode.Success
                ufinal2 = copy(integrator2.u)
                @test ufinal2 ≉ u0 # Make sure the solve did something

                DiffEqBase.reinit!(integrator2)
                @test integrator2.sol.retcode == DiffEqBase.ReturnCode.Default
                for (u, t) in DiffEqBase.TimeChoiceIterator(integrator2, 0.0:5.0:100.0)
                end
                @test isapprox(ufinal2, integrator2.u, atol=1e-8)

                DiffEqBase.reinit!(integrator2)
                @test integrator2.sol.retcode == DiffEqBase.ReturnCode.Default
                DiffEqBase.solve!(integrator2)
                @test integrator2.sol.retcode == DiffEqBase.ReturnCode.Success
                @testset "NaNs" begin
                    integrator_NaN = DiffEqBase.init(prob_NaN, tstepper1, dt=dt, verbose=true, alias_u0=false)
                    @test integrator_NaN.sol.retcode == DiffEqBase.ReturnCode.Default
                    DiffEqBase.solve!(integrator_NaN)
                    @test integrator_NaN.sol.retcode ∈ (DiffEqBase.ReturnCode.Unstable, DiffEqBase.ReturnCode.DtNaN)
                end
            end
            integrator = DiffEqBase.init(prob, timestepper, dt=dt, verbose=true, alias_u0=false)
            for (u, t) in DiffEqBase.TimeChoiceIterator(integrator, 0.0:5.0:100.0) end
            integrator_adaptive = DiffEqBase.init(prob, timestepper_adaptive, dt=dt, verbose=true, alias_u0=false)
            for (u, t) in DiffEqBase.TimeChoiceIterator(integrator_adaptive, 0.0:5.0:100.0) end
            @test  isapprox(integrator_adaptive.u, integrator.u, atol=1e-4)
            @testset "Multiple `PointwiseODEFunction`s" begin
                integrator_multiple_pwode = DiffEqBase.init(prob_multiple_pwode, timestepper2_adaptive, dt=dt, verbose=true, alias_u0=false)
                @test_throws AssertionError("No or multiple integrators using PointwiseODEFunction found") DiffEqBase.solve!(integrator_multiple_pwode)
            end
            @testset "σ_s = Inf, R = σ_c" begin
                timestepper_stepfunc_adaptive = Thunderbolt.ReactionTangentController(timestepper, Inf, 0.5, adaptive_tstep_range)
                integrator_stepfunc_adaptive = DiffEqBase.init(prob_force_half, timestepper_stepfunc_adaptive, dt=dt, verbose=true, alias_u0=false)
                DiffEqBase.solve!(integrator_stepfunc_adaptive)
                @test integrator_stepfunc_adaptive._dt == timestepper_stepfunc_adaptive.Δt_bounds[2]
            end
        end
    end
end
