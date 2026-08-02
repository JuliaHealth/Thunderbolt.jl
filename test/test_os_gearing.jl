import Thunderbolt: Thunderbolt, ThunderboltTimeIntegrator
using DiffEqBase, OrdinaryDiffEqOperatorSplitting, Thunderbolt
using OrdinaryDiffEqLowOrderRK
# using BenchmarkTools
using UnPack
using Test
import SciMLIterators: TimeChoiceIterator, intervals

# For testing purposes
struct DummyForwardEuler <: Thunderbolt.AbstractSolver end

DiffEqBase.isadaptive(::DummyForwardEuler) = false

mutable struct DummyForwardEulerCache{duType, uType, uprevType, duMatType} <:
               Thunderbolt.AbstractTimeSolverCache
    du::duType
    dumat::duMatType
    uₙ::uType
    uₙ₋₁::uprevType
    fail_at_iter::Int # 0 = never fail; there is no public transient-failure injection
    nfails::Int
end

# Dispatch for leaf construction
Thunderbolt.num_states(::ODEFunction) = 2                   # FIXME
Thunderbolt.transmembranepotential_index(::ODEFunction) = 1 # FIXME

function Thunderbolt.setup_solver_cache(
    f::Any,
    solver::DummyForwardEuler,
    t₀;
    u = nothing,
    uprev = nothing,
)
    n = u === nothing ? Thunderbolt.num_states(f) : length(u)
    du = zeros(n)
    uₙ = u === nothing ? zeros(n) : u
    uₙ₋₁ = uprev === nothing ? zeros(n) : uprev
    dumat = reshape(du, (:, 1))

    return DummyForwardEulerCache(du, dumat, uₙ, uₙ₋₁, 0, 0)
end

# Dispatch innermost solve
function Thunderbolt.OrdinaryDiffEqCore.perform_step!(
    integ::ThunderboltTimeIntegrator,
    cache::DummyForwardEulerCache,
)
    if cache.fail_at_iter != 0 && integ.iter == cache.fail_at_iter && cache.nfails == 0
        cache.nfails += 1
        integ.force_stepfail = true # what the production wrapper does on an inner failure
        return false
    end
    (; f, dt, u, p, t) = integ
    (; du) = cache

    f isa Thunderbolt.PointwiseODEFunction ? f.ode(du, u, p, t) : f(du, u, p, t)
    @. u += dt * du
    cache.dumat[:, 1] .= du

    return true
end

@testset "Operator Splitting API" begin
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
    f1dofs = [1, 2, 3]
    f2dofs = [1, 3]
    fpw = PointwiseODEFunction(f2, nothing, 1:length(f2dofs))

    fsplit1 = GenericSplitFunction((f1, fpw), (f1dofs, f2dofs))
    fsplit1b = GenericSplitFunction((f1, f2), (f1dofs, f2dofs))

    # Now the usual setup just with our new problem type.
    # u0 = rand(3)
    u0 = [
        0.7611944793397108
        0.9059606424982555
        0.5755174199139956
    ]
    tspan = (0.0, 100.0)
    prob1 = OperatorSplittingProblem(fsplit1, u0, tspan)
    probb = OperatorSplittingProblem(fsplit1b, u0, tspan)

    # Now some recursive splitting
    function ode3(du, u, p, t)
        du[1] = 0.005u[2]
        du[2] = 0.005u[1]
    end
    f3 = ODEFunction(ode3)
    # The time stepper carries the individual solver information.

    # Note that we define the dof indices w.r.t the parent function.
    # Hence the indices for `fsplit2_inner` are.
    f1dofs = [1, 2, 3]
    f2dofs = [1, 3]
    f3dofs = [1, 3]
    fsplit2_inner = GenericSplitFunction((fpw, f3), ([1, 2], [1, 2]))
    fsplit2_outer = GenericSplitFunction((f1, fsplit2_inner), (f1dofs, f2dofs))
    fsplit2_innerb = GenericSplitFunction((f2, f3), ([1, 2], [1, 2]))
    fsplit2_outerb = GenericSplitFunction((f1, fsplit2_innerb), (f1dofs, f2dofs))

    prob2 = OperatorSplittingProblem(fsplit2_outer, u0, tspan)
    prob2b = OperatorSplittingProblem(fsplit2_outerb, u0, tspan)

    function ode_NaN(du, u, p, t)
        du[1] = NaN
        du[2] = 0.01u[1]
    end

    f_NaN = ODEFunction(ode_NaN)
    fpw_NaN = PointwiseODEFunction(f_NaN, nothing, 1:2)
    f_NaN_dofs = f3dofs
    fsplit_NaN = GenericSplitFunction((f1, fpw_NaN), (f1dofs, f_NaN_dofs))
    prob_NaN = OperatorSplittingProblem(fsplit_NaN, u0, tspan)

    fsplit_NaNb = GenericSplitFunction((f1, f_NaN), (f1dofs, f_NaN_dofs))
    prob_NaNb = OperatorSplittingProblem(fsplit_NaNb, u0, tspan)

    fsplit_multiple_pwode_outer = GenericSplitFunction((fpw, fsplit2_inner), (f3dofs, f2dofs))

    prob_multiple_pwode = OperatorSplittingProblem(fsplit_multiple_pwode_outer, u0, tspan)

    function ode2_force_half(du, u, p, t)
        du[1] = 0.5
        du[2] = 0.5
    end

    fpw_force_half = PointwiseODEFunction(ODEFunction(ode2_force_half), nothing, 1:2)

    fsplit_force_half = GenericSplitFunction((f1, fpw_force_half), (f1dofs, f2dofs))
    prob_force_half = OperatorSplittingProblem(fsplit_force_half, u0, tspan)

    dt = 0.1π
    # Non-degenerate bounds: Δt_min < dt < Δt_max, so a controller that actually adapts
    # must move dt away from the initial value. (A previous revision used (dt, dt), which
    # any constant-dt implementation satisfies trivially.)
    adaptive_tstep_range = (dt * 0.5, dt * 2)
    @testset "Internal consistency" failfast = true begin
        for TimeStepperType in (LieTrotterGodunov,)
            timestepper0 = TimeStepperType((Euler(), Euler()))
            timestepper1 = TimeStepperType((DummyForwardEuler(), DummyForwardEuler()))
            timestepper1_adaptive =
                Thunderbolt.ReactionTangentController(timestepper1, 0.5, 1.0, adaptive_tstep_range)
            timestepper1_inner = TimeStepperType((DummyForwardEuler(), DummyForwardEuler()))

            timestepper2 = TimeStepperType((DummyForwardEuler(), timestepper1_inner))
            timestepper2_adaptive =
                Thunderbolt.ReactionTangentController(timestepper2, 0.5, 1.0, adaptive_tstep_range)

            @testset "$timestepper" for (prob, timestepper, adaptive) in (
                (prob1, timestepper1, false),
                (prob2, timestepper2, false),
                (prob1, timestepper1_adaptive, true),
                (prob2, timestepper2_adaptive, true),
            )
                # The remaining code works as usual.
                integrator =
                    DiffEqBase.init(prob, timestepper, dt = dt, verbose = true, alias_u0 = false)
                @test integrator.sol.retcode == DiffEqBase.ReturnCode.Default
                DiffEqBase.solve!(integrator)
                @test integrator.sol.retcode == DiffEqBase.ReturnCode.Success
                ufinal = copy(integrator.u)
                @test ufinal ≉ u0 # Make sure the solve did something
                @test integrator.uprev ≉ u0
                adaptive || @test integrator.iter == ceil(Int, (tspan[2]-tspan[1])/dt)
                @test integrator.t == tspan[2]

                DiffEqBase.reinit!(integrator)
                # integrator.dt = dt
                @test integrator.sol.retcode == DiffEqBase.ReturnCode.Default
                lastu = copy(integrator.u)
                for (u, t) in TimeChoiceIterator(integrator, 0.0:5.0:100.0)
                    lastu .= u
                end
                @test lastu ≈ integrator.u
                @test integrator.sol.retcode == DiffEqBase.ReturnCode.Default
                # The choice points enter as tstops and clip the step length, so an
                # adaptive run takes a genuinely different trajectory than plain solve!.
                @test isapprox(ufinal, integrator.u, atol = adaptive ? 1e-4 : 1e-6)
                adaptive || @test integrator.iter == ceil(Int, (tspan[2]-tspan[1])/dt)
                @test integrator.t == tspan[2]

                DiffEqBase.reinit!(integrator)
                # integrator.dt = dt
                @test integrator.sol.retcode == DiffEqBase.ReturnCode.Default
                for (uprev, tprev, u, t) in intervals(integrator)
                end
                @test integrator.sol.retcode == DiffEqBase.ReturnCode.Default
                @test isapprox(ufinal, integrator.u, atol = adaptive ? 1e-4 : 1e-6)
                adaptive || @test integrator.iter == ceil(Int, (tspan[2]-tspan[1])/dt)
                @test integrator.t == tspan[2]

                DiffEqBase.reinit!(integrator)
                # integrator.dt = dt
                @test integrator.sol.retcode == DiffEqBase.ReturnCode.Default
                DiffEqBase.solve!(integrator)
                @test integrator.sol.retcode == DiffEqBase.ReturnCode.Success
                adaptive || @test integrator.iter == ceil(Int, (tspan[2]-tspan[1])/dt)
                @test integrator.t == tspan[2]
            end

            @testset "NaNs" begin
                integrator_NaN = DiffEqBase.init(
                    prob_NaN,
                    timestepper1,
                    dt = dt,
                    verbose = true,
                    alias_u0 = false,
                )
                @test integrator_NaN.sol.retcode == DiffEqBase.ReturnCode.Default
                DiffEqBase.solve!(integrator_NaN)
                @test integrator_NaN.sol.retcode ∈
                      (DiffEqBase.ReturnCode.Unstable, DiffEqBase.ReturnCode.DtNaN)
            end

            integrator =
                DiffEqBase.init(prob1, timestepper1, dt = dt, verbose = true, alias_u0 = false)
            for (u, t) in TimeChoiceIterator(integrator, 0.0:5.0:100.0)
            end
            @test integrator.sol.retcode == DiffEqBase.ReturnCode.Default

            # ReactionTangentController is a heuristic dt = σ(R) map -- it has no error
            # estimator and never rejects on accuracy. These tests pin that the accepted
            # step size actually follows σ(R). Regression tests for the controller having
            # silently degenerated to constant dt.
            @testset "RTC adaptivity" begin
                Δt_bounds = adaptive_tstep_range

                # prob_force_half forces du = 0.5 on the pointwise operator, so R = 0.5
                # exactly and σ(R) is computable by hand.
                @testset "Sigmoid formula at R = 0.5" begin
                    σ_s, σ_c = 0.5, 1.0
                    rtc = Thunderbolt.ReactionTangentController(timestepper1, σ_s, σ_c, Δt_bounds)
                    integ = DiffEqBase.init(
                        prob_force_half,
                        rtc,
                        dt = dt,
                        verbose = true,
                        alias_u0 = false,
                    )
                    DiffEqBase.solve!(integ)
                    @test integ.sol.retcode == DiffEqBase.ReturnCode.Success
                    @test integ.t == tspan[2]
                    dt_expected =
                        (1 - 1 / (1 + exp((σ_c - 0.5) * σ_s))) * (Δt_bounds[2] - Δt_bounds[1]) +
                        Δt_bounds[1]
                    @test integ.dtcache ≈ dt_expected
                end

                # σ_s = Inf turns σ into a step function: Δt_max for R ≤ σ_c (including
                # the boundary case R = σ_c), Δt_min for R > σ_c. This asserts that dt
                # moves in the direction σ(R) predicts, to either bound.
                @testset "σ_s = Inf, σ_c = $σ_c" for (σ_c, dt_expected) in (
                    (0.75, Δt_bounds[2]),
                    (0.5, Δt_bounds[2]),
                    (0.25, Δt_bounds[1]),
                )
                    rtc = Thunderbolt.ReactionTangentController(timestepper1, Inf, σ_c, Δt_bounds)
                    integ = DiffEqBase.init(
                        prob_force_half,
                        rtc,
                        dt = dt,
                        verbose = true,
                        alias_u0 = false,
                    )
                    DiffEqBase.solve!(integ)
                    @test integ.sol.retcode == DiffEqBase.ReturnCode.Success
                    @test integ.t == tspan[2]
                    @test integ.dtcache == dt_expected
                end

                # On prob1 the reaction tangent is R = 0.01 max|u| ≪ σ_c = 1, so σ(R) sits
                # near the upper bound: the accepted dt must grow beyond the initial dt and
                # the solve must need fewer steps than the fixed-dt reference.
                @testset "dt grows for R ≪ σ_c" begin
                    integrator_adaptive = DiffEqBase.init(
                        prob1,
                        timestepper1_adaptive,
                        dt = dt,
                        verbose = true,
                        alias_u0 = false,
                    )
                    DiffEqBase.solve!(integrator_adaptive)
                    @test integrator_adaptive.sol.retcode == DiffEqBase.ReturnCode.Success
                    @test integrator_adaptive.t == tspan[2]
                    @test integrator_adaptive.dtcache > dt
                    @test integrator_adaptive.iter < ceil(Int, (tspan[2]-tspan[1])/dt)

                    integrator_reference = DiffEqBase.init(
                        prob1,
                        timestepper1,
                        dt = dt,
                        verbose = true,
                        alias_u0 = false,
                    )
                    DiffEqBase.solve!(integrator_reference)
                    @test isapprox(integrator_adaptive.u, integrator_reference.u, atol = 1e-4)
                end

                @testset "Multiple `PointwiseODEFunction`s" begin
                    integrator_multiple_pwode = DiffEqBase.init(
                        prob_multiple_pwode,
                        timestepper2_adaptive,
                        dt = dt,
                        verbose = true,
                        alias_u0 = false,
                    )
                    @test_throws AssertionError(
                        "No or multiple integrators using PointwiseODEFunction found",
                    ) DiffEqBase.solve!(integrator_multiple_pwode)
                end
            end

            # A transiently failing non-adaptive child under an adaptive root must be
            # rolled back and retried on a shrunken interval, not abort the solve.
            # Covers the child-rollback protocol: u/uprev restored from the outer
            # rollback anchor, child clocks rewound (asserted internally on every
            # accepted step), branded retcode and last_step_failed cleared.
            @testset "Transient child failure is retried" begin
                flaky = Thunderbolt.ReactionTangentController(
                    LieTrotterGodunov((DummyForwardEuler(), DummyForwardEuler())),
                    0.5,
                    1.0,
                    adaptive_tstep_range,
                )
                integ = DiffEqBase.init(prob1, flaky, dt = dt, verbose = true, alias_u0 = false)
                integ.child_subintegrators[1].cache.fail_at_iter = 4
                DiffEqBase.solve!(integ)
                @test integ.sol.retcode == DiffEqBase.ReturnCode.Success
                @test integ.t == tspan[2]
                @test integ.stats.nreject == 1
                @test integ.child_subintegrators[1].cache.nfails == 1
                for child in integ.child_subintegrators
                    @test child.t == integ.t
                    @test child.sol.retcode == DiffEqBase.ReturnCode.Success
                end

                integ_reference = DiffEqBase.init(
                    prob1,
                    Thunderbolt.ReactionTangentController(
                        LieTrotterGodunov((DummyForwardEuler(), DummyForwardEuler())),
                        0.5,
                        1.0,
                        adaptive_tstep_range,
                    ),
                    dt = dt,
                    verbose = true,
                    alias_u0 = false,
                )
                DiffEqBase.solve!(integ_reference)
                @test integ_reference.stats.nreject == 0
                @test isapprox(integ.u, integ_reference.u, atol = 1e-4)
            end
        end
    end

    @testset "OrdinaryDiffEqLowOrderRK compat" failfast=true begin
        for TimeStepperType in (LieTrotterGodunov,)
            timestepper = TimeStepperType((DummyForwardEuler(), Euler()))
            timestepper_inner = TimeStepperType((Euler(), DummyForwardEuler()))
            timestepper2 = TimeStepperType((DummyForwardEuler(), timestepper_inner))

            for (tstepper1, tstepper2) in ((timestepper, timestepper2),)
                # The remaining code works as usual.
                integrator =
                    DiffEqBase.init(probb, tstepper1, dt = dt, verbose = true, alias_u0 = false)
                @test integrator.sol.retcode == DiffEqBase.ReturnCode.Default
                DiffEqBase.solve!(integrator)
                @test integrator.sol.retcode == DiffEqBase.ReturnCode.Success
                ufinal = copy(integrator.u)
                @test ufinal ≉ u0 # Make sure the solve did something
                @test integrator.iter == ceil(Int, (tspan[2]-tspan[1])/dt)
                @test integrator.t == tspan[2]

                DiffEqBase.reinit!(integrator)
                @test integrator.sol.retcode == DiffEqBase.ReturnCode.Default
                for (u, t) in TimeChoiceIterator(integrator, 0.0:5.0:100.0)
                end
                @test isapprox(ufinal, integrator.u, atol = 1e-8)
                @test integrator.iter == ceil(Int, (tspan[2]-tspan[1])/dt)
                @test integrator.t == tspan[2]

                DiffEqBase.reinit!(integrator)
                @test integrator.sol.retcode == DiffEqBase.ReturnCode.Default
                for (uprev, tprev, u, t) in intervals(integrator)
                end
                @test isapprox(ufinal, integrator.u, atol = 1e-8)
                @test integrator.iter == ceil(Int, (tspan[2]-tspan[1])/dt)
                @test integrator.t == tspan[2]

                DiffEqBase.reinit!(integrator)
                @test integrator.sol.retcode == DiffEqBase.ReturnCode.Default
                DiffEqBase.solve!(integrator)
                @test integrator.sol.retcode == DiffEqBase.ReturnCode.Success
                @test integrator.iter == ceil(Int, (tspan[2]-tspan[1])/dt)
                @test integrator.t == tspan[2]

                integrator2 =
                    DiffEqBase.init(prob2b, tstepper2, dt = dt, verbose = true, alias_u0 = false)
                @test integrator2.sol.retcode == DiffEqBase.ReturnCode.Default
                DiffEqBase.solve!(integrator2)
                @test integrator2.sol.retcode == DiffEqBase.ReturnCode.Success
                ufinal2 = copy(integrator2.u)
                @test ufinal2 ≉ u0 # Make sure the solve did something
                @test integrator2.iter == ceil(Int, (tspan[2]-tspan[1])/dt)
                @test integrator2.t == tspan[2]

                DiffEqBase.reinit!(integrator2)
                @test integrator2.sol.retcode == DiffEqBase.ReturnCode.Default
                for (u, t) in TimeChoiceIterator(integrator2, 0.0:5.0:100.0)
                end
                @test isapprox(ufinal2, integrator2.u, atol = 1e-8)
                @test integrator2.iter == ceil(Int, (tspan[2]-tspan[1])/dt)
                @test integrator2.t == tspan[2]

                DiffEqBase.reinit!(integrator2)
                @test integrator2.sol.retcode == DiffEqBase.ReturnCode.Default
                DiffEqBase.solve!(integrator2)
                @test integrator2.sol.retcode == DiffEqBase.ReturnCode.Success
                @test integrator2.iter == ceil(Int, (tspan[2]-tspan[1])/dt)
                @test integrator2.t == tspan[2]

                @testset "NaNs" begin
                    integrator_NaN = DiffEqBase.init(
                        prob_NaNb,
                        tstepper1,
                        dt = dt,
                        verbose = true,
                        alias_u0 = false,
                    )
                    @test integrator_NaN.sol.retcode == DiffEqBase.ReturnCode.Default
                    DiffEqBase.solve!(integrator_NaN)
                    @test integrator_NaN.sol.retcode ∈
                          (DiffEqBase.ReturnCode.Unstable, DiffEqBase.ReturnCode.DtNaN)
                end
            end
            integrator =
                DiffEqBase.init(probb, timestepper, dt = dt, verbose = true, alias_u0 = false)
            for (u, t) in TimeChoiceIterator(integrator, 0.0:5.0:100.0)
            end
            @test integrator.u ≉ u0 # Make sure the solve did something
        end
    end
end

@testset "Nested split with view-wired leaves addresses the right dofs" begin
    # Solution indices are node-local, so a leaf wired as a view into the root solution
    # vector must still address the node's slice. Guards a bug fixed in
    # OrdinaryDiffEqOperatorSplitting 0.4.0; the nested-vs-flat tests above cannot see it,
    # since both sides were wrong together. dof 2 is touched only by the outer decay
    # operator, so it has a closed form.
    decay(du, u, p, t) = (@. du = -0.1 * u; nothing)
    couple_a(du, u, p, t) = (du[1] = 0.01u[2]; du[2] = 0.01u[1]; nothing)
    couple_b(du, u, p, t) = (du[1] = 0.005u[2]; du[2] = 0.005u[1]; nothing)

    f_inner = GenericSplitFunction(
        (ODEFunction(couple_a), ODEFunction(couple_b)),
        ([1, 2], [1, 2]), # node-local: these index into the node's slice [1, 3]
    )
    f_outer = GenericSplitFunction((ODEFunction(decay), f_inner), ([1, 2, 3], [1, 3]))

    u0 = [0.7611944793397108, 0.9059606424982555, 0.5755174199139956]
    tf, dt = 100.0, 0.1π
    integrator = DiffEqBase.init(
        OperatorSplittingProblem(f_outer, copy(u0), (0.0, tf)),
        LieTrotterGodunov((
            DummyForwardEuler(),
            LieTrotterGodunov((DummyForwardEuler(), DummyForwardEuler())),
        )),
        dt = dt,
        verbose = true,
        alias_u0 = false,
    )
    DiffEqBase.solve!(integrator)

    # Forward Euler on pure decay, with the final step clipped by the tf tstop.
    nfull = floor(Int, tf / dt)
    u2_exact = (1 - 0.1dt)^nfull * (1 - 0.1 * (tf - nfull * dt)) * u0[2]

    @test integrator.sol.retcode == DiffEqBase.ReturnCode.Success
    @test integrator.u[2] ≈ u2_exact rtol = 1e-12
    @test integrator.u[2] != integrator.u[3]
end
