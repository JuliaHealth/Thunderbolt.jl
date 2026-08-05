using Thunderbolt
import SciMLBase
import SciMLIterators: TimeChoiceIterator
using Test
using LinearAlgebra
using LinearSolve
using Logging

const ORTHO_MS = ConstantCoefficient(
    OrthotropicMicrostructure(Vec((1.0, 0.0, 0.0)), Vec((0.0, 1.0, 0.0)), Vec((0.0, 0.0, 1.0))),
)

"""
A short bar of `ncells` hexahedra. `dbcs` decides whether it is clamped or free floating, and
`material` which constitutive model carries the internal forces.
"""
function elastodynamic_bar(;
    ncells = (4, 1, 1),
    ρ = 1.0e3,
    material = PK1Model(Guccione1991PassiveModel(), ORTHO_MS),
    clamped = true,
)
    mesh = generate_mesh(Hexahedron, ncells, Vec((0.0, 0.0, 0.0)), Vec((1.0, 0.2, 0.2)))
    model = ElastodynamicsModel(:d, :v, material, (), ConstantCoefficient(ρ))
    dbcs = if clamped
        [Dirichlet(:d, getfacetset(mesh, "left"), (x, t) -> [0.0, 0.0, 0.0], [1, 2, 3])]
    else
        Dirichlet[]
    end
    return semidiscretize(
        model,
        FiniteElementDiscretization(Dict(:d => LagrangeCollection{1}()^3); dbcs),
        mesh,
    )
end

"""
Solve to `tend` and return the integrator. `v0` is given as a function of the dof index modulo the
spatial dimension, so a velocity field can be written without a coordinate lookup.
"""
function solve_elastodynamic(f, v0, tend, Δt; β = 1 / 4, γ = 1 / 2, adaptive = false, kwargs...)
    u0 = zeros(solution_size(f))
    Thunderbolt.default_initial_condition!(u0, f)
    problem = ElastodynamicsProblem(f, u0, v0, (0.0, tend))
    # `reltol`/`abstol` are `init` keywords, as everywhere else in SciML -- not solver fields.
    integrator = init(problem, NewmarkSolver(; β, γ), dt = Δt; adaptive, verbose = false, kwargs...)
    solve!(integrator)
    return integrator
end

# A velocity field that is a uniform translation along `dir`.
function translation_velocity(f, dir::Vec{3})
    v0 = zeros(length(Thunderbolt.velocity_dofs(f)))
    for i = 1:3:length(v0)
        v0[i], v0[i+1], v0[i+2] = dir[1], dir[2], dir[3]
    end
    return v0
end

# Transverse velocity, growing along the bar so that the free end moves fastest.
# The velocity is given in the displacement field's own numbering, which is the structural problem's.
function bending_velocity(f, amplitude)
    dh = f.structural.dh
    v0 = zeros(ndofs(dh))
    for cell in CellIterator(dh)
        for (i, node) in enumerate(getcoordinates(cell))
            dofs = celldofs(cell)[(3(i-1)+1):(3i)]
            v0[dofs[2]] = amplitude * node[1]
        end
    end
    Ferrite.apply_zero!(v0, Thunderbolt.getch(f.structural))
    return v0
end

@testset "Elastodynamics" begin
    @testset "Uniform translation is integrated exactly" begin
        # A rigid translation leaves the deformation gradient at the identity, so the internal forces
        # vanish for any hyperelastic material and `u(t) = v₀t` solves the problem exactly. Newmark
        # reproduces a constant velocity for any step size, so this pins the predictor/corrector
        # arithmetic without reference to a discretization error.
        f = elastodynamic_bar(clamped = false)
        v0 = translation_velocity(f, Vec((0.3, -0.2, 0.1)))
        tend = 0.5
        integrator = solve_elastodynamic(f, v0, tend, tend / 2)

        # The scheme is exact here; what is not is the nonlinear solve, whose default tolerance is an
        # absolute residual. The inertia enters the residual weighted by `1/(βΔt²)`, so a converged
        # residual corresponds to a displacement error of that order — hence the tolerances below are
        # a statement about the Newton, not about Newmark.
        @test integrator.sol.retcode == SciMLBase.ReturnCode.Success
        @test integrator.u[Thunderbolt.displacement_dofs(f)]≈tend .* v0 rtol=1e-7
        @test velocity(integrator)≈v0 rtol=1e-7
        @test norm(acceleration(integrator)) < 1e-6
    end

    @testset "A nonzero equilibrium stays at rest" begin
        # The equilibrium has to be nonzero to test anything: every operation in the Newmark step is
        # linear in the state, so an all-zero state is preserved under any value of β, γ or the
        # velocity slope, and under a sign-flipped predictor.
        mesh = generate_mesh(Hexahedron, (2, 1, 1), Vec((0.0, 0.0, 0.0)), Vec((1.0, 0.2, 0.2)))
        dbcs = [
            Dirichlet(:d, getfacetset(mesh, "left"), (x, t) -> [0.0, 0.0, 0.0], [1, 2, 3]),
            Dirichlet(:d, getfacetset(mesh, "right"), (x, t) -> [0.05, 0.0, 0.0], [1, 2, 3]),
        ]
        f = semidiscretize(
            ElastodynamicsModel(
                :d,
                :v,
                PK1Model(Guccione1991PassiveModel(), ORTHO_MS),
                (),
                ConstantCoefficient(1.0),
            ),
            FiniteElementDiscretization(Dict(:d => LagrangeCollection{1}()^3); dbcs),
            mesh,
        )
        # Reach the static equilibrium of the held boundary first. `γ = 1` is maximal numerical
        # dissipation, which is what drives the free vibration out; the conserving scheme below would
        # oscillate about the equilibrium forever and never settle.
        settled = solve_elastodynamic(
            f,
            zeros(length(Thunderbolt.velocity_dofs(f))),
            50.0,
            1.0;
            γ = 1.0,
            β = 1.0,
        )
        @test settled.sol.retcode == SciMLBase.ReturnCode.Success
        u_eq = copy(settled.u)
        @test norm(u_eq) > 1.0e-3                                   # genuinely nonzero
        @test norm(velocity(settled)) / norm(u_eq) < 1.0e-6         # genuinely at rest

        problem =
            ElastodynamicsProblem(f, u_eq, zeros(length(Thunderbolt.velocity_dofs(f))), (0.0, 5.0))
        integrator = init(problem, NewmarkSolver(), dt = 0.5, verbose = false)
        solve!(integrator)
        @test integrator.sol.retcode == SciMLBase.ReturnCode.Success
        @test integrator.u≈u_eq rtol=1.0e-6
        @test norm(velocity(integrator)) / norm(u_eq) < 1.0e-4
    end

    @testset "Convergence order in time" begin
        tend, Δt₀ = 0.02, 0.02 / 4
        f = elastodynamic_bar()
        v0 = bending_velocity(f, 20.0)
        run(Δt, γ) = copy(solve_elastodynamic(elastodynamic_bar(), v0, tend, Δt; γ).u)

        # One reference for both studies: every member of the Newmark family converges to the same
        # solution, so the γ = 1/2 run serves the γ = 0.7 study too. Referencing each study against
        # its own coarse fine-run instead leaves the reference's error in the ratio, which biases the
        # low order case upward by ~0.2.
        reference = run(Δt₀ / 64, 1 / 2)
        function observed_order(γ)
            errors = [norm(run(Δt₀ / refinement, γ) - reference) for refinement in (2, 4)]
            @test all(>(0), errors)
            return log2(errors[1] / errors[2])
        end

        @test observed_order(1 / 2)≈2.0 atol=0.15
        # A bound, not an equality: at a step size coarse enough to run in a test, γ = 0.7 has not
        # reached its asymptotic first order. What is pinned is that γ reaches the scheme at all.
        @test observed_order(0.7) < 1.5
    end

    @testset "Numerical dissipation follows γ" begin
        # γ = 1/2 conserves, γ > 1/2 dissipates. Measured as the decay of the swing amplitude over
        # three periods of free vibration, so that no strain energy functional has to be
        # reconstructed. The light density is what puts three periods inside a test-sized run.
        tend = 2.2
        decay = map((1 / 2, 0.6, 0.7)) do γ
            f = elastodynamic_bar(ncells = (2, 1, 1), ρ = 1.0e-2)
            integrator = init(
                ElastodynamicsProblem(
                    f,
                    zeros(solution_size(f)),
                    bending_velocity(f, 0.2),
                    (0.0, tend),
                ),
                NewmarkSolver(; γ, β = (γ + 1 / 2)^2 / 4),
                dt = 5.0e-3,
                verbose = false,
            )
            first_swing, last_swing = 0.0, 0.0
            while integrator.t < tend - 1.0e-12
                step!(integrator)
                amplitude = norm(integrator.u[Thunderbolt.displacement_dofs(f)], Inf)
                integrator.t < tend / 3 && (first_swing = max(first_swing, amplitude))
                integrator.t > 2tend / 3 && (last_swing = max(last_swing, amplitude))
            end
            @test integrator.sol.retcode == SciMLBase.ReturnCode.Success
            return last_swing / first_swing
        end

        # The ordering is the statement: dissipation increases with γ. "0.7 dissipates" alone would
        # also pass for a scheme that dissipates regardless of γ.
        @test decay[1]≈1.0 atol=0.05                   # average acceleration: no secular decay
        @test decay[3] < decay[2] < decay[1] - 0.05
    end

    # The sharpest statement available from outside is the step count: a controller using the right
    # order drives `Δt ∝ tol^(1/3)`, so the number of steps grows by `10^(1/3) ≈ 2.15` per decade of
    # tolerance. The window excludes the neighbouring exponents (order 1 → 3.16, order 3 → 1.78).
    @testset "The step count follows tol^(-1/3)" begin
        tend = 0.5   # dyadic, so a fixed step run lands on it without a closing micro-step
        bar() = elastodynamic_bar(ncells = (2, 1, 1), ρ = 1.0e-2)

        results = map((1.0e-3, 1.0e-4, 1.0e-5)) do reltol
            f = bar()
            integrator = solve_elastodynamic(
                f,
                bending_velocity(f, 0.2),
                tend,
                tend / 2^7;
                adaptive = true,
                reltol,
                abstol = reltol * 1.0e-3,
            )
            @test integrator.sol.retcode == SciMLBase.ReturnCode.Success
            @test integrator.t == tend
            # Rejections happen here, so these runs are also the end-to-end cover of the velocity and
            # acceleration rollback: without it the retried steps build their predictors from the
            # rejected state and the trajectory drifts with no other symptom.
            @test integrator.stats.nreject > 0
            return integrator.stats.naccept
        end

        for i = 1:(length(results)-1)
            @test results[i+1] / results[i]≈10^(1 / 3) rtol=0.1
        end
    end

    @testset "An adaptive run lands where a fine fixed step run does" begin
        # A single tolerance, because the *global* error is not monotone in `reltol` on an oscillatory
        # problem at fixed `tend` -- it is dominated by phase error, which the local estimate does not
        # control. Asserting a trend across tolerances would be pinning noise.
        tend = 0.5
        bar() = elastodynamic_bar(ncells = (2, 1, 1), ρ = 1.0e-2)
        reference = solve_elastodynamic(bar(), bending_velocity(bar(), 0.2), tend, tend / 2^12)
        @test reference.sol.retcode == SciMLBase.ReturnCode.Success

        f = bar()
        adaptive = solve_elastodynamic(
            f,
            bending_velocity(f, 0.2),
            tend,
            tend / 2^7;
            adaptive = true,
            reltol = 1.0e-3,
            abstol = 1.0e-6,
        )
        @test adaptive.sol.retcode == SciMLBase.ReturnCode.Success
        # Compare the displacement, not the state: the velocity block is not small next to it on an
        # oscillating bar, so a mixed norm would measure something else.
        d = Thunderbolt.displacement_dofs(f)
        @test norm(adaptive.u[d] - reference.u[d]) / norm(reference.u[d]) < 5.0e-3
    end

    @testset "The step size follows the solution" begin
        # A constant step size is optimal for the smooth bar above, so a run there cannot show that
        # the controller does anything. An activating sarcomere can: it has a fast transient while the
        # crossbridges engage and a slow approach afterwards.
        f = elastodynamic_bar(
            ncells = (2, 1, 1),
            material = ActiveStressModel(
                Guccione1991PassiveModel(),
                SimpleActiveStress(; Tmax = 220.0e3),
                Thunderbolt.CaDrivenInternalSarcomereModel(
                    Thunderbolt.RDQ20MFModel(),
                    ConstantCoefficient(1.0),
                ),
                OrthotropicMicrostructureModel(
                    ConstantCoefficient(Vec((1.0, 0.0, 0.0))),
                    ConstantCoefficient(Vec((0.0, 1.0, 0.0))),
                    ConstantCoefficient(Vec((0.0, 0.0, 1.0))),
                ),
            ),
        )
        u0 = zeros(solution_size(f))
        Thunderbolt.default_initial_condition!(u0, f)
        integrator = init(
            ElastodynamicsProblem(f, u0, zeros(length(Thunderbolt.velocity_dofs(f))), (0.0, 20.0)),
            NewmarkSolver(),
            dt = 0.05,
            adaptive = true,
            dtmax = 5.0,
            verbose = false,
        )
        ts, dts = Float64[], Float64[]
        while integrator.t < 20.0 - 1.0e-10
            step!(integrator)
            push!(ts, integrator.t)
            push!(dts, integrator.dt)
            integrator.sol.retcode == SciMLBase.ReturnCode.Success || break
        end
        @test integrator.sol.retcode == SciMLBase.ReturnCode.Success
        # Not `max/min`: the step size of a *smooth* run wanders by nearly as much, so that ratio does
        # not separate "responds to the transient" from "wanders". Every step taken while the
        # crossbridges engage is shorter than every step taken after -- a smooth run does not satisfy
        # that, its step size minimum lying in the interior.
        @test maximum(dts[ts .< 2.0]) < minimum(dts[ts .> 10.0])
    end

    @testset "The step size controller is Thunderbolt's own" begin
        # A configuration fact, so it needs no mesh: the default must not silently become a controller
        # reached through `OrdinaryDiffEqCore`'s protocol, which is what the in-package port exists to
        # avoid.
        @test Thunderbolt.default_controller(Float64, NewmarkSolver()) isa Thunderbolt.PIDController
        # The exponent the controller applies is `1/(adaptive_order+1)`; a wrong value here shows up
        # in the step count study only indirectly.
        @test Thunderbolt.adaptive_order(NewmarkSolver()) == 2
    end

    @testset "A failed solve shrinks dt once, not twice" begin
        # `dt` shrinks once per failed attempt: the step footer's `post_newton_controller!` owns the
        # solve-failure case, the controller's reject hook owns the error-estimate case.
        f = elastodynamic_bar(ncells = (2, 1, 1), ρ = 1.0e-2)
        integrator = init(
            ElastodynamicsProblem(f, zeros(solution_size(f)), bending_velocity(f, 0.2), (0.0, 0.5)),
            # A tolerance the Newton cannot reach, so every attempt fails.
            NewmarkSolver(
                inner_solver = Thunderbolt.MultiLevelNewtonRaphsonSolver(
                    newton = NewtonRaphsonSolver(
                        inner_solver = UMFPACKFactorization(),
                        max_iter = 2,
                        tol = 1.0e-30,
                    ),
                ),
            ),
            dt = 0.02,
            adaptive = true,
            verbose = false,
        )
        dt₀ = integrator.dt
        with_logger(NullLogger()) do
            try
                step!(integrator)
            catch
                # the solve gives up eventually; what is asserted is how far `dt` fell on the way
            end
        end
        ff = integrator.opts.failfactor
        @test integrator.stats.nreject > 1
        # Two-sided: `≤` alone is also satisfied by a `dt` that never shrank, which is the opposite
        # bug.
        @test ff^(integrator.stats.nreject - 1) ≤ dt₀ / integrator.dt ≤ ff^integrator.stats.nreject
    end

    @testset "The interpolant is Hermite, not linear" begin
        # `u`, `v` and `a` come from one cubic and its derivatives, so they are mutually consistent:
        # a linear interpolation of each separately does not satisfy `v = dₜu`. The endpoint
        # reproduction of `v` is what a linear interpolant cannot do at all.
        f = elastodynamic_bar(ncells = (2, 1, 1), ρ = 1.0e-2)
        integrator = init(
            ElastodynamicsProblem(f, zeros(solution_size(f)), bending_velocity(f, 0.2), (0.0, 0.5)),
            NewmarkSolver(),
            dt = 0.005,
            adaptive = false,
            verbose = false,
        )
        for _ = 1:4
            step!(integrator)
        end
        fe = Thunderbolt.displacement_dofs(f)
        tmp = zeros(solution_size(f))
        tprev, t = integrator.tprev, integrator.t
        tmid = (tprev + t) / 2

        @test integrator(tmp, tprev)[fe] == integrator.uprev[fe]
        @test integrator(tmp, t)[fe] == integrator.u[fe]
        @test velocity(integrator, tprev) == integrator.cache.vₙ₋₁
        @test velocity(integrator, t) == integrator.cache.vₙ

        # dₜ of the displacement interpolant is the velocity interpolant, and dₜ of that is the
        # acceleration one.
        h = 1.0e-6
        du = (copy(integrator(tmp, tmid + h))[fe] - copy(integrator(tmp, tmid - h))[fe]) / (2h)
        @test du≈velocity(integrator, tmid) rtol=1.0e-8
        dv = (velocity(integrator, tmid + h) - velocity(integrator, tmid)) / h
        @test dv≈acceleration(integrator, tmid) rtol=1.0e-4

        # And it is genuinely not the linear interpolant the fallback would give.
        linear = @. integrator.uprev[fe] +
           (tmid - tprev) / (t - tprev) * (integrator.u[fe] - integrator.uprev[fe])
        @test !isapprox(integrator(tmp, tmid)[fe], linear)
    end

    @testset "Velocity and acceleration interpolate to a requested time" begin
        # `TimeChoiceIterator` interpolates `u` to the requested `t` but leaves the integrator at the
        # end of the bracketing step, so the no-argument accessors report a *different* time than the
        # `u` handed to the loop body. Writing both into one output frame would be silently wrong.
        f = elastodynamic_bar(ncells = (2, 1, 1), ρ = 1.0e-2)
        integrator = init(
            ElastodynamicsProblem(f, zeros(solution_size(f)), bending_velocity(f, 0.2), (0.0, 1.0)),
            NewmarkSolver(),
            dt = 0.3,   # deliberately not a divisor of the requested spacing
            verbose = false,
        )
        mismatched = false
        for (u, t) in TimeChoiceIterator(integrator, 0.0:0.25:1.0)
            v = velocity(integrator, t)
            a = acceleration(integrator, t)
            @test all(isfinite, v)
            @test all(isfinite, a)
            # At a step boundary the two agree; strictly inside a step they must not.
            integrator.t ≈ t || (mismatched |= !(v ≈ velocity(integrator)))
        end
        @test mismatched
    end

    @testset "A rejected step rolls back the velocity and the acceleration" begin
        # The solution vector carries the displacement (and the condensed internal variables) only, so
        # the integrator's own rollback buffer cannot restore the velocity and the acceleration. They
        # are state of the same second order ODE and have to come back with it, or a retried step
        # builds its predictors from the rejected step's state and converges to a wrong answer with no
        # symptom.
        f = elastodynamic_bar(ncells = (2, 1, 1))
        integrator = init(
            ElastodynamicsProblem(f, zeros(solution_size(f)), bending_velocity(f, 0.5), (0.0, 1.0)),
            NewmarkSolver(),
            dt = 0.05,
            verbose = false,
        )
        step!(integrator)
        step!(integrator)
        u, v, a = copy(integrator.u), copy(velocity(integrator)), copy(acceleration(integrator))

        step!(integrator)
        # The step has to move all three, otherwise the rollback below asserts nothing.
        @test !isapprox(integrator.u, u)
        @test !isapprox(velocity(integrator), v)
        @test !isapprox(acceleration(integrator), a)

        Thunderbolt.reject_step!(integrator)
        @test integrator.u == u
        @test velocity(integrator) == v
        @test acceleration(integrator) == a
    end

    @testset "Condensed internal variables under Newmark" begin
        # `LinearMaxwellMaterial` carries a viscous strain governed by `dₜQ = L(F, Q)`. The local
        # problem is the same one backward Euler poses, so it needs nothing from the scheme.
        f = elastodynamic_bar(
            material = Thunderbolt.LinearMaxwellMaterial(
                E₀ = 70e3,
                E₁ = 20e3,
                μ = 1e3,
                η₁ = 1e3,
                ν = 0.3,
            ),
        )
        @test solution_size(f) > ndofs(f.dh) # there is something to condense
        integrator = solve_elastodynamic(f, bending_velocity(f, 1.0), 0.05, 0.005)
        @test integrator.sol.retcode == SciMLBase.ReturnCode.Success
        @test all(isfinite, integrator.u)
        # The viscous strain starts at zero and has to have moved: a material whose internal variable
        # never advanced would pass every assertion above. Read the internal variables by name --
        # anything else in the tail of the solution vector would make this pass for the wrong reason.
        @test maximum(abs, integrator.u[Thunderbolt.internal_variable_range(f)]) > 0
    end

    # `RDQ20MFModel` is rate coupled (`dₜQ = L(F, dₜF, Q)`), so its local problem reads the deformation
    # rate. Under Newmark that rate is `∇v`, with `∂Ḟ/∂u = γ/(βΔt)` rather than `1/Δt` — which is why
    # the element takes a velocity anchor and a coefficient instead of a timestep.
    @testset "Rate coupled sarcomere under Newmark" begin
        # `Tmax` is the cardiac value rather than the `SimpleActiveStress` default of 1.0. That matters
        # for what this testset can conclude: with a near-inert sarcomere the bar barely moves, the
        # stretch rate never leaves the noise floor, and the comparison below degenerates to a test of
        # nothing.
        active(sarcomere) = ActiveStressModel(
            Guccione1991PassiveModel(),
            SimpleActiveStress(; Tmax = 220.0e3),
            Thunderbolt.CaDrivenInternalSarcomereModel(sarcomere, ConstantCoefficient(1.0)),
            OrthotropicMicrostructureModel(
                ConstantCoefficient(Vec((1.0, 0.0, 0.0))),
                ConstantCoefficient(Vec((0.0, 1.0, 0.0))),
                ConstantCoefficient(Vec((0.0, 0.0, 1.0))),
            ),
        )
        function contract(sarcomere)
            f = elastodynamic_bar(ncells = (2, 1, 1), material = active(sarcomere))
            return solve_elastodynamic(f, zeros(length(Thunderbolt.velocity_dofs(f))), 20.0, 1.0)
        end

        integrator = contract(Thunderbolt.RDQ20MFModel())
        # Read the layout off the solved function: a bar built with a different material carries a
        # different internal variable block, so it cannot stand in for this one.
        fe = Thunderbolt.displacement_dofs(integrator.f)
        iv = Thunderbolt.internal_variable_range(integrator.f)
        @test integrator.sol.retcode == SciMLBase.ReturnCode.Success
        @test all(isfinite, integrator.u)
        # The sarcomere activates at Ca = 1 and pulls the bar in.
        @test maximum(integrator.u[iv]) > 0.1
        @test norm(integrator.u[fe]) > 0.1

        # Dropping the velocity coupling has to change the answer: if the element fed the material no
        # rate at all, the two models would agree exactly, since they differ in nothing else.
        wrapped = contract(Thunderbolt.AsRateIndependent(Thunderbolt.RDQ20MFModel()))
        @test wrapped.sol.retcode == SciMLBase.ReturnCode.Success
        @test norm(wrapped.u[fe] - integrator.u[fe]) / norm(integrator.u[fe]) > 0.05
    end
end
