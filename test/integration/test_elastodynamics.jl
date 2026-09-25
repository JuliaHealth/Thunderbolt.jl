using Thunderbolt
import SciMLBase
import SciMLIterators: TimeChoiceIterator
using Test
using LinearAlgebra
using LinearSolve
using Logging
import FerriteOperators

const ORTHO_MS = ConstantCoefficient(
    OrthotropicMicrostructure(Vec((1.0, 0.0, 0.0)), Vec((0.0, 1.0, 0.0)), Vec((0.0, 0.0, 1.0))),
)

# The state dofs carrying each field, looked up by the name the model published. These also serve as the
# check that the descriptors agree with the `state_mapping` the Newmark solver indexes with.
solution_indices_of_displacement(f) = solution_indices(f, :d)
solution_indices_of_velocity(f) = solution_indices(f, :v)

"""
A short bar of `ncells` hexahedra. `dbcs` decides whether it is clamped or free floating, and
`material` which constitutive model carries the internal forces.
"""
function elastodynamic_bar(;
    ncells = (4, 1, 1),
    ρ = 1.0e3,
    material = PK1Model(Guccione1991PassiveModel(), ORTHO_MS),
    clamped = true,
    facet_models = (),
    mass = ConsistentMass(),
)
    mesh = generate_mesh(Hexahedron, ncells, Vec((0.0, 0.0, 0.0)), Vec((1.0, 0.2, 0.2)))
    model = ElastodynamicsModel(:d, :v, material, facet_models, ConstantCoefficient(ρ))
    dbcs = if clamped
        [Dirichlet(:d, getfacetset(mesh, "left"), (x, t) -> [0.0, 0.0, 0.0], [1, 2, 3])]
    else
        Dirichlet[]
    end
    return semidiscretize(
        model,
        FiniteElementDiscretization(Dict(:d => LagrangeCollection{1}()^3); dbcs, mass),
        mesh,
    )
end

@testset "Newmark refuses a lumped mass" begin
    # The stage combines M and K on one sparsity pattern, which a `Diagonal` mass has not.
    f = elastodynamic_bar(; ncells = (1, 1, 1), mass = LumpedMass())
    @test_throws ErrorException Thunderbolt.setup_stage_operator(f, NewmarkSolver(), nothing, 0.0)
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
    v0 = zeros(length(solution_indices_of_velocity(f)))
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
        @test integrator.u[solution_indices_of_displacement(f)]≈tend .* v0 rtol=1e-7
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
            zeros(length(solution_indices_of_velocity(f))),
            50.0,
            1.0;
            γ = 1.0,
            β = 1.0,
        )
        @test settled.sol.retcode == SciMLBase.ReturnCode.Success
        u_eq = copy(settled.u)
        @test norm(u_eq) > 1.0e-3                                   # genuinely nonzero
        @test norm(velocity(settled)) / norm(u_eq) < 1.0e-6         # genuinely at rest

        problem = ElastodynamicsProblem(
            f,
            u_eq,
            zeros(length(solution_indices_of_velocity(f))),
            (0.0, 5.0),
        )
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
                amplitude = norm(integrator.u[solution_indices_of_displacement(f)], Inf)
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
        d = solution_indices_of_displacement(f)
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
            ElastodynamicsProblem(
                f,
                u0,
                zeros(length(solution_indices_of_velocity(f))),
                (0.0, 20.0),
            ),
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
        fe = solution_indices_of_displacement(f)
        tmp = zeros(solution_size(f))
        tprev, t = integrator.tprev, integrator.t
        tmid = (tprev + t) / 2

        @test integrator(tmp, tprev)[fe] == integrator.uprev[fe]
        @test integrator(tmp, t)[fe] == integrator.u[fe]
        @test velocity(integrator, tprev) == view(integrator.uprev, solution_indices_of_velocity(f))
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

    @testset "Facet models reach the assembly" begin
        # The internal force weak form is lowered to a quasi-static one, and the facet models ride
        # along with it. Nothing else in this file passes a non-empty `facet_models`, so without this
        # a lowering that silently dropped them would look identical to one that did not.
        # A large enough swing that the spring force -- which scales with the displacement it
        # sees -- is not lost against the inertia.
        v0(f) = bending_velocity(f, 5.0)
        free = elastodynamic_bar(ncells = (2, 1, 1))
        sprung = elastodynamic_bar(
            ncells = (2, 1, 1),
            # `RobinBC`, not `NormalSpringBC`: the latter resists the *normal* displacement, and
            # the free end's normal is along the bar while the swing is transverse, so it would
            # barely register. Stiff enough to matter against Guccione, too -- at 1e4 the answer
            # moves by 3e-7 and the test would pass whether or not the model was ever assembled.
            facet_models = (RobinBC(1.0e8, "right"),),
        )

        uf = solve_elastodynamic(free, v0(free), 0.05, 0.005)
        us = solve_elastodynamic(sprung, v0(sprung), 0.05, 0.005)
        @test uf.sol.retcode == SciMLBase.ReturnCode.Success
        @test us.sol.retcode == SciMLBase.ReturnCode.Success

        # A spring on the free end has to change the motion; if the facet model never reached an
        # element cache the two solves would agree to round-off.
        d = solution_indices_of_displacement(free)
        @test norm(us.u[d] - uf.u[d]) / norm(uf.u[d]) > 0.01
    end

    @testset "Prescribing the velocity is refused" begin
        # The velocity is a field of the state, but Newmark writes it from the converged displacement,
        # so a prescribed value would be overwritten. Refusing says that; without the check the
        # condition reaches the structural sub-problem, which carries no velocity field, and reports
        # that the handler knows only :d -- an internal decomposition surfacing as a user error.
        mesh = generate_mesh(Hexahedron, (2, 1, 1), Vec((0.0, 0.0, 0.0)), Vec((1.0, 0.2, 0.2)))
        model = ElastodynamicsModel(
            :d,
            :v,
            PK1Model(Guccione1991PassiveModel(), ORTHO_MS),
            (),
            ConstantCoefficient(1.0e3),
        )
        discretization = FiniteElementDiscretization(
            Dict(:d => LagrangeCollection{1}()^3);
            dbcs = [Dirichlet(:v, getfacetset(mesh, "left"), (x, t) -> [0.0, 0.0, 0.0], [1, 2, 3])],
        )
        err = try
            semidiscretize(model, discretization, mesh)
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("velocity", err.msg)
    end

    @testset "Two subdomains with different internal variable models" begin
        # The configuration the internal variable wiring is most exposed to: the two halves condense a
        # different number of unknowns per cell, so a mapping built from one subdomain's layout, or
        # from an assumed subdomain ordering, would put the sarcomere state where the viscous strain
        # belongs. Nothing else in the suite pairs unequal condensed blocks under Newmark.
        grid = generate_grid(
            Hexahedron,
            (4, 1, 1),
            Ferrite.Vec{3}((0.0, 0.0, 0.0)),
            Ferrite.Vec{3}((1.0, 0.2, 0.2)),
        )
        addcellset!(grid, "left half", x -> x[1] ≤ 0.5)
        addcellset!(grid, "right half", x -> x[1] ≥ 0.5)
        mesh = to_mesh(grid)

        viscous = ElastodynamicsModel(
            :d,
            :v,
            Thunderbolt.LinearMaxwellMaterial(E₀ = 70e3, E₁ = 20e3, μ = 1e3, η₁ = 1e3, ν = 0.3),
            (),
            ConstantCoefficient(1.0e3),
        )
        elastic = ElastodynamicsModel(
            :d,
            :v,
            PK1Model(Guccione1991PassiveModel(), ORTHO_MS),
            (),
            ConstantCoefficient(2.0e3), # a different density, so the mass term is per subdomain too
        )
        f = semidiscretize(
            Dict{String, Any}("left half" => viscous, "right half" => elastic) |>
            Thunderbolt.narrow_dict_types,
            FiniteElementDiscretization(
                Dict(:d => LagrangeCollection{1}()^3);
                dbcs = [
                    Dirichlet(:d, getfacetset(mesh, "left"), (x, t) -> [0.0, 0.0, 0.0], [1, 2, 3]),
                ],
            ),
            mesh,
        )

        # Only the viscoelastic half condenses anything, so the two halves genuinely differ.
        @test solution_size(f) > ndofs(f.dh)

        integrator = solve_elastodynamic(f, bending_velocity(f, 1.0), 0.05, 0.005)
        @test integrator.sol.retcode == SciMLBase.ReturnCode.Success
        @test all(isfinite, integrator.u)
        @test norm(integrator.u[solution_indices_of_displacement(f)]) > 0
        # The viscous strain has to have moved, and it can only be read correctly if the internal
        # variable wiring lines the two numberings up cell by cell.
        @test maximum(abs, integrator.u[Thunderbolt.internal_variable_range(f)]) > 0
    end

    @testset "A rejected step rolls back the velocity and the acceleration" begin
        # The velocity rides along in the solution vector, so the integrator's own rollback restores
        # it. The acceleration does not: it is determined by `(u, v)` rather than part of the state,
        # and is cached to avoid a mass solve per step. A step the error controller rejects has
        # already overwritten it, so without a buffer of its own the retry would build its predictors
        # from the rejected attempt and converge to a wrong answer with no symptom.
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
            return solve_elastodynamic(f, zeros(length(solution_indices_of_velocity(f))), 20.0, 1.0)
        end

        integrator = contract(Thunderbolt.RDQ20MFModel())
        # Read the layout off the solved function: a bar built with a different material carries a
        # different internal variable block, so it cannot stand in for this one.
        fe = solution_indices_of_displacement(integrator.f)
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

@testset "Deformation gradient report reads the displacement, not the state" begin
    # An elastodynamics handler carries the velocity alongside the displacement, so a report that took
    # "the first field" or differentiated the raw state vector would be reading the wrong block. The
    # displacement symbol comes from the model rather than from the handler for exactly this reason.
    f = elastodynamic_bar()
    @test Thunderbolt.displacement_symbols(f) == (:d,)
    @test Set(first(f.dh.subdofhandlers).field_names) == Set((:d, :v))

    # A uniaxial stretch on the displacement block, and a value the report must ignore on the velocity
    # block: `det F` comes out exactly 1.25 only if the velocity is left out of the differentiation.
    u = zeros(solution_size(f))
    Thunderbolt.default_initial_condition!(u, f)
    u[solution_indices(f, :v)] .= 1.0e3
    Ferrite.apply_analytical!(u, f.dh, :d, x -> Vec((0.25x[1], 0.0, 0.0)))
    report = Thunderbolt.deformation_gradient_report(f, u)
    @test report.n_subdomains == 1
    @test report.minJ ≈ 1.25
    @test report.maxJ ≈ 1.25
    @test !Thunderbolt.is_inverted(report)
end

@testset "A dashpot's tangent carries the scheme's reconstruction slope" begin
    # A dashpot is written once against the reconstructed `:v` slot rather than once per scheme. Its
    # tangent block `∂v∂u ⋅ D` is not part of ∂F/∂u -- `:v` is frozen under that request -- and is
    # served by the weighted Jacobian, whose `:v` weight is the reconstruction slope the scheme
    # supplies. Newmark's is `γ/(βΔt)`, backward Euler's is `1/Δt`.
    #
    # Asserted on the element tangent rather than by comparing two solves at different `γ`: `γ` also
    # changes the scheme's own velocity update and its numerical dissipation, so a displacement
    # difference across `γ` measures the scheme at least as much as it measures the boundary condition.
    Δt, β, γ             = 0.05, 1 / 4, 1 / 2
    newmark_slope        = γ / (β * Δt)
    backward_euler_slope = inv(Δt)
    @test newmark_slope ≉ backward_euler_slope

    # The boundary condition names its field explicitly, which is the case the `dof_range` indexing
    # exists for: this handler carries `:d` and `:v` in one `SubDofHandler`.
    f = elastodynamic_bar()
    dh = f.dh
    sdh = first(dh.subdofhandlers)
    cache = FerriteOperators.setup_facet_item_cache(
        ViscousRobinBC(1.0e4, "right", :d),
        FacetQuadratureRule{RefHexahedron}(2),
        sdh,
    )

    cell = Ferrite.CellCache(sdh)
    Ferrite.reinit!(cell, first(getfacetset(Ferrite.get_grid(dh), "right"))[1])
    n = ndofs_per_cell(sdh)
    # The damping block depends on neither slot value, only on the weight.
    args = FerriteOperators.FacetArgs(
        (u = zeros(n), v = zeros(n)),
        cell,
        nothing,
        FerriteOperators.TimeIntegrationContext(0.0, Δt, Δt),
    )

    # The declared facetset is the traversal, so a hand-driven call walks the facets of "right" that
    # this cell owns.
    right = getfacetset(Ferrite.get_grid(dh), "right")
    function damping_block(slope)
        K = zeros(n, n)
        for lfi = 1:nfacets(cell)
            if FacetIndex(cellid(cell), lfi) ∈ right
                FerriteOperators.assemble_facet!(
                    FerriteOperators.WeightedJacobianRequest(K, (u = 1.0, v = slope)),
                    cache,
                    args,
                    lfi,
                )
            end
        end
        return K
    end

    K_newmark        = damping_block(newmark_slope)
    K_backward_euler = damping_block(backward_euler_slope)
    @test !iszero(K_newmark)
    # The block is linear in the slope, so handing the facet the wrong scheme's quotient scales the
    # whole damping tangent -- for these coefficients, by a factor of two.
    @test K_newmark ≈ (newmark_slope / backward_euler_slope) .* K_backward_euler
    @test K_newmark ≉ K_backward_euler
end

@testset "A dashpot damps under Newmark" begin
    # End to end on the second order scheme, at fixed `γ` so that only the boundary condition differs.
    # The boundary condition names its field explicitly, which is the case the `dof_range` indexing
    # exists for: this handler carries `:d` and `:v` in one `SubDofHandler`, so an offset error would
    # write the traction into the velocity block instead of the displacement block.
    bare = elastodynamic_bar()
    damped = elastodynamic_bar(facet_models = (ViscousRobinBC(1.0e4, "right", :d),))
    kick(f) = translation_velocity(f, Vec((0.3, 0.0, 0.0)))

    undamped_integrator = solve_elastodynamic(bare, kick(bare), 0.05, 0.05)
    damped_integrator   = solve_elastodynamic(damped, kick(damped), 0.05, 0.05)
    @test undamped_integrator.sol.retcode == SciMLBase.ReturnCode.Success
    @test damped_integrator.sol.retcode == SciMLBase.ReturnCode.Success

    d = solution_indices_of_displacement(bare)
    @test norm(damped_integrator.u[d]) < norm(undamped_integrator.u[d])

    # How the traction scales with the viscosity is pinned exactly in `test_elements.jl`. It is not
    # asserted here: damping one end of the bar changes which mode it deforms in, so the displacement
    # norm is not monotone in the coefficient and any inequality would be reading the mode change.
end

@testset "Newmark with a plain Newton" begin
    # `NewmarkSolver` chooses its stage solver cache through the same `setup_stage_nlsolver_cache` as
    # backward Euler, so a function that condenses nothing is solvable by the plain Newton here too.
    f = elastodynamic_bar()
    @test ndofs(f.structural.lvh) == 0

    u0 = zeros(solution_size(f))
    Thunderbolt.default_initial_condition!(u0, f)
    integrator = init(
        ElastodynamicsProblem(f, u0, translation_velocity(f, Vec((0.3, 0.0, 0.0))), (0.0, 0.1)),
        NewmarkSolver(;
            β = 1 / 4,
            γ = 1 / 2,
            inner_solver = NewtonRaphsonSolver(
                inner_solver = UMFPACKFactorization(),
                max_iter = 10,
                tol = 1e-8,
            ),
        ),
        dt = 0.05,
        adaptive = false,
        verbose = false,
    )
    solve!(integrator)
    @test integrator.sol.retcode == SciMLBase.ReturnCode.Success

    # A condensing material does pose local problems, and the plain Newton has no local solver for
    # them, so the pairing is refused at setup rather than solving a system of the wrong size.
    condensing = elastodynamic_bar(
        material = Thunderbolt.LinearMaxwellMaterial(
            E₀ = 70e3,
            E₁ = 20e3,
            μ = 1e3,
            η₁ = 1e3,
            ν = 0.3,
        ),
    )
    @test ndofs(condensing.structural.lvh) > 0
    u0c = zeros(solution_size(condensing))
    Thunderbolt.default_initial_condition!(u0c, condensing)
    @test_throws "MultiLevelNewtonRaphsonSolver" init(
        ElastodynamicsProblem(
            condensing,
            u0c,
            translation_velocity(condensing, Vec((0.3, 0.0, 0.0))),
            (0.0, 0.1),
        ),
        NewmarkSolver(;
            β = 1 / 4,
            γ = 1 / 2,
            inner_solver = NewtonRaphsonSolver(inner_solver = UMFPACKFactorization()),
        ),
        dt = 0.05,
        adaptive = false,
        verbose = false,
    )
end
