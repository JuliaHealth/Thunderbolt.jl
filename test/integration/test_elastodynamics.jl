using Thunderbolt
import SciMLBase
using Test
using LinearAlgebra
using LinearSolve

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
    v0 = zeros(ndofs(f.dh))
    for i = 1:3:ndofs(f.dh)
        v0[i], v0[i+1], v0[i+2] = dir[1], dir[2], dir[3]
    end
    return v0
end

# Transverse velocity, growing along the bar so that the free end moves fastest.
function bending_velocity(f, amplitude)
    v0 = zeros(ndofs(f.dh))
    grid = Ferrite.get_grid(f.dh)
    for cell in CellIterator(f.dh)
        for (i, node) in enumerate(getcoordinates(cell))
            dofs = celldofs(cell)[(3(i-1)+1):(3i)]
            v0[dofs[2]] = amplitude * node[1]
        end
    end
    Ferrite.apply_zero!(v0, Thunderbolt.getch(f))
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
        @test integrator.u[1:ndofs(f.dh)]≈tend .* v0 rtol=1e-7
        @test Thunderbolt.velocity(integrator)≈v0 rtol=1e-7
        @test norm(Thunderbolt.acceleration(integrator)) < 1e-6
    end

    @testset "Equilibrium at rest stays at rest" begin
        f = elastodynamic_bar()
        integrator = solve_elastodynamic(f, zeros(ndofs(f.dh)), 0.1, 0.02)

        @test integrator.sol.retcode == SciMLBase.ReturnCode.Success
        @test norm(integrator.u) < 1e-10
        @test norm(Thunderbolt.velocity(integrator)) < 1e-10
    end

    # The first convergence order test in the suite. Newmark is second order for γ = 1/2 and drops to
    # first order for any other γ, which is what makes the comparison worth having: it pins that γ
    # reaches the scheme at all, rather than only that the scheme converges.
    @testset "Convergence order" begin
        tend = 0.02
        Δt₀ = tend / 4

        function observed_order(γ)
            run(Δt) = solve_elastodynamic(
                (f = elastodynamic_bar(); f),
                bending_velocity(elastodynamic_bar(), 20.0),
                tend,
                Δt;
                γ,
            ).u
            reference = copy(run(Δt₀ / 16))
            errors = [norm(run(Δt₀ / refinement) - reference) for refinement in (1, 2, 4)]
            @test all(>(0), errors)
            return log2(errors[end-1] / errors[end])
        end

        order_conserving = observed_order(1 / 2)
        order_dissipative = observed_order(0.6)

        @test order_conserving≈2.0 atol=0.3
        # Only a bound: away from γ = 1/2 the first order term dominates asymptotically, but at a step
        # size coarse enough to run in a test the measured rate still sits above one.
        @test order_dissipative < order_conserving - 0.3
    end

    @testset "Numerical dissipation follows γ" begin
        # γ = 1/2 conserves, γ > 1/2 dissipates. Measured as the decay of the swing amplitude over
        # three periods of free vibration, so that no strain energy functional has to be
        # reconstructed. The light density is what puts three periods inside a test-sized run.
        tend = 2.2
        decay = map((1 / 2, 0.7)) do γ
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
                amplitude = norm(integrator.u[1:ndofs(f.dh)], Inf)
                integrator.t < tend / 3 && (first_swing = max(first_swing, amplitude))
                integrator.t > 2tend / 3 && (last_swing = max(last_swing, amplitude))
            end
            @test integrator.sol.retcode == SciMLBase.ReturnCode.Success
            return last_swing / first_swing
        end

        @test decay[1]≈1.0 atol=0.05          # average acceleration: no secular decay
        @test decay[2] < decay[1] - 0.05      # γ > 1/2: visibly dissipative
    end

    # Zienkiewicz-Xie estimate plus the elementary controller. The sharpest statement available from
    # outside is the step count: a controller using the right order drives `Δt ∝ tol^(1/3)`, so the
    # number of steps must grow by `10^(1/3) ≈ 2.15` per decade of tolerance. Getting that exponent
    # right is exactly what "the estimator is O(Δt³)" means in practice.
    @testset "Adaptivity controls the error" begin
        tend = 0.5   # dyadic, so the fixed step reference lands on it without a closing micro-step
        bar() = elastodynamic_bar(ncells = (2, 1, 1), ρ = 1.0e-2)

        reference = solve_elastodynamic(bar(), bending_velocity(bar(), 0.2), tend, tend / 2^13)
        @test reference.sol.retcode == SciMLBase.ReturnCode.Success

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
            return (
                steps = integrator.stats.naccept,
                err = norm(integrator.u - reference.u) / norm(reference.u),
            )
        end

        # Step count follows tol^(-1/3) ...
        for i = 1:(length(results)-1)
            ratio = results[i+1].steps / results[i].steps
            @test ratio≈10^(1 / 3) rtol=0.25
        end
        # ... and the solution gets closer to the reference.
        @test results[end].err < results[1].err
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
            ElastodynamicsProblem(f, u0, zeros(ndofs(f.dh)), (0.0, 20.0)),
            NewmarkSolver(),
            dt = 0.05,
            adaptive = true,
            dtmax = 5.0,
            verbose = false,
        )
        dts = Float64[]
        while integrator.t < 20.0 - 1.0e-10
            step!(integrator)
            push!(dts, integrator.dt)
            integrator.sol.retcode == SciMLBase.ReturnCode.Success || break
        end
        @test integrator.sol.retcode == SciMLBase.ReturnCode.Success
        # The step grows by more than an order of magnitude once the transient is over.
        @test maximum(dts) / minimum(dts) > 10
    end

    @testset "The step size controller is Thunderbolt's own" begin
        # The default has to be the in-package `PIDController`, not one reached through
        # `OrdinaryDiffEqCore`'s controller protocol -- that protocol moves between versions, and
        # pinning the default to it is what this port exists to avoid.
        f = elastodynamic_bar(ncells = (2, 1, 1))
        integrator = init(
            ElastodynamicsProblem(f, (0.0, 1.0)),
            NewmarkSolver(),
            dt = 0.1,
            adaptive = true,
            verbose = false,
        )
        @test integrator.controller_cache isa Thunderbolt.PIDControllerCache

        # The exponent is `1/(adaptive_order+1)`, and getting it wrong is the failure that a step
        # count study would catch only indirectly.
        @test Thunderbolt.adaptive_order(NewmarkSolver()) == 2
    end

    @testset "A failed solve shrinks dt once, not twice" begin
        # `post_newton_controller!` applies the failure factor in the step footer, and the controller's
        # reject hook runs in the next header. Both used to divide, shrinking `dt` by `failfactor²` per
        # failed attempt — which reaches the floor in a handful of steps and looks like a diverging
        # solve rather than a bookkeeping mistake.
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
        try
            step!(integrator)
        catch
            # the solve gives up eventually; what is asserted is how far `dt` fell on the way
        end
        @test integrator.stats.nreject > 1
        @test dt₀ / integrator.dt ≤ integrator.opts.failfactor^integrator.stats.nreject
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
        u, v, a = copy(integrator.u),
        copy(Thunderbolt.velocity(integrator)),
        copy(Thunderbolt.acceleration(integrator))

        step!(integrator)
        # The step has to move all three, otherwise the rollback below asserts nothing.
        @test !isapprox(integrator.u, u)
        @test !isapprox(Thunderbolt.velocity(integrator), v)
        @test !isapprox(Thunderbolt.acceleration(integrator), a)

        Thunderbolt.reject_step!(integrator)
        @test integrator.u == u
        @test Thunderbolt.velocity(integrator) == v
        @test Thunderbolt.acceleration(integrator) == a
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
            return solve_elastodynamic(f, zeros(ndofs(f.dh)), 20.0, 1.0)
        end

        integrator = contract(Thunderbolt.RDQ20MFModel())
        nfe = ndofs(integrator.sol.prob.f.dh)
        @test integrator.sol.retcode == SciMLBase.ReturnCode.Success
        @test all(isfinite, integrator.u)
        # The sarcomere activates at Ca = 1 and pulls the bar in.
        @test maximum(integrator.u[(nfe+1):end]) > 0.1
        @test norm(integrator.u[1:nfe]) > 0.1

        # Dropping the velocity coupling has to change the answer, and by a lot: on a contracting bar
        # the rate term is a double digit fraction of the tangent. If the element fed the material no
        # rate at all, the two would agree exactly — they differ in nothing else.
        wrapped = contract(Thunderbolt.AsRateIndependent(Thunderbolt.RDQ20MFModel()))
        @test wrapped.sol.retcode == SciMLBase.ReturnCode.Success
        @test norm(wrapped.u[1:nfe] - integrator.u[1:nfe]) / norm(integrator.u[1:nfe]) > 0.05
    end
end
