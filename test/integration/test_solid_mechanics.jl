using Thunderbolt
import DiffEqBase
import SciMLBase
import SciMLIterators: intervals
using Test
using Logging
using LinearSolve
using OrderedCollections
include(joinpath(@__DIR__, "..", "testfixtures.jl"))

"""
Directory for the per-Newton-iteration VTK dumps of `VTKNewtonMonitor`. A fresh temporary directory
by default, so the writes cannot collide between parallel workers and do not depend on the cwd. Set
`THUNDERBOLT_TEST_KEEP_VTK=/some/path` to keep them somewhere durable when debugging a solve.
"""
newton_debug_dir() = get(ENV, "THUNDERBOLT_TEST_KEEP_VTK") do
    mktempdir()
end

function test_solve_passive_structure(mesh, models)
    tspan = (0.0, 1.0)
    Δt = 1.0

    # Clamp three sides
    dbcs = [
        Dirichlet(:d, getfacetset(mesh, "left"), (x, t) -> [0.0], [1])
        Dirichlet(:d, getfacetset(mesh, "front"), (x, t) -> [0.0], [2])
        Dirichlet(:d, getfacetset(mesh, "bottom"), (x, t) -> [0.0], [3])
        Dirichlet(:d, Set([1]), (x, t) -> [0.0, 0.0, 0.0], [1, 2, 3])
        Dirichlet(:d, getfacetset(mesh, "right"), (x, t) -> [0.01t], [1])
        Dirichlet(:d, getfacetset(mesh, "top"), (x, t) -> [0.02t], [2])
        Dirichlet(:d, getfacetset(mesh, "back"), (x, t) -> [0.03t], [3])
    ]

    quasistaticform = semidiscretize(
        models,
        FiniteElementDiscretization(Dict(:d => LagrangeCollection{1}()^3); dbcs),
        mesh,
    )

    problem = QuasiStaticProblem(quasistaticform, tspan)

    # Create sparse matrix and residual vector
    timestepper = HomotopyPathSolver(
        NewtonRaphsonSolver(;
            max_iter = 10,
            monitor = Thunderbolt.VTKNewtonMonitor(joinpath(newton_debug_dir(), "newton-debug")),
        ),
    )
    integrator = init(problem, timestepper, dt = Δt, verbose = true)
    u₀ = copy(integrator.u)
    solve!(integrator)
    @test integrator.sol.retcode == DiffEqBase.ReturnCode.Success
    @test integrator.u ≉ u₀
    return integrator.u
end

@testset "Passive Structure" begin

    grid = generate_grid(
        Hexahedron,
        (10, 10, 2),
        Ferrite.Vec{3}((-1.0, -1.0, -0.2)),
        Ferrite.Vec{3}((1.0, 1.0, 0.2)),
    )
    addcellset!(grid, "myocardium", x->true)
    # addcellset!(grid, "inner", x->x[3] ≤ 0.0)
    # addcellset!(grid, "outer", x->x[3] ≥ 0.0)
    mesh = to_mesh(grid)

    ortho_ms = ConstantCoefficient(
        OrthotropicMicrostructure(Vec((1.0, 0.0, 0.0)), Vec((0.0, 1.0, 0.0)), Vec((0.0, 0.0, 1.0))),
    )
    u₁ = test_solve_passive_structure(
        mesh,
        QuasiStaticModel(:d, PK1Model(HolzapfelOgden2009Model(), ortho_ms)),
    )

    u₂ = test_solve_passive_structure(
        mesh,
        QuasiStaticModel(
            :d,
            PrestressedMechanicalModel(
                PK1Model(HolzapfelOgden2009Model(), ortho_ms),
                ConstantCoefficient(Tensor{2, 3}((1.1, 0.1, 0.0, 0.2, 0.9, 0.1, -0.1, 0.0, 1.0))),
            ),
        ),
    )

    grid2 = generate_grid(
        Hexahedron,
        (10, 10, 2),
        Ferrite.Vec{3}((-1.0, -1.0, -0.2)),
        Ferrite.Vec{3}((1.0, 1.0, 0.2)),
    )
    addcellset!(grid2, "myocardium", x->true)
    # NOTE: the nodes on the nominal z=0 plane carry floating point noise of O(1e-17) from the grid
    # generator, so a bare `x[3] ≤ 0.0` predicate (which Ferrite evaluates on *all* nodes of a cell)
    # picks up only those cells whose interface nodes happen to round the right way. Use a tolerance
    # for one half and take the complement for the other, so that the two really do partition grid2.
    addcellset!(grid2, "inner", x->x[3] ≤ 1.0e-8)
    addcellset!(grid2, "outer", setdiff(OrderedSet(1:getncells(grid2)), getcellset(grid2, "inner")))
    @assert length(getcellset(grid2, "inner")) + length(getcellset(grid2, "outer")) ==
            getncells(grid2)
    mesh2 = to_mesh(grid2)

    # The prestress should force a different solution
    @test u₁ ≉ u₂

    u₃ = test_solve_passive_structure(
        mesh2,
        Dict(
            "inner" => QuasiStaticModel(:d, PK1Model(HolzapfelOgden2009Model(), ortho_ms)),
            "outer" => QuasiStaticModel(:d, PK1Model(Guccione1991PassiveModel(), ortho_ms)),
        ),
    )

    @test u₃ ≉ u₁

    u₄ = test_solve_passive_structure(
        mesh2,
        Dict(
            "inner" => QuasiStaticModel(:d, PK1Model(HolzapfelOgden2009Model(), ortho_ms)),
            "outer" => QuasiStaticModel(:d, PK1Model(HolzapfelOgden2009Model(), ortho_ms)),
        ),
    )

    @test u₄ ≉ u₃
    @test sort(u₄) ≈ sort(u₁)

    u₅ = test_solve_passive_structure(
        mesh2,
        Dict("myocardium" => QuasiStaticModel(:d, PK1Model(HolzapfelOgden2009Model(), ortho_ms))),
    )

    @test sort(u₅) ≈ sort(u₁)
end

# Counts Newton iterations through the documented monitor hook, so a test can observe how an
# iteration behaved without reaching into solver caches.
mutable struct CountingNewtonMonitor
    steps::Int
end
CountingNewtonMonitor() = CountingNewtonMonitor(0)
Thunderbolt.nonlinear_step_monitor(cache, t, f, u, m::CountingNewtonMonitor) = (m.steps += 1)
Thunderbolt.nonlinear_finalize_monitor(cache, t, f, m::CountingNewtonMonitor) = nothing

struct TestCalciumHatField end
Thunderbolt.setup_coefficient_cache(coeff::TestCalciumHatField, ::QuadratureRule, ::SubDofHandler) =
    coeff
function Thunderbolt.evaluate_coefficient(
    coeff::TestCalciumHatField,
    cell_cache::CellCache,
    qp::QuadraturePoint,
    t,
)
    Ca = t/1000.0 < 0.5 ? 2.0*t/1000.0 : 2.0-2.0*t/1000.0
    return Ca
end
# Time dependent scalar field, used to check that coefficient evaluation on a subdomain actually
# receives the *time* rather than the time integrator's parameter object.
struct TestRampField end
Thunderbolt.setup_coefficient_cache(coeff::TestRampField, ::QuadratureRule, ::SubDofHandler) = coeff
Thunderbolt.evaluate_coefficient(::TestRampField, ::CellCache, ::QuadraturePoint, t) = 0.01 * t

struct TestCalciumQuadraticHatField end
Thunderbolt.setup_coefficient_cache(
    coeff::TestCalciumQuadraticHatField,
    ::QuadratureRule,
    ::SubDofHandler,
) = coeff
Thunderbolt.evaluate_coefficient(
    coeff::TestCalciumQuadraticHatField,
    cell_cache::CellCache,
    qp::QuadraturePoint,
    t,
) = t/1000.0 < 0.5 ? (2.0*t/1000.0)^2 : 2.0-(2.0*t/1000.0)^2

function test_solve_contractile_cuboid(mesh, model, timestepper)
    integrator, u₀ = solve_contractile_cuboid(mesh, model, timestepper)
    @test integrator.sol.retcode == DiffEqBase.ReturnCode.Success
    @test integrator.u ≉ u₀
    return integrator
end

# Assertion-free variant, so tests documenting a *known broken* configuration can wrap the whole
# solve in `@test_broken` without the inner assertions firing on the way.
function solve_contractile_cuboid(mesh, model, timestepper)
    tspan = timestepper isa BackwardEulerSolver ? (0.0, 2.0) : (0.0, 300.0)
    Δt = timestepper isa BackwardEulerSolver ? 0.25 : 100.0

    # Clamp three sides
    dbcs = [
        Dirichlet(:d, getfacetset(mesh, "left"), (x, t) -> [0.0], [1])
        Dirichlet(:d, getfacetset(mesh, "front"), (x, t) -> [0.0], [2])
        Dirichlet(:d, getfacetset(mesh, "bottom"), (x, t) -> [0.0], [3])
        Dirichlet(:d, Set([1]), (x, t) -> [0.0, 0.0, 0.0], [1, 2, 3])
    ]

    quasistaticform = semidiscretize(
        model,
        FiniteElementDiscretization(Dict(:d => LagrangeCollection{1}()^3); dbcs),
        mesh,
    )

    problem = QuasiStaticProblem(quasistaticform, tspan)
    Thunderbolt.default_initial_condition!(problem.u0, problem.f)

    # Create sparse matrix and residual vector
    integrator = init(
        problem,
        timestepper,
        dt = Δt,
        verbose = true,
        adaptive = !(timestepper isa BackwardEulerSolver),
    )
    u₀ = copy(integrator.u)
    solve!(integrator)

    return integrator, u₀
end

function test_solve_contractile_ideal_lv(
    mesh,
    constitutive_model,
    tmax,
    Δt = 100.0,
    adaptive = true,
)
    tspan = (0.0, tmax)

    # Clamp three sides
    dbcs = [
        Dirichlet(:d, getnodeset(mesh, "MyocardialAnchor1"), (x, t) -> (0.0, 0.0, 0.0), [1, 2, 3]),
        Dirichlet(:d, getnodeset(mesh, "MyocardialAnchor2"), (x, t) -> (0.0, 0.0), [2, 3]),
        Dirichlet(:d, getnodeset(mesh, "MyocardialAnchor3"), (x, t) -> (0.0,), [3]),
        Dirichlet(:d, getnodeset(mesh, "MyocardialAnchor4"), (x, t) -> (0.0,), [3]),
    ]

    quasistaticform = semidiscretize(
        QuasiStaticModel(
            :d,
            constitutive_model,
            (
                RobinBC(0.1, "Epicardium"),
                NormalSpringBC(1.0, "Base"),
                PressureFieldBC(ConstantCoefficient(0.01), "Endocardium"),
            ),
        ),
        FiniteElementDiscretization(Dict(:d => LagrangeCollection{1}()^3); dbcs),
        mesh,
    )

    problem = QuasiStaticProblem(quasistaticform, tspan)

    # Create sparse matrix and residual vector
    timestepper = HomotopyPathSolver(
        NewtonRaphsonSolver(inner_solver = UMFPACKFactorization(), max_iter = 10, tol = 1e-10),
    )
    integrator =
        init(problem, timestepper, dt = Δt, verbose = true, adaptive = adaptive, maxiters = 50)
    u₀ = copy(integrator.u)
    solve!(integrator)
    @test integrator.sol.retcode == DiffEqBase.ReturnCode.Success
    @test integrator.u ≉ u₀

    return integrator
end

# Smoke tests that things do not crash and that things do at least something
@testset "Contracting cuboid" begin
    # mesh = generate_mesh(Hexahedron, (10, 10, 2), Ferrite.Vec{3}((0.0,0.0,0.0)), Ferrite.Vec{3}((1.0, 1.0, 0.2)))
    # mesh = generate_mesh(Hexahedron, (1, 1, 1), Ferrite.Vec{3}((0.0,0.0,0.0)), Ferrite.Vec{3}((1.0, 1.0, 0.2)))

    microstructure_model = OrthotropicMicrostructureModel(
        ConstantCoefficient(Vec((1.0, 0.0, 0.0))),
        ConstantCoefficient(Vec((0.0, 1.0, 0.0))),
        ConstantCoefficient(Vec((0.0, 0.0, 1.0))),
    )

    newton = NewtonRaphsonSolver(inner_solver = UMFPACKFactorization(), max_iter = 10, tol = 1e-10)

    facemodels = (
        NormalSpringBC(0.0, "right"),
        ConstantPressureBC(0.0, "back"),
        PressureFieldBC(ConstantCoefficient(0.0), "top"),
    )

    @testset "Single Subdomain" begin
        grid = generate_grid(
            Hexahedron,
            (10, 10, 2),
            Ferrite.Vec{3}((0.0, 0.0, 0.0)),
            Ferrite.Vec{3}((1.0, 1.0, 0.2)),
        )
        addcellset!(grid, "myocardium", x->true)
        mesh = to_mesh(grid)

        timestepper = HomotopyPathSolver(newton)
        test_solve_contractile_cuboid(
            mesh,
            QuasiStaticModel(
                :d,
                ExtendedHillModel(
                    HolzapfelOgden2009Model(),
                    ActiveMaterialAdapter(LinearSpringModel()),
                    GMKActiveDeformationGradientModel(),
                    Thunderbolt.CaDrivenInternalSarcomereModel(
                        PelceSunLangeveld1995Model(),
                        TestCalciumHatField(),
                    ),
                    microstructure_model,
                ),
                facemodels,
            ),
            timestepper,
        )

        test_solve_contractile_cuboid(
            mesh,
            QuasiStaticModel(
                :d,
                GeneralizedHillModel(
                    LinYinPassiveModel(),
                    ActiveMaterialAdapter(LinYinActiveModel()),
                    GMKIncompressibleActiveDeformationGradientModel(),
                    Thunderbolt.CaDrivenInternalSarcomereModel(
                        PelceSunLangeveld1995Model(),
                        TestCalciumHatField(),
                    ),
                    microstructure_model,
                ),
                facemodels,
            ),
            timestepper,
        )

        i = test_solve_contractile_cuboid(
            mesh,
            QuasiStaticModel(
                :d,
                ActiveStressModel(
                    HumphreyStrumpfYinModel(),
                    SimpleActiveStress(),
                    Thunderbolt.CaDrivenInternalSarcomereModel(
                        PelceSunLangeveld1995Model(),
                        TestCalciumHatField(),
                    ),
                    microstructure_model,
                ),
                facemodels,
            ),
            timestepper,
        )
        # VTKGridFile("SolidMechanicsIntegrationDebug", i.f.dh.grid) do vtk
        #     write_solution(vtk, i.f.dh, i.u)
        # end
    end

    @testset "Multiple subdomains" begin
        grid = generate_grid(
            Hexahedron,
            (10, 10, 2),
            Ferrite.Vec{3}((0.0, 0.0, 0.0)),
            Ferrite.Vec{3}((1.0, 1.0, 0.2)),
        )
        addcellset!(grid, "myocardium", x->true)
        addcellset!(grid, "inner", x->x[3] ≤ 0.1)
        addcellset!(grid, "outer", x->x[3] ≥ 0.1)
        addcellset!(grid, "front", x->x[1] ≤ 0.1)
        addcellset!(grid, "back", x->x[1] ≥ 0.1)
        mesh = to_mesh(grid)

        timestepper = BackwardEulerSolver(;
            inner_solver = Thunderbolt.MultiLevelNewtonRaphsonSolver(; newton = newton),
        )

        i = test_solve_contractile_cuboid(
            mesh,
            Dict(
                "front" => QuasiStaticModel(
                    :d,
                    ActiveStressModel(
                        Guccione1991PassiveModel(),
                        SimpleActiveStress(; Tmax = 220e3),
                        Thunderbolt.CaDrivenInternalSarcomereModel(
                            Thunderbolt.RDQ20MFModel(),
                            TestCalciumHatField(),
                        ),
                        microstructure_model,
                    ),
                    facemodels,
                ),
                "back" => QuasiStaticModel(
                    :d,
                    PK1Model(Guccione1991PassiveModel(), microstructure_model),
                    facemodels,
                ),
            ),
            timestepper,
        )
        # VTKGridFile(
        #     "SolidMechanicsIntegrationDebug",
        #     i.f.dh.grid,
        # ) do vtk
        #     write_solution(vtk, i.f.dh, i.u)
        # end

        test_solve_contractile_cuboid(
            mesh,
            Dict(
                "front" => QuasiStaticModel(
                    :d,
                    PK1Model(Guccione1991PassiveModel(), microstructure_model),
                    facemodels,
                ),
                # `AsRateIndependent` routes the sarcomere onto the condensed *ODE* element cache,
                # where the unwrapped model above uses the DAE one. The two no longer agree: the DAE
                # path feeds `dλdt = dλdF ⊡ Ḟ` into the local solve, the wrapped one drops it.
                "back" => QuasiStaticModel(
                    :d,
                    ActiveStressModel(
                        Guccione1991PassiveModel(),
                        SimpleActiveStress(; Tmax = 220e3),
                        Thunderbolt.CaDrivenInternalSarcomereModel(
                            Thunderbolt.AsRateIndependent(Thunderbolt.RDQ20MFModel()),
                            TestCalciumHatField(),
                        ),
                        microstructure_model,
                    ),
                    facemodels,
                ),
            ),
            timestepper,
        )

        # A rate-free (`NoEvolution`) subdomain still has to be assembled by the time integrator when
        # it sits next to a subdomain that carries an internal variable. Below, the rate-free
        # subdomain drives its sarcomere from a time dependent calcium field, which is the ordinary
        # cardiac case.
        #
        # A time dependent coefficient on a rate-free subdomain is legitimate — only a dependence on
        # the *rate* is not — and the time reaches it correctly: `QuasiStaticElementCache` lowers the
        # `gto1` parameters to their element local form and the assembly queries `get_time`. The
        # companion facet test below exercises that same path and passes.
        #
        # This one fails in the *solve* rather than in the plumbing, and is reported to be flaky, so
        # the marker stays until the mechanism is understood. Δt = 0.25 and Δt = 0.02 fail alike, so
        # shortening the step is not the answer.
        @testset "Time dependent coefficient on a rate-free subdomain" begin
            @test_broken (
                solve_contractile_cuboid(
                    mesh,
                    Dict(
                        # `PelceSunLangeveld1995Model` is a steady state model -> `NoEvolution`
                        "front" => QuasiStaticModel(
                            :d,
                            ActiveStressModel(
                                Guccione1991PassiveModel(),
                                SimpleActiveStress(; Tmax = 220e3),
                                Thunderbolt.CaDrivenInternalSarcomereModel(
                                    PelceSunLangeveld1995Model(),
                                    TestCalciumHatField(),
                                ),
                                microstructure_model,
                            ),
                            facemodels,
                        ),
                        # ... next to a subdomain that does carry an internal variable, so the
                        # problem genuinely needs the `gto1` protocol.
                        "back" => QuasiStaticModel(
                            :d,
                            ActiveStressModel(
                                Guccione1991PassiveModel(),
                                SimpleActiveStress(; Tmax = 220e3),
                                Thunderbolt.CaDrivenInternalSarcomereModel(
                                    Thunderbolt.AsRateIndependent(Thunderbolt.RDQ20MFModel()),
                                    TestCalciumHatField(),
                                ),
                                microstructure_model,
                            ),
                            facemodels,
                        ),
                    ),
                    timestepper,
                )[1].sol.retcode == DiffEqBase.ReturnCode.Success
            )
        end

        # The facet path reaches the time the same way a cell kernel does: `PressureFieldBC` reads
        # `evaluation_time(args.ctx)` and hands *that* to `evaluate_coefficient`, rather than anything
        # out of `args.p`, which carries configuration only. This pins a time dependent facet
        # coefficient on a subdomain whose element is rate-free.
        let facemodels_tdep = (
                NormalSpringBC(0.0, "right"),
                ConstantPressureBC(0.0, "back"),
                PressureFieldBC(TestRampField(), "top"),
            )
            @testset "Time dependent facet coefficient on a rate-free subdomain" begin
                @test (
                    solve_contractile_cuboid(
                        mesh,
                        Dict(
                            "front" => QuasiStaticModel(
                                :d,
                                PK1Model(Guccione1991PassiveModel(), microstructure_model),
                                facemodels_tdep,
                            ),
                            "back" => QuasiStaticModel(
                                :d,
                                PK1Model(Guccione1991PassiveModel(), microstructure_model),
                                facemodels,
                            ),
                        ),
                        timestepper,
                    )[1].sol.retcode == DiffEqBase.ReturnCode.Success
                )
            end

            # ... whereas on a subdomain that does go through `gto1` the unwrapping methods do fire.
            # Nothing else covers a time dependent facet coefficient, so this pins them down.
            @testset "Time dependent facet coefficient on a gto1 subdomain" begin
                test_solve_contractile_cuboid(
                    mesh,
                    Dict(
                        "front" => QuasiStaticModel(
                            :d,
                            PK1Model(Guccione1991PassiveModel(), microstructure_model),
                            facemodels,
                        ),
                        "back" => QuasiStaticModel(
                            :d,
                            ActiveStressModel(
                                Guccione1991PassiveModel(),
                                SimpleActiveStress(; Tmax = 220e3),
                                Thunderbolt.CaDrivenInternalSarcomereModel(
                                    Thunderbolt.AsRateIndependent(Thunderbolt.RDQ20MFModel()),
                                    TestCalciumHatField(),
                                ),
                                microstructure_model,
                            ),
                            facemodels_tdep,
                        ),
                    ),
                    timestepper,
                )
            end
        end

        # Regression: the boundary hooks of `NonlinearMultiDomainIntegrator2` used to look the
        # subdomain name up in the *surface* subdomains, which is a different namespace from the
        # volumetric one its subintegrators are keyed by. They therefore claimed nothing and silently
        # dropped every weak boundary condition.
        #
        # The testsets above do not catch it: `generate_grid` names its facetsets "front"/"back", so
        # the cellset names they use collide with facetset names and accidentally match. Here the
        # subdomains are "inner"/"outer", which no facetset is called, and the *only* load is a weak
        # boundary condition — so if it is dropped, the body simply never deforms.
        @testset "Weak boundary conditions on subdomains without a matching facetset" begin
            # Ramped from zero, so the initial state stays consistent for the homotopy solver.
            pressure_load = (PressureFieldBC(TestRampField(), "top"),)
            i = test_solve_contractile_cuboid(
                mesh,
                Dict(
                    "inner" => QuasiStaticModel(
                        :d,
                        PK1Model(Guccione1991PassiveModel(), microstructure_model),
                        pressure_load,
                    ),
                    "outer" => QuasiStaticModel(
                        :d,
                        PK1Model(Guccione1991PassiveModel(), microstructure_model),
                        pressure_load,
                    ),
                ),
                HomotopyPathSolver(newton),
            )
            @test norm(i.u) > 1.0e-8
        end

        mesh = to_mesh(generate_mixed_dimensional_grid_3D())

        timestepper = HomotopyPathSolver(newton)

        test_solve_contractile_cuboid(
            mesh,
            Dict(
                "Ventricle" => QuasiStaticModel(
                    :d,
                    ActiveStressModel(
                        HumphreyStrumpfYinModel(),
                        SimpleActiveStress(),
                        Thunderbolt.CaDrivenInternalSarcomereModel(
                            PelceSunLangeveld1995Model(),
                            TestCalciumHatField(),
                        ),
                        microstructure_model,
                    ),
                    facemodels,
                ),
            ),
            timestepper,
        )
    end
end

@testset "Idealized LV" begin
    grid = generate_ideal_lv_mesh(4, 1, 1)
    cs = compute_lv_coordinate_system(grid)
    @test !any(isnan.(cs.u_apicobasal))
    @test !any(isnan.(cs.u_transmural))
    @test !any(isnan.(cs.u_rotational))
    microstructure_parameters = ODB25LTMicrostructureParameters(αendo = deg2rad(80.0), αepi = deg2rad(-65.0))
    microstructure_model      = create_microstructure_model(cs, LagrangeCollection{1}()^3, microstructure_parameters)

    test_solve_contractile_ideal_lv(
        grid,
        ExtendedHillModel(
            HolzapfelOgden2009Model(),
            ActiveMaterialAdapter(LinearSpringModel()),
            GMKActiveDeformationGradientModel(),
            Thunderbolt.CaDrivenInternalSarcomereModel(
                PelceSunLangeveld1995Model(),
                TestCalciumHatField(),
            ),
            microstructure_model,
        ),
        300.0,
    )

    test_solve_contractile_ideal_lv(
        grid,
        GeneralizedHillModel(
            LinYinPassiveModel(),
            ActiveMaterialAdapter(LinYinActiveModel()),
            GMKIncompressibleActiveDeformationGradientModel(),
            Thunderbolt.CaDrivenInternalSarcomereModel(
                PelceSunLangeveld1995Model(),
                TestCalciumHatField(),
            ),
            microstructure_model,
        ),
        300.0,
    )

    @testset "Adaptivity does not change the result" begin
        i1 = test_solve_contractile_ideal_lv(
            grid,
            ActiveStressModel(
                HumphreyStrumpfYinModel(),
                SimpleActiveStress(),
                Thunderbolt.CaDrivenInternalSarcomereModel(
                    PelceSunLangeveld1995Model(),
                    TestCalciumQuadraticHatField(),
                ),
                microstructure_model,
            ),
            10.0,
            1.0,
            true,
        )

        i2 = test_solve_contractile_ideal_lv(
            grid,
            ActiveStressModel(
                HumphreyStrumpfYinModel(),
                SimpleActiveStress(),
                Thunderbolt.CaDrivenInternalSarcomereModel(
                    PelceSunLangeveld1995Model(),
                    TestCalciumQuadraticHatField(),
                ),
                microstructure_model,
            ),
            10.0,
            1.0,
            false,
        )

        # Test path-independence setup
        @test i1.t ≈ 10.0
        @test i2.t ≈ 10.0
        @test i1.u ≈ i2.u atol=1e-4
    end

    @testset "The load path is actually different" begin
        i1 = test_solve_contractile_ideal_lv(
            grid,
            ActiveStressModel(
                HumphreyStrumpfYinModel(),
                SimpleActiveStress(),
                Thunderbolt.CaDrivenInternalSarcomereModel(
                    PelceSunLangeveld1995Model(),
                    TestCalciumHatField(),
                ),
                microstructure_model,
            ),
            100.0,
        )

        i2 = test_solve_contractile_ideal_lv(
            grid,
            ActiveStressModel(
                HumphreyStrumpfYinModel(),
                SimpleActiveStress(),
                Thunderbolt.CaDrivenInternalSarcomereModel(
                    PelceSunLangeveld1995Model(),
                    TestCalciumQuadraticHatField(),
                ),
                microstructure_model,
            ),
            100.0,
        )

        @test i1.t ≈ 100.0
        @test i2.t ≈ 100.0
        @test !isapprox(i1.u, i2.u; atol = 1.0e-4)
    end

    # Check that the integrator reaches the final time and the solutions coincide
    @testset "Check path independence" begin
        i1 = test_solve_contractile_ideal_lv(
            grid,
            ActiveStressModel(
                HumphreyStrumpfYinModel(),
                SimpleActiveStress(),
                Thunderbolt.CaDrivenInternalSarcomereModel(
                    PelceSunLangeveld1995Model(),
                    TestCalciumHatField(),
                ),
                microstructure_model,
            ),
            500.0,
        )

        i2 = test_solve_contractile_ideal_lv(
            grid,
            ActiveStressModel(
                HumphreyStrumpfYinModel(),
                SimpleActiveStress(),
                Thunderbolt.CaDrivenInternalSarcomereModel(
                    PelceSunLangeveld1995Model(),
                    TestCalciumQuadraticHatField(),
                ),
                microstructure_model,
            ),
            500.0,
        )
        # Test path-independence
        @test i1.t ≈ 500.0
        @test i2.t ≈ 500.0
        @test i1.u ≈ i2.u atol=1e-4
    end
end

@testset "Viscoelasticity" begin
    mesh = generate_mesh(Hexahedron, (1, 1, 1))
    material = Thunderbolt.LinearMaxwellMaterial(E₀ = 70e3, E₁ = 20e3, μ = 1e3, η₁ = 1e3, ν = 0.3)
    tspan = (0.0, 1.0)
    Δt = 0.1

    # Clamp three sides
    dbcs = [
        Dirichlet(:d, getfacetset(mesh, "left"), (x, t) -> (0.0, 0.0, 0.0), [1, 2, 3]),
        Dirichlet(:d, getfacetset(mesh, "right"), (x, t) -> (0.1, 0.0, 0.0), [1, 2, 3]),
    ]

    quasistaticform = semidiscretize(
        QuasiStaticModel(:d, material, ()),
        FiniteElementDiscretization(
            Dict(:d => (LagrangeCollection{1}()^3 => QuadratureRuleCollection(1)));
            dbcs,
        ),
        mesh,
    )
    @test solution_size(quasistaticform) == 3 * 8 + 1 * 6 # Symmetric Tensor has 6 components
    problem = QuasiStaticProblem(quasistaticform, tspan)

    # Create sparse matrix and residual vector
    timestepper = BackwardEulerSolver(; inner_solver = Thunderbolt.MultiLevelNewtonRaphsonSolver(;
    # global_newton=NewtonRaphsonSolver(),
    # local_newton=NewtonRaphsonSolver(),
    ))
    integrator = init(problem, timestepper, dt = Δt, verbose = true)
    # This setup is essentially a creep test in x direction, so we check for the invariants in there
    for (uprev, tprev, u, t) in intervals(integrator)
        # Monotonicity of the solution in x direction
        @test uprev[3*8+1] ≤ u[3*8+1]
    end
    # Linear problem => check that Newton converges in 1 step.
    @test length(integrator.cache.stage.nlsolver.global_solver_cache.Θks) == 1
    @test integrator.sol.retcode == SciMLBase.ReturnCode.Success
    @test integrator.u[3*8+1] ≈ 0.05 atol=1e-5
    @test integrator.u[(3*8+2):end] ≈ zeros(5) atol=1e-5
end

@testset "Internal variables are stored per cell" begin
    # Regression test: `_query_local_state`/`_store_local_state!` used to index the *global*
    # internal variable block by quadrature point alone, without a per-cell offset, so every cell
    # read and wrote the first cell's slots. On a single cell mesh that is indistinguishable from
    # correct behaviour, which is why the smoke tests above never caught it.
    mesh = generate_mesh(Hexahedron, (2, 1, 1))
    material = Thunderbolt.LinearMaxwellMaterial(E₀ = 70e3, E₁ = 20e3, μ = 1e3, η₁ = 1e3, ν = 0.3)
    dbcs = [
        Dirichlet(:d, getfacetset(mesh, "left"), (x, t) -> (0.0, 0.0, 0.0), [1, 2, 3]),
        Dirichlet(:d, getfacetset(mesh, "right"), (x, t) -> (0.1, 0.0, 0.0), [1, 2, 3]),
    ]
    quasistaticform = semidiscretize(
        QuasiStaticModel(:d, material, ()),
        FiniteElementDiscretization(Dict(:d => LagrangeCollection{1}()^3); dbcs),
        mesh,
    )
    problem = QuasiStaticProblem(quasistaticform, (0.0, 0.3))
    timestepper = BackwardEulerSolver(
        inner_solver = Thunderbolt.MultiLevelNewtonRaphsonSolver(
            newton = NewtonRaphsonSolver(
                inner_solver = UMFPACKFactorization(),
                max_iter = 10,
                tol = 1e-8,
            ),
        ),
    )
    integrator = init(problem, timestepper, dt = 0.1, verbose = false)
    solve!(integrator)
    @test integrator.sol.retcode == SciMLBase.ReturnCode.Success

    nfe = ndofs(quasistaticform.dh)
    niv = ndofs(quasistaticform.lvh)
    ncells = getncells(mesh)
    blocksize = niv ÷ ncells
    @test niv == ncells * blocksize
    cell_block(c) = integrator.u[(nfe+(c-1)*blocksize+1):(nfe+c*blocksize)]
    # The first cell is written correctly even with the bug present, since the missing offset is
    # zero for it. The defect is that every *later* cell is left untouched.
    @test all(c -> !iszero(cell_block(c)), 1:ncells)
end

@testset "Condensed sarcomere under strong activation" begin
    # Regression test for the condensation contribution to the stress tangent, `∂P/∂Q ⊗ ∂Q/∂λ ⊗ ∂λ/∂F`.
    # The other contraction tests drive the sarcomere at Ca ≈ 0.004, where that contribution is far
    # too small for its *sign* to affect convergence -- which is how a sign error in
    # `_solve_local_sarcomere_dQdF` survived. At full activation the same error diverges the global
    # Newton within two steps, so this is the configuration that pins the tangent down.
    mesh = generate_mesh(Hexahedron, (2, 2, 1), Vec((0.0, 0.0, 0.0)), Vec((1.0, 1.0, 0.2)))
    microstructure = OrthotropicMicrostructureModel(
        ConstantCoefficient(Vec((1.0, 0.0, 0.0))),
        ConstantCoefficient(Vec((0.0, 1.0, 0.0))),
        ConstantCoefficient(Vec((0.0, 0.0, 1.0))),
    )
    model = QuasiStaticModel(
        :d,
        ActiveStressModel(
            Guccione1991PassiveModel(),
            SimpleActiveStress(; Tmax = 220e3),
            Thunderbolt.CaDrivenInternalSarcomereModel(
                Thunderbolt.RDQ20MFModel(),
                ConstantCoefficient(1.0),
            ),
            microstructure,
        ),
        (),
    )
    dbcs = [
        Dirichlet(:d, getfacetset(mesh, "left"), (x, t) -> [0.0], [1])
        Dirichlet(:d, getfacetset(mesh, "front"), (x, t) -> [0.0], [2])
        Dirichlet(:d, getfacetset(mesh, "bottom"), (x, t) -> [0.0], [3])
        Dirichlet(:d, Set([1]), (x, t) -> [0.0, 0.0, 0.0], [1, 2, 3])
    ]
    quasistaticform = semidiscretize(
        model,
        FiniteElementDiscretization(Dict(:d => LagrangeCollection{1}()^3); dbcs),
        mesh,
    )
    problem = QuasiStaticProblem(quasistaticform, (0.0, 5.0))
    Thunderbolt.default_initial_condition!(problem.u0, problem.f)
    timestepper = BackwardEulerSolver(
        inner_solver = Thunderbolt.MultiLevelNewtonRaphsonSolver(
            newton = NewtonRaphsonSolver(
                inner_solver = UMFPACKFactorization(),
                max_iter = 10,
                tol = 1e-8,
            ),
        ),
    )
    # Δt well below the ≈5 where RDQ20's Markov occupancies leave their bounds (see
    # `internal_state_in_bounds`), so this stays a tangent test rather than drifting into an
    # infeasibility test after an unrelated tweak.
    integrator = init(problem, timestepper, dt = 2.5, verbose = false)
    solve!(integrator)
    @test integrator.sol.retcode == SciMLBase.ReturnCode.Success
end

@testset "A step too long for the sarcomere fails cleanly" begin
    # RDQ20's Markov occupancies leave [0, 1] once the step outruns their own dynamics. That has to
    # surface as a return code the time integrator can act on, not as an exception out of the local
    # Newton, and the failed attempt must not be accepted. This is the only test that forces a local
    # solve to fail, so it is also what covers the per-quadrature-point failure reporting.
    mesh = generate_mesh(Hexahedron, (1, 1, 1), Vec((0.0, 0.0, 0.0)), Vec((1.0, 1.0, 0.2)))
    microstructure = OrthotropicMicrostructureModel(
        ConstantCoefficient(Vec((1.0, 0.0, 0.0))),
        ConstantCoefficient(Vec((0.0, 1.0, 0.0))),
        ConstantCoefficient(Vec((0.0, 0.0, 1.0))),
    )
    model = QuasiStaticModel(
        :d,
        ActiveStressModel(
            Guccione1991PassiveModel(),
            SimpleActiveStress(; Tmax = 220e3),
            Thunderbolt.CaDrivenInternalSarcomereModel(
                Thunderbolt.RDQ20MFModel(),
                ConstantCoefficient(1.0),
            ),
            microstructure,
        ),
        (),
    )
    dbcs = [
        Dirichlet(:d, getfacetset(mesh, "left"), (x, t) -> [0.0], [1])
        Dirichlet(:d, getfacetset(mesh, "front"), (x, t) -> [0.0], [2])
        Dirichlet(:d, getfacetset(mesh, "bottom"), (x, t) -> [0.0], [3])
        Dirichlet(:d, Set([1]), (x, t) -> [0.0, 0.0, 0.0], [1, 2, 3])
    ]
    quasistaticform = semidiscretize(
        model,
        FiniteElementDiscretization(Dict(:d => LagrangeCollection{1}()^3); dbcs),
        mesh,
    )
    problem = QuasiStaticProblem(quasistaticform, (0.0, 20.0))
    Thunderbolt.default_initial_condition!(problem.u0, problem.f)
    timestepper = BackwardEulerSolver(
        inner_solver = Thunderbolt.MultiLevelNewtonRaphsonSolver(
            newton = NewtonRaphsonSolver(
                inner_solver = UMFPACKFactorization(),
                max_iter = 10,
                tol = 1e-8,
            ),
        ),
    )
    integrator = init(problem, timestepper, dt = 20.0, verbose = false)
    # The solver warns that it cannot adapt its way out of this, which is expected here.
    with_logger(NullLogger()) do
        solve!(integrator)
    end
    @test integrator.sol.retcode == SciMLBase.ReturnCode.ConvergenceFailure
    @test integrator.t == 0.0
end

@testset "A failed homotopy solve shrinks dt once, not twice" begin
    # `dt` shrinks once per failed attempt: the step footer's `post_newton_controller!` owns the
    # solve-failure case, the controller's reject hook owns the convergence-rate case.
    mesh = generate_mesh(Hexahedron, (2, 1, 1), Vec((0.0, 0.0, 0.0)), Vec((1.0, 0.2, 0.2)))
    ms = ConstantCoefficient(
        OrthotropicMicrostructure(Vec((1.0, 0.0, 0.0)), Vec((0.0, 1.0, 0.0)), Vec((0.0, 0.0, 1.0))),
    )
    dbcs = [
        Dirichlet(:d, getfacetset(mesh, "left"), (x, t) -> [0.0, 0.0, 0.0], [1, 2, 3]),
        Dirichlet(:d, getfacetset(mesh, "right"), (x, t) -> [0.6t, 0.0, 0.0], [1, 2, 3]),
    ]
    f = semidiscretize(
        QuasiStaticModel(:d, PK1Model(Guccione1991PassiveModel(), ms), ()),
        FiniteElementDiscretization(Dict(:d => LagrangeCollection{1}()^3); dbcs),
        mesh,
    )
    integrator = init(
        QuasiStaticProblem(f, (0.0, 1.0)),
        # A tolerance the Newton cannot reach, so every attempt fails.
        HomotopyPathSolver(
            NewtonRaphsonSolver(inner_solver = UMFPACKFactorization(), max_iter = 1, tol = 1e-30),
        ),
        dt = 0.2,
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
    # Two-sided: `≤` alone is also satisfied by a `dt` that never shrank, which is the opposite bug.
    @test ff^(integrator.stats.nreject - 1) ≤ dt₀ / integrator.dt ≤ ff^integrator.stats.nreject
end

"""
The condensed cuboid of the two testsets above, solved with whichever global Newton is handed in.
Fully activated, so the local problems are genuinely nonlinear at every quadrature point.
"""
function solve_condensed_cuboid(
    sarcomere,
    newton,
    Δt,
    tend,
    local_solver = Thunderbolt.GenericLocalNonlinearSolver(),
)
    mesh = generate_mesh(Hexahedron, (2, 2, 1), Vec((0.0, 0.0, 0.0)), Vec((1.0, 1.0, 0.2)))
    microstructure = OrthotropicMicrostructureModel(
        ConstantCoefficient(Vec((1.0, 0.0, 0.0))),
        ConstantCoefficient(Vec((0.0, 1.0, 0.0))),
        ConstantCoefficient(Vec((0.0, 0.0, 1.0))),
    )
    model = QuasiStaticModel(
        :d,
        ActiveStressModel(
            Guccione1991PassiveModel(),
            SimpleActiveStress(; Tmax = 220e3),
            Thunderbolt.CaDrivenInternalSarcomereModel(sarcomere, ConstantCoefficient(1.0)),
            microstructure,
        ),
        (),
    )
    dbcs = [
        Dirichlet(:d, getfacetset(mesh, "left"), (x, t) -> [0.0], [1])
        Dirichlet(:d, getfacetset(mesh, "front"), (x, t) -> [0.0], [2])
        Dirichlet(:d, getfacetset(mesh, "bottom"), (x, t) -> [0.0], [3])
        Dirichlet(:d, Set([1]), (x, t) -> [0.0, 0.0, 0.0], [1, 2, 3])
    ]
    quasistaticform = semidiscretize(
        model,
        FiniteElementDiscretization(Dict(:d => LagrangeCollection{1}()^3); dbcs),
        mesh,
    )
    problem = QuasiStaticProblem(quasistaticform, (0.0, tend))
    Thunderbolt.default_initial_condition!(problem.u0, problem.f)
    timestepper = BackwardEulerSolver(
        inner_solver = Thunderbolt.MultiLevelNewtonRaphsonSolver(
            newton = newton,
            local_solver = local_solver,
        ),
    )
    integrator = init(problem, timestepper, dt = Δt, verbose = false)
    solve!(integrator)
    return integrator
end

function solve_viscoelastic_creep(newton)
    mesh = generate_mesh(Hexahedron, (2, 1, 1))
    material = Thunderbolt.LinearMaxwellMaterial(E₀ = 70e3, E₁ = 20e3, μ = 1e3, η₁ = 1e3, ν = 0.3)
    dbcs = [
        Dirichlet(:d, getfacetset(mesh, "left"), (x, t) -> (0.0, 0.0, 0.0), [1, 2, 3]),
        Dirichlet(:d, getfacetset(mesh, "right"), (x, t) -> (0.1, 0.0, 0.0), [1, 2, 3]),
    ]
    quasistaticform = semidiscretize(
        QuasiStaticModel(:d, material, ()),
        FiniteElementDiscretization(Dict(:d => LagrangeCollection{1}()^3); dbcs),
        mesh,
    )
    problem = QuasiStaticProblem(quasistaticform, (0.0, 0.3))
    timestepper = BackwardEulerSolver(
        inner_solver = Thunderbolt.MultiLevelNewtonRaphsonSolver(newton = newton),
    )
    integrator = init(problem, timestepper, dt = 0.1, verbose = false)
    solve!(integrator)
    return integrator
end

function solve_prestressed_sheet(newton)
    grid = generate_grid(Hexahedron, (3, 3, 1), Vec((-1.0, -1.0, -0.2)), Vec((1.0, 1.0, 0.2)))
    addcellset!(grid, "myocardium", x->true)
    mesh = to_mesh(grid)
    ortho_ms = ConstantCoefficient(
        OrthotropicMicrostructure(Vec((1.0, 0.0, 0.0)), Vec((0.0, 1.0, 0.0)), Vec((0.0, 0.0, 1.0))),
    )
    dbcs = [
        Dirichlet(:d, getfacetset(mesh, "left"), (x, t) -> [0.0], [1])
        Dirichlet(:d, getfacetset(mesh, "front"), (x, t) -> [0.0], [2])
        Dirichlet(:d, getfacetset(mesh, "bottom"), (x, t) -> [0.0], [3])
        Dirichlet(:d, Set([1]), (x, t) -> [0.0, 0.0, 0.0], [1, 2, 3])
        Dirichlet(:d, getfacetset(mesh, "right"), (x, t) -> [0.01t], [1])
        Dirichlet(:d, getfacetset(mesh, "top"), (x, t) -> [0.02t], [2])
        Dirichlet(:d, getfacetset(mesh, "back"), (x, t) -> [0.03t], [3])
    ]
    quasistaticform = semidiscretize(
        QuasiStaticModel(
            :d,
            PrestressedMechanicalModel(
                PK1Model(HolzapfelOgden2009Model(), ortho_ms),
                ConstantCoefficient(Tensor{2, 3}((1.1, 0.1, 0.0, 0.2, 0.9, 0.1, -0.1, 0.0, 1.0))),
            ),
        ),
        FiniteElementDiscretization(Dict(:d => LagrangeCollection{1}()^3); dbcs),
        mesh,
    )
    problem = QuasiStaticProblem(quasistaticform, (0.0, 1.0))
    integrator = init(problem, HomotopyPathSolver(newton), dt = 1.0, verbose = false)
    solve!(integrator)
    return integrator
end

@testset "Simplified Newton and Eisenstat-Walker forcing" begin
    # Both change how the iteration is run, not what it converges to, so the assertion throughout is
    # that the solution is the one the ordinary Newton finds.
    #
    # The residual-only assembly path they lean on is not a subset of the linearization -- it re-solves
    # the local problems and it reaches the materials through `stress_function` rather than
    # `stress_and_tangent`.
    direct = NewtonRaphsonSolver(inner_solver = UMFPACKFactorization(), max_iter = 20, tol = 1e-8)

    # Both sarcomere variants, because the rate-coupled and the rate-free local problem reach the
    # residual through different entry points.
    #
    # The wrapped model's tangent carries no `∂P/∂Ḟ · ∂Ḟ/∂u` term, so the same residual buys a larger
    # first increment and the residual rises once before quadratic convergence takes over. Two
    # consequences, both measured on this cuboid:
    #
    #   * `enforce_monotonic_convergence` would abort that as divergence, hence `false` here. It is
    #     not a concession -- the full Newton reaches `tol` one iteration after the overshoot.
    #   * the *simplified* Newton needs the shorter step, because a frozen Jacobian cannot correct
    #     the overshoot: at `Δt = 2.5` it drives a local sarcomere solve to `NaN`, and at `Δt = 1.0`
    #     it stalls at `‖r‖ ≈ 0.4`. The full Newton converges at `Δt = 2.5` for both variants.
    @testset "Condensed sarcomere, $(nameof(typeof(sarcomere)))" for (sarcomere, Δt, tend) in (
        (Thunderbolt.RDQ20MFModel(), 2.5, 5.0),
        (Thunderbolt.AsRateIndependent(Thunderbolt.RDQ20MFModel()), 0.5, 5.0),
    )
        reference = solve_condensed_cuboid(
            sarcomere,
            NewtonRaphsonSolver(
                inner_solver = UMFPACKFactorization(),
                max_iter = 20,
                tol = 1e-8,
                enforce_monotonic_convergence = false,
            ),
            Δt,
            tend,
        )
        @test reference.sol.retcode == SciMLBase.ReturnCode.Success

        # A simplified Newton converges linearly, so it needs a far more generous iteration budget
        # than the quadratic one it is compared against.
        simplified = solve_condensed_cuboid(
            sarcomere,
            NewtonRaphsonSolver(
                inner_solver = UMFPACKFactorization(),
                max_iter = 200,
                tol = 1e-8,
                simplified_newton = true,
                enforce_monotonic_convergence = false,
            ),
            Δt,
            tend,
        )
        @test simplified.sol.retcode == SciMLBase.ReturnCode.Success
        @test simplified.u ≈ reference.u rtol=1e-6
    end

    # The activated sarcomere problem is too ill-conditioned for unpreconditioned GMRES, so the
    # forcing term is exercised on the linear viscoelastic one, where GMRES is the default anyway.
    @testset "Viscoelastic creep" begin
        reference = solve_viscoelastic_creep(direct)
        @test reference.sol.retcode == SciMLBase.ReturnCode.Success

        for newton in (
            NewtonRaphsonSolver(
                tol = 1e-8,
                enforce_monotonic_convergence = false,
                simplified_newton = true,
            ),
            NewtonRaphsonSolver(
                tol = 1e-8,
                enforce_monotonic_convergence = false,
                forcing = EisenstatWalkerForcing(),
            ),
            NewtonRaphsonSolver(
                tol = 1e-8,
                forcing = EisenstatWalkerForcing(),
                simplified_newton = true,
                enforce_monotonic_convergence = false,
            ),
        )
            integrator = solve_viscoelastic_creep(newton)
            @test integrator.sol.retcode == SciMLBase.ReturnCode.Success
            @test integrator.u ≈ reference.u rtol=1e-8
        end
    end

    # `PrestressedMechanicalModel` has its own residual-only entry point, which pulls the stress back
    # from the intermediate configuration without ever forming a tangent.
    @testset "Prestressed sheet" begin
        mref = CountingNewtonMonitor()
        reference = solve_prestressed_sheet(
            NewtonRaphsonSolver(
                inner_solver = UMFPACKFactorization(),
                max_iter = 20,
                tol = 1e-8,
                monitor = mref,
            ),
        )
        @test reference.sol.retcode == SciMLBase.ReturnCode.Success

        msimplified = CountingNewtonMonitor()
        simplified = solve_prestressed_sheet(
            NewtonRaphsonSolver(
                inner_solver = UMFPACKFactorization(),
                max_iter = 100,
                tol = 1e-8,
                simplified_newton = true,
                monitor = msimplified,
            ),
        )
        @test simplified.sol.retcode == SciMLBase.ReturnCode.Success
        @test simplified.u ≈ reference.u rtol=1e-6
        # Agreement alone cannot tell a working simplified Newton from one that silently fell back
        # to the full method -- both would agree. The iteration count can: reusing the Jacobian
        # costs quadratic convergence, so it takes strictly more steps. Without this, the whole
        # residual-only assembly path could stop being exercised and every test here stay green.
        @test msimplified.steps > 2 * mref.steps
    end
end

@testset "The condensation report counts the local solves" begin
    # `condense_internal!` is what the stage reads to decide whether a step is usable, so its report
    # has to describe the solves that actually ran. Re-run on the converged step, where every
    # quadrature point poses a genuinely nonlinear local problem.
    integrator = solve_condensed_cuboid(
        Thunderbolt.RDQ20MFModel(),
        NewtonRaphsonSolver(
            inner_solver = UMFPACKFactorization(),
            max_iter = 20,
            tol = 1e-8,
            enforce_monotonic_convergence = false,
        ),
        2.5,
        2.5,
    )
    @test integrator.sol.retcode == SciMLBase.ReturnCode.Success

    sf = integrator.cache.stage.stage_function
    report = Thunderbolt.condense_internal!(
        Thunderbolt.getoperator(sf),
        Thunderbolt.stage_weights(sf),
        Thunderbolt.stage_states(sf, integrator.cache.uₙ),
        Thunderbolt.stage_user_parameters(sf),
        Thunderbolt.stage_context(sf),
    )
    ncells = getncells(Thunderbolt.get_grid(integrator.f.dh))

    @test report.converged
    # One local problem per quadrature point of every cell, and a sarcomere Newton takes at least
    # one pass -- a report of zeros is what the hook returned before it counted anything.
    @test report.solves ≥ ncells
    @test report.solves % ncells == 0
    @test report.iterations ≥ report.solves
    @test report.worst_iterations ≥ 1
    # The argmax carriers survive the fold across cells: a cellid (positive, this being the cell
    # family) and one of that cell's quadrature points.
    @test 1 ≤ report.worst_cell ≤ ncells
    @test 1 ≤ report.worst_qp ≤ report.solves ÷ ncells
    @test report.worst_iterations ≤ report.iterations
    @test isfinite(report.worst_residual) && report.worst_residual ≥ 0.0
    # No stepper here is adaptive, so the sweep asks for no step reduction.
    @test report.dt_factor == 1.0
end

@testset "The condensation phase owns the local solves" begin
    # The assembly sweeps are pure evaluations at the state `condense_internal!` wrote, correcting
    # their tangent with what it stored. Two consequences, and this is what asserts them: a sweep
    # solves nothing, and a tangent sweep will not run before a condensation has produced its
    # correction.
    integrator = solve_condensed_cuboid(
        Thunderbolt.RDQ20MFModel(),
        NewtonRaphsonSolver(
            inner_solver = UMFPACKFactorization(),
            max_iter = 20,
            tol = 1e-8,
            enforce_monotonic_convergence = false,
        ),
        2.5,
        2.5,
    )
    @test integrator.sol.retcode == SciMLBase.ReturnCode.Success

    stage  = integrator.cache.stage
    sf     = stage.stage_function
    op     = Thunderbolt.getoperator(sf)
    states = Thunderbolt.stage_states(sf, integrator.cache.uₙ)
    p      = Thunderbolt.stage_user_parameters(sf)
    ctx    = Thunderbolt.stage_context(sf)
    w      = Thunderbolt.stage_weights(sf)
    J      = Thunderbolt.getJ(op)
    r      = zeros(Thunderbolt.FerriteOperators.residual_size(op))
    lsc    = stage.nlsolver.local_solver_cache

    @test Thunderbolt.condense_internal!(op, w, states, p, ctx).converged

    # Every local solve of this iterate is recorded by the condensation phase, so a sweep that
    # records nothing is a sweep that solved nothing -- a sarcomere Newton always takes a pass.
    Thunderbolt.reset_local_solve_status!(lsc)
    Thunderbolt.evaluate!(op, r, states, p, ctx)
    Thunderbolt.assemble_weighted_jacobian!(J, op, w, states, p, ctx)
    @test all(report -> report.iterations == 0, lsc.reports.data)
    @test !any(Thunderbolt._local_solve_failed, lsc.reports.data)

    # A rejected step discards the correctors along with the trial they belong to.
    Thunderbolt.rollback_state!(integrator, integrator.cache)
    states = Thunderbolt.stage_states(sf, integrator.cache.uₙ)
    @test_throws ArgumentError Thunderbolt.assemble_weighted_jacobian!(J, op, w, states, p, ctx)
    # The residual is a function of the state alone, so it needs no corrector and stays available.
    Thunderbolt.evaluate!(op, r, states, p, ctx)
    Thunderbolt.condense_internal!(op, w, states, p, ctx)
    Thunderbolt.assemble_weighted_jacobian!(J, op, w, states, p, ctx)
end

@testset "A failed local solve says where it failed" begin
    # One local Newton pass against a tolerance no sarcomere reaches: every quadrature point exits
    # with `MaxIters`, so the condensation phase cannot converge and the step has to end. What is
    # asserted is that it ends *audibly* -- a step that aborts without naming the offender leaves
    # nothing to debug a material model with.
    logger = Test.TestLogger(min_level = Logging.Debug)
    integrator = Logging.with_logger(logger) do
        solve_condensed_cuboid(
            Thunderbolt.RDQ20MFModel(),
            NewtonRaphsonSolver(inner_solver = UMFPACKFactorization(), max_iter = 20, tol = 1e-8),
            2.5,
            2.5,
            Thunderbolt.GenericLocalNonlinearSolver(max_iters = 1, tol = 1e-14),
        )
    end
    @test integrator.sol.retcode != SciMLBase.ReturnCode.Success

    messages = [string(record.message) for record in logger.logs]
    # The folded report: that it failed, and the worst offender it carries.
    @test any(m -> occursin("Local solve did not converge", m), messages)
    @test any(m -> occursin("NOT converged", m), messages)
    @test any(m -> occursin(r"worst cell \d+ qp \d+ at \d+ iterations", m), messages)
    # The per-point detail only the multilevel solver's store can give, listing every failing point
    # rather than the one the fold kept.
    @test any(m -> occursin("Local solve failures of this pass", m), messages)
    @test any(m -> occursin(r"cell \d+ qp \d+: MaxIters", m), messages)
end

@testset "A converged condensation stays quiet" begin
    # The counterpart of the diagnostic above: it must fire on failure and only on failure, or it
    # stops carrying information.
    logger = Test.TestLogger(min_level = Logging.Debug)
    integrator = Logging.with_logger(logger) do
        solve_condensed_cuboid(
            Thunderbolt.RDQ20MFModel(),
            NewtonRaphsonSolver(
                inner_solver = UMFPACKFactorization(),
                max_iter = 20,
                tol = 1e-8,
                enforce_monotonic_convergence = false,
            ),
            2.5,
            2.5,
        )
    end
    @test integrator.sol.retcode == SciMLBase.ReturnCode.Success

    messages = [string(record.message) for record in logger.logs]
    @test !any(m -> occursin("Local solve did not converge", m), messages)
    @test !any(m -> occursin("Local solve failures of this pass", m), messages)
end

@testset "Residual-only condensation solves the same and stores nothing" begin
    # A residual sweep reads the condensed state and no corrector, so the corrector is waste there.
    # Eliding it is only admissible if it changes nothing about the state -- the election governs
    # what is formed after the local solve, never the solve -- and that is asserted bitwise, not to
    # a tolerance: a tolerance would hide exactly the kind of drift this is guarding against.
    integrator = solve_condensed_cuboid(
        Thunderbolt.RDQ20MFModel(),
        NewtonRaphsonSolver(
            inner_solver = UMFPACKFactorization(),
            max_iter = 20,
            tol = 1e-8,
            enforce_monotonic_convergence = false,
        ),
        2.5,
        2.5,
    )
    @test integrator.sol.retcode == SciMLBase.ReturnCode.Success

    stage = integrator.cache.stage
    sf    = stage.stage_function
    op    = Thunderbolt.getoperator(sf)
    p     = Thunderbolt.stage_user_parameters(sf)
    ctx   = Thunderbolt.stage_context(sf)
    w     = Thunderbolt.stage_weights(sf)
    J     = Thunderbolt.getJ(op)

    # Both routes have to start from the same point: the phase writes `q` into the very vector it
    # reads `u` from, so a second condensation would otherwise warm start off the first one's answer.
    z0     = copy(integrator.cache.uₙ)
    z      = copy(z0)
    states = Thunderbolt.stage_states(sf, z)
    nres   = Thunderbolt.FerriteOperators.residual_size(op)

    weighted_report = Thunderbolt.condense_internal!(op, w, states, p, ctx)
    @test weighted_report.converged
    q_weighted = copy(z)
    r_weighted = zeros(nres)
    Thunderbolt.evaluate!(op, r_weighted, states, p, ctx)

    z .= z0
    residual_only_report = Thunderbolt.condense_internal!(op, nothing, states, p, ctx)
    @test residual_only_report.converged
    q_residual_only = copy(z)
    r_residual_only = zeros(nres)
    Thunderbolt.evaluate!(op, r_residual_only, states, p, ctx)

    @test q_residual_only == q_weighted
    @test r_residual_only == r_weighted
    # The solves themselves are the same solves, so the report they fold is the same report.
    @test residual_only_report.solves == weighted_report.solves
    @test residual_only_report.iterations == weighted_report.iterations

    # What it does cost: the correctors are gone, and a tangent that needs them says so instead of
    # combining whatever the last weighted condensation happened to leave behind.
    err = try
        Thunderbolt.assemble_weighted_jacobian!(J, op, w, states, p, ctx)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("has no valid state", err.msg)
    # Condensing with weights again restores them.
    Thunderbolt.condense_internal!(op, w, states, p, ctx)
    Thunderbolt.assemble_weighted_jacobian!(J, op, w, states, p, ctx)

    # And the wiring: the residual half of the stage protocol is what elects it, so a tangent sweep
    # straight after one is refused for the same reason.
    Thunderbolt.condense_internal!(op, w, states, p, ctx)
    @test Thunderbolt.evaluate_stage_residual!(sf, zeros(nres), z)
    @test_throws ArgumentError Thunderbolt.assemble_weighted_jacobian!(J, op, w, states, p, ctx)
end

@testset "The consistency referee checks the weighted route" begin
    # `ConsistencyCheckWeakBoundaryCondition` differences the combination the request it is serving
    # claims to assemble. Backward Euler poses the weighted one, and that is the only route carrying
    # a dashpot's rate term into the matrix -- so it is also the only route on which that term is
    # checkable at all. A silent run is the assertion: a referee that dropped the `:v` perturbation
    # would difference the rate term against zero and warn.
    mesh = generate_mesh(Hexahedron, (2, 2, 2))
    material = Thunderbolt.LinearMaxwellMaterial(E₀ = 70e3, E₁ = 20e3, μ = 1e3, η₁ = 1e3, ν = 0.3)
    dbcs = [Dirichlet(:d, getfacetset(mesh, "left"), (x, t) -> (0.0, 0.0, 0.0), [1, 2, 3])]
    form = semidiscretize(
        QuasiStaticModel(
            :d,
            material,
            (
                ConstantPressureBC(-1e3, "right"),
                Thunderbolt.ConsistencyCheckWeakBoundaryCondition(
                    ViscousRobinBC(5e3, "right"),
                    1.0e-6,
                ),
            ),
        ),
        FiniteElementDiscretization(Dict(:d => LagrangeCollection{1}()^3); dbcs),
        mesh,
    )
    integrator = init(
        QuasiStaticProblem(form, (0.0, 0.2)),
        BackwardEulerSolver(
            inner_solver = Thunderbolt.MultiLevelNewtonRaphsonSolver(
                newton = NewtonRaphsonSolver(
                    inner_solver = UMFPACKFactorization(),
                    max_iter = 10,
                    tol = 1e-8,
                ),
            ),
        ),
        dt = 0.1,
        verbose = false,
    )
    @test_logs min_level = Logging.Warn solve!(integrator)
    @test integrator.sol.retcode == SciMLBase.ReturnCode.Success
end

@testset "A rate dependent material rejects rate-free kinematics" begin
    # `HomotopyPathSolver` is continuation, not a time scheme: it has no previous solution and no
    # timestep, so a material carrying an evolving internal variable has to be rejected -- and
    # rejected during setup, once and by name, rather than per element from the assembly loop.
    mesh = generate_mesh(Hexahedron, (1, 1, 1))
    microstructure = OrthotropicMicrostructureModel(
        ConstantCoefficient(Vec((1.0, 0.0, 0.0))),
        ConstantCoefficient(Vec((0.0, 1.0, 0.0))),
        ConstantCoefficient(Vec((0.0, 0.0, 1.0))),
    )
    model = QuasiStaticModel(
        :d,
        ActiveStressModel(
            Guccione1991PassiveModel(),
            SimpleActiveStress(; Tmax = 220e3),
            Thunderbolt.CaDrivenInternalSarcomereModel(
                Thunderbolt.RDQ20MFModel(),
                ConstantCoefficient(1.0),
            ),
            microstructure,
        ),
        (),
    )
    dbcs = [
        Dirichlet(:d, getfacetset(mesh, "left"), (x, t) -> [0.0, 0.0, 0.0], [1, 2, 3]),
        Dirichlet(:d, getfacetset(mesh, "right"), (x, t) -> [0.05, 0.0, 0.0], [1, 2, 3]),
    ]
    quasistaticform = semidiscretize(
        model,
        FiniteElementDiscretization(Dict(:d => LagrangeCollection{1}()^3); dbcs),
        mesh,
    )
    problem = QuasiStaticProblem(quasistaticform, (0.0, 1.0))
    Thunderbolt.default_initial_condition!(problem.u0, problem.f)
    timestepper = HomotopyPathSolver(
        NewtonRaphsonSolver(inner_solver = UMFPACKFactorization(), max_iter = 10, tol = 1e-8),
    )
    # Assert on the classification, not on the remedy: an earlier version of this message offered
    # `AsRateIndependent` as the way out, which does not work — the wrapper drops the velocity
    # dependence but leaves `dₜQ = L(F, Q)`, so the rejection fires again. A substring test against
    # the remedy passed throughout.
    @test_throws "RateCoupledEvolution" init(problem, timestepper, dt = 1.0, verbose = false)
end

@testset "Viscous Robin boundary conditions" begin
    # A block pushed by a constant pressure, held on one side, with a dashpot on the loaded facet.
    # The dashpot resists the *rate*, so it slows the approach to equilibrium without changing the
    # equilibrium itself — which is exactly the pair of properties asserted below.
    function solve_with(facet_models; Δt = 0.1, tend = 1.0)
        mesh = generate_mesh(Hexahedron, (2, 2, 2))
        material =
            Thunderbolt.LinearMaxwellMaterial(E₀ = 70e3, E₁ = 20e3, μ = 1e3, η₁ = 1e3, ν = 0.3)
        dbcs = [Dirichlet(:d, getfacetset(mesh, "left"), (x, t) -> (0.0, 0.0, 0.0), [1, 2, 3])]
        form = semidiscretize(
            QuasiStaticModel(:d, material, facet_models),
            FiniteElementDiscretization(Dict(:d => LagrangeCollection{1}()^3); dbcs),
            mesh,
        )
        problem = QuasiStaticProblem(form, (0.0, tend))
        timestepper = BackwardEulerSolver(
            inner_solver = Thunderbolt.MultiLevelNewtonRaphsonSolver(
                newton = NewtonRaphsonSolver(
                    inner_solver = UMFPACKFactorization(),
                    max_iter = 10,
                    tol = 1e-8,
                ),
            ),
        )
        integrator = init(problem, timestepper, dt = Δt, verbose = false)
        solve!(integrator)
        @test integrator.sol.retcode == SciMLBase.ReturnCode.Success
        return norm(@view integrator.u[1:ndofs(form.dh)])
    end

    load = (ConstantPressureBC(-1e3, "right"),)

    # Damping resists the motion, and more of it resists more. Which *directions* a dashpot resists is
    # pinned exactly in `test_elements.jl` rather than through a pair of nonlinear solves here.
    @test solve_with((load..., ViscousRobinBC(5e3, "right"))) < solve_with(load)
    @test solve_with((load..., ViscousRobinBC(1e5, "right"))) <
          solve_with((load..., ViscousRobinBC(1e3, "right")))

    # Held long enough the velocity dies out, so the dashpot contributes nothing and the damped
    # solution lands on the undamped equilibrium. This is what distinguishes a dashpot from a spring:
    # a spring would shift the equilibrium permanently.
    #
    # `tend` is set from the material rather than picked: the Maxwell branch relaxes with
    # `η₁/E₁ = 0.05`, so five time units is a hundred relaxation times and the velocity is long gone.
    @test solve_with((load..., ViscousRobinBC(5e3, "right")); Δt = 0.5, tend = 5.0) ≈
          solve_with(load; Δt = 0.5, tend = 5.0) rtol=1e-6
end

@testset "Viscous Robin is rejected by continuation" begin
    # `HomotopyPathSolver` is load stepping: it has no previous solution and no timestep, so there is
    # no velocity for a dashpot to resist. Rejecting at setup keeps the report to one named boundary
    # condition instead of a missing field deep inside the assembly loop.
    mesh = generate_mesh(Hexahedron, (1, 1, 1))
    microstructure = Thunderbolt.ConstantCoefficient(
        Thunderbolt.OrthotropicMicrostructure(
            Vec((1.0, 0.0, 0.0)),
            Vec((0.0, 1.0, 0.0)),
            Vec((0.0, 0.0, 1.0)),
        ),
    )
    model = Thunderbolt.PK1Model(Thunderbolt.LinearSpringModel(), microstructure)
    dbcs = [Dirichlet(:d, getfacetset(mesh, "left"), (x, t) -> (0.0, 0.0, 0.0), [1, 2, 3])]
    quasistaticform = semidiscretize(
        QuasiStaticModel(:d, model, (ViscousRobinBC(1.0, "right"),)),
        FiniteElementDiscretization(Dict(:d => LagrangeCollection{1}()^3); dbcs),
        mesh,
    )
    problem = QuasiStaticProblem(quasistaticform, (0.0, 1.0))
    timestepper = HomotopyPathSolver(
        NewtonRaphsonSolver(inner_solver = UMFPACKFactorization(), max_iter = 10, tol = 1e-8),
    )
    @test_throws "ViscousRobinBC" init(problem, timestepper, dt = 1.0, verbose = false)

    # The rejection unwraps the debug wrapper too, so wrapping a dashpot does not smuggle it past the
    # check and into the assembly loop.
    wrapped = semidiscretize(
        QuasiStaticModel(
            :d,
            model,
            (
                Thunderbolt.ConsistencyCheckWeakBoundaryCondition(
                    ViscousRobinBC(1.0, "right"),
                    1e-6,
                ),
            ),
        ),
        FiniteElementDiscretization(Dict(:d => LagrangeCollection{1}()^3); dbcs),
        mesh,
    )
    @test_throws "ViscousRobinBC" init(
        QuasiStaticProblem(wrapped, (0.0, 1.0)),
        timestepper,
        dt = 1.0,
        verbose = false,
    )
end

@testset "Convergence driven step size control with backward Euler" begin
    # The continuation controllers read Newton contraction rates, not a local error estimate, so they
    # are usable by any solver that answers `contraction_rate_cache` — backward Euler included. That
    # is the combination a viscously *regularized* problem wants: the rate term is there to give the
    # solve something to contract against, so its temporal accuracy is not a quantity worth
    # controlling, while the Newton convergence very much is.
    function build(facet_models)
        mesh = generate_mesh(Hexahedron, (2, 2, 2))
        material =
            Thunderbolt.LinearMaxwellMaterial(E₀ = 70e3, E₁ = 20e3, μ = 1e3, η₁ = 1e3, ν = 0.3)
        dbcs = [Dirichlet(:d, getfacetset(mesh, "left"), (x, t) -> (0.0, 0.0, 0.0), [1, 2, 3])]
        form = semidiscretize(
            QuasiStaticModel(:d, material, facet_models),
            FiniteElementDiscretization(Dict(:d => LagrangeCollection{1}()^3); dbcs),
            mesh,
        )
        return QuasiStaticProblem(form, (0.0, 1.0))
    end

    timestepper = BackwardEulerSolver(
        inner_solver = Thunderbolt.MultiLevelNewtonRaphsonSolver(
            newton = NewtonRaphsonSolver(
                inner_solver = UMFPACKFactorization(),
                max_iter = 10,
                tol = 1e-8,
            ),
        ),
    )
    bcs = (ConstantPressureBC(-1e3, "right"), ViscousRobinBC(5e3, "right"))

    # Backward Euler merely *permits* a controller. Its default is the dummy, so an ordinary `init`
    # still steps at the `dt` it was given — this is the regression guard on that, since declaring the
    # algorithm adaptive is what makes passing a controller legal in the first place.
    plain = init(build(bcs), timestepper, dt = 0.1, verbose = false)
    @test !SciMLBase.isadaptive(plain)
    solve!(plain)
    @test plain.dt == 0.1

    controlled = init(
        build(bcs),
        timestepper,
        dt = 0.1,
        verbose = false,
        controller = Deuflhard2004_B_DiscreteContinuationControllerVariant(; Θmin = 1/8, p = 1),
        dtmax = 0.5,
    )
    @test SciMLBase.isadaptive(controlled)
    solve!(controlled)
    @test controlled.sol.retcode == SciMLBase.ReturnCode.Success

    # These Newtons contract easily, so the controller grows the step -- and `adapt_dt!` clamps that
    # growth to `dtmax`. Bigger steps mean fewer of them than the 10 a fixed `dt = 0.1` would take.
    # `dtmax` is otherwise pinned only against a stub cache in `test_time_integrator.jl`, so this is
    # where the clamp is exercised through a real solve.
    @test controlled.dt > 0.1
    @test controlled.dt ≤ 0.5
    @test controlled.stats.naccept < 10
end

@testset "Backward Euler with a plain Newton" begin
    # A stage that condenses nothing needs no local solver, so the plain `NewtonRaphsonSolver` solves
    # it: the multilevel wrapper is machinery for local problems that do not exist here. Which cache a
    # stage gets is chosen by `setup_stage_nlsolver_cache` from the solver type alone.
    mesh = generate_mesh(Hexahedron, (2, 2, 2))
    microstructure = Thunderbolt.ConstantCoefficient(
        Thunderbolt.OrthotropicMicrostructure(
            Vec((1.0, 0.0, 0.0)),
            Vec((0.0, 1.0, 0.0)),
            Vec((0.0, 0.0, 1.0)),
        ),
    )
    dbcs = [Dirichlet(:d, getfacetset(mesh, "left"), (x, t) -> (0.0, 0.0, 0.0), [1, 2, 3])]

    # No internal variables, and a dashpot so that backward Euler has a rate to work with. Guccione
    # rather than a fibre spring: the latter has a rank deficient tangent in 3D, so it fails for every
    # solver and would prove nothing about the one under test.
    rate_free = Thunderbolt.PK1Model(Guccione1991PassiveModel(), microstructure)
    build(material) = QuasiStaticProblem(
        semidiscretize(
            QuasiStaticModel(
                :d,
                material,
                (ConstantPressureBC(-1e2, "right"), ViscousRobinBC(5e2, "right")),
            ),
            FiniteElementDiscretization(Dict(:d => LagrangeCollection{1}()^3); dbcs),
            mesh,
        ),
        (0.0, 1.0),
    )

    newton = NewtonRaphsonSolver(inner_solver = UMFPACKFactorization(), max_iter = 10, tol = 1e-8)

    prob = build(rate_free)
    @test ndofs(prob.f.lvh) == 0
    plain = init(prob, BackwardEulerSolver(inner_solver = newton), dt = 0.1, verbose = false)
    solve!(plain)
    @test plain.sol.retcode == SciMLBase.ReturnCode.Success

    # The two Newtons must agree on a problem both can solve — otherwise "it runs" would be the only
    # thing this test established.
    multilevel = init(
        build(rate_free),
        BackwardEulerSolver(
            inner_solver = Thunderbolt.MultiLevelNewtonRaphsonSolver(newton = newton),
        ),
        dt = 0.1,
        verbose = false,
    )
    solve!(multilevel)
    @test multilevel.sol.retcode == SciMLBase.ReturnCode.Success
    @test norm(plain.u) ≈ norm(multilevel.u) rtol=1e-6

    # A material that *does* condense has local problems the plain Newton cannot close, so the pairing
    # is refused at setup naming the count, rather than silently solving a system of the wrong size.
    condensing = Thunderbolt.LinearMaxwellMaterial(E₀ = 70e3, E₁ = 20e3, μ = 1e3, η₁ = 1e3, ν = 0.3)
    condensing_prob = build(condensing)
    @test ndofs(condensing_prob.f.lvh) > 0
    @test_throws "MultiLevelNewtonRaphsonSolver" init(
        condensing_prob,
        BackwardEulerSolver(inner_solver = newton),
        dt = 0.1,
        verbose = false,
    )
end

@testset "Deformation gradient report" begin
    mesh = generate_mesh(Hexahedron, (2, 2, 2))
    microstructure = Thunderbolt.ConstantCoefficient(
        Thunderbolt.OrthotropicMicrostructure(
            Vec((1.0, 0.0, 0.0)),
            Vec((0.0, 1.0, 0.0)),
            Vec((0.0, 0.0, 1.0)),
        ),
    )
    form = semidiscretize(
        QuasiStaticModel(:d, Thunderbolt.PK1Model(Guccione1991PassiveModel(), microstructure), ()),
        FiniteElementDiscretization(Dict(:d => LagrangeCollection{1}()^3)),
        mesh,
    )

    # Configurations written down analytically rather than solved for, so the report's numbers are
    # known exactly and a solve tolerance never enters. A solve here would only produce *some*
    # deformation, about which nothing sharp could be asserted.
    function configuration(f)
        u = zeros(solution_size(form))
        Ferrite.apply_analytical!(u, form.dh, :d, f)
        return u
    end
    stretched = configuration(x -> Vec((0.25x[1], 0.0, 0.0)))          # F₁₁ = 1.25
    folded    = configuration(x -> Vec((-1.5x[1], 0.0, 0.0)))          # F₁₁ = -0.5
    rotated   = let θ = π / 6                                          # rigid rotation about e₃
        configuration(x -> Vec((cos(θ) * x[1] - sin(θ) * x[2] - x[1], sin(θ) * x[1] + cos(θ) * x[2] - x[2], 0.0)))
    end

    @testset "the two entry points agree" begin
        # `f + u` reads the displacement symbol off the model; `dh + u + fields` is told it. They must
        # describe the same configuration, otherwise one of them is differentiating the wrong field.
        @test Thunderbolt.displacement_symbols(form) == (:d,)
        from_f  = Thunderbolt.deformation_gradient_report(form, stretched)
        from_dh = Thunderbolt.deformation_gradient_report(form.dh, stretched, :d)
        @test from_f.minJ == from_dh.minJ
        @test from_f.max_strain == from_dh.max_strain
        @test from_f.n_subdomains == 1

        # A collection is accepted, so a multi-domain model whose subdomains name the displacement
        # differently can be covered in one call.
        @test Thunderbolt.deformation_gradient_report(
            form.dh,
            stretched,
            (:displacement, :d, :u),
        ).minJ == from_f.minJ

        # A uniform uniaxial stretch, so every quadrature point carries the same known `det F`.
        @test from_f.minJ ≈ 1.25
        @test from_f.maxJ ≈ 1.25
        @test !Thunderbolt.is_inverted(from_f)
    end

    @testset "folded and rigid configurations are told apart" begin
        # A fold is a valid solve and an invalid deformation; nothing but the kinematics can say so.
        fold = Thunderbolt.deformation_gradient_report(form, folded)
        @test fold.minJ ≈ -0.5
        @test Thunderbolt.is_inverted(fold)
        @test fold.n_nonpositive == fold.n_quadrature_points

        # The motivating case for reporting the strain next to the determinant: a rigid rotation has
        # `det F == 1` and no strain at all, however far it moves. A report that watched only `det F`
        # would call this healthy, which is exactly how a solve drifting down a null space of the
        # tangent escapes notice.
        rigid = Thunderbolt.deformation_gradient_report(form, rotated)
        @test rigid.minJ ≈ 1.0
        @test rigid.maxJ ≈ 1.0
        @test !Thunderbolt.is_inverted(rigid)
        @test rigid.max_strain < 1.0e-12
        @test rigid.max_deviation > 0.5          # 2 sin(θ/2) ≈ 0.52 for θ = π/6
        # ... whereas straining moves the two together.
        stretch = Thunderbolt.deformation_gradient_report(form, stretched)
        @test stretch.max_strain > 0.1
    end

    @testset "non-mechanics subdomains are skipped" begin
        # A handler carrying a scalar field next to the displacement: the scalar has no `det F`, and
        # asking for the displacement must simply not look at it.
        dh = DofHandler(mesh)
        sdh = SubDofHandler(dh, Set(1:getncells(mesh)))
        add!(sdh, :d, Lagrange{RefHexahedron, 1}()^3)
        add!(sdh, :phi, Lagrange{RefHexahedron, 1}())
        close!(dh)
        @test Thunderbolt.deformation_gradient_report(dh, zeros(ndofs(dh)), :d).n_subdomains == 1
        # Asking for the scalar itself is a user error, and the message says why rather than throwing
        # from inside the tensor algebra.
        @test_throws "det F" Thunderbolt.deformation_gradient_report(dh, zeros(ndofs(dh)), :phi)

        # A subdomain with no displacement field at all -- the electrophysiology-next-to-mechanics
        # case -- contributes nothing and does not prevent the report.
        grid = generate_grid(Hexahedron, (2, 1, 1))
        dh2 = DofHandler(grid)
        mech = SubDofHandler(dh2, Set([1]))
        add!(mech, :d, Lagrange{RefHexahedron, 1}()^3)
        ep = SubDofHandler(dh2, Set([2]))
        add!(ep, :phi, Lagrange{RefHexahedron, 1}())
        close!(dh2)
        report = Thunderbolt.deformation_gradient_report(dh2, zeros(ndofs(dh2)), :d)
        @test report.n_subdomains == 1
        @test report.n_quadrature_points == 8   # one cell, not two

        # Nothing to report on is an error rather than a vacuous "all fine".
        @test_throws "Fields present" Thunderbolt.deformation_gradient_report(
            dh2,
            zeros(ndofs(dh2)),
            :nope,
        )
    end

    @testset "DeformationMonitor warns on a degenerate iterate" begin
        # The only user facing consumer of the report: it wraps an inner monitor and reports per Newton
        # iteration, so a solve that folds an element says so at the iterate that did it rather than
        # after the step has converged and hidden the evidence.
        dbcs = [
            Dirichlet(:d, getfacetset(mesh, "left"), (x, t) -> (0.0, 0.0, 0.0), [1, 2, 3]),
            Dirichlet(:d, getfacetset(mesh, "right"), (x, t) -> (0.05, 0.0, 0.0), [1, 2, 3]),
        ]
        monitored = semidiscretize(
            QuasiStaticModel(
                :d,
                Thunderbolt.PK1Model(Guccione1991PassiveModel(), microstructure),
                (),
            ),
            FiniteElementDiscretization(Dict(:d => LagrangeCollection{1}()^3); dbcs),
            mesh,
        )
        solve_monitored(warn_below) = solve!(
            init(
                QuasiStaticProblem(monitored, (0.0, 1.0)),
                HomotopyPathSolver(
                    NewtonRaphsonSolver(
                        inner_solver = UMFPACKFactorization(),
                        max_iter = 10,
                        tol = 1e-8,
                        monitor = DeformationMonitor(; warn_below),
                    ),
                ),
                dt = 1.0,
                verbose = false,
            ),
        )
        # This solve never folds anything, so a threshold at zero must stay silent ...
        @test_logs min_level = Logging.Warn solve_monitored(0.0)
        # ... while a threshold above the healthy `det F` fires, which is what says the monitor is
        # looking at the iterates at all rather than being wired up and never consulted.
        @test_logs (:warn,) match_mode = :any solve_monitored(2.0)
    end
end
