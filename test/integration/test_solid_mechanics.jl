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
        FiniteElementDiscretization(
            Dict(:d => LagrangeCollection{1}()^3);
            dbcs,
            assembly_strategy = Thunderbolt.PerColorAssemblyStrategy(PolyesterDevice(3)),
        ),
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
        FiniteElementDiscretization(
            Dict(:d => LagrangeCollection{1}()^3);
            dbcs,
            assembly_strategy = Thunderbolt.PerColorAssemblyStrategy(PolyesterDevice(3)),
        ),
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
        FiniteElementDiscretization(
            Dict(:d => LagrangeCollection{1}()^3);
            dbcs,
            assembly_strategy = Thunderbolt.PerColorAssemblyStrategy(PolyesterDevice(3)),
        ),
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
        # it sits next to a subdomain that carries an internal variable. Its element cache is a
        # `QuasiStaticElementCache`, which is not a `gto1` cache, so FerriteOperators' generic
        # `query_element_parameters(element, cell, ivh, p) = p` hands it the whole
        # `GenericFirstOrderTimeParameters` where the assembly expects a time.
        #
        # Everything above passes only because no coefficient on a rate-free subdomain looks at `t`.
        # Below, the rate-free subdomain drives its sarcomere from a time dependent calcium field,
        # which is the ordinary cardiac case.
        #
        # BROKEN: to be fixed together with the parameter system, which is what decides how a
        # subdomain asks for the time and for the parameters it is differentiated against.
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

        # Same defect on the facet path. The surface element cache is handed whatever
        # `query_element_parameters` produced for the *volumetric* cache of its subdomain, so on a
        # rate-free subdomain that is again the raw parameter object. The unwrapping methods this
        # branch adds are typed on `GenericFirstOrderTimeElementParameters` and never fire here.
        #
        # BROKEN, same fix as above.
        let facemodels_tdep = (
                NormalSpringBC(0.0, "right"),
                ConstantPressureBC(0.0, "back"),
                PressureFieldBC(TestRampField(), "top"),
            )
            @testset "Time dependent facet coefficient on a rate-free subdomain" begin
                @test_broken (
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

        # Regression: `setup_boundary_cache` for `NonlinearMultiDomainIntegrator2` used to look the
        # subdomain name up in the *surface* subdomains, which is a different namespace from the
        # volumetric one its subintegrators are keyed by. It therefore returned an empty cache and
        # silently dropped every weak boundary condition.
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
function solve_condensed_cuboid(sarcomere, newton, Δt, tend)
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
        inner_solver = Thunderbolt.MultiLevelNewtonRaphsonSolver(newton = newton),
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
    # They are also the only thing that exercises the residual-only assembly path: with a full Newton
    # `nlsolve!` never asks for a residual without a tangent. That path is not a subset of the
    # linearization -- it re-solves the local problems and it reaches the materials through
    # `stress_function` rather than `stress_and_tangent`.
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
