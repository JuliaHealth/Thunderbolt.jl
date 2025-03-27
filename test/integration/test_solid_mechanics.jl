using Thunderbolt

struct TestCalciumHatField end
Thunderbolt.setup_coefficient_cache(coeff::TestCalciumHatField, ::QuadratureRule, ::SubDofHandler) = coeff
Thunderbolt.evaluate_coefficient(coeff::TestCalciumHatField, cell_cache::CellCache, qp::QuadraturePoint, t) = t/1000.0 < 0.5 ? 2.0*t/1000.0 : 2.0-2.0*t/1000.0

function test_solve_contractile_cuboid(mesh, constitutive_model, timestepper, subdomains = [""])
    tspan = timestepper isa BackwardEulerSolver ? (0.0,10.0) : (0.0,300.0)
    Δt    = timestepper isa BackwardEulerSolver ? 2.0 : 100.0

    # Clamp three sides
    dbcs = [
        Dirichlet(:d, getfacetset(mesh, "left"), (x,t) -> [0.0], [1])
        Dirichlet(:d, getfacetset(mesh, "front"), (x,t) -> [0.0], [2])
        Dirichlet(:d, getfacetset(mesh, "bottom"), (x,t) -> [0.0], [3])
        Dirichlet(:d, Set([1]), (x,t) -> [0.0, 0.0, 0.0], [1, 2, 3])
    ]

    quasistaticform = semidiscretize(
        QuasiStaticModel(:d, constitutive_model, (
            NormalSpringBC(0.0, "right"),
            ConstantPressureBC(0.0, "back"),
            PressureFieldBC(ConstantCoefficient(0.0),"top")
        )),
        FiniteElementDiscretization(
            Dict(:d => LagrangeCollection{1}()^3),
            dbcs,
            subdomains,
        ),
        mesh
    )

    problem = QuasiStaticProblem(quasistaticform, tspan)
    Thunderbolt.default_initial_condition!(problem.u0, problem.f)

    # Create sparse matrix and residual vector
    integrator = init(problem, timestepper, dt=Δt, verbose=true, adaptive=!(timestepper isa BackwardEulerSolver))
    u₀ = copy(integrator.u)
    solve!(integrator)
    @test integrator.sol.retcode == DiffEqBase.ReturnCode.Success
    @test integrator.u ≉ u₀

    return integrator
end

function test_solve_contractile_ideal_lv(mesh, constitutive_model)
    tspan = (0.0,300.0)
    Δt = 100.0

    # Clamp three sides
    dbcs = [
        Dirichlet(:d, getnodeset(mesh, "MyocardialAnchor1"), (x,t) -> (0.0, 0.0, 0.0), [1,2,3]),
        Dirichlet(:d, getnodeset(mesh, "MyocardialAnchor2"), (x,t) -> (0.0, 0.0), [2,3]),
        Dirichlet(:d, getnodeset(mesh, "MyocardialAnchor3"), (x,t) -> (0.0,), [3]),
        Dirichlet(:d, getnodeset(mesh, "MyocardialAnchor4"), (x,t) -> (0.0,), [3])
    ]

    quasistaticform = semidiscretize(
        QuasiStaticModel(:d, constitutive_model, (
            NormalSpringBC(0.1, "Epicardium"),
            NormalSpringBC(0.1, "Base"),
            PressureFieldBC(ConstantCoefficient(0.01),"Endocardium")
        )),
        FiniteElementDiscretization(
            Dict(:d => LagrangeCollection{1}()^3),
            dbcs,
        ),
        mesh
    )

    problem = QuasiStaticProblem(quasistaticform, tspan)

    # Create sparse matrix and residual vector
    timestepper = HomotopyPathSolver(
        NewtonRaphsonSolver(inner_solver=Thunderbolt.LinearSolve.UMFPACKFactorization(), max_iter=10, tol=1e-10)
    )
    integrator = init(problem, timestepper, dt=Δt, verbose=true, maxiters=25)
    u₀ = copy(integrator.u)
    solve!(integrator)
    @test integrator.sol.retcode == DiffEqBase.ReturnCode.Success
    @test integrator.u ≉ u₀

    return integrator
end

# Smoke tests that things do not crash and that things do at least something
@testset "Contracting cuboid" begin
    mesh = generate_mesh(Hexahedron, (10, 10, 2), Ferrite.Vec{3}((0.0,0.0,0.0)), Ferrite.Vec{3}((1.0, 1.0, 0.2)))
    # mesh = generate_mesh(Hexahedron, (1, 1, 1), Ferrite.Vec{3}((0.0,0.0,0.0)), Ferrite.Vec{3}((1.0, 1.0, 0.2)))

    microstructure_model = OrthotropicMicrostructureModel(
        ConstantCoefficient(Vec((1.0, 0.0, 0.0))),
        ConstantCoefficient(Vec((0.0, 1.0, 0.0))),
        ConstantCoefficient(Vec((0.0, 0.0, 1.0))),
    )

    newton = NewtonRaphsonSolver(inner_solver=Thunderbolt.LinearSolve.UMFPACKFactorization(), max_iter=10, tol=1e-10)
    timestepper = BackwardEulerSolver(;
        inner_solver=Thunderbolt.MultiLevelNewtonRaphsonSolver(;
            newton=newton,
            # local_newton=NewtonRaphsonSolver(),
        )
    )
    i = test_solve_contractile_cuboid(mesh, ActiveStressModel(
        HolzapfelOgden2009Model(),
        SimpleActiveStress(;Tmax=2200.0),
        Thunderbolt.CaDrivenInternalSarcomereModel(
            Thunderbolt.RDQ20MFModel(),
            TestCalciumHatField(),
        ),
        microstructure_model
    ), timestepper)
    # VTKGridFile("SolidMechanicsIntegrationDebug", i.cache.stage.nlsolver.global_solver_cache.op.dh.grid) do vtk
    #     write_solution(vtk, i.cache.stage.nlsolver.global_solver_cache.op.dh, i.u)
    # end


    timestepper = HomotopyPathSolver(newton)
    test_solve_contractile_cuboid(mesh, ExtendedHillModel(
        HolzapfelOgden2009Model(),
        ActiveMaterialAdapter(LinearSpringModel()),
        GMKActiveDeformationGradientModel(),
        Thunderbolt.CaDrivenInternalSarcomereModel(
            PelceSunLangeveld1995Model(),
            TestCalciumHatField(),
        ),
        microstructure_model
    ), timestepper)

    test_solve_contractile_cuboid(mesh, GeneralizedHillModel(
        LinYinPassiveModel(),
        ActiveMaterialAdapter(LinYinActiveModel()),
        GMKIncompressibleActiveDeformationGradientModel(),
        Thunderbolt.CaDrivenInternalSarcomereModel(
            PelceSunLangeveld1995Model(),
            TestCalciumHatField(),
        ),
        microstructure_model
    ), timestepper)

    i = test_solve_contractile_cuboid(mesh, ActiveStressModel(
        HumphreyStrumpfYinModel(),
        SimpleActiveStress(),
        Thunderbolt.CaDrivenInternalSarcomereModel(
            PelceSunLangeveld1995Model(),
            TestCalciumHatField(),
        ),
        microstructure_model
    ), timestepper)
    VTKGridFile("SolidMechanicsIntegrationDebug", i.cache.inner_solver_cache.op.dh.grid) do vtk
        write_solution(vtk, i.cache.inner_solver_cache.op.dh, i.u)
    end

    mesh = to_mesh(generate_mixed_dimensional_grid_3D())

    test_solve_contractile_cuboid(mesh, ActiveStressModel(
        HumphreyStrumpfYinModel(),
        SimpleActiveStress(),
        Thunderbolt.CaDrivenInternalSarcomereModel(
            PelceSunLangeveld1995Model(),
            TestCalciumHatField(),
        ),
        microstructure_model
    ), timestepper, ["Ventricle"])
end

@testset "Idealized LV" begin
    grid = generate_ideal_lv_mesh(4,1,1)
    cs = compute_lv_coordinate_system(grid)
    @test !any(isnan.(cs.u_apicobasal))
    @test !any(isnan.(cs.u_transmural))
    @test !any(isnan.(cs.u_rotational))
    microstructure_model = create_simple_microstructure_model(cs, LagrangeCollection{1}()^3)

    test_solve_contractile_ideal_lv(grid, ExtendedHillModel(
        HolzapfelOgden2009Model(),
        ActiveMaterialAdapter(LinearSpringModel()),
        GMKActiveDeformationGradientModel(),
        Thunderbolt.CaDrivenInternalSarcomereModel(
            PelceSunLangeveld1995Model(),
            TestCalciumHatField(),
        ),
        microstructure_model
    ))

    test_solve_contractile_ideal_lv(grid, GeneralizedHillModel(
        LinYinPassiveModel(),
        ActiveMaterialAdapter(LinYinActiveModel()),
        GMKIncompressibleActiveDeformationGradientModel(),
        Thunderbolt.CaDrivenInternalSarcomereModel(
            PelceSunLangeveld1995Model(),
            TestCalciumHatField(),
        ),
        microstructure_model
    ))

    test_solve_contractile_ideal_lv(grid, ActiveStressModel(
        HumphreyStrumpfYinModel(),
        SimpleActiveStress(),
        Thunderbolt.CaDrivenInternalSarcomereModel(
            PelceSunLangeveld1995Model(),
            TestCalciumHatField(),
        ),
        microstructure_model
    ))

    # TODO test with filled LV
end

@testset "Viscoelasticity" begin
    mesh = generate_mesh(Hexahedron, (1,1,1))
    material = Thunderbolt.LinearMaxwellMaterial(
        E₀ = 70e3,
        E₁ = 20e3,
        μ  = 1e3,
        η₁ = 1e3,
        ν  = 0.3,
    )
    tspan = (0.0,1.0)
    Δt = 0.1

    # Clamp three sides
    dbcs = [
        Dirichlet(:d, getfacetset(mesh, "left"), (x,t) -> (0.0, 0.0, 0.0), [1,2,3]),
        Dirichlet(:d, getfacetset(mesh, "right"), (x,t) -> (0.1, 0.0, 0.0), [1,2,3]),
    ]

    quasistaticform = semidiscretize(
        QuasiStaticModel(:d, material, ()),
        FiniteElementDiscretization(
            Dict(:d => (LagrangeCollection{1}()^3 => QuadratureRuleCollection(1))),
            dbcs,
        ),
        mesh
    )
    @test solution_size(quasistaticform) == 3 * 8 + 1 * 6 # Symmetric Tensor has 6 components
    problem = QuasiStaticProblem(quasistaticform, tspan)

    # Create sparse matrix and residual vector
    timestepper = BackwardEulerSolver(;
        inner_solver=Thunderbolt.MultiLevelNewtonRaphsonSolver(;
            # global_newton=NewtonRaphsonSolver(),
            # local_newton=NewtonRaphsonSolver(),
        )
    )
    integrator = init(problem, timestepper, dt=Δt, verbose=true)
    # This setup is essentially a creep test in x direction, so we check for the invariants in there
    for (uprev, tprev, u, t) in Thunderbolt.SciMLBase.intervals(integrator)
        # Monotonicity of the solution in x direction
        @test uprev[3*8 + 1] ≤ u[3*8 + 1]
        # Linear problem => check that Newton converges in 1 step.
        @test length(integrator.cache.stage.nlsolver.global_solver_cache.Θks) == 1
    end
    @test integrator.sol.retcode == Thunderbolt.SciMLBase.ReturnCode.Success
    @test integrator.u[3*8 + 1] ≈ 0.05 atol=1e-5
    @test integrator.u[(3*8 + 2) : end] ≈ zeros(5) atol=1e-5
end
