using Thunderbolt, Test, LinearSolve

struct TimeFunctionCoefficient{F}
    f::F
end
Thunderbolt.evaluate_coefficient(c::TimeFunctionCoefficient, cell, qp, time) = c.f(time)

@testset "Land2015 benchmark problem 1" begin
    for celltype in [
        Hexahedron,
        Tetrahedron
    ]

    mesh = generate_mesh(celltype, (25, 3, 3), Ferrite.Vec{3}((0.0,0.0,0.0)), Ferrite.Vec{3}((10.0, 1.0, 1.0)))
    order = 2

    # Still not converged...
    # mesh = generate_mesh(celltype, (120, 12, 12), Ferrite.Vec{3}((0.0,0.0,0.0)), Ferrite.Vec{3}((10.0, 1.0, 1.0)))
    # order = 1

    # passive_material_model = Thunderbolt.BioNeoHooekean() 
    passive_material_model = Guccione1991PassiveModel(;C₀=2.0, Bᶠᶠ=8.0, Bˢˢ=2.0, Bⁿⁿ=2.0, Bⁿˢ=1.0, Bᶠˢ=2.0, Bᶠⁿ=2.0, mpU=SimpleCompressionPenalty(100.0))
    # passive_material_model = HolzapfelOgden2009Model()

    spatial_discretization_method = FiniteElementDiscretization(
        Dict(:displacement => LagrangeCollection{order}()^3),
        [
            Dirichlet(:displacement, getfacetset(mesh, "left"), (x,t) -> (0.0, 0.0, 0.0), [1,2,3]),
        ],
    )

    constitutive_model = PK1Model(
        passive_material_model,
        ConstantCoefficient(OrthotropicMicrostructure(
            Vec((1.0, 0.0, 0.0)),
            Vec((0.0, 1.0, 0.0)),
            Vec((0.0, 0.0, 1.0)),
        )),
    )

    pressure_profile(t) = min(t,1.0) * 0.004
    pressure_profile_coefficient = TimeFunctionCoefficient(
        pressure_profile,
    )
    bcs = Thunderbolt.ConsistencyCheckWeakBoundaryCondition(
        PressureFieldBC(pressure_profile_coefficient, "bottom"),
        1e-7,
    )
    quasistaticform = semidiscretize(
        QuasiStaticModel(:displacement, constitutive_model, (bcs,)),
        spatial_discretization_method,
        mesh
    )

    problem = QuasiStaticProblem(quasistaticform, (0.0, 1.0))

    # Create sparse matrix and residual vector
    timestepper = HomotopyPathSolver(
        NewtonRaphsonSolver(;
            tol=1e-4,
            max_iter=10,
            inner_solver=LinearSolve.UMFPACKFactorization(),
            # monitor=Thunderbolt.VTKNewtonMonitor(;outdir="./land2015-newton-debug/"),
        )
    )
    integrator = init(problem, timestepper, dt=0.2, dtmax=0.2, adaptive=true, verbose=true, maxiters=100)

    solve!(integrator)
    @test integrator.sol.retcode == DiffEqBase.ReturnCode.Success

    peh_deflection = PointEvalHandler(mesh.grid, [Vec(10.0,0.5,1.0)])
    @test isapprox(evaluate_at_points(peh_deflection, integrator.cache.inner_solver_cache.op.dh, integrator.u)[1][3], 3.17; atol=0.02)

    # VTKGridFile("Land2015Problem1-$(celltype)", mesh.grid) do vtk
    #     write_solution(vtk, integrator.cache.inner_solver_cache.op.dh, integrator.u)
    # end

    end

    # @testset "Pressure FD1" begin
    #     Δ = 1e-6

    #     J = zeros(24,24)
    #     Jfd = zeros(24,24)
    #     J2 = zeros(24,24) 
    #     u = [0.0, 0.0, 0.0, 0.007235528439120246, 0.0060324697675344045, 0.015372134435205851, 0.0071031026769519156, 2.8930799182463867e-15, 0.01835586270885209, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -4.104130228714606e-5, -0.0001549268822641951, 0.015176858354351867, -3.724030034936616e-5, 1.95950293563689e-15, 0.01818622905111361, 0.0, 0.0, 0.0]
    #     u2 = zeros(24)
    #     r = zeros(24)
    #     r2 = zeros(24)
    #     p = 0.1

    #     qr = FacetQuadratureRule{RefHexahedron}(2)
    #     ip = Lagrange{RefHexahedron,1}()^3
    #     fv = FacetValues(qr, ip)

    #     Thunderbolt.reinit!(fv, first(CellIterator(integrator.cache.inner_solver_cache.op.dh)), 1)
    #     for qp in QuadratureIterator(fv)
    #         r .= 0.0
    #         Thunderbolt.assemble_facet_pressure_qp!(J, r, u, p, qp, fv)
    #         for i in 1:24
    #             fill!(r2, 0.0)
    #             u2 .= u
    #             u2[i] += Δ
    #             Thunderbolt.assemble_facet_pressure_qp!(J2, r2, u2, p, qp, fv)
    #             r2 -= r
    #             r2 /= Δ
    #             Jfd[:,i] .+= r2
    #         end
    #         @info norm(J)
    #         @info norm(Jfd)
    #         @info Jfd[:,end], J[:,end]
    #         @test maximum(abs.(Jfd .- J)) < Δ
    #     end
    # end
end
