using Thunderbolt

function test_solve_passive_structure(constitutive_model)
    tspan = (0.0,1.0)
    Δt = 1.0
    mesh = generate_mesh(Hexahedron, (10, 10, 2), Ferrite.Vec{3}((0.0,0.0,0.0)), Ferrite.Vec{3}((1.0, 1.0, 0.2)))

    # Clamp three sides
    dbcs = [
        Dirichlet(:d, getfacetset(mesh, "left"), (x,t) -> [0.0], [1])
        Dirichlet(:d, getfacetset(mesh, "front"), (x,t) -> [0.0], [2])
        Dirichlet(:d, getfacetset(mesh, "bottom"), (x,t) -> [0.0], [3])
        Dirichlet(:d, Set([1]), (x,t) -> [0.0, 0.0, 0.0], [1, 2, 3])
        Dirichlet(:d, getfacetset(mesh, "right"), (x,t) -> [0.01t], [1])
        Dirichlet(:d, getfacetset(mesh, "top"), (x,t) -> [0.02t], [2])
        Dirichlet(:d, getfacetset(mesh, "back"), (x,t) -> [0.03t], [3])
    ]

    quasistaticform = semidiscretize(
        QuasiStaticModel(:d, constitutive_model, ()),
        FiniteElementDiscretization(
            Dict(:d => LagrangeCollection{1}()^3),
            dbcs,
        ),
        mesh
    )

    problem = QuasiStaticProblem(quasistaticform, tspan)

    # Create sparse matrix and residual vector
    timestepper = HomotopyPathSolver(
        NewtonRaphsonSolver(;max_iter=10)
    )
    integrator = init(problem, timestepper, dt=Δt, verbose=true)
    u₀ = copy(integrator.u)
    solve!(integrator)
    @test integrator.sol.retcode == DiffEqBase.ReturnCode.Success
    @test integrator.u ≉ u₀
    return integrator.u
end

@testset "Passive Structure" begin

u₁ = test_solve_passive_structure(
    PK1Model(
        HolzapfelOgden2009Model(),
        ConstantCoefficient(OrthotropicMicrostructure(
            Vec((1.0, 0.0, 0.0)),
            Vec((0.0, 1.0, 0.0)),
            Vec((0.0, 0.0, 1.0)),
        )),
    )
)
@test !iszero(u₁)

u₂ = test_solve_passive_structure(
    PrestressedMechanicalModel(
        PK1Model(
            HolzapfelOgden2009Model(),
            ConstantCoefficient(OrthotropicMicrostructure(
                Vec((1.0, 0.0, 0.0)),
                Vec((0.0, 1.0, 0.0)),
                Vec((0.0, 0.0, 1.0)),
            )),
        ),
        ConstantCoefficient(Tensor{2,3}((
            1.1,0.1,0.0,
            0.2,0.9,0.1,
            -0.1,0.0,1.0,
        ))),
    )
)

# The prestress should force a different solution
@test u₁ ≉ u₂

end
