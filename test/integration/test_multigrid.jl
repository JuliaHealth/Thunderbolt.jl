using Thunderbolt
using Test
using LinearSolve
using FerriteMultigrid
import SciMLBase

module TestMultigridHowto
mktempdir() do dir
    cd(dir) do
        include(joinpath(@__DIR__, "../../docs/src/literate-howto/multigrid.jl"))
    end
end
end

"""
A coarse-grid solver factory that counts how often it is asked for a solver. The multigrid setup
builds its coarse solver exactly once per hierarchy build, so this counts preconditioner builds.
"""
struct CountingCoarseSolverBuilder{B}
    inner::B
    builds::Base.RefValue{Int}
end

CountingCoarseSolverBuilder(inner) = CountingCoarseSolverBuilder(inner, Ref(0))

function (builder::CountingCoarseSolverBuilder)(A)
    builder.builds[] += 1
    return builder.inner(A)
end

"""
The activated cuboid of `test_solid_mechanics.jl`, at quadratic interpolation so that polynomial
multigrid has a hierarchy to coarsen, and solved with whichever global Newton is handed in.
"""
function solve_condensed_cuboid_p2(newton, Δt, tend)
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
        FiniteElementDiscretization(Dict(:d => LagrangeCollection{2}()^3); dbcs),
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

@testset "Multigrid preconditioner with condensed internal variables" begin
    # The condensed sarcomere reaches the linear solver through the multilevel Newton, which builds
    # its linear cache separately from the plain Newton the multigrid how-to exercises.
    Δt, tend = 2.5, 5.0
    nsteps = round(Int, tend / Δt)

    reference = solve_condensed_cuboid_p2(
        NewtonRaphsonSolver(inner_solver = UMFPACKFactorization(), max_iter = 20, tol = 1.0e-8),
        Δt,
        tend,
    )
    @test reference.sol.retcode == SciMLBase.ReturnCode.Success

    # An iterative linear solver caps the attainable Newton residual, hence the looser tolerance
    # here and the looser comparison below.
    function mg_newton(counter; simplified = false)
        return NewtonRaphsonSolver(
            max_iter = simplified ? 200 : 20,
            tol = 1.0e-5,
            simplified_newton = simplified,
            inner_solver = KrylovMGSolver(
                KrylovJL_GMRES(),
                PMGPrecon(; pcoarse_solver = counter),
                maxiters = 200,
            ),
        )
    end

    counter =
        CountingCoarseSolverBuilder(CachedLinearSolveCoarseSolverBuilder(UMFPACKFactorization()))
    integrator = solve_condensed_cuboid_p2(mg_newton(counter), Δt, tend)
    # Unpreconditioned GMRES does not converge on this problem, so reaching the reference solution
    # at all is what says the preconditioner is applied -- and it is rebuilt for every Jacobian.
    @test integrator.sol.retcode == SciMLBase.ReturnCode.Success
    @test integrator.u≈reference.u rtol=1.0e-4
    @test counter.builds[] > nsteps

    # A simplified Newton keeps the Jacobian, so the preconditioner built from it stays valid for
    # the whole step and must not be rebuilt: one build per full linearization, not per iteration.
    simplified_counter =
        CountingCoarseSolverBuilder(CachedLinearSolveCoarseSolverBuilder(UMFPACKFactorization()))
    simplified =
        solve_condensed_cuboid_p2(mg_newton(simplified_counter; simplified = true), Δt, tend)
    @test simplified.sol.retcode == SciMLBase.ReturnCode.Success
    @test simplified.u≈reference.u rtol=1.0e-4
    @test simplified_counter.builds[] == nsteps
end
