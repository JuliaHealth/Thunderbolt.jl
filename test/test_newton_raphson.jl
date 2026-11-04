using Test, Thunderbolt, Tensors
import LinearSolve: UMFPACKFactorization
import LinearAlgebra: norm
import SparseArrays

# What `nlsolve!` asks of the stage it solves.
mutable struct CountingStage{S} <: Thunderbolt.AbstractStageFunction
    inner::S
    njacobian::Int
    nresidual::Int
end
CountingStage(inner) = CountingStage(inner, 0, 0)
Thunderbolt.getoperator(sf::CountingStage) = Thunderbolt.getoperator(sf.inner)
Thunderbolt.getfunction(sf::CountingStage) = Thunderbolt.getfunction(sf.inner)
function Thunderbolt.update_stage_linearization!(
    sf::CountingStage,
    r::AbstractVector,
    z::AbstractVector,
)
    sf.njacobian += 1
    return Thunderbolt.update_stage_linearization!(sf.inner, r, z)
end
function Thunderbolt.evaluate_stage_residual!(
    sf::CountingStage,
    r::AbstractVector,
    z::AbstractVector,
)
    sf.nresidual += 1
    return Thunderbolt.evaluate_stage_residual!(sf.inner, r, z)
end

# A stretched cuboid: a solve from the unloaded state takes several Newton steps.
function cuboid_cache(newton; load = 1.0)
    mesh = generate_mesh(Hexahedron, (2, 2, 2))
    microstructure = ConstantCoefficient(
        OrthotropicMicrostructure(Vec((1.0, 0.0, 0.0)), Vec((0.0, 1.0, 0.0)), Vec((0.0, 0.0, 1.0))),
    )
    dbcs = [
        Dirichlet(:d, getfacetset(mesh, "left"), (x, t) -> [0.0], [1])
        Dirichlet(:d, getfacetset(mesh, "front"), (x, t) -> [0.0], [2])
        Dirichlet(:d, getfacetset(mesh, "bottom"), (x, t) -> [0.0], [3])
        Dirichlet(:d, getfacetset(mesh, "right"), (x, t) -> [0.2load * t], [1])
        Dirichlet(:d, getfacetset(mesh, "top"), (x, t) -> [0.1load * t], [2])
        Dirichlet(:d, getfacetset(mesh, "back"), (x, t) -> [-0.1load * t], [3])
    ]
    f = semidiscretize(
        QuasiStaticModel(:d, PK1Model(HolzapfelOgden2009Model(), microstructure)),
        FiniteElementDiscretization(Dict(:d => LagrangeCollection{1}()^3); dbcs),
        mesh,
    )
    integrator = init(QuasiStaticProblem(f, (0.0, 1.0)), HomotopyPathSolver(newton), dt = 1.0)
    return f, integrator.cache
end

# Start the load step from the unloaded state, as `perform_step!` would.
function prepare_step!(f, cache)
    fill!(cache.uₙ, 0.0)
    Thunderbolt.update_constraints!(f, cache, 1.0)
    Thunderbolt.set_stage_parameters!(
        cache.stage_function,
        Thunderbolt._homotopy_stage_evaluation(f, 1.0),
    )
    return cache.uₙ
end

# The textbook loop: residual and Jacobian together at every iterate, the last one included.
function eager_newton!(u, sf, tol, ch)
    J = Thunderbolt.getJ(Thunderbolt.getoperator(sf))
    r = zeros(size(J, 1))
    steps = 0
    while true
        fill!(r, 0.0)
        Thunderbolt.update_stage_linearization!(sf, r, u)
        apply_zero!(J, r, ch)
        norm(r[Ferrite.free_dofs(ch)]) < tol && steps > 0 && return steps
        u .-= SparseArrays.sparse(J) \ r
        steps += 1
    end
end

@testset "Newton assembles the Jacobian only for a step that is taken" begin
    tol = 1e-10
    direct = NewtonRaphsonSolver(; inner_solver = UMFPACKFactorization(), tol, max_iter = 20)

    @testset "a converging solve" begin
        f, cache = cuboid_cache(direct)
        u = prepare_step!(f, cache)
        stage = CountingStage(cache.stage_function)
        @test Thunderbolt.nlsolve!(u, stage, cache.inner_solver_cache, 1.0)
        solution = copy(u)
        steps = cache.inner_solver_cache.iter
        @test steps ≥ 2
        @test stage.njacobian == steps
        @test stage.nresidual == steps

        reference = prepare_step!(f, cache)
        @test eager_newton!(reference, cache.stage_function, tol, Thunderbolt.getch(f)) == steps
        @test solution ≈ reference rtol = 1e-12
    end

    @testset "simplified Newton keeps the Jacobian of iteration 0" begin
        simplified = NewtonRaphsonSolver(;
            inner_solver = UMFPACKFactorization(),
            tol = 1e-6,
            max_iter = 200,
            simplified_newton = true,
            enforce_monotonic_convergence = false,
        )
        # A frozen Jacobian only reaches a mild load.
        f, cache = cuboid_cache(simplified; load = 0.05)
        u = prepare_step!(f, cache)
        stage = CountingStage(cache.stage_function)
        @test Thunderbolt.nlsolve!(u, stage, cache.inner_solver_cache, 1.0)
        @test stage.njacobian == 1
        @test stage.nresidual == cache.inner_solver_cache.iter
    end

    @testset "a solve which runs out of iterations" begin
        limited =
            NewtonRaphsonSolver(; inner_solver = UMFPACKFactorization(), tol = 1e-14, max_iter = 1)
        f, cache = cuboid_cache(direct)
        cache_limited = Thunderbolt.setup_solver_cache(cache.stage_function, limited)
        u = prepare_step!(f, cache)
        stage = CountingStage(cache.stage_function)
        @test !Thunderbolt.nlsolve!(u, stage, cache_limited, 1.0)
        @test last(cache_limited.Θks) == Inf
        @test stage.njacobian == stage.nresidual
    end
end
