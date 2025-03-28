using Thunderbolt
using Thunderbolt.TimerOutputs

TimerOutputs.enable_debug_timings(Thunderbolt)

mesh = generate_mesh(Hexahedron, (4,4,4))
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
        Dict(:d => (LagrangeCollection{1}()^3 => QuadratureRuleCollection(2))),
        dbcs,
    ),
    mesh
)
problem = QuasiStaticProblem(quasistaticform, tspan)

# Create sparse matrix and residual vector
timestepper = BackwardEulerSolver(;
    inner_solver=Thunderbolt.MultiLevelNewtonRaphsonSolver(;
        # global_newton=NewtonRaphsonSolver(),
        # local_newton=NewtonRaphsonSolver(),
    )
)
integrator = init(problem, timestepper, dt=Δt, verbose=true)

step!(integrator) # Precompile

TimerOutputs.reset_timer!()
solve!(integrator)
TimerOutputs.print_timer()

cc = first(CellIterator(mesh.grid))
F = one(Tensor{2,3})
coefficient = nothing
qr = QuadratureRule{RefHexahedron}(2)
element_cache = Thunderbolt.setup_element_cache(integrator.cache.stage.nlsolver.global_solver_cache.op.integrator.volume_model, qr, integrator.cache.stage.nlsolver.global_solver_cache.op.dh.subdofhandlers[1])
state_cache   = element_cache.internal_cache
qp = QuadraturePoint(1, qr.points[1])

using BenchmarkTools
@btime Thunderbolt.solve_local_constraint(F, coefficient, material, state_cache, cc, qp, 0.0)
@btime Thunderbolt._query_local_state(state_cache, cc, qp)
