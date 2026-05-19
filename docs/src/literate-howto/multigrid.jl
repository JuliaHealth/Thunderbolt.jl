
using Thunderbolt, LinearSolve
using FerriteMultigrid: FerriteMultigrid, GridHierarchy, GaussSeidel
using IterativeSolvers: gauss_seidel!
using AlgebraicMultigrid
using TimerOutputs
import LinearAlgebra: diag, mul!
using HybridSmoothers
import KernelAbstractions as KA

TimerOutputs.enable_debug_timings(AlgebraicMultigrid)
TimerOutputs.enable_debug_timings(FerriteMultigrid)
reset_timer!()

tspan = (0.0, 300.0)
Δt = 100.0

# mesh = generate_mesh(Hexahedron, (10, 10, 2), Ferrite.Vec{3}((0.0,0.0,0.0)), Ferrite.Vec{3}((1.0, 1.0, 0.2)))
# gh = GridHierarchy(mesh.grid, 1)
# fine_mesh = to_mesh(gh.grids[end])
# coordinate_system = CartesianCoordinateSystem{3}()
# microstructure = OrthotropicMicrostructureModel(
#     ConstantCoefficient(Vec((1.0, 0.0, 0.0))),
#     ConstantCoefficient(Vec((0.0, 1.0, 0.0))),
#     ConstantCoefficient(Vec((0.0, 0.0, 1.0))),
# )
# weak_boundary_conditions = (RobinBC(100.0, "left"),)


mesh = generate_ring_mesh(32, 4, 4)
# gh = GridHierarchy(mesh.grid, 1)
# fine_mesh = to_mesh(gh.grids[end])
fine_mesh = mesh
coordinate_system = compute_midmyocardial_section_coordinate_system(fine_mesh)
microstructure = create_microstructure_model(
    coordinate_system,
    LagrangeCollection{1}()^3,
    ODB25LTMicrostructureParameters(),
);
weak_boundary_conditions = (RobinBC(100.0, "Epicardium"),)


# Now we describe the model which we want to use.
# The models provided by Thunderbolt are designed to be highly modular, so you can quickly swap out individual
# component or compose models with each other.
# For the active stress formulation we need first the active and passive material models.
# For this tutorial we use the models described by Guccione.
passive_material_model = Guccione1991PassiveModel()
active_material_model  = Guccione1993ActiveModel();

# Furthermore we need to describe the calcium field and associate it with the sarcomere model.
# To simplify this tutorial we will use an analytical calcium profile.
# Note that we can also use experimental data or a precomputed calcium profile here, too, by simply changing the function implementation below.
function calcium_profile_function(x,t)
    linear_interpolation(t,y1,y2,t1,t2) = y1 + (t-t1) * (y2-y1)/(t2-t1)
    ca_peak(x)                          = 1.0
    if 0 ≤ t ≤ 300.0
        return linear_interpolation(t,        0.0, ca_peak(x),   0.0, 300.0)
    elseif t ≤ 500.0
        return linear_interpolation(t, ca_peak(x),        0.0, 300.0, 500.0)
    else
        return 0.0
    end
end
calcium_field = AnalyticalCoefficient(
    calcium_profile_function,
    coordinate_system,
);

# We will use for a very simple sarcomere model which is constant in the calcium concentration.
# Note that a using a sarcomere model which has evoluation equations or rate-dependent terms will require different solvers.
sarcomere_model = CaDrivenInternalSarcomereModel(ConstantStretchModel(), calcium_field);

# Now we have everything set to describe our active stress model by passing all the model components into it.
active_stress_model = ActiveStressModel(
    passive_material_model,
    active_material_model,
    sarcomere_model,
    microstructure,
);

# We finalize the mechanical model by assigning a symbol to identify the unknown solution field and connect the active stress model with the weak boundary conditions.
mechanical_model = QuasiStaticModel(:displacement, active_stress_model, weak_boundary_conditions)

# !!! tip
#     A full list of all models can be found in the [API reference](@ref models-api).

# We now need to transform the space-time problem into a time-dependent problem by discretizing it spatially.
# This can be accomplished by the function semidiscretize, which takes a model and the disretization technique.
# We use second-order Lagrange polynomials for the displacement field, which is required for polynomial multigrid
# (p-MG coarsens from order 2 to order 1).
# !!! danger
#     The discretization API does now play well with multiple domains right now and will be updated with a possible breaking change in future releases.
spatial_discretization_method = FiniteElementDiscretization(
    Dict(:displacement => LagrangeCollection{2}()^3),
)
quasistaticform = semidiscretize(mechanical_model, spatial_discretization_method, fine_mesh);

# The remaining code is very similar to how we use SciML solvers.
# We first define our time domain, initial time step length and some dt for visualization.
dt₀ = 10.0
tspan = (0.0, 500.0)
dtvis = 25.0;
# This speeds up the CI # hide
tspan = (0.0, dtvis);   # hide

# Then we setup the problem.
# Since we have no time dependence in our active stress model the correct problem here is a quasistatic problem.
problem = QuasiStaticProblem(quasistaticform, tspan);

# Next we define the time stepper.
# Since there are no time derivatives appearing in our formulation we have to opt for a homotopy path method, which solve the time depentent problems adaptively.
# As our non-linear solver we choose the standard Newton-Raphson method.
# For the linear system inside each Newton step we use chained polynomial + geometric multigrid as preconditioner.
# The polynomial hierarchy coarsens from p=2 to p=1; the geometric hierarchy then coarsens
# the p=1 problem using the grid levels in `gh`.  Both DofHandler hierarchies are assembled automatically.
# For the theory behind homotopy path methods we refer to [the corresponding theory manual on homotopy path methods](@ref theory_homotopy-path-methods)
timestepper = HomotopyPathSolver(
    NewtonRaphsonSolver(
        max_iter     = 10,
        inner_solver = KrylovMGSolver(
            # gmres_restart matches maxiters so GMRES never restarts and loses its Krylov
            # subspace — important for a non-symmetric (Robin BC) system with a moderately
            # strong preconditioner.  If memory is a concern, reduce restart and accept
            # slightly more iterations.
            KrylovJL_GMRES(; gmres_restart = 150, verbose = 3),
            # ChainedMGPrecon(
                PMGPrecon(;
                    # Asymmetric V-cycle: forward SOR pre-smoother + backward SOR post-smoother.
                    # This is the canonical optimal V-cycle structure and is 3x cheaper per
                    # V-cycle than symmetric SSOR (4 triangular solves vs 12 for GaussSeidel
                    # with iter=3 on both sides).
                    #
                    # SOR with ω=1.3 gives a better per-sweep smoothing factor than GS (ω=1),
                    # so 2 forward + 2 backward sweeps is roughly equivalent in smoothing
                    # quality to 3 symmetric GS sweeps at 1/3 the cost.
                    #
                    # Sequential sweep ordering (not parallel L1-GS) is essential here:
                    # the Guccione material has 10-100x fiber/cross-fiber anisotropy whose
                    # coupled modes can only be attenuated by propagating information along
                    # the DOF ordering in each sweep.
                    presmoother  = AlgebraicMultigrid.SOR(1.3, AlgebraicMultigrid.ForwardSweep(), 2),
                    postsmoother = AlgebraicMultigrid.SOR(1.3, AlgebraicMultigrid.BackwardSweep(), 2),
                ),
            #     GMGPrecon(gh;
            #         presmoother  = AlgebraicMultigrid.GaussSeidel(; iter = 3),
            #         postsmoother = AlgebraicMultigrid.GaussSeidel(; iter = 3),
            #     ),
            # );
            maxiters = 150,
        ),
        ## inner_solver = LinearSolve.UMFPACKFactorization()
    )
);

# Now we initialize our time integrator as usual.
integrator = init(problem, timestepper, dt=dt₀, verbose=true, adaptive=true, dtmax=25.0);
@info length(integrator.u)

# Finally we solve the problem in time.
io = ParaViewWriter("multigrid");
for (u, t) in TimeChoiceIterator(integrator, tspan[1]:dtvis:tspan[2])
    @info t
    (; dh) = problem.f
    Thunderbolt.store_timestep!(io, t, dh.grid) do file
    Thunderbolt.store_timestep_field!(io, t, dh, u, :displacement)
    end
end;

print_timer()

# !!! tip
#     If you want to see more details of the solution process launch Julia with Thunderbolt as debug module:
#     ```
#     JULIA_DEBUG=Thunderbolt julia --project --threads=auto my_simulation_runner.jl
#     ```

#md # ## References
#md # ```@bibliography
#md # Pages = ["cm01_simple-active-stress.md"]
#md # Canonical = false
#md # ```

#md # ## [Plain program](@id mechanics-tutorial_simple-active-stress-plain-program)
#md #
#md # Here follows a version of the program without any comments.
#md # The file is also available here: [`cm01_simple-active-stress.jl`](cm01_simple-active-stress.jl).
#md #
#md # ```julia
#md # @__CODE__
#md # ```
