
using Thunderbolt, LinearSolve
using FerriteMultigrid: FerriteMultigrid, GridHierarchy, GaussSeidel
using IterativeSolvers: gauss_seidel!
using AlgebraicMultigrid
using TimerOutputs
import LinearAlgebra: diag, mul!
using HybridSmoothers
import KernelAbstractions as KA
using FerriteOperators

TimerOutputs.enable_debug_timings(AlgebraicMultigrid)  #src
TimerOutputs.enable_debug_timings(FerriteMultigrid)    #src
reset_timer!()                                         #src

# mesh = generate_ring_mesh(40, 6, 6) # 300000 dof example
mesh = generate_ring_mesh(32, 4, 4)
mesh = generate_ring_mesh(16, 1, 1) #src
gh = GridHierarchy(mesh.grid, 1)
fine_mesh = to_mesh(gh.grids[end])
coordinate_system = compute_midmyocardial_section_coordinate_system(fine_mesh)
microstructure = create_microstructure_model(
    coordinate_system,
    LagrangeCollection{1}()^3,
    ODB25LTMicrostructureParameters(),
);
weak_boundary_conditions = (RobinBC(100.0, "Epicardium"),)
passive_material_model = Guccione1991PassiveModel()
active_material_model  = Guccione1993ActiveModel();
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
)
sarcomere_model = CaDrivenInternalSarcomereModel(ConstantStretchModel(), calcium_field);
active_stress_model = ActiveStressModel(
    passive_material_model,
    active_material_model,
    sarcomere_model,
    microstructure,
)
mechanical_model = QuasiStaticModel(:displacement, active_stress_model, weak_boundary_conditions)
spatial_discretization_method = FiniteElementDiscretization(
    Dict(:displacement => LagrangeCollection{2}()^3),
    Dirichlet[],
    String[],
    PerColorAssemblyStrategy(PolyesterDevice()),
)
quasistaticform = semidiscretize(mechanical_model, spatial_discretization_method, fine_mesh);

dt₀ = 1.0
tspan = (0.0, 500.0)
dtvis = 5.0;
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
        max_iter     = 25,
        inner_solver = KrylovMGSolver(
            KrylovJL_GMRES(; verbose = 0),
            ChainedMGPrecon(
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
                    # presmoother  = HybridSmoothers.ChebyshevFirst(),
                    # postsmoother = HybridSmoothers.ChebyshevFirst(),
                ),
                GMGPrecon(gh;
                    presmoother  = AlgebraicMultigrid.SOR(1.3, AlgebraicMultigrid.ForwardSweep(), 2),
                    postsmoother = AlgebraicMultigrid.SOR(1.3, AlgebraicMultigrid.BackwardSweep(), 2),
                ),
            );
            maxiters = 100,
        ),
        ## inner_solver = LinearSolve.UMFPACKFactorization()
        # Eisenstat-Walker adaptive linear solver tolerance (Algorithm 2, E&W 1996).
        # The linear solve always starts from x₀=0 so that η is applied relative to ‖b‖.
        # ηₘₐₓ=0.5 ensures GMRES always does meaningful work even when Newton converges slowly.
        forcing = EisenstatWalkerForcing(ηₘₐₓ = 0.5),
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

print_timer() #src

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
