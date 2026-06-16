using Thunderbolt, LinearSolve
using FerriteMultigrid: FerriteMultigrid, GridHierarchy
using AlgebraicMultigrid
using TimerOutputs
import LinearAlgebra: diag, mul!

import KernelAbstractions as KA
using FerriteOperators

mesh = generate_ring_mesh(32, 4, 4)
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

tspan = (0.0, dtvis);   # hide

problem = QuasiStaticProblem(quasistaticform, tspan);

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
        # Eisenstat-Walker adaptive linear solver tolerance (Algorithm 2, E&W 1996).
        # The linear solve always starts from x₀=0 so that η is applied relative to ‖b‖.
        # ηₘₐₓ=0.5 ensures GMRES always does meaningful work even when Newton converges slowly.
        forcing = EisenstatWalkerForcing(ηₘₐₓ = 0.5),
    )
);

integrator = init(problem, timestepper, dt=dt₀, verbose=true, adaptive=true, dtmax=25.0);
@info length(integrator.u)

io = ParaViewWriter("multigrid");
for (u, t) in TimeChoiceIterator(integrator, tspan[1]:dtvis:tspan[2])
    @info t
    (; dh) = problem.f
    Thunderbolt.store_timestep!(io, t, dh.grid) do file
    Thunderbolt.store_timestep_field!(io, t, dh, u, :displacement)
    end
end;

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
