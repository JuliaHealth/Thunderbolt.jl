using Thunderbolt, LinearSolve

mesh = generate_ideal_lv_mesh(11,2,5;
    inner_radius = 0.7,
    outer_radius = 1.0,
    longitudinal_upper = 0.2,
    apex_inner = 1.3,
    apex_outer = 1.5
);

coordinate_system = compute_lv_coordinate_system(mesh);

microstructure = create_simple_microstructure_model(
    coordinate_system,
    LagrangeCollection{1}()^3;
    endo_helix_angle = deg2rad(60.0),
    epi_helix_angle = deg2rad(-60.0),
);

passive_material_model = Guccione1991PassiveModel()
active_material_model  = Guccione1993ActiveModel();

function calcium_profile_function(x::LVCoordinate,t)
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
    CoordinateSystemCoefficient(coordinate_system),
);

sarcomere_model = ConstantStretchModel(;calcium_field);

active_stress_model = ActiveStressModel(
    passive_material_model,
    active_material_model,
    sarcomere_model,
    microstructure,
);

weak_boundary_conditions = (NormalSpringBC(1.0, "Epicardium"),)

mechanical_model = StructuralModel(:displacement, active_stress_model, weak_boundary_conditions)

spatial_discretization_method = FiniteElementDiscretization(
    Dict(:displacement => LagrangeCollection{1}()^3),
)
quasistaticform = semidiscretize(mechanical_model, spatial_discretization_method, mesh);

dt₀ = 10.0
tspan = (0.0, 500.0)
dtvis = 25.0;

tspan = (0.0, dtvis);   # hide

problem = QuasiStaticProblem(quasistaticform, tspan);

timestepper = HomotopyPathSolver(
    NewtonRaphsonSolver(
        max_iter=10,
        inner_solver=LinearSolve.UMFPACKFactorization(),
    )
);

integrator = init(problem, timestepper, dt=dt₀, verbose=true, adaptive=true, dtmax=25.0);

io = ParaViewWriter("CM01_simple_lv");
for (u, t) in TimeChoiceIterator(integrator, tspan[1]:dtvis:tspan[2])
    @info t
    (; dh) = problem.f
    Thunderbolt.store_timestep!(io, t, dh.grid) do file
    Thunderbolt.store_timestep_field!(io, t, dh, u, :displacement)
    end
end;

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
