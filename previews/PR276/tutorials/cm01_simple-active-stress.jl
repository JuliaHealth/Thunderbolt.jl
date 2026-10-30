using Thunderbolt, LinearSolve

mesh = generate_ideal_lv_mesh(11,2,5;
    inner_radius = 0.7,
    outer_radius = 1.0,
    longitudinal_upper = 0.2,
    apex_inner = 1.3,
    apex_outer = 1.5
);

coordinate_system = compute_lv_coordinate_system(mesh);

microstructure = create_microstructure_model(
    coordinate_system,
    LagrangeCollection{1}()^3,
    ODB25LTMicrostructureParameters(),
);

passive_material_model = Guccione1991PassiveModel()
active_material_model  = Guccione1993ActiveModel();

function calcium_profile_function(x #=::LVCoordinate=#,t)
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

sarcomere_model = CaDrivenInternalSarcomereModel(ConstantStretchModel(), calcium_field);

active_stress_model = ActiveStressModel(
    passive_material_model,
    active_material_model,
    sarcomere_model,
    microstructure,
);

weak_boundary_conditions = (RobinBC(1.0, "Epicardium"),)

mechanical_model = QuasiStaticModel(:displacement, active_stress_model, weak_boundary_conditions)

spatial_discretization_method = FiniteElementDiscretization(
    Dict(:displacement => LagrangeCollection{1}()^3),
)
quasistaticform = semidiscretize(mechanical_model, spatial_discretization_method, mesh);

dt₀ = 5.0
tspan = (0.0, 500.0)
dtvis = 10.0;

problem = QuasiStaticProblem(quasistaticform, tspan);

timestepper = HomotopyPathSolver(
    NewtonRaphsonSolver(
        max_iter=10,
        inner_solver=LinearSolve.UMFPACKFactorization(),
    )
);

integrator = init(problem, timestepper, dt=dt₀, verbose=true, adaptive=true, dtmax=25.0);

io = ParaViewWriter("CM01_simple_lv");
d = solution_variable(quasistaticform, :displacement)
for (u, t) in TimeChoiceIterator(integrator, tspan[1]:dtvis:tspan[2])
    Thunderbolt.store_timestep!(io, t, mesh) do file
        Thunderbolt.store_timestep_field!(io, t, u, d)
    end
end;

weak_boundary_conditions_viscous = (
    RobinBC(1.0, "Epicardium"),
    ViscousNormalSpringBC(0.1, "Epicardium"),
);

mechanical_model_viscous = QuasiStaticModel(
    :displacement,
    active_stress_model,
    weak_boundary_conditions_viscous,
)
quasistaticform_viscous =
    semidiscretize(mechanical_model_viscous, spatial_discretization_method, mesh);

problem_viscous = QuasiStaticProblem(quasistaticform_viscous, tspan)
timestepper_viscous = BackwardEulerSolver(
    inner_solver =  NewtonRaphsonSolver(
        max_iter = 10,
        inner_solver = LinearSolve.UMFPACKFactorization(),
    ),
);

controller = Deuflhard2004_B_DiscreteContinuationControllerVariant(; Θmin = 1/8, p = 1)
integrator_viscous = init(
    problem_viscous,
    timestepper_viscous,
    dt = dt₀,
    verbose = true,
    controller = controller,
    dtmax = 25.0,
);

io_viscous = ParaViewWriter("CM01_simple_lv_viscous");
d = solution_variable(quasistaticform_viscous, :displacement)
for (u, t) in TimeChoiceIterator(integrator_viscous, tspan[1]:dtvis:tspan[2])
    Thunderbolt.store_timestep!(io_viscous, t, mesh) do file
        Thunderbolt.store_timestep_field!(io_viscous, t, u, d)
    end
end;

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
