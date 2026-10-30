using Thunderbolt, LinearSolve, OrdinaryDiffEqOperatorSplitting

mesh = generate_mesh(Quadrilateral, (2^6, 2^6), Vec{2}((0.0,0.0)), Vec{2}((2.5,2.5)));

Cₘ = ConstantCoefficient(1.0)
χ  = ConstantCoefficient(1.0)
κ  = ConstantCoefficient(SymmetricTensor{2,2,Float64}((4.5e-5, 0, 2.0e-5)));

stimulation_protocol = NoStimulationProtocol();

cell_model = Thunderbolt.FHNModel();

ep_model = MonodomainModel(
    Cₘ,
    χ,
    κ,
    stimulation_protocol,
    cell_model,
    CartesianCoordinateSystem(mesh),
    :φₘ, :s,
);

split_ep_model = ReactionDiffusionSplit(ep_model);

spatial_discretization_method = FiniteElementDiscretization(
    Dict(:φₘ => LagrangeCollection{1}()),
)
odeform = semidiscretize(split_ep_model, spatial_discretization_method, mesh);

u₀ = create_initial_condition(odeform, Float32);

setvariable!(u₀, odeform, :φₘ) do x
    (x[1] ≤ 1.25 && x[2] ≤ 1.25) ? 1.0f0 : 0.0f0
end
setvariable!(u₀, odeform, :s) do x
    x[2] ≥ 1.25 ? 0.1f0 : 0.0f0
end;

heat_timestepper = BackwardEulerSolver(
    inner_solver=KrylovJL_CG(atol=1e-6, rtol=1e-5),
);

cell_timestepper = AdaptiveForwardEulerSubstepper(;
    reaction_threshold=0.1,
);

timestepper = LieTrotterGodunov((heat_timestepper, cell_timestepper));

dt₀   = 1.0
dtvis = 25.0
tspan = (0.0, 1000.0);

tspan = (0.0, dtvis);   # hide

problem = OperatorSplittingProblem(odeform, u₀, tspan);

integrator = init(problem, timestepper, dt=dt₀);

io = ParaViewWriter("EP01_spiral_wave")
φₘ = solution_variable(odeform, :φₘ)
for (u, t) in TimeChoiceIterator(integrator, tspan[1]:dtvis:tspan[2])
    store_timestep!(io, t, mesh) do file
        Thunderbolt.store_timestep_field!(file, t, u, φₘ)
    end
end;

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
