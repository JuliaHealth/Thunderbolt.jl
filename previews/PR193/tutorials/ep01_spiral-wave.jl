using Thunderbolt, LinearSolve

mesh = generate_mesh(Quadrilateral, (2^6, 2^6), Vec{2}((0.0,0.0)), Vec{2}((2.5,2.5)));

Cₘ = ConstantCoefficient(1.0)
χ  = ConstantCoefficient(1.0)
κ  = ConstantCoefficient(SymmetricTensor{2,2,Float64}((4.5e-5, 0, 2.0e-5)));

stimulation_protocol = NoStimulationProtocol();

cell_model = Thunderbolt.FHNModel();

function spiral_wave_initializer!(u₀, f::GenericSplitFunction)
    # TODO cleaner implementation. We need to extract this from the types or via dispatch.
    heatfun = f.functions[1]
    heat_dofrange = f.solution_indices[1]
    odefun = f.functions[2]
    ionic_model = odefun.ode

    φ₀ = @view u₀[heat_dofrange];
    # TODO extraction these via utility functions
    dh = heatfun.dh
    s₀flat = @view u₀[(ndofs(dh)+1):end];
    # Should not be reshape but some array of arrays fun, because in general (e.g. for heterogeneous tissues) we cannot reshape into a matrix
    s₀ = reshape(s₀flat, (ndofs(dh), Thunderbolt.num_states(ionic_model)-1));

    for cell in CellIterator(dh)
        _celldofs = celldofs(cell)
        φₘ_celldofs = _celldofs[dof_range(dh, :φₘ)]
        # TODO query coordinate directly from the cell model
        coordinates = getcoordinates(cell)
        for (i, (x₁, x₂)) in zip(φₘ_celldofs,coordinates)
            if x₁ <= 1.25 && x₂ <= 1.25
                φ₀[i] = 1.0
            end
            if x₂ >= 1.25
                s₀[i,1] = 0.1
            end
        end
    end
end;

ep_model = MonodomainModel(
    Cₘ,
    χ,
    κ,
    stimulation_protocol,
    cell_model,
    :φₘ, :s,
);

split_ep_model = ReactionDiffusionSplit(ep_model);

spatial_discretization_method = FiniteElementDiscretization(
    Dict(:φₘ => LagrangeCollection{1}()),
)
odeform = semidiscretize(split_ep_model, spatial_discretization_method, mesh);

u₀ = zeros(Float32, OS.function_size(odeform))
spiral_wave_initializer!(u₀, odeform);

heat_timestepper = BackwardEulerSolver(
    inner_solver=KrylovJL_CG(atol=1e-6, rtol=1e-5),
);

cell_timestepper = AdaptiveForwardEulerSubstepper(;
    reaction_threshold=0.1,
);

timestepper = OS.LieTrotterGodunov((heat_timestepper, cell_timestepper));

dt₀ = 10.0
dtvis = 25.0;

tspan = (0.0, dtvis);   # hide

problem = OS.OperatorSplittingProblem(odeform, u₀, tspan);

integrator = OS.init(problem, timestepper, dt=dt₀);

io = ParaViewWriter("EP01_spiral_wave")
for (u, t) in OS.TimeChoiceIterator(integrator, tspan[1]:dtvis:tspan[2])
    (; dh) = odeform.functions[1]
    φ = u[odeform.solution_indices[1]]
    store_timestep!(io, t, dh.grid) do file
        Thunderbolt.store_timestep_field!(file, t, dh, φ, :φₘ)
    end
end;

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
