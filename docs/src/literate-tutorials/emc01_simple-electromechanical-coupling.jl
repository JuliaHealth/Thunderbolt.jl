
using Thunderbolt, LinearSolve, StaticArrays
using Thunderbolt: get_grid, _get_interpolation_from_discretization, add_subdomain!, ApproximationDescriptor
using DiffEqBase
import Thunderbolt: OS

include("../../../src/modeling/cells/tp2006m.jl")
include("../../../src/modeling/cells/piezo.jl")
include("../../../src/modeling/coupler/electromechanics.jl")
include("../../../src/modeling/cells/piezo/ogie2025.jl")
include("../../../src/modeling/cells/mahajan2008.jl")

function steady_state_initializer!(u₀, f::GenericSplitFunction)
    heatfun = f.functions[1]
    heat_dofrange = f.solution_indices[1]
    odefun = f.functions[2]
    ionic_model = odefun.ode

    φ₀ = @view u₀[heat_dofrange]
    dh = heatfun.dh
    s₀flat = @view u₀[(ndofs(dh)+1):end]
    s₀ = reshape(s₀flat, (ndofs(dh), Thunderbolt.num_states(ionic_model) - 1))
    default_values = Thunderbolt.default_initial_state(ionic_model)

    φ₀ .= default_values[1]
    for i ∈ 1:(Thunderbolt.num_states(ionic_model)-1)
        s₀[:, i] .= default_values[i+1]
    end
end

mesh_mechanical = generate_ideal_lv_mesh(20, 5, 20;
    inner_radius=0.7,
    outer_radius=1.0,
    longitudinal_upper=0.2,
    apex_inner=1.3,
    apex_outer=1.5
);

# Only refine transmurally since the nodal intergrid operator does not extrapolate.
mesh_electrical = generate_ideal_lv_mesh(20, 10, 20;
    inner_radius=0.7,
    outer_radius=1.0,
    longitudinal_upper=0.2,
    apex_inner=1.3,
    apex_outer=1.5
);

coordinate_system_mechanical = compute_lv_coordinate_system(mesh_mechanical);
coordinate_system_electrical = compute_lv_coordinate_system(mesh_electrical);

microstructure_mechanical = create_microstructure_model(
    coordinate_system_mechanical,
    LagrangeCollection{1}()^3,
    ODB25LTMicrostructureParameters(),
);

microstructure_electrical = create_microstructure_model(
    coordinate_system_electrical,
    LagrangeCollection{1}()^3,
    ODB25LTMicrostructureParameters(),
);

passive_material_model = LinYinPassiveModel()
active_material_model = ActiveMaterialAdapter(LinYinActiveModel());

dh_mechanical = coordinate_system_mechanical.dh
offsets = copy(dh_mechanical.cell_dofs_offset)
push!(offsets, length(dh_mechanical.cell_dofs) + 1)

ca_buf = Thunderbolt.ElementwiseData(zeros(length(dh_mechanical.cell_dofs)), offsets)

calcium_field = FieldCoefficient(
    ca_buf,
    LagrangeCollection{1}(),
);

passive_material_model = Guccione1991PassiveModel()
active_material_model = Guccione1993ActiveModel();

sarcomere_model = CaDrivenInternalSarcomereModel(PelceSunLangeveld1995Model(), calcium_field);

active_stress_model = ActiveStressModel(
    passive_material_model,
    active_material_model,
    sarcomere_model,
    microstructure_mechanical,
);

weak_boundary_conditions = (
    NormalSpringBC(1.0, "Epicardium"),
    NormalSpringBC(100.0, "Base"),
    # PressureFieldBC(Thunderbolt.AnalyticalCoefficientCache((x,t) -> t < 100 ? t * 5/100 : (t < 700 ? 5.0 : (t < 800 ? 5.0*((1+(700-t)/100)) : 0.0 )) ), "Endocardium"))# max(0.0, 0.7 * (1+π/2-t/10))
    PressureFieldBC(Thunderbolt.AnalyticalCoefficientCache((x, t) -> t < 50 ? 5.0 * sin(pi * t / 50) : 0.0), "Endocardium"))# max(0.0, 0.7 * (1+π/2-t/10))

mechanical_model = QuasiStaticModel(:displacement, active_stress_model, weak_boundary_conditions)

κ₁ = 0.17 * 0.62 / (0.17 + 0.62)
κᵣ = 0.019 * 0.24 / (0.019 + 0.24)
diffusion_tensor_field = SpectralTensorCoefficient(
    microstructure_electrical,
    ConstantCoefficient(SVector(κ₁, κᵣ, κᵣ))
)

cell_model = PiezoWrapper(
    TP2006MModel(Vector{Float64}),
    OgiePiezo1(),
    length(mesh_electrical.grid.nodes)
)

ep_model = MonodomainModel(
    ConstantCoefficient(1.0),
    ConstantCoefficient(1.0),
    diffusion_tensor_field,
    Thunderbolt.NoStimulationProtocol(),
    cell_model,
    :φₘ, :s
)

em_model = ElectroMechanicalCoupledModel(
    ReactionDiffusionSplit(ep_model),
    mechanical_model,
    ElectroMechanicalCoupler(1, 2)
)

mechanical_discretization = FiniteElementDiscretization(
    Dict(:displacement => LagrangeCollection{1}()^3),
)

ep_discretization = FiniteElementDiscretization(Dict(:φₘ => LagrangeCollection{1}()))

coupled_em_form = semidiscretize(em_model, (ep_discretization, mechanical_discretization), (mesh_electrical, mesh_mechanical));

# The remaining code is very similar to how we use SciML solvers.
# We first define our time domain, initial time step length and some dt for visualization.
dt₀ = 0.1
tspan = (0.0, 300.0)
dtvis = 1.0;
# This speeds up the CI # hide
# tspan = (0.0, 250.0);   # hide

mech_timestepper = HomotopyPathSolver(
    NewtonRaphsonSolver(
        max_iter=10,
        inner_solver=LinearSolve.UMFPACKFactorization(),
    )
);

heat_timestepper = BackwardEulerSolver(
    inner_solver=KrylovJL_CG(atol=1e-6, rtol=1e-5),
);

cell_timestepper = AdaptiveForwardEulerSubstepper(;
    reaction_threshold=0.1,
);

ep_timestepper = OS.LieTrotterGodunov((heat_timestepper, cell_timestepper));

timestepper = OS.LieTrotterGodunov((ep_timestepper, mech_timestepper));

u₀ = zeros(Float64, solution_size(coupled_em_form))
steady_state_initializer!((@view u₀[coupled_em_form.solution_indices[1]]), coupled_em_form.functions[1])

problem = OS.OperatorSplittingProblem(coupled_em_form, u₀, tspan);


# Now we initialize our time integrator as usual.
integrator = init(problem, timestepper, dt=dt₀, verbose=true, adaptive=true, dtmax=25.0);

# !!! todo
#     The post-processing API is not yet finished.
#     Please revisit the tutorial later to see how to post-process the simulation online.
#     Right now the solution is just exported into VTK, such that users can visualize the solution in e.g. ParaView.

# Finally we solve the problem in time.
io_mech = ParaViewWriter("EMC01_simple_lv_mech");
io_ep = ParaViewWriter("EMC01_simple_lv_ep");
for (u, t) in TimeChoiceIterator(integrator, tspan[1]:dtvis:tspan[2])
    dh_mech = problem.f.functions[2].dh
    Thunderbolt.store_timestep!(io_mech, t, dh_mech.grid) do file
        Thunderbolt.store_timestep_field!(io_mech, t, dh_mech, u[problem.f.solution_indices[2]], :displacement)
    end
    dh_ep = problem.f.functions[1].functions[1].dh
    Thunderbolt.store_timestep!(io_ep, t, dh_ep.grid) do file
        Thunderbolt.store_timestep_field!(io_ep, t, dh_ep, u[problem.f.functions[1].solution_indices[1]], :φₘ)
        Thunderbolt.store_timestep_field!(io_ep, t, dh_ep, cell_model.I4, :φₘ, "_I4")
    end
end;
