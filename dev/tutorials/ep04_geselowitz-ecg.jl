using Thunderbolt, LinearAlgebra, StaticArrays, OrdinaryDiffEqOperatorSplitting

Base.@kwdef struct UniformEndocardialActivation <: Function
    transmural_depth::Float64 = 0.15
end
function (p::UniformEndocardialActivation)(x::Vec{3}, t)
    τᶠ = 0.25
    # TODO source for this
    if t ≤ 2.0 && x[1] < p.transmural_depth
        return 0.5/τᶠ * exp(t/τᶠ)
    else
        return 0.0
    end
end
protocol = Thunderbolt.AnalyticalTransmembraneStimulationProtocol(
    AnalyticalCoefficient(
        UniformEndocardialActivation(),
        CartesianCoordinateSystem{3}()
    ),
    [SVector((-Inf, Inf))],
)

num_elements_heart = (32,16,16)
num_elements_heart = (8,4,4) # hide
heart_mesh = generate_mesh(Tetrahedron, num_elements_heart, Vec((1.5, 1.5, 0.0)), Vec((5.5, 3.5, 2.0)))
num_elements_torso = (56,40,28)
num_elements_torso = (14,10,7) # hide
torso_mesh = generate_mesh(Hexahedron,  num_elements_torso, Vec((0.0, 0.0, 0.0)), Vec((7.0, 5.0, 3.5)))

ground_vertex = Thunderbolt.get_closest_vertex(Vec(0.0, 0.0, 0.0), torso_mesh)
leads = [
    [Vec( 0.,   0.,  1.5), Vec( 7.,   0.,  1.5)],
    [Vec( 3.5,  0.,  1.5), Vec( 3.5,  5.,  1.5)],
]

microstructure = OrthotropicMicrostructureModel(
    ConstantCoefficient((Vec(0.0,0.0,1.0))),
    ConstantCoefficient((Vec(0.0,1.0,0.0))),
    ConstantCoefficient((Vec(1.0,0.0,0.0))),
)

κ₁ = 0.17 * 0.62 / (0.17 + 0.62)
κᵣ = 0.019 * 0.24 / (0.019 + 0.24)
diffusion_tensor_field = SpectralTensorCoefficient(
    microstructure,
    ConstantCoefficient(SVector(κ₁, κᵣ, κᵣ))
)

cellmodel = Thunderbolt.PCG2019()
heart_model = MonodomainModel(
    ConstantCoefficient(1.0),
    ConstantCoefficient(1.0),
    diffusion_tensor_field,
    protocol,
    cellmodel,
    :φₘ, :s
)
heart_odeform = semidiscretize(
    ReactionDiffusionSplit(heart_model),
    FiniteElementDiscretization(Dict(:φₘ => LagrangeCollection{1}())),
    heart_mesh,
)

u₀ = create_initial_condition(heart_odeform)
dt₀ = 0.01
dtvis = 0.5
Tₘₐₓ = 50.0
Tₘₐₓ = dtvis # hide
tspan = (0.0, Tₘₐₓ)
problem = OperatorSplittingProblem(heart_odeform, u₀, tspan)
timestepper = LieTrotterGodunov((
    BackwardEulerSolver(),
    ForwardEulerCellSolver(),
))
integrator = init(problem, timestepper, dt=dt₀, verbose=true)

torso_mesh_κᵢ = ConstantCoefficient(1.0)
torso_mesh_κ  = ConstantCoefficient(1.0)

geselowitz_ecg = Thunderbolt.Geselowitz1989ECGLeadCache(
    heart_odeform,
    torso_mesh,
    torso_mesh_κᵢ,
    torso_mesh_κ,
    leads;
    ground = Thunderbolt.OrderedSet([ground_vertex])
)

io = ParaViewWriter("ep04_ecg")
φₘ = solution_variable(heart_odeform, :φₘ)
for (u, t) in TimeChoiceIterator(integrator, tspan[1]:dtvis:tspan[2])
    store_timestep!(io, t, heart_mesh) do file
        Thunderbolt.store_timestep_field!(file, t, u, φₘ)
    end

    # To compute the ECG we just need to update the ecg cache
    Thunderbolt.update_ecg!(geselowitz_ecg, getvariable(u, φₘ))
    # which then allows us to evaluate the leads like this
    electrode_values = Thunderbolt.evaluate_ecg(geselowitz_ecg)
    @info "$t: Lead 1=$(electrode_values[1]) | Lead 2= $(electrode_values[2])"
end

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
