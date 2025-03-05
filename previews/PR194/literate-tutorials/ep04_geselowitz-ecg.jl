# # [Electrophysiology Tutorial 4: Geselowitz ECG with Monodomain Model](@id ep-tutorial_geselowitz-ecg)
#
# !!! todo
#     Show computed ECG.
#
# This tutorial shows how to setup ECG problems with monodomain models as the source and compute the QRS complex in a simple toy problem.
#
# !!! todo
#     Provide context.
#
# ## Commented Program
using Thunderbolt, LinearAlgebra, StaticArrays

# !!! todo
#     The initializer API is not yet finished and hence we deconstruct stuff here manually.
#     Please note that this method is quite fragile w.r.t. to many changes you can make in the code below.
function steady_state_initializer!(u₀, f::GenericSplitFunction)
    ## TODO cleaner implementation. We need to extract this from the types or via dispatch.
    heatfun = f.functions[1]
    heat_dofrange = f.solution_indices[1]
    odefun = f.functions[2]
    ionic_model = odefun.ode

    φ₀ = @view u₀[heat_dofrange];
    ## TODO extraction these via utility functions
    dh = heatfun.dh
    s₀flat = @view u₀[(ndofs(dh)+1):end];
    ## Should not be reshape but some array of arrays fun
    s₀ = reshape(s₀flat, (ndofs(dh), Thunderbolt.num_states(ionic_model)-1));
    default_values = Thunderbolt.default_initial_state(ionic_model)

    φ₀ .= default_values[1]
    for i ∈ 1:(Thunderbolt.num_states(ionic_model)-1)
        s₀[:, i] .= default_values[i+1]
    end
end

# We start by defining a custom activation function
Base.@kwdef struct UniformEndocardialActivation <: Function
    transmural_depth::Float64 = 0.15
end
function (p::UniformEndocardialActivation)(x::Vec{3}, t)
    τᶠ = 0.25
    ## TODO source for this
    if t ≤ 2.0 && x[1] < p.transmural_depth
        return 0.5/τᶠ * exp(t/τᶠ)
    else
        return 0.0
    end
end
protocol = Thunderbolt.AnalyticalTransmembraneStimulationProtocol(
    AnalyticalCoefficient(
        UniformEndocardialActivation(),
        CoordinateSystemCoefficient(CartesianCoordinateSystem{3}())
    ),
    [SVector((-Inf, Inf))],
)

# We also generate both meshes
num_elements_heart = (32,16,16)
num_elements_heart = (8,4,4) # hide
heart_mesh = generate_mesh(Tetrahedron, num_elements_heart, Vec((1.5, 1.5, 0.0)), Vec((5.5, 3.5, 2.0)))
num_elements_torso = (56,40,28)
num_elements_torso = (14,10,7) # hide
torso_mesh = generate_mesh(Hexahedron,  num_elements_torso, Vec((0.0, 0.0, 0.0)), Vec((7.0, 5.0, 3.5)))

# Then we place some electrodes and leads.
ground_vertex = Thunderbolt.get_closest_vertex(Vec(0.0, 0.0, 0.0), torso_mesh)
leads = [
    [Vec( 0.,   0.,  1.5), Vec( 7.,   0.,  1.5)],
    [Vec( 3.5,  0.,  1.5), Vec( 3.5,  5.,  1.5)],
]

# For our toy problem we use a very simple microstructure.
microstructure = OrthotropicMicrostructureModel(
    ConstantCoefficient((Vec(0.0,0.0,1.0))),
    ConstantCoefficient((Vec(0.0,1.0,0.0))),
    ConstantCoefficient((Vec(1.0,0.0,0.0))),
)

# With the microstructure we setup the diffusion tensor field in spectral form.
# !!! todo
#     citation
κ₁ = 0.17 * 0.62 / (0.17 + 0.62)
κᵣ = 0.019 * 0.24 / (0.019 + 0.24)
diffusion_tensor_field = SpectralTensorCoefficient(
    microstructure,
    ConstantCoefficient(SVector(κ₁, κᵣ, κᵣ))
)
# Now we setup our monodomain solver as usual.
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
u₀ = zeros(Float64, OS.function_size(heart_odeform))
steady_state_initializer!(u₀, heart_odeform)
dt₀ = 0.01
dtvis = 0.5
Tₘₐₓ = 50.0
Tₘₐₓ = dtvis # hide
tspan = (0.0, Tₘₐₓ)
problem = OS.OperatorSplittingProblem(heart_odeform, u₀, tspan)
timestepper = OS.LieTrotterGodunov((
    BackwardEulerSolver(),
    ForwardEulerCellSolver(),
))
integrator = init(problem, timestepper, dt=dt₀, verbose=true)

# Now that the time integrator is ready we setup the ECG problem.
torso_mesh_κᵢ = ConstantCoefficient(1.0)
torso_mesh_κ  = ConstantCoefficient(1.0)
# !!! todo
#     Show how to transfer `diffusion_tensor_field` onto the torso mesh.
geselowitz_ecg = Thunderbolt.Geselowitz1989ECGLeadCache(
    heart_odeform,
    torso_mesh,
    torso_mesh_κᵢ,
    torso_mesh_κ,
    leads;
    ground = Thunderbolt.OrderedSet([ground_vertex])
)
# !!! todo
#     Improve the ECG API to not spill all the internals. :)

# We compute the ECG online as follows.
io = ParaViewWriter("ep04_ecg")
for (u, t) in TimeChoiceIterator(integrator, tspan[1]:dtvis:tspan[2])
    dh = heart_odeform.functions[1].dh
    φ = u[heart_odeform.solution_indices[1]]
    store_timestep!(io, t, dh.grid) do file
        Thunderbolt.store_timestep_field!(file, t, dh, φ, :φₘ)
    end

    ## To compute the ECG we just need to update the ecg cache
    Thunderbolt.update_ecg!(geselowitz_ecg, φ)
    ## which then allows us to evaluate the leads like this
    electrode_values = Thunderbolt.evaluate_ecg(geselowitz_ecg)
    @info "$t: Lead 1=$(electrode_values[1]) | Lead 2= $(electrode_values[2])"
end


#md # ## References
#md # ```@bibliography
#md # Pages = ["ep04_geselowitz-ecg.md"]
#md # Canonical = false
#md # ```

#md # ## [Plain program](@id ep-tutorial_geselowitz-ecg-plain-program)
#md #
#md # Here follows a version of the program without any comments.
#md # The file is also available here: [`ep04_geselowitz-ecg.jl`](ep04_geselowitz-ecg.jl).
#md #
#md # ```julia
#md # @__CODE__
#md # ```
