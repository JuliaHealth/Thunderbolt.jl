# # [Electrophysiology Tutorial 5: Eikonal Models](@id ep-tutorial_eikonal)
#
# !!! todo
#     Show activation timings.
#
# This tutorial shows how to solve Eikonal models and how to recover transemembrane potential fields.
#
# !!! todo
#     Provide context.
#
# ## Commented Program
using Thunderbolt, LinearAlgebra, StaticArrays, OrdinaryDiffEqOperatorSplitting

# !!! todo
#     The initializer API is not yet finished and hence we deconstruct stuff here manually.
#     Please note that this method is quite fragile w.r.t. to many changes you can make in the code below.
function steady_state_initializer!(u₀, f::Thunderbolt.EikonalCoupledODEFunction)
    ## TODO cleaner implementation. We need to extract this from the types or via dispatch.
    odefun = f.ode_function
    ionic_model = odefun.ode

    φ₀ = @view u₀[1:ndofs(f.eikonal_function)];
    ## TODO extraction these via utility functions
    s₀flat = @view u₀[(ndofs(f.eikonal_function)+1):end];
    ## Should not be reshape but some array of arrays fun
    s₀ = reshape(s₀flat, (ndofs(f.eikonal_function), Thunderbolt.num_states(ionic_model)-1));
    default_values = Thunderbolt.default_initial_state(ionic_model)

    φ₀ .= default_values[1]
    for i ∈ 1:(Thunderbolt.num_states(ionic_model)-1)
        s₀[:, i] .= default_values[i+1]
    end
end

# We start by defining a custom activation function
protocol = Thunderbolt.NoStimulationProtocol()
# We also generate both meshes
heart_mesh =  generate_ideal_lv_mesh(11,2,5;
    inner_radius = 0.7,
    outer_radius = 1.0,
    longitudinal_upper = 0.2,
    apex_inner = 1.3,
    apex_outer = 1.5
);

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
cellmodel = Thunderbolt.StimulatedCellModel(;
    cell_model = Thunderbolt.PCG2019()
)
heart_model = MonodomainModel(
    ConstantCoefficient(1.0),
    ConstantCoefficient(1.0),
    diffusion_tensor_field,
    protocol,
    cellmodel,
    :φₘ, :s
)
heart_odeform = semidiscretize(
    ReactionEikonalSplit(heart_model),
    (
        FiniteElementDiscretization(Dict(:φₘ => LagrangeCollection{1}())),
        Thunderbolt.SimplicialEikonalDiscretization(;
        order=1,
        activation_points = [Thunderbolt.ActivationCoordinate(
            Thunderbolt.LVCoordinate(0.0, 0.0, 0.0), 0.0
        )])
    ),
    heart_mesh,
)

# u₀ = zeros(Float64, solution_size(heart_odeform))
# steady_state_initializer!(u₀, heart_odeform)
# dt₀ = 0.01
# dtvis = 0.5
# Tₘₐₓ = 50.0
# Tₘₐₓ = dtvis # hide
# tspan = (0.0, Tₘₐₓ)
# problem = OperatorSplittingProblem(heart_odeform, u₀, tspan)
# timestepper = LieTrotterGodunov((
#     BackwardEulerSolver(),
#     ForwardEulerCellSolver(),
# ))
# integrator = init(problem, timestepper, dt=dt₀, verbose=true)

# # Now that the time integrator is ready we setup the ECG problem.
# torso_mesh_κᵢ = ConstantCoefficient(1.0)
# torso_mesh_κ  = ConstantCoefficient(1.0)
# # !!! todo
# #     Show how to transfer `diffusion_tensor_field` onto the torso mesh.
# geselowitz_ecg = Thunderbolt.Geselowitz1989ECGLeadCache(
#     heart_odeform,
#     torso_mesh,
#     torso_mesh_κᵢ,
#     torso_mesh_κ,
#     leads;
#     ground = Thunderbolt.OrderedSet([ground_vertex])
# )
# # !!! todo
# #     Improve the ECG API to not spill all the internals. :)

# # We compute the ECG online as follows.
# io = ParaViewWriter("ep04_ecg")
# for (u, t) in TimeChoiceIterator(integrator, tspan[1]:dtvis:tspan[2])
#     dh = heart_odeform.functions[1].dh
#     φ = u[heart_odeform.solution_indices[1]]
#     store_timestep!(io, t, dh.grid) do file
#         Thunderbolt.store_timestep_field!(file, t, dh, φ, :φₘ)
#     end

#     ## To compute the ECG we just need to update the ecg cache
#     Thunderbolt.update_ecg!(geselowitz_ecg, φ)
#     ## which then allows us to evaluate the leads like this
#     electrode_values = Thunderbolt.evaluate_ecg(geselowitz_ecg)
#     @info "$t: Lead 1=$(electrode_values[1]) | Lead 2= $(electrode_values[2])"
# end


#md # ## References
#md # ```@bibliography
#md # Pages = ["ep05_eikonal.md"]
#md # Canonical = false
#md # ```

#md # ## [Plain program](@id ep-tutorial_eikonal-plain-program)
#md #
#md # Here follows a version of the program without any comments.
#md # The file is also available here: [`ep05_eikonal.jl`](ep05_eikonal.jl).
#md #
#md # ```julia
#md # @__CODE__
#md # ```
