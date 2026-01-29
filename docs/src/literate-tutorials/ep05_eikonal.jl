# # [Electrophysiology Tutorial 5: Eikonal Models](@id ep-tutorial_eikonal)
#
# ```@raw html
# <img src="../eikonal-wave.gif" width="50%">
# ```
#
# This tutorial shows how to solve Eikonal models and how to recover transemembrane
# potential fields and use them to calculate the ECG corresponding to the activation pattern.
#
# !!! todo
#     Provide context.
#
# # Introduction
# Depolarization can be modeled spatially as a wave propagating through cardiac tissue.
# The arrival times of this wave is the solution of the Eikonal equation in the Reaction-Eikonal
# split [NeicCamposPrassl:2017:ECE](@citet):
# ```math
# \begin{aligned}
#   C_{\textrm{m}} \partial_t \varphi &=  I_{\textrm{foot}}(t_{\textrm{arrival}}, t) - I_{\textrm{ion}}(\varphi, \boldsymbol{s}, t) & \textrm{pointwise in} \: \Omega \, , \\
#   \partial_t \boldsymbol{s} &= \mathbf{g}(\varphi, \boldsymbol{s}) & \textrm{pointwise in}  \: \Omega \, , \\
#   I_{\textrm{foot}}(t_{\textrm{arrival}}, t) &= -\frac{A}{\tau}\cdot e^{\frac{t - t_{\textrm{arrival}}}{\tau}} & \textrm{pointwise in}  \: \Omega \;\,
# \end{aligned}
# ```
# Where $C_{\textrm{m}}$ is the membrane capacitance, $\varphi$ is the transmembrane 
# potential, $I_{\textrm{foot}}$ is the stimulating current, $I_{\textrm{ion}}$ is
# the ionic cureent resulting from the cell model. $\boldsymbol{s}$ is the state vector
# for the cell model, and $\mathbf{g}$ is the function evolving the cell model states in time.
#
# Wave arrival times are obtained by solving the Eikonal equation:
# ```math
# \sqrt{\nabla t_{\textrm{arrival}}^T \boldsymbol{V} \nabla t_{\textrm{arrival}}} = 1 \qquad \textrm{in}  \: \Omega 
# ```
#
# Where $\boldsymbol{V}$ the conduction velocity, which was shown to be proportional to
# the square root of the diffusivities of the cardiac tissue [Weingart:1977:AOIC](@citet) (That's the earliest mention I 
# could find but I remember there was an older one in the 50s).
#
#
# ## Commented Program
using Thunderbolt, LinearAlgebra, StaticArrays, DifferentialEquations, Eikonal

# !!! todo
#     The initializer API is not yet finished and hence we deconstruct stuff here manually.
#     Please note that this method is quite fragile w.r.t. to many changes you can make in the code below.
function steady_state_initializer!(u₀, f::Thunderbolt.EikonalCoupledODEFunction)
    ## TODO cleaner implementation. We need to extract this from the types or via dispatch.
    odefun = f.ode_function.prob_func
    ionic_model = odefun.cellmodel

    φ₀ = @view u₀[1:solution_size(f.eikonal_function)]
    ## TODO extraction these via utility functions
    s₀flat = @view u₀[(solution_size(f.eikonal_function) + 1):end]
    ## Should not be reshape but some array of arrays fun
    s₀ = reshape(
        s₀flat, (solution_size(f.eikonal_function), Thunderbolt.num_states(ionic_model) - 1))
    default_values = Thunderbolt.default_initial_state(ionic_model)

    φ₀ .= default_values[1]
    for i in 1:(Thunderbolt.num_states(ionic_model) - 1)
        s₀[:, i] .= default_values[i + 1]
    end
end

# We start by generating a tetrahedral mesh, as the fast iterative method solver 
# accepts tetrahedral meshes only.
heart_mesh = generate_ideal_lv_mesh(12, 2, 5;
                 inner_radius = 0.7,
                 outer_radius = 1.0,
                 longitudinal_upper = 0.0,
                 apex_inner = 0.7,
                 apex_outer = 1.0
             ) |> Thunderbolt.hexahedralize |> Thunderbolt.tetrahedralize;

# !!! tip
#     We can also load realistic geometries with external formats. For this simply use either FerriteGmsh.jl
#     or one of the loader functions stated in the [mesh API](@ref mesh-utility-api).

# We also generate a coordinate system to be used for checking points for initial activation.
cs = compute_lv_coordinate_system(heart_mesh)

# For this tutorial we define a uniform endocardial activation
activation_protocol = Thunderbolt.UniformEndocardialEikonalActivationProtocol()

# For our toy problem we use a very simple microstructure.
microstructure = OrthotropicMicrostructureModel(
    ConstantCoefficient((Vec(0.0, 0.0, 1.0))),
    ConstantCoefficient((Vec(0.0, 1.0, 0.0))),
    ConstantCoefficient((Vec(1.0, 0.0, 0.0)))
)

# With the microstructure we setup the diffusion tensor field in spectral form.
# !!! todo
#     citation
κ₁ = 0.17 * 0.62 / (0.17 + 0.62)
κᵣ = 0.019 * 0.24 / (0.019 + 0.24)
diffusion_tensor_field = SpectralTensorCoefficient(
    microstructure,
    # ConstantCoefficient(SVector(κ₁, κᵣ, κᵣ))
    ConstantCoefficient(SVector(κ₁, κ₁, κ₁))
)
# Now we choose our *inner* cell model, used to compute the ionic currents, since 
# the eikonal-based foot current acts as a wrapper around ionic cell models.
cellmodel = Thunderbolt.PCG2019()

# We define the monodomain parameters to be used in the split.
heart_model = MonodomainModel(
    ConstantCoefficient(1.0),
    ConstantCoefficient(1.0),
    diffusion_tensor_field,
    NoStimulationProtocol(),
    cellmodel,
    :φₘ, :s
)

# We semidiscretize the using a reaction-diffusion split and provide the desired activation
# protocol. Here, we activate the full indocardium defined as nodes with less than 0.05
# transmural coordinate in the provided coordinate system by default.
heart_odeform = semidiscretize(
    ReactionEikonalSplit(heart_model, cs),
    (
        FiniteElementDiscretization(Dict(:φₘ => LagrangeCollection{1}())),
        Thunderbolt.SimplicialEikonalDiscretization(;
            activation_protocol
            )
    ),
    heart_mesh
)

# We allocate the solution vector for the reaction part. The activation times vector is allocated internally.
u₀ = zeros(Float64, length(heart_odeform.ode_function.prob.u0) * length(heart_mesh.grid.nodes))

# Apply the initial conditions.
steady_state_initializer!(u₀, heart_odeform)

# Here we solve for the wave arrival time using the package *Eikonal.jl* as it's
# required to be solved for once before timestepping.
Eikonal.solve!(heart_odeform, heart_mesh, diffusion_tensor_field)

# !!! danger
#     The fast iterative method solver in `Eikonal.jl` uses `@inbounds` a lot
#     in its internals. It was observed that forcing bounds checks results in an
#     enormous amount of dynamic allocations that can easily result in running out of memory.

nodal_timings = heart_odeform.ode_function.prob_func.activation_timings
VTKGridFile(("nein-activation-map-debug.vtu"), heart_mesh.grid) do vtk              # hide
    vtk.vtk["timings"] = nodal_timings    # hide
end                                                                                 # hide

dt₀ = 0.01
dtvis = 0.5
Tₘₐₓ = 50.0
Tₘₐₓ = dtvis # hide
tspan = (0.0, Tₘₐₓ)


sim = solve(heart_odeform.ode_function, Rodas5P(), saveat=collect(0.0:dtvis:Tₘₐₓ), trajectories=length(heart_odeform.ode_function.prob_func.activation_timings))
dh = cs.dh|>deepcopy
Thunderbolt.reorder_nodal!(dh)

io = ParaViewWriter("ep05_eikonal")

φₘfield = copy(nodal_timings)           # hide
for t ∈ 0.0:dtvis:Tₘₐₓ                                                          # hide
    for i in 1:length(φₘfield)                                                  # hide
        φₘfield[i] = sim.u[i](t)[1]                                             # hide
    end                                                                         # hide
    store_timestep!(io, t, dh.grid) do file                                     # hide
        Thunderbolt.store_timestep_field!(file, t, dh, φₘfield, :coordinates)   # hide
    end                                                                         # hide
end                                                                             # hide


## test the result                                                      #src
using Test                                                              #src
@test maximum(nodal_timings) ≈ 0.3/κ₁                                   #src
@test count(≈(0.3/κ₁), nodal_timings) == (5+1)*12+1                     #src
@test count(≈(0.3/κ₁*cosd(90/12)), nodal_timings) == (5+1)*12           #src


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
