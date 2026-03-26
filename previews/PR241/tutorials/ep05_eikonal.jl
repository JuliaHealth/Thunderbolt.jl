using Thunderbolt, LinearAlgebra, StaticArrays, OrdinaryDiffEqRosenbrock, FastIterativeMethod

function steady_state_initializer!(u₀, f::Thunderbolt.ReactionEikonalFunction)
    # TODO cleaner implementation. We need to extract this from the types or via dispatch.
    odefun = f.ode_function
    ionic_model = odefun.cellmodel

    φ₀ = @view u₀[1:solution_size(f.eikonal_function)]
    # TODO extraction these via utility functions
    s₀flat = @view u₀[(solution_size(f.eikonal_function) + 1):end]
    # Should not be reshape but some array of arrays fun
    s₀ = reshape(
        s₀flat, (solution_size(f.eikonal_function), Thunderbolt.num_states(ionic_model) - 1)
    )
    default_values = Thunderbolt.default_initial_state(ionic_model)

    φ₀ .= default_values[1]
    for i in 1:(Thunderbolt.num_states(ionic_model) - 1)
        s₀[:, i] .= default_values[i + 1]
    end
    return
end

heart_mesh = generate_ideal_lv_mesh(
    12, 2, 5;
    inner_radius = 0.7,
    outer_radius = 1.0,
    longitudinal_upper = 0.0,
    apex_inner = 0.7,
    apex_outer = 1.0
) |> Thunderbolt.hexahedralize |> Thunderbolt.tetrahedralize;

heart_mesh = Thunderbolt.to_mesh(heart_mesh.grid)

cs = compute_lv_coordinate_system(heart_mesh)

activation_protocol = Thunderbolt.UniformEndocardialActivationProtocol(Dict{String, Float64}(),
cs
)

microstructure = OrthotropicMicrostructureModel(
    ConstantCoefficient((Vec(0.0, 0.0, 1.0))),
    ConstantCoefficient((Vec(0.0, 1.0, 0.0))),
    ConstantCoefficient((Vec(1.0, 0.0, 0.0)))
)

κ₁ = 0.17 * 0.62 / (0.17 + 0.62)
κᵣ = 0.019 * 0.24 / (0.019 + 0.24)
diffusion_tensor_field = SpectralTensorCoefficient(
    microstructure,
    ConstantCoefficient(SVector(κ₁, κ₁, κ₁))
)

cellmodel = Thunderbolt.PCG2019()

heart_model = MonodomainModel(
    ConstantCoefficient(1.0),
    ConstantCoefficient(1.0),
    diffusion_tensor_field,
    activation_protocol,
    cellmodel,
    :φₘ, :s
)

heart_odeform = semidiscretize(
    ReactionEikonalSplit(heart_model, cs),
    (
        FiniteElementDiscretization(Dict(:φₘ => LagrangeCollection{1}())),
        Thunderbolt.SimplicialEikonalDiscretization(;
            activation_protocol,
            subdomains = String[]
        ),
    ),
    heart_mesh
)

u₀ = zeros(Float64, Thunderbolt.num_states(heart_odeform.ode_function.cellmodel) * length(heart_mesh.grid.nodes))

steady_state_initializer!(u₀, heart_odeform)

FastIterativeMethod.solve!(heart_odeform, heart_mesh)

nodal_timings = heart_odeform.ode_function.activation_timings
VTKGridFile(("ep05_eikonal_timings.vtu"), heart_mesh.grid) do vtk          # hide
    vtk.vtk["timings"] = nodal_timings    # hide
end                                                                                 # hide

dt₀ = 0.01
dtvis = 0.5
Tₘₐₓ = 50.0
Tₘₐₓ = dtvis # hide
tspan = (0.0, Tₘₐₓ)

single_prob = ODEProblem(
    (du, u, m, t) -> Thunderbolt.cell_rhs!(du, u, nothing, m.stim_offset, t, m),
    Thunderbolt.default_initial_state(cellmodel),
    (first(tspan), last(tspan)),
    Thunderbolt.StimulatedCellModel(; cell_model = cellmodel),
)
problem = SciMLBase.EnsembleProblem(single_prob, prob_func = heart_odeform.ode_function);

sim = solve(problem, Rodas5P(), saveat = collect(0.0:dtvis:Tₘₐₓ), trajectories = length(heart_odeform.ode_function.activation_timings))
dh = cs.dh |> deepcopy
Thunderbolt.reorder_nodal!(dh)

io = ParaViewWriter("ep05_eikonal")

φₘfield = copy(nodal_timings)           # hide
for t in 0.0:dtvis:Tₘₐₓ                                                          # hide
    for i in 1:length(φₘfield)                                                  # hide
        φₘfield[i] = sim.u[i](t)[1]                                             # hide
    end                                                                         # hide
    store_timestep!(io, t, dh.grid) do file                                     # hide
        Thunderbolt.store_timestep_field!(file, t, dh, φₘfield, :coordinates)   # hide
    end                                                                         # hide
end                                                                             # hide

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
