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
using DifferentialEquations

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

# We start by defining a custom activation function
protocol = Thunderbolt.NoStimulationProtocol()
# We also generate both meshes
heart_mesh = generate_ideal_lv_mesh(11, 2, 5;
                 inner_radius = 0.7,
                 outer_radius = 1.0,
                 longitudinal_upper = 0.2,
                 apex_inner = 1.3,
                 apex_outer = 1.5
             ) |> Thunderbolt.hexahedralize |> Thunderbolt.tetrahedralize;

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
    ReactionEikonalSplit(heart_model),
    (
        FiniteElementDiscretization(Dict(:φₘ => LagrangeCollection{1}())),
        Thunderbolt.SimplicialEikonalDiscretization(;
            order = 1,
            activation_points = [Thunderbolt.ActivationCoordinate(
                Thunderbolt.LVCoordinate(0.0, 0.0, 0.0), 0.0
            )])
    ),
    heart_mesh
)

u₀ = zeros(Float64, length(heart_odeform.ode_function.prob.u0) * length(heart_mesh.grid.nodes))
steady_state_initializer!(u₀, heart_odeform)

using Eikonal

conducivity_cache = Eikonal.ConductivityEvaluatorCache(heart_mesh, diffusion_tensor_field)
fim_cache = Eikonal.FIMCache(
    heart_odeform.eikonal_function.vertices,
    heart_odeform.eikonal_function.cells,
    heart_odeform.eikonal_function.vertex_to_cell,
    Eikonal.init_active_list(
        Eikonal.UniqueVector{Vector{Int}}, length(heart_odeform.eikonal_function.vertices)),
    heart_odeform.ode_function.prob_func.activation_timings,
    trues(length(heart_odeform.eikonal_function.vertices)),
    conducivity_cache)
function get_nodes_to_vertex_permutaion(dh::DofHandler)
    res = Vector{Int}(undef, dh.ndofs)
    grid = Ferrite.get_grid(dh)
    for i in 1:getncells(grid)
        res[SVector(getcells(grid, i).nodes)] .= (@view dh.cell_dofs[dh.cell_dofs_offset[i]:(dh.cell_dofs_offset[i] + Ferrite.ndofs_per_cell(dh, i) - 1)])
    end
    return sortperm(res)
end
cs = compute_lv_coordinate_system(heart_mesh)
perm = get_nodes_to_vertex_permutaion(cs.dh)|>sortperm
for node in eachindex(heart_mesh.grid.nodes)
    cs.u_transmural[perm[node]] <= 0.05 || continue
    Eikonal.make_vertex_active!(fim_cache.active_list, node)
    fim_cache.Φ[node] = 0.0
end

Eikonal.solve!(fim_cache)
VTKGridFile(("nein-activation-map-debug.vtu"), heart_mesh.grid) do vtk
    vtk.vtk["timings"] = fim_cache.Φ
end

dt₀ = 0.01
dtvis = 0.5
Tₘₐₓ = 50.0
# Tₘₐₓ = dtvis # hide
tspan = (0.0, Tₘₐₓ)

# problem = Thunderbolt.PointwiseODEProblem(heart_odeform.ode_function, u₀, tspan)
# timestepper = ForwardEulerCellSolver(solution_vector_type = Vector{Float64})
sim = solve(heart_odeform.ode_function, Rodas5P(), saveat=collect(0.0:dtvis:Tₘₐₓ), trajectories=length(fim_cache.Φ))
# integrator = init(problem,  Rodas5P(), dt=dt₀, verbose=true)
dh = cs.dh|>deepcopy
Thunderbolt.reorder_nodal!(dh)
# We compute the ECG online as follows.
io = ParaViewWriter("ep05_eikonal")
using SciMLBase
# for (uprev, tprev, u, t) in intervals(sim)
#     φ = u[1:solution_size(heart_odeform.eikonal_function)]
#     store_timestep!(io, t, dh.grid) do file
#         Thunderbolt.store_timestep_field!(file, t, dh, φ, :coordinates)
#     end
#     @show tprev, t
# end
# for (u, t) in TimeChoiceIterator(sim, tspan[1]:dtvis:tspan[2])
#     φ = u[1:solution_size(heart_odeform.eikonal_function)]
#     store_timestep!(io, t, dh.grid) do file
#         Thunderbolt.store_timestep_field!(file, t, dh, φ[perm], :φₘ)
#     end
# end

φₘfield = copy(fim_cache.Φ)
for t ∈ 0.0:dtvis:Tₘₐₓ
    for i in 1:length(fim_cache.Φ)
        φₘfield[i] = sim.u[i](t)[1]
    end
    store_timestep!(io, t, dh.grid) do file
        Thunderbolt.store_timestep_field!(file, t, dh, φₘfield, :coordinates)
    end
end
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
