using Thunderbolt, CUDA
using Thunderbolt.TimerOutputs

using Thunderbolt.StaticArrays

using LinearSolve

function steady_state_initializer!(u₀, f::GenericSplitFunction)
    # TODO cleaner implementation. We need to extract this from the types or via dispatch.
    heatfun = f.functions[1]
    heat_dofrange = f.solution_indices[1]
    odefun = f.functions[2]
    ionic_model = odefun.ode

    φ₀ = @view u₀[heat_dofrange];
    # TODO extraction these via utility functions
    dh = heatfun.dh
    s₀flat = @view u₀[(ndofs(dh)+1):end];
    # Should not be reshape but some array of arrays fun
    s₀ = reshape(s₀flat, (ndofs(dh), Thunderbolt.num_states(ionic_model)-1));
    default_values = Thunderbolt.default_initial_state(ionic_model)

    φ₀ .= default_values[1]
    for i ∈ 1:(Thunderbolt.num_states(ionic_model)-1)
        s₀[:, i] .= default_values[i+1]
    end
end

mesh = generate_mesh(Hexahedron, (80,28,12), Vec((0.0,0.0,0.0)), Vec((20.0,7.0,3.0)))
cs = CoordinateSystemCoefficient(CartesianCoordinateSystem(mesh))

tspan = (0.0, 100.0)
dtvis = 1.0
dt₀ = 0.01


κ₁ = 0.17 * 0.62 / (0.17 + 0.62)
κᵣ = 0.019 * 0.24 / (0.019 + 0.24)
model = MonodomainModel(
    ConstantCoefficient(1.0),
    ConstantCoefficient(1.0),
    ConstantCoefficient(SymmetricTensor{2,3}((
        κ₁, 0, 0,
           κᵣ, 0,
              κᵣ
    ))),
    AnalyticalTransmembraneStimulationProtocol(
        AnalyticalCoefficient((x,t) -> maximum(x) < 1.5 && t < 2.0 ? 0.5 : 0.0, cs),
        [SVector((0.0, 2.1))]
    ),
    Thunderbolt.PCG2019(),
    :φₘ, :s
)


ip_collection = LagrangeCollection{1}()

odeform = semidiscretize(
    ReactionDiffusionSplit(model),
    FiniteElementDiscretization(Dict(:φₘ => LagrangeCollection{1}())),
    mesh
)
u₀ = zeros(Float32, OS.function_size(odeform))
steady_state_initializer!(u₀, odeform)

# io = ParaViewWriter("spiral-wave-test")


timestepper = OS.LieTrotterGodunov((
    BackwardEulerSolver(
        solution_vector_type=CuVector{Float32},
        system_matrix_type=CUDA.CUSPARSE.CuSparseMatrixCSR{Float32, Int32},
        inner_solver=LinearSolve.KrylovJL_CG(atol=1.0f-6, rtol=1.0f-5),
    ),
    AdaptiveForwardEulerSubstepper(
        solution_vector_type=CuVector{Float32},
        reaction_threshold=0.1f0,
    ),
))

u₀gpu = CuVector(u₀)
problem = OS.OperatorSplittingProblem(odeform, u₀gpu, tspan)

integrator = OS.init(problem, timestepper, dt=dt₀, verbose=true)

TimerOutputs.enable_debug_timings(Thunderbolt)
step!(integrator) # precompile for benchmark below

TimerOutputs.reset_timer!()
for (u, t) in OS.TimeChoiceIterator(integrator, tspan[1]:dtvis:tspan[2])
    # dh = odeform.functions[1].dh
    # φ = u[odeform.dof_ranges[1]]
    # @info t,norm(u)
    # sflat = ....?
    # store_timestep!(io, t, dh.grid) do file
    #     Thunderbolt.store_timestep_field!(file, t, dh, φ, :φₘ)
    #     # s = reshape(sflat, (Thunderbolt.num_states(ionic_model),length(φ)))
    #     # for sidx in 1:Thunderbolt.num_states(ionic_model)
    #     #    Thunderbolt.store_timestep_field!(io, t, dh, s[sidx,:], state_symbol(ionic_model, sidx))
    #     # end
    # end
end
TimerOutputs.print_timer()
# TimerOutputs.disable_debug_timings(Thunderbolt)
