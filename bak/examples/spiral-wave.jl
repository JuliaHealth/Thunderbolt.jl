using Thunderbolt, Thunderbolt.TimerOutputs

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
    # Should not be reshape but some array of arrays fun
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
end

tspan = (0.0, 1000.0)
dtvis = 25.0
dt₀ = 1.0

model = MonodomainModel(
    ConstantCoefficient(1.0),
    ConstantCoefficient(1.0),
    ConstantCoefficient(SymmetricTensor{2,2,Float64}((4.5e-5, 0, 2.0e-5))),
    NoStimulationProtocol(),
    Thunderbolt.FHNModel(),
    :φₘ, :s,
)

mesh = generate_mesh(Quadrilateral, (2^7, 2^7), Vec{2}((0.0,0.0)), Vec{2}((2.5,2.5)))

odeform = semidiscretize(
    ReactionDiffusionSplit(model),
    FiniteElementDiscretization(Dict(:φₘ => LagrangeCollection{1}())),
    mesh
)
u₀ = zeros(Float64, OS.function_size(odeform))
spiral_wave_initializer!(u₀, odeform)

problem = OS.OperatorSplittingProblem(odeform, u₀, tspan)

timestepper = OS.LieTrotterGodunov((
    BackwardEulerSolver(),
    AdaptiveForwardEulerSubstepper(),
))

integrator = OS.init(problem, timestepper, dt=dt₀, verbose=true)

io = ParaViewWriter("spiral-wave-test")

TimerOutputs.enable_debug_timings(Thunderbolt)
TimerOutputs.reset_timer!()
for (u, t) in OS.TimeChoiceIterator(integrator, tspan[1]:dtvis:tspan[2])
    dh = odeform.functions[1].dh
    φ = u[odeform.solution_indices[1]]
    @info t,norm(u)
    # sflat = ....?
    store_timestep!(io, t, dh.grid) do file
        Thunderbolt.store_timestep_field!(file, t, dh, φ, :φₘ)
        # s = reshape(sflat, (Thunderbolt.num_states(ionic_model),length(φ)))
        # for sidx in 1:Thunderbolt.num_states(ionic_model)
        #    Thunderbolt.store_timestep_field!(io, t, dh, s[sidx,:], state_symbol(ionic_model, sidx))
        # end
    end
end
TimerOutputs.print_timer()
TimerOutputs.disable_debug_timings(Thunderbolt)

# v2
# ───────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Time                    Allocations
# ───────────────────────   ────────────────────────
# Tot / % measured:                           3.66s /  88.2%            138MiB /  17.2%

# Section                                              ncalls     time    %tot     avg     alloc    %tot      avg
# ───────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Subintegrators                                        1.00k    3.23s  100.0%  3.23ms   23.7MiB  100.0%  24.3KiB
# inner solve                                         1.00k    2.70s   83.7%  2.70ms    130KiB    0.5%     133B
# cell loop                                           1.00k    184ms    5.7%   184μs     0.00B    0.0%    0.00B
# b = M uₙ₋₁                                          1.00k    162ms    5.0%   162μs   46.9KiB    0.2%    48.0B
# implicit_euler_heat_solver_update_system_matrix!        1   1.31ms    0.0%  1.31ms   7.04MiB   29.7%  7.04MiB
# update source term                                  1.00k    223μs    0.0%   223ns     0.00B    0.0%    0.00B
# ───────────────────────────────────────────────────────────────────────────────────────────────────────────────
# v3
# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Time                    Allocations
# ───────────────────────   ────────────────────────
# Tot / % measured:                          4.31s /  81.7%            142MiB /   5.0%

# Section                                            ncalls     time    %tot     avg     alloc    %tot      avg
# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
# inner solve                                         1.00k    3.11s   88.5%  3.11ms    130KiB    1.8%     133B
# cell loop                                           1.00k    199ms    5.7%   199μs     0.00B    0.0%    0.00B
# b = M uₙ₋₁                                          1.00k    178ms    5.1%   178μs     0.00B    0.0%    0.00B
# implicit_euler_heat_solver_update_system_matrix!        1   25.1ms    0.7%  25.1ms   7.04MiB   98.2%  7.04MiB
# update source term                                  1.00k    247μs    0.0%   247ns     0.00B    0.0%    0.00B
# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
