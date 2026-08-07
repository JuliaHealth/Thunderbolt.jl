# # [Mechanics Tutorial 3: Coupling with Lumped Blood Circuits](@id mechanics-tutorial_3d0dcoupling)
# ![Pressure Volume Loop](3d0d-pv-loop.gif)
#
# This tutorial shows how to couple 3d chamber models with 0d fluid models.
#
# ## Introduction
#
# In this tutorial we will reproduce a simplified version of the model presented by [RegSalAfrFedDedQar:2022:cem](@citet).
#
# !!! warning
#     The API for 3D-0D coupling is work in progress and is hence subject to potential breaking changes.
#
# ## Commented Program
# We start by loading Thunderbolt and LinearSolve to use a custom direct solver of our choice.
using Thunderbolt, LinearSolve
# Finally, we try to approach a valid initial state by solving a simpler model first.
using OrdinaryDiffEqTsit5, OrdinaryDiffEqOperatorSplitting

fluid_model_init = RSAFDQ2022LumpedCicuitModel()
u0 = zeros(Thunderbolt.num_states(fluid_model_init))
Thunderbolt.default_initial_state!(u0, fluid_model_init)
prob = ODEProblem((du, u, p, t) -> Thunderbolt.lumped_driver!(du, u, t, [], p), u0, (0.0, 100*fluid_model_init.THB), fluid_model_init)
sol = solve(prob, Tsit5())
# #plot(sol, idxs=[1,2,3,4], tspan=(99*fluid_model_init.THB, 100*fluid_model_init.THB))

## Precomputed initial guess
u₀fluid = sol.u[end]
@info "Total blood volume: $(sum(u₀fluid[1:4])) + $(fluid_model_init.Csysₐᵣ*u₀fluid[5]) + $(fluid_model_init.Csysᵥₑₙ*u₀fluid[6]) + $(fluid_model_init.Cpulₐᵣ*u₀fluid[7]) + $(fluid_model_init.Cpulᵥₑₙ*u₀fluid[8])"

# We now generate the mechanical subproblem as in the [first tutorial](@ref mechanics-tutorial_simple-active-stress)
scaling_factor = 3.7;
# !!! warning
#     Tuning parameter until all bugs are fixed in this tutorial :)
mesh = generate_ideal_lv_mesh(16,4,19;
    inner_radius = scaling_factor*0.7,
    outer_radius = scaling_factor*1.0,
    longitudinal_upper = 0.4,
    apex_inner = scaling_factor* 1.3,
    apex_outer = scaling_factor*1.5,
    with_control_point = true,
)

coordinate_system = compute_lv_coordinate_system(mesh; subdomains = ["myocardium"])
microstructure    = create_microstructure_model(
    coordinate_system,
    LagrangeCollection{1}()^3,
    ODB25LTMicrostructureParameters();
    subdomains = ["myocardium"],
);
passive_material_model = Guccione1991PassiveModel()
active_material_model  = Guccione1993ActiveModel()
function calcium_profile_function(x::LVCoordinate,t_global)
    linear_interpolation(t,y1,y2,t1,t2) = y1 + (t-t1) * (y2-y1)/(t2-t1)
    ca_peak(x)                          = 1.0
    t = t_global % 800.0
    if 0 ≤ t ≤ 120.0
        return linear_interpolation(t,        0.0, ca_peak(x),   0.0, 120.0)
    elseif t ≤ 272.0
        return linear_interpolation(t, ca_peak(x),        0.0, 120.0, 272.0)
    else
        return 0.0
    end
end
calcium_field = AnalyticalCoefficient(
    calcium_profile_function,
    coordinate_system,
)
sarcomere_model = CaDrivenInternalSarcomereModel(ConstantStretchModel(), calcium_field)
active_stress_model = ActiveStressModel(
    passive_material_model,
    active_material_model,
    sarcomere_model,
    microstructure,
)
weak_boundary_conditions = (RobinBC(1.0, "Epicardium"),NormalSpringBC(100.0, "Base"))
solid_model = QuasiStaticModel(:displacement, active_stress_model, weak_boundary_conditions);

# The solid model is now couple with the circuit model by adding a Lagrange multipliers constraining the 3D chamber volume to match the chamber volume in the 0D model.
fluid_model = RSAFDQ2022LumpedCicuitModel(; lv_pressure_given = false)
coupler = LumpedFluidSolidCoupler(
    [
        ChamberVolumeCoupling(
            "Endocardium",
            "lv-volume-control",
            RSAFDQ2022SurrogateVolume(),
            :Vₗᵥ,
            :pₗᵥ,
            :pₗᵥ,
        )
    ],
    :displacement,
)
# The structural model is keyed by the subdomain it lives on, exactly as for an uncoupled mechanics
# problem. Note that the subdomain map goes *inside* the coupled model: `RSAFDQ2022Split` annotates
# the whole 3D-0D problem, so it is not itself something that lives on a subdomain.
coupled_model = RSAFDQ2022Model(Dict("myocardium" => solid_model), fluid_model, coupler);
# !!! todo
#     Once we figure out a nicer way to do this we should add more detailed docs here.

# Now we semidiscretize the model spatially as usual with finite elements and annotate the model with a stable split.
spatial_discretization_method = FiniteElementDiscretization(
    Dict(:displacement => LagrangeCollection{1}()^3);
    dbcs = [
        Dirichlet(:displacement, getfacetset(mesh, "Base"), (x,t) -> [0.0], [3]),
        Dirichlet(:displacement, getnodeset(mesh, "MyocardialAnchor1"), (x,t) -> (0.0, 0.0, 0.0), [1,2,3]),
        Dirichlet(:displacement, getnodeset(mesh, "MyocardialAnchor2"), (x,t) -> (0.0, 0.0), [2,3]),
        Dirichlet(:displacement, getnodeset(mesh, "MyocardialAnchor3"), (x,t) -> (0.0,), [3]),
        Dirichlet(:displacement, getnodeset(mesh, "MyocardialAnchor4"), (x,t) -> (0.0,), [3])
    ],
)
splitform = semidiscretize(
    RSAFDQ2022Split(coupled_model),
    spatial_discretization_method,
    mesh,
)

dt₀ = 1.0
dtvis = 10.0
tspan = (0.0, 3*800.0)
# This speeds up the CI #hide
tspan = (0.0, 10.0);    #hide

# The remaining code is very similar to how we use SciML solvers.
chamber_solver = HomotopyPathSolver(
    NewtonRaphsonSolver(;
        max_iter=10,
        tol=1e-2,
        inner_solver=SchurComplementLinearSolver(
            LinearSolve.UMFPACKFactorization()
        )
    )
)
blood_circuit_solver = Tsit5()
timestepper = LieTrotterGodunov((chamber_solver, blood_circuit_solver))

# The coupled problem publishes every quantity it holds by name, so the pre-paced circuit state can be
# transferred one state at a time instead of by slicing the solution vector.
u₀ = create_initial_condition(splitform)
for (sym, val) in zip(Thunderbolt.state_symbols(fluid_model), u₀fluid)
    setvariable!(u₀, splitform, sym, val)
end
# !!! tip
#     `solution_variable_names(splitform)` lists what is available here: the displacement field, the
#     chamber pressure `:pₗᵥ` introduced by the coupler, and the twelve circuit states.

problem = OperatorSplittingProblem(splitform, u₀, tspan)
integrator = init(problem, timestepper, dt=dt₀, verbose=true; dtmax=10.0);

## f2 = Figure()
## axs = [
##     Axis(f2[1, 1], title="LV"),
##     Axis(f2[1, 2], title="RV"),
##     Axis(f2[2, 1], title="LA"),
##     Axis(f2[2, 2], title="RA")
## ]

## vlv = Observable(Float64[])
## plv = Observable(Float64[])

## vrv = Observable(Float64[])
## prv = Observable(Float64[])

## vla = Observable(Float64[])
## pla = Observable(Float64[])

## vra = Observable(Float64[])
## pra = Observable(Float64[])

## lines!(axs[1], vlv, plv)
## lines!(axs[2], vrv, prv)
## lines!(axs[3], vla, pla)
## lines!(axs[4], vra, pra)
## for i in 1:4
##     xlims!(axs[1], 0.0, 180.0)
##     ylims!(axs[1], 0.0, 180.0)
## end
## display(f2)
# !!! todo
#     recover online visualization of the pressure volume loop

# Now we can finally solve the coupled problem in time.
# The displacement field and the chamber quantities are all reached by name, so the loop never touches
# the internals of the split. The descriptor is looked up once, outside the loop.
io = ParaViewWriter("CM03_3d0d-coupling");
displacement = solution_variable(splitform, :displacement)
for (u, t) in TimeChoiceIterator(integrator, tspan[1]:dtvis:tspan[2])
    store_timestep!(io, t, mesh) do file
        Thunderbolt.store_timestep_field!(file, t, u, displacement)
    end

    ## The chamber pressure is a genuine unknown of the coupled problem, and the chamber volume is a
    ## state of the 0D circuit, so both can simply be read out by name.
    ## Note that `:Vₗᵥ` is the *0D* volume; the tying constrains it to match the 3D chamber volume at
    ## convergence, but the two differ along the homotopy path.
    @info "$t: pₗᵥ = $(getvariable(u, splitform, :pₗᵥ)), Vₗᵥ = $(getvariable(u, splitform, :Vₗᵥ))"
    ## if t > 0.0
    ##     append!(vlv.val, getvariable(u, splitform, :Vₗᵥ))
    ##     append!(plv.val, getvariable(u, splitform, :pₗᵥ))
    ##     notify(vlv)
    ##     notify(plv)
    ## end
    ## TODO plot other chambers
end
# !!! tip
#     If you want to see more details of the solution process launch Julia with Thunderbolt as debug module:
#     ```
#     JULIA_DEBUG=Thunderbolt julia --project --threads=auto my_simulation_runner.jl
#     ```

#md # ## References
#md # ```@bibliography
#md # Pages = ["cm03_3d0d-coupling.md"]
#md # Canonical = false
#md # ```

#md # ## [Plain program](@id mechanics-tutorial_3d0dcoupling-plain-program)
#md #
#md # Here follows a version of the program without any comments.
#md # The file is also available here: [`cm03_3d0d-coupling.jl`](cm03_3d0d-coupling.jl).
#md #
#md # ```julia
#md # @__CODE__
#md # ```
