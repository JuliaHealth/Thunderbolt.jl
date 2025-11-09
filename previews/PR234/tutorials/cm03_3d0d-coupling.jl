using Thunderbolt, LinearSolve

using OrdinaryDiffEqTsit5, OrdinaryDiffEqOperatorSplitting

fluid_model_init = RSAFDQ2022LumpedCicuitModel()
u0 = zeros(Thunderbolt.num_states(fluid_model_init))
Thunderbolt.default_initial_condition!(u0, fluid_model_init)
prob = ODEProblem((du, u, p, t) -> Thunderbolt.lumped_driver!(du, u, t, [], p), u0, (0.0, 100*fluid_model_init.THB), fluid_model_init)
sol = solve(prob, Tsit5())

# Precomputed initial guess
u₀fluid = sol.u[end]
@info "Total blood volume: $(sum(u₀fluid[1:4])) + $(fluid_model_init.Csysₐᵣ*u₀fluid[5]) + $(fluid_model_init.Csysᵥₑₙ*u₀fluid[6]) + $(fluid_model_init.Cpulₐᵣ*u₀fluid[7]) + $(fluid_model_init.Cpulᵥₑₙ*u₀fluid[8])"

scaling_factor = 3.7;

mesh = generate_ideal_lv_mesh(8,2,5;
    inner_radius = scaling_factor*0.7,
    outer_radius = scaling_factor*1.0,
    longitudinal_upper = 0.4,
    apex_inner = scaling_factor* 1.3,
    apex_outer = scaling_factor*1.5
)
mesh = Thunderbolt.hexahedralize(mesh)

coordinate_system = compute_lv_coordinate_system(mesh)
microstructure    = create_microstructure_model(
    coordinate_system,
    LagrangeCollection{1}()^3,
    ODB25LTMicrostructureParameters(),
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

fluid_model = RSAFDQ2022LumpedCicuitModel(; lv_pressure_given = false)
coupler = LumpedFluidSolidCoupler(
    [
        ChamberVolumeCoupling(
            "Endocardium",
            RSAFDQ2022SurrogateVolume(),
            :Vₗᵥ,
            :pₗᵥ,
        )
    ],
    :displacement,
)
coupled_model = RSAFDQ2022Model(solid_model, fluid_model, coupler);

spatial_discretization_method = FiniteElementDiscretization(
    Dict(:displacement => LagrangeCollection{1}()^3),
    [
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

tspan = (0.0, 10.0)    # hide

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

u₀ = zeros(solution_size(splitform))
u₀solid_view = @view  u₀[OS.get_solution_indices(splitform, 1)]
u₀fluid_view = @view  u₀[OS.get_solution_indices(splitform, 2)]
u₀fluid_view .= u₀fluid

problem = OperatorSplittingProblem(splitform, u₀, tspan)
integrator = init(problem, timestepper, dt=dt₀, verbose=true; dtmax=10.0);

# f2 = Figure()
# axs = [
#     Axis(f2[1, 1], title="LV"),
#     Axis(f2[1, 2], title="RV"),
#     Axis(f2[2, 1], title="LA"),
#     Axis(f2[2, 2], title="RA")
# ]

# vlv = Observable(Float64[])
# plv = Observable(Float64[])

# vrv = Observable(Float64[])
# prv = Observable(Float64[])

# vla = Observable(Float64[])
# pla = Observable(Float64[])

# vra = Observable(Float64[])
# pra = Observable(Float64[])

# lines!(axs[1], vlv, plv)
# lines!(axs[2], vrv, prv)
# lines!(axs[3], vla, pla)
# lines!(axs[4], vra, pra)
# for i in 1:4
#     xlims!(axs[1], 0.0, 180.0)
#     ylims!(axs[1], 0.0, 180.0)
# end
# display(f2)

io = ParaViewWriter("CM03_3d0d-coupling");
for (u, t) in TimeChoiceIterator(integrator, tspan[1]:dtvis:tspan[2])
    chamber_function = OS.get_operator(splitform, 1)
    (; dh) = chamber_function.structural_function
    store_timestep!(io, t, dh.grid)
    usolid_view = @view u[OS.get_solution_indices(splitform, 1)]
    Thunderbolt.store_timestep_field!(io, t, dh, usolid_view, :displacement)
    Thunderbolt.finalize_timestep!(io, t)

    # if t > 0.0
    #     lv = chamber_function.tying_info.chambers[1]
    #     append!(vlv.val, lv.V⁰ᴰval)
    #     append!(plv.val, u[lv.pressure_dof_index_global])
    #     notify(vlv)
    #     notify(plv)
    # end
    # TODO plot other chambers
end

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
