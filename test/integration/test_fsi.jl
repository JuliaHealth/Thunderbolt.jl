using Test, Thunderbolt, OrdinaryDiffEqTsit5, OrdinaryDiffEqOperatorSplitting

function test_solve_contractile_ideal_lv_3D0D(mesh, constitutive_model, fluid_model_init, fluid_model, coupler, tmax, Δt, adaptive)
    u0 = zeros(Thunderbolt.num_states(fluid_model_init))
    Thunderbolt.default_initial_condition!(u0, fluid_model_init)
    prob = ODEProblem((du, u, p, t) -> Thunderbolt.lumped_driver!(du, u, t, [], p), u0, (0.0, 100*fluid_model_init.THB), fluid_model_init)
    sol = solve(prob, Tsit5())

    tspan = (0.0,tmax)

    # Clamp three sides
    dbcs = [
        Dirichlet(:d, getnodeset(mesh, "MyocardialAnchor1"), (x,t) -> (0.0, 0.0, 0.0), [1,2,3]),
        Dirichlet(:d, getnodeset(mesh, "MyocardialAnchor2"), (x,t) -> (0.0, 0.0), [2,3]),
        Dirichlet(:d, getnodeset(mesh, "MyocardialAnchor3"), (x,t) -> (0.0,), [3]),
        Dirichlet(:d, getnodeset(mesh, "MyocardialAnchor4"), (x,t) -> (0.0,), [3])
    ]

    solid_model = QuasiStaticModel(:d, constitutive_model, (
        NormalSpringBC(0.1, "Epicardium"),
        NormalSpringBC(0.1, "Base"),
    ))
    coupled_model = RSAFDQ2022Model(solid_model, fluid_model, coupler)
    splitform = semidiscretize(
        RSAFDQ2022Split(coupled_model),
        FiniteElementDiscretization(
            Dict(:d => LagrangeCollection{1}()^3),
            dbcs,
        ),
        mesh
    )

    # Create sparse matrix and residual vector
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
    solve!(integrator)
    @test integrator.sol.retcode == ReturnCode.Success
    @test integrator.u ≉ u₀

    return integrator
end

@testset "3D0D Coupled" begin
    scaling_factor = 3.0;
    mesh = generate_ideal_lv_mesh(8,2,5;
        inner_radius = scaling_factor*0.7,
        outer_radius = scaling_factor*1.0,
        longitudinal_upper = 0.4,
        apex_inner = scaling_factor* 1.3,
        apex_outer = scaling_factor*1.5
    )
    mesh = Thunderbolt.hexahedralize(mesh) # FIXME Subdomain support

    cs = compute_lv_coordinate_system(mesh)
    @test !any(isnan.(cs.u_apicobasal))
    @test !any(isnan.(cs.u_transmural))
    @test !any(isnan.(cs.u_rotational))
    microstructure_parameters = ODB25LTMicrostructureParameters(αendo=deg2rad(80.0), αepi=deg2rad(-65.0))
    microstructure_model      = create_microstructure_model(cs, LagrangeCollection{1}()^3, microstructure_parameters)
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
        cs,
    )
    test_solve_contractile_ideal_lv_3D0D(mesh,
        ActiveStressModel(
            Guccione1991PassiveModel(),
            SimpleActiveStress(),
            Thunderbolt.CaDrivenInternalSarcomereModel(
                PelceSunLangeveld1995Model(),
                calcium_field,
            ),
            microstructure_model
        ),
        RSAFDQ2022LumpedCicuitModel(),
        RSAFDQ2022LumpedCicuitModel(; lv_pressure_given = false),
        LumpedFluidSolidCoupler(
            [
                ChamberVolumeCoupling(
                    "Endocardium",
                    RSAFDQ2022SurrogateVolume(),
                    :Vₗᵥ,
                    :pₗᵥ,
                )
            ],
            :d,
        ),
        3.0, 1.0, false)

        # TODO MTK variant
end

