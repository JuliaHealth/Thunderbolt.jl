using Test
using Thunderbolt
using OrdinaryDiffEqOperatorSplitting
using DiffEqBase
using FerriteInterfaceElements, OrderedCollections, StaticArrays

@testset "EP wave propagation" begin
    function simple_initializer!(u₀, f::GenericSplitFunction)
        # TODO cleaner implementation. We need to extract this from the types or via dispatch.
        heatfun = f.functions[1]
        heat_dofrange = f.solution_indices[1]

        ϕ₀ = @view u₀[heat_dofrange];
        # TODO extraction these via utility functions
        dh = heatfun.dh
        for sdh in dh.subdofhandlers
            for cell in CellIterator(sdh)
                _celldofs = celldofs(cell)
                φₘ_celldofs = _celldofs[dof_range(sdh, :φₘ)]
                # TODO query coordinate directly from the cell model
                coordinates = getcoordinates(cell)
                for (i, x) in zip(φₘ_celldofs, coordinates)
                    ϕ₀[i] = max(1.0-norm(x), 0.0)
                end
            end
        end
    end

    function solve_waveprop(mesh, model, timestepper)
        odeform = semidiscretize(
            ReactionDiffusionSplit(model),
            FiniteElementDiscretization(
                Dict(
                    :φₘ => LagrangeCollection{1}(),
                    :φₘi => Thunderbolt.InterfaceCollection(LagrangeCollection{1}()),
                ),
            ),
            mesh,
        )

        u₀ = zeros(Float64, solution_size(odeform))
        simple_initializer!(u₀, odeform)

        tspan = (0.0, 10.0)
        problem = OperatorSplittingProblem(odeform, u₀, tspan)
        u₀ = copy(u₀)

        integrator = DiffEqBase.init(problem, timestepper, dt = 1.0, verbose = true)
        DiffEqBase.solve!(integrator)

        # for (u, t) in TimeChoiceIterator(integrator, tspan[1]:25.0:tspan[2])
        #     (; dh) = odeform.functions[1]
        #     φ = u[odeform.solution_indices[1]]
        #     store_timestep!(io, t, dh.grid) do file
        #         Ferrite.write_cellset(io.current_file, grid2)
        #         Thunderbolt.store_timestep_field!(file, t, dh, φ, :φₘ)
        #     end
        # end

        @test integrator.sol.retcode == DiffEqBase.ReturnCode.Success
        @test integrator.u ≉ u₀
        return integrator
    end

    timestepper = LieTrotterGodunov((BackwardEulerSolver(), ForwardEulerCellSolver()))
    timestepper_adaptive =
        LieTrotterGodunov((BackwardEulerSolver(), AdaptiveForwardEulerSubstepper()))
    # Bounds well apart from the base dt = 1.0, so a controller that silently ran at
    # constant dt would fail the step count assertions below.
    timestepper_rtc = Thunderbolt.ReactionTangentController(timestepper, 0.5, 1.0, (0.5, 2.0))

    @testset "Single subdomain" begin
        grid = generate_grid(Quadrilateral, (8, 8), Vec{2}((-2.5, -2.5)), Vec{2}((2.5, 2.5)))
        mesh = to_mesh(grid)
        cs = CartesianCoordinateSystem(mesh)
        coeff = ConstantCoefficient(SymmetricTensor{2, 2, Float64}((4.5e-4, 0, 2.0e-4)))
        model = MonodomainModel(
            ConstantCoefficient(1.0),
            ConstantCoefficient(1.0),
            coeff,
            Thunderbolt.AnalyticalTransmembraneStimulationProtocol(
                # Stimulate at apex
                AnalyticalCoefficient((x, t) -> norm(x) < 0.1 && t < 2.0 ? 0.01 : 0.0, cs),
                [SVector((0.0, 2.1))],
            ),
            Thunderbolt.FHNModel(),
            :φₘ,
            :s1,
        )
        integ = solve_waveprop(mesh, model, timestepper)
        integ_adaptive = solve_waveprop(mesh, model, timestepper_adaptive)
        @test integ.u ≈ integ_adaptive.u rtol = 1e-2
        # If `reaction_threshold` never trips, the substepper is bitwise a plain forward
        # Euler step and the comparison above tests nothing.
        @test !isapprox(integ.u, integ_adaptive.u; rtol = 1e-8)
        integ_rtc = solve_waveprop(mesh, model, timestepper_rtc)
        @test integ.u ≈ integ_rtc.u rtol = 1e-2
        # The reaction tangent controller must actually move dt away from dt = 1.0.
        @test integ_rtc.stats.naccept != integ.stats.naccept

        mesh = generate_ideal_lv_mesh(4, 1, 1)
        cs = CartesianCoordinateSystem(mesh)
        coeff = ConstantCoefficient(
            SymmetricTensor{2, 3, Float64}((4.5e-4, 0.0, 0.0, 2.0e-4, 0.0, 2.0e-4)),
        )
        model = MonodomainModel(
            ConstantCoefficient(1.0),
            ConstantCoefficient(1.0),
            coeff,
            Thunderbolt.AnalyticalTransmembraneStimulationProtocol(
                # Stimulate at apex
                AnalyticalCoefficient((x, t) -> norm(x) < 0.1 && t < 2.0 ? 0.01 : 0.0, cs),
                [SVector((0.0, 2.1))],
            ),
            Thunderbolt.FHNModel(),
            :φₘ,
            :s1,
        )
        integ = solve_waveprop(mesh, model, timestepper)
        integ_adaptive = solve_waveprop(mesh, model, timestepper_adaptive)
        @test integ.u ≈ integ_adaptive.u rtol = 1e-4
    end

    @testset "Pacemaker subdomain" begin
        grid = generate_grid(Quadrilateral, (64, 64), Vec{2}((-2.5, -2.5)), Vec{2}((2.5, 2.5)))
        addcellset!(grid, "Pacemaker", x->norm(x, Inf) ≤ 0.75)
        addcellset!(
            grid,
            "Myocardium",
            setdiff(OrderedSet(1:getncells(grid)), getcellset(grid, "Pacemaker")),
        )
        grid2 = insert_interfaces(grid, ["Pacemaker", "Myocardium"]) # FIXME allow to add multiple interfaces
        mesh2 = to_mesh(grid2)
        cs = CartesianCoordinateSystem(mesh2)

        coeff = ConstantCoefficient(SymmetricTensor{2, 2, Float64}((4.5e-4, 0, 2.0e-4)))
        models = Dict(
            "Pacemaker" => MonodomainModel(
                ConstantCoefficient(1.0),
                ConstantCoefficient(1.0),
                coeff,
                Thunderbolt.NoStimulationProtocol(),
                Thunderbolt.ParametrizedFHNModel{Float64}(
                    a = -0.5,
                    b = 1.0,
                    c = -0.6,
                    d = 0.0,
                    e = 0.001,
                    f = 50*0.001,
                ),
                :φₘ,
                :s1,
            ),
            "Myocardium" => MonodomainModel(
                ConstantCoefficient(1.0),
                ConstantCoefficient(1.0),
                coeff,
                Thunderbolt.NoStimulationProtocol(),
                Thunderbolt.ParametrizedFHNModel{Float64}(),
                :φₘ,
                :s2,
            ),
            # FIXME explicit name
            "interfaces" => InterfaceDiffusionModel(ConstantCoefficient(1.0), :φₘ, :φₘi),
        )
        integ = solve_waveprop(mesh2, models, timestepper)
        integ_adaptive = solve_waveprop(mesh2, models, timestepper_adaptive)
        @test integ.u ≈ integ_adaptive.u rtol = 1e-3
        integ_rtc = solve_waveprop(mesh2, models, timestepper_rtc)
        # Loose sanity bound: the adapted trajectory (dt in [0.5, 2.0] vs fixed 1.0)
        # legitimately drifts a few percent; corruption would be O(1).
        @test integ.u ≈ integ_rtc.u rtol = 5e-2
        @test integ_rtc.stats.naccept != integ.stats.naccept

        coeff = ConstantCoefficient(SymmetricTensor{2, 2, Float64}((4.5e-4, 0, 2.0e-4)))
        models = Dict(
            "Pacemaker" => MonodomainModel(
                ConstantCoefficient(1.0),
                ConstantCoefficient(1.0),
                coeff,
                Thunderbolt.NoStimulationProtocol(),
                Thunderbolt.AlievPanfilovModel(),
                :φₘ,
                :s1,
            ),
        )
        integ = solve_waveprop(mesh2, models, timestepper)
        integ_adaptive = solve_waveprop(mesh2, models, timestepper_adaptive)
        @test integ.u ≈ integ_adaptive.u rtol = 1e-3
        integ_rtc = solve_waveprop(mesh2, models, timestepper_rtc)
        # Loose sanity bound: the adapted trajectory (dt in [0.5, 2.0] vs fixed 1.0)
        # legitimately drifts a few percent; corruption would be O(1).
        @test integ.u ≈ integ_rtc.u rtol = 5e-2
        @test integ_rtc.stats.naccept != integ.stats.naccept
    end

    # TODO revive
    # mesh = to_mesh(generate_mixed_dimensional_grid_3D())
    # coeff = ConstantCoefficient(SymmetricTensor{2, 3, Float64}((4.5e-5, 0, 0, 2.0e-5, 0, 1.0e-5)))
    # u = solve_waveprop(mesh, coeff, ["Ventricle"], timestepper)
    # u_adaptive = solve_waveprop(mesh, coeff, ["Ventricle"], timestepper_adaptive)
    # @test u ≈ u_adaptive rtol = 1e-4
    # coeff = ConstantCoefficient(SymmetricTensor{2, 3, Float64}((5e-5, 0, 0, 5e-5, 0, 5e-5)))
    # u = solve_waveprop(mesh, coeff, ["Purkinje"], timestepper)
    # u_adaptive = solve_waveprop(mesh, coeff, ["Purkinje"], timestepper_adaptive)
    # @test u ≈ u_adaptive rtol = 1e-4
    # u = solve_waveprop(mesh, coeff, ["Ventricle", "Purkinje"], timestepper)
    # u_adaptive = solve_waveprop(mesh, coeff, ["Ventricle", "Purkinje"], timestepper_adaptive)
    # @test u ≈ u_adaptive rtol = 1e-4
end
