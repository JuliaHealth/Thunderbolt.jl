using Thunderbolt
using OrdinaryDiffEqOperatorSplitting
using DiffEqBase
using FerriteInterfaceElements, OrderedCollections, StaticArrays

# TODO before remove before merge
Ferrite.cell_to_vtkcell(cell::Type{InterfaceCell{RefQuadrilateral, Line, 4}}) = WriteVTK.VTKCellTypes.VTK_QUAD
Ferrite.geometric_interpolation(cell::Type{InterfaceCell{RefQuadrilateral, Line, 4}}) = Lagrange{RefQuadrilateral, 1}()
Ferrite.reference_shape_value(::InterfaceCellInterpolation{RefQuadrilateral, 1, Lagrange{RefLine, 1}}, ξ::Vec{2, Float64}, i::Int64) =
    Ferrite.reference_shape_value(Lagrange{RefQuadrilateral, 1}(), ξ, i)

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
                    ϕ₀[i] = max(1.0-norm(x)/2, 0.0)
                end
            end
        end
    end

    function solve_waveprop(mesh, coeff, subdomains, timestepper)
        cs = CartesianCoordinateSystem(mesh)
        model = MonodomainModel(
            ConstantCoefficient(1.0),
            ConstantCoefficient(1.0),
            coeff,
            Thunderbolt.AnalyticalTransmembraneStimulationProtocol(
                # Stimulate at apex
                AnalyticalCoefficient((x, t) -> norm(x) < 0.25 && t < 2.0 ? 0.5 : 0.0, cs),
                [SVector((0.0, 2.1))],
            ),
            Thunderbolt.FHNModel(),
            :φₘ,
            :s,
        )

        odeform = semidiscretize(
            ReactionDiffusionSplit(model),
            FiniteElementDiscretization(
                Dict(:φₘ => LagrangeCollection{1}());
                subdomains,
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
        @test integrator.sol.retcode == DiffEqBase.ReturnCode.Success
        @test integrator.u ≉ u₀
        return integrator.u
    end

    timestepper = LieTrotterGodunov((BackwardEulerSolver(), ForwardEulerCellSolver()))
    timestepper_adaptive =
        Thunderbolt.ReactionTangentController(timestepper, 0.5, 1.0, (0.98, 1.02))

    mesh = generate_mesh(Hexahedron, (4, 4, 4), Vec{3}((0.0, 0.0, 0.0)), Vec{3}((1.0, 1.0, 1.0)))
    coeff = ConstantCoefficient(SymmetricTensor{2, 3, Float64}((4.5e-5, 0, 0, 2.0e-5, 0, 1.0e-5)))
    u = solve_waveprop(mesh, coeff, [""], timestepper)
    u_adaptive = solve_waveprop(mesh, coeff, [""], timestepper_adaptive)
    @test u ≈ u_adaptive rtol = 1e-4

    mesh = generate_ideal_lv_mesh(4, 1, 1)
    coeff = ConstantCoefficient(SymmetricTensor{2, 3, Float64}((4.5e-5, 0, 0, 2.0e-5, 0, 1.0e-5)))
    u = solve_waveprop(mesh, coeff, ["myocardium"], timestepper)
    u_adaptive = solve_waveprop(mesh, coeff, ["myocardium"], timestepper_adaptive)
    @test u ≈ u_adaptive rtol = 1e-4

    grid = generate_grid(Quadrilateral, (64, 64), Vec{2}((-2.5, -2.5)), Vec{2}((2.5, 2.5)))
    addcellset!(grid, "Pacemaker", x->norm(x,Inf) ≤ 0.5)
    addcellset!(grid, "Myocardium", setdiff(OrderedSet(1:getncells(grid)), getcellset(grid, "Pacemaker")))
    grid2 = insert_interfaces(grid, ["Pacemaker", "Myocardium"]) # FIXME allow to add multiple interfaces

    cs = CartesianCoordinateSystem(grid2)

    coeff = ConstantCoefficient(SymmetricTensor{2, 2, Float64}((4.5e-4, 0, 2.0e-4)))
    models = Dict(
        "Pacemaker" => MonodomainModel(
            ConstantCoefficient(1.0),
            ConstantCoefficient(1.0),
            coeff,
            Thunderbolt.AnalyticalTransmembraneStimulationProtocol(
                # Stimulate at apex
                AnalyticalCoefficient((x, t) -> norm(x) < 0.25 && t < 2.0 ? 0.01 : 0.0, cs),
                [SVector((0.0, 2.1))],
            ),
            Thunderbolt.FHNModel(),
            :φₘ,
            :s1,
        ),
        "Myocardium" => MonodomainModel(
            ConstantCoefficient(1.0),
            ConstantCoefficient(1.0),
            coeff,
            NoStimulationProtocol(),
            Thunderbolt.FHNModel(),
            :φₘ,
            :s2,
        ),
        # FIXME 
        "interfaces" => InterfaceDiffusionModel(
            ConstantCoefficient(1.0),
            :φₘ,
            :φₘi,
        ),
    )

    discretization = FiniteElementDiscretization(
        Dict(
            :φₘ  => LagrangeCollection{1}(),
            :φₘi => Thunderbolt.InterfaceCollection(LagrangeCollection{1}())
        )
    )

    odeform = semidiscretize(ReactionDiffusionSplit(models), discretization, to_mesh(grid2))

    timestepper = LieTrotterGodunov((BackwardEulerSolver(), AdaptiveForwardEulerSubstepper()))
    u₀ = zeros(Float64, solution_size(odeform))
    simple_initializer!(u₀, odeform)

    tspan = (0.0, 100.0)
    problem = OperatorSplittingProblem(odeform, u₀, tspan)
    u₀ = copy(u₀)

    integrator = DiffEqBase.init(problem, timestepper, dt = 1.0, verbose = true)
    io = ParaViewWriter("ep-test")
    for (u, t) in TimeChoiceIterator(integrator, tspan[1]:1.0:tspan[2])
        (; dh) = odeform.functions[1]
        φ = u[odeform.solution_indices[1]]
        store_timestep!(io, t, dh.grid) do file
            Ferrite.write_cellset(io.current_file, grid2)
            Thunderbolt.store_timestep_field!(file, t, dh, φ, :φₘ)
        end
    end;

    mesh = to_mesh(generate_mixed_grid_2D())
    coeff = ConstantCoefficient(SymmetricTensor{2, 2, Float64}((4.5e-5, 0, 2.0e-5)))
    u = solve_waveprop(mesh, coeff, ["Pacemaker", "Myocardium"], timestepper)
    u_adaptive = solve_waveprop(mesh, coeff, ["Pacemaker", "Myocardium"], timestepper_adaptive)
    @test u ≈ u_adaptive rtol = 1e-4
    u = solve_waveprop(mesh, coeff, ["Pacemaker"], timestepper)
    u_adaptive = solve_waveprop(mesh, coeff, ["Pacemaker"], timestepper_adaptive)
    @test u ≈ u_adaptive rtol = 1e-4
    u = solve_waveprop(mesh, coeff, ["Myocardium"], timestepper)
    u_adaptive = solve_waveprop(mesh, coeff, ["Myocardium"], timestepper_adaptive)
    @test u ≈ u_adaptive rtol = 1e-4

    mesh = to_mesh(generate_mixed_dimensional_grid_3D())
    coeff = ConstantCoefficient(SymmetricTensor{2, 3, Float64}((4.5e-5, 0, 0, 2.0e-5, 0, 1.0e-5)))
    u = solve_waveprop(mesh, coeff, ["Ventricle"], timestepper)
    u_adaptive = solve_waveprop(mesh, coeff, ["Ventricle"], timestepper_adaptive)
    @test u ≈ u_adaptive rtol = 1e-4
    coeff = ConstantCoefficient(SymmetricTensor{2, 3, Float64}((5e-5, 0, 0, 5e-5, 0, 5e-5)))
    u = solve_waveprop(mesh, coeff, ["Purkinje"], timestepper)
    u_adaptive = solve_waveprop(mesh, coeff, ["Purkinje"], timestepper_adaptive)
    @test u ≈ u_adaptive rtol = 1e-4
    u = solve_waveprop(mesh, coeff, ["Ventricle", "Purkinje"], timestepper)
    u_adaptive = solve_waveprop(mesh, coeff, ["Ventricle", "Purkinje"], timestepper_adaptive)
    @test u ≈ u_adaptive rtol = 1e-4
end
