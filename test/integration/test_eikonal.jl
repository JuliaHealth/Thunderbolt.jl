using Thunderbolt
using OrdinaryDiffEqOperatorSplitting

@testset "EP wave propagation" begin
    function simple_initializer!(u₀, f::GenericSplitFunction)
        # TODO cleaner implementation. We need to extract this from the types or via dispatch.
        heatfun = f.functions[1]
        heat_dofrange = f.solution_indices[1]
        odefun = f.functions[2]
        ionic_model = odefun.ode

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
                    ϕ₀[i] = norm(x)/2
                end
            end
        end
    end

    function solve_waveprop(mesh, diffusion_tensor_field, subdomains, timestepper)
        cs = CartesianCoordinateSystem(mesh)

        activation_protocol = Thunderbolt.UniformEndocardialEikonalActivationProtocol()

        microstructure = OrthotropicMicrostructureModel(
            ConstantCoefficient((Vec(0.0, 0.0, 1.0))),
            ConstantCoefficient((Vec(0.0, 1.0, 0.0))),
            ConstantCoefficient((Vec(1.0, 0.0, 0.0)))
        )

        cellmodel = Thunderbolt.PCG2019()

        heart_model = MonodomainModel(
            ConstantCoefficient(1.0),
            ConstantCoefficient(1.0),
            diffusion_tensor_field,
            NoStimulationProtocol(),
            cellmodel,
            :φₘ, :s
        )

        heart_odeform = semidiscretize(
            ReactionEikonalSplit(heart_model, cs),
            (
                FiniteElementDiscretization(Dict(:φₘ => LagrangeCollection{1}())),
                Thunderbolt.SimplicialEikonalDiscretization(;
                    activation_protocol,
                    subdomains
                    )
            ),
            heart_mesh
        )

        u₀ = zeros(Float64, length(heart_odeform.ode_function.prob.u0) * length(heart_mesh.grid.nodes))
        simple_initializer!(u₀, heart_odeform)
        Eikonal.solve!(heart_odeform, heart_mesh, diffusion_tensor_field)
        nodal_timings = heart_odeform.ode_function.prob_func.activation_timings

        tspan = (0.0, 10.0)
        sim = solve(heart_odeform.ode_function, Rodas5P(), saveat=tspan, trajectories=length(heart_odeform.ode_function.prob_func.activation_timings))
        return sim.u
    end

    mesh = generate_mesh(Hexahedron, (4, 4, 4), Vec{3}((0.0, 0.0, 0.0)), Vec{3}((1.0, 1.0, 1.0)))
    coeff = ConstantCoefficient(SymmetricTensor{2, 3, Float64}((4.5e-5, 0, 0, 2.0e-5, 0, 1.0e-5)))
    u = solve_waveprop(mesh, coeff, [""], timestepper)
    u_adaptive = solve_waveprop(mesh, coeff, [""], timestepper_adaptive)
    @test u ≈ u_adaptive rtol = 1e-4

    mesh = generate_ideal_lv_mesh(4, 1, 1)
    coeff = ConstantCoefficient(SymmetricTensor{2, 3, Float64}((4.5e-5, 0, 0, 2.0e-5, 0, 1.0e-5)))
    u = solve_waveprop(mesh, coeff, [""], timestepper)
    u_adaptive = solve_waveprop(mesh, coeff, [""], timestepper_adaptive)
    @test u ≈ u_adaptive rtol = 1e-4

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