using Thunderbolt
using FastIterativeMethod
using LinearAlgebra

@testset "EP wave propagation" begin
    microstructure = OrthotropicMicrostructureModel(
        ConstantCoefficient((Vec(1.0, 0.0, 0.0))),
        ConstantCoefficient((Vec(0.0, 1.0, 0.0))),
        ConstantCoefficient((Vec(0.0, 0.0, 1.0))),
    )
    function solve_waveprop(mesh, diffusion_tensor_field, subdomains)
        cs = CartesianCoordinateSystem(mesh)

        activation_protocol = Thunderbolt.AnalyticalEikonalActivationProtocol(x -> norm(x) ≈ 0.0)

        cellmodel = Thunderbolt.PCG2019()

        heart_model = MonodomainModel(
            ConstantCoefficient(1.0),
            ConstantCoefficient(1.0),
            diffusion_tensor_field,
            NoStimulationProtocol(),
            cellmodel,
            :φₘ,
            :s,
        )

        heart_odeform = semidiscretize(
            ReactionEikonalSplit(heart_model, cs),
            (
                FiniteElementDiscretization(Dict(:φₘ => LagrangeCollection{1}())),
                Thunderbolt.SimplicialEikonalDiscretization(; activation_protocol, subdomains),
            ),
            mesh,
        )

        FastIterativeMethod.solve!(heart_odeform, mesh, diffusion_tensor_field)
        nodal_timings = heart_odeform.ode_function.prob_func.activation_timings
        # TODO: solve the reaction part
        return nodal_timings
    end

    @testset "Anisotropic Eikonal Cube" begin
        errors = Float64[]
        for j in (10, 20, 30, 40)
            mesh = generate_mesh(
                Tetrahedron,
                (j, j, j),
                Vec{3}((0.0, 0.0, 0.0)),
                Vec{3}((1.0, 1.0, 1.0)),
            )
            velocity = SVector((4.5e-5, 2.0e-5, 1.0e-5))
            coeff = SpectralTensorCoefficient(microstructure, ConstantCoefficient(velocity))
            u = solve_waveprop(mesh, coeff, [""])
            u_2 = Float64[]
            for (i, node) in enumerate(mesh.grid.nodes)
                coords = node.x
                unit_vec = normalize(coords)
                M = SymmetricTensor{2, 3, Float64}((
                    velocity[1]^2,
                    0.0,
                    0.0,
                    velocity[2]^2,
                    0.0,
                    velocity[3]^2,
                ))
                time = sqrt(coords ⋅ inv(M) ⋅ coords)
                push!(u_2, (time))
            end
            @test minimum(u .- u_2)≈0.0 atol=1e-8
            @test count(isapprox.(u .- u_2, 0.0, atol = 1e-8)) == 1 + 4j
            push!(errors, maximum(u .- u_2))
        end

        @test issorted(errors, rev = true)
    end

    # mesh = to_mesh(generate_mixed_grid_2D())
    # coeff = ConstantCoefficient(SymmetricTensor{2, 2, Float64}((4.5e-5, 0, 2.0e-5)))
    # u = solve_waveprop(mesh, coeff, ["Pacemaker", "Myocardium"], )
    # u_adaptive = solve_waveprop(mesh, coeff, ["Pacemaker", "Myocardium"], _adaptive)
    # @test u ≈ u_adaptive rtol = 1e-4
    # u = solve_waveprop(mesh, coeff, ["Pacemaker"], )
    # u_adaptive = solve_waveprop(mesh, coeff, ["Pacemaker"], _adaptive)
    # @test u ≈ u_adaptive rtol = 1e-4
    # u = solve_waveprop(mesh, coeff, ["Myocardium"], )
    # u_adaptive = solve_waveprop(mesh, coeff, ["Myocardium"], _adaptive)
    # @test u ≈ u_adaptive rtol = 1e-4

    # mesh = to_mesh(generate_mixed_dimensional_grid_3D())
    # coeff = ConstantCoefficient(SymmetricTensor{2, 3, Float64}((4.5e-5, 0, 0, 2.0e-5, 0, 1.0e-5)))
    # u = solve_waveprop(mesh, coeff, ["Ventricle"], )
    # u_adaptive = solve_waveprop(mesh, coeff, ["Ventricle"], _adaptive)
    # @test u ≈ u_adaptive rtol = 1e-4
    # coeff = ConstantCoefficient(SymmetricTensor{2, 3, Float64}((5e-5, 0, 0, 5e-5, 0, 5e-5)))
    # u = solve_waveprop(mesh, coeff, ["Purkinje"], )
    # u_adaptive = solve_waveprop(mesh, coeff, ["Purkinje"], _adaptive)
    # @test u ≈ u_adaptive rtol = 1e-4
    # u = solve_waveprop(mesh, coeff, ["Ventricle", "Purkinje"], )
    # u_adaptive = solve_waveprop(mesh, coeff, ["Ventricle", "Purkinje"], _adaptive)
    # @test u ≈ u_adaptive rtol = 1e-4
end
