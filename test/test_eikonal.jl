using Thunderbolt
using FastIterativeMethod
using LinearAlgebra

@testset "Single Domain" begin
    heart_mesh =
        generate_ideal_lv_mesh(
            12,
            2,
            5;
            inner_radius = 0.7,
            outer_radius = 1.0,
            longitudinal_upper = 0.0,
            apex_inner = 0.7,
            apex_outer = 1.0,
        ) |> Thunderbolt.tetrahedralize;
    heart_mesh = Thunderbolt.to_mesh(heart_mesh.grid)
    microstructure = OrthotropicMicrostructureModel(
        ConstantCoefficient((Vec(0.0, 0.0, 1.0))),
        ConstantCoefficient((Vec(0.0, 1.0, 0.0))),
        ConstantCoefficient((Vec(1.0, 0.0, 0.0))),
    )
    κ₁ = 1.0
    diffusion_tensor_field =
        SpectralTensorCoefficient(microstructure, ConstantCoefficient(SVector(κ₁, κ₁, κ₁)))
    cellmodel = Thunderbolt.PCG2019() # Does not really matter for this test
    heart_model = MonodomainModel(
        ConstantCoefficient(1.0),
        ConstantCoefficient(1.0),
        diffusion_tensor_field,
        NoStimulationProtocol(),
        cellmodel,
        :φₘ,
        :s,
    )
    results = Vector{Float64}[]
    @testset "Uniform Activation" begin
        @testset "LV-CS" begin
            cs = compute_lv_coordinate_system(heart_mesh)
            activation_protocol =
                Thunderbolt.UniformEndocardialActivationProtocol(Dict{String, Float64}(), cs)
            heart_odeform = semidiscretize(
                ReactionEikonalSplit(heart_model, cs),
                (
                    FiniteElementDiscretization(Dict(:φₘ => LagrangeCollection{1}())),
                    Thunderbolt.SimplicialEikonalDiscretization(;
                        activation_protocol,
                        subdomains = String[],
                    ),
                ),
                heart_mesh,
            )
            FastIterativeMethod.solve!(heart_odeform, heart_mesh)
            nodal_timings = heart_odeform.ode_function.activation_timings
            error_vals = Float64[]
            for cell in heart_mesh.grid.cells
                for node in cell.nodes
                    coords = heart_mesh.grid.nodes[node].x
                    push!(error_vals, nodal_timings[node] - (norm(coords) - 0.7))
                end
            end
            @test maximum(error_vals) < 0.03
            @test minimum(error_vals) ≈ 0.0 atol=1e-12
            push!(results, nodal_timings)
        end
        @testset "Cartesian-CS" begin
            cs = CartesianCoordinateSystem(heart_mesh)
            activation_protocol =
                Thunderbolt.UniformEndocardialActivationProtocol(Dict{String, Float64}(), cs)
            @test_throws "Uniformally activating the endocardium requires using either
    LV or BiV coordinate system. usage with Cartesian Coordinate System is
    restricted to AnalyticalTransmembraneStimulationProtocol" semidiscretize(
                ReactionEikonalSplit(heart_model, cs),
                (
                    FiniteElementDiscretization(Dict(:φₘ => LagrangeCollection{1}())),
                    Thunderbolt.SimplicialEikonalDiscretization(;
                        activation_protocol,
                        subdomains = String[],
                    ),
                ),
                heart_mesh,
            )
        end
        endocardial_nodes = Set{Int}()
        for (cell_idx, facet_idx_local) in heart_mesh.grid.facetsets["Endocardium"]
            cell = getcells(heart_mesh.grid, cell_idx)
            face = Ferrite.faces(cell)[facet_idx_local]
            for node in face
                push!(endocardial_nodes, node)
            end
        end
        Ferrite.addnodeset!(heart_mesh.grid, "endocardium", endocardial_nodes)
        @testset "LV-CS Nodeset" begin
            cs = compute_lv_coordinate_system(heart_mesh)
            activation_protocol =
                Thunderbolt.UniformEndocardialActivationProtocol(Dict{String, Float64}(), cs)
            heart_odeform = semidiscretize(
                ReactionEikonalSplit(heart_model, cs),
                (
                    FiniteElementDiscretization(Dict(:φₘ => LagrangeCollection{1}())),
                    Thunderbolt.SimplicialEikonalDiscretization(;
                        activation_protocol,
                        subdomains = String[],
                    ),
                ),
                heart_mesh,
            )
            FastIterativeMethod.solve!(heart_odeform, heart_mesh)
            nodal_timings = heart_odeform.ode_function.activation_timings
            error_vals = Float64[]
            for cell in heart_mesh.grid.cells
                for node in cell.nodes
                    coords = heart_mesh.grid.nodes[node].x
                    push!(error_vals, nodal_timings[node] - (norm(coords) - 0.7))
                end
            end
            @test maximum(error_vals) < 0.03
            @test minimum(error_vals) ≈ 0.0 atol=1e-12
            push!(results, nodal_timings)
        end
    end
    @testset "Analytical Activation" begin
        @testset "LV-CS" begin
            cs = compute_lv_coordinate_system(heart_mesh)
            activation_protocol = Thunderbolt.AnalyticalTransmembraneStimulationProtocol(
                AnalyticalCoefficient((x, t) -> if (x.transmural < 0.05)
                    0.0
                else
                    NaN
                end, cs),
                [SVector(0.0, 1.0)],
            )
            heart_odeform = semidiscretize(
                ReactionEikonalSplit(heart_model, cs),
                (
                    FiniteElementDiscretization(Dict(:φₘ => LagrangeCollection{1}())),
                    Thunderbolt.SimplicialEikonalDiscretization(;
                        activation_protocol,
                        subdomains = String[],
                    ),
                ),
                heart_mesh,
            )
            FastIterativeMethod.solve!(heart_odeform, heart_mesh)
            nodal_timings = heart_odeform.ode_function.activation_timings
            error_vals = Float64[]
            for cell in heart_mesh.grid.cells
                for node in cell.nodes
                    coords = heart_mesh.grid.nodes[node].x
                    push!(error_vals, nodal_timings[node] - (norm(coords) - 0.7))
                end
            end
            @test maximum(error_vals) < 0.03
            @test minimum(error_vals) ≈ 0.0 atol=1e-12
            push!(results, nodal_timings)
        end

        @testset "Cartesian-CS" begin
            cs = CartesianCoordinateSystem(heart_mesh)
            activation_protocol = Thunderbolt.AnalyticalTransmembraneStimulationProtocol(
                AnalyticalCoefficient((x, t) -> if (norm(x) < 0.70001)
                    0.0
                else
                    NaN
                end, cs),
                [SVector(0.0, 1.0)],
            )
            heart_odeform = semidiscretize(
                ReactionEikonalSplit(heart_model, cs),
                (
                    FiniteElementDiscretization(Dict(:φₘ => LagrangeCollection{1}())),
                    Thunderbolt.SimplicialEikonalDiscretization(;
                        activation_protocol,
                        subdomains = String[],
                    ),
                ),
                heart_mesh,
            )
            FastIterativeMethod.solve!(heart_odeform, heart_mesh)
            nodal_timings = heart_odeform.ode_function.activation_timings
            error_vals = Float64[]
            for cell in heart_mesh.grid.cells
                for node in cell.nodes
                    coords = heart_mesh.grid.nodes[node].x
                    push!(error_vals, nodal_timings[node] - (norm(coords) - 0.7))
                end
            end
            @test maximum(error_vals) < 0.03
            @test minimum(error_vals) ≈ 0.0 atol=1e-12                                                                             # hide
            push!(results, nodal_timings)
        end
    end
    @test results[1] ≈ results[2] ≈ results[3] ≈ results[4]
end

@testset "Subdomains" begin
    heart_mesh =
        generate_ideal_lv_mesh(
            12,
            2,
            5;
            inner_radius = 0.7,
            outer_radius = 1.0,
            longitudinal_upper = 0.0,
            apex_inner = 0.7,
            apex_outer = 1.0,
        ) |> Thunderbolt.tetrahedralize;
    addcellset!(heart_mesh.grid, "Left", x -> (x[1] < -1e-6 && x[3] < 0.35); all = false)
    addcellset!(heart_mesh.grid, "Right", x -> (x[1] > 1e-6 && x[3] < 0.35); all = false)
    addcellset!(heart_mesh.grid, "Bot", x -> x[3] >= 0.35)
    heart_mesh = Thunderbolt.to_mesh(heart_mesh.grid)
    microstructure = OrthotropicMicrostructureModel(
        ConstantCoefficient((Vec(0.0, 0.0, 1.0))),
        ConstantCoefficient((Vec(0.0, 1.0, 0.0))),
        ConstantCoefficient((Vec(1.0, 0.0, 0.0))),
    )
    κ₁ = 0.17 * 0.62 / (0.17 + 0.62) #Copied from a tutorial
    diffusion_tensor_field =
        SpectralTensorCoefficient(microstructure, ConstantCoefficient(SVector(κ₁, κ₁, κ₁)))
    cellmodel = Thunderbolt.PCG2019() # Does not really matter for this test
    heart_model = MonodomainModel(
        ConstantCoefficient(1.0),
        ConstantCoefficient(1.0),
        diffusion_tensor_field,
        NoStimulationProtocol(),
        cellmodel,
        :φₘ,
        :s,
    )
    results = Vector{Float64}[]
    @testset "Uniform Activation" begin
        @testset "LV-CS" begin
            cs = compute_lv_coordinate_system(heart_mesh, ["Left", "Right", "Bot"])
            activation_protocol = Thunderbolt.UniformEndocardialActivationProtocol(
                Dict("Left" => 1.0, "Right" => -1.0),
                cs,
            )
            heart_odeform = semidiscretize(
                ReactionEikonalSplit(heart_model, cs),
                (
                    FiniteElementDiscretization(Dict(:φₘ => LagrangeCollection{1}())),
                    Thunderbolt.SimplicialEikonalDiscretization(;
                        activation_protocol,
                        subdomains = String["Left", "Right", "Bot"],
                    ),
                ),
                heart_mesh,
            )
            FastIterativeMethod.solve!(heart_odeform, heart_mesh)
            nodal_timings = heart_odeform.ode_function.activation_timings
            push!(results, nodal_timings)
        end
        @testset "Cartesian-CS" begin
            cs = CartesianCoordinateSystem(heart_mesh)
            activation_protocol = Thunderbolt.UniformEndocardialActivationProtocol(
                Dict("Left" => 1.0, "Right" => -1.0),
                cs,
            )
            @test_throws "Uniformally activating the endocardium requires using either
    LV or BiV coordinate system. usage with Cartesian Coordinate System is
    restricted to AnalyticalTransmembraneStimulationProtocol" semidiscretize(
                ReactionEikonalSplit(heart_model, cs),
                (
                    FiniteElementDiscretization(Dict(:φₘ => LagrangeCollection{1}())),
                    Thunderbolt.SimplicialEikonalDiscretization(;
                        activation_protocol,
                        subdomains = String["Left", "Right", "Bot"],
                    ),
                ),
                heart_mesh,
            )
        end
    end
    @testset "Analytical Activation" begin
        @testset "LV-CS" begin
            cs = compute_lv_coordinate_system(heart_mesh, ["Left", "Right", "Bot"])
            activation_protocol = Thunderbolt.AnalyticalTransmembraneStimulationProtocol(
                AnalyticalCoefficient(
                    (x, t) ->
                        if (x.transmural < 0.05 && x.apicobasal > 0.88 && x.rotational > 0.5)
                            -1.0
                        elseif (x.transmural < 0.05 && x.apicobasal > 0.88)
                            1.0
                        else
                            NaN
                        end,
                    cs,
                ),
                [SVector(0.0, 1.0)],
            )
            heart_odeform = semidiscretize(
                ReactionEikonalSplit(heart_model, cs),
                (
                    FiniteElementDiscretization(Dict(:φₘ => LagrangeCollection{1}())),
                    Thunderbolt.SimplicialEikonalDiscretization(;
                        activation_protocol,
                        subdomains = String["Left", "Right", "Bot"],
                    ),
                ),
                heart_mesh,
            )
            FastIterativeMethod.solve!(heart_odeform, heart_mesh)
            nodal_timings = heart_odeform.ode_function.activation_timings
            push!(results, nodal_timings)
        end

        @testset "Cartesian-CS" begin
            cs = CartesianCoordinateSystem(heart_mesh)
            activation_protocol = Thunderbolt.AnalyticalTransmembraneStimulationProtocol(
                AnalyticalCoefficient(
                    (x, t) -> if (norm(x) < 0.70001 && x[1] > 1e-6 && x[3] <= 0.351)
                        -1.0
                    elseif (norm(x) < 0.70001 && x[1] < 1e-6 && x[3] <= 0.351)
                        1.0
                    else
                        NaN
                    end,
                    cs,
                ),
                [SVector(0.0, 1.0)],
            )
            heart_odeform = semidiscretize(
                ReactionEikonalSplit(heart_model, cs),
                (
                    FiniteElementDiscretization(Dict(:φₘ => LagrangeCollection{1}())),
                    Thunderbolt.SimplicialEikonalDiscretization(;
                        activation_protocol,
                        subdomains = String["Left", "Right", "Bot"],
                    ),
                ),
                heart_mesh,
            )
            FastIterativeMethod.solve!(heart_odeform, heart_mesh)
            nodal_timings = heart_odeform.ode_function.activation_timings
            nodal_timings = heart_odeform.ode_function.activation_timings

            push!(results, nodal_timings)
        end
    end
    # TODO: put this in a loop
    errors = Float64[]
    for cell in heart_mesh.grid.cellsets["Left"]
        for node in heart_mesh.grid.cells[cell].nodes
            coords = heart_mesh.grid.nodes[node].x
            timing = 0.133418*results[1][node] + 0.566582 #Mapping from -1:whatever to 0.7:1
            coords[1] > -0.30 && continue # boundary
            push!(errors, abs(timing - norm(coords)))
        end
    end
    @test minimum(errors) ≈ 0.0 atol=1e-15
    @test maximum(errors) < 0.03 # 0.02? (the extra node due to tetrahedralize) at the endo then propagated to 0.03? at the epi

    errors = Float64[]
    for cell in heart_mesh.grid.cellsets["Right"]
        for node in heart_mesh.grid.cells[cell].nodes
            coords = heart_mesh.grid.nodes[node].x
            timing = ((((results[1][node]+1)/2.24858)*0.3)+0.7)  #Mapping from 1:whatever to 0.7:1
            coords[1] < 0.30 && continue # boundary
            push!(errors, abs(timing - norm(coords)))
        end
    end
    @test minimum(errors) ≈ 0.0 atol=1e-6
    @test maximum(errors) < 0.03 # 0.02? (the extra node due to tetrahedralize) at the endo then propagated to 0.03? at the epi

    vals_bot = Float64[]
    for cell in heart_mesh.grid.cellsets["Bot"]
        for node in heart_mesh.grid.cells[cell].nodes
            coords = heart_mesh.grid.nodes[node].x
            coords[3] > 0.35 && continue # boundary
            push!(vals_bot, results[1][node])
        end
    end
    @test all(isinf, vals_bot)
    @test results[1] ≈ results[2] ≈ results[3]
end
