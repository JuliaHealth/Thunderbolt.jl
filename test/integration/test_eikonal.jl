using Thunderbolt
using FastIterativeMethod
using LinearAlgebra
using OrdinaryDiffEqRosenbrock
using OrdinaryDiffEqOperatorSplitting
using DiffEqBase
@testset "EP wave propagation" begin
    microstructure = OrthotropicMicrostructureModel(
        ConstantCoefficient((Vec(1.0, 0.0, 0.0))),
        ConstantCoefficient((Vec(0.0, 1.0, 0.0))),
        ConstantCoefficient((Vec(0.0, 0.0, 1.0))),
    )

    function steady_state_initializer!(u₀, f::Thunderbolt.ReactionEikonalFunction)
        ## TODO cleaner implementation. We need to extract this from the types or via dispatch.
        odefun = f.ode_function
        ionic_model = odefun.cellmodel

        φ₀ = @view u₀[1:solution_size(f.eikonal_function)]
        ## TODO extraction these via utility functions
        s₀flat = @view u₀[(solution_size(f.eikonal_function)+1):end]
        ## Should not be reshape but some array of arrays fun
        s₀ = reshape(
            s₀flat,
            (solution_size(f.eikonal_function), Thunderbolt.num_states(ionic_model) - 1),
        )
        default_values = Thunderbolt.default_initial_state(ionic_model)

        φ₀ .= default_values[1]
        for i = 1:(Thunderbolt.num_states(ionic_model)-1)
            s₀[:, i] .= default_values[i+1]
        end
        return
    end
    function create_helper_dh(mesh, ipc, sym)
        ad = Thunderbolt.ApproximationDescriptor(sym, ipc)
        dh = DofHandler(mesh)
        if isempty(mesh.grid.cellsets)
            sdh = SubDofHandler(dh, Set{Int}(1:length(mesh.grid.cells)))
            add!(sdh, sym, getinterpolation(ipc, mesh.grid.cells[first(sdh.cellset)]))
        else
            for cellset in mesh.grid.cellsets
                Thunderbolt.add_subdomain!(dh, cellset.first, [ad])
            end
        end
        close!(dh)
        return dh
    end
    function solve_waveprop(mesh, diffusion_tensor_field, subdomains)
        cs = CartesianCoordinateSystem(mesh)

        activation_protocol = Thunderbolt.AnalyticalTransmembraneStimulationProtocol(
            AnalyticalCoefficient((x, t) -> norm(x) ≈ 0.0 ? 0.0 : NaN, cs),
            [SVector(0.0, 100.0)],
        )

        cellmodel = Thunderbolt.PCG2019()

        heart_model = MonodomainModel(
            ConstantCoefficient(1.0),
            ConstantCoefficient(1.0),
            diffusion_tensor_field,
            activation_protocol,
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
        nodal_timings = heart_odeform.ode_function.activation_timings

        dtvis = 0.5
        Tₘₐₓ = 50.0
        tspan = (0.0, Tₘₐₓ)

        single_prob = ODEProblem(
            (du, u, m, t) -> Thunderbolt.cell_rhs!(du, u, m.stim_offset, t, m),
            Thunderbolt.default_initial_state(cellmodel),
            (first(tspan), last(tspan)),
            Thunderbolt.StimulatedCellModel(; cell_model = cellmodel),
        )
        problem = SciMLBase.EnsembleProblem(single_prob, prob_func = heart_odeform.ode_function);

        sim = solve(
            problem,
            Rodas5P(),
            saveat = collect(0.0:dtvis:Tₘₐₓ),
            trajectories = length(heart_odeform.ode_function.activation_timings),
        )
        @test sim.converged

        dh = create_helper_dh(mesh, LagrangeCollection{1}(), :coordinates)
        Thunderbolt.reorder_nodal!(dh)

        φₘfield = copy(nodal_timings)
        for t = 0.0:(Tₘₐₓ/2):Tₘₐₓ                                                          # hide
            for i = 1:length(φₘfield)                                                  # hide
                φₘfield[i] = sim.u[i](t)[1]                                             # hide
            end                                                                         # hide
            @test all(φₘ -> -90.0 < φₘ < 50.0, φₘfield) #PCG2019 range?
        end

        return nodal_timings
    end

    @testset "Anisotropic Eikonal Cube" begin
        errors = Float64[]
        for j in (5, 10, 15)
            mesh = generate_mesh(
                Tetrahedron,
                (j, j, j),
                Vec{3}((0.0, 0.0, 0.0)),
                Vec{3}((1.0, 1.0, 1.0)),
            )
            # velocity = SVector((4.5e-5, 2.0e-5, 1.0e-5))
            velocity = SVector(0.05 .* (1.0, 2.0, 3.0))
            coeff = SpectralTensorCoefficient(microstructure, ConstantCoefficient(velocity))
            u = solve_waveprop(mesh, coeff, String[])
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
end
