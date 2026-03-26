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
    function simple_initializer!(u₀, f::GenericSplitFunction)
        # TODO cleaner implementation. We need to extract this from the types or via dispatch.
        heatfun = f.functions[1]
        heat_dofrange = f.solution_indices[1]
        odefun = f.functions[2]
        ionic_model = odefun.ode

        ϕ₀ = @view u₀[heat_dofrange];

        s₀flat = @view u₀[(last(heat_dofrange)+1):end]
        ## Should not be reshape but some array of arrays fun
        s₀ = reshape(s₀flat, (last(heat_dofrange), Thunderbolt.num_states(ionic_model) - 1))
        default_values = Thunderbolt.default_initial_state(ionic_model)

        ϕ₀ .= default_values[1]
        for i = 1:(Thunderbolt.num_states(ionic_model)-1)
            s₀[:, i] .= default_values[i+1]
        end
    end
    function get_nodes_to_vertex_permutaion(dh::DofHandler)
        res = Vector{Int}(undef, dh.ndofs)
        grid = Ferrite.get_grid(dh)
        for i ∈ 1:getncells(grid)
            res[SVector(getcells(grid, i).nodes)] .=
                (@view dh.cell_dofs[dh.cell_dofs_offset[i]:(dh.cell_dofs_offset[i]+Ferrite.ndofs_per_cell(
                    dh,
                    i,
                )-1)])
        end
        return sortperm(res)
    end
    function steady_state_initializer!(u₀, f)
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
    function solve_waveprop_RE(mesh, diffusion_tensor_field, subdomains)
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

        FastIterativeMethod.solve!(heart_odeform, mesh)
        nodal_timings = heart_odeform.ode_function.activation_timings

        dtvis = 0.5
        Tₘₐₓ = 50.0
        tspan = (0.0, Tₘₐₓ)

        single_prob = ODEProblem(
            (du, u, m, t) -> Thunderbolt.cell_rhs!(du, u, nothing, m.stim_offset, t, m),
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

    function solve_waveprop_RED(mesh, diffusion_tensor_field, subdomains)
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
            NoStimulationProtocol(),
            cellmodel,
            :φₘ,
            :s,
        )

        heart_odeform = semidiscretize(
            ReactionEikonalDiffusionSplit(heart_model, cs),
            (
                FiniteElementDiscretization(Dict(:φₘ => LagrangeCollection{1}())),
                Thunderbolt.SimplicialEikonalDiscretization(; activation_protocol, subdomains),
            ),
            mesh,
        )

        FastIterativeMethod.solve!(heart_odeform, mesh)
        perm = get_nodes_to_vertex_permutaion(
            heart_odeform.reaction_diffusion_function.functions[1].dh,
        )
        nodal_timings = heart_odeform.activation_timings
        nodal_timings_vertexwise = copy(heart_odeform.activation_timings)
        nodal_timings .= nodal_timings[perm]
        u₀ = zeros(Float64, solution_size(heart_odeform.reaction_diffusion_function))
        simple_initializer!(u₀, heart_odeform.reaction_diffusion_function)

        tspan = (0.0, 100.0)
        problem = OperatorSplittingProblem(heart_odeform.reaction_diffusion_function, u₀, tspan)
        u₀ = copy(u₀)
        timestepper = LieTrotterGodunov((BackwardEulerSolver(), ForwardEulerCellSolver()))

        integrator = DiffEqBase.init(problem, timestepper, dt = 0.01, verbose = true)
        # io = ParaViewWriter("EP01_spiral_wave")
        # for (u, t) in TimeChoiceIterator(integrator, tspan[1]:0.1:tspan[2])
        #     (; dh) = heart_odeform.reaction_diffusion_function.functions[1]
        #     φ = u[heart_odeform.reaction_diffusion_function.solution_indices[1]]
        #     store_timestep!(io, t, dh.grid) do file
        #         Thunderbolt.store_timestep_field!(file, t, dh, φ, :φₘ)
        #     end
        # end;
        DiffEqBase.solve!(integrator)

        φₘfield = integrator.u
        @test all(φₘ -> -90.0 < φₘ < 50.0, φₘfield) #PCG2019 range?

        return nodal_timings_vertexwise
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
            u_RE = solve_waveprop_RE(mesh, coeff, String[])
            u_RED = solve_waveprop_RED(mesh, coeff, String[])
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
            @test minimum(u_RE .- u_2)≈0.0 atol=1e-8
            @test minimum(u_RED .- u_2)≈0.0 atol=1e-8
            @test count(isapprox.(u_RE .- u_2, 0.0, atol = 1e-8)) == 1 + 4j
            @test count(isapprox.(u_RED .- u_2, 0.0, atol = 1e-8)) == 1 + 4j
            push!(errors, maximum(u_RE .- u_2))
            push!(errors, maximum(u_RED .- u_2))
        end
        @test issorted(errors, rev = true)
    end
end
