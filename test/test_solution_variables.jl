using Test
using Thunderbolt
using OrdinaryDiffEqOperatorSplitting
using LinearAlgebra
using OrderedCollections
using StaticArrays
using FerriteInterfaceElements

const T = Thunderbolt

# Coordinates of the transmembrane potential dofs, for checking analytical initialisation.
function dof_coordinates(form, φsym)
    dh = form.functions[1].dh
    return T.compute_nodal_values(CartesianCoordinateSystem(dh.grid), dh, φsym)
end

# A 2D box with a monodomain model on it. `ion` and `coords` are what the individual tests vary.
function ep_form(; ion = T.FHNModel(), n = 4, coords = :cartesian, φsym = :φₘ, ssym = :s)
    mesh = generate_mesh(Quadrilateral, (n, n), Vec{2}((0.0, 0.0)), Vec{2}((1.0, 1.0)))
    cs = coords === :cartesian ? CartesianCoordinateSystem(mesh) : coords
    model = MonodomainModel(
        ConstantCoefficient(1.0),
        ConstantCoefficient(1.0),
        ConstantCoefficient(SymmetricTensor{2, 2, Float64}((1.0e-4, 0.0, 1.0e-4))),
        T.NoStimulationProtocol(),
        ion,
        cs,
        φsym,
        ssym,
    )
    form = semidiscretize(
        ReactionDiffusionSplit(model),
        FiniteElementDiscretization(Dict(φsym => LagrangeCollection{1}())),
        mesh,
    )
    return mesh, form
end

@testset "Solution variables" begin
    @testset "Publishing and lookup" begin
        mesh, form = ep_form()
        @test Set(solution_variable_names(form)) == Set([:φₘ, :s])
        @test T.validate_solution_variables(form)

        ndofsφ = ndofs(form.functions[1].dh)
        @test length(solution_indices(form, :φₘ)) == ndofsφ
        # FitzHugh-Nagumo has one recovery state, so `:s` covers one value per point.
        @test length(solution_indices(form, :s)) == ndofsφ

        # An unknown name has to say what *is* available rather than fail obscurely.
        err = try
            solution_variable(form, :nope)
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("φₘ", err.msg) && occursin("nope", err.msg)
    end

    @testset "Round trip through the three kinds" begin
        mesh, form = ep_form()
        u = create_initial_condition(form)

        setvariable!(u, form, :φₘ, x -> x[1])
        setvariable!(u, form, :s, 0.25)
        @test getvariable(u, form, :φₘ) ≈ [x[1] for x in dof_coordinates(form, :φₘ)]
        @test all(==(0.25), getvariable(u, form, :s))

        # The do-block form is the same call.
        setvariable!(u, form, :s) do x
            x[2]
        end
        @test getvariable(u, form, :s) ≈ [x[2] for x in dof_coordinates(form, :φₘ)]
    end

    @testset "The transmembrane potential may sit at any state index" begin
        # `ParametrizedAlievPanfilovModel` keeps the potential at index 2 of 2, so a heat problem wired
        # to `1:ndofs` would silently solve on the recovery variable instead.
        mesh, form = ep_form(ion = T.AlievPanfilovModel())
        ndofsφ = ndofs(form.functions[1].dh)
        @test T.transmembranepotential_index(T.AlievPanfilovModel()) == 2
        @test solution_indices(form, :φₘ) == collect((ndofsφ+1):(2*ndofsφ))
        # ... and the split's own heat index set has to agree with the descriptor.
        @test collect(form.solution_indices[1]) == solution_indices(form, :φₘ)

        _, form1 = ep_form(ion = T.FHNModel())
        @test solution_indices(form1, :φₘ) == collect(1:ndofs(form1.functions[1].dh))
    end

    @testset "Custom transmembrane potential name" begin
        mesh, form = ep_form(φsym = :V, ssym = :w)
        @test Set(solution_variable_names(form)) == Set([:V, :w])
        # The cell model's own vocabulary is untouched: it still calls its states (:φₘ, :s).
        @test T.state_symbols(T.FHNModel()) == (:φₘ, :s)
        u = create_initial_condition(form)
        setvariable!(u, form, :V, 1.5)
        @test all(==(1.5), getvariable(u, form, :V))
    end

    @testset "Defaults come from the cell model" begin
        # `default_initial_state` had no consumer before; now it is what seeds a solution vector.
        mesh, form = ep_form(ion = T.ParametrizedPCG2019Model{Float64}())
        u = create_initial_condition(form)
        expected = T.default_initial_state(T.ParametrizedPCG2019Model{Float64}())
        @test all(≈(expected[1]), getvariable(u, form, :φₘ))
        # `variable_indices` is point major, so the flat view reshapes to (ncomponents, npoints).
        s = reshape(getvariable(u, form, :s), 6, :)
        for j = 1:6
            @test all(≈(expected[j+1]), @view s[j, :])
        end
    end

    @testset "Layout agrees with the solver cache" begin
        # The descriptor derives the state layout independently of `setup_solver_cache`; if the two ever
        # disagree, a solution vector written by name would be read back scrambled by the solver.
        mesh, form = ep_form(ion = T.ParametrizedPCG2019Model{Float64}())
        odefun = form.functions[2]
        cache = T.setup_solver_cache(odefun, ForwardEulerCellSolver(), 0.0)
        points = T.solution_variable(form, :s).points
        u = zeros(solution_size(form))
        uₙmat = reshape(view(u, odefun.associated_states), size(cache.uₙmat))
        for (k, j) in ((1, 1), (3, 4), (T.npoints(points), 7))
            fill!(u, 0.0)
            uₙmat[k, j] = 1.0
            @test u[T.state_range(points, k)[j]] == 1.0
        end
    end

    @testset "Nested splits and tree-wide uniqueness" begin
        mesh, form = ep_form()
        n = solution_size(form)
        outer = GenericSplitFunction((form, T.NullFunction(3)), (1:n, (n+1):(n+3)))

        # Every symbol resolves to the same place whether asked at the root or at the inner node.
        @test Set(solution_variable_names(outer)) == Set([:φₘ, :s])
        @test solution_indices(outer, :φₘ) == solution_indices(form, :φₘ)
        @test solution_indices(outer, :s) == solution_indices(form, :s)
        @test T.validate_solution_variables(outer)

        # Initialisation composes through the nesting without clobbering the sibling block.
        u = create_initial_condition(outer)
        setvariable!(u, outer, :s, 0.75)
        @test all(==(0.75), getvariable(u, outer, :s))
        @test all(iszero, u[(n+1):(n+3)])

        # Two unrelated problems publishing the same field symbol is a modelling error, not something to
        # resolve silently by precedence.
        _, other = ep_form()
        clash = GenericSplitFunction((form, other), (1:n, 1:n))
        @test_throws ErrorException solution_variables(clash)
    end

    @testset "Subdomains with different cell models" begin
        grid = generate_grid(Quadrilateral, (8, 8), Vec{2}((-1.0, -1.0)), Vec{2}((1.0, 1.0)))
        addcellset!(grid, "Fast", x -> norm(x, Inf) ≤ 0.5)
        addcellset!(grid, "Slow", setdiff(OrderedSet(1:getncells(grid)), getcellset(grid, "Fast")))
        mesh = to_mesh(insert_interfaces(grid, ["Fast", "Slow"]))
        coeff = ConstantCoefficient(SymmetricTensor{2, 2, Float64}((1.0e-4, 0, 1.0e-4)))

        # Different state counts (2 vs 7), different transmembrane potential indices (1 vs 1), and
        # different coordinate systems per subdomain -- none of which a single stride could describe.
        models = Dict(
            "Fast" => MonodomainModel(
                ConstantCoefficient(1.0),
                ConstantCoefficient(1.0),
                coeff,
                T.NoStimulationProtocol(),
                T.FHNModel(),
                CellIndexCoordinateSystem(),
                :φₘ,
                :sfast,
            ),
            "Slow" => MonodomainModel(
                ConstantCoefficient(1.0),
                ConstantCoefficient(1.0),
                coeff,
                T.NoStimulationProtocol(),
                T.ParametrizedPCG2019Model{Float64}(),
                CartesianCoordinateSystem(mesh),
                :φₘ,
                :sslow,
            ),
            "interfaces" => InterfaceDiffusionModel(ConstantCoefficient(1.0), :φₘ, :φₘi),
        )
        form = semidiscretize(
            ReactionDiffusionSplit(models),
            FiniteElementDiscretization(
                Dict(
                    :φₘ => LagrangeCollection{1}(),
                    :φₘi => T.InterfaceCollection(LagrangeCollection{1}()),
                ),
            ),
            mesh,
        )

        @test Set(solution_variable_names(form)) == Set([:φₘ, :sfast, :sslow])
        @test T.validate_solution_variables(form)

        # Each subdomain sees its own coordinate type.
        @test eltype(T.solution_variable(form, :sfast).coordinates) === Int
        @test eltype(T.solution_variable(form, :sslow).coordinates) <: Vec

        # The heat index set is exactly the transmembrane potential slots, and it is ordered by dof of the
        # heat problem rather than by the order the subdomain dictionary happens to iterate in.
        dh = form.functions[1].dh
        @test length(solution_indices(form, :φₘ)) == ndofs(dh)
        @test collect(form.solution_indices[1]) == solution_indices(form, :φₘ)

        # Writing one subdomain's state must not touch the other's.
        u = create_initial_condition(form)
        before = copy(getvariable(u, form, :sslow))
        setvariable!(u, form, :sfast, -1.0)
        @test all(==(-1.0), getvariable(u, form, :sfast))
        @test getvariable(u, form, :sslow) == before
    end
end
