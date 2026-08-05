using Thunderbolt
using Test

# Two dof handlers over the same grid: one carrying the displacement alone, one carrying a velocity
# next to it. This is the pair a Newmark stage wires itself to, built here without a solver so that
# the wiring can be asserted directly rather than inferred from a solve.
function _handlers(; velocity_order = 1)
    mesh = generate_mesh(Hexahedron, (2, 1, 1), Vec((0.0, 0.0, 0.0)), Vec((1.0, 0.2, 0.2)))
    name = Thunderbolt.single_subdomain_or_error(mesh)
    ipc = LagrangeCollection{1}()^3

    dh_u = DofHandler(mesh)
    Thunderbolt.add_subdomain!(dh_u, name, [Thunderbolt.ApproximationDescriptor(:d, ipc)])
    close!(dh_u)

    ipv = velocity_order == 1 ? ipc : LagrangeCollection{2}()^3
    dh_uv = DofHandler(mesh)
    Thunderbolt.add_subdomain!(
        dh_uv,
        name,
        [
            Thunderbolt.ApproximationDescriptor(:d, ipc),
            Thunderbolt.ApproximationDescriptor(:v, ipv),
        ],
    )
    close!(dh_uv)

    return dh_u, dh_uv
end

# `n` condensed unknowns on every cell, laid out after the finite element block as
# `offset(cid)+1 : offset(cid+1)` with absolute, 0-based offsets.
function _ivh(dh, n)
    ncells = getncells(Thunderbolt.get_grid(dh))
    offsets = [ndofs(dh) + (cid - 1) * n for cid = 1:ncells]
    return Thunderbolt.InternalVariableHandler(offsets, ncells * n)
end

@testset "SolutionVectorMapping" begin
    dh_u, dh_uv = _handlers()
    lvh_u, lvh_uv = _ivh(dh_u, 3), _ivh(dh_uv, 3)

    mapping = Thunderbolt.SolutionVectorMapping(
        Thunderbolt.field_dof_mapping(dh_u, :d, dh_uv, :d),
        Thunderbolt.internal_variable_mapping(dh_u, lvh_u, dh_uv, lvh_uv),
    )

    nsource = ndofs(dh_uv) + ndofs(lvh_uv)
    ntarget = ndofs(dh_u) + ndofs(lvh_u)

    @testset "The dof wiring is a bijection onto the field" begin
        @test length(mapping.dofs) == ndofs(dh_u)
        @test allunique(mapping.dofs)
        @test all(d -> 1 ≤ d ≤ ndofs(dh_uv), mapping.dofs)
        # Adding a second field of the same order doubles the handler, and the displacement is half.
        @test ndofs(dh_uv) == 2 * ndofs(dh_u)
    end

    @testset "Displacement and velocity wirings are disjoint" begin
        vdofs = Thunderbolt.field_dof_mapping(dh_u, :d, dh_uv, :v)
        @test length(vdofs) == length(mapping.dofs)
        @test isempty(intersect(Set(vdofs), Set(mapping.dofs)))
        # Together they cover the whole handler: there is no third field.
        @test sort(vcat(vdofs, mapping.dofs)) == collect(1:ndofs(dh_uv))
    end

    @testset "The internal variable wiring covers the tail" begin
        @test length(mapping.internal_variables) == ndofs(lvh_u)
        @test allunique(mapping.internal_variables)
        @test sort(mapping.internal_variables) ==
              collect(Thunderbolt.internal_variable_range(dh_uv, lvh_uv))
    end

    @testset "Gather then scatter is a round trip" begin
        source = collect(1.0:nsource)
        original = copy(source)
        target = zeros(ntarget)

        Thunderbolt.gather!(target, source, mapping)
        # Wipe the source so that a scatter which failed to write would be visible as a zero rather
        # than as a value that happened to survive.
        fill!(source, 0.0)
        Thunderbolt.scatter!(source, target, mapping)

        wired = vcat(mapping.dofs, mapping.internal_variables)
        @test source[wired] == original[wired]
        @test all(iszero, source[setdiff(1:nsource, wired)])
    end

    @testset "Scatter touches nothing outside the wiring" begin
        source = collect(1.0:nsource)
        original = copy(source)
        target = fill(-1.0, ntarget)

        Thunderbolt.scatter!(source, target, mapping)

        untouched = setdiff(1:nsource, vcat(mapping.dofs, mapping.internal_variables))
        @test source[untouched] == original[untouched]
        @test all(==(-1.0), source[mapping.dofs])
    end

    @testset "The identity mapping does not copy an aliased vector" begin
        identity = Thunderbolt.IdentitySolutionVectorMapping()
        u = collect(1.0:10)
        @test Thunderbolt.gather!(u, u, identity) === u
        @test Thunderbolt.scatter!(u, u, identity) === u
        @test u == collect(1.0:10)
    end

    @testset "Mismatched interpolations are refused" begin
        # A velocity of a different order cannot be wired to the displacement dof for dof, and the
        # element evaluates the reconstructed velocity with the displacement's `CellValues`, so this
        # has to fail loudly rather than produce a partial map.
        _, dh_uv2 = _handlers(velocity_order = 2)
        @test_throws Exception Thunderbolt.field_dof_mapping(dh_u, :d, dh_uv2, :v)
    end

    @testset "Unequal condensed blocks are refused" begin
        # The two sides must agree cell by cell on how many unknowns are condensed; disagreeing would
        # silently shift every internal variable after the first differing cell.
        @test_throws Exception Thunderbolt.internal_variable_mapping(
            dh_u,
            _ivh(dh_u, 3),
            dh_uv,
            _ivh(dh_uv, 4),
        )
    end
end
