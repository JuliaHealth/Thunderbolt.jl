using Test, Thunderbolt, Ferrite, LinearAlgebra

# Dof of every node of `grid` in the continuous dof handler `dh`.
function node_to_dof(dh)
    map = zeros(Int, getnnodes(Ferrite.get_grid(dh)))
    for sdh in dh.subdofhandlers, cell in CellIterator(sdh)
        for (i, nodeid) in enumerate(getnodes(cell))
            map[nodeid] = celldofs(cell)[i]
        end
    end
    return map
end

# Widest range the rotational coordinate takes within a single element. A coordinate whose branch cut
# is smeared over a layer of elements has an element spanning nearly the whole range; one whose cut
# sits on element interfaces has none wider than the element spacing.
function max_element_rotational_spread(cs; skip_singular = false)
    spread = 0.0
    for sdh in cs.dh_rotational.subdofhandlers, cell in CellIterator(sdh)
        if skip_singular
            minimum(hypot(x[1], x[2]) for x in getcoordinates(cell)) < 1.0e-9 && continue
        end
        values = cs.u_rotational[celldofs(cell)]
        spread = max(spread, maximum(values) - minimum(values))
    end
    return spread
end

# All rotational values that the elements incident to `nodeid` assign to it.
function rotational_values_at_nodes(cs)
    values = [Float64[] for _ = 1:getnnodes(Ferrite.get_grid(cs.dh_rotational))]
    for sdh in cs.dh_rotational.subdofhandlers, cell in CellIterator(sdh)
        dofs = celldofs(cell)
        for (i, nodeid) in enumerate(getnodes(cell))
            push!(values[nodeid], cs.u_rotational[dofs[i]])
        end
    end
    return values
end

@testset "Coordinate systems" begin

    @testset "LV axes" begin
        mesh = generate_ideal_lv_mesh(8, 2, 3; apex_outer = 1.5)
        axes = compute_lv_axes(mesh)

        # The generator puts the apex on the positive z axis and the base below it.
        @test axes.longitudinal ≈ Vec((0.0, 0.0, -1.0)) atol = 1.0e-8
        @test axes.apex ≈ Vec((0.0, 0.0, 1.5)) atol = 1.0e-8
        @test axes.base_center[1] ≈ 0.0 atol = 1.0e-8
        @test axes.base_center[2] ≈ 0.0 atol = 1.0e-8

        # Orthonormal, right handed.
        @test norm(axes.lateral) ≈ 1.0
        @test axes.lateral ⋅ axes.longitudinal ≈ 0.0 atol = 1.0e-12
        @test axes.anteroposterior ≈ axes.longitudinal × axes.lateral

        # Reading the apex off the nodeset must agree with searching for it.
        @test compute_lv_axes(mesh; apex = "Apex").longitudinal ≈ axes.longitudinal atol = 1.0e-8

        # Supplying the basal plane instead of the facetset gives the same axis. The normal points
        # from the apex towards the base, which here is towards -z.
        from_plane =
            compute_lv_axes(mesh, Vec((0.0, 0.0, -0.4)), Vec((0.0, 0.0, -1.0)))
        @test from_plane.longitudinal ≈ axes.longitudinal atol = 1.0e-8
        @test from_plane.apex ≈ axes.apex atol = 1.0e-8

        @test_throws ArgumentError LVAxes(
            Vec((0.0, 0.0, 1.0)),
            Vec((0.0, 0.0, 2.0)),
            zero(Vec{3, Float64}),
            zero(Vec{3, Float64}),
        )
    end

    @testset "Apicobasal arc length recalibration" begin
        # On a ring pinned at its two flat faces the Laplace field is linear in z, so ‖∇u‖ is
        # constant, arc length is proportional to u, and the recalibration must be the identity.
        mesh = generate_ring_mesh(16, 2, 6)
        ipc = LagrangeCollection{1}()
        dh = DofHandler(mesh)
        Thunderbolt.add_subdomain!(
            dh,
            [Thunderbolt.ApproximationDescriptor(:coordinates, ipc)],
        )
        Ferrite.close!(dh)
        K = Thunderbolt._assemble_laplacian(dh, ipc)
        u = Thunderbolt._solve_dirichlet_laplace(
            K,
            dh,
            Thunderbolt.LinearSolve.KrylovJL_CG(),
            [(getfacetset(mesh, "Myocardium"), 0.0), (getfacetset(mesh, "Base"), 1.0)],
        )
        @test apicobasal_from_laplace(dh, ipc, u) ≈ clamp.(u, 0.0, 1.0) atol = 5.0e-3
    end

    @testset "Ideal LV coordinate system" begin
        num_c, num_r, num_l = 16, 2, 5
        mesh = generate_ideal_lv_mesh(num_c, num_r, num_l)
        cs = compute_lv_coordinate_system(mesh)
        n2d = node_to_dof(cs.dh)

        @testset "transmural" begin
            @test all(0.0 .≤ cs.u_transmural .≤ 1.0)
            # Exactly, not to the solver's tolerance: a harmonic coordinate that stops at
            # 1 - 3e-10 on the surface it is pinned to is one whose endpoints depend on the solver.
            for (cellid, local_facet) in getfacetset(mesh, "Endocardium")
                for nodeid in Ferrite.facets(getcells(mesh, cellid))[local_facet]
                    @test cs.u_transmural[n2d[nodeid]] == 0.0
                end
            end
            for (cellid, local_facet) in getfacetset(mesh, "Epicardium")
                for nodeid in Ferrite.facets(getcells(mesh, cellid))[local_facet]
                    @test cs.u_transmural[n2d[nodeid]] == 1.0
                end
            end
        end

        @testset "apicobasal" begin
            @test all(0.0 .≤ cs.u_apicobasal .≤ 1.0)
            for nodeid in getnodeset(mesh, "Apex")
                @test cs.u_apicobasal[n2d[nodeid]] == 0.0
            end
            for (cellid, local_facet) in getfacetset(mesh, "Base")
                for nodeid in Ferrite.facets(getcells(mesh, cellid))[local_facet]
                    @test cs.u_apicobasal[n2d[nodeid]] == 1.0
                end
            end

            # The apical end of the coordinate is an annotation, not a hardcoded name. Pinning the
            # whole apical wall rather than just its epicardial end is a materially different
            # coordinate -- see the docstring for why it is not the default.
            through_wall = compute_lv_coordinate_system(mesh; apex_nodeset = "ApexInOut")
            for nodeid in getnodeset(mesh, "ApexInOut")
                @test through_wall.u_apicobasal[n2d[nodeid]] == 0.0
            end
            @test maximum(abs, through_wall.u_apicobasal - cs.u_apicobasal) > 0.01

            # The coordinate is arc length along the wall, so refining the mesh has to bring it
            # closer to the normalized arc length of an epicardial meridian. The raw harmonic field
            # does the opposite of that: it is pinned at a single node, behaves like a point source,
            # and spends half its range on the last few percent of wall next to the apex.
            function meridian_error(nc, nr, nl; apex_nodeset)
                m = generate_ideal_lv_mesh(nc, nr, nl)
                c = compute_lv_coordinate_system(m; apex_nodeset)
                dofof = node_to_dof(c.dh)
                nodes = reshape(collect(1:(nc*(nr+1)*(nl+1))), (nc, nr + 1, nl + 1))
                ids = [nodes[1, end, k] for k = 1:(nl+1)]
                apexid = argmax([get_node_coordinate(m.grid, i)[3] for i = 1:getnnodes(m)])
                points = [get_node_coordinate(m.grid, i) for i in vcat(apexid, ids)]
                s = cumsum(vcat(0.0, [norm(points[i+1] - points[i]) for i = 1:(length(points)-1)]))
                s ./= s[end]
                ab = [c.u_apicobasal[dofof[i]] for i in vcat(apexid, ids)]
                return maximum(abs.(ab .- s))
            end
            # A meridian is a fair proxy for a trajectory of `ua` only where the level sets are
            # proper apical caps, which is what pinning both ends of the apical wall buys. The
            # default pins the epicardial apex alone, which skews them and loosens the proxy near
            # the tip -- a cost it more than repays in agreement between two differently shaped
            # ventricles, see the docstring. Both converge; they converge to different places.
            for (set, tol) in (("ApexInOut", 0.05), ("Apex", 0.08))
                coarse = meridian_error(16, 2, 5; apex_nodeset = set)
                fine = meridian_error(16, 2, 20; apex_nodeset = set)
                @test fine < coarse
                @test fine < tol
            end
        end

        @testset "rotational" begin
            values = rotational_values_at_nodes(cs)
            ridge_nodes(name) = Set{Int}(
                nodeid for (cellid, local_facet) in getfacetset(mesh, name) for
                nodeid in Ferrite.facets(getcells(mesh, cellid))[local_facet]
            )
            posterior = ridge_nodes("SRidgePost")
            anterior = ridge_nodes("SRidgeAnt")
            @test !isempty(posterior) && !isempty(anterior)

            # The chart is anchored on the ridges: 0 on the free wall side of the posterior ridge,
            # 1 on its septal side, and 2/3 on the anterior ridge from both sides.
            checked_posterior = 0
            for nodeid in setdiff(posterior, anterior)
                x = get_node_coordinate(mesh.grid, nodeid)
                hypot(x[1], x[2]) > 1.0e-9 || continue # the two ridges merge on the apex line
                @test minimum(values[nodeid]) ≈ 0.0 atol = 1.0e-8
                @test maximum(values[nodeid]) ≈ 1.0 atol = 1.0e-8
                checked_posterior += 1
            end
            @test checked_posterior > 0
            for nodeid in setdiff(anterior, posterior)
                x = get_node_coordinate(mesh.grid, nodeid)
                hypot(x[1], x[2]) > 1.0e-9 || continue
                @test all(v -> isapprox(v, 2 / 3; atol = 1.0e-8), values[nodeid])
            end

            # Away from the ridges neighbouring elements agree, so the jump really is confined to
            # the posterior ridge rather than smeared over a layer of elements.
            for nodeid = 1:getnnodes(mesh)
                nodeid in posterior && continue
                @test maximum(values[nodeid]) - minimum(values[nodeid]) < 1.0e-8
            end

            @test all(v -> all(0.0 .≤ v .≤ 1.0), values)
            # Away from the apex line no element sees more than the sliver of the range it spans. A
            # smeared branch cut would instead put an element across nearly the whole range. The
            # elements on the apex line are excluded because the two ridges merge there and the
            # coordinate genuinely has no value.
            @test max_element_rotational_spread(cs; skip_singular = true) < 3 / num_c
        end

        @testset "rotational is the same chart on differently resolved meshes" begin
            # An idealized ventricle is a surface of revolution whose ridges cut it into a third and
            # two thirds, so the Cobiveco coordinate has a closed form there: it is the azimuth,
            # running the opposite way to φ because it increases right-handed about the long axis,
            # and the long axis points at the base. A mesh independent ground truth is the sharpest
            # statement of "two meshes of the same heart agree".
            function azimuth_error(nc, nr, nl; apical_cutoff)
                m = generate_ideal_lv_mesh(nc, nr, nl)
                c = compute_lv_coordinate_system(m)
                worst = 0.0
                for sdh in c.dh_rotational.subdofhandlers, cell in CellIterator(sdh)
                    for (dof, x) in zip(celldofs(cell), getcoordinates(cell))
                        hypot(x[1], x[2]) > 1.0e-9 || continue
                        x[3] < apical_cutoff || continue
                        exact = 1 - mod(atan(x[2], x[1]), 2π) / (2π)
                        d = abs(Thunderbolt.wrap_rotational(c.u_rotational[dof]) - exact)
                        worst = max(worst, min(d, 1 - d))
                    end
                end
                return worst
            end
            # The ridges converge on the singular apex line, where they cannot both be pinned and
            # the coordinate has no value, so the closed form only holds away from it.
            for nc in (12, 24)
                @test azimuth_error(nc, 2, 8; apical_cutoff = 1.0) < 5.0e-3
            end
        end

        @testset "quadrature point evaluation" begin
            qr = getquadraturerule(QuadratureRuleCollection(2), getcells(mesh, 1))
            sdh = first(cs.dh.subdofhandlers)
            cache = Thunderbolt.duplicate_for_device(
                PolyesterDevice(),
                Thunderbolt.setup_coefficient_cache(cs, qr, sdh),
            )
            cartesian = Thunderbolt.duplicate_for_device(
                PolyesterDevice(),
                Thunderbolt.setup_coefficient_cache(
                    CartesianCoordinateSystem(mesh),
                    qr,
                    sdh,
                ),
            )
            for cell in CellIterator(sdh), qp in QuadratureIterator(qr)
                coordinate = evaluate_coefficient(cache, cell, qp, 0.0)
                # The stored dofs straddle the ridge, so this is only in range because the
                # interpolated value is wrapped back into the coordinate's own range.
                @test 0.0 ≤ coordinate.rotational < 1.0
                @test 0.0 ≤ coordinate.transmural ≤ 1.0
                @test 0.0 ≤ coordinate.apicobasal ≤ 1.0
            end
        end
    end

    @testset "The ridges determine the septum without geometry" begin
        # The ridge sheets are the interface between the two regions, so the partition they induce
        # *is* the segmentation -- no centroid needs to be classified. On a mesh whose ridges cut it
        # cleanly the two agree; where they disagree it is the arc rule that is wrong, since a cell
        # within half an element of a ridge can fall on either side of it.
        num_c = 12
        mesh = generate_ideal_lv_mesh(num_c, 2, 4)
        axes = compute_lv_axes(mesh)
        origin = Thunderbolt._to_f64(axes.base_center)
        longitudinal = Thunderbolt._to_f64(axes.longitudinal)
        provisional = Thunderbolt.AzimuthalFrame(
            origin, longitudinal, Thunderbolt._any_orthogonal(longitudinal))
        frame = Thunderbolt.AzimuthalFrame(origin, longitudinal,
            Thunderbolt._sheet_direction(mesh, "SRidgePost", provisional))
        θa = Thunderbolt.azimuth(frame, frame.origin +
            Thunderbolt._sheet_direction(mesh, "SRidgeAnt", frame))
        θp = Thunderbolt.azimuth(frame, frame.origin +
            Thunderbolt._sheet_direction(mesh, "SRidgePost", frame))

        septal = Thunderbolt._septal_cells_by_partition(mesh, "SRidgeAnt", "SRidgePost")
        @test septal !== nothing
        @test septal == Thunderbolt._septal_cells_by_arc(mesh, frame, θa, θp)
        # A third of the circumference, and the ridge facets sit on the septal side of the cut.
        @test count(septal) / getncells(mesh) ≈ 1 / 3 atol = 0.02
        for name in ("SRidgeAnt", "SRidgePost"), (cellid, _) in getfacetset(mesh, name)
            @test septal[cellid]
        end

        # The O-grid cap has no facet sheet continuing the ridges through its core, so there is no
        # partition to recover and the arc rule has to take over.
        @test Thunderbolt._septal_cells_by_partition(
            Thunderbolt.generate_ideal_lv_mesh_hex(12, 2, 4), "SRidgeAnt", "SRidgePost") === nothing
    end

    @testset "Ideal LV ridges, arbitrary circumferential count" begin
        # The ridges snap to element interfaces, so the septum does not land on exactly a third of
        # the circumference, but the chart is still anchored on them.
        for num_c in (5, 7, 13)
            mesh = generate_ideal_lv_mesh(num_c, 1, 2)
            cs = compute_lv_coordinate_system(mesh)
            @test all(isfinite, cs.u_rotational)
            @test all(0.0 .≤ cs.u_rotational .≤ 1.0)
            # A smeared branch cut shows up as an element spanning nearly the whole range.
            @test max_element_rotational_spread(cs; skip_singular = true) < 3 / num_c
        end
    end

    @testset "Midmyocardial section coordinate system" begin
        num_c = 40
        mesh = generate_ring_mesh(num_c, 2, 2)
        # A plain ring has no right ventricle attached, so it gets the azimuthal fallback.
        cs = compute_midmyocardial_section_coordinate_system(
            mesh;
            ridge_anterior = nothing,
            ridge_posterior = nothing,
        )

        @test all(isfinite, cs.u_rotational)
        @test max_element_rotational_spread(cs) ≈ 1 / num_c atol = 1.0e-8
        # `up` is +z here, so the coordinate runs the same way as φ.
        @test maximum(
            let d = abs(
                    Thunderbolt.wrap_rotational(cs.u_rotational[dof]) -
                    mod(atan(x[2], x[1]), 2π) / (2π),
                )
                min(d, 1 - d)
            end for sdh in cs.dh_rotational.subdofhandlers for cell in CellIterator(sdh)
            for (dof, x) in zip(celldofs(cell), getcoordinates(cell))
        ) < 1.0e-12
    end

    @testset "VTK output" begin
        mesh = generate_ideal_lv_mesh(8, 2, 3)
        cs = compute_lv_coordinate_system(mesh)
        mktempdir() do dir
            VTKGridFile(joinpath(dir, "continuous"), mesh.grid) do vtk
                vtk_coordinate_system(vtk, cs)
            end
            VTKGridFile(
                joinpath(dir, "discontinuous"),
                mesh.grid;
                write_discontinuous = true,
            ) do vtk
                vtk_coordinate_system(vtk, cs)
            end
            @test isfile(joinpath(dir, "continuous.vtu"))
            @test isfile(joinpath(dir, "discontinuous.vtu"))
        end
    end
end
