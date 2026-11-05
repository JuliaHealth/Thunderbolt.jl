using Test, Thunderbolt, Tensors
include(joinpath(@__DIR__, "testfixtures.jl"))
@testset "Mesh" begin
    # `uniform_refinement` splits hexahedra alone; the other cell types reach it through
    # `hexahedralize` first.
    num_refined_elements(::Type{Hexahedron}) = 8

    function test_detJ(grid)
        for cc ∈ CellIterator(grid)
            cell = getcells(grid, cellid(cc))
            ref_shape = Ferrite.getrefshape(cell)
            ip = getinterpolation(LagrangeCollection{1}(), ref_shape)
            qr = QuadratureRule{ref_shape}([1.0], [Vec(ntuple(_->0.1, Ferrite.getrefdim(cell)))]) # TODO randomize point
            gv = Ferrite.GeometryMapping{1}(Float64, ip, qr)
            x = getcoordinates(cc)
            mapping = Ferrite.calculate_mapping(gv, 1, x)
            J = Ferrite.getjacobian(mapping)
            @test Ferrite.calculate_detJ(J) > 0
        end
    end

    @testset "Cubioidal $element_type" for element_type ∈ [
        Hexahedron,
        Wedge,
        Tetrahedron,
        # Quadrilateral,
        # Triangle
    ]
        dim = Ferrite.getrefdim(element_type)
        grid = generate_grid(element_type, ntuple(_ -> 4, dim))
        addcellset!(grid, "right_cells", x -> x[1] ≥ 0.0)
        addcellset!(grid, "left_cells", x -> x[1] ≤ 0.0)

        if dim == 3
            grid_hex = Thunderbolt.hexahedralize(grid)
            @test all(typeof.(getcells(grid_hex)) .== Hexahedron) # Test if we really hit all elements
            test_detJ(grid_hex) # And for messed up elements

            # Check for correct transfer of facetsets
            addfacetset!(grid_hex, "right_new", x -> x[1] ≈ 1.0)
            @test getfacetset(grid_hex, "right") == getfacetset(grid_hex, "right_new")
            addfacetset!(grid_hex, "left_new", x -> x[1] ≈ -1.0)
            @test getfacetset(grid_hex, "left") == getfacetset(grid_hex, "left_new")
            addfacetset!(grid_hex, "top_new", x -> x[3] ≈ 1.0)
            @test getfacetset(grid_hex, "top") == getfacetset(grid_hex, "top_new")
            addfacetset!(grid_hex, "bottom_new", x -> x[3] ≈ -1.0)
            @test getfacetset(grid_hex, "bottom") == getfacetset(grid_hex, "bottom_new")
            addfacetset!(grid_hex, "front_new", x -> x[2] ≈ -1.0)
            @test getfacetset(grid_hex, "front") == getfacetset(grid_hex, "front_new")
            addfacetset!(grid_hex, "back_new", x -> x[2] ≈ 1.0)
            @test getfacetset(grid_hex, "back") == getfacetset(grid_hex, "back_new")

            # Check for correct transfer of cellsets
            addcellset!(grid_hex, "right_cells_new", x -> x[1] ≥ 0.0)
            @test getcellset(grid_hex, "right_cells") == getcellset(grid_hex, "right_cells_new")
            addcellset!(grid_hex, "left_cells_new", x -> x[1] ≤ 0.0)
            @test getcellset(grid_hex, "left_cells") == getcellset(grid_hex, "left_cells_new")
        end

        if element_type == Hexahedron
            grid_fine = Thunderbolt.uniform_refinement(grid)
            @test getncells(grid_fine) == num_refined_elements(element_type)*getncells(grid)
            @test all(typeof.(getcells(grid_fine)) .== element_type) # For the tested elements all fine elements are the same type
            test_detJ(grid_fine)
        end
    end

    @testset "Linear Hex Ring" begin
        ring_mesh = generate_ring_mesh(8, 3, 3)
        test_detJ(ring_mesh)
        open_ring_mesh = generate_open_ring_mesh(8, 3, 3, π/4)
        test_detJ(open_ring_mesh)
    end

    @testset "Quadratic Hex Ring" begin
        ring_mesh = generate_quadratic_ring_mesh(5, 3, 3)
        test_detJ(ring_mesh)
        open_ring_mesh = generate_quadratic_open_ring_mesh(8, 3, 3, π/4)
        test_detJ(open_ring_mesh)
    end

    @testset "Linear Mixed LV to Hex" begin
        lv_mesh = Thunderbolt.generate_ideal_lv_mesh(8, 4, 4)
        test_detJ(lv_mesh)
        lv_mesh_hex = Thunderbolt.hexahedralize(lv_mesh)
        test_detJ(lv_mesh_hex)
    end

    @testset "Linear Hex LV" begin
        lv_mesh = Thunderbolt.generate_ideal_lv_mesh_hex(8, 4, 4)
        test_detJ(lv_mesh)
        lv_mesh_hex = Thunderbolt.hexahedralize(lv_mesh)
        test_detJ(lv_mesh_hex)
    end

    # Every facet that belongs to exactly one cell, i.e. the boundary the mesh actually has, derived
    # from the connectivity rather than from what the generator declared.
    function boundary_skeleton(grid)
        owners = Dict{NTuple{4, Int}, Vector{FacetIndex}}()
        for cellid = 1:getncells(grid)
            for (local_facet, facet) in enumerate(Ferrite.facets(getcells(grid, cellid)))
                # Triangular and quadrilateral facets meet here, so the key is zero padded.
                sorted = sort!(collect(facet))
                key = ntuple(i -> i ≤ length(sorted) ? sorted[i] : 0, 4)
                push!(get!(owners, key, FacetIndex[]), FacetIndex(cellid, local_facet))
            end
        end
        return Set(only(v) for v in values(owners) if length(v) == 1)
    end

    # `test_detJ` samples one point per cell; a valvular plate is thin and tapered, so check the
    # whole rule.
    function min_detJdV(grid)
        smallest = Inf
        for cc in CellIterator(grid)
            cell = getcells(grid, cellid(cc))
            cv = CellValues(
                QuadratureRule{Ferrite.getrefshape(cell)}(2),
                Ferrite.geometric_interpolation(typeof(cell)),
            )
            reinit!(cv, cc)
            for qp = 1:getnquadpoints(cv)
                smallest = min(smallest, getdetJdV(cv, qp))
            end
        end
        return smallest
    end

    # `∫ π x(z)² dz` up the endocardial surface of revolution and closed by the basal plane, in the
    # polar angle, where `∫₀^θ sin³ = cos³(θ)/3 - cos(θ) + 2/3`.
    function analytic_lv_cavity(; inner_radius, apex_inner, apex_outer, longitudinal_upper)
        cap(θ) = cos(θ)^3/3 - cos(θ) + 2/3
        basal_angle = (1.0 + longitudinal_upper)*π/2
        return π * inner_radius^2 * (apex_inner*cap(π/2) + apex_outer*(cap(basal_angle) - cap(π/2)))
    end

    @testset "Ideal LV valvular plane" begin
        nc, nr, nl, np = 8, 2, 3, 3
        plain = generate_ideal_lv_mesh(nc, nr, nl)
        mesh = generate_ideal_lv_mesh(nc, nr, nl; with_valvular_plane = true)
        # The basal plane of the default geometry and the thickness of the plate centered on it.
        z_base          = 1.5*cos(1.2*π/2)
        orifice_radius  = 0.7*sin(1.2*π/2)
        plate_thickness = (1.0 - 0.7)/3

        @testset "the plate is additive" begin
            # The ventricle itself is untouched by the closure: same nodes, same cells, same sets,
            # in the same order -- `test/integration/test_fsi.jl` builds its coupled operator on this
            # mesh and pins the dof layout that this order decides.
            @test getnnodes(mesh) == getnnodes(plain) + 2*nc*(np-1) + 2
            @test getncells(mesh) == getncells(plain) + nc*np
            @test Set(getcellset(mesh, "myocardium")) == Set(1:getncells(plain))
            for i = 1:getnnodes(plain)
                @test Ferrite.get_node_coordinate(getnodes(mesh, i)) ==
                      Ferrite.get_node_coordinate(getnodes(plain, i))
            end
            for i = 1:getncells(plain)
                @test getcells(mesh, i).nodes == getcells(plain, i).nodes
            end
            for name in ("Endocardium", "Epicardium", "Base", "SRidgePost", "SRidgeAnt")
                @test getfacetset(mesh, name) == getfacetset(plain, name)
            end
            @test !Thunderbolt._has_facetset(plain, "LVChamberSurface")
        end

        @testset "boundary integrity" begin
            # With the orifice plated over, the boundary is the endocardium, the epicardium, the
            # basal annulus and the two plate faces -- and nothing else.
            parts = Tuple(
                getfacetset(mesh, name) for name in
                ("Endocardium", "Epicardium", "Base", "LVValvularPlane", "ValvularPlaneOuter")
            )
            for (i, a) in enumerate(parts), b in parts[(i+1):end]
                @test isempty(intersect(a, b))
            end
            @test Set(union(parts...)) == boundary_skeleton(mesh)
            @test getfacetset(mesh, "LVChamberSurface") ==
                  union(getfacetset(mesh, "Endocardium"), getfacetset(mesh, "LVValvularPlane"))
            plate = getcellset(mesh, "valvular-plane")
            for name in ("LVValvularPlane", "ValvularPlaneOuter")
                @test all(facet -> facet[1] ∈ plate, getfacetset(mesh, name))
            end
        end

        @testset "valvular plate" begin
            plate = getcellset(mesh, "valvular-plane")
            coordinate(n) = Ferrite.get_node_coordinate(getnodes(mesh, n))

            test_detJ(mesh)
            @test min_detJdV(mesh) > 0
            # The taper ends *on* the endocardial rim ring: every cell keeps all of its nodes
            # distinct, so no wedge collapsed onto the ring to get there.
            @test all(cellid -> allunique(getcells(mesh, cellid).nodes), plate)

            # It reaches the ring by reusing its nodes -- every endocardial one and nothing else --
            # rather than by placing a second ring on top of it.
            plate_nodes = Set(n for cellid in plate for n in getcells(mesh, cellid).nodes)
            shared = filter(plate_nodes) do n
                x = coordinate(n)
                x[3] ≈ z_base && x[1]^2 + x[2]^2 ≈ orifice_radius^2
            end
            @test length(shared) == nc
            @test nc == count(1:getnnodes(mesh)) do n
                x = coordinate(n)
                x[3] ≈ z_base && x[1]^2 + x[2]^2 ≈ orifice_radius^2
            end

            # The plate is a slab of the requested thickness, centered on the basal plane.
            zs = [coordinate(n)[3] for n in plate_nodes]
            @test minimum(zs) ≈ z_base - plate_thickness/2
            @test maximum(zs) ≈ z_base + plate_thickness/2
        end

        @testset "closed chamber surface" begin
            # The chamber surface is traversed from the cells bounding the chamber, so its normals
            # point into it and the enclosed volume comes out negative.
            translated = [
                Ferrite.get_node_coordinate(node) + Vec((3.0, -2.0, 5.0)) for node in getnodes(mesh)
            ]
            volumes = surface_volumes(mesh, getfacetset(mesh, "LVChamberSurface"))
            @test volumes[1] ≈ volumes[2] rtol = 1.0e-10
            @test volumes[1] ≈ volumes[3] rtol = 1.0e-10
            # A rigid translation moves no volume, which only a closed surface can honour.
            @test surface_volumes(mesh, getfacetset(mesh, "LVChamberSurface"), translated) ≈ volumes rtol =
                1.0e-8

            # Negative control: the endocardium alone is open at the orifice, so the three axes give
            # three different numbers and none of them is a volume.
            open_lv = surface_volumes(mesh, getfacetset(mesh, "Endocardium"))
            @test !isapprox(open_lv[1], open_lv[3]; rtol = 1.0e-6)
            @test !isapprox(open_lv[2], open_lv[3]; rtol = 1.0e-6)

            reference = analytic_lv_cavity(;
                inner_radius = 0.7,
                apex_inner = 1.3,
                apex_outer = 1.5,
                longitudinal_upper = 0.2,
            )
            fine_mesh = generate_ideal_lv_mesh(24, 2, 8; with_valvular_plane = true)
            coarse = -volumes[1]
            fine = -surface_volumes(fine_mesh, getfacetset(fine_mesh, "LVChamberSurface"))[1]
            # The facetted cavity is inscribed in the smooth one, and the ventricular half of the
            # plate displaces another ~2.5% of it, so both stay below and converge to just under it.
            @test coarse < fine < reference
            @test fine ≈ reference rtol = 0.05
        end

        @testset "parameter feasibility" begin
            @test_throws ErrorException generate_ideal_lv_mesh(
                4,
                1,
                1;
                with_valvular_plane = true,
                num_elements_valvular_plane = 1,
            )
            @test_throws ErrorException generate_ideal_lv_mesh(
                4,
                1,
                1;
                with_valvular_plane = true,
                valvular_plane_thickness = -0.1,
            )
        end
    end

    @testset "Ideal left heart" begin
        nc, nr, n_lv, n_la, np = 8, 2, 3, 2, 3
        mesh = generate_ideal_lh_mesh(nc, nr, n_lv, n_la)
        # A second resolution: the thin-walled defaults have to hold up on both.
        fine_mesh = generate_ideal_lh_mesh(24, 2, 8, 6)
        # The annulus plane of the default geometry, where the two shells meet, and the thickness of
        # the plate centered on it.
        z_rim = 1.5*cos(1.2*π/2)
        plate_thickness = (1.0 - 0.7)/3
        test_detJ(mesh)
        @test min_detJdV(fine_mesh) > 0

        @testset "layout" begin
            @test getnnodes(mesh) == nc*(nr+1)*(n_lv+1+n_la) + 2*(nr+1) + 2*nc*(np-1) + 2
            @test getncells(mesh) == nc*nr*(n_lv+n_la) + 2*nc*nr + nc*np
            @test length(getcellset(mesh, "ventricle")) == nc*nr*(n_lv+1)
            @test length(getcellset(mesh, "atrium")) == nc*nr*(n_la+1)
            @test length(getcellset(mesh, "valvular-plane")) == nc*np
            subdomains = ("ventricle", "atrium", "valvular-plane")
            @test sum(length(getcellset(mesh, name)) for name in subdomains) == getncells(mesh)
            @test length(union((getcellset(mesh, name) for name in subdomains)...)) ==
                  getncells(mesh)
        end

        @testset "boundary integrity" begin
            # With the orifice plated over, the boundary is the two endocardia, the epicardium and
            # the two plate faces -- and nothing else.
            parts = Tuple(
                getfacetset(mesh, name) for name in (
                    "LVEndocardium",
                    "LAEndocardium",
                    "Epicardium",
                    "LVValvularPlane",
                    "LAValvularPlane",
                )
            )

            @test !Thunderbolt._has_facetset(mesh, "Base")
            for (i, a) in enumerate(parts), b in parts[(i+1):end]
                @test isempty(intersect(a, b))
            end
            @test Set(union(parts...)) == boundary_skeleton(mesh)
            # The annulus is where the two shells meet, so its facets are interior.
            @test isempty(intersect(getfacetset(mesh, "MitralAnnulus"), boundary_skeleton(mesh)))
        end

        @testset "set consistency" begin
            @test getfacetset(mesh, "Endocardium") ==
                  union(getfacetset(mesh, "LVEndocardium"), getfacetset(mesh, "LAEndocardium"))
            @test getfacetset(mesh, "Epicardium") ==
                  union(getfacetset(mesh, "LVEpicardium"), getfacetset(mesh, "LAEpicardium"))
            # The endocardia stay anatomical; the chamber surfaces are the closed versions.
            @test getfacetset(mesh, "LVChamberSurface") ==
                  union(getfacetset(mesh, "LVEndocardium"), getfacetset(mesh, "LVValvularPlane"))
            @test getfacetset(mesh, "LAChamberSurface") ==
                  union(getfacetset(mesh, "LAEndocardium"), getfacetset(mesh, "LAValvularPlane"))
            # `compute_lv_coordinate_system(mesh; subdomains = ["ventricle"])` reads these, so they
            # may not reach into the atrium.
            ventricle = getcellset(mesh, "ventricle")
            for name in
                ("SRidgePost", "SRidgeAnt", "LVEndocardium", "LVEpicardium", "MitralAnnulus")
                @test all(facet -> facet[1] ∈ ventricle, getfacetset(mesh, name))
            end
            @test all(
                facet -> facet[1] ∈ getcellset(mesh, "atrium"),
                getfacetset(mesh, "LAEndocardium"),
            )
            plate = getcellset(mesh, "valvular-plane")
            for name in ("LVValvularPlane", "LAValvularPlane")
                @test all(facet -> facet[1] ∈ plate, getfacetset(mesh, name))
            end
        end

        @testset "valvular plate" begin
            plate = getcellset(mesh, "valvular-plane")
            rim = getnodeset(mesh, "MitralAnnulusRing")
            coordinate(n) = Ferrite.get_node_coordinate(getnodes(mesh, n))

            @test min_detJdV(mesh) > 0
            # The taper ends *on* the shared ring: every cell keeps all of its nodes distinct, so no
            # wedge collapsed onto the ring to get there.
            @test all(cellid -> allunique(getcells(mesh, cellid).nodes), plate)

            # It reaches the ring by reusing its nodes -- every endocardial one and nothing else --
            # rather than by placing a second ring on top of it.
            plate_nodes = Set(n for cellid in plate for n in getcells(mesh, cellid).nodes)
            shared = intersect(plate_nodes, rim)
            @test length(shared) == nc
            endocardial_rim_radius = 0.7*sin(1.2*π/2)
            @test all(shared) do n
                x = coordinate(n)
                x[1]^2 + x[2]^2 ≈ endocardial_rim_radius^2
            end
            @test nc == count(1:getnnodes(mesh)) do n
                x = coordinate(n)
                x[3] ≈ z_rim && x[1]^2 + x[2]^2 ≈ endocardial_rim_radius^2
            end

            # The plate is a slab of the requested thickness, centered on the annulus plane.
            zs = [coordinate(n)[3] for n in plate_nodes]
            @test minimum(zs) ≈ z_rim - plate_thickness/2
            @test maximum(zs) ≈ z_rim + plate_thickness/2
        end

        @testset "closed chamber surfaces" begin
            # The chamber surfaces are traversed from the cells bounding the chamber, so their
            # normals point into it and the enclosed volume comes out negative.
            translated = [
                Ferrite.get_node_coordinate(node) + Vec((3.0, -2.0, 5.0)) for node in getnodes(mesh)
            ]
            for name in ("LVChamberSurface", "LAChamberSurface")
                volumes = surface_volumes(mesh, getfacetset(mesh, name))
                @test volumes[1] ≈ volumes[2] rtol = 1.0e-10
                @test volumes[1] ≈ volumes[3] rtol = 1.0e-10
                # A rigid translation moves no volume, which only a closed surface can honour.
                @test surface_volumes(mesh, getfacetset(mesh, name), translated) ≈ volumes rtol =
                    1.0e-8
            end

            # Negative control: the endocardium alone is open at the orifice, so the three axes give
            # three different numbers and none of them is a volume.
            open_lv = surface_volumes(mesh, getfacetset(mesh, "LVEndocardium"))
            @test !isapprox(open_lv[1], open_lv[3]; rtol = 1.0e-6)
            @test !isapprox(open_lv[2], open_lv[3]; rtol = 1.0e-6)

            reference = analytic_lv_cavity(;
                inner_radius = 0.7,
                apex_inner = 1.3,
                apex_outer = 1.5,
                longitudinal_upper = 0.2,
            )
            coarse = -surface_volumes(mesh, getfacetset(mesh, "LVChamberSurface"))[1]
            fine = -surface_volumes(fine_mesh, getfacetset(fine_mesh, "LVChamberSurface"))[1]
            # The facetted cavity is inscribed in the smooth one, and the ventricular half of the
            # plate displaces another ~2.5% of it, so both stay below and converge to just under it.
            @test coarse < fine < reference
            @test fine ≈ reference rtol = 0.05
        end

        @testset "rim sharing" begin
            rim = getnodeset(mesh, "MitralAnnulusRing")
            @test length(rim) == nc*(nr+1)

            chambers_of = Dict(n => Set{String}() for n in rim)
            for name in ("ventricle", "atrium"), cellid in getcellset(mesh, name)
                for n in getcells(mesh, cellid).nodes
                    n ∈ rim && push!(chambers_of[n], name)
                end
            end
            @test all(==(Set(["ventricle", "atrium"])), values(chambers_of))

            # Both shells are truncated in the same plane, which is what lets them share the ring.
            @test all(n -> Ferrite.get_node_coordinate(getnodes(mesh, n))[3] ≈ z_rim, rim)
        end

        @testset "chamber separation helper" begin
            endocardium = getfacetset(mesh, "Endocardium")
            plane_point = Vec((0.0, 0.0, z_rim))
            above, below =
                separate_chamber_surfaces(mesh, endocardium, plane_point, Vec((0.0, 0.0, 1.0)))
            @test above == getfacetset(mesh, "LVEndocardium")
            @test below == getfacetset(mesh, "LAEndocardium")

            # Flipping the normal swaps the two returns, which is the documented convention.
            flipped_above, flipped_below =
                separate_chamber_surfaces(mesh, endocardium, plane_point, Vec((0.0, 0.0, -1.0)))
            @test flipped_above == below
            @test flipped_below == above
        end

        @testset "parameter feasibility" begin
            # The atrium has to be wide enough at the annulus to stand on the ventricular rim, on
            # its endocardial and on its epicardial layer.
            @test_throws ErrorException generate_ideal_lh_mesh(4, 1, 1, 1; la_cavity_radius = 0.1)
            @test_throws ErrorException generate_ideal_lh_mesh(
                4,
                1,
                1,
                1;
                la_cavity_radius = 0.7,
                la_wall_thickness = 0.01,
            )
            @test_throws ErrorException generate_ideal_lh_mesh(4, 1, 1, 1; la_wall_thickness = -0.1)
            @test_throws ErrorException generate_ideal_lh_mesh(
                4,
                1,
                1,
                1;
                num_elements_valvular_plane = 1,
            )
        end

        @testset "single chamber generator unchanged" begin
            # The refactor that gave both generators their shared builders may not move the
            # single-chamber mesh: `test/integration/test_fsi.jl` builds its coupled operator on it.
            lv = generate_ideal_lv_mesh(4, 2, 2)
            @test getnnodes(lv) == 4*3*3 + 3
            @test getncells(lv) == 4*2*2 + 4*2
            @test length(getfacetset(lv, "Endocardium")) == 4*2 + 4
            @test length(getfacetset(lv, "Epicardium")) == 4*2 + 4
            @test length(getfacetset(lv, "Base")) == 4*2
            @test length(getfacetset(lv, "SRidgePost")) == 2*2 + 2
            @test length(getfacetset(lv, "SRidgeAnt")) == 2*2 + 2

            θ_first, θ_base = 1.2*π/2/3, 1.2*π/2
            @test Ferrite.get_node_coordinate(getnodes(lv, 1)) ≈
                  Vec((0.7*sin(θ_first), 0.0, 1.3*cos(θ_first)))
            @test Ferrite.get_node_coordinate(getnodes(lv, 2*4*3 + 1)) ≈
                  Vec((0.7*sin(θ_base), 0.0, 1.5*cos(θ_base)))
            @test Ferrite.get_node_coordinate(getnodes(lv, getnnodes(lv))) ≈ Vec((0.0, 0.0, 1.5))
        end
    end

    @testset "IO" begin
        dirname = @__DIR__

        @testset "voom2 legacy" begin
            voom2_grid = load_voom2_grid(dirname * "/data/voom2/ex1")

            @test length(voom2_grid.nodes) == 9
            @test typeof(voom2_grid.cells[1]) == Line
            @test typeof(voom2_grid.cells[2]) == Hexahedron
            @test length(voom2_grid.cells) == 2
            test_detJ(voom2_grid)
        end

        @testset "mfem v1.0 $filename" for (filename, element_type) in [
            ("ref-segment.mesh", Line),
            ("ref-triangle.mesh", Triangle),
            ("ref-square.mesh", Quadrilateral),
            ("ref-tetrahedron.mesh", Tetrahedron),
            ("ref-cube.mesh", Hexahedron),
            ("ref-prism.mesh", Wedge),
            ("ref-pyramid.mesh", Pyramid),
        ]
            mfem_grid = load_mfem_grid(dirname * "/data/mfem/" * filename)

            @test all(typeof.(mfem_grid.cells) .== element_type)
            test_detJ(mfem_grid)
        end

        @testset "openCARP $filename" for (filename, element_type) in [
            ("ref-segment", Line),
            ("ref-triangle", Triangle),
            ("ref-square", Quadrilateral),
            ("ref-tetrahedron", Tetrahedron),
            ("ref-cube", Hexahedron),
            ("ref-prism", Wedge),
            # ("ref-pyramid", Pyramid),
        ]
            carp_grid = load_carp_grid(dirname * "/data/openCARP/" * filename)

            @test all(typeof.(carp_grid.cells) .== element_type)
            test_detJ(carp_grid)
        end
    end

    @testset "Surface extraction" begin
        LV_mesh = generate_ideal_lv_mesh(4, 2, 2)
        surface_mesh = Thunderbolt.extract_outer_surface_mesh(LV_mesh)

        @test length(surface_mesh.cellsets) == 3
        @test length(surface_mesh.cellsets["Epicardium"]) == 3*4
        @test length(surface_mesh.cellsets["Endocardium"]) == 3*4
        @test length(surface_mesh.cellsets["Base"]) == 4*2
        @test length(surface_mesh.nodes) == 2 + 3*2*4 + 4 # nodes at base + nodes on loopes endo&epi + node loop on base
        @test length(surface_mesh.cells) == 2*4 + 2*4*(2+1) # cells at base + cells inside&outside
    end

    @testset "Geometry Tools" begin
        ring_mesh = generate_ring_mesh(5, 4, 4)
        @test Thunderbolt.compute_center_of_mass(ring_mesh) ≈ Vec((0.0, 0.0, 0.0)) atol=1e-16
        @test Thunderbolt.compute_center_of_surface(ring_mesh, "Endocardium") ≈ Vec((0.0, 0.0, 0.0)) atol=1e-16
    end

    @testset "reorder_cells!" begin
        @testset "identity permutation leaves grid unchanged" begin
            grid = generate_grid(Quadrilateral, (3, 2))
            cells_before = deepcopy(getcells(grid))
            cellsets_before = deepcopy(Ferrite.getcellsets(grid))
            facetsets_before = deepcopy(Ferrite.getfacetsets(grid))

            reorder_cells!(grid, collect(1:getncells(grid)))

            @test getcells(grid) == cells_before
            @test Ferrite.getcellsets(grid) == cellsets_before
            @test Ferrite.getfacetsets(grid) == facetsets_before
        end

        @testset "cells are reordered according to permutation" begin
            grid = generate_grid(Line, (4,))
            # cells 1 and 2 are in the left half
            addcellset!(grid, "left", Set([1, 2]))
            original_cells = deepcopy(getcells(grid))
            perm = [4, 3, 2, 1]

            reorder_cells!(grid, perm)

            for (new_idx, old_idx) in enumerate(perm)
                @test getcells(grid, new_idx) == original_cells[old_idx]
            end

            @test getcellset(grid, "left") == OrderedSet([3, 4])
        end

        @testset "facetsets are updated by permutation" begin
            grid = generate_grid(Quadrilateral, (3, 1))
            # "left" facetset uses FacetIndex(1, 4) by default from generate_grid
            perm = [3, 2, 1]  # reverse
            # old cell 1 -> new position invperm([3,2,1])[1] = 3

            reorder_cells!(grid, perm)

            left_set = getfacetset(grid, "left")
            @test all(fi -> fi[1] == 3, left_set)

            right_set = getfacetset(grid, "right")
            @test all(fi -> fi[1] == 1, right_set)
        end

        @testset "vertexsets are updated by permutation" begin
            grid = generate_grid(Quadrilateral, (2, 2))
            addvertexset!(grid, "bottom-left-corner", x -> x[1] ≈ -1.0 && x[2] ≈ -1.0)
            # The bottom-left corner vertex belongs to cell 1, vertex 1
            @test !isempty(getvertexset(grid, "bottom-left-corner"))
            original_cell_ids = Set(vi[1] for vi in getvertexset(grid, "bottom-left-corner"))

            perm = [4, 3, 2, 1]  # reverse
            iperm = invperm(perm)

            reorder_cells!(grid, perm)

            new_cell_ids = Set(vi[1] for vi in getvertexset(grid, "bottom-left-corner"))
            @test new_cell_ids == Set(iperm[id] for id in original_cell_ids)
        end
    end

    @testset "compute_sfc_ordering" begin
        @testset "permutation is valid" begin
            for (CT, dim) in
                [(Line, 1), (Quadrilateral, 2), (Triangle, 2), (Hexahedron, 3), (Tetrahedron, 3)]
                nels = ntuple(_ -> 3, dim)
                grid = generate_grid(CT, nels)
                perm = compute_sfc_ordering(grid)
                @test isperm(perm)
                @test length(perm) == getncells(grid)
            end
        end

        @testset "2D grid: SFC ordering reduces average neighbor distance" begin
            # A shuffled 2D grid should have better locality after SFC ordering
            grid = generate_grid(Quadrilateral, (4, 4))
            Random.seed!(42)
            shuffled_perm = randperm(getncells(grid))
            reorder_cells!(grid, shuffled_perm)

            function mean_consecutive_center_dist(g)
                centers = [
                    sum(get_node_coordinate(g, n) for n in Ferrite.get_node_ids(getcells(g, i))) / length(Ferrite.get_node_ids(getcells(g, i))) for i = 1:getncells(g)
                ]
                dists = [norm(centers[i+1] - centers[i]) for i = 1:(length(centers)-1)]
                sum(dists) / length(dists)
            end

            dist_before = mean_consecutive_center_dist(grid)
            sfc_perm = compute_sfc_ordering(grid)
            reorder_cells!(grid, sfc_perm)
            dist_after = mean_consecutive_center_dist(grid)

            @test dist_after <= dist_before
        end
    end

end
