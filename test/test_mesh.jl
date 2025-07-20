@testset "Mesh" begin
    num_refined_elements(::Type{Hexahedron}) = 8
    num_refined_elements(::Type{Tetrahedron}) = 8
    num_refined_elements(::Type{Triangle}) = 4
    num_refined_elements(::Type{Quadrilateral}) = 4

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
        ring_mesh = generate_ring_mesh(8,3,3)
        test_detJ(ring_mesh)
        open_ring_mesh = generate_open_ring_mesh(8,3,3,π/4)
        test_detJ(open_ring_mesh)
    end

    @testset "Quadratic Hex Ring" begin
        ring_mesh = generate_quadratic_ring_mesh(5,3,3)
        test_detJ(ring_mesh)
        open_ring_mesh = generate_quadratic_open_ring_mesh(8,3,3,π/4)
        test_detJ(open_ring_mesh)
    end

    @testset "Linear Hex LV" begin
        lv_mesh = Thunderbolt.generate_ideal_lv_mesh(8,4,4)
        test_detJ(lv_mesh)
        lv_mesh_hex = Thunderbolt.hexahedralize(lv_mesh)
        test_detJ(lv_mesh_hex)
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
        LV_mesh = generate_ideal_lv_mesh(4,2,2)
        surface_mesh = Thunderbolt.extract_outer_surface_mesh(LV_mesh)

        @test length(surface_mesh.cellsets) == 3
        @test length(surface_mesh.cellsets["Epicardium"]) == 3*4
        @test length(surface_mesh.cellsets["Endocardium"]) == 3*4
        @test length(surface_mesh.cellsets["Base"]) == 4*2
        @test length(surface_mesh.nodes) == 2 + 3*2*4 + 4 # nodes at base + nodes on loopes endo&epi + node loop on base
        @test length(surface_mesh.cells) == 2*4 + 2*4*(2+1) # cells at base + cells inside&outside
    end
end
