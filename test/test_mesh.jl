@testset "Mesh" begin
    num_refined_elements(::Type{Hexahedron}) = 8
    num_refined_elements(::Type{Tetrahedron}) = 8
    num_refined_elements(::Type{Triangle}) = 4
    num_refined_elements(::Type{Quadrilateral}) = 4

    function test_detJ(grid)
        dim = Ferrite.getdim(grid)

        for cell ∈ CellIterator(grid)
            ref_shape = Ferrite.getrefshape(getcells(grid, cellid(cell)))
            ip = Lagrange{ref_shape, 1}()
            qr = QuadratureRule{ref_shape, Float64}([1.0], [Vec(ntuple(_->0.1, dim))]) # TODO randomize point
            gv = Ferrite.GeometryMapping{1}(Float64, ip, qr)
            x = getcoordinates(cell)
            mapping = Ferrite.calculate_mapping(gv, 1, x)
            J = Ferrite.getjacobian(mapping)
            @test Ferrite.calculate_detJ(J) > 0
        end
    end

    @testset "Cubioidal $element_type" for element_type ∈ [
        Hexahedron,
        Wedge,
        # Tetrahedron,
        # Quadrilateral,
        # Triangle
    ]
        dim = Ferrite.getdim(element_type)
        grid = generate_grid(element_type, ntuple(_ -> 3, dim))
        if dim == 3
            grid_hex = Thunderbolt.hexahedralize(grid)
            @test all(typeof.(getcells(grid_hex)) .== Hexahedron) # Test if we really hit all elements
            test_detJ(grid_hex) # And for messed up elements
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
    end

    @testset "Quadratic Hex Ring" begin
        ring_mesh = generate_quadratic_ring_mesh(5,3,3)
        test_detJ(ring_mesh)
    end

    @testset "Linear Hex LV" begin
        lv_mesh = Thunderbolt.generate_ideal_lv_mesh(8,4,4)
        test_detJ(lv_mesh)
        lv_mesh_hex = Thunderbolt.hexahedralize(lv_mesh)
        test_detJ(lv_mesh_hex)
    end
end
