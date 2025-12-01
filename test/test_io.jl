using Thunderbolt, SHA

@testset "IO" begin
    @testset "ParaViewWriter" begin
        grid = generate_grid(Hexahedron, (2,2,2))
        gip =  Ferrite.geometric_interpolation(typeof(grid.cells[1]))
        dh = DofHandler(grid)
        add!(dh, :apicobasal, gip)
        add!(dh, :transmural, gip)
        add!(dh, :rotational, gip)
        add!(dh, :transventricular, gip)
        close!(dh)

        coordinate_data = zeros(ndofs(dh))
        apply_analytical!(coordinate_data, dh, :apicobasal, x->0.5x[1]+0.5)
        apply_analytical!(coordinate_data, dh, :transmural, x->1.5-0.5x[2])
        apply_analytical!(coordinate_data, dh, :rotational, x->0.5(x[3]^3+1))
        apply_analytical!(coordinate_data, dh, :transventricular, x->0.5(x[3]+1)^2)

        pvd = ParaViewWriter(joinpath("testdata","cobivec"))
        @t
        store_timestep!(pvd, 0.0, grid) do vtk
            store_timestep_field!(vtk, 0.0, dh, coordinate_data, :apicobasal)
            store_timestep_field!(vtk, 0.0, dh, coordinate_data, :transmural)
            store_timestep_field!(vtk, 0.0, dh, coordinate_data, :rotational)
            store_timestep_field!(vtk, 0.0, dh, coordinate_data, :transventricular)
        end

        @test bytes2hex(open(SHA.sha1, joinpath("testdata","cobivec.pvd")))       == "957f820ffedfd8e3486643dc76c0231d4a836d61"
        @test bytes2hex(open(SHA.sha1, joinpath("testdata","cobivec","0.0.vtu"))) == "a4f496ac89b21eef7a20ecdeef55b64c5441249f"
    end

    @testset "CoBiVeC" begin
        grid = generate_grid(Hexahedron, (2,2,2))
        gip =  Ferrite.geometric_interpolation(typeof(grid.cells[1]))
        dh = DofHandler(grid)
        add!(dh, :apicobasal, gip)
        add!(dh, :transmural, gip)
        add!(dh, :rotational, gip)
        add!(dh, :transventricular, gip)
        close!(dh)

        coordinate_data = zeros(ndofs(dh))
        apply_analytical!(coordinate_data, dh, :apicobasal, x->0.5x[1]+0.5)
        apply_analytical!(coordinate_data, dh, :transmural, x->1.5-0.5x[2])
        apply_analytical!(coordinate_data, dh, :rotational, x->0.5(x[3]^3+1))
        apply_analytical!(coordinate_data, dh, :transventricular, x->0.5(x[3]+1)^2)

        filename = "cobivec.vtu"
        VTKGridFile(joinpath("testdata",filename), grid) do vtk
            Ferrite.write_solution(vtk, dh, coordinate_data)
        end

        cobivec = Thunderbolt.read_vtk_cobivec(joinpath("testdata",filename), "transmural", "apicobasal", "rotational", "transventricular")
        @test cobivec.dh.grid.cells == grid.cells
        @test cobivec.dh.grid.nodes == grid.nodes
        for (cell1,cell2) in zip(CellIterator(dh), CellIterator(cobivec.dh))
            all_dofs = celldofs(cell1)
            @test cobivec.u_apicobasal[celldofs(cell2)] == coordinate_data[all_dofs[dof_range(dh, :apicobasal)]]
            @test cobivec.u_transmural[celldofs(cell2)] == coordinate_data[all_dofs[dof_range(dh, :transmural)]]
            @test cobivec.u_rotational[celldofs(cell2)] == coordinate_data[all_dofs[dof_range(dh, :rotational)]]
            @test cobivec.u_transventricular[celldofs(cell2)] == coordinate_data[all_dofs[dof_range(dh, :transventricular)]]
        end
    end
end
