@testset "Transfer Opeartors" begin
    source_mesh = Thunderbolt.generate_simple_disc_mesh(Quadrilateral, 40)

    target_mesh = generate_mesh(Triangle, (10, 11))
    target_mesh_nonmatching = generate_mesh(Triangle, (40, 44), Vec((-2.0,-2.0)), Vec((2.0,2.0)))
    cells_hole = Set{Int}()
    cells_remaining = Set{Int}()
    for cc in CellIterator(target_mesh.grid)
        if all(norm.(getcoordinates(cc)) .≤ 1)
            push!(cells_hole, cellid(cc))
        else
            push!(cells_remaining, cellid(cc))
        end
    end

    @testset "Nodal Intergrid Interpolation on Matching Grids" begin
        source_dh = DofHandler(source_mesh)
        add!(source_dh, :z, Lagrange{RefQuadrilateral,1}())
        add!(source_dh, :u, Lagrange{RefQuadrilateral,2}())
        add!(source_dh, :v, Lagrange{RefQuadrilateral,3}())
        close!(source_dh)

        source_u = ones(ndofs(source_dh))
        apply_analytical!(source_u, source_dh, :v, x->-norm(x))
        apply_analytical!(source_u, source_dh, :z, x-> norm(x))

        target_dh = DofHandler(target_mesh)
        target_sdh_hole = SubDofHandler(target_dh, cells_hole)
        add!(target_sdh_hole, :v, Lagrange{RefTriangle,2}())
        add!(target_sdh_hole, :w, Lagrange{RefTriangle,1}())
        close!(target_dh)

        v_range = dof_range(target_dh.subdofhandlers[1], :v)
        w_range = dof_range(target_dh.subdofhandlers[1], :w)
    
        op = Thunderbolt.NodalIntergridInterpolation(source_dh, target_dh, :v)
        target_u = [NaN for i in 1:ndofs(target_dh)]
        Thunderbolt.transfer!(target_u, op, source_u)
        cvv = CellValues(QuadratureRule{RefTriangle}(1), Lagrange{RefTriangle,2}())
        for cc in CellIterator(target_dh.subdofhandlers[1])
            reinit!(cvv, cc)
            dofs_v = @view celldofs(cc)[v_range]
            dofs_w = @view celldofs(cc)[w_range]
            for qp in QuadratureIterator(cvv)
                x = Thunderbolt.spatial_coordinate(Lagrange{RefTriangle,1}(), qp.ξ, getcoordinates(cc))
                @test function_value(cvv, qp, target_u[dofs_v]) ≈ -norm(x) atol=3e-1
                @test all(isnan.(target_u[dofs_w]))
            end
        end

        op = Thunderbolt.NodalIntergridInterpolation(source_dh, target_dh, :z, :w)
        target_u = [NaN for i in 1:ndofs(target_dh)]
        Thunderbolt.transfer!(target_u, op, source_u)
        cvw = CellValues(QuadratureRule{RefTriangle}(1), Lagrange{RefTriangle,1}())
        for cc in CellIterator(target_dh.subdofhandlers[1])
            reinit!(cvw, cc)
            dofs_v = @view celldofs(cc)[v_range]
            dofs_w = @view celldofs(cc)[w_range]
            for qp in QuadratureIterator(cvw)
                x = Thunderbolt.spatial_coordinate(Lagrange{RefTriangle,1}(), qp.ξ, getcoordinates(cc))
                @test all(isnan.(target_u[dofs_v]))
                @test function_value(cvw, qp, target_u[dofs_w]) ≈ norm(x) atol=3e-1
            end
        end
    end

    target_grid_nonmatching = generate_grid(Triangle, (40, 44), Vec((-2.0,-2.0)), Vec((2.0,2.0)))
    addcellset!(target_grid_nonmatching, "hole", x->norm(x) ≤ 1.0)
    addcellset!(target_grid_nonmatching, "remaining", x->norm(x) ≥ 1.0)
    target_mesh_nonmatching = to_mesh(target_grid_nonmatching)

    @testset "Nodal Intergrid Interpolation on Non-Matching Grids" begin
        source_dh = DofHandler(source_mesh)
        add!(source_dh, :z, Lagrange{RefQuadrilateral,1}())
        add!(source_dh, :u, Lagrange{RefQuadrilateral,2}())
        add!(source_dh, :v, Lagrange{RefQuadrilateral,3}())
        close!(source_dh)

        source_u = ones(ndofs(source_dh))
        apply_analytical!(source_u, source_dh, :v, x->-norm(x))
        apply_analytical!(source_u, source_dh, :z, x-> norm(x))

        target_dh = DofHandler(target_mesh_nonmatching)
        target_sdh_hole = SubDofHandler(target_dh, getcellset(target_mesh_nonmatching, "hole"))
        add!(target_sdh_hole, :v, Lagrange{RefTriangle,2}())
        add!(target_sdh_hole, :w, Lagrange{RefTriangle,1}())
        target_sdh_remaining = SubDofHandler(target_dh, getcellset(target_mesh_nonmatching, "remaining"))
        add!(target_sdh_remaining, :v, Lagrange{RefTriangle,2}())
        add!(target_sdh_remaining, :w, Lagrange{RefTriangle,1}())
        close!(target_dh)

        v_range = dof_range(target_dh.subdofhandlers[1], :v)
        w_range = dof_range(target_dh.subdofhandlers[1], :w)

        target_sdhids = Thunderbolt.get_subdofhandler_indices_on_subdomains(target_dh, ["hole"])

        op = Thunderbolt.NodalIntergridInterpolation(source_dh, target_dh, :v, :v; subdomains_to=target_sdhids)
        target_u = [NaN for i in 1:ndofs(target_dh)]
        Thunderbolt.transfer!(target_u, op, source_u)
        cvv = CellValues(QuadratureRule{RefTriangle}(1), Lagrange{RefTriangle,2}())
        for cc in CellIterator(target_dh.subdofhandlers[1])
            reinit!(cvv, cc)
            dofs_v = @view celldofs(cc)[v_range]
            dofs_w = @view celldofs(cc)[w_range]
            for qp in QuadratureIterator(cvv)
                x = Thunderbolt.spatial_coordinate(Lagrange{RefTriangle,1}(), qp.ξ, getcoordinates(cc))
                @test function_value(cvv, qp, target_u[dofs_v]) ≈ -norm(x) atol=3e-1
                @test all(isnan.(target_u[dofs_w]))
            end
        end

        op = Thunderbolt.NodalIntergridInterpolation(source_dh, target_dh, :z, :w; subdomains_to=target_sdhids)
        target_u = [NaN for i in 1:ndofs(target_dh)]
        Thunderbolt.transfer!(target_u, op, source_u)
        cvw = CellValues(QuadratureRule{RefTriangle}(1), Lagrange{RefTriangle,1}())
        for cc in CellIterator(target_dh.subdofhandlers[1])
            reinit!(cvw, cc)
            dofs_v = @view celldofs(cc)[v_range]
            dofs_w = @view celldofs(cc)[w_range]
            for qp in QuadratureIterator(cvw)
                x = Thunderbolt.spatial_coordinate(Lagrange{RefTriangle,1}(), qp.ξ, getcoordinates(cc))
                @test all(isnan.(target_u[dofs_v]))
                @test function_value(cvw, qp, target_u[dofs_w]) ≈ norm(x) atol=3e-1
            end
        end
    end
end
