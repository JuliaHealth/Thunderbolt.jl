@testset "Transfer Opeartors" begin
    function test_transfer(source_mesh, target_mesh, transfer_operator)
        @testset "Matching Grids" begin
            source_dh = DofHandler(source_mesh)
            add!(source_dh, :z, Lagrange{RefQuadrilateral, 1}())
            add!(source_dh, :u, Lagrange{RefQuadrilateral, 2}())
            add!(source_dh, :v, Lagrange{RefQuadrilateral, 3}())
            close!(source_dh)

            source_u = ones(ndofs(source_dh))
            apply_analytical!(source_u, source_dh, :v, x->-norm(x))
            apply_analytical!(source_u, source_dh, :z, x -> norm(x))

            target_dh = DofHandler(target_mesh)
            target_sdh_hole = SubDofHandler(target_dh, cells_hole)
            add!(target_sdh_hole, :v, Lagrange{RefTriangle, 2}())
            add!(target_sdh_hole, :w, Lagrange{RefTriangle, 1}())
            close!(target_dh)

            v_range = dof_range(target_dh.subdofhandlers[1], :v)
            w_range = dof_range(target_dh.subdofhandlers[1], :w)

            op = transfer_operator(source_dh, target_dh, :v)

            target_u = [NaN for i = 1:ndofs(target_dh)]
            Thunderbolt.transfer!(target_u, op, source_u)

            cvv = CellValues(QuadratureRule{RefTriangle}(1), Lagrange{RefTriangle, 2}())
            for cc in CellIterator(target_dh.subdofhandlers[1])
                Ferrite.reinit!(cvv, cc)
                dofs_v = @view celldofs(cc)[v_range]
                dofs_w = @view celldofs(cc)[w_range]
                for qp in QuadratureIterator(cvv)
                    x = Thunderbolt.spatial_coordinate(
                        Lagrange{RefTriangle, 1}(),
                        qp.ξ,
                        getcoordinates(cc),
                    )
                    @test function_value(cvv, qp, target_u[dofs_v]) ≈ -norm(x) atol=3e-1
                end
                @test all(isnan.(target_u[dofs_w]))
            end

            op = transfer_operator(source_dh, target_dh, :z, :w)
            target_u = [NaN for i = 1:ndofs(target_dh)]
            Thunderbolt.transfer!(target_u, op, source_u)
            cvw = CellValues(QuadratureRule{RefTriangle}(1), Lagrange{RefTriangle, 1}())
            for cc in CellIterator(target_dh.subdofhandlers[1])
                Ferrite.reinit!(cvw, cc)
                dofs_v = @view celldofs(cc)[v_range]
                dofs_w = @view celldofs(cc)[w_range]
                for qp in QuadratureIterator(cvw)
                    x = Thunderbolt.spatial_coordinate(
                        Lagrange{RefTriangle, 1}(),
                        qp.ξ,
                        getcoordinates(cc),
                    )
                    @test function_value(cvw, qp, target_u[dofs_w]) ≈ norm(x) atol=3e-1
                end
                @test all(isnan.(target_u[dofs_v]))
            end
        end

        target_grid_nonmatching =
            generate_grid(Triangle, (40, 44), Vec((-2.0, -2.0)), Vec((2.0, 2.0)))
        addcellset!(target_grid_nonmatching, "hole", x->norm(x) ≤ 1.0)
        addcellset!(target_grid_nonmatching, "remaining", x->norm(x) ≥ 1.0)
        target_mesh_nonmatching = to_mesh(target_grid_nonmatching)

        @testset "Non-Matching Grids" begin
            source_dh = DofHandler(source_mesh)
            add!(source_dh, :z, Lagrange{RefQuadrilateral, 1}())
            add!(source_dh, :u, Lagrange{RefQuadrilateral, 2}())
            add!(source_dh, :v, Lagrange{RefQuadrilateral, 3}())
            close!(source_dh)

            source_u = ones(ndofs(source_dh))
            apply_analytical!(source_u, source_dh, :v, x->-norm(x))
            apply_analytical!(source_u, source_dh, :z, x -> norm(x))

            target_dh = DofHandler(target_mesh_nonmatching)
            target_sdh_hole = SubDofHandler(target_dh, getcellset(target_mesh_nonmatching, "hole"))
            add!(target_sdh_hole, :v, Lagrange{RefTriangle, 2}())
            add!(target_sdh_hole, :w, Lagrange{RefTriangle, 1}())
            target_sdh_remaining =
                SubDofHandler(target_dh, getcellset(target_mesh_nonmatching, "remaining"))
            add!(target_sdh_remaining, :v, Lagrange{RefTriangle, 2}())
            add!(target_sdh_remaining, :w, Lagrange{RefTriangle, 1}())
            close!(target_dh)

            v_range = dof_range(target_dh.subdofhandlers[1], :v)
            w_range = dof_range(target_dh.subdofhandlers[1], :w)

            target_sdhids = Thunderbolt.get_subdofhandler_indices_on_subdomains(target_dh, ["hole"])

            op = transfer_operator(source_dh, target_dh, :v, :v; subdomains_to = target_sdhids)

            # Note that here we assume that all sdh have the same ndofspercell.
            vdofs_from = sort(
                unique(
                    reduce(
                        vcat,
                        [
                            celldofs(source_dh, i)[dof_range(source_dh.subdofhandlers[1], :v)]
                            for i in source_dh.subdofhandlers[1].cellset
                        ],
                    ),
                ),
            )
            vdofs_to = sort(
                unique(
                    reduce(
                        vcat,
                        [
                            celldofs(target_dh, i)[v_range] for
                            i in target_dh.subdofhandlers[target_sdhids[]].cellset
                        ],
                    ),
                ),
            )

            # Since it is a single subdomain new "nodes" ordering does not match that of dofs
            # TODO: find a better test, these are constructed almost the same way as the function being tested
            @test op.mapping.node_to_dof_map_from == vdofs_from
            @test op.mapping.node_to_dof_map_to == vdofs_to

            target_u = [NaN for i = 1:ndofs(target_dh)]
            Thunderbolt.transfer!(target_u, op, source_u)
            cvv = CellValues(QuadratureRule{RefTriangle}(1), Lagrange{RefTriangle, 2}())
            for cc in CellIterator(target_dh.subdofhandlers[1])
                Ferrite.reinit!(cvv, cc)
                dofs_v = @view celldofs(cc)[v_range]
                dofs_w = @view celldofs(cc)[w_range]
                for qp in QuadratureIterator(cvv)
                    x = Thunderbolt.spatial_coordinate(
                        Lagrange{RefTriangle, 1}(),
                        qp.ξ,
                        getcoordinates(cc),
                    )
                    @test function_value(cvv, qp, target_u[dofs_v]) ≈ -norm(x) atol=3e-1
                end
                @test all(isnan.(target_u[dofs_w]))
            end

            op = transfer_operator(source_dh, target_dh, :z, :w; subdomains_to = target_sdhids)
            target_u = [NaN for i = 1:ndofs(target_dh)]
            Thunderbolt.transfer!(target_u, op, source_u)
            cvw = CellValues(QuadratureRule{RefTriangle}(1), Lagrange{RefTriangle, 1}())
            for cc in CellIterator(target_dh.subdofhandlers[1])
                Ferrite.reinit!(cvw, cc)
                dofs_v = @view celldofs(cc)[v_range]
                dofs_w = @view celldofs(cc)[w_range]
                for qp in QuadratureIterator(cvw)
                    x = Thunderbolt.spatial_coordinate(
                        Lagrange{RefTriangle, 1}(),
                        qp.ξ,
                        getcoordinates(cc),
                    )
                    @test function_value(cvw, qp, target_u[dofs_w]) ≈ norm(x) atol=3e-1
                end
                @test all(isnan.(target_u[dofs_v]))
            end
        end

        @testset "Convenience Constructor" begin
            source_dh = DofHandler(source_mesh)
            add!(source_dh, :v, Lagrange{RefQuadrilateral, 1}())
            close!(source_dh)

            source_u = ones(ndofs(source_dh))
            apply_analytical!(source_u, source_dh, :v, x -> norm(x))

            target_dh = DofHandler(target_mesh_nonmatching)
            add!(target_dh, :v, Lagrange{RefTriangle, 2}())
            close!(target_dh)

            v_range = dof_range(target_dh.subdofhandlers[1], :v)

            op = transfer_operator(source_dh, target_dh)
            target_u = [NaN for i = 1:ndofs(target_dh)]
            Thunderbolt.transfer!(target_u, op, source_u)
            cvv = CellValues(QuadratureRule{RefTriangle}(1), Lagrange{RefTriangle, 2}())
            for cc in CellIterator(target_dh.subdofhandlers[1])
                Ferrite.reinit!(cvv, cc)
                dofs_v = @view celldofs(cc)[v_range]
                for qp in QuadratureIterator(cvv)
                    x = Thunderbolt.spatial_coordinate(
                        Lagrange{RefTriangle, 1}(),
                        qp.ξ,
                        getcoordinates(cc),
                    )
                    any(norm.(getcoordinates(cc)) .> 1.0) && continue
                    @test function_value(cvv, qp, target_u[dofs_v]) ≈ norm(x) atol=3e-1
                end
            end
        end
    end
    function test_pe_true_subdomain()
        # tests that PointEvalHandler does not search the full grid but only
        # cells where the field is defined
        source_grid = generate_grid(Quadrilateral, (40, 40), Vec((-2.0, -2.0)), Vec((2.0, 2.0)))
        addcellset!(source_grid, "hole", x -> norm(x) ≤ 1.0)

        subdomain = getcellset(source_grid, "hole")
        source_dh = DofHandler(source_grid)
        sdh = SubDofHandler(source_dh, subdomain)
        add!(sdh, :v, Lagrange{RefQuadrilateral, 1}())
        close!(source_dh)

        target_mesh = generate_grid(Triangle, (4, 4), Vec((-2.0, -2.0)), Vec((2.0, 2.0)))
        addcellset!(target_mesh, "hole", x -> norm(x) ≤ 1.0)

        subdomain = getcellset(target_mesh, "hole")
        target_dh = DofHandler(target_mesh)
        sdh = SubDofHandler(target_dh, subdomain)
        add!(sdh, :v, Lagrange{RefTriangle, 1}())
        close!(target_dh)

        op = NodalIntergridInterpolation(source_dh, target_dh)

        # Since it is a single subdomain new "nodes" ordering matches that of dofs
        @test op.mapping.node_to_dof_map_from == 1:ndofs(source_dh)
        @test op.mapping.node_to_dof_map_to == 1:ndofs(target_dh)

        target_u = zeros(ndofs(target_dh))
        Thunderbolt.transfer!(target_u, op, zeros(ndofs(source_dh)))

        @test_broken !any(isnan.(target_u))
        return
    end
    source_mesh = Thunderbolt.generate_simple_disc_mesh(Quadrilateral, 40)

    target_mesh = generate_mesh(Triangle, (10, 11))
    target_mesh_nonmatching = generate_mesh(Triangle, (40, 44), Vec((-2.0, -2.0)), Vec((2.0, 2.0)))
    cells_hole = Set{Int}()
    cells_remaining = Set{Int}()
    for cc in CellIterator(target_mesh.grid)
        if all(norm.(getcoordinates(cc)) .≤ 1)
            push!(cells_hole, cellid(cc))
        else
            push!(cells_remaining, cellid(cc))
        end
    end

    rbf_test_cases = reduce(
        vcat,
        [
            [
                (
                    "RL-RBF α = $α, M = $M, k = $k",
                    (dh_from, dh_to, args...; kwargs...) ->
                        RescaledRadialBasisFunctionTransferOperator(
                            k,
                            M,
                            α,
                            dh_from,
                            dh_to,
                            args...;
                            kwargs...,
                        ),
                ),
                (
                    "L-RBF α = $α, M = $M, k = $k",
                    (dh_from, dh_to, args...; kwargs...) ->
                        RadialBasisFunctionTransferOperator(
                            k,
                            M,
                            α,
                            dh_from,
                            dh_to,
                            args...;
                            kwargs...,
                        ),
                ),
                (
                    "RL-RBF-G α = $α, M = $M, k = $k",
                    (dh_from, dh_to, args...; kwargs...) ->
                        RescaledRadialBasisFunctionGeodesicTransferOperator(
                            k,
                            M,
                            α,
                            dh_from,
                            dh_to,
                            args...;
                            β = 0.5,
                            kwargs...,
                        ),
                ),
                (
                    "L-RBF-G α = $α, M = $M, k = $k",
                    (dh_from, dh_to, args...; kwargs...) ->
                        RadialBasisFunctionGeodesicTransferOperator(
                            k,
                            M,
                            α,
                            dh_from,
                            dh_to,
                            args...;
                            β = 0.5,
                            kwargs...,
                        ),
                ),
            ] for α ∈ 1.5:1.5:3.0, M ∈ 1:2, k ∈ 0:2 # Due to how the circle connectivity the tests for lower alphas fail
        ],
    )
    @testset "Transfer Operator: $name" for (name, transfer_operator) in (
        ("NodalIntergridInterpolation", NodalIntergridInterpolation),
        rbf_test_cases...,
    )
        test_transfer(source_mesh, target_mesh, transfer_operator)
    end

    @testset "Ferrite.jl#1182" begin
        test_pe_true_subdomain()
    end

end
