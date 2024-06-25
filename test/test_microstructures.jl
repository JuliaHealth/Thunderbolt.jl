@testset "microstructures" begin
    ring_grid = generate_ring_mesh(80,1,1)

    qr_collection = QuadratureRuleCollection(2)
    ip_collection = LagrangeCollection{1}()^3

    ring_cs = compute_midmyocardial_section_coordinate_system(ring_grid)

    cartesian_coefficient = CoordinateSystemCoefficient(CartesianCoordinateSystem(ring_grid))
    qr = getquadraturerule(qr_collection, getcells(ring_grid, 1))

    dh = DofHandler(ring_grid)
    add!(dh, :u, Lagrange{RefHexahedron,1}())
    close!(dh)
    sdh = first(dh.subdofhandlers)

    cs_cache = Thunderbolt.setup_coefficient_cache(cartesian_coefficient, qr, sdh)

    @testset "Midmyocardial coordinate system" begin
        csc = CoordinateSystemCoefficient(ring_cs)
        cache2 = Thunderbolt.setup_coefficient_cache(csc, qr, sdh)
        for cellcache in CellIterator(ring_cs.dh)
            for qp in QuadratureIterator(qr)
                x = evaluate_coefficient(cs_cache, cellcache, qp, 0.0)
                transmural = 4*(norm(Vec(x[1:2]..., 0.))-0.75)
                coord = evaluate_coefficient(cache2, cellcache, qp, 0.0)
                @test transmural ≈ coord.transmural atol=0.01
            end
        end
    end

    @testset "OrthotropicMicrostructureModel" begin
        ms = create_simple_microstructure_model(ring_cs, ip_collection,
            endo_helix_angle = deg2rad(0.0),
            epi_helix_angle = deg2rad(0.0),
            endo_transversal_angle = 0.0,
            epi_transversal_angle = 0.0,
            sheetlet_pseudo_angle = deg2rad(0)
        )
        cache2 = Thunderbolt.setup_coefficient_cache(ms, qr, sdh)
        for cellcache in CellIterator(ring_grid)
            for qp in QuadratureIterator(qr)
                x = evaluate_coefficient(cs_cache, cellcache, qp, 0.0)
                # If we set all angles to 0, then the generator on the ring simply generates sheetlets which point in positive z direction, where the normal point radially outwards.
                ndir = Vec(x[1:2]..., 0.)/norm(x[1:2])
                sdir = Vec(0., 0., 1.)
                fdir = sdir × ndir

                fsn = evaluate_coefficient(cache2, cellcache, qp, 0.0)
                @test fsn.f ≈ fdir atol=0.05
                @test fsn.s ≈ sdir
                @test fsn.n ≈ ndir atol=0.05
            end
        end
    end
end
