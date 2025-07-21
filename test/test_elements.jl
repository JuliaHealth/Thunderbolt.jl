@testset "Element API" begin
    import Thunderbolt: assemble_element!, assemble_facet!
    import Thunderbolt: setup_element_cache, setup_boundary_cache
    import Thunderbolt: BilinearMassIntegrator, BilinearDiffusionIntegrator
    import Thunderbolt: CompositeVolumetricElementCache, CompositeSurfaceElementCache

    grid = generate_grid(Hexahedron, (1,1,1))
    qrc  = QuadratureRuleCollection(3)
    qr   = QuadratureRule{RefHexahedron}(3)
    qrcf = QuadratureRuleCollection(3)
    qrf  = FacetQuadratureRule{RefHexahedron}(3)
    ip   = Lagrange{RefHexahedron,1}()

    dhs  = DofHandler(grid)
    add!(dhs, :u, ip)
    close!(dhs)
    sdhs = first(dhs.subdofhandlers)
    cell_cache_s = Ferrite.CellCache(sdhs)
    reinit!(cell_cache_s, 1)
    uₑs = [
        -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0
    ].*1e-4

    ipv = ip^3
    dhv  = DofHandler(grid)
    add!(dhv, :u, ipv)
    close!(dhv)
    sdhv = first(dhv.subdofhandlers)
    cell_cache_v = Ferrite.CellCache(sdhv)
    reinit!(cell_cache_v, 1)
    uₑv = [
        -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0,
        -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0,
        -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0,
    ].*1e-4

    # We check for pairwise consistency of the assembly operations
    # First we check if the empty caches work correctly
    @testset "Empty caches" begin
        rₑ¹ = zeros(ndofs(dhs))
        rₑ² = zeros(ndofs(dhs))
        Kₑ¹ = zeros(ndofs(dhs), ndofs(dhs))
        Kₑ² = zeros(ndofs(dhs), ndofs(dhs))

        # Volume
        assemble_element!(Kₑ¹, rₑ¹, uₑs, cell_cache_s, Thunderbolt.EmptyVolumetricElementCache(), 0.0)
        @test iszero(Kₑ¹)
        @test iszero(rₑ¹)

        assemble_element!(     rₑ², uₑs, cell_cache_s, Thunderbolt.EmptyVolumetricElementCache(), 0.0)
        @test iszero(rₑ²)

        assemble_element!(Kₑ²,      uₑs, cell_cache_s, Thunderbolt.EmptyVolumetricElementCache(), 0.0)
        @test iszero(Kₑ²)

        # Surface
        for local_facet_index in 1:nfacets(cell_cache_s)
            assemble_facet!(Kₑ¹, rₑ¹, uₑs, cell_cache_s, local_facet_index, Thunderbolt.EmptySurfaceElementCache(), 0.0)
            @test iszero(Kₑ¹)
            @test iszero(rₑ¹)

            assemble_facet!(     rₑ², uₑs, cell_cache_s, local_facet_index, Thunderbolt.EmptySurfaceElementCache(), 0.0)
            @test iszero(rₑ²)

            assemble_facet!(Kₑ²,      uₑs, cell_cache_s, local_facet_index, Thunderbolt.EmptySurfaceElementCache(), 0.0)
            @test iszero(Kₑ²)
        end
    end

    # No we check some examples for the implemented physics
    @testset "Scalar volumetric bilinear elements: $model" for model in (
        BilinearMassIntegrator(
            ConstantCoefficient(1.0),
            qrc,
            :u,
        ),
        BilinearDiffusionIntegrator(
            ConstantCoefficient(one(Tensor{2,3})),
            qrc,
            :u,
        )
    )
        Kₑ¹ = zeros(ndofs(dhs), ndofs(dhs))
        Kₑ² = zeros(ndofs(dhs), ndofs(dhs))

        element_cache = setup_element_cache(model, sdhs)

        assemble_element!(Kₑ¹, cell_cache_s, element_cache, 0.0)
        @test !iszero(Kₑ¹)

        composite_element_cache = CompositeVolumetricElementCache((
            element_cache,
            element_cache,
        ))

        assemble_element!(Kₑ², cell_cache_s, composite_element_cache, 0.0)
        @test 2Kₑ¹ ≈ Kₑ²
    end

    @testset "Vectorial volumetric nonlinear elements: $model" for model in (
        PK1Model(
            HolzapfelOgden2009Model(),
            ConstantCoefficient(
                OrthotropicMicrostructure(
                    Vec((1.0, 0.0, 0.0)),
                    Vec((0.0, 1.0, 0.0)),
                    Vec((0.0, 0.0, 1.0)),
                )
            )
        ),
    )
        rₑ¹ = zeros(ndofs(dhv))
        rₑ² = zeros(ndofs(dhv))
        Kₑ¹ = zeros(ndofs(dhv), ndofs(dhv))
        Kₑ² = zeros(ndofs(dhv), ndofs(dhv))

        element_cache = setup_element_cache(QuasiStaticModel(:u, model, ()), qr, sdhv)

        @test_opt assemble_element!(Kₑ¹, rₑ¹, uₑv, cell_cache_v, element_cache, 0.0)
                  assemble_element!(Kₑ¹, rₑ¹, uₑv, cell_cache_v, element_cache, 0.0)
        @test !iszero(Kₑ¹)
        @test !iszero(rₑ¹)

        @test_opt assemble_element!(     rₑ², uₑv, cell_cache_v, element_cache, 0.0)
                  assemble_element!(     rₑ², uₑv, cell_cache_v, element_cache, 0.0)
        @test rₑ² ≈ rₑ¹

        @test_opt assemble_element!(Kₑ²,      uₑv, cell_cache_v, element_cache, 0.0)
                  assemble_element!(Kₑ²,      uₑv, cell_cache_v, element_cache, 0.0)
        @test Kₑ² ≈ Kₑ¹

        composite_element_cache = CompositeVolumetricElementCache((
            element_cache,
            element_cache,
        ))

        Kₑ¹ .= 0.0
        rₑ¹ .= 0.0
        @test_opt assemble_element!(Kₑ¹, rₑ¹, uₑv, cell_cache_v, composite_element_cache, 0.0)
                  assemble_element!(Kₑ¹, rₑ¹, uₑv, cell_cache_v, composite_element_cache, 0.0)
        @test 2Kₑ² ≈ Kₑ¹
        @test 2rₑ² ≈ rₑ¹

        rₑ² .= 0.0
        @test_opt assemble_element!(     rₑ², uₑv, cell_cache_v, composite_element_cache, 0.0)
                  assemble_element!(     rₑ², uₑv, cell_cache_v, composite_element_cache, 0.0)
        @test rₑ² ≈ rₑ¹

        Kₑ² .= 0.0
        @test_opt assemble_element!(Kₑ²,      uₑv, cell_cache_v, composite_element_cache, 0.0)
                  assemble_element!(Kₑ²,      uₑv, cell_cache_v, composite_element_cache, 0.0)
        @test Kₑ² ≈ Kₑ¹
    end

    # No we check some examples for the implemented physics
    @testset "Vectorial surface elements: $model" for (model, has_jac) in (
        (RobinBC(1.0, "left"), true),
        (NormalSpringBC(1.0, "left"), true),
        (BendingSpringBC(1.0, "left"), true),
        (ConstantPressureBC(1.0, "left"), true),
        (PressureFieldBC(ConstantCoefficient(1.0), "left"), true),
    )
        rₑ¹ = zeros(ndofs(dhv))
        rₑ² = zeros(ndofs(dhv))
        Kₑ¹ = zeros(ndofs(dhv), ndofs(dhv))
        Kₑ² = zeros(ndofs(dhv), ndofs(dhv))

        element_cache = setup_boundary_cache(model, qrf, sdhv)

        for local_facet_index in 1:nfacets(cell_cache_v)
            assemble_facet!(Kₑ¹, rₑ¹, uₑv, cell_cache_v, local_facet_index, element_cache, 0.0)
            @test iszero(Kₑ¹) != has_jac
            @test iszero(rₑ¹) != has_jac

            assemble_facet!(     rₑ², uₑv, cell_cache_v, local_facet_index, element_cache, 0.0)
            @test rₑ² ≈ rₑ¹

            assemble_facet!(Kₑ²,      uₑv, cell_cache_v, local_facet_index, element_cache, 0.0)
            @test Kₑ² ≈ Kₑ¹
        end

        composite_element_cache = CompositeSurfaceElementCache((
            element_cache,
            element_cache,
        ))

        Kₑ¹ .= 0.0
        rₑ¹ .= 0.0
        for local_facet_index in 1:nfacets(cell_cache_v)
            assemble_facet!(Kₑ¹, rₑ¹, uₑv, cell_cache_v, local_facet_index, composite_element_cache, 0.0)
        end
        @test 2Kₑ² ≈ Kₑ¹
        @test 2rₑ² ≈ rₑ¹

        Kₑ² .= 0.0
        rₑ² .= 0.0
        for local_facet_index in 1:nfacets(cell_cache_v)
            assemble_facet!(     rₑ², uₑv, cell_cache_v, local_facet_index, composite_element_cache, 0.0)
            assemble_facet!(Kₑ²,      uₑv, cell_cache_v, local_facet_index, composite_element_cache, 0.0)
        end
        @test Kₑ² ≈ Kₑ¹
        @test rₑ² ≈ rₑ¹
    end
end
