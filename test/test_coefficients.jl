using Test, Thunderbolt, Tensors, StaticArrays
using JET: @test_opt, @test_call
@testset "Coefficient API" begin
    device = PolyesterDevice()

    grid = generate_grid(Line, (2,))
    cell_cache = Ferrite.CellCache(grid)
    qp1 = QuadraturePoint(1, Vec((0.0,)))
    qp2 = QuadraturePoint(2, Vec((0.1,)))
    qr = QuadratureRule{RefLine}([1.0, 1.0], [Vec{1}((0.0,)), Vec{1}((0.1,))])
    ip_collection = LagrangeCollection{1}()
    dh = DofHandler(grid)
    add!(dh, :u, getinterpolation(ip_collection, first(grid.cells)))
    close!(dh)
    sdh = first(dh.subdofhandlers)

    function setup_test_cache(coefficient)
        return Thunderbolt.duplicate_for_device(
            device,
            Thunderbolt.setup_coefficient_cache(coefficient, qr, sdh),
        )
    end

    @testset "ConstantCoefficient($val" for val ∈ [1.0, one(Tensor{2, 2})]
        cc = ConstantCoefficient(val)
        coeff_cache = setup_test_cache(cc)
        Ferrite.reinit!(cell_cache, 1)
        @test_opt evaluate_coefficient(coeff_cache, cell_cache, qp1, 0.0)
        @test evaluate_coefficient(coeff_cache, cell_cache, qp1, 0.0) ≈ val
        @test evaluate_coefficient(coeff_cache, cell_cache, qp2, 1.0) ≈ val
        Ferrite.reinit!(cell_cache, 2)
        @test evaluate_coefficient(coeff_cache, cell_cache, qp1, 0.0) ≈ val
        @test evaluate_coefficient(coeff_cache, cell_cache, qp2, 1.0) ≈ val

        @test_opt evaluate_coefficient(coeff_cache, cell_cache, qp1, 0.0)
    end

    @testset "FieldCoefficient" begin
        data_scalar = zeros(2, 2)
        data_scalar[1, 1] = 1.0
        data_scalar[1, 2] = -1.0
        data_scalar[2, 1] = -1.0
        fcs = FieldCoefficient(data_scalar, ip_collection)
        coeff_cache = setup_test_cache(fcs)
        Ferrite.reinit!(cell_cache, 1)
        @test evaluate_coefficient(coeff_cache, cell_cache, qp1, 0.0) ≈ 0.0
        @test evaluate_coefficient(coeff_cache, cell_cache, qp1, 1.0) ≈ 0.0
        @test evaluate_coefficient(coeff_cache, cell_cache, qp2, 0.0) ≈ -0.1
        @test evaluate_coefficient(coeff_cache, cell_cache, qp2, 1.0) ≈ -0.1
        Ferrite.reinit!(cell_cache, 2)
        @test evaluate_coefficient(coeff_cache, cell_cache, qp1, 0.0) ≈ -0.5
        @test evaluate_coefficient(coeff_cache, cell_cache, qp1, 1.0) ≈ -0.5
        @test evaluate_coefficient(coeff_cache, cell_cache, qp2, 0.0) ≈ (0.1+1.0)/2.0-1.0
        @test evaluate_coefficient(coeff_cache, cell_cache, qp2, 1.0) ≈ (0.1+1.0)/2.0-1.0

        data_vector = zeros(Vec{2, Float64}, 2, 2)
        data_vector[1, 1] = Vec((1.0, 0.0))
        data_vector[1, 2] = Vec((0.0, -1.0))
        data_vector[2, 1] = Vec((-1.0, -0.0))
        fcv = FieldCoefficient(data_vector, ip_collection^2)
        coeff_cache = setup_test_cache(fcv)
        Ferrite.reinit!(cell_cache, 1)
        @test evaluate_coefficient(coeff_cache, cell_cache, qp1, 0.0) ≈ Vec((0.0, 0.0))
        @test evaluate_coefficient(coeff_cache, cell_cache, qp2, 1.0) ≈ Vec((-0.1, 0.0))
        Ferrite.reinit!(cell_cache, 2)
        @test evaluate_coefficient(coeff_cache, cell_cache, qp1, 0.0) ≈ Vec((0.0, -0.5))
        @test evaluate_coefficient(coeff_cache, cell_cache, qp2, 1.0) ≈
              Vec((0.0, (0.1+1.0)/2.0-1.0))

        @test_opt evaluate_coefficient(coeff_cache, cell_cache, qp1, 0.0)
    end

    @testset "Cartesian Coordinate System" begin
        ccsc = CartesianCoordinateSystem(grid)
        coeff_cache = setup_test_cache(ccsc)
        Ferrite.reinit!(cell_cache, 1)
        @test evaluate_coefficient(coeff_cache, cell_cache, qp1, 0.0) ≈ Vec((-0.5,))
        @test evaluate_coefficient(coeff_cache, cell_cache, qp1, 1.0) ≈ Vec((-0.5,))
        @test evaluate_coefficient(coeff_cache, cell_cache, qp2, 0.0) ≈ Vec((-0.45,))
        @test evaluate_coefficient(coeff_cache, cell_cache, qp2, 1.0) ≈ Vec((-0.45,))
        Ferrite.reinit!(cell_cache, 2)
        @test evaluate_coefficient(coeff_cache, cell_cache, qp1, 0.0) ≈ Vec((0.5,))
        @test evaluate_coefficient(coeff_cache, cell_cache, qp1, 1.0) ≈ Vec((0.5,))
        @test evaluate_coefficient(coeff_cache, cell_cache, qp2, 0.0) ≈ Vec((0.55,))
        @test evaluate_coefficient(coeff_cache, cell_cache, qp2, 1.0) ≈ Vec((0.55,))

        @test_opt evaluate_coefficient(coeff_cache, cell_cache, qp1, 0.0)
    end

    @testset "AnalyticalCoefficient" begin
        ac = AnalyticalCoefficient((x, t) -> norm(x)+t, CartesianCoordinateSystem(grid))
        coeff_cache = setup_test_cache(ac)
        Ferrite.reinit!(cell_cache, 1)
        @test evaluate_coefficient(coeff_cache, cell_cache, qp1, 0.0) ≈ 0.5
        @test evaluate_coefficient(coeff_cache, cell_cache, qp2, 0.0) ≈ 0.45
        @test evaluate_coefficient(coeff_cache, cell_cache, qp1, 1.0) ≈ 1.5
        @test evaluate_coefficient(coeff_cache, cell_cache, qp2, 1.0) ≈ 1.45
        Ferrite.reinit!(cell_cache, 2)
        @test evaluate_coefficient(coeff_cache, cell_cache, qp1, 0.0) ≈ 0.5
        @test evaluate_coefficient(coeff_cache, cell_cache, qp2, 0.0) ≈ 0.55
        @test evaluate_coefficient(coeff_cache, cell_cache, qp1, 1.0) ≈ 1.5
        @test evaluate_coefficient(coeff_cache, cell_cache, qp2, 1.0) ≈ 1.55

        @test_opt evaluate_coefficient(coeff_cache, cell_cache, qp1, 0.0)
    end

    @testset "SpectralTensorCoefficient" begin
        eigvec = Vec((1.0, 0.0))
        eigval = -1.0
        stc = SpectralTensorCoefficient(
            ConstantCoefficient(TransverselyIsotropicMicrostructure(eigvec)),
            ConstantCoefficient(SVector((eigval, 0.0))),
        )
        st = Tensor{2, 2}((-1.0, 0.0, 0.0, 0.0))
        coeff_cache = setup_test_cache(stc)
        for i = 1:2
            Ferrite.reinit!(cell_cache, i)
            @test_opt evaluate_coefficient(coeff_cache, cell_cache, qp1, 0.0)
            @test evaluate_coefficient(coeff_cache, cell_cache, qp1, 0.0) ≈ st
            @test evaluate_coefficient(coeff_cache, cell_cache, qp2, 0.0) ≈ st
            @test evaluate_coefficient(coeff_cache, cell_cache, qp1, 1.0) ≈ st
            @test evaluate_coefficient(coeff_cache, cell_cache, qp2, 1.0) ≈ st
        end

        st2 = Tensor{2, 2}((-1.0, 0.0, 0.0, -1.0))
        stc2 = SpectralTensorCoefficient(
            ConstantCoefficient(TransverselyIsotropicMicrostructure(eigvec)),
            ConstantCoefficient(SVector((eigval, eigval))),
        )
        coeff_cache = setup_test_cache(stc2)
        @test evaluate_coefficient(coeff_cache, cell_cache, qp1, 0.0) ≈ st2

        stc3 = SpectralTensorCoefficient(
            ConstantCoefficient(AnisotropicPlanarMicrostructure(Vec((1.0, 0.0)), Vec((0.0, 1.0)))),
            ConstantCoefficient(SVector((eigval, eigval))),
        )
        coeff_cache = setup_test_cache(stc3)
        @test evaluate_coefficient(coeff_cache, cell_cache, qp1, 0.0) ≈ st2

        @test_opt evaluate_coefficient(coeff_cache, cell_cache, qp1, 0.0)
    end

    @testset "SpatiallyHomogeneousDataField" begin
        shdc = SpatiallyHomogeneousDataField([1.0, 2.0], [Vec((0.1,)), Vec((0.2,)), Vec((0.3,))])
        coeff_cache = setup_test_cache(shdc)
        for i = 1:2
            Ferrite.reinit!(cell_cache, i)
            @test_opt evaluate_coefficient(coeff_cache, cell_cache, qp1, 0.0)
            @test evaluate_coefficient(coeff_cache, cell_cache, qp1, 0.0) ≈ Vec((0.1,))
            @test evaluate_coefficient(coeff_cache, cell_cache, qp2, 0.0) ≈ Vec((0.1,))
            @test evaluate_coefficient(coeff_cache, cell_cache, qp1, 1.0) ≈ Vec((0.1,))
            @test evaluate_coefficient(coeff_cache, cell_cache, qp2, 1.0) ≈ Vec((0.1,))
            @test evaluate_coefficient(coeff_cache, cell_cache, qp1, 1.1) ≈ Vec((0.2,))
            @test evaluate_coefficient(coeff_cache, cell_cache, qp2, 1.1) ≈ Vec((0.2,))
            @test evaluate_coefficient(coeff_cache, cell_cache, qp1, 2.0) ≈ Vec((0.2,))
            @test evaluate_coefficient(coeff_cache, cell_cache, qp2, 2.0) ≈ Vec((0.2,))
            @test evaluate_coefficient(coeff_cache, cell_cache, qp1, 2.1) ≈ Vec((0.3,))
            @test evaluate_coefficient(coeff_cache, cell_cache, qp2, 2.1) ≈ Vec((0.3,))

            @test_opt evaluate_coefficient(coeff_cache, cell_cache, qp1, 0.0)
        end
    end

    @testset "ConductivityToDiffusivityCoefficient" begin
        eigvec = Vec((1.0, 0.0))
        eigval = -1.0
        stc = SpectralTensorCoefficient(
            ConstantCoefficient(TransverselyIsotropicMicrostructure(eigvec)),
            ConstantCoefficient(SVector((eigval, 0.0))),
        )
        st = Tensor{2, 2}((-1.0, 0.0, 0.0, 0.0))
        ctdc = Thunderbolt.ConductivityToDiffusivityCoefficient(
            stc,
            ConstantCoefficient(2.0),
            ConstantCoefficient(0.5),
        )
        coeff_cache = setup_test_cache(ctdc)
        for i = 1:2
            Ferrite.reinit!(cell_cache, i)
            @test_opt evaluate_coefficient(coeff_cache, cell_cache, qp1, 0.0)
            @test evaluate_coefficient(coeff_cache, cell_cache, qp1, 0.0) ≈ st
            @test evaluate_coefficient(coeff_cache, cell_cache, qp2, 0.0) ≈ st
            @test evaluate_coefficient(coeff_cache, cell_cache, qp1, 1.0) ≈ st
            @test evaluate_coefficient(coeff_cache, cell_cache, qp2, 1.0) ≈ st

            @test_opt evaluate_coefficient(coeff_cache, cell_cache, qp1, 0.0)
        end
    end

    @testset "Static interpolation values" begin
        # The static values reimplement the cell mapping so that it can run without a `reinit!` and
        # on a device. Nothing in this suite used to call them, which is how they came to reference
        # Ferrite internals that no longer exist -- so pin them against `CellValues` itself, on
        # deliberately distorted cells of every element type in play.
        cells = (
            Lagrange{RefHexahedron, 1}() => [
                Vec((0.0, 0.0, 0.0)), Vec((1.3, 0.1, 0.0)),
                Vec((1.1, 1.4, -0.2)), Vec((0.2, 1.0, 0.1)),
                Vec((-0.1, 0.2, 1.2)), Vec((1.5, 0.0, 1.0)),
                Vec((1.2, 1.1, 1.4)), Vec((0.0, 1.3, 1.1)),
            ],
            Lagrange{RefTetrahedron, 1}() => [
                Vec((0.0, 0.0, 0.0)), Vec((1.7, 0.2, 0.1)),
                Vec((0.3, 1.4, -0.1)), Vec((0.1, 0.2, 1.9)),
            ],
            Lagrange{RefPrism, 1}() => [
                Vec((0.0, 0.0, 0.0)), Vec((1.2, 0.1, 0.0)), Vec((0.1, 1.3, 0.2)),
                Vec((0.0, 0.1, 1.4)), Vec((1.1, 0.0, 1.2)), Vec((0.2, 1.2, 1.5)),
            ],
        )
        for (ip, coords) in cells
            qrc = QuadratureRule{Ferrite.getrefshape(ip)}(2)
            cv = CellValues(qrc, ip, ip^3)
            reinit!(cv, coords)
            fv = Thunderbolt.FerriteUtils.StaticInterpolationValues(cv.fun_values)
            gm = Thunderbolt.FerriteUtils.StaticInterpolationValues(cv.geo_mapping)
            for q = 1:getnquadpoints(cv)
                mapping = Ferrite.calculate_mapping(gm, q, coords)
                N, dNdx = Thunderbolt.FerriteUtils.calculate_mapped_values(fv, q, mapping)
                detJ = Ferrite.calculate_detJ(Ferrite.getjacobian(mapping))
                @test detJ * Ferrite.getweights(qrc)[q] ≈ getdetJdV(cv, q)
                for i = 1:getnbasefunctions(cv)
                    @test N[i] ≈ shape_value(cv, q, i)
                    @test dNdx[i] ≈ shape_gradient(cv, q, i)
                end
            end
        end
    end

    @testset "StaticCellValues" begin
        # `StaticCellValues` is what the GPU path adapts a `CellValues` into, so it has to answer
        # every query the same way -- but it stores no cell geometry, so it has to be handed the
        # coordinates instead of being `reinit!`ed.
        ip = Lagrange{RefHexahedron, 1}()
        coords = [
            Vec((0.0, 0.0, 0.0)), Vec((1.3, 0.1, 0.0)),
            Vec((1.1, 1.4, -0.2)), Vec((0.2, 1.0, 0.1)),
            Vec((-0.1, 0.2, 1.2)), Vec((1.5, 0.0, 1.0)),
            Vec((1.2, 1.1, 1.4)), Vec((0.0, 1.3, 1.1)),
        ]
        qrc = QuadratureRule{RefHexahedron}(2)
        cv = CellValues(qrc, ip, ip^3)
        reinit!(cv, coords)
        scv = Thunderbolt.FerriteUtils.StaticCellValues(cv)

        @test getnquadpoints(scv) == getnquadpoints(cv)
        @test getnbasefunctions(scv) == getnbasefunctions(cv)
        @test Ferrite.getngeobasefunctions(scv) == Ferrite.getngeobasefunctions(cv)
        # Storing no geometry is the point of the type, so reinit! is a no-op rather than an error.
        @test Ferrite.reinit!(scv, coords) === nothing

        ue = [0.3, -1.2, 0.7, 2.1, 0.4, -0.6, 1.5, 0.9]
        for q = 1:getnquadpoints(cv)
            qv = Thunderbolt.FerriteUtils.quadrature_point_values(scv, q, coords)
            @test getdetJdV(qv) ≈ getdetJdV(cv, q)
            @test getnbasefunctions(qv) == getnbasefunctions(cv)
            for i = 1:getnbasefunctions(cv)
                @test shape_value(qv, i) ≈ shape_value(cv, q, i)
                @test shape_gradient(qv, i) ≈ shape_gradient(cv, q, i)
            end
            # The AbstractQuadratureValues interface on top of those accessors.
            @test function_value(qv, ue) ≈ function_value(cv, q, ue)
            @test function_gradient(qv, ue) ≈ function_gradient(cv, q, ue)
            @test spatial_coordinate(qv, coords) ≈ spatial_coordinate(cv, q, coords)
        end
    end

    @testset "QuadratureValuesIterator" begin
        ip = Lagrange{RefTetrahedron, 1}()
        coords = [
            Vec((0.0, 0.0, 0.0)), Vec((1.7, 0.2, 0.1)),
            Vec((0.3, 1.4, -0.1)), Vec((0.1, 0.2, 1.9)),
        ]
        qrc = QuadratureRule{RefTetrahedron}(2)
        cv = CellValues(qrc, ip, ip^3)
        reinit!(cv, coords)
        ue = [0.4, 1.1, -0.8, 0.25]

        # Over plain CellValues the iterator hands out views of the reinit!ed values ...
        n = 0
        for (q, qv) in enumerate(Thunderbolt.FerriteUtils.QuadratureValuesIterator(cv))
            @test getdetJdV(qv) ≈ getdetJdV(cv, q)
            @test function_gradient(qv, ue) ≈ function_gradient(cv, q, ue)
            n += 1
        end
        @test n == getnquadpoints(cv)
        @test length(Thunderbolt.FerriteUtils.QuadratureValuesIterator(cv)) == n
        @test keys(Thunderbolt.FerriteUtils.QuadratureValuesIterator(cv)) == 1:n

        # ... and over StaticCellValues it computes them per point from the coordinates it carries.
        scv = Thunderbolt.FerriteUtils.StaticCellValues(cv)
        m = 0
        for (q, qv) in
            enumerate(Thunderbolt.FerriteUtils.QuadratureValuesIterator(scv, coords))
            @test getdetJdV(qv) ≈ getdetJdV(cv, q)
            @test function_value(qv, ue) ≈ function_value(cv, q, ue)
            @test function_gradient(qv, ue) ≈ function_gradient(cv, q, ue)
            @test spatial_coordinate(qv, coords) ≈ spatial_coordinate(cv, q, coords)
            m += 1
        end
        @test m == getnquadpoints(cv)
    end
end
