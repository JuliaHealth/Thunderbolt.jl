@testset "Coefficient API" begin
    grid = generate_grid(Line, (2,))
    cell_cache = Ferrite.CellCache(grid)
    qp1 = QuadraturePoint(1, Vec((0.0,)))
    qp2 = QuadraturePoint(2, Vec((0.1,)))
    qr  = QuadratureRule{RefLine}([1.0, 1.0], [Vec{1}((0.0,)), Vec{1}((0.1,))])
    ip_collection = LagrangeCollection{1}()
    dh = DofHandler(grid)
    add!(dh, :u, getinterpolation(ip_collection, first(grid.cells)))
    close!(dh)
    sdh = first(dh.subdofhandlers)

    @testset "ConstantCoefficient($val" for val ∈ [1.0, one(Tensor{2,2})]
        cc = ConstantCoefficient(val)
        coeff_cache = Thunderbolt.setup_coefficient_cache(cc, qr, sdh)
        reinit!(cell_cache, 1)
        @test_opt evaluate_coefficient(coeff_cache, cell_cache, qp1, 0.0)
        @test evaluate_coefficient(coeff_cache, cell_cache, qp1, 0.0) ≈ val
        @test evaluate_coefficient(coeff_cache, cell_cache, qp2, 1.0) ≈ val
        reinit!(cell_cache, 2)
        @test evaluate_coefficient(coeff_cache, cell_cache, qp1, 0.0) ≈ val
        @test evaluate_coefficient(coeff_cache, cell_cache, qp2, 1.0) ≈ val
    end

    @testset "FieldCoefficient" begin
        data_scalar = zeros(2,2)
        data_scalar[1,1] =  1.0
        data_scalar[1,2] = -1.0
        data_scalar[2,1] = -1.0
        fcs = FieldCoefficient(data_scalar, ip_collection)
        coeff_cache = Thunderbolt.setup_coefficient_cache(fcs, qr, sdh)
        reinit!(cell_cache, 1)
        @test evaluate_coefficient(coeff_cache, cell_cache, qp1, 0.0) ≈  0.0
        @test evaluate_coefficient(coeff_cache, cell_cache, qp1, 1.0) ≈  0.0
        @test evaluate_coefficient(coeff_cache, cell_cache, qp2, 0.0) ≈ -0.1
        @test evaluate_coefficient(coeff_cache, cell_cache, qp2, 1.0) ≈ -0.1
        reinit!(cell_cache, 2)
        @test evaluate_coefficient(coeff_cache, cell_cache, qp1, 0.0) ≈ -0.5
        @test evaluate_coefficient(coeff_cache, cell_cache, qp1, 1.0) ≈ -0.5
        @test evaluate_coefficient(coeff_cache, cell_cache, qp2, 0.0) ≈ (0.1+1.0)/2.0-1.0
        @test evaluate_coefficient(coeff_cache, cell_cache, qp2, 1.0) ≈ (0.1+1.0)/2.0-1.0

        data_vector = zeros(Vec{2,Float64},2,2)
        data_vector[1,1] = Vec((1.0,0.0))
        data_vector[1,2] = Vec((0.0,-1.0))
        data_vector[2,1] = Vec((-1.0,-0.0))
        fcv = FieldCoefficient(data_vector, ip_collection^2)
        coeff_cache = Thunderbolt.setup_coefficient_cache(fcv, qr, sdh)
        reinit!(cell_cache, 1)
        @test evaluate_coefficient(coeff_cache, cell_cache, qp1, 0.0) ≈ Vec((0.0,0.0))
        @test evaluate_coefficient(coeff_cache, cell_cache, qp2, 1.0) ≈ Vec((-0.1,0.0))
        reinit!(cell_cache, 2)
        @test evaluate_coefficient(coeff_cache, cell_cache, qp1, 0.0) ≈ Vec((0.0,-0.5))
        @test evaluate_coefficient(coeff_cache, cell_cache, qp2, 1.0) ≈ Vec((0.0,(0.1+1.0)/2.0-1.0))
    end

    @testset "Cartesian Coordinate System" begin
        ccsc = CartesianCoordinateSystem(grid)
        coeff_cache = Thunderbolt.setup_coefficient_cache(ccsc, qr, sdh)
        reinit!(cell_cache, 1)
        @test evaluate_coefficient(coeff_cache, cell_cache, qp1, 0.0) ≈ Vec((-0.5,))
        @test evaluate_coefficient(coeff_cache, cell_cache, qp1, 1.0) ≈ Vec((-0.5,))
        @test evaluate_coefficient(coeff_cache, cell_cache, qp2, 0.0) ≈ Vec((-0.45,))
        @test evaluate_coefficient(coeff_cache, cell_cache, qp2, 1.0) ≈ Vec((-0.45,))
        reinit!(cell_cache, 2)
        @test evaluate_coefficient(coeff_cache, cell_cache, qp1, 0.0) ≈ Vec((0.5,))
        @test evaluate_coefficient(coeff_cache, cell_cache, qp1, 1.0) ≈ Vec((0.5,))
        @test evaluate_coefficient(coeff_cache, cell_cache, qp2, 0.0) ≈ Vec((0.55,))
        @test evaluate_coefficient(coeff_cache, cell_cache, qp2, 1.0) ≈ Vec((0.55,))
    end

    @testset "AnalyticalCoefficient" begin
        ac = AnalyticalCoefficient(
            (x,t) -> norm(x)+t,
            CartesianCoordinateSystem(grid),
        )
        coeff_cache = Thunderbolt.setup_coefficient_cache(ac, qr, sdh)
        reinit!(cell_cache, 1)
        @test evaluate_coefficient(coeff_cache, cell_cache, qp1, 0.0) ≈  0.5
        @test evaluate_coefficient(coeff_cache, cell_cache, qp2, 0.0) ≈  0.45
        @test evaluate_coefficient(coeff_cache, cell_cache, qp1, 1.0) ≈  1.5
        @test evaluate_coefficient(coeff_cache, cell_cache, qp2, 1.0) ≈  1.45
        reinit!(cell_cache, 2)
        @test evaluate_coefficient(coeff_cache, cell_cache, qp1, 0.0) ≈  0.5
        @test evaluate_coefficient(coeff_cache, cell_cache, qp2, 0.0) ≈  0.55
        @test evaluate_coefficient(coeff_cache, cell_cache, qp1, 1.0) ≈  1.5
        @test evaluate_coefficient(coeff_cache, cell_cache, qp2, 1.0) ≈  1.55
    end

    @testset "SpectralTensorCoefficient" begin
        eigvec = Vec((1.0,0.0))
        eigval = -1.0
        stc = SpectralTensorCoefficient(
            ConstantCoefficient(TransverselyIsotropicMicrostructure(eigvec)),
            ConstantCoefficient(SVector((eigval,0.0))),
        )
        st = Tensor{2,2}((-1.0,0.0,0.0,0.0))
        coeff_cache = Thunderbolt.setup_coefficient_cache(stc, qr, sdh)
        for i in 1:2
            reinit!(cell_cache, i)
            @test_opt evaluate_coefficient(coeff_cache, cell_cache, qp1, 0.0)
            @test evaluate_coefficient(coeff_cache, cell_cache, qp1, 0.0) ≈ st
            @test evaluate_coefficient(coeff_cache, cell_cache, qp2, 0.0) ≈ st
            @test evaluate_coefficient(coeff_cache, cell_cache, qp1, 1.0) ≈ st
            @test evaluate_coefficient(coeff_cache, cell_cache, qp2, 1.0) ≈ st
        end

        st2 = Tensor{2,2}((-1.0,0.0,0.0,-1.0))
        stc2 = SpectralTensorCoefficient(
            ConstantCoefficient(TransverselyIsotropicMicrostructure(eigvec)),
            ConstantCoefficient(SVector((eigval,eigval))),
        )
        coeff_cache = Thunderbolt.setup_coefficient_cache(stc2, qr, sdh)
        @test evaluate_coefficient(coeff_cache, cell_cache, qp1, 0.0) ≈ st2

        stc3 = SpectralTensorCoefficient(
            ConstantCoefficient(AnisotropicPlanarMicrostructure(Vec((1.0,0.0)), Vec((0.0,1.0)))),
            ConstantCoefficient(SVector((eigval,eigval))),
        )
        coeff_cache = Thunderbolt.setup_coefficient_cache(stc3, qr, sdh)
        @test evaluate_coefficient(coeff_cache, cell_cache, qp1, 0.0) ≈ st2
    end

    @testset "SpatiallyHomogeneousDataField" begin
        shdc = SpatiallyHomogeneousDataField(
            [1.0, 2.0],
            [Vec((0.1,)), Vec((0.2,)), Vec((0.3,))]
        )
        coeff_cache = Thunderbolt.setup_coefficient_cache(shdc, qr, sdh)
        for i in 1:2
            reinit!(cell_cache, i)
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
        end
    end

    @testset "ConductivityToDiffusivityCoefficient" begin
        eigvec = Vec((1.0,0.0))
        eigval = -1.0
        stc = SpectralTensorCoefficient(
            ConstantCoefficient(TransverselyIsotropicMicrostructure(eigvec)),
            ConstantCoefficient(SVector((eigval,0.0))),
        )
        st = Tensor{2,2}((-1.0,0.0,0.0,0.0))
        ctdc = Thunderbolt.ConductivityToDiffusivityCoefficient(
            stc,
            ConstantCoefficient(2.0),
            ConstantCoefficient(0.5),
        )
        coeff_cache = Thunderbolt.setup_coefficient_cache(ctdc, qr, sdh)
        for i in 1:2
            reinit!(cell_cache, i)
            @test_opt evaluate_coefficient(coeff_cache, cell_cache, qp1, 0.0)
            @test evaluate_coefficient(coeff_cache, cell_cache, qp1, 0.0) ≈ st
            @test evaluate_coefficient(coeff_cache, cell_cache, qp2, 0.0) ≈ st
            @test evaluate_coefficient(coeff_cache, cell_cache, qp1, 1.0) ≈ st
            @test evaluate_coefficient(coeff_cache, cell_cache, qp2, 1.0) ≈ st
        end
    end
end
