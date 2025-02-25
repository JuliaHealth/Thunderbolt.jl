
"""
    coeffs_kernel!(Vals, sdh, coeff_cache, cv, t)

Compute and store coefficient values at quadrature points for all cells in the grid 
    (i.e. we store n_cells * n_quadoints values).

# Arguments
- `Vals::Vector{<:Any}`: Output vector to store computed coefficient values. This vector has length 
    equal to n_cells * n_quadpoints. 
- `sdh::DeviceSubDofHandler`: device instance of `SubDofHandler`.
- `coeff_cache`: Cache object used for coefficient evaluation.
- `cv::CellValues`: CellValues object containing quadrature rule and basis function information.
- `t::Real`: Current time (used for time-dependent coefficients).

# Implementation Details
1. Loops through each subdofhandler in the DofHandler
2. For each cell in the subdofhandler:
   - Retrieves cell coordinates and ID
   - Iterates over quadrature points using `QuadratureValuesIterator`
   - Evaluates coefficients at each quadrature point using `evaluate_coefficient`
   - Stores results in `Vals` with cell-specific offset calculation
"""
function coeffs_kernel!(Vals, sdh, coeff_cache, cv, t)
    for cell in CellIterator(sdh) # iterate over all cells in the subdomain
        cell_id = cellid(cell)
        coords = getcoordinates(cell)
        quadrature_iterator = QuadratureValuesIterator(cv, coords)
        n_quad_points = length(quadrature_iterator) # number of quadrature points
        for (i, qv) in pairs(quadrature_iterator) # pairs will fetch the current index (i) and the `StaticQuadratureValue` (qv)
            qp = QuadraturePoint(i, qv.ξ)
            fx = evaluate_coefficient(coeff_cache, cell, qp, t)
            Vals[(cell_id-1)*n_quad_points+i] = fx
        end
    end
    return nothing
end

import Thunderbolt: setup_coefficient_cache, evaluate_coefficient, ConductivityToDiffusivityCoefficient, deep_adapt,QuadraturePoint
import Thunderbolt.FerriteUtils:
    CellIterator, QuadratureValuesIterator, cellid, getcoordinates, getnbasefunctions
import CUDA: @cuda
import Adapt: adapt_structure
import Tensors: Vec

@testset "Coefficient API" begin
    left = Tensor{1,1,Float32}((-1.0,)) # define the left bottom corner of the grid.
    right = Tensor{1,1,Float32}((1.0,)) # define the right top corner of the grid.
    grid = generate_grid(Line, (2,), left, right)
    dh = DofHandler(grid)
    ip_collection = LagrangeCollection{1}()
    ip = getinterpolation(ip_collection, first(grid.cells))
    add!(dh, :u, getinterpolation(ip_collection, first(grid.cells)))
    close!(dh)
    qr = QuadratureRule{RefLine}([1.0f0, 1.0f0], [Vec{1}((0.0f0,)), Vec{1}((0.1f0,))])
    n_quad = qr.weights |> length
    cellvalues = CellValues(Float32, qr, ip)
    n_cells = grid.cells |> length
    sdh = first(dh.subdofhandlers)

    strategy = Thunderbolt.CudaAssemblyStrategy(Float32, Int32)
    cu_dh = adapt_structure(strategy, dh)
    cu_sdh = cu_dh.subdofhandlers[1]
    @testset "ConstantCoefficient($val" for val ∈ [1.0f0]
        cc = ConstantCoefficient(val)
        coeff_cache = setup_coefficient_cache(cc, qr, sdh)
        correct_vals = ones(Float32, n_cells * n_quad)
        Vals = zeros(Float32, n_cells * n_quad) |> cu
        cuda_strategy = Thunderbolt.CudaAssemblyStrategy(Float32, Int32)

        @cuda blocks = 1 threads = n_cells coeffs_kernel!(Vals, cu_sdh, coeff_cache, cellvalues, 0.0f0)
        @test Vector(Vals) ≈ correct_vals

    end

    @testset "FieldCoefficient" begin
        data_scalar = zeros(Float32, 2, 2)
        data_scalar[1, 1] = 1.0f0
        data_scalar[1, 2] = -1.0f0
        data_scalar[2, 1] = -1.0f0
        fcs = FieldCoefficient(data_scalar, ip_collection)
        coeff_cache = adapt_structure(strategy, setup_coefficient_cache(fcs, qr, sdh))
        correct_vals = [0.0f0, -0.1f0, -0.5f0, (0.1f0 + 1.0f0) / 2.0f0 - 1.0f0]
        Vals = zeros(Float32, correct_vals |> length) |> cu
        cuda_strategy = Thunderbolt.CudaAssemblyStrategy(Float32, Int32)

        CUDA.@cuda blocks = 1 threads = n_cells coeffs_kernel!(Vals, cu_sdh, coeff_cache, cellvalues, 0.0f0)
        @test Vector(Vals) ≈ correct_vals


        data_vector = zeros(Vec{2,Float32}, 2, 2)
        data_vector[1, 1] = Vec((1.0f0, 0.0f0))
        data_vector[1, 2] = Vec((0.0f0, -1.0f0))
        data_vector[2, 1] = Vec((-1.0f0, -0.0f0))
        fcv = FieldCoefficient(data_vector, ip_collection^2)
        coeff_cache = adapt_structure(strategy, setup_coefficient_cache(fcv, qr, sdh))
        correct_vals = [Vec((0.0f0, 0.0f0)), Vec((-0.1f0, 0.0f0)), Vec((0.0f0, -0.5f0)), Vec((0.0f0, (0.1f0 + 1.0f0) / 2.0f0 - 1.0f0))]
        Vals = correct_vals |> similar |> cu
        cuda_strategy = Thunderbolt.CudaAssemblyStrategy(Float32, Int32)

        @cuda blocks = 1 threads = n_cells coeffs_kernel!(Vals, cu_sdh, coeff_cache, cellvalues, 0.0f0)
        @test Vector(Vals) ≈ correct_vals

    end

    @testset "Cartesian CoordinateSystemCoefficient" begin
        ccsc = CoordinateSystemCoefficient(CartesianCoordinateSystem(grid))
        coeff_cache = setup_coefficient_cache(ccsc, qr, sdh)
        correct_vals = [Vec((-0.5f0,)), Vec((-0.45f0,)), Vec((0.5f0,)), Vec((0.55f0,))]
        Vals = correct_vals |> similar |> cu
        cuda_strategy = Thunderbolt.CudaAssemblyStrategy(Float32, Int32)

        @cuda blocks = 1 threads = n_cells coeffs_kernel!(Vals, cu_sdh, coeff_cache, cellvalues, 0.0f0)
        @test Vector(Vals) ≈ correct_vals

    end

    @testset "AnalyticalCoefficient" begin
        ac = AnalyticalCoefficient(
            (x, t) -> norm(x) + t,
            CoordinateSystemCoefficient(CartesianCoordinateSystem(grid))
        )
        coeff_cache = Thunderbolt.setup_coefficient_cache(ac, qr, sdh)
        correct_vals = [0.5f0, 0.45f0, 0.5f0, 0.55f0]
        Vals = correct_vals |> similar |> cu
        cuda_strategy = Thunderbolt.CudaAssemblyStrategy(Float32, Int32)

        @cuda blocks = 1 threads = n_cells coeffs_kernel!(Vals, cu_sdh, coeff_cache, cellvalues, 0.0f0)
        @test Vector(Vals) ≈ correct_vals

        correct_vals = [1.5f0, 1.45f0, 1.5f0, 1.55f0]
        Vals = correct_vals |> similar |> cu
        cuda_strategy = Thunderbolt.CudaAssemblyStrategy(Float32, Int32)

        @cuda blocks = 1 threads = n_cells coeffs_kernel!(Vals, cu_sdh, coeff_cache, cellvalues, 1.0f0)
        @test Vector(Vals) ≈ correct_vals

    end

    @testset "SpectralTensorCoefficient" begin
        eigvec = Vec((1.0f0, 0.0f0))
        eigval = -1.0f0
        stc = SpectralTensorCoefficient(
            ConstantCoefficient(TransverselyIsotropicMicrostructure(eigvec)),
            ConstantCoefficient(SVector((eigval, 0.0f0))),
        )
        st = Tensor{2,2,Float32}((-1.0, 0.0, 0.0, 0.0))
        coeff_cache = setup_coefficient_cache(stc, qr, sdh)
        correct_vals = [st, st, st, st]
        Vals = correct_vals |> similar |> cu
        cuda_strategy = Thunderbolt.CudaAssemblyStrategy(Float32, Int32)

        @cuda blocks = 1 threads = n_cells coeffs_kernel!(Vals, cu_sdh, coeff_cache, cellvalues, 0.0f0)
        @test Vector(Vals) ≈ correct_vals


        st2 = Tensor{2,2,Float32}((-1.0, 0.0, 0.0, -1.0))
        stc2 = SpectralTensorCoefficient(
            ConstantCoefficient(TransverselyIsotropicMicrostructure(eigvec)),
            ConstantCoefficient(SVector((eigval, eigval))),
        )
        coeff_cache = setup_coefficient_cache(stc2, qr, sdh)
        correct_vals = [st2, st2, st2, st2]
        Vals = correct_vals |> similar |> cu
        cuda_strategy = Thunderbolt.CudaAssemblyStrategy(Float32, Int32)

        @cuda blocks = 1 threads = n_cells coeffs_kernel!(Vals, cu_sdh, coeff_cache, cellvalues, 0.0f0)
        @test Vector(Vals) ≈ correct_vals


        stc3 = SpectralTensorCoefficient(
            ConstantCoefficient(AnisotropicPlanarMicrostructure(Vec((1.0f0, 0.0f0)), Vec((0.0f0, 1.0f0)))),
            ConstantCoefficient(SVector((eigval, eigval))),
        )
        coeff_cache = setup_coefficient_cache(stc3, qr, sdh)
        correct_vals = [st2, st2, st2, st2]
        Vals = correct_vals |> similar |> cu
        cuda_strategy = Thunderbolt.CudaAssemblyStrategy(Float32, Int32)

        @cuda blocks = 1 threads = n_cells coeffs_kernel!(Vals, cu_sdh, coeff_cache, cellvalues, 0.0f0)
        @test Vector(Vals) ≈ correct_vals

    end

    @testset "SpatiallyHomogeneousDataField" begin
        shdc = SpatiallyHomogeneousDataField(
            [1.0f0, 2.0f0],
            [Vec((0.1f0,)), Vec((0.2f0,)), Vec((0.3f0,))]
        )
        coeff_cache = adapt_structure(strategy, Thunderbolt.setup_coefficient_cache(shdc, qr, sdh))
        correct_vals = [Vec((0.1f0,)), Vec((0.1f0,)), Vec((0.1f0,)), Vec((0.1f0,))]
        Vals = correct_vals |> similar |> cu
        cuda_strategy = Thunderbolt.CudaAssemblyStrategy(Float32, Int32)

        @cuda blocks = 1 threads = n_cells coeffs_kernel!(Vals, cu_sdh, coeff_cache, cellvalues, 0.0f0)
        @test Vector(Vals) ≈ correct_vals


        correct_vals = [Vec((0.1f0,)), Vec((0.1f0,)), Vec((0.1f0,)), Vec((0.1f0,))]
        Vals = correct_vals |> similar |> cu
        cuda_strategy = Thunderbolt.CudaAssemblyStrategy(Float32, Int32)

        @cuda blocks = 1 threads = n_cells coeffs_kernel!(Vals, cu_sdh, coeff_cache, cellvalues, 1.0f0)
        @test Vector(Vals) ≈ correct_vals


        correct_vals = [Vec((0.2f0,)), Vec((0.2f0,)), Vec((0.2f0,)), Vec((0.2f0,))]
        Vals = correct_vals |> similar |> cu
        cuda_strategy = Thunderbolt.CudaAssemblyStrategy(Float32, Int32)

        @cuda blocks = 1 threads = n_cells coeffs_kernel!(Vals, cu_sdh, coeff_cache, cellvalues, 1.1f0)
        @test Vector(Vals) ≈ correct_vals


        correct_vals = [Vec((0.2f0,)), Vec((0.2f0,)), Vec((0.2f0,)), Vec((0.2f0,))]
        Vals = correct_vals |> similar |> cu
        cuda_strategy = Thunderbolt.CudaAssemblyStrategy(Float32, Int32)

        @cuda blocks = 1 threads = n_cells coeffs_kernel!(Vals, cu_sdh, coeff_cache, cellvalues, 2.0f0)
        @test Vector(Vals) ≈ correct_vals


        correct_vals = [Vec((0.3f0,)), Vec((0.3f0,)), Vec((0.3f0,)), Vec((0.3f0,))]
        Vals = correct_vals |> similar |> cu
        cuda_strategy = Thunderbolt.CudaAssemblyStrategy(Float32, Int32)

        @cuda blocks = 1 threads = n_cells coeffs_kernel!(Vals, cu_sdh, coeff_cache, cellvalues, 2.1f0)
        @test Vector(Vals) ≈ correct_vals

    end


    @testset "ConductivityToDiffusivityCoefficient" begin
        eigvec = Vec((1.0f0, 0.0f0))
        eigval = -1.0f0
        stc = SpectralTensorCoefficient(
            ConstantCoefficient(TransverselyIsotropicMicrostructure(eigvec)),
            ConstantCoefficient(SVector((eigval, 0.0f0))),
        )
        ctdc = ConductivityToDiffusivityCoefficient(
            stc,
            ConstantCoefficient(2.0f0),
            ConstantCoefficient(0.5f0),
        )
        coeff_cache = setup_coefficient_cache(ctdc, qr, sdh)
        correct_vals = [Tensor{2,2,Float32}((-1.0, 0.0, 0.0, 0.0)), Tensor{2,2,Float32}((-1.0, 0.0, 0.0, 0.0)), Tensor{2,2,Float32}((-1.0, 0.0, 0.0, 0.0)), Tensor{2,2,Float32}((-1.0, 0.0, 0.0, 0.0))]
        Vals = correct_vals |> similar |> cu
        cuda_strategy = Thunderbolt.CudaAssemblyStrategy(Float32, Int32)

        @cuda blocks = 1 threads = n_cells coeffs_kernel!(Vals, cu_sdh, coeff_cache, cellvalues, 0.0f0)
        @test Vector(Vals) ≈ correct_vals

        correct_vals = [Tensor{2,2,Float32}((-1.0, 0.0, 0.0, 0.0)), Tensor{2,2,Float32}((-1.0, 0.0, 0.0, 0.0)), Tensor{2,2,Float32}((-1.0, 0.0, 0.0, 0.0)), Tensor{2,2,Float32}((-1.0, 0.0, 0.0, 0.0))]
        Vals = correct_vals |> similar |> cu
        cuda_strategy = Thunderbolt.CudaAssemblyStrategy(Float32, Int32)

        @cuda blocks = 1 threads = n_cells coeffs_kernel!(Vals, cu_sdh, coeff_cache, cellvalues, 1.0f0)
        @test Vector(Vals) ≈ correct_vals
    end
end
