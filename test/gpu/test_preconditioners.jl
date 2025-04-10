using SparseArrays
using LinearSolve
using SparseMatricesCSR
using KernelAbstractions

function make_test_matrix(N)
    # Poisson's equation in 1D with Dirichlet BCs
    # Symmetric tridiagonal matrix (CSC)
    return spdiagm(0 => 2 * ones(N), -1 => -ones(N-1), 1 => -ones(N-1))
end


function expected_result(x)
    # Expected result after applying L1 preconditioner with partition size 2:
    # y[i] = x[i] / (A[i,i] + sum(|A[i,j]| for j not in partition))
    y =  x ./ (2 .+ 1)
    y[1] = x[1] / 2
    y[end] = x[end] / 2
    return y
end

function test_sym_csc(A,x,partsize,backend)
    expected_y = expected_result(x)
    cudabuilder = L1GSPrecBuilder(backend)
    @testset "$backend CSC Symmetric" begin
        N = size(A, 1)
        for nparts in 1:partsize:N
            P = cudabuilder(A, partsize, nparts)
            # y = similar(x)
            # LinearSolve.ldiv!(y, P, x)
            y = P \ x
            @test isapprox(y, expected_y; atol=1e-10)
        end
    end
end

function test_sym_csr(A,x,partsize,backend)
    expected_y = expected_result(x)
    B = SparseMatrixCSR(A)  # CSR version of A
    cudabuilder = L1GSPrecBuilder(backend)
    @testset "$backend CSR Symmetric" begin
        N = size(A, 1)
        for nparts in 1:partsize:N
            P = cudabuilder(B, partsize, nparts)
            # y = similar(x)
            # LinearSolve.ldiv!(y, P, x)
            y = P \ x
            @test isapprox(y, expected_y; atol=1e-10)
        end
    end
end


@testset "L1 Preconditioner" begin
    N = 8
    A = make_test_matrix(N)
    x = Float32.(0:N-1)

    test_sym_csc(A, x, 2, CUDABackend())
    test_sym_csr(A, x, 2, CUDABackend())
    test_sym_csc(A, x, 2, CPU())
    test_sym_csr(A, x, 2, CPU())

    #### Non-symmetric CSC (GPU/CPU) ####
    @testset "Non-Symmetric CSC" begin
        A2 = copy(A)
        A2[1, 8] = -1.0  # won't affect the result
        A2[2, 8] = -1.0  # needs to  be handled
        expected_y2 = expected_result(x)  
        expected_y2[2] = x[2] / (A2[2,2] + abs(A2[2,3]) + abs(A2[2,8]) )  # Adjusted for non-symmetric case

        # GPU
        P_gpu = L1GSPrecBuilder(CUDABackend())(A2, 2, 1)
        y_gpu = similar(x)
        LinearSolve.ldiv!(y_gpu, P_gpu, x)
        @test isapprox(y_gpu, expected_y2; atol=1e-10)

        # CPU
        P_cpu = L1GSPrecBuilder(CPU())(A2, 2, 1)
        # y_cpu = similar(x)
        # LinearSolve.ldiv!(y_cpu, P_cpu, x)
        y_cpu = P_cpu \ x
        @test isapprox(y_cpu, expected_y2; atol=1e-10)
    end
    
end
