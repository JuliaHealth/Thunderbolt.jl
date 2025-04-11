using SparseArrays
using LinearSolve
using SparseMatricesCSR
using KernelAbstractions
using MatrixDepot


##########################################
## L1 Gauss Seidel Preconditioner - GPU ##
##########################################

function poisson_test_matrix(N)
    # Poisson's equation in 1D with Dirichlet BCs
    # Symmetric tridiagonal matrix (CSC)
    return spdiagm(0 => 2 * ones(N), -1 => -ones(N - 1), 1 => -ones(N - 1))
end


function poisson_l1gs_expected_result(x)
    # Expected result after applying L1 preconditioner with partition size 2:
    # y[i] = x[i] / (A[i,i] + sum(|A[i,j]| for j not in partition))
    y = x ./ (2 .+ 1)
    y[1] = x[1] / 2
    y[end] = x[end] / 2
    return y
end

function test_sym_csc(A, x, partsize, backend)
    expected_y = poisson_l1gs_expected_result(x)
    builder = L1GSPrecBuilder(backend)
    @testset "$backend CSC Symmetric" begin
        N = size(A, 1)
        for nparts in 1:partsize:N
            P = builder(A, partsize, nparts)
            y = P \ x
            @test isapprox(y, expected_y; atol=1e-10)
        end
    end
end

function test_sym_csr(A, x, partsize, backend)
    expected_y = poisson_l1gs_expected_result(x)
    B = SparseMatrixCSR(A)  # CSR version of A
    builder = L1GSPrecBuilder(backend)
    @testset "$backend CSR Symmetric" begin
        N = size(A, 1)
        for nparts in 1:partsize:N
            P = builder(B, partsize, nparts)
            y = P \ x
            @test isapprox(y, expected_y; atol=1e-10)
        end
    end
end

function test_l1gs_prec(A, b)
    nparts = 20
    partsize = 256

    prob = LinearProblem(A, b)
    sol_unprec = solve(prob, KrylovJL_GMRES())
    @test isapprox(A * sol_unprec.u, b, rtol=1e-1, atol=1e-1)

    # Test L1GS Preconditioner
    P = L1GSPrecBuilder(CUDABackend())(A, partsize, nparts)
    sol_prec = solve(prob, KrylovJL_GMRES(P); Pl=P)
    println("Unprec. no. iters: $(sol_unprec.iters), time: $(sol_unprec.stats.timer)")
    println("Prec. no. iters: $(sol_prec.iters), time: $(sol_prec.stats.timer)")
    @test isapprox(A * sol_prec.u, b, rtol=1e-1, atol=1e-1)
    @test sol_prec.iters < sol_unprec.iters
    @test sol_prec.resid < sol_unprec.resid
    @test sol_prec.stats.timer < sol_unprec.stats.timer
end

@testset "L1 Gauss Seidel Preconditioner - GPU" begin

    @testset "Algorithm" begin
        N = 8
        A = poisson_test_matrix(N)
        x = Float32.(0:N-1)

        test_sym_csc(A, x, 2, CUDABackend())
        test_sym_csr(A, x, 2, CUDABackend())

        #### Non-symmetric CSC (GPU/CPU) ####
        @testset "Non-Symmetric CSC" begin
            A2 = copy(A)
            A2[1, 8] = -1.0  # won't affect the result
            A2[2, 8] = -1.0  # needs to  be handled
            expected_y2 = poisson_l1gs_expected_result(x)
            expected_y2[2] = x[2] / (A2[2, 2] + abs(A2[2, 3]) + abs(A2[2, 8]))  # Adjusted for non-symmetric case

            # GPU
            P_gpu = L1GSPrecBuilder(CUDABackend())(A2, 2, 1)
            y_gpu = similar(x)
            LinearSolve.ldiv!(y_gpu, P_gpu, x)
            @test isapprox(y_gpu, expected_y2; atol=1e-10)
        end

    end

    @testset "Solution with LinearSolve" begin

        @testset "Unsymmetric A" begin
            md = mdopen("HB/sherman5")
            A = md.A
            b = md.b[:, 1]
            test_l1gs_prec(A, b)
        end

        @testset "Symmetric A" begin
            md = mdopen("HB/bcsstk15") # ill-conditioned matrix
            A = md.A
            b = ones(size(A, 1))
            test_l1gs_prec(A, b)
        end
    end
end
