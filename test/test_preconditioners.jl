using MatrixDepot, LinearSolve, SparseArrays, SparseMatricesCSR
using KernelAbstractions
using ThreadPinning

##########################################
## L1 Gauss Seidel Preconditioner - CPU ##
##########################################

pinthreads(:cores) # to map each thread to core and avoid hyperthreading

function poisson_test_matrix(N)
    # Poisson's equation in 1D with Dirichlet BCs
    # Symmetric tridiagonal matrix (CSC)
    return spdiagm(0 => 2 * ones(N), -1 => -ones(N - 1), 1 => -ones(N - 1))
end


function poisson_l1gs_expected_result(x)
    # Expected result after applying L1 preconditioner with partition size 2:
    # y[i] = x[i] / (A[i,i] + sum(|A[i,j]| for j not in partition))
    return [0, 1/3, 2/3, 11/9, 4/3, 19/9, 2, 9/2]
end

function test_sym_csc(A, x, partsize)
    expected_y = poisson_l1gs_expected_result(x)
    backend = CPU()
    builder = L1GSPrecBuilder(backend)
    @testset "$backend CSC Symmetric" begin
        N = size(A, 1)
        for nparts in 1:partsize:N
            P = builder(A, partsize)
            y = P \ x
            @test isapprox(y, expected_y; atol=1e-10)
        end
    end
end


function test_sym_csr(A, x, partsize)
    expected_y = poisson_l1gs_expected_result(x)
    backend = CPU()
    builder = L1GSPrecBuilder(backend)
    B = SparseMatrixCSR(A)
    @testset "$backend CSR Symmetric" begin
        N = size(A, 1)
        for nparts in 1:partsize:N
            P = builder(B, partsize)
            y = P \ x
            @test isapprox(y, expected_y; atol=1e-10)
        end
    end
end


function test_l1gs_prec(A, b)
    nparts = ThreadPinning.ncores()
    partsize = size(A, 1) / nparts |> ceil |> Int

    prob = LinearProblem(A, b)
    sol_unprec = solve(prob, KrylovJL_GMRES())
    @test isapprox(A * sol_unprec.u, b, rtol=1e-1, atol=1e-1)

    # Test L1GS Preconditioner
    P = L1GSPrecBuilder(CPU())(A, partsize, nparts)
    sol_prec = solve(prob, KrylovJL_GMRES(P); Pl=P)
    println("Unprec. no. iters: $(sol_unprec.iters), time: $(sol_unprec.stats.timer)")
    println("Prec. no. iters: $(sol_prec.iters), time: $(sol_prec.stats.timer)")
    @test isapprox(A * sol_prec.u, b, rtol=1e-1, atol=1e-1)
    @test sol_prec.iters < sol_unprec.iters
    @test sol_prec.resid < sol_unprec.resid
    @test sol_prec.stats.timer < sol_unprec.stats.timer
end

@testset "L1GS Preconditioner" begin

    @testset "Algorithm" begin
        N = 8
        A = poisson_test_matrix(N)
        x = Float32.(0:N-1)
        test_sym_csc(A, x, 2)
        test_sym_csr(A, x, 2)

        @testset "Non-Symmetric CSC" begin
            A2 = copy(A)
            A2[1, 8] = -1.0  # won't affect the result
            A2[2, 8] = -1.0  # needs to  be handled
            expected_y2 = poisson_l1gs_expected_result(x)
            expected_y2[2] = x[2] / (A2[2, 2] + abs(A2[2, 3]) + abs(A2[2, 8]))  # Adjusted for non-symmetric case

            P_cpu = L1GSPrecBuilder(CPU())(A2, 2, 1)
            y_cpu = P_cpu \ x
            @test isapprox(y_cpu, expected_y2; atol=1e-10)
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

