using MatrixDepot, LinearSolve, SparseArrays, SparseMatricesCSR
using KernelAbstractions
import Thunderbolt: ThreadedSparseMatrixCSR

##########################################
## L1 Gauss Seidel Preconditioner - CPU ##
##########################################

function poisson_test_matrix(N)
    # Poisson's equation in 1D with Dirichlet BCs
    # Symmetric tridiagonal matrix (CSC)
    return spdiagm(0 => 2 * ones(N), -1 => -ones(N - 1), 1 => -ones(N - 1))
end

function test_sym(testname,A, x,y_exp,D_Dl1_exp,SLbuffer_exp, partsize)
    @testset "$testname Symmetric" begin
        total_ncores = 8 # Assuming 8 cores for testing
        for ncores in 1:total_ncores # testing for multiple cores to check that the answer is independent of the number of cores
            builder = L1GSPrecBuilder(CPUSetting(ncores))
            P = builder(A, partsize)
            @test P.D_Dl1 ≈ D_Dl1_exp
            @test P.SLbuffer ≈ SLbuffer_exp
            y = P \ x
            @test y ≈ y_exp
        end
    end
end

function test_l1gs_prec(A, b)
    ncores = 8 # Assuming 8 cores for testing
    partsize = size(A, 1) / ncores |> ceil |> Int

    prob = LinearProblem(A, b)
    sol_unprec = solve(prob, KrylovJL_GMRES())
    @test isapprox(A * sol_unprec.u, b, rtol=1e-1, atol=1e-1)

    # Test L1GS Preconditioner
    P = L1GSPrecBuilder(CPUSetting(ncores))(A, partsize)
    sol_prec = solve(prob, KrylovJL_GMRES(P); Pl=P)
    println("Unprec. no. iters: $(sol_unprec.iters), time: $(sol_unprec.stats.timer)")
    println("Prec. no. iters: $(sol_prec.iters), time: $(sol_prec.stats.timer)")
    @test isapprox(A * sol_prec.u, b, rtol=1e-1, atol=1e-1)
    @test sol_prec.iters < sol_unprec.iters
    @test sol_prec.resid < sol_unprec.resid
    @test sol_prec.stats.timer < sol_unprec.stats.timer
end

@testset "L1GS Preconditioner" begin
    @testset "CPUSetting" begin
        # Test the default CPUSetting
        builder = L1GSPrecBuilder(CPU())
        backsetting = builder.backsetting
        @test backsetting.backend == CPU()
        @test backsetting.ncores == Threads.nthreads()

        # Test the full constructor
        ncores = rand(1:Threads.nthreads())
        builder = L1GSPrecBuilder(CPUSetting(ncores))
        backsetting = builder.backsetting
        @test backsetting.backend == CPU()
        @test backsetting.ncores == ncores
    end
    @testset "Algorithm" begin
        N = 9
        A = poisson_test_matrix(N)
        x = 0:N-1 |> collect .|> Float64
        y_exp = [0, 1/3, 2/3, 11/9, 4/3, 19/9, 2, 3.0, 8/3]
        D_Dl1_exp = Float64.([2,3,3,3,3,3,3,3,3])
        SLbuffer_exp = Float64.([-1,-1,-1,-1])
        test_sym("CPU CSC",A, x,y_exp,D_Dl1_exp,SLbuffer_exp, 2)
        B = SparseMatrixCSR(A)
        test_sym("CPU, CSR",B, x,y_exp,D_Dl1_exp,SLbuffer_exp, 2)
        C = ThreadedSparseMatrixCSR(B)
        test_sym("CPU, Threaded CSR",C, x,y_exp,D_Dl1_exp,SLbuffer_exp, 2)


        @testset "Non-Symmetric CSC" begin
            A2 = copy(A)
            A2[1, 8] = -1.0  # won't affect the result
            A2[2, 8] = -1.0  # 1/3 → 1/4
            y2_exp = [0, 1/4, 2/3, 11/9, 4/3, 19/9, 2, 3.0, 8/3]
            D_Dl1_exp2 = Float64.([3, 4, 3, 3, 3, 3, 3, 3, 3])
            SLbuffer_exp2 = Float64.([-1, -1, -1, -1])

            builder = L1GSPrecBuilder(CPUSetting(2))
            P = builder(A2, 2)
            @test P.D_Dl1 ≈ D_Dl1_exp2
            @test P.SLbuffer ≈ SLbuffer_exp2
            y_cpu = P \ x
            @test y_cpu ≈ y2_exp
        end

        @testset "Partsize" begin
            partsize = 3
            ncores = 2
            D_Dl1_exp = Float64.([2,2,3,3,2,3,3,2,2])
            SLbuffer_exp = Float64.([-1,0,-1,-1,0,-1,-1,0,-1])
            builder = L1GSPrecBuilder(CPUSetting(ncores))
            P = builder(A, partsize)
            @test P.D_Dl1 ≈ D_Dl1_exp
            @test P.SLbuffer ≈ SLbuffer_exp
        end
    end
    @testset "Solution with LinearSolve" begin

        @testset "Non-Symmetric A" begin
            md = mdopen("HB/sherman5")
            A = md.A
            b = md.b[:, 1]
            test_l1gs_prec(A, b)
        end

        @testset "Symmetric A" begin
            md = mdopen("HB/bcsstk15") 
            A = md.A
            b = ones(size(A, 1))
            test_l1gs_prec(A, b)
        end
    end
end

