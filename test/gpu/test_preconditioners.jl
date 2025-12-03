using Test
using SparseArrays
using LinearSolve
using SparseMatricesCSR
using KernelAbstractions
using MatrixDepot
using Thunderbolt
import Thunderbolt: CudaDevice

##########################################
## L1 Gauss Seidel Preconditioner - GPU ##
##########################################

function poisson_test_matrix(N)
    # Poisson's equation in 1D with Dirichlet BCs
    # Symmetric tridiagonal matrix (CSC)
    return spdiagm(0 => 2 * ones(N), -1 => -ones(N - 1), 1 => -ones(N - 1))
end

function test_sym(testname,A, x,y_exp,D_Dl1_exp,SLbuffer_exp, partsize)
    @testset "$testname Symmetric" begin
        total_nblocks = 10
        total_nthreads = 10
        for nblocks in 1:total_nblocks # testing for multiple `nblocks` and `nthreads` to check that the answer is independent of the config.
            for nthreads in 1:total_nthreads
                builder = L1GSPrecBuilder(CudaDevice(nblocks, nthreads))
                P = builder(A, partsize)
                @test Vector(P.D_Dl1) ≈ D_Dl1_exp
                @test Vector(P.SLbuffer) ≈ SLbuffer_exp
                y = P \ x
                @test y ≈ y_exp
            end
        end
    end
end

function test_l1gs_prec(A, b)
    prob = LinearProblem(A, b)
    sol_unprec = solve(prob, KrylovJL_GMRES())
    @test isapprox(A * sol_unprec.u, b, rtol=1e-1, atol=1e-1)
    
    # Test L1GS Preconditioner
    nblocks = 20
    nthreads = 256

    partsize = 10
    builder = L1GSPrecBuilder(CudaDevice(nblocks, nthreads))
    P = builder(A, partsize)
    sol_prec = solve(prob, KrylovJL_GMRES(P); Pl=P)
    println("Unprec. no. iters: $(sol_unprec.iters), time: $(sol_unprec.stats.timer)")
    println("Prec. no. iters: $(sol_prec.iters), time: $(sol_prec.stats.timer)")
    @test isapprox(A * sol_prec.u, b, rtol=1e-1, atol=1e-1)
    @test sol_prec.iters < sol_unprec.iters
    @test sol_prec.resid < sol_unprec.resid
    @test sol_prec.stats.timer < sol_unprec.stats.timer
end

@testset "L1GS Preconditioner - GPU" begin
    @testset "Algorithm" begin
        N = 9
        A = poisson_test_matrix(N)
        x = 0:N-1 |> collect .|> Float64
        y_exp = [0, 1 / 2, 1.0, 2.0, 2.0 , 3.5 ,3.0, 5.0, 4.0]
        D_Dl1_exp = Float64.([2, 2, 2, 2, 2, 2, 2, 2, 2])  # η=1.5: all rows satisfy a_ii >= η*dl1_ii (2 >= 1.5*1)
        SLbuffer_exp = Float64.([-1,-1,-1,-1])
        test_sym("GPU CSC",A, x,y_exp,D_Dl1_exp,SLbuffer_exp, 2)
        B = SparseMatrixCSR(A)
        test_sym("GPU CSR",B, x,y_exp,D_Dl1_exp,SLbuffer_exp, 2)

        @testset "Non-Symmetric CSC" begin
            A2 = copy(A)
            A2[1, 8] = -1.0  # won't affect the result
            A2[2, 8] = -1.0  # 1/2 → 1/3
            y2_exp = [0, 1 / 3, 1.0, 2.0, 2.0 , 3.5 ,3.0, 5.0, 4.0]
            D_Dl1_exp2 = Float64.([2, 3, 2, 2, 2, 2, 2, 2, 2])  # η=1.5: only row 1 has a_ii < η*dl1_ii (2 < 1.5*2)
            SLbuffer_exp2 = Float64.([-1, -1, -1, -1])

            builder = L1GSPrecBuilder(CudaDevice(2,2))
            P = builder(A2, 2)
            @test Vector(P.D_Dl1) ≈ D_Dl1_exp2
            @test Vector(P.SLbuffer) ≈ SLbuffer_exp2
            y_cpu = P \ x
            @test y_cpu ≈ y2_exp
        end

        @testset "Partsize" begin
            partsize = 3
            D_Dl1_exp = Float64.([2,2,3,3,2,3,3,2,2])
            SLbuffer_exp = Float64.([-1,0,-1,-1,0,-1,-1,0,-1])
            builder = L1GSPrecBuilder(CudaDevice(2,2))
            P = builder(A, partsize)
            @test Vector(P.D_Dl1) ≈ D_Dl1_exp
            @test Vector(P.SLbuffer) ≈ SLbuffer_exp
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
