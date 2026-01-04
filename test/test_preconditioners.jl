using MatrixDepot, LinearSolve, SparseArrays, SparseMatricesCSR
using KernelAbstractions
using JLD2: load
import Thunderbolt: ThreadedSparseMatrixCSR
using TimerOutputs
using LinearAlgebra: Symmetric

##########################################
## L1 Gauss Seidel Preconditioner - CPU ##
##########################################

# Enable debug timings for Thunderbolt
TimerOutputs.enable_debug_timings(Thunderbolt.Preconditioners)

function poisson_test_matrix(N)
    # Poisson's equation in 1D with Dirichlet BCs
    # Symmetric tridiagonal matrix (CSC)
    return spdiagm(0 => 2 * ones(N), -1 => -ones(N - 1), 1 => -ones(N - 1))
end

function test_sym(testname, A, x, y_exp, D_Dl1_exp, SLbuffer_exp, partsize)
function test_sym(testname, A, x, y_exp, D_Dl1_exp, SLbuffer_exp, partsize)
    @testset "$testname Symmetric" begin
        total_ncores = 8 # Assuming 8 cores for testing
        for ncores = 1:total_ncores # testing for multiple cores to check that the answer is independent of the number of cores
            builder = L1GSPrecBuilder(PolyesterDevice(ncores))
            P = A isa Symmetric ? builder(A, partsize;sweep=ForwardSweep(),storage=PackedBuffer()) : builder(A, partsize; isSymA=true,sweep=ForwardSweep(),storage=PackedBuffer())
            @test P.D_Dl1 ≈ D_Dl1_exp
            @test P.sweepstorage.SLbuffer ≈ SLbuffer_exp
            y = P \ x
            @test y ≈ y_exp
        end
    end
end

function test_l1gs_prec(A, b, partsize=nothing)
    ncores = 8 # Assuming 8 cores for testing
    partsize = partsize === nothing ? size(A, 1) / ncores |> ceil |> Int : partsize

    prob = LinearProblem(A, b)
    sol_unprec = solve(prob, KrylovJL_GMRES())
    @test isapprox(A * sol_unprec.u, b, rtol=1e-1, atol=1e-1)
    TimerOutputs.reset_timer!()
    P = L1GSPrecBuilder(PolyesterDevice(ncores))(A, partsize;sweep = ForwardSweep())

    println("\n" * "="^60)
    println("Preconditioner Construction Timings:")
    TimerOutputs.print_timer()
    println("="^60)

    # Reset timer before testing ldiv!
    TimerOutputs.reset_timer!()
    sol_prec = solve(prob, KrylovJL_GMRES(P); Pl=P)

    println("\n" * "="^60)
    println("ldiv! Timings during solve:")
    TimerOutputs.print_timer()
    println("="^60)

    println("Unprec. no. iters: $(sol_unprec.iters), time: $(sol_unprec.stats.timer)")
    println("Prec. no. iters: $(sol_prec.iters), time: $(sol_prec.stats.timer)")
    @test isapprox(A * sol_prec.u, b, rtol = 1e-1, atol = 1e-1)
    @test sol_prec.iters < sol_unprec.iters
end

@testset "L1GS Preconditioner" begin
    @testset "Algorithm" begin
        N = 9
        A = poisson_test_matrix(N)
        x = 0:N-1 |> collect .|> Float64
        y_exp = [0, 1 / 2, 1.0, 2.0, 2.0 , 3.5 ,3.0, 5.0, 4.0]
        D_Dl1_exp = Float64.([2, 2, 2, 2, 2, 2, 2, 2, 2])  # η=1.5: all rows satisfy a_ii >= η*dl1_ii (2 >= 1.5*1)
        SLbuffer_exp = Float64.([-1, -1, -1, -1])
        test_sym("CPU CSC", A, x, y_exp, D_Dl1_exp, SLbuffer_exp, 2)
        B = SparseMatrixCSR(A)
        test_sym("CPU, CSR", B, x, y_exp, D_Dl1_exp, SLbuffer_exp, 2)
        test_sym("CPU, CSR", B, x, y_exp, D_Dl1_exp, SLbuffer_exp, 2)
        C = ThreadedSparseMatrixCSR(B)
        test_sym("CPU, Threaded CSR", C, x, y_exp, D_Dl1_exp, SLbuffer_exp, 2)

        @testset "η parameter" begin
            # Test with η = 2.0 (more strict than default 1.5)
            # For Poisson matrix: a_ii = 2, dl1_ii = 1 for all rows
            # Since a_ii = 2 >= η*dl1_ii = 2*1 = 2, all rows satisfy condition
            # Therefore dl1star_ii = 0 for all rows, D_Dl1 = a_ii = 2
            η = 2.0
            D_Dl1_exp = Float64.([2, 2, 2, 2, 2, 2, 2, 2, 2])
            SLbuffer_exp = Float64.([-1, -1, -1, -1])
            builder = L1GSPrecBuilder(PolyesterDevice(2))
            P = builder(A, 2; η=η, storage=PackedBuffer())
            @test P.D_Dl1 ≈ D_Dl1_exp
            @test P.sweepstorage.SLbuffer ≈ SLbuffer_exp

            # Test with η = 3.0 (very strict)
            # a_ii = 2 < η*dl1_ii = 3*1 = 3, condition NOT satisfied
            # Therefore dl1star_ii = dl1_ii/2 = 1/2 = 0.5
            # D_Dl1 = a_ii + dl1star_ii = 2 + 0.5 = 2.5
            η = 3.0
            D_Dl1_exp = Float64.([2.0, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5])
            P = builder(A, 2; η=η, storage=PackedBuffer())
            @test P.D_Dl1 ≈ D_Dl1_exp
            @test P.sweepstorage.SLbuffer ≈ SLbuffer_exp

            # Test with η = 1.0 (less strict than default)
            # a_ii = 2 >= η*dl1_ii = 1*1 = 1, condition satisfied
            # Therefore dl1star_ii = 0
            # D_Dl1 = a_ii = 2
            η = 1.0
            D_Dl1_exp = Float64.([2, 2, 2, 2, 2, 2, 2, 2, 2])
            P = builder(A, 2; η=η, storage=PackedBuffer())
            @test P.D_Dl1 ≈ D_Dl1_exp
            @test P.sweepstorage.SLbuffer ≈ SLbuffer_exp

            # Verify preconditioner still works correctly with different η
            y_exp = [0, 1 / 2, 1.0, 2.0, 2.0, 3.5, 3.0, 5.0, 4.0]
            η = 2.0
            P = builder(A, 2; η=η, storage=PackedBuffer())
            y = P \ x
            @test y ≈ y_exp
        end


        @testset "Non-Symmetric CSC" begin
            A2 = copy(A)
            A2[1, 8] = -1.0  # won't affect the result
            A2[2, 8] = -1.0  # 1/2 → 1/3
            y2_exp = [0, 1 / 3, 1.0, 2.0, 2.0 , 3.5 ,3.0, 5.0, 4.0]
            D_Dl1_exp2 = Float64.([2, 3, 2, 2, 2, 2, 2, 2, 2])  # η=1.5: only row 1 has a_ii < η*dl1_ii (2 < 1.5*2)
            SLbuffer_exp2 = Float64.([-1, -1, -1, -1])

            builder = L1GSPrecBuilder(PolyesterDevice(2))
            P = builder(A2, 2; storage=PackedBuffer())
            @test P.D_Dl1 ≈ D_Dl1_exp2
            @test P.sweepstorage.SLbuffer ≈ SLbuffer_exp2
            y_cpu = P \ x
            @test y_cpu ≈ y2_exp
        end

        @testset "Partsize" begin
            partsize = 3
            ncores = 2
            D_Dl1_exp = Float64.([2, 2, 2, 2, 2, 2, 2, 2, 2])  # η=1.5: all rows satisfy a_ii >= η*dl1_ii
            SLbuffer_exp = Float64.([-1, 0, -1, -1, 0, -1, -1, 0, -1])
            builder = L1GSPrecBuilder(PolyesterDevice(ncores))
            P = builder(A, partsize; storage=PackedBuffer())
            @test P.D_Dl1 ≈ D_Dl1_exp
            @test P.sweepstorage.SLbuffer ≈ SLbuffer_exp
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
            A = md.A # type here is Symmetric{Float64, SparseMatrixCSC{Float64, Int64}}
            b = ones(size(A, 1))
            test_l1gs_prec(A, b)
        end
    end
end
