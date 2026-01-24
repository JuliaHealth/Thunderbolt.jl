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

function test_sym_result(testname, A, x, y_exp, D_Dl1_exp, partsize, sweep= ForwardSweep(),cache_strategy=PackedBufferCache(), test_buffer_fn = (P)-> nothing)
    @testset "$testname Symmetric" begin
        total_ncores = 8 # Assuming 8 cores for testing
        for ncores = 1:total_ncores # testing for multiple cores to check that the answer is independent of the number of cores
            builder = L1GSPrecBuilder(PolyesterDevice(ncores))
            P =
                A isa Symmetric ?
                builder(A, partsize; sweep = sweep, cache_strategy = cache_strategy) :
                builder(
                    A,
                    partsize;
                    isSymA = true,
                    sweep = sweep,
                    cache_strategy = cache_strategy,
                )
            if sweep isa SymmetricSweep
                @test P.sweep.lop.D_DL1 ≈ D_Dl1_exp
                @test P.sweep.uop.D_DL1 ≈ D_Dl1_exp
            else
                @test P.sweep.op.D_DL1 ≈ D_Dl1_exp
            end
            test_buffer_fn(P)
            y = P \ x
            @test y ≈ y_exp
        end
    end
end

function test_l1gs_prec(A, b, partsize = nothing)
    ncores = 8 # Assuming 8 cores for testing
    partsize = partsize === nothing ? size(A, 1) / ncores |> ceil |> Int : partsize

    prob = LinearProblem(A, b)
    sol_unprec = solve(prob, KrylovJL_GMRES())
    @test isapprox(A * sol_unprec.u, b, rtol = 1e-1, atol = 1e-1)
    TimerOutputs.reset_timer!()
    P = L1GSPrecBuilder(PolyesterDevice(ncores))(A, partsize; sweep = ForwardSweep())

    println("\n" * "="^60)
    println("Preconditioner Construction Timings:")
    TimerOutputs.print_timer()
    println("="^60)

    # Reset timer before testing ldiv!
    TimerOutputs.reset_timer!()
    sol_prec = solve(prob, KrylovJL_GMRES(P); Pl = P)

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
        x = 0:(N-1) |> collect .|> Float64
        y_exp_fwd = [0, 1 / 2, 1.0, 2.0, 2.0, 3.5, 3.0, 5.0, 4.0]
        y_exp_bwd = [0.25, 0.5, 1.75, 1.5, 3.25, 2.5, 4.75, 3.5, 4.0]
        y_exp_sym = [0.125, 0.25, 1.0, 1.0, 1.875, 1.75, 2.75, 2.5,2.0]
        D_Dl1_exp = Float64.([2, 2, 2, 2, 2, 2, 2, 2, 2])  # η=1.5: all rows satisfy a_ii >= η*dl1_ii (2 >= 1.5*1)
        SLbuffer_exp = Float64.([-1, -1, -1, -1])
        SUbuffer_exp = Float64.([-1, -1, -1, -1])
        
        # Packed buffer tests # 
        test_fwd_buffer_fn = (P) -> @test P.sweep.op.L.SLbuffer ≈ SLbuffer_exp
        test_bwd_buffer_fn = (P) -> @test P.sweep.op.U.SUbuffer ≈ SUbuffer_exp
        test_sym_buffer_fn = (P) -> begin
            @test P.sweep.lop.L.SLbuffer ≈ SLbuffer_exp
            @test P.sweep.uop.U.SUbuffer ≈ SUbuffer_exp
        end
        # Forward sweep packed buffer tests
        test_sym_result("Packed, Forward, CPU CSC", A, x, y_exp_fwd, D_Dl1_exp, 2, ForwardSweep(), PackedBufferCache(), test_fwd_buffer_fn)
        B = SparseMatrixCSR(A)
        test_sym_result("Packed, Forward, CPU CSR", B, x, y_exp_fwd, D_Dl1_exp, 2, ForwardSweep(), PackedBufferCache(), test_fwd_buffer_fn)
        test_sym_result("Packed, Forward, CPU CSR", B, x, y_exp_fwd, D_Dl1_exp, 2)
        C = ThreadedSparseMatrixCSR(B)
        test_sym_result("Packed, Forward, CPU Threaded CSR", C, x, y_exp_fwd, D_Dl1_exp, 2, ForwardSweep(), PackedBufferCache(), test_fwd_buffer_fn)

        # Backward sweep packed buffer tests
        test_sym_result("Packed, Backward, CPU CSC", A, x, y_exp_bwd, D_Dl1_exp, 2, BackwardSweep(), PackedBufferCache(), test_bwd_buffer_fn)
        test_sym_result("Packed, Backward, CPU CSR", B, x, y_exp_bwd, D_Dl1_exp, 2, BackwardSweep(), PackedBufferCache(), test_bwd_buffer_fn)
        test_sym_result("Packed, Backward, CPU CSR", B, x, y_exp_bwd, D_Dl1_exp, 2, BackwardSweep())
        test_sym_result("Packed, Backward, CPU Threaded CSR", C, x, y_exp_bwd, D_Dl1_exp, 2, BackwardSweep(), PackedBufferCache(), test_bwd_buffer_fn)

        # Symmetric sweep packed buffer tests
        test_sym_result("Packed, Symmetric, CPU CSC", A, x, y_exp_sym, D_Dl1_exp, 2, SymmetricSweep(), PackedBufferCache(), test_sym_buffer_fn)
        test_sym_result("Packed, Symmetric, CPU CSR", B, x, y_exp_sym, D_Dl1_exp, 2, SymmetricSweep(), PackedBufferCache(), test_sym_buffer_fn)
        test_sym_result("Packed, Symmetric, CPU CSR", B, x, y_exp_sym, D_Dl1_exp, 2, SymmetricSweep())
        test_sym_result("Packed, Symmetric, CPU Threaded CSR", C, x, y_exp_sym, D_Dl1_exp, 2, SymmetricSweep(), PackedBufferCache(), test_sym_buffer_fn)

        # MatrixViewCache tests
        # Forward sweep MatrixViewCache tests
        test_sym_result("MatrixView, Forward, CPU CSC", A, x, y_exp_fwd, D_Dl1_exp, 2, ForwardSweep(), MatrixViewCache())
        test_sym_result("MatrixView, Forward, CPU CSR", B, x, y_exp_fwd, D_Dl1_exp, 2, ForwardSweep(), MatrixViewCache())
        test_sym_result("MatrixView, Forward, CPU Threaded CSR", C, x, y_exp_fwd, D_Dl1_exp, 2, ForwardSweep(), MatrixViewCache())

        # Backward sweep MatrixViewCache tests
        test_sym_result("MatrixView, Backward, CPU CSC", A, x, y_exp_bwd, D_Dl1_exp, 2, BackwardSweep(), MatrixViewCache())
        test_sym_result("MatrixView, Backward, CPU CSR", B, x, y_exp_bwd, D_Dl1_exp, 2, BackwardSweep(), MatrixViewCache())
        test_sym_result("MatrixView, Backward, CPU Threaded CSR", C, x, y_exp_bwd, D_Dl1_exp, 2, BackwardSweep(), MatrixViewCache())

        # Symmetric sweep MatrixViewCache tests
        test_sym_result("MatrixView, Symmetric, CPU CSC", A, x, y_exp_sym, D_Dl1_exp, 2, SymmetricSweep(), MatrixViewCache())
        test_sym_result("MatrixView, Symmetric, CPU CSR", B, x, y_exp_sym, D_Dl1_exp, 2, SymmetricSweep(), MatrixViewCache())
        test_sym_result("MatrixView, Symmetric, CPU Threaded CSR", C, x, y_exp_sym, D_Dl1_exp, 2, SymmetricSweep(), MatrixViewCache())

        @testset "η parameter" begin
            # Test with η = 2.0 (more strict than default 1.5)
            # For Poisson matrix: a_ii = 2, dl1_ii = 1 for all rows
            # Since a_ii = 2 >= η*dl1_ii = 2*1 = 2, all rows satisfy condition
            # Therefore dl1star_ii = 0 for all rows, D_Dl1 = a_ii = 2
            η = 2.0
            D_Dl1_exp = Float64.([2, 2, 2, 2, 2, 2, 2, 2, 2])
            SLbuffer_exp = Float64.([-1, -1, -1, -1])
            builder = L1GSPrecBuilder(PolyesterDevice(2))
            P = builder(A, 2; η = η, cache_strategy = PackedBufferCache())
            @test P.sweep.op.D_DL1 ≈ D_Dl1_exp
            @test P.sweep.op.L.SLbuffer ≈ SLbuffer_exp

            # Test with η = 3.0 (very strict)
            # a_ii = 2 < η*dl1_ii = 3*1 = 3, condition NOT satisfied
            # Therefore dl1star_ii = dl1_ii/2 = 1/2 = 0.5
            # D_Dl1 = a_ii + dl1star_ii = 2 + 0.5 = 2.5
            η = 3.0
            D_Dl1_exp = Float64.([2.0, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5])
            P = builder(A, 2; η = η, cache_strategy = PackedBufferCache())
            @test P.sweep.op.D_DL1 ≈ D_Dl1_exp
            @test P.sweep.op.L.SLbuffer ≈ SLbuffer_exp

            # Test with η = 1.0 (less strict than default)
            # a_ii = 2 >= η*dl1_ii = 1*1 = 1, condition satisfied
            # Therefore dl1star_ii = 0
            # D_Dl1 = a_ii = 2
            η = 1.0
            D_Dl1_exp = Float64.([2, 2, 2, 2, 2, 2, 2, 2, 2])
            P = builder(A, 2; η = η, cache_strategy = PackedBufferCache())
            @test P.sweep.op.D_DL1 ≈ D_Dl1_exp
            @test P.sweep.op.L.SLbuffer ≈ SLbuffer_exp

            # Verify preconditioner still works correctly with different η
            y_exp = [0, 1 / 2, 1.0, 2.0, 2.0, 3.5, 3.0, 5.0, 4.0]
            η = 2.0
            P = builder(A, 2; η = η, cache_strategy = PackedBufferCache())
            y = P \ x
            @test y ≈ y_exp
        end


        @testset "Non-Symmetric CSC" begin
            A2 = copy(A)
            A2[1, 8] = -1.0  # won't affect the result
            A2[2, 8] = -1.0  # 1/2 → 1/3
            y2_exp = [0, 1 / 3, 1.0, 2.0, 2.0, 3.5, 3.0, 5.0, 4.0]
            D_Dl1_exp2 = Float64.([2, 3, 2, 2, 2, 2, 2, 2, 2])  # η=1.5: only row 1 has a_ii < η*dl1_ii (2 < 1.5*2)
            SLbuffer_exp2 = Float64.([-1, -1, -1, -1])

            builder = L1GSPrecBuilder(PolyesterDevice(2))
            P = builder(A2, 2; cache_strategy = PackedBufferCache())
            @test P.sweep.op.D_DL1 ≈ D_Dl1_exp2
            @test P.sweep.op.L.SLbuffer ≈ SLbuffer_exp2
            y_cpu = P \ x
            @test y_cpu ≈ y2_exp
        end

        @testset "Partsize" begin
            partsize = 3
            ncores = 2
            D_Dl1_exp = Float64.([2, 2, 2, 2, 2, 2, 2, 2, 2])  # η=1.5: all rows satisfy a_ii >= η*dl1_ii
            SLbuffer_exp = Float64.([-1, 0, -1, -1, 0, -1, -1, 0, -1])
            builder = L1GSPrecBuilder(PolyesterDevice(ncores))
            P = builder(A, partsize; cache_strategy = PackedBufferCache())
            @test P.sweep.op.D_DL1 ≈ D_Dl1_exp
            @test P.sweep.op.L.SLbuffer ≈ SLbuffer_exp
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
