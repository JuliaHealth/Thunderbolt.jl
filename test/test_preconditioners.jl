using MatrixDepot, LinearSolve, SparseArrays, SparseMatricesCSR
using KernelAbstractions
import Thunderbolt: ThreadedSparseMatrixCSR
using TimerOutputs
using LinearAlgebra: Symmetric, norm, tril, triu, diag, I
import Thunderbolt.Preconditioners: _apply_sweep!

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

function test_sym_result(
    testname,
    A,
    x,
    y_exp,
    D_Dl1_exp,
    partsize,
    sweep = ForwardSweep(),
    cache_strategy = PackedBufferCache(),
    test_buffer_fn = (P) -> nothing,
)
    @testset "$testname Symmetric" begin
        total_ncores = 8 # Assuming 8 cores for testing
        for ncores = 1:total_ncores # testing for multiple cores to check that the answer is independent of the number of cores
            builder = L1GSPrecBuilder(PolyesterDevice(ncores))
            P =
                A isa Symmetric ?
                builder(A, partsize; sweep = sweep, cache_strategy = cache_strategy) :
                builder(A, partsize; isSymA = true, sweep = sweep, cache_strategy = cache_strategy)
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

function test_l1gs_prec(
    testname,
    A,
    b,
    sweep = ForwardSweep(),
    cache_strategy = PackedBufferCache();
    ncores = 20,
    partsize = nothing,
    isSymA = false,
    solver = KrylovJL_GMRES(),
)
    @testset "$testname" begin
        ncores = ncores
        partsize = partsize === nothing ? ceil(Int, size(A, 1) / ncores) : partsize

        prob = LinearProblem(A, b)

        sol_unprec = solve(prob, solver)
        @test isapprox(A * sol_unprec.u, b, rtol = 1e-1, atol = 1e-1)
        TimerOutputs.reset_timer!()
        P = if A isa Symmetric
            L1GSPrecBuilder(PolyesterDevice(ncores))(
                A,
                partsize;
                sweep = sweep,
                cache_strategy = cache_strategy,
            )
        else
            L1GSPrecBuilder(PolyesterDevice(ncores))(
                A,
                partsize;
                sweep = sweep,
                cache_strategy = cache_strategy,
                isSymA = isSymA,
            )
        end

        println("\n" * "="^60)
        println("Preconditioner Construction Timings:")
        TimerOutputs.print_timer()
        println("="^60)

        # Reset timer before testing ldiv!
        TimerOutputs.reset_timer!()

        sol_prec = solve(prob, solver; Pl = P)

        println("\n" * "="^60)
        println("ldiv! Timings during solve:")
        TimerOutputs.print_timer()
        println("="^60)

        println("Unprec. no. iters: $(sol_unprec.iters), time: $(sol_unprec.stats.timer)")
        println("Prec. no. iters: $(sol_prec.iters), time: $(sol_prec.stats.timer)")
        @test isapprox(A * sol_prec.u, b, rtol = 1e-1, atol = 1e-1)
        @test sol_prec.iters <= sol_unprec.iters
    end
end


@testset "L1GS Preconditioner" begin
    @testset "Algorithm" begin
        N = 9
        A = poisson_test_matrix(N)
        x = 0:(N-1) |> collect .|> Float64
        y_exp_fwd = [0, 1 / 2, 1.0, 2.0, 2.0, 3.5, 3.0, 5.0, 4.0]
        y_exp_bwd = [0.25, 0.5, 1.75, 1.5, 3.25, 2.5, 4.75, 3.5, 4.0]
        y_exp_sym = [0.25, 0.5, 2.0, 2.0, 3.75, 3.5, 5.5, 5.0, 4.0]
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
        test_sym_result(
            "Packed, Forward, CPU CSC",
            A,
            x,
            y_exp_fwd,
            D_Dl1_exp,
            2,
            ForwardSweep(),
            PackedBufferCache(),
            test_fwd_buffer_fn,
        )
        B = SparseMatrixCSR(A)
        test_sym_result(
            "Packed, Forward, CPU CSR",
            B,
            x,
            y_exp_fwd,
            D_Dl1_exp,
            2,
            ForwardSweep(),
            PackedBufferCache(),
            test_fwd_buffer_fn,
        )
        test_sym_result("Packed, Forward, CPU CSR", B, x, y_exp_fwd, D_Dl1_exp, 2, ForwardSweep())
        C = ThreadedSparseMatrixCSR(B)
        test_sym_result(
            "Packed, Forward, CPU Threaded CSR",
            C,
            x,
            y_exp_fwd,
            D_Dl1_exp,
            2,
            ForwardSweep(),
            PackedBufferCache(),
            test_fwd_buffer_fn,
        )

        # Backward sweep packed buffer tests
        test_sym_result(
            "Packed, Backward, CPU CSC",
            A,
            x,
            y_exp_bwd,
            D_Dl1_exp,
            2,
            BackwardSweep(),
            PackedBufferCache(),
            test_bwd_buffer_fn,
        )
        test_sym_result(
            "Packed, Backward, CPU CSR",
            B,
            x,
            y_exp_bwd,
            D_Dl1_exp,
            2,
            BackwardSweep(),
            PackedBufferCache(),
            test_bwd_buffer_fn,
        )
        test_sym_result("Packed, Backward, CPU CSR", B, x, y_exp_bwd, D_Dl1_exp, 2, BackwardSweep())
        test_sym_result(
            "Packed, Backward, CPU Threaded CSR",
            C,
            x,
            y_exp_bwd,
            D_Dl1_exp,
            2,
            BackwardSweep(),
            PackedBufferCache(),
            test_bwd_buffer_fn,
        )

        # Symmetric sweep packed buffer tests
        test_sym_result(
            "Packed, Symmetric, CPU CSC",
            A,
            x,
            y_exp_sym,
            D_Dl1_exp,
            2,
            SymmetricSweep(),
            PackedBufferCache(),
            test_sym_buffer_fn,
        )
        test_sym_result(
            "Packed, Symmetric, CPU CSR",
            B,
            x,
            y_exp_sym,
            D_Dl1_exp,
            2,
            SymmetricSweep(),
            PackedBufferCache(),
            test_sym_buffer_fn,
        )
        test_sym_result(
            "Packed, Symmetric, CPU CSR",
            B,
            x,
            y_exp_sym,
            D_Dl1_exp,
            2,
            SymmetricSweep(),
        )
        test_sym_result(
            "Packed, Symmetric, CPU Threaded CSR",
            C,
            x,
            y_exp_sym,
            D_Dl1_exp,
            2,
            SymmetricSweep(),
            PackedBufferCache(),
            test_sym_buffer_fn,
        )

        # MatrixViewCache tests
        # Forward sweep MatrixViewCache tests
        test_sym_result(
            "MatrixView, Forward, CPU CSC",
            A,
            x,
            y_exp_fwd,
            D_Dl1_exp,
            2,
            ForwardSweep(),
            MatrixViewCache(),
        )
        test_sym_result(
            "MatrixView, Forward, CPU CSR",
            B,
            x,
            y_exp_fwd,
            D_Dl1_exp,
            2,
            ForwardSweep(),
            MatrixViewCache(),
        )
        test_sym_result(
            "MatrixView, Forward, CPU Threaded CSR",
            C,
            x,
            y_exp_fwd,
            D_Dl1_exp,
            2,
            ForwardSweep(),
            MatrixViewCache(),
        )

        # Backward sweep MatrixViewCache tests
        test_sym_result(
            "MatrixView, Backward, CPU CSC",
            A,
            x,
            y_exp_bwd,
            D_Dl1_exp,
            2,
            BackwardSweep(),
            MatrixViewCache(),
        )
        test_sym_result(
            "MatrixView, Backward, CPU CSR",
            B,
            x,
            y_exp_bwd,
            D_Dl1_exp,
            2,
            BackwardSweep(),
            MatrixViewCache(),
        )
        test_sym_result(
            "MatrixView, Backward, CPU Threaded CSR",
            C,
            x,
            y_exp_bwd,
            D_Dl1_exp,
            2,
            BackwardSweep(),
            MatrixViewCache(),
        )

        # Symmetric sweep MatrixViewCache tests
        test_sym_result(
            "MatrixView, Symmetric, CPU CSC",
            A,
            x,
            y_exp_sym,
            D_Dl1_exp,
            2,
            SymmetricSweep(),
            MatrixViewCache(),
        )
        test_sym_result(
            "MatrixView, Symmetric, CPU CSR",
            B,
            x,
            y_exp_sym,
            D_Dl1_exp,
            2,
            SymmetricSweep(),
            MatrixViewCache(),
        )
        test_sym_result(
            "MatrixView, Symmetric, CPU Threaded CSR",
            C,
            x,
            y_exp_sym,
            D_Dl1_exp,
            2,
            SymmetricSweep(),
            MatrixViewCache(),
        )

        @testset "η parameter" begin
            # Test with η = 2.0 (more strict than default 1.5)
            # For Poisson matrix: a_ii = 2, dl1_ii = 1 for all rows
            # Since a_ii = 2 >= η*dl1_ii = 2*1 = 2, all rows satisfy condition
            # Therefore dl1star_ii = 0 for all rows, D_Dl1 = a_ii = 2
            η = 2.0
            D_Dl1_exp = Float64.([2, 2, 2, 2, 2, 2, 2, 2, 2])
            SLbuffer_exp = Float64.([-1, -1, -1, -1])
            builder = L1GSPrecBuilder(PolyesterDevice(2))
            P = builder(A, 2; η = η, sweep = ForwardSweep(), cache_strategy = PackedBufferCache())
            @test P.sweep.op.D_DL1 ≈ D_Dl1_exp
            @test P.sweep.op.L.SLbuffer ≈ SLbuffer_exp

            # Test with η = 3.0 (very strict)
            # a_ii = 2 < η*dl1_ii = 3*1 = 3, condition NOT satisfied
            # Therefore dl1star_ii = dl1_ii/2 = 1/2 = 0.5
            # D_Dl1 = a_ii + dl1star_ii = 2 + 0.5 = 2.5
            η = 3.0
            D_Dl1_exp = Float64.([2.0, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5])
            P = builder(A, 2; η = η, sweep = ForwardSweep(), cache_strategy = PackedBufferCache())
            @test P.sweep.op.D_DL1 ≈ D_Dl1_exp
            @test P.sweep.op.L.SLbuffer ≈ SLbuffer_exp

            # Test with η = 1.0 (less strict than default)
            # a_ii = 2 >= η*dl1_ii = 1*1 = 1, condition satisfied
            # Therefore dl1star_ii = 0
            # D_Dl1 = a_ii = 2
            η = 1.0
            D_Dl1_exp = Float64.([2, 2, 2, 2, 2, 2, 2, 2, 2])
            P = builder(A, 2; η = η, sweep = ForwardSweep(), cache_strategy = PackedBufferCache())
            @test P.sweep.op.D_DL1 ≈ D_Dl1_exp
            @test P.sweep.op.L.SLbuffer ≈ SLbuffer_exp

            # Verify preconditioner still works correctly with different η
            y_exp = [0, 1 / 2, 1.0, 2.0, 2.0, 3.5, 3.0, 5.0, 4.0]
            η = 2.0
            P = builder(A, 2; η = η, sweep = ForwardSweep(), cache_strategy = PackedBufferCache())
            y = P \ x
            @test y ≈ y_exp
        end


        @testset "Non-Symmetric CSC" begin
            A2 = copy(A)
            A2[1, 8] = -1.0  # won't affect the result
            A2[2, 8] = -1.0  # 1/2 → 1/3
            y2_fwd_exp = [0, 1.0 / 3.0, 1.0, 2.0, 2.0, 3.5, 3.0, 5.0, 4.0]
            y2_bwd_exp = [1.0 / 6.0, 1.0 / 3.0, 1.75, 1.5, 3.25, 2.5, 4.75, 3.5, 4.0] # since backward sweep, then change in the second element propagtes to the first element as well
            D_Dl1_exp2 = Float64.([2, 3, 2, 2, 2, 2, 2, 2, 2])  # η=1.5: only row 1 has a_ii < η*dl1_ii (2 < 1.5*2)
            SLbuffer_exp2 = Float64.([-1, -1, -1, -1])
            SUbuffer_exp2 = Float64.([-1, -1, -1, -1])

            builder = L1GSPrecBuilder(PolyesterDevice(2))

            # Forward sweep with PackedBufferCache
            P = builder(A2, 2; sweep = ForwardSweep(), cache_strategy = PackedBufferCache())
            @test P.sweep.op.D_DL1 ≈ D_Dl1_exp2
            @test P.sweep.op.L.SLbuffer ≈ SLbuffer_exp2
            @test P \ x ≈ y2_fwd_exp

            # Forward sweep with MatrixViewCache
            P = builder(A2, 2; sweep = ForwardSweep(), cache_strategy = MatrixViewCache())
            @test P.sweep.op.D_DL1 ≈ D_Dl1_exp2
            @test P \ x ≈ y2_fwd_exp

            # Backward sweep with PackedBufferCache
            P = builder(A2, 2; sweep = BackwardSweep(), cache_strategy = PackedBufferCache())
            @test P.sweep.op.D_DL1 ≈ D_Dl1_exp2
            @test P.sweep.op.U.SUbuffer ≈ SUbuffer_exp2
            @test P \ x ≈ y2_bwd_exp

            # Backward sweep with MatrixViewCache
            P = builder(A2, 2; sweep = BackwardSweep(), cache_strategy = MatrixViewCache())
            @test P.sweep.op.D_DL1 ≈ D_Dl1_exp2
            @test P \ x ≈ y2_bwd_exp
        end

        @testset "Partsize" begin
            partsize = 3
            ncores = 2
            D_Dl1_exp = Float64.([2, 2, 2, 2, 2, 2, 2, 2, 2])  # η=1.5: all rows satisfy a_ii >= η*dl1_ii
            SLbuffer_exp = Float64.([-1, 0, -1, -1, 0, -1, -1, 0, -1])
            builder = L1GSPrecBuilder(PolyesterDevice(ncores))
            P = builder(A, partsize; sweep = ForwardSweep(), cache_strategy = PackedBufferCache())
            @test P.sweep.op.D_DL1 ≈ D_Dl1_exp
            @test P.sweep.op.L.SLbuffer ≈ SLbuffer_exp
        end
    end
    @testset "Solution with LinearSolve" begin

        @testset "Non-Symmetric A (HB/sherman5)" begin
            md = mdopen("HB/sherman5")
            A = md.A
            b = md.b[:, 1]
            # Forward and Backward sweeps with both cache strategies
            test_l1gs_prec(
                "PackedBuffer, ForwardSweep HB/sherman5",
                A,
                b,
                ForwardSweep(),
                PackedBufferCache(),
            )
            test_l1gs_prec(
                "MatrixView, ForwardSweep HB/sherman5",
                A,
                b,
                ForwardSweep(),
                MatrixViewCache(),
            )
            test_l1gs_prec(
                "PackedBuffer, BackwardSweep HB/sherman5",
                A,
                b,
                BackwardSweep(),
                PackedBufferCache(),
            )
            test_l1gs_prec(
                "MatrixView, BackwardSweep HB/sherman5",
                A,
                b,
                BackwardSweep(),
                MatrixViewCache(),
            )
        end

        @testset "Symmetric A" begin
            @testset "Wathen Matrix" begin
                # md = mdopen("HB/bcsstk15")
                # A = md.A
                A = matrixdepot("wathen", 120)
                b = ones(size(A, 1))

                test_l1gs_prec(
                    "PackedBuffer, ForwardSweep wathen",
                    A,
                    b,
                    ForwardSweep(),
                    PackedBufferCache();
                    isSymA = true,
                    partsize = 10,
                    solver = KrylovJL_CG(),
                )
                test_l1gs_prec(
                    "MatrixView,  ForwardSweep wathen",
                    A,
                    b,
                    ForwardSweep(),
                    MatrixViewCache();
                    isSymA = true,
                    partsize = 10,
                    solver = KrylovJL_CG(),
                )

                test_l1gs_prec(
                    "PackedBuffer, BackwardSweep wathen",
                    A,
                    b,
                    BackwardSweep(),
                    PackedBufferCache();
                    isSymA = true,
                    partsize = 10,
                    solver = KrylovJL_CG(),
                )
                test_l1gs_prec(
                    "MatrixView,  BackwardSweep wathen",
                    A,
                    b,
                    BackwardSweep(),
                    MatrixViewCache();
                    isSymA = true,
                    partsize = 10,
                    solver = KrylovJL_CG(),
                )

                test_l1gs_prec(
                    "PackedBuffer, SymmetricSweep wathen",
                    A,
                    b,
                    SymmetricSweep(),
                    PackedBufferCache();
                    isSymA = true,
                    partsize = 10,
                    solver = KrylovJL_CG(),
                )
                test_l1gs_prec(
                    "MatrixView,  SymmetricSweep wathen",
                    A,
                    b,
                    SymmetricSweep(),
                    MatrixViewCache();
                    isSymA = true,
                    partsize = 10,
                    solver = KrylovJL_CG(),
                )
            end

            @testset "HB/bcsstk18 (Symmetric type)" begin
                md = mdopen("HB/bcsstk10")
                A = md.A
                b = ones(size(A, 1))

                test_l1gs_prec(
                    "MatrixViewCache, ForwardSweep bcsstk10",
                    A,
                    b,
                    ForwardSweep(),
                    MatrixViewCache();
                )
                test_l1gs_prec(
                    "PackedBufferCache, ForwardSweep bcsstk10",
                    A,
                    b,
                    ForwardSweep(),
                    PackedBufferCache();
                    partsize = 10,
                )

                test_l1gs_prec(
                    "MatrixViewCache, BackwardSweep bcsstk10",
                    A,
                    b,
                    BackwardSweep(),
                    MatrixViewCache();
                )
                test_l1gs_prec(
                    "PackedBufferCache, BackwardSweep bcsstk10",
                    A,
                    b,
                    BackwardSweep(),
                    PackedBufferCache();
                    partsize = 10,
                )

                test_l1gs_prec(
                    "MatrixViewCache, SymmetricSweep bcsstk10",
                    A,
                    b,
                    SymmetricSweep(),
                    MatrixViewCache();
                )
                test_l1gs_prec(
                    "PackedBufferCache, SymmetricSweep bcsstk10",
                    A,
                    b,
                    SymmetricSweep(),
                    PackedBufferCache();
                    partsize = 10,
                )
            end
        end
    end
    @testset "Sanity Checks" begin
        # Additional sanity checks can be added here
        @testset "Dense GS Sanity Check" begin
            # Dense Gauss-Seidel sanity check to verify sweep convergence properties:
            # 1. Forward and Backward sweeps should converge in the same number of iterations
            # 2. Symmetric sweep should converge in approximately half the iterations

            function poisson_dense(N)
                A = zeros(N, N)
                for i = 1:N
                    A[i, i] = 2.0
                    i > 1 && (A[i, i-1] = -1.0)
                    i < N && (A[i, i+1] = -1.0)
                end
                return A
            end

            # Forward sweep: for i = 1:n, x[i] = (b[i] - sum_{j!=i} A[i,j]*x[j]) / A[i,i]
            function gs_forward_sweep!(x, A, b)
                n = length(x)
                for i = 1:n
                    σ = 0.0
                    for j = 1:n
                        j != i && (σ += A[i, j] * x[j])
                    end
                    x[i] = (b[i] - σ) / A[i, i]
                end
            end

            # Backward sweep: same but iterate i from n to 1
            function gs_backward_sweep!(x, A, b)
                n = length(x)
                for i = n:-1:1
                    σ = 0.0
                    for j = 1:n
                        j != i && (σ += A[i, j] * x[j])
                    end
                    x[i] = (b[i] - σ) / A[i, i]
                end
            end

            # Symmetric sweep: forward then backward
            function gs_symmetric_sweep!(x, A, b)
                gs_forward_sweep!(x, A, b)
                gs_backward_sweep!(x, A, b)
            end

            # Solve Ax=b using GS as standalone solver
            function gs_solve(A, b, sweep_fn!; tol = 1e-10, maxiter = 10000)
                x = zeros(length(b))
                for iter = 1:maxiter
                    sweep_fn!(x, A, b)
                    residual = norm(A * x - b)
                    if residual < tol
                        return x, iter
                    end
                end
                return x, maxiter
            end

            N = 50
            A = poisson_dense(N)
            b = ones(N)

            _, iters_fwd = gs_solve(A, b, gs_forward_sweep!)
            _, iters_bwd = gs_solve(A, b, gs_backward_sweep!)
            _, iters_sym = gs_solve(A, b, gs_symmetric_sweep!)

            println(
                "Dense GS iterations: Forward=$iters_fwd, Backward=$iters_bwd, Symmetric=$iters_sym",
            )
            println("Ratio Forward/Symmetric = $(round(iters_fwd / iters_sym, digits=2))")

            # Forward and backward should converge in the same number of iterations
            @test iters_fwd == iters_bwd

            # Symmetric should converge in approximately half the iterations (within 1%)
            @test abs(iters_sym - iters_fwd ÷ 2) <= iters_fwd ÷ 100
        end

        @testset "Sparse L1GS Sweep Sanity Check" begin
            # Test L1GS implementation using internal _apply_sweep! as a GS solver
            # This verifies that the sparse implementation matches the dense reference
            N = 50
            A = poisson_test_matrix(N)
            b = ones(N)

            # Extract L and U for computing b - U*x and b - L*x
            D = spdiagm(0 => diag(A))
            L = tril(A, -1)  # strict lower triangular
            U = triu(A, 1)   # strict upper triangular

            # Build L1GS preconditioners with partsize=N (full matrix, standard GS)
            builder = L1GSPrecBuilder(PolyesterDevice(1))
            P_fwd = builder(
                A,
                N;
                isSymA = true,
                sweep = ForwardSweep(),
                cache_strategy = MatrixViewCache(),
            )
            P_bwd = builder(
                A,
                N;
                isSymA = true,
                sweep = BackwardSweep(),
                cache_strategy = MatrixViewCache(),
            )
            P_sym = builder(
                A,
                N;
                isSymA = true,
                sweep = SymmetricSweep(),
                cache_strategy = MatrixViewCache(),
            )

            tol = 1e-10
            maxiter = 10000

            # Forward GS iteration: x^{k+1} = (D+L)^{-1} * (b - U*x^k)
            x_fwd = zeros(N)
            iters_fwd = maxiter
            for iter = 1:maxiter
                rhs = b - U * x_fwd
                x_fwd .= rhs
                _apply_sweep!(x_fwd, P_fwd.partitioning, P_fwd.sweep)
                if norm(A * x_fwd - b) < tol
                    iters_fwd = iter
                    break
                end
            end

            # Backward GS iteration: x^{k+1} = (D+U)^{-1} * (b - L*x^k)
            x_bwd = zeros(N)
            iters_bwd = maxiter
            for iter = 1:maxiter
                rhs = b - L * x_bwd
                x_bwd .= rhs
                _apply_sweep!(x_bwd, P_bwd.partitioning, P_bwd.sweep)
                if norm(A * x_bwd - b) < tol
                    iters_bwd = iter
                    break
                end
            end

            # Symmetric GS iteration: forward half-step then backward half-step
            x_sym = zeros(N)
            iters_sym = maxiter
            for iter = 1:maxiter
                # Forward: x_half = (D+L)^{-1} * (b - U*x)
                rhs_fwd = b - U * x_sym
                x_half = copy(rhs_fwd)
                _apply_sweep!(x_half, P_fwd.partitioning, P_fwd.sweep)

                # Backward: x_new = (D+U)^{-1} * (b - L*x_half)
                rhs_bwd = b - L * x_half
                x_sym .= rhs_bwd
                _apply_sweep!(x_sym, P_bwd.partitioning, P_bwd.sweep)

                if norm(A * x_sym - b) < tol
                    iters_sym = iter
                    break
                end
            end

            println(
                "Sparse L1GS iterations: Forward=$iters_fwd, Backward=$iters_bwd, Symmetric=$iters_sym",
            )
            println("Ratio Forward/Symmetric = $(round(iters_fwd / iters_sym, digits=2))")

            # Forward and backward should converge in the same number of iterations
            @test iters_fwd == iters_bwd

            # Symmetric should converge in approximately half the iterations (within 1%)
            @test abs(iters_sym - iters_fwd ÷ 2) <= iters_fwd ÷ 100
        end
    end
end
