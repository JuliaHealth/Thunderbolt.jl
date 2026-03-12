using Test
using SparseArrays
using LinearSolve
using SparseMatricesCSR
using KernelAbstractions
using MatrixDepot
using Thunderbolt, CUDA
using LinearAlgebra: Symmetric
import Thunderbolt: CudaDevice

##########################################
## L1 Gauss Seidel Preconditioner - GPU ##
##########################################

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
    D_DL1_exp,
    partsize,
    sweep = ForwardSweep(),
    cache_strategy = PackedBufferCache(),
    test_buffer_fn = (P) -> nothing,
)
    @testset "$testname Symmetric" begin
        total_nblocks = 10
        total_nthreads = 10
        for nblocks = 1:total_nblocks # testing for multiple `nblocks` and `nthreads` to check that the answer is independent of the config.
            for nthreads = 1:total_nthreads
                builder = L1GSPrecBuilder(CudaDevice(nblocks, nthreads))
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
                    @test Vector(P.sweep.lop.D_DL1) ≈ D_DL1_exp
                    @test Vector(P.sweep.uop.D_DL1) ≈ D_DL1_exp
                else
                    @test Vector(P.sweep.op.D_DL1) ≈ D_DL1_exp
                end
                test_buffer_fn(P)
                y = P \ x
                @test y ≈ y_exp
            end
        end
    end
end

function test_l1gs_prec(
    testname,
    A,
    b,
    sweep = ForwardSweep(),
    cache_strategy = PackedBufferCache();
    nblocks = 20,
    nthreads = 256,
    partsize = nothing,
    isSymA = false,
    solver = KrylovJL_GMRES(),
)
    @testset "$testname" begin
        partsize = partsize === nothing ? ceil(Int, size(A, 1) / (nblocks * nthreads)) : partsize

        prob = LinearProblem(A, b)

        sol_unprec = solve(prob, solver)
        @test isapprox(A * sol_unprec.u, b, rtol = 1e-1, atol = 1e-1)

        P = if A isa Symmetric
            L1GSPrecBuilder(CudaDevice(nblocks, nthreads))(
                A,
                partsize;
                sweep = sweep,
                cache_strategy = cache_strategy,
            )
        else
            L1GSPrecBuilder(CudaDevice(nblocks, nthreads))(
                A,
                partsize;
                sweep = sweep,
                cache_strategy = cache_strategy,
                isSymA = isSymA,
            )
        end

        sol_prec = solve(prob, solver; Pl = P)

        println("Unprec. no. iters: $(sol_unprec.iters), time: $(sol_unprec.stats.timer)")
        println("Prec. no. iters: $(sol_prec.iters), time: $(sol_prec.stats.timer)")
        @test isapprox(A * sol_prec.u, b, rtol = 1e-1, atol = 1e-1)
        @test sol_prec.iters <= sol_unprec.iters
    end
end


@testset "L1GS Preconditioner - GPU" begin
    @testset "Algorithm" begin
        N = 9
        A = poisson_test_matrix(N)
        x = 0:(N-1) |> collect .|> Float64
        y_exp_fwd = [0, 1 / 2, 1.0, 2.0, 2.0, 3.5, 3.0, 5.0, 4.0]
        y_exp_bwd = [0.25, 0.5, 1.75, 1.5, 3.25, 2.5, 4.75, 3.5, 4.0]
        y_exp_sym = [0.25, 0.5, 2.0, 2.0, 3.75, 3.5, 5.5, 5.0, 4.0]
        D_DL1_exp = Float64.([2, 2, 2, 2, 2, 2, 2, 2, 2])  # η=1.5: all rows satisfy a_ii >= η*dl1_ii (2 >= 1.5*1)
        SLbuffer_exp = Float64.([-1, -1, -1, -1])
        SUbuffer_exp = Float64.([-1, -1, -1, -1])

        # Packed buffer tests #
        test_fwd_buffer_fn = (P) -> @test Vector(P.sweep.op.L.SLbuffer) ≈ SLbuffer_exp
        test_bwd_buffer_fn = (P) -> @test Vector(P.sweep.op.U.SUbuffer) ≈ SUbuffer_exp
        test_sym_buffer_fn = (P) -> begin
            @test Vector(P.sweep.lop.L.SLbuffer) ≈ SLbuffer_exp
            @test Vector(P.sweep.uop.U.SUbuffer) ≈ SUbuffer_exp
        end

        # Forward sweep packed buffer tests
        test_sym_result(
            "Packed, Forward, GPU CSC",
            A,
            x,
            y_exp_fwd,
            D_DL1_exp,
            2,
            ForwardSweep(),
            PackedBufferCache(),
            test_fwd_buffer_fn,
        )
        B = SparseMatrixCSR(A)
        test_sym_result(
            "Packed, Forward, GPU CSR",
            B,
            x,
            y_exp_fwd,
            D_DL1_exp,
            2,
            ForwardSweep(),
            PackedBufferCache(),
            test_fwd_buffer_fn,
        )

        # Backward sweep packed buffer tests
        test_sym_result(
            "Packed, Backward, GPU CSC",
            A,
            x,
            y_exp_bwd,
            D_DL1_exp,
            2,
            BackwardSweep(),
            PackedBufferCache(),
            test_bwd_buffer_fn,
        )
        test_sym_result(
            "Packed, Backward, GPU CSR",
            B,
            x,
            y_exp_bwd,
            D_DL1_exp,
            2,
            BackwardSweep(),
            PackedBufferCache(),
            test_bwd_buffer_fn,
        )

        # Symmetric sweep packed buffer tests
        test_sym_result(
            "Packed, Symmetric, GPU CSC",
            A,
            x,
            y_exp_sym,
            D_DL1_exp,
            2,
            SymmetricSweep(),
            PackedBufferCache(),
            test_sym_buffer_fn,
        )
        test_sym_result(
            "Packed, Symmetric, GPU CSR",
            B,
            x,
            y_exp_sym,
            D_DL1_exp,
            2,
            SymmetricSweep(),
            PackedBufferCache(),
            test_sym_buffer_fn,
        )

        # MatrixViewCache tests
        # Forward sweep MatrixViewCache tests
        test_sym_result(
            "MatrixView, Forward, GPU CSC",
            A,
            x,
            y_exp_fwd,
            D_DL1_exp,
            2,
            ForwardSweep(),
            MatrixViewCache(),
        )
        test_sym_result(
            "MatrixView, Forward, GPU CSR",
            B,
            x,
            y_exp_fwd,
            D_DL1_exp,
            2,
            ForwardSweep(),
            MatrixViewCache(),
        )

        # Backward sweep MatrixViewCache tests
        test_sym_result(
            "MatrixView, Backward, GPU CSC",
            A,
            x,
            y_exp_bwd,
            D_DL1_exp,
            2,
            BackwardSweep(),
            MatrixViewCache(),
        )
        test_sym_result(
            "MatrixView, Backward, GPU CSR",
            B,
            x,
            y_exp_bwd,
            D_DL1_exp,
            2,
            BackwardSweep(),
            MatrixViewCache(),
        )

        # Symmetric sweep MatrixViewCache tests
        test_sym_result(
            "MatrixView, Symmetric, GPU CSC",
            A,
            x,
            y_exp_sym,
            D_DL1_exp,
            2,
            SymmetricSweep(),
            MatrixViewCache(),
        )
        test_sym_result(
            "MatrixView, Symmetric, GPU CSR",
            B,
            x,
            y_exp_sym,
            D_DL1_exp,
            2,
            SymmetricSweep(),
            MatrixViewCache(),
        )

        @testset "η parameter" begin
            # Test with η = 2.0 (more strict than default 1.5)
            # For Poisson matrix: a_ii = 2, dl1_ii = 1 for all rows
            # Since a_ii = 2 >= η*dl1_ii = 2*1 = 2, all rows satisfy condition
            # Therefore dl1star_ii = 0 for all rows, D_DL1 = a_ii = 2
            η = 2.0
            D_DL1_exp = Float64.([2, 2, 2, 2, 2, 2, 2, 2, 2])
            SLbuffer_exp = Float64.([-1, -1, -1, -1])
            builder = L1GSPrecBuilder(CudaDevice(2, 2))
            P = builder(A, 2; η = η, sweep = ForwardSweep(), cache_strategy = PackedBufferCache())
            @test Vector(P.sweep.op.D_DL1) ≈ D_DL1_exp
            @test Vector(P.sweep.op.L.SLbuffer) ≈ SLbuffer_exp

            # Test with η = 3.0 (very strict)
            # a_ii = 2 < η*dl1_ii = 3*1 = 3, condition NOT satisfied
            # Therefore dl1star_ii = dl1_ii/2 = 1/2 = 0.5
            # D_DL1 = a_ii + dl1star_ii = 2 + 0.5 = 2.5
            η = 3.0
            D_DL1_exp = Float64.([2.0, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5])
            P = builder(A, 2; η = η, sweep = ForwardSweep(), cache_strategy = PackedBufferCache())
            @test Vector(P.sweep.op.D_DL1) ≈ D_DL1_exp
            @test Vector(P.sweep.op.L.SLbuffer) ≈ SLbuffer_exp

            # Test with η = 1.0 (less strict than default)
            # a_ii = 2 >= η*dl1_ii = 1*1 = 1, condition satisfied
            # Therefore dl1star_ii = 0
            # D_DL1 = a_ii = 2
            η = 1.0
            D_DL1_exp = Float64.([2, 2, 2, 2, 2, 2, 2, 2, 2])
            P = builder(A, 2; η = η, sweep = ForwardSweep(), cache_strategy = PackedBufferCache())
            @test Vector(P.sweep.op.D_DL1) ≈ D_DL1_exp
            @test Vector(P.sweep.op.L.SLbuffer) ≈ SLbuffer_exp

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
            y2_bwd_exp = [1.0 / 6.0, 1.0 / 3.0, 1.75, 1.5, 3.25, 2.5, 4.75, 3.5, 4.0]
            D_DL1_exp2 = Float64.([2, 3, 2, 2, 2, 2, 2, 2, 2])  # η=1.5: only row 1 has a_ii < η*dl1_ii (2 < 1.5*2)
            SLbuffer_exp2 = Float64.([-1, -1, -1, -1])
            SUbuffer_exp2 = Float64.([-1, -1, -1, -1])

            builder = L1GSPrecBuilder(CudaDevice(2, 2))

            # Forward sweep with PackedBufferCache
            P = builder(A2, 2; sweep = ForwardSweep(), cache_strategy = PackedBufferCache())
            @test Vector(P.sweep.op.D_DL1) ≈ D_DL1_exp2
            @test Vector(P.sweep.op.L.SLbuffer) ≈ SLbuffer_exp2
            @test P \ x ≈ y2_fwd_exp

            # Forward sweep with MatrixViewCache
            P = builder(A2, 2; sweep = ForwardSweep(), cache_strategy = MatrixViewCache())
            @test Vector(P.sweep.op.D_DL1) ≈ D_DL1_exp2
            @test P \ x ≈ y2_fwd_exp

            # Backward sweep with PackedBufferCache
            P = builder(A2, 2; sweep = BackwardSweep(), cache_strategy = PackedBufferCache())
            @test Vector(P.sweep.op.D_DL1) ≈ D_DL1_exp2
            @test Vector(P.sweep.op.U.SUbuffer) ≈ SUbuffer_exp2
            @test P \ x ≈ y2_bwd_exp

            # Backward sweep with MatrixViewCache
            P = builder(A2, 2; sweep = BackwardSweep(), cache_strategy = MatrixViewCache())
            @test Vector(P.sweep.op.D_DL1) ≈ D_DL1_exp2
            @test P \ x ≈ y2_bwd_exp
        end

        @testset "Partsize" begin
            partsize = 3
            D_DL1_exp = Float64.([2, 2, 2, 2, 2, 2, 2, 2, 2])  # η=1.5: all rows satisfy a_ii >= η*dl1_ii
            SLbuffer_exp = Float64.([-1, 0, -1, -1, 0, -1, -1, 0, -1])
            builder = L1GSPrecBuilder(CudaDevice(2, 2))
            P = builder(
                A,
                partsize;
                isSymA = true,
                sweep = ForwardSweep(),
                cache_strategy = PackedBufferCache(),
            )
            @test Vector(P.sweep.op.D_DL1) ≈ D_DL1_exp
            @test Vector(P.sweep.op.L.SLbuffer) ≈ SLbuffer_exp
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
                    "MatrixView, ForwardSweep wathen",
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
                    "MatrixView, BackwardSweep wathen",
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
                    "MatrixView, SymmetricSweep wathen",
                    A,
                    b,
                    SymmetricSweep(),
                    MatrixViewCache();
                    isSymA = true,
                    partsize = 10,
                    solver = KrylovJL_CG(),
                )
            end

            @testset "HB/bcsstk10 (Symmetric type)" begin
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
                    partsize = 10,
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
end
