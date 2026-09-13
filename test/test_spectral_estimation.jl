using Thunderbolt
using Test
using LinearAlgebra
using Random
using SparseArrays
using Tensors

import Thunderbolt:
    SpectralRadiusWorkspace,
    estimate_rho!,
    _gershgorin_bound,
    _should_reestimate,
    LumpedMassRateOperator,
    mul_rate!,
    add_source_rate!,
    compute_lumped_inverse_mass!,
    BilinearMassIntegrator,
    BilinearDiffusionIntegrator,
    setup_operator,
    update_operator!,
    AssemblyStrategy,
    SequentialCPUDevice,
    TimeIntegrationContext
import FerriteOperators

# A callable struct rather than a closure over `A`, so the allocation tests below measure
# `estimate_rho!`/`mul_rate!` themselves rather than a boxed capture (same reasoning as
# `LinearDecay` in test_sts_families.jl).
struct DenseApply{MT}
    A::MT
end
(f::DenseApply)(w, v) = mul!(w, f.A, v)

# A random negative-definite matrix: `-(RᵀR) - I` has eigenvalues ≤ -1 for any `R`, so the
# dominant-magnitude eigenvalue is real and power iteration applies directly.
random_negdef(rng, ::Type{T}, n) where {T} = (R = randn(rng, T, n, n); -(R' * R) - I)

@testset "estimate_rho! vs eigen: dense negative-definite matrices" begin
    # Tight settings, isolating whether the power iteration itself converges to the right
    # dominant eigenvalue -- independent of any particular random draw's eigenvalue gap (the
    # loose *default* reltol=1e-2 is deliberately gap-sensitive; that is covered separately
    # below on a matrix with a known, healthy gap).
    rng = MersenneTwister(20260901)
    for n in (5, 20), trial in 1:3
        A = random_negdef(rng, Float64, n)
        λmax_abs = maximum(abs, eigvals(A))

        ws = SpectralRadiusWorkspace(zeros(n))
        ρ = estimate_rho!(ws, DenseApply(A); maxiters = 20_000, reltol = 1.0e-12, safety = 1.0)

        @test ρ ≈ λmax_abs rtol = 1.0e-4
        @test ws.ρ == ρ
    end
end

@testset "estimate_rho! with default settings on a well-separated spectrum" begin
    # A well-separated spectrum (ratio 1:10:100) so the *default* reltol=1e-2 stopping rule
    # -- which reads only the last step size, not the true remaining error -- is not misled by
    # a slow-converging near-degenerate pair the way an arbitrary random draw could.
    A = Diagonal([-1.0, -10.0, -100.0])
    λmax_abs = 100.0

    ws = SpectralRadiusWorkspace(zeros(3))
    ρ = estimate_rho!(ws, DenseApply(A))

    @test ρ ≈ 1.1 * λmax_abs rtol = 0.05
end

@testset "estimate_rho!: warm start converges faster on a repeated operator" begin
    rng = MersenneTwister(1)
    A = random_negdef(rng, Float64, 30)
    ws = SpectralRadiusWorkspace(zeros(30))
    estimate_rho!(ws, DenseApply(A); maxiters = 200, safety = 1.0)
    first_iters = ws.iters_done
    estimate_rho!(ws, DenseApply(A); maxiters = 200, safety = 1.0)
    # Warm-started from the already-converged direction: at most as many iterations again.
    @test ws.iters_done ≤ first_iters
end

@testset "estimate_rho!: runaway guard" begin
    # `apply!` that ignores `v` entirely and always returns the same (bad) result -- a stand-in for
    # an operator evaluated somewhere it should not have been trusted, where reseeding the iterate
    # and retrying cannot possibly help either.
    struct ConstantApply{T}
        value::T
        calls::Ref{Int}
    end
    ConstantApply(value) = ConstantApply(value, Ref(0))
    function (f::ConstantApply)(w, v)
        f.calls[] += 1
        fill!(w, f.value)
        return w
    end

    @testset "non-finite result retries once, then errors" for badvalue in (Inf, NaN)
        ws = SpectralRadiusWorkspace(zeros(5))
        bad_apply! = ConstantApply(badvalue)
        e = @test_throws ErrorException estimate_rho!(ws, bad_apply!)
        @test occursin("non-finite", e.value.msg)
        @test occursin("retrying once", e.value.msg)
        @test bad_apply!.calls[] == 2 # the attempt, then the retry -- both fail on their first iterate
    end

    @testset "a finite but absurd jump vs. the previous estimate also triggers the guard" begin
        ws = SpectralRadiusWorkspace(zeros(3))
        ws.ρ = 1.0 # a modest "previous" estimate to jump away from
        huge_apply! = ConstantApply(1.0e30)
        e = @test_throws ErrorException estimate_rho!(ws, huge_apply!)
        @test occursin("jump", e.value.msg)
        @test occursin("retrying once", e.value.msg)
    end

    @testset "describe context reaches the error" begin
        ws = SpectralRadiusWorkspace(zeros(4))
        e = @test_throws ErrorException estimate_rho!(
            ws,
            ConstantApply(Inf);
            describe = () -> " EMRKC-specific context.",
        )
        @test occursin("EMRKC-specific context.", e.value.msg)
    end

    @testset "a benign estimator is unaffected" begin
        # The guard sits on the same success path every other testset in this file already
        # exercises (dense negative-definite matrices, the FE heat problem, warm start, Float32):
        # those passing unchanged is the regression check that ordinary use never retries.
        A = Diagonal([-1.0, -10.0, -100.0])
        ws = SpectralRadiusWorkspace(zeros(3))
        ρ = estimate_rho!(ws, DenseApply(A))
        @test ρ ≈ 1.1 * 100.0 rtol = 0.05
    end
end

# A tiny FE heat problem, assembled exactly as `Thunderbolt._assemble_laplacian` does: a
# bilinear integrator handed to `setup_operator`, assembled by `update_operator!`, read back
# through FerriteOperators' documented `get_matrix` accessor.
function assemble_heat_operators(n = 4)
    grid = generate_grid(Quadrilateral, (n, n))
    dh   = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral, 1}())
    close!(dh)

    qrc      = QuadratureRuleCollection(2)
    strategy = AssemblyStrategy(SequentialCPUDevice())
    ctx      = TimeIntegrationContext(0.0, 0.0, 0.0)

    Mop = setup_operator(strategy, BilinearMassIntegrator(ConstantCoefficient(1.0), qrc, :u), dh)
    Kop = setup_operator(
        strategy,
        BilinearDiffusionIntegrator(ConstantCoefficient(one(Tensor{2, 2})), qrc, :u),
        dh,
    )
    update_operator!(Mop, nothing, ctx)
    update_operator!(Kop, nothing, ctx)
    return Mop, Kop, ndofs(dh)
end

@testset "Real FE case: heat problem row-sum lumping + ρ_F" begin
    Mop, Kop, n = assemble_heat_operators()
    M = FerriteOperators.get_matrix(Mop)
    K = FerriteOperators.get_matrix(Kop)

    invM = zeros(n)
    ones_tmp = zeros(n)
    compute_lumped_inverse_mass!(invM, Mop, ones_tmp)

    # (a) invM is the row-sum lumped inverse mass of the assembled mass matrix.
    @test invM ≈ 1 ./ vec(sum(M, dims = 2))

    dense_rate = Diagonal(invM) * Matrix(K)
    λmax_abs   = maximum(abs, eigvals(dense_rate))

    # (b) estimate_rho! over mul_rate! agrees with eigen of the dense rate matrix (safety = 1
    # here isolates the power-iteration estimate itself from the safety margin, which the
    # dense-matrix testset above already covers).
    op = LumpedMassRateOperator(Kop, invM)
    ws = SpectralRadiusWorkspace(zeros(n))
    ρ = estimate_rho!(ws, (w, v) -> mul_rate!(w, op, v); maxiters = 300, safety = 1.0)
    @test ρ ≈ λmax_abs rtol = 0.05

    # (c) the Gershgorin bound is a genuine upper bound on the same spectral radius.
    @test _gershgorin_bound(K, invM) ≥ λmax_abs
end

@testset "compute_lumped_inverse_mass!: row-sum positivity guard" begin
    invM, ones_tmp = zeros(3), zeros(3)

    Mbad = Matrix(Diagonal([1.0, 0.0, 2.0]))
    @test_throws ErrorException compute_lumped_inverse_mass!(invM, Mbad, ones_tmp)

    Mgood = Matrix(Diagonal([1.0, 2.0, 4.0]))
    compute_lumped_inverse_mass!(invM, Mgood, ones_tmp)
    @test invM ≈ [1.0, 0.5, 0.25]
end

@testset "mul_rate! / add_source_rate!" begin
    K = sparse(Diagonal([2.0, -4.0, 6.0]))
    invM = [1.0, 0.5, 1.0 / 3.0]
    op = LumpedMassRateOperator(K, invM)

    y = zeros(3)
    x = [1.0, 1.0, 1.0]
    mul_rate!(y, op, x)
    @test y ≈ [2.0, -2.0, 2.0]

    add_source_rate!(y, op, [1.0, 2.0, 3.0])
    @test y ≈ [2.0 + 1.0, -2.0 + 1.0, 2.0 + 1.0]
end

@testset "_should_reestimate policy shapes" begin
    @testset ":once" begin
        @test _should_reestimate(:once, -1, false) == true    # never yet
        @test _should_reestimate(:once, 0, false) == false
        @test _should_reestimate(:once, 1_000, false) == false
        @test _should_reestimate(:once, 0, true) == true       # stepfail forces regardless
        @test_throws ErrorException _should_reestimate(:always, 0, false)
    end

    @testset "n::Int" begin
        @test _should_reestimate(3, -1, false) == true         # never yet
        @test _should_reestimate(3, 0, false) == false
        @test _should_reestimate(3, 2, false) == false
        @test _should_reestimate(3, 3, false) == true
        @test _should_reestimate(3, 5, false) == true
        @test _should_reestimate(3, 0, true) == true           # stepfail forces regardless
    end

    @testset "callable" begin
        even_only = s -> iseven(s)
        @test _should_reestimate(even_only, -1, false) == true # never yet, callable not even called
        @test _should_reestimate(even_only, 4, false) == true
        @test _should_reestimate(even_only, 5, false) == false
        @test _should_reestimate(even_only, 5, true) == true    # stepfail forces regardless
    end
end

# Function barriers so the allocation checks measure the callee, not boxed captures at this
# scope (same reasoning as `warmup_then_allocated` in test_sts_families.jl).
function warmup_then_allocated_mul_rate(op, y, x)
    mul_rate!(y, op, x)
    return @allocated mul_rate!(y, op, x)
end
function warmup_then_allocated_estimate_rho(ws, apply!)
    estimate_rho!(ws, apply!)
    return @allocated estimate_rho!(ws, apply!)
end

@testset "Allocation-free after warmup" begin
    n = 40
    K = sparse(random_negdef(MersenneTwister(2), Float64, n))
    invM = ones(n)
    op = LumpedMassRateOperator(K, invM)
    y, x = zeros(n), randn(MersenneTwister(3), n)

    @test warmup_then_allocated_mul_rate(op, y, x) == 0

    ws = SpectralRadiusWorkspace(zeros(n))
    apply! = DenseApply(K)
    @test warmup_then_allocated_estimate_rho(ws, apply!) == 0
end

@testset "Float32 end-to-end" begin
    rng = MersenneTwister(7)
    n = 12
    A = random_negdef(rng, Float32, n)
    λmax_abs = maximum(abs, eigvals(Float64.(A)))

    ws = SpectralRadiusWorkspace(zeros(Float32, n))
    @test eltype(ws.v) === Float32

    ρ = estimate_rho!(ws, DenseApply(A); maxiters = 200, safety = 1.1)
    @test ρ isa Float32
    @test Float64(ρ) ≈ 1.1 * λmax_abs rtol = 0.1
end
