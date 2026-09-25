using Thunderbolt
using Test
using LinearAlgebra
using Random
using SparseArrays
using Tensors

import Thunderbolt:
    SpectralRadiusWorkspace,
    estimate_rho!,
    EMRKCDivergence,
    _gershgorin_bound,
    _should_reestimate,
    RateOperator,
    mul_rate!,
    add_source_rate!,
    BilinearMassIntegrator,
    BilinearDiffusionIntegrator,
    setup_operator,
    update_operator!,
    AssemblyStrategy,
    SequentialCPUDevice,
    TimeIntegrationContext
import FerriteOperators

# A callable struct rather than a closure over `A`, so the allocation tests below measure
# `estimate_rho!`/`mul_rate!` rather than a boxed capture.
struct DenseApply{MT}
    A::MT
end
(f::DenseApply)(w, v) = mul!(w, f.A, v)

# A random negative-definite matrix: `-(RᵀR) - I` has eigenvalues ≤ -1 for any `R`, so the
# dominant-magnitude eigenvalue is real and power iteration applies directly.
random_negdef(rng, ::Type{T}, n) where {T} = (R = randn(rng, T, n, n); -(R' * R) - I)

@testset "estimate_rho! vs eigen: dense negative-definite matrices" begin
    # Tight settings, so this measures whether the power iteration converges to the right dominant
    # eigenvalue independently of a random draw's eigenvalue gap. The gap-sensitive *default*
    # reltol = 1e-2 is covered separately below, on a matrix with a known healthy gap.
    rng = MersenneTwister(20260901)
    for n in (5, 20), trial = 1:3
        A = random_negdef(rng, Float64, n)
        λmax_abs = maximum(abs, eigvals(A))

        ws = SpectralRadiusWorkspace(zeros(n))
        ρ = estimate_rho!(ws, DenseApply(A); maxiters = 20_000, reltol = 1.0e-12, safety = 1.0)

        @test ρ ≈ λmax_abs rtol = 1.0e-4
        @test ws.ρ == ρ
    end
end

@testset "estimate_rho! with default settings on a well-separated spectrum" begin
    # Ratio 1:10:100, so the default reltol = 1e-2 stopping rule -- which reads only the last step
    # size, not the true remaining error -- is not misled by a near-degenerate pair.
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
    # An `apply!` that ignores `v` and always returns the same bad result: an operator evaluated
    # where it should not be trusted, which reseeding and retrying cannot help either.
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
        e = @test_throws EMRKCDivergence estimate_rho!(ws, bad_apply!)
        @test occursin("non-finite", e.value.msg)
        @test occursin("retrying once", e.value.msg)
        @test bad_apply!.calls[] == 2 # the attempt, then the retry -- both fail on their first iterate
    end

    @testset "a finite but absurd jump vs. the previous estimate also triggers the guard" begin
        ws = SpectralRadiusWorkspace(zeros(3))
        ws.ρ = 1.0 # a modest "previous" estimate to jump away from
        huge_apply! = ConstantApply(1.0e30)
        e = @test_throws EMRKCDivergence estimate_rho!(ws, huge_apply!)
        @test occursin("jump", e.value.msg)
        @test occursin("retrying once", e.value.msg)
    end

    @testset "describe context reaches the error" begin
        ws = SpectralRadiusWorkspace(zeros(4))
        e = @test_throws EMRKCDivergence estimate_rho!(
            ws,
            ConstantApply(Inf);
            describe = () -> " EMRKC-specific context.",
        )
        @test occursin("EMRKC-specific context.", e.value.msg)
    end

    @testset "maxiters exhaustion is a silent, documented under-estimate" begin
        # Power iteration's per-step error is bounded by the eigenvalue ratio (1/10 here), so two
        # steps cannot reach reltol = 1e-15: exhaustion without tripping the non-finite/jump guard.
        # An under-budgeted `maxiters`/`reltol` pair is not a bad operator and must keep working.
        A = Diagonal([-1.0, -10.0, -100.0])
        ws = SpectralRadiusWorkspace(zeros(3))
        ρ = estimate_rho!(ws, DenseApply(A); maxiters = 2, reltol = 1.0e-15)
        @test isfinite(ρ) && ρ > 0
        @test ws.iters_done == 2 # the exhaustion signal a caller can check itself
    end

    @testset "a benign estimator is unaffected" begin
        # The guard sits on the success path every other testset here exercises; ordinary use must
        # never retry.
        A = Diagonal([-1.0, -10.0, -100.0])
        ws = SpectralRadiusWorkspace(zeros(3))
        ρ = estimate_rho!(ws, DenseApply(A))
        @test ρ ≈ 1.1 * 100.0 rtol = 0.05
    end
end

# A tiny FE heat problem, assembled as `Thunderbolt._assemble_laplacian` does it.
function assemble_heat_operators(n = 4)
    grid = generate_grid(Quadrilateral, (n, n))
    dh   = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral, 1}())
    close!(dh)

    qrc      = QuadratureRuleCollection(2)
    strategy = AssemblyStrategy(SequentialCPUDevice())
    ctx      = TimeIntegrationContext(0.0, 0.0, 0.0)

    mass      = BilinearMassIntegrator(ConstantCoefficient(1.0), qrc, :u)
    diffusion = BilinearDiffusionIntegrator(ConstantCoefficient(one(Tensor{2, 2})), qrc, :u)
    Mop       = setup_operator(strategy, mass, dh)
    Kop       = setup_operator(strategy, diffusion, dh)
    update_operator!(Mop, nothing, ctx)
    update_operator!(Kop, nothing, ctx)
    # The rate form the discretization's lumped mass gives `EMRKC`.
    rate = setup_operator(
        strategy,
        FerriteOperators.RateFormIntegrator(diffusion, FerriteOperators.RowSumLumped(mass)),
        dh;
        initial_context = ctx,
    )
    update_operator!(rate, nothing, ctx)
    return Mop, Kop, ndofs(dh), rate
end

@testset "Real FE case: heat problem lumped rate + ρ_F" begin
    Mop, Kop, n, rate = assemble_heat_operators()
    M = FerriteOperators.get_matrix(Mop)
    K = FerriteOperators.get_matrix(Kop)

    # (a) the lumped mass the discretization builds is the row-sum lumped consistent one.
    invM = FerriteOperators.rate_form_inverse_mass(rate).diag
    @test invM ≈ 1 ./ vec(sum(M, dims = 2))

    dense_rate = Diagonal(invM) * Matrix(K)
    λmax_abs   = maximum(abs, eigvals(dense_rate))

    # (b) estimate_rho! over mul_rate! agrees with eigen of the dense rate matrix; safety = 1
    # isolates the estimate from the safety margin, covered by the dense-matrix testset above.
    op = RateOperator(rate, invM, 1)
    ws = SpectralRadiusWorkspace(zeros(n))
    ρ = estimate_rho!(ws, (w, v) -> mul_rate!(w, op, v); maxiters = 300, safety = 1.0)
    @test ρ ≈ λmax_abs rtol = 0.05

    # (c) the Gershgorin bound is a genuine upper bound on the same spectral radius.
    @test _gershgorin_bound(K, invM) ≥ λmax_abs
    @test Thunderbolt._gershgorin_bound(op) ≥ λmax_abs
end

@testset "mul_rate! / add_source_rate!" begin
    invM = [1.0, 0.5, 1.0 / 3.0]
    A = sparse(Diagonal(invM) * Diagonal([2.0, -4.0, 6.0]))   # M⁻¹K, already composed
    op = RateOperator(A, invM, 1)

    y = zeros(3)
    x = [1.0, 1.0, 1.0]
    mul_rate!(y, op, x)
    @test y ≈ [2.0, -2.0, 2.0]

    add_source_rate!(y, op, [1.0, 2.0, 3.0])
    @test y ≈ [2.0 + 1.0, -2.0 + 1.0, 2.0 + 1.0]

    # `sign = -1` for an element assembling the POSITIVE stiffness convention: the same operator
    # comes out negated, assigned rather than accumulated into.
    neg = RateOperator(A, invM, -1)
    fill!(y, 17.0)
    mul_rate!(y, neg, x)
    @test y ≈ -[2.0, -2.0, 2.0]

    # No inverse mass beside a fused store: a source has no route and says so.
    @test_throws ErrorException add_source_rate!(zeros(3), RateOperator(A, nothing, 1), ones(3))

    @test rate_sign(
        BilinearDiffusionIntegrator(
            ConstantCoefficient(one(Tensor{2, 2})),
            QuadratureRuleCollection(2),
            :u,
        ),
    ) == 1
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
        # Period n: re-estimating exactly every n steps means steps_since ≥ n - 1 is due.
        @test _should_reestimate(3, -1, false) == true         # never yet
        @test _should_reestimate(3, 0, false) == false
        @test _should_reestimate(3, 1, false) == false
        @test _should_reestimate(3, 2, false) == true
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

# Function barriers so the allocation checks measure the callee, not boxed captures at this scope.
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
    op = RateOperator(K, ones(n), 1)
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
