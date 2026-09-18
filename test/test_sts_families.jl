using Thunderbolt
using Test

import Thunderbolt: sts_stage_count, sts_coefficient_state, sts_stage_coefficients, sts_sweep!

# Independent reference polynomials, so the sweep is checked against a closed form rather than
# against its own recurrence.
function chebyshev1_and_deriv(x, s)
    Tjm2, Tjm1 = 1.0, x
    Tjm2p, Tjm1p = 0.0, 1.0
    for _ in 2:s
        Tj = 2x * Tjm1 - Tjm2
        Tjp = 2Tjm1 + 2x * Tjm1p - Tjm2p
        Tjm2, Tjm1 = Tjm1, Tj
        Tjm2p, Tjm1p = Tjm1p, Tjp
    end
    return Tjm1, Tjm1p
end
chebyshev1(x, s) = s == 0 ? 1.0 : s == 1 ? x : chebyshev1_and_deriv(x, s)[1]

function legendre(x, s)
    s == 0 && return 1.0
    Pjm2, Pjm1 = 1.0, x
    for n in 2:s
        Pjm2, Pjm1 = Pjm1, ((2n - 1) * x * Pjm1 - (n - 1) * Pjm2) / n
    end
    return Pjm1
end

function gegenbauer32(x, s)
    s == 0 && return 1.0
    α = 1.5
    Cjm2, Cjm1 = 1.0, 2α * x
    for n in 2:s
        Cjm2, Cjm1 = Cjm1, (2x * (n + α - 1) * Cjm1 - (n + 2α - 2) * Cjm2) / n
    end
    return Cjm1
end

# R_s(τλ) from the family's closed-form stability polynomial. `s` is the stage count the sweep runs
# with (it fixes ω0/w1), `τλ` the actual signed product used in the test's y' = λy problem.
function closed_form_R(::RKC1, ε, s, τλ)
    ω0 = 1.0 + ε / s^2
    Ts, Tsp = s == 1 ? (ω0, 1.0) : chebyshev1_and_deriv(ω0, s)
    ω1 = Ts / Tsp
    return chebyshev1(ω0 + ω1 * τλ, s) / Ts
end
closed_form_R(::RKL1, s, τλ) = legendre(1 + (2.0 / (s^2 + s)) * τλ, s)
function closed_form_R(::RKG1, s, τλ)
    bs = 2.0 / ((s + 1) * (s + 2))
    w1 = 4.0 / (s * (s + 3))
    return bs * gegenbauer32(1 + w1 * τλ, s)
end

# The true real stability boundary |τλ_max| at stage count s, for sampling τλ safely inside the
# interval. RKL1/RKG1 are exact (see sts.jl); RKC1's is the exact root τλ = -2ω0/ω1 with
# ω1 = T_s(ω0)/T_s′(ω0), not the asymptotic law `sts_stage_count` uses.
function boundary_magnitude(fam::RKC1, s)
    ω0 = 1.0 + fam.ε / s^2
    Ts, Tsp = s == 1 ? (ω0, 1.0) : chebyshev1_and_deriv(ω0, s)
    return 2ω0 * Tsp / Ts
end
boundary_magnitude(::RKL1, s) = s^2 + s
boundary_magnitude(::RKG1, s) = s * (s + 3) / 2

# A non-allocating y' = λy right hand side; a callable struct rather than a closure, so it is safe
# in the allocation test.
struct LinearDecay{T}
    λ::T
end
(f::LinearDecay)(du, Y, t) = (du .= f.λ .* Y; nothing)

families() = (RKC1(0.05), RKL1(), RKG1())

@testset "Validation: sweep matches the closed-form stability polynomial" begin
    for fam in families(), s in (1, 2, 3, 5, 10, 20)
        zmax = boundary_magnitude(fam, s)
        for frac in (0.05, 0.3, 0.6, 0.9, 0.999)
            τλ = -frac * zmax
            λ, τ = τλ, 1.0 # λ carries the sign; τ = 1 so `τλ` above IS the product used below
            y0, Ya, Yb, du = [1.0], [0.0], [0.0], [0.0]
            Y = sts_sweep!(LinearDecay(λ), Ya, Yb, du, y0, 0.0, τ, s, fam)
            R = fam isa RKC1 ? closed_form_R(fam, fam.ε, s, τλ) : closed_form_R(fam, s, τλ)
            # atol guards the (odd s, small |R|) combinations where R sits near a polynomial root:
            # the two independently rounded paths differ sub-ULP in absolute terms, which is a large
            # *relative* disagreement against a tiny R.
            @test Y[1] ≈ R rtol = 1.0e-13 atol = 1.0e-13
        end
    end
end

@testset "Stage-count laws" begin
    @testset "RKC1: minimal and monotone" begin
        ε = 0.05
        β = 2 - 4ε / 3
        for z in (0.5, 3.7, 50.0, 500.0)
            s = sts_stage_count(RKC1(ε), z)
            @test z ≤ s^2 * β
            s > 1 && @test z > (s - 1)^2 * β
        end
        zs = (0.1, 1.0, 5.0, 20.0, 100.0, 1000.0)
        ss = [sts_stage_count(RKC1(ε), z) for z in zs]
        @test issorted(ss)
    end

    @testset "RKL1: minimal, monotone, always odd" begin
        for z in (0.5, 5.0, 30.0, 500.0)
            s = sts_stage_count(RKL1(), z)
            @test isodd(s)
            @test z ≤ s^2 + s
            s > 1 && @test z > (s - 2)^2 + (s - 2) # previous ODD stage count
        end
        zs = (0.1, 1.0, 5.0, 20.0, 100.0, 1000.0)
        ss = [sts_stage_count(RKL1(), z) for z in zs]
        @test issorted(ss)
        @test all(isodd, ss)
    end

    @testset "RKG1: minimal and monotone" begin
        for z in (0.5, 5.0, 30.0, 500.0)
            s = sts_stage_count(RKG1(), z)
            @test z ≤ s * (s + 3) / 2
            s > 1 && @test z > (s - 1) * (s + 2) / 2
        end
        zs = (0.1, 1.0, 5.0, 20.0, 100.0, 1000.0)
        ss = [sts_stage_count(RKG1(), z) for z in zs]
        @test issorted(ss)
    end
end

@testset "Extreme z raises a pointed error, not InexactError" begin
    # The threshold past typemax(Int) is family-dependent (~3.4e37 for RKC1 at ε = 0.05); 1e40 clears
    # it for all three.
    for fam in (RKC1(0.05), RKL1(), RKG1())
        e = @test_throws Thunderbolt.EMRKCDivergence sts_stage_count(fam, 1.0e40)
        @test !(e.value isa InexactError)
        @test occursin("does not fit", e.value.msg)
    end
end

@testset "sts_sweep!: mismatched buffer lengths raise DimensionMismatch" begin
    y0, Ya, Yb, du = [1.0, 2.0], [0.0, 0.0], [0.0, 0.0], [0.0] # du too short
    @test_throws DimensionMismatch sts_sweep!(
        LinearDecay(-1.0), Ya, Yb, du, y0, 0.0, 0.1, 2, RKC1(0.05),
    )
end

@testset "Internal consistency: c_s == 1" begin
    for fam in families(), s in (1, 2, 3, 5, 10, 20)
        st = sts_coefficient_state(fam, s, Float64)
        for j in 1:s
            (_, _, _, c), st = sts_stage_coefficients(fam, st, j)
            j == s && @test c ≈ 1.0 atol = 1.0e-12
        end
    end
end

@testset "Float32 state matches Float64 to 1e-5" begin
    for fam in families(), s in (1, 2, 5, 10)
        zmax = boundary_magnitude(fam, s)
        τλ = -0.5 * zmax

        y0f, Yaf, Ybf, duf = [1.0], [0.0], [0.0], [0.0]
        Yf64 = sts_sweep!(LinearDecay(τλ), Yaf, Ybf, duf, y0f, 0.0, 1.0, s, fam)

        y0s, Yas, Ybs, dus = Float32[1.0], Float32[0.0], Float32[0.0], Float32[0.0]
        Yf32 = sts_sweep!(LinearDecay(Float32(τλ)), Yas, Ybs, dus, y0s, 0.0f0, 1.0f0, s, fam)

        # rtol: the coefficient recurrence itself runs in Float32, so up to s = 10 stages accumulate
        # single-precision rounding. atol covers frac = 0.5 landing on the polynomials' midpoint
        # argument, a root for every odd s here, where the ratio of two near-zero numbers is
        # meaningless.
        @test Float64(Yf32[1]) ≈ Yf64[1] rtol = 1.0e-5 atol = 1.0e-5
        @test eltype(Yf32) == Float32
    end
end

@testset "Float32 purity: coefficients and sweep stay Float32, not promoted to Float64" begin
    for fam in families(), s in (1, 2, 5, 10)
        # `RKC1(0.05)`'s ε is a Float64 field; the coefficient state must convert it to `T` rather
        # than let it promote every downstream scalar back to Float64.
        st = sts_coefficient_state(fam, s, Float32)
        for j = 1:s
            (μ, ν, μ̃, c), st = sts_stage_coefficients(fam, st, j)
            @test μ isa Float32 && ν isa Float32 && μ̃ isa Float32 && c isa Float32
        end

        y0, Ya, Yb, du = Float32[1.0], Float32[0.0], Float32[0.0], Float32[0.0]
        Y = sts_sweep!(LinearDecay(-1.0f0), Ya, Yb, du, y0, 0.0f0, 0.01f0, s, fam)
        @test eltype(Y) == Float32
    end
end

# A function barrier: `for fam in families()` makes `fam` `Union`-typed at top-level scope, so a
# direct call would measure the dynamic dispatch rather than the sweep.
function warmup_then_allocated(rhs, Ya, Yb, du, y0, t0, τ, s, fam)
    sts_sweep!(rhs, Ya, Yb, du, y0, t0, τ, s, fam) # warmup: compile + fill buffers
    return @allocated sts_sweep!(rhs, Ya, Yb, du, y0, t0, τ, s, fam)
end

@testset "Allocation-free after warmup" begin
    for fam in families()
        s = 7
        rhs = LinearDecay(-1.0)
        y0, Ya, Yb, du = [1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        bytes = warmup_then_allocated(rhs, Ya, Yb, du, y0, 0.0, 0.01, s, fam)
        @test bytes == 0
    end
end

@testset "Stability near the boundary" begin
    for fam in families(), s in (1, 2, 3, 5, 10, 20)
        zmax = boundary_magnitude(fam, s)
        for frac in (0.5, 0.9, 0.999)
            τλ = -frac * zmax
            y0, Ya, Yb, du = [1.0], [0.0], [0.0], [0.0]
            Y = sts_sweep!(LinearDecay(τλ), Ya, Yb, du, y0, 0.0, 1.0, s, fam)
            @test abs(Y[1]) ≤ 1.0 + 1.0e-9
        end
    end
end
