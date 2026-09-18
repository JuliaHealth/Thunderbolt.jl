#####################################################################
#  Super-time-stepping (STS) core: RKC1/RKL1/RKG1 stage families    #
#####################################################################
"""
    AbstractSTSFamily

A first-order explicit Runge-Kutta family whose stage count `s` buys an extended real stability
interval, so one step of length `τ` integrates a stiff, real-negative-spectrum right hand side in
`s` stages instead of `O(s²)` forward-Euler steps.
See [`RKC1`](@ref), [`RKL1`](@ref), [`RKG1`](@ref), [`sts_sweep!`](@ref).
"""
abstract type AbstractSTSFamily end

"""
    RKC1(ε = 0.05) <: AbstractSTSFamily

First-order Runge-Kutta-Chebyshev family. `ε` damps the stability boundary away from the imaginary
axis (`ε = 0` is undamped and numerically fragile there).
Reference: van der Houwen & Sommeijer (1980); Rosilho de Souza et al., arXiv:2401.01745, Algorithm 1.
"""
struct RKC1{T <: Real} <: AbstractSTSFamily
    ε::T
end
RKC1() = RKC1(0.05)

"""
    RKL1 <: AbstractSTSFamily

First-order Runge-Kutta-Legendre family (undamped shifted-Legendre stability polynomial).
[`sts_stage_count`](@ref) always returns an odd stage count for this family.
Reference: Meyer, Balsara & Aslam, J. Comput. Phys. 257 (2014) 594-626.
"""
struct RKL1 <: AbstractSTSFamily end

"""
    RKG1 <: AbstractSTSFamily

First-order Runge-Kutta-Gegenbauer family (shifted Gegenbauer polynomial, α = 3/2). Unlike
[`RKL1`](@ref) it keeps the convex monotone property under a Dirichlet boundary condition.
Reference: Skaras, Saxton, Meyer & Aslam, J. Comput. Phys. 425 (2021) 109879.
"""
struct RKG1 <: AbstractSTSFamily end

# Coefficients throughout this file are (μⱼ, νⱼ, μ̃ⱼ) in
#   Yⱼ = μⱼYⱼ₋₁ + νⱼYⱼ₋₂ + μ̃ⱼτf(Yⱼ₋₁),   μⱼ + νⱼ = 1,
# matching OrdinaryDiffEqStabilizedRK's RKL1/RKG1 naming. The RKC reference this RKC1 is transcribed
# from (Rosilho de Souza et al., arXiv:2401.01745, Algorithm 1) names the same three the other way
# round: its `mu` is our μ̃, its `nu` our μ, its `kappa` our ν.

"""
    sts_stage_count(fam::AbstractSTSFamily, z) -> Int

The smallest stage count `s` whose real stability interval admits `z = τ·ρ` (`τ` the outer step,
`ρ` a spectral radius estimate; any safety margin is the caller's responsibility).
"""
function sts_stage_count end

@inline function _sts_safe_ceil(z, x)
    x < typemax(Int) || throw(EMRKCDivergence(
        "sts_stage_count: the stage count for z = $z does not fit in a machine `Int`. This " *
        "almost always means a diverged state or an unusable `rho_*_estimate` override, not a " *
        "genuine stiffness measure.",
    ))
    return ceil(Int, x)
end

# van der Houwen & Sommeijer (1980); Verwer, Hundsdorfer & Sommeijer, Numer. Math. 57 (1990) 157-178.
function sts_stage_count(fam::RKC1, z)
    β = 2 - 4fam.ε / 3
    return max(1, _sts_safe_ceil(z, sqrt(z / β)))
end

# Smallest odd s with z ≤ s² + s (Meyer, Balsara & Aslam, JCP 257 (2014), §2-3), matching
# OrdinaryDiffEqStabilizedRK's RKL1 law.
function sts_stage_count(::RKL1, z)
    s = max(1, _sts_safe_ceil(z, (sqrt(1 + 4z) - 1) / 2))
    return isodd(s) ? s : s + 1
end

# Smallest s with z ≤ s(s+3)/2: R_s(z) = b_s C_s^(3/2)(1 + w₁z) with b_s = 2/((s+1)(s+2)) and
# w₁ = 4/(s(s+3)) (Skaras, Saxton, Meyer & Aslam, JCP 425 (2021) 109879, eqs. 22-24), whose extremal
# magnitude on [-1,1] is attained at the endpoints (checked here to s = 30 against the paper's
# eq. 25 α-recurrence), normalized by b_s to 1 -- so |R_s| ≤ 1 exactly up to z = 2/w₁ = s(s+3)/2.
#
# NOTE: the installed OrdinaryDiffEqStabilizedRK (`rkc_perform_step.jl`, RKG1) uses z ≤ s(s+3)/4,
# half of that. Evaluating R_s at both confirms s(s+3)/2 is the true root (|R_s| = 1 there to full
# precision, > 1 just beyond) while s(s+3)/4 leaves |R_s| well under 1: that implementation is
# conservative by roughly √2 in s, not unstable. We use the literature-derived s(s+3)/2.
function sts_stage_count(::RKG1, z)
    return max(1, _sts_safe_ceil(z, (sqrt(9 + 8z) - 3) / 2))
end

"""
    sts_stability_boundary(fam::AbstractSTSFamily, s::Integer) -> Float64

The real stability boundary of an `s`-stage sweep: the largest `z` with `|R_s(-z)| ≤ 1`. Forward
partner of [`sts_stage_count`](@ref), which inverts it, so
`sts_stage_count(fam, sts_stability_boundary(fam, s)) == s` for every family and every `s` it admits.
"""
function sts_stability_boundary end

sts_stability_boundary(fam::RKC1, s::Integer) = (2 - 4fam.ε / 3) * float(s)^2
sts_stability_boundary(::RKL1, s::Integer) = float(s)^2 + float(s)
sts_stability_boundary(::RKG1, s::Integer) = float(s) * (float(s) + 3) / 2

# On-the-fly scalar recurrence for the order-1 stage coefficients of an `s`-stage sweep:
# `sts_coefficient_state` builds the state before stage `j = 1`, which is then threaded sequentially
# through `sts_stage_coefficients` for `j = 1, …, s` -- out of order or with a stage skipped is
# unsupported. `cⱼ` is the stage time fraction (`t0 + cⱼτ`; `cₛ == 1`). Every scalar is computed in
# `T`, a family struct's own fields (e.g. `RKC1.ε`) included, so a `T`-typed sweep's broadcasts are
# not promoted back to the field's type.
function sts_coefficient_state end
function sts_stage_coefficients end

struct RKC1CoeffState{T}
    ω0::T
    ω1::T
    Tjm2::T # Tⱼ₋₂(ω0)
    Tjm1::T # Tⱼ₋₁(ω0)
    cjm2::T
    cjm1::T
end

function sts_coefficient_state(fam::RKC1, s::Integer, ::Type{T}) where {T}
    ε = T(fam.ε)
    ω0 = one(T) + ε / T(s)^2
    Ts, Tsp = s == 1 ? (ω0, one(T)) : _chebyshev1_at_degree(ω0, s)
    ω1 = Ts / Tsp
    return RKC1CoeffState{T}(ω0, ω1, one(T), ω0, zero(T), zero(T))
end

function _chebyshev1_at_degree(ω0::T, s) where {T}
    Tjm2, Tjm1 = one(T), ω0
    Tjm2p, Tjm1p = zero(T), one(T)
    for _ = 2:s
        Tj = 2ω0 * Tjm1 - Tjm2
        Tjp = 2Tjm1 + 2ω0 * Tjm1p - Tjm2p
        Tjm2, Tjm1 = Tjm1, Tj
        Tjm2p, Tjm1p = Tjm1p, Tjp
    end
    return Tjm1, Tjm1p
end

function sts_stage_coefficients(::RKC1, st::RKC1CoeffState{T}, j::Integer) where {T}
    (; ω0, ω1) = st
    if j == 1
        μ̃ = ω1 / ω0
        return (one(T), zero(T), μ̃, μ̃), RKC1CoeffState{T}(ω0, ω1, st.Tjm2, st.Tjm1, zero(T), μ̃)
    end
    Tj = 2ω0 * st.Tjm1 - st.Tjm2
    bj, bjm1, bjm2 = one(T) / Tj, one(T) / st.Tjm1, one(T) / st.Tjm2
    μ = 2ω0 * bj / bjm1
    ν = -bj / bjm2
    μ̃ = 2ω1 * bj / bjm1
    c = μ * st.cjm1 + ν * st.cjm2 + μ̃
    return (μ, ν, μ̃, c), RKC1CoeffState{T}(ω0, ω1, st.Tjm1, Tj, st.cjm1, c)
end

# RKL1 and RKG1 have a closed form bⱼ(j), so their state is just w1 plus the running stage-time pair.
struct RKLGCoeffState{T}
    w1::T
    cjm2::T
    cjm1::T
end

sts_coefficient_state(::RKL1, s::Integer, ::Type{T}) where {T} =
    RKLGCoeffState{T}(T(2) / T(s^2 + s), zero(T), zero(T))

function sts_stage_coefficients(::RKL1, st::RKLGCoeffState{T}, j::Integer) where {T}
    μ, ν = T(2j - 1) / T(j), -T(j - 1) / T(j) # reduces to (1, 0) at j = 1
    μ̃ = μ * st.w1
    c = μ * st.cjm1 + ν * st.cjm2 + μ̃
    return (μ, ν, μ̃, c), RKLGCoeffState{T}(st.w1, st.cjm1, c)
end

sts_coefficient_state(::RKG1, s::Integer, ::Type{T}) where {T} =
    RKLGCoeffState{T}(T(4) / T(s * (s + 3)), zero(T), zero(T))

function sts_stage_coefficients(::RKG1, st::RKLGCoeffState{T}, j::Integer) where {T}
    # bⱼ(j) = 2/((j+1)(j+2)); νⱼ needs bⱼ₋₂, which is singular at j = 1 (no Yⱼ₋₂ there anyway).
    bj(k) = T(2) / (T(k + 1) * T(k + 2))
    μ, ν = j == 1 ? (one(T), zero(T)) :
        (T(2j + 1) / T(j) * bj(j) / bj(j - 1), -T(j + 1) / T(j) * bj(j) / bj(j - 2))
    μ̃ = μ * st.w1
    c = μ * st.cjm1 + ν * st.cjm2 + μ̃
    return (μ, ν, μ̃, c), RKLGCoeffState{T}(st.w1, st.cjm1, c)
end

"""
    sts_sweep!(rhs!, Ya, Yb, du, y0, t0, τ, s, fam::AbstractSTSFamily) -> Y

Advances `y0` by one outer step of length `τ` through `s` stages of `fam`, calling `rhs!(du, Y, t)`
once per stage at the stage time `t0 + cⱼ₋₁·τ`. `Ya` and `Yb` rotate as `Yⱼ₋₁`/`Yⱼ₋₂`; `y0` is
read-only throughout and must not be one of them. Returns whichever of `Ya`/`Yb` holds `Yₛ` --
callers must not assume it is `Ya`. `s == 1` degenerates to a forward-Euler step of size `μ̃₁τ`.
Allocation-free once warmed up, for an `rhs!` that does not allocate.

`Ya`, `Yb`, `du` and `y0` must have equal `length`, checked via `@boundscheck`: the per-stage
broadcast is `@inbounds`, so a short buffer would be a silent out-of-bounds write.
"""
function sts_sweep!(rhs!, Ya, Yb, du, y0, t0, τ, s::Integer, fam::AbstractSTSFamily)
    @boundscheck (length(Ya) == length(Yb) == length(du) == length(y0)) ||
        throw(DimensionMismatch("sts_sweep!: Ya, Yb, du and y0 must have equal length; got " *
            "$(length(Ya)), $(length(Yb)), $(length(du)), $(length(y0))."))
    st = sts_coefficient_state(fam, s, eltype(y0))
    for j = 1:s
        Yjm1 = j == 1 ? y0 : (isodd(j - 1) ? Ya : Yb)
        Yjm2 = j <= 2 ? y0 : (isodd(j - 2) ? Ya : Yb)
        dest = isodd(j) ? Ya : Yb
        rhs!(du, Yjm1, t0 + st.cjm1 * τ)
        (μ, ν, μ̃, _), st = sts_stage_coefficients(fam, st, j)
        @inbounds @.. dest = μ * Yjm1 + ν * Yjm2 + (μ̃ * τ) * du
    end
    return isodd(s) ? Ya : Yb
end

# Exact value at `η` of dx/dt = λ(x - y∞), the gate normal form of `gating_symbols`, written with
# `expm1` because `e^{ηλ} - 1` cancels catastrophically for the small ηλ an inner STS sub-step
# produces. Intended for decay gates, λ ≤ 0: at λ > 0, or λ = 0 with an infinite `y∞`, the result can
# be non-finite and nothing here guards against that -- the caller's `isfinite` check on the swept
# state (e.g. `EMRKC`'s step) is where it is caught.
@inline exponential_gate_step(x, λ, y∞, η) = x + expm1(η * λ) * (x - y∞)
