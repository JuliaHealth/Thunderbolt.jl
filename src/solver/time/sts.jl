#####################################################################
#  Super-time-stepping (STS) core: RKC1/RKL1/RKG1 stage families    #
#####################################################################
"""
    AbstractSTSFamily

A super-time-stepping (STS) stage family: a first-order explicit Runge-Kutta scheme whose
stage count `s` trades accuracy order for an extended real stability interval, so that one
outer step of length `τ` can integrate a stiff, real-negative-spectrum right hand side in `s`
stages instead of `O(s²)` plain forward-Euler steps. See [`RKC1`](@ref), [`RKL1`](@ref),
[`RKG1`](@ref) and [`sts_sweep!`](@ref).
"""
abstract type AbstractSTSFamily end

"""
    RKC1(ε = 0.05) <: AbstractSTSFamily

First-order Runge-Kutta-Chebyshev family. `ε` damps the stability boundary away from the
imaginary axis (`ε = 0` is undamped and numerically fragile there).
Reference: van der Houwen & Sommeijer (1980); Rosilho de Souza et al., arXiv:2401.01745,
Algorithm 1.
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

# Coefficient naming used throughout this file: (μⱼ, νⱼ, μ̃ⱼ) with μⱼ the Yⱼ₋₁ coefficient, νⱼ
# the Yⱼ₋₂ coefficient, μ̃ⱼ the step coefficient --
#   Yⱼ = μⱼYⱼ₋₁ + νⱼYⱼ₋₂ + μ̃ⱼτf(Yⱼ₋₁),   μⱼ + νⱼ = 1.
# This matches OrdinaryDiffEqStabilizedRK's RKL1/RKG1 naming directly. The reference RKC
# implementation this RKC1 is transcribed from (Rosilho de Souza et al., arXiv:2401.01745,
# Algorithm 1; mRKC's `ChebyshevMethods::CoefficientsRKC1`) names the SAME three quantities
# the other way around: its `mu` is our μ̃ (step), its `nu` is our μ (Yⱼ₋₁), its `kappa` is
# our ν (Yⱼ₋₂).

"""
    sts_stage_count(fam::AbstractSTSFamily, z) -> Int

The smallest stage count `s` whose real stability interval admits `z = τ·ρ` (`τ` the outer
step, `ρ` a spectral radius estimate; any safety margin is the caller's responsibility).
"""
function sts_stage_count end

# s = ceil(√(z/β)), β = 2 - 4ε/3: the standard asymptotic RKC stage-count law (van der Houwen
# & Sommeijer 1980; Verwer, Hundsdorfer & Sommeijer, Numer. Math. 57 (1990) 157-178).
function sts_stage_count(fam::RKC1, z)
    β = 2 - 4fam.ε / 3
    return max(1, ceil(Int, sqrt(z / β)))
end

# Smallest odd s with z ≤ s² + s. This is the RKL1 stability boundary of Meyer, Balsara &
# Aslam (JCP 257 (2014), §2-3): the stability polynomial is R_s(z) = P_s(1 + w₁z) with
# w₁ = 2/(s²+s) for the shifted Legendre polynomial P_s, which is bounded by 1 in magnitude on
# [-1,1] with equality only at the endpoints; z ≤ s²+s is exactly where 1+w₁z reaches -1.
# Matches OrdinaryDiffEqStabilizedRK's RKL1 stage-count law verbatim.
function sts_stage_count(::RKL1, z)
    s = max(1, ceil(Int, (sqrt(1 + 4z) - 1) / 2))
    return isodd(s) ? s : s + 1
end

# Smallest s with z ≤ s(s+3)/2.
#
# Derivation (Skaras, Saxton, Meyer & Aslam, JCP 425 (2021) 109879, eqs. 22-24): the RKG1
# stability polynomial is R_s(z) = b_s C_s^(3/2)(1 + w₁z) with b_s = 2/((s+1)(s+2)) and
# w₁ = 4/(s(s+3)) for the shifted Gegenbauer polynomial C_s^(3/2) (α = 3/2). As for Legendre
# above, C_s^(3/2) attains its extremal magnitude on [-1,1] at the endpoints (checked here up
# to s = 30 against the α-recurrence of the paper's eq. 25, since the paper itself only states
# the CMP result, not this classical stability fact); the b_s normalization makes that extremal
# value 1, so |R_s(z)| ≤ 1 exactly up to z = 2/w₁ = s(s+3)/2.
#
# NOTE: the installed OrdinaryDiffEqStabilizedRK (`rkc_perform_step.jl`, RKG1) computes its
# stage count from z ≤ s(s+3)/4 -- exactly half of the bound above. Evaluating R_s at both
# candidates confirms s(s+3)/2 is the true root (|R_s| = 1 there to full precision, and > 1
# just beyond it), while s(s+3)/4 leaves |R_s| well under 1: that implementation is not
# unstable, just needlessly conservative by roughly a factor √2 in s. We use the
# literature-derived s(s+3)/2 here.
function sts_stage_count(::RKG1, z)
    return max(1, ceil(Int, (sqrt(9 + 8z) - 3) / 2))
end

"""
    sts_coefficient_state(fam::AbstractSTSFamily, s::Integer, ::Type{T})
    sts_stage_coefficients(fam::AbstractSTSFamily, state, j::Integer) -> ((μⱼ, νⱼ, μ̃ⱼ, cⱼ), state′)

On-the-fly scalar recurrence for the order-1 stage coefficients of an `s`-stage sweep.
`sts_coefficient_state` builds the initial state (before stage `j = 1`); `state` is then
threaded sequentially through `sts_stage_coefficients` calls for `j = 1, …, s` -- calling out
of order or skipping a stage is not supported. `cⱼ` is the stage time fraction
(`t0 + cⱼτ`; `cₛ == 1`). All scalars are computed in `Float64` regardless of `T`; `T` names
the state-vector eltype the caller intends and is otherwise unused here -- the conversion
happens at the broadcast in [`sts_sweep!`](@ref).
"""
function sts_coefficient_state end

"""
See [`sts_coefficient_state`](@ref) -- this is the second half of that same contract.
"""
function sts_stage_coefficients end

struct RKC1CoeffState
    ω0::Float64
    ω1::Float64
    Tjm2::Float64 # Tⱼ₋₂(ω0), ready for the next sts_stage_coefficients call
    Tjm1::Float64 # Tⱼ₋₁(ω0)
    cjm2::Float64
    cjm1::Float64
end

function sts_coefficient_state(fam::RKC1, s::Integer, ::Type{T}) where {T}
    ω0 = 1.0 + fam.ε / s^2
    # T_s(ω0) and T_s′(ω0) at the full degree s, via the Chebyshev recurrence and its
    # derivative recurrence, to seed ω1 = T_s(ω0)/T_s′(ω0) once for the whole sweep.
    Ts, Tsp = s == 1 ? (ω0, 1.0) : _chebyshev1_at_degree(ω0, s)
    ω1 = Ts / Tsp
    return RKC1CoeffState(ω0, ω1, 1.0, ω0, 0.0, 0.0)
end

function _chebyshev1_at_degree(ω0, s)
    Tjm2, Tjm1 = 1.0, ω0
    Tjm2p, Tjm1p = 0.0, 1.0
    for _ in 2:s
        Tj = 2ω0 * Tjm1 - Tjm2
        Tjp = 2Tjm1 + 2ω0 * Tjm1p - Tjm2p
        Tjm2, Tjm1 = Tjm1, Tj
        Tjm2p, Tjm1p = Tjm1p, Tjp
    end
    return Tjm1, Tjm1p
end

function sts_stage_coefficients(::RKC1, st::RKC1CoeffState, j::Integer)
    (; ω0, ω1) = st
    if j == 1
        μ̃ = ω1 / ω0
        return (1.0, 0.0, μ̃, μ̃), RKC1CoeffState(ω0, ω1, st.Tjm2, st.Tjm1, 0.0, μ̃)
    end
    Tj = 2ω0 * st.Tjm1 - st.Tjm2
    bj, bjm1, bjm2 = 1 / Tj, 1 / st.Tjm1, 1 / st.Tjm2
    μ = 2ω0 * bj / bjm1
    ν = -bj / bjm2
    μ̃ = 2ω1 * bj / bjm1
    c = μ * st.cjm1 + ν * st.cjm2 + μ̃
    return (μ, ν, μ̃, c), RKC1CoeffState(ω0, ω1, st.Tjm1, Tj, st.cjm1, c)
end

# RKL1 and RKG1 both have a closed form bⱼ(j) (no incremental polynomial recurrence needed),
# so their coefficient state is just the family's w1 plus the running stage-time pair.
struct RKLGCoeffState
    w1::Float64
    cjm2::Float64
    cjm1::Float64
end

sts_coefficient_state(::RKL1, s::Integer, ::Type{T}) where {T} =
    RKLGCoeffState(2.0 / (s^2 + s), 0.0, 0.0)

function sts_stage_coefficients(::RKL1, st::RKLGCoeffState, j::Integer)
    μ, ν = (2j - 1) / j, -(j - 1) / j # reduces to (1, 0) at j = 1
    μ̃ = μ * st.w1
    c = μ * st.cjm1 + ν * st.cjm2 + μ̃
    return (μ, ν, μ̃, c), RKLGCoeffState(st.w1, st.cjm1, c)
end

sts_coefficient_state(::RKG1, s::Integer, ::Type{T}) where {T} =
    RKLGCoeffState(4.0 / (s * (s + 3)), 0.0, 0.0)

function sts_stage_coefficients(::RKG1, st::RKLGCoeffState, j::Integer)
    # bⱼ(j) = 2/((j+1)(j+2)); νⱼ needs bⱼ₋₂, which is singular at j = 1 (no Yⱼ₋₂ there anyway).
    bj(k) = 2.0 / ((k + 1) * (k + 2))
    μ, ν = j == 1 ? (1.0, 0.0) : ((2j + 1) / j * bj(j) / bj(j - 1), -(j + 1) / j * bj(j) / bj(j - 2))
    μ̃ = μ * st.w1
    c = μ * st.cjm1 + ν * st.cjm2 + μ̃
    return (μ, ν, μ̃, c), RKLGCoeffState(st.w1, st.cjm1, c)
end

"""
    sts_sweep!(rhs!, Ya, Yb, du, y0, t0, τ, s, fam::AbstractSTSFamily) -> Y

Advances `y0` by one outer step of length `τ` through `s` stages of `fam`, calling
`rhs!(du, Y, t)` once per stage at the stage time `t0 + cⱼ₋₁·τ`. `Ya` and `Yb` are two
work buffers that rotate as `Yⱼ₋₁`/`Yⱼ₋₂` (a completed stage overwrites the buffer holding
the now-consumed `Yⱼ₋₂`); `y0` is read-only throughout and never one of the rotating buffers.
Returns whichever of `Ya`/`Yb` holds `Yₛ` -- callers must not assume it is always `Ya`.
`s == 1` degenerates to a single forward-Euler step of size `μ̃₁τ`. Allocation-free once
warmed up, for an `rhs!` that itself does not allocate.
"""
function sts_sweep!(rhs!, Ya, Yb, du, y0, t0, τ, s::Integer, fam::AbstractSTSFamily)
    st = sts_coefficient_state(fam, s, eltype(y0))
    for j in 1:s
        Yjm1 = j == 1 ? y0 : (isodd(j - 1) ? Ya : Yb)
        Yjm2 = j <= 2 ? y0 : (isodd(j - 2) ? Ya : Yb)
        dest = isodd(j) ? Ya : Yb
        rhs!(du, Yjm1, t0 + st.cjm1 * τ)
        (μ, ν, μ̃, _), st = sts_stage_coefficients(fam, st, j)
        @inbounds @.. dest = μ * Yjm1 + ν * Yjm2 + (μ̃ * τ) * du
    end
    return isodd(s) ? Ya : Yb
end
