using Thunderbolt
using Test
using StaticArrays

import Thunderbolt: exponential_gate_step

# Minimal LCG for the bit-identity regression below: `Random` is not a declared test dependency
# of this package, and this test only needs varied inputs, not statistical quality.
mutable struct _LCG
    state::UInt64
end
function _lcg_value!(g::_LCG)
    g.state = g.state * 0x5DEECE66D + 0xB
    return Float64(g.state >> 11) / Float64(UInt64(1) << 53) # in [0, 1)
end

# Frozen (φ, gate-state) samples spanning the physiological φ range, reused across the exactness,
# expm1-regression and consistency testsets below.
const _FROZEN_PHI = (-90.0, -55.0, -20.0, 10.0, 40.0)
const _GATE_STATE = SVector(0.05, 0.15, 0.35, 0.55, 0.75, 0.95) # h, m, f, s, xs, xr

# A model whose declared gate does not name a real state, to exercise `gating_indices`'s
# not-found error. `struct` must sit at top level, so it cannot live inside the `@testset` below.
struct _BogusGatingModel <: Thunderbolt.AbstractIonicModel end
Thunderbolt.state_symbols(::Type{_BogusGatingModel}) = (:φₘ, :s1)
Thunderbolt.gating_symbols(::Type{_BogusGatingModel}) = (:nope,)

@testset "gating_indices" begin
    @test gating_symbols(Thunderbolt.PCG2019()) == (:h, :m, :f, :s, :xs, :xr)
    @test gating_indices(Thunderbolt.PCG2019()) == (2, 3, 4, 5, 6, 7)

    @test gating_symbols(Thunderbolt.FHNModel()) == (:s,)
    @test gating_indices(Thunderbolt.FHNModel()) == (2,)

    # Default: no declared gates (degenerate coverage, e.g. Aliev-Panfilov).
    @test gating_symbols(Thunderbolt.AlievPanfilovModel()) == ()
    @test gating_indices(Thunderbolt.AlievPanfilovModel()) == ()

    @test_throws ErrorException gating_indices(_BogusGatingModel())
end

# `atol` matters here as much as `rtol`: at large η a fast gate's exponential term underflows and
# both sides converge on `y∞` through subtractive cancellation (`x - (x - y∞)`), so two correctly
# rounded ~1e-16-absolute errors can differ by many orders of magnitude in *relative* terms once
# `y∞` itself is tiny. `atol` catches that regime; `rtol` still governs it everywhere else.

@testset "Exactness: PCG2019 gate primitive vs analytic exponential decay" begin
    p = Thunderbolt.PCG2019()
    for φ in _FROZEN_PHI
        # `x` is the FULL local state row (φ, then the gates), as production passes it -- not the
        # gates-only vector `_GATE_STATE` is on its own.
        λ, y∞ = gate_coefficients(p, φ, SVector(φ, _GATE_STATE...), 0.0)
        for i in eachindex(λ), η in (1.0e-3, 0.1, 1.0, 10.0, 100.0)
            x0 = _GATE_STATE[i]
            expected = y∞[i] + (x0 - y∞[i]) * exp(η * λ[i])
            got = exponential_gate_step(x0, λ[i], y∞[i], η)
            @test got ≈ expected rtol = 1.0e-12 atol = 1.0e-12
        end
    end
end

@testset "Exactness: FHN gate primitive vs analytic exponential decay" begin
    p = Thunderbolt.FHNModel()
    for φ in (-1.0, 0.0, 0.3, 0.7, 1.5)
        λ, y∞ = gate_coefficients(p, φ, SVector(φ, 0.2), 0.0) # full row: (φ, s)
        x0 = 0.2
        for η in (1.0e-3, 0.1, 1.0, 10.0, 100.0)
            expected = y∞[1] + (x0 - y∞[1]) * exp(η * λ[1])
            got = exponential_gate_step(x0, λ[1], y∞[1], η)
            @test got ≈ expected rtol = 1.0e-12 atol = 1.0e-12
        end
    end
end

@testset "expm1 regression: η = 1e-12·τ keeps full precision" begin
    p = Thunderbolt.PCG2019()
    for φ in _FROZEN_PHI
        λ, y∞ = gate_coefficients(p, φ, SVector(φ, _GATE_STATE...), 0.0)
        for i in eachindex(λ)
            τ = -1 / λ[i]
            η = 1.0e-12 * τ
            x0 = _GATE_STATE[i]
            # First-order Taylor limit, computed directly (no `exp` call): this is what `expm1`
            # is supposed to reproduce to full precision at such a tiny `η*λ`, and what a naive
            # `x + (exp(η*λ) - 1)*(x - y∞)` loses through catastrophic cancellation in `exp - 1`.
            expected = x0 + η * λ[i] * (x0 - y∞[i])
            got = exponential_gate_step(x0, λ[i], y∞[i], η)
            @test got ≈ expected rtol = 1.0e-15
        end
    end
end

@testset "Consistency: dη-derivative at η=0 matches cell_rhs!" begin
    η = 1.0e-6
    # Central difference: PCG2019's fastest gate (τ_m = 0.12 ms) has |λ| ~ 8, and a one-sided
    # difference's O(η) truncation term (~λ²η/2) is already a few ppm there -- comparable to the
    # rtol below. Central differencing drops the truncation term to O(η²), several orders smaller.
    central_fd(x0, λi, y∞i) =
        (exponential_gate_step(x0, λi, y∞i, η) - exponential_gate_step(x0, λi, y∞i, -η)) / 2η
    @testset "PCG2019" begin
        p = Thunderbolt.PCG2019()
        for φ in _FROZEN_PHI
            u = vcat(φ, Vector(_GATE_STATE))
            du = zeros(7)
            Thunderbolt.cell_rhs!(du, u, nothing, 0.0, p)
            λ, y∞ = gate_coefficients(p, φ, u, 0.0) # full row, as production passes it
            for (k, idx) in enumerate(gating_indices(p))
                x0 = _GATE_STATE[k]
                fd = central_fd(x0, λ[k], y∞[k])
                @test fd ≈ du[idx] rtol = 1.0e-6
            end
        end
    end
    @testset "FHN" begin
        p = Thunderbolt.FHNModel()
        for φ in (-1.0, 0.0, 0.3, 0.7, 1.5)
            s0 = 0.2
            u = [φ, s0]
            du = zeros(2)
            Thunderbolt.cell_rhs!(du, u, nothing, 0.0, p)
            λ, y∞ = gate_coefficients(p, φ, u, 0.0) # full row, as production passes it
            idx = only(gating_indices(p))
            fd = central_fd(s0, λ[1], y∞[1])
            @test fd ≈ du[idx] rtol = 1.0e-6
        end
    end
end

# Pre-factoring reference: a verbatim copy of `cell_rhs_fast!`/`cell_rhs_slow!` as they read
# before the shared `_pcg2019_*_gate` helpers were factored out, so the factoring can be checked
# against the original expressions rather than against itself.
function _reference_cell_rhs_fast!(du, φ, state, x, t, p::Thunderbolt.ParametrizedPCG2019Model{T}) where {T}
    sigmoid(φ, E_Y, k_Y, sign) = 1.0 / (1.0 + exp(sign * (φ - E_Y) / k_Y))

    C_m = T(1.0)

    Thunderbolt.@unpack g_Na, g_K1, g_to, g_CaL, g_Kr, g_Ks           = p
    Thunderbolt.@unpack E_K, E_Na, E_Ca, E_r, E_d, E_z, E_y, E_h, E_m = p
    Thunderbolt.@unpack k_r, k_d, k_z, k_y, k_h, k_m                  = p

    Thunderbolt.@unpack τ_h0, δ_h, τ_m = p

    h  = state[1]
    m  = state[2]
    f  = state[3]
    s  = state[4]
    xs = state[5]
    xr = state[6]

    r∞ = sigmoid(φ, E_r, k_r, -1.0)
    d∞ = sigmoid(φ, E_d, k_d, -1.0)
    z∞ = sigmoid(φ, E_z, k_z, 1.0)
    y∞ = sigmoid(φ, E_y, k_y, 1.0)

    I_Na  = g_Na * m * m * m * h * h * (φ - E_Na)
    I_K1  = g_K1 * z∞ * (φ - E_K)
    I_to  = g_to * r∞ * s * (φ - E_K)
    I_CaL = g_CaL * d∞ * f * (φ - E_Ca)
    I_Kr  = g_Kr * xr * y∞ * (φ - E_K)
    I_Ks  = g_Ks * xs * (φ - E_K)

    I_total = I_Na + I_K1 + I_to + I_CaL + I_Kr + I_Ks

    du[1] = -I_total/C_m

    τ_h = (2.0 * τ_h0 * exp(δ_h * (φ - E_h) / k_h)) / (1.0 + exp((φ - E_h) / k_h))
    h∞ = sigmoid(φ, E_h, k_h, 1.0)
    du[2] = (h∞-h)/τ_h

    m∞ = sigmoid(φ, E_m, k_m, -1.0)
    du[3] = (m∞-m)/τ_m
end

function _reference_cell_rhs_slow!(du, φ, state, x, t, p::Thunderbolt.ParametrizedPCG2019Model)
    sigmoid(φ, E_Y, k_Y, sign) = 1.0 / (1.0 + exp(sign * (φ - E_Y) / k_Y))

    Thunderbolt.@unpack E_f, E_s, E_xs, E_xr = p
    Thunderbolt.@unpack k_f, k_s, k_xs, k_xr = p
    Thunderbolt.@unpack τ_f, τ_s, τ_xs, τ_xr = p

    f  = state[3]
    s  = state[4]
    xs = state[5]
    xr = state[6]

    f∞ = sigmoid(φ, E_f, k_f, 1.0)
    du[4] = (f∞-f)/τ_f

    s∞ = sigmoid(φ, E_s, k_s, 1.0)
    du[5] = (s∞-s)/τ_s

    xs∞ = sigmoid(φ, E_xs, k_xs, -1.0)
    du[6] = (xs∞-xs)/τ_xs

    xr∞ = sigmoid(φ, E_xr, k_xr, -1.0)
    du[7] = (xr∞-xr)/τ_xr
end

function _reference_cell_rhs!(du, u, x, t, p::Thunderbolt.ParametrizedPCG2019Model)
    φₘ = u[1]
    s = @view u[2:end]
    _reference_cell_rhs_fast!(du, φₘ, s, x, t, p)
    _reference_cell_rhs_slow!(du, φₘ, s, x, t, p)
    return nothing
end

@testset "PCG2019 factoring: bit-identical to the pre-factoring reference" begin
    p = Thunderbolt.PCG2019()
    g = _LCG(0x2545F4914F6CDD1D)
    for _ in 1:20
        φ = 160.0 * _lcg_value!(g) - 100.0
        state = [_lcg_value!(g) for _ in 1:6]
        u = vcat(φ, state)

        du = zeros(7)
        Thunderbolt.cell_rhs!(du, u, nothing, 0.0, p)

        du_ref = zeros(7)
        _reference_cell_rhs!(du_ref, u, nothing, 0.0, p)

        @test du == du_ref
    end
end

@testset "Float32 eltype" begin
    p32 = Thunderbolt.ParametrizedPCG2019Model{Float32}()
    φ = 0.0f0
    x = SVector{7,Float32}(φ, 0.1f0, 0.2f0, 0.3f0, 0.4f0, 0.5f0, 0.6f0) # full row: (φ, gates...)
    λ, y∞ = gate_coefficients(p32, φ, x, 0.0f0)
    @test λ isa SVector{6,Float32}
    @test y∞ isa SVector{6,Float32}

    gates = SVector{6,Float32}(x[2], x[3], x[4], x[5], x[6], x[7]) # the gate rows of the full row
    got = exponential_gate_step.(gates, λ, y∞, 0.01f0)
    @test eltype(got) == Float32
end
