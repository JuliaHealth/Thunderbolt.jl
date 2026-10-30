"""
The canine ventricular cardiomyocyte electrophysiology model by [PatCorGra:2019:cuq](@citet).
"""
Base.@kwdef struct ParametrizedPCG2019Model{T} <: AbstractIonicModel
    # ------ I_Na -------
    g_Na::T = 12.0    # [mS/µF]
    E_m::T  = -52.244 # [mV]
    k_m::T  = 6.5472  # [mV]
    τ_m::T  = 0.12    # [ms]
    E_h::T  = -78.7   # [mV]
    k_h::T  = 5.93    # [mV]
    δ_h::T  = 0.799163 # dimensionless
    τ_h0::T = 6.80738  # [ms]
    # ------ I_K1 -------
    g_K1::T = 0.73893  # [mS/µF]
    E_z::T  = -91.9655 # [mV]
    k_z::T  = 12.4997  # [mV]
    # ------ I_to -------
    g_to::T = 0.1688   # [mS/µF]
    E_r::T  = 14.3116  # [mV]
    k_r::T  = 11.462   # [mV]
    E_s::T  = -47.9286 # [mV]
    k_s::T  = 4.9314   # [mV]
    τ_s::T  = 9.90669  # [ms]
    # ------ I_CaL -------
    g_CaL::T = 0.11503 # [mS/µF]
    E_d::T   = 0.7     # [mV]
    k_d::T   = 4.3     # [mV]
    E_f::T   = -15.7   # [mV]
    k_f::T   = 4.6     # [mV]
    τ_f::T   = 30.0    # [ms]
    # ------ I_Kr -------
    g_Kr::T = 0.056 # [mS/µF]
    E_xr::T = -26.6 # [mV]
    k_xr::T = 6.5   # [mV]
    τ_xr::T = 334.0 # [ms]
    E_y::T = -49.6 # [mV]
    k_y::T = 23.5  # [mV]
    # ------- I_Ks --------
    g_Ks::T = 0.008 # [mS/µF]
    E_xs::T = 24.6  # [mV]
    k_xs::T = 12.1  # [mV]
    τ_xs::T = 628.0 # [ms]
    # ------- Other --------
    E_Na::T = 65.0  # [mV]
    E_K::T  = -85.0 # [mV]
    E_Ca::T = 50.0  # [mV]
end

const PCG2019 = ParametrizedPCG2019Model{Float64};

# Shared between `cell_rhs_fast!`/`cell_rhs_slow!` and `gate_coefficients`: each `_pcg2019_*_gate`
# returns the (x∞, τ) pair of one state's gate normal form dx/dt = (x∞(φ) - x)/τ(φ). Every literal is
# tied to the type of the value it meets (`one(e)`, an `Int` sign that promotes rather than widens),
# so a `ParametrizedPCG2019Model{Float32}` evaluates its gates entirely in `Float32`; a bare
# `Float64` literal would put the hottest loop of a GPU "Float32" run into double precision.
@inline function _pcg2019_sigmoid(φ, E_Y, k_Y, sign)
    e = exp(sign * (φ - E_Y) / k_Y)
    return one(e) / (one(e) + e)
end

@inline function _pcg2019_h_gate(φ, p::ParametrizedPCG2019Model)
    @unpack E_h, k_h, τ_h0, δ_h = p
    e = exp((φ - E_h) / k_h)
    τ_h = (2 * τ_h0 * exp(δ_h * (φ - E_h) / k_h)) / (one(e) + e)
    h∞ = _pcg2019_sigmoid(φ, E_h, k_h, 1)
    return h∞, τ_h
end

@inline function _pcg2019_m_gate(φ, p::ParametrizedPCG2019Model)
    @unpack E_m, k_m, τ_m = p
    m∞ = _pcg2019_sigmoid(φ, E_m, k_m, -1)
    return m∞, τ_m
end

@inline function _pcg2019_f_gate(φ, p::ParametrizedPCG2019Model)
    @unpack E_f, k_f, τ_f = p
    f∞ = _pcg2019_sigmoid(φ, E_f, k_f, 1)
    return f∞, τ_f
end

@inline function _pcg2019_s_gate(φ, p::ParametrizedPCG2019Model)
    @unpack E_s, k_s, τ_s = p
    s∞ = _pcg2019_sigmoid(φ, E_s, k_s, 1)
    return s∞, τ_s
end

@inline function _pcg2019_xs_gate(φ, p::ParametrizedPCG2019Model)
    @unpack E_xs, k_xs, τ_xs = p
    xs∞ = _pcg2019_sigmoid(φ, E_xs, k_xs, -1)
    return xs∞, τ_xs
end

@inline function _pcg2019_xr_gate(φ, p::ParametrizedPCG2019Model)
    @unpack E_xr, k_xr, τ_xr = p
    xr∞ = _pcg2019_sigmoid(φ, E_xr, k_xr, -1)
    return xr∞, τ_xr
end

function cell_rhs_fast!(du, φ, state, x, t, p::ParametrizedPCG2019Model{T}) where {T}
    C_m = T(1.0) # TODO pass!

    @unpack g_Na, g_K1, g_to, g_CaL, g_Kr, g_Ks = p
    @unpack E_K, E_Na, E_Ca, E_r, E_d, E_z, E_y = p
    @unpack k_r, k_d, k_z, k_y                  = p

    h  = state[1]
    m  = state[2]
    f  = state[3]
    s  = state[4]
    xs = state[5]
    xr = state[6]

    # Instantaneous gates
    r∞ = _pcg2019_sigmoid(φ, E_r, k_r, -1)
    d∞ = _pcg2019_sigmoid(φ, E_d, k_d, -1)
    z∞ = _pcg2019_sigmoid(φ, E_z, k_z, 1)
    y∞ = _pcg2019_sigmoid(φ, E_y, k_y, 1)

    # Currents
    I_Na  = g_Na * m * m * m * h * h * (φ - E_Na)
    I_K1  = g_K1 * z∞ * (φ - E_K)
    I_to  = g_to * r∞ * s * (φ - E_K)
    I_CaL = g_CaL * d∞ * f * (φ - E_Ca)
    I_Kr  = g_Kr * xr * y∞ * (φ - E_K)
    I_Ks  = g_Ks * xs * (φ - E_K)

    I_total = I_Na + I_K1 + I_to + I_CaL + I_Kr + I_Ks

    du[1] = -I_total/C_m

    h∞, τ_h = _pcg2019_h_gate(φ, p)
    du[2] = (h∞-h)/τ_h

    m∞, τ_m = _pcg2019_m_gate(φ, p)
    du[3] = (m∞-m)/τ_m
end

function cell_rhs_slow!(du, φ, state, x, t, p::ParametrizedPCG2019Model)
    f  = state[3]
    s  = state[4]
    xs = state[5]
    xr = state[6]

    f∞, τ_f = _pcg2019_f_gate(φ, p)
    du[4] = (f∞-f)/τ_f

    s∞, τ_s = _pcg2019_s_gate(φ, p)
    du[5] = (s∞-s)/τ_s

    xs∞, τ_xs = _pcg2019_xs_gate(φ, p)
    du[6] = (xs∞-xs)/τ_xs

    xr∞, τ_xr = _pcg2019_xr_gate(φ, p)
    du[7] = (xr∞-xr)/τ_xr
end

function cell_rhs!(
    du::TD,
    u::TU,
    x::TX,
    t::TT,
    cell_parameters::TP,
) where {TD, TU, TX, TT, TP <: ParametrizedPCG2019Model}
    φₘ = u[1]
    s = @view u[2:end]
    cell_rhs_fast!(du, φₘ, s, x, t, cell_parameters)
    cell_rhs_slow!(du, φₘ, s, x, t, cell_parameters)
    return nothing
end

num_states(::Type{<:ParametrizedPCG2019Model}) = 7
state_symbols(::Type{<:ParametrizedPCG2019Model}) = (:φₘ, :h, :m, :f, :s, :xs, :xr)

gating_symbols(::Type{<:ParametrizedPCG2019Model}) = (:h, :m, :f, :s, :xs, :xr)

function gate_coefficients(p::ParametrizedPCG2019Model{T}, φ, x, t) where {T}
    h∞, τ_h   = _pcg2019_h_gate(φ, p)
    m∞, τ_m   = _pcg2019_m_gate(φ, p)
    f∞, τ_f   = _pcg2019_f_gate(φ, p)
    s∞, τ_s   = _pcg2019_s_gate(φ, p)
    xs∞, τ_xs = _pcg2019_xs_gate(φ, p)
    xr∞, τ_xr = _pcg2019_xr_gate(φ, p)
    λ         = SVector{6, T}(-1/τ_h, -1/τ_m, -1/τ_f, -1/τ_s, -1/τ_xs, -1/τ_xr)
    y∞        = SVector{6, T}(h∞, m∞, f∞, s∞, xs∞, xr∞)
    return λ, y∞
end

function default_initial_state(p::ParametrizedPCG2019Model{T}) where {T}
    @unpack E_K, E_h, E_m, E_f, E_s, E_xs, E_xr = p
    @unpack k_h, k_m, k_f, k_s, k_xs, k_xr      = p

    u₀ = zeros(T, 7)
    u₀[1] = E_K
    u₀[2] = _pcg2019_sigmoid(u₀[1], E_h, k_h, 1)
    u₀[3] = _pcg2019_sigmoid(u₀[1], E_m, k_m, -1)
    u₀[4] = _pcg2019_sigmoid(u₀[1], E_f, k_f, 1)
    u₀[5] = _pcg2019_sigmoid(u₀[1], E_s, k_s, 1)
    u₀[6] = _pcg2019_sigmoid(u₀[1], E_xs, k_xs, -1)
    u₀[7] = _pcg2019_sigmoid(u₀[1], E_xr, k_xr, -1)
    return u₀
end
