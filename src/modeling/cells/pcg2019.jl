"""
The canine ventricular cardiomyocyte electrophysiology model by [PatCorGra:2019:cuq](@citet).
"""
Base.@kwdef struct ParametrizedPCG2019Model{T} <: AbstractIonicModel
    # ------ I_Na -------
    g_Na::T  = 12.0    # [mS/µF]
    E_m::T   = -52.244 # [mV]
    k_m::T   = 6.5472  # [mV]
    τ_m::T  = 0.12    # [ms]
    E_h::T   = -78.7   # [mV]
    k_h::T   = 5.93    # [mV]
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
    τ_s::T = 9.90669  # [ms]
    # ------ I_CaL -------
    g_CaL::T = 0.11503 # [mS/µF]
    E_d::T   = 0.7     # [mV]
    k_d::T   = 4.3     # [mV]
    E_f::T   = -15.7   # [mV]
    k_f::T   = 4.6     # [mV]
    τ_f::T  = 30.0    # [ms]
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

function cell_rhs_fast!(du, φ, state, x, t, p::ParametrizedPCG2019Model{T}) where {T}
    sigmoid(φ, E_Y, k_Y, sign) = 1.0 / (1.0 + exp(sign * (φ - E_Y) / k_Y))

    C_m = T(1.0) # TODO pass!

    @unpack g_Na, g_K1, g_to, g_CaL, g_Kr, g_Ks           = p
    @unpack E_K, E_Na, E_Ca, E_r, E_d, E_z, E_y, E_h, E_m = p
    @unpack k_r, k_d, k_z, k_y, k_h, k_m                  = p

    @unpack τ_h0, δ_h, τ_m = p

    h  = state[1]
    m  = state[2]
    f  = state[3]
    s  = state[4]
    xs = state[5]
    xr = state[6]

    # Instantaneous gates
    r∞ = sigmoid(φ, E_r, k_r, -1.0)
    d∞ = sigmoid(φ, E_d, k_d, -1.0)
    z∞ = sigmoid(φ, E_z, k_z, 1.0)
    y∞ = sigmoid(φ, E_y, k_y, 1.0)

    # Currents
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

function cell_rhs_slow!(du, φ, state, x, t, p::ParametrizedPCG2019Model)
    sigmoid(φ, E_Y, k_Y, sign) = 1.0 / (1.0 + exp(sign * (φ - E_Y) / k_Y))

    @unpack E_f, E_s, E_xs, E_xr = p
    @unpack k_f, k_s, k_xs, k_xr = p
    @unpack τ_f, τ_s, τ_xs, τ_xr = p

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

transmembranepotential_index(cell_model::ParametrizedPCG2019Model) = 1
num_states(::ParametrizedPCG2019Model) = 7
function default_initial_state(p::ParametrizedPCG2019Model{T}) where {T}
    sigmoid(φ, E_Y, k_Y, sign) = 1.0 / (1.0 + exp(sign * (φ - E_Y) / k_Y))

    @unpack E_K, E_h, E_m, E_f, E_s, E_xs, E_xr = p
    @unpack k_h, k_m, k_f, k_s, k_xs, k_xr      = p

    u₀ = zeros(T, 7)
    u₀[1] = E_K
    u₀[2] = sigmoid(u₀[1], E_h, k_h, 1.0)
    u₀[3] = sigmoid(u₀[1], E_m, k_m, -1.0)
    u₀[4] = sigmoid(u₀[1], E_f, k_f, 1.0)
    u₀[5] = sigmoid(u₀[1], E_s, k_s, 1.0)
    u₀[6] = sigmoid(u₀[1], E_xs, k_xs, -1.0)
    u₀[7] = sigmoid(u₀[1], E_xr, k_xr, -1.0)
    return u₀
end
