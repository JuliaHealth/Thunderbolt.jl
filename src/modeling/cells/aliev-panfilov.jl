Base.@kwdef struct ParametrizedAlievPanfilovModel{T} <: AbstractIonicModel
    cₜ::T = T(1.0/12.9)
    k::T = T(8.0)
    a::T = T(0.05)
    ϵ₀::T = T(0.002)
    μ₁::T = T(0.2)
    μ₂::T = T(0.3)
end

const AlievPanfilovModel = ParametrizedAlievPanfilovModel{Float64};

transmembranepotential_index(cell_model::ParametrizedAlievPanfilovModel) = 2
num_states(::ParametrizedAlievPanfilovModel) = 2
default_initial_state(::ParametrizedAlievPanfilovModel) = [0.0, 0.0]

function cell_rhs!(
    du::TD,
    u::TU,
    x::TX,
    t::TT,
    cell_parameters::TP,
) where {TD, TU, TX, TT, TP <: ParametrizedAlievPanfilovModel}
    (; cₜ, k, a, ϵ₀, μ₁, μ₂) = cell_parameters
    φₘ = u[2]
    s = u[1]
    ε = ϵ₀ + s * μ₁ / (φₘ + μ₂)
    du[2] = cₜ * (k * φₘ * (φₘ - 1.0) * (φₘ - a) - φₘ * s)
    du[1] = cₜ * ε * (-s - k * φₘ * (φₘ - a - 1.0))
    return nothing
end

@inline function reaction_rhs!(
    dφₘ::TD,
    φₘ::TV,
    s::TS,
    x::TX,
    t::TT,
    cell_parameters::ParametrizedAlievPanfilovModel,
) where {TD <: SubArray, TV, TS, TX, TT}
    (; cₜ, k, a, ϵ₀, μ₁, μ₂) = cell_parameters
    φₘ = u[2]
    s = u[1]
    dφₘ .= cₜ * (k*φₘ*(φₘ-a)*(1-φₘ) - φₘ*s)
    return nothing
end

@inline function state_rhs!(
    ds::TD,
    φₘ::TV,
    s::TS,
    x::TX,
    t::TT,
    cell_parameters::ParametrizedAlievPanfilovModel,
) where {TD <: SubArray, TV, TS, TX, TT}
    (; cₜ, k, a, ϵ₀, μ₁, μ₂) = cell_parameters
    φₘ = u[2]
    s = u[1]
    ε = ϵ₀ + s * μ₁ / (φₘ + μ₂)
    ds .= cₜ* ε * (-s - k*φₘ*(φₘ-a-1))
    return nothing
end


function f(du, u, p, t)
    (; cₜ, k, a, ϵ₀, μ₁, μ₂) = p
    φₘ = u[2]
    s = u[1]
    ε = ϵ₀ + s * μ₁ / (φₘ + μ₂)
    du[2] = cₜ * (k*φₘ*(φₘ-a)*(1-φₘ) - φₘ*s)
    du[1] = cₜ* ε * (-s - k*φₘ*(φₘ-a-1))
end
