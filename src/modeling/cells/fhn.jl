"""
The classical neuron electrophysiology model independently found by [Fit:1961:ips](@citet) and [NagARiYos:1962:apt](@citet).
This model is less stiff and cheaper than any cardiac electrophysiology model, which maks it
a good choice for quick testing if things work at all.
"""
Base.@kwdef struct ParametrizedFHNModel{T} <: AbstractIonicModel
    a::T = T(0.1)
    b::T = T(0.5)
    c::T = T(1.0)
    d::T = T(0.0)
    e::T = T(0.01)
end;

const FHNModel = ParametrizedFHNModel{Float64};

transmembranepotential_index(cell_model::ParametrizedFHNModel) = 1
num_states(::ParametrizedFHNModel) = 2
default_initial_state(::ParametrizedFHNModel) = [0.0, 0.0]

function cell_rhs!(
    du::TD,
    u::TU,
    x::TX,
    t::TT,
    cell_parameters::TP,
) where {TD, TU, TX, TT, TP <: ParametrizedFHNModel}
    @unpack a, b, c, d, e = cell_parameters
    φₘ = u[1]
    s = u[2]
    du[1] = φₘ*(1-φₘ)*(φₘ-a) - s
    du[2] = e*(b*φₘ - c*s - d)
    return nothing
end

@inline function reaction_rhs!(
    dφₘ::TD,
    φₘ::TV,
    s::TS,
    x::TX,
    t::TT,
    cell_parameters::ParametrizedFHNModel,
) where {TD <: SubArray, TV, TS, TX, TT}
    @unpack a = cell_parameters
    φₘ = u[1]
    s = u[2]
    dφₘ .= φₘ*(1-φₘ)*(φₘ-a) - s
    return nothing
end

@inline function state_rhs!(
    ds::TD,
    φₘ::TV,
    s::TS,
    x::TX,
    t::TT,
    cell_parameters::ParametrizedFHNModel,
) where {TD <: SubArray, TV, TS, TX, TT}
    @unpack b, c, d, e = cell_parameters
    φₘ = u[1]
    s = u[2]
    ds .= e*(b*φₘ - c*s - d)
    return nothing
end
