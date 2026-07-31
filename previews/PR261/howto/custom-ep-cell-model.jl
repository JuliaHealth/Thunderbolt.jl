using Thunderbolt

Base.@kwdef struct HeterogeneousFHNModel{T, T2} <: Thunderbolt.AbstractIonicModel
    a::T = T(0.1)
    b::T = T(0.5)
    c::T = T(1.0)
    d::T = T(0.0)
    e::T2 = (x,t)->0.01
end
HeterogeneousFHNModel(::Type{T}, e::F) where {T,F} = HeterogeneousFHNModel{T,F}(0.1,0.5,1.0,0.0,e)

Thunderbolt.transmembranepotential_index(cell_model::HeterogeneousFHNModel) = 1

Thunderbolt.num_states(::HeterogeneousFHNModel) = 1

Thunderbolt.default_initial_state(::HeterogeneousFHNModel) = [0.0, 0.0]

function Thunderbolt.cell_rhs!(du::TD,u::TU,x::TX,t::TT,p::TP) where {TD,TU,TX,TT,TP <: HeterogeneousFHNModel}
    # Flatten out parameters ...
    (;a,b,c,d) = p
    # ... and the state variables
    φₘ = u[1]
    s  = u[2]
    # Now we assign the rates.
    du[1] = φₘ*(1-φₘ)*(φₘ-a) - s
    du[2] = p.e(x,t)*(b*φₘ - c*s - d)
    # Return nothing so no oopsies happen. :)
    return nothing
end

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
