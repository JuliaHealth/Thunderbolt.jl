# # [Adding EP Cell Models](@id how-to-custom-ep-cell-model)
using Thunderbolt

# We first need to define a struct holding all the parameters.
# If you want to have parameters with spatial variation, which can be exchanged easily, then simply add a field with custom type and a function which accepts a coordinate `x` and a time `t` as input, as for example here for the parameter `e`:
# !!! tip
#     Parametrize the cell model parameters with the used float type to easily change precision for GPU simulations, as some GPUs perform very bad with Float64.
Base.@kwdef struct HeterogeneousFHNModel{T, T2} <: Thunderbolt.AbstractIonicModel
    a::T = T(0.1)
    b::T = T(0.5)
    c::T = T(1.0)
    d::T = T(0.0)
    e::T2 = (x,t)->0.01
end
HeterogeneousFHNModel(::Type{T}, e::F) where {T,F} = HeterogeneousFHNModel{T,F}(0.1,0.5,1.0,0.0,e)

# We now need to dispatch all functions of the cell EP API.
# First we need to give an index for the transmembrane potential, which we e.g. need for operator splitting methods and during post-processing.
Thunderbolt.transmembranepotential_index(cell_model::HeterogeneousFHNModel) = 1
# Now we need the number of state variables in the model.
# Here we have only a single internal state $s$.
Thunderbolt.num_states(::HeterogeneousFHNModel) = 1
# For convenience, we should dispatch this function which contains some admissible initial state for the model in its default parametrization.
Thunderbolt.default_initial_state(::HeterogeneousFHNModel) = [0.0, 0.0]
# Finally we also need to provide the right hand side of the model.
# The API is similar to what we have in SciML, but we have one additional input `x`.
# `x` contains spatial information to distinguish individual cells, allowing spatial gradients of cellular behavior.
# If no spatial information is provded, then `x === nothing`.
# Usually the types for x are either Vec{sdim}, if the coordinate is carthesian, or some generalized coordinate.
# Please consult [the coordinate system API docs](@ref coordinate-system-api) for more details.
# !!! note
#     You may have noticed that all inputs are parametrized.
#     This seems to be necessary to force specialization for the GPU code -- removing the type parameters leads to cryptic CUDA.jl errors.
function Thunderbolt.cell_rhs!(du::TD,u::TU,x::TX,t::TT,p::TP) where {TD,TU,TX,TT,TP <: HeterogeneousFHNModel}
    ## Flatten out parameters ... 
    (;a,b,c,d) = p
    ## ... and the state variables
    φₘ = u[1]
    s  = u[2]
    ## Now we assign the rates.
    du[1] = φₘ*(1-φₘ)*(φₘ-a) - s
    du[2] = p.e(x,t)*(b*φₘ - c*s - d)
    ## Return nothing so no oopsies happen. :)
    return nothing
end

# !!! todo
#     Show how users can control which the coordiante system information is passed into the cell model.
