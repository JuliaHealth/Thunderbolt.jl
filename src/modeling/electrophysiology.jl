"""
"""
abstract type AbstractIonChannel end;

"""
"""
abstract type MarkovTypeIonChannel <: AbstractIonChannel end;

"""
Helper to properly dispatch individual gating variables.
"""
abstract type HodgkinHuxleyTypeGate end;

"""
Parameters for a generic sigmoid function of the form

σ(x) = \frac{A + B x}{C + D \exp{\frac{E + F x}{G}}}
"""
struct GenericSigmoidParameters{T}
    A::T
    B::T
    C::T
    D::T
    E::T
    F::T
    G::T
end

@inline σ(s::T1, p::GenericSigmoidParameters{T2}) where {T1, T2} =
    (p.A + p.B*s)/(p.C + p.D*exp(p.E + p.F*s) / p.G)

"""
The classical gate formulation is stated in the normalized affine form:

∂ₜ𝐬ₖ = αₖ(φₘ)𝐬ₖ + βₖ(φₘ)

where αₖ(φₘ) = \frac{A + B φₘ}{C + D \exp{\frac{E + F φₘ}{G}}}

Note that the original formulation is

∂ₜ𝐬ₖ = aₖ(φₘ)𝐬ₖ + bₖ(φₘ)(1 - 𝐬ₖ)

where αₖ = aₖ - bₖ and βₖ = bₖ.
"""
struct GenericHodgkinHuxleyGate{T} <: HodgkinHuxleyTypeGate where {T}
    αₚ::GenericSigmoidParameters{T}
    βₚ::GenericSigmoidParameters{T}
end

@inline α(g::GenericHodgkinHuxleyGate{T}, φₘ::T) where {T} = σ(φₘ, g.αₚ)
@inline β(g::GenericHodgkinHuxleyGate{T}, φₘ::T) where {T} = σ(φₘ, g.βₚ)

# Spatially varying parameters
@inline α(g::GenericHodgkinHuxleyGate{T1}, φₘ::T2, x::T3) where {T1, T2, T3} = σ(φₘ, g.αₚ(x))
@inline β(g::GenericHodgkinHuxleyGate{T1}, φₘ::T2, x::T3) where {T1, T2, T3} = σ(φₘ, g.βₚ(x))

"""
Probabilistic ion channels with diagonal, semi-affine internal structure.

∂ₜ𝐬₁ = g₁(φₘ, 𝐬) = α₁(φₘ)𝐬₁ + β₁(φₘ)
        .
        .
        .
∂ₜ𝐬ₙ = gₙ(φₘ, 𝐬) = αₙ(φₘ)𝐬ₙ + βₙ(φₘ)

They can be derived as special cases of Markov type ion channels with
tensor-product structure (TODO citation). 𝐬 is called the gating vector
and its entries are the gating variables.
"""
struct HodgkinHuxleyTypeIonChannel{NGates} <: AbstractIonChannel where {NGates}
    gates::SVector{NGates, HodgkinHuxleyTypeGate}
    powers::SVector{NGates, Int}
end;

@inline function g(gate::HodgkinHuxleyTypeGate, φₘ::T, 𝐬ᵢ::T) where {T}
    α(gate, φₘ)*𝐬ᵢ + β(gate, φₘ)
end

@inline function g(gate::HodgkinHuxleyTypeGate, φₘ::T, 𝐬ᵢ::T, x::AbstractVector{T}) where {T}
    α(gate, φₘ, x)*𝐬ᵢ + β(gate, φₘ, x)
end

"""
Ohmic current of the form

Iⱼ = ̄gⱼ pⱼ (φₘ - Eⱼ)

where ̄gⱼ is the maximal conductance, pᵢ the open probability of the associated channel and Eⱼ the equilibrium potential.
"""
struct OhmicCurrent{T, NChannels}
    g::T
    channels::SVector{NChannels, HodgkinHuxleyTypeIonChannel}
end

"""
Supertype for all ionic models in Thunderbolt.
"""
abstract type AbstractIonicModel end

"""
    num_states(ionic_model) -> Int
    num_states(::Type{<:AbstractIonicModel})

Length of the model's local state vector. Like [`state_symbols`](@ref) this is a property of the model
*type*; implementations dispatch on the type and the instance form forwards.
"""
num_states(model::AbstractIonicModel) = num_states(typeof(model))

"""
    state_symbols(ionic_model)              -> NTuple{num_states(ionic_model), Symbol}
    state_symbols(::Type{<:AbstractIonicModel})

Names of the local state variables, in the order they occupy the model's local state vector.

Names are a property of the model *type*, so implementations dispatch on the type and the instance form
forwards. The default is `(:φₘ, :s1, :s2, …)`; override it when the transmembrane potential does not come
first, or to give the remaining states meaningful names.
"""
state_symbols(model::AbstractIonicModel) = state_symbols(typeof(model))
function state_symbols(::Type{M}) where {M <: AbstractIonicModel}
    return ntuple(i -> i == 1 ? transmembranepotential_symbol(M) : Symbol(:s, i-1), num_states(M))
end

"""
    transmembranepotential_symbol(ionic_model) -> Symbol
    transmembranepotential_symbol(::Type{<:AbstractIonicModel})

Which entry of [`state_symbols`](@ref) plays the role of the transmembrane potential. Defaults to `:φₘ`.

This is a *role*, not a user-facing name: the field a `MonodomainModel` exposes is named independently, so
the two need not agree.
"""
transmembranepotential_symbol(model::AbstractIonicModel) =
    transmembranepotential_symbol(typeof(model))
transmembranepotential_symbol(::Type{<:AbstractIonicModel}) = :φₘ

"""
    transmembranepotential_index(ionic_model) -> Int

Position of the transmembrane potential in the local state vector, derived from [`state_symbols`](@ref) and
[`transmembranepotential_symbol`](@ref) so that a model's state names and the index cannot disagree. The
potential may therefore sit at any index.

Both inputs depend only on the model *type*, so this folds to a literal and no `Symbol` survives into the
pointwise kernels. It is deliberately *not* a `@generated` function: a generator runs in the world age of
the caller's compilation, so a cell model defined by a user after Thunderbolt was loaded would be invisible
to it.
"""
@inline function transmembranepotential_index(ionic_model::AbstractIonicModel)
    idx = findfirst(==(transmembranepotential_symbol(ionic_model)), state_symbols(ionic_model))
    idx === nothing && _no_transmembrane_potential(ionic_model)
    return idx
end

@noinline function _no_transmembrane_potential(ionic_model)
    φsym = transmembranepotential_symbol(ionic_model)
    return error(
        "Cannot locate the transmembrane potential in $(nameof(typeof(ionic_model))): its " *
        "`transmembranepotential_symbol` is $(repr(φsym)), but `state_symbols` returns " *
        "$(state_symbols(ionic_model)). Implement " *
        "`Thunderbolt.state_symbols(::Type{<:$(nameof(typeof(ionic_model)))})` naming every state " *
        "(including $(repr(φsym))), or override `Thunderbolt.transmembranepotential_symbol` if the " *
        "potential goes by another name in this model.",
    )
end

"""
Models where all states are described by Hodgkin-Huxley type ion channels.
"""
abstract type HodgkinHuxleyTypeModel <: AbstractIonicModel end;

"""
The model from the seminal paper of Hodgkin and Huxley (1952).

(TODO citation)
"""
struct HodgkinHuxleyModel <: HodgkinHuxleyTypeModel end;

"""
Simplest model with qubic reaction and no state.

(TODO citation)
"""
struct NagumoModel <: HodgkinHuxleyTypeModel end;

"""
Simple model with qubic reaction and linear state.

(TODO citation)
"""
struct FitzHughNagumoModel <: HodgkinHuxleyTypeModel end;

abstract type AbstractEPModel end;

"""
    has_pointwise_reaction_part(model) -> Bool

Capability trait: does `model` contribute a pointwise reaction ODE that a reaction-diffusion split
can peel off into its own subproblem?

This is a trait rather than an `isa AbstractEPModel` check so that models outside Thunderbolt's own
type hierarchy - including types owned by other packages, which cannot be retrofitted with a
supertype - can declare the capability. Models answering `true` must also implement
[`reaction_model`](@ref) and [`reaction_solution_symbol`](@ref).
"""
has_pointwise_reaction_part(model) = false
has_pointwise_reaction_part(::AbstractEPModel) = true

"""
    reaction_model(model)

The pointwise ODE model driving the reaction part of `model`, for models with
[`has_pointwise_reaction_part`](@ref).
"""
reaction_model(model::AbstractEPModel) = model.ion

"""
    reaction_solution_symbol(model)

The field variable the reaction part of `model` acts on, for models with
[`has_pointwise_reaction_part`](@ref).
"""
function reaction_solution_symbol end

abstract type AbstractStimulationProtocol <: AbstractSourceTerm end;

@doc raw"""
Supertype for all stimulation protocols fulfilling $I_{\rm{stim,e}} = I_{\rm{stim,i}}$.
"""
abstract type TransmembraneStimulationProtocol <: AbstractStimulationProtocol end;

"""
A dummy protocol describing the absence of stimuli for a simulation.
"""
struct NoStimulationProtocol <: TransmembraneStimulationProtocol end

function setup_element_cache(protocol::NoStimulationProtocol, qr, sdh::SubDofHandler)
    return EmptyVolumetricElementCache()
end

"""
Describe the transmembrane stimulation by some analytical function on a given set of time intervals.
"""
struct AnalyticalTransmembraneStimulationProtocol{
    F <: AnalyticalCoefficient,
    T,
    VectorType <: AbstractVector{SVector{2, T}},
} <: TransmembraneStimulationProtocol
    f::F
    nonzero_intervals::VectorType # Helper for sparsity in time
end

function setup_element_cache(
    protocol::AnalyticalTransmembraneStimulationProtocol,
    qr,
    sdh::SubDofHandler,
)
    @assert length(sdh.dh.field_names) == 1 "Support for multiple fields not yet implemented."
    field_name = first(sdh.dh.field_names)
    ip = Ferrite.getfieldinterpolation(sdh, field_name)
    ip_geo = geometric_subdomain_interpolation(sdh)
    AnalyticalCoefficientElementCache(
        setup_coefficient_cache(protocol.f, qr, sdh),
        protocol.nonzero_intervals,
        CellValues(qr, ip, ip_geo), # TODO something more lightweight
    )
end

"""
The original model formulation (TODO citation) with the structure

 χCₘ∂ₜφₘ = ∇⋅κᵢ∇φᵢ + χ(Iᵢₒₙ(φₘ,𝐬,x) + Iₛₜᵢₘ,ᵢ(x,t))
 χCₘ∂ₜφₘ = ∇⋅κₑ∇φₑ - χ(Iᵢₒₙ(φₘ,𝐬,x) + Iₛₜᵢₘ,ₑ(x,t))
    ∂ₜ𝐬  = g(φₘ,𝐬,x)
 φᵢ - φₑ = φₘ

!!! warn
    Not implemented yet.
"""
struct ParabolicParabolicBidomainModel <: AbstractEPModel
    χ::Any
    Cₘ::Any
    κᵢ::Any
    κₑ::Any
    stim::AbstractStimulationProtocol
    ion::AbstractIonicModel
end

"""
Transformed bidomain model with the structure

 χCₘ∂ₜφₘ = ∇⋅κᵢ∇φₘ + ∇⋅κᵢ∇φₑ      + χ(Iᵢₒₙ(φₘ,𝐬,x) + Iₛₜᵢₘ(x,t))
      0  = ∇⋅κᵢ∇φₘ + ∇⋅(κᵢ+κₑ)∇φₑ +  Iₛₜᵢₘ,ₑ(t) - Iₛₜᵢₘ,ᵢ(t)
    ∂ₜ𝐬  = g(φₘ,𝐬,x)
      φᵢ = φₘ + φₑ

This formulation is a transformation of the parabolic-parabolic
form (c.f. TODO citation) and has been derived by (TODO citation) first.

!!! warn
    Not implemented yet.
"""
struct ParabolicEllipticBidomainModel <: AbstractEPModel
    χ::Any
    Cₘ::Any
    κᵢ::Any
    κₑ::Any
    stim::AbstractStimulationProtocol
    ion::AbstractIonicModel
end

"""
Simplification of the bidomain model with the structure

 χCₘ∂ₜφₘ = ∇⋅κ∇φₘ + χ(Iᵢₒₙ(φₘ,𝐬) + Iₛₜᵢₘ(t))
    ∂ₜ𝐬  = g(φₘ,𝐬)

(TODO citation). Can be derived through the assumption (TODO), but also when the
assumption is violated we can construct optimal κ (TODO citation+example) for the
reconstruction of φₘ.
"""
struct MonodomainModel{
    F1,
    F2,
    F3,
    STIM <: TransmembraneStimulationProtocol,
    ION <: AbstractIonicModel,
    CS,
} <: AbstractEPModel
    χ::F1
    Cₘ::F2
    κ::F3
    stim::STIM
    ion::ION
    # Which coordinate the cell model is handed as its `x`, as a coefficient -- a Cartesian position, a
    # generalized coordinate, a cell index. It sits here next to `κ` and `stim` because it is a modelling
    # choice like any other spatially varying input, and because a `Dict` of subdomain models can then
    # give a 1D Purkinje network and 3D tissue different coordinate systems. `nothing` when the cell model
    # ignores its coordinate.
    cell_coordinates::CS
    transmembrane_solution_symbol::Symbol
    internal_state_symbol::Symbol
end

MonodomainModel(χ, Cₘ, κ, stim, ion, φsym::Symbol, ssym::Symbol) =
    MonodomainModel(χ, Cₘ, κ, stim, ion, nothing, φsym, ssym)

get_field_variable_names(model::MonodomainModel) = (model.transmembrane_solution_symbol,)

reaction_solution_symbol(model::MonodomainModel) = model.transmembrane_solution_symbol

# The cell state is quadrature-point-local state in exactly the sense the solid mechanics materials mean
# it, so it is declared through the same interface. The transmembrane potential is excluded: it is a field
# the heat problem solves for, and it aliases one slot of every point's local state vector.
function gather_internal_variable_infos(model::MonodomainModel)
    return (InternalVariableInfo(model.internal_state_symbol, num_states(model.ion) - 1),)
end

"""
    internal_state_components(model) -> Vector{Int}

Which slots of a point's local state vector the model's *internal* state occupies, i.e. everything except
the transmembrane potential. Not necessarily contiguous - `ParametrizedAlievPanfilovModel` carries the
potential at index 2 of 2, so its internal state is slot 1.
"""
function internal_state_components(model::MonodomainModel)
    ion = reaction_model(model)
    φidx = transmembranepotential_index(ion)
    return [i for i = 1:num_states(ion) if i != φidx]
end

"""
    ReactionDiffusionSplit(model)

Annotation for the classical reaction-diffusion split of a given model.

The coordinate handed to the reaction model lives on the model itself
([`MonodomainModel`](@ref)'s `cell_coordinates`), not here, so that it survives any other split
annotation and can differ between subdomains.
"""
struct ReactionDiffusionSplit{mType}
    model::mType
end

include("cells/aliev-panfilov.jl")
include("cells/fhn.jl")
include("cells/pcg2019.jl")
