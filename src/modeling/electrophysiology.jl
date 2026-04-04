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

state_symbol(ionic_model::AbstractIonicModel, sidx::Int) = Symbol("s$sidx")

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

abstract type AbstractStimulationProtocol <: AbstractSourceTerm end;

@doc raw"""
Supertype for all stimulation protocols fulfilling $I_{\rm{stim,e}} = I_{\rm{stim,i}}$.
"""
abstract type TransmembraneStimulationProtocol <: AbstractStimulationProtocol end;

"""
A dummy protocol describing the absence of stimuli for a simulation.
"""
struct NoStimulationProtocol <: TransmembraneStimulationProtocol end

"""
Activation protocol for uniformally activating the endocardium with zero wave arrival time.
"""
struct UniformEndocardialActivationProtocol{CST} <: TransmembraneStimulationProtocol
    subdomains_offsets::Dict{String, Float64}
    cs::CST
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
} <: AbstractEPModel
    χ::F1
    Cₘ::F2
    κ::F3
    stim::STIM
    ion::ION
    # TODO the variables below should be queried from the ionic model
    transmembrane_solution_symbol::Symbol
    internal_state_symbol::Symbol
end

get_field_variable_names(model::MonodomainModel) = (model.transmembrane_solution_symbol,)

"""
    ReactionDiffusionSplit(model)
    ReactionDiffusionSplit(model, coeff)
Annotation for the classical reaction-diffusion split of a given model. The
second argument is a coefficient describing the input `x` for the reaction model rhs,
which is usually some generalized coordinate.
"""
struct ReactionDiffusionSplit{mType, csType}
    model::mType
    cs::csType
end

ReactionDiffusionSplit(model) = ReactionDiffusionSplit(model, nothing)

"""
    ReactionEikonalSplit(model, cs)
Annotation for the reaction-Eikonal split of a given model. The
argument `cs` is a coefficient describing the input `x` for the reaction model rhs,
which is usually some generalized coordinate.
"""
struct ReactionEikonalSplit{mType, csType}
    model::mType
    cs::csType
end

"""
    ReactionEikonalDiffusionSplit(model, cs)
Annotation for the reaction-Eikonal-diffusion split of a given model. The
argument `cs` is a coefficient describing the input `x` for the reaction model rhs,
which is usually some generalized coordinate.
"""
struct ReactionEikonalDiffusionSplit{mType, csType}
    model::mType
    cs::csType
end

"""
    StimulatedCellModel(;cell_model, [stim_offset=0.0, stim_length=1.0, stim_strength= 0.91])

Wrapper around ionic cell models that adds foot current according to [NeicCamposPrassl:2017:ECE](@citet)
"""
@kwdef struct StimulatedCellModel{TC, OffestT, T} <: AbstractIonicModel
    cell_model::TC
    stim_offset::OffestT = 0.0
    stim_length::T = 1.0
    stim_strength::T = 0.91
    τᶠ::T = 0.25
end
num_states(m::StimulatedCellModel) = num_states(m.cell_model)
default_initial_state(m::StimulatedCellModel) = default_initial_state(m.cell_model)
function cell_rhs!(du, u, i, x, t, m::StimulatedCellModel)
    cell_rhs!(du, u, i, nothing, t, m.cell_model)
    if m.stim_offset ≤ t ≤ m.stim_offset + m.stim_length
        idx = transmembranepotential_index(m.cell_model)
        du[idx] += m.stim_strength / m.τᶠ * exp((t - m.stim_offset) / m.τᶠ)
    end
    return nothing
end

function cell_rhs!(du, u, i, x, t, m::StimulatedCellModel{<:Any, <:AbstractArray})
    cell_rhs!(du, u, i, nothing, t, m.cell_model)
    if m.stim_offset[i] ≤ t ≤ m.stim_offset[i] + m.stim_length
        idx = transmembranepotential_index(m.cell_model)
        du[idx] += m.stim_strength / m.τᶠ * exp((t - m.stim_offset[i]) / m.τᶠ)
    end
    return nothing
end

include("cells/fhn.jl")
include("cells/pcg2019.jl")
