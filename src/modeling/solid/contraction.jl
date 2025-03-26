"""
TODO wrap the existing models into this one
"""
abstract type SteadyStateSarcomereModel <: AbstractInternalModel end

abstract type AbstractInternalMaterialStateCache end

# Material states without evolution equations. I.e. the states variables are at most a function of space (reference) and time.
abstract type TrivialInternalMaterialStateCache <: AbstractInternalMaterialStateCache end

abstract type RateIndependentMaterialStateCache <: AbstractInternalMaterialStateCache end

# Base.@kwdef struct SarcomereModel{ModelType, TCF, TSL}
#     model::ModelType
#     calcium_field::TCF
#     sarcomere_length::TSL
# end

# Base.@kwdef struct RateDependentSarcomereModel{ModelType, TCF, TSL, TSV}
#     model::ModelType
#     calcium_field::TCF
#     sarcomere_length::TSL
#     sarcomere_velocity::TSV
# end

"""
TODO citation pelce paper

TODO remove explicit calcium field dependence

!!! warning
    It should be highlighted that this model directly gives the steady state
    for the active stretch.
"""
Base.@kwdef struct PelceSunLangeveld1995Model{TD, CF} <: SteadyStateSarcomereModel
    β::TD = 3.0
    λᵃₘₐₓ::TD = 0.7
    calcium_field::CF
end

function compute_λᵃ(Ca, mp::PelceSunLangeveld1995Model)
    @unpack β, λᵃₘₐₓ = mp
    f(c) = c > 0.0 ? 0.5 + atan(β*log(c))/π  : 0.0
    return 1.0 / (1.0 + f(Ca)*(1.0/λᵃₘₐₓ - 1.0))
end

function gather_internal_variable_infos(model::PelceSunLangeveld1995Model)
    return InternalVariableInfo(:s, 0)
end

function default_initial_condition!(u::AbstractVector, model::PelceSunLangeveld1995Model)
    return nothing
end

"""
"""
struct PelceSunLangeveld1995Cache{CF} <: TrivialInternalMaterialStateCache
    calcium_cache::CF
end

function state(model_cache::PelceSunLangeveld1995Cache, geometry_cache, qp::QuadraturePoint, time)
    return evaluate_coefficient(model_cache.calcium_cache, geometry_cache, qp, time)
end

function setup_contraction_model_cache(contraction_model::PelceSunLangeveld1995Model, qr::QuadratureRule, sdh::SubDofHandler)
    return PelceSunLangeveld1995Cache(
        setup_coefficient_cache(contraction_model.calcium_field, qr, sdh)
    )
end

update_contraction_model_cache!(cache::PelceSunLangeveld1995Cache, time, cell, cv) = nothing

function 𝓝(state, F, coefficients, mp::PelceSunLangeveld1995Model)
    return state
end

"""
TODO remove explicit calcium field dependence
"""
Base.@kwdef struct ConstantStretchModel{TD, CF} <: SteadyStateSarcomereModel
    λ::TD = 1.0
    calcium_field::CF
end
compute_λᵃ(Ca, mp::ConstantStretchModel) = mp.λ

struct ConstantStretchCache{CF} <: TrivialInternalMaterialStateCache
    calcium_cache::CF
end

function state(model_cache::ConstantStretchCache, geometry_cache, qp::QuadraturePoint, time)
    return evaluate_coefficient(model_cache.calcium_cache, geometry_cache, qp, time)
end

function setup_contraction_model_cache(contraction_model::ConstantStretchModel, qr::QuadratureRule, sdh::SubDofHandler)
    return ConstantStretchCache(
        setup_coefficient_cache(contraction_model.calcium_field, qr, sdh)
    )
end

update_contraction_model_cache!(cache::ConstantStretchCache, time, cell, cv) = nothing

𝓝(state, F, coefficients, mp::ConstantStretchModel) = state

@doc raw"""
Mean-field variant of the sarcomere model presented by [RegDedQua:2020:bdm](@citet).

The default parameters for the model are taken from the same paper for human cardiomyocytes at body temperature.

!!! note
    For this model in combination with active stress framework the assumption is
    taken that the sarcomere stretch is exactly equivalent to $\sqrt{I^f_4}$.

!!! note
    In ,,Active contraction of cardiac cells: a reduced model for sarcomere
    dynamics with cooperative interactions'' there is a proposed evolution law for
    active stretches.
"""
Base.@kwdef struct RDQ20MFModel{TD, TCF}
    # Geoemtric parameters
    LA::TD = 1.25 # µm
    LM::TD = 1.65 # µm
    LB::TD = 0.18 # µm
    SL₀::TD = 2.2 # µm
    # RU steady state parameter
    Q::TD = 2.0 # unitless
    Kd₀::TD = 0.381 # µm
    αKd::TD = -0.571 # # µM/µm
    μ::TD = 10.0 # unitless
    γ::TD = 12.0 # unitless
    # RU kinetics parameters
    Koff::TD = 0.1 # 1/ms
    Kbasic::TD = 0.013 # 1/ms
    # XB cycling paramerters
    r₀::TD = 0.13431 # 1/ms
    α::TD = 25.184 # unitless
    μ₀_fP::TD = 0.032653 # 1/ms
    μ₁_fP::TD = 0.000778 # 1/ms
    # Upscaling parameter
    a_XB::TD = 22.894e3 # kPa
    #
    calcium_field::TCF
end

struct RDQ20MFCache{CF} <: RateIndependentMaterialStateCache
    calcium_cache::CF
end

function state(model_cache::RDQ20MFModel, geometry_cache, qp::QuadraturePoint, time)
    return evaluate_coefficient(model_cache.calcium_cache, geometry_cache, qp, time)
end

function setup_contraction_model_cache(contraction_model::RDQ20MFModel, qr::QuadratureRule, sdh::SubDofHandler)
    return RDQ20MFCache(
        setup_coefficient_cache(contraction_model.calcium_field, qr, sdh)
    )
end


function default_initial_condition!(u::AbstractVector, model::RDQ20MFModel)
    u[1] = 1.0
    u[2:end] .= 0.0
end

num_states(model::RDQ20MFModel) = 20

function rhs_fast!(dRU, uRU::SArray{Tuple{2,2,2,2}}, λ, Ca, t, p::RDQ20MFModel)
    # Initialize helper rates
    sarcomere_length = p.SL₀*λ
    dC1 = p.Koff / (p.Kd₀ - p.αKd * (2.15 - sarcomere_length)) * Ca
    dC = @SMatrix [
        dC1     dC1;
        p.Koff  p.Koff / p.μ
    ]

    dT = @MArray zeros(typeof(p.Q * p.Kbasic * p.γ / p.µ),2,2,2,2)
    @inbounds for TL ∈ 1:2, TR ∈ 1:2
        permissive_neighbors = TL + TR - 2
        dT[TL,2,TR,1] = p.Kbasic * p.γ^(2-permissive_neighbors);
        dT[TL,2,TR,2] = p.Kbasic * p.γ^(2-permissive_neighbors);
        dT[TL,1,TR,1] = p.Q * p.Kbasic * p.γ^permissive_neighbors / p.μ;
        dT[TL,1,TR,2] = p.Q * p.Kbasic * p.γ^permissive_neighbors;
    end

    # Update RU
    ## Compute fluxes associated with center unit
    ΦT_C = @SArray [uRU[TL,TC,TR,CC] * dT[TL,TC,TR,CC] for TL ∈ 1:2, TC ∈ 1:2, TR ∈ 1:2, CC ∈ 1:2]
    ΦC_C = @SArray [uRU[TL,TC,TR,CC] * dC[CC,TC] for TL ∈ 1:2, TC ∈ 1:2, TR ∈ 1:2, CC ∈ 1:2]

    ## Sum probabilities over CC
    suRU4  = sum(uRU;  dims=4)
    sΦT_C4 = sum(ΦT_C; dims=4)
    
    ## Compute rates associated with left unit
    flux_tot_L = sum(sΦT_C4; dims=3)
    prob_tot_L = sum(suRU4;  dims=3)
    dT_L = @SMatrix [prob_tot_L[TL,TC,1,1] > 1e-12 ? flux_tot_L[TL,TC,1,1] / prob_tot_L[TL,TC,1,1] : 0.0 for TL ∈ 1:2, TC ∈ 1:2]

    ## Compute rates associated with right unit
    flux_tot_R = sum(sΦT_C4; dims=1)
    prob_tot_R = sum(suRU4;  dims=1)
    dT_R = @SMatrix [prob_tot_R[1,TC,TR,1] > 1e-12 ? flux_tot_R[1,TC,TR,1] / prob_tot_R[1,TC,TR,1] : 0.0 for TR ∈ 1:2, TC ∈ 1:2]

    ## Compute fluxes associated with external units
    ## TODO investigate why the indices of dT_L and dT_R are flipped
    ΦT_L = @SArray [uRU[TL,TC,TR,CC] * dT_L[TC,TL] for TL ∈ 1:2, TC ∈ 1:2, TR ∈ 1:2, CC ∈ 1:2]
    ΦT_R = @SArray [uRU[TL,TC,TR,CC] * dT_R[TC,TR]  for TL ∈ 1:2, TC ∈ 1:2, TR ∈ 1:2, CC ∈ 1:2]

    ## Store RU rates
    @inbounds for TL ∈ 1:2, TC ∈ 1:2, TR ∈ 1:2, CC ∈ 1:2
        dRU[TL,TC,TR,CC] = 
            -ΦT_L[TL,TC,TR,CC] + ΦT_L[3 - TL,TC,TR,CC] -
             ΦT_C[TL,TC,TR,CC] + ΦT_C[TL,3 - TC,TR,CC] -
             ΦT_R[TL,TC,TR,CC] + ΦT_R[TL,TC,3 - TR,CC] -
             ΦC_C[TL,TC,TR,CC] + ΦC_C[TL,TC,TR,3 - CC]
    end

    return nothing
end

function rhs_fast(uRU::SArray{Tuple{2,2,2,2}}, λ, Ca, t, p::RDQ20MFModel)
    # Initialize helper rates
    sarcomere_length = p.SL₀*λ
    dC1 = p.Koff / (p.Kd₀ - p.αKd * (2.15 - sarcomere_length)) * Ca
    dC = @SMatrix [
        dC1     dC1;
        p.Koff  p.Koff / p.μ
    ]

    dT = @MArray zeros(typeof(p.Q * p.Kbasic * p.γ / p.µ),2,2,2,2)
    @inbounds for TL ∈ 1:2, TR ∈ 1:2
        permissive_neighbors = TL + TR - 2
        dT[TL,2,TR,1] = p.Kbasic * p.γ^(2-permissive_neighbors);
        dT[TL,2,TR,2] = p.Kbasic * p.γ^(2-permissive_neighbors);
        dT[TL,1,TR,1] = p.Q * p.Kbasic * p.γ^permissive_neighbors / p.μ;
        dT[TL,1,TR,2] = p.Q * p.Kbasic * p.γ^permissive_neighbors;
    end

    # Update RU
    ## Compute fluxes associated with center unit
    ΦT_C = @SArray [uRU[TL,TC,TR,CC] * dT[TL,TC,TR,CC] for TL ∈ 1:2, TC ∈ 1:2, TR ∈ 1:2, CC ∈ 1:2]
    ΦC_C = @SArray [uRU[TL,TC,TR,CC] * dC[CC,TC] for TL ∈ 1:2, TC ∈ 1:2, TR ∈ 1:2, CC ∈ 1:2]

    ## Sum probabilities over CC
    suRU4  = sum(uRU;  dims=4)
    sΦT_C4 = sum(ΦT_C; dims=4)

    ## Compute rates associated with left unit
    flux_tot_L = sum(sΦT_C4; dims=3)
    prob_tot_L = sum(suRU4;  dims=3)
    dT_L = @SMatrix [prob_tot_L[TL,TC,1,1] > 1e-12 ? flux_tot_L[TL,TC,1,1] / prob_tot_L[TL,TC,1,1] : 0.0 for TL ∈ 1:2, TC ∈ 1:2]

    ## Compute rates associated with right unit
    flux_tot_R = sum(sΦT_C4; dims=1)
    prob_tot_R = sum(suRU4;  dims=1)
    dT_R = @SMatrix [prob_tot_R[1,TC,TR,1] > 1e-12 ? flux_tot_R[1,TC,TR,1] / prob_tot_R[1,TC,TR,1] : 0.0 for TR ∈ 1:2, TC ∈ 1:2]

    ## Compute fluxes associated with external units
    ΦT_L = @SArray [uRU[TL,TC,TR,CC] * dT_L[TC,TL] for TL ∈ 1:2, TC ∈ 1:2, TR ∈ 1:2, CC ∈ 1:2]
    ΦT_R = @SArray [uRU[TL,TC,TR,CC] * dT_R[TC,TR]  for TL ∈ 1:2, TC ∈ 1:2, TR ∈ 1:2, CC ∈ 1:2]

    ## Return RU rates
    return @SArray [
        -ΦT_L[TL,TC,TR,CC] + ΦT_L[3 - TL,TC,TR,CC] -
         ΦT_C[TL,TC,TR,CC] + ΦT_C[TL,3 - TC,TR,CC] -
         ΦT_R[TL,TC,TR,CC] + ΦT_R[TL,TC,3 - TR,CC] -
         ΦC_C[TL,TC,TR,CC] + ΦC_C[TL,TC,TR,3 - CC]
    for TL ∈ 1:2, TC ∈ 1:2, TR ∈ 1:2, CC ∈ 1:2]
end

function rhs_fast_dλ!(dRU, uRU::AbstractArray{T}, λ, Ca, t, p::RDQ20MFModel) where T
    ΦC_C = @MArray zeros(T,2,2,2,2)
    dC = @MMatrix zeros(T,2,2)

    # Initialize helper rates
    sarcomere_length = p.SL₀*λ
    dC[1,1] = - p.SL₀*p.αKd * p.Koff / (p.Kd₀ - p.αKd * (2.15 - sarcomere_length))^2 * Ca
    dC[1,2] = - p.SL₀*p.αKd * p.Koff / (p.Kd₀ - p.αKd * (2.15 - sarcomere_length))^2 * Ca
    # dC[2,1] = 0.0
    # dC[2,2] = 0.0

    # Update RU
    ## Compute fluxes associated with center unit
    @inbounds for TL ∈ 1:2, TC ∈ 1:2, TR ∈ 1:2, CC ∈ 1:1
        ΦC_C[TL,TC,TR,CC] = uRU[TL,TC,TR,CC] * dC[CC,TC]
    end

    # Store RU rates
    @inbounds for TL ∈ 1:2, TC ∈ 1:2, TR ∈ 1:2, CC ∈ 1:2
        dRU[TL,TC,TR,CC] = -ΦC_C[TL,TC,TR,CC] + ΦC_C[TL,TC,TR,3 - CC]
    end

    return dRU
end

function sarcomere_rhs!(du, u, λ, dλdt, Ca, t, p::RDQ20MFModel)
    # Direct translation from https://github.com/FrancescoRegazzoni/cardiac-activation/blob/master/models_cpp/model_RDQ20_MF.cpp
    uRU_flat = @view u[1:16]
    uRU = SArray{Tuple{2,2,2,2}}(reshape(uRU_flat, (2,2,2,2)))
    uXB = @view u[17:20]

    dRU_flat = @view du[1:16]
    dRU = reshape(dRU_flat, (2,2,2,2))
    dXB = @view du[17:20]

    rhs_fast!(dRU, uRU, λ, Ca, t, p)

    dT = @MArray zeros(2,2,2,2)
    @inbounds for TL ∈ 1:2, TR ∈ 1:2
        permissive_neighbors = TL + TR - 2
        dT[TL,2,TR,1] = p.Kbasic * p.γ^(2-permissive_neighbors);
        dT[TL,2,TR,2] = p.Kbasic * p.γ^(2-permissive_neighbors);
        dT[TL,1,TR,1] = p.Q * p.Kbasic * p.γ^permissive_neighbors / p.μ;
        dT[TL,1,TR,2] = p.Q * p.Kbasic * p.γ^permissive_neighbors;
    end

    flux_PN = 0.0
    flux_NP = 0.0
    permissivity = 0.0
    @inbounds for TL ∈ 1:2, TR ∈ 1:2, CC ∈ 1:2
        permissivity += uRU[TL,2,TR,CC]
        flux_PN += uRU[TL,2,TR,CC] * dT[TL,2,TR,CC]
        flux_NP += uRU[TL,1,TR,CC] * dT[TL,1,TR,CC]
    end

    k_PN = if permissivity >= 1e-12
        flux_PN / permissivity
    else
        0.0
    end
    k_NP = if 1.0 - permissivity >= 1e-12
        flux_NP / (1.0 - permissivity)
    else
        0.0
    end

    #           q(v) = α|v|
    r = p.r₀ + p.α*abs(dλdt)
    diag_P = r + k_PN
    diag_N = r + k_NP

    XB_A = @SMatrix [
        -diag_P      0.0     k_NP      0.0
           dλdt  -diag_P      0.0     k_NP
           k_PN      0.0  -diag_N      0.0
            0.0     k_PN     dλdt  -diag_N
    ];

    dXB .= XB_A*uXB
    dXB[1] += p.μ₀_fP * permissivity
    dXB[2] += p.μ₁_fP * permissivity
end

function fraction_single_overlap(model::RDQ20MFModel, λ)
    SL = λ*model.SL₀
    LMh = (model.LM - model.LB) * 0.5;
    if (SL > model.LA && SL <= model.LM)
      return (SL - model.LA) / LMh;
    elseif (SL > model.LM && SL <= 2 * model.LA - model.LB)
      return (SL + model.LM - 2 * model.LA) * 0.5 / LMh;
    elseif (SL > 2 * model.LA - model.LB && SL <= 2 * model.LA + model.LB)
      return 1.0;
    elseif (SL > 2 * model.LA + model.LB && SL <= 2 * model.LA + model.LM)
      return (model.LM + 2 * model.LA - SL) * 0.5 / LMh;
    else
      return 0.0;
    end
end

function compute_active_tension(model::RDQ20MFModel, state, sarcomere_stretch)
    model.a_XB * (state[18] + state[20]) * fraction_single_overlap(model, sarcomere_stretch)
end

function compute_active_stiffness(model::RDQ20MFModel, state, sarcomere_stretch)
    model.a_XB * (state[17] + state[19]) * fraction_single_overlap(model, sarcomere_stretch)
end

function gather_internal_variable_infos(model::RDQ20MFModel)
    return InternalVariableInfo(:s, 20)
end

function 𝓝(state, F, coefficients, model::RDQ20MFModel)
    f = F ⋅ coefficients.f
    sarcomere_stretch = √(f ⋅ f)
    return (state[18] + state[20]) * fraction_single_overlap(model, sarcomere_stretch)
end
