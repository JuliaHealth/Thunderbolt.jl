abstract type AbstractSarcomereModel <: AbstractInternalModel end
abstract type AbstractSteadyStateSarcomereModel <: AbstractSarcomereModel end
abstract type AbstractRateIndependentSarcomereModel <: AbstractSarcomereModel end
abstract type AbstractRateDependentSarcomereModel <: AbstractSarcomereModel end

abstract type AbstractCondensationMaterialStateCache end

# Material states without evolution equations. I.e. the states variables are at most a function of space (reference) and time.
abstract type TrivialCondensationMaterialStateCache <: AbstractCondensationMaterialStateCache end
struct EmptyTrivialCondensationMaterialStateCache <: TrivialCondensationMaterialStateCache end


abstract type RateIndependentCondensationMaterialStateCache <:
              AbstractCondensationMaterialStateCache end
struct EmptyRateIndependentCondensationMaterialStateCache <:
       RateIndependentCondensationMaterialStateCache end


abstract type RateDependentCondensationMaterialStateCache <: AbstractCondensationMaterialStateCache end
struct EmptyRateDependentCondensationMaterialStateCache <:
       RateDependentCondensationMaterialStateCache end


# Most models do not need additional scratch space for the evaluation
function setup_contraction_model_cache(
        contraction_model::AbstractSteadyStateSarcomereModel,
        qr::QuadratureRule,
        sdh::SubDofHandler
)
    return EmptyTrivialCondensationMaterialStateCache()
end
function duplicate_for_device(device, cache::EmptyTrivialCondensationMaterialStateCache)
    return EmptyTrivialCondensationMaterialStateCache()
end
function setup_contraction_model_cache(
        contraction_model::AbstractRateIndependentSarcomereModel,
        qr::QuadratureRule,
        sdh::SubDofHandler
)
    return EmptyRateIndependentCondensationMaterialStateCache()
end
function duplicate_for_device(device, cache::EmptyRateIndependentCondensationMaterialStateCache)
    return EmptyRateIndependentCondensationMaterialStateCache()
end
function setup_contraction_model_cache(
        contraction_model::AbstractRateDependentSarcomereModel,
        qr::QuadratureRule,
        sdh::SubDofHandler
)
    return EmptyRateDependentCondensationMaterialStateCache()
end
function duplicate_for_device(device, cache::EmptyRateDependentCondensationMaterialStateCache)
    return EmptyRateDependentCondensationMaterialStateCache()
end

# Some defaults
function default_initial_state!(Q::AbstractVector, model::AbstractSteadyStateSarcomereModel)
    return nothing
end
function gather_internal_variable_infos(model::AbstractSteadyStateSarcomereModel)
    return nothing
end
function 𝓝(state, F, coefficients, mp::AbstractSteadyStateSarcomereModel)
    return state
end

# Ignore rate dependency of a rate dependent model to emulate rate independency.
struct RateIndependentSarcomereModelWrapper{T <: AbstractRateDependentSarcomereModel} <:
       AbstractRateIndependentSarcomereModel
    model::T
end
function setup_contraction_model_cache(
        wrapper::RateIndependentSarcomereModelWrapper,
        qr::QuadratureRule,
        sdh::SubDofHandler
)
    @assert setup_contraction_model_cache(wrapper.model, qr, sdh) isa
            EmptyRateDependentCondensationMaterialStateCache "Wrapping non-trivial material state caches not supported."
    return EmptyRateIndependentCondensationMaterialStateCache()
end
num_states(wrapper::RateIndependentSarcomereModelWrapper) = num_states(wrapper.model)

# We use this for testing and fitting purposes
Base.@kwdef struct StandaloneSarcomereModel{ModelType, CaF, SF, VF}
    model::ModelType
    calcium::CaF
    fiber_stretch::SF
    fiber_velocity::VF
end
function (model_wrapper::StandaloneSarcomereModel)(du, u, p, t)
    λ    = model_wrapper.fiber_stretch(t)
    dλdt = model_wrapper.fiber_velocity(t)
    Ca   = model_wrapper.calcium(t)
    sarcomere_rhs!(du, u, λ, dλdt, Ca, t, model_wrapper.model)
end
num_states(wrapper::StandaloneSarcomereModel) = num_states(wrapper.model)

# Wrapper to mark a sarcomere model as an internal model
Base.@kwdef struct CaDrivenInternalSarcomereModel{ModelType, CalciumFieldType} <:
                   AbstractSarcomereModel
    model::ModelType
    calcium_field::CalciumFieldType
end
num_states(wrapper::CaDrivenInternalSarcomereModel) = num_states(wrapper.model)
function 𝓝(state, F, coefficients, wrapper::CaDrivenInternalSarcomereModel)
    𝓝(state, F, coefficients, wrapper.model)
end
compute_λᵃ(state, wrapper::CaDrivenInternalSarcomereModel) = compute_λᵃ(state, wrapper.model)
function sarcomere_rhs!(dQ, Q, λ, dλdt, Ca, time, wrapper::CaDrivenInternalSarcomereModel)
    sarcomere_rhs!(dQ, Q, λ, dλdt, Ca, time, wrapper.model)
end
function gather_internal_variable_infos(wrapper::CaDrivenInternalSarcomereModel)
    gather_internal_variable_infos(wrapper.model)
end
function default_initial_state!(Q::AbstractVector, wrapper::CaDrivenInternalSarcomereModel)
    default_initial_state!(Q, wrapper.model)
end

struct TrivialCaDrivenCondensationSarcomereCache{ModelType, ModelCacheType, CalciumCacheType} <:
       TrivialCondensationMaterialStateCache
    model::ModelType
    model_cache::ModelCacheType
    calcium_cache::CalciumCacheType
end
function duplicate_for_device(device, cache::TrivialCaDrivenCondensationSarcomereCache)
    return TrivialCaDrivenCondensationSarcomereCache(
        duplicate_for_device(device, cache.model),
        duplicate_for_device(device, cache.model_cache),
        duplicate_for_device(device, cache.calcium_cache)
    )
end
struct RateIndependentCaDrivenCondensationSarcomereCache{
    ModelType,
    ModelCacheType,
    CalciumCacheType
} <: RateIndependentCondensationMaterialStateCache
    model::ModelType
    model_cache::ModelCacheType
    calcium_cache::CalciumCacheType
end
function duplicate_for_device(device, cache::RateIndependentCaDrivenCondensationSarcomereCache)
    return RateIndependentCaDrivenCondensationSarcomereCache(
        duplicate_for_device(device, cache.model),
        duplicate_for_device(device, cache.model_cache),
        duplicate_for_device(device, cache.calcium_cache)
    )
end
struct RateDependentCaDrivenCondensationSarcomereCache{
    ModelType,
    ModelCacheType,
    CalciumCacheType
} <: RateDependentCondensationMaterialStateCache
    model::ModelType
    model_cache::ModelCacheType
    calcium_cache::CalciumCacheType
end
function duplicate_for_device(device, cache::RateDependentCaDrivenCondensationSarcomereCache)
    return RateDependentCaDrivenCondensationSarcomereCache(
        duplicate_for_device(device, cache.model),
        duplicate_for_device(device, cache.model_cache),
        duplicate_for_device(device, cache.calcium_cache)
    )
end
function setup_contraction_model_cache(
        wrapper::CaDrivenInternalSarcomereModel,
        qr::QuadratureRule,
        sdh::SubDofHandler
)
    return setup_contraction_model_cache_from_wrapper(wrapper.model, wrapper.calcium_field, qr, sdh)
end
function setup_contraction_model_cache_from_wrapper(
        model::AbstractSteadyStateSarcomereModel,
        calcium_field,
        qr::QuadratureRule,
        sdh::SubDofHandler
)
    return TrivialCaDrivenCondensationSarcomereCache(
        model,
        setup_contraction_model_cache(model, qr, sdh),
        setup_coefficient_cache(calcium_field, qr, sdh)
    )
end
function setup_contraction_model_cache_from_wrapper(
        model::AbstractRateIndependentSarcomereModel,
        calcium_field,
        qr::QuadratureRule,
        sdh::SubDofHandler
)
    return RateIndependentCaDrivenCondensationSarcomereCache(
        model,
        setup_contraction_model_cache(model, qr, sdh),
        setup_coefficient_cache(calcium_field, qr, sdh)
    )
end
function setup_contraction_model_cache_from_wrapper(
        model::AbstractRateDependentSarcomereModel,
        calcium_field,
        qr::QuadratureRule,
        sdh::SubDofHandler
)
    return RateDependentCaDrivenCondensationSarcomereCache(
        model,
        setup_contraction_model_cache(model, qr, sdh),
        setup_coefficient_cache(calcium_field, qr, sdh)
    )
end

# Evalaute wrapped model's state
function state(
        wrapper_cache::TrivialCaDrivenCondensationSarcomereCache,
        geometry_cache,
        qp::QuadraturePoint,
        time
)
    return state(wrapper_cache, wrapper_cache.model, geometry_cache, qp, time)
end
# Should usually be just the calcium state
function state(
        wrapper_cache::TrivialCaDrivenCondensationSarcomereCache,
        model::AbstractSteadyStateSarcomereModel,
        geometry_cache,
        qp::QuadraturePoint,
        time
)
    return evaluate_coefficient(wrapper_cache.calcium_cache, geometry_cache, qp, time)
end

"""
TODO citation pelce paper

!!! warning
    It should be highlighted that this model directly gives the steady state
    for the active stretch.
"""
Base.@kwdef struct PelceSunLangeveld1995Model{TD} <: AbstractSteadyStateSarcomereModel
    β::TD = 3.0
    λᵃₘₐₓ::TD = 0.7
end
num_states(::PelceSunLangeveld1995Model) = 0
function compute_λᵃ(Ca, mp::PelceSunLangeveld1995Model)
    @unpack β, λᵃₘₐₓ = mp
    f(c) = c > 0.0 ? 0.5 + atan(β * log(c)) / π : 0.0
    return 1.0 / (1.0 + f(Ca) * (1.0 / λᵃₘₐₓ - 1.0))
end

"""
    Debug model applying a constant stretch state.
"""
Base.@kwdef struct ConstantStretchModel{TD} <: AbstractSteadyStateSarcomereModel
    λ::TD = 1.0
end
num_states(::ConstantStretchModel) = 0
compute_λᵃ(Ca, mp::ConstantStretchModel) = mp.λ


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
Base.@kwdef struct RDQ20MFModel{TD} <: AbstractRateDependentSarcomereModel
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
end

function default_initial_state!(Q::AbstractVector, model::RDQ20MFModel)
    # This should always work fine
    Q[1] = 1.0
    Q[2:end] .= 0.0
end
num_states(model::RDQ20MFModel) = 20

function rhs_fast!(dRU, uRU::SArray{Tuple{2, 2, 2, 2}}, λ, Ca, t, p::RDQ20MFModel)
    # Initialize helper rates
    sarcomere_length = p.SL₀ * λ
    dC1 = p.Koff / (p.Kd₀ - p.αKd * (2.15 - sarcomere_length)) * Ca
    dC = @SMatrix [dC1 dC1;
                   p.Koff p.Koff/p.μ]

    dT = @MArray zeros(typeof(p.Q * p.Kbasic * p.γ / p.µ), 2, 2, 2, 2)
    @inbounds for TL in 1:2, TR in 1:2
        permissive_neighbors = TL + TR - 2
        dT[TL, 2, TR, 1] = p.Kbasic * p.γ^(2 - permissive_neighbors)
        dT[TL, 2, TR, 2] = p.Kbasic * p.γ^(2 - permissive_neighbors)
        dT[TL, 1, TR, 1] = p.Q * p.Kbasic * p.γ^permissive_neighbors / p.μ
        dT[TL, 1, TR, 2] = p.Q * p.Kbasic * p.γ^permissive_neighbors
    end

    # Update RU
    ## Compute fluxes associated with center unit
    ΦT_C = @SArray [uRU[TL, TC, TR, CC] * dT[TL, TC, TR, CC]
                    for TL in 1:2, TC in 1:2, TR in 1:2, CC in 1:2]
    ΦC_C = @SArray [uRU[TL, TC, TR, CC] * dC[CC, TC] for TL in 1:2, TC in 1:2, TR in 1:2, CC in 1:2]

    ## Sum probabilities over CC
    suRU4 = sum(uRU; dims = 4)
    sΦT_C4 = sum(ΦT_C; dims = 4)

    ## Compute rates associated with left unit
    flux_tot_L = sum(sΦT_C4; dims = 3)
    prob_tot_L = sum(suRU4; dims = 3)
    dT_L = @SMatrix [prob_tot_L[TL, TC, 1, 1] > 1e-12 ?
                     flux_tot_L[TL, TC, 1, 1] / prob_tot_L[TL, TC, 1, 1] :
                     0.0 for TL in 1:2, TC in 1:2]

    ## Compute rates associated with right unit
    flux_tot_R = sum(sΦT_C4; dims = 1)
    prob_tot_R = sum(suRU4; dims = 1)
    dT_R = @SMatrix [prob_tot_R[1, TC, TR, 1] > 1e-12 ?
                     flux_tot_R[1, TC, TR, 1] / prob_tot_R[1, TC, TR, 1] :
                     0.0 for TR in 1:2, TC in 1:2]

    ## Compute fluxes associated with external units
    ## TODO investigate why the indices of dT_L and dT_R are flipped
    ΦT_L = @SArray [uRU[TL, TC, TR, CC] * dT_L[TC, TL]
                    for TL in 1:2, TC in 1:2, TR in 1:2, CC in 1:2]
    ΦT_R = @SArray [uRU[TL, TC, TR, CC] * dT_R[TC, TR]
                    for TL in 1:2, TC in 1:2, TR in 1:2, CC in 1:2]

    ## Store RU rates
    @inbounds for TL in 1:2, TC in 1:2, TR in 1:2, CC in 1:2
        dRU[TL, TC, TR, CC] = -ΦT_L[TL, TC, TR, CC] + ΦT_L[3 - TL, TC, TR, CC] -
                              ΦT_C[TL, TC, TR, CC] +
                              ΦT_C[TL, 3 - TC, TR, CC] - ΦT_R[TL, TC, TR, CC] +
                              ΦT_R[TL, TC, 3 - TR, CC] -
                              ΦC_C[TL, TC, TR, CC] + ΦC_C[TL, TC, TR, 3 - CC]
    end

    return nothing
end

function rhs_fast(uRU::SArray{Tuple{2, 2, 2, 2}}, λ, Ca, t, p::RDQ20MFModel)
    # Initialize helper rates
    sarcomere_length = p.SL₀ * λ
    dC1 = p.Koff / (p.Kd₀ - p.αKd * (2.15 - sarcomere_length)) * Ca
    dC = @SMatrix [dC1 dC1;
                   p.Koff p.Koff/p.μ]

    dT = @MArray zeros(typeof(p.Q * p.Kbasic * p.γ / p.µ), 2, 2, 2, 2)
    @inbounds for TL in 1:2, TR in 1:2
        permissive_neighbors = TL + TR - 2
        dT[TL, 2, TR, 1] = p.Kbasic * p.γ^(2 - permissive_neighbors)
        dT[TL, 2, TR, 2] = p.Kbasic * p.γ^(2 - permissive_neighbors)
        dT[TL, 1, TR, 1] = p.Q * p.Kbasic * p.γ^permissive_neighbors / p.μ
        dT[TL, 1, TR, 2] = p.Q * p.Kbasic * p.γ^permissive_neighbors
    end

    # Update RU
    ## Compute fluxes associated with center unit
    ΦT_C = @SArray [uRU[TL, TC, TR, CC] * dT[TL, TC, TR, CC]
                    for TL in 1:2, TC in 1:2, TR in 1:2, CC in 1:2]
    ΦC_C = @SArray [uRU[TL, TC, TR, CC] * dC[CC, TC] for TL in 1:2, TC in 1:2, TR in 1:2, CC in 1:2]

    ## Sum probabilities over CC
    suRU4 = sum(uRU; dims = 4)
    sΦT_C4 = sum(ΦT_C; dims = 4)

    ## Compute rates associated with left unit
    flux_tot_L = sum(sΦT_C4; dims = 3)
    prob_tot_L = sum(suRU4; dims = 3)
    dT_L = @SMatrix [prob_tot_L[TL, TC, 1, 1] > 1e-12 ?
                     flux_tot_L[TL, TC, 1, 1] / prob_tot_L[TL, TC, 1, 1] :
                     0.0 for TL in 1:2, TC in 1:2]

    ## Compute rates associated with right unit
    flux_tot_R = sum(sΦT_C4; dims = 1)
    prob_tot_R = sum(suRU4; dims = 1)
    dT_R = @SMatrix [prob_tot_R[1, TC, TR, 1] > 1e-12 ?
                     flux_tot_R[1, TC, TR, 1] / prob_tot_R[1, TC, TR, 1] :
                     0.0 for TR in 1:2, TC in 1:2]

    ## Compute fluxes associated with external units
    ΦT_L = @SArray [uRU[TL, TC, TR, CC] * dT_L[TC, TL]
                    for TL in 1:2, TC in 1:2, TR in 1:2, CC in 1:2]
    ΦT_R = @SArray [uRU[TL, TC, TR, CC] * dT_R[TC, TR]
                    for TL in 1:2, TC in 1:2, TR in 1:2, CC in 1:2]

    ## Return RU rates
    return @SArray [-ΦT_L[TL, TC, TR, CC] + ΦT_L[3 - TL, TC, TR, CC] - ΦT_C[TL, TC, TR, CC] +
                    ΦT_C[TL, 3 - TC, TR, CC] - ΦT_R[TL, TC, TR, CC] + ΦT_R[TL, TC, 3 - TR, CC] -
                    ΦC_C[TL, TC, TR, CC] + ΦC_C[TL, TC, TR, 3 - CC]
                    for TL in 1:2, TC in 1:2, TR in 1:2, CC in 1:2]
end

function rhs_fast_dλ!(dRU, uRU::AbstractArray{T}, λ, Ca, t, p::RDQ20MFModel) where {T}
    ΦC_C = @MArray zeros(T, 2, 2, 2, 2)
    dC = @MMatrix zeros(T, 2, 2)

    # Initialize helper rates
    sarcomere_length = p.SL₀ * λ
    dC[1, 1] = -p.SL₀ * p.αKd * p.Koff / (p.Kd₀ - p.αKd * (2.15 - sarcomere_length))^2 * Ca
    dC[1, 2] = -p.SL₀ * p.αKd * p.Koff / (p.Kd₀ - p.αKd * (2.15 - sarcomere_length))^2 * Ca
    # dC[2,1] = 0.0
    # dC[2,2] = 0.0

    # Update RU
    ## Compute fluxes associated with center unit
    @inbounds for TL in 1:2, TC in 1:2, TR in 1:2, CC in 1:1
        ΦC_C[TL, TC, TR, CC] = uRU[TL, TC, TR, CC] * dC[CC, TC]
    end

    # Store RU rates
    @inbounds for TL in 1:2, TC in 1:2, TR in 1:2, CC in 1:2
        dRU[TL, TC, TR, CC] = -ΦC_C[TL, TC, TR, CC] + ΦC_C[TL, TC, TR, 3 - CC]
    end

    return dRU
end

function sarcomere_rhs!(du, u, λ, dλdt, Ca, t, p::RDQ20MFModel)
    # Direct translation from https://github.com/FrancescoRegazzoni/cardiac-activation/blob/master/models_cpp/model_RDQ20_MF.cpp
    uRU_flat = @view u[1:16]
    uRU = SArray{Tuple{2, 2, 2, 2}}(reshape(uRU_flat, (2, 2, 2, 2)))
    uXB = @view u[17:20]

    dRU_flat = @view du[1:16]
    dRU = reshape(dRU_flat, (2, 2, 2, 2))
    dXB = @view du[17:20]

    rhs_fast!(dRU, uRU, λ, Ca, t, p)

    dT = @MArray zeros(2, 2, 2, 2)
    @inbounds for TL in 1:2, TR in 1:2
        permissive_neighbors = TL + TR - 2
        dT[TL, 2, TR, 1] = p.Kbasic * p.γ^(2 - permissive_neighbors)
        dT[TL, 2, TR, 2] = p.Kbasic * p.γ^(2 - permissive_neighbors)
        dT[TL, 1, TR, 1] = p.Q * p.Kbasic * p.γ^permissive_neighbors / p.μ
        dT[TL, 1, TR, 2] = p.Q * p.Kbasic * p.γ^permissive_neighbors
    end

    flux_PN = 0.0
    flux_NP = 0.0
    permissivity = 0.0
    @inbounds for TL in 1:2, TR in 1:2, CC in 1:2
        permissivity += uRU[TL, 2, TR, CC]
        flux_PN += uRU[TL, 2, TR, CC] * dT[TL, 2, TR, CC]
        flux_NP += uRU[TL, 1, TR, CC] * dT[TL, 1, TR, CC]
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
    r = p.r₀ + p.α * abs(dλdt)
    diag_P = r + k_PN
    diag_N = r + k_NP

    XB_A = @SMatrix [-diag_P 0.0 k_NP 0.0
                     dλdt -diag_P 0.0 k_NP
                     k_PN 0.0 -diag_N 0.0
                     0.0 k_PN dλdt -diag_N]

    dXB .= XB_A * uXB
    dXB[1] += p.μ₀_fP * permissivity
    dXB[2] += p.μ₁_fP * permissivity
end

function fraction_single_overlap(model::RDQ20MFModel, λ)
    SL = λ * model.SL₀
    LMh = (model.LM - model.LB) * 0.5
    if (SL > model.LA && SL <= model.LM)
        return (SL - model.LA) / LMh
    elseif (SL > model.LM && SL <= 2 * model.LA - model.LB)
        return (SL + model.LM - 2 * model.LA) * 0.5 / LMh
    elseif (SL > 2 * model.LA - model.LB && SL <= 2 * model.LA + model.LB)
        return 1.0
    elseif (SL > 2 * model.LA + model.LB && SL <= 2 * model.LA + model.LM)
        return (model.LM + 2 * model.LA - SL) * 0.5 / LMh
    else
        return 0.0
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
