# TODO (FILE) I think we should change the design here. Instea of dispatching on Ψ we should make the material callable or equip it with a function.

abstract type AbstractMaterialModel end

function material_routine(material_model::AbstractMaterialModel, F::Tensor{2}, coefficient_cache, ::EmptyInternalCache, geometry_cache::Ferrite.CellCache, qp::QuadraturePoint, time)
    coefficients = evaluate_coefficient(coefficient_cache, geometry_cache, qp, time)
    return stress_and_tangent(material_model, F, coefficients, EmptyInternalModel())
end

function material_routine(material_model::AbstractMaterialModel, F::Tensor{2}, coefficient_cache, state_cache::TrivialInternalMaterialStateCache, geometry_cache::Ferrite.CellCache, qp::QuadraturePoint, time)
    coefficients = evaluate_coefficient(coefficient_cache, geometry_cache, qp, time)
    Q = state(state_cache, geometry_cache, qp, time)
    return stress_and_tangent(material_model, F, coefficients, Q)
end

function material_routine(material_model::AbstractMaterialModel, F::Tensor{2}, coefficient_cache, state_cache::RateIndependentMaterialStateCache, geometry_cache::Ferrite.CellCache, qp::QuadraturePoint, time)
    coefficients = evaluate_coefficient(coefficient_cache, geometry_cache, qp, time)
    Q, ∂P∂QdQdF  = solve_local_constraint(F, coefficients, material_model, state_cache, geometry_cache, qp, time)
    P, ∂P∂F      = stress_and_tangent(material_model, F, coefficients, Q)
    return P, ∂P∂F + ∂P∂QdQdF
end

@doc raw"""
    PrestressedMechanicalModel(inner_model, prestress_field)

Models the stress formulated in the 1st Piola-Kirchhoff stress tensor based on a multiplicative split
of the deformation gradient $$F = F_{\textrm{e}} F_{0}$$ where we compute $$P(F_{\textrm{e}}) = P(F F^{-1}_{0})$$.

Please note that it is assumed that $$F^{-1}_{0}$$ is the quantity computed by `prestress_field`.
"""
struct PrestressedMechanicalModel{MM, FF} <: AbstractMaterialModel
    inner_model::MM
    prestress_field::FF
end

struct PrestressedMechanicalModelCoefficientCache{T1, T2}
    inner_cache::T1
    prestress_cache::T2
end

function setup_coefficient_cache(m::PrestressedMechanicalModel, qr::QuadratureRule, sdh::SubDofHandler)
    PrestressedMechanicalModelCoefficientCache(
        setup_coefficient_cache(m.inner_model, qr, sdh),
        setup_coefficient_cache(m.prestress_field, qr, sdh),
    )
end

material_routine(material_model::PrestressedMechanicalModel, F::Tensor{2}, coefficient_cache, state_cache::EmptyInternalCache, geometry_cache::Ferrite.CellCache, qp::QuadraturePoint, time) = prestressed_material_routine(material_model, F, coefficient_cache, state_cache, geometry_cache, qp, time)
material_routine(material_model::PrestressedMechanicalModel, F::Tensor{2}, coefficient_cache, state_cache::TrivialInternalMaterialStateCache, geometry_cache::Ferrite.CellCache, qp::QuadraturePoint, time) = prestressed_material_routine(material_model, F, coefficient_cache, state_cache, geometry_cache, qp, time)
material_routine(material_model::PrestressedMechanicalModel, F::Tensor{2}, coefficient_cache, state_cache::RateIndependentMaterialStateCache, geometry_cache::Ferrite.CellCache, qp::QuadraturePoint, time) = prestressed_material_routine(material_model, F, coefficient_cache, state_cache, geometry_cache, qp, time)
function prestressed_material_routine(material_model::PrestressedMechanicalModel, F::Tensor{2}, coefficient_cache, state_cache, geometry_cache::Ferrite.CellCache, qp::QuadraturePoint, time)
    F₀inv = evaluate_coefficient(coefficient_cache.prestress_cache, geometry_cache, qp, time)
    Fᵉ = F ⋅ F₀inv
    ∂Ψᵉ∂Fᵉ, ∂²Ψᵉ∂Fᵉ² = material_routine(material_model.inner_model, Fᵉ, coefficient_cache.inner_cache, state_cache, geometry_cache, qp, time)
    Pᵉ = ∂Ψᵉ∂Fᵉ # Elastic PK1
    P  = Pᵉ ⋅ transpose(F₀inv) # Obtained by Coleman-Noll procedure
    Aᵉ = ∂²Ψᵉ∂Fᵉ² # Elastic mixed modulus
    # TODO condense these steps into a single operation "A_imkn F_jm F_ln"
    # Pull elastic modulus from intermediate to reference configuration
    ∂Pᵉ∂F = Aᵉ ⋅ transpose(F₀inv)
    ∂P∂F = dot_2_1t(∂Pᵉ∂F, F₀inv)
    return P, ∂P∂F
end

setup_internal_cache(material_model::PrestressedMechanicalModel, qr::QuadratureRule, sdh::SubDofHandler) = setup_internal_cache(material_model.inner_model, qr, sdh)

@doc raw"""
    PK1Model(material, coefficient_field)
    PK1Model(material, internal_model, coefficient_field)

Models the stress formulated in the 1st Piola-Kirchhoff stress tensor. If the material is energy-based,
then the term is formulated as follows:
$$\int_{\Omega_0} P(u,s) \cdot \delta F dV = \int_{\Omega_0} \partial_{F} \psi(u,s) \cdot \delta \nabla u $$
"""
struct PK1Model{PMat, IMod, CFType} <: AbstractMaterialModel
    material::PMat
    internal_model::IMod
    coefficient_field::CFType
end

PK1Model(material, coefficient_field) = PK1Model(material, EmptyInternalModel(), coefficient_field)

function setup_coefficient_cache(m::PK1Model, qr::QuadratureRule, sdh::SubDofHandler)
    return setup_coefficient_cache(m.coefficient_field, qr, sdh)
end

setup_internal_cache(material_model::PK1Model, qr::QuadratureRule, sdh::SubDofHandler) = setup_internal_cache(material_model.internal_model, qr, sdh)

function stress_and_tangent(model::PK1Model, F::Tensor{2}, coefficients, ::EmptyInternalModel)
    ∂²Ψ∂F², ∂Ψ∂F = Tensors.hessian(
        F_ad ->
              Ψ(F_ad, coefficients, model.material),
        F, :all)

    return ∂Ψ∂F, ∂²Ψ∂F²
end

@doc raw"""
    GeneralizedHillModel(passive_spring_model, active_spring_model, active_deformation_gradient_model,contraction_model, microstructure_model)

The generalized Hill framework as proposed by [GokMenKuh:2014:ghm](@citet).

In this framework the model is formulated as an energy minimization problem with the following additively split energy:

$W(\mathbf{F}, \mathbf{F}^{\rm{a}}) = W_{\rm{passive}}(\mathbf{F}) + W_{\rm{active}}(\mathbf{F}\mathbf{F}^{-\rm{a}})$

Where $W_{\rm{passive}}$ is the passive material response and $W_{\rm{active}}$ the active response
respectvely.
"""
struct GeneralizedHillModel{PMat, AMat, ADGMod, CMod, MS} <: AbstractMaterialModel
    passive_spring::PMat
    active_spring::AMat
    active_deformation_gradient_model::ADGMod
    contraction_model::CMod
    microstructure_model::MS
end

function setup_coefficient_cache(m::GeneralizedHillModel, qr::QuadratureRule, sdh::SubDofHandler)
    return setup_coefficient_cache(m.microstructure_model, qr, sdh)
end

function stress_and_tangent(model::GeneralizedHillModel, F::Tensor{2}, coefficients, state)
    # TODO what is a good abstraction here?
    Fᵃ = compute_Fᵃ(state, coefficients, model.contraction_model, model.active_deformation_gradient_model)

    ∂²Ψ∂F², ∂Ψ∂F = Tensors.hessian(
        F_ad ->
              Ψ(F_ad,     coefficients, model.passive_spring)
            + Ψ(F_ad, Fᵃ, coefficients, model.active_spring),
        F, :all)

    return ∂Ψ∂F, ∂²Ψ∂F²
end


@doc raw"""
    ExtendedHillModel(passive_spring_model, active_spring_model, active_deformation_gradient_model,contraction_model, microstructure_model)

The extended (generalized) Hill model as proposed by [OgiBalPer:2023:aeg](@citet). The original formulation dates back to [StaKlaHol:2008:smc](@citet) for smooth muscle tissues.

In this framework the model is formulated as an energy minimization problem with the following additively split energy:

$W(\mathbf{F}, \mathbf{F}^{\rm{a}}) = W_{\rm{passive}}(\mathbf{F}) + \mathcal{N}(\bm{\alpha})W_{\rm{active}}(\mathbf{F}\mathbf{F}^{-\rm{a}})$

Where $W_{\rm{passive}}$ is the passive material response and $W_{\rm{active}}$ the active response
respectvely. $\mathcal{N}$ is the amount of formed crossbridges. We refer to the original paper [OgiBalPer:2023:aeg](@cite) for more details.
"""
struct ExtendedHillModel{PMat, AMat, ADGMod, CMod, MS} <: AbstractMaterialModel
    passive_spring::PMat
    active_spring::AMat
    active_deformation_gradient_model::ADGMod
    contraction_model::CMod
    microstructure_model::MS
end

function setup_coefficient_cache(m::ExtendedHillModel, qr::QuadratureRule, sdh::SubDofHandler)
    return setup_coefficient_cache(m.microstructure_model, qr, sdh)
end

function stress_and_tangent(model::ExtendedHillModel, F::Tensor{2}, coefficients, cell_state)
    # TODO what is a good abstraction here?
    Fᵃ = compute_Fᵃ(cell_state, coefficients, model.contraction_model, model.active_deformation_gradient_model)
    N = 𝓝(cell_state, model.contraction_model)

    ∂²Ψ∂F², ∂Ψ∂F = Tensors.hessian(
        F_ad ->
                Ψ(F_ad,     coefficients, model.passive_spring)
            + N*Ψ(F_ad, Fᵃ, coefficients, model.active_spring),
        F, :all)

    return ∂Ψ∂F, ∂²Ψ∂F²
end


@doc raw"""
    ActiveStressModel(material_model, active_stress_model, contraction_model, microstructure_model)

The active stress model as originally proposed by [GucWalMcC:1993:mac](@citet).

In this framework the model is formulated via balance of linear momentum in the first Piola Kirchhoff $\mathbf{P}$:

$\mathbf{P}(\mathbf{F},T^{\rm{a}}) := \partial_{\mathbf{F}} W_{\rm{passive}}(\mathbf{F}) + \mathbf{P}^{\rm{a}}(\mathbf{F}, T^{\rm{a}})$

where the passive material response can be described by an energy $W_{\rm{passive}$ and $T^{\rm{a}}$ the active tension generated by the contraction model.
"""
struct ActiveStressModel{Mat, ASMod, CMod, MS} <: AbstractMaterialModel
    material_model::Mat
    active_stress_model::ASMod
    contraction_model::CMod
    microstructure_model::MS
end

function setup_coefficient_cache(m::ActiveStressModel, qr::QuadratureRule, sdh::SubDofHandler)
    return setup_coefficient_cache(m.microstructure_model, qr, sdh)
end

function stress_and_tangent(model::ActiveStressModel, F::Tensor{2}, coefficients, cell_state)
    ∂²Ψ∂F², ∂Ψ∂F = Tensors.hessian(
        F_ad ->
              Ψ(F_ad, coefficients, model.material_model),
        F, :all)

    ∂2 = Tensors.gradient(
        F_ad -> ∂(model.active_stress_model, cell_state, F_ad, coefficients),
    F)
    N = 𝓝(cell_state, model.contraction_model)
    return ∂Ψ∂F + N*∂(model.active_stress_model, cell_state, F, coefficients), ∂²Ψ∂F² + N*∂2
end

setup_internal_cache(material_model::Union{<:ActiveStressModel, <:ExtendedHillModel, <:GeneralizedHillModel}, qr::QuadratureRule, sdh::SubDofHandler) = setup_contraction_model_cache(material_model.contraction_model, qr, sdh)
setup_internal_cache(material_model::Union{<:ElastodynamicsModel{<:ActiveStressModel}, <:ElastodynamicsModel{<:ExtendedHillModel}, <:ElastodynamicsModel{<:GeneralizedHillModel}}, qr::QuadratureRule, sdh::SubDofHandler) = setup_contraction_model_cache(material_model.rhs.contraction_model, qr, sdh)

# Some debug materials
Base.@kwdef struct LinearMaxwellMaterial{T} <: AbstractMaterialModel
    E₀::T
    E₁::T
    μ::T
    η₁::T
    ν::T
end

# TODO this actually belongs to the multi-level newton file :)
# Dual (global cache and element-level cache) use for now to make it non-allocating.
struct GenericFirstOrderRateIndependentMaterialStateCache{LocalModelType, QType, QType2, T, LVH} <: RateIndependentMaterialStateCache
    # The actual model
    model::LocalModelType
    # Internal state at t and tprev
    # TODO play around with using a Qvector here and throw out lvh
    Q::QType
    Qprev::QType
    # t - tprev
    Δt::T
    # local_solver::...?
    lvh::LVH
    # These are used locally
    localQ::QType2
    localQprev::QType2
end

local_function_size(model::QuasiStaticModel) = local_function_size(model.material_model)
function local_function_size(model::AbstractMaterialModel)
    return _compute_local_function_size(0, gather_internal_variable_infos(model))
end

function _compute_local_function_size(total, lvis::Tuple)
    for lvi in lvis
        total += _compute_local_function_size(total, lvi)
    end
    return total
end

function _compute_local_function_size(total, lvi::InternalVariableInfo)
    return lvi.size
end

function _query_local_state(state_cache::GenericFirstOrderRateIndependentMaterialStateCache, geometry_cache, qp)
    dh = state_cache.lvh.dh
    dofs = celldofsview(dh, cellid(geometry_cache))
    size = local_function_size(state_cache.model)
    range_begin = 1+(qp.i-1)*size
    range_end   = qp.i*size
    Qv  = @view state_cache.Q[range_begin:range_end]
    Qpv = @view state_cache.Qprev[range_begin:range_end]
    @inbounds @.. state_cache.localQ     = Qv
    @inbounds @.. state_cache.localQprev = Qpv

    return state_cache.localQ, state_cache.localQprev
end

function _store_local_state!(state_cache::GenericFirstOrderRateIndependentMaterialStateCache, geometry_cache, qp)
    dh = state_cache.lvh.dh
    dofs = celldofsview(dh, cellid(geometry_cache))
    # TODO properly via gather_internal_variable_infos :)
    size = local_function_size(state_cache.model)
    range_begin = 1+(qp.i-1)*size
    range_end   = qp.i*size
    Qv  = @view state_cache.Q[range_begin:range_end]
    @inbounds @.. Qv = state_cache.localQ

    return nothing
end

function solve_local_constraint(F::Tensor{2,dim}, coefficients, material_model::LinearMaxwellMaterial, state_cache::GenericFirstOrderRateIndependentMaterialStateCache, geometry_cache, qp, time) where dim
    # Concept only for now.
    function solve_internal_timestep(material::LinearMaxwellMaterial, state_cache::GenericFirstOrderRateIndependentMaterialStateCache, ε, εᵛflat, εᵛprevflat)
        @unpack Δt = state_cache
        εᵛ₁ = SymmetricTensor{2,dim}(εᵛflat)
        εᵛ₀ = SymmetricTensor{2,dim}(εᵛprevflat)
        #     dεᵛdt = E₁/η₁ c : (ε - εᵛ)
        # <=> (εᵛ₁ - εᵛ₀) / Δt = E₁/η₁ c : (ε - εᵛ₁) = E₁/η₁ c : ε - E₁/η₁ c : εᵛ₁
        # <=> εᵛ₁ / Δt + E₁/η₁ c : εᵛ₁ = εᵛ₀/Δt + E₁/η₁ c : ε
        # <=> (𝐈 / Δt + E₁/η₁ c) : εᵛ₁ = εᵛ₀/Δt + E₁/η₁ c : ε

        (; E₀, E₁, μ, η₁, ν) = material
        I = one(ε)
        c₁ = ν / ((ν + 1)*(1-2ν)) * I ⊗ I
        c₂ = 1 / (1+ν) * one(c₁)
        ℂ = c₁ + c₂

        # FIXME non-allocating version by using state_cache nlsolver
        A = tomandel(SMatrix, one(ℂ)/Δt + E₁/η₁ * ℂ)
        b = tomandel(SVector, εᵛ₀/Δt + E₁/η₁ * ℂ ⊡ ε)
        return frommandel(typeof(ε), A \ b)
    end

    Qflat, Qprevflat = _query_local_state(state_cache, geometry_cache, qp)
    ε = symmetric(F - one(F))
    Q = solve_internal_timestep(material_model, state_cache, ε, Qflat, Qprevflat)
    Qflat .= Q.data
    _store_local_state!(state_cache, geometry_cache, qp)

    # Corrector
    # Concept only for now.
    function solve_internal_timestep_corrector(material::LinearMaxwellMaterial, state_cache::GenericFirstOrderRateIndependentMaterialStateCache, ε, εᵛflat, εᵛprevflat)
        @unpack Δt = state_cache
        εᵛ₁ = SymmetricTensor{2,dim}(εᵛflat)
        εᵛ₀ = SymmetricTensor{2,dim}(εᵛprevflat)
        # Local problem: (𝐈 / Δt + E₁/η₁ c) : εᵛ₁ = εᵛ₀/Δt + E₁/η₁ c : ε
        # =>  dLdQ = 𝐈 / Δt + E₁/η₁ c   := A
        # => -dLdF = E₁/η₁ c            := B

        (; E₀, E₁, μ, η₁, ν) = material
        I = one(ε)
        c₁ = ν / ((ν + 1)*(1-2ν)) * I ⊗ I
        c₂ = 1 / (1+ν) * one(c₁)
        ℂ = c₁ + c₂

        # FIXME non-allocating version by using state_cache nlsolver
        A = tomandel(SMatrix, one(ℂ)/Δt + E₁/η₁ * ℂ)
        B = tomandel(SMatrix, E₁/η₁ * ℂ)
        return frommandel(typeof(ℂ), A \ B)
    end
    dQdF = solve_internal_timestep_corrector(material_model, state_cache, ε, Qflat, Qprevflat)
    ∂P∂Q = Tensors.gradient(εᵛ->stress_function(material_model, ε, coefficients, εᵛ), Q)

    return Q, ∂P∂Q ⊡ dQdF
end

function stress_function(material::LinearMaxwellMaterial, ε, coefficients, εᵛ)
    (; E₀, E₁, μ, η₁, ν) = material
    I = one(ε)
    c₁ = ν / ((ν + 1)*(1-2ν)) * I ⊗ I
    c₂ = 1 / (1+ν) * one(c₁)
    ℂ = c₁ + c₂
    return E₀ * ℂ ⊡ ε + E₁ * ℂ ⊡ (ε - εᵛ)
end

function stress_and_tangent(material_model::LinearMaxwellMaterial, F::Tensor{2}, coefficients, εᵛ)
    ε = symmetric(F - one(F))
    ∂σ∂ε, σ = Tensors.gradient(ε->stress_function(material_model, ε, coefficients, εᵛ), ε, :all)
    return σ, ∂σ∂ε
end

function setup_coefficient_cache(m::LinearMaxwellMaterial, qr::QuadratureRule, sdh::SubDofHandler)
    return NoMicrostructureModel() # FIXME what should we do here? :)
end

function gather_internal_variable_infos(model::LinearMaxwellMaterial)
    return (InternalVariableInfo(:εᵛ, 6),) # TODO iterator and dimension info
end
