# TODO (FILE) I think we should change the design here. Instea of dispatching on Ψ we should make the material callable or equip it with a function.

abstract type AbstractMaterialModel end


function material_routine(constitutive_model::AbstractMaterialModel, cc, F, internal_state, geometry_cache::Ferrite.CellCache, qp::QuadraturePoint, time)
    coefficients = evaluate_coefficient(cc, geometry_cache, qp, time)
    return material_routine(F, coefficients, internal_state, constitutive_model)
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

function material_routine(constitutive_model::PrestressedMechanicalModel, coefficient_cache::PrestressedMechanicalModelCoefficientCache, F, internal_state, geometry_cache::Ferrite.CellCache, qp::QuadraturePoint, time)
    F₀inv = evaluate_coefficient(coefficient_cache.prestress_cache, geometry_cache, qp, time)
    Fᵉ = F ⋅ F₀inv
    ∂Ψᵉ∂Fᵉ, ∂²Ψᵉ∂Fᵉ² = material_routine(constitutive_model.inner_model, coefficient_cache.inner_cache, Fᵉ, internal_state, geometry_cache, qp, time)
    Pᵉ = ∂Ψᵉ∂Fᵉ # Elastic PK1
    P  = Pᵉ ⋅ transpose(F₀inv) # Obtained by Coleman-Noll procedure
    Aᵉ = ∂²Ψᵉ∂Fᵉ² # Elastic mixed modulus
    # TODO condense these steps into a single operation "A_imkn F_jm F_ln"
    # Pull elastic modulus from intermediate to reference configuration
    ∂Pᵉ∂F = Aᵉ ⋅ transpose(F₀inv)
    ∂P∂F = dot_2_1t(∂Pᵉ∂F, F₀inv)
    return P, ∂P∂F
end

setup_internal_model_cache(constitutive_model::PrestressedMechanicalModel, qr::QuadratureRule, sdh::SubDofHandler) = setup_internal_model_cache(constitutive_model.inner_model, qr, sdh)

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

function material_routine(model::PK1Model, cc, F, internal_state, geometry_cache::Ferrite.CellCache, qp::QuadraturePoint, time)
    coefficients = evaluate_coefficient(cc, geometry_cache, qp, time)
    ∂²Ψ∂F², ∂Ψ∂F = Tensors.hessian(
            F_ad -> Ψ(F_ad, coefficients, model.material),
        F, :all
    )

    return ∂Ψ∂F, ∂²Ψ∂F²
end

setup_internal_model_cache(constitutive_model::PK1Model, qr::QuadratureRule, sdh::SubDofHandler) = setup_internal_model_cache(constitutive_model.internal_model, qr, sdh)

function material_routine(F::Tensor{2}, coefficients, ::EmptyInternalModel, model::PK1Model)
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

function material_routine(F::Tensor{2}, coefficients, internal_state, model::GeneralizedHillModel)
    # TODO what is a good abstraction here?
    Fᵃ = compute_Fᵃ(internal_state, coefficients, model.contraction_model, model.active_deformation_gradient_model)

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

function material_routine(F::Tensor{2,dim}, coefficients, cell_state, model::ExtendedHillModel) where {dim}
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

function material_routine(F::Tensor{2,dim}, coefficients, cell_state, model::ActiveStressModel) where {dim}
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


"""
    ElastodynamicsModel(::AbstractMaterialModel, ρ::Coefficient)
"""
struct ElastodynamicsModel{RHSModel <: AbstractMaterialModel, CoefficientType}
    rhs::RHSModel
    ρ::CoefficientType
end

function setup_coefficient_cache(m::ElastodynamicsModel, qr::QuadratureRule, sdh::SubDofHandler)
    return setup_coefficient_cache(m.rhs, qr, sdh)
end

setup_internal_model_cache(constitutive_model::Union{<:ActiveStressModel, <:ExtendedHillModel, <:GeneralizedHillModel}, qr::QuadratureRule, sdh::SubDofHandler) = setup_contraction_model_cache(constitutive_model.contraction_model, qr, sdh)
setup_internal_model_cache(constitutive_model::Union{<:ElastodynamicsModel{<:ActiveStressModel}, <:ElastodynamicsModel{<:ExtendedHillModel}, <:ElastodynamicsModel{<:GeneralizedHillModel}}, qr::QuadratureRule, sdh::SubDofHandler) = setup_contraction_model_cache(constitutive_model.rhs.contraction_model, qr, sdh)
