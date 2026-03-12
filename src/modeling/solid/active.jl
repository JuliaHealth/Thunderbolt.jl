"""
A simple helper to use a passive material model as an active material for [GeneralizedHillModel](@ref), [ExtendedHillModel](@ref) and [ActiveStressModel](@ref).
"""
struct ActiveMaterialAdapter{Mat}
    mat::Mat
end

"""
"""
function Ψ(F, Fᵃ, coeff::AbstractOrthotropicMicrostructure, adapter::ActiveMaterialAdapter)
    f₀, s₀, n₀ = coeff.f, coeff.s, coeff.n
    f̃ = Fᵃ ⋅ f₀ / norm(Fᵃ ⋅ f₀)
    s̃ = Fᵃ ⋅ s₀ / norm(Fᵃ ⋅ s₀)
    ñ = Fᵃ ⋅ n₀ / norm(Fᵃ ⋅ n₀)

    Fᵉ = F ⋅ inv(Fᵃ)
    coeff = OrthotropicMicrostructure(f̃, s̃, ñ)
    Ψᵃ = Ψ(Fᵉ, coeff, adapter.mat)
    return Ψᵃ
end

@doc raw"""
The active deformation gradient formulation by [GokMenKuh:2014:ghm](@citet).

$F^{\rm{a}} = (\lambda^{\rm{a}}-1) f_0 \otimes f_0$ + I$

See also [OgiBalPer:2023:aeg](@cite) for a further analysis.
"""
struct GMKActiveDeformationGradientModel end

function compute_Fᵃ(
        state,
        coeff::AbstractTransverselyIsotropicMicrostructure,
        contraction_model::AbstractSarcomereModel,
        ::GMKActiveDeformationGradientModel
)
    f₀ = coeff.f
    λᵃ = compute_λᵃ(state, contraction_model)
    Fᵃ = Tensors.unsafe_symmetric(one(SymmetricTensor{2, 3}) + (λᵃ - 1.0) * f₀ ⊗ f₀)
    return Fᵃ
end


@doc raw"""
An incompressivle version of the active deformation gradient formulation by [GokMenKuh:2014:ghm](@citet).

$F^{\rm{a}} = \lambda^{\rm{a}} f_0 \otimes f_0 + \frac{1}{\sqrt{\lambda^{\rm{a}}}}(s_0 \otimes s_0 + n_0 \otimes n_0)$

See also [OgiBalPer:2023:aeg](@cite) for a further analysis.
"""
struct GMKIncompressibleActiveDeformationGradientModel end

function compute_Fᵃ(
        state,
        coeff::AbstractOrthotropicMicrostructure,
        contraction_model::AbstractSarcomereModel,
        ::GMKIncompressibleActiveDeformationGradientModel
)
    f₀, s₀, n₀ = coeff.f, coeff.s, coeff.n
    λᵃ = compute_λᵃ(state, contraction_model)
    Fᵃ = λᵃ * f₀ ⊗ f₀ + 1.0 / sqrt(λᵃ) * s₀ ⊗ s₀ + 1.0 / sqrt(λᵃ) * n₀ ⊗ n₀
    return Fᵃ
end

@doc raw"""
The active deformation gradient formulation by [RosLasRuiSeqQua:2014:tco](@citet).

$F^{\rm{a}} = \lambda^{\rm{a}} f_0 \otimes f_0 + (1+\kappa(\lambda^{\rm{a}}-1)) s_0 \otimes s_0 + \frac{1}{1+\kappa(\lambda^{\rm{a}}-1))\lambda^{\rm{a}}} n_0 \otimes n_0$

Where $\kappa \geq 0$ is the sheelet part.

See also [OgiBalPer:2023:aeg](@cite) for a further analysis.
"""
struct RLRSQActiveDeformationGradientModel{TD}
    sheetlet_part::TD
end

function compute_Fᵃ(
        state,
        coeff::AbstractOrthotropicMicrostructure,
        contraction_model::AbstractSarcomereModel,
        Fᵃ_model::RLRSQActiveDeformationGradientModel
)
    @unpack sheetlet_part = Fᵃ_model
    f₀, s₀, n₀ = coeff.f, coeff.s, coeff.n
    λᵃ = compute_λᵃ(state, contraction_model)
    Fᵃ = λᵃ * f₀ ⊗ f₀ +
         (1.0 + sheetlet_part * (λᵃ - 1.0)) * s₀ ⊗ s₀ +
         1.0 / ((1.0 + sheetlet_part * (λᵃ - 1.0)) * λᵃ) * n₀ ⊗ n₀
    return Fᵃ
end


@doc raw"""
A simple active stress component.

$T^{\rm{a}} = T^{\rm{max}} \, [Ca_{\rm{i}}] \frac{(F \cdot f_0) \otimes f_0}{||F \cdot f_0||}$
"""
Base.@kwdef struct SimpleActiveStress{TD}
    Tmax::TD = 1.0
end

function active_stress(
        sas::SimpleActiveStress,
        F::Tensor{2, dim},
        coeff::AbstractTransverselyIsotropicMicrostructure
) where {dim}
    sas.Tmax * (F ⋅ coeff.f) ⊗ coeff.f / norm(F ⋅ coeff.f)
end


@doc raw"""
The active stress component described by [PieRegSalCorVerQua:2022:clm](@citet) (Eq. 3).

$T^{\rm{a}} = T^{\rm{max}} \, [Ca_{\rm{i}}] \left(p^f \frac{(F \cdot f_0) \otimes f_0}{||F \cdot f_0||} + p^{\rm{s}} \frac{(F \cdot s_0) \otimes s_0}{||F \cdot s_0||} + p^{\rm{n}} \frac{(F \cdot n_0) \otimes n_0}{||F \cdot n_0||}\right)$
"""
Base.@kwdef struct PiersantiActiveStress{TD}
    Tmax::TD = 1.0
    pf::TD = 1.0
    ps::TD = 0.75
    pn::TD = 0.0
end

function active_stress(
        sas::PiersantiActiveStress,
        F::Tensor{2, dim},
        coeff::AbstractOrthotropicMicrostructure
) where {dim}
    sas.Tmax * (
        sas.pf * (F ⋅ coeff.f) ⊗ coeff.f / norm(F ⋅ coeff.f) +
        sas.ps * (F ⋅ coeff.s) ⊗ coeff.s / norm(F ⋅ coeff.s) +
        sas.pn * (F ⋅ coeff.n) ⊗ coeff.n / norm(F ⋅ coeff.n)
    )
end


@doc raw"""
The active stress component as described by [GucWalMcC:1993:mac](@citet).

$T^{\rm{a}} = T^{\rm{max}} \, [Ca_{\rm{i}}] (F \cdot f_0) \otimes f_0$

"""
Base.@kwdef struct Guccione1993ActiveModel
    # Default values from Marina Kampers PhD thesis
    Tmax::Float64   = 135.0 #kPa
    l₀::Float64     = 1.45  #µm
    lR::Float64     = 1.8   #µm
    Ca₀::Float64    = 4.35  #µM
    Ca₀max::Float64 = 4.35  #µM
    B::Float64      = 3.8   #1/µm
end

function active_stress(
        sas::Guccione1993ActiveModel,
        F::Tensor{2, dim},
        coeff::AbstractTransverselyIsotropicMicrostructure
) where {dim}
    @unpack l₀, Ca₀, lR, Ca₀max, Tmax, B = sas
    f = F ⋅ coeff.f
    λf = norm(f)
    l = lR * λf
    ECa₅₀² = Ca₀max^2 / (exp(B * (l - l₀)) - 1.0)
    T₀ = Tmax * Ca₀^2 / (Ca₀^2 + ECa₅₀²)
    return T₀ * (f / λf) ⊗ coeff.f # We normalize here the fiber direction, as T₀ should contain all the active stress associated with the direction
end

# This is merely a helper function to use the same code path for active and passive simulations.
function active_stress(::NullEnergyModel, F::Tensor, coeff)
    return zero(F)
end
