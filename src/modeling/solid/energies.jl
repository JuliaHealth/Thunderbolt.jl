# This file contains some common energies.

@doc raw"""
A simple dummy energy with $\Psi = 0$.
"""
struct NullEnergyModel end
Ψ(F, coefficients, mp::NullEnergyModel) = 0.0


@doc raw"""
A simple dummy compression model with $U(I_3) = 0$.
"""
struct NullCompressionPenalty end
U(I₃, mp::NullCompressionPenalty) = 0.0


@doc raw"""
An isochoric compression model where

$U(I_3) = \beta (I_3^b + I_3^{-b} -2)^a$

with $a,b \geq 1$.


Entry 1 from table 3 in [HarNef:2003:pgp](@cite).
"""
Base.@kwdef struct HartmannNeffCompressionPenalty1{TD1, TD2}
    a::TD1 = 1
    b::TD1 = 2
    β::TD2 = 1.0
end
function U(I₃, mp::HartmannNeffCompressionPenalty1)
    I₃ < 0 && return NaN
    return mp.β * (I₃^mp.b + 1/(I₃^mp.b) - 2)^mp.a
end


@doc raw"""
An isochoric compression model where

$U(I_3) = \beta (\sqrt{I_3}-1)^a$

with $a > 1$.

Entry 2 from table 3 in [HarNef:2003:pgp](@cite).
"""
Base.@kwdef struct HartmannNeffCompressionPenalty2{TD1, TD2}
    a::TD1 = 1.1
    β::TD2 = 1.0
end
function U(I₃, mp::HartmannNeffCompressionPenalty2)
    I₃ < 0 && return NaN
    return mp.β * (√I₃-1)^mp.a
end


@doc raw"""
An isochoric compression model where

$U(I_3) = \beta (I_3 - 2\log(\sqrt{I_3}) + 4\log(\sqrt{I_3})^2) - 1)$

Entry 3 from table 3 in [HarNef:2003:pgp](@cite).
"""
Base.@kwdef struct HartmannNeffCompressionPenalty3{T}
    β::T = 1.0
end
function U(I₃, mp::HartmannNeffCompressionPenalty3)
    I₃ < 0 && return NaN
    J = √I₃
    mp.β * (I₃ - 2log(J) + 4log(J)^2 - 1)
end


@doc raw"""
A compression model with $U(I_3) = \beta (I_3 -1 - 2\log(\sqrt{I_3}))^a$.

!!! note
    Citation missing. How is this one called in literature?
"""
Base.@kwdef struct SimpleCompressionPenalty{TD}
    β::TD = 1.0
end
function U(I₃::T, mp::SimpleCompressionPenalty) where {T}
    I₃ < 0 && return NaN
    J = √I₃
    mp.β * (I₃ - 1 - 2*log(J))
end


"""
https://onlinelibrary.wiley.com/doi/epdf/10.1002/cnm.2866
"""
Base.@kwdef struct TransverseIsotopicNeoHookeanModel{TD1, TD2, TU}
    a₁::TD1 = 2.6
    a₂::TD1 = 2.82
    α₁::TD2 = 30.48
    α₂::TD2 = 7.25

    mpU::TU = HartmannNeffCompressionPenalty1()
end
function Ψ(
    F,
    coeff::AbstractTransverselyIsotropicMicrostructure,
    mp::TransverseIsotopicNeoHookeanModel,
)
    @unpack a₁, a₂, α₁, α₂, mpU = mp
    f₀ = coeff.f

    C = tdot(F)
    I₁ = tr(C)
    I₃ = det(C)

    Ī₁ = I₁/cbrt(I₃)
    # this is a hotfix to fight numerical noise when returning to the equilibrium state...
    if -1e-8 < Ī₁ - 3.0 < 0.0
        Ī₁ = 3.0
    end

    I₄ = tr(C ⋅ f₀ ⊗ f₀)

    Ψᵖ = α₁*(Ī₁ - 3)^a₁ + U(I₃, mpU)
    if I₄ ≥ 1
        Ψᵖ += α₂*(I₄ - 1)^a₂
    end

    return Ψᵖ
end


@doc raw"""
The well-known orthotropic material model for the passive response of
cardiac tissues by [HolOgd:2009:cmp](@citet).

$\Psi = \frac{a}{2b} e^{b(I_1-3)} + \sum_{i\in\{\rm{f},\rm{s}\}} \frac{a^i}{2b^i}(e^{b^i<I_4^i - 1>^2}-1) + \frac{a^{\rm{fs}}}{2b^{\rm{fs}}}(e^{b^{\rm{fs}}{I_8^{\rm{fs}}}^2}-1)$
"""
Base.@kwdef struct HolzapfelOgden2009Model{TD, TU} #<: OrthotropicMaterialModel
    a::TD = 0.059
    b::TD = 8.023
    aᶠ::TD = 18.472
    bᶠ::TD = 16.026
    aˢ::TD = 2.581
    bˢ::TD = 11.120
    aᶠˢ::TD = 0.216
    bᶠˢ::TD = 11.436
    mpU::TU = SimpleCompressionPenalty()
end
function Ψ(F, coeff::AbstractOrthotropicMicrostructure, mp::HolzapfelOgden2009Model)
    # Modified version of https://onlinelibrary.wiley.com/doi/epdf/10.1002/cnm.2866
    @unpack a, b, aᶠ, bᶠ, aˢ, bˢ, aᶠˢ, bᶠˢ, mpU = mp
    f₀, s₀, n₀ = coeff.f, coeff.s, coeff.n

    C = tdot(F)
    I₃ = det(C)
    I₁ = tr(C/cbrt(I₃))
    I₄ᶠ = f₀ ⋅ C ⋅ f₀
    I₄ˢ = s₀ ⋅ C ⋅ s₀
    I₈ᶠˢ = (f₀ ⋅ C ⋅ s₀ + s₀ ⋅ C ⋅ f₀)/2.0

    Ψᵖ = a/(2.0*b)*(exp(b*(I₁-3.0))-1.0) + aᶠˢ/(2.0*bᶠˢ)*(exp(bᶠˢ*I₈ᶠˢ^2)-1.0) + U(I₃, mpU)
    if I₄ᶠ >= 1.0
        Ψᵖ += aᶠ/(2.0*bᶠ)*(exp(bᶠ*(I₄ᶠ - 1)^2)-1.0)
    end
    if I₄ˢ >= 1.0
        Ψᵖ += aˢ/(2.0*bˢ)*(exp(bˢ*(I₄ˢ - 1)^2)-1.0)
    end

    return Ψᵖ
end


@doc raw"""
This is the Fung-type transverse isotropic material model for the passive
response of cardiac tissue proposed by [LinYIn:1998:mcl](@citet).


$\Psi = C_1(e^{C_2(I_1-3)^2 + C_3(I_1-3)(I_4-1) + C_4(I_4-1)^2}-1)$
"""
Base.@kwdef struct LinYinPassiveModel{TD, TU} #<: TransverseIsotropicMaterialModel
    C₁::TD = 1.05
    C₂::TD = 9.13
    C₃::TD = 2.32
    C₄::TD = 0.08
    mpU::TU = SimpleCompressionPenalty()
end
function Ψ(F, coeff::AbstractTransverselyIsotropicMicrostructure, model::LinYinPassiveModel)
    @unpack C₁, C₂, C₃, C₄, mpU = model
    f₀ = coeff.f

    C = tdot(F) # = FᵀF

    # Invariants
    I₁ = tr(C)
    I₃ = det(C)
    I₄ = f₀ ⋅ C ⋅ f₀ # = C : (f ⊗ f)

    # Exponential portion
    Q = C₂*(I₁-3)^2 + C₃*(I₁-3)*(I₄-1) + C₄*(I₄-1)^2
    return C₁*(exp(Q)-1) + U(I₃, mpU)
end

@doc raw"""
This is the transverse isotropic material model for the active
response of cardiac tissue proposed by [LinYIn:1998:mcl](@citet).

$\Psi=C_0 + C_1*(I_1-3)(I_4-1) + C_2(I_1-3)^2 + C_3*(I_4-1)^2 + C_3*(I_1-3) + C_5*(I_4-1)$
"""
Base.@kwdef struct LinYinActiveModel{TD, TU} #<: TransverseIsotropicMaterialModel
    C₀::TD = 0.0
    C₁::TD = -13.03
    C₂::TD = 36.65
    C₃::TD = 35.42
    C₄::TD = 15.52
    C₅::TD = 1.62
    mpU::TU = SimpleCompressionPenalty()
end
function Ψ(F, coeff::AbstractTransverselyIsotropicMicrostructure, model::LinYinActiveModel)
    @unpack C₀, C₁, C₂, C₃, C₄, C₅, mpU = model
    f₀ = coeff.f
    C = tdot(F) # = FᵀF

    # Invariants
    I₁ = tr(C)
    I₃ = det(C)
    I₄ = f₀ ⋅ C ⋅ f₀ # = C : (f ⊗ f)

    return C₀ + C₁*(I₁-3)*(I₄-1) + C₂*(I₁-3)^2 + C₃*(I₄-1)^2 + C₄*(I₁-3) + C₅*(I₄-1) + U(I₃, mpU)
end

@doc raw"""
This is the transverse isotropic material model for the active
response of cardiac tissue proposed by [HumStrYin:1990:dcr](@citet).

$\Psi = C_1(\sqrt{I_4}-1)^2 + C_2(\sqrt{I_4}-1)^3 + C_3(\sqrt{I_4}-1)(I_1-3) + C_3(I_1-3)^2$
"""
Base.@kwdef struct HumphreyStrumpfYinModel{TD, TU} #<: TransverseIsotropicMaterialModel
    C₁::TD = 15.93
    C₂::TD = 55.85
    C₃::TD = 3.59
    C₄::TD = 30.21
    mpU::TU = SimpleCompressionPenalty()
end
function Ψ(F, coeff::AbstractTransverselyIsotropicMicrostructure, model::HumphreyStrumpfYinModel)
    @unpack C₁, C₂, C₃, C₄, mpU = model
    f₀ = coeff.f
    C = tdot(F) # = FᵀF

    # Invariants
    I₁ = tr(C)
    I₃ = det(C)
    I₄ = f₀ ⋅ C ⋅ f₀ # = C : (f ⊗ f)

    return C₁*(√I₄-1)^2 + C₂*(√I₄-1)^3 + C₃*(√I₄-1)*(I₁-3) + C₄*(I₁-3)^2 + U(I₃, mpU)
end


@doc raw"""
A simple linear fiber spring model for testing purposes.

$\Psi^{\rm{a}} = \frac{a^{\rm{f}}}{2}(I_e^{\rm{e}}-1)^2$
"""
Base.@kwdef struct LinearSpringModel{TD, TU}
    η::TD = 10.0
    mpU::TU = NullCompressionPenalty()
end
function Ψ(F, coeff::AbstractTransverselyIsotropicMicrostructure, mp::LinearSpringModel)
    @unpack η = mp
    f₀ = coeff.f

    M = Tensors.unsafe_symmetric(f₀ ⊗ f₀)
    FMF = Tensors.unsafe_symmetric(F ⋅ M ⋅ transpose(F))
    I₄ = tr(FMF)

    return η / 2.0 * (I₄ - 1)^2.0
end


@doc raw"""
An orthotropic material model for the passive myocardial tissue response by [GucMcCWal:1991:pmp](@citet).

$\Psi = B^{\rm{ff}} {E^{\rm{ff}}}^2 + B^{\rm{ss}}{E^{\rm{ss}}}^2 + B^{\rm{nn}}{E^{\rm{nn}}}^2 + B^{\rm{ns}}({E^{\rm{ns}}}^2+{E^{\rm{sn}}}^2) + B^{\rm{fs}}({E^{\rm{fs}}}^2+{E^{\rm{sf}}}^2) + B^{\rm{fn}}({E^{\rm{fn}}}^2+{E^{\rm{nf}}}^2)$

The default parameterization is taken from from [ZheChaNieScoFerLeoYap:2023:ems](@cite).
"""
Base.@kwdef struct Guccione1991PassiveModel{CPT}
    C₀::Float64 = 0.1
    Bᶠᶠ::Float64 = 29.8
    Bˢˢ::Float64 = 14.9
    Bⁿⁿ::Float64 = 14.9
    Bⁿˢ::Float64 = 9.3
    Bᶠˢ::Float64 = 19.2
    Bᶠⁿ::Float64 = 14.4
    mpU::CPT = SimpleCompressionPenalty(50.0)
end
function Ψ(F, coeff::AbstractOrthotropicMicrostructure, mp::Guccione1991PassiveModel)
    @unpack C₀, Bᶠᶠ, Bˢˢ, Bⁿⁿ, Bⁿˢ, Bᶠˢ, Bᶠⁿ, mpU = mp
    f₀, s₀, n₀ = coeff.f, coeff.s, coeff.n

    C = tdot(F)
    I₃ = det(C)

    E = (C-one(F))/2.0

    Eᶠᶠ = f₀ ⋅ E ⋅ f₀
    Eˢˢ = s₀ ⋅ E ⋅ s₀
    Eⁿⁿ = n₀ ⋅ E ⋅ n₀

    Eᶠˢ = f₀ ⋅ E ⋅ s₀
    Eˢᶠ = s₀ ⋅ E ⋅ f₀

    Eˢⁿ = s₀ ⋅ E ⋅ n₀
    Eⁿˢ = n₀ ⋅ E ⋅ s₀

    Eᶠⁿ = f₀ ⋅ E ⋅ n₀
    Eⁿᶠ = n₀ ⋅ E ⋅ f₀

    Q =
        Bᶠᶠ*Eᶠᶠ^2 +
        Bˢˢ*Eˢˢ^2 +
        Bⁿⁿ*Eⁿⁿ^2 +
        Bⁿˢ*(Eⁿˢ^2+Eˢⁿ^2) +
        Bᶠˢ*(Eᶠˢ^2+Eˢᶠ^2) +
        Bᶠⁿ*(Eᶠⁿ^2+Eⁿᶠ^2)

    return C₀*exp(Q)/2.0 + U(I₃, mpU)
end

@doc raw"""
    SimpleActiveSpring

A simple linear fiber spring as for example found in [GokMenKuh:2014:ghm](@cite).

$\Psi^{\rm{a}} = \frac{a^{\rm{f}}}{2}(I_e^{\rm{e}}-1)^2$
"""
Base.@kwdef struct SimpleActiveSpring
    aᶠ::Float64 = 1.0
end

function Ψ(F, Fᵃ, coeff::AbstractTransverselyIsotropicMicrostructure, mp::SimpleActiveSpring)
    @unpack aᶠ = mp
    f₀ = coeff.f

    Fᵉ = F ⋅ inv(Fᵃ)
    Cᵉ = tdot(Fᵉ)
    Iᵉ₄ᶠ = f₀ ⋅ Cᵉ ⋅ f₀

    return aᶠ/2.0*(Iᵉ₄ᶠ-1.0)^2
end

# """
# """
# Base.@kwdef struct BarbarottaRossiDedeQuarteroni2018 #<: OrthotropicMaterialModel
# 	a   =  0.2
# 	b   =  4.614
# 	aᶠ  =  4.1907
# 	bᶠ  =  7.8565
# 	aˢ  =  2.564
# 	bˢ  = 10.446
# 	aᶠˢ =  0.1304
# 	bᶠˢ = 15.255
# 	K   = 1.000 # Value not stated in paper
# 	k   = 4.0
# end

# """
# """
# function Ψ(F, f₀, s₀, Caᵢ, mp::BarbarottaRossiDedeQuarteroni2018)
# 	# Reproduction of https://onlinelibrary.wiley.com/doi/10.1002/cnm.3137
#     @unpack a, b, aᶠ, bᶠ, aˢ, bˢ, aᶠˢ, bᶠˢ, K, k = mp

# 	# Gram-Schmidt step to enforce strict orthogonality
# 	s₀ = s₀ - (f₀⋅s₀)*f₀
# 	# Reconstruct normal
# 	n₀ = cross(f₀, s₀)
# 	n₀ /= norm(n₀)

# 	# Transverse isotropic active contraction (incompressible) - Also Rossi?
# 	λ = λᵃ(Caᵢ)
# 	Fᵃ = λ*f₀⊗f₀ + (one(F) - f₀⊗f₀)/sqrt(λ)

# 	# Orthotropic active contraction (incompressible) - Rossi
# 	# ξᶠ = (λᵃ(Caᵢ)-1.0) # microscopic shortening
# 	# γf = ξᶠ
# 	# γn = ξᶠ
# 	# γs = 1.0/((1.0+γf)*(1.0+γn)) - 1.0
# 	# Fᵃ = one(F) + γf * f₀⊗f₀ + γs * s₀⊗s₀ + γn * n₀⊗n₀

# 	# Everything is computed in Fᵉ
# 	Fᵉ = F⋅inv(Fᵃ)
# 	C = tdot(Fᵉ)
# 	I₃ = det(C)
# 	J = sqrt(I₃)
# 	#J = det(Fᵉ)
#     I₁ = tr(C/cbrt(J^2))
# 	I₄ᶠ = f₀ ⋅ C ⋅ f₀
# 	I₄ˢ = s₀ ⋅ C ⋅ s₀
# 	I₈ᶠˢ = (f₀ ⋅ C ⋅ s₀ + s₀ ⋅ C ⋅ f₀)/2

# 	U = K/4.0 * (J^2 - 1 - 2*log(J))

# 	Ψ = a/(2.0*b)*exp(b*(I₁-3.0)) + aᶠˢ/(2.0*bᶠˢ)*(exp(bᶠˢ*I₈ᶠˢ^2)-1.0) + U
# 	if I₄ᶠ > 1.0
# 		Ψ += aᶠ/(2.0*bᶠ)*(exp(bᶠ*(I₄ᶠ - 1)^2)-1.0)
# 	end
# 	if I₄ˢ > 1.0
# 		Ψ += aˢ/(2.0*bˢ)*(exp(bˢ*(I₄ˢ - 1)^2)-1.0)
# 	end

#     return Ψ
# end

# """
# """
# struct Passive2017Energy #<: IsotropicMaterialModel
# 	a
# 	a₁
# 	a₂
# 	b
# 	α₁
# 	α₂
# 	β
# 	η
# end

# """
# """
# function Ψ(F, f₀, s₀, Caᵢ, mp::Passive2017Energy)
# 	# Modified version of https://onlinelibrary.wiley.com/doi/epdf/10.1002/cnm.2866
#     @unpack a, a₁, a₂, b, α₁, α₂, β, η = mp
# 	C = tdot(F)
#     I₁ = tr(C)
# 	I₃ = det(C)
# 	I₄ = tr(C ⋅ f₀ ⊗ f₀)

# 	I₃ᵇ = I₃^b
# 	U = β * (I₃ᵇ + 1/I₃ᵇ - 2)^a
# 	# Ψᵖ = α₁*(I₁/cbrt(I₃) - 3)^a₁ + α₂*max((I₄ - 1), 0.0)^a₂ + U
# 	Ψᵖ = α₁*(I₁/cbrt(I₃) - 3)^a₁ + α₂*(I₄ - 1)^2 + U

# 	M = Tensors.unsafe_symmetric(f₀ ⊗ f₀)
# 	Fᵃ = Tensors.unsafe_symmetric(one(F) + (λᵃ(Caᵢ) - 1.0) * M)
# 	f̃ = Fᵃ ⋅ f₀ / norm(Fᵃ ⋅ f₀)
# 	M̃ = f̃ ⊗ f̃

# 	Fᵉ = F - (1 - 1.0/λᵃ(Caᵢ)) * ((F ⋅ f₀) ⊗ f₀)
# 	FMF = Tensors.unsafe_symmetric(Fᵉ ⋅ M̃ ⋅ transpose(Fᵉ))
# 	Iᵉ₄ = tr(FMF)
# 	Ψᵃ = η / 2 * (Iᵉ₄ - 1)^2

#     return Ψᵖ + Ψᵃ
# end



@doc raw"""
    BioNeoHookean

A simple isotropic Neo-Hookean model of the form

$\Psi = \alpha (\bar{I_1}-3)$
"""
Base.@kwdef struct BioNeoHookean{TD, TU} #<: IsotropicMaterialModel
    α::TD = 1.0
    mpU::TU = SimpleCompressionPenalty()
end
function Ψ(F, coeff, mp::BioNeoHookean)
    # Modified version of https://onlinelibrary.wiley.com/doi/epdf/10.1002/cnm.2866
    @unpack α, mpU = mp
    C = tdot(F)
    I₁ = tr(C)
    I₃ = det(C)

    return α*(I₁/cbrt(I₃) - 3) + U(I₃, mpU)
end
