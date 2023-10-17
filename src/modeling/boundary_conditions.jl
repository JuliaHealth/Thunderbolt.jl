
@doc raw"""
RobinBC

```math
\bm{P}(\bm{u}) \cdot \bm{n}_0 = - \alpha \bm{u} \quad \textbf{x} \in \partial \Omega,
```
"""
struct RobinBC
α::Float64
boundary_name::String
end

#TODO Energy-based interface?
struct NormalSpringBC
kₛ::Float64
boundary_name::String
end

struct BendingSpringBC
kᵇ::Float64
boundary_name::String
end

struct ConstantPressureBC
p::Float64
boundary_name::String
end

struct SimpleFaceCache{MP, FV}
mp::MP
# time::Float64
# microstructure_model::MM
# coordinate_system::CS
fv::FV
end

getboundaryname(face_cache::FC) where {FC} = face_cache.mp.boundary_name

function setup_face_cache(bcd::BCD, fv::FV) where {BCD, FV}
SimpleFaceCache(bcd, fv)
end

function assemble_face!(Kₑ::Matrix, residualₑ::Vector, uₑ::Vector, cache::SimpleFaceCache{RobinBC,FV}) where {FV}
@unpack mp, fv = cache
@unpack α = mp

ndofs_face = getnbasefunctions(fv)
for qp in 1:getnquadpoints(fv)
    dΓ = getdetJdV(fv, qp)
    N = getnormal(fv, qp)

    u_q = function_value(fv, qp, uₑ)
    ∂²Ψ∂u², ∂Ψ∂u = Tensors.hessian(u -> α*u⋅u, u_q, :all)

    # Add contribution to the residual from this test function
    for i in 1:ndofs_face
        δuᵢ = shape_value(fv, qp, i)
        residualₑ[i] += δuᵢ ⋅ ∂Ψ∂u * dΓ

        for j in 1:ndofs_face
            δuⱼ = shape_value(fv, qp, j)
            # Add contribution to the tangent
            Kₑ[i, j] += ( δuᵢ ⋅ ∂²Ψ∂u² ⋅ δuⱼ ) * dΓ
        end
    end
end
end

function assemble_face!(Kₑ::Matrix, residualₑ::Vector, uₑ::Vector, cache::SimpleFaceCache{NormalSpringBC,FV}) where {FV}
@unpack mp, fv = cache
@unpack kₛ = mp

ndofs_face = getnbasefunctions(fv)
for qp in 1:getnquadpoints(fv)
    dΓ = getdetJdV(fv, qp)
    N = getnormal(fv, qp)

    u_q = function_value(fv, qp, uₑ)
    ∂²Ψ∂u², ∂Ψ∂u = Tensors.hessian(u -> 0.5*kₛ*(u⋅N)^2, u_q, :all)

    # Add contribution to the residual from this test function
    for i in 1:ndofs_face
        δuᵢ = shape_value(fv, qp, i)
        residualₑ[i] += δuᵢ ⋅ ∂Ψ∂u * dΓ

        for j in 1:ndofs_face
            δuⱼ = shape_value(fv, qp, j)
            # Add contribution to the tangent
            Kₑ[i, j] += ( δuᵢ ⋅ ∂²Ψ∂u² ⋅ δuⱼ ) * dΓ
        end
    end
end
end

function assemble_face!(Kₑ::Matrix, residualₑ::Vector, uₑ::Vector, cache::SimpleFaceCache{BendingSpringBC,FV}) where {FV}
@unpack mp, fv = cache
@unpack kᵇ = mp

ndofs_face = getnbasefunctions(fv)
for qp in 1:getnquadpoints(fv)
    dΓ = getdetJdV(fv, qp)
    N = getnormal(fv, qp)

    ∇u = function_gradient(fv, qp, uₑ)
    F = one(∇u) + ∇u

    ∂²Ψ∂F², ∂Ψ∂F = Tensors.hessian(F_ -> 0.5*kᵇ*(transpose(inv(F_))⋅N - N)⋅(transpose(inv(F_))⋅N - N), F, :all)

    # Add contribution to the residual from this test function
    for i in 1:ndofs_face
        ∇δui = shape_gradient(fv, qp, i)
        residualₑ[i] -= ∇δui ⊡ ∂Ψ∂F * dΓ

        ∇δui∂P∂F = ∇δui ⊡ ∂²Ψ∂F² # Hoisted computation
        for j in 1:ndofs_face
            ∇δuj = shape_gradient(fv, qp, j)
            # Add contribution to the tangent
            Kₑ[i, j] += ( ∇δui∂P∂F ⊡ ∇δuj ) * dΓ
        end
    end
end
end

function assemble_face!(Kₑ::Matrix, residualₑ::Vector, uₑ::Vector, cache::SimpleFaceCache{ConstantPressureBC,FV}) where {FV}
@unpack mp, fv = cache
@unpack p = mp

ndofs_face = getnbasefunctions(fv)
for qp in 1:getnquadpoints(fv)
    dΓ = getdetJdV(fv, qp)

    n₀ = getnormal(fv, qp)

    ∇u = function_gradient(fv, qp, uₑ)
    F = one(∇u) + ∇u

    # ∂P∂F = Tensors.gradient(
    #     F_ad -> p * det(F_ad) * transpose(inv(F_ad)),
    # F)

    # Add contribution to the residual from this test function
    cofF = transpose(inv(F))
    neumann_term = p * det(F) * cofF
    for i in 1:ndofs_face
        δuᵢ = shape_value(fv, qp, i)
        residualₑ[i] += neumann_term ⋅ n₀ ⋅ δuᵢ * dΓ

        # ∂P∂Fδui =   ∂P∂F ⊡ (n₀ ⊗ δuᵢ) # Hoisted computation
        for j in 1:ndofs_face
            ∇δuⱼ = shape_gradient(fv, qp, j)
            # Add contribution to the tangent
            # Kₑ[i, j] += (n₀ ⊗ δuⱼ) ⊡ ∂P∂Fδui * dΓ
            Kₑ[i, j] += δuᵢ ⋅ (((cofF ⊡ ∇δuⱼ) * one(cofF) - cofF ⋅ transpose(∇δuⱼ)) ⋅ neumann_term) ⋅ n₀ * dΓ
        end
    end
end
end

function update_face_cache(cell::CC, face_cache::SimpleFaceCache{MP}) where {CC, MP}
end
