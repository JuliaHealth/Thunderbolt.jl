function setup_boundary_cache(boundary_models::Tuple, qr::FacetQuadratureRule, sdh::SubDofHandler)
    length(boundary_models) == 0 && return EmptySurfaceElementCache()
    return CompositeSurfaceElementCache(
        ntuple(i->setup_boundary_cache(boundary_models[i], qr, sdh), length(boundary_models))
    )
end

@doc raw"""
Any boundary condition stated in the weak form
```math
\int f(u, n_0), \delta u \mathrm{d} \partial \Omega_0
```
"""
abstract type AbstractWeakBoundaryCondition end

@doc raw"""
    RobinBC(α, boundary_name::String)

```math
\bm{P}(\bm{u}) \cdot \bm{n}_0 = - \alpha \bm{u} \quad \textbf{x} \in \partial \Omega_0,
```
"""
struct RobinBC <: AbstractWeakBoundaryCondition
    α::Float64
    boundary_name::String
end

@doc raw"""
    NormalSpringBC(kₛ boundary_name::String)

```math
\bm{P}(\bm{u}) \cdot \bm{n}_0 = - k_s \bm{u} \cdot n_0 \quad \textbf{x} \in \partial \Omega_0,
```
"""
struct NormalSpringBC <: AbstractWeakBoundaryCondition
    kₛ::Float64
    boundary_name::String
end

@doc raw"""
    BendingSpringBC(kᵇ, boundary_name::String)

```math
\bm{P}(\bm{u}) \cdot \bm{n}_0 = - \partial_F \frac{1}{2} k_b \left (cof(F) n_0 - n_0 \right) \quad \textbf{x} \in \partial \Omega_0,
```
"""
struct BendingSpringBC <: AbstractWeakBoundaryCondition
    kᵇ::Float64
    boundary_name::String
end

@doc raw"""
    ConstantPressureBC(p::Real, boundary_name::String)

```math
\bm{P}(\bm{u}) \cdot \bm{n}_0 = - p n_0 \quad \textbf{x} \in \partial \Omega_0,
```
"""
struct ConstantPressureBC <: AbstractWeakBoundaryCondition
    p::Float64
    boundary_name::String
end

@doc raw"""
    PressureFieldBC(pressure_field, boundary_name::String)

```math
\bm{P}(\bm{u}) \cdot \bm{n}_0 = - k_s \bm{u} \cdot n_0 \quad \textbf{x} \in \partial \Omega_0,
```
"""
struct PressureFieldBC{C} <: AbstractWeakBoundaryCondition
    pc::C
    boundary_name::String
end

"""
Standard cache for surface integrals.
"""
struct SimpleFacetCache{MP, FV} <: AbstractSurfaceElementCache
    mp::MP
    fv::FV
end
@inline is_facet_in_cache(facet::FacetIndex, cell::CellCache, face_cache::SimpleFacetCache) = facet ∈ getfacetset(cell.grid, getboundaryname(face_cache))
@inline getboundaryname(face_cache::SimpleFacetCache) = face_cache.mp.boundary_name

function setup_boundary_cache(face_model::AbstractWeakBoundaryCondition, qr::FacetQuadratureRule, sdh::SubDofHandler)
    @assert length(sdh.dh.field_names) == 1 "Support for multiple fields not yet implemented."
    field_name = first(sdh.dh.field_names)
    ip         = Ferrite.getfieldinterpolation(sdh, field_name)
    ip_geo = geometric_subdomain_interpolation(sdh)
    return SimpleFacetCache(face_model, FacetValues(qr, ip, ip_geo))
end

function assemble_face!(Kₑ::AbstractMatrix, residualₑ::AbstractVector, uₑ::AbstractVector, cell, local_face_index::Int, cache::SimpleFacetCache{<:RobinBC}, time)
    @unpack mp, fv = cache
    @unpack α = mp

    reinit!(fv, cell, local_face_index)

    ndofs_face = getnbasefunctions(fv)
    for qp in QuadratureIterator(fv)
        dΓ = getdetJdV(fv, qp)

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

function assemble_face!(Kₑ::AbstractMatrix, uₑ::AbstractVector, cell, local_face_index, cache::SimpleFacetCache{<:RobinBC}, time)
    @unpack mp, fv = cache
    @unpack α = mp

    reinit!(fv, cell, local_face_index)

    ndofs_face = getnbasefunctions(fv)
    for qp in QuadratureIterator(fv)
        dΓ = getdetJdV(fv, qp)

        u_q = function_value(fv, qp, uₑ)
        ∂²Ψ∂u², ∂Ψ∂u = Tensors.hessian(u -> α*u⋅u, u_q, :all)

        # Add contribution to the residual from this test function
        for i in 1:ndofs_face
            δuᵢ = shape_value(fv, qp, i)
    
            for j in 1:ndofs_face
                δuⱼ = shape_value(fv, qp, j)
                # Add contribution to the tangent
                Kₑ[i, j] += ( δuᵢ ⋅ ∂²Ψ∂u² ⋅ δuⱼ ) * dΓ
            end
        end
    end
end

function assemble_face!(residualₑ::AbstractVector, uₑ::AbstractVector, cell, local_face_index::Int, cache::SimpleFacetCache{<:RobinBC}, time)
    @unpack mp, fv = cache
    @unpack α = mp

    reinit!(fv, cell, local_face_index)

    ndofs_face = getnbasefunctions(fv)
    for qp in QuadratureIterator(fv)
        dΓ = getdetJdV(fv, qp)

        u_q = function_value(fv, qp, uₑ)
        ∂Ψ∂u = Tensors.gradient(u -> α*u⋅u, u_q)

        # Add contribution to the residual from this test function
        for i in 1:ndofs_face
            δuᵢ = shape_value(fv, qp, i)
            residualₑ[i] += δuᵢ ⋅ ∂Ψ∂u * dΓ
        end
    end
end



function assemble_face!(Kₑ::AbstractMatrix, residualₑ::AbstractVector, uₑ::AbstractVector, cell, local_face_index, cache::SimpleFacetCache{<:NormalSpringBC}, time)
    @unpack mp, fv = cache
    @unpack kₛ = mp

    reinit!(fv, cell, local_face_index)

    ndofs_face = getnbasefunctions(fv)
    for qp in QuadratureIterator(fv)
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

function assemble_face!(Kₑ::AbstractMatrix, uₑ::AbstractVector, cell, local_face_index, cache::SimpleFacetCache{<:NormalSpringBC}, time)
    @unpack mp, fv = cache
    @unpack kₛ = mp

    reinit!(fv, cell, local_face_index)

    ndofs_face = getnbasefunctions(fv)
    for qp in QuadratureIterator(fv)
        dΓ = getdetJdV(fv, qp)
        N = getnormal(fv, qp)
    
        u_q = function_value(fv, qp, uₑ)
        ∂²Ψ∂u², ∂Ψ∂u = Tensors.hessian(u -> 0.5*kₛ*(u⋅N)^2, u_q, :all)
    
        # Add contribution to the residual from this test function
        for i in 1:ndofs_face
            δuᵢ = shape_value(fv, qp, i)
    
            for j in 1:ndofs_face
                δuⱼ = shape_value(fv, qp, j)
                # Add contribution to the tangent
                Kₑ[i, j] += ( δuᵢ ⋅ ∂²Ψ∂u² ⋅ δuⱼ ) * dΓ
            end
        end
    end
end

function assemble_face!(residualₑ::AbstractVector, uₑ::AbstractVector, cell, local_face_index, cache::SimpleFacetCache{<:NormalSpringBC}, time)
    @unpack mp, fv = cache
    @unpack kₛ = mp

    reinit!(fv, cell, local_face_index)

    ndofs_face = getnbasefunctions(fv)
    for qp in QuadratureIterator(fv)
        dΓ = getdetJdV(fv, qp)
        N = getnormal(fv, qp)

        u_q = function_value(fv, qp, uₑ)
        ∂Ψ∂u = Tensors.gradient(u -> 0.5*kₛ*(u⋅N)^2, u_q)

        # Add contribution to the residual from this test function
        for i in 1:ndofs_face
            δuᵢ = shape_value(fv, qp, i)
            residualₑ[i] += δuᵢ ⋅ ∂Ψ∂u * dΓ
        end
    end
end



function assemble_face!(Kₑ::AbstractMatrix, residualₑ::AbstractVector, uₑ::AbstractVector, cell, local_face_index, cache::SimpleFacetCache{<:BendingSpringBC}, time)
    @unpack mp, fv = cache
    @unpack kᵇ = mp

    reinit!(fv, cell, local_face_index)

    ndofs_face = getnbasefunctions(fv)
    for qp in QuadratureIterator(fv)
        dΓ = getdetJdV(fv, qp)
        N = getnormal(fv, qp)

        ∇u = function_gradient(fv, qp, uₑ)
        F = one(∇u) + ∇u

        ∂²Ψ∂F², ∂Ψ∂F = Tensors.hessian(F_ -> 0.5*kᵇ*(transpose(inv(F_))⋅N - N)⋅(transpose(inv(F_))⋅N - N), F, :all)

        # Add contribution to the residual from this test function
        for i in 1:ndofs_face
            ∇δui = shape_gradient(fv, qp, i)
            residualₑ[i] += ∇δui ⊡ ∂Ψ∂F * dΓ

            ∇δui∂P∂F = ∇δui ⊡ ∂²Ψ∂F² # Hoisted computation
            for j in 1:ndofs_face
                ∇δuj = shape_gradient(fv, qp, j)
                # Add contribution to the tangent
                Kₑ[i, j] += ( ∇δui∂P∂F ⊡ ∇δuj ) * dΓ
            end
        end
    end
end

function assemble_face!(Kₑ::AbstractMatrix, uₑ::AbstractVector, cell, local_face_index, cache::SimpleFacetCache{<:BendingSpringBC}, time)
    @unpack mp, fv = cache
    @unpack kᵇ = mp

    reinit!(fv, cell, local_face_index)

    ndofs_face = getnbasefunctions(fv)
    for qp in QuadratureIterator(fv)
        dΓ = getdetJdV(fv, qp)
        N = getnormal(fv, qp)
    
        ∇u = function_gradient(fv, qp, uₑ)
        F = one(∇u) + ∇u
    
        ∂²Ψ∂F², ∂Ψ∂F = Tensors.hessian(F_ -> 0.5*kᵇ*(transpose(inv(F_))⋅N - N)⋅(transpose(inv(F_))⋅N - N), F, :all)
    
        # Add contribution to the residual from this test function
        for i in 1:ndofs_face
            ∇δui = shape_gradient(fv, qp, i)
    
            ∇δui∂P∂F = ∇δui ⊡ ∂²Ψ∂F² # Hoisted computation
            for j in 1:ndofs_face
                ∇δuj = shape_gradient(fv, qp, j)
                # Add contribution to the tangent
                Kₑ[i, j] += ( ∇δui∂P∂F ⊡ ∇δuj ) * dΓ
            end
        end
    end
end

function assemble_face!(residualₑ::AbstractVector, uₑ::AbstractVector, cell, local_face_index, cache::SimpleFacetCache{<:BendingSpringBC}, time)
    @unpack mp, fv = cache
    @unpack kᵇ = mp

    reinit!(fv, cell, local_face_index)

    ndofs_face = getnbasefunctions(fv)
    for qp in QuadratureIterator(fv)
        dΓ = getdetJdV(fv, qp)
        N = getnormal(fv, qp)

        ∇u = function_gradient(fv, qp, uₑ)
        F = one(∇u) + ∇u

        ∂Ψ∂F = Tensors.gradient(F_ -> 0.5*kᵇ*(transpose(inv(F_))⋅N - N)⋅(transpose(inv(F_))⋅N - N), F)

        # Add contribution to the residual from this test function
        for i in 1:ndofs_face
            ∇δui = shape_gradient(fv, qp, i)
            residualₑ[i] += ∇δui ⊡ ∂Ψ∂F * dΓ
        end
    end
end



function assemble_face_pressure_qp!(Kₑ::AbstractMatrix, residualₑ::AbstractVector, uₑ::AbstractVector, p, qp, fv::FacetValues)
    ndofs_face = getnbasefunctions(fv)

    dΓ = getdetJdV(fv, qp)
    n₀ = getnormal(fv, qp)

    ∇u = function_gradient(fv, qp, uₑ)
    F = one(∇u) + ∇u

    invF = inv(F)
    cofF = transpose(invF)
    J = det(F)
    neumann_term = p * J * cofF
    for i in 1:ndofs_face
        δuᵢ = shape_value(fv, qp, i)
        residualₑ[i] += neumann_term ⋅ n₀ ⋅ δuᵢ * dΓ

        for j in 1:ndofs_face
            ∇δuⱼ = shape_gradient(fv, qp, j)
            # Add contribution to the tangent
            #   δF^-1 = -F^-1 δF F^-1
            #   δJ = J tr(δF F^-1)
            # Product rule
            δcofF = -transpose(invF ⋅ ∇δuⱼ ⋅ invF)
            δJ = J * tr(∇δuⱼ ⋅ invF)
            δJcofF = δJ * cofF + J * δcofF
            Kₑ[i, j] += p * (δJcofF ⋅ n₀) ⋅ δuᵢ * dΓ
        end
    end
end

function assemble_face_pressure_qp!(Kₑ::AbstractMatrix, uₑ::AbstractVector, p, qp, fv::FacetValues)
    ndofs_face = getnbasefunctions(fv)

    dΓ = getdetJdV(fv, qp)
    n₀ = getnormal(fv, qp)

    ∇u = function_gradient(fv, qp, uₑ)
    F = one(∇u) + ∇u
    
    invF = inv(F)
    cofF = transpose(invF)
    J = det(F)
    # neumann_term = p * J * cofF
    for i in 1:ndofs_face
        δuᵢ = shape_value(fv, qp, i)

        for j in 1:ndofs_face
            ∇δuⱼ = shape_gradient(fv, qp, j)
            # Add contribution to the tangent
            #   δF^-1 = -F^-1 δF F^-1
            #   δJ = J tr(δF F^-1)
            # Product rule
            δcofF = -transpose(invF ⋅ ∇δuⱼ ⋅ invF)
            δJ = J * tr(∇δuⱼ ⋅ invF)
            δJcofF = δJ * cofF + J * δcofF
            Kₑ[i, j] += p * (δJcofF ⋅ n₀) ⋅ δuᵢ * dΓ
        end
    end
end

function assemble_face_pressure_qp!(residualₑ::AbstractVector, uₑ::AbstractVector, p, qp, fv::FacetValues)
    ndofs_face = getnbasefunctions(fv)

    dΓ = getdetJdV(fv, qp)
    n₀ = getnormal(fv, qp)

    ∇u = function_gradient(fv, qp, uₑ)
    F = one(∇u) + ∇u

    invF = inv(F)
    cofF = transpose(invF)
    J = det(F)
    neumann_term = p * J * cofF
    for i in 1:ndofs_face
        δuᵢ = shape_value(fv, qp, i)
        residualₑ[i] += neumann_term ⋅ n₀ ⋅ δuᵢ * dΓ
    end
end


function assemble_face!(Kₑ::AbstractMatrix, residualₑ::AbstractVector, uₑ::AbstractVector, cell, local_face_index, cache::SimpleFacetCache{<:PressureFieldBC}, time)
    @unpack mp, fv = cache
    @unpack pc = mp

    reinit!(fv, cell, local_face_index)

    for qp in QuadratureIterator(fv)
        p = evaluate_coefficient(pc, cell, qp, time)
        assemble_face_pressure_qp!(Kₑ, residualₑ, uₑ, p, qp, fv)
    end
end

function assemble_face!(Kₑ::AbstractMatrix, uₑ::AbstractVector, cell, local_face_index, cache::SimpleFacetCache{<:PressureFieldBC}, time)
    @unpack mp, fv = cache
    @unpack pc = mp

    reinit!(fv, cell, local_face_index)

    for qp in QuadratureIterator(fv)
        # Add contribution to the residual from this test function
        p = evaluate_coefficient(pc, cell, qp, time)
        assemble_face_pressure_qp!(Kₑ, uₑ, p, qp, fv)
    end
end

function assemble_face!(residualₑ::AbstractVector, uₑ::AbstractVector, cell, local_face_index, cache::SimpleFacetCache{<:PressureFieldBC}, time)
    @unpack mp, fv = cache
    @unpack pc = mp

    reinit!(fv, cell, local_face_index)

    for qp in QuadratureIterator(fv)
        p = evaluate_coefficient(pc, cell, qp, time)
        assemble_face_pressure_qp!(residualₑ, uₑ, p, qp, fv)
    end
end



function assemble_face!(Kₑ::AbstractMatrix, residualₑ::AbstractVector, uₑ::AbstractVector, cell, local_face_index, cache::SimpleFacetCache{<:ConstantPressureBC}, time)
    @unpack mp, fv = cache
    @unpack p = mp

    reinit!(fv, cell, local_face_index)

    for qp in QuadratureIterator(fv)
        assemble_face_pressure_qp!(Kₑ, residualₑ, uₑ, p, qp, fv)
    end
end

function assemble_face!(Kₑ::AbstractMatrix, uₑ::AbstractVector, cell, local_face_index, cache::SimpleFacetCache{<:ConstantPressureBC}, time)
    @unpack mp, fv = cache
    @unpack p = mp

    reinit!(fv, cell, local_face_index)

    for qp in QuadratureIterator(fv)
        assemble_face_pressure_qp!(Kₑ, uₑ, p, qp, fv)
    end
end

function assemble_face!(residualₑ::AbstractVector, uₑ::AbstractVector, cell, local_face_index, cache::SimpleFacetCache{<:ConstantPressureBC}, time)
    @unpack mp, fv = cache
    @unpack p = mp

    reinit!(fv, cell, local_face_index)

    for qp in QuadratureIterator(fv)
        assemble_face_pressure_qp!(residualₑ, uₑ, p, qp, fv)
    end
end
