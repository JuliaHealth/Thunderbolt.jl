using Thunderbolt, UnPack

import Thunderbolt.WriteVTK: paraview_collection 

# # 0 = G(du,u,q,p⁰,t)
# # 0 = L(dq,u,q,p ,t) = (L₁(Pₑ¹dq, Pₐ¹u, Pₑ¹q, p¹, t), ..., Lₙ(Pₑⁿdq, Pₐⁿu, Pₑⁿq, pⁿ, t))
# # Where Pₐⁱ,Pₑⁱ are the mappings from global to the local chunk for the respective variables.
# struct LocalGlobalFunction{GType, LTYpe} <: Thunderbolt.SciMLBase.AbstactNonlinearFunction
#     G::GType
#     L::LType
# end

# struct LocalGlobalProblem{F <: LocalGlobalFunction}
#     f::F
# end

# # TODO citation of original paper and Hartmann's benchmark paper
# struct MultilevelNewton
#     max_iter::Int
# end

# struct MultilevelNewtonCache{VT, JT, LCTG, LCTL}
#     max_iter::Int
#     residual_global::VT
#     residual_local::VT
#     δu::VT
#     J::JT
#     lincache_global::LCTG # Global linear solver
#     lincache_local::LCTL # Local linear solver
# end

# function OrdinaryDiffEq.build_nlsolver(
#     alg, nlalg::MultilevelNewton,
#     u, uprev, p, t, dt,
#     f::F, rate_prototype, ::Type{uEltypeNoUnits},
#     ::Type{uBottomEltypeNoUnits}, ::Type{tTypeNoUnits},
#     γ, c, α,
#     ::Val{true}) where {F, uEltypeNoUnits, uBottomEltypeNoUnits,
#     tTypeNoUnits}
# end

# struct QuasiStaticFunction{GType, LTYpe} <: OrdinaryDiffEq.AbstractDEFunction
#     G::GType
#     L::LType
# end

# struct QuasiStaticProblem{GType, LTYpe} <: OrdinaryDiffEq.AbstractDEFunction
#     G::GType
#     L::LType
# end

# function OrdinaryDiffEq.__solve(prob::QuasiStaticProblem)
# end

struct LinearViscoelasticity{T,TV,FT}
    K::T
    G::T
    viscosity::TV
    traction::FT
end

# Ferrite-specific
struct MaterialCache{M, DH, CV, FV}
    material::M
    dh::DH
    cv::CV
    fv::FV
end

function assemble_cell_Ju!(ke, u, q, cell, cellvalues, material::LinearViscoelasticity)
    q_offset     = getnquadpoints(cellvalues)*(cellid(cell)-1)
    @unpack Gᵥ   = material.viscosity
    @unpack K, G = material
    for q_point in 1:getnquadpoints(cellvalues)
        Eᵥ = frommandel(SymmetricTensor{2,3}, @view q[q_offset+q_point, :])
        # Get the integration weight for the quadrature point
        dΩ = getdetJdV(cellvalues, q_point)
        for i in 1:getnbasefunctions(cellvalues)
            # Gradient of the trial function
            ∇Nᵢ = shape_symmetric_gradient(cellvalues, q_point, i)
            # Stress (Eq. 3)
            E = ∇Nᵢ
            T = K*tr(E)*one(E) + 2*(G+Gᵥ)*dev(E) - 2*Gᵥ*dev(Eᵥ)
            for j in 1:getnbasefunctions(cellvalues)
                # Symmetric gradient of the test function
                ∇ˢʸᵐNⱼ = shape_symmetric_gradient(cellvalues, q_point, j)
                ke[i, j] += (∇ˢʸᵐNⱼ ⊡ T) * dΩ
            end
        end
    end
    return nothing
end

function assemble_cell_Jq!(ke, u, q, cell, cellvalues, material::LinearViscoelasticity)
    q_offset     = getnquadpoints(cellvalues)*(cellid(cell)-1)
    @unpack Gᵥ   = material.viscosity
    @unpack K, G = material
    for q_point in 1:getnquadpoints(cellvalues)
        Eᵥ = frommandel(SymmetricTensor{2,3}, @view q[q_offset+q_point, :])
        # Get the integration weight for the quadrature point
        dΩ = getdetJdV(cellvalues, q_point)
        for i in 1:getnbasefunctions(cellvalues)
            C = gradient(ϵ -> -2*Gᵥ*dev(ϵ), Eᵥ);
            for j in 1:getnbasefunctions(cellvalues)
                # Gradient of the test function
                ∇Nⱼ = shape_symmetric_gradient(cellvalues, q_point, j)
                ke[j, q_point] += (∇Nⱼ ⊡ C) * dΩ
            end
        end
    end
    return nothing
end

function assemble_cell!(ke, fe, u, q, cell, cellvalues, material::LinearViscoelasticity)
    assemble_cell_Ju!(ke, u, q, cell, cellvalues, material)
    return nothing
end

# Standard assembly loop for global problem
function stress!(du, u, q, cache::MaterialCache, t)
    K = allocate_matrix(cache.dh)
    f = zeros(ndofs(cache.dh))
    # Allocate the element stiffness matrix
    n_basefuncs = getnbasefunctions(cache.cv)
    ke = zeros(n_basefuncs, n_basefuncs)
    fe = zeros(n_basefuncs)
    # Create an assembler
    assembler = start_assemble(K, f)
    # Loop over all cells
    for cell in CellIterator(cache.dh)
        # Update the shape function gradients based on the cell coordinates
        reinit!(cache.cv, cell)
        # Reset the element stiffness matrix
        fill!(ke, 0.0)
        fill!(fe, 0.0)
        # Compute element contribution
        assemble_cell!(ke, fe, u, q, cell, cache.cv, material)
        # Assemble ke into K
        assemble!(assembler, celldofs(cell), ke, fe)
    end
    for face in FacetIterator(cache.dh, getfacetset(cache.dh.grid,"right"))
        # Update the facetvalues to the correct facet number
        reinit!(cache.fv, face)
        # Reset the temporary array for the next facet
        fill!(fe, 0.0)
        # Access the cell's coordinates
        cell_coordinates = getcoordinates(face)
        for qp in 1:getnquadpoints(cache.fv)
            # Calculate the global coordinate of the quadrature point.
            x = spatial_coordinate(cache.fv, qp, cell_coordinates)
            tₚ = cache.material.traction(x, t)
            # Get the integration weight for the current quadrature point.
            dΓ = getdetJdV(cache.fv, qp)
            for i in 1:getnbasefunctions(cache.fv)
                Nᵢ = shape_value(cache.fv, qp, i)
                fe[i] -= tₚ ⋅ Nᵢ * dΓ
            end
        end
        # Add the local contributions to the correct indices in the global external force vector
        assemble!(f, celldofs(face), fe)
    end
    du .= K*u .- f
end

function stress_Ju!(K, u, q, cache::MaterialCache, t)
    # Allocate the element stiffness matrix
    n_basefuncs = getnbasefunctions(cache.cv)
    ke = zeros(n_basefuncs, n_basefuncs)
    # Create an assembler
    assembler = start_assemble(K; fillzero=false)
    # Loop over all cells
    for cell in CellIterator(dh)
        # Update the shape function gradients based on the cell coordinates
        reinit!(cache.cv, cell)
        # Reset the element stiffness matrix
        fill!(ke, 0.0)
        # Compute element contribution
        assemble_cell_Ju!(ke, u, q, cell, cache.cv, cache.material)
        # Assemble ke into K
        assemble!(assembler, celldofs(cell), ke)
    end
end

function stress_Jq!(K, u, q, material::MaterialCache, t)
    # Allocate the element stiffness matrix
    n_basefuncs = getnbasefunctions(cache.cv)
    ke = zeros(n_basefuncs, n_basefuncs)
    # Create an assembler
    assembler = start_assemble(K)
    # Loop over all cells
    for cell in CellIterator(dh)
        # Update the shape function gradients based on the cell coordinates
        reinit!(cache.cv, cell)
        # Reset the element stiffness matrix
        fill!(ke, 0.0)
        # Compute element contribution
        assemble_cell_Jq!(ke, u, q, cell, cache.cv, material)
        # Assemble ke into K
        assemble!(assembler, celldofs(cell), ke)
    end
end

struct OverstressedViscosity{T}
    Gᵥ::T
    η₀::T
    s₀::T
end

struct FerriteElementChunkInfo{CV}
    cv::CV
end

struct FerriteQuadratureInfo{CV}
    cv::CV
    qpᵢ::Int
end

function viscosity_evolution(u, q, p::OverstressedViscosity, t, local_chunk_info)
    @unpack Gᵥ, η₀, s₀ = p
    @unpack cv, qpᵢ = local_chunk_info

    E   = function_symmetric_gradient(cv, qpᵢ, u)
    Eᵥ  = frommandel(SymmetricTensor{2,3}, q)
    Tₒᵥ = 2Gᵥ * (E-Eᵥ) #dev(E-Eᵥ)
    η   = η₀ #* exp(-s₀*norm(Tₒᵥ))

    return tomandel(Tₒᵥ/η)
end

function viscosity_evolution_Jq(u, q, p::OverstressedViscosity, t, local_chunk_info)
    @unpack Gᵥ, η₀, s₀ = p
    @unpack cv, qpᵢ = local_chunk_info

    E   = function_symmetric_gradient(cv, qpᵢ, u)
    Tₒᵥ(Eᵥ) = 2Gᵥ * (E-Eᵥ) #dev(E-Eᵥ)
    η(Eᵥ)   = η₀ #* exp(-s₀*norm(Tₒᵥ(Eᵥ)))

    Eᵥin = frommandel(SymmetricTensor{2,3}, q)
    C = gradient(Eᵥ -> Tₒᵥ(Eᵥ)/η(Eᵥ), Eᵥin)
    return tomandel(C)
end

function viscosity_evolution_Ju(u, q, p::OverstressedViscosity, t, local_chunk_info)
    @unpack Gᵥ, η₀, s₀ = p
    @unpack cv, qpᵢ = local_chunk_info

    Ju = zeros(SymmetricTensor{2,3}, getnbasefunctions(cv))

    # Ein    = function_symmetric_gradient(cv, qpᵢ, u)
    Eᵥ     = frommandel(SymmetricTensor{2,3}, q)
    Tₒᵥ(E) = 2Gᵥ * (E-Eᵥ) #dev(E-Eᵥ)
    η(E)   = η₀ #* exp(-s₀*norm(Tₒᵥ))

    for i in 1:getnbasefunctions(cv)
        ∇Nᵢ = shape_symmetric_gradient(cv, qpᵢ, i)
        C   = gradient(E -> Tₒᵥ(E)/η(E), ∇Nᵢ)
        Ju[i] = C ⊡ ∇Nᵢ
    end

    return Ju
end
# function solve_local()
#     # for each local chunk # Element loop
#     #     for each element in chunk # Quadrature loop
#     #         solve stage ,,L(uₗ, qₗ) = 0'' with fixed uₗ for qₗ
#     #         solve dQ/dU = [∂L/∂Q]^-1 ∂L/∂
#     #         update ∂G/∂U with dQ/dU
#     #     end
#     # end
# end

function traction(x, t)
    return Vec(0.0, 0.0, 0.0)
end

# Hartmann 2005 Table 5
material = LinearViscoelasticity(
    1667.00,
       8.75,
    OverstressedViscosity(
        150.0,
        13.0,
        1.1,
    ),
    traction,
)

function generate_linear_elasticity_problem()
    grid = generate_grid(Hexahedron, (2, 1, 1));

    dim = 3
    order = 1 # linear interpolation
    ip = Lagrange{RefHexahedron, order}()^dim; # vector valued interpolation

    qr = QuadratureRule{RefHexahedron}(2)
    qr_face = FacetQuadratureRule{RefHexahedron}(2);

    cellvalues = CellValues(qr, ip)
    facetvalues = FacetValues(qr_face, ip);

    dh = DofHandler(grid)
    add!(dh, :u, ip)
    close!(dh);

    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, getfacetset(grid, "left"),   (x, t) -> Vec((0.0,0.0,0.0)), [1,2,3]))
    add!(ch, Dirichlet(:u, getfacetset(grid, "right"),   (x, t) -> Vec((0.1,0.0,0.0)), [1,2,3]))
    close!(ch);

    return dh, ch, cellvalues, facetvalues
end

dh,ch,cv,fv = generate_linear_elasticity_problem()
cache = MaterialCache(
    material,
    dh,
    cv,
    fv,
)
u₀ = zeros(ndofs(dh))
q₀ = zeros(getnquadpoints(cv)*getncells(dh.grid)*6)

dt = 0.01
T  = 1.0

# init
J = allocate_matrix(dh)
global_residual = zeros(ndofs(dh))
u = copy(u₀)
q = copy(q₀)
qprev = copy(q₀)
qmat     = reshape(q,     (getnquadpoints(cv)*getncells(dh.grid), 6)) # This can be relaxed
qprevmat = reshape(qprev, (getnquadpoints(cv)*getncells(dh.grid), 6)) # This can be relaxed

max_iter = 100 # Should be 1

# solve

# Translation Ferrite->SciML
global_f          = stress!
global_jacobian_u = stress_Ju!
global_jacobian_q = stress_Jq!
chunk_jacobian_q  = assemble_cell_Jq!
local_jacobian_q  = viscosity_evolution_Jq
local_jacobian_u  = viscosity_evolution_Ju
local_f           = viscosity_evolution

pvd = paraview_collection("linear-viscoelasticity")
for t ∈ 0.0:dt:T
    # For each stage ... this is a backward Euler for now, so just 1 stage :)
    # Outer newton loop
    for outer_iter in 1:max_iter
        #Compute residual
        apply!(u, ch)
        global_f(global_residual, u, qmat, cache, t+dt)

        #Setup Jacobian
        # 1. Local solves and Jacobian corrections (normally these are done IN global_jacobian_u for efficiency reasons)
        # for each local chunk # Element loop
        ∂G∂Qₑ = zeros(SymmetricTensor{2,3}, (getnbasefunctions(cache.cv), getnquadpoints(cache.cv))) # TODO flatten to second order tensor
        dQdUₑ = zeros(SymmetricTensor{2,3}, (getnquadpoints(cache.cv), getnbasefunctions(cache.cv)))
        assembler = start_assemble(J; fillzero=true)
        ke = zeros(getnbasefunctions(cache.cv), getnbasefunctions(cache.cv))
        for cell in CellIterator(dh)
            # prepare iteration
            reinit!(cache.cv, cell)
            fill!(ke, 0.0)
            fill!(∂G∂Qₑ, zero(SymmetricTensor{2,3}))

            # chunk_info = FerriteElementChunkInfo(cache.cv, qp, keq)
            ue = @views u[celldofs(cell)] # TODO copy for better cache utilization
            q_offset     = getnquadpoints(cv)*(cellid(cell)-1)
            # for each item in local chunk # Quadrature loop
            for qp in 1:getnquadpoints(cache.cv)
                chunk_item = FerriteQuadratureInfo(cache.cv, qp)
                # solve local problem
                local_q     = @view qmat[q_offset+qp, :]
                local_qprev = @view qprevmat[q_offset+qp, :]
                for inner_iter in 1:max_iter
                    # setup system for local backward Euler solve (qₙ₊₁ - qₙ)/Δt - rhs(...) = 0
                    local_J        = local_jacobian_q(ue, local_q, cache.material.viscosity, t+dt, chunk_item)
                    local_dq       = local_f(ue, local_q, cache.material.viscosity, t+dt, chunk_item)
                    local_residual = (local_q .- local_qprev) .- dt .* local_dq
                    # solve linear sytem and update local solution
                    local_Δq       = (one(local_J) - (dt .*local_J)) \ local_residual
                    local_q      .-= local_Δq
                    # check convergence
                    resnorm        = norm(local_residual)
                    # println("Progress... $inner_iter cell=$(cellid(cell))|qp=$qp at t=$t resnorm=$resnorm")
                    (resnorm < 1e-8 || norm(local_Δq) < 1e-8) && break
                    inner_iter == max_iter && error("Newton diverged for cell=$(cellid(cell))|qp=$qp at t=$t resnorm=$resnorm")
                end
                # Contribution to corrector part ,,dQdUₑ``
                local_J     = local_jacobian_q(ue, local_q, cache.material.viscosity, t+dt, chunk_item)
                invlocal_Jq = inv(local_J)
                ∂L∂U = local_jacobian_u(u, q, cache.material.viscosity, t+dt, chunk_item)
                for j in 1:getnbasefunctions(cv)
                    dQdUₑ[qp,j] = frommandel(SymmetricTensor{2,3}, invlocal_Jq * tomandel(∂L∂U[j]))
                end
            end
            # update local Jacobian contribution
            # Contribution to corrector part ,,∂G∂Qₑ``
            chunk_jacobian_q(∂G∂Qₑ, ue, qmat, cell, cache.cv, cache.material)
            # Correction of global Jacobian
            for i in 1:getnbasefunctions(cache.cv)
                for j in 1:getnbasefunctions(cache.cv)
                    for k in 1:getnquadpoints(cache.cv)
                        ke[i,j] += ∂G∂Qₑ[i,k] ⊡ dQdUₑ[k,j]
                    end
                end
            end
            assemble!(assembler, celldofs(cell), ke)
        end
        # 2. Jacobian of global function G w.r.t. to global vector u
        global_jacobian_u(J, u, qmat, cache, t+dt)
        # 3. Apply boundary conditions
        apply_zero!(J, global_residual, ch)
        # 4. Solve linear system
        Δu = J \ global_residual
        # 5. Update solution
        u .-= Δu
        #Check convergence
        rnorm = norm(global_residual)
        @info "$t: Iteration=$outer_iter rnorm=$rnorm"
        if rnorm < 1e-7
            break
        end
        if norm(Δu) < 1e-8
            break
        end
        outer_iter == max_iter && error("max iter")
    end
    qprev .= q

    VTKGridFile("linear-viscoelasticity-$t.vtu", dh) do vtk
        write_solution(vtk, dh, u)
        # for (i, key) in enumerate(("11", "22", "12"))
        #     write_cell_data(vtk, avg_cell_stresses[i], "sigma_" * key)
        # end
        # write_projection(vtk, proj, stress_field, "stress field")
        # Ferrite.write_cellset(vtk, grid)
        pvd[t] = vtk
    end
end
# close(pvd)
