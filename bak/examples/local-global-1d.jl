using Thunderbolt, UnPack

import Thunderbolt.WriteVTK: paraview_collection 

struct LinearViscosity1D{T}
    E1::T
    eta1::T
end

struct LinearViscoelasticity1D{T,TV,FT}
    E0::T
    viscosity::TV
    traction::FT
end

# Ferrite-specific
struct ElementCache{M, DH, CV, FV}
    material::M
    dh::DH
    cv::CV
    fv::FV
end

function assemble_cell_Ju!(ke, u, q, cell, cellvalues, material::LinearViscoelasticity1D)
    q_offset     = getnquadpoints(cellvalues)*(cellid(cell)-1)
    @unpack E1   = material.viscosity
    @unpack E0   = material
    for q_point in 1:getnquadpoints(cellvalues)
        # Get the integration weight for the quadrature point
        dΩ = getdetJdV(cellvalues, q_point)
        for i in 1:getnbasefunctions(cellvalues)
            # Gradient of the trial function
            ∇Nᵢ = shape_gradient(cellvalues, q_point, i)[1]
            # Stress (Eq. 3)
            T = (E0+E1)*∇Nᵢ #- E1*Eᵥ->0
            for j in 1:getnbasefunctions(cellvalues)
                # Symmetric gradient of the test function
                ∇ˢʸᵐNⱼ = shape_gradient(cellvalues, q_point, j)[1]
                ke[i, j] += (∇ˢʸᵐNⱼ * T) * dΩ
            end
        end
    end
    return nothing
end

function assemble_cell_Jq!(ke, u, q, cell, cellvalues, material::LinearViscoelasticity1D)
    q_offset     = getnquadpoints(cellvalues)*(cellid(cell)-1)
    @unpack E1   = material.viscosity
    @unpack E0   = material
    for q_point in 1:getnquadpoints(cellvalues)
        Eᵥ = q[q_offset+q_point]
        # Get the integration weight for the quadrature point
        dΩ = getdetJdV(cellvalues, q_point)
        # for i in 1:getnbasefunctions(cellvalues)
            C = -E1
            for j in 1:getnbasefunctions(cellvalues)
                # Gradient of the test function
                ∇Nⱼ = shape_gradient(cellvalues, q_point, j)[1]
                ke[j, q_point] += (∇Nⱼ * C) * dΩ
            end
        # end
    end
    return nothing
end

function assemble_cell!(ke, fe, u, q, cell, cellvalues, material::LinearViscoelasticity1D)
    q_offset     = getnquadpoints(cellvalues)*(cellid(cell)-1)
    @unpack E1   = material.viscosity
    @unpack E0   = material
    for q_point in 1:getnquadpoints(cellvalues)
        E  = function_gradient(cv, q_point, u)[1]
        Eᵥ = q[q_offset+q_point]
        # Get the integration weight for the quadrature point
        dΩ = getdetJdV(cellvalues, q_point)
        for i in 1:getnbasefunctions(cellvalues)
            # Gradient of the trial function
            ∇Nᵢ = shape_gradient(cellvalues, q_point, i)[1]
            T = (E0+E1)*∇Nᵢ
            for j in 1:getnbasefunctions(cellvalues)
                # Symmetric gradient of the test function
                ∇ˢʸᵐNⱼ = shape_gradient(cellvalues, q_point, j)[1]
                ke[i, j] += (∇ˢʸᵐNⱼ * T) * dΩ
            end
            fe[i] += ∇Nᵢ * ((E0+E1)*E - E1*Eᵥ) * dΩ
        end
    end
    return nothing
end

# Standard assembly loop for global problem
function stress!(residual, u, q, cache::ElementCache, t)
    K = allocate_matrix(cache.dh)
    # Allocate the element stiffness matrix
    n_basefuncs = getnbasefunctions(cache.cv)
    ke = zeros(n_basefuncs, n_basefuncs)
    fe = zeros(n_basefuncs)
    # Create an assembler
    assembler = start_assemble(K, residual)
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
                fe[i] += tₚ * Nᵢ * dΓ
            end
        end
        # Add the local contributions to the correct indices in the global external force vector
        assemble!(residual, celldofs(face), fe)
    end
end

function stress_Ju!(K, u, q, cache::ElementCache, t)
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

function stress_Jq!(K, u, q, cache::ElementCache, t)
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
        assemble_cell_Jq!(ke, u, q, cell, cache.cv, cache.material)
        # Assemble ke into K
        assemble!(assembler, celldofs(cell), ke)
    end
end

struct FerriteElementChunkInfo{CV}
    cv::CV
end

struct FerriteQuadratureInfo{CV}
    cv::CV
    qpᵢ::Int
end

function viscosity_evolution(u, q, p::LinearViscosity1D, t, local_chunk_info)
    @unpack E1, eta1 = p
    @unpack cv, qpᵢ = local_chunk_info

    E   = function_gradient(cv, qpᵢ, u)[1]
    Eᵥ  = q[1]
    Tₒᵥ = E1*(E-Eᵥ)

    return Tₒᵥ/eta1
end

function viscosity_evolution_Jq(u, q, p::LinearViscosity1D, t, local_chunk_info)
    @unpack E1, eta1 = p
    return -E1/eta1
end

function viscosity_evolution_Ju(u, q, p::LinearViscosity1D, t, local_chunk_info)
    @unpack E1, eta1 = p
    @unpack cv, qpᵢ = local_chunk_info
    return [shape_gradient(cv, qpᵢ, i)[1]*E1/eta1 for i in 1:getnbasefunctions(cv)]
end

function traction(x, t)
    return 0.1
end

# Hartmann 2005 Table 5
material = LinearViscoelasticity1D(
    1.0, #70
    LinearViscosity1D(
        1.0,#20
        1.0,
    ),
    traction,
)

function generate_linear_elasticity_problem()
    grid = generate_grid(Line, (1,));

    order = 1 # linear interpolation
    ip = Lagrange{RefLine, order}() # vector valued interpolation

    qr = QuadratureRule{RefLine}(2)
    qr_face = FacetQuadratureRule{RefLine}(1);

    cellvalues = CellValues(qr, ip)
    facetvalues = FacetValues(qr_face, ip);

    dh = DofHandler(grid)
    add!(dh, :u, ip)
    close!(dh);

    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, getfacetset(grid, "left"),   (x, t) -> 0.0))
    # add!(ch, Dirichlet(:u, getfacetset(grid, "right"),   (x, t) -> 0.1))
    close!(ch);

    return dh, ch, cellvalues, facetvalues
end

dh,ch,cv,fv = generate_linear_elasticity_problem()
cache = ElementCache(
    material,
    dh,
    cv,
    fv,
)
u₀ = zeros(ndofs(dh))
q₀ = zeros(getnquadpoints(cv)*getncells(dh.grid))

dt = 1.0
T  = 2.0

# init
J = allocate_matrix(dh)
global_residual = zeros(ndofs(dh))
u = copy(u₀)
q = copy(q₀)
qprev = copy(q₀)
qmat     = reshape(q,     (getnquadpoints(cv)*getncells(dh.grid), 1)) # This can be relaxed
qprevmat = reshape(qprev, (getnquadpoints(cv)*getncells(dh.grid), 1)) # This can be relaxed

max_iter   = 3 # Should be 2
local_tol⁰ = 1e-16
global_tol = 1e-8

# solve
Δu = zeros(ndofs(dh))
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
        # Apply constraints to solution
        apply!(u, ch)

        #Setup Jacobian
        # 1. Local solves and Jacobian corrections (normally these are done IN global_jacobian_u for efficiency reasons)
        # for each local chunk # Element loop
        ∂G∂Qₑ = zeros((getnbasefunctions(cache.cv), getnquadpoints(cache.cv))) # TODO flatten to second order tensor
        dQdUₑ = zeros((getnquadpoints(cache.cv), getnbasefunctions(cache.cv)))

        assembler = start_assemble(J; fillzero=true)
        for cell in CellIterator(dh)
            # prepare iteration
            reinit!(cache.cv, cell)
            ue          = @views u[celldofs(cell)] # TODO copy for better cache utilization
            q_offset    = getnquadpoints(cv)*(cellid(cell)-1)
            # for each item in local chunk # Quadrature loop
            for qp in 1:getnquadpoints(cache.cv)
                chunk_item = FerriteQuadratureInfo(cache.cv, qp)
                # solve local problem with Newton-Raphson
                local_q     = @view qmat[q_offset+qp, :]
                local_qprev = @view qprevmat[q_offset+qp, :]
                for inner_iter in 1:max_iter
                    # setup system for local backward Euler solve L(qₙ₊₁, uₙ₊₁) = (qₙ₊₁ - qₙ) - Δt*rhs(qₙ₊₁, uₙ₊₁) = 0 w.r.t. qₙ₊₁ with uₙ₊₁ frozen
                    local_J        = local_jacobian_q(ue, local_q, cache.material.viscosity, t+dt, chunk_item)
                    local_dq       = local_f(ue, local_q, cache.material.viscosity, t+dt, chunk_item)
                    local_residual = (local_q .- local_qprev) .- dt .* local_dq # = L(u,q)
                    # solve linear sytem and update local solution
                    ∂L∂q           = one(local_J) - (dt .* local_J)
                    local_Δq       = ∂L∂q \ local_residual
                    local_q      .-= local_Δq
                    # check convergence
                    resnorm        = norm(local_residual)
                    # println("Progress... $inner_iter cell=$(cellid(cell))|qp=$qp at t=$t resnorm=$resnorm")
                    local_tol = local_tol⁰ # min(norm(Δu)*norm(Δu), local_tol⁰)
                    (resnorm ≤ local_tol || norm(local_Δq) ≤ local_tol) && break
                    inner_iter == max_iter && error("Newton diverged for cell=$(cellid(cell))|qp=$qp at t=$t resnorm=$resnorm")
                end
                # Contribution to corrector part ,,dQdUₑ`` by solving the linear system with multiple right hand sides
                local_J     = local_jacobian_q(ue, local_q, cache.material.viscosity, t+dt, chunk_item)
                ∂L∂q        = (one(local_J) - (dt .* local_J))
                inv∂L∂q     = inv(∂L∂q)
                ∂L∂U        = -dt * local_jacobian_u(ue, local_q, cache.material.viscosity, t+dt, chunk_item)
                for j in 1:getnbasefunctions(cv)
                    dQdUₑ[qp,j] = inv∂L∂q * -∂L∂U[j]
                end
            end
            # update local Jacobian contribution
            # Contribution to corrector part ,,∂G∂Qₑ``
            fill!(∂G∂Qₑ, 0.0)
            chunk_jacobian_q(∂G∂Qₑ, ue, qmat, cell, cache.cv, cache.material)
            # Correction of global Jacobian
            ke = ∂G∂Qₑ * dQdUₑ
            assemble!(assembler, celldofs(cell), ke)
        end
        # 2. Residual and Jacobian of global function G(qₙ₊₁, uₙ₊₁) w.r.t. to global vector uₙ₊₁ with qₙ₊₁ frozen
        global_f(global_residual, u, qmat, cache, t+dt)
        global_jacobian_u(J, u, qmat, cache, t+dt)
        @show J, global_residual
        # 3. Apply boundary conditions
        apply_zero!(J, global_residual, ch)
        # 4. Solve linear system
        Δu .= J \ global_residual
        # 5. Update solution
        u .-= Δu
        #Check convergence
        rnorm = norm(global_residual)
        @info "$t: Iteration=$outer_iter rnorm=$rnorm"
        if rnorm ≤ global_tol
            break
        end
        if norm(Δu) ≤ global_tol
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
