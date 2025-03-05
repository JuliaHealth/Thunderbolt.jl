using Thunderbolt, UnPack

import Thunderbolt.WriteVTK: paraview_collection 

# Local out-of-place functions of pure ODE form dq = f(u,q,p,t)
abstract type LocalODEModel end

# ------------------------------------------------------------------
# -----------vvv HERE WE DEFINE THE LOCAL MODELS vvv----------------
# -------------------------- 1D ------------------------------------
# Dispatch helper and parameters for local function
struct LinearViscosity1D{T} <: LocalODEModel
    E₁::T
    η₁::T
end

# this is "g" in "dq = g(u,q,p,t)"
function viscosity_evolution(E, Eᵥ, p::LinearViscosity1D, t)
    @unpack E₁, η₁ = p
    return E₁/η₁*(E-Eᵥ)
end

# this is "dg(u,q)/dq"
function viscosity_evolution_Jq(E, Eᵥ, p, t)
    return gradient(Eᵥ2 -> viscosity_evolution(E, Eᵥ2, p, t), Eᵥ)
    # @unpack E₁, η₁ = p
    # return Tensor{2,1}((-E₁/η₁,))
end

# this is "dg(u,q)/du"
function viscosity_evolution_Ju(E, Eᵥ, p, t)
    return gradient(E2 -> viscosity_evolution(E2, Eᵥ, p, t), E)
    # @unpack E₁, η₁ = p
    # return Tensor{2,1}((E₁/η₁,))
end

# Transformation from vector to tensor form
function q_to_local_form(q::AbstractArray, model::LinearViscosity1D)
    return Tensor{2,1}((q[1],))
end

# Transformation from tensor form to vector
function store_local_form_q_to!(qstorage::AbstractArray, qform::Tensor{2}, model::LinearViscosity1D)
    qstorage[1] = qform[1]
end

# Local storage determinator
q_size(model::LinearViscosity1D) = 1


# # -------------------------- 3D ------------------------------------
# struct LinearViscosity3D{T} <: LocalODEModel
#     E₁::T
#     η₁::T
# end

# # this is "g" in "dq = g(u,q,p,t)"
# function viscosity_evolution(E, Eᵥ, p::LinearViscosity3D, t)
#     @unpack E₁, η₁ = p
#     return E₁/η₁*(E-Eᵥ)
# end

# # Transformation from vector to tensor form
# function q_to_local_form(q::AbstractArray, model::LinearViscosity3D)
#     return SymmetricTensor{2,3}(ntuple(i->q[i],6))
# end

# # Transformation from tensor form to vector
# function store_local_form_q_to!(qstorage::AbstractArray, qform::Tensor{2}, model::LinearViscosity3D)
#     qstorage .= qform.data
# end

# # Local storage determinator
# q_size(model::LinearViscosity3D) = 6
# -----------^^^ HERE WE DEFINE THE LOCAL MODEL ^^^----------------
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# -----------vvv HERE WE DEFINE THE GLOBAL MODEL vvv----------------
# -------------------------- 1D ------------------------------------
struct LinearViscoelasticity1D{T,TV,FT}
    E₀::T
    viscosity::TV
    traction::FT
end
q_size(model::LinearViscoelasticity1D) = q_size(model.viscosity)

# Smart unpacking of the materials
global_∂q(E,Eᵥ,model::LinearViscoelasticity1D,cache) = global_∂q(u,q,model.viscosity,cache)
global_∂q(E,Eᵥ,model::LinearViscosity1D,cache) = -model.E₁
# -------------------------- 2D and 3D ------------------------------
struct LinearViscoelasticity{T,TV,FT}
    E₀::T
    viscosity::TV
    traction::FT
end

# -----------^^^ HERE WE DEFINE THE GLOBAL MODEL ^^^----------------
# ------------------------------------------------------------------

# Ferrite-specific variables
struct StructuralElementCache{M, DH, CH, CV, FV}
    material::M
    dh::DH
    ch::CH
    cv::CV
    fv::FV
end

struct QuasiStaticSolutionState{UT, QT} # = "LocalGlobalSolution
    u::UT
    q::QT
end

function structural_weak_form_integrand(uq::QuasiStaticSolutionState, model::LinearViscoelasticity1D)
    E    = uq.u
    Eᵥ   = Tensor{2,1}((uq.q[1],))
    σ    = model.E₀*E + model.viscosity.E₁*(E-Eᵥ)
    ∂σ∂E = Tensor{4,1}(((model.E₀ + model.viscosity.E₁),))
    return σ, ∂σ∂E
end

# struct DynamicSolutionState{UT, VT, QT}
#     u::UT
#     v::VT
#     q::QT
# end

function generate_linear_viscoelasticity_function(material::LinearViscoelasticity1D)
    grid = generate_grid(Line, (1,));

    order = 1 # linear interpolation
    ip = Lagrange{RefLine, order}()^1 # vector valued interpolation

    qr = QuadratureRule{RefLine}(2)
    qr_face = FacetQuadratureRule{RefLine}(1);

    cellvalues = CellValues(qr, ip)
    facetvalues = FacetValues(qr_face, ip);

    dh = DofHandler(grid)
    add!(dh, :u, ip)
    close!(dh);

    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, getfacetset(grid, "left"),   (x, t) -> 0.0))
    close!(ch);

    return StructuralElementCache(material, dh, ch, cellvalues, facetvalues)
end

function generate_linear_viscoelasticity_function(material::LinearViscoelasticity)
    grid = generate_grid(Hexahedron, (1,1,1));

    order = 1 # linear interpolation
    ip = Lagrange{RefHexahedron, order}() # vector valued interpolation

    qr = QuadratureRule{RefHexahedron}(2)
    qr_face = FacetQuadratureRule{RefHexahedron}(2);

    cellvalues = CellValues(qr, ip)
    facetvalues = FacetValues(qr_face, ip);

    dh = DofHandler(grid)
    add!(dh, :u, ip)
    close!(dh);

    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, getfacetset(grid, "left"),   (x, t) -> 0.0))
    close!(ch);

    return StructuralElementCache(material, dh, ch, cellvalues, facetvalues)
end

abstract type AbstractStageCache end

abstract type AbstractMultiLevelStageCache <: AbstractStageCache end

struct MultiLevelBackwardEulerStage{T, ST} <: AbstractMultiLevelStageCache
    t::T   # Integrator
    dt::T  # Integrator
    solprev::ST
    # Multilevel solver info
    tol::T
    max_iter::Int
end

struct LocalBackwardEulerCache{T, ST, ST2}
    t::T   # Integrator
    dt::T  # Integrator
    sol::ST # View into local part of global u and q
    solprev::ST2 # View into local part of global uprev and qprev
    # NLSolver
    tol::T
    max_iter::Int
end

struct QuadraturePointStageInfo{iType, CV}
    qpi::iType
    cellvalues::CV
    # TODO QuadratureHandler
end

function local_solve(A::FourthOrderTensor, b::AbstractTensor,cache)
    return inv(A) ⊡ b
end
function local_solve(A::SecondOrderTensor, b::AbstractTensor, cache)
    return A \ b
end

# This function MUST NOT allocate or have any runtime dispatch (to support GPU)
function build_local_cache(sol::QuasiStaticSolutionState, cell::CellCache, info::QuadraturePointStageInfo, stage::MultiLevelBackwardEulerStage)
    @unpack cellvalues = info
    udofs = celldofs(cell)
    uprev = @view stage.solprev.u[udofs]
    Eprev = function_gradient(cellvalues, info.qpi, uprev)

    # qdofs = @view quadraturedofs(...) # TODO implement and use this instead of the manual extration below
    sol_size_local = length(sol.q)
    nqp            = getnquadpoints(cellvalues)*getncells(cell.grid)
    qprev_t3 = reshape(stage.solprev.q, (sol_size_local, nqp))
    qoffset  = getnquadpoints(cellvalues)*(cellid(cell)-1)
    qprev    = @view qprev_t3[:, qoffset + info.qpi]

    # Wrap into correct solution type
    solprev = QuasiStaticSolutionState(
        Eprev,
        qprev,
    )

    # Construct local cache object to control the dispatches
    return LocalBackwardEulerCache(
        stage.t,
        stage.dt,
        sol,
        solprev,
        stage.tol,
        stage.max_iter,
    )
end

function local_problem_residual(cache::LocalBackwardEulerCache, model::LocalODEModel)
    @unpack t, dt = cache
    u     = cache.sol.u
    uprev = cache.solprev.u
    q     = q_to_local_form(cache.sol.q, model)
    qprev = q_to_local_form(cache.solprev.q, model)
    dq       = local_rhs(u, q, model, t)
    residual = (q - qprev) - dt * dq # = L(u,q)
    return residual
end

solve_local_problem(cache::LocalBackwardEulerCache, model::LinearViscoelasticity1D) = solve_local_problem(cache, model.viscosity)

function solve_local_problem(cache::LocalBackwardEulerCache, model::LocalODEModel)
    @unpack t, dt, tol = cache
    u     = cache.sol.u
    uprev = cache.solprev.u
    q     = q_to_local_form(cache.sol.q, model)
    qprev = q_to_local_form(cache.solprev.q, model)

    # solve local problem with Newton-Raphson
    for inner_iter in 1:max_iter
        # setup system for local backward Euler solve L(qₙ₊₁, uₙ₊₁) = (qₙ₊₁ - qₙ) - Δt*rhs(qₙ₊₁, uₙ₊₁) = 0 w.r.t. qₙ₊₁ with uₙ₊₁ frozen
        residual = local_problem_residual(cache, model) # = L(u,q)
        # solve linear sytem and update local solution
        J        = local_∂q(u, q, model, t)
        ∂L∂q     = one(J) - (dt * J)
        Δq       = local_solve(∂L∂q, residual, cache)
        q       -= Δq
        # Update cache
        store_local_form_q_to!(cache.sol.q, q, model)
        # check convergence
        resnorm  = norm(residual)
        (resnorm ≤ tol || norm(Δq) ≤ tol) && break
        # inner_iter == max_iter && error("Newton diverged for cell=$(cellid(cell))|qp=$qp at t=$t resnorm=$resnorm")
        inner_iter == max_iter && error("t=$t resnorm=$resnorm")
    end

    return q
end

solve_corrector_problem(cache::LocalBackwardEulerCache, model::LinearViscoelasticity1D) = solve_corrector_problem(cache, model.viscosity)

function solve_corrector_problem(cache::LocalBackwardEulerCache, model::LocalODEModel)
    @unpack t, dt = cache
    u     = cache.sol.u
    q     = q_to_local_form(cache.sol.q, model)

    J    = local_∂q(u, q, model, t)
    ∂L∂Q = (one(J) - (dt * J))
    ∂L∂U = -dt * local_∂u(u, q, model, t)
    return local_solve(∂L∂Q, -∂L∂U, cache) # = dQdU
end

function element_routine!(ke, fe, sol::QuasiStaticSolutionState, cell, cellvalues, material::LinearViscoelasticity1D, stage_cache::AbstractMultiLevelStageCache)
    ke2 = zero(ke)
    # Unpack local vectors
    ue = sol.u
    qe = sol.q
    # Easy access
    qemat = reshape(qe, (q_size(material), getnquadpoints(cellvalues)))
    # Iterator info
    for q_point in 1:getnquadpoints(cellvalues)
        # Still iterator info
        info      = QuadraturePointStageInfo(q_point, cellvalues)           # This encodes framework-specific info relevant for the local solve
        # Common information
        E  = function_gradient(cellvalues, q_point, ue)                     # Frozen u for inner function - dereference here because of 1D function
        # 1. Solve for Qₙ₊₁
        Eᵥref     = @view qemat[:, q_point]                                 # Get reference for Qₙ₊₁
        local_sol = QuasiStaticSolutionState(E, Eᵥref)                      # ,,u and q'' for the local problem after another chain rule application
        cache     = build_local_cache(local_sol, cell, info, stage_cache)   # Setup cache for local solver in a non-allocating way
        Eᵥ        = solve_local_problem(cache, material)                    # Solve for new local solution
        # 2. Contribution to corrector part ,,dQdU`` by solving the linear system
        dQdE = solve_corrector_problem(cache, material) # Order 2 tensor
        # 3. ∂G∂Q
        ∂G∂Q = global_∂q(E, Eᵥ, material, cache)        # Order 2 tensor
        # 4. Evaluate weak form at quadrature point
        σ, ∂σ∂E = structural_weak_form_integrand(local_sol, material)
        # 5. Assemble element matrix
        # Get the integration weight for the quadrature point
        dΩ = getdetJdV(cellvalues, q_point)
        for i in 1:getnbasefunctions(cellvalues)
            # Gradient of the trial function
            ∇Nᵢ = shape_gradient(cellvalues, q_point, i)
            fe[i] += σ ⊡ ∇Nᵢ * dΩ
            σδE =  (∂σ∂E + ∂G∂Q ⊗ dQdE) ⊡ ∇Nᵢ
            for j in 1:getnbasefunctions(cellvalues)
                # Symmetric gradient of the test function
                ∇ˢʸᵐNⱼ = shape_gradient(cellvalues, q_point, j)
                ke[i, j] += (∇ˢʸᵐNⱼ ⊡ σδE) * dΩ
                ke2[i, j] += (∇ˢʸᵐNⱼ ⊡ ∂σ∂E ⊡ ∇Nᵢ) * dΩ
            end
        end
    end
    @show ke2
    return nothing
end

# Standard assembly loop for global problem
function stress!(J, residual, sol::QuasiStaticSolutionState, cache::StructuralElementCache, stage_cache::AbstractMultiLevelStageCache)
    @unpack u,q = sol

    qncomps = q_size(cache.material)

    f = zeros(ndofs(cache.dh))
    # Allocate the element stiffness matrix
    n_basefuncs = getnbasefunctions(cache.cv)
    ke = zeros(n_basefuncs, n_basefuncs)
    fe = zeros(n_basefuncs)
    ue = zeros(n_basefuncs)
    qe = zeros(qncomps*getnquadpoints(cache.cv))
    # Create an assembler
    assembler = start_assemble(J, residual)
    # Assemble volume terms
    for cell in CellIterator(cache.dh)
        # Update the shape function gradients based on the cell coordinates
        reinit!(cache.cv, cell)
        # Reset the element stiffness matrix
        fill!(ke, 0.0)
        fill!(fe, 0.0)
        # Extract field dofs
        ue .= u[celldofs(cell)]
        # Extract internal dofs
        qoffset1 = (cellid(cell)-1)*qncomps*getnquadpoints(cache.cv) + 1
        qoffset2 = qoffset1 + qncomps*getnquadpoints(cache.cv) - 1
        qrange  = qoffset1:qoffset2
        qe .= q[qrange]
        # Compute element contribution
        element_routine!(ke, fe, QuasiStaticSolutionState(ue, qe), cell, cache.cv, cache.material, stage_cache)
        # Assemble ke into K
        assemble!(assembler, celldofs(cell), ke, fe)
        # Update internal implicit solution
        q[qrange] .= qe
    end
    # Assemble surface terms
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
            tₚ = cache.material.traction(x, stage_cache.t)
            # Get the integration weight for the current quadrature point.
            dΓ = getdetJdV(cache.fv, qp)
            for i in 1:getnbasefunctions(cache.fv)
                Nᵢ = shape_value(cache.fv, qp, i)
                fe[i] -= tₚ ⋅ Nᵢ * dΓ
            end
        end
        # Add the local contributions to the correct indices in the global external force vector
        assemble!(residual, celldofs(face), fe)
    end
end

local_rhs = viscosity_evolution
local_∂u  = viscosity_evolution_Ju
local_∂q  = viscosity_evolution_Jq

function traction1(x, t)
    return Vec((0.1,))
end

function traction3(x, t)
    return Vec((0.1,0.0,0.0))
end

# Hartmann 2005 Table 5
material = LinearViscoelasticity1D(
    1.0, #70
    LinearViscosity1D(
        1.0,#20
        1.0,
    ),
    traction1,
)

f = generate_linear_viscoelasticity_function(material)

u₀ = zeros(ndofs(f.dh))
q₀ = zeros(getnquadpoints(f.cv)*getncells(f.dh.grid))

dt = 0.1
T  = 10.0

# init
J = allocate_matrix(f.dh)
global_residual = zeros(ndofs(f.dh))
u = copy(u₀)
uprev = copy(u₀)
q = copy(q₀)
qprev = copy(q₀)

sol = QuasiStaticSolutionState(u, q)
solprev = QuasiStaticSolutionState(uprev, qprev)

max_iter   = 10 # Should be 2
local_tol⁰ = 1e-16
global_tol = 1e-8

# solve
Δu = zeros(ndofs(f.dh))
# Translation Ferrite->SciML
global_fJu        = stress!

displacements = []

pvd = paraview_collection("linear-viscoelasticity")
for t ∈ 0.0:dt:T
    # For each stage ... this is a backward Euler for now, so just 1 stage :)
    # Outer newton loop
    update!(f.ch, t)
    for outer_iter in 1:max_iter
        # Apply constraints to solution
        apply!(u, f.ch)

        #Setup Jacobian for global problem
        local_tol = outer_iter == 1 ? local_tol⁰ : min(local_tol⁰, norm(Δu)^2) # A sufficient condition for quadratic convergence is that the local residual converges quadratically w.r.t. to the global residual (see Rabbat et al. [1979])
        local_solver_info = MultiLevelBackwardEulerStage(t, dt, solprev, local_tol, max_iter)

        global_fJu(J, global_residual, sol, f, local_solver_info)

        # 3. Apply boundary conditions
        apply_zero!(J, global_residual, f.ch)
        # 4. Solve linear system
        Δu .= J \ global_residual
        # 5. Update solution
        u .-= Δu
        #Check convergence
        rnorm = norm(global_residual)
        inorm = norm(Δu)
        @info "$t: Iteration=$outer_iter rnorm=$rnorm inorm=$inorm"
        if rnorm ≤ global_tol
            break
        end
        if norm(Δu) ≤ global_tol
            break
        end
        outer_iter == max_iter && error("max iter")
    end
    qprev .= q
    uprev .= u

    VTKGridFile("linear-viscoelasticity-$t.vtu", f.dh) do vtk
        write_solution(vtk, f.dh, u)
        # for (i, key) in enumerate(("11", "22", "12"))
        #     write_cell_data(vtk, avg_cell_stresses[i], "sigma_" * key)
        # end
        # write_projection(vtk, proj, stress_field, "stress field")
        # Ferrite.write_cellset(vtk, grid)
        pvd[t] = vtk
    end
    
    push!(displacements, u[end])
end
# close(pvd)
