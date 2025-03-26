
# TODO where to put this?
function construct_qvector(::Type{StorageType}, ::Type{IndexType}, mesh::SimpleMesh, qrc::QuadratureRuleCollection, subdomains::Vector{String} = [""]) where {StorageType, IndexType}
    num_points = 0
    num_cells  = 0
    for subdomain in subdomains
        for (celltype, cellset) in pairs(mesh.volumetric_subdomains[subdomain].data)
            qr = getquadraturerule(qrc, getcells(mesh, first(cellset).idx))
            num_points += getnquadpoints(qr)*length(cellset)
            num_cells  += length(cellset)
        end
    end
    data    = zeros(eltype(StorageType), num_points)
    offsets = zeros(num_cells+1)

    offsets[1]        = 1
    next_point_offset = 1
    next_cell         = 1
    for subdomain in subdomains
        for (celltype, cellset) in pairs(mesh.volumetric_subdomains[subdomain].data)
            qr = getquadraturerule(qrc, getcells(mesh, first(cellset).idx))
            for cellidx in cellset
                next_point_offset += getnquadpoints(qr)
                next_cell += 1
                offsets[next_cell] = next_point_offset
            end
        end
    end

    return DenseDataRange(StorageType(data), IndexType(offsets))
end

function compute_quadrature_fluxes!(fluxdata, dh, u, field_name, integrator)
    grid = get_grid(dh)
    sdim = getspatialdim(grid)
    for sdh in dh.subdofhandlers
        ip          = Ferrite.getfieldinterpolation(sdh, field_name)
        firstcell   = getcells(grid, first(sdh.cellset))
        ip_geo      = Ferrite.geometric_interpolation(typeof(firstcell))^sdim
        element_qr  = getquadraturerule(integrator.qrc, firstcell)
        cv = CellValues(element_qr, ip, ip_geo)
        _compute_quadrature_fluxes_on_subdomain!(fluxdata,sdh,cv,u,integrator)
    end
end

function _compute_quadrature_fluxes_on_subdomain!(κ∇u,sdh,cv,u,integrator::BilinearDiffusionIntegrator)
    n_basefuncs = getnbasefunctions(cv)
    for cell ∈ CellIterator(sdh)
        κ∇ucell = get_data_for_index(κ∇u, cellid(cell))

        reinit!(cv, cell)
        uₑ = @view u[celldofs(cell)]

        for qp in QuadratureIterator(cv)
            D_loc = evaluate_coefficient(integrator.D, cell, qp, time)
            # dΩ = getdetJdV(cellvalues, qp)
            for i in 1:n_basefuncs
                ∇Nᵢ = shape_gradient(cv, qp, i)
                κ∇ucell[qp.i] += D_loc ⋅ ∇Nᵢ * uₑ[i]
            end
        end
    end
end

"""
    Plonsey1964ECGGaussCache(op::AbstractBilinearOperator, φₘ::AbstractVector)

Here φₘ is the solution vector containing the transmembranepotential, op is the associated diffusion opeartor and
κₜ is the torso's conductivity.

Returns a cache to compute the lead field with the form proposed in [Plo:1964:vcf](@cite)
with the Gauss theorem applied to it, as for example described in [OgiBalPer:2021:ema](@cite).
Calling [`evaluate_ecg`](@ref) with this method simply evaluates the following integral efficiently:

\$\\varphi_e(x)=\\frac{1}{4 \\pi \\kappa_t} \\int_\\Omega \\frac{ \\kappa_ ∇φₘ \\cdot (\\tilde{x}-x)}{||(\\tilde{x}-x)||^3}\\mathrm{d}\\tilde{x}\$

The important simplifications taken are:
   1. Surrounding volume is an infinite, homogeneous sphere with isotropic conductivity
   2. The extracellular space and surrounding volume share the same isotropic, homogeneous conductivity tensor
"""
struct Plonsey1964ECGGaussCache{BufferType, OperatorType}
    # Buffer for storing "κ(x) ∇φₘ(x,t)" at the quadrature points
    κ∇φₘ::BufferType
    op::OperatorType
end

function Plonsey1964ECGGaussCache(op::AssembledBilinearOperator, φₘ::AbstractVector{T}) where T
    @unpack dh, integrator = op
    @assert length(dh.field_names) == 1 "Multiple fields detected. Problem setup might be broken..."
    grid = get_grid(dh)
    sdim = Ferrite.getspatialdim(grid)
    κ∇φₘ = construct_qvector(Vector{Vec{sdim,T}}, Vector{Int64}, grid, integrator.qrc)
    compute_quadrature_fluxes!(κ∇φₘ, dh, φₘ, dh.field_names[1], integrator)
    Plonsey1964ECGGaussCache(κ∇φₘ, op)
end

"""
    evaluate_ecg(method::Plonsey1964ECGGaussCache, x::Vec, κₜ::Real)

Compute the pseudo ECG at a given point x by evaluating:

\$\\varphi_e(x)=\\frac{1}{4 \\pi \\kappa_t} \\int_\\Omega \\frac{ \\kappa_ ∇φₘ \\cdot (\\tilde{x}-x)}{||(\\tilde{x}-x)||^3}\\mathrm{d}\\tilde{x}\$

For more information please read the docstring for [`Plonsey1964ECGGaussCache`](@ref)
"""
function evaluate_ecg(method::Plonsey1964ECGGaussCache, x::Vec, κₜ::Real)
    φₑ = 0.0
    @unpack κ∇φₘ, op = method
    @unpack dh = op
    @assert length(dh.field_names) == 1 "Multiple fields detected. Problem setup might be broken..."
    grid = get_grid(dh)
    sdim = getspatialdim(grid)
    for sdh in dh.subdofhandlers
        ip          = Ferrite.getfieldinterpolation(sdh, first(dh.field_names))
        firstcell   = getcells(grid, first(sdh.cellset))
        ip_geo      = Ferrite.geometric_interpolation(typeof(firstcell))^sdim
        element_qr  = getquadraturerule(op.integrator.qrc, firstcell)
        cv = CellValues(element_qr, ip, ip_geo)
        # Function barrier
        φₑ += _evaluate_ecg_inner!(κ∇φₘ, method, x, κₜ, sdh, cv)
    end

    return -φₑ / (4π*κₜ)
end

function evaluate_ecg(method::Plonsey1964ECGGaussCache, x::AbstractVector{<:Vec}, κₜ::Real)
    φₑ = zeros(length(x))
    for i in eachindex(x)
        φₑ = evaluate_ecg(method, x[i], κₜ)
    end
    return φₑ
end

function _evaluate_ecg_inner!(κ∇φₘ, method::Plonsey1964ECGGaussCache, x::Vec, κₜ::Real, sdh, cv)
    φₑ = 0.0
    for cell ∈ CellIterator(sdh)
        reinit!(cv, cell)
        coords = getcoordinates(cell)
        κ∇φₘe = get_data_for_index(κ∇φₘ, cellid(cell))
        φₑ += _evaluate_ecg_plonsey_gauss(κ∇φₘe, coords, cv, x)
    end
    return φₑ
end

function _evaluate_ecg_plonsey_gauss(κ∇φₘ, coords::AbstractVector{Vec{sdim,T}}, cv, x::Vec{sdim,T}) where {sdim, T}
    φₑ_local = 0.0
    n_geom_basefuncs = Ferrite.getngeobasefunctions(cv)
    @inbounds for (qp, w) in pairs(Ferrite.getweights(cv.qr))
        # Compute dΩ
        mapping = Ferrite.calculate_mapping(cv.geo_mapping, qp, coords)
        dΩ = Ferrite.calculate_detJ(Ferrite.getjacobian(mapping)) * w
        # Compute x̃
        x̃ = spatial_coordinate(cv, qp, coords)
        # Evaluate κ∇φₘ*(x̃-x)/||x̃-x||³
        φₑ_local += κ∇φₘ[qp] ⋅ (x̃-x)/norm((x̃-x))^3 * dΩ
    end
    return φₑ_local
end


function update_ecg!(cache::Plonsey1964ECGGaussCache, φₘ::AbstractVector{T}) where T
    @unpack op = cache
    @unpack dh, integrator = op
    grid = get_grid(dh)
    sdim = Ferrite.getspatialdim(grid)
    fill!(cache.κ∇φₘ.data, zero(eltype(cache.κ∇φₘ)))
    compute_quadrature_fluxes!(cache.κ∇φₘ, dh, φₘ, dh.field_names[1], integrator)
end

"""
    PoissonECGReconstructionCache(fₑₚ::GenericSplitFunction, Ωₜ::AbstractMesh, κᵢ, κ, electrodes::AbstractVector{<:Vec}; ground, linear_solver, solution_vector_type, system_matrix_type)

Sets up a cache for calculating ``\\varphi_\\mathrm{e}`` by solving the Poisson problem
```math
\\nabla \\cdot (\\boldsymbol{\\kappa}_{\\mathrm{i}} + \\boldsymbol{\\kappa}_{\\mathrm{e}}) \\nabla \\varphi_{\\mathrm{e}}=-\\nabla \\cdot\\left(\\boldsymbol{\\kappa}_{\\mathrm{i}} \\nabla \\varphi_\\mathrm{m}\\right)
```
as for example proposed in [PotDubRicVinGul:2006:cmb](@cite) and investigated in [OgiBalPer:2021:ema](@cite) (as well as other studies). Here κₑ is the extracellular conductivity tensor and κᵢ is the intracellular conductivity tensor. The cache includes the assembled
stiffness matrix with applied homogeneous Dirichlet boundary condition at the first vertex of the mesh. As the problem is solved for each timestep with only the right hand side changing.

## Keyword Arguments
* `ground               = Set([VertexIndex(1, 1)])`
* `linear_solver        = LinearSolve.KrylovJL_CG()`
* `solution_vector_type = Vector{Float64}`
* `system_matrix_type   = ThreadedSparseMatrixCSR{Float64,Int64}`

"""
struct PoissonECGReconstructionCache{DiffusionOperatorType1, DiffusionOperatorType2, TransferOperatorType, SolutionVectorType, SolverCacheType, PHType, CHType}
    torso_op::DiffusionOperatorType1  # Operator on the torso mesh for ∇κ∇
    heart_op::DiffusionOperatorType2  # Operator on the heart mesh for ∇κᵢ∇
    transfer_op::TransferOperatorType # Transfer from heart to torso mesh
    ϕₑ::SolutionVectorType            # Solution vector buffer
    κ∇φₘ_h::SolutionVectorType        # Source term buffer on heart
    κ∇φₘ_t::SolutionVectorType        # Source term buffer on torso
    inner_solver::SolverCacheType     # Linear solver
    ph::PHType                        # PointEvalHandler on the torso
    ch::CHType                        # ConstraintHandler on the torso
end

# Convenience ctor to unpack default setup
function PoissonECGReconstructionCache(
    epfun::GenericSplitFunction,
    torso_grid::AbstractGrid,
    heart_diffusion_tensor_field, # κᵢ - diffusion tensor description for heart on heart grid
    torso_diffusion_tensor_field, # κ - diffusion tensor description for heart and torso on torso grid
    electrode_positions::AbstractVector{<:Vec};
    ground               = Set([VertexIndex(1, 1)]),
    linear_solver        = LinearSolve.KrylovJL_CG(),
    solution_vector_type = Vector{Float64},
    system_matrix_type   = ThreadedSparseMatrixCSR{Float64,Int64},
)
    PoissonECGReconstructionCache(
        epfun.functions[1],
        torso_grid,
        heart_diffusion_tensor_field,
        torso_diffusion_tensor_field,
        electrode_positions;
        ground,
        linear_solver,
        solution_vector_type,
        system_matrix_type,
    )
end

function PoissonECGReconstructionCache(
    heart_fun::AffineODEFunction,
    torso_grid::AbstractGrid,
    heart_diffusion_tensor_field, # κᵢ - diffusion tensor description for heart on heart grid
    torso_diffusion_tensor_field, # κ - diffusion tensor description for heart and torso on torso grid
    electrode_positions::AbstractVector{<:Vec};
    ipc                  = LagrangeCollection{1}(),
    qrc                  = QuadratureRuleCollection(2),
    ground               = OrderedSet([VertexIndex(1, 1)]),
    linear_solver        = LinearSolve.KrylovJL_CG(),
    solution_vector_type = Vector{Float64},
    system_matrix_type   = ThreadedSparseMatrixCSR{Float64,Int64},
    extracellular_potential_symbol = :φₑ,
)
    heart_dh      = heart_fun.dh
    heart_grid = get_grid(heart_dh)
    length(heart_dh.field_names) == 1 || @warn "Multiple fields detected. Setup might be broken..."

    torso_model = SteadyDiffusionModel(
        torso_diffusion_tensor_field,
        NoStimulationProtocol(), #ConstantCoefficient(NaN), # FIXME Poisoning to detecte if we accidentally touch these
        extracellular_potential_symbol
    )

    torso_fun = semidiscretize(
        torso_model,
        FiniteElementDiscretization(
            Dict(extracellular_potential_symbol => ipc),
            [Dirichlet(extracellular_potential_symbol, ground, (x,t) -> 0.0)]
        ),
        torso_grid
    )

    torso_dh = torso_fun.dh
    torso_ch = torso_fun.ch

    transfer_op = NodalIntergridInterpolation(
        heart_dh,
        torso_dh,
        first(heart_dh.field_names),
        first(torso_dh.field_names),
    )

    heart_op = setup_assembled_operator(
        BilinearDiffusionIntegrator(
            heart_diffusion_tensor_field,
            qrc,
            extracellular_potential_symbol,
        ),
        system_matrix_type,
        heart_dh,
    )
    update_operator!(heart_op, 0.) # Trigger assembly

    torso_op = setup_assembled_operator(
        BilinearDiffusionIntegrator(
            torso_diffusion_tensor_field,
            qrc,
            extracellular_potential_symbol,
        ),
        system_matrix_type,
        torso_dh,
    )
    update_operator!(torso_op, 0.) # Trigger assembly

    # Setup electrodes
    ph = PointEvalHandler(torso_grid, electrode_positions; warn=false)
    if !all(x -> x !== nothing, ph.cells)
        error("Poisson reconstruction setup failed! Some electrodes are not found in the torso mesh ($(ph.cells)).")
    end

    PoissonECGReconstructionCache(
        heart_fun,
        torso_fun,
        heart_op,
        torso_op,
        transfer_op,
        ph;
        linear_solver,
        solution_vector_type,
    )
end

function PoissonECGReconstructionCache(
    heart_fun::AffineODEFunction,
    torso_fun::AffineSteadyStateFunction,
    heart_op::AssembledBilinearOperator,
    torso_op::AssembledBilinearOperator,
    transfer_op::AbstractTransferOperator,
    ph::PointEvalHandler;
    linear_solver        = LinearSolve.KrylovJL_CG(),
    solution_vector_type = Vector{Float64},
)
    torso_dh = torso_op.dh
    torso_ch = torso_fun.ch
    grid = get_grid(torso_dh)
    length(torso_dh.field_names) == 1 || @warn "Multiple fields detected. Setup might be broken..."

    κ∇φₘh = create_system_vector(solution_vector_type, heart_fun) # RHS buffer before transfer
    κ∇φₘt = create_system_vector(solution_vector_type, torso_fun) # RHS buffer after transfer
    κ∇φₘt .= 0.0
    ϕₑ    = create_system_vector(solution_vector_type, torso_fun) # Solution vector

    linprob  = LinearSolve.LinearProblem(
        torso_op.A, κ∇φₘt; u0=ϕₑ
    )
    lincache = init(linprob, linear_solver)

    return PoissonECGReconstructionCache(torso_op, heart_op, transfer_op, ϕₑ, κ∇φₘh, κ∇φₘt, lincache, ph, torso_ch)
end

function update_ecg!(cache::PoissonECGReconstructionCache, φₘ::AbstractVector)
    # Compute κᵢ∇φₘ on the heart
    mul!(cache.κ∇φₘ_h, cache.heart_op, φₘ)
    # Transfer κᵢ∇φₘ to the torso
    transfer!(cache.κ∇φₘ_t, cache.transfer_op, cache.κ∇φₘ_h)
    cache.κ∇φₘ_t[isnan.(cache.κ∇φₘ_t)] .= 0.0 # FIXME
    # "Move to right hand side
    cache.κ∇φₘ_t .*= -1.0
    # Apply BC to linear system
    apply_zero!(cache.inner_solver.A, cache.inner_solver.b, cache.ch)
    # Solve κ∇φₑ = -κᵢ∇φₘ for φₑ
    LinearSolve.solve!(cache.inner_solver)
    return nothing
end

# Batch evaluate all electrodes
function evaluate_ecg(cache::PoissonECGReconstructionCache)
    dh = cache.torso_op.dh
    return evaluate_at_points(cache.ph, dh, cache.ϕₑ, first(dh.field_names))
end

"""
    Geselowitz1989ECGLeadCache(problem, torso_grid, κ, κᵢ, electrode_sets, [ground, linear_solver, solution_vector_type, system_matrix_type])

Here the lead field, `Z`, is computed using the discretization of `problem`.
The lead field is computed as the solution of
```math
\\nabla \\cdot(\\mathbf{\\kappa} \\nabla Z)=\\left\\{\\begin{array}{cl}
-1 & \\text { at the positive electrode } \\\\
1 & \\text { at the negative electrode } \\\\
0 & \\text { else where }
\\end{array}\\right.
```
Where ``\\kappa`` is the bulk conductivity tensor.

Returns a cache contain the lead fields that are used to compute the lead potentials as proposed in [Ges:1989:ote](@cite).
Calling [`reinit!`](@ref) with this method simply evaluates the following integral efficiently:

```math
V(t)=\\int \\nabla Z(\\boldsymbol{x}) \\cdot \\boldsymbol{\\kappa}_\\mathrm{i} \\nabla \\varphi_\\mathrm{m} \\,\\mathrm{d}\\boldsymbol{x}.
```
"""
struct Geselowitz1989ECGLeadCache{TZ <: AbstractMatrix, DiffusionOperatorType, TransferOperatorType, SolutionVectorType <: AbstractVector, ElectrodesVecType}
    heart_op::DiffusionOperatorType   # Operator on the heart mesh for ∇κᵢ∇
    transfer_op::TransferOperatorType # Transfer from heart to torso mesh
    κ∇φₘ_h::SolutionVectorType        # Source term buffer on heart
    κ∇φₘ_t::SolutionVectorType        # Source term buffer on torso
    Z::TZ                             # Lead field
    electrode_positions::ElectrodesVecType
end

function Geselowitz1989ECGLeadCache(
    heart_fun::GenericSplitFunction,
    torso_grid::AbstractGrid,
    heart_diffusion_tensor_field, # κᵢ - diffusion tensor description for heart on heart grid
    full_diffusion_tensor_field,  # κ - diffusion tensor description for heart and torso on torso grid
    electrode_positions::AbstractVector{<:Vector{<:Vec}};
    ipc                  = LagrangeCollection{1}(),
    qrc                  = QuadratureRuleCollection(2),
    ground               = OrderedSet([VertexIndex(1, 1)]),
    linear_solver        = LinearSolve.KrylovJL_CG(),
    solution_vector_type = Vector{Float64},
    system_matrix_type   = ThreadedSparseMatrixCSR{Float64,Int64},
)
    return Geselowitz1989ECGLeadCache(
        heart_fun.functions[1],
        torso_grid,
        heart_diffusion_tensor_field, # κᵢ - diffusion tensor description for heart on heart grid
        full_diffusion_tensor_field,  # κ - diffusion tensor description for heart and torso on torso grid
        [[get_closest_vertex(position, torso_grid) for position in positions] for positions in electrode_positions];
        ipc                 ,
        qrc                 ,
        ground              ,
        linear_solver       ,
        solution_vector_type,
        system_matrix_type  ,
    )
end

function Geselowitz1989ECGLeadCache(
    heart_fun::AffineODEFunction,
    torso_grid::AbstractGrid,
    heart_diffusion_tensor_field, # κᵢ - diffusion tensor description for heart on heart grid
    full_diffusion_tensor_field,  # κ - diffusion tensor description for heart and torso on torso grid
    electrode_positions::AbstractVector{<:Vector{<:Vec}};
    ipc                  = LagrangeCollection{1}(),
    qrc                  = QuadratureRuleCollection(2),
    ground               = OrderedSet([VertexIndex(1, 1)]),
    linear_solver        = LinearSolve.KrylovJL_CG(),
    solution_vector_type = Vector{Float64},
    system_matrix_type   = ThreadedSparseMatrixCSR{Float64,Int64},
)
    return Geselowitz1989ECGLeadCache(
        heart_fun,
        torso_grid,
        heart_diffusion_tensor_field, # κᵢ - diffusion tensor description for heart on heart grid
        full_diffusion_tensor_field,  # κ - diffusion tensor description for heart and torso on torso grid
        [[get_closest_vertex(position, torso_grid) for position in positions] for positions in electrode_positions];
        ipc                 ,
        qrc                 ,
        ground              ,
        linear_solver       ,
        solution_vector_type,
        system_matrix_type  ,
    )
end

function Geselowitz1989ECGLeadCache(
    heart_fun::AffineODEFunction,
    torso_grid::AbstractGrid,
    heart_diffusion_tensor_field, # κᵢ - diffusion tensor description for heart on heart grid
    full_diffusion_tensor_field,  # κ - diffusion tensor description for heart and torso on torso grid
    electrode_positions::AbstractVector{Vector{VertexIndex}};
    ipc                  = LagrangeCollection{1}(),
    qrc                  = QuadratureRuleCollection(2),
    ground               = OrderedSet([VertexIndex(1, 1)]),
    linear_solver        = LinearSolve.KrylovJL_CG(),
    solution_vector_type = Vector{Float64},
    system_matrix_type   = ThreadedSparseMatrixCSR{Float64,Int64},
    lead_field_sym       = :Z,
)
    tmpsym = heart_fun.bilinear_term.sym
    lead_field_model = SteadyDiffusionModel(
        full_diffusion_tensor_field,
        NoStimulationProtocol(), #ConstantCoefficient(NaN), # FIXME Poisoning to detecte if we accidentally touch these
        lead_field_sym
    )

    source_model = SteadyDiffusionModel(
        heart_diffusion_tensor_field,
        NoStimulationProtocol(), #ConstantCoefficient(NaN), # FIXME Poisoning to detecte if we accidentally touch these
        tmpsym
    )

    lead_field_fun = semidiscretize(
        lead_field_model,
        FiniteElementDiscretization(
            Dict(lead_field_sym => ipc),
            [Dirichlet(lead_field_sym, ground, (x,t) -> 0.0)]
        ),
        torso_grid
    )


    sourcefun = semidiscretize(
        source_model,
        FiniteElementDiscretization(
            Dict(tmpsym => ipc),
            Dirichlet[]
        ),
        get_grid(heart_fun.dh)
    )

    ϕₘ_op = setup_assembled_operator(
        BilinearDiffusionIntegrator(
            heart_diffusion_tensor_field,
            qrc,
            tmpsym,
        ),
        system_matrix_type,
        sourcefun.dh,
    )
    update_operator!(ϕₘ_op, 0.) # Trigger assembly

    lead_op = setup_assembled_operator(
        BilinearDiffusionIntegrator(
            full_diffusion_tensor_field,
            qrc,
            lead_field_sym,
        ),
        system_matrix_type,
        lead_field_fun.dh,
    )
    update_operator!(lead_op, 0.) # Trigger assembly

    lead_field_dh = lead_field_fun.dh
    heart_dh = heart_fun.dh

    transfer_op = NodalIntergridInterpolation(
        heart_dh,
        lead_field_dh,
        first(heart_dh.field_names),
        first(lead_field_dh.field_names),
    )

    Geselowitz1989ECGLeadCache(
        heart_fun,
        lead_field_fun,
        lead_op,
        ϕₘ_op,
        transfer_op,
        electrode_positions;
        linear_solver,
        solution_vector_type,
        lead_field_sym,
    )
end

function Geselowitz1989ECGLeadCache(
    heart_fun::AffineODEFunction,
    lead_fun::AffineSteadyStateFunction,
    lead_op::AssembledBilinearOperator,
    source_op::AssembledBilinearOperator,
    transfer_op,
    electrode_positions::AbstractVector{Vector{VertexIndex}};
    linear_solver        = LinearSolve.KrylovJL_CG(),
    solution_vector_type = Vector{Float64},
    lead_field_sym       = :Z,
)
    lead_dh  = lead_op.dh
    length(lead_dh.field_names) == 1 || @warn "Multiple fields detected. Setup might be broken..."
    nelectrodes = length(electrode_positions)
    ∇Njκ∇φₘ_h   = create_system_vector(solution_vector_type, heart_fun) # Solution vector
    ∇Njκ∇φₘ_t   = create_system_vector(solution_vector_type, lead_fun)  # Solution vector
    Z           = zeros(eltype(∇Njκ∇φₘ_t), nelectrodes, length(∇Njκ∇φₘ_t))
    ϕₑ          = zeros(nelectrodes)

    lead_rhs = zeros(eltype(∇Njκ∇φₘ_t), nelectrodes, length(∇Njκ∇φₘ_t))

    @views for (i, electrode_set) in enumerate(electrode_positions)
        @assert length(electrode_set) ≥ 2 "Electrode set $i has too few electrodes ($(length(electrode_set))<2)"
        current_rhs = lead_rhs[i,:]
        _add_electrode!(current_rhs, lead_dh, electrode_set[1], 1.0, lead_field_sym)
        for j in 2:length(electrode_set)
            _add_electrode!(current_rhs, lead_dh, electrode_set[j], -1.0/(length(electrode_set)-1), lead_field_sym)
        end
        leadprob = LinearSolve.LinearProblem(
            lead_op.A, current_rhs; u0=Z[i,:]
        )
        lincache = init(leadprob, linear_solver)
        LinearSolve.solve!(lincache)
    end

    return Geselowitz1989ECGLeadCache(source_op, transfer_op, ∇Njκ∇φₘ_h, ∇Njκ∇φₘ_t, Z, electrode_positions)
end

function _add_electrode!(f::AbstractVector{T}, dh::DofHandler, electrode::VertexIndex, weight, lead_field_sym::Symbol) where {T<:Number}
    local_dof = Ferrite.vertexdof_indices(Ferrite.getfieldinterpolation(dh.subdofhandlers[1], lead_field_sym))[electrode[2]][1]::Int
    global_dof = celldofs(dh, electrode[1])[local_dof]::Int
    f[global_dof] = -weight
    return nothing
end

function update_ecg!(cache::Geselowitz1989ECGLeadCache, φₘ::AbstractVector)
    # Compute κᵢ∇φₘ on the heart
    mul!(cache.κ∇φₘ_h, cache.heart_op, φₘ)
    # Transfer κᵢ∇φₘ to the torso
    transfer!(cache.κ∇φₘ_t, cache.transfer_op, cache.κ∇φₘ_h)
    cache.κ∇φₘ_t[isnan.(cache.κ∇φₘ_t)] .= 0.0 # FIXME
    return nothing
end

# Batch evaluate all electrodes
function evaluate_ecg(cache::Geselowitz1989ECGLeadCache)
    return -cache.Z * cache.κ∇φₘ_t
end
