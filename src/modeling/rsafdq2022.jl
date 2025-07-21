##########################################################################

mutable struct RSAFDQ2022SingleChamberTying{CVM}
    const pressure_dof_index_local::Int
    pressure_dof_index_global::Int
    const facets::OrderedSet{FacetIndex}
    const volume_method::CVM
    const displacement_symbol::Symbol
    V⁰ᴰval::Float64
    const V⁰ᴰidx_local::Int
    V⁰ᴰidx_global::Int
end

struct RSAFDQ2022TyingCache{FV <: FacetValues, CVM}
    fv::FV
    chambers::Vector{RSAFDQ2022SingleChamberTying{CVM}}
end

struct RSAFDQ2022TyingInfo{CVM}
    chambers::Vector{RSAFDQ2022SingleChamberTying{CVM}}
end

solution_size(problem::RSAFDQ2022TyingInfo) = length(problem.chambers)

function setup_tying_cache(tying_info::RSAFDQ2022TyingInfo, qr, sdh::SubDofHandler)
    @assert length(sdh.dh.field_names) == 1 "Support for multiple fields not yet implemented."
    field_name = first(sdh.dh.field_names)
    ip          = Ferrite.getfieldinterpolation(sdh, field_name)
    ip_geo = geometric_subdomain_interpolation(sdh)
    RSAFDQ2022TyingCache(FacetValues(qr, ip, ip_geo), tying_info.chambers)
end

function get_tying_dofs(tying_cache::RSAFDQ2022TyingCache, u)
    return [u[chamber.pressure_dof_index_local] for chamber in tying_cache.chambers]
end

"""
Pressure contribution (i.e. variation w.r.t. p) for the term
    ∫ p n(u) δu ∂Ω
 [= ∫ p J(u) F(u)^-T n₀ δu ∂Ω₀]
where p is the unknown chamber pressure and u contains the unknown deformation field.
"""
# Residual and Jacobian
function assemble_LFSI_coupling_contribution_col!(C, R, dh::AbstractDofHandler, u::AbstractVector, pressure, method::RSAFDQ2022SingleChamberTying)
    grid = dh.grid
    ip = Ferrite.getfieldinterpolation(dh.subdofhandlers[1], method.displacement_symbol)
    ip_geo = Ferrite.geometric_interpolation(typeof(getcells(grid, 1)))
    intorder = 2*Ferrite.getorder(ip)
    ref_shape = Ferrite.getrefshape(ip)
    qr_facet = FacetQuadratureRule{ref_shape}(intorder)
    fv = FacetValues(qr_facet, ip, ip_geo)

    for facet ∈ FacetIterator(dh, method.facets)
        assemble_LFSI_coupling_contribution_col_inner!(C, R, u, pressure, facet, dh, fv, method.displacement_symbol)
    end
end
# Residual only
function assemble_LFSI_coupling_contribution_col!(C, dh::AbstractDofHandler, u::AbstractVector, pressure, method::RSAFDQ2022SingleChamberTying)
    grid = dh.grid
    ip = Ferrite.getfieldinterpolation(dh.subdofhandlers[1], method.displacement_symbol)
    ip_geo = Ferrite.geometric_interpolation(typeof(getcells(grid, 1)))
    intorder = 2*Ferrite.getorder(ip)
    ref_shape = Ferrite.getrefshape(ip)
    qr_facet = FacetQuadratureRule{ref_shape}(intorder)
    fv = FacetValues(qr_facet, ip, ip_geo)

    for facet ∈ FacetIterator(dh, method.facets)
        assemble_LFSI_coupling_contribution_col_inner!(C, u, pressure, facet, dh, fv, method.displacement_symbol)
    end
end


function compute_chamber_volume(dh, u, setname, method::RSAFDQ2022SingleChamberTying)
    check_subdomains(dh)
    grid = dh.grid
    ip = Ferrite.getfieldinterpolation(dh.subdofhandlers[1], method.displacement_symbol)
    ip_geo = Ferrite.geometric_interpolation(typeof(getcells(grid, 1)))
    intorder = 2*Ferrite.getorder(ip)
    ref_shape = Ferrite.getrefshape(ip)
    qr_facet = FacetQuadratureRule{ref_shape}(intorder)
    fv = FacetValues(qr_facet, ip, ip_geo)

    volume = 0.0
    drange = dof_range(dh,method.displacement_symbol)
    for facet ∈ FacetIterator(dh, getfacetset(grid, setname))
        reinit!(fv, facet)

        coords = getcoordinates(facet)
        ddofs = @view celldofs(facet)[drange]
        uₑ = @view u[ddofs]

        for qp in QuadratureIterator(fv)
            dΓ = getdetJdV(fv, qp)
            N = getnormal(fv, qp)

            ∇u = function_gradient(fv, qp, uₑ)
            F = one(∇u) + ∇u

            d = function_value(fv, qp, uₑ)

            x = spatial_coordinate(fv, qp, coords)

            volume += volume_integral(x, d, F, N, method.volume_method) * dΓ
        end
    end
    return volume
end

"""
Compute the chamber volume as a surface integral via the integral
  -∫ det(F) ((h ⊗ h)(x + d - b)) adj(F) N ∂Ωendo

as proposed by [RegSalAfrFedDedQar:2022:cem](@citet).

!!! note 
    This integral basically measures the volume via displacement on a given axis.
"""
Base.@kwdef struct RSAFDQ2022SurrogateVolume{T}
    h::Vec{3,T} = Vec((0.0, 1.0, 0.0))
    b::Vec{3,T} = Vec((0.0, 0.0, -0.1))
end

function volume_integral(x::Vec, d::Vec, F::Tensor, N::Vec, method::RSAFDQ2022SurrogateVolume)
    @unpack h, b = method
    val = det(F) * ((h ⊗ h) ⋅ (x + d - b)) ⋅ (transpose(inv(F)) ⋅  N)
    # val < 0.0 && @error val, d, x, N
    -val #det(F) * ((h ⊗ h) ⋅ (x + d - b)) ⋅ (transpose(inv(F)) ⋅  N)
end

##########################################################################

"""
    RSAFDQ20223DFunction{MT, CT}

Generic description of the function associated with the RSAFDQModel.
"""
struct RSAFDQ20223DFunction{MT <: QuasiStaticFunction, TP <: RSAFDQ2022TyingInfo} <: AbstractSemidiscreteBlockedFunction
    structural_function::MT
    tying_info::TP
end
BlockArrays.blocksizes(f::RSAFDQ20223DFunction) = (solution_size(f.structural_function), solution_size(f.tying_info))

getch(f::AbstractSemidiscreteFunction) = f.ch
getch(f::AbstractSemidiscreteBlockedFunction) = error("Overlaod getch to get the constraint handler for a blocked function")
getch(f::RSAFDQ20223DFunction) = getch(f.structural_function)

# struct RSAFDQ2022VolumeFunction{MT <: QuasiStaticFunction, TP <: RSAFDQ2022TyingInfo} <: AbstractSemidiscreteBlockedFunction
#     structural_function::MT
#     tying_info::TP
# end

##########################################################################

"""
The split model described by [RegSalAfrFedDedQar:2022:cem](@citet) alone.
"""
struct RSAFDQ2022Model{SM #=<: QuasiStaticModel =#, CM <: AbstractLumpedCirculatoryModel, CT <: LumpedFluidSolidCoupler}
    structural_model::SM
    circuit_model::CM
    coupler::CT
end

"""
Annotation for the split described by [RegSalAfrFedDedQar:2022:cem](@citet).
"""
struct RSAFDQ2022Split{MODEL <: Union{CoupledModel, RSAFDQ2022Model}}
    model::MODEL
end

function assemble_tying_facet_rsadfq!(Jₑ, residualₑ, uₑ, p, cell, local_facet_index, fv, time)
    reinit!(fv, cell, local_facet_index)

    for qp in QuadratureIterator(fv)
        assemble_facet_pressure_qp!(Jₑ, residualₑ, uₑ, p, qp, fv)
    end
end

function assemble_tying_facet_rsadfq!(Jₑ, uₑ, p, cell, local_facet_index, fv, time)
    reinit!(fv, cell, local_facet_index)

    for qp in QuadratureIterator(fv)
        assemble_facet_pressure_qp!(Jₑ, uₑ, p, qp, fv)
    end
end

function assemble_tying!(Jₑ, residualₑ, uₑ, uₜ, cell, tying_cache::RSAFDQ2022TyingCache, time)
    for local_facet_index ∈ 1:nfacets(cell)
        for (chamber_index,chamber) in pairs(tying_cache.chambers)
            if (cellid(cell), local_facet_index) ∈ chamber.facets
                assemble_tying_facet_rsadfq!(Jₑ, residualₑ, uₑ, uₜ[chamber_index], cell, local_facet_index, tying_cache.fv, time)
            end
        end
    end
end

function assemble_tying!(Jₑ, uₑ, uₜ, cell, tying_cache::RSAFDQ2022TyingCache, time)
    for local_facet_index ∈ 1:nfacets(cell)
        for (chamber_index,chamber) in pairs(tying_cache.chambers)
            if (cellid(cell), local_facet_index) ∈ chamber.facets
                assemble_tying_facet_rsadfq!(Jₑ, uₑ, uₜ[chamber_index], cell, local_facet_index, tying_cache.fv, time)
            end
        end
    end
end

#################################################################################

function create_chamber_tyings(coupler::LumpedFluidSolidCoupler{CVM}, structural_problem, circuit_model) where CVM
    num_unknowns_structure = solution_size(structural_problem)
    chamber_tyings = RSAFDQ2022SingleChamberTying{CVM}[]
    for i in 1:length(coupler.chamber_couplings)
        # Get i-th ChamberVolumeCoupling
        coupling = coupler.chamber_couplings[i]
        # The pressure dof is just the last dof index for the structurel problem + the current chamber index
        pressure_dof_index = num_unknowns_structure + i
        (; dh) = structural_problem
        chamber_facetset = getfacetset(get_grid(dh), coupling.chamber_surface_setname)
        chamber_volume_idx_lumped = get_variable_symbol_index(circuit_model, coupling.lumped_model_symbol)
        # TODO rethink the next two lines
        tying = RSAFDQ2022SingleChamberTying(
            pressure_dof_index,
            pressure_dof_index,
            chamber_facetset,
            coupling.chamber_volume_method,
            coupler.displacement_symbol,
            NaN,
            chamber_volume_idx_lumped,
            num_unknowns_structure+num_unknown_pressures(circuit_model)+chamber_volume_idx_lumped,
        )
        tying.V⁰ᴰval = initial_volume_lumped = compute_chamber_volume(dh, zeros(ndofs(dh)), coupling.chamber_surface_setname, tying)
        push!(chamber_tyings, tying)
    end
    return chamber_tyings
end

function semidiscretize(split::RSAFDQ2022Split, discretization::FiniteElementDiscretization, mesh::AbstractGrid)
    @unpack model = split
    @unpack structural_model, circuit_model, coupler = model
    @assert length(coupler.chamber_couplings) ≥ 1 "Provide at least one coupling for the semi-discretization of an RSAFDQ2022 model"
    @assert coupler.displacement_symbol == structural_model.displacement_symbol "Coupler is not compatible with structural model"

    # Discretize individual problems
    structural_problem = semidiscretize(model.structural_model, discretization, mesh)
    num_chambers_lumped = num_unknown_pressures(model.circuit_model)

    # ODE problem for blood circuit
    circuit_fun = ODEFunction( #Not ModelingToolkit.ODEFunction :)
            model.circuit_model,
        (du,u,t,chamber_pressures) -> lumped_driver!(du, u, t, chamber_pressures, model.circuit_model),
        zeros(num_chambers_lumped) # Initialize with 0 pressure in the chambers - TODO replace this hack with a proper transfer operator!
    )

    # Tie problems
    # Fix dispatch....
    chamber_tyings = create_chamber_tyings(coupler, structural_problem, circuit_model)
    @debug "Chamber tyings:"
    for chamber_tying in chamber_tyings
        @debug "$(chamber_tying.pressure_dof_index_local), $(chamber_tying.pressure_dof_index_global), $(chamber_tying.volume_method), $(chamber_tying.displacement_symbol), $(chamber_tying.V⁰ᴰidx_local), $(chamber_tying.V⁰ᴰidx_global)"
    end
    @assert num_chambers_lumped == length(chamber_tyings) "Number of chambers in structural model ($(length(chamber_tyings))) and circuit model ($num_chambers_lumped) differs."

    tying_info = RSAFDQ2022TyingInfo(chamber_tyings)
    structural_fun = RSAFDQ20223DFunction(
        structural_problem,
        tying_info  # TODO replace with proper function
    )

    offset = solution_size(structural_fun)
    splitfun = GenericSplitFunction(
        (
            structural_fun,
            circuit_fun
        ),
        (
            1:offset,
            (offset+1):(offset+solution_size(circuit_fun))
        ),
        (
            VolumeTransfer0D3D(tying_info),
            PressureTransfer3D0D(tying_info),
        ),
    )

    return splitfun
end


function semidiscretize(split::RSAFDQ2022Split{<:CoupledModel}, discretization::FiniteElementDiscretization, grid::AbstractGrid)
    ets = elementtypes(grid)
    @assert length(ets) == 1 "Multiple element types not supported"
    @assert length(split.model.base_models) == 2 "I can only handle pure mechanics coupled to pure circuit."
    error("Implementation for RSAFDQ2022Split{<:CoupledModel} currently broken. 💔")
end

#################################################################################

function residual_norm(cache::AbstractNonlinearSolverCache, f::RSAFDQ2022TyingInfo)
    norm(cache.residual[Block(2)])
end

eliminate_constraints_from_increment!(Δu, f::RSAFDQ2022TyingInfo, solver_cache::AbstractNonlinearSolverCache) = nothing
function eliminate_constraints_from_linearization!(solver_cache::AbstractNonlinearSolverCache, f::RSAFDQ20223DFunction)
    @unpack structural_function = f
    @unpack op = solver_cache
    ch = getch(structural_function)
    # Eliminate residual
    residual_block = @view solver_cache.residual[Block(1)]
    # Elimiante diagonal
    # apply_zero!(getJ(op, Block(1,1)), residual_block, ch) # FIXME crashes
    apply!(getJ(op, Block(1,1)), ch)
    apply_zero!(residual_block, ch)
    # Eliminate rows
    getJ(op, Block((1,2)))[ch.prescribed_dofs, :] .= 0.0
    # Eliminate columns
    getJ(op, Block((2,1)))[:, ch.prescribed_dofs] .= 0.0
end

update_constraints_block!(::RSAFDQ2022TyingInfo, ::BlockArrays.Block, ::Thunderbolt.HomotopyPathSolverCache, ::Float64) = nothing
