##########################################################################

mutable struct RSAFDQ2022SingleChamberTying{CVM}
    const pressure_dof_index_local::Int
    const pressure_dof_index_global::Int
    const pressure_symbol::Symbol
    const pressure_parameter_index_local
    const facets::OrderedSet{FacetIndex}
    const volume_method::CVM
    const displacement_symbol::Symbol
    V⁰ᴰval::Float64
    const V⁰ᴰidx_global::Int
end

struct RSAFDQ2022TyingInfo{CVM}
    chambers::Vector{RSAFDQ2022SingleChamberTying{CVM}}
end

solution_size(problem::RSAFDQ2022TyingInfo) = length(problem.chambers)

# TODO use an operator for this
function compute_chamber_volume(dh, u, setname, method::RSAFDQ2022SingleChamberTying)
    grid = dh.grid

    volume = 0.0
    # TODO function barrier
    for facetset in values(grid.surface_subdomains[setname].data)
        # TODO move out of loop and refactor
        cell = getcells(grid, first(facetset)[1])
        sdhi = typeof(cell) == Hexahedron ? 1 : min(2, length(dh.subdofhandlers)) # :) We can find it by searching the element index of the first element in the facetset in the sdh cellsets.
        sdh = dh.subdofhandlers[sdhi]
        ip = Ferrite.getfieldinterpolation(sdh, method.displacement_symbol)
        drange = dof_range(sdh, method.displacement_symbol)
        for facet ∈ FacetIterator(sdh, facetset)
            ip_geo = Ferrite.geometric_interpolation(typeof(cell))
            intorder = 2*Ferrite.getorder(ip)
            ref_shape = Ferrite.getrefshape(ip)
            qr_facet = FacetQuadratureRule{ref_shape}(intorder)
            fv = FacetValues(qr_facet, ip, ip_geo)

            Ferrite.reinit!(fv, facet)

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
    h::Vec{3, T} = Vec((0.0, 1.0, 0.0))
    b::Vec{3, T} = Vec((0.0, 0.0, -0.1))
end

function volume_integral(x::Vec, d::Vec, F::Tensor, N::Vec, method::RSAFDQ2022SurrogateVolume)
    @unpack h, b = method
    val = det(F) * ((h ⊗ h) ⋅ (x + d - b)) ⋅ (transpose(inv(F)) ⋅ N)
    # val < 0.0 && @error val, d, x, N
    -val #det(F) * ((h ⊗ h) ⋅ (x + d - b)) ⋅ (transpose(inv(F)) ⋅  N)
end

##########################################################################

"""
    RSAFDQ20223DFunction{MT, CT}

Generic description of the function associated with the RSAFDQModel.
"""
struct RSAFDQ20223DFunction{MT <: QuasiStaticFunction, TP <: RSAFDQ2022TyingInfo} <:
       AbstractSemidiscreteBlockedFunction
    structural_function::MT
    tying_info::TP
end
BlockArrays.blocksizes(f::RSAFDQ20223DFunction) =
    (solution_size(f.structural_function), solution_size(f.tying_info))

getch(f::AbstractSemidiscreteFunction) = f.ch
getch(f::AbstractSemidiscreteBlockedFunction) =
    error("Overlaod getch to get the constraint handler for a blocked function")
getch(f::RSAFDQ20223DFunction) = getch(f.structural_function)

BlockArrays.blocks(f::RSAFDQ20223DFunction) = (f.structural_function, f.tying_info)

# The chamber pressures are genuine unknowns appended after the structural block. `pressure_dof_index_local`
# and `pressure_dof_index_global` hold the same value today, but `_local` is the correct one here: the
# enclosing split translates, so a descriptor must report positions local to this function.
function solution_variables(f::RSAFDQ20223DFunction)
    vars = solution_variables(f.structural_function)
    for chamber in f.tying_info.chambers
        push!(vars, GlobalVariable(chamber.pressure_symbol, chamber.pressure_dof_index_local))
    end
    return merge_and_check_unique(vars)
end

##########################################################################

"""
The split model described by [RegSalAfrFedDedQar:2022:cem](@citet) alone.
"""
struct RSAFDQ2022Model{
    SM#=<: QuasiStaticModel =#,
    CM <: AbstractLumpedCirculatoryModel,
    CT <: LumpedFluidSolidCoupler,
}
    structural_model::SM
    circuit_model::CM
    coupler::CT
end

"""
Annotation for the split described by [RegSalAfrFedDedQar:2022:cem](@citet).
"""
struct RSAFDQ2022Split{MODEL <: RSAFDQ2022Model}
    model::MODEL
end

#################################################################################

"""
    _check_rsafdq_internal_variables(structural_problem)

The 3D-0D coupling does not support materials carrying internal variables yet. Two independent
reasons, both of which need the redesign to resolve properly:

 1. The 3D block's *solution vector* is laid out `[u_dofs | internal_vars | pressures]`
    (`blocksizes`, sized via `solution_size(::QuasiStaticFunction) = ndofs(dh) + ndofs(lvh)`),
    while its *system matrix* is laid out `[u_dofs | pressures]`
    (`block_sizes = [ndofs(dh), num_chambers]` in `setup_operator`). The tying assembly in
    `_update_tying_subdomain_Jr` builds a single `dofs` array and uses it to index both, so the two
    layouts must coincide - which they only do when there are no internal variables.
 2. `RSAFDQ2022Split` is solved through `HomotopyPathSolver`, which never routes through
    `BackwardEulerStageAnnotation`. Internal variables would therefore never be advanced in time
    even if the indexing were corrected, which is a worse failure mode than an error.

Fixing this properly requires distinguishing the solution-vector layout from the system layout,
which is what the planned solution-partition concept is for.
"""
function _check_rsafdq_internal_variables(structural_problem)
    niv = ndofs(structural_problem.lvh)
    niv == 0 || error(
        "The RSAFDQ2022 3D-0D coupling does not support materials with internal variables yet " *
        "(got $niv internal variable dofs). See `_check_rsafdq_internal_variables` for why. " *
        "Use a material without internal variables, e.g. a plain `PK1Model`.",
    )
    return nothing
end

function create_chamber_tyings(
    coupler::LumpedFluidSolidCoupler{CVM},
    structural_problem,
    circuit_model,
) where {CVM}
    num_unknowns_structure = solution_size(structural_problem)
    chamber_tyings = RSAFDQ2022SingleChamberTying{CVM}[]
    for i = 1:length(coupler.chamber_couplings)
        # Get i-th ChamberVolumeCoupling
        coupling = coupler.chamber_couplings[i]
        # The pressure dof is just the last dof index for the structurel problem + the current chamber index
        pressure_dof_index          = num_unknowns_structure + i
        (; dh)                      = structural_problem
        chamber_facetset            = getfacetset(get_grid(dh), coupling.chamber_surface_setname)
        chamber_volume_idx_lumped   = get_variable_symbol_index(circuit_model, coupling.lumped_volume_symbol)
        chamber_pressure_idx_lumped = get_parameter_symbol_index(circuit_model, coupling.lumped_pressure_symbol)
        # TODO rethink the next two lines
        tying = RSAFDQ2022SingleChamberTying(
            pressure_dof_index,
            pressure_dof_index,
            coupling.pressure_symbol_3D,
            chamber_pressure_idx_lumped,
            chamber_facetset,
            coupling.chamber_volume_method,
            coupler.displacement_symbol,
            NaN,
            num_unknowns_structure+num_unknown_pressures(circuit_model)+chamber_volume_idx_lumped,
        )
        tying.V⁰ᴰval =
            compute_chamber_volume(dh, zeros(ndofs(dh)), coupling.chamber_surface_setname, tying)
        push!(chamber_tyings, tying)
    end
    return chamber_tyings
end

function semidiscretize(
    split::RSAFDQ2022Split,
    discretization::FiniteElementDiscretization,
    mesh::AbstractGrid,
)
    @unpack model = split
    @unpack structural_model, circuit_model, coupler = model
    @assert length(coupler.chamber_couplings) ≥ 1 "Provide at least one coupling for the semi-discretization of an RSAFDQ2022 model"
    @assert coupler.displacement_symbol == structural_displacement_symbol(structural_model) "Coupler is not compatible with structural model"

    # Discretize individual problems
    structural_problem = semidiscretize(model.structural_model, discretization, mesh)
    _check_rsafdq_internal_variables(structural_problem)
    num_chambers_lumped = num_unknown_pressures(model.circuit_model)

    # ODE problem for blood circuit
    circuit_fun = ODEFunction(model.circuit_model) #Not ModelingToolkit.ODEFunction :)

    # Tie problems
    # Fix dispatch....
    chamber_tyings = create_chamber_tyings(coupler, structural_problem, circuit_model)
    @debug "Chamber tyings:"
    for chamber_tying in chamber_tyings
        @debug "Chamber:" chamber_tying.pressure_dof_index_local chamber_tying.pressure_dof_index_global chamber_tying.volume_method chamber_tying.displacement_symbol chamber_tying.V⁰ᴰidx_global
    end
    @assert num_chambers_lumped == length(chamber_tyings) "Number of chambers in structural model ($(length(chamber_tyings))) and circuit model ($num_chambers_lumped) differs."

    tying_info = RSAFDQ2022TyingInfo(chamber_tyings)
    structural_fun = RSAFDQ20223DFunction(
        structural_problem,
        tying_info,  # TODO replace with proper function
    )

    offset = solution_size(structural_fun)
    splitfun = GenericSplitFunction(
        (structural_fun, circuit_fun),
        (1:offset, (offset+1):(offset+solution_size(model.circuit_model))),
        (VolumeTransfer0D3D(tying_info), PressureTransfer3D0D(tying_info)),
    )

    return splitfun
end

#################################################################################

function residual_norm(cache::AbstractNonlinearSolverCache, f::RSAFDQ2022TyingInfo)
    norm(cache.residual[Block(2)])
end

eliminate_constraints_from_increment!(
    Δu,
    f::RSAFDQ2022TyingInfo,
    solver_cache::AbstractNonlinearSolverCache,
) = nothing
function eliminate_constraints_from_linearization!(
    solver_cache::AbstractNonlinearSolverCache,
    op,
    f::RSAFDQ20223DFunction,
)
    @unpack structural_function = f
    ch = getch(structural_function)
    # Eliminate residual
    residual_block = @view solver_cache.residual[Block(1)]
    # Elimiante diagonal
    # apply_zero!(getJ(op, Block(1,1)), residual_block, ch) # FIXME crashes
    apply!(getJ(op, Block(1, 1)), ch)
    apply_zero!(residual_block, ch)
    # Eliminate rows
    getJ(op, Block((1, 2)))[ch.prescribed_dofs, :] .= 0.0
    # Eliminate columns
    getJ(op, Block((2, 1)))[:, ch.prescribed_dofs] .= 0.0
end

update_constraints_block!(
    ::RSAFDQ2022TyingInfo,
    ::BlockArrays.Block,
    ::HomotopyPathSolverCache,
    ::Float64,
) = nothing
