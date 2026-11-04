##########################################################################

mutable struct RSAFDQ2022SingleChamberTying
    # The chamber pressure's global dof, from `algebraic_dofs`. It is also its index in the enclosing
    # split vector, because the 3D block leads that vector.
    const pressure_dof_index::Int
    const pressure_symbol::Symbol
    const pressure_parameter_index_local
    const facets::OrderedSet{FacetIndex}
    const displacement_symbol::Symbol
    # The buffer between the two sides of the split: `VolumeTransfer0D3D` writes the 0D solve's
    # chamber volume here, and the 3D stage copies it into its parameter bag once per step.
    V⁰ᴰval::Float64
    const V⁰ᴰidx_global::Int
end

struct RSAFDQ2022TyingInfo
    chambers::Vector{RSAFDQ2022SingleChamberTying}
end

solution_size(problem::RSAFDQ2022TyingInfo) = length(problem.chambers)

"""
    compute_chamber_volume(dh, u, setname, method) -> V³ᴰ

`∫_Γ V³ᴰ(u) dΓ` over a chamber surface, evaluated directly on the `DofHandler`.

This is the setup-time route: `create_chamber_tyings` needs the reference volume while the model is
being semidiscretized, before any operator over the structural problem exists.
[`Thunderbolt.chamber_volume`](@ref) is the route for everything that has one — same integrand
([`Thunderbolt._chamber_volume_facet`](@ref)), reduced over the operator's own tying facets.
"""
function compute_chamber_volume(dh, u, setname, method::RSAFDQ2022SingleChamberTying)
    volume = 0.0
    for (sdhi, facets) in _chamber_facets_per_subdofhandler(dh, setname)
        volume += _chamber_volume_contribution(dh.subdofhandlers[sdhi], u, facets, method)
    end
    return volume
end

# A chamber surface is bounded by whatever subdomains touch it -- the ventricular wall and the
# valvular plate, hexahedra and wedges -- so the facets are grouped by the subdofhandler owning
# their cell. Resolving the owner is what keeps the interpolation the one that cell actually
# carries; the surface's own partition is by cell type, which does not determine the subdomain.
function _chamber_facets_per_subdofhandler(dh, setname)
    grouped = OrderedDict{Int, Vector{FacetIndex}}()
    for facetset in values(get_grid(dh).surface_subdomains[setname].data), facet in facetset
        sdhi = dh.cell_to_subdofhandler[facet[1]]
        sdhi == 0 && continue
        push!(get!(Vector{FacetIndex}, grouped, sdhi), facet)
    end
    return grouped
end

function _chamber_volume_contribution(sdh, u, facets, method::RSAFDQ2022SingleChamberTying)
    ip     = Ferrite.getfieldinterpolation(sdh, method.displacement_symbol)
    drange = dof_range(sdh, method.displacement_symbol)
    qr     = FacetQuadratureRule{Ferrite.getrefshape(ip)}(2*Ferrite.getorder(ip))
    fv     = FacetValues(qr, ip, Ferrite.geometric_interpolation(getcelltype(sdh)))

    volume = 0.0
    for facet ∈ FacetIterator(sdh, facets)
        Ferrite.reinit!(fv, facet)
        ddofs = @view celldofs(facet)[drange]
        dₑ = @view u[ddofs]
        volume += _chamber_volume_facet(fv, getcoordinates(facet), dₑ)
    end
    return volume
end


##########################################################################

"""
    RSAFDQ20223DFunction{MT, CT}

Generic description of the function associated with the RSAFDQModel.
"""
struct RSAFDQ20223DFunction{MT <: QuasiStaticFunction} <: AbstractSemidiscreteFunction
    structural_function::MT
    tying_info::RSAFDQ2022TyingInfo
end

# The chamber pressures are algebraic variables of the structural `DofHandler`, so this function's
# unknowns *are* the structural ones. Where a block split is needed -- `setup_stage_operator`
# building the `BlockMatrix` Schur solves on -- it is derived there from the chamber count.
solution_size(f::RSAFDQ20223DFunction) = solution_size(f.structural_function)

getch(f::AbstractSemidiscreteFunction) = f.ch
getch(f::RSAFDQ20223DFunction) = getch(f.structural_function)

get_strategy(f::RSAFDQ20223DFunction) = get_strategy(f.structural_function)

# The checks are the structural model's; continuation rejects the same materials and boundary
# conditions whether or not a circuit is tied to them.
check_internal_variables_are_rate_free(f::RSAFDQ20223DFunction) =
    check_internal_variables_are_rate_free(f.structural_function)
check_weak_boundary_conditions_are_rate_free(f::RSAFDQ20223DFunction) =
    check_weak_boundary_conditions_are_rate_free(f.structural_function)

function solution_variables(f::RSAFDQ20223DFunction)
    vars = solution_variables(f.structural_function)
    for chamber in f.tying_info.chambers
        push!(vars, GlobalVariable(chamber.pressure_symbol, chamber.pressure_dof_index))
    end
    return merge_and_check_unique(vars)
end

# The chamber balance rows read `V⁰ᴰ` off `p`, not off `chamber.V⁰ᴰval` directly (see
# `ChamberBalanceCache`): copied into a plain vector here, once per step, rather than aliased, so an
# assembly worker never sees a write racing its own sweep.
_homotopy_stage_evaluation(f::RSAFDQ20223DFunction, t) = StageEvaluation(;
    p         = (V⁰ᴰ = [chamber.V⁰ᴰval for chamber in f.tying_info.chambers],),
    ctx       = TimeIntegrationContext(t, zero(t), zero(t)),
    condensed = has_internal_variables(f.structural_function),
)

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

function create_chamber_tyings(coupler::LumpedFluidSolidCoupler, structural_problem, circuit_model)
    num_unknowns_structure = solution_size(structural_problem)
    chamber_tyings = RSAFDQ2022SingleChamberTying[]
    for i = 1:length(coupler.chamber_couplings)
        # Get i-th ChamberVolumeCoupling
        coupling                    = coupler.chamber_couplings[i]
        (; dh)                      = structural_problem
        pressure_dof_index          = only(algebraic_dofs(dh, coupling.pressure_symbol_3D))
        chamber_facetset            = getfacetset(get_grid(dh), coupling.chamber_surface_setname)
        chamber_volume_idx_lumped   = get_variable_symbol_index(circuit_model, coupling.lumped_volume_symbol)
        chamber_pressure_idx_lumped = get_parameter_symbol_index(circuit_model, coupling.lumped_pressure_symbol)
        # TODO rethink the next two lines
        tying = RSAFDQ2022SingleChamberTying(
            pressure_dof_index,
            coupling.pressure_symbol_3D,
            chamber_pressure_idx_lumped,
            chamber_facetset,
            coupler.displacement_symbol,
            NaN,
            # The chamber volume, in the circuit block that follows the 3D unknowns.
            num_unknowns_structure + chamber_volume_idx_lumped,
        )
        tying.V⁰ᴰval =
            compute_chamber_volume(dh, zeros(ndofs(dh)), coupling.chamber_surface_setname, tying)
        push!(chamber_tyings, tying)
    end
    return chamber_tyings
end

"""
    _chamber_coupler_models(coupler)

One [`Pressure3D0DVolumeCoupler`](@ref) per chamber, in `coupler.chamber_couplings` order.
"""
_chamber_coupler_models(coupler::LumpedFluidSolidCoupler) = Tuple(
    Pressure3D0DVolumeCoupler(
        coupling.chamber_surface_setname,
        coupler.displacement_symbol,
        coupling.pressure_symbol_3D,
    ) for coupling in coupler.chamber_couplings
)

"""
    _with_chamber_couplers(structural_model, coupler)

The structural model with the chamber tying terms appended to its facet terms.

That is what makes the chamber pressures unknowns of the *model*: the terms reading them declare
them, and the discretization only has to put what the model declares into the `DofHandler`. They go
at the end, so the pressures are numbered in `coupler.chamber_couplings` order. Every subdomain model
carries the same terms -- the pressures are unknowns of the whole 3D system, not of one subdomain --
which is also what lets the domain split agree on one declaration.
"""
_with_chamber_couplers(model::QuasiStaticModel, coupler) = QuasiStaticModel(
    model.displacement_symbol,
    model.material_model,
    (_facet_model_tuple(model.facet_models)..., _chamber_coupler_models(coupler)...),
)

_with_chamber_couplers(models::Dict{String}, coupler) = Dict{String, QuasiStaticModel}(
    name => _with_chamber_couplers(model, coupler) for (name, model) in models
)

function semidiscretize(
    split::RSAFDQ2022Split,
    discretization::FiniteElementDiscretization,
    mesh::AbstractGrid,
)
    @unpack model = split
    @unpack structural_model, circuit_model, coupler = model
    @assert length(coupler.chamber_couplings) ≥ 1 "Provide at least one coupling for the semi-discretization of an RSAFDQ2022 model"
    @assert coupler.displacement_symbol == structural_displacement_symbol(structural_model) "Coupler is not compatible with structural model"

    # Discretize individual problems. The chamber pressures are unknowns of the 3D system, so they
    # enter its `DofHandler` as algebraic variables rather than being appended to the solution vector.
    structural_problem =
        semidiscretize(_with_chamber_couplers(structural_model, coupler), discretization, mesh)
    # The 3D half is stepped by `HomotopyPathSolver`. Refused here rather than at `init`, so that the
    # message names the material before anything else is set up.
    check_internal_variables_are_rate_free(structural_problem)
    num_chambers_lumped = num_unknown_pressures(model.circuit_model)

    # ODE problem for blood circuit
    circuit_fun = ODEFunction(model.circuit_model) #Not ModelingToolkit.ODEFunction :)

    # Tie problems
    # Fix dispatch....
    chamber_tyings = create_chamber_tyings(coupler, structural_problem, circuit_model)
    @debug "Chamber tyings:"
    for chamber_tying in chamber_tyings
        @debug "Chamber:" chamber_tying.pressure_dof_index chamber_tying.displacement_symbol chamber_tying.V⁰ᴰidx_global
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
