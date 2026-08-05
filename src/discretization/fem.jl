# The general FEM semidiscretize algorithm should be like this
#
# for each subdomain+model pair
#     for each weak form
#         for each field in the weak form
#             register field in dof handler
#         query quadrature rule from discretization or compute from weak form order
#         for each internal variable
#             register internal variables at the quadrature points
#         register integrator
# return function type matching the integrator list

"""
Descriptor for a finite element discretization of a part of a PDE over some subdomain.

!!! note
    The current implementation is restricted to Bubnov-Galerkin methods. Petrov-Galerkin support will come in the future.
"""
struct FiniteElementDiscretization
    """
    """
    interpolations::Dict{Symbol}
    """
    """
    dbcs::Vector{Dirichlet} # TODO descriptor instead of Dirichlet. This allows us to distinguish different cases.
    """
    Each model comes with a set of symbols identifying the weak forms.
    These fields map user-provided quadrature rules.
    """
    qrcs::Dict{Symbol}
    fqrcs::Dict{Symbol}
    """
    This field might be removed in future updates.
    """
    assembly_strategy::AbstractAssemblyStrategy
    """
    """
    function FiniteElementDiscretization(
        ips::Dict{Symbol};
        dbcs::Vector{Dirichlet} = Dirichlet[],
        qrcs::Dict{Symbol} = Dict{Symbol, Any}(),
        fqrcs::Dict{Symbol} = Dict{Symbol, Any}(),
        assembly_strategy = SequentialAssemblyStrategy(SequentialCPUDevice()),
    )
        new(ips, dbcs, qrcs, fqrcs, assembly_strategy)
    end
end

_extract_ipc(ipc::InterpolationCollection) = ipc
_extract_ipc(p::Pair{<:InterpolationCollection, <:QuadratureRuleCollection}) = first(p)

function _extract_qrc(ipc::InterpolationCollection)
    ansatzorder = getorder(ipc)
    return QuadratureRuleCollection(max(2ansatzorder-1, 2))
end
_extract_qrc(p::Pair{<:InterpolationCollection, <:QuadratureRuleCollection}) = last(p)

# Internal utility with proper error message
function _get_interpolation_from_discretization(disc::FiniteElementDiscretization, sym::Symbol)
    if !haskey(disc.interpolations, sym)
        error(
            "Finite element discretization does not have an interpolation for $sym. Available symbols: $(collect(keys(disc.interpolations))).",
        )
    end
    return _extract_ipc(disc.interpolations[sym])
end
function _get_quadrature_from_discretization(disc::FiniteElementDiscretization, sym::Symbol)
    # Step 1: Try to query from qrcs discretization table
    if haskey(disc.qrcs, sym)
        return disc.qrcs[sym]
    end
    # Step 2: Deduce from interpolation order
    if haskey(disc.interpolations, sym)
        return _extract_qrc(disc.interpolations[sym])
    end
    error(
        "Finite element discretization does not have an interpolation or quadrature rule for $sym. Available symbols: $(collect(keys(disc.interpolations))) and $(collect(keys(disc.qrcs))).",
    )
end
function _get_facet_quadrature_from_discretization(disc::FiniteElementDiscretization, sym::Symbol)
    # Step 1: Try to query from qrcs discretization table
    if haskey(disc.fqrcs, sym)
        return disc.fqrcs[sym]
    end
    # Step 2: Deduce from interpolation order
    if haskey(disc.interpolations, sym)
        intorder = getorder(_extract_ipc(disc.interpolations[sym]))
        return FacetQuadratureRuleCollection(intorder)
    end
    error(
        "Finite element discretization does not have an interpolation or facet quadrature rule for $sym. Available symbols: $(collect(keys(disc.interpolations))) and $(collect(keys(disc.fqrcs))).",
    )
end

"""
    _approximation_descriptors(disc::FiniteElementDiscretization, model)

Gather the [`ApproximationDescriptor`](@ref)s for all field variables of `model`.

All fields of one model on one subdomain must be registered in a *single* `add_subdomain!` call,
because every call constructs a fresh `SubDofHandler`. Registering them one by one would yield one
single-field `SubDofHandler` per field over the same cellset instead of one handler carrying all
fields.
"""
function _approximation_descriptors(disc::FiniteElementDiscretization, model)
    return [
        ApproximationDescriptor(sym, _get_interpolation_from_discretization(disc, sym)) for
        sym in get_field_variable_names(model)
    ]
end

"""
    _subdomain_cellids(mesh, name)

All cell ids of the volumetric subdomain `name`, flattened over the per-celltype split.
"""
function _subdomain_cellids(mesh, name)
    haskey(mesh.volumetric_subdomains, name) || error(
        "Volumetric subdomain $name not found on mesh. Available subdomains: $(collect(keys(mesh.volumetric_subdomains))).",
    )
    return Set(idx.idx for (_, cellset) in mesh.volumetric_subdomains[name].data for idx in cellset)
end

"""
    _add_dirichlet_conditions!(ch, dbcs)

Add all Dirichlet conditions to `ch`, re-raising Ferrite's binding failure with the information
needed to diagnose it: which fields and how much of the mesh the dof handler actually covers. The
usual cause is a condition whose set reaches into cells that carry no model, and hence no dofs.
"""
function _add_dirichlet_conditions!(ch::ConstraintHandler, dbcs)
    dh = ch.dh
    for (i, dbc) ∈ enumerate(dbcs)
        try
            Ferrite.add!(ch, dbc)
        catch e
            e isa ErrorException || rethrow()
            ncovered = sum(length(sdh.cellset) for sdh in dh.subdofhandlers; init = 0)
            error(
                "Failed to add Dirichlet condition $i on field :$(dbc.field_name) - $(e.msg)\n" *
                "The dof handler carries fields $(dh.field_names) on $ncovered of " *
                "$(getncells(get_grid(dh))) cells. If the condition reaches cells outside the " *
                "subdomains carrying a model, those cells have no dofs to constrain.",
            )
        end
    end
    return ch
end

"""
    _check_model_subdomains_disjoint(mesh, names)

Ferrite associates each cell with at most one `SubDofHandler`, so the subdomains carrying models
must not overlap. Overlapping *cellsets* on the mesh are fine - only the subset used by the models
is constrained. Errors with the offending pair and the size of the overlap.
"""
function _check_model_subdomains_disjoint(mesh, names)
    cellids = Dict(name => _subdomain_cellids(mesh, name) for name in names)
    for (i, name1) in enumerate(names), name2 in Iterators.drop(names, i)
        overlap = intersect(cellids[name1], cellids[name2])
        isempty(overlap) || error(
            "Subdomains \"$name1\" and \"$name2\" both carry a model but share $(length(overlap)) cells " *
            "(e.g. cell $(first(sort(collect(overlap))))). Ferrite associates every cell with at most one " *
            "SubDofHandler, so subdomains carrying models must be pairwise disjoint.",
        )
    end
    return nothing
end

function semidiscretize(
    model::TransientDiffusionModel,
    discretization::FiniteElementDiscretization,
    mesh::AbstractGrid,
)
    @assert length(discretization.dbcs) == 0 "Dirichlet conditions not supported yet for TransientDiffusionProblem"

    sym = model.solution_variable_symbol
    ipc = _get_interpolation_from_discretization(discretization, sym)
    qrc = _get_quadrature_from_discretization(discretization, sym)
    dh = DofHandler(mesh)
    add_subdomain!(dh, single_subdomain_or_error(mesh), [ApproximationDescriptor(sym, ipc)])
    close!(dh)

    T = get_coordinate_eltype(get_grid(dh))
    return AffineODEFunction(
        BilinearMassIntegrator(
            ConstantCoefficient(T(1.0)),
            haskey(discretization.qrcs, :mass) ? discretization.qrcs[:mass] : qrc, # Allow e.g. mass lumping for explicit integrators.
            sym,
        ),
        BilinearDiffusionIntegrator(model.κ, qrc, sym),
        LinearIntegrator(model.source, qrc),
        dh,
        discretization.assembly_strategy,
    )
end

function register_affine_ode_integrators!(
    mass_integrators,
    rhs_integrators,
    linear_integrators,
    dh,
    name,
    discretization::FiniteElementDiscretization,
    model::TransientDiffusionModel,
)
    sym = model.solution_variable_symbol
    ipc = _get_interpolation_from_discretization(discretization, sym)
    add_subdomain!(dh, name, [ApproximationDescriptor(sym, ipc)])

    T = get_coordinate_eltype(get_grid(dh))

    qrc = _get_quadrature_from_discretization(discretization, sym)
    # TODO allow e.g. mass lumping for explicit integrators.
    mass_integrators[name] = BilinearMassIntegrator(
        ConstantCoefficient(T(1.0)),
        haskey(discretization.qrcs, :mass) ? discretization.qrcs[:mass] : qrc,
        sym,
    )
    rhs_integrators[name] = BilinearDiffusionIntegrator(model.κ, qrc, sym)
    linear_integrators[name] = LinearIntegrator(model.source, qrc)
end

function register_affine_ode_integrators!(
    mass_integrators,
    rhs_integrators,
    linear_integrators,
    dh,
    name,
    discretization::FiniteElementDiscretization,
    model::InterfaceDiffusionModel,
)
    sym = model.solution_variable_symbol
    ipc =
        _get_interpolation_from_discretization(discretization, model.interface_interpolation_symbol)
    add_subdomain!(dh, name, [ApproximationDescriptor(sym, ipc)])

    T = get_coordinate_eltype(get_grid(dh))

    qrc = _get_quadrature_from_discretization(discretization, sym)
    rhs_integrators[name] =
        BilinearInterfaceDiffusionIntegrator(model.G, qrc, sym, model.solution_variable_symbol)
end

function semidiscretize(
    models::Dict{String, <:Union{<:TransientDiffusionModel, <:InterfaceDiffusionModel}},
    discretization::FiniteElementDiscretization,
    mesh::AbstractGrid,
)
    @assert length(discretization.dbcs) == 0 "Dirichlet conditions not supported yet for transient diffusion models"

    _check_model_subdomains_disjoint(mesh, collect(keys(models)))

    dh = DofHandler(mesh)

    # 3 weak forms
    rhs_integrators    = Dict{String, AbstractBilinearIntegrator}()
    mass_integrators   = Dict{String, AbstractBilinearIntegrator}()
    linear_integrators = Dict{String, AbstractLinearIntegrator}()

    for (name, model) in models
        if typeof(model) <: TransientDiffusionModel
            register_affine_ode_integrators!(
                mass_integrators,
                rhs_integrators,
                linear_integrators,
                dh,
                name,
                discretization,
                model,
            )
        end
    end
    # We register the interfaces in a separate step, so they do not interfere with the dof distribution
    for (name, model) in models
        if typeof(model) <: InterfaceDiffusionModel
            register_affine_ode_integrators!(
                mass_integrators,
                rhs_integrators,
                linear_integrators,
                dh,
                name,
                discretization,
                model,
            )
        end
    end

    close!(dh)

    return AffineODEFunction(
        BilinearMultiIntegrator(mass_integrators),
        BilinearMultiIntegrator(rhs_integrators),
        LinearMultiIntegrator(linear_integrators),
        dh,
        discretization.assembly_strategy,
    )
end

function semidiscretize(
    model::SteadyDiffusionModel,
    discretization::FiniteElementDiscretization,
    mesh::AbstractGrid,
)
    sym = model.solution_variable_symbol
    ipc = _get_interpolation_from_discretization(discretization, sym)
    qrc = _get_quadrature_from_discretization(discretization, sym)
    dh = DofHandler(mesh)
    add_subdomain!(dh, single_subdomain_or_error(mesh), [ApproximationDescriptor(sym, ipc)])
    close!(dh)

    ch = ConstraintHandler(dh)
    _add_dirichlet_conditions!(ch, discretization.dbcs)
    close!(ch)

    return AffineSteadyStateFunction(
        BilinearDiffusionIntegrator(model.κ, qrc, sym),
        LinearIntegrator(model.source, qrc),
        dh,
        ch,
        discretization.assembly_strategy,
    )
end

function semidiscretize(
    models::Dict{String, <:SteadyDiffusionModel},
    discretization::FiniteElementDiscretization,
    mesh::AbstractGrid,
)
    _check_model_subdomains_disjoint(mesh, collect(keys(models)))

    dh = DofHandler(mesh)

    rhs_integrators    = Dict{String, AbstractBilinearIntegrator}()
    linear_integrators = Dict{String, AbstractLinearIntegrator}()

    for (name, model) in models
        sym = model.solution_variable_symbol
        ipc = _get_interpolation_from_discretization(discretization, sym)
        add_subdomain!(dh, name, [ApproximationDescriptor(sym, ipc)])

        qrc                      = _get_quadrature_from_discretization(discretization, sym)
        rhs_integrators[name]    = BilinearDiffusionIntegrator(model.κ, qrc, sym)
        linear_integrators[name] = LinearIntegrator(model.source, qrc)
    end

    close!(dh)

    ch = ConstraintHandler(dh)
    _add_dirichlet_conditions!(ch, discretization.dbcs)
    close!(ch)

    return AffineSteadyStateFunction(
        BilinearMultiIntegrator(rhs_integrators),
        LinearMultiIntegrator(linear_integrators),
        dh,
        ch,
        discretization.assembly_strategy,
    )
end

function semidiscretize(
    split::ReactionDiffusionSplit{<:MonodomainModel},
    discretization::FiniteElementDiscretization,
    mesh::AbstractGrid,
)
    epmodel = split.model
    φsym = epmodel.transmembrane_solution_symbol

    heat_model = TransientDiffusionModel(
        ConductivityToDiffusivityCoefficient(epmodel.κ, epmodel.Cₘ, epmodel.χ),
        epmodel.stim,
        φsym,
    )

    heatfun = semidiscretize(heat_model, discretization, mesh)

    dh = heatfun.dh
    ndofsφ = ndofs(dh)
    # TODO we need some information about the discretization of this one, e.g. dofs a nodes vs dofs at quadrature points
    # TODO we should call semidiscretize here too - This is a placeholder for the nodal discretization
    nstates_per_point = num_states(epmodel.ion)
    odefun = PointwiseODEFunction(
        # TODO epmodel.Cₘ(x)
        epmodel.ion,
        split.cs === nothing ? nothing : compute_nodal_values(split.cs, dh, φsym),
        1:(nstates_per_point*ndofsφ),
    )
    # TODO this assumes that the transmembrane potential is the first field. Relax this.
    heat_dofrange = 1:ndofsφ
    ode_dofrange = 1:(nstates_per_point*ndofsφ)
    #
    semidiscrete_ode = GenericSplitFunction(
        (heatfun, odefun),
        (heat_dofrange, ode_dofrange),
        # No transfer operators needed, because the the solutions variables overlap with the subproblems perfectly
    )

    return semidiscrete_ode
end

function semidiscretize_map_diffusion_part(epmodel::MonodomainModel)
    return TransientDiffusionModel(
        ConductivityToDiffusivityCoefficient(epmodel.κ, epmodel.Cₘ, epmodel.χ),
        epmodel.stim,
        epmodel.transmembrane_solution_symbol,
    )
end

function semidiscretize_map_diffusion_part(model::InterfaceDiffusionModel)
    return model
end

function semidiscretize(
    split::ReactionDiffusionSplit{Dict{String, Any}},
    discretization::FiniteElementDiscretization,
    mesh::AbstractGrid,
)
    models = narrow_dict_types(split.model)
    semidiscretize(ReactionDiffusionSplit(models, split.cs), discretization, mesh)
end

function semidiscretize(
    split::ReactionDiffusionSplit{
        <:Dict{String, <: Union{<:AbstractEPModel, <:InterfaceDiffusionModel}},
    },
    discretization::FiniteElementDiscretization,
    mesh::AbstractGrid,
)
    epmodels = split.model

    heat_models =
        Dict(name => semidiscretize_map_diffusion_part(epmodel) for (name, epmodel) in epmodels)

    heatfun = semidiscretize(heat_models, discretization, mesh)

    dh = heatfun.dh
    ndofsφ = ndofs(dh)

    # TODO we need some information about the discretization of this one, e.g. dofs a nodes vs dofs at quadrature points
    # TODO we should call semidiscretize here too - This is a placeholder for the nodal discretization
    inner_functions = PointwiseODEFunction[]
    offset = 0
    # Only bulk models own a block of the solution vector; coupling models attach to fields that
    # bulk models introduced and are handled entirely by the heat function.
    bulk_models = filter(nm -> !is_coupling_model(last(nm)), epmodels)
    for (name, model) in epmodels
        if is_coupling_model(model) && has_pointwise_reaction_part(model)
            error(
                "Coupling model on \"$name\" ($(typeof(model))) carries a pointwise reaction part, " *
                "which is not supported yet: a coupling model does not own a block of the solution " *
                "vector, so its reaction state has nowhere to live.",
            )
        end
    end

    epmodel_symbols = Set(
        reaction_solution_symbol(model) for
        model in values(bulk_models) if has_pointwise_reaction_part(model)
    )
    @assert length(epmodel_symbols) == 1 "All EP models in a domain split must share the same transmembrane potential symbol, got $(epmodel_symbols)."
    φsym = first(epmodel_symbols)
    xφ = (split.cs === nothing ? nothing : compute_nodal_values(split.cs, dh, φsym))
    heat_dofrange = Int[]
    for (name, model) in bulk_models
        if has_pointwise_reaction_part(model) # Bulk models without a reaction part need no ODE block
            # Extract dofs associated with subdomain
            subdofs = collect_dofs_on_subdomain(dh, mesh, name)
            # Compute range for states
            mindof, maxdof = extrema(subdofs)
            @assert length(mindof:maxdof) == length(subdofs) "$(mindof:maxdof) does not match length(subdofs)=$(length(subdofs)) => Subdomain is not isolated. "
            nstates = num_states(reaction_model(model))
            # The state vector is packed subdomain by subdomain in iteration order, *not* by
            # global dof index, because the number of states per point varies between subdomains.
            # Hence the range is derived from the running offset, exactly as the heat dof map below.
            state_range = (offset+1):(offset+nstates*length(subdofs))
            @info "Mapping state range on $name to $state_range from $(mindof:maxdof)"

            # Create ode function for subdomain
            push!(
                inner_functions,
                PointwiseODEFunction(
                    reaction_model(model),
                    xφ === nothing ? nothing : view(xφ, mindof:maxdof),
                    state_range,
                ),
            )

            # Map heat dofs
            heat_dofs_submodel =
                ((subdofs .- minimum(subdofs)) .* nstates) .+
                transmembranepotential_index(reaction_model(model)) .+ offset
            append!(heat_dofrange, heat_dofs_submodel)

            # Update total number of dofs in split
            offset += length(subdofs)*num_states(reaction_model(model))
        end
    end
    ionicfun = PointwiseMultiODEFunction(inner_functions, xφ)
    ionic_dofrange = 1:offset

    # NOTE: these two index sets deliberately *overlap* - `heat_dofrange` picks the transmembrane
    # potential entries out of the ionic state vector. Operator splitting splits the operator, not
    # the unknowns, so this is not a partition of the solution vector.
    semidiscrete_ode = GenericSplitFunction(
        (heatfun, ionicfun),
        (heat_dofrange, ionic_dofrange),
        # No transfer operators needed, because the the solutions variables overlap with the subproblems perfectly
    )

    return semidiscrete_ode
end

"""
    _setup_internal_variable_handler(integrator, dh)

Build the [`InternalVariableHandler`](@ref) for `integrator` using `FerriteOperators`, rather than
laying the condensed unknowns out by hand.

`FerriteOperators` normally does this inside `setup_subdomain_caches`, i.e. at operator setup - but
`solution_size` has to be known already at `semidiscretize` time in order to size the initial
solution vector, so it is built eagerly here. The element caches are constructed once more later
during operator setup; that is setup-only cost.
"""
function _setup_internal_variable_handler(integrator, dh)
    element_caches = [setup_element_cache(integrator, sdh) for sdh in dh.subdofhandlers]
    return FerriteOperators.setup_internal_variable_handler(integrator, element_caches, dh)
end

"""
    _rebase_internal_variable_handler(lvh, from_dh, to_dh)

The same condensed layout, re-expressed as offsets into a solution vector whose finite element block
is `to_dh`'s rather than `from_dh`'s.

How many unknowns a cell condenses is a property of its material, so it does not change when a second
field is added next to the displacement -- only where they start does. Rebuilding the handler from the
element caches instead would require setting them up against a multi-field `SubDofHandler`, which the
element layer does not support and does not need to.
"""
function _rebase_internal_variable_handler(
    lvh::InternalVariableHandler,
    from_dh::Ferrite.AbstractDofHandler,
    to_dh::Ferrite.AbstractDofHandler,
)
    offsets = lvh.internal_variable_offsets
    offsets === nothing && return lvh # nothing is condensed, so there is nothing to rebase
    return InternalVariableHandler(offsets .+ (ndofs(to_dh) - ndofs(from_dh)), ndofs(lvh))
end

# Solid mechanics semidiscretize interface
function semidiscretize(
    model::QuasiStaticModel,
    discretization::FiniteElementDiscretization,
    mesh::AbstractGrid,
)
    sym = model.displacement_symbol
    qrc = _get_quadrature_from_discretization(discretization, sym)
    fqrc = _get_facet_quadrature_from_discretization(discretization, sym)
    dh = DofHandler(mesh)
    name = single_subdomain_or_error(get_grid(dh))
    add_subdomain!(dh, name, _approximation_descriptors(discretization, model))
    close!(dh)

    integrator =
        NonlinearIntegrator(model, model.facet_models, get_field_variable_names(model), qrc, fqrc)
    lvh = _setup_internal_variable_handler(integrator, dh)

    ch = ConstraintHandler(dh)
    _add_dirichlet_conditions!(ch, discretization.dbcs)
    close!(ch)

    semidiscrete_problem =
        QuasiStaticFunction(dh, ch, lvh, integrator, discretization.assembly_strategy)

    return semidiscrete_problem
end

function semidiscretize(
    model::ElastodynamicsModel,
    discretization::FiniteElementDiscretization,
    mesh::AbstractGrid,
)
    sym = model.displacement_symbol
    _reject_velocity_constraints(discretization, model.velocity_symbol)

    # The internal force contribution of an elastodynamics model *is* the quasi-static one, so the
    # model is lowered rather than the path reimplemented: element caches, material routines and the
    # condensation of quadrature point local internal variables are reused unchanged. What is
    # genuinely new is the mass term next to it.
    quasistaticform = semidiscretize(
        QuasiStaticModel(sym, model.material_model, model.facet_models),
        discretization,
        mesh,
    )

    # The mass integrand `ρ Nᵢ⋅Nⱼ` is of degree `2p` per direction, where the stiffness terms carrying
    # gradients are `2(p-1)`, so the mass matrix is the one that decides whether a rule is rich
    # enough. Sharing the displacement rule is safe rather than merely convenient: `_extract_qrc`
    # gives `max(2p-1, 2)` points, exact through degree `3` for `p = 1` and `4p-3` for `p ≥ 2`, both
    # of which cover `2p`. A rule passed explicitly via `qrcs` is the user's to get right.
    mass_term = BilinearMassIntegrator(
        model.ρ,
        _get_quadrature_from_discretization(discretization, sym),
        sym,
    )

    return _elastodynamics_function(
        quasistaticform,
        mass_term,
        discretization,
        mesh,
        sym,
        model.velocity_symbol,
        [single_subdomain_or_error(mesh)],
    )
end

"""
    _reject_velocity_constraints(discretization, vsym)

Refuse a Dirichlet condition on the velocity field.

The velocity is a field of the state, but a scheme that reconstructs it -- [`NewmarkSolver`](@ref)
solves for the displacement and writes the velocity from it -- has no way to honour a prescribed
value: whatever the constraint wrote would be overwritten by the reconstruction. Left unchecked, the
condition reaches the structural sub-problem, which carries no velocity field, and reports that the
handler knows only the displacement -- an internal decomposition surfacing as a user-facing error.
"""
function _reject_velocity_constraints(discretization::FiniteElementDiscretization, vsym::Symbol)
    for dbc in discretization.dbcs
        dbc.field_name === vsym && error(
            "Dirichlet conditions on the velocity field $(vsym) are not supported. A time scheme " *
            "that reconstructs the velocity from the displacement would overwrite them. Constrain " *
            "the displacement instead, or set the initial velocity through the problem's `u0`.",
        )
    end
    return nothing
end

"""
    _elastodynamics_function(quasistaticform, mass_term, discretization, mesh, sym, vsym, names)

Wrap a structural problem and a mass term into an [`ElastodynamicsFunction`](@ref).

The velocity is a genuine field of the *state*, so that it is interpolated with the displacement and
can be written out by name. It is deliberately not a field of the structural problem: the internal
forces have no velocity equation, and a scheme that reconstructs the velocity would otherwise
assemble empty rows for it. The two numberings are wired rather than assumed to coincide.

Prescribing the velocity is refused rather than supported -- see
`_reject_velocity_constraints`.
"""
function _elastodynamics_function(
    quasistaticform,
    mass_term,
    discretization,
    mesh,
    sym,
    vsym,
    names,
)
    ipc = _get_interpolation_from_discretization(discretization, sym)
    _assert_shared_interpolation(discretization, sym, vsym, ipc)

    dh = DofHandler(mesh)
    for name in names
        add_subdomain!(
            dh,
            name,
            [ApproximationDescriptor(sym, ipc), ApproximationDescriptor(vsym, ipc)],
        )
    end
    close!(dh)

    ch = ConstraintHandler(dh)
    _add_dirichlet_conditions!(ch, discretization.dbcs)
    close!(ch)

    lvh = _rebase_internal_variable_handler(quasistaticform.lvh, quasistaticform.dh, dh)

    state_mapping = SolutionVectorMapping(
        field_dof_mapping(quasistaticform.dh, sym, dh, sym),
        internal_variable_mapping(quasistaticform.dh, quasistaticform.lvh, dh, lvh),
    )
    velocity_dofs = field_dof_mapping(quasistaticform.dh, sym, dh, vsym)

    _assert_constraints_transfer(quasistaticform.ch, ch, state_mapping)

    return ElastodynamicsFunction(
        dh,
        ch,
        lvh,
        quasistaticform,
        mass_term,
        state_mapping,
        velocity_dofs,
        quasistaticform.assembly_strategy,
    )
end

"""
    _assert_constraints_transfer(structural_ch, state_ch, mapping)

The stage solves against the structural constraint handler while the state carries its own, both
built from the same `Dirichlet` conditions. This checks that they agree: every displacement dof
prescribed on one side must be prescribed on the other.

Without it, a condition that bound on one handler and silently failed to bind on the other -- a
misspelled field, a set that reaches cells outside a subdomain -- would leave the two solving
different problems.
"""
function _assert_constraints_transfer(structural_ch, state_ch, mapping)
    structural = Set(structural_ch.prescribed_dofs)
    state = Set(state_ch.prescribed_dofs)
    image = Set(mapping.dofs[d] for d in structural)
    image ⊆ state || error(
        "$(length(setdiff(image, state))) displacement dofs are constrained on the structural " *
        "problem but not on the state.",
    )
    extra = setdiff(intersect(state, Set(mapping.dofs)), image)
    isempty(extra) || error(
        "$(length(extra)) displacement dofs are constrained on the state but not on the structural " *
        "problem, which is what the stage solves.",
    )
    return nothing
end

"""
    _assert_shared_interpolation(discretization, sym, vsym, ipc)

The velocity shares the displacement's interpolation.

This is assumed throughout: the element evaluates the reconstructed velocity with the *displacement*
`CellValues` (`compute_kinematic_quantities`), and the schemes size velocity vectors by the
displacement dof count. A mismatch is silent nonsense rather than an error, which is what the
assertion is for.
"""
function _assert_shared_interpolation(discretization, sym, vsym, ipc)
    haskey(discretization.interpolations, vsym) || return nothing
    discretization.interpolations[vsym] === ipc || error(
        "The velocity field $(vsym) must share the interpolation of the displacement field $(sym), " *
        "got $(discretization.interpolations[vsym]) and $(ipc).",
    )
    return nothing
end

"""
    semidiscretize(models::Dict{String, <:ElastodynamicsModel}, discretization, mesh)

Elastodynamics over several subdomains, each with its own material and density.

The structural sub-problem is the multi-subdomain quasi-static one, so materials with different
internal variable models sit side by side exactly as they do without inertia. The mass term becomes a
[`BilinearMultiIntegrator`](@ref) so that each subdomain contributes its own `ρ`.
"""
function semidiscretize(
    models::Dict{String, <: ElastodynamicsModel},
    discretization::FiniteElementDiscretization,
    mesh::AbstractGrid,
)
    sym = structural_displacement_symbol(models)
    vsym = _shared_symbol_or_error(models, model -> model.velocity_symbol, "velocity")
    _reject_velocity_constraints(discretization, vsym)

    quasistaticform = semidiscretize(
        Dict{String, QuasiStaticModel}(
            name => QuasiStaticModel(
                model.displacement_symbol,
                model.material_model,
                model.facet_models,
            ) for (name, model) in models
        ),
        discretization,
        mesh,
    )

    mass_term = BilinearMultiIntegrator(
        Dict{String, AbstractBilinearIntegrator}(
            name => BilinearMassIntegrator(
                model.ρ,
                _get_quadrature_from_discretization(discretization, sym),
                sym,
            ) for (name, model) in models
        ),
    )

    return _elastodynamics_function(
        quasistaticform,
        mass_term,
        discretization,
        mesh,
        sym,
        vsym,
        collect(keys(models)),
    )
end

"""
    _shared_symbol_or_error(models, accessor, what)

The symbol every model in a domain split agrees on. Distinct symbols would mean distinct fields, and
the state carries one field per role.
"""
function _shared_symbol_or_error(models::Dict{String}, accessor, what::AbstractString)
    symbols = Set(accessor(model) for model in values(models))
    length(symbols) == 1 ||
        error("All models in a domain split must share the same $what symbol, got $(symbols).")
    return first(symbols)
end

function semidiscretize(
    models::Dict{String, <: QuasiStaticModel},
    discretization::FiniteElementDiscretization,
    mesh::AbstractGrid,
)
    _check_model_subdomains_disjoint(mesh, collect(keys(models)))

    dh = DofHandler(mesh)
    integrators = Dict{String, NonlinearIntegrator}()
    for (name, model) in models
        add_subdomain!(dh, name, _approximation_descriptors(discretization, model))

        form_names = get_volumetric_weak_form_names(model)
        @assert length(form_names) == 1
        form_name = first(form_names)
        qrc = _get_quadrature_from_discretization(discretization, form_name) # FIXME we want a more intrusive approach which also takes the model into account here

        fqrc = _get_facet_quadrature_from_discretization(discretization, form_name)

        integrators[name] = NonlinearIntegrator(
            model,
            model.facet_models,
            get_field_variable_names(model),
            qrc,
            fqrc,
        )
    end
    close!(dh)

    lvh = _setup_internal_variable_handler(NonlinearMultiDomainIntegrator2(integrators), dh)

    ch = ConstraintHandler(dh)
    _add_dirichlet_conditions!(ch, discretization.dbcs)
    # FIXME add affine constraints due to AMR
    close!(ch)

    semidiscrete_problem = QuasiStaticFunction(
        dh,
        ch,
        lvh,
        NonlinearMultiDomainIntegrator2(integrators),
        discretization.assembly_strategy,
    )

    return semidiscrete_problem
end
