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
        qrcs::Dict{Symbol} = Dict{Symbol,Any}(),
        fqrcs::Dict{Symbol} = Dict{Symbol,Any}(),
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
        "Finite element discretization does not have an interpolation for $sym. Available symbols: $(collect(keys(disc.interpolations))) and $(collect(keys(disc.qrcs))).",
    )
end

semidiscretize(::CoupledModel, discretization, mesh::AbstractGrid) =
    @error "No implementation for the generic discretization of coupled problems available yet."

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

    if !isempty(discretization.subdomains)
        for name in discretization.subdomains
            add_subdomain!(dh, name, [ApproximationDescriptor(sym, ipc)])
        end
    else
        add_subdomain!(dh, single_subdomain_or_error(mesh), [ApproximationDescriptor(sym, ipc)])
    end
    close!(dh)

    T = get_coordinate_eltype(get_grid(dh))
    return AffineODEFunction(
        BilinearMassIntegrator(
            ConstantCoefficient(T(1.0)),
            haskey(discretization.qrcs, :mass) ? discretization.qrcs[:mass]  : qrc, # Allow e.g. mass lumping for explicit integrators.
            sym,
        ),
        BilinearDiffusionIntegrator(model.κ, qrc, sym),
        LinearIntegrator(model.source, qrc),
        dh,
        discretization.assembly_strategy,
    )
end

function register_affine_ode_integrators!(mass_integrators, rhs_integrators, linear_integrators, dh, name, discretization::FiniteElementDiscretization, model::TransientDiffusionModel)
    sym = model.solution_variable_symbol
    ipc = _get_interpolation_from_discretization(discretization, sym)
    add_subdomain!(dh, name, [ApproximationDescriptor(sym, ipc)])

    T = get_coordinate_eltype(get_grid(dh))

    qrc  = _get_quadrature_from_discretization(discretization, sym)
    # TODO allow e.g. mass lumping for explicit integrators.
    mass_integrators[name] = BilinearMassIntegrator(
        ConstantCoefficient(T(1.0)),
        haskey(discretization.qrcs, :mass) ? discretization.qrcs[:mass]  : qrc,
        sym,
    )
    rhs_integrators[name]  = BilinearDiffusionIntegrator(model.κ, qrc, sym)
    linear_integrators[name] = LinearIntegrator(model.source, qrc)
end

function register_affine_ode_integrators!(mass_integrators, rhs_integrators, linear_integrators, dh, name, discretization::FiniteElementDiscretization, model::InterfaceDiffusionModel)
    sym = model.solution_variable_symbol_1
    ipc = _get_interpolation_from_discretization(discretization, sym)
    add_subdomain!(dh, name, [ApproximationDescriptor(sym, ipc)])

    T = get_coordinate_eltype(get_grid(dh))

    qrc  = _get_quadrature_from_discretization(discretization, sym)
    # TODO allow e.g. mass lumping for explicit integrators.
    mass_integrators[name] = BilinearMassIntegrator(
        ConstantCoefficient(T(1.0)),
        haskey(discretization.qrcs, :mass) ? discretization.qrcs[:mass]  : qrc,
        sym,
    )
    rhs_integrators[name]  = BilinearInterfaceDiffusionIntegrator(model.G, qrc, sym,  model.solution_variable_symbol_2)
end

function semidiscretize(
    models::Dict{String, <:Union{<:TransientDiffusionModel, <:InterfaceDiffusionModel}},
    discretization::FiniteElementDiscretization,
    mesh::AbstractGrid,
)
    @assert length(discretization.dbcs) == 0 "Dirichlet conditions not supported yet for transient diffusion models"

    dh = DofHandler(mesh)

    # 3 weak forms
    rhs_integrators    = Dict{String, AbstractBilinearIntegrator}()
    mass_integrators   = Dict{String, AbstractBilinearIntegrator}()
    linear_integrators = Dict{String, AbstractLinearIntegrator}()

    for (name, model) in models
        register_affine_ode_integrators!(mass_integrators, rhs_integrators, linear_integrators, dh, name, discretization, model)
    end

    close!(dh)

    return AffineODEFunction(
        mass_integrators,
        rhs_integrators,
        linear_integrators,
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
    if !isempty(discretization.subdomains)
        for name in discretization.subdomains
            add_subdomain!(dh, name, [ApproximationDescriptor(sym, ipc)])
        end
    else
        add_subdomain!(dh, single_subdomain_or_error(mesh), [ApproximationDescriptor(sym, ipc)])
    end
    close!(dh)

    ch = ConstraintHandler(dh)
    for dbc ∈ discretization.dbcs
        Ferrite.add!(ch, dbc)
    end
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
    odefun = PointwiseODEFunction(
        # TODO epmodel.Cₘ(x)
        epmodel.ion,
        split.cs === nothing ? nothing : compute_nodal_values(split.cs, dh, φsym),
        1:ndofsφ,
        0,
    )
    nstates_per_point = num_states(odefun.ode)
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
    semidiscretize(ReactionDiffusionSplit(models), discretization, mesh)
end

function semidiscretize(
    split::ReactionDiffusionSplit{<:Dict{String, <: Union{<:AbstractEPModel, <:InterfaceDiffusionModel}}},
    discretization::FiniteElementDiscretization,
    mesh::AbstractGrid,
)
    epmodels = split.model

    heat_models = Dict(name => semidiscretize_map_diffusion_part(epmodel) for (name, epmodel) in epmodels)

    heatfun = semidiscretize(heat_models, discretization, mesh)

    dh = heatfun.dh
    ndofsφ = ndofs(dh)

    # TODO we need some information about the discretization of this one, e.g. dofs a nodes vs dofs at quadrature points
    # TODO we should call semidiscretize here too - This is a placeholder for the nodal discretization
    inner_functions = PointwiseODEFunction[]
    offset = 0
    xφ = (split.cs === nothing ? nothing : compute_nodal_values(split.cs, dh, epmodel.transmembrane_solution_symbol))
    for (name, model) in epmodels
        if typeof(model) <: AbstractEPModel # Only handle the EP models
            subdofs = collect_dofs_on_subdomain(dh, mesh, name)
            push!(inner_functions,
                PointwiseODEFunction(
                    model.ion,
                    xφ,
                    subdofs,
                    offset,
                )
            )
            offset += length(subdofs)*(num_states(model.ion)-1)
        end
    end
    odefun = PointwiseMultiODEFunction(
        inner_functions,
    )
    # TODO this assumes that the transmembrane potential is the first field. Relax this.
    heat_dofrange = 1:ndofsφ
    ode_dofrange = 1:offset
    #
    semidiscrete_ode = GenericSplitFunction(
        (heatfun, odefun),
        (heat_dofrange, ode_dofrange),
        # No transfer operators needed, because the the solutions variables overlap with the subproblems perfectly
    )

    return semidiscrete_ode
end

# Solid mechanics semidiscretize interface
function semidiscretize_register_subdomains!(
    dh,
    lvh,
    model,
    discretization::FiniteElementDiscretization,
    subdomains,
)
    semidiscretize_register_subdomains!(
        dh,
        lvh,
        model,
        model.material_model,
        discretization,
        subdomains,
    )
end
function semidiscretize_register_subdomains!(
    dh,
    lvh,
    model,
    material_model::AbstractMaterialModel,
    discretization::FiniteElementDiscretization,
    subdomains,
)
    sym = model.displacement_symbol
    ipc = _get_interpolation_from_discretization(discretization, sym)
    qrc = _get_quadrature_from_discretization(discretization, sym)
    if !isempty(subdomains)
        for name in subdomains
            add_subdomain!(dh, name, [ApproximationDescriptor(sym, ipc)])
            add_subdomain!(lvh, name, gather_internal_variable_infos(material_model), qrc, dh)
        end
    else
        name = single_subdomain_or_error(get_grid(dh))
        add_subdomain!(dh, name, [ApproximationDescriptor(sym, ipc)])
        add_subdomain!(lvh, name, gather_internal_variable_infos(material_model), qrc, dh)
    end
end

function semidiscretize_register_subdomains!(
    dh,
    lvh,
    model,
    material_models::MultiMaterialModel,
    discretization::FiniteElementDiscretization,
    subdomains,
)
    if length(subdomains) > 1
        @warn "Multimaterials ignore discretization subdomains for now."
    end
    semidiscretize_register_subdomains_multi!(
        dh,
        lvh,
        model,
        material_models.materials,
        material_models.domains,
        material_models.domain_names,
        discretization,
    )
end
@unroll function semidiscretize_register_subdomains_multi!(
    dh,
    lvh,
    model,
    material_models,
    domains,
    domain_names,
    discretization,
)
    sym = model.displacement_symbol
    ipc = _get_interpolation_from_discretization(discretization, sym)
    qrc = _get_quadrature_from_discretization(discretization, sym)
    idx = 1
    @unroll for material_model in material_models
        add_subdomain!(dh, domain_names[idx], [ApproximationDescriptor(sym, ipc)])
        add_subdomain!(
            lvh,
            domain_names[idx],
            gather_internal_variable_infos(material_model),
            qrc,
            dh,
        )
        idx += 1
    end
end

# FIXME redirect to multi-domain version
function semidiscretize(
    model::QuasiStaticModel,
    discretization::FiniteElementDiscretization,
    mesh::AbstractGrid,
)
    sym = model.displacement_symbol
    qrc = _get_quadrature_from_discretization(discretization, sym)
    fqrc = _get_facet_quadrature_from_discretization(discretization, sym)
    dh = DofHandler(mesh)
    lvh = InternalVariableHandler(mesh)
    semidiscretize_register_subdomains!(dh, lvh, model, discretization, discretization.subdomains)
    close!(dh)
    close!(lvh)

    ch = ConstraintHandler(dh)
    for dbc ∈ discretization.dbcs
        Ferrite.add!(ch, dbc)
    end
    close!(ch)

    semidiscrete_problem = QuasiStaticFunction(
        dh,
        ch,
        lvh,
        NonlinearIntegrator(
            model,
            model.facet_models,
            [sym],
            qrc,
            fqrc,
        ),
        discretization.assembly_strategy,
    )

    return semidiscrete_problem
end

function semidiscretize(
    models::Dict{String, QuasiStaticModel},
    discretization::FiniteElementDiscretization,
    mesh::AbstractGrid,
)
    dh = DofHandler(mesh)
    lvh = InternalVariableHandler(mesh)
    integrators = Dict{String, NonlinearIntegrator}()
    for (name, model) in models
        for sym in get_field_variable_names(model)
            ipc = _get_interpolation_from_discretization(discretization, sym)
            add_subdomain!(dh, name, [ApproximationDescriptor(sym, ipc)])
        end

        form_names = get_volumetric_weak_form_names(model)
        @assert length(form_names) == 1
        form_name = first(form_names)
        qrc  = _get_quadrature_from_discretization(discretization, form_name) # FIXME we want a more intrusive approach which also takes the model into account here
        add_subdomain!(lvh, name, gather_internal_variable_infos(model), qrc, dh)

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
    close!(lvh)

    ch = ConstraintHandler(dh)
    for dbc ∈ discretization.dbcs
        Ferrite.add!(ch, dbc)
    end
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
