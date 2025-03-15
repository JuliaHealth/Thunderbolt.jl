"""
Descriptor for a finite element discretization of a part of a PDE over some subdomain.
    
!!! note
    The current implementation is restricted to Bubnov-Galerkin methods. Petrov-Galerkin support will come in the future.
"""
struct FiniteElementDiscretization
    """
    """
    interpolations::Dict{Symbol}#, Union{<:InterpolationCollection, Pair{<:InterpolationCollection, <:QuadratureRuleCollection}}}
    """
    """
    dbcs::Vector{Dirichlet}
    """
    """
    subdomains::Vector{String}
    """
    """
    mass_qrc::Union{<:QuadratureRuleCollection,Nothing} # TODO maybe an "extras" field should be used instead :)
    """
    """
    function FiniteElementDiscretization(ips::Dict{Symbol}, dbcs::Vector{Dirichlet} = Dirichlet[], subdomains::Vector{String} = [""], mass_qrc = nothing)
        new(ips, dbcs, subdomains, mass_qrc)
    end
end

_extract_ipc(ipc::InterpolationCollection) = ipc
_extract_ipc(p::Pair{<:InterpolationCollection, <:QuadratureRuleCollection}) = first(p)

function _extract_qrc(ipc::InterpolationCollection)
    ansatzorder = getorder(ipc)
    return QuadratureRuleCollection(max(2ansatzorder-1,2))
end
_extract_qrc(p::Pair{<:InterpolationCollection, <:QuadratureRuleCollection}) = last(p)

# Internal utility with proper error message
function _get_interpolation_from_discretization(disc::FiniteElementDiscretization, sym::Symbol)
    if !haskey(disc.interpolations, sym)
        error("Finite element discretization does not have an interpolation for $sym. Available symbols: $(collect(keys(disc.interpolations))).")
    end
    return _extract_ipc(disc.interpolations[sym])
end
function _get_quadrature_from_discretization(disc::FiniteElementDiscretization, sym::Symbol)
    if !haskey(disc.interpolations, sym)
        error("Finite element discretization does not have an interpolation for $sym. Available symbols: $(collect(keys(disc.interpolations))).")
    end
    return _extract_qrc(disc.interpolations[sym])
end
function _get_facet_quadrature_from_discretization(disc::FiniteElementDiscretization, sym::Symbol)
    if !haskey(disc.interpolations, sym)
        error("Finite element discretization does not have an interpolation for $sym. Available symbols: $(collect(keys(disc.interpolations))).")
    end
    intorder = getorder(_extract_ipc(disc.interpolations[sym]))
    return FacetQuadratureRuleCollection(intorder)
end

semidiscretize(::CoupledModel, discretization, mesh::AbstractGrid) = @error "No implementation for the generic discretization of coupled problems available yet."

function semidiscretize(model::TransientDiffusionModel, discretization::FiniteElementDiscretization, mesh::AbstractGrid)
    @assert length(discretization.dbcs) == 0 "Dirichlet conditions not supported yet for TransientDiffusionProblem"

    sym = model.solution_variable_symbol
    ipc = _get_interpolation_from_discretization(discretization, sym)
    qrc = _get_quadrature_from_discretization(discretization, sym)
    dh = DofHandler(mesh)
    for name in discretization.subdomains
        add_subdomain!(dh, name, [ApproximationDescriptor(sym, ipc)])
    end
    close!(dh)

    T = get_coordinate_eltype(get_grid(dh))
    return AffineODEFunction(
        BilinearMassIntegrator(
            ConstantCoefficient(T(1.0)),
            discretization.mass_qrc === nothing ? qrc : mass_qrc, # Allow e.g. mass lumping for explicit integrators.
            sym,
        ),
        BilinearDiffusionIntegrator(
            model.κ,
            qrc,
            sym,
        ),
        model.source, # TODO qrc for source term
        dh,
    )
end

function semidiscretize(model::SteadyDiffusionModel, discretization::FiniteElementDiscretization, mesh::AbstractGrid)
    sym = model.solution_variable_symbol
    ipc = _get_interpolation_from_discretization(discretization, sym)
    qrc = _get_quadrature_from_discretization(discretization, sym)
    dh = DofHandler(mesh)
    for name in discretization.subdomains
        add_subdomain!(dh, name, [ApproximationDescriptor(sym, ipc)])
    end
    close!(dh)

    ch = ConstraintHandler(dh)
    for dbc ∈ discretization.dbcs
        Ferrite.add!(ch, dbc)
    end
    close!(ch)

    return AffineSteadyStateFunction(
        BilinearDiffusionIntegrator(
            model.κ,
            qrc,
            sym,
        ),
        model.source, # TODO qrc for source term
        dh,
        ch,
    )
end

function semidiscretize(split::ReactionDiffusionSplit{<:MonodomainModel}, discretization::FiniteElementDiscretization, mesh::AbstractGrid)
    epmodel = split.model
    φsym = epmodel.transmembrane_solution_symbol

    heat_model = TransientDiffusionModel(
        ConductivityToDiffusivityCoefficient(epmodel.κ, epmodel.Cₘ, epmodel.χ),
        epmodel.stim,
        φsym,
    )

    heatfun = semidiscretize(
        heat_model,
        discretization,
        mesh,
    )

    dh = heatfun.dh
    ndofsφ = ndofs(dh)
    # TODO we need some information about the discretization of this one, e.g. dofs a nodes vs dofs at quadrature points
    # TODO we should call semidiscretize here too - This is a placeholder for the nodal discretization
    odefun = PointwiseODEFunction(
        # TODO epmodel.Cₘ(x)
        ndofsφ,
        epmodel.ion,
        split.cs === nothing ? nothing : compute_nodal_values(split.cs, dh, φsym)
    )
    nstates_per_point = num_states(odefun.ode)
    # TODO this assumes that the transmembrane potential is the first field. Relax this.
    heat_dofrange = 1:ndofsφ
    ode_dofrange = 1:nstates_per_point*ndofsφ
    #
    semidiscrete_ode = GenericSplitFunction(
        (heatfun, odefun),
        (heat_dofrange, ode_dofrange),
        # No transfer operators needed, because the the solutions variables overlap with the subproblems perfectly
    )

    return semidiscrete_ode
end

function semidiscretize(model::QuasiStaticModel, discretization::FiniteElementDiscretization, mesh::AbstractGrid)
    sym = model.displacement_symbol
    ipc = _get_interpolation_from_discretization(discretization, sym)
    qrc = _get_quadrature_from_discretization(discretization, sym)
    dh = DofHandler(mesh)
    lvh = InternalVariableHandler(mesh)
    for name in discretization.subdomains
        add_subdomain!(dh, name, [ApproximationDescriptor(sym, ipc)])
        add_subdomain!(lvh, name, gather_internal_variable_infos(model.material_model), qrc, dh)
    end
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
            model.face_models,
            [sym],
            qrc,
            _get_facet_quadrature_from_discretization(discretization, sym),
        ),
    )

    return semidiscrete_problem
end
