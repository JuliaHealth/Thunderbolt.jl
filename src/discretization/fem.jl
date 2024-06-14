"""
"""
struct FiniteElementDiscretization
    """
    """
    interpolations::Dict{Symbol, InterpolationCollection}
    """
    """
    dbcs::Vector{Dirichlet}
    """
    """
    function FiniteElementDiscretization(ips::Dict{Symbol, <: InterpolationCollection})
        new(ips, Dirichlet[])
    end
    
    function FiniteElementDiscretization(ips::Dict{Symbol, <: InterpolationCollection}, dbcs::Vector{Dirichlet})
        new(ips, dbcs)
    end
end

semidiscretize(::CoupledModel, discretization, grid) = @error "No implementation for the generic discretization of coupled problems available yet."

function semidiscretize(model::TransientHeatModel, discretization::FiniteElementDiscretization, grid::AbstractGrid)
    ets = elementtypes(grid)
    @assert length(ets) == 1

    sym = model.solution_variable_symbol

    # TODO factor this out to make a isolated transient heat problem and call semidiscretize here. This should simplify testing.
    ip = getinterpolation(discretization.interpolations[sym], getcells(grid, 1))
    dh = DofHandler(grid)
    Ferrite.add!(dh, sym, ip)
    close!(dh);
    return TransientHeatFunction(
        model.κ,
        model.source,
        dh
    )
end

function semidiscretize(split::ReactionDiffusionSplit{<:MonodomainModel}, discretization::FiniteElementDiscretization, grid::AbstractGrid)
    epmodel = split.model
    φsym = epmodel.transmembrane_solution_symbol
    ets = elementtypes(grid)
    @assert length(ets) == 1

    heat_model = TransientHeatModel(
        ConductivityToDiffusivityCoefficient(epmodel.κ, epmodel.Cₘ, epmodel.χ),
        epmodel.stim,
        φsym,
    )

    heatfun = semidiscretize(
        heat_model,
        discretization,
        grid,
    )

    dh = heatfun.dh
    ndofsφ = ndofs(dh)
    # TODO we need some information about the discretization of this one, e.g. dofs a nodes vs dofs at quadrature points
    # TODO we should call semidiscretize here too - This is a placeholder for the nodal discretization
    odefun = PointwiseODEFunction(
        # TODO epmodel.Cₘ(x) and coordinates
        ndofsφ,
        epmodel.ion
    )
    # TODO This is a faulty assumption. We can have varying ndofs per cell.
    nstates_per_cell = num_states(odefun.ode)
    # TODO this assumes that the transmembrane potential is the first field. Relax this.
    heat_dofrange = 1:ndofsφ
    ode_dofrange = 1:nstates_per_cell*ndofsφ
    #
    semidiscrete_ode = GenericSplitFunction(
        (heatfun, odefun),
        (heat_dofrange, ode_dofrange),
        # No transfer operators needed, because the the solutions variables overlap with the subproblems perfectly
    )

    return semidiscrete_ode
end

function semidiscretize(model::StructuralModel{<:QuasiStaticModel}, discretization::FiniteElementDiscretization, grid::AbstractGrid)
    ets = elementtypes(grid)
    @assert length(ets) == 1 "Multiple elements not supported yet."

    ip = getinterpolation(discretization.interpolations[model.displacement_symbol], getcells(grid, 1))
    ip_geo = Ferrite.default_geometric_interpolation(ip) # TODO get interpolation from cell
    dh = DofHandler(grid)
    Ferrite.add!(dh, model.displacement_symbol, ip)
    close!(dh);

    ch = ConstraintHandler(dh)
    for dbc ∈ discretization.dbcs
        Ferrite.add!(ch, dbc)
    end
    close!(ch)

    semidiscrete_problem = QuasiStaticNonlinearFunction(
        dh,
        ch,
        model.mechanical_model,
        model.face_models
    )

    return semidiscrete_problem
end
