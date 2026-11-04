"""
    _chamber_coupling(displacement_symbol, chamber)

The sparsity the chamber pressure needs, as Ferrite's coupling descriptor.

The pressure is the facet items' own tail (`facet_item_global_dofs`), so the only local systems it
enters are the tying facets': `FacetCoupling` over the chamber surface allocates the displacement
dofs of the adjacent cells and nothing beyond them. The cell sweep of those subdomains assembles the
pure displacement system and never addresses a pressure entry.
"""
_chamber_coupling(displacement_symbol, chamber) = FacetCoupling(
    chamber.facets;
    algebraic_coupling = ((displacement_symbol, chamber.pressure_symbol),),
)

setup_local_solver_cache(f::RSAFDQ20223DFunction, solver::HomotopyPathSolver) =
    setup_local_solver_cache(f.structural_function, solver)

function setup_stage_operator(
    f::RSAFDQ20223DFunction,
    solver::HomotopyPathSolver,
    local_solver_cache,
    t₀,
)
    (; tying_info, structural_function) = f
    (; dh, ch, integrator) = structural_function
    chambers = tying_info.chambers
    n_chambers = length(chambers)
    n_u = ndofs(dh) - n_chambers

    # The tying facets write into one dof shared by every chamber facet, which no coloring can make
    # race free, so the scheduling is sequential regardless of what the discretization asked for.
    couplings =
        Tuple(_chamber_coupling(chamber.displacement_symbol, chamber) for chamber in chambers)
    # CSC blocks, not the CSR of FerriteOperators' own blocked-assembly example:
    # `SchurComplementLinearSolver`'s inner `UMFPACKFactorization` factorizes the (1,1) block, which
    # needs CSC.
    strategy = AssemblyStrategy(
        FullAssembly(
            FerriteOperators.BlockedOperatorSpecification(
                [n_u, n_chambers],
                BlockMatrix{Float64, Matrix{SparseMatrixCSC{Float64, Int}}};
                algebraic_couplings = couplings,
                constraint_handler = ch,
            ),
        ),
        SequentialScheduling(),
        get_strategy(f).device,
    )

    return setup_operator(
        strategy,
        _annotate_with_local_solver_cache(integrator, local_solver_cache),
        dh;
        slots = THUNDERBOLT_STAGE_SLOTS,
    )
end
