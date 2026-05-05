# TODO try to reproduce this via the BlockOperator
@concrete struct AssembledRSAFDQ2022Operator <: AbstractBlockOperator
    J
    strategy
    subdomain_caches
    dh
    integrator
    chambers
    tying_caches
end

# Interface
function FerriteOperators.update_linearization!(
    op::AssembledRSAFDQ2022Operator,
    u_::AbstractVector,
    p,
)
    error("Not implemented yet.")
end
function FerriteOperators.update_linearization!(
    op::AssembledRSAFDQ2022Operator,
    residual_::AbstractVector,
    u_::AbstractVector,
    p,
)
    (; J, strategy, subdomain_caches, chambers, tying_caches, dh) = op

    bs = blocksizes(J)
    s1 = bs[1, 1][1]
    s2 = bs[2, 2][1]
    u  = BlockedVector(u_, [s1, s2])
    ud = @view u[Block(1)]
    up = @view u[Block(2)]

    residual  = BlockedVector(residual_, [s1, s2])
    residuald = @view residual[Block(1)]
    residualp = @view residual[Block(2)]
    fill!(residuald, 0.0)
    fill!(residualp, 0.0)

    Jdd = @view J[Block(1, 1)]
    Jpd = @view J[Block(2, 1)]
    Jdp = @view J[Block(1, 2)]
    fill!(Jpd, 0.0)
    fill!(Jdp, 0.0)

    # Pass 1: Assemble volume as usual
    assembler = start_assemble(strategy, J, residual)
    task = FerriteOperators.AssembleLinearizationJR(assembler, u, p)

    FerriteOperators.execute_on_subdomains!(task, strategy, subdomain_caches)

    # Pass 2: Assemble forward and backward coupling contributions
    # TODO wrap into task system as boundary integration
    for (chamber_index, chamber) ∈ enumerate(chambers)
        V⁰ᴰ = chamber.V⁰ᴰval
        chamber_pressure = u[chamber.pressure_dof_index_local] # We can also make this up[pressure_dof_index] with local index

        Jpd_current = @view Jpd[chamber_index, :]
        Jdp_current = @view Jdp[:, chamber_index]

        residualp_current = @view residualp[chamber_index]

        for (sdh, tying_cache) in tying_caches[chamber_index]
            # FIXME allocator api
            Kₑ = zeros(ndofs_per_cell(sdh)+1, ndofs_per_cell(sdh)+1)
            rₑ = zeros(ndofs_per_cell(sdh)+1)
            uₑ = zeros(ndofs_per_cell(sdh)+1)
            for facet in FacetIterator(sdh, tying_cache.facets)
                # FIXME loader function
                dofs = [celldofs(facet); chamber.pressure_dof_index_local]
                uₑ .= u[dofs]
                fill!(Kₑ, 0.0)
                fill!(rₑ, 0.0)
                # FIXME use facet directly
                assemble_facet!(Kₑ, rₑ, uₑ, facet.cc, facet.current_facet_id, tying_cache, p)
                assemble!(assembler, dofs, Kₑ, rₑ)
            end
        end

        residualp_current[1] -= V⁰ᴰ

        @info "Chamber $chamber_index p=$chamber_pressure, V0=$V⁰ᴰ"
    end

    FerriteOperators.finalize_assembly!(assembler)
end
function FerriteOperators.residual!(
    op::AssembledRSAFDQ2022Operator,
    residual_::AbstractVector,
    u_::AbstractVector,
    p,
)
    error("Not implemented yet.")
end

getJ(op::AssembledRSAFDQ2022Operator) = op.J
getJ(op::AssembledRSAFDQ2022Operator, i::Block) = @view op.J[i]

function _find_sdhs(dh, facetset)
    facet = first(facetset)
    sdhs = SubDofHandler[]
    for sdh in dh.subdofhandlers
        if facet[1] ∈ sdh.cellset
            push!(sdhs, sdh)
        end
    end
    return sdhs
end

function setup_operator(f::RSAFDQ20223DFunction, solver::AbstractNonlinearSolver)
    (; tying_info, structural_function) = f
    (; dh, integrator, assembly_strategy) = structural_function

    operator_strategy =
        FerriteOperators.setup_operator_strategy_cache(assembly_strategy, integrator, dh)
    # TODO we are missing a way to dynamically extend the sparsity pattern in FerriteOperators
    J                = FerriteOperators.create_system_matrix(operator_strategy, dh)
    subdomain_caches = FerriteOperators.setup_subdomain_caches(operator_strategy, integrator, dh)
    # TODO this is also not possible yet
    tying_caches = [
        [
            (
                sdh,
                setup_boundary_cache(
                    Pressure3D0DVolumeCouplerIntegrator(
                        integrator.fqrc,
                        integrator.volume_model.displacement_symbol,
                        :pₗᵥ, #chamber.pressure_symbol # FIXME
                        chamber.facets,
                        chamber.volume_method,
                    ),
                    sdh,
                ),
            ) for sdh in _find_sdhs(dh, chamber.facets)
        ] for chamber in tying_info.chambers
    ]

    num_chambers = length(tying_info.chambers)
    block_sizes = [ndofs(dh), num_chambers]
    total_size = sum(block_sizes)
    # First we initialize an empty dummy block array
    Jblock = BlockArray(spzeros(total_size, total_size), block_sizes, block_sizes)
    Jblock[Block(1, 1)] = J
    # TODO optimize storage
    Jblock[Block(1, 2)] = sparse(ones(ndofs(dh), num_chambers))
    Jblock[Block(2, 1)] = sparse(ones(num_chambers, ndofs(dh)))
    Jblock[Block(2, 2)] = sparse(ones(num_chambers, num_chambers))
    Ferrite.fillzero!(Jblock)

    return AssembledRSAFDQ2022Operator(
        Jblock,
        operator_strategy,
        subdomain_caches,
        dh,
        integrator,
        tying_info.chambers,
        tying_caches,
    )
end
