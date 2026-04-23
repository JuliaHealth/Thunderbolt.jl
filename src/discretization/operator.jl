# This function is defined to make things sufficiently type-stable
function _sequential_update_linearization_on_subdomain_Jr!(
    assembler,
    sdh,
    element_cache,
    facet_cache,
    tying_cache,
    u,
    time,
)
    # Prepare standard values
    ndofs = ndofs_per_cell(sdh)
    Jₑ = zeros(ndofs, ndofs)
    uₑ = zeros(ndofs)
    rₑ = zeros(ndofs)
    uₜ = get_tying_dofs(tying_cache, u)
    @inbounds for cell in CellIterator(sdh)
        fill!(Jₑ, 0)
        fill!(rₑ, 0)
        dofs = celldofs(cell)

        uₑ .= @view u[dofs]
        @timeit_debug "assemble element" assemble_element!(Jₑ, rₑ, uₑ, cell, element_cache, time)
        # TODO maybe it makes sense to merge this into the element routine in a modular fasion?
        # TODO benchmark against putting this into the FacetIterator
        @timeit_debug "assemble facets" assemble_element!(Jₑ, rₑ, uₑ, cell, facet_cache, time)
        @timeit_debug "assemble tying" assemble_tying!(Jₑ, rₑ, uₑ, uₜ, cell, tying_cache, time)
        assemble!(assembler, dofs, Jₑ, rₑ)
    end
end

function needs_update(op::LinearFerriteOperator, t)
    return _needs_update(op, op.integrator.integrand, t)
end

function _needs_update(op::LinearFerriteOperator, protocol::AnalyticalTransmembraneStimulationProtocol, t)
    for nonzero_interval ∈ protocol.nonzero_intervals
        nonzero_interval[1] ≤ t ≤ nonzero_interval[2] && return true
    end
    return false
end

function _needs_update(op::LinearFerriteOperator, protocol::NoStimulationProtocol, t)
    return false
end
