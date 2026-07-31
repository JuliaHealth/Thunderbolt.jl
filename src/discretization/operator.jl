
function needs_update(op::LinearFerriteOperator, t)
    return needs_update(op, op.integrator, t)
end

function needs_update(op::LinearFerriteOperator, integrator, t)
    return _needs_update(op, integrator.integrand, t)
end

function needs_update(op::LinearFerriteOperator, integrator::LinearMultiIntegrator, t)
    return any([
        _needs_update(op, subintegrator.integrand, t) for
        subintegrator in values(integrator.subintegrators)
    ])
end

function _needs_update(
    op::LinearFerriteOperator,
    protocol::AnalyticalTransmembraneStimulationProtocol,
    t,
)
    for nonzero_interval ∈ protocol.nonzero_intervals
        nonzero_interval[1] ≤ t ≤ nonzero_interval[2] && return true
    end
    return false
end

function _needs_update(op::LinearFerriteOperator, protocol::NoStimulationProtocol, t)
    return false
end

needs_update(::LinearNullOperator, t) = false
