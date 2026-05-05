
function needs_update(op::LinearFerriteOperator, t)
    return _needs_update(op, op.integrator.integrand, t)
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
