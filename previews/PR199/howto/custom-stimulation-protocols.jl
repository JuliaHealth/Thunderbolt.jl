using Thunderbolt, StaticArrays

struct SimpleS1S2Protocol{S1Type, S2Type, WT} <: Function
    S1::S1Type
    S1_window::WT
    S2::S2Type
    S2_window::WT
end

function (protocol::SimpleS1S2Protocol)(x,t)
    if protocol.S1_window[1] < t < protocol.S1_window[2]
        return protocol.S1(x,t-protocol.S1_window[1])
    elseif protocol.S2_window[1] < t < protocol.S2_window[2]
        return protocol.S2(x,t-protocol.S2_window[1])
    end
    return 0.0
end

coordinate_system_coefficient = CoordinateSystemCoefficient(CartesianCoordinateSystem{3}()) # Or some cardiac coordinate system
stimulus_around_zero(x,t) = max(1.0-norm(x),0.0)
stimulus_around_one(x,t)  = max(1.0-norm(x+one(x)),0.0)
s1s2fun = SimpleS1S2Protocol(
    stimulus_around_zero, SVector((  0.0,   1.0)),
    stimulus_around_one , SVector((200.0, 201.0)),
)
protocol_nonzero_intervals = [s1s2fun.S1_window, s1s2fun.S2_window]
protocol = Thunderbolt.AnalyticalTransmembraneStimulationProtocol(
    AnalyticalCoefficient(
        s1s2fun,
        coordinate_system_coefficient,
    ),
    protocol_nonzero_intervals,
)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
