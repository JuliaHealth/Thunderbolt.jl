# # [Custom Stimulation Protocols](@id how-to-custom-stim-protocol)

# ## Analytical protocols
# The easiest way to implement a custom stimulation protocol is to use the [AnalyticalTransmembraneStimulationProtocol](@ref)
#
# We start by define a struct holding all the parameters and make it callable.
# The return value is the stimulus strength.
# If you want to have parameters with spatial variation, which can be exchanged easily, then simply add a field with custom type and a function which accepts a coordinate `x` and a time `t` as input.
# 
# Here we want to have a very simple S1S2 protocol with two spherical stimulus applied in a modular fashion.
# Hence, we define a struct holding two callable functions and the windows for the Stimuli.
using Thunderbolt, StaticArrays

struct SimpleS1S2Protocol{S1Type, S2Type, WT} <: Function
    S1::S1Type
    S1_window::WT
    S2::S2Type
    S2_window::WT
end

# The function call then simply selects the correct stimulus with time offset, and return 0 outside of the intervals.
function (protocol::SimpleS1S2Protocol)(x,t)
    if protocol.S1_window[1] < t < protocol.S1_window[2]
        return protocol.S1(x,t-protocol.S1_window[1])
    elseif protocol.S2_window[1] < t < protocol.S2_window[2]
        return protocol.S2(x,t-protocol.S2_window[1])
    end
    return 0.0
end

# It is now possible to use the protocol as follows
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
# where the coordinate_system_coefficient determines the type of coordiante passed into the protocol.

# !!! todo
#     We should develop a way to define custom stimulation protocols symbolically, e.g. via MTK.
