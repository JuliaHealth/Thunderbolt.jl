"""
    MTKLumpedCicuitModel

A lumped (0D) circulatory model for LV simulations as presented in [RegSalAfrFedDedQar:2022:cem](@citet).
"""
Base.@kwdef struct MTKLumpedCicuitModel{ProbType <: SciMLBase.ODEProblem} <:
                   AbstractLumpedCirculatoryModel
    # We generate a dummy problem to query the parameters
    prob::ProbType
    pressure_symbols::Vector{ModelingToolkit.Num}
end

function MTKLumpedCicuitModel(
    sys::ModelingToolkit.ODESystem,
    u0,
    pressure_symbols::Vector{ModelingToolkit.Num},
)
    # To construct the ODEProblem we need to provide an initial value for the pressures
    ps = [sym => 0.0 for sym in pressure_symbols]
    prob = SciMLBase.ODEProblem(sys, merge(Dict(u0), Dict(ps)), (0.0, 0.0))
    return MTKLumpedCicuitModel(prob, pressure_symbols)
end

function ODEFunction(model::MTKLumpedCicuitModel)
    return model.prob.f.sys
end

solution_size(model::MTKLumpedCicuitModel) = length(model.prob.u0)
num_states(model::MTKLumpedCicuitModel) = length(model.prob.u0)
num_unknown_pressures(model::MTKLumpedCicuitModel) = length(model.pressure_symbols)
function get_variable_symbol_index(model::MTKLumpedCicuitModel, symbol::ModelingToolkit.Num)
    SymbolicIndexingInterface.variable_index(model.prob, symbol)
end
function get_parameter_symbol_index(model::MTKLumpedCicuitModel, symbol::ModelingToolkit.Num)
    SymbolicIndexingInterface.parameter_index(model.prob, symbol)
end

function default_initial_condition!(u, model::MTKLumpedCicuitModel)
    u .= model.prob.u0
end
