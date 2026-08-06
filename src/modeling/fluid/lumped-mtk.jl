"""
    MTKLumpedCicuitModel

A lumped (0D) circulatory model for LV simulations as presented in [RegSalAfrFedDedQar:2022:cem](@citet).

!!! note
    Building one of these needs `ModelingToolkit`. The type lives here so that it stays exported and
    dispatchable, but its constructor and the symbol-lookup methods are supplied by
    `ThunderboltMTKExt`, which loads as soon as `ModelingToolkit` is available. See also
    [`mtk_models`](@ref).
"""
Base.@kwdef struct MTKLumpedCicuitModel{ProbType <: SciMLBase.ODEProblem, PSType} <:
                   AbstractLumpedCirculatoryModel
    # We generate a dummy problem to query the parameters
    prob::ProbType
    # `Vector{ModelingToolkit.Num}` in practice, but naming that type here would make the struct
    # undefinable without ModelingToolkit loaded.
    pressure_symbols::PSType
end

# Stub for the user-facing constructor. The real method dispatches on `ModelingToolkit.ODESystem` and
# lives in `ThunderboltMTKExt`, where it wins by specificity. This signature cannot mention the MTK
# types at all — they do not exist until the extension loads.
function MTKLumpedCicuitModel(sys, u0, pressure_symbols)
    return error(
        "Constructing an `MTKLumpedCicuitModel` requires ModelingToolkit. Run `using ModelingToolkit` " *
        "to load `ThunderboltMTKExt`, which supplies this constructor.",
    )
end

function ODEFunction(model::MTKLumpedCicuitModel)
    return model.prob.f.sys
end

solution_size(model::MTKLumpedCicuitModel) = length(model.prob.u0)
num_states(model::MTKLumpedCicuitModel) = length(model.prob.u0)
num_unknown_pressures(model::MTKLumpedCicuitModel) = length(model.pressure_symbols)

function default_initial_state!(u, model::MTKLumpedCicuitModel)
    u .= model.prob.u0
end
