"""
    ThunderboltMTKExt

Everything in Thunderbolt that needs `ModelingToolkit`. Loaded automatically once `ModelingToolkit`
is available.

The split follows one rule: a type stays in `Thunderbolt` if it is user-facing and can be *defined*
without MTK; only the things that genuinely name an MTK type live here. That keeps
`MTKLumpedCicuitModel`, `ChamberVolumeCoupling` and `LumpedFluidSolidCoupler` exported and
dispatchable from the base package, with `Thunderbolt` supplying an erroring stub constructor that
this extension overrides by specificity.

The prebuilt circuit definitions need `@mtkmodel`, which only `SciCompDSL` provides, so they live in
the separate `ThunderboltMTKModelsExt` — otherwise merely loading `ModelingToolkit` would drag in
SciCompDSL's Symbolics/SymbolicUtils closure and undo most of the load-time saving.
"""
module ThunderboltMTKExt

using Thunderbolt
using ModelingToolkit

import Thunderbolt:
    Thunderbolt,
    MTKLumpedCicuitModel,
    ThreadedSparseMatrixCSR,
    mul,
    mtk_parameter_query_filter,
    get_variable_symbol_index,
    get_parameter_symbol_index

import Thunderbolt.SciMLBase
import Thunderbolt.SymbolicIndexingInterface
import Base: *
import Thunderbolt.LinearAlgebra: mul!

# ------------------------------------------------------------------------------------------------
# MTKLumpedCicuitModel — the real constructor. More specific than the erroring stub in
# `src/modeling/fluid/lumped-mtk.jl`, so it wins whenever this extension is loaded.

function Thunderbolt.MTKLumpedCicuitModel(
    sys::ModelingToolkit.ODESystem,
    u0,
    pressure_symbols::Vector{ModelingToolkit.Num},
)
    # To construct the ODEProblem we need to provide an initial value for the pressures
    ps = [sym => 0.0 for sym in pressure_symbols]
    prob = SciMLBase.ODEProblem(sys, merge(Dict(u0), Dict(ps)), (0.0, 0.0))
    return MTKLumpedCicuitModel(prob, pressure_symbols)
end

function Thunderbolt.get_variable_symbol_index(
    model::MTKLumpedCicuitModel,
    symbol::ModelingToolkit.Num,
)
    return SymbolicIndexingInterface.variable_index(model.prob, symbol)
end
function Thunderbolt.get_parameter_symbol_index(
    model::MTKLumpedCicuitModel,
    symbol::ModelingToolkit.Num,
)
    return SymbolicIndexingInterface.parameter_index(model.prob, symbol)
end

# ------------------------------------------------------------------------------------------------
# Symbol queries — the fallback returning `false` lives in `src/utils.jl`.

Thunderbolt.mtk_parameter_query_filter(param::ModelingToolkit.BasicSymbolic, sym) = true

# ------------------------------------------------------------------------------------------------
# Dispatch disambiguation against MTK-owned array types; counterpart of `src/disambiguation.jl`.

*(::ThreadedSparseMatrixCSR, ::ModelingToolkit.Symbolics.Arr{<:Any, 1}) = @error "Not implemented"

mul!(
    ::ModelingToolkit.ModelingToolkitBase.JumpProcesses.ExtendedJumpArray,
    ::ThreadedSparseMatrixCSR,
    ::AbstractVector{<:Number},
) = @error "Not implemented"

end # module
