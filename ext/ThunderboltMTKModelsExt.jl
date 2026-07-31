"""
    ThunderboltMTKModelsExt

The prebuilt ModelingToolkit circuit definitions (`RSAFDQ2022CircuitMTK`, …), reached from user code
via [`Thunderbolt.mtk_models`](@ref).

These are separated from `ThunderboltMTKExt` because `@mtkmodel` is provided by `SciCompDSL`, not by
`ModelingToolkit`, and SciCompDSL's dependency closure (Symbolics, SymbolicUtils, ModelingToolkitBase)
is most of what makes loading expensive. Triggering on both packages means `using ModelingToolkit`
alone still gets you the 3D-0D coupling without paying for the model DSL.
"""
module ThunderboltMTKModelsExt

using Thunderbolt
using ModelingToolkit
using SciCompDSL

include("mtkmodels.jl")

end # module
