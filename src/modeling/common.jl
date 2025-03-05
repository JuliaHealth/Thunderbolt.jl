# Common modeling primitives are found here
"""
This described anything that is possibly condensed at element level.
"""
abstract type AbstractInternalModel end

struct EmptyInternalModel <: AbstractInternalModel
end

struct EmptyInternalCache
end

setup_internal_cache(::EmptyInternalModel, ::QuadratureRule, ::SubDofHandler) = EmptyInternalCache()

# function state(model_cache::EmptyInternalCache, geometry_cache, qp::QuadraturePoint, time)
#     return EmptyInternal()
# end

abstract type AbstractSourceTerm end

include("core/coordinate_systems.jl")

include("core/coefficients.jl")
include("core/analytical_coefficient.jl")

include("core/element_interface.jl")
include("core/composite_elements.jl")
include("core/weak_boundary_conditions.jl")

abstract type AbstractBilinearIntegrator end
include("core/mass.jl")
include("core/diffusion.jl")
include("core/nonlinear.jl")
