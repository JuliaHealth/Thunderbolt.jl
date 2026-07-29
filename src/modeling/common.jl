# Common modeling primitives are found here
"""
This described anything that is possibly condensed at element level.
"""
abstract type AbstractInternalModel end

struct EmptyInternalModel <: AbstractInternalModel end

struct EmptyInternalCache end

setup_internal_cache(::EmptyInternalModel, ::QuadratureRule, ::SubDofHandler) = EmptyInternalCache()


abstract type AbstractSourceTerm end

"""
    is_coupling_model(model) -> Bool

Capability trait: does `model` describe a *coupling* between existing fields rather than a physics
domain of its own?

A coupling model attaches to field variables introduced by other models - typically across an
interface between subdomains - and therefore does not own a block of the solution vector the way a
bulk model does. `InterfaceDiffusionModel` is the current example.

This is deliberately independent of [`has_pointwise_reaction_part`](@ref): the two answer different
questions, and a coupling model may well carry its own reaction dynamics (e.g. gap-junction
kinetics on an interface). Code deciding *whether a model owns a domain block* must ask this trait,
not infer it from the presence or absence of a reaction part.
"""
is_coupling_model(model) = false

include("core/coordinate_systems.jl")

include("core/coefficients.jl")
include("core/analytical_coefficient.jl")

include("core/weak_boundary_conditions.jl")

include("core/mass.jl")
include("core/diffusion.jl")
include("core/linear.jl")
include("core/nonlinear.jl")
include("core/multi-integrator.jl")
