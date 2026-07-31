# Common modeling primitives are found here
"""
This described anything that is possibly condensed at element level.
"""
abstract type AbstractInternalModel end

struct EmptyInternalModel <: AbstractInternalModel end

struct EmptyInternalCache end

setup_internal_cache(::EmptyInternalModel, ::QuadratureRule, ::SubDofHandler) = EmptyInternalCache()

"""
    InternalVariableEvolution

Holy trait classifying the evolution law of a condensed internal variable `Q`, and with it the class
of the resulting system:

| trait                 | local problem per quadrature point | resulting system        |
| :-------------------- | :--------------------------------- | :---------------------- |
| `NoEvolution`         | none, or algebraic `L(F, Q) = 0`    | rate free               |
| `FirstOrderEvolution` | `dₜQ = L(F, Q)`                     | ODE in mass matrix form |
| `RateCoupledEvolution`| `dₜQ = L(F, dₜF, Q)`                | true DAE                |

This is a property of the *model*, deliberately not of the state cache it is lowered into. The
`Empty…CondensationMaterialStateCache` types say only that a model needs no extra scratch space for
its evaluation; they say nothing about whether it carries an internal variable or how that variable
evolves. Reading the classification off them conflates the two questions.

It is also askable before a mesh exists, which the cache-based answer is not.
"""
abstract type InternalVariableEvolution end
struct NoEvolution <: InternalVariableEvolution end
struct FirstOrderEvolution <: InternalVariableEvolution end
struct RateCoupledEvolution <: InternalVariableEvolution end

"""
    internal_variable_evolution(model) -> InternalVariableEvolution

The [`InternalVariableEvolution`](@ref) of `model`. Material models delegate to whatever internal
model they carry, mirroring `setup_internal_cache`.
"""
internal_variable_evolution(model) = error(
    "$(typeof(model)) does not declare how its internal variable evolves. Add a method " *
    "`Thunderbolt.internal_variable_evolution(::$(typeof(model)))` returning `NoEvolution()`, " *
    "`FirstOrderEvolution()` or `RateCoupledEvolution()`, or delegate to the internal model it wraps.",
)
internal_variable_evolution(::EmptyInternalModel) = NoEvolution()


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
