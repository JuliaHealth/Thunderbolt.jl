"""
    semidiscretize(model, discretization, mesh)

Transform a space-time model into a pure time-dependent problem.

The result is a semidiscrete *function* describing the spatial discretization. Which concrete
function type comes out depends on the model, on the discretization, and on any annotation wrapping
the model (e.g. [`ReactionDiffusionSplit`](@ref)).

## Implementing a new model

There is deliberately no common model supertype - a model is whatever implements the interface, so
that types owned by other packages can act as models too. At minimum a model needs

  - [`get_field_variable_names`](@ref) - the field variables it introduces, as a `Tuple` of `Symbol`
  - `get_volumetric_weak_form_names` - the volumetric weak forms it contributes, as a `Tuple`
  - a `semidiscretize` method

and, if it carries quadrature-point-local state, [`gather_internal_variable_infos`](@ref).
"""
semidiscretize

function semidiscretize(model, discretization, mesh)
    error(
        "semidiscretize is not implemented for a model of type $(typeof(model)) with a " *
        "discretization of type $(typeof(discretization)) on a mesh of type $(typeof(mesh)). " *
        "Either the model is not supported yet, or it is wrapped in the wrong annotation. " *
        "See the `semidiscretize` docstring for the interface a model has to implement.",
    )
end

function semidiscretize(models::Dict{String, Any}, discretization, mesh)
    semidiscretize(narrow_dict_types(models), discretization, mesh)
end
