"""
    semidiscretize(model, discretization, mesh)

Transform a space-time model into a pure time-dependent problem.
"""
semidiscretize

semidiscretize(model, discretization, mesh) = error("Not implemented yet.")

function semidiscretize(models::Dict{String, Any}, discretization, mesh)
    semidiscretize(narrow_dict_types(models), discretization, mesh)
end
