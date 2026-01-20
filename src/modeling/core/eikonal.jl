@doc raw"""
    TransientDiffusionModel(conductivity_coefficient, source_term, solution_variable_symbol)

Model formulated as ``\partial_t u = \nabla \cdot \kappa(x) \nabla u + f``
"""
struct EikonalModel{DiffusivityCoefficientType}
    κ::DiffusivityCoefficientType
end