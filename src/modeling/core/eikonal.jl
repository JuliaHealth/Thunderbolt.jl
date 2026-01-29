@doc raw"""
    EikonalModel(DiffusivityCoefficientType)

Model formulated as ``\sqrt{{\nabla t_a}^T \kappa(x) \nabla t_a} = 1``
"""
struct EikonalModel{DiffusivityCoefficientType}
    κ::DiffusivityCoefficientType
end