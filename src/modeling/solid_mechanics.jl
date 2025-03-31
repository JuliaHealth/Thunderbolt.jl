
"""
    QuasiStaticModel(displacement_sym, mechanical_model, face_models)

A generic model for quasi-static mechanical problems.
"""
struct QuasiStaticModel{MM #= <: AbstractMaterialModel =#, FM}
    displacement_symbol::Symbol
    material_model::MM
    face_models::FM
end

get_field_variable_names(model::QuasiStaticModel) = (model.displacement_symbol, )

"""
    ElastodynamicsModel(displacement_sym, velocity_symbol, material_model::AbstractMaterialModel, face_model, ρ::Coefficient)
"""
struct ElastodynamicsModel{RHSModel #= <: AbstractMaterialModel =#, FM, CoefficientType}
    displacement_symbol::Symbol
    velocity_symbol::Symbol
    material_model::RHSModel
    face_models::FM
    ρ::CoefficientType
end

include("solid/energies.jl")
include("solid/contraction.jl")
include("solid/active.jl")
include("solid/materials.jl")
include("solid/elements.jl")
