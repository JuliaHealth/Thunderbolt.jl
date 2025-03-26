
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

include("solid/energies.jl")
include("solid/contraction.jl")
include("solid/active.jl")
include("solid/materials.jl")
include("solid/elements.jl")
