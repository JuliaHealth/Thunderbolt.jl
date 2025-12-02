
"""
    QuasiStaticModel(displacement_sym, mechanical_model, facet_models)

A generic model for quasi-static mechanical problems.
"""
struct QuasiStaticModel{MM #= <: AbstractMaterialModel =#, FM}
    displacement_symbol::Symbol
    material_model::MM
    facet_models::FM
end

get_field_variable_names(model::QuasiStaticModel) = (model.displacement_symbol, )

"""
    ElastodynamicsModel(displacement_sym, velocity_symbol, material_model::AbstractMaterialModel, facet_model, ρ::Coefficient)
"""
struct ElastodynamicsModel{RHSModel #= <: AbstractMaterialModel =#, FM, CoefficientType}
    displacement_symbol::Symbol
    velocity_symbol::Symbol
    material_model::RHSModel
    facet_models::FM
    ρ::CoefficientType
end

include("solid/energies.jl")
include("solid/contraction.jl")
include("solid/active.jl")
include("solid/materials.jl")
include("solid/elements.jl")
