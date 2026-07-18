
"""
    QuasiStaticModel(displacement_sym, mechanical_model, facet_models)

A generic model for quasi-static mechanical problems.
"""
struct QuasiStaticModel{MM#= <: AbstractMaterialModel =#, FM}
    displacement_symbol::Symbol
    material_model::MM
    facet_models::FM
end

QuasiStaticModel(displacement_symbol, material_model) =
    QuasiStaticModel(displacement_symbol, material_model, ())

get_field_variable_names(model::QuasiStaticModel) = [model.displacement_symbol]

get_volumetric_weak_form_names(model::QuasiStaticModel) = [model.displacement_symbol]

"""
    structural_displacement_symbol(model)

Returns the displacement symbol of a (possibly multi-domain) structural model, i.e. either a single
[`QuasiStaticModel`](@ref) or a `Dict{String, <:QuasiStaticModel}` describing one model per subdomain.
Errors if a multi-domain model does not agree on a single displacement symbol across all subdomains.
"""
structural_displacement_symbol(model::QuasiStaticModel) = model.displacement_symbol
function structural_displacement_symbol(models::Dict{String, <:QuasiStaticModel})
    symbols = Set(model.displacement_symbol for model in values(models))
    @assert length(symbols) == 1 "All structural models in a domain split must share the same displacement symbol, got $(symbols)."
    return first(symbols)
end

"""
    ElastodynamicsModel(displacement_sym, velocity_symbol, material_model::AbstractMaterialModel, facet_model, ρ::Coefficient)
"""
struct ElastodynamicsModel{RHSModel#= <: AbstractMaterialModel =#, FM, CoefficientType}
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
