
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

get_field_variable_names(model::QuasiStaticModel) = (model.displacement_symbol,)

get_volumetric_weak_form_names(model::QuasiStaticModel) = (model.displacement_symbol,)

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

@doc raw"""
    ElastodynamicsModel(displacement_sym, velocity_symbol, material_model, facet_models, ρ)

Balance of momentum including the inertia term,
```math
\int_\Omega \rho\, \delta u \cdot \ddot{u} \,\mathrm{d}\Omega
+ \int_\Omega \mathrm{grad}(\delta u) : P(u) \,\mathrm{d}\Omega = \text{(facet terms)} ,
```
i.e. the [`QuasiStaticModel`](@ref) with a mass term on top.

!!! note "The velocity is not a degree of freedom"
    `velocity_symbol` names the velocity for output only. Time integrators for this model
    (see [`NewmarkSolver`](@ref)) discretize in displacement form: the global unknown is the
    displacement field alone, and velocity and acceleration are reconstructed by the scheme and
    kept in the solver cache. A formulation carrying the velocity as a genuine unknown would be a
    different model with two field variables.
"""
# Not an `AbstractMaterialModel`, and it never reaches the material path: `semidiscretize` lowers it
# to a `QuasiStaticModel`, which is what the element caches are built from.
struct ElastodynamicsModel{MaterialModel#= <: AbstractMaterialModel =#, FM, CoefficientType}
    displacement_symbol::Symbol
    velocity_symbol::Symbol
    material_model::MaterialModel
    facet_models::FM
    ρ::CoefficientType
end

ElastodynamicsModel(displacement_symbol, velocity_symbol, material_model, ρ) =
    ElastodynamicsModel(displacement_symbol, velocity_symbol, material_model, (), ρ)

# Only the displacement is discretized, hence a single field variable -- see the note above.
get_field_variable_names(model::ElastodynamicsModel) = (model.displacement_symbol,)

get_volumetric_weak_form_names(model::ElastodynamicsModel) = (model.displacement_symbol,)

structural_displacement_symbol(model::ElastodynamicsModel) = model.displacement_symbol

include("solid/energies.jl")
include("solid/contraction.jl")
include("solid/active.jl")
include("solid/materials.jl")
include("solid/elements.jl")
