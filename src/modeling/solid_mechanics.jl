
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

@doc raw"""
    ElastodynamicsModel(displacement_sym, velocity_symbol, material_model, facet_models, ρ)

Balance of momentum including the inertia term,
```math
\int_\Omega \rho\, \delta u \cdot \ddot{u} \,\mathrm{d}\Omega
+ \int_\Omega \mathrm{grad}(\delta u) : P(u) \,\mathrm{d}\Omega = \text{(facet terms)} ,
```
i.e. the [`QuasiStaticModel`](@ref) with a mass term on top.

!!! note "The velocity is a field, but not a Newton unknown"
    `velocity_symbol` names a genuine field: it shares the displacement's interpolation, occupies a
    block of the solution vector, and can be written out by name. What it is *not* is an unknown of
    the nonlinear solve, and a `Dirichlet` condition on it is refused -- a scheme that reconstructs
    the velocity from the displacement would overwrite whatever the constraint wrote. Time integrators for this model (see
    [`NewmarkSolver`](@ref)) discretize in displacement form, so a step solves for the displacement
    and reconstructs the velocity from it. The acceleration is not stored either way: it follows from
    the balance of momentum.
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

# The velocity is part of the state, so it is a field variable. The *weak form* below names the
# displacement alone: the internal forces have no velocity equation, and the problem the nonlinear
# solver is handed is posed on the displacement.
get_field_variable_names(model::ElastodynamicsModel) =
    (model.displacement_symbol, model.velocity_symbol)

get_volumetric_weak_form_names(model::ElastodynamicsModel) = (model.displacement_symbol,)

structural_displacement_symbol(model::ElastodynamicsModel) = model.displacement_symbol

# Stated once for both model families: a domain split must agree on one displacement field. The generic
# `_shared_symbol_or_error` does the work, so this stays one line rather than a second implementation.
structural_displacement_symbol(
    models::Dict{String, <:Union{QuasiStaticModel, ElastodynamicsModel}},
) = _shared_symbol_or_error(models, model -> model.displacement_symbol, "displacement")

include("solid/energies.jl")
include("solid/contraction.jl")
include("solid/active.jl")
include("solid/materials.jl")
include("solid/elements.jl")
