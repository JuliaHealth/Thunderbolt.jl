@doc raw"""
    NonlinearIntegrator

Represents the integrand a the nonlinear form over some function space.
"""
struct NonlinearIntegrator{
    VM,
    FM,
    SYMS <: Base.AbstractVecOrTuple{Symbol},
    QRC <: Union{<:QuadratureRuleCollection, Nothing},
    FQRC <: Union{<:FacetQuadratureRuleCollection, Nothing},
} <: FerriteOperators.AbstractCondensedNonlinearIntegrator
    volume_model::VM
    facet_model::FM
    syms::SYMS  # The symbols for all unknowns in the submodels.
    qrc::QRC
    fqrc::FQRC
end

function setup_element_cache(i::NonlinearIntegrator, sdh::SubDofHandler)
    return setup_element_cache(i.volume_model, getquadraturerule(i.qrc, sdh), sdh)
end

function setup_boundary_cache(i::NonlinearIntegrator, sdh::SubDofHandler)
    return setup_boundary_cache(i.facet_model, getquadraturerule(i.fqrc, sdh), sdh)
end

# `get_number_of_internal_dofs_per_element` dispatches on the *element cache*, since that is what
# determines how many condensed unknowns a cell carries; per cache type methods live next to the
# cache they describe. Subdomains carrying no volumetric model contribute none.
FerriteOperators.get_number_of_internal_dofs_per_element(
    integrator,
    ::FerriteOperators.EmptyVolumetricElementCache,
    sdh::SubDofHandler,
) = Iterators.repeated(0, length(sdh.cellset))
