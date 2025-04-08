@doc raw"""
    LinearIntegrator

Represents the integrand a the linear form over some function space.
"""
struct LinearIntegrator{IntegrandType, QRC <: Union{<:QuadratureRuleCollection, Nothing}}
    integrand::IntegrandType
    qrc::QRC
end

function setup_element_cache(i::LinearIntegrator, sdh::SubDofHandler)
    return setup_element_cache(i.integrand, getquadraturerule(i.qrc, sdh), sdh)
end
