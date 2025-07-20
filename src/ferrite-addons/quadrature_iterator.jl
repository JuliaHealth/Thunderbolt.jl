"""
    QuadraturePoint{dim, T}

A simple helper to carry quadrature point information.
"""
struct QuadraturePoint{dim, T}
    i::Int
    ξ::Vec{dim, T}
end

"""
    QuadratureIterator(::QuadratureRule)
    QuadratureIterator(::FacetQuadratureRule, local_facet_idx::Int)
    QuadratureIterator(::CellValues)
    QuadratureIterator(::FacetValues)

A helper to loop over the quadrature points in some rule or cache with type [`QuadraturePoint`](@ref).
"""
struct QuadratureIterator{QR<:QuadratureRule}
    qr::QR
end
QuadratureIterator(fqr::FacetQuadratureRule, local_facet_idx::Int) = QuadratureIterator(fqr.facet_rules[local_facet_idx])
QuadratureIterator(cv::CellValues) = QuadratureIterator(cv.qr)
QuadratureIterator(fv::FacetValues) = QuadratureIterator(fv.fqr.facet_rules[fv.current_facet[]])

function Base.iterate(iterator::QuadratureIterator, i = 1)
    i > getnquadpoints(iterator.qr) && return nothing
    return (QuadraturePoint(i,Ferrite.getpoints(iterator.qr)[i]), i + 1)
end
Base.eltype(::Type{<:QuadraturePoint{<:QuadratureRule{<:AbstractRefShape, dim, T}}}) where {dim, T} = QuadraturePoint{dim, T}
Base.length(iterator::QuadratureIterator) = length(Ferrite.getnquadpoints(iterator.qr))

Ferrite.spatial_coordinate(fe_v::Ferrite.AbstractValues, qp::QuadraturePoint, x::AbstractVector{<:Vec}) = Ferrite.spatial_coordinate(fe_v, qp.i, x)

Ferrite.getdetJdV(cv::CellValues, qp::QuadraturePoint) = Ferrite.getdetJdV(cv, qp.i)
Ferrite.shape_value(cv::CellValues, qp::QuadraturePoint, base_fun_idx::Int) = Ferrite.shape_value(cv, qp.i, base_fun_idx)
Ferrite.shape_gradient(cv::CellValues, qp::QuadraturePoint, base_fun_idx::Int) = Ferrite.shape_gradient(cv, qp.i, base_fun_idx)
Ferrite.function_value(cv::CellValues, qp::QuadraturePoint, ue) = Ferrite.function_value(cv, qp.i, ue)
Ferrite.function_gradient(cv::CellValues, qp::QuadraturePoint, ue) = Ferrite.function_gradient(cv, qp.i, ue)

Ferrite.getdetJdV(fv::FacetValues, qp::QuadraturePoint) = Ferrite.getdetJdV(fv, qp.i)
Ferrite.shape_value(fv::FacetValues, qp::QuadraturePoint, base_fun_idx::Int) = Ferrite.shape_value(fv, qp.i, base_fun_idx)
Ferrite.shape_gradient(fv::FacetValues, qp::QuadraturePoint, base_fun_idx::Int) = Ferrite.shape_gradient(fv, qp.i, base_fun_idx)
Ferrite.function_value(fv::FacetValues, qp::QuadraturePoint, ue) = Ferrite.function_value(fv, qp.i, ue)
Ferrite.function_gradient(fv::FacetValues, qp::QuadraturePoint, ue) = Ferrite.function_gradient(fv, qp.i, ue)
Ferrite.getnormal(fv::FacetValues, qp::QuadraturePoint) = Ferrite.getnormal(fv, qp.i)

Ferrite.shape_value(cv::FerriteUtils.StaticInterpolationValues, qp::QuadraturePoint, base_fun_idx::Int) = Ferrite.shape_value(cv, qp.i, base_fun_idx)
Ferrite.function_value(cv::FerriteUtils.StaticInterpolationValues, qp::QuadraturePoint, ue) = Ferrite.function_value(cv, qp.i, ue)
