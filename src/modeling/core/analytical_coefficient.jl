
"""
    AnalyticalCoefficient(f::Function, cs::CoordinateSystemCoefficient)

A coefficient given as the analytical function f(x,t) in the specified coordiante system.
"""
struct AnalyticalCoefficient{F<:Function, CSYS<:CoordinateSystemCoefficient}
    f::F
    coordinate_system_coefficient::CSYS
end

struct AnalyticalCoefficientCache{F<:Function, CSC}
    f::F
    coordinate_system_cache::CSC
end

function setup_coefficient_cache(coeff::AnalyticalCoefficient, qr::QuadratureRule, sdh::SubDofHandler)
    return AnalyticalCoefficientCache(
        coeff.f,
        setup_coefficient_cache(coeff.coordinate_system_coefficient, qr, sdh)
    )
end

duplicate_for_parallel(cache::AnalyticalCoefficientCache) = AnalyticalCoefficientCache(cache.f, duplicate_for_parallel(cache.coordinate_system_cache))

@inline function evaluate_coefficient(coeff::F, cell_cache::FerriteUtils.AnyCellCache, qp::QuadraturePoint{<:Any,T}, t) where {F <: AnalyticalCoefficientCache, T}
    x = evaluate_coefficient(coeff.coordinate_system_cache, cell_cache, qp, t)
    return coeff.f(x, t)
end

"""
    AnalyticalCoefficientElementCache(f(x,t)->..., nonzero_in_time_intervals, cellvalues)

Analytical coefficient described by a function in space and time.
Can be sparse in time.
"""
struct AnalyticalCoefficientElementCache{CoefficientCacheType <: AnalyticalCoefficientCache, T, VectorType <: AbstractVector{SVector{2,T}}, FEValueType}
    cc::CoefficientCacheType
    nonzero_intervals::VectorType
    cv::FEValueType
end
duplicate_for_parallel(ec::AnalyticalCoefficientElementCache) = AnalyticalCoefficientElementCache(duplicate_for_parallel(ec.cc), ec.nonzero_intervals, ec.cv)

@inline function assemble_element!(bₑ::AbstractVector, geometry_cache::FerriteUtils.AnyCellCache, element_cache::AnalyticalCoefficientElementCache, time)
    _assemble_element!(bₑ, geometry_cache, getcoordinates(geometry_cache), element_cache::AnalyticalCoefficientElementCache, time)
end

# We want this to be as fast as possible, so throw away everything unused
@inline function _assemble_element!(bₑ::AbstractVector, geometry_cache::CellCache, coords::AbstractVector{<:Vec{dim,T}}, element_cache::AnalyticalCoefficientElementCache, time) where {dim,T}
    @unpack cc, cv = element_cache
    n_geom_basefuncs = Ferrite.getngeobasefunctions(cv)
    @inbounds for (qpi, w) in pairs(Ferrite.getweights(cv.qr))
        # Compute dΩ
        mapping = Ferrite.calculate_mapping(cv.geo_mapping, qpi, coords)
        dΩ = Ferrite.calculate_detJ(Ferrite.getjacobian(mapping)) * w
        # Evaluate f
        fx = evaluate_coefficient(cc, geometry_cache, QuadraturePoint(qpi, cv.qr.points[qpi]), time)
        # Evaluate all basis functions
        @inbounds for j ∈ 1:getnbasefunctions(cv)
            δu = shape_value(cv, qpi, j)
            bₑ[j] += fx * δu * dΩ
        end
    end
end

@inline function _assemble_element!(bₑ, geometry_cache::FerriteUtils.DeviceCellCache, coords, element_cache::AnalyticalCoefficientElementCache, time) 
    @unpack cc, cv = element_cache
    for qv in FerriteUtils.QuadratureValuesIterator(cv, coords)
        dΩ = FerriteUtils.getdetJdV(qv)
        idx = qv.idx
        ξ = qv.ξ
        qp = QuadraturePoint(idx, ξ)
        @inbounds for j ∈ 1:getnbasefunctions(cv)
            fx =  evaluate_coefficient(cc, geometry_cache, qp, time) 
            δu = FerriteUtils.shape_value(qv, j)
            bₑ[j] += fx * δu * dΩ
        end
    end
end
