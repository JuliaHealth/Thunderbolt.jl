# in subdofhandlers and in grid there are some vectors that incorporate many objects of different types but they 
# only share the same abstract type. 
# one way to solve this (not the optimal way memory wise) is to convert the vector to a vector of Union types if
# there are multiple concrete types in the vector , otherwise convert the vector to a vector of the concrete type.
function convert_vec_to_concrete(vec::Vector)
    # Get all unique concrete types in the vector
    Ts = unique(typeof.(vec))

    if length(Ts) == 1
        # All elements are the same concrete type
        T = Ts[1]
        return collect(T, vec)
    else
        # Create a union of all observed concrete types
        U = Union{Ts...}
        return collect(U, vec)
    end
end


#######################
## Assembly Strategy ##
#######################
abstract type AbstractAssemblyStrategy end

# TODO decouple assembly strategy type from actual device type
# encompass the all the required data types that needs to be worked with on the GPU
struct CudaAssemblyStrategy <: AbstractAssemblyStrategy
    floattype::Type
    inttype::Type
end

floattype(strategy::CudaAssemblyStrategy) = strategy.floattype
inttype(strategy::CudaAssemblyStrategy) = strategy.inttype

CudaAssemblyStrategy() = CudaAssemblyStrategy(Float32, Int32)


##########################################
## Thunderbolt general objects adaption ##
##########################################
Adapt.@adapt_structure AnalyticalCoefficientCache
Adapt.@adapt_structure CartesianCoordinateSystemCache
Adapt.@adapt_structure FieldCoefficientCache
Adapt.@adapt_structure AnalyticalCoefficientElementCache
Adapt.@adapt_structure SpatiallyHomogeneousDataField

function Adapt.adapt_structure(::AbstractAssemblyStrategy, dh::DofHandler)
    error("GPU specific implementation for `adapt_structure(to,dh::DofHandler)` is not implemented yet")
end

function deep_adapt(::AbstractAssemblyStrategy, dh::Thunderbolt.FerriteUtils.DeviceDofHandlerData)
    error("GPU specific implementation for `deep_adapt(strategy::CudaAssemblyStrategy, dh::DeviceDofHandlerData)` is not implemented yet")
end

function Adapt.adapt_structure(::AbstractAssemblyStrategy, element_cache::AnalyticalCoefficientElementCache)
    error("GPU specific implementation for `adapt_structure(to, element_cache::AnalyticalCoefficientElementCache)` is not implemented yet")
end

function Adapt.adapt_structure(::AbstractAssemblyStrategy, cysc::FieldCoefficientCache)
    error("GPU specific implementation for `adapt_structure(to, cysc::FieldCoefficientCache)` is not implemented yet")
end

function Adapt.adapt_structure(::AbstractAssemblyStrategy, sphdf::SpatiallyHomogeneousDataField)
    error("GPU specific implementation for `adapt_structure(to, cysc::FieldCoefficientCache)` is not implemented yet")
end
