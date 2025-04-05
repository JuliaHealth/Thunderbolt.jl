abstract type AbstractAssemblyStrategy end

"""
    SequentialAssemblyStrategy()
"""
struct SequentialAssemblyStrategy <: AbstractAssemblyStrategy
    device::AbstractDevice
end

struct SequentialAssemblyStrategyCache{DeviceCacheType}
    # Scratch for the device to store its data
    device_cache::DeviceCacheType
end


"""
    PerColorAssemblyStrategy(chunksize, coloralg)
"""
struct PerColorAssemblyStrategy <: AbstractAssemblyStrategy
    device::AbstractDevice
    coloralg::Symbol
end

struct PerColorAssemblyStrategyCache{DeviceCacheType, ColorCacheType}
    # Scratch for the device to store its data
    device_cache::DeviceCacheType
    # Everythign related to the coloring is stored here
    color_cache::ColorCacheType
end



"""
    ElementAssemblyStrategy
"""
struct ElementAssemblyStrategy <: AbstractAssemblyStrategy
    device::AbstractDevice
end

struct ElementAssemblyStrategyCache{DeviceCacheType, EADataType}
    # Scratch for the device to store its data
    device_cache::DeviceCacheType
    # Everythign related to the coloring is stored here
    ea_data::EADataType
end
