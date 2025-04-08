abstract type AbstractAssemblyStrategy end

"""
    SequentialAssemblyStrategy()
"""
struct SequentialAssemblyStrategy{DeviceType} <: AbstractAssemblyStrategy
    device::DeviceType
end

struct SequentialAssemblyStrategyCache{DeviceCacheType}
    # Scratch for the device to store its data
    device_cache::DeviceCacheType
end


"""
    PerColorAssemblyStrategy(chunksize, coloralg)
"""
struct PerColorAssemblyStrategy{DeviceType} <: AbstractAssemblyStrategy
    device::DeviceType
    # coloralg::Symbol # TODO
end

struct PerColorAssemblyStrategyCache{DeviceCacheType, ColorCacheType}
    # Scratch for the device to store its data
    device_cache::DeviceCacheType
    # Everythign related to the coloring is stored here
    color_cache::ColorCacheType
end

function create_dh_coloring(dh::DofHandler; alg = Ferrite.ColoringAlgorithm.WorkStream)
    grid = get_grid(dh)
    return [Ferrite.create_coloring(grid, sdh.cellset; alg) for sdh in dh.subdofhandlers]
end

"""
    ElementAssemblyStrategy
"""
struct ElementAssemblyStrategy{DeviceType} <: AbstractAssemblyStrategy
    device::DeviceType
end

struct ElementAssemblyStrategyCache{DeviceCacheType, EADataType}
    # Scratch for the device to store its data
    device_cache::DeviceCacheType
    # Everythign related to the coloring is stored here
    ea_data::EADataType
end
