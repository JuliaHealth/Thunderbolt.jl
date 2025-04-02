"""
This cache allows to combine multiple elements over the same volume.
If surface caches are passed they are handled properly. This requred dispatching
    is_facet_in_cache(facet::FacetIndex, geometry_cache, my_cache::MyCacheType)
"""
struct CompositeVolumetricElementCache{CacheTupleType <: Tuple} <: AbstractVolumetricElementCache
    inner_caches::CacheTupleType
end
# Main entry point for bilinear operators
assemble_element!(Kₑ::AbstractMatrix, cell::CellCache, element_cache::CompositeVolumetricElementCache, time) = assemble_composite_element!(Kₑ, cell, element_cache.inner_caches, time)
@unroll function assemble_composite_element!(Kₑ::AbstractMatrix, cell::CellCache, inner_caches::CacheTupleType, time) where CacheTupleType <: Tuple
    @unroll for inner_cache ∈ inner_caches
        assemble_element!(Kₑ, cell, inner_cache, time)
    end
end
# Update element matrix in nonlinear operators
assemble_element!(Kₑ::AbstractMatrix, uₑ::AbstractVector, cell::CellCache, element_cache::CompositeVolumetricElementCache, time) = assemble_composite_element!(Kₑ, uₑ, cell, element_cache.inner_caches, time)
@unroll function assemble_composite_element!(Kₑ::AbstractMatrix, uₑ::AbstractVector, cell::CellCache, inner_caches::CacheTupleType, time) where CacheTupleType <: Tuple
    @unroll for inner_cache ∈ inner_caches
        assemble_element!(Kₑ, uₑ, cell, inner_cache, time)
    end
end
# Update element matrix and residual in nonlinear operators
assemble_element!(Kₑ::AbstractMatrix, residualₑ::AbstractVector, uₑ::AbstractVector, cell::CellCache, element_cache::CompositeVolumetricElementCache, time) = assemble_composite_element!(Kₑ, residualₑ, uₑ, cell, element_cache.inner_caches, time)
@unroll function assemble_composite_element!(Kₑ::AbstractMatrix, residualₑ::AbstractVector, uₑ::AbstractVector, cell::CellCache, inner_caches::CacheTupleType, time) where CacheTupleType <: Tuple
    @unroll for inner_cache ∈ inner_caches
        assemble_element!(Kₑ, residualₑ, uₑ, cell, inner_cache, time)
    end
end
# Update residual in nonlinear operators
assemble_element!(residualₑ::AbstractVector, uₑ::AbstractVector, cell::CellCache, element_cache::CompositeVolumetricElementCache, time) = assemble_composite_element!(residualₑ, uₑ, cell, element_cache.inner_caches, time)
@unroll function assemble_composite_element!(residualₑ::AbstractVector, uₑ::AbstractVector, cell::CellCache, inner_caches::CacheTupleType, time) where CacheTupleType <: Tuple
    @unroll for inner_cache ∈ inner_caches
        assemble_element!(residualₑ, uₑ, cell, inner_cache, time)
    end
end

# If we compose a face cache into an element cache, then we loop over the faces of the elements and try to assemble
function assemble_element!(Kₑ::AbstractMatrix, uₑ::AbstractVector, cell::CellCache, facet_cache::AbstractSurfaceElementCache, time)
    for local_facet_index ∈ 1:nfacets(cell)
        if is_facet_in_cache(FacetIndex(cellid(cell), local_facet_index), cell, facet_cache)
            assemble_face!(Kₑ, uₑ, cell, local_facet_index, facet_cache, time)
        end
    end
end
function assemble_element!(Kₑ::AbstractMatrix, residualₑ::AbstractVector, uₑ::AbstractVector, cell::CellCache, facet_cache::AbstractSurfaceElementCache, time)
    for local_facet_index ∈ 1:nfacets(cell)
        if is_facet_in_cache(FacetIndex(cellid(cell), local_facet_index), cell, facet_cache)
            assemble_face!(Kₑ, residualₑ, uₑ, cell, local_facet_index, facet_cache, time)
        end
    end
end
function assemble_element!(residualₑ::AbstractVector, uₑ::AbstractVector, cell::CellCache, facet_cache::AbstractSurfaceElementCache, time)
    for local_facet_index ∈ 1:nfacets(cell)
        if is_facet_in_cache(FacetIndex(cellid(cell), local_facet_index), cell, facet_cache)
            assemble_face!(residualₑ, uₑ, cell, local_facet_index, facet_cache, time)
        end
    end
end



"""
This cache allows to combine multiple elements over the same surface.
"""
struct CompositeSurfaceElementCache{CacheTupleType <: Tuple} <: AbstractSurfaceElementCache
    inner_caches::CacheTupleType
end
# Main entry point for bilinear operators
assemble_face!(Kₑ::AbstractMatrix, cell::CellCache, local_facet_index::Int, surface_cache::CompositeSurfaceElementCache, time) = assemble_composite_face!(Kₑ, cell, surface_cache.inner_caches, local_facet_index, time)
@unroll function assemble_composite_face!(Kₑ::AbstractMatrix, cell::CellCache, local_facet_index::Int, inner_caches::CacheTupleType, time) where CacheTupleType <: Tuple
    @unroll for inner_cache ∈ inner_caches
        assemble_face!(Kₑ, cell, local_facet_index, inner_cache, time)
    end
end
# Update element matrix in nonlinear operators
assemble_face!(Kₑ::AbstractMatrix, uₑ::AbstractVector, cell::CellCache, local_facet_index::Int, surface_cache::CompositeSurfaceElementCache, time) = assemble_composite_face!(Kₑ, uₑ, cell, local_facet_index, surface_cache.inner_caches, time)
@unroll function assemble_composite_face!(Kₑ::AbstractMatrix, uₑ::AbstractVector, cell::CellCache, local_facet_index::Int, inner_caches::CacheTupleType, time) where CacheTupleType <: Tuple
    @unroll for inner_cache ∈ inner_caches
        assemble_face!(Kₑ, uₑ, cell, local_facet_index, inner_cache, time)
    end
end
# Update element matrix and residual in nonlinear operators
assemble_face!(Kₑ::AbstractMatrix, residualₑ::AbstractVector, uₑ::AbstractVector, cell::CellCache, local_facet_index::Int, surface_cache::CompositeSurfaceElementCache, time) = assemble_composite_face!(Kₑ, residualₑ, uₑ, cell, local_facet_index, surface_cache.inner_caches, time)
@unroll function assemble_composite_face!(Kₑ::AbstractMatrix, residualₑ::AbstractVector, uₑ::AbstractVector, cell::CellCache, local_facet_index::Int, inner_caches::CacheTupleType, time) where CacheTupleType <: Tuple
    @unroll for inner_cache ∈ inner_caches
        assemble_face!(Kₑ, residualₑ, uₑ, cell, local_facet_index, inner_cache, time)
    end
end
# Update residual in nonlinear operators
assemble_face!(residualₑ::AbstractVector, uₑ::AbstractVector, cell::CellCache, local_facet_index::Int, surface_cache::CompositeSurfaceElementCache, time) = assemble_composite_face!(residualₑ, uₑ, cell, local_facet_index, surface_cache.inner_caches, time)
@unroll function assemble_composite_face!(residualₑ::AbstractVector, uₑ::AbstractVector, cell::CellCache, local_facet_index::Int, inner_caches::CacheTupleType, time) where CacheTupleType <: Tuple
    @unroll for inner_cache ∈ inner_caches
        assemble_face!(residualₑ, uₑ, cell, local_facet_index, inner_cache, time)
    end
end

# If we compose a face cache into an element cache, then we loop over the faces of the elements and try to assemble
# Update element matrix in nonlinear operators
assemble_element!(Kₑ::AbstractMatrix, uₑ::AbstractVector, cell::CellCache, surface_cache::CompositeSurfaceElementCache, time) = assemble_composite_element!(Kₑ, uₑ, cell, surface_cache.inner_caches, time)
# Update element matrix and residual in nonlinear operators
assemble_element!(Kₑ::AbstractMatrix, residualₑ::AbstractVector, uₑ::AbstractVector, cell::CellCache, surface_cache::CompositeSurfaceElementCache, time) = assemble_composite_element!(Kₑ, residualₑ, uₑ, cell, surface_cache.inner_caches, time)
# Update residual in nonlinear operators
assemble_element!(residualₑ::AbstractVector, uₑ::AbstractVector, cell::CellCache, surface_cache::CompositeSurfaceElementCache, time) = assemble_composite_element!(residualₑ, uₑ, cell, surface_cache.inner_caches, time)


"""
This cache allows to combine multiple elements over the same interface.
"""
struct CompositeInterfaceElementCache{CacheTupleType <: Tuple} <: AbstractInterfaceElementCache
    inner_caches::CacheTupleType
end
# Main entry point for bilinear operators
assemble_interface!(Kₑ::AbstractMatrix, interface, interface_cache::CompositeInterfaceElementCache, time) = assemble_composite_interface!(Kₑ, interface, interface_cache.inner_caches, time)
@unroll function assemble_composite_interface!(Kₑ::AbstractMatrix, interface, inner_caches::CacheTupleType, time) where CacheTupleType <: Tuple
    @unroll for inner_cache ∈ inner_caches
        assemble_interface!(Kₑ, interface, inner_cache, time)
    end
end
# Update element matrix in nonlinear operators
assemble_interface!(Kₑ::AbstractMatrix, uₑ::AbstractVector, interface, interface_cache::CompositeInterfaceElementCache, time) = assemble_composite_interface!(Kₑ, uₑ, interface, interface_cache.inner_caches, time)
@unroll function assemble_composite_interface!(Kₑ::AbstractMatrix, uₑ::AbstractVector, interface, inner_caches::CacheTupleType, time) where CacheTupleType <: Tuple
    @unroll for inner_cache ∈ inner_caches
        assemble_interface!(Kₑ, uₑ, interface, inner_cache, time)
    end
end
# Update element matrix and residual in nonlinear operators
assemble_interface!(Kₑ::AbstractMatrix, residualₑ::AbstractVector, uₑ::AbstractVector, interface, interface_cache::CompositeInterfaceElementCache, time) = assemble_composite_interface!(Kₑ, residualₑ, uₑ, interface, interface_cache.inner_caches, time)
@unroll function assemble_composite_interface!(Kₑ::AbstractMatrix, residualₑ::AbstractVector, uₑ::AbstractVector, interface, inner_caches::CacheTupleType, time) where CacheTupleType <: Tuple
    @unroll for inner_cache ∈ inner_caches
        assemble_interface!(Kₑ, residualₑ, uₑ, interface, inner_cache, time)
    end
end
# Update residual in nonlinear operators
assemble_interface!(residualₑ::AbstractVector, uₑ::AbstractVector, interface, interface_cache::CompositeInterfaceElementCache, time) = assemble_composite_interface!(Kₑ, interface, interface_cache.inner_caches, time)
@unroll function assemble_composite_interface!(residualₑ::AbstractVector, uₑ::AbstractVector, interface, inner_caches::CacheTupleType, time) where CacheTupleType <: Tuple
    @unroll for inner_cache ∈ inner_caches
        assemble_interface!(residualₑ, uₑ, interface, inner_cache, time)
    end
end
