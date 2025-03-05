"""
Supertype for all caches to integrate over volumes.

Interface:

    setup_element_cache(model, qr, sdh)

"""
abstract type AbstractVolumetricElementCache end

# setup_element_cache(model, qr, sdh) = setup_element_cache(model, qr, Ferrite.getfieldinterpolation(sdh,1), sdh) # FIXME

@doc raw"""
    assemble_element!(Kₑ::AbstractMatrix, cell::CellCache, element_cache::AbstractVolumetricElementCache, time)
Main entry point for bilinear operators

    assemble_element!(Kₑ::AbstractMatrix, uₑ::AbstractVector, cell::CellCache, element_cache::AbstractVolumetricElementCache, time)
Update element matrix in nonlinear operators

    assemble_element!(Kₑ::AbstractMatrix, residualₑ::AbstractVector, uₑ::AbstractVector, cell::CellCache, element_cache::AbstractVolumetricElementCache, time)
Update element matrix and residual in nonlinear operators

    assemble_element!(residualₑ::AbstractVector, uₑ::AbstractVector, cell::CellCache, element_cache::AbstractVolumetricElementCache, time)
Update residual in nonlinear operators

The notation is as follows.
* $K_e$ the element stiffness matrix
* $u_e$ the element unknowns
* $residual_e$ the element residual
"""
assemble_element!


"""
    Utility to execute noop assembly.
"""
struct EmptyVolumetricElementCache <: AbstractVolumetricElementCache end
# Main entry point for bilinear operators
assemble_element!(Kₑ::AbstractMatrix, cell::CellCache, element_cache::EmptyVolumetricElementCache, time) = nothing
# Update element matrix in nonlinear operators
assemble_element!(Kₑ::AbstractMatrix, uₑ::AbstractVector, cell::CellCache, element_cache::EmptyVolumetricElementCache, time) = nothing
# Update element matrix and residual in nonlinear operators
assemble_element!(Kₑ::AbstractMatrix, residualₑ::AbstractVector, uₑ::AbstractVector, cell::CellCache, element_cache::EmptyVolumetricElementCache, time) = nothing
# Update residual in nonlinear operators
assemble_element!(residualₑ::AbstractVector, uₑ::AbstractVector, cell::CellCache, element_cache::EmptyVolumetricElementCache, time) = nothing



"""
Supertype for all caches to integrate over surfaces.

Interface:

    setup_boundary_cache(model, qr, sdh)

"""
abstract type AbstractSurfaceElementCache end

# setup_boundary_cache(model, qr, sdh) = setup_boundary_cache(model, qr, Ferrite.getfieldinterpolation(sdh,1), sdh) # FIXME

@doc raw"""
    assemble_face!(Kₑ::AbstractMatrix, cell::CellCache, face_cache::AbstractSurfaceElementCache, time)
Main entry point for bilinear operators

    assemble_face!(Kₑ::AbstractMatrix, uₑ::AbstractVector, cell::CellCache, face_cache::AbstractSurfaceElementCache, time)
Update face matrix in nonlinear operators

    assemble_face!(Kₑ::AbstractMatrix, residualₑ::AbstractVector, uₑ::AbstractVector, cell::CellCache, face_cache::AbstractSurfaceElementCache, time)
Update face matrix and residual in nonlinear operators

    assemble_face!(residualₑ::AbstractVector, uₑ::AbstractVector, cell::CellCache, face_cache::AbstractSurfaceElementCache, time)
Update residual in nonlinear operators

The notation is as follows.
* $K_e$ the element stiffness matrix
* $u_e$ the element unknowns
* $residual_e$ the element residual
"""
assemble_face!


"""
    Utility to execute noop assembly.
"""
struct EmptySurfaceElementCache <: AbstractSurfaceElementCache end
# Update element matrix in nonlinear operators
assemble_face!(Kₑ::AbstractMatrix, uₑ::AbstractVector, cell::CellCache, local_face_index::Int, face_caches::EmptySurfaceElementCache, time)    = nothing
assemble_element!(Kₑ::AbstractMatrix, uₑ::AbstractVector, cell::CellCache, local_face_index::Int, face_caches::EmptySurfaceElementCache, time) = nothing
# Update element matrix and residual in nonlinear operators
assemble_face!(Kₑ::AbstractMatrix, residualₑ::AbstractVector, uₑ::AbstractVector, cell, local_face_index::Int, face_caches::EmptySurfaceElementCache, time)    = nothing
assemble_element!(Kₑ::AbstractMatrix, residualₑ::AbstractVector, uₑ::AbstractVector, cell, local_face_index::Int, face_caches::EmptySurfaceElementCache, time) = nothing
# Update residual in nonlinear operators
assemble_face!(residualₑ::AbstractVector, uₑ::AbstractVector, cell, local_face_index::Int, face_caches::EmptySurfaceElementCache, time)    = nothing
assemble_element!(residualₑ::AbstractVector, uₑ::AbstractVector, cell, local_face_index::Int, face_caches::EmptySurfaceElementCache, time) = nothing
@inline is_facet_in_cache(::FacetIndex, cell, ::EmptySurfaceElementCache) = false

"""
Supertype for all caches to integrate over interfaces.

Interface:

    setup_interface_cache(model, qr, ip, sdh)

"""
abstract type AbstractInterfaceElementCache end

@doc raw"""
    assemble_interface!(Kₑ::AbstractMatrix, cell::CellCache, face_cache::AbstractSurfaceElementCache, time)
Main entry point for bilinear operators

    assemble_interface!(Kₑ::AbstractMatrix, uₑ::AbstractVector, cell::CellCache, face_cache::AbstractSurfaceElementCache, time)
Update face matrix in nonlinear operators

    assemble_interface!(Kₑ::AbstractMatrix, residualₑ::AbstractVector, uₑ::AbstractVector, cell::CellCache, face_cache::AbstractSurfaceElementCache, time)
Update face matrix and residual in nonlinear operators

    assemble_interface!(residualₑ::AbstractVector, uₑ::AbstractVector, cell::CellCache, face_cache::AbstractSurfaceElementCache, time)
Update residual in nonlinear operators

The notation is as follows.
* $K_e$ the element pair stiffness matrix
* $u_e$ the element pair unknowns
* $residual_e$ the element pair residual
"""
assemble_interface!


"""
Utility to execute noop assembly.
"""
struct EmptyInterfaceCache <: AbstractInterfaceElementCache end
# Update element matrix in nonlinear operators
assemble_interface!(Kₑ::AbstractMatrix, uₑ::AbstractVector, cell::CellCache, local_face_index::Int, face_caches::EmptyInterfaceCache, time)            = nothing
# Update element matrix and residual in nonlinear operators
assemble_interface!(Kₑ::AbstractMatrix, residualₑ::AbstractVector, uₑ::AbstractVector, cell, local_face_index::Int, face_caches::EmptyInterfaceCache, time) = nothing
# Update residual in nonlinear operators
assemble_interface!(residualₑ::AbstractVector, uₑ::AbstractVector, cell, local_face_index::Int, face_caches::EmptyInterfaceCache, time) = nothing

