
"""
TODO document me!
"""
# Is this really what we want? Or should the face integrals be directly merged into the mechanical model?
# Does this abstraction layer have another benefit?
struct StructuralModel{MM, FM}
    mechanical_model::MM
    face_models::FM
end


"""
    StructuralElementCache

TODO document me!
"""
# TODO rename contraction_model_cache
struct StructuralElementCache{M, CMCache, CV}
    constitutive_model::M
    contraction_model_cache::CMCache
    cv::CV
end

# TODO how to control dispatch on required input for the material routin?
# TODO finer granularity on the dispatch here. depending on the evolution law of the internal variable this routine looks slightly different.
function assemble_element!(Kₑ::Matrix, residualₑ, uₑ, geometry_cache, element_cache::StructuralElementCache, time)
    @unpack constitutive_model, contraction_model_cache, cv = element_cache
    ndofs = getnbasefunctions(cv)

    reinit!(cv, geometry_cache)

    @inbounds for qp ∈ QuadratureIterator(cv)
        dΩ = getdetJdV(cv, qp)

        # Compute deformation gradient F
        ∇u = function_gradient(cv, qp, uₑ)
        F = one(∇u) + ∇u

        # Compute stress and tangent
        contraction_state = state(contraction_model_cache, geometry_cache, qp, time)
        P, ∂P∂F = material_routine(constitutive_model, F, contraction_state, geometry_cache, qp, time)

        # Loop over test functions
        for i in 1:ndofs
            ∇δui = shape_gradient(cv, qp, i)

            # Add contribution to the residual from this test function
            residualₑ[i] += ∇δui ⊡ P * dΩ

            ∇δui∂P∂F = ∇δui ⊡ ∂P∂F # Hoisted computation
            for j in 1:ndofs
                ∇δuj = shape_gradient(cv, qp, j)
                # Add contribution to the tangent
                Kₑ[i, j] += ( ∇δui∂P∂F ⊡ ∇δuj ) * dΩ
            end
        end
    end
end
