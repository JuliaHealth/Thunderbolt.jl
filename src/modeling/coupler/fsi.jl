"""
Descriptor for which volume to couple with which variable for the constraint.
"""
struct ChamberVolumeCoupling{CVM}
    chamber_surface_setname::String
    chamber_volume_method::CVM
    lumped_volume_symbol::Union{Symbol, ModelingToolkit.Num}
    lumped_pressure_symbol::Union{Symbol, ModelingToolkit.Num}
end

"""
Enforce the constraints that
  chamber volume 3D (solid model) = chamber volume 0D (lumped circuit)
via Lagrange multiplied, where a surface pressure integral is introduced such that
  ∫  ∂Ωendo
Here `chamber_volume_method` is responsible to compute the 3D volume.

This approach has been proposed by [RegSalAfrFedDedQar:2022:cem](@citet).
"""
struct LumpedFluidSolidCoupler{CVM} <: AbstractCoupler
    chamber_couplings::Vector{ChamberVolumeCoupling{CVM}}
    displacement_symbol::Union{Symbol, ModelingToolkit.Num}
end

is_bidrectional(::LumpedFluidSolidCoupler) = true

"""
    Debug helper for FSI. Just keeps the chamber volume constant.
"""
struct ConstantChamberVolume
    volume::Float64
end

function volume_integral(x, d, F, N, method::ConstantChamberVolume)
    method.volume
end

"""
Chamber volume estimator as presented in [HirBasJagWilGee:2017:mcc](@cite).

Compute the chamber volume as a surface integral via the integral
   - ∫ (x + d) det(F) cof(F) N ∂Ωendo
where it is assumed that the chamber is convex, zero displacement in
apicobasal direction at the valvular plane occurs and the plane normal is aligned
with the z axis, where the origin is at z=0.
"""
struct Hirschvogel2017SurrogateVolume end

function volume_integral(x::Vec, d::Vec, F::Tensor, N::Vec, method::Hirschvogel2017SurrogateVolume)
    val = det(F) * (x + d) ⋅ inv(transpose(F)) ⋅ N
    return -val
end

"""
Chamber volume contribution for the 3D-0D constraint
    ∫ V³ᴰ(u) ∂Ω - V⁰ᴰ(c)
where u are the unkowns in the 3D problem and c the unkowns in the 0D problem.

Pressure contribution (i.e. variation w.r.t. p) for the term
    ∫ p n(u) δu ∂Ω
 [= ∫ p J(u) F(u)^-T n₀ δu ∂Ω₀]
where p is the unknown chamber pressure and u contains the unknown deformation field.
"""
struct Pressure3D0DVolumeCouplerIntegrator <: AbstractNonlinearIntegrator
    fqrc::FacetQuadratureRuleCollection
    displacement_symbol::Symbol
    pressure_symbol::Symbol
    # boundary_name::String
    facets::OrderedSet
    volume_method
end

@concrete struct VolumeTimeParameters
    V⁰ᴰs
    time
end

@concrete struct Pressure3D0DVolumeCouplerCache <: AbstractSurfaceElementCache
    fv
    displacement_range
    pressure_index
    facets
    volume_method
end

function FerriteOperators.setup_boundary_cache(model::Pressure3D0DVolumeCouplerIntegrator, sdh)
    qr     = getquadraturerule(model.fqrc, sdh)
    ip     = Ferrite.getfieldinterpolation(sdh, model.displacement_symbol)
    ip_geo = geometric_subdomain_interpolation(sdh)
    fv     = FacetValues(qr, ip, ip_geo)

    displacement_range = dof_range(sdh, model.displacement_symbol)

    # This is the non-local coupling
    # @assert lenght(sdh_chamber.cellset) == 1 "0D subdomain has more than one cell ($(lenght(sdh_chamber.cellset)))"
    # pressure_symbol = model.pressure_symbol
    # pressure_dofs = dof_range(sdh, pressure_symbol)
    # @assert lenght(pressure_dofs) == 1 "Pressure ,,$(pressure_symbol)'' is associated with more than one dof ($(lenght(pressure_dofs)))"

    all_facets = model.facets #getfacetset(get_grid(sdh.dh), model.boundary_name)
    pressure_dof = ndofs_per_cell(sdh)+1 #first(pressure_dofs)
    return Pressure3D0DVolumeCouplerCache(
        fv,
        displacement_range,
        pressure_dof,
        filter(facet -> facet[1] ∈ sdh.cellset, all_facets),
        model.volume_method,
    )
end

@inline FerriteOperators.is_facet_in_cache(
    facet::FacetIndex,
    cell::CellCache,
    facet_cache::Pressure3D0DVolumeCouplerCache,
) = facet ∈ facet_cache.facets

function FerriteOperators.assemble_facet!(
    Kₑ::AbstractMatrix,
    residualₑ::AbstractVector,
    uₑ::AbstractVector,
    geometry_cache::CellCache,
    local_facet_index::Int,
    element_cache::Pressure3D0DVolumeCouplerCache,
    t,
)
    (; fv, displacement_range, pressure_index, volume_method) = element_cache

    reinit!(fv, geometry_cache, local_facet_index)

    # Displacement
    pdof = pressure_index # celldofs(geometry_cache)[pressure_index]
    dₑ = @view uₑ[displacement_range]
    p = uₑ[pdof]
    coords = getcoordinates(geometry_cache)

    for qp in QuadratureIterator(fv)
        # Part 1: Surface pressure part
        ∂Ω₀ = getdetJdV(fv, qp)

        ∇d = function_gradient(fv, qp, dₑ)
        F = one(∇d) + ∇d
        J = det(F)
        invF = inv(F)
        cofF = transpose(invF)

        n₀ = getnormal(fv, qp)
        n = cofF ⋅ n₀

        for i ∈ 1:getnbasefunctions(fv)
            δuᵢ = shape_value(fv, qp, i)
            residualₑ[displacement_range[i]] += p * J * n ⋅ δuᵢ * ∂Ω₀
            for j ∈ 1:getnbasefunctions(fv)
                ∇δuⱼ = shape_gradient(fv, qp, j)
                # Add contribution to the tangent
                #   δF^-1 = -F^-1 δF F^-1
                #   δJ = J tr(δF F^-1)
                # Product rule
                δcofF = -transpose(invF ⋅ ∇δuⱼ ⋅ invF)
                δJ = J * tr(∇δuⱼ ⋅ invF)
                δJcofF = δJ * cofF + J * δcofF
                Kₑ[displacement_range[i], displacement_range[j]] += p * (δJcofF ⋅ n₀) ⋅ δuᵢ * ∂Ω₀
            end
            Kₑ[displacement_range[i], pdof] += J * n ⋅ δuᵢ * ∂Ω₀
        end

        # Part 2: Chamber volume constraint part
        d = function_value(fv, qp, dₑ)
        x = spatial_coordinate(fv, qp, coords)

        residualₑ[pdof] += volume_integral(x, d, F, n₀, volume_method) * ∂Ω₀

        # Via chain rule we obtain:
        #   δV(u,F(u)) = δu ⋅ dVdu + δF : dVdF
        ∂V∂u = Tensors.gradient(u_ -> volume_integral(x, u_, F, n₀, volume_method), d)
        ∂V∂F = Tensors.gradient(u_ -> volume_integral(x, d, u_, n₀, volume_method), F)
        for j ∈ 1:getnbasefunctions(fv)
            δuⱼ = shape_value(fv, qp, j)
            ∇δuⱼ = shape_gradient(fv, qp, j)
            Kₑ[pdof, displacement_range[j]] += (∂V∂u ⋅ δuⱼ + ∂V∂F ⊡ ∇δuⱼ) * ∂Ω₀
        end

        # Kₑ[pdof, pdof] += 0
    end
end

struct Volume0DResidualIntegrator <: AbstractNonlinearIntegrator
    pressure_symbol::Symbol
    chamber_index::Int
    V⁰ᴰs::AbstractVector{Float64}
end

@concrete struct Volume0DResidualCache <: AbstractVolumetricElementCache
    pressure_dof_index
    chamber_index
    V⁰ᴰs
end

function FerriteOperators.setup_element_cache(model::Volume0DResidualIntegrator, sdh)
    pressure_dof_range = dof_range(sdh, model.pressure_symbol)
    @assert lenght(pressure_dof_range) == 1 "Pressure ,,$(pressure_symbol)'' is associated with more than one dof ($(lenght(pressure_dof_range)))"

    return Volume0DResidualCache(first(pressure_dof_range), model.chamber_index, model.V⁰ᴰs)
end

function FerriteOperators.assemble_element!(
    Kₑ::AbstractMatrix,
    residualₑ::AbstractVector,
    uₑ::AbstractVector,
    geometry_cache::CellCache,
    element::Volume0DResidualCache,
    t,
)
    R[element.pressure_dof_index] -= element.V⁰ᴰs[element.chamber_index]
end

function FerriteOperators.assemble_element!(
    Kₑ::AbstractMatrix,
    uₑ::AbstractVector,
    geometry_cache::CellCache,
    element::Volume0DResidualCache,
    t,
) end

function FerriteOperators.assemble_element!(
    residualₑ::AbstractVector,
    uₑ::AbstractVector,
    geometry_cache::CellCache,
    element::Volume0DResidualCache,
    t,
)
    R[element.pressure_dof_index] -= element.V⁰ᴰs[element.chamber_index]
end
