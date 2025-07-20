"""
Descriptor for which volume to couple with which variable for the constraint.
"""
struct ChamberVolumeCoupling{CVM}
    chamber_surface_setname::String
    chamber_volume_method::CVM
    lumped_model_symbol::Union{Symbol,ModelingToolkit.Num}
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
    displacement_symbol::Union{Symbol,ModelingToolkit.Num}
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

function assemble_LFSI_coupling_contribution_row_inner!(Jₑ, rₑ, uₑ, p, facet, dh, fv, method)
    reinit!(fv, facet)

    coords = getcoordinates(facet)

    for qp in QuadratureIterator(fv)
        dΓ = getdetJdV(fv, qp)
        N = getnormal(fv, qp)

        ∇u = function_gradient(fv, qp, uₑ)
        F = one(∇u) + ∇u

        d = function_value(fv, qp, uₑ)

        x = spatial_coordinate(fv, qp, coords)

        rₑ[1] += volume_integral(x, d, F, N, method) * dΓ
        # Via chain rule we obtain:
        #   δV(u,F(u)) = δu ⋅ dVdu + δF : dVdF
        ∂V∂u = Tensors.gradient(u_ -> volume_integral(x, u_, F, N, method), d)
        ∂V∂F = Tensors.gradient(u_ -> volume_integral(x, d, u_, N, method), F)
        for j ∈ 1:getnbasefunctions(fv)
            δuⱼ = shape_value(fv, qp, j)
            ∇δuⱼ = shape_gradient(fv, qp, j)
            Jₑ[j] += (∂V∂u ⋅ δuⱼ + ∂V∂F ⊡ ∇δuⱼ) * dΓ
        end
    end
end

"""
Chamber volume contribution for the 3D-0D constraint
    ∫ V³ᴰ(u) ∂Ω = V⁰ᴰ(c)
where u are the unkowns in the 3D problem and c the unkowns in the 0D problem.
"""
function assemble_LFSI_coupling_contribution_row!(C, R, dh, u, p, V⁰ᴰ, method)
    grid = dh.grid
    ip = Ferrite.getfieldinterpolation(dh.subdofhandlers[1], method.displacement_symbol) # TODO TYPE INSTABILITY - remove this as the interpolation query is instable
    ip_geo = Ferrite.geometric_interpolation(typeof(getcells(grid, 1)))
    intorder = 2*Ferrite.getorder(ip)
    ref_shape = Ferrite.getrefshape(ip)
    qr_facet = FacetQuadratureRule{ref_shape}(intorder)
    fv = FacetValues(qr_facet, ip, ip_geo)

    ndofs = getnbasefunctions(ip)

    uₑ = zeros(ndofs)
    Jₑ = zeros(ndofs)
    rₑ = zeros(1)

    drange = dof_range(dh, method.displacement_symbol)

    for facet ∈ FacetIterator(dh, method.facets)
        ddofs = @view celldofs(facet)[drange]
        uₑ .= u[ddofs]
        fill!(Jₑ, 0.0)
        fill!(rₑ, 0.0)
        assemble_LFSI_coupling_contribution_row_inner!(Jₑ, rₑ, uₑ, p, facet, dh, fv, method.volume_method)
        C[ddofs] .+= Jₑ
        R[1] += rₑ[1]
    end

    # R = ∫ V³ᴰ(u) ∂Ω - V⁰ᴰ(c)
    @info "Volume 3D $(R[1])"
    R[1] -= V⁰ᴰ
end

function assemble_LFSI_coupling_contribution_col_inner!(C, R, u, p, facet, dh, fv::FacetValues, symbol::Symbol)
    reinit!(fv, facet)
    drange = dof_range(dh, symbol)

    ddofs = @view celldofs(facet)[drange]
    uₑ = @view u[ddofs]

    for qp in QuadratureIterator(fv)
        ∂Ω₀ = getdetJdV(fv, qp)
        n₀ = getnormal(fv, qp)

        ∇u = function_gradient(fv, qp, uₑ)
        F = one(∇u) + ∇u
        J = det(F)
        invF = inv(F)
        cofF = transpose(invF)

        for j ∈ 1:getnbasefunctions(fv)
            δuⱼ = shape_value(fv, qp, j)
            R[ddofs[j]] += p * J * cofF ⋅ n₀ ⋅ δuⱼ * ∂Ω₀
            C[ddofs[j], 1] += J * cofF ⋅ n₀ ⋅ δuⱼ * ∂Ω₀
        end
    end
end


function assemble_LFSI_coupling_contribution_col_inner!(C, u, p, facet, dh, fv::FacetValues, symbol::Symbol)
    reinit!(fv, facet)
    drange = dof_range(dh, symbol)

    ddofs = @view celldofs(facet)[drange]
    uₑ = @view u[ddofs]

    for qp in QuadratureIterator(fv)
        ∂Ω₀ = getdetJdV(fv, qp)
        n₀ = getnormal(fv, qp)

        ∇u = function_gradient(fv, qp, uₑ)
        F = one(∇u) + ∇u
        J = det(F)
        invF = inv(F)
        cofF = transpose(invF)

        for j ∈ 1:getnbasefunctions(fv)
            δuⱼ = shape_value(fv, qp, j)
            C[ddofs[j], 1] += J * cofF ⋅ n₀ ⋅ δuⱼ * ∂Ω₀
        end
    end
end

function assemble_LFSI_volumetric_corrector_inner!(Kₑ::Matrix, residualₑ, uₑ, p, fv, symbol)
    reinit!(fv, facet[1], facet[2])

    ndofs_facet = getnbasefunctions(fv)
    for qp in QuadratureIterator(fv)
        dΓ = getdetJdV(fv, qp)

        n₀ = getnormal(fv, qp)

        ∇u = function_gradient(fv, qp, uₑ)
        F = one(∇u) + ∇u

        # Add contribution to the residual from this test function
        # TODO fix the "nothing" here
        invF = inv(F)
        cofF = transpose(invF)
        J = det(F)
        neumann_term = p * J * cofF
        for i in 1:J
            δuᵢ = shape_value(fv, qp, i)
            residualₑ[i] += neumann_term ⋅ n₀ ⋅ δuᵢ * dΓ

            # ∂P∂Fδui =   ∂P∂F ⊡ (n₀ ⊗ δuᵢ) # Hoisted computation
            for j in 1:ndofs_facet
                ∇δuⱼ = shape_gradient(fv, qp, j)
                # Add contribution to the tangent
                # Kₑ[i, j] += (n₀ ⊗ δuⱼ) ⊡ ∂P∂Fδui * dΓ
                #   δF^-1 = -F^-1 δF F^-1
                #   δJ = J tr(δF F^-1)
                # Product rule
                δcofF = -transpose(invF ⋅ ∇δuⱼ ⋅ invF)
                δJ = J * tr(∇δuⱼ ⋅ invF)
                δJcofF = δJ * cofF + J * δcofF
                Kₑ[j, i] += p * (δJcofF ⋅ n₀) ⋅ δuᵢ * dΓ
            end
        end
    end
end

function assemble_LFSI_volumetric_corrector!(J, residual, dh, u, p, setname, method)
    grid = dh.grid
    ip = Ferrite.getfieldinterpolation(dh.subdofhandlers[1], method.displacement_symbol)
    ip_geo = Ferrite.geometric_interpolation(typeof(getcells(grid, 1)))
    intorder = 2*Ferrite.getorder(ip)
    ref_shape = Ferrite.getrefshape(ip)
    qr_facet = FacetQuadratureRule{ref_shape}(intorder)
    fv = FacetValues(qr_facet, ip, ip_geo)

    drange = dof_range(dh, method.displacement_symbol)

    assembler = start_assemble(J, residual, false)

    ndofs = ndofs_per_cell(dh)
    Jₑ = zeros(ndofs, ndofs)
    rₑ = zeros(ndofs)
    uₑ = zeros(ndofs)

    for facet ∈ FacetIterator(dh, getfacetset(grid, setname))
        dofs = celldofs(facet)
        fill!(Jₑ, 0)
        fill!(rₑ, 0)
        uₑ .= @view u[dofs]
        assemble_LFSI_volumetric_corrector_inner!(Jₑ, rₑ, uₑ, p, fv, method.displacement_symbol)
        assemble!(assembler, dofs, Jₑ, rₑ)
    end
end
