#####################################################################
#  Lumped-mass rate operators: f(x) = Mₗ⁻¹ K x                      #
#####################################################################
# The rate map `x ↦ Mₗ⁻¹Kx` of a lumped-mass semidiscretization. `invM` is applied AFTER the unlumped
# action of `K`, so `K` need only support `mul!`.
struct LumpedMassRateOperator{KOpT, VT <: AbstractVector}
    K::KOpT
    invM::VT
end

function mul_rate!(y, op::LumpedMassRateOperator, x)
    mul!(y, op.K, x)
    y .*= op.invM
    return y
end

function add_source_rate!(y, op::LumpedMassRateOperator, b)
    y .+= op.invM .* b
    return y
end

# Row-sum mass lumping `invM ← 1 ./ (Mop * 𝟙)`, through a single operator product so that it runs on
# a device vector too. Row-sum lumping is only exact for a P1 mass matrix, whose row sums are
# guaranteed positive, which is what the guard below checks.
function compute_lumped_inverse_mass!(invM, Mop, ones_tmp)
    ones_tmp .= one(eltype(ones_tmp))
    mul!(invM, Mop, ones_tmp)
    minimum(invM) > 0 || error(
        "Row-sum mass lumping requires strictly positive row sums (guaranteed for a P1 " *
        "mass matrix); got a non-positive row sum.",
    )
    invM .= 1 ./ invM
    return invM
end

#####################################################################
#  Fused rate operators: f(x) = -A x, with A = M⁻¹K already fused    #
#####################################################################

"""
The rate map `x ↦ -Ax` of a semidiscretization whose diffusion operator `A` already carries the
inverse mass -- the fused `M⁻¹K` a `BlockRowAssembly(; premultiply_inverse_mass = ...)` store holds,
where `M` is block diagonal by cell over a discontinuous space and is inverted exactly, per cell, at
fill time. There is no lumping and no separate mass matrix anywhere in this path.

**The sign is this type's whole content beyond the product.** The fused operator carries the POSITIVE
stiffness convention `a(u,v) = +∫ D∇u·∇v`, which is the opposite of
[`BilinearDiffusionIntegrator`](@ref)'s. [`LumpedMassRateOperator`](@ref) takes Thunderbolt's own
already-negated `K` and applies no sign; this one takes the positive one and negates. Handing either
the other's operator integrates the diffusion backwards in time -- a blow-up, not a subtle error.

The negation rides the 5-argument `mul!` rather than a second pass over `y`, so the whole rate is one
sweep of the action.
"""
struct FusedInverseMassRateOperator{OpT}
    A::OpT
end

function mul_rate!(y, op::FusedInverseMassRateOperator, x)
    mul!(y, op.A, x, -one(eltype(y)), zero(eltype(y)))
    return y
end

# A source enters the rate as `M⁻¹b`, and the fused store does not expose `M⁻¹` apart from the
# product it was folded into -- the per-cell inverse blocks are formed and dropped at fill time. The
# seam is a rate operator that keeps them (a block-diagonal `M⁻¹` and its device kernel); until then
# a stimulus on this path has to be written as an initial condition.
@noinline add_source_rate!(y, op::FusedInverseMassRateOperator, b) = error(
    "A source term cannot enter the rate of a `FusedInverseMassRateOperator`: the inverse mass was " *
    "folded into the diffusion store at fill time and is not available on its own. Write the " *
    "stimulus as an initial condition, or extend this operator to carry the per-cell inverse mass " *
    "blocks.",
)
