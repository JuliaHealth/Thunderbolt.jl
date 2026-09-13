#####################################################################
#  Lumped-mass rate operators: f(x) = Mₗ⁻¹ K x                      #
#####################################################################
"""
    LumpedMassRateOperator{KOpT, VT}(K, invM)

The rate map `x ↦ Mₗ⁻¹Kx` of a lumped-mass semidiscretization, with the lumped inverse
mass `invM` applied AFTER the unlumped action `K` rather than folded into a combined
matrix. `K` need only support `mul!` -- a raw sparse matrix or any operator FerriteOperators
defines `mul!` for both work.
"""
struct LumpedMassRateOperator{KOpT, VT <: AbstractVector}
    K::KOpT
    invM::VT
end

"""
    mul_rate!(y, op::LumpedMassRateOperator, x)

`y ← invM .* (K * x)`. THE SEAM every lumped-mass rate operator implements; a future
block-diagonal-mass (DG) operator is the second implementor.
"""
function mul_rate!(y, op::LumpedMassRateOperator, x)
    mul!(y, op.K, x)
    y .*= op.invM
    return y
end

"""
    add_source_rate!(y, op::LumpedMassRateOperator, b)

`y ← y + invM .* b`. THE SEAM every lumped-mass rate operator implements alongside
[`mul_rate!`](@ref).
"""
function add_source_rate!(y, op::LumpedMassRateOperator, b)
    y .+= op.invM .* b
    return y
end

"""
    compute_lumped_inverse_mass!(invM, Mop, ones_tmp)

Row-sum mass lumping: `invM ← 1 ./ (Mop * 𝟙)`, via a single operator product (no scalar
indexing, so this runs on a device vector too). Errors if any row sum is non-positive --
row-sum lumping is only exact for a P1 mass matrix, whose row sums are guaranteed positive.
"""
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
