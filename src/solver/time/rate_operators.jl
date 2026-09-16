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
