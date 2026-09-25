#####################################################################
#  The rate operator f_F(x) = σ·M⁻¹Kx                                #
#####################################################################

"""
    rate_sign(integrator) -> ±1

The sign that turns the operator a bilinear diffusion term assembles into the rate
`du/dt = σ M⁻¹ K u`. Thunderbolt's [`BilinearDiffusionIntegrator`](@ref) assembles the already
negated form `-∫ D∇u·∇v`, so its sign is `+1`, the default; an element assembling the positive form
`+∫ D∇u·∇v` declares `-1`. The wrong sign integrates the diffusion backwards in time.
"""
rate_sign(::AbstractBilinearIntegrator) = 1

# `op` is `x ↦ M⁻¹Kx`: FerriteOperators' rate-form operator, or its mirrored form
# (`MirroredRateFormOperator`). `minv` is what a source is scaled by, resolved once at setup
# (`_rate_source_inverse_mass`): the diagonal's reciprocal, a per-cell inverse operator, or `nothing`.
struct RateOperator{OpT, MinvT}
    op::OpT
    minv::MinvT
    sign::Int
end

function mul_rate!(y, r::RateOperator, x)
    r.sign == 1 ? mul!(y, r.op, x) : mul!(y, r.op, x, r.sign, zero(eltype(y)))
    return y
end

# `y += M⁻¹b`.
add_source_rate!(y, r::RateOperator, b) = (_add_inverse_mass!(y, r.minv, b); y)
_add_inverse_mass!(y, minv::AbstractVector, b) = (y .+= minv .* b; nothing)
_add_inverse_mass!(y, minv, b) = (mul!(y, minv, b, one(eltype(y)), one(eltype(y))); nothing)
@noinline _add_inverse_mass!(y, ::Nothing, b) = error(
    "This rate operator carries no inverse mass to scale a source by: `M⁻¹` was fused into the " *
    "diffusion store at fill time and no per-cell inverse was built beside it.",
)
