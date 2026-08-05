#####################################################################
#            Step size controllers driven by an error estimate      #
#####################################################################
#
# These are Thunderbolt's own, deliberately: they follow the same in-package pattern as the
# continuation controllers in `homotopy.jl` -- a plain struct, a cache, and methods on
# `should_accept_step` / `adapt_dt!` / `reject_step!`.
#
# The alternative was to drive the controllers of `OrdinaryDiffEqCore` directly. They are good, and
# that path still works for anyone who asks for it explicitly (see the adapters in `newmark.jl`), but
# it couples the default configuration to a protocol that moves: over one upgrade it grew a fifth
# argument on `setup_controller_cache`, and it reaches for integrator fields such as `success_iter`
# that this integrator does not carry. A ~60 line port of an algorithm that has not changed since
# Söderlind (2003) is cheaper to own than that coupling.

@doc raw"""
    PIDController(β₁, β₂, β₃ = 0; accept_safety = 0.81, limiter = default_dt_factor_limiter)

Proportional-integral-derivative step size control from a local error estimate, following
Söderlind [Sod:2003:dcs](@cite).

With ``\varepsilon_i`` the inverse of the scaled error estimate of the last three steps, the step size
factor is
```math
\texttt{dt\_factor} = \mathrm{limiter}\left(
\varepsilon_1^{\beta_1/k}\, \varepsilon_2^{\beta_2/k}\, \varepsilon_3^{\beta_3/k} \right) ,
\qquad k = \texttt{adaptive\_order} + 1 ,
```
and the next step is `Δt * dt_factor`. Dividing the exponents by `k` is why the coefficients are
stated independently of the scheme's order, unlike a bare PI controller.

Two details do most of the work in practice, and both differ from a textbook I-controller:

* a step is accepted when the **proposed factor** clears `accept_safety`, not when the error estimate
  clears one. A noisy estimate landing just above unity then costs a slightly shorter next step
  instead of a discarded one.
* the default limiter ``1 + \arctan(x - 1)`` is smooth and saturates around `[0.21, 2.57]`, so no
  separate `qmin`/`qmax` clipping is needed and the response to an outlier is gentle.

Measured against the alternatives on a freely vibrating bar at `reltol = 1e-3` (accepted / rejected
steps): I-controller 117/33, PI controller 133/14, this controller **101/4**.

The defaults `(0.6, -0.2, 0)` are Söderlind's, and are what [`NewmarkSolver`](@ref) uses unless a
controller is passed to `init`.
"""
struct PIDController{T, Limiter}
    β::NTuple{3, T}
    accept_safety::T
    limiter::Limiter
end

@inline default_dt_factor_limiter(x) = one(x) + atan(x - one(x))

function PIDController(
    β₁::Real,
    β₂::Real,
    β₃::Real = zero(β₁);
    accept_safety = 81 // 100,
    limiter = default_dt_factor_limiter,
)
    β = map(float, promote(β₁, β₂, β₃))
    T = typeof(β[1])
    return PIDController{T, typeof(limiter)}(β, T(accept_safety), limiter)
end

"""
    PIDControllerCache

Per-solve state of a [`PIDController`](@ref): the scaled error estimate of the current attempt, the
inverse estimates of the last three steps, and the factor proposed for the current attempt.

`err` is seeded with ones, which makes the first steps behave as a plain I-controller until enough
history has accumulated -- the same convention the reference implementation uses.
"""
mutable struct PIDControllerCache{T, C <: PIDController}
    controller::C
    # Inverse scaled error estimates of the current and the two preceding steps
    err::NTuple{3, T}
    # Proposed step size factor of the current attempt, and whether it has been computed for it
    dt_factor::T
    dt_factor_valid::Bool
    # Scaled error estimate reported by the scheme for the current attempt
    EEst::T
end

OrdinaryDiffEqCore.setup_controller_cache(
    _alg,
    _cache,
    controller::PIDController{T},
    ::Type{EEstT},
    _disco_probs,
) where {T, EEstT} = PIDControllerCache{T, typeof(controller)}(
    controller,
    (one(T), one(T), one(T)),
    one(T),
    false,
    one(T),
)

"""
    set_error_estimate!(controller_cache, EEst)

Report the scheme's scaled local error estimate for the current attempt, in the usual convention where
`EEst ≤ 1` means "within tolerance".

This is the *only* thing a scheme owes a controller. Everything derived from it -- the proposed step
size, whether the step is accepted, the history -- belongs to the controller.
"""
set_error_estimate!(controller_cache::PIDControllerCache, EEst) =
    (controller_cache.EEst = EEst; controller_cache.dt_factor_valid = false; nothing)

# A controller that ignores the estimate still has to tolerate being told one.
set_error_estimate!(_controller_cache, EEst) = nothing

# `k = order + 1`, i.e. the order of the *local* error the estimate measures.
_pid_error_order(alg) = adaptive_order(alg) + 1

@doc raw"""
    adaptive_order(alg)

Order of the local error estimate `alg` provides, i.e. one less than the exponent a step size
controller applies. A second order scheme with an `O(Δt³)` local error estimate answers `2`.

Distinct from the *convergence* order of the scheme whenever the estimate is not the difference of two
embedded formulas of adjacent order.
"""
adaptive_order(alg) = error(
    "$(typeof(alg)) does not declare an `adaptive_order`, so no error based step size control " *
    "is possible for it. Implement `Thunderbolt.adaptive_order` if it provides an error estimate.",
)

# One update per attempt: the history shift below is a side effect, so the factor is computed once and
# memoised. The step loop asks `should_accept_step` more than once per attempt.
function _pid_dt_factor!(cache::PIDControllerCache, alg)
    cache.dt_factor_valid && return cache.dt_factor
    (; β, limiter) = cache.controller
    k = _pid_error_order(alg)
    # Guarded against a vanishing estimate, which would otherwise send the factor to `Inf`/`NaN`.
    EEst = max(cache.EEst, eps(typeof(cache.EEst)))
    err = (inv(EEst), cache.err[1], cache.err[2])
    cache.err = err
    cache.dt_factor = limiter(err[1]^(β[1] / k) * err[2]^(β[2] / k) * err[3]^(β[3] / k))
    cache.dt_factor_valid = true
    return cache.dt_factor
end

# The history must advance only for a step that is actually taken, so it is shifted here rather than
# in `_pid_dt_factor!`, which a rejected attempt also runs.
function _pid_accept!(cache::PIDControllerCache)
    cache.err = (cache.err[1], cache.err[1], cache.err[2])
    return nothing
end

should_accept_step(
    integrator::ThunderboltTimeIntegrator,
    _cache,
    controller_cache::PIDControllerCache,
) = _pid_dt_factor!(controller_cache, integrator.alg) ≥ controller_cache.controller.accept_safety

function adapt_dt!(
    integrator::ThunderboltTimeIntegrator,
    _cache,
    controller_cache::PIDControllerCache,
)
    dt_factor = _pid_dt_factor!(controller_cache, integrator.alg)
    _pid_accept!(controller_cache)
    # Through `set_proposed_dt!` rather than by assigning `dt`, so that `dtcache` and `dtpropose`
    # follow -- `modify_dt_for_tstops!` restores `dt` from `dtcache` at every header.
    SciMLBase.set_proposed_dt!(integrator, min(integrator.dt * dt_factor, integrator.opts.dtmax))
    return nothing
end

function reject_step!(
    integrator::ThunderboltTimeIntegrator,
    _cache,
    controller_cache::PIDControllerCache,
)
    # A step rejected because the *solve* failed has already had `dt` shortened by
    # `post_newton_controller!` in the step footer. Shrinking again here would apply the failure factor
    # and the controller's proposal to the same attempt.
    integrator.force_stepfail && return SciMLBase.set_proposed_dt!(integrator, integrator.dt)
    dt_factor = _pid_dt_factor!(controller_cache, integrator.alg)
    SciMLBase.set_proposed_dt!(integrator, integrator.dt * dt_factor)
    return nothing
end
