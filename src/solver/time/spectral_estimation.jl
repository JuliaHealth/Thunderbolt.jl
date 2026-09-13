#####################################################################
#  Spectral radius estimation: power iteration + host-only bounds   #
#####################################################################
"""
    SpectralRadiusWorkspace{VT, T}

Warm-startable state for [`estimate_rho!`](@ref): the current unit-norm iterate `v`, a
matching scratch buffer `w`, the last estimated spectral radius `ρ` (safety-multiplied),
and the iteration count `iters_done` the last call needed.
"""
mutable struct SpectralRadiusWorkspace{VT <: AbstractVector, T <: Real}
    v::VT
    w::VT
    ρ::T
    iters_done::Int
end

"""
    SpectralRadiusWorkspace(template::AbstractVector)

Allocate a workspace shaped like `template`, seeded (via [`_reseed!`](@ref)) with a fixed
non-uniform unit vector rather than `template` itself, whose caller-supplied content may be zero
or (unluckily) an eigenvector the iteration cannot then escape.
"""
function SpectralRadiusWorkspace(template::AbstractVector)
    T = real(eltype(template))
    ws = SpectralRadiusWorkspace(similar(template), similar(template), zero(T), 0)
    _reseed!(ws)
    return ws
end

"""
    _reseed!(ws::SpectralRadiusWorkspace)

Reset `ws.v` to the same fixed non-uniform unit vector the constructor seeds from, discarding
whatever the iterate currently holds. [`estimate_rho!`](@ref) calls this to retry from a clean
start after a poisoned iterate -- e.g. an `apply!` that returned Inf/NaN, which
`v .= w ./ nw` then spreads into every entry -- rather than warm-starting from it again.
"""
function _reseed!(ws::SpectralRadiusWorkspace)
    v = ws.v
    # One broadcast, not a scalar write into `v[end]`: a device vector forbids scalar indexing, and
    # the seed values are the same ones either way.
    v .= one(eltype(v)) .+ (eachindex(v) .== lastindex(v))
    v ./= norm(v)
    return ws
end

"""
    estimate_rho!(ws::SpectralRadiusWorkspace, apply!; maxiters = 50, reltol = 1.0e-2, safety = 1.1,
                  jump_factor = 1.0e6, describe = () -> "")

Normalized power iteration for the spectral radius of the (implicit) linear operator
`apply!(w, v) -> w`, warm-started from `ws.v` and converging once `ρ = ‖apply!(w, v)‖ / ‖v‖`
changes by less than `reltol` relative between iterations. Returns `safety * ρ` and stores
that same safety-multiplied value into `ws.ρ` (`ws.v`/`ws.w`/`ws.iters_done` are updated too).
Reads its vectors through `norm`/broadcast only, so it runs unchanged on a device vector as
long as `apply!` does.

A non-finite result, or one more than `jump_factor` times the previous `ws.ρ` (skipped on the
first call for a workspace, where `ws.ρ == 0` and there is nothing to compare against), is
treated as `apply!` having been evaluated somewhere it should not have been trusted rather than a
genuine spectral radius: it retries exactly once from a freshly [`_reseed!`](@ref)ed iterate (the
current `ws.v` may itself be poisoned by the bad result), and raises an error -- never a silently
clamped value -- if the retry is no better. `describe` is appended verbatim to that error; a
caller with more context than this generic operator (the state a Jacobian-free difference was
evaluated at, its own knobs) can use it to name that in the message.
"""
function estimate_rho!(
    ws::SpectralRadiusWorkspace{VT, T},
    apply!;
    maxiters::Integer = 50,
    reltol = 1.0e-2,
    safety = 1.1,
    jump_factor = 1.0e6,
    describe = () -> "",
) where {VT, T}
    ρ_prev = ws.ρ
    ρ = T(safety) * _power_iterate!(ws, apply!, maxiters, T(reltol))
    if !_rho_is_sane(ρ, ρ_prev, jump_factor)
        _reseed!(ws)
        ρ = T(safety) * _power_iterate!(ws, apply!, maxiters, T(reltol))
        _rho_is_sane(ρ, ρ_prev, jump_factor) || _rho_runaway_error(ρ, ρ_prev, jump_factor, describe)
    end
    ws.ρ = ρ
    return ws.ρ
end

# The iteration proper, factored out of `estimate_rho!` so the runaway guard can rerun it against
# a reseeded `ws.v` without duplicating the loop. Pre-safety: `estimate_rho!` applies `safety`.
function _power_iterate!(ws::SpectralRadiusWorkspace{VT, T}, apply!, maxiters, tol) where {VT, T}
    v, w = ws.v, ws.w
    ρ = zero(T)
    ρ_prev = ρ
    iters = 0
    for k = 1:maxiters
        iters = k
        nv = norm(v)
        nv > 0 || error("SpectralRadiusWorkspace: iterate collapsed to zero.")
        apply!(w, v)
        nw = norm(w)
        ρ = nw / nv
        if !isfinite(ρ) || nw == 0
            # A non-finite `ρ` (bad `apply!`) or nw == 0 (v lies in the operator's null space) both
            # leave no direction left to refine -- and breaking here, before `v .= w ./ nw`, keeps a
            # non-finite `nw` from spreading `NaN` into `v` and tripping the zero-collapse check above
            # on the loop's next iteration.
            break
        end
        v .= w ./ nw
        k > 1 && abs(ρ - ρ_prev) ≤ tol * abs(ρ_prev) && break
        ρ_prev = ρ
    end
    ws.iters_done = iters
    return ρ
end

_rho_is_sane(ρ, ρ_prev, jump_factor) = isfinite(ρ) && (ρ_prev == 0 || ρ ≤ jump_factor * ρ_prev)

@noinline function _rho_runaway_error(ρ, ρ_prev, jump_factor, describe)
    cause = isfinite(ρ) ?
        "a $(round(ρ / ρ_prev, sigdigits = 3))x jump over the previous estimate $(ρ_prev) " *
        "(threshold $(jump_factor)x)" : "a non-finite value"
    error(
        "estimate_rho!: the power iteration produced $ρ -- $cause -- even after retrying once " *
        "from a freshly reseeded iterate.$(describe())",
    )
end

"""
    _gershgorin_bound(K, invM::AbstractVector)

Host-only upper bound on the spectral radius of `Diagonal(invM) * K`:
`maxᵢ invM[i] * Σⱼ|K[i,j]|`. One method per sparse storage `K` may take -- row access
differs between column-major (`SparseMatrixCSC`) and row-major (`ThreadedSparseMatrixCSR`)
layouts.
"""
function _gershgorin_bound(K::SparseMatrixCSC, invM::AbstractVector)
    T = promote_type(eltype(K), eltype(invM))
    rowsums = zeros(T, size(K, 1))
    rv, nzv = rowvals(K), nonzeros(K)
    for col = 1:size(K, 2), idx in nzrange(K, col)
        rowsums[rv[idx]] += abs(nzv[idx])
    end
    return maximum(invM[i] * rowsums[i] for i in eachindex(rowsums))
end

function _gershgorin_bound(K::ThreadedSparseMatrixCSR, invM::AbstractVector)
    A = K.A
    T = promote_type(eltype(K), eltype(invM))
    bound = zero(T)
    for row = 1:size(A, 1)
        rowsum = zero(T)
        for nz in nzrange(A, row)
            rowsum += abs(A.nzval[nz])
        end
        bound = max(bound, invM[row] * rowsum)
    end
    return bound
end

"""
    _should_reestimate(policy, steps_since, stepfail::Bool) -> Bool

Whether ρ needs re-estimating this step. `steps_since` counts steps since the last
estimate; a NEGATIVE value means none has run yet and, like a step failure (`stepfail`),
always forces one regardless of `policy`. Otherwise:

- `:once` -- never again (paper default: estimate once, trust it for the whole run).
- `n::Int` -- every `n` steps (`steps_since ≥ n`).
- a callable -- `policy(steps_since)`.
"""
function _should_reestimate(policy, steps_since, stepfail::Bool)
    (stepfail || steps_since < 0) && return true
    return _reestimate_due(policy, steps_since)
end

_reestimate_due(policy::Symbol, steps_since) =
    policy === :once ? false :
    error("Unknown rho_recompute policy :$policy -- expected :once, an Int, or a callable.")
_reestimate_due(n::Integer, steps_since) = steps_since ≥ n
_reestimate_due(f, steps_since) = f(steps_since)
