#####################################################################
#  Spectral radius estimation: power iteration + host-only bounds   #
#####################################################################
# Warm-startable state for `estimate_rho!`.
mutable struct SpectralRadiusWorkspace{VT <: AbstractVector, T <: Real}
    v::VT
    w::VT
    ρ::T # the last estimate, already safety-multiplied
    iters_done::Int
end

# Seeded with a fixed non-uniform unit vector rather than `template`, whose caller-supplied content
# may be zero or an eigenvector the iteration cannot then escape.
function SpectralRadiusWorkspace(template::AbstractVector)
    T = real(eltype(template))
    ws = SpectralRadiusWorkspace(similar(template), similar(template), zero(T), 0)
    _reseed!(ws)
    return ws
end

# Back to that same fixed seed, discarding a poisoned iterate -- e.g. an Inf/NaN from `apply!` that
# `v .= w ./ nw` has already spread into every entry.
function _reseed!(ws::SpectralRadiusWorkspace)
    v = ws.v
    # One broadcast, not a scalar write into `v[end]`: a device vector forbids scalar indexing.
    v .= one(eltype(v)) .+ (eachindex(v) .== lastindex(v))
    v ./= norm(v)
    return ws
end

# Normalized power iteration for the spectral radius of the (implicit) linear operator
# `apply!(w, v) -> w`, warm-started from `ws.v` and converged once ρ = ‖apply!(w, v)‖/‖v‖ changes by
# less than `reltol` relative between iterations. Returns `safety * ρ` and stores that same value in
# `ws.ρ`. Reads its vectors through `norm`/broadcast only, so it runs on a device vector as long as
# `apply!` does.
#
# A non-finite result, or one more than `jump_factor` times the previous `ws.ρ` (skipped on a
# workspace's first call, where `ws.ρ == 0`), is treated as `apply!` having been evaluated somewhere
# it should not be trusted: retried exactly once from a freshly reseeded iterate, then raised as an
# error -- never a silently clamped value. `describe` is appended verbatim to that error.
#
# Exhausting `maxiters` without meeting `reltol` returns silently; power iteration converges to the
# dominant eigenvalue from below, so such a value under-estimates the true radius, which `safety` was
# not sized to cover. A caller that needs to know compares `ws.iters_done` against its `maxiters`.
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
    raw = _power_iterate!(ws, apply!, maxiters, T(reltol))
    ρ = T(safety) * raw
    if !_rho_is_sane(ρ, ρ_prev, jump_factor)
        _reseed!(ws)
        raw = _power_iterate!(ws, apply!, maxiters, T(reltol))
        ρ = T(safety) * raw
        _rho_is_sane(ρ, ρ_prev, jump_factor) || _rho_runaway_error(ρ, ρ_prev, jump_factor, describe)
    end
    ws.ρ = ρ
    return ws.ρ
end

# Pre-safety; factored out so the runaway guard can rerun it against a reseeded `ws.v`.
# `ws.iters_done == maxiters` is the under-estimation signal; the non-finite/null-space exit below is
# a definitive stop, not starvation, so it can report fewer even when nothing further would help.
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
            # Bad `apply!`, or `v` in the operator's null space: no direction left to refine. Breaking
            # before `v .= w ./ nw` also keeps a non-finite `nw` from spreading NaN into `v`.
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
    cause =
        isfinite(ρ) ?
        "a $(round(ρ / ρ_prev, sigdigits = 3))x jump over the previous estimate $(ρ_prev) " *
        "(threshold $(jump_factor)x)" : "a non-finite value"
    throw(
        EMRKCDivergence(
            "estimate_rho!: the power iteration produced $ρ -- $cause -- even after retrying once " *
            "from a freshly reseeded iterate.$(describe())",
        ),
    )
end

# Host-only upper bound on the spectral radius of `Diagonal(invM) * K`: `maxᵢ invM[i] * Σⱼ|K[i,j]|`.
# One method per sparse storage -- row access differs between column-major and row-major layouts.
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
