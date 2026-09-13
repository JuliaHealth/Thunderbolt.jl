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

Allocate a workspace shaped like `template`, seeded with a fixed non-uniform unit vector
rather than `template` itself, whose caller-supplied content may be zero or (unluckily) an
eigenvector the iteration cannot then escape.
"""
function SpectralRadiusWorkspace(template::AbstractVector)
    T = real(eltype(template))
    v = similar(template)
    v .= one(eltype(template))
    length(v) > 1 && (v[end] += one(eltype(template)))
    v ./= norm(v)
    w = similar(template)
    return SpectralRadiusWorkspace(v, w, zero(T), 0)
end

"""
    estimate_rho!(ws::SpectralRadiusWorkspace, apply!; maxiters = 50, reltol = 1.0e-2, safety = 1.1)

Normalized power iteration for the spectral radius of the (implicit) linear operator
`apply!(w, v) -> w`, warm-started from `ws.v` and converging once `ρ = ‖apply!(w, v)‖ / ‖v‖`
changes by less than `reltol` relative between iterations. Returns `safety * ρ` and stores
that same safety-multiplied value into `ws.ρ` (`ws.v`/`ws.w`/`ws.iters_done` are updated too).
Reads its vectors through `norm`/broadcast only, so it runs unchanged on a device vector as
long as `apply!` does.
"""
function estimate_rho!(
    ws::SpectralRadiusWorkspace{VT, T},
    apply!;
    maxiters::Integer = 50,
    reltol = 1.0e-2,
    safety = 1.1,
) where {VT, T}
    v, w = ws.v, ws.w
    tol = T(reltol)
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
        if nw == 0
            break # v lies in the operator's null space -- no direction left to refine
        end
        v .= w ./ nw
        k > 1 && abs(ρ - ρ_prev) ≤ tol * abs(ρ_prev) && break
        ρ_prev = ρ
    end
    ws.ρ = T(safety) * ρ
    ws.iters_done = iters
    return ws.ρ
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
