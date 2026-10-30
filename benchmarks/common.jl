# What more than one benchmark in this directory needs: the Jacobi preconditioner the splitting arms
# solve with, the cgroup guard every timed run has to pass, and the timing harness itself.
#
# Included (not `using`d) by `benchmark-emrkc.jl`, `benchmark-discretization-variants.jl` and
# `benchmark-gpu-split.jl`, so everything here lands in the script's own scope. The measurement
# parameters -- how many steps a pass is, how long the warmup runs -- stay with the benchmark that
# defines them and are passed in, because they are part of what each file reports.

using Thunderbolt
using CUDA
using LinearAlgebra
using OrdinaryDiffEqOperatorSplitting

import SparseArrays: nonzeros
import SparseMatricesCSR: getrowptr, getcolval, getnzval
import Thunderbolt: ThreadedSparseMatrixCSR

####################################
## Jacobi preconditioning
####################################

"""
The reciprocal main diagonal of the backward Euler operator `M - Δt K`, as a left preconditioner for
the conjugate gradient.

Filled on first `ldiv!` rather than at `init`: `LinearSolve.init` calls the `precs` callback while
the system matrix is still the freshly allocated all-zero sparsity pattern, and the affine backward
Euler path then fills that same object in place through `nonzeros(A)` without ever reassigning
`cache.A`, which is what would otherwise mark the preconditioner stale. Holding on to `A` and reading
its diagonal on first use means the matrix is assembled by the time it is read.

Every arm runs at a fixed `Δt`, so `A` is assembled once and the one-shot fill is valid for the life
of the arm. A varying `Δt` would have to invalidate `ready`.
"""
mutable struct JacobiPrecon{MatType, VecType}
    A::MatType
    inv_diag::VecType
    ready::Bool
end

function JacobiPrecon(A)
    d = similar(nonzeros(A), size(A, 1))
    fill!(d, one(eltype(d)))
    return JacobiPrecon(A, d, false)
end

# Neither `ThreadedSparseMatrixCSR` nor `CuSparseMatrixCSR` has a `diag` method or a scalar
# `getindex`, so each row is walked for its own column index. Everything else takes the generic
# `diag` path.
function _fill_inv_diag!(d, A::ThreadedSparseMatrixCSR)
    rowptr, colval, nzval = getrowptr(A), getcolval(A), getnzval(A)
    @inbounds for i in eachindex(d)
        v = zero(eltype(d))
        for k in rowptr[i]:(rowptr[i + 1] - 1)
            colval[k] == i && (v = nzval[k])
        end
        d[i] = iszero(v) ? one(v) : inv(v)
    end
    return d
end

function _inv_diag_kernel!(d, rowPtr, colVal, nzVal)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    @inbounds if i ≤ length(d)
        v = zero(eltype(d))
        for k in rowPtr[i]:(rowPtr[i + 1] - 1)
            colVal[k] == i && (v = nzVal[k])
        end
        d[i] = iszero(v) ? one(v) : inv(v)
    end
    return nothing
end

function _fill_inv_diag!(d::CuVector, A::CUDA.CUSPARSE.CuSparseMatrixCSR)
    threads = 256
    CUDA.@cuda threads = threads blocks = cld(length(d), threads) _inv_diag_kernel!(
        d, A.rowPtr, A.colVal, A.nzVal,
    )
    return d
end

function _fill_inv_diag!(d, A)
    v = diag(A)
    d .= ifelse.(iszero.(v), one(eltype(d)), inv.(v))
    return d
end

function LinearAlgebra.ldiv!(y::AbstractVector, P::JacobiPrecon, x::AbstractVector)
    if !P.ready
        _fill_inv_diag!(P.inv_diag, P.A)
        P.ready = true
    end
    y .= P.inv_diag .* x
    return y
end
LinearAlgebra.ldiv!(P::JacobiPrecon, x::AbstractVector) = ldiv!(x, P, x)

# `KrylovJL`'s own contract: `(A, p) -> (Pl, Pr)`. CG takes left/centered preconditioning only, so the
# right slot stays the identity.
jacobi_precs(A, p = nothing) = (JacobiPrecon(A), LinearAlgebra.I)

####################################
## Machine discipline
####################################

"""
The effective `memory.max` (bytes) of this process's cgroup v2 leaf, found by walking
`/proc/self/cgroup`'s `0::<path>` up through `/sys/fs/cgroup<path>`. `nothing` for "max" (unset) or
when no such file is found.
"""
function _cgroup_memory_limit()
    lines = try
        readlines("/proc/self/cgroup")
    catch
        return nothing
    end
    idx = findfirst(l -> startswith(l, "0::"), lines)
    idx === nothing && return nothing
    dir = "/sys/fs/cgroup" * split(lines[idx], "0::")[2]
    while true
        f = joinpath(dir, "memory.max")
        if isfile(f)
            v = strip(read(f, String))
            return v == "max" ? nothing : parse(Int, v)
        end
        parent = dirname(dir)
        parent == dir && return nothing
        dir = parent
    end
end

"""
Refuses to run outside a memory-capped cgroup: an uncapped run's GC sizes its heap against the whole
machine rather than the 8G these benchmarks are meant to run in. `BENCHMARK_UNCAPPED=1` overrides.
`script` is the caller's own path, so the invocation the message prints is the one that reruns it.
"""
function _assert_memory_capped(script::AbstractString)
    get(ENV, "BENCHMARK_UNCAPPED", "0") == "1" && return nothing
    limit = _cgroup_memory_limit()
    capped = limit !== nothing && limit ≤ 12 * 1024^3
    capped || error(
        "No memory-capped cgroup detected (effective memory.max = $(limit === nothing ? "unset" : limit) " *
        "bytes). An uncapped run's GC sizes itself against the whole machine. Run:\n" *
        "  systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 env JULIA_NUM_THREADS=2 " *
        "julia -t2 --heap-size-hint=3G --project=test/gpu $script\n" *
        "or set BENCHMARK_UNCAPPED=1 to run uncapped deliberately.",
    )
    Base.JLOptions().heap_size_hint == 0 &&
        println("WARNING: no --heap-size-hint given -- GC growth is unbounded even inside the cgroup.")
    return nothing
end

####################################
## Stepping and timing
####################################

# `init` takes the initial condition as the integrator's own state, so every arm gets a copy -- a
# shared host or device initial condition would otherwise be consumed by the first arm to run.
build(form, u0, alg, Δt, tend) =
    init(OperatorSplittingProblem(form, copy(u0), (zero(Δt), tend)), alg; dt = Δt, verbose = false)

sync(::Vector) = nothing
sync(::CuVector) = CUDA.synchronize()

function gpu_clocks()
    out = read(
        `nvidia-smi --query-gpu=clocks.sm,clocks.mem --format=csv,noheader,nounits`, String,
    )
    sm, mem = parse.(Int, strip.(split(first(split(strip(out), '\n')), ',')))
    return sm, mem
end

"""
Step without interruption for `warmup_seconds`, reading the clocks back *while still stepping*. On a
card that idles at 300 MHz a short warmup measures the ramp rather than the kernel, and the returned
clocks are what says this one did not. They must be sampled mid-flight: the card drops back within
tens of milliseconds of going idle, less than one `nvidia-smi` query takes.
"""
function prewarm!(integrator, on_device, warmup_seconds)
    t0 = time_ns()
    clocks = (0, 0)
    sampled = false
    while (time_ns() - t0) / 1.0e9 < warmup_seconds
        step!(integrator)
        if on_device && !sampled && (time_ns() - t0) / 1.0e9 > warmup_seconds / 2
            clocks = gpu_clocks()
            sampled = true
        end
    end
    sync(integrator.u)
    return clocks
end

"Minimum seconds per step over `npass` passes of `nsteps` steps."
function measure!(integrator, nsteps, npass)
    best = Inf
    for _ = 1:npass
        t0 = time_ns()
        for _ = 1:nsteps
            step!(integrator)
        end
        sync(integrator.u)
        best = min(best, (time_ns() - t0) / 1.0e9 / nsteps)
    end
    return best
end
