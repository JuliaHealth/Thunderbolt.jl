"""
    ThunderboltFerriteMultigridExt

Package extension activated when both `Thunderbolt` and `FerriteMultigrid` are loaded.

Overrides `Thunderbolt._materialize_inner_solver` for `PMGPrecon`, `GMGPrecon`, and
`ChainedMGPrecon` so that `KrylovMGSolver` automatically builds the full multigrid
hierarchy from the semidiscrete problem's `DofHandler`.
"""
module ThunderboltFerriteMultigridExt

using Thunderbolt
using FerriteMultigrid
using Ferrite
using LinearAlgebra
using SparseArrays
using LinearSolve
using Polyester
import AlgebraicMultigrid as AMG

import FerriteMultigrid: GridHierarchy

import Thunderbolt: _materialize_inner_solver, KrylovMGSolver,
                    PMGPrecon, GMGPrecon, ChainedMGPrecon, OrderedSet

function GridHierarchy(coarse_grid::Thunderbolt.SimpleMesh, n_refinements::Int)
    @assert n_refinements >= 1 "Need at least one refinement level"
    grids            = [coarse_grid]
    fine2coarse_maps = Vector{Int}[]
    crc_maps         = Vector[]

    for _ in 1:n_refinements
        fine_grid, f2c, crc = uniform_refinement(grids[end].grid)
        push!(grids, to_mesh(fine_grid))
        push!(fine2coarse_maps, f2c)
        push!(crc_maps, crc)
    end

    return GridHierarchy(grids, fine2coarse_maps, crc_maps)
end

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

_default_cycle()      = AlgebraicMultigrid.V()
_default_pmg_config() = pmultigrid_config()
_default_gmg_config() = gmultigrid_config()

"""
    _estimate_spectral_radius_dinv_A(A; power_iters = 50)

Estimate ρ(D⁻¹A) via power iteration with a deterministic starting vector.

Using `ones` (normalised) avoids the random-seed dependence that caused intermittent
over- or under-estimation of λ_max on non-symmetric matrices, which in turn made the
damped-Jacobi smoother diverge unpredictably.
"""
function _estimate_spectral_radius_dinv_A(A; power_iters = 50)
    d   = diag(A)
    n   = size(A, 1)
    v   = fill(inv(sqrt(Float64(n))), n)
    tmp = similar(v)
    λ   = 1.0
    for _ in 1:power_iters
        mul!(tmp, A, v)   # tmp = A·v
        tmp ./= d          # tmp = D⁻¹A·v
        λ    = norm(tmp)
        tmp ./= λ
        copyto!(v, tmp)
    end
    return λ
end

"""
    _chebyshev_jacobi_ω(A; power_iters = 50)

Return the Chebyshev-optimal damping parameter ω for a damped-Jacobi smoother.

The standard formula targeting the upper 2/3 of the spectrum is:

    ω = 4 / (3 · λ_max)      (Shewchuk / Stüben convention)

This guarantees:
  |1 − ω · λ|  ≤ 1/3   for  λ ∈ [λ_max/3, λ_max]    (high-freq attenuation)
  |1 − ω · λ|  ≤  1    for  λ ∈ [0, λ_max/3]          (low-freq preserved for coarse grid)

For the non-symmetric Robin-BC case, eigenvalues of D⁻¹A may be weakly complex.
The 4/(3λ_max) formula remains valid provided the eigenvalue arguments stay below
arccos(2/3) ≈ 48°, which is typical for mildly non-symmetric FEM systems where
the boundary contribution is small relative to the bulk stiffness.
"""
function _chebyshev_jacobi_ω(A; power_iters = 50)
    λ_max = _estimate_spectral_radius_dinv_A(A; power_iters)
    ω     = 4.0 / (3.0 * λ_max)
    @info "Jacobi smoother: estimated λ_max(D⁻¹A) = $λ_max → ω = $ω"
    return ω
end

"""
    DampedJacobi(iter)

Damped-Jacobi smoother: x ← x + ω·D⁻¹·(b − A·x), applied `iter` times.
Uses `mul!(tmp, A, x)` for correct `A·x` on any (non-symmetric) sparsity pattern.
"""
struct DampedJacobi
    iter::Int
end
struct DampedJacobiSmoother{At, Dt}
    iter::Int
    ω::Float64
    A::At
    d::Dt
    tmp::Dt
end

function AMG.setup_smoother(config::DampedJacobi, A, symmetry)
    d = collect(diag(A))
    return DampedJacobiSmoother(
        config.iter,
        _chebyshev_jacobi_ω(A),
        A,
        d,
        similar(d),
    )
end

function AMG.smooth!(x::AbstractVector, s::DampedJacobiSmoother, b::AbstractVector)
    (; A, iter, ω, d, tmp) = s
    for _ in 1:iter
        mul!(tmp, A, x)
        @batch for i in 1:length(x)
            x[i] += s.ω * (b[i] - tmp[i]) / d[i]
        end
    end
    return nothing
end

"""
    LazyPrecon

A mutable, type-stable preconditioner wrapper.

`precs` in `LinearSolve` is called both at `init` time (when the matrix is zero/unassembled)
and at each `solve!` call (when the real Jacobian is ready).  Because `LinearCache` fixes
the type of `Pl` at `init` time, the `precs` function must always return the *same type*.

`LazyPrecon` solves this: it is returned at `init` time with `inner = nothing` (identity),
and its `inner` field is replaced in-place on every subsequent `precs` call with the real
multigrid preconditioner.  `LinearAlgebra.ldiv!` dispatches to the inner object when set,
or falls back to copy (identity) when `inner === nothing`.
"""
mutable struct LazyPrecon
    inner  # nothing ≡ identity; any AMG/MG preconditioner otherwise
end
LazyPrecon() = LazyPrecon(nothing)

function LinearAlgebra.ldiv!(x::AbstractVector, P::LazyPrecon, b::AbstractVector)
    P.inner === nothing ? copyto!(x, b) : ldiv!(x, P.inner, b)
end
function LinearAlgebra.ldiv!(P::LazyPrecon, b::AbstractVector)
    P.inner === nothing || ldiv!(P.inner, b)
    return b
end
LinearAlgebra.adjoint(P::LazyPrecon) = P  # symmetric smoother, adjoint ≈ self

"""
    _vcycle_diagnostic(prec, A)

Print a one-shot V-cycle quality report in both Euclidean and A-norm:
- Step-by-step residual (Euclidean) through the V-cycle
- A-norm error reduction (using a direct solve for the exact solution)
- Prolongator partition-of-unity check
- 20-step stationary V-cycle iteration to confirm per-iteration convergence rate
"""
function _vcycle_diagnostic(prec, A)
    n        = size(A, 1)
    tmp      = zeros(n)

    smoother = prec.ml.presmoother
    P        = prec.ml.levels[1].P
    R        = prec.ml.levels[1].R
    A_c      = prec.ml.final_A

    b_test   = randn(n)
    norm_b   = norm(b_test)

    # Exact solution via UMFPACK for A-norm error measurement
    x_exact  = lu(A) \ b_test
    a_norm(e) = sqrt(max(dot(e, A * e), 0.0))
    e0_A     = a_norm(x_exact)   # A-norm of initial error (x=0 → e=x_exact)

    # 1. Pre-smooth
    x_s = zeros(n)
    smoother(A, x_s, b_test)
    mul!(tmp, A, x_s)
    r_s  = b_test - tmp
    es_A = a_norm(x_exact - x_s)
    @info "V-cycle step 1 – pre-smooth" ratio_2norm=(norm(r_s)/norm_b) ratio_Anorm=(es_A/e0_A)

    # 2. Restrict residual
    r_c = Vector(R * r_s)
    @info "V-cycle step 2 – restrict" norm_r_c=norm(r_c) norm_r_s=norm(r_s)

    # 3. Coarse solve
    e_c = zeros(size(A_c, 1))
    prec.ml.coarse_solver(e_c, r_c)
    coarse_res = norm(A_c * e_c - r_c) / (norm(r_c) + eps())
    # A-norm improvement from coarse correction: should be r_c^T e_c (= ‖P^T r_s‖²_{A_c⁻¹})
    coarse_gain_Anorm = dot(r_c, e_c)
    @info "V-cycle step 3 – coarse solve" coarse_res_ratio=coarse_res e_c_norm=norm(e_c) A_norm_gain_sq=coarse_gain_Anorm

    # 4. Prolongate and add correction
    correction = Vector(P * e_c)
    x_c        = x_s + correction
    mul!(tmp, A, x_c)
    r_c2 = b_test - tmp
    ec_A = a_norm(x_exact - x_c)
    @info "V-cycle step 4 – coarse correction" ratio_2norm=(norm(r_c2)/norm_b) ratio_Anorm=(ec_A/e0_A) correction_norm=norm(correction)

    # 5. Post-smooth
    smoother(A, x_c, b_test)
    mul!(tmp, A, x_c)
    r_f  = b_test - tmp
    ef_A = a_norm(x_exact - x_c)
    @info "V-cycle step 5 – post-smooth" ratio_2norm=(norm(r_f)/norm_b) ratio_Anorm=(ef_A/e0_A)

    # Full V-cycle via ldiv!
    x_v = zeros(n)
    ldiv!(x_v, prec, b_test)
    mul!(tmp, A, x_v)
    ev_A = a_norm(x_exact - x_v)
    @info "Full V-cycle (ldiv!)" ratio_2norm=(norm(b_test-tmp)/norm_b) ratio_Anorm=(ev_A/e0_A)

    # Prolongator partition-of-unity check
    nc  = size(P, 2)
    err = norm(P * ones(nc) .- 1.0, Inf)
    @info "p-MG prolongator" size_P=size(P) constant_preservation_err=err

    # Standalone stationary V-cycle iteration: x_{k+1} = x_k + M^{-1}(b - Ax_k)
    @info "Standalone stationary V-cycle convergence (20 iterations):"
    x_si = zeros(n)
    Δx   = zeros(n)
    for k = 1:20
        mul!(tmp, A, x_si)
        r_k    = b_test .- tmp
        r_norm = norm(r_k)
        e_k_A  = a_norm(x_exact - x_si)
        @info "  iter" k rel_res=(r_norm/norm_b) rel_Anorm=(e_k_A/e0_A)
        fill!(Δx, 0.0)
        ldiv!(Δx, prec, r_k)
        x_si .+= Δx
    end
end

"""
    _inject_precs(krylov, precs_fn)

Reconstruct `krylov` (a `LinearSolve.KrylovJL`) with `precs_fn` injected, preserving
all other settings (algorithm, restart, window, verbosity, …).
"""
function _inject_precs(krylov::LinearSolve.KrylovJL, precs_fn)
    LinearSolve.KrylovJL(krylov.args...;
        KrylovAlg     = krylov.KrylovAlg,
        gmres_restart = krylov.gmres_restart,
        window        = krylov.window,
        precs         = precs_fn,
        krylov.kwargs...)
end

"""
    _lazy_precs(builder)

Return a `precs(A, p)` closure suitable for use with any `KrylovJL` solver.

A single `LazyPrecon` instance is captured.  On every `precs` call, if `A` is non-zero
the builder is invoked and the lazy wrapper's inner preconditioner is replaced.  This
allows `LinearSolve` to fix `cache.Pl::LazyPrecon` at `init` time while still receiving
the real (matrix-dependent) preconditioner at each solve.
"""
function _lazy_precs(builder)
    lazy = LazyPrecon()
    return (A, p = nothing) -> begin
        if !iszero(norm(A, Inf))
            lazy.inner = first(builder(A, p))
        end
        return (lazy, I)
    end
end

"""
Build a two-level `DofHandlerHierarchy` for polynomial multigrid on the same grid.

The coarse `DofHandler` mirrors `f.dh` (one `SubDofHandler` per cell type) but uses
polynomial order `p-1` for every field interpolation.  Mixed meshes are handled
transparently.
"""
function _build_pmg_dhh(f, field_name::Symbol)
    dh_fine = f.dh
    grid    = Ferrite.get_grid(dh_fine)

    dh_coarse = DofHandler(grid)
    for sdh in dh_fine.subdofhandlers
        ip_fine   = Ferrite.getfieldinterpolation(sdh, field_name)
        ip_coarse = Ferrite.getlowerorder(ip_fine)
        sdh_c     = SubDofHandler(dh_coarse, sdh.cellset)
        Ferrite.add!(sdh_c, field_name, ip_coarse)
    end
    Ferrite.close!(dh_coarse)

    # index 1 = coarsest, index 2 = finest (= f.dh)
    return DofHandlerHierarchy([dh_coarse, dh_fine])
end

"""
Build a `DofHandlerHierarchy` for geometric multigrid.

`dh_fine` is reused verbatim as the finest g-MG level, preserving its exact grid object
and DOF ordering.  Coarser levels are created on `gh.grids[k]` mirroring the SubDofHandler
structure of `dh_fine`: each SDH's cell set is mapped to coarser levels by composing the
`gh.fine2coarse` maps from finest down to the target level.  This preserves the DOF
numbering and field interpolation for each region exactly as in `dh_fine`.

**Critical for `ChainedMGPrecon`**: pass the p-MG coarse DH (i.e. `dhh_pmg[1]`) as
`dh_fine`.  The p-MG algorithm produces `A_pmg_coarse = P_p' * A_fine * P_p` indexed by
`dhh_pmg[1]`'s DOF numbering.  If the g-MG fine-level DH were a freshly constructed
`DofHandler` on the same grid, its cellset would be built via `Set(1:N)` → hash-order
iteration → a permuted `OrderedSet` → a *different* DOF numbering.  The Galerkin product
`P_g' * A_pmg_coarse * P_g` would then mix incompatible index spaces, producing a
structurally degenerate coarse matrix.  Reusing the exact same DH object eliminates this
risk by construction.

For standalone `GMGPrecon`, pass `f.dh` directly.
"""
function _build_gmg_dhh(dh_fine::DofHandler, field_name::Symbol, gh::GridHierarchy)
    n_levels = length(gh)  # gh.grids[1] = coarsest, gh.grids[n_levels] = finest

    # Collect per-SDH (interpolation, cell set on the finest grid).
    # cell sets are converted to plain Set{Int} for efficient map application.
    sdh_ips   = [Ferrite.getfieldinterpolation(sdh, field_name) for sdh in dh_fine.subdofhandlers]
    sdh_cells = [sdh.cellset                          for sdh in dh_fine.subdofhandlers]

    # Build coarse DH for levels 1..n_levels-1, working from finest-1 down to 1.
    # gh.fine2coarse[k] maps cell ids on grids[k+1] → cell ids on grids[k].
    coarse_handlers = Vector{DofHandler}(undef, n_levels - 1)
    for k in (n_levels - 1):-1:1
        f2c = gh.fine2coarse[k]
        # Propagate each cell set one level coarser.
        sdh_cells = [OrderedSet{Int}(f2c[c] for c in cells) for cells in sdh_cells]

        dh = DofHandler(gh.grids[k])
        for (ip, cells) in zip(sdh_ips, sdh_cells)
            isempty(cells) && continue
            sdh_c = SubDofHandler(dh, cells)
            Ferrite.add!(sdh_c, field_name, ip)
        end
        Ferrite.close!(dh)
        coarse_handlers[k] = dh
    end

    return DofHandlerHierarchy([coarse_handlers..., dh_fine])
end
# -----------------------------------------------------------------------
# PMGStationaryAlgorithm — standalone V-cycle iterative solver
# -----------------------------------------------------------------------

"""
    PMGStationaryAlgorithm

A `LinearSolve.SciMLLinearSolveAlgorithm` that drives convergence by repeated V-cycle
applications (stationary iteration):

    x_{k+1} = x_k + M^{-1}(b - A·x_k)

`build_prec(A)` is called at each `solve!` to construct a fresh multigrid preconditioner
from the current Jacobian.  The iteration stops when the relative 2-norm residual drops
below `reltol` or `maxiter` cycles are exhausted.
"""
struct PMGStationaryAlgorithm <: LinearSolve.SciMLLinearSolveAlgorithm
    build_prec::Any    # A::SparseMatrixCSC → ldiv!-able V-cycle preconditioner
    reltol::Float64
    maxiter::Int
    verbose::Bool
end

function LinearSolve.init_cacheval(
    ::PMGStationaryAlgorithm, A, b, u, Pl, Pr,
    maxiters::Int, abstol, reltol, verbose::Bool,
    assumptions::LinearSolve.OperatorAssumptions,
)
    return nothing
end

function LinearSolve.solve!(
    cache::LinearSolve.LinearCache,
    alg::PMGStationaryAlgorithm;
    kwargs...,
)
    A = cache.A
    b = cache.b
    x = cache.u
    fill!(x, 0.0)

    norm_b = norm(b)
    if norm_b < eps(eltype(b))
        return LinearSolve.SciMLBase.build_linear_solution(
            alg, x, nothing, cache; retcode = LinearSolve.ReturnCode.Success)
    end

    prec = alg.build_prec(A)

    tmp = similar(x)
    Δx  = similar(x)
    for k = 1:alg.maxiter
        mul!(tmp, A, x)
        tmp .= b .- tmp
        rel = norm(tmp) / norm_b
        alg.verbose && @info "PMG stationary iter" k rel_res=rel
        rel < alg.reltol && break
        fill!(Δx, 0.0)
        ldiv!(Δx, prec, tmp)
        x .+= Δx
    end

    return LinearSolve.SciMLBase.build_linear_solution(
        alg, x, nothing, cache; retcode = LinearSolve.ReturnCode.Success)
end

# -----------------------------------------------------------------------
# _materialize_inner_solver — PMGPrecon
# -----------------------------------------------------------------------

function Thunderbolt._materialize_inner_solver(
    f,
    solver::KrylovMGSolver{<:Any, <:PMGPrecon},
)
    isempty(f.dh.subdofhandlers) &&
        error("DofHandler has no subdofhandlers; cannot build p-MG hierarchy.")

    field_name = first(Ferrite.getfieldnames(f.dh))
    mg  = solver.mg

    cycle        = mg.cycle        === nothing ? _default_cycle()      : mg.cycle
    pgrid_config = mg.pgrid_config === nothing ? _default_pmg_config() : mg.pgrid_config

    dhh = _build_pmg_dhh(f, field_name)

    pcoarse_solver = mg.pcoarse_solver === nothing ?
        CachedLinearSolveCoarseSolverBuilder(UMFPACKFactorization()) : mg.pcoarse_solver

    # If the user supplied custom smoother callables, bake them into a fixed-smoother
    # PMultigridPreconBuilder (same pattern as GMGPrecon / ChainedMGPrecon).
    # Otherwise build an adaptive-ω Jacobi builder that re-estimates λ_max(D⁻¹A) for
    # every new Jacobian — necessary because the spectral radius changes across Newton steps.
    builder = if mg.presmoother !== nothing
        pre  = mg.presmoother
        post = mg.postsmoother === nothing ? mg.presmoother : mg.postsmoother
        PMultigridPreconBuilder(
            dhh, pgrid_config;
            cycle          = cycle,
            pcoarse_solver = pcoarse_solver,
            presmoother    = pre,
            postsmoother   = post,
            symmetry       = AMG.NoSymmetry(),
        )
    else
        # Default: 2-iteration damped Jacobi with per-Newton adaptive ω.
        # The PMultigridPreconBuilder is created once (geometry cached); each call to
        # builder(A, p) re-estimates λ_max(D⁻¹A) inside setup_smoother and rebuilds
        # only the numeric phase (RAP + smoother).
        smoother = DampedJacobi(5)
        PMultigridPreconBuilder(
            dhh, pgrid_config;
            cycle          = cycle,
            pcoarse_solver = pcoarse_solver,
            presmoother    = smoother,
            postsmoother   = smoother,
            symmetry       = AMG.NoSymmetry(),
        )
    end

    return _inject_precs(solver.krylov, _lazy_precs(builder))
end

function Thunderbolt._materialize_inner_solver(
    f,
    solver::KrylovMGSolver{<:Any, <:GMGPrecon},
)
    isempty(f.dh.subdofhandlers) &&
        error("DofHandler has no subdofhandlers; cannot build g-MG hierarchy.")

    field_name = first(Ferrite.getfieldnames(f.dh))
    mg  = solver.mg

    gconfig = mg.gconfig === nothing ? _default_gmg_config() : mg.gconfig
    dhh     = _build_gmg_dhh(f.dh, field_name, mg.gh)

    pcoarse_solver = mg.pcoarse_solver === nothing ?
        CachedLinearSolveCoarseSolverBuilder(UMFPACKFactorization()) : mg.pcoarse_solver

    pre  = mg.presmoother  === nothing ? DampedJacobi(5) : mg.presmoother
    post = mg.postsmoother === nothing ? DampedJacobi(5) : mg.postsmoother

    geo_gmg_ref = Ref{Any}(nothing)
    builder = (A, p = nothing) -> begin
        A_csc = SparseMatrixCSC(A)
        if geo_gmg_ref[] === nothing
            geo_gmg_ref[] = gmultigrid_symbolic(mg.gh, dhh, gconfig, A_csc)
        end
        ml = gmultigrid_numeric!(geo_gmg_ref[], A_csc, mg.gh, dhh, nothing, gconfig, pcoarse_solver;
                        presmoother = pre, postsmoother = post, symmetry = AMG.NoSymmetry())
        return (AlgebraicMultigrid.aspreconditioner(ml), I)
    end
    return _inject_precs(solver.krylov, _lazy_precs(builder))
end

# -----------------------------------------------------------------------
# _materialize_inner_solver — ChainedMGPrecon
# -----------------------------------------------------------------------

function Thunderbolt._materialize_inner_solver(
    f,
    solver::KrylovMGSolver{<:Any, <:ChainedMGPrecon},
)
    isempty(f.dh.subdofhandlers) &&
        error("DofHandler has no subdofhandlers; cannot build chained MG hierarchy.")

    field_name = first(Ferrite.getfieldnames(f.dh))
    mg  = solver.mg
    pmg = mg.pmg
    gmg = mg.gmg

    cycle        = pmg.cycle        === nothing ? _default_cycle()      : pmg.cycle
    pgrid_config = pmg.pgrid_config === nothing ? _default_pmg_config() : pmg.pgrid_config
    gconfig      = gmg.gconfig      === nothing ? _default_gmg_config() : gmg.gconfig

    dhh_pmg = _build_pmg_dhh(f, field_name)
    # Reuse dhh_pmg[1] as the g-MG fine level: the p-MG Galerkin matrix A_pmg_coarse is
    # indexed by dhh_pmg[1]'s DOF numbering.  Reusing the exact same DH object (not a new
    # one on the same grid) eliminates any risk of DOF-ordering mismatch.
    dhh_gmg = _build_gmg_dhh(dhh_pmg[1], field_name, gmg.gh)

    gmg_pcoarse = gmg.pcoarse_solver === nothing ?
        CachedLinearSolveCoarseSolverBuilder(UMFPACKFactorization()) : gmg.pcoarse_solver

    # G-MG smoothers used at the coarse correction levels.
    pre_gmg  = gmg.presmoother  === nothing ? DampedJacobi(5) : gmg.presmoother
    post_gmg = gmg.postsmoother === nothing ? pre_gmg         : gmg.postsmoother

    # For Galerkin g-MG: geometry (including RAP workspaces) is built on first call with A.
    gmg_geo_ref = Ref{Any}(nothing)

    gmg_coarse_solver = (A::SparseMatrixCSC) -> begin
        if gmg_geo_ref[] === nothing
            gmg_geo_ref[] = gmultigrid_symbolic(gmg.gh, dhh_gmg, gconfig, A)
        end
        ml = gmultigrid_numeric!(gmg_geo_ref[], A, gmg.gh, dhh_gmg, nothing, gconfig, gmg_pcoarse;
                        presmoother = pre_gmg, postsmoother = post_gmg,
                        symmetry = AMG.NoSymmetry())
        return GMultigridCoarseSolver(ml)
    end

    # P-MG smoothers (fine-level smoothing); fall back to 2-iteration damped Jacobi when not set.
    pre_pmg  = pmg.presmoother
    post_pmg = pmg.postsmoother === nothing ? pmg.presmoother : pmg.postsmoother

    builder = if pre_pmg !== nothing
        PMultigridPreconBuilder(
            dhh_pmg, pgrid_config;
            cycle          = cycle,
            pcoarse_solver = gmg_coarse_solver,
            presmoother    = pre_pmg,
            postsmoother   = post_pmg,
        )
    else
        smoother = DampedJacobi(5)
        PMultigridPreconBuilder(
            dhh_pmg, pgrid_config;
            cycle          = cycle,
            pcoarse_solver = gmg_coarse_solver,
            presmoother    = smoother,
            postsmoother   = smoother,
        )
    end
    return _inject_precs(solver.krylov, _lazy_precs(builder))
end

end # module ThunderboltFerriteMultigridExt
