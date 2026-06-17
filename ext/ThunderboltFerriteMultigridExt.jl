# The formatter messes up for this file.
#! format: off
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
AMG.adjoint(P::LazyPrecon) = P  # symmetric smoother, adjoint ≈ self

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
        smoother = AMG.Jacobi(0.5, iter=5)
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

    pre  = mg.presmoother  === nothing ? AMG.Jacobi(0.5, iter=5) : mg.presmoother
    post = mg.postsmoother === nothing ? AMG.Jacobi(0.5, iter=5) : mg.postsmoother

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
    pre_gmg  = gmg.presmoother  === nothing ? AMG.Jacobi(0.5, iter=5) : gmg.presmoother
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
        smoother = AMG.Jacobi(0.5, iter=5)
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
#! format: on
