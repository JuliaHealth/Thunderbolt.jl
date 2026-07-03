"""
    AbstractMGPrecon

Abstract supertype for multigrid preconditioner configuration structs.

The concrete subtypes `PMGPrecon`, `GMGPrecon`, and `ChainedMGPrecon` are available in
this package.  Their `_materialize_inner_solver` implementations (which build the actual
multigrid hierarchy) are provided by the `ThunderboltFerriteMultigridExt` extension when
both `Thunderbolt` and `FerriteMultigrid` are loaded together.
"""
abstract type AbstractMGPrecon end

"""
    PMGPrecon(; cycle, pcoarse_solver, pgrid_config, presmoother, postsmoother)

Polynomial multigrid preconditioner configuration for use with [`KrylovMGSolver`](@ref).

The polynomial hierarchy is built automatically at `setup_solver_cache` time from the
semidiscrete problem's `DofHandler`, handling mixed meshes transparently.

!!! note
    Requires `FerriteMultigrid` to be loaded.  The finite-element discretization must use
    polynomial order ≥ 2 (e.g. `LagrangeCollection{2}()^3`).

# Keyword arguments
- `cycle`          – AMG cycle type (default: `AMG.V()`)
- `pcoarse_solver` – coarse-grid solver factory (default: `SmoothedAggregationCoarseSolver()`)
- `pgrid_config`   – polynomial MG configuration (default: `pmultigrid_config()`)
- `presmoother`    – smoother callable `(A, x, b) -> nothing` applied before the coarse
                     correction (default: `nothing` → damped Jacobi with Chebyshev-optimal ω)
- `postsmoother`   – smoother callable `(A, x, b) -> nothing` applied after the coarse
                     correction (default: `nothing` → same as `presmoother`)
"""
struct PMGPrecon{CY, CS, PC, PRE, POST} <: AbstractMGPrecon
    cycle::CY
    pcoarse_solver::CS
    pgrid_config::PC
    presmoother::PRE
    postsmoother::POST
end

# Bare constructor with `nothing` placeholders — real defaults set by the extension.
PMGPrecon(;
    cycle = nothing,
    pcoarse_solver = nothing,
    pgrid_config = nothing,
    presmoother = nothing,
    postsmoother = nothing,
) = PMGPrecon(cycle, pcoarse_solver, pgrid_config, presmoother, postsmoother)

"""
    GMGPrecon(gh; gconfig, pcoarse_solver, presmoother, postsmoother)

Geometric multigrid preconditioner configuration for use with [`KrylovMGSolver`](@ref).

The user provides a `GridHierarchy` (from FerriteMultigrid); the `DofHandlerHierarchy`
and `ConstraintHandlerHierarchy` are built automatically from the problem's `DofHandler`.

!!! note
    Requires `FerriteMultigrid` to be loaded.  Geometric coarsening via
    `uniform_refinement` only supports pure Hexahedron or Tetrahedron meshes.  For mixed
    meshes use `PMGPrecon` alone or [`ChainedMGPrecon`](@ref).

# Keyword arguments
- `gconfig`        – geometric MG configuration (default: `gmultigrid_config()`)
- `pcoarse_solver` – coarse-grid solver factory (default: UMFPACK direct solver)
- `presmoother`    – smoother callable `(A, x, b) -> nothing` applied before the coarse
                     correction (default: `nothing` → `AlgebraicMultigrid.GaussSeidel()`)
- `postsmoother`   – smoother callable `(A, x, b) -> nothing` applied after the coarse
                     correction (default: `nothing` → same as `presmoother`)
"""
struct GMGPrecon{GH, GC, CS, PRE, POST} <: AbstractMGPrecon
    gh::GH
    gconfig::GC
    pcoarse_solver::CS
    presmoother::PRE
    postsmoother::POST
end

GMGPrecon(
    gh;
    gconfig = nothing,
    pcoarse_solver = nothing,
    presmoother = nothing,
    postsmoother = nothing,
) = GMGPrecon(gh, gconfig, pcoarse_solver, presmoother, postsmoother)

"""
    ChainedMGPrecon(pmg::PMGPrecon, gmg::GMGPrecon)

Chains polynomial multigrid (fine levels) with geometric multigrid as the coarse solver.

!!! note
    Requires `FerriteMultigrid` to be loaded.
"""
struct ChainedMGPrecon{PM <: PMGPrecon, GM <: GMGPrecon} <: AbstractMGPrecon
    pmg::PM
    gmg::GM
end

"""
    KrylovMGSolver(krylov, mg_config; maxiters = nothing)
    KrylovMGSolver(mg_config; maxiters = nothing)

Pairs a Krylov solver algorithm with a multigrid preconditioner configuration.

When `FerriteMultigrid` is also loaded, the `DofHandlerHierarchy` (and optionally
`ConstraintHandlerHierarchy`) needed by the preconditioner are built automatically from
the semidiscrete problem's `DofHandler`—no manual hierarchy construction is required.

The `precs` callback of the `krylov` algorithm is replaced by the multigrid preconditioner;
all other settings (restart length, verbosity, …) are preserved.

When called with a single argument the krylov solver defaults to
`KrylovJL_GMRES(gmres_restart=400)`.

The `maxiters` keyword caps the maximum number of Krylov iterations per linear solve.
When `nothing` (the default), LinearSolve uses `length(b)` as the limit.

!!! warning
    `KrylovJL_FGMRES` is currently **not supported**: a bug in LinearSolve.jl prevents
    the left preconditioner from being passed to Krylov.jl's `fgmres!`.  Use
    `KrylovJL_GMRES` (with an appropriate `gmres_restart`) instead.

# Example
```julia
using Thunderbolt, LinearSolve, FerriteMultigrid

timestepper = HomotopyPathSolver(
    NewtonRaphsonSolver(
        max_iter     = 10,
        inner_solver = KrylovMGSolver(
            KrylovJL_GMRES(gmres_restart = 400, verbose = 1),
            GMGPrecon(gh;
                presmoother  = (A, x, b) -> gauss_seidel!(x, A, b),
                postsmoother = (A, x, b) -> gauss_seidel!(x, A, b),
            );
            maxiters = 100,
        ),
    )
)
```

!!! note
    Polynomial multigrid (`PMGPrecon`) requires the finite-element discretization to use
    polynomial order ≥ 2.  Change `LagrangeCollection{1}()` to `LagrangeCollection{2}()`
    in your `FiniteElementDiscretization` to enable p-MG.
"""
struct KrylovMGSolver{KA, MG <: AbstractMGPrecon}
    krylov::KA
    mg::MG
    maxiters::Union{Int, Nothing}
end

KrylovMGSolver(krylov, mg::AbstractMGPrecon; maxiters = nothing) =
    KrylovMGSolver(krylov, mg, maxiters)

KrylovMGSolver(mg::AbstractMGPrecon; maxiters = nothing) =
    KrylovMGSolver(LinearSolve.KrylovJL_GMRES(gmres_restart = 400), mg, maxiters)

"""
    _linear_maxiters(solver) -> Union{Int, Nothing}

Return the maximum number of Krylov iterations for the linear solve, or `nothing` to use
LinearSolve's default (`length(b)`).

Override this for custom solver types that carry a `maxiters` setting.
"""
_linear_maxiters(_) = nothing
_linear_maxiters(solver::KrylovMGSolver) = solver.maxiters

"""
    _materialize_inner_solver(f, solver)

Extension hook called inside `setup_solver_cache` to allow `KrylovMGSolver` to build
the actual `LinearSolve` algorithm (with the `precs` callable) from the semidiscrete
function `f`.

The default implementation is a no-op that returns `solver` unchanged.  The
`ThunderboltFerriteMultigridExt` extension overrides this for concrete `AbstractMGPrecon`
subtypes.
"""
_materialize_inner_solver(_, solver) = solver
