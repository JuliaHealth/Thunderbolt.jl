residual_norm(cache::AbstractNonlinearSolverCache, f::AbstractSemidiscreteFunction) =
    norm(cache.residual)
residual_norm(cache::AbstractNonlinearSolverCache, f::AbstractSolidMechanicsFunction) =
    norm(cache.residual[Ferrite.free_dofs(getch(f))])
residual_norm(cache::AbstractNonlinearSolverCache, f::NullFunction, i::Block) = 0.0
residual_norm(cache::AbstractNonlinearSolverCache, f::NullFunction) = 0.0

# Through `getJ` rather than `op.J`, so that an operator which contributes terms of its own --
# `InertiaStageOperator` adds the inertia -- can forward to the matrix it shares with the assembly.
# The operator is passed in rather than read off the cache: it belongs to the stage, which is the one
# thing that knows which nonlinear problem is being solved.
eliminate_constraints_from_linearization!(
    cache::AbstractNonlinearSolverCache,
    op,
    f::AbstractSemidiscreteFunction,
) = apply_zero!(getJ(op), cache.residual, getch(f))

eliminate_constraints_from_residual!(
    cache::AbstractNonlinearSolverCache,
    f::AbstractSemidiscreteFunction,
) = apply_zero!(cache.residual, getch(f))
eliminate_constraints_from_increment!(
    Δu::AbstractVector,
    f::AbstractSemidiscreteFunction,
    cache::AbstractNonlinearSolverCache,
) = apply_zero!(Δu, getch(f))
eliminate_constraints_from_increment!(
    Δu::AbstractVector,
    f::NullFunction,
    cache::AbstractNonlinearSolverCache,
) = nothing
