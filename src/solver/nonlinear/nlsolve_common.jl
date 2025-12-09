residual_norm(cache::AbstractNonlinearSolverCache, f::AbstractSemidiscreteFunction) =
    norm(cache.residual)
residual_norm(cache::AbstractNonlinearSolverCache, f::AbstractQuasiStaticFunction) =
    norm(cache.residual[Ferrite.free_dofs(getch(f))])
residual_norm(cache::AbstractNonlinearSolverCache, f::NullFunction, i::Block) = 0.0
residual_norm(cache::AbstractNonlinearSolverCache, f::NullFunction) = 0.0

eliminate_constraints_from_linearization!(
    cache::AbstractNonlinearSolverCache,
    f::AbstractSemidiscreteFunction,
) = apply_zero!(cache.op.J, cache.residual, getch(f))
eliminate_constraints_from_increment!(
    Δu::AbstractVector,
    f::AbstractSemidiscreteFunction,
    cache::AbstractNonlinearSolverCache,
) = apply_zero!(Δu, getch(f))
function eliminate_constraints_from_increment!(
    Δu::AbstractVector,
    f::AbstractSemidiscreteBlockedFunction,
    cache::AbstractNonlinearSolverCache,
)
    # TODO be smarter about the block extraction and use info from f
    Δublocked = BlockedVector(Δu, [blocksizes(f)...])
    for (i, fi) ∈ enumerate(blocks(f))
        Δublockedi = @view Δublocked[Block(i)]
        eliminate_constraints_from_increment!(Δublockedi, fi, cache)
    end
end
eliminate_constraints_from_increment!(
    Δu::AbstractVector,
    f::NullFunction,
    cache::AbstractNonlinearSolverCache,
) = nothing

function eliminate_constraints_from_linearization!(
    cache::AbstractNonlinearSolverCache,
    f::AbstractSemidiscreteBlockedFunction,
)
    for (i, _) ∈ enumerate(blocks(f))
        eliminate_constraints_from_linearization_blocked!(cache, problem, Block(i))
    end
end

function eliminate_constraints_from_linearization_blocked!(
    cache::AbstractNonlinearSolverCache,
    f::AbstractSemidiscreteBlockedFunction,
    i_::Block,
)
    @assert length(i_.n) == 1
    i = i_.n[1]
    fi = blocks(f)[i]
    hasproperty(fi, :ch) || return nothing
    ch = getch(fi)
    # TODO optimize this
    for j = 1:length(blocks(f))
        if i == j
            jacobian_block = getJ(cache.op, Block((i, i)))
            # Eliminate diagonal entry only
            residual_block = @view cache.residual[i_]
            apply_zero!(jacobian_block, residual_block, ch)
        else
            # Eliminate rows
            jacobian_block = getJ(cache.op, Block((i, j)))
            jacobian_block[ch.prescribed_dofs, :] .= 0.0
            # Eliminate columns
            jacobian_block = getJ(cache.op, Block((j, i)))
            jacobian_block[:, ch.prescribed_dofs] .= 0.0
        end
    end

    return nothing
end
