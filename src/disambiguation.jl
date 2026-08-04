# Some dispatches to make the dispatcher happy.
# The two ModelingToolkit-typed ones live in `ThunderboltMTKExt`.
*(::ThreadedSparseMatrixCSR, ::SciMLBase.AbstractNoTimeSolution{T, 1} where {T}) =
    @error "Not implemented"
*(A::ThreadedSparseMatrixCSR, v::BlockArrays.FillArrays.AbstractZeros{<:Any, 1}) = mul(A, v)
*(A::ThreadedSparseMatrixCSR, v::BlockArrays.ArrayLayouts.LayoutVector) = mul(A, v)
*(
    A::ThreadedSparseMatrixCSR,
    v::DynamicQuantities.QuantityArray{T, 1, D, Q, V},
) where {
    T,
    D <: DynamicQuantities.AbstractDimensions,
    Q <: DynamicQuantities.UnionAbstractQuantity{T, D},
    V <: AbstractVector{T},
} = mul(A, v)
*(
    A::ThreadedSparseMatrixCSR,
    v::DynamicQuantities.QuantityArray{T, 2, D, Q, V},
) where {
    T,
    D <: DynamicQuantities.AbstractDimensions,
    Q <: DynamicQuantities.UnionAbstractQuantity{T, D},
    V <: AbstractMatrix{T},
} = mul(A, v)

function Ferrite.start_assemble(strategy::FerriteOperators.AbstractAssemblyStrategy, J::BlockMatrix, residual::AbstractVector; fillzero::Bool=true)
    FerriteOperators.strategy_needs_atomic(strategy) && @warn "Assembling into BlockMatrix without atomics with a strategy that might be not thread-safe. Results might be corrupted."
    Ferrite.start_assemble(J, residual; fillzero)
end
