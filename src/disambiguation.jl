# Some dispatches to make the dispatcher happy
*(::ThreadedSparseMatrixCSR, ::ModelingToolkit.Symbolics.Arr{<:Any, 1}) = @error "Not implemented"
*(::ThreadedSparseMatrixCSR, ::SciMLBase.AbstractNoTimeSolution{T, 1} where T) = @error "Not implemented"
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

mul!(
    ::ModelingToolkit.ModelingToolkitBase.JumpProcesses.ExtendedJumpArray,
    ::ThreadedSparseMatrixCSR,
    ::AbstractVector{<:Number},
) = @error "Not implemented"
