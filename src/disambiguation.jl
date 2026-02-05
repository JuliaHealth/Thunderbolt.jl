# Some dispatches to make the dispatcher happy
*(::ThreadedSparseMatrixCSR, ::Symbolics.Arr{<:Any, 1}) = @error "Not implemented"
mul!(
    ::JumpProcesses.ExtendedJumpArray,
    ::ThreadedSparseMatrixCSR,
    ::AbstractVector{<:Number},
) = @error "Not implemented"
