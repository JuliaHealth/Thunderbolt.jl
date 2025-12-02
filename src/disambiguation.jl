# Some dispatches to make the dispatcher happy
*(::ThreadedSparseMatrixCSR, ::ModelingToolkit.Symbolics.Arr{<:Any, 1}) = @error "Not implemented"
mul!(::ModelingToolkit.JumpProcesses.ExtendedJumpArray, ::ThreadedSparseMatrixCSR, ::AbstractVector{<:Number})= @error "Not implemented"
