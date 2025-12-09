#########################################
## CUDA L1 Gauss Seidel Preconditioner ##
#########################################

# workaround for the issue with SparseMatricesCSR
# TODO: find a more robust solution to dispatch the correct function
Preconditioners.colvals(A::CUSPARSE.CuSparseDeviceMatrixCSR{Tv, Ti, 1}) where {Tv, Ti} = A.colVal
Preconditioners.getrowptr(A::CUSPARSE.CuSparseDeviceMatrixCSR{Tv, Ti, 1}) where {Tv, Ti} = A.rowPtr

Preconditioners.sparsemat_format_type(
    ::CUSPARSE.CuSparseDeviceMatrixCSC{Tv, Ti, 1},
) where {Tv, Ti} = CSCFormat()
Preconditioners.sparsemat_format_type(
    ::CUSPARSE.CuSparseDeviceMatrixCSR{Tv, Ti, 1},
) where {Tv, Ti} = CSRFormat()
