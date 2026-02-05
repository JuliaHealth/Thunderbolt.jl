#########################################
## CUDA L1 Gauss Seidel Preconditioner ##
#########################################

using CUDA.GPUArrays: GPUSparseDeviceMatrixCSR, GPUSparseDeviceMatrixCSC

# workaround for the issue with SparseMatricesCSR
# TODO: find a more robust solution to dispatch the correct function
Preconditioners.colvals(M::GPUSparseDeviceMatrixCSR) = M.colVal
Preconditioners.getrowptr(M::GPUSparseDeviceMatrixCSR) = M.rowPtr

Preconditioners.sparsemat_format_type(::GPUSparseDeviceMatrixCSC) = CSCFormat()
Preconditioners.sparsemat_format_type(::GPUSparseDeviceMatrixCSR) = CSRFormat()
