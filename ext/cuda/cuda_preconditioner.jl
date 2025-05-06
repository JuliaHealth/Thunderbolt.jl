#########################################
## CUDA L1 Gauss Seidel Preconditioner ##
#########################################

# PIRACY ALERT: the following code is commented out to avoid piracy
# SparseMatricesCSR.colvals(A::CUSPARSE.CuSparseDeviceMatrixCSR{Tv,Ti,1}) where {Tv,Ti} = A.colVal
# SparseMatricesCSR.getrowptr(A::CUSPARSE.CuSparseDeviceMatrixCSR{Tv,Ti,1}) where {Tv,Ti} = A.rowPtr

# workaround for the issue with SparseMatricesCSR
# TODO: find a more robust solution to dispatch the correct function
Preconditioners.colvals(A::CUSPARSE.CuSparseDeviceMatrixCSR{Tv,Ti,1}) where {Tv,Ti} = A.colVal
Preconditioners.getrowptr(A::CUSPARSE.CuSparseDeviceMatrixCSR{Tv,Ti,1}) where {Tv,Ti} = A.rowPtr

Preconditioners.sparsemat_format_type(::CUSPARSE.CuSparseDeviceMatrixCSC{Tv,Ti,1}) where {Tv,Ti} = CSCFormat()
Preconditioners.sparsemat_format_type(::CUSPARSE.CuSparseDeviceMatrixCSR{Tv,Ti,1}) where {Tv,Ti} = CSRFormat()

function Preconditioners.default_device_config(::CUDABackend)
    dev = device()
    nblocks = CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT) # no. SMs
    nthreads = convert(typeof(nblocks),256)
    return nblocks,nthreads
end
