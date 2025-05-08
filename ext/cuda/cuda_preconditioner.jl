#########################################
## CUDA L1 Gauss Seidel Preconditioner ##
#########################################

# workaround for the issue with SparseMatricesCSR
# TODO: find a more robust solution to dispatch the correct function
Preconditioners.colvals(A::CUSPARSE.CuSparseDeviceMatrixCSR{Tv,Ti,1}) where {Tv,Ti} = A.colVal
Preconditioners.getrowptr(A::CUSPARSE.CuSparseDeviceMatrixCSR{Tv,Ti,1}) where {Tv,Ti} = A.rowPtr

Preconditioners.sparsemat_format_type(::CUSPARSE.CuSparseDeviceMatrixCSC{Tv,Ti,1}) where {Tv,Ti} = CSCFormat()
Preconditioners.sparsemat_format_type(::CUSPARSE.CuSparseDeviceMatrixCSR{Tv,Ti,1}) where {Tv,Ti} = CSRFormat()

function Preconditioners.default_device_config(::CUDABackend)
    dev = device()
    nblocks = CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT) # no. SMs
    # CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR
    nthreads = CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR) # no. threads per SM
    return nblocks,nthreads
end
