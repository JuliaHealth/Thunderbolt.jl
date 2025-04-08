#########################################
## CUDA L1 Gauss Seidel Preconditioner ##
#########################################

Adapt.adapt(::CUDABackend, A::CUSPARSE.AbstractCuSparseMatrix) = A
Adapt.adapt(::CUDABackend,A::AbstractSparseMatrix) = A |> cu
Adapt.adapt(::CUDABackend, x::Vector) = x |> cu
Adapt.adapt(::CUDABackend, x::CuVector) = x

# For some reason, these properties are not automatically defined for Device Arrays, 
# so we need to define them, so that we can call these attributes generically regardless the device backend.  
SparseArrays.rowvals(A::CUSPARSE.CuSparseDeviceMatrixCSC{Tv,Ti,1}) where {Tv,Ti} = A.rowVal
SparseArrays.getcolptr(A::CUSPARSE.CuSparseDeviceMatrixCSC{Tv,Ti,1}) where {Tv,Ti} = A.colPtr
SparseArrays.getnzval(A::CUSPARSE.CuSparseDeviceMatrixCSC{Tv,Ti,1}) where {Tv,Ti} = A.nzVal

SparseMatricesCSR.colvals(A::CUSPARSE.CuSparseDeviceMatrixCSR{Tv,Ti,1}) where {Tv,Ti} = A.colVal
SparseMatricesCSR.getrowptr(A::CUSPARSE.CuSparseDeviceMatrixCSR{Tv,Ti,1}) where {Tv,Ti} = A.rowPtr
SparseMatricesCSR.getnzval(A::CUSPARSE.CuSparseDeviceMatrixCSR{Tv,Ti,1}) where {Tv,Ti} = A.nzVal



Thunderbolt.sparsemat_format_type(::CUSPARSE.CuSparseDeviceMatrixCSC{Tv,Ti,1}) where {Tv,Ti} = CSCFormat
Thunderbolt.sparsemat_format_type(::CUSPARSE.CuSparseDeviceMatrixCSR{Tv,Ti,1}) where {Tv,Ti} = CSRFormat
