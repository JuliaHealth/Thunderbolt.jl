#########################################
## CUDA L1 Gauss Seidel Preconditioner ##
#########################################

# WARNING: This code exhibit piratic nature because both `adapt` and its arguments are foreign objects.
# Therefore, `adapt` behavior is going to be different depending on whether `Thunderbolt` and `CuThunderboltExt` are loaded or not.
# Reference: https://juliatesting.github.io/Aqua.jl/stable/piracies/
# Note: the problem is with `AbstractSparseMatrix` as the default behavior of `adapt` is to return the same object, whatever the backend is.
# Adapt.adapt(::CUDABackend, A::CUSPARSE.AbstractCuSparseMatrix) = A
# Adapt.adapt(::CUDABackend,A::AbstractSparseMatrix) = A |> cu
# Adapt.adapt(::CUDABackend, x::Vector) = x |> cu # not needed 
# Adapt.adapt(::CUDABackend, x::CuVector) = x # not needed

# workaround for the issue with adapt
Preconditioners.convert_to_backend(::CUDABackend, A::AbstractSparseMatrix) = A |> cu
Preconditioners.convert_to_backend(::CUDABackend, A::CUSPARSE.AbstractCuSparseMatrix) = A 

# For some reason, these properties are not automatically defined for Device Arrays, 
# so we need to define them, so that we can call these attributes generically regardless the device backend.  
SparseArrays.rowvals(A::CUSPARSE.CuSparseDeviceMatrixCSC{Tv,Ti,1}) where {Tv,Ti} = A.rowVal
SparseArrays.getcolptr(A::CUSPARSE.CuSparseDeviceMatrixCSC{Tv,Ti,1}) where {Tv,Ti} = A.colPtr
SparseArrays.getnzval(A::CUSPARSE.CuSparseDeviceMatrixCSC{Tv,Ti,1}) where {Tv,Ti} = A.nzVal

SparseMatricesCSR.colvals(A::CUSPARSE.CuSparseDeviceMatrixCSR{Tv,Ti,1}) where {Tv,Ti} = A.colVal
SparseMatricesCSR.getrowptr(A::CUSPARSE.CuSparseDeviceMatrixCSR{Tv,Ti,1}) where {Tv,Ti} = A.rowPtr
SparseMatricesCSR.getnzval(A::CUSPARSE.CuSparseDeviceMatrixCSR{Tv,Ti,1}) where {Tv,Ti} = A.nzVal



Preconditioners.sparsemat_format_type(::CUSPARSE.CuSparseDeviceMatrixCSC{Tv,Ti,1}) where {Tv,Ti} = CSCFormat
Preconditioners.sparsemat_format_type(::CUSPARSE.CuSparseDeviceMatrixCSR{Tv,Ti,1}) where {Tv,Ti} = CSRFormat
