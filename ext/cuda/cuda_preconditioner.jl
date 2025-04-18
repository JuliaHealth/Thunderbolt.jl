#########################################
## CUDA L1 Gauss Seidel Preconditioner ##
#########################################

# PIRACY ALERT: this code exhibit piratic nature because both `adapt` and its arguments are foreign objects.
# Therefore, `adapt` behavior is going to be different depending on whether `Thunderbolt` and `CuThunderboltExt` are loaded or not.
# Reference: https://juliatesting.github.io/Aqua.jl/stable/piracies/
# Note: the problem is with `AbstractSparseMatrix` as the default behavior of `adapt` is to return the same object, whatever the backend is.
# Adapt.adapt(::CUDABackend, A::CUSPARSE.AbstractCuSparseMatrix) = A
# Adapt.adapt(::CUDABackend,A::AbstractSparseMatrix) = A |> cu
# Adapt.adapt(::CUDABackend, x::Vector) = x |> cu # not needed 
# Adapt.adapt(::CUDABackend, x::CuVector) = x # not needed

# TODO: remove this function if back compatibility is not needed
Preconditioners.convert_to_backend(::CUDABackend, A::AbstractSparseMatrix) = adapt(CUDABackend(), A)


# For some reason, these properties are not automatically defined for Device Arrays, 
# TODO: remove the following code when https://github.com/JuliaGPU/CUDA.jl/pull/2738 is merged
#SparseArrays.rowvals(A::CUSPARSE.CuSparseDeviceMatrixCSC{Tv,Ti,1}) where {Tv,Ti} = A.rowVal
#SparseArrays.getcolptr(A::CUSPARSE.CuSparseDeviceMatrixCSC{Tv,Ti,1}) where {Tv,Ti} = A.colPtr
#SparseArrays.getnzval(A::CUSPARSE.CuSparseDeviceMatrixCSC{Tv,Ti,1}) where {Tv,Ti} = A.nzVal
#SparseMatricesCSR.getnzval(A::CUSPARSE.CuSparseDeviceMatrixCSR{Tv,Ti,1}) where {Tv,Ti} = A.nzVal

# PIRACY ALERT: the following code is commented out to avoid piracy
# SparseMatricesCSR.colvals(A::CUSPARSE.CuSparseDeviceMatrixCSR{Tv,Ti,1}) where {Tv,Ti} = A.colVal
# SparseMatricesCSR.getrowptr(A::CUSPARSE.CuSparseDeviceMatrixCSR{Tv,Ti,1}) where {Tv,Ti} = A.rowPtr

# workaround for the issue with SparseMatricesCSR
# TODO: find a more robust solution to dispatch the correct function
Preconditioners.colvals(A::CUSPARSE.CuSparseDeviceMatrixCSR{Tv,Ti,1}) where {Tv,Ti} = A.colVal
Preconditioners.getrowptr(A::CUSPARSE.CuSparseDeviceMatrixCSR{Tv,Ti,1}) where {Tv,Ti} = A.rowPtr

Preconditioners.sparsemat_format_type(::CUSPARSE.CuSparseDeviceMatrixCSC{Tv,Ti,1}) where {Tv,Ti} = CSCFormat
Preconditioners.sparsemat_format_type(::CUSPARSE.CuSparseDeviceMatrixCSR{Tv,Ti,1}) where {Tv,Ti} = CSRFormat
