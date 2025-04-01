#########################################
## CUDA L1 Gauss Seidel Preconditioner ##
#########################################

Thunderbolt.convert_to_backend(A::CUSPARSE.AbstractCuSparseMatrix, ::CUDABackend) = A
Thunderbolt.convert_to_backend(A::AbstractSparseMatrix, ::CUDABackend) = A |> cu
Thunderbolt.convert_to_backend(x::Vector, ::CUDABackend) = x |> cu
Thunderbolt.convert_to_backend(x::CuVector, ::CUDABackend) = x


Thunderbolt.sparsemat_format_type(::CUSPARSE.CuSparseDeviceMatrixCSC{Tv,Ti,1}) where {Tv,Ti} = CSC
Thunderbolt.sparsemat_format_type(::CUSPARSE.CuSparseDeviceMatrixCSR{Tv,Ti,1}) where {Tv,Ti} = CSR
