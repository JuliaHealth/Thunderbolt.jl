using SparseArrays
using CUDA
using Thunderbolt
using LinearSolve
using SparseMatricesCSR

N = 8
A = spdiagm(0 => 2 * ones(N), -1 => -ones(N-1), 1 => -ones(N-1))

cudal1prec =  Thunderbolt.CudaL1PrecBuilder()
P = cudal1prec(A; n_threads=2, n_blocks=1)
x = eltype(A).(collect(0:N-1))
y = x
LinearSolve.ldiv!(y, P, x)


B = SparseMatrixCSR(A)
x = eltype(A).(collect(0:N-1))
y = x
P = cudal1prec(B; n_threads=2, n_blocks=1)
LinearSolve.ldiv!(y, P, x)
## TODO: Add tests for the above code snippet



abstract type AbstractMatrixSymmetry end

struct SymmetricMatrix <: AbstractMatrixSymmetry end
struct NonSymmetricMatrix <: AbstractMatrixSymmetry end

struct DeviceDiagonalIterator{MatrixType, MatrixSymmetry <: AbstractMatrixSymmetry}
    A::MatrixType
end

matrix_symmetry_type(A::AbstractSparseMatrix)  = isapprox(A, A',rtol=1e-12) ? SymmetricMatrix : NonSymmetricMatrix

DiagonalIterator(A::MatrixType) where {MatrixType} = DeviceDiagonalIterator{MatrixType,matrix_symmetry(A)}(A)
