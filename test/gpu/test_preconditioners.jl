using SparseArrays
using CUDA
using Thunderbolt
using LinearSolve
using SparseMatricesCSR
using KernelAbstractions

N = 8
A = spdiagm(0 => 2 * ones(N), -1 => -ones(N-1), 1 => -ones(N-1))

cudal1prec =  Thunderbolt.GpuL1PrecBuilder(CUDABackend())
P = cudal1prec(A,2,1)
x = eltype(A).(collect(0:N-1))
y = x
LinearSolve.ldiv!(y, P, x)


B = SparseMatrixCSR(A)
x = eltype(A).(collect(0:N-1))
y = x
P = cudal1prec(B; n_threads=2, n_blocks=1)
LinearSolve.ldiv!(y, P, x)



cpul1builder = Thunderbolt.CpuL1PrecBuilder()
P = cpul1builder(A; partsize=2)
x = eltype(A).(collect(0:N-1))
y = x
LinearSolve.ldiv!(y, P, x)

## TODO: Add tests for the above code snippet
