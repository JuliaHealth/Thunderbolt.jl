using SparseArrays
using CUDA
using Thunderbolt
using LinearSolve

N = 4
A = spdiagm(0 => 2 * ones(N), -1 => -ones(N-1), 1 => -ones(N-1))

cudal1prec=  Thunderbolt.CudaL1PrecBuilder()
P = cudal1prec(A;n_threads=1, n_blocks=1)
x = eltype(A).(collect(0:N-1))
y = x
LinearSolve.ldiv!(y, P, x)
