KernelAbstractions.functional(::CUDABackend) = CUDA.functional()

# remove once PR (https://github.com/JuliaGPU/CUDA.jl/pull/2720) is merged
CUDA.CUSPARSE.CuSparseMatrixCSR{T}(Mat::SparseMatrixCSR) where {T} =
           CUDA.CUSPARSE.CuSparseMatrixCSR{T}(CuVector{Cint}(Mat.rowptr), CuVector{Cint}(Mat.colval),
                                CuVector{T}(Mat.nzval), size(Mat))
