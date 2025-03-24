####################################
## L1 Gauss Seidel Preconditioner ##
####################################

## Interfaces for the L1 Gauss Seidel preconditioner
abstract type AbstractPartitioning end

struct L1Preconditioner{Partitioning,MatrixType}
    partitioning::Partitioning
    A::MatrixType
end


LinearSolve.ldiv!(::VectorType, ::L1Preconditioner{Partitioning}, ::VectorType) where {VectorType <: AbstractVector, Partitioning} = 
    error("Not implemented")

abstract type AbstractL1PrecBuilder end
struct CudaL1PrecBuilder <: AbstractL1PrecBuilder end

function build_l1prec(::AbstractL1PrecBuilder, ::AbstractMatrix)
    error("Not implemented")
end

(builder::AbstractL1PrecBuilder)(A::AbstractMatrix) = build_l1prec(builder, A)
