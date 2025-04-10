module Preconditioners

using SparseArrays, SparseMatricesCSR
using LinearSolve
import LinearSolve: \
using Adapt
using UnPack
import KernelAbstractions: Backend, @kernel, @index, @ndrange, @groupsize, @print, functional,
    CPU,synchronize
import SparseArrays: getcolptr,getnzval
import SparseMatricesCSR: getrowptr,getnzval


## Generic Code #

# CSR and CSC are exact the same in symmetric matrices,so we need to hold symmetry info
# in order to be exploited in cases in which one format has better access pattern than the other.
abstract type AbstractMatrixSymmetry end
struct SymmetricMatrix <: AbstractMatrixSymmetry end 
struct NonSymmetricMatrix <: AbstractMatrixSymmetry end

abstract type AbstractMatrixFormat end
struct CSRFormat <: AbstractMatrixFormat end
struct CSCFormat <: AbstractMatrixFormat end


Adapt.adapt(::CPU, A::AbstractSparseMatrix) = A
Adapt.adapt(::CPU, x::Vector) = x

sparsemat_format_type(::SparseMatrixCSC) = CSCFormat
sparsemat_format_type(::SparseMatrixCSR) = CSRFormat



include("l1_gauss_seidel.jl")

export L1GSPrecBuilder

end
