module Preconditioners

using SparseArrays, SparseMatricesCSR
using LinearSolve
import LinearSolve: \
using Adapt
using UnPack
import KernelAbstractions: Backend, @kernel, @index, @ndrange, @groupsize, @print, functional,
    CPU,synchronize,GPU
import SparseArrays: getcolptr,getnzval
import SparseMatricesCSR: getnzval
import LinearAlgebra: Symmetric

## Generic Code #

# CSR and CSC are exact the same in symmetric matrices,so we need to hold symmetry info
# in order to be exploited in cases in which one format has better access pattern than the other.
abstract type AbstractMatrixSymmetry end
struct SymmetricMatrix <: AbstractMatrixSymmetry end 
struct NonSymmetricMatrix <: AbstractMatrixSymmetry end

abstract type AbstractMatrixFormat end
struct CSRFormat <: AbstractMatrixFormat end
struct CSCFormat <: AbstractMatrixFormat end

# Why using these traits?
# Since we are targeting multiple backends, but unfortunately, all the sparse matrix CSC/CSR on all
# backends don't share the same supertype (e.g. AbstractSparseMatrixCSC/AbstractSparseMatrixCSR)
# e.g. CUSPARSE.CuSparseDeviceMatrixCSC <:SparseArrays.AbstractSparseMatrixCSC → false
# So we need to define our own traits to identify the format of the sparse matrix
sparsemat_format_type(::SparseMatrixCSC) = CSCFormat()
sparsemat_format_type(::SparseMatrixCSR) = CSRFormat()

# Why? because we want to circumvent piracy when extending these functions for device backend (e.g. CuSparseDeviceMatrixCSR)
# TODO: find a more robust solution to dispatch the correct function
colvals(A::SparseMatrixCSR) = SparseMatricesCSR.colvals(A)
getrowptr(A::SparseMatrixCSR) = SparseMatricesCSR.getrowptr(A)

include("l1_gauss_seidel.jl")

export L1GSPrecBuilder,CPUSetting,GPUSetting

end
