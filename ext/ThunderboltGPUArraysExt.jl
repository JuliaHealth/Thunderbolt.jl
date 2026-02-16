module ThunderboltGPUArraysExt

using Thunderbolt
using GPUArrays

import Thunderbolt.Preconditioners: Preconditioners, CSCFormat, CSRFormat

Preconditioners.colvals(A::GPUSparseDeviceMatrixCSR) = A.colVal
Preconditioners.getrowptr(A::GPUSparseDeviceMatrixCSR) = A.rowPtr

Preconditioners.sparsemat_format_type(::GPUSparseDeviceMatrixCSR) = CSCFormat()
Preconditioners.sparsemat_format_type(::GPUSparseDeviceMatrixCSC) = CSRFormat()

end
