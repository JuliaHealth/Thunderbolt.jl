module CuThunderboltExt

import LinearAlgebra

# CUDA support is limited to what this extension declares:
#   * the pointwise cell-model solve, whose outer loop becomes a CUDA kernel launch;
#   * `CuVector`/`CuSparseMatrix` system allocation for the solver interface;
#   * mirroring a host-assembled bilinear operator into a device matrix;
#   * moving a coefficient's coordinate vector to the device.
# Assembly on the GPU goes through `FerriteOperators`' device seam, not through here.

using Thunderbolt

import CUDA:
    CUDA, CuArray, CuVector, CUSPARSE, blockDim, blockIdx, threadIdx, @cuda, launch_configuration

import Thunderbolt:
    AbstractSemidiscreteFunction,
    AbstractPointwiseFunction,
    AbstractPointwiseSolverCache,
    AbstractBilinearIntegrator,
    AbstractLinearIntegrator,
    AbstractCPUDevice,
    AssemblyStrategy,
    FullAssembly,
    SequentialScheduling,
    MirroredBilinearOperator,
    MirroredLinearOperator,
    num_states,
    solution_size

import FerriteOperators

import Ferrite: AbstractDofHandler

import SparseArrays: SparseMatrixCSC

##########################
## Pointwise solvers
##########################

# `npoints` is passed in rather than read off a cache field: the outer kernel below already knows it
# from the pointwise function, and a launch sized from the cache would tie every pointwise cache in
# Thunderbolt to one field name.
function _gpu_pointwise_step_inner_kernel_wrapper!(
    f, t, Δt, cache::AbstractPointwiseSolverCache, npoints,
)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    i > npoints && return nothing
    Thunderbolt._pointwise_step_inner_kernel!(f, i, t, Δt, cache)
    return nothing
end

# This controls the outer loop over the ODEs
function Thunderbolt._pointwise_step_outer_kernel!(
    f::AbstractPointwiseFunction,
    t::Real,
    Δt::Real,
    cache::AbstractPointwiseSolverCache,
    ::Union{<:CuVector, SubArray{<:Any, 1, <:CuVector}},
)
    npoints = length(f.associated_states) ÷ num_states(f.ode)
    kernel = @cuda launch=false _gpu_pointwise_step_inner_kernel_wrapper!(
        f.ode, t, Δt, cache, npoints,
    )
    config = launch_configuration(kernel.fun)
    threads = min(npoints, config.threads)
    blocks = cld(npoints, threads)
    kernel(f.ode, t, Δt, cache, npoints; threads, blocks)
    return true
end

##########################
## System allocation
##########################

Thunderbolt.create_system_vector(::Type{<:CuVector{T}}, f::AbstractSemidiscreteFunction) where {T} = CUDA.zeros(T, solution_size(f))
Thunderbolt.create_system_vector(::Type{<:CuVector{T}}, dh::DofHandler) where {T}                  = CUDA.zeros(T, ndofs(dh))

function Thunderbolt.create_system_matrix(
    ::Type{<:CUSPARSE.CuSparseMatrixCSC{Tv, Ti}},
    dh::AbstractDofHandler,
) where {Tv, Ti}
    Acsc = convert(SparseMatrixCSC{Tv, Ti}, allocate_matrix(dh))
    return CUSPARSE.CuSparseMatrixCSC{Tv, Ti}(
        CuVector{Ti}(Acsc.colptr),
        CuVector{Ti}(Acsc.rowval),
        CuVector{Tv}(Acsc.nzval),
        size(Acsc),
    )
end

# The CSC arrays are handed to the CSR type unchanged, so what this returns is the CSR of `Aᵀ`, and
# every operator assembled into it is assumed symmetric -- the same assumption
# `create_system_matrix(::Type{<:ThreadedSparseMatrixCSR}, dh)` in `src/solver/interface.jl` carries,
# for the same reason: what the affine backward Euler stage combines is the entrywise `nonzeros`
# correspondence between this matrix and the operators mirrored into the same pattern, and CUSPARSE's
# faithful CSC→CSR conversion would reorder the entries and break it. The two have to be changed
# together.
function Thunderbolt.create_system_matrix(
    ::Type{<:CUSPARSE.CuSparseMatrixCSR{Tv, Ti}},
    dh::AbstractDofHandler,
) where {Tv, Ti}
    Acsc = convert(SparseMatrixCSC{Tv, Ti}, allocate_matrix(dh))
    return CUSPARSE.CuSparseMatrixCSR{Tv, Ti}(
        CuVector{Ti}(Acsc.colptr),
        CuVector{Ti}(Acsc.rowval),
        CuVector{Tv}(Acsc.nzval),
        reverse(size(Acsc)),
    )
end

##########################
## Host assembly, device A
##########################

function Thunderbolt.setup_assembled_operator(
    strategy::AssemblyStrategy{<:FullAssembly, SequentialScheduling, <:AbstractCPUDevice},
    integrator::AbstractBilinearIntegrator,
    system_matrix_type::Type{<:Union{CUSPARSE.CuSparseMatrixCSC, CUSPARSE.CuSparseMatrixCSR}},
    dh::AbstractDofHandler;
    kwargs...,
)
    return MirroredBilinearOperator(
        Thunderbolt.setup_operator(strategy, integrator, dh; kwargs...),
        Thunderbolt.create_system_matrix(system_matrix_type, dh),
    )
end

# A host-assembled rate form: the rhs mirrors as any bilinear operator, the diagonal inverse mass
# rides a device vector. A dense per-cell mass has no mirror.
function Thunderbolt.setup_assembled_operator(
    strategy::AssemblyStrategy{<:FullAssembly, SequentialScheduling, <:AbstractCPUDevice},
    integrator::FerriteOperators.BilinearRateFormIntegrator,
    system_matrix_type::Type{<:Union{CUSPARSE.CuSparseMatrixCSC, CUSPARSE.CuSparseMatrixCSR}},
    dh::AbstractDofHandler;
    kwargs...,
)
    host = Thunderbolt.setup_operator(strategy, integrator, dh; kwargs...)
    minv = FerriteOperators.rate_form_inverse_mass(host)
    minv isa LinearAlgebra.Diagonal || error(
        "A host-assembled rate form mirrors onto a device only with a diagonal mass " *
        "(`LumpedMass()`/`CollocatedMass()`); a dense per-cell mass has no mirror. Assemble on the " *
        "device instead.",
    )
    rhs = MirroredBilinearOperator(
        FerriteOperators.rate_form_rhs(host),
        Thunderbolt.create_system_matrix(system_matrix_type, dh),
    )
    T = eltype(rhs.A)
    return Thunderbolt.MirroredRateFormOperator(
        host, rhs, CuVector{T}(minv.diag), CuVector{T}(undef, length(minv.diag)),
    )
end

# The vector-side counterpart, for a host-assembled source feeding a device-resident right hand
# side: mirror its load vector into a persistent device buffer instead of uploading fresh every
# `update_operator!`.
function Thunderbolt.setup_assembled_operator(
    strategy::AssemblyStrategy{<:FullAssembly, SequentialScheduling, <:AbstractCPUDevice},
    integrator::AbstractLinearIntegrator,
    solution_vector_type::Type{<:CuVector},
    dh::AbstractDofHandler,
)
    return MirroredLinearOperator(
        Thunderbolt.setup_operator(strategy, integrator, dh),
        Thunderbolt.create_system_vector(solution_vector_type, dh),
    )
end

##########################
## Cross device vectors
##########################

Thunderbolt.adapt_vector_type(::Type{<:CuVector}, v::VT) where {VT <: Vector} = CuVector(v)

# Adds a source operator's device-resident payload into `b`: direct for a device-assembled
# source, and via its `MirroredLinearOperator` wrapper's refreshed buffer for a host-assembled
# one. `LinearNullOperator` stays a no-op.
Thunderbolt._add_source_term!(b::CuVector, source::FerriteOperators.LinearNullOperator) = b
function Thunderbolt._add_source_term!(b::CuVector, source::FerriteOperators.AbstractLinearOperator)
    b .+= FerriteOperators.operator_payload(source)
    return b
end

end
