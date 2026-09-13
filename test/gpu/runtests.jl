using Test
using CUDA

# This suite needs a FerriteOperators newer than the registered 0.4.0 -- the KernelAbstractionsDevice
# GPU surface below is unreleased (do/gpu) -- which is why `[sources]` in this environment's
# Project.toml points FerriteOperators at a local checkout; run with
# `julia --project=test/gpu test/gpu/runtests.jl`.
#
# Everything below needs a device. Without one the suite reports that it did nothing rather than
# failing, so it can be included unconditionally by a runner that does not know the machine.
if !CUDA.functional()
    @warn "CUDA is not functional here -- skipping the Thunderbolt GPU test suite."
else
    using Thunderbolt
    using Ferrite
    using LinearSolve
    using LinearAlgebra
    using OrdinaryDiffEqOperatorSplitting
    using SparseArrays
    using StaticArrays

    import Adapt: adapt
    import FerriteOperators: KernelAbstractionsDevice
    import KernelAbstractions as KA
    import KernelAbstractions: @kernel, @index, @Const
    import Thunderbolt: num_states, solution_size, ThreadedSparseMatrixCSR
    import Thunderbolt:
        AssemblyStrategy,
        ColoredScheduling,
        FullAssembly,
        SequentialCPUDevice,
        StandardOperatorSpecification,
        TimeIntegrationContext

    const CuCSC = CUDA.CUSPARSE.CuSparseMatrixCSC{Float32, Int32}
    const CuCSR = CUDA.CUSPARSE.CuSparseMatrixCSR{Float32, Int32}

    """
        device_assembly_strategy(; matrix_type = nothing)

    The assembly strategy the device arms below hand to `FiniteElementDiscretization`: a
    `KernelAbstractionsDevice` over the CUDA backend, in the precision a device solve runs in.
    Coloring is not a tuning choice -- Ferrite's device matrix assembler accumulates without atomics,
    so `FerriteOperators` rejects any other scheduling for a device.
    """
    device_assembly_strategy(; matrix_type = nothing) = AssemblyStrategy(
        FullAssembly(StandardOperatorSpecification(; matrix_type)),
        ColoredScheduling(),
        KernelAbstractionsDevice(
            CUDABackend();
            value_type = Float32,
            index_type = Int32,
            items_per_worker = 2,
            max_workgroup_size = 256,
        ),
    )

    "The host reference the device arms are compared against, at matched precision."
    host_assembly_strategy() = AssemblyStrategy(SequentialCPUDevice{Float32, Int32}())

    @testset "Thunderbolt GPU" begin
        include("test_coefficients.jl")
        include("test_operators.jl")
        include("test_assembly.jl")
        include("test_pointwise.jl")
        include("test_diffusion.jl")
        include("test_split.jl")
        include("test_emrkc.jl")
    end
end
