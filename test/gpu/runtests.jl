using Test
using CUDA

# This suite's device surface (KernelAbstractionsDevice) needs the FerriteOperators the package
# compat names (0.4.2); this environment's Project.toml nevertheless points `[sources]` at a local
# checkout. Run with `julia --project=test/gpu test/gpu/runtests.jl`.
#
# This environment doubles as the environment of `benchmarks/`: the benchmarks are CUDA-dependent
# for the same reason this suite is, and they run as `julia --project=test/gpu benchmarks/<f>.jl`.
# `FerriteOperatorsExampleElements` and `Serialization` are here for them -- nothing under `test/`
# consumes either.
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
