# Per step cost of the monodomain operator splitting solve of the first EP tutorial, on the host
# against the device.
#
# What this measures, and why each number is here:
#
#  * wall time per outer Lie-Trotter-Godunov step, host against device, on a grid large enough that
#    the linear solve dominates. The problem is memory bound, so a modest speedup is the expected
#    outcome; the number that would be a defect is a slowdown.
#  * host allocations per step on the device arm. A device solve that keeps allocating on the host
#    every step is paying for a transfer somewhere, and the split's index sets are the usual place:
#    `@views parent.u[idxs]` uploads a `Vector{Int}` index set to the device on every access, once
#    per child per sync. The `gathered` arm below is the same problem with the heat child's index set
#    forced back to a `Vector{Int}`, which is what that costs.
#  * whether `OrdinaryDiffEqOperatorSplitting`'s `need_sync` elides the forward and backward copies
#    between parent and child. It does exactly when the child's buffer is a `SubArray` of the
#    parent's -- which is what indexing a host vector gives, and what indexing a device vector with a
#    contiguous range does *not*: `GPUArrays` derives a new `CuArray` over the same memory instead,
#    and `need_sync` cannot see the aliasing.
#  * the two device solve shapes against each other: host assembly mirrored into a device matrix, and
#    assembly on the device. On the tutorial's time independent coefficients both assemble once at
#    setup, so the per step numbers separate the *setup* cost, not the step cost -- which is why the
#    setup time is reported beside it, and why the assembly throughput below is measured on its own.
#  * assembly throughput of the mass and diffusion forms, sequential against threaded against the
#    device, at a mesh size where the launch overhead is amortized. These kernels evaluate a
#    coefficient per quadrature point, so the numbers are not comparable to a plain element benchmark.
#
# CUDA is a weak dependency, so this runs in the GPU test environment rather than the package one:
# `julia --project=test/gpu benchmarks/benchmark-gpu-split.jl`.

using Thunderbolt
using CUDA
using LinearSolve
using OrdinaryDiffEqOperatorSplitting
using SparseArrays

import FerriteOperators: KernelAbstractionsDevice
import Thunderbolt: ThreadedSparseMatrixCSR
import Thunderbolt:
    AssemblyStrategy, ColoredScheduling, FullAssembly, PolyesterDevice, SequentialCPUDevice,
    StandardOperatorSpecification, TimeIntegrationContext
import OrdinaryDiffEqOperatorSplitting: GenericSplitFunction, need_sync

# `sync` and the Jacobi preconditioner, shared with the other benchmarks in this directory. Nothing
# here uses `build`/`prewarm!`/`measure!` from it: this file times differently (see
# `measure_with_allocations!`) and builds its own splitting integrator below.
include(joinpath(@__DIR__, "common.jl"))

const N       = 256
const NASM    = 512   # assembly throughput mesh: 262144 cells, enough to amortize the launch
const NSTEPS  = 20
const NBLOCKS = 5
const NSETUP  = 3
const NASMREP = 10
const Δt      = 1.0f0
const CuCSC   = CUDA.CUSPARSE.CuSparseMatrixCSC{Float32, Int32}
const CuCSR   = CUDA.CUSPARSE.CuSparseMatrixCSR{Float32, Int32}

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

function monodomain_form(
    n;
    assembly_strategy = Thunderbolt.default_strategy(),
    qrcs = Dict{Symbol, Any}(),
)
    mesh = generate_mesh(Quadrilateral, (n, n), Vec{2}((0.0, 0.0)), Vec{2}((2.5, 2.5)))
    ep_model = MonodomainModel(
        ConstantCoefficient(1.0),
        ConstantCoefficient(1.0),
        ConstantCoefficient(SymmetricTensor{2, 2, Float64}((4.5e-5, 0.0, 2.0e-5))),
        NoStimulationProtocol(),
        Thunderbolt.ParametrizedFHNModel{Float32}(),
        CartesianCoordinateSystem(mesh),
        :φₘ,
        :s,
    )
    return semidiscretize(
        ReactionDiffusionSplit(ep_model),
        FiniteElementDiscretization(
            Dict(:φₘ => LagrangeCollection{1}());
            qrcs,
            assembly_strategy,
        ),
        mesh,
    )
end

# The heat index set as a `Vector{Int}`, i.e. the shape the split emitted before it reported
# contiguous runs as ranges.
gathered_indices(form) = GenericSplitFunction(
    form.functions,
    (collect(form.solution_indices[1]), form.solution_indices[2]),
    form.synchronizers,
)

function initial_condition(form)
    u₀ = create_initial_condition(form, Float32)
    setvariable!(u₀, form, :φₘ) do x
        (x[1] ≤ 1.25 && x[2] ≤ 1.25) ? 1.0f0 : 0.0f0
    end
    setvariable!(u₀, form, :s) do x
        x[2] ≥ 1.25 ? 0.1f0 : 0.0f0
    end
    return u₀
end

function build_split_integrator(form, u0, VT, SpMatType)
    timestepper = LieTrotterGodunov((
        BackwardEulerSolver(
            solution_vector_type = VT,
            system_matrix_type   = SpMatType,
            inner_solver         = KrylovJL_CG(atol = 1.0f-8, rtol = 1.0f-6),
        ),
        AdaptiveForwardEulerSubstepper(
            solution_vector_type = VT,
            reaction_threshold   = 0.1f0,
        ),
    ))
    return init(
        OperatorSplittingProblem(form, u0, (0.0f0, Float32(1000 * Δt))),
        timestepper;
        dt = Δt,
    )
end

"""
Minimum per step wall time over `NBLOCKS` blocks of `NSTEPS` steps, and the host allocations one
block of steps costs, divided by the step count. Named apart from `benchmarks/common.jl`'s
`measure!`, which reports the time alone: the host allocations are half of what this file is for.
"""
function measure_with_allocations!(integrator)
    for _ = 1:NSTEPS # warmup
        step!(integrator)
    end
    sync(integrator.u)

    best = Inf
    for _ = 1:NBLOCKS
        t0 = time_ns()
        for _ = 1:NSTEPS
            step!(integrator)
        end
        sync(integrator.u)
        best = min(best, (time_ns() - t0) / 1.0e9 / NSTEPS)
    end

    allocated = @allocated begin
        for _ = 1:NSTEPS
            step!(integrator)
        end
        sync(integrator.u)
    end

    return best, allocated / NSTEPS
end

pad(x, n) = rpad(string(x), n)

"""
Minimum wall time of `init`, which is where every arm assembles its operators, over `NSETUP` builds
after a warmup -- a single build is dominated by compilation and by whatever the collector does next.
The integrator returned is the last one, which is what the per step measurement then runs on.
"""
function setup_time(f)
    f()  # warmup, so the number is not a compilation time
    CUDA.synchronize()
    best = Inf
    integrator = nothing
    for _ = 1:NSETUP
        GC.gc()
        t0 = time_ns()
        integrator = f()
        CUDA.synchronize()
        best = min(best, (time_ns() - t0) / 1.0e9)
    end
    return integrator, best
end

####################################
## Assembly throughput
####################################

function assembly_operators(strategy, matrix_type)
    grid = generate_grid(Quadrilateral, (NASM, NASM), Vec{2}((0.0f0, 0.0f0)), Vec{2}((2.5f0, 2.5f0)))
    dh = DofHandler(grid)
    add!(dh, :φₘ, Lagrange{RefQuadrilateral, 1}())
    close!(dh)
    qrc = QuadratureRuleCollection(Float32, 2)
    solver = BackwardEulerSolver(
        solution_vector_type = Vector{Float32},
        system_matrix_type   = matrix_type,
    )
    mass = Thunderbolt.BilinearMassIntegrator(ConstantCoefficient(1.0f0), qrc, :φₘ)
    diffusion = Thunderbolt.BilinearDiffusionIntegrator(
        ConstantCoefficient(SymmetricTensor{2, 2, Float32}((4.5f-5, 0.0f0, 2.0f-5))),
        qrc,
        :φₘ,
    )
    return (
        Thunderbolt.setup_operator(strategy, mass, solver, dh),
        Thunderbolt.setup_operator(strategy, diffusion, solver, dh),
    )
end

function measure_assembly(strategy, matrix_type, on_device::Bool)
    operators = assembly_operators(strategy, matrix_type)
    ctx = TimeIntegrationContext(0.0f0, 0.0f0, 0.0f0)
    sweep() = for op in operators
        Thunderbolt.update_operator!(op, nothing, ctx)
    end

    sweep()
    on_device && CUDA.synchronize()
    best = Inf
    for _ = 1:NASMREP
        t0 = time_ns()
        sweep()
        on_device && CUDA.synchronize()
        best = min(best, (time_ns() - t0) / 1.0e9)
    end
    allocated = @allocated begin
        sweep()
        on_device && CUDA.synchronize()
    end
    return best, allocated
end

function assembly_benchmark()
    println(
        "\nAssembly of the mass and diffusion forms, ", NASM, " x ", NASM,
        " (", NASM^2, " cells), Float32, min of ", NASMREP, ":",
    )
    println("  ", pad("device", 22), pad("s/sweep", 14), pad("host B/sweep", 16), "speedup")
    arms = (
        ("sequential", AssemblyStrategy(SequentialCPUDevice{Float32, Int32}()), SparseMatrixCSC{Float32, Int32}, false),
        ("Polyester($(Threads.nthreads()))", AssemblyStrategy(PolyesterDevice{Float32, Int32}(32)), SparseMatrixCSC{Float32, Int32}, false),
        ("CUDA", device_assembly_strategy(), CuCSC, true),
    )
    baseline = nothing
    for (label, strategy, matrix_type, on_device) in arms
        t, allocs = measure_assembly(strategy, matrix_type, on_device)
        baseline === nothing && (baseline = t)
        println(
            "  ", pad(label, 22),
            pad(round(t; sigdigits = 4), 14),
            pad(round(Int, allocs), 16),
            round(baseline / t; digits = 2), "x",
        )
        GC.gc()
        CUDA.reclaim()
    end
    return nothing
end

####################################
## Split solve
####################################

function split_benchmark()
    form     = monodomain_form(N)
    # `value_type` above is the global system's; the element precision is elected here.
    devform = monodomain_form(
        N;
        assembly_strategy = device_assembly_strategy(),
        qrcs = Dict(:φₘ => QuadratureRuleCollection(Float32, 2)),
    )
    gathered = gathered_indices(form)
    u₀       = initial_condition(form)
    println(
        "Monodomain ", N, " x ", N,
        ", ", ndofs(form.functions[1].dh), " dofs, ",
        Thunderbolt.solution_size(form), " states, Δt = ", Δt,
    )

    # The last arm assembles on the device, which is also why its matrix type is CSC: Ferrite ships a
    # device assembler for that format only, while the mirrored arms above hand the CSR type the
    # solve prefers a copy of a host matrix.
    specs = (
        ("CPU Float32 CSR",  () -> build_split_integrator(form,     copy(u₀),     Vector{Float32},   ThreadedSparseMatrixCSR{Float32, Int32})),
        ("GPU host asm CSR", () -> build_split_integrator(form,     CuVector(u₀), CuVector{Float32}, CuCSR)),
        ("GPU gathered idx", () -> build_split_integrator(gathered, CuVector(u₀), CuVector{Float32}, CuCSR)),
        ("GPU device asm",   () -> build_split_integrator(devform,  CuVector(u₀), CuVector{Float32}, CuCSC)),
    )
    arms = map(spec -> (spec[1], setup_time(spec[2])...), specs)

    println("\nSynchronization elision between parent and child:")
    for (label, integrator, _) in arms
        parent = integrator.u
        for (i, child) in enumerate(integrator.child_subintegrators)
            slice = @views parent[integrator.child_solution_indices[i]]
            println(
                "  ", pad(label, 18), " child ", i,
                "  u::", pad(nameof(typeof(child.u)), 10),
                " slice::", pad(nameof(typeof(slice)), 10),
                " need_sync=", need_sync(child.u, slice),
            )
        end
    end

    println(
        "\n  ", pad("arm", 18), pad("s/step", 14), pad("host B/step", 14),
        pad("setup s", 12), "speedup",
    )
    baseline = nothing
    for (label, integrator, setup) in arms
        t, allocs = measure_with_allocations!(integrator)
        baseline === nothing && (baseline = t)
        println(
            "  ", pad(label, 18),
            pad(round(t; sigdigits = 4), 14),
            pad(round(Int, allocs), 14),
            pad(round(setup; sigdigits = 4), 12),
            round(baseline / t; digits = 2), "x",
        )
    end
    return nothing
end

function main()
    CUDA.functional() || error("This benchmark needs a functional CUDA device.")
    split_benchmark()
    GC.gc()
    CUDA.reclaim()
    assembly_benchmark()
    return nothing
end

# Only when this file is what was run: `include`ing it from a REPL session loads the definitions
# without starting the measurement.
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
