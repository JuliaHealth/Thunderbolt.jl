# The heat subproblem on its own, which is the path a device system matrix takes: the operators are
# assembled on the host and mirrored into a `CuSparseMatrix`, and everything the backward Euler stage
# does afterwards -- combining the mass and diffusion nonzeros, `M uₙ₋₁`, the CG solve -- happens on
# the device. A stimulus is applied so the source operator is non-null and its host vector actually
# has to reach the device solution vector.

function _diffusion_form(; n = 32)
    mesh = generate_mesh(Quadrilateral, (n, n), Vec{2}((0.0, 0.0)), Vec{2}((2.5, 2.5)))
    cs = CartesianCoordinateSystem(mesh)
    model = TransientDiffusionModel(
        ConstantCoefficient(SymmetricTensor{2, 2, Float64}((4.5e-5, 0.0, 2.0e-5))),
        AnalyticalTransmembraneStimulationProtocol(
            AnalyticalCoefficient((x, t) -> exp(-norm(x - Vec((1.25, 1.25)))^2), cs),
            [SVector((0.0, 5.0))],
        ),
        :u,
    )
    return semidiscretize(
        model,
        FiniteElementDiscretization(Dict(:u => LagrangeCollection{1}())),
        mesh,
    )
end

@testset "Transient diffusion with a device system matrix" begin
    odefun = _diffusion_form()
    tspan  = (0.0f0, 1.0f0)
    Δt     = 0.1f0
    nsteps = 5

    u₀ = zeros(Float32, solution_size(odefun))
    u₀[1:2:end] .= 1.0f0

    inner_solver() = KrylovJL_CG(atol = 1.0f-10, rtol = 1.0f-8)

    cpu = init(
        Thunderbolt.ODEProblem(odefun, copy(u₀), tspan),
        BackwardEulerSolver(
            solution_vector_type = Vector{Float32},
            system_matrix_type   = ThreadedSparseMatrixCSR{Float32, Int32},
            inner_solver         = inner_solver(),
        );
        dt = Δt,
    )
    for _ = 1:nsteps
        step!(cpu)
    end
    @test cpu.u ≉ u₀

    # Both device formats are exercised, because they carry different assumptions: the CSR is built
    # by handing the CSC arrays over unchanged and is therefore the transpose (see
    # `create_system_matrix` in `ext/CuThunderboltExt.jl`), while the CSC is a faithful upload.
    # Agreeing with the host arm is what says the symmetry that licenses the former holds here.
    @testset "$SpMatType" for SpMatType in (
        CUDA.CUSPARSE.CuSparseMatrixCSR{Float32, Int32},
        CUDA.CUSPARSE.CuSparseMatrixCSC{Float32, Int32},
    )
        gpu = init(
            Thunderbolt.ODEProblem(odefun, CuVector(u₀), tspan),
            BackwardEulerSolver(
                solution_vector_type = CuVector{Float32},
                system_matrix_type   = SpMatType,
                inner_solver         = inner_solver(),
            );
            dt = Δt,
        )
        @test gpu.cache.stage.M isa Thunderbolt.MirroredBilinearOperator
        @test gpu.cache.stage.M.A isa SpMatType
        @test gpu.cache.stage.linear_solver.A isa SpMatType

        # `operator_payload` is what the inherited `Base.eltype`/`Base.size` (from
        # `AbstractBilinearOperator <: AbstractNonlinearOperator`) read; without it they `MethodError`.
        @test eltype(gpu.cache.stage.M) === eltype(gpu.cache.stage.M.A)
        @test size(gpu.cache.stage.M) === size(gpu.cache.stage.M.A)

        for _ = 1:nsteps
            step!(gpu)
        end
        # Same Float32 arithmetic on both arms; what separates them is the reduction order of the
        # two CG implementations. See the tolerance note in `test_split.jl`.
        @test Array(gpu.u) ≈ cpu.u rtol = 1.0f-5
    end
end

@testset "Host-assembled source into a device right hand side: zero per-step allocation" begin
    odefun = _diffusion_form()
    u₀ = zeros(Float32, solution_size(odefun))
    u₀[1:2:end] .= 1.0f0
    gpu = init(
        Thunderbolt.ODEProblem(odefun, CuVector(u₀), (0.0f0, 1.0f0)),
        BackwardEulerSolver(
            solution_vector_type = CuVector{Float32},
            system_matrix_type   = CUDA.CUSPARSE.CuSparseMatrixCSR{Float32, Int32},
            inner_solver         = KrylovJL_CG(atol = 1.0f-10, rtol = 1.0f-8),
        );
        dt = 0.1f0,
    )
    source = gpu.cache.stage.source_term
    @test source isa Thunderbolt.MirroredLinearOperator

    b = gpu.cache.stage.linear_solver.b
    Thunderbolt._add_source_term!(b, source, 0.1f0) # warmup
    @test CUDA.@allocated(Thunderbolt._add_source_term!(b, source, 0.1f0)) == 0

    ctx = TimeIntegrationContext(0.3f0, 0.0f0, 0.0f0)
    Thunderbolt.update_operator!(source, nothing, ctx) # warmup
    @test CUDA.@allocated(Thunderbolt.update_operator!(source, nothing, ctx)) == 0
end

@testset "Staleness guard: a source that changes mid-run" begin
    # `needs_update` is false at the start, true while `t` crosses the interval, false again
    # afterwards -- the buffer refresh has to track exactly that, on both arms.
    mesh = generate_mesh(Quadrilateral, (16, 16), Vec{2}((0.0, 0.0)), Vec{2}((2.5, 2.5)))
    cs = CartesianCoordinateSystem(mesh)
    model = TransientDiffusionModel(
        ConstantCoefficient(SymmetricTensor{2, 2, Float64}((4.5e-5, 0.0, 2.0e-5))),
        AnalyticalTransmembraneStimulationProtocol(
            AnalyticalCoefficient((x, t) -> exp(-norm(x - Vec((1.25, 1.25)))^2), cs),
            [SVector((0.3, 0.6))],
        ),
        :u,
    )
    odefun = semidiscretize(
        model,
        FiniteElementDiscretization(Dict(:u => LagrangeCollection{1}())),
        mesh,
    )

    tspan = (0.0f0, 1.0f0)
    Δt = 0.1f0
    nsteps = 10
    step_times = Δt .* (1:nsteps)
    active = 0.3f0 .<= step_times .<= 0.6f0
    @test any(active)   # the run actually reaches a step that needs the update ...
    @test any(.!active) # ... and steps before/after it that don't

    u₀ = zeros(Float32, solution_size(odefun))
    u₀[1:2:end] .= 1.0f0
    inner_solver() = KrylovJL_CG(atol = 1.0f-10, rtol = 1.0f-8)

    cpu = init(
        Thunderbolt.ODEProblem(odefun, copy(u₀), tspan),
        BackwardEulerSolver(
            solution_vector_type = Vector{Float32},
            system_matrix_type   = ThreadedSparseMatrixCSR{Float32, Int32},
            inner_solver         = inner_solver(),
        );
        dt = Δt,
    )
    gpu = init(
        Thunderbolt.ODEProblem(odefun, CuVector(u₀), tspan),
        BackwardEulerSolver(
            solution_vector_type = CuVector{Float32},
            system_matrix_type   = CUDA.CUSPARSE.CuSparseMatrixCSR{Float32, Int32},
            inner_solver         = inner_solver(),
        );
        dt = Δt,
    )
    @test gpu.cache.stage.source_term isa Thunderbolt.MirroredLinearOperator

    for _ = 1:nsteps
        step!(cpu)
        step!(gpu)
        @test Array(gpu.u) ≈ cpu.u rtol = 1.0f-4
    end
end

# The "Cross device vector addition" testset that stood here exercised
# `FerriteOperators.__add_to_vector!` methods added for `Vector`/`CuVector` in
# `ext/CuThunderboltExt.jl` — type piracy on FO's one private (double-underscore)
# name, unreachable from any real Thunderbolt path (`Ferrite.add!(::AbstractVector,
# ::AbstractLinearOperator)`, the only caller, is never invoked with mismatched
# host/device vectors anywhere in this package). Removed along with the piracy;
# no public path exercises the same cross-device add to rework the test onto.
