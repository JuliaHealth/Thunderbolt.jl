# The emRKC step on the device, in the two shapes `test_split.jl` runs the split solvers in: host
# assembly mirrored into a `CuSparseMatrix`, and assembly on the device itself. All three arms run
# the same Float32 arithmetic on the same problem, so they may only differ in reduction order.
#
# The stimulus interval ends mid-run, so the per-outer-stage source refresh is exercised and then
# stops. The conductivity is the first EP tutorial's scaled by 1000: at the tutorial's value the fast
# sweep degenerates to a single stage, and only a multi-stage sweep rotates the inner stage buffers
# through device arrays.

import FerriteOperators

function _emrkc_monodomain_form(;
    n = 32,
    assembly_strategy = Thunderbolt.default_strategy(),
    qrcs = Dict{Symbol, Any}(),
    stim_until = 2.5,
)
    mesh = generate_mesh(Quadrilateral, (n, n), Vec{2}((0.0, 0.0)), Vec{2}((2.5, 2.5)))
    cs = CartesianCoordinateSystem(mesh)
    ep_model = MonodomainModel(
        ConstantCoefficient(1.0),
        ConstantCoefficient(1.0),
        ConstantCoefficient(SymmetricTensor{2, 2, Float64}((4.5e-2, 0.0, 2.0e-2))),
        AnalyticalTransmembraneStimulationProtocol(
            AnalyticalCoefficient(
                (x, t) -> 0.5 * exp(-norm(x - Vec((1.25, 1.25)))^2) * cospi(t / 5),
                cs,
            ),
            [SVector((0.0, stim_until))],
        ),
        Thunderbolt.ParametrizedFHNModel{Float32}(),
        cs,
        :φₘ,
        :s,
    )
    return semidiscretize(
        ReactionDiffusionSplit(ep_model),
        FiniteElementDiscretization(
            Dict(:φₘ => LagrangeCollection{1}());
            qrcs,
            assembly_strategy,
            mass = LumpedMass(),
        ),
        mesh,
    )
end

@testset "emRKC, host versus device" begin
    stim_until = 2.5
    odeform = _emrkc_monodomain_form(; stim_until)

    u₀ = create_initial_condition(odeform, Float32)
    setvariable!(u₀, odeform, :φₘ) do x
        (x[1] ≤ 1.25 && x[2] ≤ 1.25) ? 1.0f0 : 0.0f0
    end
    setvariable!(u₀, odeform, :s) do x
        x[2] ≥ 1.25 ? 0.1f0 : 0.0f0
    end

    tspan  = (0.0f0, 5.0f0)
    Δt     = 1.0f0
    nsteps = 5

    # The interval boundary falls strictly inside the run: the steps starting at 0, 1 and 2 refresh
    # the source, the ones starting at 3 and 4 lie beyond the window and must not.
    step_starts = Δt .* (0:(nsteps-1))
    @test any(≤(stim_until), step_starts) && any(>(stim_until), step_starts)

    build(form, u0, VT, SpMatType) = init(
        OperatorSplittingProblem(form, u0, tspan),
        EMRKC(solution_vector_type = VT, system_matrix_type = SpMatType);
        dt = Δt,
    )

    # As in `test_split.jl`: the model side's `assembly_strategy` picks which device assembles, the
    # device assembler exists for CSC only, and the quadrature collection elects element precision.
    devform = _emrkc_monodomain_form(;
        stim_until,
        assembly_strategy = device_assembly_strategy(),
        qrcs = Dict(:φₘ => QuadratureRuleCollection(Float32, 2)),
    )

    cpu = build(odeform, copy(u₀), Vector{Float32}, ThreadedSparseMatrixCSR{Float32, Int32})
    gpu = build(odeform, CuVector(u₀), CuVector{Float32}, CuCSR)
    gpu_assembled = build(devform, CuVector(u₀), CuVector{Float32}, CuCSC)

    # Host assembly mirrors, device assembly owns its matrix outright -- the same split as the
    # backward Euler stage in `test_split.jl`, here on the rate operator's diffusion matrix and the
    # source. The payload checks are what say the source really lives on the device.
    @test cpu.cache isa Thunderbolt.EMRKCCache
    @test cpu.cache.source_op isa Thunderbolt.LinearFerriteOperator
    @test gpu.cache.op isa Thunderbolt.RateOperator
    @test gpu.cache.op.op isa Thunderbolt.MirroredRateFormOperator
    @test gpu.cache.op.minv isa CuVector{Float32}
    @test gpu.cache.source_op isa Thunderbolt.MirroredLinearOperator
    @test FerriteOperators.operator_payload(gpu.cache.source_op) isa CuVector{Float32}
    let c = gpu_assembled.cache
        @test c.op.op isa FerriteOperators.RateFormFerriteOperator
        @test FerriteOperators.get_matrix(FerriteOperators.rate_form_rhs(c.op.op)) isa CuCSC
        @test c.op.minv isa CuVector{Float32}
        @test FerriteOperators.operator_payload(c.source_op) isa CuVector{Float32}
    end

    sizing(integ) = Thunderbolt._emrkc_step_sizing(integ.alg, Δt, integ.cache.ρS, integ.cache.ρF)

    φₘ = solution_variable(odeform, :φₘ)
    for step = 1:nsteps
        step!(cpu)
        step!(gpu)
        step!(gpu_assembled)

        if step == 1
            # The first step estimated both radii on every arm over its own operators. Those may
            # differ in reduction order, so the integer stage counts they round into are compared
            # rather than the raw radii; η is a function of the outer count alone and agrees exactly.
            s_cpu, η_cpu, m_cpu = sizing(cpu)
            @test m_cpu > 1 # the inner sweep really rotates its stage buffers (see header note)
            for integ in (gpu, gpu_assembled)
                @test isfinite(integ.cache.ρS) && integ.cache.ρS > 0
                @test isfinite(integ.cache.ρF) && integ.cache.ρF > 0
                s, η, m = sizing(integ)
                @test (s, m) == (s_cpu, m_cpu)
                @test η == η_cpu
            end
        end

        @test gpu.t == cpu.t
        @test gpu_assembled.t == cpu.t
        # Same tolerance and reasoning as `test_split.jl`: identical Float32 arithmetic at identical
        # stage counts, so what separates the arms is reduction order.
        @test getvariable(Array(gpu.u), φₘ) ≈ getvariable(cpu.u, φₘ) rtol = 1.0f-5
        @test getvariable(Array(gpu_assembled.u), φₘ) ≈ getvariable(cpu.u, φₘ) rtol = 1.0f-5
    end
    @test Array(gpu.u) ≈ cpu.u rtol = 1.0f-5
    @test Array(gpu_assembled.u) ≈ cpu.u rtol = 1.0f-5
    # The wave moved, so the agreement above is not three copies of the initial condition.
    @test cpu.u ≉ u₀
end
