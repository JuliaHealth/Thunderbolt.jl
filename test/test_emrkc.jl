using Thunderbolt
using OrdinaryDiffEqOperatorSplitting
using DiffEqBase
using Test
using LinearAlgebra
using SparseArrays
using StaticArrays
import SciMLBase
import FerriteOperators

import Thunderbolt:
    ExponentialMultirateSTSAlgorithm,
    PassiveChildSolver,
    PassiveChildCache,
    _emrkc_step_sizing,
    sts_stage_count,
    sts_stability_boundary,
    gate_coefficients,
    gating_indices,
    cell_rhs!,
    num_states,
    needs_update,
    update_operator!,
    TimeIntegrationContext

#####################################################################
#  Problem fixtures                                                 #
#####################################################################

# A monodomain problem small enough to reference-implement, with a smooth always-active stimulus so
# that the source really is reassembled at every outer stage rather than elided by `needs_update`.
function emrkc_problem(;
    n = 4,
    ion = Thunderbolt.FHNModel(),
    κ = 1.0e-3,
    stimulate = true,
    tspan = (0.0, 1.0),
)
    grid = generate_grid(Quadrilateral, (n, n), Vec{2}((-1.0, -1.0)), Vec{2}((1.0, 1.0)))
    mesh = to_mesh(grid)
    cs = CartesianCoordinateSystem(mesh)
    stim = if stimulate
        AnalyticalTransmembraneStimulationProtocol(
            AnalyticalCoefficient((x, t) -> 0.5 * exp(-2 * norm(x)^2) * cospi(t), cs),
            [SVector((0.0, 1.0e6))], # always "on"
        )
    else
        NoStimulationProtocol()
    end
    model = MonodomainModel(
        ConstantCoefficient(1.0),
        ConstantCoefficient(1.0),
        ConstantCoefficient(SymmetricTensor{2, 2, Float64}((κ, 0.0, κ))),
        stim,
        ion,
        :φₘ,
        :s,
    )
    odeform = semidiscretize(
        ReactionDiffusionSplit(model),
        FiniteElementDiscretization(Dict(:φₘ => LagrangeCollection{1}())),
        mesh,
    )
    return OperatorSplittingProblem(odeform, emrkc_initial_state(odeform, ion), tspan), odeform
end

# The cell model's resting state everywhere, plus a smooth bump and one spike on the transmembrane
# potential. The spike gives the slow force's Jacobian a clearly dominant eigenvalue, which is what
# makes the ρ_S power iteration below a sharp test rather than a race between near-degenerate modes.
function emrkc_initial_state(odeform, ion)
    u₀ = zeros(Float64, solution_size(odeform))
    nV = length(odeform.solution_indices[1])
    rest = Thunderbolt.default_initial_state(ion)
    for k = 1:num_states(ion), j = 1:nV
        u₀[(k-1)*nV+j] = rest[k]
    end
    φoffset = (Thunderbolt.transmembranepotential_index(ion) - 1) * nV
    for j = 1:nV
        u₀[φoffset+j] += 0.4 * sinpi(j / nV) + 0.1 * cospi(3j / nV)
    end
    u₀[φoffset+1] += 1.0
    return u₀
end

#####################################################################
#  Naive reference implementation of Algorithm 3                    #
#####################################################################
# Written straight from the paper (Rosilho de Souza, Grote, Pezzuto & Krause, arXiv:2401.01745,
# Algorithm 3): direct, allocating, full width, with neither the V-row restriction nor the analytic
# finish the production path optimizes with. It shares with production only what a *model* and an
# *assembly* are -- `gate_coefficients`, `cell_rhs!` and the assembled operators -- and reimplements
# every piece of the scheme itself, stage coefficients and stage count law included.
#
# The three placements a wrong wiring gets wrong are all visible here, and none of them is visible
# in a convergence test: the exponential is taken with `η` (not Δt) once per OUTER stage, the inner
# sweep starts at `y_E`, and the difference quotient is taken against the outer stage value `Y`.

ref_stage_count(z, ε) = max(1, ceil(Int, sqrt(z / (2 - 4ε / 3))))

# (μⱼ, νⱼ, μ̃ⱼ) and the stage times of the s-stage first order RKC sweep, from the Chebyshev
# polynomial at ω₀ = 1 + ε/s² with ω₁ = T_s(ω₀)/T_s'(ω₀). `c[j]` is cⱼ₋₁, so `c[1] = c₀ = 0`.
function ref_rkc1_coefficients(s::Int, ε::Float64)
    ω₀ = 1 + ε / s^2
    T = Vector{Float64}(undef, s + 1)   # T[j+1] = Tⱼ(ω₀)
    Tp = Vector{Float64}(undef, s + 1)
    T[1], Tp[1] = 1.0, 0.0
    T[2], Tp[2] = ω₀, 1.0
    for j = 2:s
        T[j+1]  = 2ω₀ * T[j] - T[j-1]
        Tp[j+1] = 2T[j] + 2ω₀ * Tp[j] - Tp[j-1]
    end
    ω₁ = T[s+1] / Tp[s+1]

    μ, ν, μ̃ = zeros(s), zeros(s), zeros(s)
    c = zeros(s + 1)
    μ[1], ν[1], μ̃[1] = 1.0, 0.0, ω₁ / ω₀
    c[2] = μ̃[1]
    for j = 2:s
        bj, bjm1, bjm2 = 1 / T[j+1], 1 / T[j], 1 / T[j-1]
        μ[j] = 2ω₀ * bj / bjm1
        ν[j] = -bj / bjm2
        μ̃[j] = 2ω₁ * bj / bjm1
        c[j+1] = μ[j] * c[j] + ν[j] * c[j-1] + μ̃[j]
    end
    return μ, ν, μ̃, c
end

# Everything the reference needs about one problem, read off the production cache so that the two
# integrate the same discretization.
struct RefContext{IonType, MatType, SrcType}
    ion::IonType
    K::MatType
    invM::Vector{Float64}
    source_op::SrcType
    V::UnitRange{Int}
    npoints::Int
    nstates::Int
    φidx::Int
    gidx::Vector{Int}   # declared gate positions ...
    gsel::Vector{Bool}  # ... and which of them this partition integrates exponentially
    ρS::Float64
    ρF::Float64
    εo::Float64
    εi::Float64
end

function RefContext(integrator, ion, gates)
    cache = integrator.cache
    gsyms = gating_symbols(ion)
    return RefContext(
        ion,
        FerriteOperators.get_matrix(cache.op.K),
        collect(cache.op.invM),
        cache.source_op,
        integrator.f.solution_indices[1],
        length(integrator.f.solution_indices[1]),
        num_states(ion),
        Thunderbolt.transmembranepotential_index(ion),
        collect(gating_indices(ion)),
        Bool[gates === :all ? true : (g ∈ gates) for g in gsyms],
        cache.ρS,
        cache.ρF,
        integrator.alg.outer.ε,
        integrator.alg.inner.ε,
    )
end

ref_slot(ctx, k, p) = (k - 1) * ctx.npoints + p
ref_local(ctx, u, p) = [u[ref_slot(ctx, k, p)] for k = 1:ctx.nstates]

# y_E: the outer stage value with every selected gate advanced exactly over the window η.
function ref_gate_step(Y, t, η, ctx)
    yE = copy(Y)
    any(ctx.gsel) || return yE
    for p = 1:ctx.npoints
        loc = ref_local(ctx, Y, p)
        λ, y∞ = gate_coefficients(ctx.ion, loc[ctx.φidx], loc, t)
        for k in eachindex(ctx.gidx)
            ctx.gsel[k] || continue
            slot = ref_slot(ctx, ctx.gidx[k], p)
            yE[slot] = y∞[k] + (Y[slot] - y∞[k]) * exp(η * λ[k])
        end
    end
    return yE
end

# f_S: the cell right hand side with the exponentially integrated rows removed. No source -- that
# is not part of the reaction and enters only on the transmembrane rows.
function ref_slow_force(y, t, ctx)
    fS = zeros(length(y))
    du = zeros(ctx.nstates)
    for p = 1:ctx.npoints
        fill!(du, 0.0)
        cell_rhs!(du, ref_local(ctx, y, p), nothing, t, ctx.ion)
        for k in eachindex(ctx.gidx)
            ctx.gsel[k] && (du[ctx.gidx[k]] = 0.0)
        end
        for k = 1:ctx.nstates
            fS[ref_slot(ctx, k, p)] = du[k]
        end
    end
    return fS
end

function ref_averaged_force(Y, t, η, m, ctx)
    yE = ref_gate_step(Y, t, η, ctx)
    fS = ref_slow_force(yE, t, ctx)

    needs_update(ctx.source_op, t) &&
        update_operator!(ctx.source_op, nothing, TimeIntegrationContext(t, 0.0, 0.0))
    if !(ctx.source_op isa Thunderbolt.LinearNullOperator)
        fS[ctx.V] .+= ctx.invM .* FerriteOperators.operator_payload(ctx.source_op)
    end

    # The inner sweep, over the FULL state, starting from y_E, with f_S frozen: the rows the
    # production path finishes analytically go through the recurrence here.
    inner_rhs(w) = (r = copy(fS); r[ctx.V] .+= ctx.invM .* (ctx.K * w[ctx.V]); r)
    μ, ν, μ̃, _ = ref_rkc1_coefficients(m, ctx.εi)
    Wjm2, Wjm1 = copy(yE), copy(yE)
    W = copy(yE)
    for j = 1:m
        W = μ[j] * Wjm1 + ν[j] * Wjm2 + (μ̃[j] * η) * inner_rhs(Wjm1)
        Wjm2, Wjm1 = Wjm1, W
    end

    return (W .- Y) ./ η
end

function ref_emrkc_step(u, t, Δt, ctx)
    s = ref_stage_count(Δt * ctx.ρS, ctx.εo)
    η = 2Δt / ((2 - 4ctx.εo / 3) * s^2)
    m = ref_stage_count(η * ctx.ρF, ctx.εi)

    μ, ν, μ̃, c = ref_rkc1_coefficients(s, ctx.εo)
    Yjm2, Yjm1 = copy(u), copy(u)
    Y = copy(u)
    for j = 1:s
        fbar = ref_averaged_force(Yjm1, t + c[j] * Δt, η, m, ctx)
        Y = μ[j] * Yjm1 + ν[j] * Yjm2 + (μ̃[j] * Δt) * fbar
        Yjm2, Yjm1 = Yjm1, Y
    end
    return Y, (s, η, m)
end

# A dense, centrally differenced Jacobian of the same masked slow force, for the ρ_S front end.
function ref_slow_jacobian(u, t, ctx)
    n = length(u)
    J = zeros(n, n)
    for j = 1:n
        h = 1.0e-6 * max(1.0, abs(u[j]))
        up, um = copy(u), copy(u)
        up[j] += h
        um[j] -= h
        J[:, j] = (ref_slow_force(up, t, ctx) .- ref_slow_force(um, t, ctx)) ./ (2h)
    end
    return J
end

#####################################################################
#  THE discriminating test                                          #
#####################################################################

@testset "emRKC step matches a naive Algorithm 3 reference" begin
    # The stage counts are forced through raw-number spectral radii (`rho_safety = 1`, so the
    # numbers are used verbatim), which is the only way to reach the (s, m) corners on a problem
    # small enough to reference-implement.
    fhn_cases = [
        # (ρ_S, ρ_F, expected s, expected m)
        (10.0, 10.0, 1, 1),
        (10.0, 120.0, 1, 3),
        (10.0, 1000.0, 1, 8),
        (60.0, 40.0, 2, 1),
        (60.0, 500.0, 2, 3),
        (60.0, 4000.0, 2, 8),
        (450.0, 200.0, 5, 1),
        (450.0, 3000.0, 5, 3),
        (450.0, 25000.0, 5, 8),
    ]
    Δt = 0.1
    nsteps = 3

    @testset "FHN gates=$(gates) (s,m)=($s_exp,$m_exp)" for gates in (:all, ()),
        (ρS, ρF, s_exp, m_exp) in fhn_cases

        prob, odeform = emrkc_problem(; tspan = (0.0, nsteps * Δt))
        ion = odeform.functions[2].ode
        alg = EMRKC(; gates = gates, rho_S_estimate = ρS, rho_F_estimate = ρF, rho_safety = 1.0)
        integ = DiffEqBase.init(prob, alg; dt = Δt, verbose = false)

        u0 = copy(integ.u)
        produced = [
            (DiffEqBase.step!(integ); copy(integ.u)) for _ = 1:nsteps
        ]

        ctx = RefContext(integ, ion, gates)
        @test (ctx.ρS, ctx.ρF) == (ρS, ρF)

        uref = copy(u0)
        for step = 1:nsteps
            uref, (s, η, m) = ref_emrkc_step(uref, (step - 1) * Δt, Δt, ctx)
            @test (s, m) == (s_exp, m_exp)
            @test produced[step] ≈ uref rtol = 1.0e-13
        end
    end

    # A partial mask over a six-gate model: catches an index or mask wiring error that a
    # single-gate model cannot see.
    @testset "PCG2019 gates=$(gates)" for gates in (:all, (:h, :m), ())
        Δtp = 1.0e-3
        prob, odeform =
            emrkc_problem(; n = 3, ion = Thunderbolt.PCG2019(), tspan = (0.0, nsteps * Δtp))
        ion = odeform.functions[2].ode
        alg = EMRKC(;
            gates = gates,
            rho_S_estimate = 2000.0,
            rho_F_estimate = 1.0e5,
            rho_safety = 1.0,
        )
        integ = DiffEqBase.init(prob, alg; dt = Δtp, verbose = false)

        u0 = copy(integ.u)
        produced = [
            (DiffEqBase.step!(integ); copy(integ.u)) for _ = 1:nsteps
        ]

        ctx = RefContext(integ, ion, gates)
        uref = copy(u0)
        for step = 1:nsteps
            uref, _ = ref_emrkc_step(uref, (step - 1) * Δtp, Δtp, ctx)
            @test produced[step] ≈ uref rtol = 1.0e-13
        end
    end
end

#####################################################################

@testset "(s, η, m) against hand-computed Algorithm 3 values" begin
    alg = EMRKC()               # RKC1(0.05) outer and inner
    β = 2 - 4 * 0.05 / 3

    # s = ⌈√(Δt ρ_S / β)⌉, η = 2Δt/(β s²), m = ⌈√(η ρ_F / β)⌉.
    @test _emrkc_step_sizing(alg, 0.1, 10.0, 10.0) == (1, 2 * 0.1 / (β * 1^2), 1)
    @test _emrkc_step_sizing(alg, 0.1, 60.0, 500.0) == (2, 2 * 0.1 / (β * 2^2), 3)
    @test _emrkc_step_sizing(alg, 0.1, 450.0, 25000.0) == (5, 2 * 0.1 / (β * 5^2), 8)

    # A vanishing slow force is the degenerate single stage case, not a division by zero.
    s, η, m = _emrkc_step_sizing(alg, 0.25, 0.0, 0.0)
    @test (s, m) == (1, 1)
    @test η ≈ 2 * 0.25 / β

    # The window is the outer family's stability boundary read forward, so the sizing law composes
    # with any pair of families rather than hardcoding RKC's β s².
    for (fam_o, fam_i) in ((RKC1(0.05), RKC1(0.05)), (RKL1(), RKG1()), (RKG1(), RKL1()))
        a = EMRKC(outer = fam_o, inner = fam_i)
        s, η, m = _emrkc_step_sizing(a, 0.1, 300.0, 5000.0)
        @test s == sts_stage_count(fam_o, 0.1 * 300.0)
        @test η ≈ 2 * 0.1 / sts_stability_boundary(fam_o, s)
        @test m == sts_stage_count(fam_i, η * 5000.0)
    end

    # `max_stages` refuses rather than silently truncating the stability it buys.
    @test_throws ErrorException _emrkc_step_sizing(EMRKC(max_stages = 3), 1.0, 1.0e4, 1.0)
    @test_throws ErrorException _emrkc_step_sizing(alg, 1.0, NaN, 1.0)
end

@testset "sts_stage_count inverts sts_stability_boundary" begin
    for fam in (RKC1(0.05), RKC1(0.0), RKL1(), RKG1()), s = 1:20
        fam isa RKL1 && iseven(s) && continue # RKL1 only ever reports odd stage counts
        # Just inside the boundary: `s` stages suffice, and `s - 1` would not.
        @test sts_stage_count(fam, sts_stability_boundary(fam, s) * (1 - 1.0e-12)) == s
    end
end

@testset "Spectral radius front ends" begin
    prob, odeform = emrkc_problem()
    ion = odeform.functions[2].ode

    # A raw-number override is a ρ like any other: `rho_safety` multiplies it too, so swapping an
    # estimator for a measured number cannot silently drop the margin.
    integ = DiffEqBase.init(
        prob,
        EMRKC(rho_S_estimate = 3.0, rho_F_estimate = 7.0, rho_safety = 1.1);
        dt = 0.1,
        verbose = false,
    )
    DiffEqBase.step!(integ)
    @test integ.cache.ρS ≈ 1.1 * 3.0
    @test integ.cache.ρF ≈ 1.1 * 7.0

    integ = DiffEqBase.init(prob, EMRKC(rho_safety = 1.0); dt = 0.1, verbose = false)
    DiffEqBase.step!(integ)

    # ρ_F is the spectral radius of the assembled lumped rate operator Mₗ⁻¹K ...
    invM = collect(integ.cache.op.invM)
    K = Matrix(FerriteOperators.get_matrix(integ.cache.op.K))
    ρF_exact = maximum(abs, eigvals(Diagonal(invM) * K))
    @test integ.cache.ρF ≈ ρF_exact rtol = 0.05

    # ... and ρ_S that of the Jacobian of the *masked* slow force, which is what the Jacobian free
    # directional finite difference front end has to reproduce.
    ctx = RefContext(integ, ion, :all)
    ρS_exact = maximum(abs, eigvals(ref_slow_jacobian(prob.u0, 0.0, ctx)))
    @test integ.cache.ρS ≈ ρS_exact rtol = 0.05

    # Removing the gates from the slow force is what emRKC buys over mRKC: on a model with a fast
    # gate the exponentially integrated partition is far less stiff than the full reaction.
    pprob, podeform = emrkc_problem(n = 3, ion = Thunderbolt.PCG2019())
    ρS_of(gates) = begin
        i = DiffEqBase.init(pprob, EMRKC(gates = gates, rho_safety = 1.0); dt = 1.0e-3, verbose = false)
        DiffEqBase.step!(i)
        i.cache.ρS
    end
    @test ρS_of(:all) < ρS_of(()) / 3

    # :gershgorin is an upper bound on the same ρ_F, and is a usable policy on the host.
    integ = DiffEqBase.init(
        prob,
        EMRKC(rho_F_estimate = :gershgorin, rho_safety = 1.0);
        dt = 0.1,
        verbose = false,
    )
    DiffEqBase.step!(integ)
    @test integ.cache.ρF ≥ ρF_exact

    for bad in (EMRKC(rho_F_estimate = :nonsense), EMRKC(rho_S_estimate = :nonsense))
        i = DiffEqBase.init(prob, bad; dt = 0.1, verbose = false)
        @test_throws ErrorException DiffEqBase.step!(i)
    end
end

@testset "ρ recompute policy" begin
    prob, _ = emrkc_problem(tspan = (0.0, 1.0))

    # `:once` estimates on the first step of a run and then never again ...
    integ = DiffEqBase.init(prob, EMRKC(); dt = 0.1, verbose = false)
    for expected in (0, 1, 2)
        DiffEqBase.step!(integ)
        @test integ.cache.steps_since_estimate == expected
    end

    # ... but a `reinit!` starts a new run, whose state the previous ρ_S says nothing about.
    DiffEqBase.reinit!(integ, prob.u0)
    DiffEqBase.step!(integ)
    @test integ.cache.steps_since_estimate == 0

    # `n::Int` re-estimates every n steps.
    integ = DiffEqBase.init(prob, EMRKC(rho_recompute = 2); dt = 0.1, verbose = false)
    for expected in (0, 1, 2, 0, 1)
        DiffEqBase.step!(integ)
        @test integ.cache.steps_since_estimate == expected
    end
end

@testset "Setup validation" begin
    prob, odeform = emrkc_problem()
    u = copy(prob.u0)

    @test OS.init_cache(odeform, EMRKC(); uprev = copy(u), u = copy(u)) isa Thunderbolt.EMRKCCache

    # Not a reaction-diffusion split at all.
    bogus = GenericSplitFunction(
        (ODEFunction((du, u, p, t) -> (du .= -u)), ODEFunction((du, u, p, t) -> (du .= u))),
        ([1, 2, 3], [1, 2, 3]),
    )
    @test_throws ErrorException OS.init_cache(bogus, EMRKC(); uprev = zeros(3), u = zeros(3))

    # A gate the model never declared, and options that are not a partition at all.
    for bad in (EMRKC(gates = (:nope,)), EMRKC(gates = :some), EMRKC(gates = :s))
        @test_throws ErrorException OS.init_cache(odeform, bad; uprev = copy(u), u = copy(u))
    end
end

@testset "Degenerate mode: gates = () runs a model that declares none" begin
    # Aliev-Panfilov declares no gates and implements no `gate_coefficients`, so an exponential
    # stage would `MethodError`. The empty selection has to be lowered away entirely.
    prob, _ = emrkc_problem(ion = Thunderbolt.AlievPanfilovModel(), tspan = (0.0, 1.0))
    for gates in (:all, ())
        integ = DiffEqBase.init(prob, EMRKC(gates = gates); dt = 0.05, verbose = false)
        DiffEqBase.solve!(integ)
        @test integ.sol.retcode == SciMLBase.ReturnCode.Success
        @test all(isfinite, integ.u)
    end
end

@testset "Operator splitting interface: clocks, children, reinit" begin
    prob, _ = emrkc_problem(tspan = (0.0, 1.0))
    integ = DiffEqBase.init(prob, EMRKC(); dt = 0.1, verbose = false)

    @test SciMLBase.isadaptive(integ.alg) == false
    @test all(c -> c.alg isa PassiveChildSolver, integ.child_subintegrators)
    @test all(c -> c.cache isa PassiveChildCache, integ.child_subintegrators)

    u₀ = copy(integ.u)
    DiffEqBase.solve!(integ)
    @test integ.sol.retcode == SciMLBase.ReturnCode.Success
    @test integ.t ≈ 1.0
    @test integ.u ≉ u₀
    @test all(isfinite, integ.u)
    # The passive children are the clock-keeping half of the monolithic step: their clocks must
    # track the parent's and their state must be the parent's slice.
    @test all(c -> c.t ≈ integ.t, integ.child_subintegrators)
    for (i, c) in enumerate(integ.child_subintegrators)
        @test c.u ≈ integ.u[integ.child_solution_indices[i]]
    end

    uend = copy(integ.u)
    DiffEqBase.reinit!(integ, u₀)
    @test integ.t ≈ 0.0
    @test integ.u ≈ u₀
    @test all(c -> c.t ≈ 0.0, integ.child_subintegrators)
    DiffEqBase.solve!(integ)
    @test integ.sol.retcode == SciMLBase.ReturnCode.Success
    @test integ.u ≈ uend

    # `uprev` is the rollback anchor of the surrounding integrator and must survive a step
    # untouched until that integrator advances it itself.
    integ2 = DiffEqBase.init(prob, EMRKC(); dt = 0.1, verbose = false)
    before = copy(integ2.uprev)
    DiffEqBase.step!(integ2)
    @test integ2.uprev ≈ before
end

@testset "A diverging sweep fails the step instead of writing a non-finite solution" begin
    prob, _ = emrkc_problem(tspan = (0.0, 1.0))
    integ = DiffEqBase.init(
        prob,
        EMRKC(rho_S_estimate = 1.0, rho_F_estimate = 1.0);
        dt = 0.1,
        verbose = false,
    )
    # A state whose cubic reaction term overflows, so the sweep produces non-finite stage values.
    # The step has to be reported as failed with the solution vector left as it was, which lets the
    # surrounding integrator roll back to its anchor rather than carry a NaN forward.
    integ.u .= 1.0e120
    DiffEqBase.solve!(integ)
    @test integ.sol.retcode != SciMLBase.ReturnCode.Success
    @test integ.t == 0.0 # no step was ever accepted
    @test all(isfinite, integ.u)
end
