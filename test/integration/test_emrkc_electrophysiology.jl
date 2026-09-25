using Thunderbolt
using OrdinaryDiffEqOperatorSplitting
using DiffEqBase
using Test
using LinearAlgebra
import SciMLBase

# emRKC against the production reaction-diffusion split on a propagating monodomain wave: a planar
# PCG2019 activation front, stepped by `EMRKC()` and by
# `LieTrotterGodunov(BackwardEulerSolver(), ForwardEulerCellSolver())`.
#
# Both schemes are first order in time, so at a matched step size their temporal errors and the
# difference between them are all O(Δt). That is not what separates them: `BackwardEulerSolver`
# carries the consistent mass matrix and emRKC's rate operator a row-sum lumped one, so the two solve
# semidiscretizations differing at O(h²) whatever Δt is. The first testset measures both
# contributions and asserts which dominates -- that is where the tolerances come from.

# 10 mm square, κ/(Cₘχ) = 0.4 mm²/ms, first order Lagrange: h = 0.21 mm, a few elements across the
# front. The O(h²) between the two mass matrices is 1.5e-2 at h = 0.31 mm and 3.3e-3 at h = 0.16 mm,
# so anything coarser outgrows the activation tolerances below.
const EMRKC_WAVE_L    = 10.0
const EMRKC_WAVE_N    = 48
const EMRKC_WAVE_TEND = 20.0  # ms, long enough for the front to cross the domain

function emrkc_wave_form(; mass = LumpedMass())
    mesh = generate_mesh(
        Quadrilateral, (EMRKC_WAVE_N, EMRKC_WAVE_N),
        Vec{2}((0.0, 0.0)), Vec{2}((EMRKC_WAVE_L, EMRKC_WAVE_L)),
    )
    cs = CartesianCoordinateSystem(mesh)
    model = MonodomainModel(
        ConstantCoefficient(1.0),
        ConstantCoefficient(1.0),
        ConstantCoefficient(SymmetricTensor{2, 2, Float64}((0.4, 0.0, 0.4))),
        NoStimulationProtocol(),
        Thunderbolt.PCG2019(),
        cs,
        :φₘ, :s,
    )
    return semidiscretize(
        ReactionDiffusionSplit(model),
        FiniteElementDiscretization(Dict(:φₘ => LagrangeCollection{1}()); mass),
        mesh,
    )
end

# An S1 stimulus written as an initial condition -- the left twelfth raised above threshold, the rest
# resting -- so no applied-current amplitude has to be tuned against the model's excitability.
function emrkc_wave_u0(form)
    u₀ = create_initial_condition(form, Float64)
    setvariable!(u₀, form, :φₘ) do x
        x[1] ≤ 0.12EMRKC_WAVE_L ? 20.0 : -85.0
    end
    return u₀
end

# Solve to `EMRKC_WAVE_TEND`, returning the final transmembrane potential and the per-dof activation
# time -- the linearly interpolated first upstroke crossing of -20 mV, `NaN` where the front never
# arrived. Unlike the potential itself, activation times are insensitive to where in the plateau the
# two schemes sit.
function emrkc_wave_solve(form, u₀, alg, Δt)
    φₘ = solution_variable(form, :φₘ)
    integ = DiffEqBase.init(
        OperatorSplittingProblem(form, copy(u₀), (0.0, EMRKC_WAVE_TEND)), alg;
        dt = Δt, verbose = false,
    )
    prev = copy(getvariable(integ.u, φₘ))
    act  = fill(NaN, length(prev))
    while integ.t < EMRKC_WAVE_TEND - 1.0e-9
        DiffEqBase.step!(integ)
        cur = getvariable(integ.u, φₘ)
        @inbounds for i in eachindex(act)
            if isnan(act[i]) && prev[i] < -20.0 ≤ cur[i]
                act[i] = integ.t - Δt * (cur[i] + 20.0) / (cur[i] - prev[i])
            end
        end
        copyto!(prev, cur)
    end
    # Stepping by hand is what makes the per-step readout above possible; the retcode still comes
    # from the normal path, which by now has nothing left to step.
    DiffEqBase.solve!(integ)
    return act, copy(getvariable(integ.u, φₘ)), integ
end

relgap(a, b) = norm(a .- b) / norm(b)

@testset "emRKC wave propagation against the operator splitting baseline" begin
    # The mass treatment is the discretization's: the implicit baseline takes the consistent mass,
    # emRKC the lumped one. Same mesh and interpolation, so the dof layout and `u₀` are shared.
    formL = emrkc_wave_form(; mass = ConsistentMass())
    formE = emrkc_wave_form()
    u₀   = emrkc_wave_u0(formE)
    baseline = LieTrotterGodunov((BackwardEulerSolver(), ForwardEulerCellSolver()))
    Δt = 0.01

    actL, φL, integL = emrkc_wave_solve(formL, u₀, baseline, Δt)
    actE, φE, integE = emrkc_wave_solve(formE, u₀, EMRKC(), Δt)
    _, φL2, _ = emrkc_wave_solve(formL, u₀, baseline, Δt / 2)
    _, φE2, _ = emrkc_wave_solve(formE, u₀, EMRKC(), Δt / 2)

    @test integL.sol.retcode == SciMLBase.ReturnCode.Success
    @test integE.sol.retcode == SciMLBase.ReturnCode.Success

    # The front crossed most of the domain; without this the agreement below could be two resting
    # states.
    @test count(!isnan, actE) > 0.8 * length(actE)
    @test abs(count(!isnan, actE) - count(!isnan, actL)) < 0.02 * length(actE)

    activated = .!isnan.(actL) .& .!isnan.(actE)
    Δact = abs.(actL[activated] .- actE[activated])
    # The upstroke lasts ~1 ms: the two schemes place the front within half an upstroke of each other
    # everywhere, and within a fifth on average.
    @test maximum(Δact) < 0.5
    @test sum(Δact) / length(Δact) < 0.25

    gap = relgap(φE, φL)
    @test gap < 2.0e-2

    # ... and that gap is spatial, not temporal: halving Δt leaves it where it was while moving each
    # scheme's own solution at least five times less. What the two disagree about is the mass matrix,
    # not the step size -- a first order temporal difference would have halved here. This is the
    # assertion the tolerances above are derived from.
    @test relgap(φE2, φL2) > 0.7gap
    @test relgap(φL, φL2) < 0.2gap
    @test relgap(φE, φE2) < 0.2gap
end

@testset "emRKC steps past the explicit stability limit of the same discretization" begin
    form = emrkc_wave_form()
    u₀   = emrkc_wave_u0(form)

    # The forward Euler limit of the lumped diffusion operator, off the algorithm's own estimate with
    # the safety factor removed.
    probe = DiffEqBase.init(
        OperatorSplittingProblem(form, copy(u₀), (0.0, EMRKC_WAVE_TEND)), EMRKC(rho_safety = 1.0);
        dt = 1.0e-3, verbose = false,
    )
    DiffEqBase.step!(probe)
    Δt_explicit = 2 / probe.cache.ρF
    Δt = 10Δt_explicit

    _, φ, integ = emrkc_wave_solve(form, u₀, EMRKC(), Δt)
    _, φ_fine, _ = emrkc_wave_solve(form, u₀, EMRKC(), Δt_explicit / 2)

    @test integ.sol.retcode == SciMLBase.ReturnCode.Success
    @test all(isfinite, φ)
    # Ten times past the limit the wave is still a wave: the potential stays inside the model's range
    # and the trajectory inside 15% of the converged one. Accuracy is not the claim, but a solution
    # that is neither accurate nor physiological is not stability either.
    #
    # Ten, not the hundred `test_emrkc_convergence.jl` reaches on pure diffusion: there the reaction
    # is switched off and diffusion alone bounds the step. Here a hundred times this limit is 5 ms,
    # where emRKC still returns a finite, non-growing solution but PCG2019's ~1 ms upstroke is long
    # gone. On a full monodomain the reaction's timescale, not the diffusion's stability limit, is
    # what bounds a usable step.
    @test all(φᵢ -> -90.0 ≤ φᵢ ≤ 60.0, φ)
    @test relgap(φ, φ_fine) < 0.15

    # The control: the same step size with the inner sweep forced to a single stage is a plain
    # forward Euler diffusion step of size ≈ Δt, which this step size is past. The stabilized inner
    # sweep is what buys the step, not a mild problem.
    control = DiffEqBase.init(
        OperatorSplittingProblem(form, copy(u₀), (0.0, EMRKC_WAVE_TEND)),
        EMRKC(rho_F_estimate = 1.0e-12); dt = Δt, verbose = false,
    )
    DiffEqBase.solve!(control)
    φ_control = getvariable(control.u, solution_variable(form, :φₘ))
    @test control.sol.retcode != SciMLBase.ReturnCode.Success ||
        norm(φ_control) > 10norm(φ_fine)
end
