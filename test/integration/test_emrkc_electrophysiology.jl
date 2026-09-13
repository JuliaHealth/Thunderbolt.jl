using Thunderbolt
using OrdinaryDiffEqOperatorSplitting
using DiffEqBase
using Test
using LinearAlgebra
import SciMLBase

# emRKC against the production reaction-diffusion split on a propagating monodomain wave: a planar
# PCG2019 activation front, stepped by `EMRKC()` and by the
# `LieTrotterGodunov(BackwardEulerSolver(), ForwardEulerCellSolver())` pair that
# `test/integration/test_electrophysiology.jl` exercises.
#
# Both schemes are first order in time, so at a matched step size their temporal errors are each
# O(Δt) and so is the difference between them. That is *not* what separates them here:
# `BackwardEulerSolver` carries the consistent mass matrix and emRKC's rate operator a row-sum lumped
# one, so the two solve different semidiscretizations of the same monodomain problem, differing at
# O(h²) whatever Δt is. The first testset measures both contributions and asserts which one dominates
# -- that measurement is where the tolerances come from, rather than from a bound on Δt alone.

# 10 mm square, κ/(Cₘχ) = 0.4 mm²/ms, first order Lagrange: h = 0.21 mm, a few elements across the
# front. Coarser than this and the O(h²) between the two mass matrices grows past what the
# activation tolerances below are worth stating at all: it is 1.5e-2 at h = 0.31 mm and 3.3e-3 at
# h = 0.16 mm.
const EMRKC_WAVE_L    = 10.0
const EMRKC_WAVE_N    = 48
const EMRKC_WAVE_TEND = 20.0  # ms, long enough for the front to cross the domain

function emrkc_wave_form()
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
        FiniteElementDiscretization(Dict(:φₘ => LagrangeCollection{1}())),
        mesh,
    )
end

# The cell model's resting state everywhere, with the left twelfth of the domain raised above
# threshold: an S1 stimulus written as an initial condition, which launches a planar front without
# the amplitude of an applied current having to be tuned against the model's excitability.
function emrkc_wave_u0(form)
    u₀ = create_initial_condition(form, Float64)
    setvariable!(u₀, form, :φₘ) do x
        x[1] ≤ 0.12EMRKC_WAVE_L ? 20.0 : -85.0
    end
    return u₀
end

"""
Solve to `EMRKC_WAVE_TEND` and return the final transmembrane potential together with the
per-dof activation time -- the linearly interpolated first upstroke crossing of -20 mV, `NaN` where
the front never arrived. Activation times are what a conduction study compares, and unlike the
potential itself they are insensitive to where in the plateau the two schemes sit.
"""
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
    # Stepping by hand is what makes the per-step readout above possible; the retcode still has to
    # come from the normal path, and by now there is nothing left for it to step.
    DiffEqBase.solve!(integ)
    return act, copy(getvariable(integ.u, φₘ)), integ
end

relgap(a, b) = norm(a .- b) / norm(b)

@testset "emRKC wave propagation against the operator splitting baseline" begin
    form = emrkc_wave_form()
    u₀   = emrkc_wave_u0(form)
    baseline = LieTrotterGodunov((BackwardEulerSolver(), ForwardEulerCellSolver()))
    Δt = 0.01

    actL, φL, integL = emrkc_wave_solve(form, u₀, baseline, Δt)
    actE, φE, integE = emrkc_wave_solve(form, u₀, EMRKC(), Δt)
    _, φL2, _ = emrkc_wave_solve(form, u₀, baseline, Δt / 2)
    _, φE2, _ = emrkc_wave_solve(form, u₀, EMRKC(), Δt / 2)

    @test integL.sol.retcode == SciMLBase.ReturnCode.Success
    @test integE.sol.retcode == SciMLBase.ReturnCode.Success

    # The front actually crossed most of the domain -- without this the agreement below could be two
    # copies of a resting state.
    @test count(!isnan, actE) > 0.8 * length(actE)
    @test abs(count(!isnan, actE) - count(!isnan, actL)) < 0.02 * length(actE)

    activated = .!isnan.(actL) .& .!isnan.(actE)
    Δact = abs.(actL[activated] .- actE[activated])
    # The front takes ~15 ms to cross and the upstroke lasts ~1 ms. The two schemes place it within
    # half an upstroke of each other everywhere, and within a fifth of one on average.
    @test maximum(Δact) < 0.5
    @test sum(Δact) / length(Δact) < 0.25

    gap = relgap(φE, φL)
    @test gap < 2.0e-2

    # ... and that gap is spatial, not temporal. Halving Δt leaves it where it was, while it moves
    # each scheme's own solution at least five times less: what the two disagree about is the mass
    # matrix (consistent under backward Euler, row-sum lumped under emRKC), not the step size. This
    # is the assertion the tolerances above are derived from -- a bound on Δt alone would not reach
    # them, and a first order temporal difference would have halved here.
    @test relgap(φE2, φL2) > 0.7gap
    @test relgap(φL, φL2) < 0.2gap
    @test relgap(φE, φE2) < 0.2gap
end

@testset "emRKC steps past the explicit stability limit of the same discretization" begin
    form = emrkc_wave_form()
    u₀   = emrkc_wave_u0(form)

    # The forward Euler limit of the lumped diffusion operator, read off the algorithm's own estimate
    # with the safety factor removed.
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
    # Ten times past the limit the wave is still a wave: the potential stays inside the model's own
    # range and the trajectory inside 15% of the converged one (measured: 8%). Accuracy is not the
    # claim -- the step is ten times an explicit scheme's -- but a solution that is neither accurate
    # nor physiological is not stability either.
    #
    # Ten, and not the hundred `test_emrkc_convergence.jl` reaches on pure diffusion: there the
    # reaction is switched off and the diffusion is the only thing bounding the step. Here the step
    # a hundred times past this limit is 5 ms, and while emRKC still returns a finite, non-growing
    # solution there, PCG2019's ~1 ms upstroke is long gone at that step size and the result leaves
    # the physiological range. On a full monodomain it is the reaction's own timescale, not the
    # diffusion's stability limit, that bounds a usable step -- which is exactly what emRKC is for.
    @test all(φᵢ -> -90.0 ≤ φᵢ ≤ 60.0, φ)
    @test relgap(φ, φ_fine) < 0.15

    # The control: the same step size with the inner sweep forced to a single stage is a plain
    # forward Euler diffusion step of size ≈ Δt, which this step size is past. It is the stabilized
    # inner sweep that buys the step, not something about the problem being mild.
    control = DiffEqBase.init(
        OperatorSplittingProblem(form, copy(u₀), (0.0, EMRKC_WAVE_TEND)),
        EMRKC(rho_F_estimate = 1.0e-12); dt = Δt, verbose = false,
    )
    DiffEqBase.solve!(control)
    φ_control = getvariable(control.u, solution_variable(form, :φₘ))
    @test control.sol.retcode != SciMLBase.ReturnCode.Success ||
        norm(φ_control) > 10norm(φ_fine)
end
