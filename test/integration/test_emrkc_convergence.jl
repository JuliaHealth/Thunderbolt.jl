using Thunderbolt
using OrdinaryDiffEqOperatorSplitting
using DiffEqBase
using Test
using LinearAlgebra
using StaticArrays
import SciMLBase

# Observed order of convergence in time, and what the multirate machinery buys at a step size no
# explicit scheme survives. A scheme that forms its averaged force from the wrong window still
# *converges* -- to the wrong limit -- so the order measured here says nothing on its own about
# which limit; that the limit is the right one is what `test/test_emrkc.jl` pins, against a
# reference implementation of the paper's Algorithm 3.
#
# Every initial condition below is a smooth function of the coordinates rather than of the dof
# index. The unresolved end of a stiff spectrum converges at no order at all, so an initial
# condition with content there would measure the damping of modes none of these step sizes resolve.

"""
Ratios of successive solution differences under repeated step halving. A scheme of order `p` sends
these to `2^p`. Same measure as `test/integration/test_temporal_convergence.jl` uses: the
differences are norms of the *difference vectors*, so a sign change across components cannot be
mistaken for convergence.
"""
function convergence_ratios(solve_to_end, Δts)
    us = [solve_to_end(Δt) for Δt in Δts]
    diffs = [norm(us[i+1] .- us[i]) for i = 1:(length(us)-1)]
    return [diffs[i] / diffs[i+1] for i = 1:(length(diffs)-1)]
end

function emrkc_monodomain(ion; n, κ, stimulate = true)
    grid = generate_grid(Quadrilateral, (n, n), Vec{2}((-1.0, -1.0)), Vec{2}((1.0, 1.0)))
    mesh = to_mesh(grid)
    cs = CartesianCoordinateSystem(mesh)
    stim = if stimulate
        AnalyticalTransmembraneStimulationProtocol(
            AnalyticalCoefficient((x, t) -> 2.0 * exp(-4 * norm(x)^2) * cospi(t), cs),
            [SVector((0.0, 1.0e6))],
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
    return semidiscretize(
        ReactionDiffusionSplit(model),
        FiniteElementDiscretization(Dict(:φₘ => LagrangeCollection{1}())),
        mesh,
    )
end

# The cell model's resting state everywhere, with a smooth Gaussian on the transmembrane potential.
function emrkc_u0(odeform, ion)
    u₀ = zeros(Float64, solution_size(odeform))
    nV = length(odeform.solution_indices[1])
    rest = Thunderbolt.default_initial_state(ion)
    for k = 1:Thunderbolt.num_states(ion), j = 1:nV
        u₀[(k-1)*nV+j] = rest[k]
    end
    φidx = Thunderbolt.transmembranepotential_index(ion)
    φoffset = (φidx - 1) * nV
    dh = odeform.functions[1].dh
    for sdh in dh.subdofhandlers, cell in CellIterator(sdh)
        dofs = celldofs(cell)[dof_range(sdh, :φₘ)]
        for (i, x) in zip(dofs, getcoordinates(cell))
            u₀[φoffset+i] = rest[φidx] + exp(-3 * norm(x)^2)
        end
    end
    return u₀
end

emrkc_integrator(odeform, u₀, alg, Δt, tspan) = DiffEqBase.init(
    OperatorSplittingProblem(odeform, copy(u₀), tspan),
    alg;
    dt = Δt,
    verbose = false,
)

function emrkc_solve(odeform, u₀, alg, Δt, tspan)
    integ = emrkc_integrator(odeform, u₀, alg, Δt, tspan)
    DiffEqBase.solve!(integ)
    return integ
end

#####################################################################

@testset "Pure diffusion: first order, and stable far past the explicit limit" begin
    # FHN with its reaction switched off (e = f = 0) leaves exactly the lumped mass heat equation
    # with a source. The slow force vanishes identically, so the outer sweep degenerates to a
    # single stage and the inner sweep carries the whole problem -- the arm where the multirate
    # machinery is all there is.
    ion = Thunderbolt.ParametrizedFHNModel{Float64}(e = 0.0, f = 0.0)
    tspan = (0.0, 1.0)
    Δts = (0.05, 0.025, 0.0125, 0.00625, 0.003125)

    odeform = emrkc_monodomain(ion; n = 16, κ = 1.0)
    u₀ = emrkc_u0(odeform, ion)
    V = odeform.solution_indices[1]

    ratios = convergence_ratios(Δts) do Δt
        integ = emrkc_solve(odeform, u₀, EMRKC(gates = ()), Δt, tspan)
        @test integ.sol.retcode == SciMLBase.ReturnCode.Success
        return copy(integ.u)
    end
    @test all(r -> 1.7 ≤ r ≤ 2.3, ratios)

    # The stability claim, on the same discretization without a drive, so that the exact solution
    # decays monotonically and any growth is the scheme's own.
    odeform_free = emrkc_monodomain(ion; n = 16, κ = 1.0, stimulate = false)
    probe = emrkc_integrator(odeform_free, u₀, EMRKC(gates = (), rho_safety = 1.0), 0.001, tspan)
    DiffEqBase.step!(probe)
    Δt_explicit = 2 / probe.cache.ρF          # the forward Euler limit of this discretization
    Δt_big = 100 * Δt_explicit

    coarse = emrkc_solve(odeform_free, u₀, EMRKC(gates = ()), Δt_big, tspan)
    fine   = emrkc_solve(odeform_free, u₀, EMRKC(gates = ()), Δt_explicit / 2, tspan)
    @test coarse.sol.retcode == SciMLBase.ReturnCode.Success
    @test all(isfinite, coarse.u)
    @test norm(coarse.u[V]) < norm(u₀[V])     # diffusion only: no growth
    # Stable, and still in the right place -- loosely, because Δt_big covers the whole interval in
    # under two steps and this is a first order scheme. Stability is the claim; accuracy at a
    # hundred times the explicit limit is not.
    @test norm(coarse.u .- fine.u) < 0.5 * norm(fine.u)

    # ... and that step size really is past the explicit limit: forcing a single inner stage makes
    # the inner sweep a plain forward Euler step of size ≈ Δt, which grows where pure diffusion
    # can only decay.
    explicit =
        emrkc_solve(odeform_free, u₀, EMRKC(gates = (), rho_F_estimate = 1.0e-12), Δt_big, tspan)
    @test explicit.sol.retcode != SciMLBase.ReturnCode.Success ||
          norm(explicit.u[V]) > 10 * norm(u₀[V])
end

@testset "Monodomain + PCG2019 is first order" begin
    ion = Thunderbolt.PCG2019()
    tspan = (0.0, 1.0) # ms
    odeform = emrkc_monodomain(ion; n = 8, κ = 5.0)
    u₀ = emrkc_u0(odeform, ion)

    # The refinement starts where the inner sweep already takes three stages and ends where it
    # takes one. An observed order is only meaningful over step sizes that resolve what is being
    # measured, and one step size above this window the inner sweep's own damping of the
    # unresolved end of the diffusion spectrum still dominates the difference.
    Δts = (0.025, 0.0125, 0.00625, 0.003125, 0.0015625)
    ratios = convergence_ratios(Δts) do Δt
        integ = emrkc_solve(odeform, u₀, EMRKC(), Δt, tspan)
        @test integ.sol.retcode == SciMLBase.ReturnCode.Success
        return copy(integ.u)
    end
    @test all(r -> 1.7 ≤ r ≤ 2.3, ratios)
end

@testset "Partition invariance: the gate selection is a method choice, not a model change" begin
    # Which declared gates are integrated exponentially picks a different *method* for the same
    # problem, so the three partitions must disagree at any finite Δt and agree in the limit -- and
    # the disagreement is itself a first order quantity, since both methods are first order
    # approximations of the same solution.
    ion = Thunderbolt.PCG2019()
    tspan = (0.0, 1.0)
    odeform = emrkc_monodomain(ion; n = 8, κ = 5.0)
    u₀ = emrkc_u0(odeform, ion)

    Δts = (0.05, 0.025, 0.0125, 0.00625)
    partitions = (:all, (:h, :m), ())
    solutions = Dict(
        p => [copy(emrkc_solve(odeform, u₀, EMRKC(gates = p), Δt, tspan).u) for Δt in Δts] for
        p in partitions
    )

    @testset "$(pa) vs $(pb)" for (pa, pb) in ((:all, (:h, :m)), (:all, ()), ((:h, :m), ()))
        gaps = [norm(solutions[pa][i] .- solutions[pb][i]) for i in eachindex(Δts)]
        # Different methods: they do not agree at a finite step size ...
        @test gaps[1] > 0
        # ... the disagreement halves with the step size ...
        @test all(i -> 1.7 ≤ gaps[i] / gaps[i+1] ≤ 2.3, 1:(length(gaps)-1))
        # ... down to a residual that is negligible against the solution.
        @test gaps[end] < 1.0e-3 * norm(solutions[pa][end])
    end
end
