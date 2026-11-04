using Test, Thunderbolt, OrdinaryDiffEqTsit5, OrdinaryDiffEqOperatorSplitting
using LinearSolve
using OrdinaryDiffEqNonlinearSolve
using ModelingToolkit
using SciCompDSL
using DynamicQuantities
# Named rather than `using`d: the MTK stack in this module exports a lot, and an explicit import
# cannot become ambiguous with it.
import BlockArrays: BlockMatrix, blocklengths, blocks
import SparseArrays: SparseMatrixCSC, nnz, nzrange
import Tensors: Tensors, Tensor, det, tdot, tr, ⊗
include(joinpath(@__DIR__, "..", "testfixtures.jl"))

# The 3D-0D coupled path, whole and in pieces.
#
# The transient solves below assert that the coupled problem converges, which a wrong Jacobian or a
# dropped coupling term can survive. The operator testsets after them pin the structure the solver
# rests on -- the dof layout, the tying-facet coverage, the coupling sparsity, the chamber-volume
# identities and the analytic tying tangent -- on the very same model, at a state that is
# deliberately not a solution.

function calcium_profile_function(x::LVCoordinate, t_global)
    linear_interpolation(t, y1, y2, t1, t2) = y1 + (t-t1) * (y2-y1)/(t2-t1)
    ca_peak(x)                              = 1.0
    t                                       = t_global % 800.0
    if 0 ≤ t ≤ 120.0
        return linear_interpolation(t, 0.0, ca_peak(x), 0.0, 120.0)
    elseif t ≤ 272.0
        return linear_interpolation(t, ca_peak(x), 0.0, 120.0, 272.0)
    else
        return 0.0
    end
end

# The single-chamber ideal LV, built once and shared by every testset in this file: what the operator
# testsets pin is then the model that is solved, not a look-alike.
scaling_factor = 3.9
ideal_lv_mesh = generate_ideal_lv_mesh(
    6,
    1,
    2;
    inner_radius        = scaling_factor * 0.7,
    outer_radius        = scaling_factor * 1.0,
    longitudinal_upper  = 0.4,
    apex_inner          = scaling_factor * 1.3,
    apex_outer          = scaling_factor * 1.5,
    with_valvular_plane = true,
)
ideal_lv_cs = compute_lv_coordinate_system(ideal_lv_mesh; subdomains = ["myocardium"])
ideal_lv_constitutive_model = ActiveStressModel(
    Guccione1991PassiveModel(),
    SimpleActiveStress(),
    Thunderbolt.CaDrivenInternalSarcomereModel(
        PelceSunLangeveld1995Model(),
        AnalyticalCoefficient(calcium_profile_function, ideal_lv_cs),
    ),
    create_microstructure_model(
        ideal_lv_cs,
        LagrangeCollection{1}()^3,
        ODB25LTMicrostructureParameters(αendo = deg2rad(80.0), αepi = deg2rad(-65.0));
        subdomains = ["myocardium"],
    ),
)

"""
    ideal_lv_3d0d_splitform(fluid_model, coupler; material = ideal_lv_constitutive_model)

The semidiscretized 3D-0D problem on `ideal_lv_mesh`. The boundary conditions and the valvular plane
are fixed; the material, the circuit and the coupling that ties it to the chamber surface are what
vary between call sites.
"""
function ideal_lv_3d0d_splitform(fluid_model, coupler; material = ideal_lv_constitutive_model)
    # No Dirichlet condition: the epicardial Robin spring pins the rigid body modes, and the base is
    # left free to move -- the valvular plane is what keeps the chamber volume well defined then.
    solid_model =
        QuasiStaticModel(:d, material, (RobinBC(0.1, "Epicardium"), NormalSpringBC(0.1, "Base")))
    # The cap closing the basal orifice: a stiff isotropic closure that carries the chamber pressure
    # on its ventricular face without bulging, compressible so the single element it is thick does
    # not lock. Same parameters as the `cm03_3d0d-coupling` tutorial, which states the contract.
    valvular_plane_model = QuasiStaticModel(
        :d,
        PK1Model(
            BioNeoHookean(; α = 16000.0, mpU = SimpleCompressionPenalty(16000.0)),
            NoMicrostructureModel(),
        ),
    )
    coupled_model = RSAFDQ2022Model(
        Dict("myocardium" => solid_model, "valvular-plane" => valvular_plane_model),
        fluid_model,
        coupler,
    )
    return semidiscretize(
        RSAFDQ2022Split(coupled_model),
        FiniteElementDiscretization(Dict(:d => LagrangeCollection{1}()^3)),
        ideal_lv_mesh,
    )
end

function test_solve_contractile_ideal_lv_3D0D(u0fluid, fluid_model, coupler, tmax, Δt)
    tspan = (0.0, tmax)

    splitform = ideal_lv_3d0d_splitform(fluid_model, coupler)

    # Create sparse matrix and residual vector
    chamber_solver = HomotopyPathSolver(
        NewtonRaphsonSolver(;
            max_iter = 10,
            tol = 1e-2,
            # The (1,1) block is the structural Jacobian, invertible; the (2,2) block is zero by
            # construction (`ChamberBalanceCache`'s row is constant in `u`) -- exactly the saddle
            # point `SchurComplementLinearSolver` factors.
            inner_solver = SchurComplementLinearSolver(LinearSolve.UMFPACKFactorization()),
        ),
    )
    blood_circuit_solver = Tsit5()
    timestepper = LieTrotterGodunov((chamber_solver, blood_circuit_solver))

    u₀ = zeros(solution_size(splitform))
    u₀solid_view = @view u₀[OS.get_solution_indices(splitform, 1)]
    u₀fluid_view = @view u₀[OS.get_solution_indices(splitform, 2)]
    u₀fluid_view .= u0fluid

    problem = OperatorSplittingProblem(splitform, u₀, tspan)
    integrator = init(problem, timestepper, dt = Δt, verbose = true; dtmax = 10.0);
    solve!(integrator)
    @test integrator.sol.retcode == ReturnCode.Success
    @test integrator.u ≉ u₀

    return integrator
end

@testset "3D0D Coupled" begin
    fluid_model_init = RSAFDQ2022LumpedCicuitModel()
    u0 = zeros(Thunderbolt.num_states(fluid_model_init))
    Thunderbolt.default_initial_state!(u0, fluid_model_init)
    prob = ODEProblem(
        (du, u, p, t) -> Thunderbolt.lumped_driver!(du, u, t, [], p),
        u0,
        (0.0, 10*fluid_model_init.THB),
        fluid_model_init,
    )
    sol = solve(prob, Tsit5())

    @test !any(isnan.(ideal_lv_cs.u_apicobasal))
    @test !any(isnan.(ideal_lv_cs.u_transmural))
    @test !any(isnan.(ideal_lv_cs.u_rotational))

    test_solve_contractile_ideal_lv_3D0D(
        sol.u[end],
        RSAFDQ2022LumpedCicuitModel(; lv_pressure_given = false),
        LumpedFluidSolidCoupler([ChamberVolumeCoupling("LVChamberSurface", :Vₗᵥ, :pₗᵥ, :pₗᵥ)], :d),
        1.0,
        1.0,
    )

    @mtkcompile rsafdq2022mtk_init = Thunderbolt.mtk_models().RSAFDQ2022CircuitMTK()
    τ = 800.0# TODO query ...?
    prob = ODEProblem(rsafdq2022mtk_init, [], (0.0, 10*τ))
    sol = solve(prob, Tsit5())
    @mtkcompile rsafdq2022mtk =
        Thunderbolt.mtk_models().RSAFDQ2022CircuitMTK(; lv_pressure_given = false)
    test_solve_contractile_ideal_lv_3D0D(
        sol.u[end],
        MTKLumpedCicuitModel(
            rsafdq2022mtk,
            Dict(unknowns(rsafdq2022mtk) .=> sol.u[end]),
            [rsafdq2022mtk.external_input_lv_p],
        ),
        LumpedFluidSolidCoupler(
            [
                ChamberVolumeCoupling(
                    "LVChamberSurface",
                    rsafdq2022mtk.Vₗᵥ,
                    rsafdq2022mtk.external_input_lv_p,
                    :pₗᵥ,
                ),
            ],
            :d,
        ),
        1.0,
        1.0,
    )
end

@testset "3D0D prescribed chamber volume" begin
    coupler = LumpedFluidSolidCoupler([ChamberVolumeCoupling("LVChamberSurface", :V, :p, :pₗᵥ)], :d)
    probe = ideal_lv_3d0d_splitform(Thunderbolt.DummyLumpedCircuitModel(t -> 0.0), coupler)
    V₀ = only(probe.functions[1].tying_info.chambers).V⁰ᴰval

    T, Δt = 4.0, 1.0
    smoothstep(τ) = (τ = clamp(τ, 0.0, 1.0); τ * τ * (3 - 2τ))
    V(t) = V₀ * (1 - 0.01 * smoothstep(t / T))

    tmax = T + 2Δt
    integrator = test_solve_contractile_ideal_lv_3D0D(
        [V(0.0)],
        Thunderbolt.DummyLumpedCircuitModel(V),
        coupler,
        tmax,
        Δt,
    )

    f = probe.functions[1]
    chamber = only(f.tying_info.chambers)
    # The 0D state is the prescribed volume itself, not an approximation chasing it.
    @test integrator.u[chamber.V⁰ᴰidx_global] ≈ V(tmax) rtol = 1e-12
    # The 3D problem follows it, within the tolerance of its Newton solve, and the pressure is its answer.
    u₃ᴰ = integrator.u[1:Thunderbolt.solution_size(f)]
    V³ᴰ = Thunderbolt.compute_chamber_volume(
        f.structural_function.dh,
        u₃ᴰ,
        "LVChamberSurface",
        chamber,
    )
    @test V³ᴰ ≈ V(tmax) atol = 1e-2
    @test integrator.u[chamber.pressure_dof_index] != 0.0
end

# A material with an internal variable on the 3D-0D coupled path. Its local problem has a closed form,
#
#   W(F, Q) = a/2 (I₁ - 3) + μ/2 (I₁ - 3 - Q)² + c/2 Q² + κ/2 (J - 1)² - a ln J,
#   0 = ∂W/∂Q  ⇒  Q = μ/(μ + c) (I₁ - 3),
#
# and condensing it leaves the hyperelastic energy with `μc/(μ + c)/2 (I₁ - 3)²` in place of the two
# terms in `Q`. So the condensed solve has a reference that shares none of its machinery.
Base.@kwdef struct CondensedEnergy
    a::Float64 = 2.0
    μ::Float64 = 8.0
    c::Float64 = 4.0
    κ::Float64 = 100.0
end
struct ReducedEnergy
    m::CondensedEnergy
end

function Thunderbolt.Ψ(F, coefficients, e::ReducedEnergy)
    (; a, μ, c, κ) = e.m
    I₁ = tr(tdot(F))
    J = det(F)
    return a / 2 * (I₁ - 3) + μ * c / (μ + c) / 2 * (I₁ - 3)^2 + κ / 2 * (J - 1)^2 - a * log(J)
end

_frozen_energy(F, Q, (; a, μ, c, κ)) =
    a / 2 * (tr(tdot(F)) - 3) +
    μ / 2 * (tr(tdot(F)) - 3 - Q)^2 +
    c / 2 * Q^2 +
    κ / 2 * (det(F) - 1)^2 - a * log(det(F))

const CondensedPK1 = PK1Model{<:CondensedEnergy}

Thunderbolt.internal_variable_evolution(::CondensedPK1) = Thunderbolt.SteadyStateEvolution()
Thunderbolt.default_initial_state!(uq, ::CondensedPK1) = fill!(uq, 0.0)
Thunderbolt.gather_internal_variable_infos(::CondensedPK1) =
    (Thunderbolt.InternalVariableInfo(:Q, 1),)

function Thunderbolt.solve_local_state!(
    F,
    coefficients,
    m::CondensedPK1,
    state_cache,
    geometry_cache,
    qp,
    time,
    Qflat,
)
    (; μ, c) = m.material
    Qflat[1] = μ / (μ + c) * (tr(tdot(F)) - 3)
    return nothing
end

function Thunderbolt.solve_local_constraint(
    F,
    coefficients,
    m::CondensedPK1,
    state_cache,
    geometry_cache,
    qp,
    time,
    Qflat,
)
    Thunderbolt.solve_local_state!(F, coefficients, m, state_cache, geometry_cache, qp, time, Qflat)
    # ∂P/∂Q = -2μF and dQ/dF = 2μ/(μ + c) F
    (; μ, c) = m.material
    return Qflat, -4μ^2 / (μ + c) * (F ⊗ F)
end

Thunderbolt.stress_function(m::CondensedPK1, F::Tensor{2}, coefficients, Q::AbstractVector) =
    Tensors.gradient(X -> _frozen_energy(X, Q[1], m.material), F)
function Thunderbolt.stress_and_tangent(
    m::CondensedPK1,
    F::Tensor{2},
    coefficients,
    Q::AbstractVector,
)
    ∂²W∂F², ∂W∂F = Tensors.hessian(X -> _frozen_energy(X, Q[1], m.material), F, :all)
    return ∂W∂F, ∂²W∂F²
end

coupled_splitform(material) = ideal_lv_3d0d_splitform(
    RSAFDQ2022LumpedCicuitModel(; lv_pressure_given = false),
    LumpedFluidSolidCoupler([ChamberVolumeCoupling("LVChamberSurface", :Vₗᵥ, :pₗᵥ, :pₗᵥ)], :d);
    material,
)

function solve_coupled(material)
    splitform = coupled_splitform(material)
    f = splitform.functions[1]
    chamber = only(f.tying_info.chambers)

    u₀ = zeros(solution_size(splitform))
    circuit = @view u₀[OS.get_solution_indices(splitform, 2)]
    Thunderbolt.default_initial_state!(circuit, RSAFDQ2022LumpedCicuitModel())
    # Start the chamber a little below the mesh's own volume.
    u₀[chamber.V⁰ᴰidx_global] = 0.995 * chamber.V⁰ᴰval

    timestepper = LieTrotterGodunov((
        HomotopyPathSolver(
            NewtonRaphsonSolver(;
                max_iter = 15,
                tol = 1e-9,
                inner_solver = SchurComplementLinearSolver(LinearSolve.UMFPACKFactorization()),
            ),
        ),
        Tsit5(),
    ))
    integrator = init(
        OperatorSplittingProblem(splitform, u₀, (0.0, 2.0)),
        timestepper;
        dt = 1.0,
        verbose = false,
        dtmax = 10.0,
    )
    solve!(integrator)
    return (; f, chamber, u = copy(integrator.u), retcode = integrator.sol.retcode)
end

@testset "3D0D coupling of a material with internal variables" begin
    m = CondensedEnergy()
    condensed = solve_coupled(PK1Model(m, NoMicrostructureModel()))
    reduced = solve_coupled(PK1Model(ReducedEnergy(m), NoMicrostructureModel()))
    @test condensed.retcode == ReturnCode.Success
    @test reduced.retcode == ReturnCode.Success

    # The condensed state follows the displacement block, the circuit follows it.
    f = condensed.f
    nu = Thunderbolt.ndofs(f.structural_function.dh)
    nq = length(Thunderbolt.internal_variable_range(f.structural_function))
    @test nq > 0
    @test solution_size(f) == nu + nq
    @test condensed.chamber.V⁰ᴰidx_global > nu + nq

    # What the coupling sees is the material the condensation reduces to.
    @test condensed.u[1:nu] ≈ reduced.u[1:nu] rtol = 1e-6
    @test condensed.u[condensed.chamber.pressure_dof_index] ≈
          reduced.u[reduced.chamber.pressure_dof_index] rtol = 1e-6
    @test condensed.u[(nu+nq+1):end] ≈ reduced.u[(nu+1):end] rtol = 1e-6

    # And the local problem was solved at the last state, not left at the initial one.
    @test any(!iszero, condensed.u[(nu+1):(nu+nq)])
end

@testset "3D0D coupling refuses a material with an evolving internal variable" begin
    microstructure = ConstantCoefficient(
        OrthotropicMicrostructure(Vec((1.0, 0.0, 0.0)), Vec((0.0, 1.0, 0.0)), Vec((0.0, 0.0, 1.0))),
    )
    material = ActiveStressModel(
        Guccione1991PassiveModel(),
        SimpleActiveStress(),
        Thunderbolt.CaDrivenInternalSarcomereModel(
            Thunderbolt.RDQ20MFModel(),
            ConstantCoefficient(1.0),
        ),
        microstructure,
    )
    @test_throws "RateCoupledEvolution" coupled_splitform(material)
end

# Pinned solver-supplied data for the operator testsets below. Within one 3D solve `V⁰ᴰ` is a
# constant, not an unknown; pinning it here rather than taking it from a transfer keeps the operator
# testsets independent of the 0D solve.
const PINNED_V⁰ᴰ = 120.0
# The pseudo-time the calcium transient is sampled at, and the pinned chamber pressure.
const PINNED_T = 100.0
const PINNED_PRESSURE = 1.3

# The state is drawn here rather than from `Random`: a failure of the testsets below has to be
# reproducible across Julia versions, which the `rand` stream does not promise. Knuth's LCG, taking
# the top 53 bits of each word.
function _pinned_uniform(n::Int, seed::Integer)
    u = Vector{Float64}(undef, n)
    x = UInt64(seed)
    for i = 1:n
        x = 0x5851f42d4c957f2d * x + 0x14057b7ef767814f
        u[i] = Float64(x >> 11) / 9007199254740992.0
    end
    return u
end

"""
    assemble_pair(op, u, p, ctx) -> (Matrix(J), r)

One fused linearization sweep at `u`, as dense matrix and residual.
"""
function assemble_pair(op, u, p, ctx)
    J = Thunderbolt.getJ(op)
    r = zeros(size(J, 1))
    Thunderbolt.update_linearization!(op, r, (u = u,), p, ctx)
    return (Matrix(J), r)
end

# Entry for entry, because the global dof numbering is the same by construction. `≈` rather than
# `==`: the summation order into a dof shared by several facets is not pinned.
function approx_entrywise(A, B; rtol = 1.0e-10)
    scale = maximum(abs, B)
    return all(isapprox.(A, B; rtol, atol = rtol * (scale > 0 ? scale : one(scale))))
end

"""
    single_chamber_state(; seed = 42)

The single-chamber ideal-LV model solved above, its stage operator, and a fixed pseudo-random state
to assemble at.

The state is deliberately *not* a solution: at equilibrium the residual is zero, which makes the
residual assertions vacuous, and the chamber row is only nonzero away from equilibrium.

Returns `(; f, op, u, p, ctx, pressure_symbol, n_chambers)`.
"""
function single_chamber_state(; seed = 42)
    splitform = ideal_lv_3d0d_splitform(
        RSAFDQ2022LumpedCicuitModel(; lv_pressure_given = false),
        LumpedFluidSolidCoupler([ChamberVolumeCoupling("LVChamberSurface", :Vₗᵥ, :pₗᵥ, :pₗᵥ)], :d),
    )
    f = splitform.functions[1]

    solver = HomotopyPathSolver(NewtonRaphsonSolver(; max_iter = 10, tol = 1e-2))
    op = Thunderbolt.setup_stage_operator(
        f,
        solver,
        Thunderbolt.setup_local_solver_cache(f, solver),
        PINNED_T,
    )

    n = Thunderbolt.solution_size(f)
    n_chambers = length(f.tying_info.chambers)
    u = 0.05 .* (_pinned_uniform(n, seed) .- 0.5)
    u[(n-n_chambers+1):n] .= PINNED_PRESSURE
    for chamber in f.tying_info.chambers
        chamber.V⁰ᴰval = PINNED_V⁰ᴰ
    end

    evaluation = Thunderbolt._homotopy_stage_evaluation(f, PINNED_T)
    return (; f, op, u, p = evaluation.p, ctx = evaluation.ctx, pressure_symbol = :pₗᵥ, n_chambers)
end

@testset "RSAFDQ2022 3D-0D operator" begin
    state = single_chamber_state()
    dh = state.f.structural_function.dh
    J, r = assemble_pair(state.op, state.u, state.p, state.ctx)

    pressure_dofs = algebraic_dofs(dh, state.pressure_symbol)
    n_field_dofs = ndofs(dh) - state.n_chambers

    @testset "dof numbering" begin
        # The chamber pressures are the trailing block and the field dofs keep the leading one, which
        # is what the block split below rests on. The dof the tying info carries is the same one the
        # handler hands out -- it is the row the chamber balance is assembled into.
        @test pressure_dofs ==
              [chamber.pressure_dof_index for chamber in state.f.tying_info.chambers]
        @test pressure_dofs == n_field_dofs .+ (1:state.n_chambers)
    end

    @testset "block layout" begin
        # `SchurComplementLinearSolver` factors the (1,1) block directly, which needs `op.J` to be a
        # `BlockMatrix` split as [field dofs | chamber pressures].
        @test state.op.J isa BlockMatrix{Float64, Matrix{SparseMatrixCSC{Float64, Int}}}
        @test blocklengths(axes(state.op.J, 1)) == [n_field_dofs, state.n_chambers]
    end

    @testset "facet coverage" begin
        # Every declared tying facet must be assembled by exactly one subdomain. A boundary set
        # spanning several subdomains (apex wedges beside the hexahedra) once lost the facets
        # outside the first facet's subdomain. The declared set *is* the traversal, and a facet
        # whose cell a subdomain does not own is a setup error, so what is left to check here is
        # that the per-subdomain declarations cover the chamber surfaces without duplication.
        #
        # The declaration is the union over every facet term, so it also carries the two springs the
        # myocardium wears; what may not happen is a facet appearing twice, which would assemble it
        # twice.
        integrator = state.f.structural_function.integrator
        declared = [
            facet for sdh in dh.subdofhandlers for
            facet in Thunderbolt.FerriteOperators.facet_items(integrator, sdh)
        ]
        chamber_facets = union((Set(chamber.facets) for chamber in state.f.tying_info.chambers)...)
        @test length(declared) == length(unique(declared))
        @test chamber_facets ⊆ Set(declared)
        @test setdiff(Set(declared), chamber_facets) ⊆
              union(Set(getfacetset(dh.grid, "Epicardium")), Set(getfacetset(dh.grid, "Base")))
    end

    pdof = only(pressure_dofs)
    @testset "coupling is exercised" begin
        @test maximum(abs, J[pdof, :]) > 0
        @test maximum(abs, J[:, pdof]) > 0
        @test abs(r[pdof]) > 0
    end

    @testset "coupling sparsity is the tying surface" begin
        # The pressure is the facet items' tail alone, so the cell sweep never addresses a pressure
        # entry and the allocated coupling is `FacetCoupling` over the chamber surface: the
        # displacement dofs of the cells carrying a tying facet, and no other. This mesh is one
        # element thick, so every cell carries one and the count coincides with the whole-handler
        # `CellCoupling`'s -- the two-chamber operator below is where the sets differ.
        chamber = only(state.f.tying_info.chambers)
        adjacent = unique!(reduce(vcat, [celldofs(dh, facet[1]) for facet in chamber.facets]))
        @test nnz(blocks(state.op.J)[1, 2]) == length(adjacent)
        @test nnz(blocks(state.op.J)[2, 1]) == length(adjacent)
    end

    @testset "chamber volume as a facet functional" begin
        # `r[p] = ∫_Γ V³ᴰ(u) dΓ - V⁰ᴰ`, so the reduction has to reproduce the row's integral half
        # exactly -- same integrand, same facets, only the destination differs.
        V = Thunderbolt.chamber_volume(state.op, state.pressure_symbol, state.u)
        @test V ≈ r[pdof] + PINNED_V⁰ᴰ rtol = 1.0e-12
        @test abs(V) > 0

        # ... and the direct, operator-free evaluation `create_chamber_tyings` still uses for the
        # reference volume: one integrand, two routes, `≈` because the traversals differ in order.
        chamber = only(state.f.tying_info.chambers)
        @test V ≈ Thunderbolt.compute_chamber_volume(dh, state.u, "LVChamberSurface", chamber) rtol =
            1.0e-12

        # Nothing was written into the operator by evaluating it.
        J₂, r₂ = assemble_pair(state.op, state.u, state.p, state.ctx)
        @test approx_entrywise(r₂, r)
        @test approx_entrywise(J₂, J)
    end

    @testset "derivatives" begin
        # The independent referee for the analytic tying tangent: central finite differences against
        # the operator's own residual, chamber rows included. The `-V⁰ᴰ` row is constant in `u`, so
        # its block should come out zero on both sides of the comparison.
        result = Thunderbolt.FerriteOperators.check_derivatives(
            state.op,
            (u = state.u,),
            state.p,
            state.ctx,
        )
        @test result.passed
        for (name, check) in pairs(result.checks)
            check.passed || @info "check_derivatives($name)" check
        end
    end
end

"""
    two_chamber_state(; seed = 7, device = SequentialCPUDevice())

A passive left heart with one [`Thunderbolt.Pressure3D0DVolumeCoupler`](@ref) per chamber, its
operator, and a fixed pseudo-random state to assemble at.

The couplers tie over the *closed* chamber surfaces, endocardium plus the matching face of the
valvular plate, which is what makes their volume integral the enclosed volume. The two chamber
pressures are the only unknowns outside the mesh, and they exist because the two facet terms say
so -- nothing here declares them a second time. The state is deliberately not a solution, for the
reason given at [`single_chamber_state`](@ref).

`device` is the operator's assembly device, sequential by default; the threaded-vs-sequential
equivalence testset builds the fixture a second time with `PolyesterDevice()`. Everything upstream of
the operator -- the coordinate system included -- is assembled sequentially either way, so that the
two arms of that comparison differ in the operator's device and in nothing else.

Returns `(; f, op, u, pressure_symbols, chamber_surface_names, pressure_dofs, n_u, models)`.
"""
function two_chamber_state(; seed = 7, device = Thunderbolt.SequentialCPUDevice())
    scaling_factor = 3.9
    mesh = generate_ideal_lh_mesh(
        6,
        1,
        2,
        2;
        inner_radius       = scaling_factor * 0.7,
        outer_radius       = scaling_factor * 1.0,
        longitudinal_upper = 0.4,
        apex_inner         = scaling_factor * 1.3,
        apex_outer         = scaling_factor * 1.5,
    )

    # The coordinate system is ventricular -- the atrium has no long axis of its own -- so it is
    # built on that subdomain alone, off the ventricular surfaces, with the interior annulus sheet
    # standing in for the basal one. The atrium carries a constant frame instead.
    cs = compute_lv_coordinate_system(
        mesh;
        subdomains       = ["ventricle"],
        axes             = compute_lv_axes(mesh; base = "MitralAnnulus", apex = "Apex"),
        base_name        = "MitralAnnulus",
        endocardium_name = "LVEndocardium",
        epicardium_name  = "LVEpicardium",
        strategy         = Thunderbolt.AssemblyStrategy(Thunderbolt.SequentialCPUDevice()),
    )
    ventricle_microstructure = create_microstructure_model(
        cs,
        LagrangeCollection{1}()^3,
        ODB25LTMicrostructureParameters(αendo = deg2rad(80.0), αepi = deg2rad(-65.0));
        subdomains = ["ventricle"],
    )
    atrium_microstructure = ConstantCoefficient(
        OrthotropicMicrostructure(Vec((1.0, 0.0, 0.0)), Vec((0.0, 1.0, 0.0)), Vec((0.0, 0.0, 1.0))),
    )

    pressure_symbols = (:pₗᵥ, :pₗₐ)
    # The closed chamber surfaces: endocardium plus the plate face that caps it.
    chamber_surface_names = ("LVChamberSurface", "LAChamberSurface")
    couplers = ntuple(2) do i
        Pressure3D0DVolumeCoupler(chamber_surface_names[i], :d, pressure_symbols[i])
    end
    # The plate is a stiff closure, not tissue: both chamber pressures act on it, and it carries
    # them without bulging. Same parameters as the `cm03_3d0d-coupling` tutorial.
    plate_material = PK1Model(
        BioNeoHookean(; α = 16000.0, mpU = SimpleCompressionPenalty(16000.0)),
        NoMicrostructureModel(),
    )
    # All three subdomains carry both couplers: the declarations have to agree across a domain
    # split, and each coupler only ever traverses the facets of its own chamber. The plate is not
    # along for the ride here -- both of its faces are tying facets, so its own model is what
    # declares them, and the pressure of a closed cavity does act on it.
    models = Dict(
        "ventricle" => QuasiStaticModel(
            :d,
            PK1Model(Guccione1991PassiveModel(), ventricle_microstructure),
            couplers,
        ),
        "atrium" => QuasiStaticModel(
            :d,
            PK1Model(Guccione1991PassiveModel(), atrium_microstructure),
            couplers,
        ),
        "valvular-plane" => QuasiStaticModel(:d, plate_material, couplers),
    )

    dbcs = [
        Dirichlet(:d, getnodeset(mesh, "MyocardialAnchor1"), (x, t) -> (0.0, 0.0, 0.0), [1, 2, 3]),
        Dirichlet(:d, getnodeset(mesh, "MyocardialAnchor2"), (x, t) -> (0.0, 0.0), [2, 3]),
        Dirichlet(:d, getnodeset(mesh, "MyocardialAnchor3"), (x, t) -> (0.0,), [3]),
        Dirichlet(:d, getnodeset(mesh, "MyocardialAnchor4"), (x, t) -> (0.0,), [3]),
    ]
    # The default facet rule is exact for this bilinear-geometry volume functional, and the chamber
    # volume is the enclosed volume only where the surface is closed *and* the rule is exact.
    f = semidiscretize(
        models,
        FiniteElementDiscretization(
            Dict(:d => LagrangeCollection{1}()^3);
            dbcs,
            assembly_strategy = Thunderbolt.AssemblyStrategy(device),
        ),
        mesh,
    )

    dh = f.dh
    pressure_dofs = [only(algebraic_dofs(dh, sym)) for sym in pressure_symbols]
    n_u = ndofs(dh) - length(pressure_symbols)

    # Each pressure sits in the tail of its own tying facets' local systems and nowhere else, so the
    # sparsity it needs is a `FacetCoupling` over its closed chamber surface -- what
    # `Thunderbolt._chamber_coupling` declares on the RSAFDQ path.
    couplings = Tuple(
        Thunderbolt.FerriteOperators.FacetCoupling(
            getfacetset(dh.grid, name);
            algebraic_coupling = ((:d, sym),),
        ) for (name, sym) in zip(chamber_surface_names, pressure_symbols)
    )
    strategy = Thunderbolt.AssemblyStrategy(
        Thunderbolt.FullAssembly(
            Thunderbolt.FerriteOperators.StandardOperatorSpecification(;
                algebraic_couplings = couplings,
                constraint_handler = f.ch,
            ),
        ),
        Thunderbolt.SequentialScheduling(),
        Thunderbolt.get_strategy(f).device,
    )
    op = Thunderbolt.setup_operator(
        strategy,
        f.integrator,
        dh;
        slots = Thunderbolt.THUNDERBOLT_STAGE_SLOTS,
    )

    u = 0.05 .* (_pinned_uniform(Thunderbolt.solution_size(f), seed) .- 0.5)
    u[pressure_dofs] .= PINNED_PRESSURE

    return (; f, op, u, pressure_symbols, chamber_surface_names, pressure_dofs, n_u, models)
end

"""
    deformed_coordinates(dh, u)

The grid's nodes moved by the displacement field in `u`.

With a linear geometry and a linear displacement interpolation the deformed mesh is exactly the mesh
with moved nodes, so the deformed chamber volume follows from node positions alone -- independently
of how the operator computes it.
"""
function deformed_coordinates(dh, u)
    grid = dh.grid
    displacements = zeros(Vec{3, Float64}, getnnodes(grid))
    for sdh in dh.subdofhandlers
        displacement_range = dof_range(sdh, :d)
        for cellid in sdh.cellset
            dofs = celldofs(dh, cellid)[displacement_range]
            for (i, nodeid) in enumerate(getcells(grid, cellid).nodes)
                displacements[nodeid] = Vec(ntuple(c -> u[dofs[3*(i-1)+c]], 3))
            end
        end
    end
    return [
        Ferrite.get_node_coordinate(node) + displacements[i] for
        (i, node) in enumerate(getnodes(grid))
    ]
end

@testset "Two-chamber 3D-0D operator" begin
    state = two_chamber_state()
    dh = state.f.dh
    integrator = state.f.integrator
    n_chambers = length(state.pressure_symbols)

    @testset "declared unknowns" begin
        # The couplers own the pressures, so the model derives them rather than being told.
        for model in values(state.models)
            @test collect(Thunderbolt.algebraic_variables(model)) == collect(state.pressure_symbols)
        end
        @test dh.algebraic_names == collect(state.pressure_symbols)
        @test state.pressure_dofs == state.n_u .+ (1:n_chambers)
        # Every subdomain's facet items see the same tail, in the same order -- and the cell sweep of
        # the same subdomain sees none of it.
        for sdh in dh.subdofhandlers
            @test Thunderbolt.FerriteOperators.facet_item_global_dofs(integrator, sdh) ==
                  state.pressure_dofs
            @test isempty(Thunderbolt.FerriteOperators.global_dofs(integrator, sdh))
        end
        @test Thunderbolt.FerriteOperators.algebraic_items(integrator, dh) ==
              [[dof] for dof in state.pressure_dofs]
    end

    @testset "facet coverage" begin
        declared = [
            facet for sdh in dh.subdofhandlers for
            facet in Thunderbolt.FerriteOperators.facet_items(integrator, sdh)
        ]
        surfaces = union(
            getfacetset(dh.grid, "LVChamberSurface"),
            getfacetset(dh.grid, "LAChamberSurface"),
        )
        @test length(declared) == length(surfaces)
        @test Set(declared) == Set(surfaces)
        for name in ("LVEndocardium", "LAEndocardium", "LVValvularPlane", "LAValvularPlane")
            @test Set(getfacetset(dh.grid, name)) ⊆ Set(declared)
        end

        # `facet_items` filters per subdomain, so the plate's faces reach the operator only because
        # the plate's own model declares them. Its wedges and its hexahedra are handled apart, so
        # the subdomain is several subhandlers.
        plate_cells = getcellset(dh.grid, "valvular-plane")
        plate_sdhs = [sdh for sdh in dh.subdofhandlers if all(∈(plate_cells), sdh.cellset)]
        @test !isempty(plate_sdhs)
        plate_declared = [
            facet for sdh in plate_sdhs for
            facet in Thunderbolt.FerriteOperators.facet_items(integrator, sdh)
        ]
        @test !isempty(plate_declared)
        @test Set(plate_declared) == union(
            getfacetset(dh.grid, "LVValvularPlane"),
            getfacetset(dh.grid, "LAValvularPlane"),
        )
    end

    V⁰ᴰ = [120.0, 60.0]
    ctx = Thunderbolt.TimeIntegrationContext(PINNED_T, 0.0, 0.0)
    J, r = assemble_pair(state.op, state.u, (V⁰ᴰ = V⁰ᴰ,), ctx)

    @testset "both chambers are exercised" begin
        @test size(J) == (state.n_u + n_chambers, state.n_u + n_chambers)
        for dof in state.pressure_dofs
            @test maximum(abs, J[dof, :]) > 0
            @test maximum(abs, J[:, dof]) > 0
            @test abs(r[dof]) > 0
        end
    end

    @testset "coupling sparsity is each tying surface" begin
        # One `FacetCoupling` per chamber: the pressure column carries the displacement dofs of the
        # cells owning one of ITS tying facets, plus its own diagonal, and nothing of the rest of
        # the mesh.
        for (i, name) in enumerate(state.chamber_surface_names)
            adjacent = unique!(
                reduce(vcat, [celldofs(dh, facet[1]) for facet in getfacetset(dh.grid, name)]),
            )
            @test length(nzrange(state.op.J, state.pressure_dofs[i])) == length(adjacent) + 1
            @test length(adjacent) < state.n_u
        end
    end

    @testset "each chamber row is its closed volume" begin
        # The chamber row measures `-∮ (x + d) ⋅ n̂ dΓ / 3`, which is the enclosed volume only on a
        # closed surface -- and then it agrees with the single-axis form `-∮ xᵢ n̂ᵢ dΓ` for every
        # axis. The plate is meshed, so the closure moves with the wall and the deformed surface is
        # still closed.
        deformed = deformed_coordinates(dh, state.u)
        for (i, name) in enumerate(("LVChamberSurface", "LAChamberSurface"))
            volumes = surface_volumes(dh.grid, getfacetset(dh.grid, name), deformed)
            @test volumes[1] ≈ volumes[2] rtol = 1.0e-10
            @test volumes[1] ≈ volumes[3] rtol = 1.0e-10
            @test r[state.pressure_dofs[i]] ≈ -volumes[1] - V⁰ᴰ[i] rtol = 1.0e-10
        end
    end

    @testset "each chamber volume is its own surface" begin
        # One sweep per chamber over an operator whose facet items carry both: a facet of the other
        # chamber contributes nothing, which is what makes the two values differ and what makes each
        # of them the row's own integral half.
        volumes =
            [Thunderbolt.chamber_volume(state.op, sym, state.u) for sym in state.pressure_symbols]
        for (i, V) in pairs(volumes)
            @test V ≈ r[state.pressure_dofs[i]] + V⁰ᴰ[i] rtol = 1.0e-12
        end
        @test !isapprox(volumes[1], volumes[2]; rtol = 1.0e-3)

        # A name no coupler declared would have every facet decline, which the engine reads as a
        # legitimate empty sum -- so the entry point rejects it rather than reporting a zero volume.
        err =
            @test_throws ArgumentError Thunderbolt.chamber_volume(state.op, :not_a_chamber, state.u)
        @test occursin("no tying facets for a chamber named", err.value.msg)
    end

    @testset "V⁰ᴰ enters each chamber row alone" begin
        # `r[p] -= V⁰ᴰ` is the whole dependence, so swapping the two reference volumes has to move
        # the two chamber entries by the difference and nothing else at all.
        swapped = reverse(V⁰ᴰ)
        J′, r′ = assemble_pair(state.op, state.u, (V⁰ᴰ = swapped,), ctx)
        Δ = r′ - r
        @test Δ[state.pressure_dofs] ≈ V⁰ᴰ - swapped rtol = 1.0e-12
        others = setdiff(1:length(r), state.pressure_dofs)
        @test maximum(abs, Δ[others]) ≈ 0 atol = 1.0e-12 * maximum(abs, r)
        @test approx_entrywise(J′, J)
    end
end

@testset "Threaded vs sequential two-chamber equivalence" begin
    # The only genuine multi-worker exercise of the full coupled operator (cells + facet items +
    # algebraic items + global dofs): the atomic scatter under `PolyesterDevice` has to reproduce
    # the sequential assembly, entry for entry up to summation order (never `==`, see
    # `approx_entrywise`). Needs a worker with more than one thread -- `runtests.jl` gives every
    # worker `INTEGRATION_THREADS` of them.
    seq = two_chamber_state()
    par = two_chamber_state(device = Thunderbolt.PolyesterDevice())

    V⁰ᴰ = [120.0, 60.0]
    ctx = Thunderbolt.TimeIntegrationContext(PINNED_T, 0.0, 0.0)
    J_seq, r_seq = assemble_pair(seq.op, seq.u, (V⁰ᴰ = V⁰ᴰ,), ctx)
    J_par, r_par = assemble_pair(par.op, par.u, (V⁰ᴰ = V⁰ᴰ,), ctx)

    @test approx_entrywise(J_par, J_seq)
    @test approx_entrywise(r_par, r_seq)

    # The chamber-volume reduction takes the same threaded route: per-worker partials folded in
    # worker order, so the value is the sequential one up to summation order.
    for sym in seq.pressure_symbols
        @test Thunderbolt.chamber_volume(par.op, sym, par.u) ≈
              Thunderbolt.chamber_volume(seq.op, sym, seq.u) rtol = 1.0e-12
    end
end
