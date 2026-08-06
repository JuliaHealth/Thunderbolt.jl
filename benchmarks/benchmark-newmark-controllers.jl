# Step size control for `NewmarkSolver`, on a freely vibrating bar.
#
# Two things this measures, both of which back choices made in `src/solver/time/controllers.jl`:
#
#  * the Soederlind coefficients of the default `PIDController(3//5, -1//5, 0)` against a pure
#    integral and a PI setting, all through the same acceptance rule;
#  * the step count law `tol^(-1/3)`, which is the sharp check that `adaptive_order` and the
#    Zienkiewicz-Xie estimate's order agree. A wrong exponent shows up here and nowhere else.
#
# What it deliberately does *not* measure is Thunderbolt's controller against the ones in
# `OrdinaryDiffEqCore`. Those cannot drive this integrator: `should_accept_step` has methods for
# `PIDControllerCache` and the continuation controllers only, so an upstream controller cache reaches
# no method. That is the state the vendoring left behind, and it is why a cross-controller comparison
# has to be a coefficient comparison instead.
#
# Run with `julia --project=. benchmarks/benchmark-newmark-controllers.jl`.

using Thunderbolt
using LinearAlgebra
import SciMLBase

const ORTHO_MS = ConstantCoefficient(
    OrthotropicMicrostructure(Vec((1.0, 0.0, 0.0)), Vec((0.0, 1.0, 0.0)), Vec((0.0, 0.0, 1.0))),
)

"""
A clamped bar, light enough that several periods of free vibration fit into a short run. The light
density is what makes the step size controller work for its living: at `ρ = 1e3` the motion is slow
enough that any controller looks fine.
"""
function vibrating_bar(; ncells = (2, 1, 1), ρ = 1.0e-2)
    mesh = generate_mesh(Hexahedron, ncells, Vec((0.0, 0.0, 0.0)), Vec((1.0, 0.2, 0.2)))
    model = ElastodynamicsModel(
        :d,
        :v,
        PK1Model(Guccione1991PassiveModel(), ORTHO_MS),
        (),
        ConstantCoefficient(ρ),
    )
    dbcs = [Dirichlet(:d, getfacetset(mesh, "left"), (x, t) -> [0.0, 0.0, 0.0], [1, 2, 3])]
    return semidiscretize(
        model,
        FiniteElementDiscretization(Dict(:d => LagrangeCollection{1}()^3); dbcs),
        mesh,
    )
end

# Transverse velocity growing along the bar, so the free end swings fastest.
function bending_velocity(f, amplitude)
    dh = f.structural.dh
    v0 = zeros(ndofs(dh))
    for cell in CellIterator(dh)
        for (i, node) in enumerate(getcoordinates(cell))
            dofs = celldofs(cell)[(3(i - 1) + 1):(3i)]
            v0[dofs[2]] = amplitude * node[1]
        end
    end
    Ferrite.apply_zero!(v0, Thunderbolt.getch(f.structural))
    return v0
end

function run_controller(controller, reltol; tend = 2.2, dt = 5.0e-3)
    f = vibrating_bar()
    u0 = zeros(solution_size(f))
    Thunderbolt.default_initial_condition!(u0, f)
    integrator = init(
        ElastodynamicsProblem(f, u0, bending_velocity(f, 0.2), (0.0, tend)),
        NewmarkSolver(),
        dt = dt,
        adaptive = true,
        controller = controller,
        reltol = reltol,
        abstol = reltol * 1.0e-3,
        verbose = false,
    )
    solve!(integrator)
    return (
        retcode = integrator.sol.retcode,
        accepted = integrator.stats.naccept,
        rejected = integrator.stats.nreject,
        u = copy(integrator.u),
    )
end

# Coefficient settings of the *same* controller, so the acceptance rule and the limiter are held
# fixed and only the Soederlind exponents vary.
const CONTROLLERS = (
    "integral only" => Thunderbolt.PIDController(1 // 1, 0 // 1, 0 // 1),
    "proportional-integral" => Thunderbolt.PIDController(7 // 20, 1 // 5, 0 // 1),
    "default" => Thunderbolt.PIDController(3 // 5, -1 // 5, 0 // 1),
)

function compare(; reltols = (1.0e-3, 1.0e-4))
    for reltol in reltols
        println("\nreltol = $reltol")
        println(rpad("controller", 16), lpad("accepted", 10), lpad("rejected", 10), lpad("rej %", 8))
        for (name, controller) in CONTROLLERS
            r = run_controller(controller, reltol)
            total = r.accepted + r.rejected
            pct = total == 0 ? 0.0 : round(100 * r.rejected / total; digits = 1)
            println(rpad(name, 16), lpad(r.accepted, 10), lpad(r.rejected, 10), lpad(pct, 8))
            r.retcode == SciMLBase.ReturnCode.Success || println("    ! retcode $(r.retcode)")
        end
    end
end

"""
Step count against tolerance. The Zienkiewicz-Xie estimate is `O(dt^3)`, so a controller applying the
matching exponent gives a step count scaling like `tol^(-1/3)`, i.e. successive ratios near
`10^(1/3) = 2.154`. A wrong `adaptive_order` shows up here immediately and nowhere else.
"""
function step_count_law(; reltols = (1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6))
    controller = last(last(CONTROLLERS))
    counts = Int[]
    println("\nstep count law (target ratio $(round(10^(1 / 3); digits = 3)))")
    for reltol in reltols
        r = run_controller(controller, reltol)
        push!(counts, r.accepted)
        ratio = length(counts) > 1 ? round(counts[end] / counts[end-1]; digits = 3) : ""
        println("  reltol $(rpad(reltol, 9)) accepted $(lpad(r.accepted, 6))   ratio $ratio")
    end
    return counts
end

if abspath(PROGRAM_FILE) == @__FILE__
    compare()
    step_count_law()
end
