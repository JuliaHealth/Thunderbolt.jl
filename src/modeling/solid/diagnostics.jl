# Kinematic diagnostics for solid mechanics solves.
#
# Two failure modes motivate these, and neither one announces itself.
#
# *Element inversion.* A hyperelastic energy is only a model of the material on `det F > 0`, and
# several of the compressibility penalties in `energies.jl` are written in `I₃ = det(C) = (det F)²`,
# which cannot see the sign of `det F` at all — so the folded configuration is an equally admissible
# minimum as far as Newton is concerned. A solve that folds an element does not fail; it converges.
#
# *Drift along a null space of the tangent.* A body held only by boundary conditions that leave some
# rigid motion free — a `NormalSpringBC`, which resists the normal displacement but not tangential
# sliding — can rotate at no energy cost. That too converges, and `det F` stays exactly 1 while the
# displacement grows without bound. Only comparing the displacement gradient against the *strain*
# separates "the body moved" from "the body deformed", which is why both are reported.

"""
    DeformationGradientReport

Summary of the kinematics of a solve over its quadrature points, as produced by
[`deformation_gradient_report`](@ref).

* `n_nonpositive > 0` — the configuration is folded, and no amount of solver convergence makes it a
  valid deformation.
* `max_deviation` large with `max_strain` near zero — the body is moving rigidly rather than
  deforming, which is the signature of a solve sliding down a null space of the tangent.

`n_subdomains` is how many `SubDofHandler`s actually contributed. It is worth checking on a
multiphysics or mixed-dimensional problem, where the ones carrying no displacement field are skipped.
"""
struct DeformationGradientReport
    minJ::Float64
    maxJ::Float64
    # Where the minimum sits, so a suspicious solve can be looked at in the right place.
    cell::Int
    qp::Int
    n_nonpositive::Int
    n_quadrature_points::Int
    # max ||F - I||: how far the configuration is from the reference one.
    max_deviation::Float64
    # max ||E||, E = (FᵀF - I)/2: how much of that is actual straining. A rigid motion has E = 0
    # however large it is.
    max_strain::Float64
    n_subdomains::Int
end

"""
    is_inverted(report)

Whether any quadrature point of `report` has a non-positive `det F`.
"""
is_inverted(report::DeformationGradientReport) = report.n_nonpositive > 0

function Base.show(io::IO, r::DeformationGradientReport)
    print(
        io,
        "DeformationGradientReport(det F ∈ [",
        round(r.minJ, sigdigits = 6),
        ", ",
        round(r.maxJ, sigdigits = 6),
        "], min at cell ",
        r.cell,
        " qp ",
        r.qp,
        ", ",
        r.n_nonpositive,
        "/",
        r.n_quadrature_points,
        " non-positive, max ||F-I|| = ",
        round(r.max_deviation, sigdigits = 4),
        ", max ||E|| = ",
        round(r.max_strain, sigdigits = 4),
        ", ",
        r.n_subdomains,
        " subdomain(s))",
    )
end

# A single symbol is the common case; a multi-domain model may name its displacement differently per
# subdomain, so a collection is accepted too.
_field_symbols(field::Symbol) = (field,)
_field_symbols(fields) = Tuple(fields)

# The first requested field this subdomain actually carries, or `nothing` if it carries none -- which
# is how a subdomain that is not doing mechanics (an electrophysiology block, a Purkinje network) is
# passed over rather than differentiated as though it were.
function _displacement_field(sdh::SubDofHandler, syms)
    for sym in syms
        sym ∈ sdh.field_names && return sym
    end
    return nothing
end

@doc raw"""
    deformation_gradient_report(f, u; qr_order = 2)
    deformation_gradient_report(dh::Ferrite.AbstractDofHandler, u, fields; qr_order = 2)

Kinematic summary of the configuration `u`: the range of ``\det \bm{F}``, how many quadrature points
are folded, and how far the body has moved (``\max \lVert \bm{F} - \bm{I} \rVert``) against how much it
has actually strained (``\max \lVert \bm{E} \rVert``).

The first form takes a semidiscrete function and reads the displacement symbols off its model. The
second takes them explicitly, as a `Symbol` or a collection of them, which is what to use for a bare
`DofHandler` or when the field to differentiate is not the one the model would name.

Only subdomains carrying one of `fields` are visited, so a multiphysics handler -- mechanics next to
electrophysiology, or a mixed-dimensional grid with a Purkinje network -- reports on its mechanical
part alone. A subdomain whose displacement field cannot yield a square ``\bm{F}`` (an embedded shell
or cable, where the reference dimension differs from the component count) is skipped for the same
reason: ``\det \bm{F}`` is not defined there. `n_subdomains` on the result says how many contributed,
and it is an error for that to be zero.

Reach for this when a solve "converges" to something implausible, and especially when a mechanics
problem diverges only for *some* inputs -- a perturbed microstructure, a different step size. That
pattern usually means the solve is finding a fold or a free rigid mode rather than meeting a genuinely
harder problem, and neither is visible in the residual.

`qr_order` need not match the quadrature the solve used; this samples the field, it does not integrate
it.
"""
function deformation_gradient_report(
    dh::Ferrite.AbstractDofHandler,
    u::AbstractVector,
    fields;
    qr_order::Int = 2,
)
    syms       = _field_symbols(fields)
    minJ, maxJ = Inf, -Inf
    cell, qp   = 0, 0
    nnonpos    = 0
    npoints    = 0
    maxdev     = 0.0
    maxstrain  = 0.0
    nsub       = 0
    nskipped   = 0
    for sdh in dh.subdofhandlers
        field_name = _displacement_field(sdh, syms)
        field_name === nothing && continue
        ip = Ferrite.getfieldinterpolation(sdh, field_name)
        # `∇u` is square only when the field has one component per reference direction. Anything else
        # is not a bulk displacement and has no `det F`.
        if Ferrite.n_components(ip) != Ferrite.getrefdim(ip)
            nskipped += 1
            continue
        end
        ip_geo = geometric_subdomain_interpolation(sdh)
        cv     = CellValues(QuadratureRule{Ferrite.getrefshape(ip)}(qr_order), ip, ip_geo)
        dofr   = Ferrite.dof_range(sdh, field_name)
        nsub   += 1
        for cc in CellIterator(sdh)
            reinit!(cv, cc)
            uₑ = @view u[celldofs(cc)]
            for q = 1:getnquadpoints(cv)
                ∇u        = function_gradient(cv, q, @view uₑ[dofr])
                F         = one(∇u) + ∇u
                J         = det(F)
                npoints   += 1
                maxdev    = max(maxdev, norm(∇u))
                maxstrain = max(maxstrain, norm((tdot(F) - one(F)) / 2))
                J ≤ 0 && (nnonpos += 1)
                if J < minJ
                    minJ, cell, qp = J, cellid(cc), q
                end
                J > maxJ && (maxJ = J)
            end
        end
    end
    nsub == 0 && _no_mechanics_subdomain_error(dh, syms, nskipped)
    return DeformationGradientReport(
        minJ,
        maxJ,
        cell,
        qp,
        nnonpos,
        npoints,
        maxdev,
        maxstrain,
        nsub,
    )
end

function _no_mechanics_subdomain_error(dh, syms, nskipped)
    available = unique(Iterators.flatten(sdh.field_names for sdh in dh.subdofhandlers))
    error(
        "No subdomain of this handler carries a differentiable displacement field out of $(syms). " *
        "Fields present: $(collect(available))." *
        (
            nskipped > 0 ?
            " $(nskipped) subdomain(s) do carry one, but on cells whose reference dimension differs " *
            "from its component count, where `det F` is undefined." : ""
        ),
    )
end

deformation_gradient_report(f::AbstractSolidMechanicsFunction, u::AbstractVector; kwargs...) =
    deformation_gradient_report(f.dh, u, displacement_symbols(f); kwargs...)

"""
    displacement_symbols(f)

The field symbols a solid mechanics function solves for, as named by its model.

Read off the integrator rather than off the `DofHandler`, because the handler does not distinguish a
displacement from any other field it happens to carry: an `ElastodynamicsFunction`'s handler also holds
the velocity, and a multiphysics one holds fields belonging to other physics entirely.
"""
displacement_symbols(f::AbstractSolidMechanicsFunction) =
    _integrator_symbols(get_volume_integrator(f))

_integrator_symbols(integrator::NonlinearIntegrator) = integrator.syms
_integrator_symbols(integrator::NonlinearMultiDomainIntegrator2) = Tuple(
    unique(
        Iterators.flatten(
            _integrator_symbols(subintegrator) for
            subintegrator in values(integrator.subintegrators)
        ),
    ),
)

"""
    DeformationMonitor(; inner_monitor, warn_below = 0.0, qr_order = 2)

Nonlinear solver monitor that reports the first Newton iterate whose ``\\det \\bm{F}`` drops to
`warn_below` or less.

Plugged into a solver the same way [`VTKNewtonMonitor`](@ref) is
(`NewtonRaphsonSolver(monitor = DeformationMonitor())`), and it wraps an inner monitor so it composes
with the ordinary progress reporting rather than replacing it.

Warning *per Newton iteration* rather than per step is the point: an iterate that folds an element is
usually the one that sends the solve onto the spurious branch, and by the time the step has converged
the evidence of where it went wrong is gone. `warn_below` can be raised above zero to catch a solve
that is merely approaching a fold.
"""
Base.@kwdef struct DeformationMonitor
    # Untyped for the same reason `NewtonRaphsonSolver.monitor` is: entered once per Newton iteration.
    inner_monitor::Any = DefaultProgressMonitor()
    warn_below::Float64 = 0.0
    qr_order::Int = 2
end

function nonlinear_step_monitor(cache, time, f, u, monitor::DeformationMonitor)
    nonlinear_step_monitor(cache, time, f, u, monitor.inner_monitor)
    report = deformation_gradient_report(f, u; qr_order = monitor.qr_order)
    if report.minJ ≤ monitor.warn_below
        @warn "Deformation gradient degenerate at t=$time, Newton iteration $(cache.iter): $report"
    end
    return nothing
end

nonlinear_finalize_monitor(nlcache, time, f, monitor::DeformationMonitor) =
    nonlinear_finalize_monitor(nlcache, time, f, monitor.inner_monitor)

# --- energies ----------------------------------------------------------------------------------------
#
# Both are functionals of an operator, reduced by `FerriteOperators.evaluate_functional` with the
# strategy and the element caches the operator was set up with. The kernels live with the elements:
# `StrainEnergyFunctional` in `elements.jl`, `KineticEnergyFunctional` in `core/mass.jl`.

"""
    strain_energy(op, states, t) -> Float64
    strain_energy(f, u, t) -> Float64

The strain energy ``\\int_{\\Omega_0} \\Psi \\, \\mathrm{d}V`` at time `t`, reduced over the cells of
a solid mechanics operator with [`StrainEnergyFunctional`](@ref). Only the energetic part of each
material counts -- see [`strain_energy_density`](@ref) -- and boundary terms such as a `RobinBC` do
not, so for a model with elastic supports this is not the whole stored energy.

`op` is an operator over the integrator of a `QuasiStaticFunction` `f`, e.g. one from
`setup_evaluation_operator`, which holds no matrix. `states` carries its slots: `(u = u,)` with `u`
in the numbering of `f.dh`, and `q = InternalSource(u)` on top where a material has condensed
state. Nothing is written into `op`.

The second form takes a `QuasiStaticFunction` or an `ElastodynamicsFunction` with its
solution vector, and sets an evaluation operator up for the one call. Where the energy is wanted
repeatedly, hold the operator instead:

    op = Thunderbolt.setup_evaluation_operator(
        Thunderbolt.get_strategy(f), f.integrator, f.dh; slots = (:u, :q),
    )

For an `ElastodynamicsFunction` these are the fields of `f.structural`. The structural operator
of a Newmark stage is `stage_op.op`, its mass operator `stage_op.M`.
"""
function strain_energy(op, states::NamedTuple, t)
    !haskey(states, :q) ||
        states.q isa InternalSource ||
        throw(ArgumentError("the `q` slot takes `InternalSource(u)`, got a $(typeof(states.q))"))
    return FerriteOperators.evaluate_functional(
        op,
        StrainEnergyFunctional(),
        states,
        nothing,
        TimeIntegrationContext(t, zero(t), zero(t)),
    )
end

function strain_energy(f::QuasiStaticFunction, u::AbstractVector, t)
    op = setup_evaluation_operator(get_strategy(f), f.integrator, f.dh; slots = (:u, :q))
    states = has_internal_variables(f) ? (u = u, q = InternalSource(u)) : (u = u,)
    return strain_energy(op, states, t)
end

# The state carries the velocity next to the displacement; the energy is a functional of the
# structural problem's numbering.
function strain_energy(f::ElastodynamicsFunction, u::AbstractVector, t)
    z = zeros(eltype(u), solution_size(f.structural))
    gather!(z, u, f.state_mapping)
    return strain_energy(f.structural, z, t)
end

"""
    kinetic_energy(op, states, t) -> Float64
    kinetic_energy(f::ElastodynamicsFunction, u, t) -> Float64

The kinetic energy ``\\tfrac12 \\int_{\\Omega_0} \\rho \\, v \\cdot v \\, \\mathrm{d}V`` at time `t`,
reduced over the cells of a mass operator with [`KineticEnergyFunctional`](@ref). The mass element
integrates it with its own quadrature, so this is exactly ``\\tfrac12 v^{\\mathrm{T}} M v`` of the
consistent mass.

`op` is an operator over a consistent mass integrator, and `states = (u = v,)` carries the velocity
in its `u` slot, in the numbering of the operator's dof handler. The second form reads the velocity
out of the state `u` and sets an evaluation operator up for the one call, as [`strain_energy`](@ref)
does.
"""
kinetic_energy(op, states::NamedTuple, t) = FerriteOperators.evaluate_functional(
    op,
    KineticEnergyFunctional(),
    states,
    nothing,
    TimeIntegrationContext(t, zero(t), zero(t)),
)

function kinetic_energy(f::ElastodynamicsFunction, u::AbstractVector, t)
    op = setup_evaluation_operator(get_strategy(f), f.mass_term, f.structural.dh)
    return kinetic_energy(op, (u = u[f.velocity_dofs],), t)
end
