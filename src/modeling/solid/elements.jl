# The quasi-static element caches, one per problem class. All three hold the same data; they differ
# only in their supertype, which is what selects the assembly protocol:
#
# | cache                                  | local problem per quadrature point | resulting system            |
# | :------------------------------------- | :--------------------------------- | :-------------------------- |
# | `QuasiStaticElementCache`              | none, or algebraic `L(F, Q) = 0`   | rate free (e.g. homotopy)   |
# | `QuasiStaticCondensedODEElementCache`  | `dₜQ = L(F, Q)`                    | ODE in mass matrix form     |
# | `QuasiStaticCondensedDAEElementCache`  | `dₜQ = L(F, dₜF, Q)`               | true DAE                    |
#
# Only the two condensed caches are `gto1` caches, so only they receive `uₑprev` and `Δt`. That is the
# point of the split: `QuasiStaticElementCache` keeps its bare-`time` methods, and because the type
# sets no longer overlap the dispatch ambiguity against FerriteOperators' `gto1` methods cannot arise
# at all — previously it had to be resolved by hand in `src/disambiguation.jl`.
#
# Julia's single inheritance means these cannot also share a Thunderbolt abstract parent (their
# FerriteOperators supertypes differ), hence the `AnyQuasiStaticElementCache` union below for the
# methods that genuinely do not care.

"""
    QuasiStaticElementCache

A generic cache to assemble elements coming from a [StructuralModel](@ref).

Right now the model has to be formulated in the first Piola Kirchhoff stress tensor and F.

This is the **rate-free** variant: its local problem, if it has one at all, carries no time derivative,
so it needs neither a timestep nor a previous state. It is what continuation solvers such as
`HomotopyPathSolver` use.
"""
struct QuasiStaticElementCache{M, CCache, CMCache, CV} <:
       FerriteOperators.AbstractVolumetricElementCache
    # This one determines the exact material
    constitutive_model::M
    # This one is a helper to evaluate coefficients in a type stable way without allocations
    coefficient_cache::CCache
    # This one is a helper to condense local variables
    internal_cache::CMCache
    # FEValue scratch for the ansatz space
    cv::CV
end

"""
    QuasiStaticCondensedODEElementCache

Quasi-static element whose internal variable follows `dₜQ = L(F, Q)`. Together with the (singular)
mass matrix of the internal variables this is an ODE in mass matrix form. Assembled through the `gto1`
protocol, so it receives `uₑprev` and `Δt`.
"""
struct QuasiStaticCondensedODEElementCache{M, CCache, CMCache, CV} <:
       FerriteOperators.AbstractGenericFirstOrderTimeVolumetricElementCache
    constitutive_model::M
    coefficient_cache::CCache
    internal_cache::CMCache
    cv::CV
end

"""
    QuasiStaticCondensedDAEElementCache

Quasi-static element whose internal variable follows `dₜQ = L(F, dₜF, Q)`. The dependence on the rate
of the deformation gradient makes this a genuine DAE rather than a mass matrix ODE. Assembled through
the `gto1` protocol.
"""
struct QuasiStaticCondensedDAEElementCache{M, CCache, CMCache, CV} <:
       FerriteOperators.AbstractGenericFirstOrderTimeVolumetricElementCache
    constitutive_model::M
    coefficient_cache::CCache
    internal_cache::CMCache
    cv::CV
end

# Methods that only touch the shared fields dispatch on this union.
const AnyQuasiStaticElementCache = Union{
    QuasiStaticElementCache,
    QuasiStaticCondensedODEElementCache,
    QuasiStaticCondensedDAEElementCache,
}

# The two `gto1` caches, i.e. those that get `uₑprev` and `Δt`.
const QuasiStaticCondensedElementCache =
    Union{QuasiStaticCondensedODEElementCache, QuasiStaticCondensedDAEElementCache}

"""
    get_number_of_internal_dofs_per_element(integrator, element_cache, sdh)

Number of condensed unknowns each cell of `sdh` carries, as an iterable of `length(sdh.cellset)`.
Used by `FerriteOperators` to lay out the [`InternalVariableHandler`](@ref). Dispatches on the
element cache, since that is what determines how many condensed unknowns a cell carries.
"""
function FerriteOperators.get_number_of_internal_dofs_per_element(
    integrator,
    element_cache::AnyQuasiStaticElementCache,
    sdh::SubDofHandler,
)
    nqp          = getnquadpoints(element_cache.cv)
    ndofs_per_qp = internal_variable_size(element_cache.constitutive_model, nothing, nothing)
    return Iterators.repeated(ndofs_per_qp*nqp, length(sdh.cellset))
end

# The condensed unknowns of a cell are appended after its finite element dofs, so the element
# unknown vector is laid out as `[fe_dofs | internal_variables]`. Sizes are computed from the cache
# rather than stored, which keeps `QuasiStaticElementCache` unchanged.
_qs_nbase(e::AnyQuasiStaticElementCache) = getnbasefunctions(e.cv)
_qs_ninternal(e::AnyQuasiStaticElementCache) =
    internal_variable_size(e.constitutive_model, nothing, nothing)*getnquadpoints(e.cv)

FerriteOperators.allocate_element_unknown_vector(e::AnyQuasiStaticElementCache, _) =
    zeros(_qs_nbase(e) + _qs_ninternal(e))

# Absolute index range of this cell's condensed block in the global solution vector. Deliberately
# shared by `load_element_unknowns!` and `store_condensed_element_unknowns!` so the two cannot drift
# apart — mismatched load/store indexing is exactly the bug class that produced the "internal
# variables only written for the first cell" defect.
@inline function _qs_internal_index_range(cell, ivh, e::AnyQuasiStaticElementCache)
    offset = internal_variable_offset(ivh, cellid(cell))
    return (offset+1):(offset+_qs_ninternal(e))
end

function FerriteOperators.load_element_unknowns!(uₑ, u, cell, ivh, e::AnyQuasiStaticElementCache)
    n, ni = _qs_nbase(e), _qs_ninternal(e)
    @views uₑ[1:n] .= u[celldofs(cell)]
    if ni > 0
        @views uₑ[(n+1):(n+ni)] .= u[_qs_internal_index_range(cell, ivh, e)]
    end
    return nothing
end

# Counterpart of `load_element_unknowns!`: write the condensed block back after the element solved it
# locally. Only correct because the `gto1` assembly writes the local solve result into `uₑ`'s tail; on
# the older path the material wrote straight to the global vector and this would copy a stale tail
# back over it.
function FerriteOperators.store_condensed_element_unknowns!(
    uₑ,
    u,
    cell,
    ivh,
    e::AnyQuasiStaticElementCache,
)
    n, ni = _qs_nbase(e), _qs_ninternal(e)
    if ni > 0
        @views u[_qs_internal_index_range(cell, ivh, e)] .= uₑ[(n+1):(n+ni)]
    end
    return nothing
end

"""
    _qs_split_unknowns(element_cache, uₑ)

Split an element unknown vector laid out `[fe_dofs | internal_variables]` into the displacement dofs
and the internal variables reshaped as `(size_per_quadrature_point, nquadpoints)`, so a quadrature
loop can address its own slice as `@view Qₑ[:, qp.i]`.

The internal block is empty for materials without condensed state, in which case the reshape yields a
`0 × nqp` array and every per-point slice is empty.
"""
@inline function _qs_split_unknowns(e::AnyQuasiStaticElementCache, uₑ)
    n, ni = _qs_nbase(e), _qs_ninternal(e)
    nqp   = getnquadpoints(e.cv)
    dₑ    = @view uₑ[1:n]
    Qₑ    = @view uₑ[(n+1):(n+ni)]
    return dₑ, reshape(Qₑ, (ni ÷ nqp, nqp))
end

# TODO how to control dispatch on required input for the material routin?
# TODO finer granularity on the dispatch here. depending on the evolution law of the internal variable this routine looks slightly different.
function assemble_element!(
    Kₑ::AbstractMatrix,
    residualₑ::AbstractVector,
    uₑ::AbstractVector,
    geometry_cache::CellCache,
    element_cache::QuasiStaticElementCache,
    time,
)
    @unpack constitutive_model, internal_cache, cv, coefficient_cache = element_cache
    ndofs = getnbasefunctions(cv)
    dₑ = @view uₑ[1:ndofs]

    reinit!(cv, geometry_cache)

    @inbounds for qp ∈ QuadratureIterator(cv)
        dΩ = getdetJdV(cv, qp)

        # A rate-free scheme has no previous configuration to form a rate from.
        ∇u = function_gradient(cv, qp, dₑ)
        kinematics = DeformationGradient(one(∇u) + ∇u)

        # Compute stress and tangent
        P, sensitivities = material_routine(
            constitutive_model,
            kinematics,
            coefficient_cache,
            internal_cache,
            geometry_cache,
            qp,
            time,
        )
        tangent = consistent_tangent(sensitivities)

        # Loop over test functions
        for i = 1:ndofs
            ∇δui = shape_gradient(cv, qp, i)

            # Add contribution to the residual from this test function
            residualₑ[i] += ∇δui ⊡ P * dΩ

            ∇δui_tangent = ∇δui ⊡ tangent # Hoisted computation
            for j = 1:ndofs
                ∇δuj = shape_gradient(cv, qp, j)
                # Add contribution to the tangent
                Kₑ[i, j] += (∇δui_tangent ⊡ ∇δuj) * dΩ
            end
        end
    end
end

function assemble_element!(
    Kₑ::AbstractMatrix,
    uₑ::AbstractVector,
    geometry_cache::CellCache,
    element_cache::QuasiStaticElementCache,
    time,
)
    @unpack constitutive_model, internal_cache, cv, coefficient_cache = element_cache
    ndofs = getnbasefunctions(cv)
    dₑ = @view uₑ[1:ndofs]

    reinit!(cv, geometry_cache)

    @inbounds for qp ∈ QuadratureIterator(cv)
        dΩ = getdetJdV(cv, qp)

        ∇u = function_gradient(cv, qp, dₑ)
        kinematics = DeformationGradient(one(∇u) + ∇u)

        # Compute "tangent only"
        _, sensitivities = material_routine(
            constitutive_model,
            kinematics,
            coefficient_cache,
            internal_cache,
            geometry_cache,
            qp,
            time,
        )
        tangent = consistent_tangent(sensitivities)

        # Loop over test functions
        for i = 1:ndofs
            ∇δui = shape_gradient(cv, qp, i)

            # Add contribution to the residual from this test function
            # residualₑ[i] += ∇δui ⊡ P * dΩ

            ∇δui_tangent = ∇δui ⊡ tangent # Hoisted computation
            for j = 1:ndofs
                ∇δuj = shape_gradient(cv, qp, j)
                # Add contribution to the tangent
                Kₑ[i, j] += (∇δui_tangent ⊡ ∇δuj) * dΩ
            end
        end
    end
end

function assemble_element!(
    residualₑ::AbstractVector,
    uₑ::AbstractVector,
    geometry_cache::CellCache,
    element_cache::QuasiStaticElementCache,
    time,
)
    @unpack constitutive_model, internal_cache, cv, coefficient_cache = element_cache
    ndofs = getnbasefunctions(cv)
    dₑ = @view uₑ[1:ndofs]

    reinit!(cv, geometry_cache)

    @inbounds for qp ∈ QuadratureIterator(cv)
        dΩ = getdetJdV(cv, qp)

        ∇u = function_gradient(cv, qp, dₑ)
        kinematics = DeformationGradient(one(∇u) + ∇u)

        # Compute stress only
        P = reduced_material_routine(
            constitutive_model,
            kinematics,
            coefficient_cache,
            internal_cache,
            geometry_cache,
            qp,
            time,
        )

        # Loop over test functions
        for i = 1:ndofs
            ∇δui = shape_gradient(cv, qp, i)

            # Add contribution to the residual from this test function
            residualₑ[i] += ∇δui ⊡ P * dΩ
        end
    end
end

# --- gto1 ---------------------------------------------------------------------------------------
#
# NOTE ON `t`: this is the time the step is solved *at*, i.e. `t + Δt` in the time integrator's own
# naming (`perform_backward_euler_step!` passes `t + Δt` into `GenericFirstOrderTimeParameters`). The
# argument is called `t` to match FerriteOperators' signature, but it is the END of the step, not its
# beginning. Verified against a two-step solve: t = 0.1 then 0.2 for tspan (0.0, 0.2) with dt = 0.1.
#
# The `gto1` ("generic time order 1") entry points take the previous element unknowns and the
# timestep as arguments instead of reading them from a cache that a solver mutated. `Qₑ`/`Qₑprev` are
# views into `uₑ`/`uₑprev` and therefore strictly cell-local — there is no global vector to offset
# into. The local solve writes its result back into the `Qₑ` view; `store_condensed_element_unknowns!`
# is what copies that tail back into the global solution vector.

# Forming the rate and stating how it linearizes are the *scheme's* two contributions, and these are
# the only places in the element layer that know which scheme is running. Within `gto1` the rate is
# the backward difference `Ḟ = dₜ(I + ∇u)`, hence `∂Ḟ/∂u = 1/Δt`.
#
# The two are defined side by side deliberately: a scheme that changes how the rate is formed must
# change its linearization in the same breath, and splitting them across the file is how a `1/Δt`
# quietly outlives the difference quotient it belongs to. Newmark replaces both with `γ/(βΔt)`;
# an SDIRK stage in rate form moves the timestep onto `∂F∂u` instead (see `KinematicLinearization`).
#
# Dispatching on the cache means the ODE cache does not pay for the second gradient evaluation: its
# local problem `dₜQ = L(F, Q)` cannot read a rate, so offering one would be dead work — and its
# linearization correspondingly has no rate slot.
@inline function compute_kinematic_quantities(
    e::QuasiStaticCondensedODEElementCache,
    qp,
    dₑ,
    dₑprev,
    Δt,
)
    ∇u = function_gradient(e.cv, qp, dₑ)
    return DeformationGradient(one(∇u) + ∇u)
end

@inline function compute_kinematic_quantities(
    e::QuasiStaticCondensedDAEElementCache,
    qp,
    dₑ,
    dₑprev,
    Δt,
)
    ∇u     = function_gradient(e.cv, qp, dₑ)
    ∇uprev = function_gradient(e.cv, qp, dₑprev)
    return DeformationGradientWithRate(one(∇u) + ∇u, (∇u - ∇uprev) / Δt)
end

@inline compute_kinematic_linearization(::QuasiStaticCondensedODEElementCache, Δt) =
    KinematicLinearization(one(Δt))

@inline compute_kinematic_linearization(::QuasiStaticCondensedDAEElementCache, Δt) =
    KinematicLinearization(one(Δt), inv(Δt))

function FerriteOperators.assemble_element_gto1!(
    Kₑ::AbstractMatrix,
    residualₑ::AbstractVector,
    uₑ::AbstractVector,
    uₑprev::AbstractVector,
    geometry_cache::CellCache,
    element_cache::QuasiStaticCondensedElementCache,
    p,
    t,
    Δt,
)
    @unpack constitutive_model, internal_cache, cv, coefficient_cache = element_cache
    ndofs = getnbasefunctions(cv)
    dₑ, Qₑ = _qs_split_unknowns(element_cache, uₑ)
    dₑprev, Qₑprev = _qs_split_unknowns(element_cache, uₑprev)

    reinit!(cv, geometry_cache)

    @inbounds for qp ∈ QuadratureIterator(cv)
        dΩ = getdetJdV(cv, qp)

        kinematics = compute_kinematic_quantities(element_cache, qp, dₑ, dₑprev, Δt)

        P, sensitivities = material_routine(
            constitutive_model,
            kinematics,
            coefficient_cache,
            internal_cache,
            geometry_cache,
            qp,
            t,
            @view(Qₑ[:, qp.i]),
            @view(Qₑprev[:, qp.i]),
            Δt,
        )
        tangent =
            consistent_tangent(sensitivities, compute_kinematic_linearization(element_cache, Δt))

        for i = 1:ndofs
            ∇δui = shape_gradient(cv, qp, i)
            residualₑ[i] += ∇δui ⊡ P * dΩ

            ∇δui_tangent = ∇δui ⊡ tangent # Hoisted computation
            for j = 1:ndofs
                ∇δuj = shape_gradient(cv, qp, j)
                Kₑ[i, j] += (∇δui_tangent ⊡ ∇δuj) * dΩ
            end
        end
    end
end

function FerriteOperators.assemble_element_gto1!(
    Kₑ::AbstractMatrix,
    uₑ::AbstractVector,
    uₑprev::AbstractVector,
    geometry_cache::CellCache,
    element_cache::QuasiStaticCondensedElementCache,
    p,
    t,
    Δt,
)
    @unpack constitutive_model, internal_cache, cv, coefficient_cache = element_cache
    ndofs = getnbasefunctions(cv)
    dₑ, Qₑ = _qs_split_unknowns(element_cache, uₑ)
    dₑprev, Qₑprev = _qs_split_unknowns(element_cache, uₑprev)

    reinit!(cv, geometry_cache)

    @inbounds for qp ∈ QuadratureIterator(cv)
        dΩ = getdetJdV(cv, qp)

        kinematics = compute_kinematic_quantities(element_cache, qp, dₑ, dₑprev, Δt)

        # Tangent only
        _, sensitivities = material_routine(
            constitutive_model,
            kinematics,
            coefficient_cache,
            internal_cache,
            geometry_cache,
            qp,
            t,
            @view(Qₑ[:, qp.i]),
            @view(Qₑprev[:, qp.i]),
            Δt,
        )
        tangent =
            consistent_tangent(sensitivities, compute_kinematic_linearization(element_cache, Δt))

        for i = 1:ndofs
            ∇δui = shape_gradient(cv, qp, i)
            ∇δui_tangent = ∇δui ⊡ tangent # Hoisted computation
            for j = 1:ndofs
                ∇δuj = shape_gradient(cv, qp, j)
                Kₑ[i, j] += (∇δui_tangent ⊡ ∇δuj) * dΩ
            end
        end
    end
end

function FerriteOperators.assemble_element_gto1!(
    residualₑ::AbstractVector,
    uₑ::AbstractVector,
    uₑprev::AbstractVector,
    geometry_cache::CellCache,
    element_cache::QuasiStaticCondensedElementCache,
    p,
    t,
    Δt,
)
    @unpack constitutive_model, internal_cache, cv, coefficient_cache = element_cache
    ndofs = getnbasefunctions(cv)
    dₑ, Qₑ = _qs_split_unknowns(element_cache, uₑ)
    dₑprev, Qₑprev = _qs_split_unknowns(element_cache, uₑprev)

    reinit!(cv, geometry_cache)

    @inbounds for qp ∈ QuadratureIterator(cv)
        dΩ = getdetJdV(cv, qp)

        kinematics = compute_kinematic_quantities(element_cache, qp, dₑ, dₑprev, Δt)

        # Stress only
        P = reduced_material_routine(
            constitutive_model,
            kinematics,
            coefficient_cache,
            internal_cache,
            geometry_cache,
            qp,
            t,
            @view(Qₑ[:, qp.i]),
            @view(Qₑprev[:, qp.i]),
            Δt,
        )

        for i = 1:ndofs
            ∇δui = shape_gradient(cv, qp, i)
            residualₑ[i] += ∇δui ⊡ P * dΩ
        end
    end
end

# FerriteOperators computes a *single* element parameter object from the volumetric cache and passes
# it to the boundary cache as well (`operators/nonlinear.jl`), whose generic `assemble_element!`
# forwards it verbatim to `assemble_facet!`. Weak boundary conditions are functions of `(u, t)` only,
# so the `gto1` payload is unwrapped to the time here rather than teaching every facet cache about it.
#
# CONSTRAINT: if a surface cache ever genuinely needs `uₑprev`/`Δt`, it must subtype
# `FerriteOperators.AbstractGenericFirstOrderTimeSurfaceElementCache` *and* these methods must be
# narrowed, because as written they would strip the payload before it reaches it.
#
# These methods also discard `pfot.pₑ`, which is correct only for as long as the facet caches treat
# their single trailing argument as the time — they pass it straight to `evaluate_coefficient`. This
# is the surface half of the `t`/`p` conflation described in the `nlsolve!` docstring: once `p`
# carries differentiable material parameters, boundary conditions that depend on such a parameter
# (a pressure amplitude, say) will need it forwarded rather than dropped.
# FerriteOperators computes a *single* element parameter object from the volumetric cache and passes
# it to the boundary cache as well (`operators/nonlinear.jl:12,19`), whose generic `assemble_element!`
# forwards it verbatim to `assemble_facet!`. Weak boundary conditions are functions of `(u, t)` only,
# so the payload is unwrapped to the time here rather than teaching every facet cache about it.
#
# `CompositeSurfaceElementCache` needs its own copy of each method: FerriteOperators specializes
# `assemble_element!` on it, so against the abstract-cache method neither signature would dominate
# (one is more specific in the cache, the other in the parameters) and the pair would be ambiguous.
#
# CONSTRAINT: if a surface cache ever genuinely needs `uₑprev`/`Δt`, it must subtype
# `FerriteOperators.AbstractGenericFirstOrderTimeSurfaceElementCache` *and* these methods must be
# narrowed, because as written they strip the payload before it reaches it.
#
# These methods also discard `pfot.pₑ`, which is correct only while the facet caches treat their single
# trailing argument as the time — they pass it straight to `evaluate_coefficient`. This is the surface
# half of the `t`/`p` conflation described in the `nlsolve!` docstring: once `p` carries parameters
# being optimized, a boundary condition depending on such a parameter needs it forwarded, not dropped.
#
# The right long-term fix is upstream: `AssembleLinearizationJR` should query element parameters for
# the boundary cache separately instead of reusing the volumetric one.

FerriteOperators.assemble_element!(
    Kₑ::AbstractMatrix,
    residualₑ::AbstractVector,
    uₑ::AbstractVector,
    geometry_cache::CellCache,
    facet_cache::FerriteOperators.AbstractSurfaceElementCache,
    pfot::FerriteOperators.GenericFirstOrderTimeElementParameters,
) = FerriteOperators.assemble_element!(Kₑ, residualₑ, uₑ, geometry_cache, facet_cache, pfot.t)

FerriteOperators.assemble_element!(
    Kₑ::AbstractMatrix,
    uₑ::AbstractVector,
    geometry_cache::CellCache,
    facet_cache::FerriteOperators.AbstractSurfaceElementCache,
    pfot::FerriteOperators.GenericFirstOrderTimeElementParameters,
) = FerriteOperators.assemble_element!(Kₑ, uₑ, geometry_cache, facet_cache, pfot.t)

FerriteOperators.assemble_element!(
    residualₑ::AbstractVector,
    uₑ::AbstractVector,
    geometry_cache::CellCache,
    facet_cache::FerriteOperators.AbstractSurfaceElementCache,
    pfot::FerriteOperators.GenericFirstOrderTimeElementParameters,
) = FerriteOperators.assemble_element!(residualₑ, uₑ, geometry_cache, facet_cache, pfot.t)

FerriteOperators.assemble_element!(
    Kₑ::AbstractMatrix,
    residualₑ::AbstractVector,
    uₑ::AbstractVector,
    geometry_cache::CellCache,
    facet_cache::FerriteOperators.CompositeSurfaceElementCache,
    pfot::FerriteOperators.GenericFirstOrderTimeElementParameters,
) = FerriteOperators.assemble_element!(Kₑ, residualₑ, uₑ, geometry_cache, facet_cache, pfot.t)

FerriteOperators.assemble_element!(
    Kₑ::AbstractMatrix,
    uₑ::AbstractVector,
    geometry_cache::CellCache,
    facet_cache::FerriteOperators.CompositeSurfaceElementCache,
    pfot::FerriteOperators.GenericFirstOrderTimeElementParameters,
) = FerriteOperators.assemble_element!(Kₑ, uₑ, geometry_cache, facet_cache, pfot.t)

FerriteOperators.assemble_element!(
    residualₑ::AbstractVector,
    uₑ::AbstractVector,
    geometry_cache::CellCache,
    facet_cache::FerriteOperators.CompositeSurfaceElementCache,
    pfot::FerriteOperators.GenericFirstOrderTimeElementParameters,
) = FerriteOperators.assemble_element!(residualₑ, uₑ, geometry_cache, facet_cache, pfot.t)

# ------------------------------------------------------------------------------------------------

"""
    quasistatic_element_cache_type(evolution::InternalVariableEvolution)

Which quasi-static element cache a material needs, decided by the evolution law of its internal
variable — not by the time integrator. A material with no internal variable, or with one that carries
no time derivative, is rate free; `dₜQ = L(F, Q)` is a mass matrix ODE; `dₜQ = L(F, dₜF, Q)` is a DAE.

The question is asked of the model via [`internal_variable_evolution`](@ref) rather than of the state
cache: the `Empty…CondensationMaterialStateCache` types record only that a model needs no extra
scratch space, which is a different question and gives the wrong answer for an unwrapped rate
dependent model.
"""
quasistatic_element_cache_type(::NoEvolution) = QuasiStaticElementCache
quasistatic_element_cache_type(::FirstOrderEvolution) = QuasiStaticCondensedODEElementCache
quasistatic_element_cache_type(::RateCoupledEvolution) = QuasiStaticCondensedDAEElementCache

function setup_quasistatic_element_cache(
    material_model::AbstractMaterialModel,
    qr::QuadratureRule,
    sdh::SubDofHandler,
    cv::CellValues,
)
    internal_cache = setup_internal_cache(material_model, qr, sdh)
    return quasistatic_element_cache_type(internal_variable_evolution(material_model))(
        material_model,
        setup_coefficient_cache(material_model, qr, sdh),
        internal_cache,
        cv,
    )
end
function setup_element_cache(model::QuasiStaticModel, qr::QuadratureRule, sdh::SubDofHandler)
    @assert length(sdh.dh.field_names) == 1 "Support for multiple fields not yet implemented."
    field_name = first(sdh.dh.field_names)
    ip         = Ferrite.getfieldinterpolation(sdh, field_name)
    ip_geo     = geometric_subdomain_interpolation(sdh)
    cv         = CellValues(qr, ip, ip_geo)
    return setup_quasistatic_element_cache(model.material_model, qr, sdh, cv)
end

duplicate_for_device(device, model::AbstractMaterialModel) = model
# Reconstruct the *same* concrete cache type, so the problem class survives the move to a device.
function duplicate_for_device(device, cache::AnyQuasiStaticElementCache)
    return (typeof(cache).name.wrapper)(
        duplicate_for_device(device, cache.constitutive_model),
        duplicate_for_device(device, cache.coefficient_cache),
        duplicate_for_device(device, cache.internal_cache),
        duplicate_for_device(device, cache.cv),
    )
end
