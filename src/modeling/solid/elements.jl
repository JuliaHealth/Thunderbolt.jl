# The quasi-static element caches, one per problem class. All hold the same data; they differ only in
# the kinematics their local problem consumes, which is what selects the slots they read and the
# request kinds they serve:
#
# | cache                                    | local problem per quadrature point | slots read              |
# | :--------------------------------------- | :--------------------------------- | :---------------------- |
# | `QuasiStaticElementCache`                | none                               | `u`                     |
# | `QuasiStaticCondensedSteadyElementCache` | algebraic `0 = L(F, Q)`            | `u`, `q`                |
# | `QuasiStaticCondensedODEElementCache`    | `dₜQ = L(F, Q)`                    | `u`, `q`, `qprev`       |
# | `QuasiStaticCondensedDAEElementCache`    | `dₜQ = L(F, dₜF, Q)`               | `u`, `q`, `qprev`, `v`  |
#
# The DAE cache is the only one whose residual depends on the deformation rate, so it is the only one
# whose tangent carries the scheme's rate chain rule — see `assemble_cell!(::WeightedJacobianRequest, …)`.
#
# Julia's single inheritance means these cannot also share a Thunderbolt abstract parent, hence the
# `AnyQuasiStaticElementCache` union below for the methods that genuinely do not care.

"""
    QuasiStaticElementCache

A generic cache to assemble elements coming from a [StructuralModel](@ref).

Right now the model has to be formulated in the first Piola Kirchhoff stress tensor and F.

This is the variant for a material with **no condensed state at all**: nothing is solved per
quadrature point, so the kernels evaluate the material directly and there is no corrector to store.
A material that condenses an algebraic constraint gets
[`QuasiStaticCondensedSteadyElementCache`](@ref) instead.
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
    QuasiStaticCondensedSteadyElementCache

Quasi-static element whose internal variable solves the algebraic constraint `0 = L(F, Q)`. Condensed
like the two below, but rate free: no timestep, no previous state, hence no `qprev` slot. That is
what makes it the one condensed element a continuation solver such as [`HomotopyPathSolver`](@ref)
can carry.

`Q` persists in the solution vector between calls, so every local solve is warm started from the
previous trial point rather than from a fixed guess.

`correctors` holds the per-quadrature-point tangent correction `∂P/∂Q · dQ/dF`, exactly as for
[`QuasiStaticCondensedODEElementCache`](@ref).
"""
struct QuasiStaticCondensedSteadyElementCache{M, CCache, CMCache, CV, Corr} <:
       FerriteOperators.AbstractVolumetricElementCache
    constitutive_model::M
    coefficient_cache::CCache
    internal_cache::CMCache
    cv::CV
    correctors::Corr
end

"""
    QuasiStaticCondensedODEElementCache

Quasi-static element whose internal variable follows `dₜQ = L(F, Q)`. Together with the (singular)
mass matrix of the internal variables this is an ODE in mass matrix form. Its local problem reads the
deformation gradient alone, so its tangent is a plain `∂F/∂u`.

`correctors` holds the per-quadrature-point tangent correction `∂P/∂Q · dQ/dF` that
[`FerriteOperators.condense_cell!`](@ref) stored, keyed by cellid; the `Consistent` kernels read it
through [`FerriteOperators.item_state`](@ref), which names the cell if no condensation phase ran.
"""
struct QuasiStaticCondensedODEElementCache{M, CCache, CMCache, CV, Corr} <:
       FerriteOperators.AbstractVolumetricElementCache
    constitutive_model::M
    coefficient_cache::CCache
    internal_cache::CMCache
    cv::CV
    correctors::Corr
end

"""
    QuasiStaticCondensedDAEElementCache

Quasi-static element whose internal variable follows `dₜQ = L(F, dₜF, Q)`. The dependence on the rate
of the deformation gradient makes this a genuine DAE rather than a mass matrix ODE, and it is what
makes the `v` slot and the weighted tangent below part of this cache's contract.

`correctors` carries both corrections of the local solve, `∂P/∂Q · dQ/dF` and `∂P/∂Q · dQ/dḞ`, since
a weighted sweep weighs the two separately.
"""
struct QuasiStaticCondensedDAEElementCache{M, CCache, CMCache, CV, Corr} <:
       FerriteOperators.AbstractVolumetricElementCache
    constitutive_model::M
    coefficient_cache::CCache
    internal_cache::CMCache
    cv::CV
    correctors::Corr
end

# Methods that only touch the shared fields dispatch on this union.
const AnyQuasiStaticElementCache = Union{
    QuasiStaticElementCache,
    QuasiStaticCondensedSteadyElementCache,
    QuasiStaticCondensedODEElementCache,
    QuasiStaticCondensedDAEElementCache,
}

# Every cache that condenses, i.e. carries a `q` slot and a corrector store.
const AnyQuasiStaticCondensedElementCache = Union{
    QuasiStaticCondensedSteadyElementCache,
    QuasiStaticCondensedODEElementCache,
    QuasiStaticCondensedDAEElementCache,
}

# The two caches whose internal variable is condensed out of a *time dependent* local problem, i.e.
# those that get `q`, `qprev` and a timestep.
const QuasiStaticCondensedElementCache =
    Union{QuasiStaticCondensedODEElementCache, QuasiStaticCondensedDAEElementCache}

FerriteOperators.reinit_values!(e::AnyQuasiStaticElementCache, cell) = reinit!(e.cv, cell)
Ferrite.getnquadpoints(e::AnyQuasiStaticElementCache) = getnquadpoints(e.cv)

# The condensed state is written by `condense_cell!` and gathered back as the `q` slot; declaring this
# is what makes the sensitivity admissibility rules apply to these caches.
FerriteOperators.has_internal_state(::Type{<:QuasiStaticCondensedSteadyElementCache}) = true
FerriteOperators.has_internal_state(::Type{<:QuasiStaticCondensedODEElementCache}) = true
FerriteOperators.has_internal_state(::Type{<:QuasiStaticCondensedDAEElementCache}) = true

"""
    get_number_of_internal_dofs_per_element(integrator, element_cache, sdh)

Number of condensed unknowns each cell of `sdh` carries, as an iterable of `length(sdh.cellset)`.
Used by `FerriteOperators` to lay out the [`InternalVariableHandler`](@ref). Dispatches on the
element cache, since that is what determines how many condensed unknowns a cell carries.
"""
function get_number_of_internal_dofs_per_element(
    integrator,
    element_cache::AnyQuasiStaticElementCache,
    sdh::SubDofHandler,
)
    nqp = getnquadpoints(element_cache.cv)
    # `nothing, nothing`: this asks for *one* size and hands it to every cell and every quadrature point
    # of the subdomain, so it only holds for a material whose local state size is constant. A model whose
    # size varies per point -- FE², see `internal_variable_size` -- cannot be sized here, and the
    # `InternalVariableHandler` block this feeds could not express the result either.
    ndofs_per_qp = internal_variable_size(element_cache.constitutive_model, nothing, nothing)
    return Iterators.repeated(ndofs_per_qp*nqp, length(sdh.cellset))
end

_qs_nbase(e::AnyQuasiStaticElementCache) = getnbasefunctions(e.cv)
# Multiplying one size by the quadrature count assumes every point of the cell carries the same amount of
# state; see `internal_variable_size` for when that stops being true.
_qs_ninternal(e::AnyQuasiStaticElementCache) =
    internal_variable_size(e.constitutive_model, nothing, nothing)*getnquadpoints(e.cv)

"""
    _qs_internal_block(element_cache, Qₑ)

View the cell's condensed unknowns as `(size_per_quadrature_point, nquadpoints)`, so a quadrature loop
can address its own slice as `@view Qₑ[:, qp.i]`.

The block is empty for materials without condensed state, in which case the reshape yields a
`0 × nqp` array and every per-point slice is empty.

The reshape is what makes "one size per quadrature point" a structural assumption of the element
layout, not merely of the caller: a material whose state size varies between the points of a cell
(FE², see [`internal_variable_size`](@ref)) has no rectangular block to reshape into. `_qs_ninternal`
sizes the block by multiplying, so the division here is exact by construction.
"""
@inline function _qs_internal_block(e::AnyQuasiStaticElementCache, Qₑ)
    nqp = getnquadpoints(e.cv)
    return reshape(Qₑ, (length(Qₑ) ÷ nqp, nqp))
end

# A weighted sweep that does not name `:u` must contribute nothing from a displacement-only term.
@inline _qs_state_weight(weights::NamedTuple) = haskey(weights, :u) ? weights.u : false

# --- quadrature point kernels ---------------------------------------------------------------------

"""
    _qs_assemble_qp!(req, ip, cv, qp, P, tangent, dΩ)

Add one quadrature point's `∫ ∇δu ⊡ tangent ⊡ ∇δu dΩ` to the element matrix of `req` and, if `req`
carries a residual, its `∫ ∇δu ⊡ P dΩ`. `ip` is the interpolation `cv` holds.

This is also the function barrier of the element kernels: a material whose routine cannot be inferred
costs one dynamic dispatch per quadrature point here, not one per matrix entry.
"""
function _qs_assemble_qp!(req, ip, cv, qp, P, tangent, dΩ)
    ndofs = getnbasefunctions(cv)
    @inbounds for i = 1:ndofs
        ∇δui = shape_gradient(cv, qp, i)
        _qs_accumulate_residual!(req, i, ∇δui ⊡ P * dΩ)

        ∇δui_tangent = ∇δui ⊡ tangent # Hoisted computation
        for j = 1:ndofs
            ∇δuj = shape_gradient(cv, qp, j)
            req.K[i, j] += (∇δui_tangent ⊡ ∇δuj) * dΩ
        end
    end
end

# The gradient of the shape function `3(I-1) + a` of a vectorized scalar interpolation is `eₐ ⊗ ∇N_I`.
# Contracting by node needs 3 instead of 9 products per matrix entry.
@inline function _nodal_shape_gradient(cv, qp, I)
    e₁∇N = shape_gradient(cv, qp, 3I - 2)
    return Vec{3}((e₁∇N[1, 1], e₁∇N[1, 2], e₁∇N[1, 3]))
end

function _qs_assemble_qp!(
    req,
    ::VectorizedInterpolation{3},
    cv,
    qp,
    P::Tensor{2, 3},
    tangent::Tensor{4, 3},
    dΩ,
)
    nnodes = getnbasefunctions(cv) ÷ 3
    @inbounds for I = 1:nnodes
        ∇Nᵢ = _nodal_shape_gradient(cv, qp, I)
        Pᵢ = P ⋅ ∇Nᵢ
        for a = 1:3
            i = 3(I - 1) + a
            _qs_accumulate_residual!(req, i, Pᵢ[a] * dΩ)

            aᵢ = Tensor{2, 3}(
                (b, l) ->
                    (
                        ∇Nᵢ[1] * tangent[a, 1, b, l] +
                        ∇Nᵢ[2] * tangent[a, 2, b, l] +
                        ∇Nᵢ[3] * tangent[a, 3, b, l]
                    ) * dΩ,
            ) # Hoisted computation
            for J = 1:nnodes
                v              = aᵢ ⋅ _nodal_shape_gradient(cv, qp, J)
                req.K[i, 3J-2] += v[1]
                req.K[i, 3J-1] += v[2]
                req.K[i, 3J]   += v[3]
            end
        end
    end
end

@inline _qs_assemble_qp!(req, cv, qp, P, tangent, dΩ) =
    _qs_assemble_qp!(req, Ferrite.function_interpolation(cv), cv, qp, P, tangent, dΩ)

@inline _qs_accumulate_residual!(req::FerriteOperators.JacobianResidualRequest, i, value) =
    (req.r[i] += value)
@inline _qs_accumulate_residual!(req, i, value) = nothing

# --- rate-free elements --------------------------------------------------------------------------

FerriteOperators.provides_analytic(
    ::Type{<:QuasiStaticElementCache},
    ::FerriteOperators.JacobianKind{:u},
) = true
FerriteOperators.provides_analytic(
    ::Type{<:QuasiStaticElementCache},
    ::FerriteOperators.JacobianResidualKind,
) = true

# TODO how to control dispatch on required input for the material routine?
# TODO finer granularity on the dispatch here. depending on the evolution law of the internal variable this routine looks slightly different.
function FerriteOperators.assemble_cell!(
    req::FerriteOperators.JacobianResidualRequest,
    element_cache::QuasiStaticElementCache,
    args::FerriteOperators.CellArgs,
)
    @unpack constitutive_model, internal_cache, cv, coefficient_cache = element_cache
    dₑ = args.states.u
    time = FerriteOperators.evaluation_time(args.ctx)

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
            args.cell,
            qp,
            time,
        )
        _qs_assemble_qp!(req, cv, qp, P, consistent_tangent(sensitivities), dΩ)
    end
end

function _assemble_quasistatic_jacobian!(req, element_cache::QuasiStaticElementCache, args, w)
    @unpack constitutive_model, internal_cache, cv, coefficient_cache = element_cache
    dₑ = args.states.u
    time = FerriteOperators.evaluation_time(args.ctx)

    @inbounds for qp ∈ QuadratureIterator(cv)
        dΩ = getdetJdV(cv, qp)

        ∇u = function_gradient(cv, qp, dₑ)
        kinematics = DeformationGradient(one(∇u) + ∇u)

        # Compute "tangent only"
        P, sensitivities = material_routine(
            constitutive_model,
            kinematics,
            coefficient_cache,
            internal_cache,
            args.cell,
            qp,
            time,
        )
        _qs_assemble_qp!(req, cv, qp, P, consistent_tangent(sensitivities), w * dΩ)
    end
end

# `true` rather than `1.0`: the unweighted path must not promote the element matrix' eltype.
FerriteOperators.assemble_cell!(
    req::FerriteOperators.JacobianRequest{:u},
    element_cache::QuasiStaticElementCache,
    args::FerriteOperators.CellArgs,
) = _assemble_quasistatic_jacobian!(req, element_cache, args, true)

# A rate-free element is a function of `(u, t)` alone, so only the `:u` weight acts on it.
FerriteOperators.provides_analytic(
    ::Type{<:QuasiStaticElementCache},
    ::FerriteOperators.WeightedJacobianKind,
) = true
FerriteOperators.assemble_cell!(
    req::FerriteOperators.WeightedJacobianRequest,
    element_cache::QuasiStaticElementCache,
    args::FerriteOperators.CellArgs,
) = _assemble_quasistatic_jacobian!(req, element_cache, args, _qs_state_weight(req.weights))

function FerriteOperators.assemble_cell!(
    req::FerriteOperators.ResidualRequest,
    element_cache::QuasiStaticElementCache,
    args::FerriteOperators.CellArgs,
)
    @unpack constitutive_model, internal_cache, cv, coefficient_cache = element_cache
    ndofs = getnbasefunctions(cv)
    dₑ = args.states.u
    time = FerriteOperators.evaluation_time(args.ctx)

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
            args.cell,
            qp,
            time,
        )

        for i = 1:ndofs
            ∇δui = shape_gradient(cv, qp, i)
            req.r[i] += ∇δui ⊡ P * dΩ
        end
    end
end

# --- condensed elements --------------------------------------------------------------------------
#
# The scheme hands the element **two** unrelated time quantities, and the context keeps them apart:
#
#  * `stage_scaling(ctx)` -- the interval the *internal variable* integrates over. `dₜQ = L(F, Q)` is
#    first order no matter what the global scheme does with `u`, so its local problem is unchanged by
#    the choice of global scheme.
#  * the `v` slot -- how the deformation rate is formed. `AffineRate` reconstructs it from the
#    unknown displacement at gather time, so the element never learns the scheme's coefficients.
#
# Collapsing the two is what makes a rate-coupled material silently wrong under any scheme but
# backward Euler, where they happen to be reciprocals.

# Forming the rate is the scheme's contribution, and this is the only place in the element layer that
# reads the reconstruction. Dispatching on the cache means the ODE cache does not pay for the second
# gradient evaluation: its local problem `dₜQ = L(F, Q)` cannot read a rate.
@inline function compute_kinematic_quantities(
    e::QuasiStaticCondensedODEElementCache,
    qp,
    dₑ,
    states,
)
    ∇u = function_gradient(e.cv, qp, dₑ)
    return DeformationGradient(one(∇u) + ∇u)
end

@inline function compute_kinematic_quantities(
    e::QuasiStaticCondensedDAEElementCache,
    qp,
    dₑ,
    states,
)
    ∇u = function_gradient(e.cv, qp, dₑ)
    # The `v` slot already carries `∂v∂u (u - uᵥ)`, and `function_gradient` is linear, so its gradient
    # is the deformation rate the material asks for.
    ∇v = function_gradient(e.cv, qp, states.v)
    return DeformationGradientWithRate(one(∇u) + ∇u, ∇v)
end

# How the tangent weighs the two kinematic sensitivities. A reconstructed slot is frozen under
# `JacobianKind{:u}`, so the rate weight is zero there and is the scheme's chain-rule scalar in a
# weighted sweep.
@inline _qs_linearization(::QuasiStaticCondensedODEElementCache, wu, wv) =
    KinematicLinearization(wu)
@inline _qs_linearization(::QuasiStaticCondensedDAEElementCache, wu, wv) =
    KinematicLinearization(wu, wv)

"""
    _qs_rate_weight(weights)

The scheme's `∂v/∂u`, read out of a weighted request's chain-rule scalars.

A weighted sweep over a rate-coupled material must name the `v` slot: without it the local solve's
rate sensitivity would be dropped silently, leaving a descent direction rather than a Newton one.
"""
@inline function _qs_rate_weight(weights::NamedTuple)
    haskey(weights, :v) || throw(
        ArgumentError(
            "A weighted Jacobian over a rate-coupled material must weigh the `v` slot; got slots " *
            "$(keys(weights)). The rate sensitivity ∂P/∂Ḟ enters through it and through nothing else.",
        ),
    )
    return weights.v
end

"""
    _qs_correctors(element_cache, args)

The cell's per-quadrature-point tangent corrections, as
[`FerriteOperators.condense_cell!`](@ref) stored them.

Reading a cell no condensation phase has visited throws and names it: the corrections are the local
solve's, and a `Consistent` tangent cannot be formed without them.
"""
@inline _qs_correctors(e::AnyQuasiStaticCondensedElementCache, args) =
    FerriteOperators.item_state(e.correctors, cellid(args.cell))

# The stored corrections are one `SVector` per cell, so the buffer that fills it is sized off the
# store's own element type rather than off a runtime quadrature point count.
@inline _qs_corrector_buffer(::FerriteOperators.ItemStates{SVector{NQP, Corr}}) where {NQP, Corr} =
    MVector{NQP, Corr}(undef)

@inline _qs_store_correctors!(store::FerriteOperators.ItemStates{S}, id, buffer) where {S} =
    FerriteOperators.set_item_state!(store, id, S(buffer))

FerriteOperators.invalidate_correctors!(e::AnyQuasiStaticCondensedElementCache) =
    (FerriteOperators.invalidate_item_states!(e.correctors); nothing)

FerriteOperators.provides_analytic(
    ::Type{<:QuasiStaticCondensedODEElementCache},
    ::FerriteOperators.JacobianKind{:u},
) = true
FerriteOperators.provides_analytic(
    ::Type{<:QuasiStaticCondensedODEElementCache},
    ::FerriteOperators.JacobianResidualKind,
) = true
FerriteOperators.provides_analytic(
    ::Type{<:QuasiStaticCondensedDAEElementCache},
    ::FerriteOperators.JacobianKind{:u},
) = true
FerriteOperators.provides_analytic(
    ::Type{<:QuasiStaticCondensedDAEElementCache},
    ::FerriteOperators.JacobianResidualKind,
) = true
FerriteOperators.provides_analytic(
    ::Type{<:QuasiStaticCondensedDAEElementCache},
    ::FerriteOperators.WeightedJacobianKind,
) = true

"""
    condense_cell!(cache, args, weights)

Solve every quadrature point's local problem, write the trial internal state into `args.states.q`,
store the tangent correction each solve produced, and report what the solves did
([`cell_condensation_report`](@ref)).

This is the only hook that solves; the assembly kernels are pure evaluations at the state it wrote,
correcting their tangent with what it stored.

`weights` are not read: the local problem takes `F` and `Ḟ` as independent inputs, so its two
corrections stay separated and the scheme's chain-rule scalars weigh them where every other
sensitivity is weighed, in [`consistent_tangent`](@ref).
"""
function FerriteOperators.condense_cell!(
    element_cache::QuasiStaticCondensedElementCache,
    args::FerriteOperators.CellArgs,
    weights::NamedTuple,
)
    @unpack constitutive_model, internal_cache, cv, coefficient_cache = element_cache
    dₑ = args.states.u
    Qₑ = _qs_internal_block(element_cache, args.states.q)
    Qₑprev = _qs_internal_block(element_cache, args.states.qprev)
    t = FerriteOperators.evaluation_time(args.ctx)
    Δt = FerriteOperators.stage_scaling(args.ctx)
    correctors = _qs_corrector_buffer(element_cache.correctors)

    @inbounds for qp ∈ QuadratureIterator(cv)
        kinematics = compute_kinematic_quantities(element_cache, qp, dₑ, args.states)
        correctors[qp.i] = condense_material_routine(
            constitutive_model,
            kinematics,
            coefficient_cache,
            internal_cache,
            args.cell,
            qp,
            t,
            @view(Qₑ[:, qp.i]),
            @view(Qₑprev[:, qp.i]),
            Δt,
        )
    end
    _qs_store_correctors!(element_cache.correctors, cellid(args.cell), correctors)
    # Read after the loop, so the slots folded here are the ones these solves just wrote.
    return cell_condensation_report(
        internal_cache.local_solver_cache,
        cellid(args.cell),
        getnquadpoints(cv),
    )
end

"""
    condense_cell!(cache, args, ::Nothing)

The residual-only election: solve every quadrature point's local problem and write the trial internal
state, forming no tangent correction.

The cell's corrector slot is left invalid — `condense_internal!` dropped it before the sweep — so a
`Consistent` tangent at this state names the cell and refuses rather than combining the corrections of
whatever trial point stored them last. Condensing again with weights makes it assemblable.
"""
function FerriteOperators.condense_cell!(
    element_cache::QuasiStaticCondensedElementCache,
    args::FerriteOperators.CellArgs,
    ::Nothing,
)
    @unpack constitutive_model, internal_cache, cv, coefficient_cache = element_cache
    dₑ = args.states.u
    Qₑ = _qs_internal_block(element_cache, args.states.q)
    Qₑprev = _qs_internal_block(element_cache, args.states.qprev)
    t = FerriteOperators.evaluation_time(args.ctx)
    Δt = FerriteOperators.stage_scaling(args.ctx)

    @inbounds for qp ∈ QuadratureIterator(cv)
        kinematics = compute_kinematic_quantities(element_cache, qp, dₑ, args.states)
        condense_material_state!(
            constitutive_model,
            kinematics,
            coefficient_cache,
            internal_cache,
            args.cell,
            qp,
            t,
            @view(Qₑ[:, qp.i]),
            @view(Qₑprev[:, qp.i]),
            Δt,
        )
    end
    return cell_condensation_report(
        internal_cache.local_solver_cache,
        cellid(args.cell),
        getnquadpoints(cv),
    )
end

function _assemble_condensed_cell!(
    req,
    element_cache::QuasiStaticCondensedElementCache,
    args::FerriteOperators.CellArgs,
    linearization,
)
    @unpack constitutive_model, internal_cache, cv, coefficient_cache = element_cache
    dₑ = args.states.u
    Qₑ = _qs_internal_block(element_cache, args.states.q)
    correctors = _qs_correctors(element_cache, args)
    t = FerriteOperators.evaluation_time(args.ctx)

    @inbounds for qp ∈ QuadratureIterator(cv)
        dΩ = getdetJdV(cv, qp)

        kinematics = compute_kinematic_quantities(element_cache, qp, dₑ, args.states)

        P, sensitivities = frozen_material_routine(
            constitutive_model,
            kinematics,
            coefficient_cache,
            internal_cache,
            args.cell,
            qp,
            t,
            @view(Qₑ[:, qp.i]),
        )
        tangent = consistent_tangent(sensitivities + correctors[qp.i], linearization)
        _qs_assemble_qp!(req, cv, qp, P, tangent, dΩ)
    end
end

FerriteOperators.assemble_cell!(
    req::FerriteOperators.JacobianResidualRequest,
    element_cache::QuasiStaticCondensedElementCache,
    args::FerriteOperators.CellArgs,
) = _assemble_condensed_cell!(
    req,
    element_cache,
    args,
    _qs_linearization(element_cache, true, false),
)

FerriteOperators.assemble_cell!(
    req::FerriteOperators.JacobianRequest{:u},
    element_cache::QuasiStaticCondensedElementCache,
    args::FerriteOperators.CellArgs,
) = _assemble_condensed_cell!(
    req,
    element_cache,
    args,
    _qs_linearization(element_cache, true, false),
)

FerriteOperators.provides_analytic(
    ::Type{<:QuasiStaticCondensedODEElementCache},
    ::FerriteOperators.WeightedJacobianKind,
) = true
FerriteOperators.assemble_cell!(
    req::FerriteOperators.WeightedJacobianRequest,
    element_cache::QuasiStaticCondensedODEElementCache,
    args::FerriteOperators.CellArgs,
) = _assemble_condensed_cell!(
    req,
    element_cache,
    args,
    _qs_linearization(element_cache, _qs_state_weight(req.weights), false),
)

FerriteOperators.assemble_cell!(
    req::FerriteOperators.WeightedJacobianRequest,
    element_cache::QuasiStaticCondensedDAEElementCache,
    args::FerriteOperators.CellArgs,
) = _assemble_condensed_cell!(
    req,
    element_cache,
    args,
    _qs_linearization(element_cache, _qs_state_weight(req.weights), _qs_rate_weight(req.weights)),
)

function FerriteOperators.assemble_cell!(
    req::FerriteOperators.ResidualRequest,
    element_cache::QuasiStaticCondensedElementCache,
    args::FerriteOperators.CellArgs,
)
    @unpack constitutive_model, internal_cache, cv, coefficient_cache = element_cache
    ndofs = getnbasefunctions(cv)
    dₑ = args.states.u
    Qₑ = _qs_internal_block(element_cache, args.states.q)
    t = FerriteOperators.evaluation_time(args.ctx)

    @inbounds for qp ∈ QuadratureIterator(cv)
        dΩ = getdetJdV(cv, qp)

        kinematics = compute_kinematic_quantities(element_cache, qp, dₑ, args.states)

        # Stress only, at the state the condensation phase wrote
        P = frozen_reduced_material_routine(
            constitutive_model,
            kinematics,
            coefficient_cache,
            internal_cache,
            args.cell,
            qp,
            t,
            @view(Qₑ[:, qp.i]),
        )

        for i = 1:ndofs
            ∇δui = shape_gradient(cv, qp, i)
            req.r[i] += ∇δui ⊡ P * dΩ
        end
    end
end

# --- steady state condensed elements --------------------------------------------------------------
#
# Same three phases as above -- condense, then evaluate at the state it wrote, correcting the tangent
# -- with the time discretization removed: the local problem is `0 = L(F, Q)`, so it takes neither
# `Δt` nor `qprev` and the material entry points lose those two arguments.

FerriteOperators.provides_analytic(
    ::Type{<:QuasiStaticCondensedSteadyElementCache},
    ::FerriteOperators.JacobianKind{:u},
) = true
FerriteOperators.provides_analytic(
    ::Type{<:QuasiStaticCondensedSteadyElementCache},
    ::FerriteOperators.JacobianResidualKind,
) = true
FerriteOperators.provides_analytic(
    ::Type{<:QuasiStaticCondensedSteadyElementCache},
    ::FerriteOperators.WeightedJacobianKind,
) = true

@inline compute_kinematic_quantities(e::QuasiStaticCondensedSteadyElementCache, qp, dₑ, states) =
    (∇u = function_gradient(e.cv, qp, dₑ); DeformationGradient(one(∇u) + ∇u))

@inline _qs_linearization(::QuasiStaticCondensedSteadyElementCache, wu, wv) =
    KinematicLinearization(wu)

function FerriteOperators.condense_cell!(
    element_cache::QuasiStaticCondensedSteadyElementCache,
    args::FerriteOperators.CellArgs,
    weights::NamedTuple,
)
    @unpack constitutive_model, internal_cache, cv, coefficient_cache = element_cache
    dₑ = args.states.u
    Qₑ = _qs_internal_block(element_cache, args.states.q)
    t = FerriteOperators.evaluation_time(args.ctx)
    correctors = _qs_corrector_buffer(element_cache.correctors)

    @inbounds for qp ∈ QuadratureIterator(cv)
        kinematics = compute_kinematic_quantities(element_cache, qp, dₑ, args.states)
        correctors[qp.i] = condense_material_routine(
            constitutive_model,
            kinematics,
            coefficient_cache,
            internal_cache,
            args.cell,
            qp,
            t,
            @view(Qₑ[:, qp.i]),
        )
    end
    _qs_store_correctors!(element_cache.correctors, cellid(args.cell), correctors)
    return cell_condensation_report(
        internal_cache.local_solver_cache,
        cellid(args.cell),
        getnquadpoints(cv),
    )
end

function FerriteOperators.condense_cell!(
    element_cache::QuasiStaticCondensedSteadyElementCache,
    args::FerriteOperators.CellArgs,
    ::Nothing,
)
    @unpack constitutive_model, internal_cache, cv, coefficient_cache = element_cache
    dₑ = args.states.u
    Qₑ = _qs_internal_block(element_cache, args.states.q)
    t = FerriteOperators.evaluation_time(args.ctx)

    @inbounds for qp ∈ QuadratureIterator(cv)
        kinematics = compute_kinematic_quantities(element_cache, qp, dₑ, args.states)
        condense_material_state!(
            constitutive_model,
            kinematics,
            coefficient_cache,
            internal_cache,
            args.cell,
            qp,
            t,
            @view(Qₑ[:, qp.i]),
        )
    end
    return cell_condensation_report(
        internal_cache.local_solver_cache,
        cellid(args.cell),
        getnquadpoints(cv),
    )
end

function _assemble_steady_condensed_cell!(
    req,
    element_cache::QuasiStaticCondensedSteadyElementCache,
    args::FerriteOperators.CellArgs,
    linearization,
)
    @unpack constitutive_model, internal_cache, cv, coefficient_cache = element_cache
    dₑ = args.states.u
    Qₑ = _qs_internal_block(element_cache, args.states.q)
    correctors = _qs_correctors(element_cache, args)
    t = FerriteOperators.evaluation_time(args.ctx)

    @inbounds for qp ∈ QuadratureIterator(cv)
        dΩ = getdetJdV(cv, qp)

        kinematics = compute_kinematic_quantities(element_cache, qp, dₑ, args.states)

        P, sensitivities = frozen_material_routine(
            constitutive_model,
            kinematics,
            coefficient_cache,
            internal_cache,
            args.cell,
            qp,
            t,
            @view(Qₑ[:, qp.i]),
        )
        tangent = consistent_tangent(sensitivities + correctors[qp.i], linearization)
        _qs_assemble_qp!(req, cv, qp, P, tangent, dΩ)
    end
end

FerriteOperators.assemble_cell!(
    req::FerriteOperators.JacobianResidualRequest,
    element_cache::QuasiStaticCondensedSteadyElementCache,
    args::FerriteOperators.CellArgs,
) = _assemble_steady_condensed_cell!(req, element_cache, args, KinematicLinearization(true))

FerriteOperators.assemble_cell!(
    req::FerriteOperators.JacobianRequest{:u},
    element_cache::QuasiStaticCondensedSteadyElementCache,
    args::FerriteOperators.CellArgs,
) = _assemble_steady_condensed_cell!(req, element_cache, args, KinematicLinearization(true))

FerriteOperators.assemble_cell!(
    req::FerriteOperators.WeightedJacobianRequest,
    element_cache::QuasiStaticCondensedSteadyElementCache,
    args::FerriteOperators.CellArgs,
) = _assemble_steady_condensed_cell!(
    req,
    element_cache,
    args,
    KinematicLinearization(_qs_state_weight(req.weights)),
)

function FerriteOperators.assemble_cell!(
    req::FerriteOperators.ResidualRequest,
    element_cache::QuasiStaticCondensedSteadyElementCache,
    args::FerriteOperators.CellArgs,
)
    @unpack constitutive_model, internal_cache, cv, coefficient_cache = element_cache
    ndofs = getnbasefunctions(cv)
    dₑ = args.states.u
    Qₑ = _qs_internal_block(element_cache, args.states.q)
    t = FerriteOperators.evaluation_time(args.ctx)

    @inbounds for qp ∈ QuadratureIterator(cv)
        dΩ = getdetJdV(cv, qp)

        kinematics = compute_kinematic_quantities(element_cache, qp, dₑ, args.states)

        P = frozen_reduced_material_routine(
            constitutive_model,
            kinematics,
            coefficient_cache,
            internal_cache,
            args.cell,
            qp,
            t,
            @view(Qₑ[:, qp.i]),
        )

        for i = 1:ndofs
            ∇δui = shape_gradient(cv, qp, i)
            req.r[i] += ∇δui ⊡ P * dΩ
        end
    end
end

# ------------------------------------------------------------------------------------------------

"""
    quasistatic_element_cache_type(evolution::InternalVariableEvolution)

Which quasi-static element cache a material needs, decided by the evolution law of its internal
variable — not by the time integrator. No internal variable means nothing to condense; `0 = L(F, Q)`
condenses without a timestep; `dₜQ = L(F, Q)` is a mass matrix ODE; `dₜQ = L(F, dₜF, Q)` is a DAE.

The question is asked of the model via [`internal_variable_evolution`](@ref) rather than of the state
cache: the `Empty…CondensationMaterialStateCache` types record only that a model needs no extra
scratch space, which is a different question and gives the wrong answer for an unwrapped rate
dependent model.
"""
quasistatic_element_cache_type(::NoEvolution) = QuasiStaticElementCache
quasistatic_element_cache_type(::SteadyStateEvolution) = QuasiStaticCondensedSteadyElementCache
quasistatic_element_cache_type(::FirstOrderEvolution) = QuasiStaticCondensedODEElementCache
quasistatic_element_cache_type(::RateCoupledEvolution) = QuasiStaticCondensedDAEElementCache

"""
    setup_condensation_correctors(evolution, material_model, qr, sdh)

The corrector store the condensed element caches carry, spliced into their constructor -- empty for
[`NoEvolution`](@ref), which has no local solve to correct with.

One `SVector` of per-quadrature-point corrections per cell, in the currency the element's kinematics
fix: a rate coupled local problem contributes to both sensitivities, a first order one only to
`∂P/∂F`. Sized over the whole grid, since a cellid is what keys it.
"""
setup_condensation_correctors(
    ::NoEvolution,
    material_model,
    qr::QuadratureRule,
    sdh::SubDofHandler,
) = ()

setup_condensation_correctors(
    evolution::Union{SteadyStateEvolution, FirstOrderEvolution, RateCoupledEvolution},
    material_model,
    qr::QuadratureRule,
    sdh::SubDofHandler,
) = (_condensation_corrector_store(evolution, material_model, qr, sdh),)

function _condensation_corrector_store(evolution, material_model, qr, sdh)
    grid       = Ferrite.get_grid(sdh.dh)
    correction = local_correction_type(material_model, Val(getspatialdim(grid)))
    corrector  = _corrector_currency(evolution, correction)
    return FerriteOperators.ItemStates{SVector{getnquadpoints(qr), corrector}}(getncells(grid))
end

_corrector_currency(::SteadyStateEvolution, ::Type{C}) where {C} = KinematicSensitivities{C}
_corrector_currency(::FirstOrderEvolution, ::Type{C}) where {C} = KinematicSensitivities{C}
_corrector_currency(::RateCoupledEvolution, ::Type{C}) where {C} =
    KinematicSensitivitiesWithRate{C, C}

function setup_quasistatic_element_cache(
    material_model::AbstractMaterialModel,
    qr::QuadratureRule,
    sdh::SubDofHandler,
    cv::CellValues,
)
    internal_cache = setup_internal_cache(material_model, qr, sdh)
    evolution      = internal_variable_evolution(material_model)
    return quasistatic_element_cache_type(evolution)(
        material_model,
        setup_coefficient_cache(material_model, qr, sdh),
        internal_cache,
        cv,
        setup_condensation_correctors(evolution, material_model, qr, sdh)...,
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
# Reconstruct the *same* concrete cache type, so the problem class survives the move to a device. The
# corrector store is spliced exactly as at setup, and shares rather than copies: cells are disjoint
# between workers, so a condensation's writes are visible to whichever worker sweeps them.
function duplicate_for_device(device, cache::AnyQuasiStaticElementCache)
    return (typeof(cache).name.wrapper)(
        duplicate_for_device(device, cache.constitutive_model),
        duplicate_for_device(device, cache.coefficient_cache),
        duplicate_for_device(device, cache.internal_cache),
        duplicate_for_device(device, cache.cv),
        map(store -> duplicate_for_device(device, store), _qs_corrector_stores(cache))...,
    )
end

_qs_corrector_stores(cache::QuasiStaticElementCache) = ()
_qs_corrector_stores(cache::AnyQuasiStaticCondensedElementCache) = (cache.correctors,)
