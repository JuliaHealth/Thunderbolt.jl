# TODO (FILE) I think we should change the design here. Instea of dispatching on Ψ we should make the material callable or equip it with a function.

abstract type AbstractMaterialModel end

function default_initial_state!(uq, ::AbstractMaterialModel)
    error("Initial condition setup not implemented yet.")
end

function material_routine(
        material_model::AbstractMaterialModel,
        F::Tensor{2},
        coefficient_cache,
        ::EmptyInternalCache,
        geometry_cache::Ferrite.CellCache,
        qp::QuadraturePoint,
        time
)
    coefficients = evaluate_coefficient(coefficient_cache, geometry_cache, qp, time)
    return stress_and_tangent(material_model, F, coefficients, EmptyInternalModel())
end

function material_routine(
        material_model::AbstractMaterialModel,
        F::Tensor{2},
        coefficient_cache,
        state_cache::TrivialCondensationMaterialStateCache,
        geometry_cache::Ferrite.CellCache,
        qp::QuadraturePoint,
        time
)
    coefficients = evaluate_coefficient(coefficient_cache, geometry_cache, qp, time)
    Q = state(state_cache, geometry_cache, qp, time)
    return stress_and_tangent(material_model, F, coefficients, Q)
end

function material_routine(
        material_model::AbstractMaterialModel,
        F::Tensor{2},
        coefficient_cache,
        state_cache::RateIndependentCondensationMaterialStateCache,
        geometry_cache::Ferrite.CellCache,
        qp::QuadraturePoint,
        time
)
    coefficients = evaluate_coefficient(coefficient_cache, geometry_cache, qp, time)
    Q, ∂P∂QdQdF = solve_local_constraint(
        F, coefficients, material_model, state_cache, geometry_cache, qp, time)
    P, ∂P∂F = stress_and_tangent(material_model, F, coefficients, Q)
    return P, ∂P∂F + ∂P∂QdQdF
end

function reduced_material_routine(
        material_model::AbstractMaterialModel,
        F::Tensor{2},
        coefficient_cache,
        ::EmptyInternalCache,
        geometry_cache::Ferrite.CellCache,
        qp::QuadraturePoint,
        time
)
    coefficients = evaluate_coefficient(coefficient_cache, geometry_cache, qp, time)
    return stress_function(material_model, F, coefficients, EmptyInternalModel())
end

function reduced_material_routine(
        material_model::AbstractMaterialModel,
        F::Tensor{2},
        coefficient_cache,
        state_cache::TrivialCondensationMaterialStateCache,
        geometry_cache::Ferrite.CellCache,
        qp::QuadraturePoint,
        time
)
    coefficients = evaluate_coefficient(coefficient_cache, geometry_cache, qp, time)
    Q = state(state_cache, geometry_cache, qp, time)
    return stress_function(material_model, F, coefficients, Q)
end

function reduced_material_routine(
        material_model::AbstractMaterialModel,
        F::Tensor{2},
        coefficient_cache,
        state_cache::RateIndependentCondensationMaterialStateCache,
        geometry_cache::Ferrite.CellCache,
        qp::QuadraturePoint,
        time
)
    coefficients = evaluate_coefficient(coefficient_cache, geometry_cache, qp, time)
    Q            = solve_local_constraint_state_only(F, coefficients, material_model, state_cache, geometry_cache, qp, time)
    P            = stress_function(material_model, F, coefficients, Q)
    return P, ∂P∂F + ∂P∂QdQdF
end

@doc raw"""
    PrestressedMechanicalModel(inner_model, prestress_field)

Models the stress formulated in the 1st Piola-Kirchhoff stress tensor based on a multiplicative split
of the deformation gradient $$F = F_{\textrm{e}} F_{0}$$ where we compute $$P(F_{\textrm{e}}) = P(F F^{-1}_{0})$$.

Please note that it is assumed that $$F^{-1}_{0}$$ is the quantity computed by `prestress_field`.
"""
struct PrestressedMechanicalModel{MM, FF} <: AbstractMaterialModel
    inner_model::MM
    prestress_field::FF
end

struct PrestressedMechanicalModelCoefficientCache{T1, T2}
    inner_cache::T1
    prestress_cache::T2
end

function default_initial_state!(uq, model::PrestressedMechanicalModel)
    default_initial_state!(uq, model.inner_model)
end

function setup_coefficient_cache(
        m::PrestressedMechanicalModel,
        qr::QuadratureRule,
        sdh::SubDofHandler
)
    return PrestressedMechanicalModelCoefficientCache(
        setup_coefficient_cache(m.inner_model, qr, sdh),
        setup_coefficient_cache(m.prestress_field, qr, sdh)
    )
end
function duplicate_for_device(device, cache::PrestressedMechanicalModelCoefficientCache)
    return PrestressedMechanicalModelCoefficientCache(
        duplicate_for_device(device, cache.inner_cache),
        duplicate_for_device(device, cache.prestress_cache)
    )
end

function material_routine(
        material_model::PrestressedMechanicalModel,
        F::Tensor{2},
        coefficient_cache,
        state_cache::EmptyInternalCache,
        geometry_cache::Ferrite.CellCache,
        qp::QuadraturePoint,
        time
)
    prestressed_material_routine(
        material_model,
        F,
        coefficient_cache,
        state_cache,
        geometry_cache,
        qp,
        time
    )
end
function material_routine(
        material_model::PrestressedMechanicalModel,
        F::Tensor{2},
        coefficient_cache,
        state_cache::TrivialCondensationMaterialStateCache,
        geometry_cache::Ferrite.CellCache,
        qp::QuadraturePoint,
        time
)
    prestressed_material_routine(
        material_model,
        F,
        coefficient_cache,
        state_cache,
        geometry_cache,
        qp,
        time
    )
end
function material_routine(
        material_model::PrestressedMechanicalModel,
        F::Tensor{2},
        coefficient_cache,
        state_cache::RateIndependentCondensationMaterialStateCache,
        geometry_cache::Ferrite.CellCache,
        qp::QuadraturePoint,
        time
)
    prestressed_material_routine(
        material_model,
        F,
        coefficient_cache,
        state_cache,
        geometry_cache,
        qp,
        time
    )
end
function prestressed_material_routine(
        material_model::PrestressedMechanicalModel,
        F::Tensor{2},
        coefficient_cache,
        state_cache,
        geometry_cache::Ferrite.CellCache,
        qp::QuadraturePoint,
        time
)
    F₀inv = evaluate_coefficient(coefficient_cache.prestress_cache, geometry_cache, qp, time)
    Fᵉ = F ⋅ F₀inv
    ∂Ψᵉ∂Fᵉ, ∂²Ψᵉ∂Fᵉ² = material_routine(
        material_model.inner_model,
        Fᵉ,
        coefficient_cache.inner_cache,
        state_cache,
        geometry_cache,
        qp,
        time
    )
    Pᵉ = ∂Ψᵉ∂Fᵉ # Elastic PK1
    P = Pᵉ ⋅ transpose(F₀inv) # Obtained by Coleman-Noll procedure
    Aᵉ = ∂²Ψᵉ∂Fᵉ² # Elastic mixed modulus
    # TODO condense these steps into a single operation "A_imkn F_jm F_ln"
    # Pull elastic modulus from intermediate to reference configuration
    ∂Pᵉ∂F = Aᵉ ⋅ transpose(F₀inv)
    ∂P∂F = dot_2_1t(∂Pᵉ∂F, F₀inv)
    return P, ∂P∂F
end

function reduced_material_routine(
        material_model::PrestressedMechanicalModel,
        F::Tensor{2},
        coefficient_cache,
        state_cache::EmptyInternalCache,
        geometry_cache::Ferrite.CellCache,
        qp::QuadraturePoint,
        time
)
    reduced_prestressed_material_routine(
        material_model,
        F,
        coefficient_cache,
        state_cache,
        geometry_cache,
        qp,
        time
    )
end
function reduced_material_routine(
        material_model::PrestressedMechanicalModel,
        F::Tensor{2},
        coefficient_cache,
        state_cache::TrivialCondensationMaterialStateCache,
        geometry_cache::Ferrite.CellCache,
        qp::QuadraturePoint,
        time
)
    reduced_prestressed_material_routine(
        material_model,
        F,
        coefficient_cache,
        state_cache,
        geometry_cache,
        qp,
        time
    )
end
function reduced_material_routine(
        material_model::PrestressedMechanicalModel,
        F::Tensor{2},
        coefficient_cache,
        state_cache::RateIndependentCondensationMaterialStateCache,
        geometry_cache::Ferrite.CellCache,
        qp::QuadraturePoint,
        time
)
    reduced_prestressed_material_routine(
        material_model,
        F,
        coefficient_cache,
        state_cache,
        geometry_cache,
        qp,
        time
    )
end
function reduced_prestressed_material_routine(
        material_model::PrestressedMechanicalModel,
        F::Tensor{2},
        coefficient_cache,
        state_cache,
        geometry_cache::Ferrite.CellCache,
        qp::QuadraturePoint,
        time
)
    F₀inv = evaluate_coefficient(coefficient_cache.prestress_cache, geometry_cache, qp, time)
    Fᵉ = F ⋅ F₀inv
    ∂Ψᵉ∂Fᵉ = reduced_material_routine(
        material_model.inner_model,
        Fᵉ,
        coefficient_cache.inner_cache,
        state_cache,
        geometry_cache,
        qp,
        time
    )
    Pᵉ = ∂Ψᵉ∂Fᵉ # Elastic PK1
    P = Pᵉ ⋅ transpose(F₀inv) # Obtained by Coleman-Noll procedure
    return P
end
function setup_internal_cache(
        material_model::PrestressedMechanicalModel,
        qr::QuadratureRule,
        sdh::SubDofHandler
)
    setup_internal_cache(material_model.inner_model, qr, sdh)
end

@doc raw"""
    PK1Model(material, coefficient_field)
    PK1Model(material, internal_model, coefficient_field)

Models the stress formulated in the 1st Piola-Kirchhoff stress tensor. If the material is energy-based,
then the term is formulated as follows:
$$\int_{\Omega_0} P(u,s) \cdot \delta F dV = \int_{\Omega_0} \partial_{F} \psi(u,s) \cdot \delta \nabla u $$
"""
struct PK1Model{PMat, IMod, CFType} <: AbstractMaterialModel
    material::PMat
    internal_model::IMod
    coefficient_field::CFType
end

PK1Model(material, coefficient_field) = PK1Model(material, EmptyInternalModel(), coefficient_field)

function setup_coefficient_cache(m::PK1Model, qr::QuadratureRule, sdh::SubDofHandler)
    return setup_coefficient_cache(m.coefficient_field, qr, sdh)
end

default_initial_state!(uq, model::PK1Model) = default_initial_state!(uq, model.internal_model)

function setup_internal_cache(material_model::PK1Model, qr::QuadratureRule, sdh::SubDofHandler)
    setup_internal_cache(material_model.internal_model, qr, sdh)
end

function stress_function(model::PK1Model, F::Tensor{2}, coefficients, ::EmptyInternalModel)
    ∂Ψ∂F = Tensors.gradient(F_ad -> Ψ(F_ad, coefficients, model.material), F)

    return ∂Ψ∂F
end

function stress_and_tangent(model::PK1Model, F::Tensor{2}, coefficients, ::EmptyInternalModel)
    ∂²Ψ∂F², ∂Ψ∂F = Tensors.hessian(F_ad -> Ψ(F_ad, coefficients, model.material), F, :all)

    return ∂Ψ∂F, ∂²Ψ∂F²
end

@doc raw"""
    GeneralizedHillModel(passive_spring_model, active_spring_model, active_deformation_gradient_model,contraction_model, microstructure_model)

The generalized Hill framework as proposed by [GokMenKuh:2014:ghm](@citet).

In this framework the model is formulated as an energy minimization problem with the following additively split energy:

$W(\mathbf{F}, \mathbf{F}^{\rm{a}}) = W_{\rm{passive}}(\mathbf{F}) + W_{\rm{active}}(\mathbf{F}\mathbf{F}^{-\rm{a}})$

Where $W_{\rm{passive}}$ is the passive material response and $W_{\rm{active}}$ the active response
respectvely.
"""
struct GeneralizedHillModel{PMat, AMat, ADGMod, CMod, MS} <: AbstractMaterialModel
    passive_spring::PMat
    active_spring::AMat
    active_deformation_gradient_model::ADGMod
    contraction_model::CMod
    microstructure_model::MS
end

function setup_coefficient_cache(m::GeneralizedHillModel, qr::QuadratureRule, sdh::SubDofHandler)
    return setup_coefficient_cache(m.microstructure_model, qr, sdh)
end

function stress_function(model::GeneralizedHillModel, F::Tensor{2}, coefficients, state)
    # TODO what is a good abstraction here?
    Fᵃ = compute_Fᵃ(
        state,
        coefficients,
        model.contraction_model,
        model.active_deformation_gradient_model
    )

    ∂Ψ∂F = Tensors.gradient(
        F_ad -> Ψ(F_ad, coefficients, model.passive_spring) +
                Ψ(F_ad, Fᵃ, coefficients, model.active_spring),
        F
    )

    return ∂Ψ∂F
end

function stress_and_tangent(model::GeneralizedHillModel, F::Tensor{2}, coefficients, state)
    # TODO what is a good abstraction here?
    Fᵃ = compute_Fᵃ(
        state,
        coefficients,
        model.contraction_model,
        model.active_deformation_gradient_model
    )

    ∂²Ψ∂F², ∂Ψ∂F = Tensors.hessian(
        F_ad -> Ψ(F_ad, coefficients, model.passive_spring) +
                Ψ(F_ad, Fᵃ, coefficients, model.active_spring),
        F,
        :all
    )

    return ∂Ψ∂F, ∂²Ψ∂F²
end


@doc raw"""
    ExtendedHillModel(passive_spring_model, active_spring_model, active_deformation_gradient_model,contraction_model, microstructure_model)

The extended (generalized) Hill model as proposed by [OgiBalPer:2023:aeg](@citet). The original formulation dates back to [StaKlaHol:2008:smc](@citet) for smooth muscle tissues.

In this framework the model is formulated as an energy minimization problem with the following additively split energy:

$W(\mathbf{F}, \mathbf{F}^{\rm{a}}) = W_{\rm{passive}}(\mathbf{F}) + \mathcal{N}(\bm{\alpha})W_{\rm{active}}(\mathbf{F}\mathbf{F}^{-\rm{a}})$

Where $W_{\rm{passive}}$ is the passive material response and $W_{\rm{active}}$ the active response
respectvely. $\mathcal{N}$ is the amount of formed crossbridges. We refer to the original paper [OgiBalPer:2023:aeg](@cite) for more details.
"""
struct ExtendedHillModel{PMat, AMat, ADGMod, CMod, MS} <: AbstractMaterialModel
    passive_spring::PMat
    active_spring::AMat
    active_deformation_gradient_model::ADGMod
    contraction_model::CMod
    microstructure_model::MS
end

function setup_coefficient_cache(m::ExtendedHillModel, qr::QuadratureRule, sdh::SubDofHandler)
    return setup_coefficient_cache(m.microstructure_model, qr, sdh)
end

function stress_function(model::ExtendedHillModel, F::Tensor{2}, coefficients, cell_state)
    # TODO what is a good abstraction here?
    Fᵃ = compute_Fᵃ(
        cell_state,
        coefficients,
        model.contraction_model,
        model.active_deformation_gradient_model
    )
    N = 𝓝(cell_state, F, coefficients, model.contraction_model)

    ∂Ψ∂F = Tensors.gradient(
        F_ad -> Ψ(F_ad, coefficients, model.passive_spring) +
                N * Ψ(F_ad, Fᵃ, coefficients, model.active_spring),
        F
    )

    return ∂Ψ∂F
end

function stress_and_tangent(model::ExtendedHillModel, F::Tensor{2}, coefficients, cell_state)
    # TODO what is a good abstraction here?
    Fᵃ = compute_Fᵃ(
        cell_state,
        coefficients,
        model.contraction_model,
        model.active_deformation_gradient_model
    )
    N = 𝓝(cell_state, F, coefficients, model.contraction_model)

    ∂²Ψ∂F², ∂Ψ∂F = Tensors.hessian(
        F_ad -> Ψ(F_ad, coefficients, model.passive_spring) +
                N * Ψ(F_ad, Fᵃ, coefficients, model.active_spring),
        F,
        :all
    )

    return ∂Ψ∂F, ∂²Ψ∂F²
end


@doc raw"""
    ActiveStressModel(material_model, active_stress_model, contraction_model, microstructure_model)

The active stress model as originally proposed by [GucWalMcC:1993:mac](@citet).

In this framework the model is formulated via balance of linear momentum in the first Piola Kirchhoff $\mathbf{P}$:

$\mathbf{P}(\mathbf{F},T^{\rm{a}}) := \partial_{\mathbf{F}} W_{\rm{passive}}(\mathbf{F}) + \mathbf{P}^{\rm{a}}(\mathbf{F}, T^{\rm{a}})$

where the passive material response can be described by an energy $W_{\rm{passive}$ and $T^{\rm{a}}$ the active tension generated by the contraction model.
"""
struct ActiveStressModel{Mat, ASMod, CMod, MS} <: AbstractMaterialModel
    material_model::Mat
    active_stress_model::ASMod
    contraction_model::CMod
    microstructure_model::MS
end

function default_initial_state!(
        uq,
        model::Union{GeneralizedHillModel, ExtendedHillModel, ActiveStressModel}
)
    default_initial_state!(uq, model.contraction_model)
end

function setup_coefficient_cache(m::ActiveStressModel, qr::QuadratureRule, sdh::SubDofHandler)
    return setup_coefficient_cache(m.microstructure_model, qr, sdh)
end

function stress_function(model::ActiveStressModel, F::Tensor{2}, coefficients, cell_state)
    ∂Ψ∂F = Tensors.gradient(F_ad -> Ψ(F_ad, coefficients, model.material_model), F)

    P2 = 𝓝(cell_state, F, coefficients, model.contraction_model) *
         active_stress(model.active_stress_model, F, coefficients)
    return ∂Ψ∂F + P2
end
function stress_and_tangent(model::ActiveStressModel, F::Tensor{2}, coefficients, cell_state)
    ∂²Ψ∂F², ∂Ψ∂F = Tensors.hessian(F_ad -> Ψ(F_ad, coefficients, model.material_model), F, :all)

    ∂2, P2 = Tensors.gradient(
        F_ad -> 𝓝(cell_state, F_ad, coefficients, model.contraction_model) *
                active_stress(model.active_stress_model, F_ad, coefficients),
        F,
        :all
    )
    return ∂Ψ∂F + P2, ∂²Ψ∂F² + ∂2
end

function gather_internal_variable_infos(model::ActiveStressModel)
    return gather_internal_variable_infos(model.contraction_model)
end

function setup_internal_cache(
        material_model::Union{<:ActiveStressModel, <:ExtendedHillModel, <:GeneralizedHillModel},
        qr::QuadratureRule,
        sdh::SubDofHandler
)
    setup_contraction_model_cache(material_model.contraction_model, qr, sdh)
end
function setup_internal_cache(
        material_model::Union{
            <:ElastodynamicsModel{<:ActiveStressModel},
            <:ElastodynamicsModel{<:ExtendedHillModel},
            <:ElastodynamicsModel{<:GeneralizedHillModel}
        },
        qr::QuadratureRule,
        sdh::SubDofHandler
)
    setup_contraction_model_cache(material_model.rhs.contraction_model, qr, sdh)
end

# TODO this actually belongs to the multi-level newton file :)
# Dual (global cache and element-level cache) use for now to make it non-allocating.
struct GenericFirstOrderRateIndependentCondensationMaterialStateCache{
    LocalModelType,
    LocalModelCacheType,
    LocalSolverType,
    QType,
    QType2,
    T,
    LVH
} <: RateIndependentCondensationMaterialStateCache
    # The actual model
    model::LocalModelType
    model_cache::LocalModelCacheType
    # Internal state at t and tprev
    # TODO play around with using a Qvector here and throw out lvh
    Q::QType
    Qprev::QType
    # t - tprev
    Δt::T
    local_solver_cache::LocalSolverType
    lvh::LVH
    # These are used locally
    localQ::QType2
    localQprev::QType2
end

function _solve_local_sarcomere_dQdF(
        dQdλ,
        dλdF,
        λ,
        F,
        coefficients,
        active_term_model,
        wrapper::CaDrivenInternalSarcomereModel
)
    return _solve_local_sarcomere_dQdF(
        dQdλ,
        dλdF,
        λ,
        F,
        coefficients,
        active_term_model,
        wrapper.model
    )
end

function _solve_local_sarcomere_dQdF(
        dQdλ,
        dλdF,
        λ,
        F,
        coefficients,
        active_term_model,
        sacromere_model::RDQ20MFModel
)
    dfgdQ = active_stress(active_term_model, F, coefficients) * fraction_single_overlap(sacromere_model, λ)
    dQdF  = (dQdλ[18] + dQdλ[20]) * dfgdQ ⊗ dλdF
    return -dQdF
end

# Local solve
function solve_internal_timestep(
        material_model::ActiveStressModel,
        state_cache::GenericFirstOrderRateIndependentCondensationMaterialStateCache,
        λ,
        dλdt,
        Q,
        Qprev,
        Ca
)
    @unpack Δt = state_cache
    #     dsdt = sarcomere_rhs(s,λ,t)
    # <=> (sₜ₁ - sₜ₀) / Δt = sarcomere_rhs(sₜ₁,λₜ₁,t1)

    function local_residual!(R, Q, λ, dλdt)
        dQ = zeros(eltype(Q), length(Q)) # TODO preallocate during setup
        sarcomere_rhs!(dQ, Q, λ, dλdt, Ca, time, material_model.contraction_model)
        @.. R = (Q - Qprev) / state_cache.Δt - dQ
        return nothing
    end

    function local_residual_jac_wrap!(R, Q)
        return local_residual!(R, Q, λ, dλdt)
    end

    R = state_cache.local_solver_cache.residual
    J = state_cache.local_solver_cache.J
    rtol = min(state_cache.local_solver_cache.params.tol, state_cache.local_solver_cache.outer_tol)
    for newton_iter in 1:(state_cache.local_solver_cache.params.max_iters)
        ForwardDiff.jacobian!(J, local_residual_jac_wrap!, R, Q)
        local_residual!(R, Q, λ, dλdt)
        ΔQ = J \ R
        Q .-= ΔQ
        residualnorm = norm(R)
        if residualnorm < state_cache.local_solver_cache.params.tol
            break
        elseif newton_iter == state_cache.local_solver_cache.params.max_iters
            state_cache.local_solver_cache.retcode = SciMLBase.ReturnCode.MaxIters
            @debug "Reached maximum local Newton iterations at cell $(cellid(geometry_cache)) qp $(qp.i). Aborting. ||r|| = $(residualnorm)" _group=:nlsolve
            return Q, J
        elseif isnan(residualnorm)
            state_cache.local_solver_cache.retcode = SciMLBase.ReturnCode.ConvergenceFailure
            @debug "Newton-Raphson diverged. Aborting. ||r|| = $residualnorm" _group=:nlsolve
            return Q, J
        end
    end
    ForwardDiff.jacobian!(J, local_residual_jac_wrap!, R, Q)
    state_cache.local_solver_cache.retcode = SciMLBase.ReturnCode.Success
    return Q, J
end

function solve_local_constraint(
        F::Tensor{2, dim},
        coefficients,
        material_model::ActiveStressModel,
        state_cache::GenericFirstOrderRateIndependentCondensationMaterialStateCache,
        geometry_cache,
        qp,
        time
) where {dim}
    Qflat, Qprevflat = _query_local_state(state_cache, geometry_cache, qp)
    # Early out if any of the previous local solves failed
    if state_cache.local_solver_cache.retcode ∉
       (SciMLBase.ReturnCode.Default, SciMLBase.ReturnCode.Success)
        return Qflat, zero(Tensor{4, dim, Float64, 4^dim})
    end

    function computeλ(F)
        f = F ⋅ coefficients.f
        return √(f ⋅ f)
    end

    # Frozen variables
    dλdF, λ = Tensors.gradient(computeλ, F, :all)
    dλdt = 0.0 # TODO query
    Ca = evaluate_coefficient(state_cache.model_cache.calcium_cache, geometry_cache, qp, time)

    Q, J = solve_internal_timestep(material_model, state_cache, λ, dλdt, Qflat, Qprevflat, Ca)
    # Abort if local solve failed
    if state_cache.local_solver_cache.retcode ∉
       (SciMLBase.ReturnCode.Default, SciMLBase.ReturnCode.Success)
        return Qflat, zero(Tensor{4, dim, Float64, 4^dim})
    end
    # Qflat .= Q
    _store_local_state!(state_cache, geometry_cache, qp)

    # Solve corrector problem
    function local_residual_rhs_wrap!(R, λ)
        dQ = zeros(eltype(λ), length(Q)) # TODO preallocate during setup
        sarcomere_rhs!(dQ, Q, λ, dλdt, Ca, time, material_model.contraction_model)
        @.. R = (Q - Qprevflat) / state_cache.Δt - dQ
        return nothing
    end
    R = state_cache.local_solver_cache.residual
    ∂fₗ∂λ = state_cache.local_solver_cache.rhs_corrector
    ForwardDiff.derivative!(∂fₗ∂λ, local_residual_rhs_wrap!, R, λ)
    dQdλ = J \ -∂fₗ∂λ

    return Q,
    _solve_local_sarcomere_dQdF(
        dQdλ,
        dλdF,
        λ,
        F,
        coefficients,
        material_model.active_stress_model,
        material_model.contraction_model
    )
end

function solve_local_constraint_state_only(
        F::Tensor{2, dim},
        coefficients,
        material_model::ActiveStressModel,
        state_cache::GenericFirstOrderRateIndependentCondensationMaterialStateCache,
        geometry_cache,
        qp,
        time
) where {dim}
    Qflat, Qprevflat = _query_local_state(state_cache, geometry_cache, qp)
    # Early out if any of the previous local solves failed
    if state_cache.local_solver_cache.retcode ∉
       (SciMLBase.ReturnCode.Default, SciMLBase.ReturnCode.Success)
        return Qflat, zero(Tensor{4, dim, Float64, 4^dim})
    end

    function computeλ(F)
        f = F ⋅ coefficients.f
        return √(f ⋅ f)
    end

    # Frozen variables
    dλdF, λ = Tensors.gradient(computeλ, F, :all)
    dλdt = 0.0 # TODO query
    Ca = evaluate_coefficient(state_cache.model_cache.calcium_cache, geometry_cache, qp, time)

    Q, J = solve_internal_timestep(material_model, state_cache, λ, dλdt, Qflat, Qprevflat, Ca)
    # Abort if local solve failed
    if state_cache.local_solver_cache.retcode ∉
       (SciMLBase.ReturnCode.Default, SciMLBase.ReturnCode.Success)
        return Qflat, zero(Tensor{4, dim, Float64, 4^dim})
    end
    # Qflat .= Q
    _store_local_state!(state_cache, geometry_cache, qp)

    return Q
end

# Some debug materials
Base.@kwdef struct LinearMaxwellMaterial{T, sdim} <: AbstractMaterialModel
    E₀::T
    E₁::T
    μ::T
    η₁::T
    ν::T
end
function LinearMaxwellMaterial(E₀::T, Eₗ::T, μ::T, η₁::T, ν::T) where {T}
    LinearMaxwellMaterial{T, 3}(E₀, Eₗ, μ, η₁, ν)
end

local_function_size(model::QuasiStaticModel) = local_function_size(model.material_model)
function local_function_size(model::AbstractMaterialModel)
    return _compute_local_function_size(0, gather_internal_variable_infos(model))
end

function _compute_local_function_size(total, lvis::Base.AbstractVecOrTuple)
    for lvi in lvis
        total += _compute_local_function_size(total, lvi)
    end
    return total
end

function _compute_local_function_size(total, lvi::InternalVariableInfo)
    return lvi.size
end

function _query_local_state(
        state_cache::GenericFirstOrderRateIndependentCondensationMaterialStateCache,
        geometry_cache,
        qp
)
    dh                                   = state_cache.lvh.dh
    dofs                                 = celldofsview(dh, cellid(geometry_cache))
    size                                 = local_function_size(state_cache.model)
    range_begin                          = 1 + (qp.i - 1) * size
    range_end                            = qp.i * size
    Qv                                   = @view state_cache.Q[range_begin:range_end]
    Qpv                                  = @view state_cache.Qprev[range_begin:range_end]
    @inbounds @.. state_cache.localQ     = Qv
    @inbounds @.. state_cache.localQprev = Qpv

    return state_cache.localQ, state_cache.localQprev
end

function _store_local_state!(
        state_cache::GenericFirstOrderRateIndependentCondensationMaterialStateCache,
        geometry_cache,
        qp
)
    dh = state_cache.lvh.dh
    dofs = celldofsview(dh, cellid(geometry_cache))
    # TODO properly via gather_internal_variable_infos :)
    size = local_function_size(state_cache.model)
    range_begin = 1 + (qp.i - 1) * size
    range_end = qp.i * size
    Qv = @view state_cache.Q[range_begin:range_end]
    @inbounds @.. Qv = state_cache.localQ

    return nothing
end

function solve_internal_timestep(
        material::LinearMaxwellMaterial,
        state_cache::GenericFirstOrderRateIndependentCondensationMaterialStateCache,
        ε::SymmetricTensor{2, dim},
        εᵛflat,
        εᵛprevflat
) where {dim}
    @unpack Δt = state_cache
    εᵛ₁ = SymmetricTensor{2, dim}(εᵛflat)
    εᵛ₀ = SymmetricTensor{2, dim}(εᵛprevflat)
    #     dεᵛdt = E₁/η₁ c : (ε - εᵛ)
    # <=> (εᵛ₁ - εᵛ₀) / Δt = E₁/η₁ c : (ε - εᵛ₁) = E₁/η₁ c : ε - E₁/η₁ c : εᵛ₁
    # <=> εᵛ₁ / Δt + E₁/η₁ c : εᵛ₁ = εᵛ₀/Δt + E₁/η₁ c : ε
    # <=> (𝐈 / Δt + E₁/η₁ c) : εᵛ₁ = εᵛ₀/Δt + E₁/η₁ c : ε

    (; E₀, E₁, μ, η₁, ν) = material
    I = one(ε)
    c₁ = ν / ((ν + 1) * (1 - 2ν)) * I ⊗ I
    c₂ = 1 / (1 + ν) * one(c₁)
    ℂ = c₁ + c₂

    # FIXME non-allocating version by using state_cache nlsolver
    A = tomandel(SMatrix, one(ℂ) / Δt + E₁ / η₁ * ℂ)
    b = tomandel(SVector, εᵛ₀ / Δt + E₁ / η₁ * ℂ ⊡ ε)
    return frommandel(typeof(ε), A \ b)
end

function solve_local_constraint(
        F::Tensor{2, dim},
        coefficients,
        material_model::LinearMaxwellMaterial,
        state_cache::GenericFirstOrderRateIndependentCondensationMaterialStateCache,
        geometry_cache,
        qp,
        time
) where {dim}
    Qflat, Qprevflat = _query_local_state(state_cache, geometry_cache, qp)
    ε = symmetric(F - one(F))
    Q = solve_internal_timestep(material_model, state_cache, ε, Qflat, Qprevflat)
    Qflat .= Q.data
    _store_local_state!(state_cache, geometry_cache, qp)

    # Corrector
    function solve_internal_timestep_corrector(
            material::LinearMaxwellMaterial,
            state_cache::GenericFirstOrderRateIndependentCondensationMaterialStateCache,
            ε,
            εᵛflat,
            εᵛprevflat,
            coefficients
    )
        @unpack Δt = state_cache
        εᵛ₁ = SymmetricTensor{2, dim}(εᵛflat)
        εᵛ₀ = SymmetricTensor{2, dim}(εᵛprevflat)
        # Local problem: (𝐈 / Δt + E₁/η₁ c) : εᵛ₁ = εᵛ₀/Δt + E₁/η₁ c : ε
        # =>  dLdQ = 𝐈 / Δt + E₁/η₁ c   := A
        # => -dLdF = E₁/η₁ c            := B

        (; E₀, E₁, μ, η₁, ν) = material
        I = one(ε)
        c₁ = ν / ((ν + 1) * (1 - 2ν)) * I ⊗ I
        c₂ = 1 / (1 + ν) * one(c₁)
        ℂ = c₁ + c₂

        # FIXME non-allocating version by using state_cache nlsolver
        A = tomandel(SMatrix, one(ℂ) / Δt + E₁ / η₁ * ℂ)
        B = tomandel(SMatrix, E₁ / η₁ * ℂ)
        return frommandel(typeof(ℂ), A \ B)
    end
    dQdF = solve_internal_timestep_corrector(
        material_model,
        state_cache,
        ε,
        Qflat,
        Qprevflat,
        coefficients
    )
    ∂P∂Q = Tensors.gradient(εᵛ -> stress_function(material_model, ε, coefficients, εᵛ), Q)

    return Q, ∂P∂Q ⊡ dQdF
end

function solve_local_constraint_state_only(
        F::Tensor{2, dim},
        coefficients,
        material_model::LinearMaxwellMaterial,
        state_cache::GenericFirstOrderRateIndependentCondensationMaterialStateCache,
        geometry_cache,
        qp,
        time
) where {dim}
    Qflat, Qprevflat = _query_local_state(state_cache, geometry_cache, qp)
    ε = symmetric(F - one(F))
    Q = solve_internal_timestep(material_model, state_cache, ε, Qflat, Qprevflat)
    Qflat .= Q.data
    _store_local_state!(state_cache, geometry_cache, qp)

    return Q
end

function stress_function(material::LinearMaxwellMaterial, ε, coefficients, εᵛ)
    (; E₀, E₁, μ, η₁, ν) = material
    I = one(ε)
    c₁ = ν / ((ν + 1) * (1 - 2ν)) * I ⊗ I
    c₂ = 1 / (1 + ν) * one(c₁)
    ℂ = c₁ + c₂
    return E₀ * ℂ ⊡ ε + E₁ * ℂ ⊡ (ε - εᵛ)
end

function stress_and_tangent(
        material_model::LinearMaxwellMaterial,
        F::Tensor{2},
        coefficients,
        εᵛ::SymmetricTensor{2}
)
    ε = symmetric(F - one(F))
    ∂σ∂ε, σ = Tensors.gradient(ε -> stress_function(material_model, ε, coefficients, εᵛ), ε, :all)
    return σ, ∂σ∂ε
end

function setup_coefficient_cache(m::LinearMaxwellMaterial, qr::QuadratureRule, sdh::SubDofHandler)
    return NoMicrostructureModel() # FIXME what should we do here? :)
end

function setup_internal_cache(
        material_model::LinearMaxwellMaterial,
        qr::QuadratureRule,
        sdh::SubDofHandler
)
    return EmptyRateIndependentCondensationMaterialStateCache()
end

function gather_internal_variable_infos(model::LinearMaxwellMaterial{T, sdim}) where {T, sdim}
    if sdim == 3
        return InternalVariableInfo(:εᵛ, 6)
    else
        return InternalVariableInfo(:εᵛ, 4)
    end
end
