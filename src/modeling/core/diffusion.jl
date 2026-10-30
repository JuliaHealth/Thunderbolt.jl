@doc raw"""
    BilinearDiffusionIntegrator{CoefficientType}

Represents the integrand of the bilinear form ``a(u,v) = -\int \nabla v(x) \cdot D(x) \nabla u(x) dx`` for a given diffusion tensor ``D(x)`` and ``u,v`` from the same function space.
"""
struct BilinearDiffusionIntegrator{CoefficientType, QRC <: QuadratureRuleCollection} <:
       AbstractBilinearIntegrator
    D::CoefficientType
    qrc::QRC
    sym::Symbol
end

"""
The cache associated with [`BilinearDiffusionIntegrator`](@ref) to assemble element diffusion matrices.
"""
struct BilinearDiffusionElementCache{CoefficientCacheType, CV} <: AbstractVolumetricElementCache
    Dcache::CoefficientCacheType
    cellvalues::CV
end

function duplicate_for_device(device, cache::BilinearDiffusionElementCache)
    return BilinearDiffusionElementCache(
        duplicate_for_device(device, cache.Dcache),
        duplicate_for_device(device, cache.cellvalues),
    )
end

Ferrite.getnquadpoints(element_cache::BilinearDiffusionElementCache) =
    getnquadpoints(element_cache.cellvalues)
FerriteOperators.reinit_values!(element_cache::BilinearDiffusionElementCache, cell) =
    reinit!(element_cache.cellvalues, cell)

FerriteOperators.provides_analytic(
    ::Type{<:BilinearDiffusionElementCache},
    ::FerriteOperators.JacobianKind{:u},
) = true

function FerriteOperators.assemble_cell!(
    req::FerriteOperators.JacobianRequest{:u},
    element_cache::BilinearDiffusionElementCache,
    args::FerriteOperators.CellArgs,
)
    @unpack cellvalues, Dcache = element_cache
    Kₑ = req.K
    cell = args.cell
    time = FerriteOperators.evaluation_time(args.ctx)
    n_basefuncs = getnbasefunctions(cellvalues)

    for qp in QuadratureIterator(cellvalues)
        D_loc = evaluate_coefficient(Dcache, cell, qp, time)
        dΩ = getdetJdV(cellvalues, qp)
        for i = 1:n_basefuncs
            ∇Nᵢ = shape_gradient(cellvalues, qp, i)
            for j = 1:n_basefuncs
                ∇Nⱼ = shape_gradient(cellvalues, qp, j)
                Kₑ[i, j] -= _inner_product_helper(∇Nⱼ, D_loc, ∇Nᵢ) * dΩ
            end
        end
    end
end

# The bilinear form induces a linear operator, so its residual is the element diffusion matrix
# acting on the element vector -- mandatory, so the element composes into nonlinear operators and
# AD-based sensitivities.
function FerriteOperators.assemble_cell!(
    req::FerriteOperators.ResidualRequest,
    element_cache::BilinearDiffusionElementCache,
    args::FerriteOperators.CellArgs,
)
    @unpack cellvalues, Dcache = element_cache
    cell = args.cell
    uₑ = args.states.u
    time = FerriteOperators.evaluation_time(args.ctx)
    n_basefuncs = getnbasefunctions(cellvalues)

    for qp in QuadratureIterator(cellvalues)
        D_loc = evaluate_coefficient(Dcache, cell, qp, time)
        dΩ = getdetJdV(cellvalues, qp)
        ∇u = function_gradient(cellvalues, qp, uₑ)
        for i = 1:n_basefuncs
            ∇Nᵢ = shape_gradient(cellvalues, qp, i)
            req.r[i] -= _inner_product_helper(∇u, D_loc, ∇Nᵢ) * dΩ
        end
    end
end

FerriteOperators.element_value_type(c::BilinearDiffusionElementCache) =
    FerriteOperators.element_value_type(c.cellvalues)

function setup_element_cache(element_model::BilinearDiffusionIntegrator, sdh::SubDofHandler)
    qr     = getquadraturerule(element_model.qrc, sdh)
    ip     = Ferrite.getfieldinterpolation(sdh, element_model.sym)
    ip_geo = geometric_subdomain_interpolation(sdh)
    return BilinearDiffusionElementCache(
        setup_coefficient_cache(element_model.D, qr, sdh),
        CellValues(FerriteOperators.element_value_type(qr), qr, ip, ip_geo),
    )
end

@doc raw"""
    TransientDiffusionModel(conductivity_coefficient, source_term, solution_variable_symbol)

Model formulated as ``\partial_t u = \nabla \cdot \kappa(x) \nabla u + f``
"""
struct TransientDiffusionModel{ConductivityCoefficientType, SourceType <: AbstractSourceTerm}
    κ::ConductivityCoefficientType
    source::SourceType
    solution_variable_symbol::Symbol
end

get_field_variable_names(model::TransientDiffusionModel) = (model.solution_variable_symbol,)
get_volumetric_weak_form_names(model::TransientDiffusionModel) = (model.solution_variable_symbol,)

@doc raw"""
    BilinearDiffusionIntegrator{CoefficientType}

Represents the integrand of the bilinear form ``a(u,v) = -\int \nabla v(x) \cdot D(x) \nabla u(x) dx`` for a given diffusion tensor ``D(x)`` and ``u,v`` from the same function space.
"""
struct BilinearInterfaceDiffusionIntegrator{CoefficientType, QRC <: QuadratureRuleCollection} <:
       AbstractBilinearIntegrator
    D::CoefficientType
    qrc::QRC
    sym1::Symbol
    sym2::Symbol
end

"""
The cache associated with [`BilinearDiffusionIntegrator`](@ref) to assemble element diffusion matrices.
"""
struct BilinearInterfaceDiffusionElementCache{CoefficientCacheType, CV} <:
       AbstractVolumetricElementCache
    Dcache::CoefficientCacheType
    cellvalues::CV
end

# FerriteInterfaceElements does not know the device API, and `InterfaceCellValues` has no
# field-wise constructor: a deep copy is the correct CPU duplicate — private worker scratch,
# internal here/there aliasing preserved. A non-CPU device stays a loud MethodError.
duplicate_for_device(device::AbstractCPUDevice, cv::InterfaceCellValues) = deepcopy(cv)

function duplicate_for_device(device, cache::BilinearInterfaceDiffusionElementCache)
    return BilinearInterfaceDiffusionElementCache(
        duplicate_for_device(device, cache.Dcache),
        duplicate_for_device(device, cache.cellvalues),
    )
end

Ferrite.getnquadpoints(element_cache::BilinearInterfaceDiffusionElementCache) =
    getnquadpoints(element_cache.cellvalues)
FerriteOperators.reinit_values!(element_cache::BilinearInterfaceDiffusionElementCache, cell) =
    reinit!(element_cache.cellvalues, cell)

FerriteOperators.provides_analytic(
    ::Type{<:BilinearInterfaceDiffusionElementCache},
    ::FerriteOperators.JacobianKind{:u},
) = true

function FerriteOperators.assemble_cell!(
    req::FerriteOperators.JacobianRequest{:u},
    element_cache::BilinearInterfaceDiffusionElementCache,
    args::FerriteOperators.CellArgs,
)
    (; cellvalues, Dcache) = element_cache
    Kₑ = req.K
    cell = args.cell
    time = FerriteOperators.evaluation_time(args.ctx)

    for qp = 1:getnquadpoints(cellvalues)
        D_loc = evaluate_coefficient(Dcache, cell, qp, time)
        dΩ = getdetJdV_average(cellvalues, qp)
        for i = 1:getnbasefunctions(cellvalues)
            jump_δu = shape_value_jump(cellvalues, qp, i)
            for j = 1:getnbasefunctions(cellvalues)
                jump_u = shape_value_jump(cellvalues, qp, j)
                Kₑ[i, j] -= (jump_δu * D_loc * jump_u) * dΩ
            end
        end
    end
end

# The bilinear form induces a linear operator, so its residual is the element interface matrix
# acting on the element vector -- mandatory, so the element composes into nonlinear operators and
# AD-based sensitivities.
function FerriteOperators.assemble_cell!(
    req::FerriteOperators.ResidualRequest,
    element_cache::BilinearInterfaceDiffusionElementCache,
    args::FerriteOperators.CellArgs,
)
    (; cellvalues, Dcache) = element_cache
    cell = args.cell
    uₑ = args.states.u
    time = FerriteOperators.evaluation_time(args.ctx)

    for qp = 1:getnquadpoints(cellvalues)
        D_loc = evaluate_coefficient(Dcache, cell, qp, time)
        dΩ = getdetJdV_average(cellvalues, qp)
        jump_u = function_value_jump(cellvalues, qp, uₑ)
        for i = 1:getnbasefunctions(cellvalues)
            jump_δu = shape_value_jump(cellvalues, qp, i)
            req.r[i] -= (jump_δu * D_loc * jump_u) * dΩ
        end
    end
end

function setup_element_cache(
    element_model::BilinearInterfaceDiffusionIntegrator,
    sdh::SubDofHandler,
)
    qr = getquadraturerule(element_model.qrc, sdh)
    ip = Ferrite.getfieldinterpolation(sdh, element_model.sym1)
    cv = InterfaceCellValues(qr, ip)
    return BilinearInterfaceDiffusionElementCache(
        setup_coefficient_cache(element_model.D, qr, sdh),
        cv,
    )
end


@doc raw"""
    TransientDiffusionModel(conductivity_coefficient, source_term, solution_variable_symbol)

Model formulated as ``\int_{\Gamma^{\text{P}/\text{M}}} [\![ \delta u ]\!] G [\![ u ]\!] \mathrm{d}\Gamma``.
"""
@concrete struct InterfaceDiffusionModel
    G
    solution_variable_symbol::Symbol
    interface_interpolation_symbol::Symbol
end

get_field_variable_names(model::InterfaceDiffusionModel) = (model.solution_variable_symbol,)
get_volumetric_weak_form_names(model::InterfaceDiffusionModel) = (model.solution_variable_symbol,)

is_coupling_model(::InterfaceDiffusionModel) = true

@doc raw"""
    SteadyDiffusionModel(conductivity_coefficient, source_term, solution_variable_symbol)

Model formulated as ``\nabla \cdot \kappa(x) \nabla u = f``
"""
struct SteadyDiffusionModel{ConductivityCoefficientType, SourceType <: AbstractSourceTerm}
    κ::ConductivityCoefficientType
    source::SourceType
    solution_variable_symbol::Symbol
end

get_field_variable_names(model::SteadyDiffusionModel) = (model.solution_variable_symbol,)
get_volumetric_weak_form_names(model::SteadyDiffusionModel) = (model.solution_variable_symbol,)
