abstract type AbstractIsotropicMicrostructure end

# These are for nice error messages
struct NoMicrostructure <: AbstractIsotropicMicrostructure end

struct NoMicrostructureModel end

setup_coefficient_cache(
    coefficient::NoMicrostructureModel,
    qr::QuadratureRule,
    sdh::SubDofHandler,
) = coefficient

function evaluate_coefficient(fsn::NoMicrostructureModel, cell_cache, qp::QuadraturePoint, t)
    return NoMicrostructure()
end

"""
Struct must define f::Vec
"""
abstract type AbstractTransverselyIsotropicMicrostructure <: AbstractIsotropicMicrostructure end

"""
Struct must define f::Vec, s::Vec and in 3D n::Vec
"""
abstract type AbstractOrthotropicMicrostructure <: AbstractTransverselyIsotropicMicrostructure end

struct AnisotropicPlanarMicrostructure{T} <: AbstractOrthotropicMicrostructure
    f::Vec{2, T}
    s::Vec{2, T}
end
Base.zero(::Type{AnisotropicPlanarMicrostructure{T}}) where {T} =
    AnisotropicPlanarMicrostructure(zero(Vec{3, T}), zero(Vec{3, T}))

# Compat with spectral coefficient
@inline function _eval_st_coefficient(M::AnisotropicPlanarMicrostructure, λ::SVector{2})
    return λ[1] * M.f ⊗ M.f + λ[2] * M.s ⊗ M.s
end

struct AnisotropicPlanarMicrostructureModel{FiberCoefficientType, SheetletCoefficientType}
    fiber_coefficient::FiberCoefficientType
    sheetlet_coefficient::SheetletCoefficientType
end

struct AnisotropicPlanarMicrostructureCache{FC, SC}
    fiber_cache::FC
    sheetlet_cache::SC
end

function duplicate_for_device(device, cache::AnisotropicPlanarMicrostructureCache)
    return AnisotropicPlanarMicrostructureCache(
        duplicate_for_device(device, cache.fiber_cache),
        duplicate_for_device(device, cache.sheetlet_cache),
        duplicate_for_device(device, cache.normal_cache),
    )
end

function setup_coefficient_cache(
    coefficient::AnisotropicPlanarMicrostructureModel,
    qr::QuadratureRule,
    sdh::SubDofHandler,
)
    return AnisotropicPlanarMicrostructureCache(
        setup_coefficient_cache(coefficient.fiber_coefficient, qr, sdh),
        setup_coefficient_cache(coefficient.sheetlet_coefficient, qr, sdh),
    )
end

function evaluate_coefficient(
    fsn::AnisotropicPlanarMicrostructureCache,
    cell_cache,
    qp::QuadraturePoint{2},
    t,
)
    f = evaluate_coefficient(fsn.fiber_cache, cell_cache, qp, t)
    s = evaluate_coefficient(fsn.sheetlet_cache, cell_cache, qp, t)

    return AnisotropicPlanarMicrostructure(orthogonalize_system(f, s)...)
end


struct TransverselyIsotropicMicrostructure{dim, T} <: AbstractTransverselyIsotropicMicrostructure
    f::Vec{dim, T}
end
Base.zero(::Type{TransverselyIsotropicMicrostructure{T}}) where {T} =
    TransverselyIsotropicMicrostructure(zero(Vec{dim, T}))

# Compat with spectral coefficient
@inline function _eval_st_coefficient(M::TransverselyIsotropicMicrostructure, λ::SVector{2})
    Mf = M.f ⊗ M.f
    return λ[1] * Mf + λ[2] * (one(Mf) - Mf)
end

struct TransverselyIsotropicMicrostructureModel{FiberCoefficientType}
    fiber_coefficient::FiberCoefficientType
end

struct TransverselyIsotropicMicrostructureCache{FC}
    fiber_cache::FC
end

function duplicate_for_device(device, cache::TransverselyIsotropicMicrostructureCache)
    return TransverselyIsotropicMicrostructureCache(duplicate_for_device(device, cache.fiber_cache))
end

function setup_coefficient_cache(
    coefficient::TransverselyIsotropicMicrostructureModel,
    qr::QuadratureRule,
    sdh::SubDofHandler,
)
    return AnisotropicPlanarMicrostructureCache(
        setup_coefficient_cache(coefficient.fiber_coefficient, qr, sdh),
    )
end

function evaluate_coefficient(
    fsn::TransverselyIsotropicMicrostructureCache,
    cell_cache,
    qp::QuadraturePoint{3},
    t,
)
    f = evaluate_coefficient(fsn.fiber_cache, cell_cache, qp, t)
    return TransverselyIsotropicMicrostructure(f)
end


struct OrthotropicMicrostructure{T} <: AbstractOrthotropicMicrostructure
    f::Vec{3, T}
    s::Vec{3, T}
    n::Vec{3, T}
end
Base.zero(::Type{OrthotropicMicrostructure{T}}) where {T} =
    OrthotropicMicrostructure(zero(Vec{dim, T}), zero(Vec{dim, T}), zero(Vec{dim, T}))

# Compat with spectral coefficient
@inline function _eval_st_coefficient(M::OrthotropicMicrostructure, λ::SVector{3})
    return λ[1] * M.f ⊗ M.f + λ[2] * M.s ⊗ M.s + λ[3] * M.n ⊗ M.n
end

struct OrthotropicMicrostructureModel{
    FiberCoefficientType,
    SheetletCoefficientType,
    NormalCoefficientType,
}
    fiber_coefficient::FiberCoefficientType
    sheetlet_coefficient::SheetletCoefficientType
    normal_coefficient::NormalCoefficientType
end

struct OrthotropicMicrostructureCache{FC, SC, NC}
    fiber_cache::FC
    sheetlet_cache::SC
    normal_cache::NC
end

function duplicate_for_device(device, cache::OrthotropicMicrostructureCache)
    return OrthotropicMicrostructureCache(
        duplicate_for_device(device, cache.fiber_cache),
        duplicate_for_device(device, cache.sheetlet_cache),
        duplicate_for_device(device, cache.normal_cache),
    )
end

function setup_coefficient_cache(
    coefficient::OrthotropicMicrostructureModel,
    qr::QuadratureRule,
    sdh::SubDofHandler,
)
    return OrthotropicMicrostructureCache(
        setup_coefficient_cache(coefficient.fiber_coefficient, qr, sdh),
        setup_coefficient_cache(coefficient.sheetlet_coefficient, qr, sdh),
        setup_coefficient_cache(coefficient.normal_coefficient, qr, sdh),
    )
end

function evaluate_coefficient(
    fsn::OrthotropicMicrostructureCache,
    cell_cache,
    qp::QuadraturePoint{3},
    t,
)
    f = evaluate_coefficient(fsn.fiber_cache, cell_cache, qp, t)
    s = evaluate_coefficient(fsn.sheetlet_cache, cell_cache, qp, t)
    n = evaluate_coefficient(fsn.normal_cache, cell_cache, qp, t)

    return OrthotropicMicrostructure(orthogonalize_system(f, s, n)...)
end

"""
Linear transmural distribution of the microstructure with three angles to describe all reachable physiological angles.
"""
Base.@kwdef struct ODB25LTMicrostructureParameters{T}
    αendo::T = deg2rad(60.0)
    αepi::T  = deg2rad(-60.0)
    βendo::T = 0.0
    βepi::T  = 0.0
    γendo::T = 0.0
    γepi::T  = 0.0
end

"""
    compute_local_microstructure(params::..., x, axes)

Compute fiber, sheetlet and normal direction from the transmural, circumferential, apicobasal directions
in addition to given helix, transversal and sheetlet angles. The theory is based on the classical work by
[StreSpoPatRosSon:1969:foc](@citet).
"""
function compute_local_microstructure(
    p::ODB25LTMicrostructureParameters,
    x::Union{LVCoordinate, BiVCoordinate},
    axes::Union{LVCoordinate, BiVCoordinate},
)
    (; αendo, αepi, βendo, βepi, γendo, γepi) = p

    # linear interpolation of rotation angle
    α = (1-x.transmural) * αendo + (x.transmural) * αepi
    β = (1-x.transmural) * βendo + (x.transmural) * βepi
    γ = (1-x.transmural) * γendo + (x.transmural) * γepi

    circumferential_direction = axes.rotational
    transmural_direction      = axes.transmural
    apicobasal_direction      = axes.apicobasal

    # First we abc_fsn the helix rotation ...
    f₀ = rotate_around(circumferential_direction, transmural_direction, α)
    f₀ = normalize(f₀)
    # ... followed by the transversal_angle ...
    f₀ = rotate_around(f₀, apicobasal_direction, -β)
    f₀ = normalize(f₀)

    # Then we construct the the orthogonal sheetlet vector ...
    s₀ = rotate_around(circumferential_direction, transmural_direction, α+π/2.0)
    s₀ = normalize(s₀)
    s₀ = orthogonalize(s₀, f₀)
    s₀ = normalize(s₀)
    s₀ = rotate_around(s₀, f₀, -γ)
    s₀ = normalize(s₀)

    # Compute normal :)
    n₀ = f₀ × s₀
    n₀ = normalize(n₀)

    return OrthotropicMicrostructure(f₀, s₀, n₀)
end

"""
    create_simple_microstructure_model(coordinate_system, ip_component::VectorInterpolationCollection; endo_helix_angle = deg2rad(80.0), epi_helix_angle = deg2rad(-65.0), endo_transversal_angle = 0.0, epi_transversal_angle = 0.0, sheetlet_angle = 0.0, make_orthogonal=true)

Create a rotating fiber field by deducing the circumferential direction from apicobasal and transmural gradients.
"""
function create_simple_microstructure_model(
    coordinate_system,
    ip_collection::VectorizedInterpolationCollection{3};
    endo_helix_angle = deg2rad(80.0),
    epi_helix_angle = deg2rad(-65.0),
    endo_transversal_angle = 0.0,
    epi_transversal_angle = 0.0,
    endo_rot_angle = 0.0,
    epi_rot_angle = 0.0,
)
    return create_microstructure_model(
        coordinate_system,
        ip_collection,
        ODB25LTMicrostructureParameters(
            endo_helix_angle,
            epi_helix_angle,
            endo_transversal_angle,
            epi_transversal_angle,
            endo_rot_angle,
            epi_rot_angle,
        ),
    )
end

"""
    create_microstructure_model(coordinate_system::CoordinateSystemCoefficient, ip::VectorInterpolationCollection, parameters)

Create a rotating fiber field by deducing the circumferential direction from apicobasal and transmural gradients.
"""
function create_microstructure_model(
    coordinate_system::CoordinateSystemCoefficient,
    ip_collection::VectorizedInterpolationCollection{3},
    parameters,
)
    @unpack dh = coordinate_system

    # TODO this storage is redundant, can we reduce the memory footprint?
    offsets = copy(dh.cell_dofs_offset)
    push!(offsets, length(dh.cell_dofs)+1)

    # The vectors follow the spatial dimension and precision of the grid
    Tv = get_coordinate_type(get_grid(dh))
    f_buf = ElementwiseData(zeros(Tv, length(dh.cell_dofs)), offsets)
    s_buf = ElementwiseData(zeros(Tv, length(dh.cell_dofs)), offsets)
    n_buf = ElementwiseData(zeros(Tv, length(dh.cell_dofs)), offsets)

    qrs = NodalQuadratureRuleCollection(ip_collection.base)
    cvs = CellValueCollection(qrs, ip_collection.base)
    for sdh in dh.subdofhandlers
        first_cell = getcells(Ferrite.get_grid(dh), first(sdh.cellset))
        cv = getcellvalues(cvs, first_cell)
        for cell in CellIterator(sdh)
            cellindex = cellid(cell)
            reinit!(cv, cell)
            dof_indices = celldofs(cell)

            for qp in QuadratureIterator(cv)
                # TODO grab these via some interface!
                apicobasal_direction =
                    function_gradient(cv, qp, coordinate_system.u_apicobasal[dof_indices])
                apicobasal_direction /= norm(apicobasal_direction)
                transmural_direction =
                    function_gradient(cv, qp, coordinate_system.u_transmural[dof_indices])
                transmural_direction /= norm(transmural_direction)
                apicobasal_direction =
                    apicobasal_direction -
                    (apicobasal_direction ⋅ transmural_direction) * transmural_direction # We do this fix to ensure local orthogonality
                circumferential_direction = transmural_direction × apicobasal_direction
                circumferential_direction /= norm(circumferential_direction)
                axes = LVCoordinate(;
                    transmural = transmural_direction,
                    rotational = circumferential_direction,
                    apicobasal = apicobasal_direction,
                )
                # TODO grab these via some interface!
                x = LVCoordinate(;
                    transmural = function_value(
                        cv,
                        qp,
                        coordinate_system.u_transmural[dof_indices],
                    ),
                    rotational = function_value(
                        cv,
                        qp,
                        coordinate_system.u_rotational[dof_indices],
                    ),
                    apicobasal = function_value(
                        cv,
                        qp,
                        coordinate_system.u_apicobasal[dof_indices],
                    ),
                )

                coeff = compute_local_microstructure(parameters, x, axes)

                f_buf[qp.i, cellindex] = Tv(coeff.f)
                s_buf[qp.i, cellindex] = Tv(coeff.s)
                n_buf[qp.i, cellindex] = Tv(coeff.n)
            end
        end
    end

    OrthotropicMicrostructureModel(
        FieldCoefficient(f_buf, ip_collection),
        FieldCoefficient(s_buf, ip_collection),
        FieldCoefficient(n_buf, ip_collection),
    )
end
