abstract type AbstractPointwiseSolver <: AbstractSolver end
abstract type AbstractPointwiseSolverCache <: AbstractTimeSolverCache end

SciMLBase.isadaptive(::AbstractPointwiseSolver) = false

# Auxilliary functions to query the coordinate
@inline getcoordinate(f::F, i::I)       where {F<:AbstractPointwiseSolverCache,I}                  = getcoordinate(f, i, f.xs)
@inline getcoordinate(f::F, i::I, x::X) where {F<:AbstractPointwiseSolverCache,I,X<:Nothing}       = nothing
@inline getcoordinate(f::F, i::I, x::X) where {F<:AbstractPointwiseSolverCache,I,X<:AbstractArray} = x[i]

# Redirect to inner solve
function perform_step!(f::PointwiseODEFunction, cache::AbstractPointwiseSolverCache, t::Real, Δt::Real)
    @timeit_debug "reaction solve" _pointwise_step_outer_kernel!(f, t, Δt, cache, cache.uₙ)
end

# This controls the outer loop over the ODEs
function _pointwise_step_outer_kernel!(f::PointwiseODEFunction, t::Real, Δt::Real, cache::AbstractPointwiseSolverCache, ::Union{Vector,SubArray{<:Any,1,<:Vector}})
    @unpack npoints = f

    @batch minbatch=cache.batch_size_hint for i ∈ 1:npoints
        _pointwise_step_inner_kernel!(f.ode, i, t, Δt, cache) || return false
    end

    return true
end

"""
Simple forward euler to solve the cell model.
"""
Base.@kwdef struct ForwardEulerCellSolver{SolutionVectorType} <: AbstractPointwiseSolver
    solution_vector_type::Type{SolutionVectorType} = Vector{Float64}
    batch_size_hint::Int                           = 32
end

# Fully accelerator compatible
struct ForwardEulerCellSolverCache{duType, uType, dumType, umType, xType} <: AbstractPointwiseSolverCache
    du::duType
    # These vectors hold the data
    uₙ::uType
    uₙ₋₁::uType
    # These array view the data above to give easy indices of the form [ode index, local state index]
    dumat::dumType
    uₙmat::umType
    # uₙ₋₁mat::umType
    batch_size_hint::Int
    xs::xType
end
Adapt.@adapt_structure ForwardEulerCellSolverCache

# This is the actual solver
@inline function _pointwise_step_inner_kernel!(cell_model::F, i::I, t::T, Δt::T, cache::C) where {F, C <: ForwardEulerCellSolverCache, T <: Real, I <: Integer}
    u_local    = @view cache.uₙmat[i, :]
    du_local   = @view cache.dumat[i, :]
    x          = getcoordinate(cache, i)

    # TODO get Cₘ
    cell_rhs!(du_local, u_local, x, t, cell_model)

    @inbounds for j in 1:length(u_local)
        u_local[j] += Δt*du_local[j]
    end

    return true
end

function setup_solver_cache(f::PointwiseODEFunction, solver::ForwardEulerCellSolver, t₀; u = nothing, uprev=nothing)
    @unpack npoints, ode = f
    ndofs_local = num_states(ode)

    du      = create_system_vector(solver.solution_vector_type, f)
    dumat   = reshape(du, (npoints,ndofs_local))
    uₙ      = u === nothing ? create_system_vector(solver.solution_vector_type, f) : u
    uₙ₋₁    = uₙ
    uₙmat   = reshape(uₙ, (npoints,ndofs_local))
    xs      = f.x === nothing ? nothing : Adapt.adapt(solver.solution_vector_type, f.x)

    return ForwardEulerCellSolverCache(du, uₙ, uₙ₋₁, dumat, uₙmat, solver.batch_size_hint, xs)
end

Base.@kwdef struct AdaptiveForwardEulerSubstepper{T, SolutionVectorType <: AbstractVector{T}} <: AbstractPointwiseSolver
    substeps::Int                                  = 10
    reaction_threshold::T                          = 0.1
    solution_vector_type::Type{SolutionVectorType} = Vector{Float64}
    batch_size_hint::Int                           = 32
end

# Fully accelerator compatible
struct AdaptiveForwardEulerSubstepperCache{T, duType, uType, dumType, umType, xType} <: AbstractPointwiseSolverCache
    du::duType
    # These vectors hold the data
    uₙ::uType
    uₙ₋₁::uType
    # These array view the data above to give easy indices of the form [ode index, local state index]
    dumat::dumType
    uₙmat::umType
    # Solver parameters
    substeps::Int
    reaction_threshold::T
    batch_size_hint::Int
    xs::xType
end
Adapt.@adapt_structure AdaptiveForwardEulerSubstepperCache

@inline function _pointwise_step_inner_kernel!(cell_model::F, i::I, t::T, Δt::T, cache::C) where {F, C <: AdaptiveForwardEulerSubstepperCache, T <: Real, I <: Integer}
    u_local    = @view cache.uₙmat[i, :]
    du_local   = @view cache.dumat[i, :]
    x          = getcoordinate(cache, i)

    φₘidx = transmembranepotential_index(cell_model)

    # TODO get Cₘ
    cell_rhs!(du_local, u_local, x, t, cell_model)

    if abs(du_local[φₘidx]) < cache.reaction_threshold
        for j in 1:length(u_local)
            u_local[j] += Δt*du_local[j]
        end
    else
        Δtₛ = Δt/cache.substeps
        for j in 1:length(u_local)
            u_local[j] += Δtₛ*du_local[j]
        end

        for substep ∈ 2:cache.substeps
            tₛ = t + substep*Δtₛ
            #TODO Cₘ
            cell_rhs!(du_local, u_local, x, t, cell_model)

            for j in 1:length(u_local)
                u_local[j] += Δtₛ*du_local[j]
            end
        end
    end

    return true
end

function setup_solver_cache(f::PointwiseODEFunction, solver::AdaptiveForwardEulerSubstepper, t₀; u = nothing, uprev=nothing)
    @unpack npoints, ode = f
    ndofs_local = num_states(ode)

    du      = create_system_vector(solver.solution_vector_type, f)
    dumat   = reshape(du, (npoints,ndofs_local))
    uₙ      = u === nothing ? create_system_vector(solver.solution_vector_type, f) : u
    uₙ₋₁    = uₙ
    uₙmat   = reshape(uₙ, (npoints,ndofs_local))
    xs      = if f.x === nothing
        nothing
    else
        adapt_vector_type(solver.solution_vector_type, f.x)
    end

    return AdaptiveForwardEulerSubstepperCache(du, uₙ, uₙ₋₁, dumat, uₙmat, solver.substeps, solver.reaction_threshold, solver.batch_size_hint, xs)
end
