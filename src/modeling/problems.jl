"""
    AbstractSemidiscreteProblem <: DiffEqBase.AbstractDEProblem

Supertype for all problems coming from PDE discretizations.

## Interface

solution_size(::AbstractSemidiscreteProblem)
"""
abstract type AbstractSemidiscreteProblem <: DiffEqBase.AbstractDEProblem end
DiffEqBase.has_kwargs(::AbstractSemidiscreteProblem) = false
DiffEqBase.isinplace(::AbstractSemidiscreteProblem) = true
solution_size(prob::AbstractSemidiscreteProblem) = solution_size(prob.f)


abstract type AbstractPointwiseProblem <: AbstractSemidiscreteProblem end

# """
#     AbstractSemidiscreteDAEProblem <: AbstractSemidiscreteProblem

# Supertype for discretizations of time-dependent PDEs in residual form.
# """
# abstract type AbstractSemidiscreteDAEProblem <: AbstractSemidiscreteProblem end


# """
#     AbstractSemidiscreteODEProblem <: AbstractSemidiscreteProblem

# Supertype for discretizations of time-dependent PDEs in mass matrix form.
# """
# abstract type AbstractSemidiscreteODEProblem <: AbstractSemidiscreteProblem end


# """
#     AbstractSemidiscreteNonlinearProblem <: AbstractSemidiscreteProblem

# Supertype for discretizations of time-dependent PDEs without explicit time derivatives.
# """
# abstract type AbstractSemidiscreteNonlinearProblem <: AbstractSemidiscreteProblem end

function DiffEqBase.build_solution(prob::AbstractSemidiscreteProblem,
        alg, t, u; timeseries_errors = length(u) > 2,
        dense = false, dense_errors = dense,
        calculate_error = true,
        k = nothing,
        alg_choice = nothing,
        interp = DiffEqBase.LinearInterpolation(t, u),
        retcode = DiffEqBase.ReturnCode.Default, destats = missing, stats = nothing,
        kwargs...)
    T = eltype(eltype(u))
    N = 2 # Why?

    resid = nothing
    original = nothing

    return DiffEqBase.SciMLBase.ODESolution{T, N}(u,
        nothing,
        nothing,
        t, k,
        nothing,
        prob,
        alg,
        interp,
        dense,
        0,
        stats,
        alg_choice,
        retcode,
        resid,
        original)

    # xref https://github.com/xtalax/MethodOfLines.jl/blob/bc0bf8c4fcd2376dc5c3df9642806749bc0c1cdd/src/interface/solution/timedep.jl#L11
    # original_sol = u
    # TODO custom AbstractPDETimeSeriesSolution
    # return DiffEqBase.SciMLBase.PDETimeSeriesSolution{T, N, typeof(u), Nothing, typeof(u)}(u,
    #     original_sol,
    #     nothing,
    #     t, k,
    #     nothing, # ivdomain
    #     nothing, # ivs
    #     nothing, # dvs
    #     nothing, # disc_data
    #     prob,
    #     alg,
    #     interp,
    #     dense,
    #     0,
    #     stats,
    #     retcode,
    #     stats)
end

# abstract type AbstractCoupledProblem <: AbstractSemidiscreteProblem end

# """
#     CoupledProblem{MT, CT}

# Generic description of a coupled problem.
# """
# struct CoupledProblem{MT <: Tuple, CT <: Tuple} <: AbstractCoupledProblem
#     base_problems::MT
#     couplings::CT
# end

# base_problems(problem::CoupledProblem) = problem.base_problems

# solution_size(problem::AbstractCoupledProblem) = sum([solution_size(p) for p ∈ base_problems(problem)])

# function get_coupler(problem::CoupledProblem, i::Int, j::Int)
#     for coupling in problem.couplers
#         @unpack coupler = coupling
#         is_correct_coupler(coupling.coupler, i, j) && return
#     end
#     return NullCoupler()
# end

# relevant_couplings(problem::CoupledProblem, i::Int) = [coupling for coupling in problem.couplings if is_relevant_coupling(coupling)]

struct QuasiStaticProblem{fType <: AbstractQuasiStaticFunction, uType, tType, pType} <: AbstractSemidiscreteProblem
    f::fType
    u0::uType
    tspan::tType
    p::pType
end

QuasiStaticProblem(f::AbstractQuasiStaticFunction, tspan::Tuple{<:Real, <:Real}) = QuasiStaticProblem(f, zeros(solution_size(f)), tspan, DiffEqBase.NullParameters())
QuasiStaticProblem(f::AbstractQuasiStaticFunction, u0::AbstractVector, tspan::Tuple{<:Real, <:Real}) = QuasiStaticProblem(f, u0, tspan, DiffEqBase.NullParameters())


struct PointwiseODEProblem{fType <: AbstractPointwiseFunction, uType, tType, pType} <: AbstractPointwiseProblem
    f::fType
    u0::uType
    tspan::tType
    p::pType
end

PointwiseODEProblem(f::AbstractPointwiseFunction, tspan::Tuple{<:Real, <:Real}) = PointwiseODEProblem(f, zeros(solution_size(f)), tspan, DiffEqBase.NullParameters())
PointwiseODEProblem(f::AbstractPointwiseFunction, u0::AbstractVector, tspan::Tuple{<:Real, <:Real}) = PointwiseODEProblem(f, u0, tspan, DiffEqBase.NullParameters())


struct ODEProblem{fType <: AbstractSemidiscreteFunction, uType, tType, pType} <: AbstractSemidiscreteProblem
    f::fType
    u0::uType
    tspan::tType
    p::pType
end

ODEProblem(f::AbstractSemidiscreteFunction, tspan::Tuple{<:Real, <:Real})  = ODEProblem(f, zeros(ndofs(f.dh)), tspan, DiffEqBase.NullParameters())
ODEProblem(f::AbstractSemidiscreteFunction, u0::AbstractVector, tspan::Tuple{<:Real, <:Real})  = ODEProblem(f, u0, tspan, DiffEqBase.NullParameters())
