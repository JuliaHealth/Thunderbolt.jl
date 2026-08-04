"""
    AbstractSemidiscreteProblem <: SciMLBase.AbstractDEProblem

Supertype for all problems coming from PDE discretizations.

## Interface

solution_size(::AbstractSemidiscreteProblem)
"""
abstract type AbstractSemidiscreteProblem <: SciMLBase.AbstractDEProblem end
SciMLBase.has_kwargs(::AbstractSemidiscreteProblem) = false
SciMLBase.isinplace(::AbstractSemidiscreteProblem) = true
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

function SciMLBase.build_solution(
    prob::AbstractSemidiscreteProblem,
    alg,
    t,
    u;
    timeseries_errors = length(u) > 2,
    dense = false,
    dense_errors = dense,
    calculate_error = true,
    k = nothing,
    alg_choice = nothing,
    interp = SciMLBase.LinearInterpolation(t, u),
    retcode = SciMLBase.ReturnCode.Default,
    destats = missing,
    stats = nothing,
    kwargs...,
)
    T = eltype(eltype(u))
    N = 2 # Why?

    resid = nothing
    original = nothing

    return SciMLBase.ODESolution{T, N}(
        u,
        nothing,
        nothing,
        t,
        k,
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
        original,
        nothing,
    )

    # xref https://github.com/xtalax/MethodOfLines.jl/blob/bc0bf8c4fcd2376dc5c3df9642806749bc0c1cdd/src/interface/solution/timedep.jl#L11
    # original_sol = u
    # TODO custom AbstractPDETimeSeriesSolution
    # return SciMLBase.PDETimeSeriesSolution{T, N, typeof(u), Nothing, typeof(u)}(u,
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

# Bounded by the concrete quasi-static function rather than by `AbstractSolidMechanicsFunction`: an
# `ElastodynamicsFunction` shares the spatial problem but carries a mass term, and posing it as a
# quasi-static problem would silently drop the inertia.
struct QuasiStaticProblem{fType <: QuasiStaticFunction, uType, tType, pType} <:
       AbstractSemidiscreteProblem
    f::fType
    u0::uType
    tspan::tType
    p::pType
end

QuasiStaticProblem(f::QuasiStaticFunction, tspan::Tuple{<:Real, <:Real}) =
    QuasiStaticProblem(f, zeros(solution_size(f)), tspan, SciMLBase.NullParameters())
QuasiStaticProblem(f::QuasiStaticFunction, u0::AbstractVector, tspan::Tuple{<:Real, <:Real}) =
    QuasiStaticProblem(f, u0, tspan, SciMLBase.NullParameters())

"""
    ElastodynamicsProblem(f::ElastodynamicsFunction, [u0, [v0,]] tspan)

Second order in time structural problem. `u0` has the full solution size (displacement dofs followed
by the condensed internal variables), while `v0` is the initial velocity of the **displacement dofs
only** — the internal variables have no velocity, and the velocity is not a degree of freedom.

The initial acceleration is not an input: it is determined by the balance of momentum at `t₀` and is
computed during solver setup.
"""
struct ElastodynamicsProblem{fType <: ElastodynamicsFunction, uType, vType, tType, pType} <:
       AbstractSemidiscreteProblem
    f::fType
    u0::uType
    v0::vType
    tspan::tType
    p::pType

    function ElastodynamicsProblem(f, u0, v0, tspan, p)
        length(u0) == solution_size(f) || error(
            "Initial displacement has length $(length(u0)), but the solution size is $(solution_size(f)).",
        )
        length(v0) == ndofs(f.dh) || error(
            "Initial velocity has length $(length(v0)), but the displacement field has $(ndofs(f.dh)) dofs. " *
            "The velocity covers the displacement dofs only, not the condensed internal variables.",
        )
        return new{typeof(f), typeof(u0), typeof(v0), typeof(tspan), typeof(p)}(f, u0, v0, tspan, p)
    end
end

ElastodynamicsProblem(f::ElastodynamicsFunction, tspan::Tuple{<:Real, <:Real}) =
    ElastodynamicsProblem(f, zeros(solution_size(f)), zeros(ndofs(f.dh)), tspan)
ElastodynamicsProblem(f::ElastodynamicsFunction, u0::AbstractVector, tspan::Tuple{<:Real, <:Real}) =
    ElastodynamicsProblem(f, u0, zeros(ndofs(f.dh)), tspan)
ElastodynamicsProblem(
    f::ElastodynamicsFunction,
    u0::AbstractVector,
    v0::AbstractVector,
    tspan::Tuple{<:Real, <:Real},
) = ElastodynamicsProblem(f, u0, v0, tspan, SciMLBase.NullParameters())


struct PointwiseODEProblem{fType <: AbstractPointwiseFunction, uType, tType, pType} <:
       AbstractPointwiseProblem
    f::fType
    u0::uType
    tspan::tType
    p::pType
end

PointwiseODEProblem(f::AbstractPointwiseFunction, tspan::Tuple{<:Real, <:Real}) =
    PointwiseODEProblem(f, zeros(solution_size(f)), tspan, SciMLBase.NullParameters())
PointwiseODEProblem(
    f::AbstractPointwiseFunction,
    u0::AbstractVector,
    tspan::Tuple{<:Real, <:Real},
) = PointwiseODEProblem(f, u0, tspan, SciMLBase.NullParameters())


struct ODEProblem{fType <: AbstractSemidiscreteFunction, uType, tType, pType} <:
       AbstractSemidiscreteProblem
    f::fType
    u0::uType
    tspan::tType
    p::pType
end

ODEProblem(f::AbstractSemidiscreteFunction, tspan::Tuple{<:Real, <:Real}) =
    ODEProblem(f, zeros(ndofs(f.dh)), tspan, SciMLBase.NullParameters())
ODEProblem(f::AbstractSemidiscreteFunction, u0::AbstractVector, tspan::Tuple{<:Real, <:Real}) =
    ODEProblem(f, u0, tspan, SciMLBase.NullParameters())
