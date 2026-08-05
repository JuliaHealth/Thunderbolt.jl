"""
    AbstractSolutionVectorMapping

Bidirectional wiring between two solution vectors laid out by different dof handlers and different
internal variable handlers.

Beyond the stage solves below this is what a domain decomposition method (overlapping Schwarz) or a
nonlinear preconditioner needs in order to move between a global problem and a local one, which is
why it is named after what it does rather than after its first consumer.

## Interface

    gather!(target, source, mapping)   # source numbering -> target numbering
    scatter!(source, target, mapping)  # target numbering -> source numbering
"""
abstract type AbstractSolutionVectorMapping end

"""
    IdentitySolutionVectorMapping()

The mapping between a numbering and itself. Both directions are no-ops, and a target that aliases the
source is left untouched rather than copied onto itself.
"""
struct IdentitySolutionVectorMapping <: AbstractSolutionVectorMapping end

function gather!(target::AbstractVector, source::AbstractVector, ::IdentitySolutionVectorMapping)
    target === source && return target
    copyto!(target, source)
    return target
end

function scatter!(source::AbstractVector, target::AbstractVector, ::IdentitySolutionVectorMapping)
    source === target && return source
    copyto!(source, target)
    return source
end

"""
    SolutionVectorMapping(dofs, internal_variables)

`dofs[i]` is the source index feeding the target's `i`-th finite element dof, and
`internal_variables[i]` the source index feeding its `i`-th condensed internal variable.

The target's own layout is `[dofs | internal_variables]` because that is the layout every vector a
dof handler assembles into already has -- the condensed unknowns are appended after `ndofs` -- so
this is the target handler's invariant rather than an assumption about the source.
"""
struct SolutionVectorMapping{DofMapType, IVMapType} <: AbstractSolutionVectorMapping
    dofs::DofMapType
    internal_variables::IVMapType
end

SolutionVectorMapping(dofs) = SolutionVectorMapping(dofs, Int[])

function gather!(target::AbstractVector, source::AbstractVector, m::SolutionVectorMapping)
    ndofs = length(m.dofs)
    @inbounds for i ∈ eachindex(m.dofs)
        target[i] = source[m.dofs[i]]
    end
    @inbounds for i ∈ eachindex(m.internal_variables)
        target[ndofs+i] = source[m.internal_variables[i]]
    end
    return target
end

function scatter!(source::AbstractVector, target::AbstractVector, m::SolutionVectorMapping)
    ndofs = length(m.dofs)
    @inbounds for i ∈ eachindex(m.dofs)
        source[m.dofs[i]] = target[i]
    end
    @inbounds for i ∈ eachindex(m.internal_variables)
        source[m.internal_variables[i]] = target[ndofs+i]
    end
    return source
end

Base.length(m::SolutionVectorMapping) = length(m.dofs) + length(m.internal_variables)

"""
    AbstractStageFunction

The nonlinear problem one *stage* of a time integration scheme poses.

A stage has its own unknowns, which are neither required to be a subset of the semidiscrete
function's unknowns nor to correspond to any part of the solution vector. It knows how to build them
from the current state and how to write the state back once they are solved for. That is the whole
content of the abstraction, and it is what makes Newmark (which condenses the velocity), an IMEX
split (which condenses the explicitly advanced block) and backward Euler (which condenses nothing)
the same object -- and what leaves room for FIRK, whose stage is `s` times larger than the state.

## Interface

    getoperator(sf)          # the nonlinear operator, *including* the terms the scheme adds
    getfunction(sf)          # the semidiscrete function the stage was built from
    stage_mapping(sf)        # bidirectional wiring to the function's numbering
    stage_parameters(sf)     # the element facing `p` of the current step
    stage_size(sf)           # length of the stage unknown vector
    init_stage!(z, sf, u)    # predictor: current state -> stage unknowns
    update_state!(u, sf, z)  # converged stage unknowns -> state, reconstructing what was condensed

The nonlinear solver sees only this. It never learns what time it is -- everything the operator needs
travels in `stage_parameters` -- which is what lets one solver serve a continuation, a time step and,
later, a coupled multi-stage solve.
"""
abstract type AbstractStageFunction end

stage_mapping(::AbstractStageFunction) = IdentitySolutionVectorMapping()
stage_size(sf::AbstractStageFunction) = solution_size(getfunction(sf))
init_stage!(z::AbstractVector, sf::AbstractStageFunction, u::AbstractVector) =
    gather!(z, u, stage_mapping(sf))
update_state!(u::AbstractVector, sf::AbstractStageFunction, z::AbstractVector) =
    scatter!(u, z, stage_mapping(sf))

# The constraints, the residual norm and the monitors are properties of the *function*; a stage whose
# unknowns carry no constraints of their own forwards rather than reimplementing. Override on a stage
# that constrains its own unknowns differently.
getch(sf::AbstractStageFunction) = getch(getfunction(sf))
residual_norm(cache::AbstractNonlinearSolverCache, sf::AbstractStageFunction) =
    residual_norm(cache, getfunction(sf))
eliminate_constraints_from_linearization!(
    cache::AbstractNonlinearSolverCache,
    sf::AbstractStageFunction,
) = eliminate_constraints_from_linearization!(cache, getoperator(sf), getfunction(sf))
eliminate_constraints_from_residual!(
    cache::AbstractNonlinearSolverCache,
    sf::AbstractStageFunction,
) = eliminate_constraints_from_residual!(cache, getfunction(sf))
eliminate_constraints_from_increment!(
    Δu::AbstractVector,
    sf::AbstractStageFunction,
    cache::AbstractNonlinearSolverCache,
) = eliminate_constraints_from_increment!(Δu, getfunction(sf), cache)

"""
    FullStateStage(f, op, p)

The stage whose unknowns *are* the function's unknowns: nothing is condensed and nothing is
reconstructed.

Backward Euler and [`HomotopyPathSolver`](@ref) both pose this, and it is what a scheme uses whenever
the quantity it solves for is the whole state. The mapping is the identity and the stage vector
aliases the state, so neither transfer hook copies anything.

`p` is rewritten once per step by the scheme. It is a plain field rather than an argument threaded
through the solver because the operator, not the solver, is what consumes it.
"""
mutable struct FullStateStage{FType, OpType, PType} <: AbstractStageFunction
    const f::FType
    const op::OpType
    p::PType
end

getoperator(sf::FullStateStage) = sf.op
getfunction(sf::FullStateStage) = sf.f
stage_parameters(sf::FullStateStage) = sf.p

"""
    set_stage_parameters!(sf, p)

Hand the stage the element facing parameters of the step about to be solved.

This is the counterpart of upstream `OrdinaryDiffEq` writing `γ`, `c` and `tmp` onto its `NLSolver`
before calling it: the map from stage unknowns back to state, and everything the operator needs to
evaluate, belongs to the stage rather than to the step function.
"""
function set_stage_parameters!(sf::FullStateStage, p)
    sf.p = p
    return sf
end
