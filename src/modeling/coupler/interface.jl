abstract type AbstractCoupler end

function is_correct_coupler(coupler::AbstractCoupler, i::Int, j::Int)
    @unpack problem_1_index, problem_2_index = coupler
    problem_1_index == i && problem_2_index == j && return true
    if is_bidrectional(coupler)
        problem_2_index == i && problem_1_index == j && return true
    end
    return false
end

function is_relevant_coupler(coupler::AbstractCoupler, i::Int)
    @unpack problem_1_index, problem_2_index = coupler
    if problem_1_index == i || problem_2_index == j
        return true
    end
    return false
end

is_bidrectional(coupler::AbstractCoupler) = false

"""
Indcator that blocks are not coupled.
"""
struct NullCoupler <: AbstractCoupler end

"""
Abstract supertype for all interface coupling schemes.
"""
abstract type InterfaceCoupler <: AbstractCoupler end

"""
Abstract supertype for all volume coupling schemes.
"""
abstract type VolumeCoupler <: AbstractCoupler end

"""
Helper to describe the coupling between problems.
"""
struct Coupling{CouplerType <: AbstractCoupler}
    problem_1_index::Int
    problem_2_index::Int
    coupler::CouplerType
end

"""
A descriptor for a coupled model.
"""
struct CoupledModel{MT <: Tuple, CT <: Tuple}
    base_models::MT
    couplings::CT
end

CoupledModel(base_models::Tuple, coupler::Coupling) = CoupledModel(base_models, (coupler,))
CoupledModel(base_models::AbstractVector, coupler::Coupling) =
    CoupledModel(ntuple(i -> base_models[i], length(base_models)), (coupler,))
CoupledModel(base_models::AbstractVector, couplers::Tuple) =
    CoupledModel(ntuple(i -> base_models[i], length(base_models)), couplers)
CoupledModel(base_models::Tuple, couplers::AbstractVector) =
    CoupledModel(base_models, ntuple(i -> couplers[i], length(couplers)))
CoupledModel(base_models::AbstractVector, couplers::AbstractVector) = CoupledModel(
    ntuple(i -> base_models[i], length(base_models)),
    ntuple(i -> couplers[i], length(couplers)),
)

is_relevant_coupling(coupler::Coupling, i::Int) = is_relevant_coupler(coupler.coupling, i)

function get_coupler(model::CoupledModel, i::Int, j::Int)
    for coupling in model.couplers
        @unpack coupler = coupling
        is_correct_coupler(coupling.coupler, i, j) && return
    end
    return NullCoupler()
end

relevant_couplings(model::CoupledModel, i::Int) =
    [coupling for coupling in model.couplings if is_relevant_coupling(coupling, i)]
