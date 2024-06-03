"""
    GenericSplitFunction(functions::Tuple, dof_ranges::Tuple)
    GenericSplitFunction(functions::Tuple, dof_ranges::Tuple, syncronizers::Tuple)

This type of function describes a set of connected inner functions in mass-matrix form, as usually found in operator splitting procedures.

TODO "Automatic sync"
     we should be able to get rid of the synchronizer and handle the connection of coefficients and solutions in semidiscretize.
"""
struct GenericSplitFunction{fSetType <: Tuple, idxSetType <: Tuple, sSetType <: Tuple} <: AbstractOperatorSplitFunction
    # The atomic ode functions
    functions::fSetType
    # The ranges for the values in the solution vector.
    dof_ranges::idxSetType
    # Operators to update the ode function parameters
    synchronizers::sSetType
    function GenericSplitFunction(fs::Tuple, drs::Tuple, syncers::Tuple)
        @assert length(fs) == length(drs) == length(syncers)
        new{typeof(fs), typeof(drs), typeof(syncers)}(fs, drs, syncers)
    end
end

function function_size(gsf::GenericSplitFunction)
    # TODO optimize
    alldofs = Set{Int}()
    for dof_range in gsf.dof_ranges
        union!(alldofs, dof_range)
    end
    return length(alldofs)
end

struct NoExternalSynchronization end

GenericSplitFunction(fs::Tuple, drs::Tuple) = GenericSplitFunction(fs, drs, ntuple(_->NoExternalSynchronization(), length(fs)))

@inline get_operator(f::GenericSplitFunction, i::Integer) = f.functions[i]

recursive_null_parameters(f::AbstractOperatorSplitFunction) = @error "Not implemented"
recursive_null_parameters(f::GenericSplitFunction) = ntuple(i->recursive_null_parameters(get_operator(f, i)), length(f.functions));
recursive_null_parameters(f::DiffEqBase.AbstractODEFunction) = DiffEqBase.NullParameters()
