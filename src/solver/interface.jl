abstract type AbstractSolver <: SciMLBase.AbstractDEAlgorithm end
abstract type AbstractNonlinearSolver <: AbstractSolver end

abstract type AbstractNonlinearSolverCache end

abstract type AbstractTimeSolverCache end

# TODO remove
getJ(op) = op.J

# Nonlinear
setup_stage_operator(f::NullFunction, solver::AbstractSolver, local_solver_cache, t₀) =
    NullOperator{Float64, solution_size(f), solution_size(f)}()

# Linear
# An absent stimulus assembles nothing at all, under every strategy: no element loop, no storage,
# and no dependence on what the strategy would have assembled into. One method rather than one per
# strategy shape -- it is strictly more specific than the generic linear method below in both the
# strategy and the integrator, and disjoint from every bilinear method in the integrator, so no
# ambiguity is introduced (`test/test_aqua.jl` checks that).
function setup_operator(
    strategy::AbstractAssemblyStrategy,
    ::LinearIntegrator{<:NoStimulationProtocol},
    solver::AbstractSolver,
    dh::AbstractDofHandler,
)
    LinearNullOperator{value_type(strategy.device), ndofs(dh)}()
end

function setup_operator(
    strategy,
    integrator::AbstractLinearIntegrator,
    solver::AbstractSolver,
    dh::AbstractDofHandler,
)
    return setup_assembled_operator(strategy, integrator, solver.solution_vector_type, dh)
end

"""
    setup_assembled_operator(strategy, integrator::AbstractLinearIntegrator, solution_vector_type, dh)

Vector-side counterpart of the bilinear `setup_assembled_operator`: the default assembles on the host
and returns that operator unchanged; a device-crossing override mirrors its load vector instead.
"""
function setup_assembled_operator(
    strategy,
    integrator::AbstractLinearIntegrator,
    solution_vector_type::Type,
    dh::AbstractDofHandler,
)
    setup_operator(strategy, integrator, dh)
end

# Bilinear
function setup_operator(
    strategy::Union{
        AssemblyStrategy{<:FullAssembly, SequentialScheduling, <:AbstractCPUDevice},
        AssemblyStrategy{<:FullAssembly, <:ColoredScheduling, <:AbstractCPUDevice},
    },
    integrator::AbstractBilinearIntegrator,
    solver::AbstractSolver,
    dh::AbstractDofHandler,
)
    setup_assembled_operator(strategy, integrator, solver.system_matrix_type, dh)
end

function setup_operator(
    strategy::AssemblyStrategy{<:FullAssembly, <:Any, <:AbstractGPUDevice},
    integrator::AbstractBilinearIntegrator,
    solver::AbstractSolver,
    dh::AbstractDofHandler,
)
    setup_assembled_operator(strategy, integrator, solver.system_matrix_type, dh)
end

"""
A matrix-free operator holds no system matrix, so `solver.system_matrix_type` has nothing to name and
is not read: where the assembled forms above pick a storage format, here the strategy's own
`MatrixFreeAction` storage election already fixes what is kept and what is recomputed per action, and
the device the strategy names is the device the action runs on. There is consequently no
`setup_assembled_operator` step and no mirror -- the operator is whatever `FerriteOperators` builds
for this strategy, on both host and device.

The solver still has to want one: an operator of this kind serves `mul!`, not a linear solve, so it
reaches only the solvers that step through the operator's action -- [`EMRKC`](@ref) today.
"""
function setup_operator(
    strategy::AssemblyStrategy{<:MatrixFreeAction},
    integrator::AbstractBilinearIntegrator,
    solver::AbstractSolver,
    dh::AbstractDofHandler,
)
    return setup_operator(strategy, integrator, dh)
end

"""
    setup_assembled_operator(strategy, integrator, system_matrix_type, dh)

Materialize `integrator` against `dh` into the operator a solver requesting `system_matrix_type` can
step with.

The device that assembles and the format the solver wants its system matrix in are independent
choices -- the model carries `strategy`, the solver carries `system_matrix_type` -- and this is where
they meet. Where both live on the host they may still differ in *format*: the affine backward Euler
stage reads the mass and diffusion matrices only through `nonzeros`, and every host allocator emits
the same ordering for one dof handler, so the requested format is simply not needed here. Across a
device boundary that no longer holds, and the extension method returns a
[`MirroredBilinearOperator`](@ref) instead.

Where the strategy names a *device*, there is no mirror: the operator assembles straight into
`system_matrix_type`, so the format is threaded onto the operator specification and the two knobs
have to name the same one. Device assembly needs a CSC device matrix -- Ferrite ships no device
assembler for CSR, so `FerriteOperators` rejects that pairing at setup -- and a CSR system matrix
stays available only through a *host* assembly strategy, which assembles on the host and mirrors
into the device matrix.
"""
function setup_assembled_operator(
    strategy::AssemblyStrategy{<:FullAssembly, SequentialScheduling, <:AbstractCPUDevice},
    integrator::AbstractBilinearIntegrator,
    system_matrix_type::Type,
    dh::AbstractDofHandler,
)
    setup_operator(strategy, integrator, dh)
end

@doc (@doc setup_assembled_operator)
function setup_assembled_operator(
    strategy::AssemblyStrategy{<:FullAssembly, <:Any, <:AbstractGPUDevice},
    integrator::AbstractBilinearIntegrator,
    system_matrix_type::Type,
    dh::AbstractDofHandler,
)
    return setup_operator(_device_assembly_strategy(strategy, system_matrix_type), integrator, dh)
end

# The two knobs meet here. A device assembly writes the entries of `system_matrix_type` itself, so it
# has to be a format Ferrite ships a device assembler for -- CSC, not CSR -- and the operator
# specification is where `FerriteOperators` reads the type from.
function _device_assembly_strategy(
    strategy::AssemblyStrategy{<:FullAssembly, <:Any, <:AbstractGPUDevice},
    system_matrix_type::Type,
)
    # Read through FerriteOperators' own accessor rather than off the field: the accessor is the
    # supported spelling and answers for every form, while the field exists only on the ones that
    # carry global storage.
    spec = FerriteOperators.operator_specification(strategy.form)
    # Anything but the standard specification is rejected for a device by FerriteOperators, with a
    # message naming the limitation. Hand it over untouched rather than rebuilding it into one.
    spec isa StandardOperatorSpecification || return strategy
    # FerriteOperators performs the identical `hasmethod(Ferrite.start_assemble, ...)` capability
    # check at its own setup (setup.jl:328-331, "CSC device matrices only"), so `setup_operator`
    # below already rejects an unassemblable `system_matrix_type` -- a mirror here would outlive an
    # upstream fix. See the docstring above for the CSR/host-mirror hint that check's message lacks.
    declared = spec.matrix_type
    declared === nothing ||
        declared === system_matrix_type ||
        error(
            "The assembly strategy's operator specification names the matrix type $declared while " *
            "the solver asks for $system_matrix_type. The device assembles directly into the " *
            "solver's system matrix, so the two have to name the same type -- drop the one on the " *
            "specification.",
        )
    return AssemblyStrategy(
        FullAssembly(
            StandardOperatorSpecification(;
                algebraic_couplings = spec.algebraic_couplings,
                constraint_handler  = spec.constraint_handler,
                matrix_type         = system_matrix_type,
            ),
        ),
        strategy.scheduling,
        strategy.device,
    )
end

"""
    MirroredBilinearOperator(host_operator, A)

A bilinear operator assembled by `host_operator` on the host, whose matrix is mirrored into `A`.

`update_operator!` runs the host assembly and then refreshes `A`, so everything downstream --
`mul!`, and the `nonzeros` the affine backward Euler stage combines -- sees only `A`. That
correspondence is entrywise, so `A` has to be allocated by the same [`create_system_matrix`](@ref)
call as the stage matrix it is combined into; nothing here can check that beyond the entry count.

The real precondition is stronger than the entry count, and the constructor cannot see the
difference: the mirror copies `nonzeros` positionally, so the two matrices must ALSO agree on the
ORDER their nonzeros are stored in. A host CSR and a device CSC of the same pattern have the same
`nnz` and pass the check; they carry the same order only because `ThreadedSparseMatrixCSR` is the CSR
of the TRANSPOSE and therefore shares the CSC ordering of the original, and because the forms
mirrored this way are symmetric, so that transpose is the same matrix. An ASYMMETRIC form mirrored
across that pairing would be silently transposed. Mirror a CSR host matrix only where both hold.
"""
struct MirroredBilinearOperator{OperatorType, MatrixType, BufferType} <: AbstractBilinearOperator
    host_operator::OperatorType
    A::MatrixType
    # Staging buffer in the mirror's value type. The host assembles in the strategy's value type,
    # which is not in general the solver's, and a cross-device `copyto!` needs the two to match.
    nzbuffer::BufferType
end

function MirroredBilinearOperator(host_operator, A)
    nnz_host   = length(nonzeros(host_operator.A))
    nnz_mirror = length(nonzeros(A))
    nnz_host == nnz_mirror || error(
        "Cannot mirror a $(nnz_host) entry matrix into a $(nnz_mirror) entry one. Both are " *
        "allocated from the same dof handler, so the two allocators disagree about the sparsity " *
        "pattern and the entrywise correspondence the mirror relies on does not exist.",
    )
    return MirroredBilinearOperator(
        host_operator,
        A,
        Vector{eltype(nonzeros(A))}(undef, nnz_mirror),
    )
end

function update_operator!(op::MirroredBilinearOperator, p, ctx = nothing)
    update_operator!(op.host_operator, p, ctx)
    op.nzbuffer .= nonzeros(op.host_operator.A)
    copyto!(nonzeros(op.A), op.nzbuffer)
    return nothing
end

mul!(out::AbstractVector, op::MirroredBilinearOperator, in::AbstractVector) = mul!(out, op.A, in)
mul!(out::AbstractVector, op::MirroredBilinearOperator, in::AbstractVector, α, β) =
    mul!(out, op.A, in, α, β)

# `MirroredBilinearOperator <: AbstractBilinearOperator <: AbstractNonlinearOperator`, whose
# `Base.eltype`/`Base.size` read `FerriteOperators.operator_payload`; without this method they
# `MethodError` instead of answering for the mirrored matrix `A`, which is what everything
# downstream (`mul!`, the stage assembly) actually reads.
FerriteOperators.operator_payload(op::MirroredBilinearOperator) = op.A

"""
    MirroredLinearOperator(host_operator, b)

Vector-side counterpart of [`MirroredBilinearOperator`](@ref): a source operator assembled by
`host_operator` on the host, whose load vector is mirrored into the device buffer `b`.

`update_operator!` runs the host assembly and refreshes `b`; `_add_source_term!` then adds only from
`b` (via `operator_payload`, which for `AbstractLinearOperator` reads the `b` field directly), so a
step where the source does not change costs no upload.
"""
struct MirroredLinearOperator{OperatorType, VectorType, BufferType} <: AbstractLinearOperator
    host_operator::OperatorType
    b::VectorType
    # Host staging buffer in the mirror's value type -- same cross-device eltype mismatch
    # `MirroredBilinearOperator.nzbuffer` stages for, here for the load vector.
    buffer::BufferType
end

function MirroredLinearOperator(host_operator, b)
    n_host = length(FerriteOperators.operator_payload(host_operator))
    n_host == length(b) || error(
        "Cannot mirror a $(n_host) entry vector into a $(length(b)) entry one. Both are allocated " *
        "from the same dof handler, so the two allocators disagree about the vector length.",
    )
    return MirroredLinearOperator(host_operator, b, Vector{eltype(b)}(undef, length(b)))
end

function update_operator!(op::MirroredLinearOperator, p, ctx = nothing)
    update_operator!(op.host_operator, p, ctx)
    op.buffer .= FerriteOperators.operator_payload(op.host_operator)
    copyto!(op.b, op.buffer)
    return nothing
end

needs_update(op::MirroredLinearOperator, t) = needs_update(op.host_operator, t)

"""
    refresh_source_operator!(op, t)

Re-assemble a source operator for the time `t`, where it says it depends on one.

The single owner of that step: every solver carrying a source term does it once per step -- the
affine backward Euler stage before it builds its right hand side, [`EMRKC`](@ref) once per outer
stage -- and a protocol that is stationary in time is skipped rather than reassembled.
"""
refresh_source_operator!(op, t) =
    needs_update(op, t) &&
    update_operator!(op, nothing, TimeIntegrationContext(t, zero(t), zero(t)))

# Nonlinear
"""
    setup_stage_operator(f, solver, local_solver_cache, t₀)

The operator the stage of `solver` solves with, for the function `f`.

Dispatches on the **pair**, because what is assembled and which dof handler it is assembled against
are both properties of the combination: a continuation poses the internal forces alone, backward Euler
poses them with a condensed local problem underneath, and Newmark poses them together with an inertia
term. Each method therefore names its own handler; nothing infers one from `f`.

Distinct from the `setup_operator(strategy, integrator, dh)` family, which materializes one integrator
against one handler and knows nothing about who will solve with it.

`local_solver_cache` is the per-quadrature-point scratch from [`setup_local_solver_cache`](@ref), or
`nothing` where nothing is condensed. `t₀` is the time any operator that has to be assembled once at
setup is assembled at.
"""
function setup_stage_operator end

function update_constraints!(
    f::AbstractSemidiscreteFunction,
    solver_cache::AbstractTimeSolverCache,
    t,
)
    Ferrite.update!(getch(f), t)
    apply!(solver_cache.uₙ, getch(f))
end

update_constraints!(f, solver_cache::AbstractTimeSolverCache, t) = nothing

create_system_matrix(T::Type{<:AbstractMatrix}, f::AbstractSemidiscreteFunction) =
    create_system_matrix(T, f.dh)

# The CSC arrays are handed to the CSR type unchanged, so what this returns is the CSR of `Aᵀ`, and
# every operator assembled into it is assumed symmetric. Deliberate, not incidental: the entrywise
# `nonzeros` correspondence between this matrix and the CSC ones the operators own is what
# `_implicit_euler_heat_solver_update_system_matrix!` combines them through, and a faithful CSC→CSR
# conversion would reorder the entries and break it. The CUDA extension's
# `create_system_matrix(::Type{<:CuSparseMatrixCSR}, dh)` carries the same assumption for the same
# reason; the two have to be changed together.
function create_system_matrix(
    ::Type{<:ThreadedSparseMatrixCSR{Tv, Ti}},
    dh::AbstractDofHandler,
) where {Tv, Ti}
    Acsct = transpose(convert(SparseMatrixCSC{Tv, Ti}, allocate_matrix(dh)))
    return ThreadedSparseMatrixCSR(Acsct)
end

function create_system_matrix(SpMatType::Type{<:SparseMatrixCSC}, dh::AbstractDofHandler)
    A = convert(SpMatType, allocate_matrix(dh))
    return A
end

function create_system_vector(::Type{<:Vector{T}}, f::AbstractSemidiscreteFunction) where {T}
    return zeros(T, solution_size(f))
end

function create_system_vector(::Type{<:Vector{T}}, dh::DofHandler) where {T}
    return zeros(T, ndofs(dh))
end
