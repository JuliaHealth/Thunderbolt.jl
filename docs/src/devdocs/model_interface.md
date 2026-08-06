```@meta
DocTestSetup = :(using Thunderbolt)
```

# Model Interface

This page collects the generic functions a *model* is expected to implement. Until now this contract
existed only implicitly, spread across call sites.

## No common supertype

There is deliberately **no** `AbstractModel` supertype. A model is whatever implements the interface
below. The reason is extensibility: Julia has single inheritance and no retroactive supertyping, so
requiring `<: AbstractModel` would forbid types owned by other packages from acting as models —
which matters because Thunderbolt aims to interoperate with the wider SciML ecosystem.

Where a family of models genuinely needs to be distinguished, use a **capability trait** rather than
an `isa` check against an abstract type. Traits attach to any type, compose along independent axes,
and say what a model *does* rather than what it *is*.

## Core contract

Every model participating in `semidiscretize` implements:

| Function | Returns | Notes |
| :------- | :------ | :---- |
| `semidiscretize(model, discretization, mesh)` | a semidiscrete function | the entry point |
| `get_field_variable_names(model)` | `Tuple` of `Symbol` | field variables the model introduces |
| `get_volumetric_weak_form_names(model)` | `Tuple` of `Symbol` | volumetric weak forms contributed |
| `gather_internal_variable_infos(model)` | `Tuple` of `InternalVariableInfo` | empty tuple if none |

!!! note "Return tuples, not vectors"
    These return **tuples**, including the empty tuple for "none". Consumers must never have to
    branch on the shape of the result, which was previously the case: implementations variously
    returned a `Vector`, a `Tuple`, a bare element, or `nothing`.

## Capability traits

These answer **independent** questions and must not be used as proxies for one another. In
particular, whether a model owns a block of the solution vector is asked with
[`is_coupling_model`](@ref Thunderbolt.is_coupling_model), never inferred from the presence of a
reaction part — a coupling model may perfectly well carry its own reaction dynamics.

```@docs
Thunderbolt.is_coupling_model
Thunderbolt.has_pointwise_reaction_part
Thunderbolt.reaction_model
Thunderbolt.reaction_solution_symbol
```

## Internal variables

Quadrature-point-local state ("internal variables") is declared through:

```@docs
Thunderbolt.gather_internal_variable_infos
Thunderbolt.internal_variable_size
```

together with `default_initial_state!(Q, model)`.

!!! warning "The size is not a constant in general"
    `internal_variable_size` takes a cell and a quadrature point because a model may size its local state
    per point — FE² and computational homogenization solve a nested boundary value problem at every
    quadrature point, and those problems need not have the same number of dofs.

    Every material in the package today is size-constant, and the machinery around them currently
    *assumes* it in four places: `get_number_of_internal_dofs_per_element` sizes a whole subdomain from
    one number, `_qs_split_unknowns` reshapes a cell's block into a rectangular
    `(size_per_qp, nqp)` array, `setup_local_solve_reports` divides to recover `nqp`, and the local
    solver cache allocates one `J`/residual/corrector. `InternalVariableHandler` itself stores one offset
    per *cell*, so a ragged per-point layout is not expressible yet either.

    Supporting a varying size therefore means changing the storage layout, not only this function. The
    argument list is what keeps that door open, and each of the sites above says so where it assumes
    otherwise.

## Ionic cell models

Implement `num_states(::Type{Model})`, `state_symbols(::Type{Model})`,
`default_initial_state(model)` and `cell_rhs!(du, u, x, t, cell_parameters)`, where `u` is the full
local state vector. Models that also provide the reaction/state split implement `reaction_rhs!` and
`state_rhs!`, which an IMEX or Rush–Larsen integrator can exploit.

The state count and the state names are properties of the model *type*, so they dispatch on the type;
instance-level forwarders are provided.

`transmembranepotential_index(model)` is **derived**, not implemented: it is the position of
[`transmembranepotential_symbol`](@ref Thunderbolt.transmembranepotential_symbol) within
[`state_symbols`](@ref Thunderbolt.state_symbols). The potential may therefore sit at any index — as it
does in `ParametrizedAlievPanfilovModel`, which carries it second — and a model whose names and role
symbol disagree fails loudly at setup instead of silently reading the wrong state.

```@docs
Thunderbolt.state_symbols
Thunderbolt.transmembranepotential_symbol
Thunderbolt.transmembranepotential_index
```

See `src/modeling/cells/fhn.jl` for a minimal example.

## Naming quantities in the solution vector

The three interfaces above describe a *model*. What a semidiscrete **function**'s solution vector holds
is answered by [`solution_variables`](@ref Thunderbolt.solution_variables), which every consumer —
initial conditions, post-processing, IO — is built on. A function type implements that one method; the
descriptors it returns carry everything else.

```@docs
Thunderbolt.solution_variables
Thunderbolt.SolutionVariable
Thunderbolt.StatePointSet
```

## Material models

Subtype `AbstractMaterialModel` — this one *is* a genuine supertype, because it carries shared
fallback implementations — and implement `stress_and_tangent(model, F, coefficients, state)`,
`stress_function(...)` for the residual-only path, and
`setup_coefficient_cache(model, qr, sdh)`. Materials carrying internal state additionally implement
the three functions listed under [Internal variables](@ref).

### Kinematics: how the deformation reaches a material

The element does not hand a material a bare `F`. It hands it an `AbstractKinematics`, which is what
the *time scheme* is able to offer at that quadrature point:

| type | carries | offered by |
| :--- | :------ | :--------- |
| `DeformationGradient(F)` | `F` | rate-free schemes, e.g. `HomotopyPathSolver` |
| `DeformationGradientWithRate(F, Ḟ)` | `F` and `Ḟ` | schemes that reconstruct a velocity |

Read them with `deformation_gradient(kinematics)` and `deformation_rate(kinematics)`. Offering more
than a material reads is fine — a rate-independent material accepts `DeformationGradientWithRate` and
ignores the rate. Offering *less* raises an error naming the two ways out, because a material that
needs `Ḟ` cannot silently be given a wrong one.

The seam stops at `material_routine`: `stress_and_tangent`, `stress_function` and `Ψ` keep taking
bare tensors, so automatic differentiation closures still capture leaf values rather than a container.

A material whose stress reads the rate declares `rate_dependence(model) = RateDependent()` and
implements the five-argument `stress_and_tangent(model, F, Ḟ, coefficients, state)` returning
`(P, ∂P∂F, ∂P∂Ḟ)`. It never learns how the rate was formed: the scheme hands the element an
`AffineVelocity`, carrying the slope `∂Ḟ/∂u` and the displacement at which the reconstructed velocity
vanishes — `1/Δt` with the previous solution for backward Euler, `γ/(βΔt)` with a different reference
for Newmark (see [Newmark-β for second order systems](@ref theory_newmark)).

## Naming: three distinct "initial state" concepts

These are easily confused. They are *not* variants of one function:

| Function | Scope |
| :------- | :---- |
| `default_initial_state!(Q, model)` | one evaluation point of a model — a material at a quadrature point, a cell model at a nodal point, a lumped circuit |
| `default_initial_state(ionic_model)` | the same, for cell models that return their state rather than write it |
| `default_initial_condition!(u, f)` | an entire semidiscrete **function** |

The first two are the *model's* business and the third the *function's*: `default_initial_condition!`
walks [`solution_variables`](@ref Thunderbolt.solution_variables) and asks each model for its own
default, so a model author never writes solution-vector indices. `create_initial_condition(f)`
allocates and does this in one step.

!!! note "Write only what you own"
    `default_initial_condition!` may assume `u` is already zeroed and must write only the entries it
    owns. That is what lets it recurse into an operator split without a child clobbering its siblings.

## Assembly-facing protocol

Anything contributing to a residual or matrix additionally implements the `FerriteOperators`
protocol — `setup_element_cache`, `setup_boundary_cache`, `assemble_element!` and
`duplicate_for_device`. `duplicate_for_device` is not optional: shared-memory parallel assembly
gives every worker its own cache, so any cache holding mutable scratch must implement it.
