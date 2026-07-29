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
```

together with `internal_variable_size(model, cid, qp)` and `default_initial_state!(Q, model)`.

## Ionic cell models

Implement `num_states(model)`, `transmembranepotential_index(model)`,
`default_initial_state(model)` and `cell_rhs!(du, u, x, t, cell_parameters)`, where `u` is the full
local state vector. Models that also provide the reaction/state split implement `reaction_rhs!` and
`state_rhs!`, which an IMEX or Rush–Larsen integrator can exploit.

See `src/modeling/cells/fhn.jl` for a minimal example.

## Material models

Subtype `AbstractMaterialModel` — this one *is* a genuine supertype, because it carries shared
fallback implementations — and implement `stress_and_tangent(model, F, coefficients, state)`,
`stress_function(...)` for the residual-only path, and
`setup_coefficient_cache(model, qr, sdh)`. Materials carrying internal state additionally implement
the three functions listed under [Internal variables](@ref).

## Naming: three distinct "initial state" concepts

These are easily confused. They are *not* variants of one function:

| Function | Scope |
| :------- | :---- |
| `default_initial_state!(Q, material_model)` | one quadrature point of a material |
| `default_initial_state(ionic_model)` | one point of a pointwise cell model |
| `default_initial_condition!(u, f)` | an entire semidiscrete function |

!!! warning
    `default_initial_state(ionic_model)` currently has no consumers inside the package, although it
    is taught as required API in the how-to guide. Initial conditions for EP problems must still be
    set by hand.

## Assembly-facing protocol

Anything contributing to a residual or matrix additionally implements the `FerriteOperators`
protocol — `setup_element_cache`, `setup_boundary_cache`, `assemble_element!` and
`duplicate_for_device`. `duplicate_for_device` is not optional: shared-memory parallel assembly
gives every worker its own cache, so any cache holding mutable scratch must implement it.
