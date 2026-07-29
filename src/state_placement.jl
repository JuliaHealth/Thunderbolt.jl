"""
    AbstractStatePlacement

Where auxiliary state attached to the mesh lives.

Auxiliary state is any per-point state that is not an ordinary finite element field variable:
ionic states of cell models, sarcomere states, viscoelastic history variables.

The two placements are **not** two storage layouts for one kind of object - they differ in whether
state can be shared between cells:

  - [`QuadraturePointPlacement`](@ref) state is strictly cell-local. No quadrature point belongs to
    two cells, so this state is never shared.
  - [`DofPlacement`](@ref) state sits on dofs, which may be shared between neighbouring cells.

That distinction has real consequences - for parallel assembly, and for whether the state can be
advanced by a purely pointwise update - which is why the objects managing the two
(`InternalVariableHandler` and `PointwiseODEFunction`) are genuinely different rather than two
implementations of the same thing.

Placement is a property of the **spatial** discretization, not of the time discretization. Choosing
it therefore belongs to `semidiscretize`, and it must never be inferred from the time integrator:
the same discretized problem has to be usable under different time schemes.
"""
abstract type AbstractStatePlacement end

"""
    DofPlacement(field::Symbol)
    DofPlacement(fields::Tuple{Vararg{Symbol}})

Entries live on the **dofs** of `fields`, one copy per dof. Dofs may be shared between neighbouring
cells, so entries under this placement may be too.

Note that these are dofs, *not* mesh nodes - the two coincide only for first order Lagrange
interpolations. This is the placement used by the classical operator-split electrophysiology
formulation, where the cell model is solved pointwise per transmembrane potential dof, and by
ionic-current-interpolation (ICI) style schemes. It also describes an ordinary finite element block,
which may carry several fields at once.
"""
struct DofPlacement <: AbstractStatePlacement
    fields::Tuple{Vararg{Symbol}}
end
DofPlacement(field::Symbol) = DofPlacement((field,))

"""
    QuadraturePointPlacement()

Auxiliary state lives at the **quadrature points**, one copy of the state per quadrature point.
Quadrature points belong to exactly one cell, so this state is never shared between cells.

This is the placement used by material models carrying internal variables, and the one required by
state-variable-interpolation (SVI) style electrophysiology schemes, where the ionic model is
evaluated where the integrand is evaluated.
"""
struct QuadraturePointPlacement <: AbstractStatePlacement end
