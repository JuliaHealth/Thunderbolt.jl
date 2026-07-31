"""
    AbstractCoupler

Supertype for descriptions of how two model components are coupled.

!!! note
    A generic coupling graph (`CoupledModel`, `Coupling`, `NullCoupler`, `InterfaceCoupler`,
    `VolumeCoupler` and their accessors) used to live here. It was removed because it never
    executed - its accessors referenced undefined variables and non-existent struct fields, so no
    code path could ever have called them, while it was nevertheless exported and documented as the
    generic multiphysics interface.

    Coupling is being redesigned. Until then the only concrete coupler is
    [`LumpedFluidSolidCoupler`](@ref), and every coupling in the package is bespoke.
"""
abstract type AbstractCoupler end
