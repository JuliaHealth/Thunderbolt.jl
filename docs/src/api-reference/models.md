```@meta
DocTestSetup = :(using Thunderbolt)
```

# [Models](@id models-api)

## Coefficient

```@docs
ConstantCoefficient
FieldCoefficient
AnalyticalCoefficient
SpectralTensorCoefficient
SpatiallyHomogeneousDataField
setup_coefficient_cache
evaluate_coefficient
ElastodynamicsModel
```

## [Microstructure](@id microstructure-api)

```@docs
AnisotropicPlanarMicrostructureModel
OrthotropicMicrostructureModel
create_microstructure_model
ODB25LTMicrostructureParameters
```

## Boundary Conditions

```@docs
RobinBC
NormalSpringBC
BendingSpringBC
ConstantPressureBC
PressureFieldBC
```

### Viscous Boundary Conditions

The rate analogues of the Robin family. These resist the velocity rather than the displacement, so they
require a time integrator: the element reads the velocity the scheme reconstructs from the displacement
out of the `:v` slot.

```@docs
ViscousRobinBC
ViscousNormalSpringBC
Thunderbolt.AbstractViscousWeakBoundaryCondition
Thunderbolt.damping_tensor
Thunderbolt.get_time
```

## Kinematic Diagnostics

A converged solve is not necessarily a valid deformation: a folded element and a body drifting along a
free rigid mode both satisfy the residual. These report on the kinematics so that neither passes
unnoticed.

```@docs
Thunderbolt.deformation_gradient_report
Thunderbolt.DeformationGradientReport
Thunderbolt.is_inverted
Thunderbolt.displacement_symbols
DeformationMonitor
```

## Solid Mechanics

```@docs
QuasiStaticModel
ExtendedHillModel
GeneralizedHillModel
ActiveStressModel
PK1Model
PrestressedMechanicalModel
```

The energetic part of a material, for the energy diagnostics:

```@docs
Thunderbolt.strain_energy_density
```

### Passive Energies

Every energy is implemented in the right Cauchy-Green tensor, `Ψᶜ(C, coefficients, model)`, with
`Ψ(F, coefficients, model) = Ψᶜ(tdot(F), coefficients, model)` the form the stress paths call. `C`
is typed `SymmetricTensor{2}` and `F` `Tensor{2}`, so passing one function the other's argument is a
`MethodError`.

```@docs
NullEnergyModel
LinearSpringModel
TransverseIsotopicNeoHookeanModel
HolzapfelOgden2009Model
LinYinPassiveModel
LinYinActiveModel
HumphreyStrumpfYinModel
Guccione1991PassiveModel
BioNeoHookean
```

### Active Energies

```@docs
SimpleActiveSpring
ActiveMaterialAdapter
```

### Active Deformation Gradients

```@docs
GMKActiveDeformationGradientModel
GMKIncompressibleActiveDeformationGradientModel
RLRSQActiveDeformationGradientModel
```

### Active Stresses

```@docs
SimpleActiveStress
PiersantiActiveStress
Guccione1993ActiveModel
```

### Compression

```@docs
NullCompressionPenalty
SimpleCompressionPenalty
HartmannNeffCompressionPenalty1
HartmannNeffCompressionPenalty2
HartmannNeffCompressionPenalty3
```

## Electrophysiology

```@docs
Thunderbolt.TransientDiffusionModel
Thunderbolt.SteadyDiffusionModel
MonodomainModel
Thunderbolt.ParabolicParabolicBidomainModel
Thunderbolt.ParabolicEllipticBidomainModel
ReactionDiffusionSplit
```

```@docs
NoStimulationProtocol
TransmembraneStimulationProtocol
AnalyticalTransmembraneStimulationProtocol
```

## Cells

!!! warning
    These are intended to be replaced by ModelingToolkit analogues!

```@docs
Thunderbolt.ParametrizedFHNModel
Thunderbolt.ParametrizedPCG2019Model
```


## Fluid Mechanics

### Lumped Models

```@docs
Thunderbolt.DummyLumpedCircuitModel
MTKLumpedCicuitModel
RSAFDQ2022LumpedCicuitModel
```

## Multiphysics

!!! warning
    There is no generic coupling interface yet. Every coupling in the package is currently bespoke;
    the only concrete coupler is [`LumpedFluidSolidCoupler`](@ref) below. A generic interface is
    being designed.

```@docs
Thunderbolt.AbstractCoupler
```

### FSI

```@docs
LumpedFluidSolidCoupler
Pressure3D0DVolumeCoupler
ChamberVolumeCoupling
Thunderbolt.volume_integral
RSAFDQ2022Split
RSAFDQ2022Model
chamber_volume
ChamberVolumeFunctional
```
