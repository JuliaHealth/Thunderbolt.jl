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
evaluate_coefficient
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

## Solid Mechanics

```@docs
QuasiStaticModel
ExtendedHillModel
GeneralizedHillModel
ActiveStressModel
PK1Model
PrestressedMechanicalModel
```

### Passive Energies

```@docs
NullEnergyModel
LinearSpringModel
TransverseIsotopicNeoHookeanModel
HolzapfelOgden2009Model
LinYinPassiveModel
LinYinActiveModel
HumphreyStrumpfYinModel
Guccione1991PassiveModel
Thunderbolt.BioNeoHooekean
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

### Generic Interface

```@docs
Thunderbolt.InterfaceCoupler
Thunderbolt.VolumeCoupler
Coupling
CoupledModel
```

### FSI

```@docs
LumpedFluidSolidCoupler
Hirschvogel2017SurrogateVolume
RSAFDQ2022SurrogateVolume
RSAFDQ2022Split
RSAFDQ2022Model
```
