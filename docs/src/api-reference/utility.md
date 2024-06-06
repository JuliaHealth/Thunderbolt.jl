```@meta
DocTestSetup = :(using Thunderbolt)
```

# Utility

## Collections

```@docs
Thunderbolt.InterpolationCollection
getinterpolation
Thunderbolt.ScalarInterpolationCollection
Thunderbolt.VectorInterpolationCollection
Thunderbolt.VectorizedInterpolationCollection
LagrangeCollection
QuadratureRuleCollection
getquadraturerule
CellValueCollection
FacetValueCollection
```

## Iteration

```@docs
QuadraturePoint
QuadratureIterator
```

TODO TimeChoiceIterator https://github.com/termi-official/Thunderbolt.jl/issues/32

## IO

```@docs
ParaViewWriter
JLD2Writer
store_timestep!
store_timestep_celldata!
store_timestep_field!
store_coefficient!
store_green_lagrange!
finalize_timestep!
finalize!
```

## Transfer Operators

```@docs
Thunderbolt.NodalIntergridInterpolation
Thunderbolt.transfer!
```

## Postprocessing


### ECG

```@docs
Thunderbolt.Plonsey1964ECGGaussCache
Thunderbolt.evaluate_ecg
```
