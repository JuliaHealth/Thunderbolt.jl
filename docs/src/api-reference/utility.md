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

Field transfer operators move solution values between compatible Ferrite dof handlers.
The public API currently exposes nodal intergrid interpolation together with the generic
`transfer!` method for applying the constructed transfer operator.

Additionally, convenience RBF operator constructors are provided for building
radial basis function transfer operators with Euclidean or geodesic distance metrics.
The geodesic-distance variant follows the hybrid-distance formulation described in [BucRegDedQua:2024:rrb](@cite).

## Postprocessing


### ECG

```@docs
Thunderbolt.PoissonECGReconstructionCache
Thunderbolt.Plonsey1964ECGGaussCache
Thunderbolt.Geselowitz1989ECGLeadCache
Thunderbolt.evaluate_ecg
```
