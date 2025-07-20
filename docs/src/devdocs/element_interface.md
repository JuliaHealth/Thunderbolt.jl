```@meta
DocTestSetup = :(using Thunderbolt)
```

# Element Interface

## Entry Points

```@docs
Thunderbolt.AbstractVolumetricElementCache
Thunderbolt.AbstractSurfaceElementCache
Thunderbolt.AbstractInterfaceElementCache
Thunderbolt.assemble_element!
Thunderbolt.assemble_facet!
Thunderbolt.assemble_interface!
```


## Common

```@docs
Thunderbolt.AnalyticalCoefficientElementCache
Thunderbolt.SimpleFacetCache
```

## Composite

```@docs
Thunderbolt.CompositeVolumetricElementCache
Thunderbolt.CompositeSurfaceElementCache
Thunderbolt.CompositeInterfaceElementCache
```

## Bilinear

```@docs
Thunderbolt.BilinearMassIntegrator
Thunderbolt.BilinearMassElementCache
Thunderbolt.BilinearDiffusionIntegrator
Thunderbolt.BilinearDiffusionElementCache
```


## Solid Mechanics

### Elements

```@docs
Thunderbolt.StructuralElementCache
```
