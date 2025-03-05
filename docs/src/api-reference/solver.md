```@meta
DocTestSetup = :(using Thunderbolt)
```

# Solver

## Linear

```@docs
SchurComplementLinearSolver
```

## Nonlinear

```@docs
NewtonRaphsonSolver
MultiLevelNewtonRaphsonSolver
```


## Time

```@docs
BackwardEulerSolver
ForwardEulerSolver
ForwardEulerCellSolver
AdaptiveForwardEulerSubstepper
LoadDrivenSolver
```

## Operator Splitting Module

```@docs
Thunderbolt.OS.LieTrotterGodunov
Thunderbolt.OS.GenericSplitFunction
Thunderbolt.OS.OperatorSplittingIntegrator
```

## Operator Splitting Adaptivity

```@docs
Thunderbolt.ReactionTangentController
```
