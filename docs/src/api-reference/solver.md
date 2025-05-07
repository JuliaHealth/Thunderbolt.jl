```@meta
DocTestSetup = :(using Thunderbolt)
```

# Solver

## Linear

```@docs
SchurComplementLinearSolver
```

## Preconditioners

```@docs
Thunderbolt.Preconditioners.CPUSetting
Thunderbolt.Preconditioners.GPUSetting
Thunderbolt.Preconditioners.BlockPartitioning
Thunderbolt.Preconditioners.L1GSPrecBuilder
Thunderbolt.Preconditioners.L1GSPreconditioner
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
