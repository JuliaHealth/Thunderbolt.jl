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
ForwardEulerCellSolver
AdaptiveForwardEulerSubstepper
HomotopyPathSolver
```

## Operator Splitting Adaptivity

```@docs
Thunderbolt.ReactionTangentController
```
