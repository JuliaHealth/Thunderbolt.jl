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
Thunderbolt.Preconditioners.L1GSPrecBuilder
Thunderbolt.Preconditioners.L1GSPreconditioner
Thunderbolt.Preconditioners.ForwardSweep
Thunderbolt.Preconditioners.BackwardSweep
Thunderbolt.Preconditioners.SymmetricSweep
Thunderbolt.Preconditioners.PackedBufferCache
Thunderbolt.Preconditioners.MatrixViewCache
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
