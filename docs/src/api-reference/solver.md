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
Thunderbolt.AbstractStageFunction
Thunderbolt.update_stage_linearization!
Thunderbolt.evaluate_stage_residual!
Thunderbolt.condense_stage!
```


## Time

```@docs
BackwardEulerSolver
ForwardEulerCellSolver
AdaptiveForwardEulerSubstepper
HomotopyPathSolver
NewmarkSolver
```

## Super-time-stepping

```@docs
Thunderbolt.AbstractSTSFamily
RKC1
RKL1
RKG1
Thunderbolt.sts_sweep!
Thunderbolt.sts_stage_count
Thunderbolt.sts_stability_boundary
Thunderbolt.ExponentialMultirateSTSAlgorithm
EMRKC
Thunderbolt.rate_sign
Thunderbolt.PassiveChildSolver
gating_symbols
gate_coefficients
gating_indices
```

## Operator Splitting Adaptivity

```@docs
Thunderbolt.ReactionTangentController
```

## Step size control

```@docs
Thunderbolt.PIDController
Thunderbolt.adaptive_order
Thunderbolt.set_error_estimate!
Thunderbolt.velocity
Thunderbolt.acceleration
```

## Energy diagnostics

Both energies are functionals of an operator, reduced over its cells by `FerriteOperators.evaluate_functional`.

```@docs
Thunderbolt.strain_energy
Thunderbolt.kinetic_energy
Thunderbolt.StrainEnergyFunctional
Thunderbolt.KineticEnergyFunctional
```
