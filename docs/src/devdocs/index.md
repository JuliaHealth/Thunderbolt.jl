# Developer documentation
## Design details

### Models

In Thunderbolt a model essentially describes a set of PDEs, their boundary conditions and their coupling information on a high level.

### Functions

Functions are simply semidiscretizations together with boundary condition and coupling information for the semidiscrete form.

### Problems

A function equipped with a time interval and an initial guess.

### Operators

Operators decouple the function description from their evaluation.

`assemble_element`

Questions:
1. How to deal with quasi-static problems which also need velocities?
2. How to make recursive assemble_element definitions for volume coupled problems?

### Solver

Solvers construct operators from given functions and solve some problem with the function info.

`setup_solver_caches(problem, solver, t₀)` takes the problem and a solver to setup the operators needed during solve.

Questions:
1. How to control which exact operator?
1. When and how to check if the operator is compatible with the solver?

See also [my brain flushing in real time some design decisions](DifferentialEquationsjl-issue.md).
