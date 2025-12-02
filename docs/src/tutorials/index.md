# Tutorials

On this page you find an overview of Thunderbolt tutorials.
The tutorials explain and show the indended usage of Thunderbolt to simulate cardiac problems.
Tutorials are sorted by their physical domain and written in a way that users can start with reading the tutorials (in order) of the physics they are interested in, without necessarily going through the tutorials for the other phyiscal domains.

The tutorials all follow roughly the same structure:
 - **Introduction** introduces the problem to be solved and discusses the learning outcomes
   of the tutorial.
 - **Commented program** is the code for solving the problem with explanations and comments.
 - **Plain program** is the raw source code of the program.

When studying the tutorials it is a good idea to obtain a local copy of the code and run it
on your own machine as you read along. Some of the tutorials also include suggestions for
tweaks to the program that you can try out on your own.

!!! danger
    The tutorials are work in progress and not all necessary features are implemented yet.

!!! tip
    Parallel assembly and solvers are automatically enabled when you start julia with multiple threads, e.g. via
    ```repl
    julia --project --threads=auto
    ```

### Cardiac Mechanics

This section explains how cardiac solid mechanics simulations can be carried out, how these simulations can be coupled with blood circuit models and how to add mechanical custom models.

---
#### [Mechanics Tutorial 01: Simple Active Stress](@ref mechanics-tutorial_simple-active-stress)
In this tutorial you will learn how to:
* setup a basic contraction simulation
* use the homotpy path solver (,,Load Stepping'') and choose a custom inner solver
* extract quantities to post-process solutions online and offline
---
#### [Mechanics Tutorial 02: Prestress](@ref mechanics-tutorial_prestress) (TODO)
In this tutorial you will learn how to:
* add residual strains to a model
* add preload to a model
* find the new reference configuration
---
#### [Mechanics Tutorial 03: Coupling A Ciculatory System Model](@ref mechanics-tutorial_3d0dcoupling)
In this tutorial you will learn how to:
* define a blood circuit model
* couple the blood circuit model with a single heart chamber
* visualize pressure-volume loops and the blood circuit solution along the
---
#### [Mechanics Tutorial 04: Pericardial Boundary Condtions](@ref mechanics-tutorial_pericardium) (TODO)
In this tutorial you will learn how to:
* generate a pericardium
* add pericardial boundary condtions to a model
---
#### [Mechanics Tutorial 05: Four Chamber Models](@ref mechanics-tutorial_fourchamber) (TODO)
In this tutorial you will learn how to:
* handle and couple multiple subdomains
* couple the blood circuit model with a multiple heart chambers
---
#### [Mechanics Tutorial 06: Heart Valves](@ref mechanics-tutorial_heartvalves) (TODO)
In this tutorial you will learn how to:
* handle 1D and 2D elements in 3D
* add a 3D fluid model to the heart
* couple a 3D fluid model an external blood circuit model

---
### Cardiac Electrophysiology

This section explains how cardiac electrophysiology simulations can be carried out, how these simulations can be coupled with Purkinje network models, how to extract the ECG and how to add mechanical electrophysiology models.

---
#### [EP Tutorial 01: Spiral Waves with a Monodomain Model](@ref ep-tutorial_spiral-wave)
In this tutorial you will learn how to:
* setup a basic electrophysiology simulations
* set initial conditions
* define a custom stimulation protocols
* run an electrophysiology simulation on a GPU
---
#### [EP Tutorial 02: Activating a Left Ventricle via Purkinje Network](@ref ep-tutorial_purkinje) (TODO)
In this tutorial you will learn how to:
* generate a Purkinje Network
* couple the chamber tissue with the network
---
#### [EP Tutorial 03: ECG with a Bidomain Model](@ref ep-tutorial_purkinje) (TODO)
In this tutorial you will learn how to:
* handle multiple subdomains with different physics
* handle coefficients when facing multiple subdomains
* add ground boundary conditons
* induce Torsade de pointes
* add defibrillation boundary conditions
---
#### [EP Tutorial 04: ECG with a Monodomain Model](@ref ep-tutorial_geselowitz-ecg)
In this tutorial you will learn how to:
* transfer coefficients and solutions between overlapping domains
* compute the ECG form a monodomain model as a postprocessing step
---
#### [EP Tutorial 05: Reaction-Eikonal ECG](@ref ep-tutorial_eikonal) (TODO)
In this tutorial you will learn how to:
* perform ECG simulations with a simplified activation dynamics model
---
#### [EP Tutorial 06: Including Pacemakers](@ref ep-tutorial_pacemaker) (TODO)
In this tutorial you will learn how to:
* Handling Heterogeneous Tissues
