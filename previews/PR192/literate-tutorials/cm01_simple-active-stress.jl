# # [Mechanics Tutorial 1: Simple Contracting Ventricle](@id mechanics-tutorial_simple-active-stress)
# ![Contracting Left Ventricle](contracting-left-ventricle.gif)
#
# This tutorial shows how to perform a simulation for simple active mechanical behavior of heart chambers.
#
# ## Introduction
#
# A general model to simulate the contractile behavior of cardiact issues it the *active stress model*.
# Let us denote with $\Omega_{\mathrm{H}}$ our heart domain and with $u : \Omega_{\mathrm{H}} \to \mathbb{R}^3$ the unknown displacement field in three dimensional space.
# This induces a deformation gradient $\bm{F} = \bm{I} + \nabla \bm{u}$.
# With this formulation we can define a large class of active stress models in the first Piola-Kirchhoff stress with the following form:
#
# $$\bm{P} = \partial_{\bm{F}} \psi_{\mathrm{p}} + \mathcal{N}(\bm{\alpha}) \, \partial_{\bm{F}} \psi_{\mathrm{a}}$$
#
# According to [Cha:1982:mlv](@citet) the additive split of the stress in active and passive parts dates back to unpublished Peskin and has been popularized by [GucWalMcC:1993:mac](@citet).
#
# ## Commented Program
# We start by loading Thunderbolt and LinearSolve to use a custom direct solver of our choice.
using Thunderbolt, LinearSolve

# Our goal is to simulate the contraction of a left ventricle with a very simple active stress formulation.
# Hence in a first step we need to load a suitable mesh.
# Thunderbolt can generate idealized geometries as follows.
mesh = generate_ideal_lv_mesh(11,2,5;
    inner_radius = 0.7,
    outer_radius = 1.0,
    longitudinal_upper = 0.2,
    apex_inner = 1.3,
    apex_outer = 1.5
);
# Here the first 3 parameters control the number of elements in circumferential, radial and longitudinal directions.
# The number of elements is very low, so users have an easy time to play around with it.
# For scientific studies the mesh needs to be finer, such that the simulation converges properly.
# The remaining parameters control the chamber geometry shape itself.

# !!! tip
#     We can also load realistic geometries with external formats. For this simply use either FerriteGmsh.jl
#     or one of the loader functions stated in the [mesh API](@ref mesh-utility-api).

# Next we will define a coordinate system, which helps us to work with cardiac geometries.
# This way we can reuse different methods, like for example fiber generators, across geometries.
coordinate_system = compute_lv_coordinate_system(mesh);

# In this coordinate system we will now create a microstructure with linearly varying helix angle in transmural direction.
# The compute microstructure field will be generated on the function space of piecewise continuous first order Lagrange polynomials.
microstructure = create_simple_microstructure_model(
    coordinate_system,
    LagrangeCollection{1}()^3;
    endo_helix_angle = deg2rad(60.0),
    epi_helix_angle = deg2rad(-60.0),
);

# Now we describe the model which we want to use.
# The models provided by Thunderbolt are designed to be highly modular, so you can quickly swap out individual
# component or compose models with each other.
# For the active stress formulation we need first the active and passive material models.
# For this tutorial we use the models described by Guccione.
passive_material_model = Guccione1991PassiveModel()
active_material_model  = Guccione1993ActiveModel();

# Furthermore we need to describe the calcium field and associate it with the sarcomere model.
# To simplify this tutorial we will use an analytical calcium profile.
# Note that we can also use experimental data or a precomputed calcium profile here, too, by simply changing the function implementation below.
function calcium_profile_function(x::LVCoordinate,t)
    linear_interpolation(t,y1,y2,t1,t2) = y1 + (t-t1) * (y2-y1)/(t2-t1)
    ca_peak(x)                          = 1.0
    if 0 ≤ t ≤ 300.0
        return linear_interpolation(t,        0.0, ca_peak(x),   0.0, 300.0)
    elseif t ≤ 500.0
        return linear_interpolation(t, ca_peak(x),        0.0, 300.0, 500.0)
    else
        return 0.0
    end
end
calcium_field = AnalyticalCoefficient(
    calcium_profile_function,
    CoordinateSystemCoefficient(coordinate_system),
);

# We will use for a very simple sarcomere model which is constant in the calcium concentration.
# Note that a using a sarcomere model which has evoluation equations or rate-dependent terms will require different solvers.
sarcomere_model = ConstantStretchModel(;calcium_field);

# Now we have everything set to describe our active stress model by passing all the model components into it.
active_stress_model = ActiveStressModel(
    passive_material_model,
    active_material_model,
    sarcomere_model,
    microstructure,
);

# Next we define some boundary conditions.
# In order to have a very rough approximation of the effect of the pericardium, we use a Robin boundary condition in normal direction.
weak_boundary_conditions = (NormalSpringBC(1.0, "Epicardium"),)

# We finalize the mechanical model by assigning a symbol to identify the unknown solution field and connect the active stress model with the weak boundary conditions.
mechanical_model = StructuralModel(:displacement, active_stress_model, weak_boundary_conditions)

# !!! tip
#     A full list of all models can be found in the [API reference](https://termi-official.github.io/Thunderbolt.jl/dev/api-reference/models/#Solid-Mechanics).

# We now need to transform the space-time problem into a time-dependent problem by discretizing it spatially.
# This can be accomplished by the function semidiscretize, which takes a model and the disretization technique.
# Here we use a finite element discretization in space with first order Lagrange polynomials to discretize the displacement field.
# !!! danger
#     The discretization API does now play well with multiple domains right now and will be updated with a possible breaking change in future releases.
spatial_discretization_method = FiniteElementDiscretization(
    Dict(:displacement => LagrangeCollection{1}()^3),
)
quasistaticform = semidiscretize(mechanical_model, spatial_discretization_method, mesh);

# The remaining code is very similar to how we use SciML solvers.
# We first define our time domain, initial time step length and some dt for visualization.
dt₀ = 10.0
tspan = (0.0, 500.0)
dtvis = 25.0;
# This speeds up the CI # hide
tspan = (0.0, dtvis);   # hide

# Then we setup the problem.
# Since we have no time dependence in our active stress model the correct problem here is a quasistatic problem.
problem = QuasiStaticProblem(quasistaticform, tspan);

# Next we define the time stepper.
# Since there are no time derivatives appearing in our formulation we have to opt for a homotopy path method, which solve the time depentent problems adaptively.
# As our non-linear solver we choose the standard Newton-Raphson method and a direct solver for the inner linear system.
# For the theory behind homotopy path methods we refer to [the corresponding theory manual on homotopy path methods](@ref theory_homotopy-path-methods)
timestepper = HomotopyPathSolver(
    NewtonRaphsonSolver(
        max_iter=10,
        inner_solver=LinearSolve.UMFPACKFactorization(),
    )
);

# Now we initialize our time integrator as usual.
integrator = init(problem, timestepper, dt=dt₀, verbose=true, adaptive=true, dtmax=25.0);

# !!! todo
#     The post-processing API is not yet finished.
#     Please revisit the tutorial later to see how to post-process the simulation online.
#     Right now the solution is just exported into VTK, such that users can visualize the solution in e.g. ParaView.

# Finally we solve the problem in time.
io = ParaViewWriter("CM01_simple_lv");
for (u, t) in TimeChoiceIterator(integrator, tspan[1]:dtvis:tspan[2])
    @info t
    (; dh) = problem.f
    Thunderbolt.store_timestep!(io, t, dh.grid) do file
    Thunderbolt.store_timestep_field!(io, t, dh, u, :displacement)
    end
end;

# !!! tip
#     If you want to see more details of the solution process launch Julia with Thunderbolt as debug module:
#     ```
#     JULIA_DEBUG=Thunderbolt julia --project --threads=auto my_simulation_runner.jl
#     ```

#md # ## References
#md # ```@bibliography
#md # Pages = ["cm01_simple-active-stress.md"]
#md # Canonical = false
#md # ```

#md # ## [Plain program](@id mechanics-tutorial_simple-active-stress-plain-program)
#md #
#md # Here follows a version of the program without any comments.
#md # The file is also available here: [`cm01_simple-active-stress.jl`](cm01_simple-active-stress.jl).
#md #
#md # ```julia
#md # @__CODE__
#md # ```
