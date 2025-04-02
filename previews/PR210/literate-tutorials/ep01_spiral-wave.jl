# # [Electrophysiology Tutorial 1: Simple Spiral Wave](@id ep-tutorial_spiral-wave)
# ![Spiral Wave](spiral-wave.gif)
#
# This tutorial shows how to perform a simulation of electrophysiological behavior of cardiac tissue.
#
# ## Introduction
#
# The most widespread model of cardiac electrophysiology is the monodomain model.
# It can be defined on a domain $\Omega$ as the system of partial differential equations
#
# ```math
# \begin{aligned}
#   \chi C_{\textrm{m}} \partial_t \varphi &= \nabla \cdot \boldsymbol{\kappa} \nabla \varphi - \chi I'(\varphi, \boldsymbol{s}, t) & \textrm{in} \: \Omega \, , \\
#   \partial_t \boldsymbol{s} &= \mathbf{g}(\varphi, \boldsymbol{s}) & \textrm{in}  \: \Omega \, , \\
#   0 &= \boldsymbol{\kappa} \nabla \varphi \cdot \mathbf{n} & \mathrm{on} \: \partial \Omega \, ,
# \end{aligned}
# ```
#
# together with admissible initial conditions and a cellular ionic model to determine $I'$ and $\mathbf{g}$. 
# ${\boldsymbol{\kappa}}$ denotes the conductivity tensor, $\varphi$ is the transmembrane potential field, $\chi$ is the volume to membrane surface ratio, $C_{\mathrm{m}}$ is the membrane capacitance, and $I'(\varphi, \boldsymbol{s}, t) := I_{\textrm{ion}}(\varphi, \boldsymbol{s}) + I_{\textrm{stim}}(t)$ denotes the sum of the ionic current due to the cell model and the applied stimulus current, respectively.
#
# In this tutorial we will apply a reaction-diffusion split to this model and solve it with an operator splitting solver.
# For some theory on operator splitting we refer to the [theory manual on operator splitting](@ref theory_operator-splitting).
#
# ## Commented Program
# We start by loading Thunderbolt and LinearSolve to use a custom direct solver of our choice.
using Thunderbolt, LinearSolve

# We start by constructing a square domain for our simulation.
mesh = generate_mesh(Quadrilateral, (2^6, 2^6), Vec{2}((0.0,0.0)), Vec{2}((2.5,2.5)));
# Here the first parameter is the element type and the second parameter is a tuple holding the number of subdivisions per dimension.
# The last two parameters are the corners defining the rectangular domain.
# !!! tip
#     We can also load realistic geometries with external formats. For this simply use either FerriteGmsh.jl
#     or one of the loader functions stated in the [mesh API](@ref mesh-utility-api).

# We now define the parameters appearing in the model.
# For simplciity we assume $C_{\mathrm{m}} = \chi = 1.0$ and a homogeneous, anisotropic symmetric conductivity tensor.
Cₘ = ConstantCoefficient(1.0)
χ  = ConstantCoefficient(1.0)
κ  = ConstantCoefficient(SymmetricTensor{2,2,Float64}((4.5e-5, 0, 2.0e-5)));
# !!! tip
#     If the mesh is properly annotated, then we can generate (or even load) a cardiac coordinate system.
#     Consult [coordinate system API documentation](@ref coordinate-system-api) for details.
#     With this information we can [construct idealized microstructures](@ref microstructure-api) to define heterogeneous conductivity tensors e.g. as
#     ```julia
#     microstructure = create_simple_microstructure_model(
#       coordinate_system,
#       LagrangeCollection{1}()^3;
#       endo_helix_angle = deg2rad(60.0),
#       epi_helix_angle = deg2rad(-60.0),
#     )
#     κ = SpectralTensorCoefficient(
#         microstructure,
#         ConstantCoefficient(SVector(κ₁, κ₂, κ₃))
#     )
#     ```
#     where κ₁, κ₂, κ₃ are the eigenvalues for the fiber, sheet and normal direction.

# The spiral wave will unfold due to the specific construction of the initial conditions, hence we do not need to apply a stimulus.
stimulation_protocol = NoStimulationProtocol();

# Now we choose a cell model.
# For simplicity we choose a neuronal electrophysiology model, which is a nice playground.
cell_model = Thunderbolt.FHNModel();
# !!! tip
#     A full list of all models can be found in the [API reference](api-reference/models/#Cells).
#     To implement a custom cell model please consult [the how-to section](@ref how-to-custom-ep-cell-model).

# !!! todo
#     The initializer API is not yet finished and hence we deconstruct stuff here manually.
#     Please note that this method is quite fragile w.r.t. to many changes you can make in the code below.
# Spiral wave initializer for the FitzHugh-Nagumo
function spiral_wave_initializer!(u₀, f::GenericSplitFunction)
    ## TODO cleaner implementation. We need to extract this from the types or via dispatch.
    heatfun = f.functions[1]
    heat_dofrange = f.solution_indices[1]
    odefun = f.functions[2]
    ionic_model = odefun.ode

    φ₀ = @view u₀[heat_dofrange];
    ## TODO extraction these via utility functions
    dh = heatfun.dh
    s₀flat = @view u₀[(ndofs(dh)+1):end];
    ## Should not be reshape but some array of arrays fun, because in general (e.g. for heterogeneous tissues) we cannot reshape into a matrix
    s₀ = reshape(s₀flat, (ndofs(dh), Thunderbolt.num_states(ionic_model)-1));

    for cell in CellIterator(dh)
        _celldofs = celldofs(cell)
        φₘ_celldofs = _celldofs[dof_range(dh, :φₘ)]
        ## TODO query coordinate directly from the cell model
        coordinates = getcoordinates(cell)
        for (i, (x₁, x₂)) in zip(φₘ_celldofs,coordinates)
            if x₁ <= 1.25 && x₂ <= 1.25
                φ₀[i] = 1.0
            end
            if x₂ >= 1.25
                s₀[i,1] = 0.1
            end
        end
    end
end;

# Now we put the components together by instantiating the monodomain model.
ep_model = MonodomainModel(
    Cₘ,
    χ,
    κ,
    stimulation_protocol,
    cell_model,
    :φₘ, :s,
);

# We now annotate the model to be reaction-diffusion split.
# Special solvers need special forms for the model.
# However, the same solver can work with different forms.
# In the case of operator splitting users might choose to split the equations differently.
# Hence we leave it as a user option which split they prefer, or if they even want work on the full problem.
split_ep_model = ReactionDiffusionSplit(ep_model);

# !!! todo
#     Show how to use solvers different that LTG (and implement them).

# We now need to transform the space-time problem into a time-dependent problem by discretizing it spatially.
# This can be accomplished by the function semidiscretize, which takes a model and the disretization technique.
# Here we use a finite element discretization in space with first order Lagrange polynomials to discretize the displacement field.
# !!! danger
#     The discretization API does now play well with multiple domains right now and will be updated with a possible breaking change in future releases.
spatial_discretization_method = FiniteElementDiscretization(
    Dict(:φₘ => LagrangeCollection{1}()),
)
odeform = semidiscretize(split_ep_model, spatial_discretization_method, mesh);

# We now allocate a solution vector and set the initial condition.
u₀ = zeros(Float32, OS.function_size(odeform))
spiral_wave_initializer!(u₀, odeform);


# We proceed by defining the time integration algorithms for each subproblem.
# First, there is the heat problem, which we will solve with a low-storage backward Euler method
heat_timestepper = BackwardEulerSolver(
    inner_solver=KrylovJL_CG(atol=1e-6, rtol=1e-5),
);
# !!! tip
#     On non-trivial geometries it is highly recommended to use a preconditioner.
#     Please consult the [LinearSolve.jl docs](https://docs.sciml.ai/LinearSolve/stable/basics/Preconditioners/) for details.

# And then there is the reaction subproblem, which decouples locally into "number of dofs in the discrete heat problem" separate ODE.
# We will solve these locally adaptive with forward Euler steps.
cell_timestepper = AdaptiveForwardEulerSubstepper(;
    reaction_threshold=0.1,
);

# Now we can just instantiate the operator splitting algorithm of our choice.
# Since our time integrators are both first order in time we opt for the standard first order accurrate operator splitting technique by Lie-Trotter (or Godunov).
timestepper = OS.LieTrotterGodunov((heat_timestepper, cell_timestepper));

# The remaining code is very similar to how we use SciML solvers.
# We first define our time domain, initial time step length and some dt for visualization.
dt₀ = 10.0
dtvis = 25.0;
# This speeds up the CI # hide
tspan = (0.0, dtvis);   # hide

# Then we setup the problem.
# We have a split function, so the correct problem is an OperatorSplittingProblem.
problem = OS.OperatorSplittingProblem(odeform, u₀, tspan);

# !!! tip
#     If we want to solve the problem on the GPU, or if we want to use special matrix and vector formats, we just need to adjust the vector and matrix types.
#     For example, if we want to problem to be solved on a CUDA GPU with 32 bit precision, then we need to adjust the types as follows.
#     ```
#     u₀gpu = CuVector(u₀)
#     heat_timestepper = BackwardEulerSolver(
#       solution_vector_type=CuVector{Float32},
#       system_matrix_type=CUDA.CUSPARSE.CuSparseMatrixCSR{Float32, Int32},
#       inner_solver=KrylovJL_CG(atol=1.0f-6, rtol=1.0f-5),
#     )
#     cell_timestepper = AdaptiveForwardEulerSubstepper(
#         solution_vector_type=CuVector{Float32},
#         reaction_threshold=0.1f0,
#     )
#     ...
#     problem = OS.OperatorSplittingProblem(odeform, u₀gpu, tspan)
#     ```

# Now we initialize our time integrator as usual.
integrator = OS.init(problem, timestepper, dt=dt₀);

# !!! todo
#     The post-processing API is not yet finished.
#     Please revisit the tutorial later to see how to post-process the simulation online.
#     Right now the solution is just exported into VTK, such that users can visualize the solution in e.g. ParaView.

# And finally we solve the problem in time.
io = ParaViewWriter("EP01_spiral_wave")
for (u, t) in OS.TimeChoiceIterator(integrator, tspan[1]:dtvis:tspan[2])
    (; dh) = odeform.functions[1]
    φ = u[odeform.solution_indices[1]]
    store_timestep!(io, t, dh.grid) do file
        Thunderbolt.store_timestep_field!(file, t, dh, φ, :φₘ)
    end
end;

# !!! tip
#     If you want to see more details of the solution process launch Julia with Thunderbolt as debug module:
#     ```
#     JULIA_DEBUG=Thunderbolt julia --project --threads=auto my_simulation_runner.jl
#     ```

#md # ## References
#md # ```@bibliography
#md # Pages = ["ep01_spiral_wave.md"]
#md # Canonical = false
#md # ```

#md # ## [Plain program](@id mechanics-tutorial_spiral-wave-plain-program)
#md #
#md # Here follows a version of the program without any comments.
#md # The file is also available here: [`ep01_spiral_wave.jl`](ep01_spiral_wave.jl).
#md #
#md # ```julia
#md # @__CODE__
#md # ```
