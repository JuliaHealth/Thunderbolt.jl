using Thunderbolt, Thunderbolt.TimerOutputs, CUDA, UnPack
using LinearSolve, KrylovPreconditioners

Base.@kwdef struct HeterogeneousFHNModel{T, T2} <: Thunderbolt.AbstractIonicModel
    a::T = T(0.1)
    b::T = T(0.5)
    c::T = T(1.0)
    d::T = T(0.0)
    e::T2 = T2((x,t)->0.01)
end;
HeterogeneousFHNModel(::Type{T}, e::F) where {T,F} = HeterogeneousFHNModel{T,F}(0.1,0.5,1.0,0.0,e)

Thunderbolt.transmembranepotential_index(cell_model::HeterogeneousFHNModel) = 1
Thunderbolt.num_states(::HeterogeneousFHNModel) = 2
Thunderbolt.default_initial_state(::HeterogeneousFHNModel) = [0.0, 0.0]

function Thunderbolt.cell_rhs!(du::TD,u::TU,x::TX,t::TT,cell_parameters::TP) where {TD,TU,TX,TT,TP <: HeterogeneousFHNModel}
    @unpack a,b,c,d,e = cell_parameters
    φₘ = u[1]
    s  = u[2]
    du[1] = φₘ*(1-φₘ)*(φₘ-a) - s
    du[2] = e(x,t)*(b*φₘ - c*s - d)
    return nothing
end

function spiral_wave_initializer!(u₀, f::GenericSplitFunction)
    # TODO cleaner implementation. We need to extract this from the types or via dispatch.
    heatfun = f.functions[1]
    heat_dofrange = f.dof_ranges[1]
    odefun = f.functions[2]
    ionic_model = odefun.ode

    φ₀ = @view u₀[heat_dofrange];
    # TODO extraction these via utility functions
    dh = heatfun.dh
    s₀flat = @view u₀[(ndofs(dh)+1):end];
    # Should not be reshape but some array of arrays fun
    s₀ = reshape(s₀flat, (ndofs(dh), Thunderbolt.num_states(ionic_model)-1));

    for cell in CellIterator(dh)
        _celldofs = celldofs(cell)
        φₘ_celldofs = _celldofs[dof_range(dh, :φₘ)]
        # TODO query coordinate directly from the cell model
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
end

tspan = (0.0, 1000.0)
dtvis = 25.0
dt₀ = 1.0

function parabolic_e(x,t)
    clamp(0.01 + 0.1x[1], 0.009, 0.011)
end

cell_model = HeterogeneousFHNModel(
    Float32, parabolic_e
)

model = MonodomainModel(
    ConstantCoefficient(1.0),
    ConstantCoefficient(1.0),
    ConstantCoefficient(SymmetricTensor{2,2,Float32}((4.5e-5, 0, 2.0e-5))),
    NoStimulationProtocol(),
    cell_model,
    :φₘ, :s,
)

mesh = generate_mesh(Quadrilateral, (2^7, 2^7), Vec{2}((0.0,0.0)), Vec{2}((2.5,2.5)))

csc = CoordinateSystemCoefficient(
    CartesianCoordinateSystem(mesh)
)

odeform = semidiscretize(
    ReactionDiffusionSplit(model, csc),
    FiniteElementDiscretization(Dict(:φₘ => LagrangeCollection{1}())),
    mesh
)
u₀ = zeros(Float32, OS.function_size(odeform))
spiral_wave_initializer!(u₀, odeform)

u₀gpu = CuVector(u₀)
problem = OS.OperatorSplittingProblem(odeform, u₀gpu, tspan)

# 
function BJPprec(A,p)
    P = BlockJacobiPreconditioner(A, 1000, CUDABackend())
    KrylovPreconditioners.update!(P, A)
    return (P,I)
end
timestepper = OS.LieTrotterGodunov((
    BackwardEulerSolver(
        solution_vector_type=CuVector{Float32},
        system_matrix_type=CUDA.CUSPARSE.CuSparseMatrixCSR{Float32, Int32},
        inner_solver=LinearSolve.KrylovJL_CG(atol=1.0f-6, rtol=1.0f-5, precs=BJPprec, ldiv=false),
    ),
    AdaptiveForwardEulerSubstepper(
        solution_vector_type=CuVector{Float32},
        reaction_threshold=0.1f0,
    ),
))

integrator = OS.init(problem, timestepper, dt=dt₀, verbose=true)

io = ParaViewWriter("spiral-wave-test")

# TimerOutputs.enable_debug_timings(Thunderbolt)
TimerOutputs.reset_timer!()
for (u, t) in OS.TimeChoiceIterator(integrator, tspan[1]:dtvis:tspan[2])
    dh = odeform.functions[1].dh
    φ = u[odeform.dof_ranges[1]]
    @info t,norm(u)
    # sflat = ....?
    store_timestep!(io, t, dh.grid) do file
        Thunderbolt.store_timestep_field!(file, t, dh, Vector(φ), :φₘ)
        # s = reshape(sflat, (Thunderbolt.num_states(ionic_model),length(φ)))
        # for sidx in 1:Thunderbolt.num_states(ionic_model)
        #    Thunderbolt.store_timestep_field!(io, t, dh, s[sidx,:], state_symbol(ionic_model, sidx))
        # end
    end
end
TimerOutputs.print_timer()
# TimerOutputs.disable_debug_timings(Thunderbolt)
