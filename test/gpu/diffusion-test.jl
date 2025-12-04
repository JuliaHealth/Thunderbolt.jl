using Thunderbolt, CUDA

import Thunderbolt: solution_size, ThreadedSparseMatrixCSR

mesh = generate_mesh(Quadrilateral, (2^7, 2^7), Vec{2}((0.0, 0.0)), Vec{2}((2.5, 2.5)))

model = TransientDiffusionModel(
    ConstantCoefficient(SymmetricTensor{2, 2, Float32}((4.5e-5, 0, 2.0e-5))),
    # ConstantCoefficient(SymmetricTensor{2,2,Float32}((1.0, 0, 1.0))),
    NoStimulationProtocol(),
    :u,
)

odefun =
    semidiscretize(model, FiniteElementDiscretization(Dict(:u => LagrangeCollection{1}())), mesh)

u₀ = rand(Float32, solution_size(odefun))

tspan = (0.0f0, 10.0f0)
dt₀ = 0.1f0
dtvis = 1.0f0

cputimestepper = BackwardEulerSolver(solution_vector_type = Vector{Float32}, system_matrix_type = ThreadedSparseMatrixCSR{Float32, Int32})
cpuproblem     = Thunderbolt.ODEProblem(odefun, u₀, tspan)
cpuintegrator  = init(cpuproblem, cputimestepper, dt = dt₀)

for (u, t) in TimeChoiceIterator(cpuintegrator, tspan[1]:dtvis:tspan[2])
    @info (norm(u), t)
    t > 0.0 && @assert u₀ ≉ u
end

@assert u₀ ≉ cpuintegrator.u


u₀gpu        = CuVector(u₀)
gputimestepper = BackwardEulerSolver(solution_vector_type = CuVector{Float32}, system_matrix_type = CUDA.CUSPARSE.CuSparseMatrixCSC{Float32, Int32})
gpuproblem     = Thunderbolt.ODEProblem(odefun, u₀gpu, tspan)
gpuintegrator  = init(gpuproblem, gputimestepper, dt = dt₀)

for (u, t) in TimeChoiceIterator(gpuintegrator, tspan[1]:dtvis:tspan[2])
    @info (norm(u), t)
    t > 0.0 && @assert u₀ ≉ Vector(u)
end

@assert Vector(gpuintegrator.u) ≈ cpuintegrator.u
