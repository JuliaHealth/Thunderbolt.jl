using Thunderbolt, CUDA

import Thunderbolt: solution_size, num_states, FHNModel

function uniform_initializer!(u₀, f::PointwiseODEFunction)
    ionic_model = f.ode
    ndofs       = solution_size(f)
    nstates     = num_states(ionic_model)
    u₀mat     = reshape(u₀, (ndofs ÷ nstates, nstates))
    φ₀       = @view u₀mat[:, 1];
    s₀        = @view u₀mat[:, 2:end];

    for i = 1:f.npoints
        φ₀[i] = 1.0
        s₀[i, 1] = 0.1
    end
end

odefun = PointwiseODEFunction(2^8, Thunderbolt.ParametrizedFHNModel{Float32}())

u₀ = zeros(Float32, solution_size(odefun))
uniform_initializer!(u₀, odefun)

tspan = (0.0f0, 10.0f0)
dt₀ = 0.1f0
dtvis = 1.0f0

cputimestepper = ForwardEulerCellSolver(solution_vector_type = Vector{Float32})
cpuproblem     = PointwiseODEProblem(odefun, u₀, tspan)
cpuintegrator  = init(cpuproblem, cputimestepper, dt = dt₀)

for (u, t) in TimeChoiceIterator(cpuintegrator, tspan[1]:dtvis:tspan[2])
    @info (norm(u), t)
    t > 0.0 && @assert u₀ ≉ u
end

@assert u₀ ≉ cpuintegrator.u

uniform_initializer!(u₀, odefun)
u₀gpu        = CuVector(u₀)
gputimestepper = ForwardEulerCellSolver(solution_vector_type = CuVector{Float32}, batch_size_hint = 32)
gpuproblem     = PointwiseODEProblem(odefun, u₀gpu, tspan)
gpuintegrator  = init(gpuproblem, gputimestepper, dt = dt₀)

for (u, t) in TimeChoiceIterator(gpuintegrator, tspan[1]:dtvis:tspan[2])
    @info (norm(u), t)
    t > 0.0 && @assert u₀ ≉ Vector(u)
end

@assert Vector(gpuintegrator.u) ≈ cpuintegrator.u
