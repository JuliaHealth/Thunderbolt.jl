using BenchmarkTools, Thunderbolt, StaticArrays,Ferrite , CUDA

# The Following is needed to enforce grid to use Float32.
left = Vec((-1.f0, -1.f0,-1.f0)) 
right = Vec((1.f0, 1.f0, 1.f0)) 

grid = generate_grid(Hexahedron , (500,100,100),left,right)

ip_collection = LagrangeCollection{1}()
qr_collection = QuadratureRuleCollection(2)
dh = DofHandler(grid)
add!(dh, :u, getinterpolation(ip_collection, first(grid.cells)))
close!(dh)
cs = CartesianCoordinateSystem(grid)
protocol = AnalyticalTransmembraneStimulationProtocol(
                AnalyticalCoefficient((x,t) -> cos(2π * t) * exp(-norm(x)^2), CoordinateSystemCoefficient(cs)),
                [SVector((0.f0, 1.f0))]
            )



#############################
# CPU operator Benchmarking #
#############################

linop = Thunderbolt.LinearOperator(
    zeros(Float32,ndofs(dh)),
    protocol,
    qr_collection,
    dh,
)

@benchmark Thunderbolt.update_operator!($linop,$0.f0)


#############################
# GPU operator Benchmarking #
#############################

cuda_strategy = Thunderbolt.CudaAssemblyStrategy()
# Notes on launch configuration:
# These values are based on optimal occupancy of my GPU (Nvidia GeForce RTX 3050 Ti w 4GB VRAM) from Nsight Compute.
# The number of threads per block is 384, and the number of blocks (no. SMs) is 20.
cuda_op = Thunderbolt.init_linear_operator(cuda_strategy,protocol, qr_collection, dh;n_threads = 384,n_blocks = 20);

## benchmark with BenchmarkTools
@btime Thunderbolt.update_operator!($cuda_op,$0.f0)

# benchmark with CUDA/Nvidia tools
# Nsight Compute command: ncu --mode=launch julia
# Note: run twice to get the correct time.
Thunderbolt.update_operator!(cuda_op,0.f0) # warm up
CUDA.@profile trace=true Thunderbolt.update_operator!(cuda_op,0.f0)


######################################
# CPU threaded operator Benchmarking #
######################################

plinop = Thunderbolt.PEALinearOperator(
    zeros(Float32,ndofs(dh)),
    qr_collection,
    protocol,
    dh,
);

@benchmark Thunderbolt.update_operator!($plinop,$0.f0)
