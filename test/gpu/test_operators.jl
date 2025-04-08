@testset "Operator API" begin
    left  = Vec((-1.f0, -1.f0)) # define the left bottom corner of the grid.
    right = Vec((1.f0, 1.f0)) # define the right top corner of the grid.
    grid = generate_grid(Quadrilateral, (287,1),left,right)
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral,1}())
    close!(dh)
    qrc = QuadratureRuleCollection{2}()

    cs = CartesianCoordinateSystem(grid)

    protocol = AnalyticalTransmembraneStimulationProtocol(
                    AnalyticalCoefficient((x,t) -> cos(2π * t) * exp(-norm(x)^2), cs),
                    [SVector((0.f0, 1.f0))]
                )

    
    cpustrategy = Thunderbolt.SequentialAssemblyStrategy(Thunderbolt.SequentialCPUDevice{Float32,Int32}())
    linop = Thunderbolt.setup_operator(
        cpustrategy,
        protocol,
        BackwardEulerSolver(),
        dh,
        qrc,
    )
    Thunderbolt.update_operator!(linop,0.f0)

    gpustrategy = Thunderbolt.ElementAssemblyStrategy(Thunderbolt.CudaDevice())
    cuda_op = Thunderbolt.setup_operator(
        gpustrategy,
        protocol,
        BackwardEulerSolver(),
        dh,
        qrc,
    )

    Thunderbolt.update_operator!(cuda_op,0.f0)

    @test Vector(cuda_op.b) ≈ linop.b
end
