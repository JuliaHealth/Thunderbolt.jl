@testset "Operator API" begin
    left  = Vec((-1.0f0, -1.0f0)) # define the left bottom corner of the grid.
    right = Vec((1.0f0, 1.0f0)) # define the right top corner of the grid.
    grid  = generate_grid(Quadrilateral, (287, 1), left, right)
    dh    = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral, 1}())
    close!(dh)
    qrc = QuadratureRuleCollection{2}()

    cs = CartesianCoordinateSystem(grid)

    linint = Thunderbolt.LinearIntegrator(
        AnalyticalTransmembraneStimulationProtocol(
            AnalyticalCoefficient((x, t) -> cos(2π * t) * exp(-norm(x)^2), cs),
            [SVector((0.0f0, 1.0f0))],
        ),
        qrc,
    )

    cpustrategy =
        Thunderbolt.SequentialAssemblyStrategy(Thunderbolt.SequentialCPUDevice{Float32, Int32}())
    linop = Thunderbolt.setup_operator(cpustrategy, linint, BackwardEulerSolver(), dh)
    Thunderbolt.update_operator!(linop, 0.0f0)

    gpustrategy = Thunderbolt.ElementAssemblyStrategy(Thunderbolt.CudaDevice())
    cuda_op = Thunderbolt.setup_operator(gpustrategy, linint, BackwardEulerSolver(), dh)

    Thunderbolt.update_operator!(cuda_op, 0.0f0)

    @test Vector(cuda_op.b) ≈ linop.b
end
