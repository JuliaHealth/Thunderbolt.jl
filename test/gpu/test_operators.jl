# The stimulus linear form, assembled on the device against the same form assembled on the host.
# `AnalyticalCoefficientElementCache` is the one element cache on the electrophysiology path that
# maps the quadrature points itself instead of `reinit!`ing its values object, so it exercises the
# other half of the device element contract: a cache whose values object is shared read-only.

@testset "Linear operator, host versus device assembly" begin
    left  = Vec((-1.0f0, -1.0f0))
    right = Vec((1.0f0, 1.0f0))
    grid  = generate_grid(Quadrilateral, (287, 1), left, right)
    dh    = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral, 1}())
    close!(dh)
    qrc = QuadratureRuleCollection(Float32, 2)
    cs  = CartesianCoordinateSystem(grid)

    linint = Thunderbolt.LinearIntegrator(
        AnalyticalTransmembraneStimulationProtocol(
            AnalyticalCoefficient((x, t) -> cos(2π * t) * exp(-norm(x)^2), cs),
            [SVector((0.0f0, 1.0f0))],
        ),
        qrc,
    )
    ctx = TimeIntegrationContext(0.0f0, 0.0f0, 0.0f0)

    host = Thunderbolt.setup_operator(host_assembly_strategy(), linint, dh)
    Thunderbolt.update_operator!(host, nothing, ctx)

    device = Thunderbolt.setup_operator(device_assembly_strategy(), linint, dh)
    # A linear operator's vector is allocated by the device, so no matrix type is involved.
    @test device.b isa CuVector{Float32}
    Thunderbolt.update_operator!(device, nothing, ctx)

    # Both arms sum the same Float32 quadrature contributions; what separates them is the order the
    # colored device sweep accumulates a shared dof in, measured here below 1e-7 relative.
    @test Array(device.b) ≈ host.b rtol = 1.0f-5

    # Coloring fixes that order per entry, so a repeated sweep reproduces the previous one exactly.
    first_run = Array(device.b)
    Thunderbolt.update_operator!(device, nothing, ctx)
    @test first_run == Array(device.b)

    # A device-assembled source's payload is already a `CuVector`: adding it into a device right
    # hand side is a direct axpy, with no per-step upload.
    b = CUDA.zeros(Float32, length(device.b))
    Thunderbolt._add_source_term!(b, device, 0.1f0) # warmup
    @test CUDA.@allocated(Thunderbolt._add_source_term!(b, device, 0.1f0)) == 0
end
