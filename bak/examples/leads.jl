using Thunderbolt, StaticArrays, DelimitedFiles

mesh = generate_mesh(Hexahedron, (80,28,12), Vec((0.0,0.0,0.0)), Vec((20.0,7.0,3.0)))
cs = CartesianCoordinateSystem(mesh)

qr_collection = QuadratureRuleCollection(2)
ip_collection = LagrangeCollection{1}()

ms = OrthotropicMicrostructureModel(
    ConstantCoefficient(Vec(0., 0., 1.)),
    ConstantCoefficient(Vec(1., 0., 0.)),
    ConstantCoefficient(Vec(0., 1., 0.))
)

κ = SpectralTensorCoefficient(
    ms, 
    ConstantCoefficient(SVector(0.3, 0.12, 0.12) + SVector(0.3, 0.03, 0.03)))
κᵢ = SpectralTensorCoefficient(
    ms, 
    ConstantCoefficient(SVector(0.3, 0.12, 0.12)))


model = MonodomainModel(
    ConstantCoefficient(1),
    ConstantCoefficient(1),
    κᵢ,
    Thunderbolt.AnalyticalTransmembraneStimulationProtocol(
        AnalyticalCoefficient((x,t) -> norm(x) < 1.5 && t < 2.0 ? 0.5 : 0.0, Thunderbolt.CoordinateSystemCoefficient(cs)),
        [SVector((0.0, 2.1))]
    ),
    Thunderbolt.PCG2019(),
    :φₘ, :s
)

problem = semidiscretize(
    ReactionDiffusionSplit(model),
    FiniteElementDiscretization(Dict(:φₘ => ip_collection)),
    mesh
)

lead_field_cache = Thunderbolt.Geselowitz1989ECGLeadCache(mesh, κ, κᵢ, [Vec(0., 0., 0.) => Vec(20., 0., 0.)])

ground_vertex = Vec(0.0, 0.0, 0.0)

heart_model = TransientDiffusionModel(
    ConstantCoefficient(κ),
    NoStimulationProtocol(), # Poisoning to detecte if we accidentally touch these
    :φₘ
)
heart_fun = semidiscretize(
    heart_model,
    FiniteElementDiscretization(
        Dict(:φₘ => LagrangeCollection{1}()),
        Dirichlet[]
    ),
    mesh
)


io = ParaViewWriter("Lead")

store_timestep!(io, 0.0, lead_field_cache.lead_field_op.dh.grid) do vtk
    store_timestep_field!(io, 0.0, lead_field_cache.lead_field_op.dh, lead_field_cache.Z[1, :], :Z, "Z")
end
