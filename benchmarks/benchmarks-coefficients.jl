using Thunderbolt, BenchmarkTools, StaticArrays

grid = generate_grid(Line, (2,))

cell_cache = Ferrite.CellCache(grid)
qp1 = QuadraturePoint(1, Vec((0.0,)))
ip_collection = LagrangeCollection{1}()
reinit!(cell_cache, 1)

for val ∈ [1.0, one(Tensor{2,2})]
    cc = ConstantCoefficient(val)
    @btime evaluate_coefficient($cc, $cell_cache, $qp1, 0.0)
end

data_scalar = zeros(2,2)
data_scalar[1,1] =  1.0
data_scalar[1,2] = -1.0
data_scalar[2,1] = -1.0
fcs = FieldCoefficient(data_scalar, ip_collection)
@btime evaluate_coefficient($fcs, $cell_cache, $qp1, 0.0)

data_vector = zeros(Vec{2,Float64},2,2)
data_vector[1,1] = Vec((1.0,0.0))
data_vector[1,2] = Vec((-1.0,-0.0))
data_vector[2,1] = Vec((0.0,-1.0))
fcv = FieldCoefficient(data_vector, ip_collection^2)
@btime evaluate_coefficient($fcv, $cell_cache, $qp1, 0.0)

ccsc = CoordinateSystemCoefficient(CartesianCoordinateSystem(grid))
@btime evaluate_coefficient($ccsc, $cell_cache, $qp1, 0.0)

ac = AnalyticalCoefficient(
    (x,t) -> norm(x)+t,
    ccsc
)
@btime evaluate_coefficient($ac, $cell_cache, $qp1, 0.0)

eigvec = Vec((1.0,0.0))
eigval = -1.0
stc = SpectralTensorCoefficient(
    ConstantCoefficient(SVector((eigvec,))),
    ConstantCoefficient(SVector((eigval,))),
)
st = Tensor{2,2}((-1.0,0.0,0.0,0.0))
@btime evaluate_coefficient($stc, $cell_cache, $qp1, 0.0)

shdc = SpatiallyHomogeneousDataField(
    [1.0, 2.0],
    [Vec((0.1,)), Vec((0.2,)), Vec((0.3,))]
)
@btime evaluate_coefficient($shdc, $cell_cache, $qp1, 0.0)

eigvec = Vec((1.0,0.0))
eigval = -1.0
stc = SpectralTensorCoefficient(
    ConstantCoefficient(SVector((eigvec,))),
    ConstantCoefficient(SVector((eigval,))),
)
st = Tensor{2,2}((-1.0,0.0,0.0,0.0))
ctdc = Thunderbolt.ConductivityToDiffusivityCoefficient(
    stc,
    ConstantCoefficient(2.0),
    ConstantCoefficient(0.5),
)

@btime evaluate_coefficient($stc, $cell_cache, $qp1, 0.0)

struct LineMicrostructureModel{FiberCoefficientType}
    fiber_coefficient::FiberCoefficientType
end
function Thunderbolt.evaluate_coefficient(fsn::LineMicrostructureModel, cell_cache, qp::QuadraturePoint{1}, t)
    f = evaluate_coefficient(fsn.fiber_coefficient, cell_cache, qp, t)

    return SVector((f,))
end

using StaticArrays
data_vector = zeros(Vec{1,Float64},2,2)
data_vector[1,1] = Vec(( 1.0,))
data_vector[1,2] = Vec((-1.0,))
data_vector[2,1] = Vec((-1.0,))
fcv = FieldCoefficient(data_vector, ip_collection^1)
stc = SpectralTensorCoefficient(
    LineMicrostructureModel(FieldCoefficient(data_vector, ip_collection^1)),
    ConstantCoefficient(SVector((1.0,))),
)
integral = Thunderbolt.BilinearDiffusionIntegrator(stc)
qr = QuadratureRule{RefLine}(1)
ip = Lagrange{RefLine,1}()
ip_geo = Lagrange{RefLine,1}()
ec = Thunderbolt.setup_element_cache(integral, qr, ip, ip_geo)
Kₑ = zeros(2,2)
@btime Thunderbolt.assemble_element!($Kₑ, $cell_cache, $ec, 0.0)
