using BenchmarkTools, Thunderbolt, StaticArrays
cell_cache = Ferrite.CellCache(generate_grid(Hexahedron, (1,1,1)))
reinit!(cell_cache,1)
ref_shape = RefHexahedron
order = 1
ip_collection = LagrangeCollection{order}()
ip = getinterpolation(ip_collection, ref_shape)
qr = QuadratureRule{ref_shape}(2)
cv = CellValues(qr, ip)
ac = AnalyticalCoefficient(
    (x,t) -> norm(x)+t,
    CartesianCoordinateSystemCoefficient(
        getinterpolation(LagrangeCollection{1}()^3)
    )
)
b = zeros(8)
element_cache = Thunderbolt.AnalyticalCoefficientElementCache(ac, [SVector((0.0,1.0))], cv)
@btime Thunderbolt.assemble_element!($b, $cell_cache, $element_cache, 0.0)
