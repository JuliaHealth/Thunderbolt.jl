using BenchmarkTools, Thunderbolt, StaticArrays,Ferrite

grid = generate_grid(Hexahedron , (1,1,1))
cell_cache = Ferrite.CellCache(grid)
reinit!(cell_cache,1)

ip_collection = LagrangeCollection{1}()
ip = getinterpolation(ip_collection, grid.cells[1])
qr_collection = QuadratureRuleCollection(2)
qr = getquadraturerule(qr_collection, grid.cells[1])
cv = CellValues(qr, ip)
dh = DofHandler(grid)
add!(dh, :u, getinterpolation(ip_collection, first(grid.cells)))
close!(dh)
sdh = first(dh.subdofhandlers)
ac = AnalyticalCoefficient(
    (x,t) -> norm(x)+t,
    CoordinateSystemCoefficient(
        CartesianCoordinateSystem(grid)
    )
)

coeff_cache = Thunderbolt.setup_coefficient_cache(ac, qr, sdh)

b = zeros(8)
element_cache = Thunderbolt.AnalyticalCoefficientElementCache(coeff_cache, [SVector((0.0,1.0))], cv)
@btime Thunderbolt.assemble_element!($b, $cell_cache, $element_cache, 0.0)
