# Shared test fixtures.
#
# Included by every test file that needs them, rather than being defined in `runtests.jl`'s top-level
# scope, so that each file can be run on its own:
#
#     julia --project=. -e 'using Pkg; Pkg.activate("test-env"); include("test/test_mesh.jl")'
#
# Keep this file small. Helpers used by exactly one test file belong in that file — hoisting them here
# would make every test sandbox pay for them, and anything that adds methods to `Thunderbolt`
# functions (the `Test*Field` coefficient types, `DummyForwardEuler`, …) would then be re-`@eval`'d
# into every process for no reason.

using Thunderbolt
import Thunderbolt: OrderedSet
using Random

"""
Mixed-dimensional 3D grid: one `Hexahedron` ("Ventricle") plus one `Line` ("Purkinje"), with the six
facet sets of the hexahedron named as usual. Used to exercise the mixed-dimensional subdomain paths.
"""
function generate_mixed_dimensional_grid_3D()
    nodes = Node.([
        Vec((-1.0, -1.0, -1.0)),
        Vec((1.0, -1.0, -1.0)),
        Vec((-1.0, 1.0, -1.0)),
        Vec((1.0, 1.0, -1.0)),
        Vec((-1.0, -1.0, 1.0)),
        Vec((1.0, -1.0, 1.0)),
        Vec((-1.0, 1.0, 1.0)),
        Vec((1.0, 1.0, 1.0)),
        Vec((0.0, 0.0, 0.0)),
    ])
    elements = [Hexahedron((1, 2, 4, 3, 5, 6, 8, 7)), Line((8, 9))]
    cellsets = Dict(("Ventricle" => OrderedSet([1]), "Purkinje" => OrderedSet([2])))
    facetsets = Dict((
        "bottom" => OrderedSet([FacetIndex(1, 1)]),
        "front" => OrderedSet([FacetIndex(1, 2)]),
        "right" => OrderedSet([FacetIndex(1, 3)]),
        "back" => OrderedSet([FacetIndex(1, 4)]),
        "left" => OrderedSet([FacetIndex(1, 5)]),
        "top" => OrderedSet([FacetIndex(1, 6)]),
    ))
    return Grid(elements, nodes; cellsets, facetsets)
end

"""
    surface_volumes(grid, facets, coordinates = nothing)

`∫ xᵢ nᵢ dA` over `facets`, once per axis, with `n` the outward normal of the cell each facet
belongs to and `coordinates` an optional stand-in for the grid's node coordinates -- a deformed
configuration, say.

On a closed surface the divergence theorem makes all three components the enclosed volume, signed by
the orientation; on an open one they differ and none of them is a volume. The rule is second order
because the integrand is quadratic on a bilinear facet, and that discrete identity holds only where
the quadrature is exact.
"""
function surface_volumes(grid, facets, coordinates = nothing)
    facetvalues = Dict{DataType, Any}()
    volumes = zeros(3)
    for facet in facets
        cell = getcells(grid, facet[1])
        fv = get!(facetvalues, typeof(cell)) do
            FacetValues(
                FacetQuadratureRule{Ferrite.getrefshape(cell)}(2),
                Ferrite.geometric_interpolation(typeof(cell)),
            )
        end
        x =
            coordinates === nothing ? getcoordinates(grid, facet[1]) :
            [coordinates[nodeid] for nodeid in cell.nodes]
        # Qualified: the SciML stack exports a `reinit!` of its own, and this file is included into
        # test modules that load it.
        Ferrite.reinit!(fv, cell, x, facet[2])
        for qp = 1:getnquadpoints(fv)
            n  = getnormal(fv, qp)
            xq = spatial_coordinate(fv, qp, x)
            dΓ = getdetJdV(fv, qp)
            for d = 1:3
                volumes[d] += xq[d]*n[d]*dΓ
            end
        end
    end
    return volumes
end
