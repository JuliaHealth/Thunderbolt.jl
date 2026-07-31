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
