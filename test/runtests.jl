using Test, Tensors, Thunderbolt, StaticArrays

import Thunderbolt: OrderedSet, to_mesh

using JET: @test_call, @test_opt

function generate_mixed_grid_2D()
    nodes = Node.([
        Vec((-1.0,-1.0)),
        Vec(( 0.0,-1.0)),
        Vec(( 1.0,-1.0)),
        Vec((-1.0, 1.0)),
        Vec(( 0.0, 1.0)),
        Vec(( 1.0, 1.0)),
    ])
    elements = [
        Triangle((1,2,5)),
        Triangle((1,5,4)),
        Quadrilateral((2,3,6,5)),
    ]
    cellsets = Dict((
        "Pacemaker" => OrderedSet([1]),
        "Myocardium" => OrderedSet([2,3])
    ))
    return Grid(elements, nodes; cellsets)
end

function generate_mixed_dimensional_grid_3D()
    nodes = Node.([
        Vec((-1.0, -1.0, -1.0)),
        Vec((1.0, -1.0, -1.0)),
        Vec((1.0, 1.0, -1.0)),
        Vec((-1.0, 1.0, -1.0)),
        Vec((-1.0, -1.0, 1.0)),
        Vec((1.0, -1.0, 1.0)),
        Vec((1.0, 1.0, 1.0)),
        Vec((-1.0, 1.0, 1.0)),
        Vec((0.0,0.0,0.0)),
    ])
    elements = [
        Hexahedron((1,2,3,4,5,6,7,8)),
        Line((8,9)),
    ]
    cellsets = Dict((
        "Ventricle" => OrderedSet([1]),
        "Purkinje" => OrderedSet([2])
    ))
    facetsets = Dict((
        "left" => OrderedSet([FacetIndex(1,1)]),
        "right" => OrderedSet([FacetIndex(1,2)]),
        "top" => OrderedSet([FacetIndex(1,3)]),
        "bottom" => OrderedSet([FacetIndex(1,4)]),
        "front" => OrderedSet([FacetIndex(1,5)]),
        "back" => OrderedSet([FacetIndex(1,6)]),
    ))
    return Grid(elements, nodes; cellsets, facetsets)
end

include("test_elements.jl")
include("test_sarcomere.jl")
include("test_operators.jl")

include("test_solver.jl")

include("test_transfer.jl")

include("test_integrators.jl")

include("test_type_stability.jl")
include("test_mesh.jl")
include("test_coefficients.jl")
include("test_microstructures.jl")

include("integration/test_passive_structure.jl")
include("integration/test_solid_mechanics.jl")
include("integration/test_electrophysiology.jl")
include("integration/test_ecg.jl")

include("test_aqua.jl")
