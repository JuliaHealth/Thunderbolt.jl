"""
Descriptor for a tetrahedral element discretization used for solving the Eikonal equation explicitly.
"""
@kwdef struct SimplicialEikonalDiscretization{
    ActivationProtocolT <: TransmembraneStimulationProtocol,
}
    order::Int = 1
    activation_protocol::ActivationProtocolT
    subdomains::Vector{String}
end

"""
Returns the indicies of the nodes belonging to the endocardium, defaulting to 5% of the
transmural coordinate range as endocardium.
"""
function get_nodes(protocol::UniformEndocardialActivationProtocol, mesh)
    cs = protocol.cs
    qrc = NodalQuadratureRuleCollection{}(LagrangeCollection{1}())
    if haskey(mesh.grid.nodesets, "endocardium")
        return collect(mesh.grid.nodesets["endocardium"])
    end
    nodes = Int[]
    for sdh in cs.dh.subdofhandlers
        qr = getquadraturerule(qrc, sdh)
        csc = setup_coefficient_cache(cs, qr, sdh)
        for cell in CellIterator(sdh)
            for qp in QuadratureIterator(qr)
                coords = evaluate_coefficient(csc, cell, qp, 0.0)
                coords.transmural <= 0.05 || continue
                push!(nodes, cell.nodes[qp.i])
            end
        end
    end
    return nodes
end

function get_nodes(protocol::AnalyticalTransmembraneStimulationProtocol{
                <:AnalyticalCoefficient{
                    <:Function,
                    <:CartesianCoordinateSystem
                }},
                mesh)

    f = protocol.f.f
    nodes = Int[]
    for subdomain in mesh.grid.cellsets
        # TODO: use domains defined in discretization instead?
        cellset = last(subdomain)
        for cellidx in cellset
            for node in mesh.grid.cells[cellidx[]].nodes
                stim_val = f(get_node_coordinate(mesh.grid, node), 0.0)
                isnan(stim_val) && continue
                push!(nodes, node)
            end
        end
    end
    return nodes
end

function get_nodes(protocol::AnalyticalTransmembraneStimulationProtocol, mesh)
    cs = protocol.f.coordinate_system_coefficient
    qrc = NodalQuadratureRuleCollection{}(LagrangeCollection{1}())
    f = protocol.f.f
    nodes = Int[]
    for sdh in cs.dh.subdofhandlers
        qr = getquadraturerule(qrc, sdh)
        csc = setup_coefficient_cache(cs, qr, sdh)
        for cell in CellIterator(sdh)
            for qp in QuadratureIterator(qr)
                coord = evaluate_coefficient(csc, cell, qp, 0.0)
                stim_val = f(coord, 0.0)
                isnan(stim_val) && continue
                push!(nodes, cell.nodes[qp.i])
            end
        end
    end
    return nodes
end

function get_nodes(
        ::UniformEndocardialActivationProtocol{<:CartesianCoordinateSystem},
        mesh
)
    throw(error("Uniformally activating the endocardium requires using either
    LV or BiV coordinate system. usage with Cartesian Coordinate System is
    restricted to AnalyticalTransmembraneStimulationProtocol"))
end

function semidiscretize(
        ::EikonalModel,
        discretization::SimplicialEikonalDiscretization{<:UniformEndocardialActivationProtocol},
        mesh::SimpleMesh
)
    activation_points = get_nodes(discretization.activation_protocol, mesh)
    vertices = getproperty.(mesh.grid.nodes, :x)
    activation_points_offsets = fill(NaN, length(vertices))
    cells_ferrite = [Tetrahedron((0, 0, 0, 0)) for _ in 1:length(mesh.grid.cells)]
    if isempty(discretization.subdomains)
        for cellidx in 1:length(mesh.grid.cells)
            for node in mesh.grid.cells[cellidx].nodes
                isnan(activation_points_offsets[node]) || continue
                activation_points_offsets[node] = 0.0 #TODO: Maybe allow this to change?
            end
        end
    else
        for subdomain in discretization.subdomains
            haskey(discretization.activation_protocol.subdomains_offsets, subdomain) || continue
            cellset = getcellset(mesh, subdomain)
            for cellidx in cellset
                for node in mesh.grid.cells[cellidx[]].nodes
                    isnan(activation_points_offsets[node]) || continue
                    activation_points_offsets[node] = discretization.activation_protocol.subdomains_offsets[subdomain]
                end
            end
        end
    end
    cells_ferrite .= mesh.grid.cells

    activation_points_offsets = activation_points_offsets[activation_points]
    cells = getproperty.(cells_ferrite, :nodes)
    nnodes = getnnodes(mesh.grid)

    max_vertices, max_edges, max_faces = Ferrite._max_nentities_per_cell(getcells(mesh.grid))
    vertex_to_cell = (Ferrite.build_vertex_to_cell(mesh.grid.cells; max_vertices, nnodes))
    return EikonalFunction(
        vertices, cells, vertex_to_cell, activation_points, activation_points_offsets)
end


function semidiscretize(
        ::EikonalModel,
        discretization::SimplicialEikonalDiscretization{
            <:AnalyticalTransmembraneStimulationProtocol{
                <:AnalyticalCoefficient{
                    <:Function,
                    <:CartesianCoordinateSystem
                }}},
        mesh::SimpleMesh
) 
    activation_points = get_nodes(discretization.activation_protocol, mesh)
    vertices = getproperty.(mesh.grid.nodes, :x)
    f = discretization.activation_protocol.f.f
    activation_points_offsets = f.(get_node_coordinate.(Ref(mesh.grid), activation_points), 0.0)
    cells_ferrite = [Tetrahedron((0, 0, 0, 0)) for _ in 1:length(mesh.grid.cells)]
    
    cells_ferrite .= mesh.grid.cells

    cells = getproperty.(cells_ferrite, :nodes)
    nnodes = getnnodes(mesh.grid)

    max_vertices, max_edges, max_faces = Ferrite._max_nentities_per_cell(getcells(mesh.grid))
    vertex_to_cell = (Ferrite.build_vertex_to_cell(mesh.grid.cells; max_vertices, nnodes))
    return EikonalFunction(
        vertices, cells, vertex_to_cell, activation_points, activation_points_offsets)
end

function semidiscretize(
        ::EikonalModel,
        discretization::SimplicialEikonalDiscretization{
            <:AnalyticalTransmembraneStimulationProtocol},
        mesh::SimpleMesh
)
    activation_points = get_nodes(discretization.activation_protocol, mesh)
    vertices = getproperty.(mesh.grid.nodes, :x)
    f = discretization.activation_protocol.f.f
    cs = discretization.activation_protocol.f.coordinate_system_coefficient
    qrc = NodalQuadratureRuleCollection{}(LagrangeCollection{1}())
    activation_points_offsets = Float64[]
    activation_points = Int[]
    for sdh in cs.dh.subdofhandlers
        any(subdomain -> first(sdh.cellset) ∈ subdomain, discretization.subdomains) && continue
        qr = getquadraturerule(qrc, sdh)
        csc = setup_coefficient_cache(cs, qr, sdh)
        for cell in CellIterator(sdh)
            for qp in QuadratureIterator(qr)
                coord = evaluate_coefficient(csc, cell, qp, 0.0)
                stim_val = f(coord, 0.0)
                isnan(stim_val) && continue
                push!(activation_points_offsets, stim_val)
                push!(activation_points, cell.nodes[qp.i])
            end
        end
    end
    cells_ferrite = [Tetrahedron((0, 0, 0, 0)) for _ in 1:length(mesh.grid.cells)]
    
    cells_ferrite .= mesh.grid.cells

    cells = getproperty.(cells_ferrite, :nodes)
    nnodes = getnnodes(mesh.grid)

    max_vertices, max_edges, max_faces = Ferrite._max_nentities_per_cell(getcells(mesh.grid))
    vertex_to_cell = (Ferrite.build_vertex_to_cell(mesh.grid.cells; max_vertices, nnodes))
    return EikonalFunction(
        vertices, cells, vertex_to_cell, activation_points, activation_points_offsets)
end
