"""
    CartesianCoordinateSystem(mesh)

Standard cartesian coordinate system.
"""
struct CartesianCoordinateSystem{sdim}
end

value_type(::CartesianCoordinateSystem{sdim}) where sdim = Vec{sdim, Float32}

CartesianCoordinateSystem(mesh::AbstractGrid{sdim}) where sdim = CartesianCoordinateSystem{sdim}()

"""
    getcoordinateinterpolation(cs::CartesianCoordinateSystem, cell::AbstractCell)

Get interpolation function for the cartesian coordinate system.
"""
getcoordinateinterpolation(cs::CartesianCoordinateSystem{sdim}, cell::CellType) where {sdim, CellType <: AbstractCell} = Ferrite.geometric_interpolation(CellType)^sdim


"""
    LVCoordinateSystem(dh, u_transmural, u_apicobasal)

Simplified universal ventricular coordinate on LV only, containing the transmural, apicobasal and
rotational coordinates. See [`compute_lv_coordinate_system`](@ref) to construct it.
"""
struct LVCoordinateSystem{DH <: AbstractDofHandler, IPC}
    dh::DH
    ip_collection::IPC # TODO special dof handler with type stable interpolation
    u_transmural::Vector{Float64}
    u_apicobasal::Vector{Float64}
    u_rotational::Vector{Float64}
end


"""
    LVCoordinate{T}

LV only part of the universal ventricular coordinate, containing
    * transmural
    * apicobasal
    * rotational
"""
struct LVCoordinate{T}
    transmural::T
    apicaobasal::T
    rotational::T
end

value_type(::LVCoordinateSystem) = LVCoordinate{Float32}


"""
    getcoordinateinterpolation(cs::LVCoordinateSystem, cell::AbstractCell)

Get interpolation function for the LV coordinate system.
"""
getcoordinateinterpolation(cs::LVCoordinateSystem, cell::AbstractCell) = getinterpolation(cs.ip_collection, cell)


"""
    compute_lv_coordinate_system(mesh::SimpleMesh)

Requires a mesh with facetsets
    * Base
    * Epicardium
    * Endocardium
and a nodeset
    * Apex
"""
function compute_lv_coordinate_system(mesh::SimpleMesh{3}, subdomains::Vector{String} = [""]; up = Vec((0.0,0.0,1.0)))
    @assert up ≈ Vec((0.0,0.0,1.0)) "Custom up vector not yet supported."
    ip_collection = LagrangeCollection{1}()
    qr_collection = QuadratureRuleCollection(2)
    cv_collection = CellValueCollection(qr_collection, ip_collection)

    dh = DofHandler(mesh)
    for name in subdomains
        add_subdomain!(dh, name, [ApproximationDescriptor(:coordinates, ip_collection)])
    end
    Ferrite.close!(dh)

    # Assemble Laplacian
    # TODO use bilinear operator for performance
    K = allocate_matrix(dh)

    assembler = start_assemble(K)
    for sdh in dh.subdofhandlers
        cellvalues = getcellvalues(cv_collection, getcells(mesh, first(sdh.cellset)))
        n_basefuncs = getnbasefunctions(cellvalues)
        Ke = zeros(n_basefuncs, n_basefuncs)
        @inbounds for cell in CellIterator(sdh)
            fill!(Ke, 0)

            reinit!(cellvalues, cell)

            for qp in QuadratureIterator(cellvalues)
                dΩ = getdetJdV(cellvalues, qp)

                for i in 1:n_basefuncs
                    ∇v = shape_gradient(cellvalues, qp, i)
                    for j in 1:n_basefuncs
                        ∇u = shape_gradient(cellvalues, qp, j)
                        Ke[i, j] += (∇v ⋅ ∇u) * dΩ
                    end
                end
            end

            assemble!(assembler, celldofs(cell), Ke)
        end
    end

    # Transmural coordinate
    ch = ConstraintHandler(dh);
    dbc = Dirichlet(:coordinates, getfacetset(mesh, "Endocardium"), (x, t) -> 0)
    Ferrite.add!(ch, dbc);
    dbc = Dirichlet(:coordinates, getfacetset(mesh, "Epicardium"), (x, t) -> 1)
    Ferrite.add!(ch, dbc);
    close!(ch)
    update!(ch, 0.0);

    K_transmural = copy(K)
    f = zeros(ndofs(dh))

    apply!(K_transmural, f, ch)
    transmural = K_transmural \ f;

    # Apicobasal coordinate
    #TODO refactor check for node set existence
    if !haskey(mesh.grid.nodesets, "Apex") #TODO this is just a hotfix, assuming that z points towards the apex
        apex_node_index = 1
        nodes = getnodes(mesh)
        for (i,node) ∈ enumerate(nodes)
            if nodes[i].x[3] > nodes[apex_node_index].x[3]
                apex_node_index = i
            end
        end
        addnodeset!(mesh, "Apex", OrderedSet{Int}((apex_node_index)))
    end

    ch = ConstraintHandler(dh);
    dbc = Dirichlet(:coordinates, getfacetset(mesh, "Base"), (x, t) -> 0)
    Ferrite.add!(ch, dbc);
    dbc = Dirichlet(:coordinates, getnodeset(mesh, "Apex"), (x, t) -> 1)
    Ferrite.add!(ch, dbc);
    close!(ch)
    update!(ch, 0.0);

    K_apicobasal = copy(K)
    f = zeros(ndofs(dh))

    apply!(K_apicobasal, f, ch)
    apicobasal = K_apicobasal \ f;

    rotational = zeros(ndofs(dh))
    rotational .= NaN

    qrn  = NodalQuadratureRuleCollection(ip_collection) # FIXME ip_collection from grid
    cvcn = CellValueCollection(qrn, ip_collection)
    for sdh in dh.subdofhandlers
        cellvalues = getcellvalues(cvcn, getcells(mesh, first(sdh.cellset)))
        @inbounds for cell in CellIterator(sdh)
            reinit!(cellvalues, cell)
            coords = getcoordinates(cell)
            dofs = celldofs(cell)

            for qp in QuadratureIterator(cellvalues)
                dΩ = getdetJdV(cellvalues, qp)

                x_dof = spatial_coordinate(cellvalues, qp, coords)

                x_planar = x_dof - (x_dof ⋅ up) * up # Project into plane
                xlen = norm(x_planar)
                if xlen < 1e-8
                    rotational[dofs[qp.i]] = 0.0
                else
                    x = x_planar / xlen
                    rotational[dofs[qp.i]] = (π + atan(x[1], x[2]))/2 # TODO tilted coordinate system
                end
            end
        end
    end

    return LVCoordinateSystem(dh, ip_collection, transmural, apicobasal, rotational)
end

"""
    compute_midmyocardial_section_coordinate_system(mesh::SimpleMesh)

Requires a mesh with facetsets
    * Base
    * Epicardium
    * Endocardium
    * Myocardium
"""
function compute_midmyocardial_section_coordinate_system(mesh::SimpleMesh{3}, subdomains::Vector{String} = [""]; up = Vec((0.0,0.0,1.0)))
    @assert up ≈ Vec((0.0,0.0,1.0)) "Custom up vector not yet supported."
    ip_collection = LagrangeCollection{1}()
    qr_collection = QuadratureRuleCollection(2)
    cv_collection = CellValueCollection(qr_collection, ip_collection)

    dh = DofHandler(mesh)
    for name in subdomains
        add_subdomain!(dh, name, [ApproximationDescriptor(:coordinates, ip_collection)])
    end
    Ferrite.close!(dh)

    # Assemble Laplacian
    # TODO use bilinear operator
    K = allocate_matrix(dh)

    assembler = start_assemble(K)
    for sdh in dh.subdofhandlers
        cellvalues = getcellvalues(cv_collection, getcells(mesh, first(sdh.cellset)))
        n_basefuncs = getnbasefunctions(cellvalues)
        Ke = zeros(n_basefuncs, n_basefuncs)
        @inbounds for cell in CellIterator(sdh)
            fill!(Ke, 0)

            reinit!(cellvalues, cell)

            for qp in QuadratureIterator(cellvalues)
                dΩ = getdetJdV(cellvalues, qp)

                for i in 1:n_basefuncs
                    ∇v = shape_gradient(cellvalues, qp, i)
                    for j in 1:n_basefuncs
                        ∇u = shape_gradient(cellvalues, qp, j)
                        Ke[i, j] += (∇v ⋅ ∇u) * dΩ
                    end
                end
            end

            assemble!(assembler, celldofs(cell), Ke)
        end
    end

    # Transmural coordinate
    ch = ConstraintHandler(dh);
    dbc = Dirichlet(:coordinates, getfacetset(mesh, "Endocardium"), (x, t) -> 0)
    Ferrite.add!(ch, dbc);
    dbc = Dirichlet(:coordinates, getfacetset(mesh, "Epicardium"), (x, t) -> 1)
    Ferrite.add!(ch, dbc);
    close!(ch)
    update!(ch, 0.0);

    K_transmural = copy(K)
    f = zeros(ndofs(dh))

    apply!(K_transmural, f, ch)
    transmural = K_transmural \ f;

    ch = ConstraintHandler(dh);
    dbc = Dirichlet(:coordinates, getfacetset(mesh, "Base"), (x, t) -> 0)
    Ferrite.add!(ch, dbc);
    dbc = Dirichlet(:coordinates, getfacetset(mesh, "Myocardium"), (x, t) -> 0.15)
    Ferrite.add!(ch, dbc);
    close!(ch)
    update!(ch, 0.0);

    K_apicobasal = copy(K)
    f = zeros(ndofs(dh))

    apply!(K_apicobasal, f, ch)
    apicobasal = K_apicobasal \ f;

    rotational = zeros(ndofs(dh))
    rotational .= NaN

    qrn  = NodalQuadratureRuleCollection(ip_collection) # FIXME ip_collection from grid
    cvcn = CellValueCollection(qrn, ip_collection)
    for sdh in dh.subdofhandlers
        cellvalues = getcellvalues(cvcn, getcells(mesh, first(sdh.cellset)))
        @inbounds for cell in CellIterator(sdh)
            reinit!(cellvalues, cell)
            coords = getcoordinates(cell)
            dofs = celldofs(cell)
            for qp in QuadratureIterator(cellvalues)
                dΩ = getdetJdV(cellvalues, qp)

                x_dof = spatial_coordinate(cellvalues, qp, coords)

                x_planar = x_dof - (x_dof ⋅ up) * up # Project into plane
                x = x_planar / norm(x_planar)

                rotational[dofs[qp.i]] = (π + atan(x[1], x[2]))/2 # TODO tilted coordinate system
            end
        end
    end

    return LVCoordinateSystem(dh, ip_collection, transmural, apicobasal, rotational)
end

"""
    vtk_coordinate_system(vtk, cs::LVCoordinateSystem)

Store the LV coordinate system in a vtk file.
"""
function vtk_coordinate_system(vtk, cs::LVCoordinateSystem)
    Ferrite.write_solution(vtk, cs.dh, cs.u_apicobasal, "apicobasal_")
    Ferrite.write_solution(vtk, cs.dh, cs.u_rotational, "rotational_")
    Ferrite.write_solution(vtk, cs.dh, cs.u_transmural, "transmural_")
end

"""
    BiVCoordinateSystem(dh, u_transmural, u_apicobasal, u_rotational, u_transventricular)

Universal ventricular coordinate, containing the transmural, apicobasal, rotational 
and transventricular coordinates.
"""
struct BiVCoordinateSystem{DH <: Ferrite.AbstractDofHandler}
    dh::DH
    u_transmural::Vector{Float32}
    u_apicobasal::Vector{Float32}
    u_rotational::Vector{Float32}
    u_transventricular::Vector{Float32}
end


"""
BiVCoordinate{T}

Biventricular universal coordinate, containing
    * transmural
    * apicobasal
    * rotational
    * transventricular
"""
struct BiVCoordinate{T}
    transmural::T
    apicaobasal::T
    rotational::T
    transventricular::T
end

value_type(::BiVCoordinateSystem) = BiVCoordinate

getcoordinateinterpolation(cs::BiVCoordinateSystem, cell::Ferrite.AbstractCell) = Ferrite.getfieldinterpolation(cs.dh, (1,1))

function vtk_coordinate_system(vtk, cs::BiVCoordinateSystem)
    Ferrite.write_solution(vtk, bivcs.dh, bivcs.u_transmural, "_transmural")
    Ferrite.write_solution(vtk, bivcs.dh, bivcs.u_apicobasal, "_apicobasal")
    Ferrite.write_solution(vtk, bivcs.dh, bivcs.u_rotational, "_rotational")
    Ferrite.write_solution(vtk, bivcs.dh, bivcs.u_transventricular, "_transventricular")
end

