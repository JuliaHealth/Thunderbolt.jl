# Long axis of a ventricular geometry.
#
# There is no single reliable estimator: the annotation-based one inherits the placement of a single
# apex node, and the inertia-based one needs near-axisymmetry that real chambers do not have. So all
# of them are computed, one is selected, and the disagreements are reported as diagnostics.

"""
    LongAxisInfo

The LV long axis together with the data it was derived from and two diagnostics, so that a bad
coordinate field or a mis-recovered facetset shows up as a number rather than as a wrong marker.

`axis` is a unit vector pointing **from the apex towards the base**, anchored at `apex`. Which
estimator produces it is selected by `axis_from`; all of them are computed and reported regardless,
each with its own reliability measure, because none is trustworthy on every geometry.

- `:basal_plane` (**default**) — normal of a least-squares plane fitted to the basal surface,
  oriented away from the apex. This is the most robust option for the meshes used here, because the
  base is deliberately clipped as planar as possible and the fit averages over the whole surface.
  Reliability: `base_rms_residual`, the RMS distance of the basal surface from the fitted plane in
  mesh length units — read it directly as "how planar is my clip".
- `:apex_base` — from the apex to the area-weighted base centroid. Uses both annotations, but note
  the apex is a **single node** on the `.vtu` meshes, so this direction inherits that one node's
  placement error. Reliability: `apex_base_discrepancy` against the fitted plane normal.
- `:inertia` — symmetry axis of the volume-weighted covariance tensor. Only meaningful for a
  near-axisymmetric chamber. `inertia_conditioning` near 1 means the geometry is too asymmetric (or
  too close to spherical) for it to say anything. Measured: ~10¹⁵ on the idealized ellipsoid
  (exactly axisymmetric, 0° discrepancy) but **1.27 on the idealized S17**, which is built with
  `septum_flatness = 0.325` and `axis_ratio = 0.925` and so has no degenerate transverse pair at
  all — there it is 60° off and must be ignored. Kept for diagnosis, not recommended as the axis.

The discrepancies are all measured against `axis`, so the one belonging to the selected estimator is
zero by construction.
"""
struct LongAxisInfo{T}
    axis::Vec{3,T}
    axis_from::Symbol
    apex::Vec{3,T}
    base_center::Vec{3,T}
    chamber_length::T
    base_normal::Vec{3,T}
    base_normal_discrepancy::T
    base_rms_residual::T
    apex_base_axis::Vec{3,T}
    apex_base_discrepancy::T
    inertia_axis::Vec{3,T}
    inertia_discrepancy::T
    inertia_conditioning::T
end

function Base.show(io::IO, lai::LongAxisInfo)
    print(io, "LongAxisInfo(", lai.axis_from, " axis = ", lai.axis,
        ", length = ", round(lai.chamber_length, digits=3),
        "; basal plane rms ", round(lai.base_rms_residual, digits=5),
        ", off by ", round(lai.base_normal_discrepancy, digits=2), "°",
        "; apex-base off by ", round(lai.apex_base_discrepancy, digits=2), "°",
        "; inertia off by ", round(lai.inertia_discrepancy, digits=2), "°",
        " (conditioning ", round(lai.inertia_conditioning, digits=3), "))")
end

"""
    compute_principal_axis(mesh; domain_name)

Symmetry axis of the volume-weighted covariance tensor `∫ (x-c)⊗(x-c) dΩ` about the center of mass.

Note this is **not** simply the direction of largest spread. An LV is approximately transversely
isotropic about its long axis, so the covariance has a near-degenerate *pair* of transverse
eigenvalues and one isolated eigenvalue along the axis — but whether the isolated one is the largest
or the smallest depends on how elongated the chamber is. A tall idealized ellipsoid (9.7 cm long,
3.2 cm radius) has its largest spread along the axis; the S17 geometry (6.7 cm long, ~3.5 cm radius)
is wider than it is tall and has its largest spread *transverse* to it. Picking the largest
eigenvalue therefore returns an axis ~89° wrong on the latter.

So: identify the closest pair of eigenvalues as the transverse plane and return the remaining
eigenvector. Returns `(axis, separation)`, where `separation` is the ratio of the larger to the
smaller eigenvalue gap — how confidently the isolated eigenvalue can be told apart from the pair.
Near 1 the body is too close to spherically symmetric for this estimator to mean anything.

The result is a line, not a direction: its sign is arbitrary and must be fixed by an annotation.
"""
function compute_principal_axis(
    mesh::SimpleMesh{3};
    domain_name = first(mesh.volumetric_subdomains.keys),
)
    c = compute_center_of_mass(mesh; domain_name)

    J = zero(SymmetricTensor{2,3,Float64})

    order = Ferrite.getorder(Ferrite.geometric_interpolation(getcells(mesh, 1)))
    ipc = LagrangeCollection{order}()
    dh = DofHandler(mesh)
    add_subdomain!(dh, domain_name, :u => ipc)
    close!(dh)

    qrc = QuadratureRuleCollection(max(2order - 1, 2))
    for sdh in dh.subdofhandlers
        gip = geometric_interpolation(get_first_cell(sdh))
        ip  = getinterpolation(ipc, sdh)
        qr  = getquadraturerule(qrc, sdh)
        cv  = CellValues(qr, ip, gip)
        for cell in CellIterator(sdh)
            Ferrite.reinit!(cv, cell)
            coords = getcoordinates(cell)
            for qp in QuadratureIterator(cv)
                dΩ = getdetJdV(cv, qp)
                r  = spatial_coordinate(cv, qp, coords) - c
                J += symmetric(r ⊗ r) * dΩ
            end
        end
    end

    E = eigen(J)
    λ = eigvals(E)   # ascending
    V = eigvecs(E)

    gap_low  = λ[2] - λ[1]   # pairing (λ1,λ2) as transverse leaves v3 as the axis
    gap_high = λ[3] - λ[2]   # pairing (λ2,λ3) as transverse leaves v1 as the axis
    axis = gap_low ≤ gap_high ? Vec{3}(V[:, 3]) : Vec{3}(V[:, 1])

    lo, hi = minmax(gap_low, gap_high)
    separation = lo ≈ 0.0 ? Inf : hi / lo
    return axis, separation
end

"""
    fit_basal_plane(mesh; domain_name, second_restriction)

Least-squares plane through the basal surface. Returns `(normal, centroid, rms_residual)`.

The base of these meshes is deliberately clipped as planar as possible, which makes this the
best-conditioned way to get a long axis: unlike the inertia estimator it does not care how
axisymmetric the chamber is, and unlike averaging facet normals it degrades gracefully and yields a
residual you can read directly. The normal is the eigenvector of the *smallest* principal value of
the area-weighted covariance of the basal surface points; `rms_residual = √λ_min` is the RMS
distance of that surface from the fitted plane, in mesh length units.

The result is a line, not a direction: its sign is arbitrary and must be fixed by an annotation.
"""
function fit_basal_plane(
    mesh::SimpleMesh{3};
    domain_name = "Base",
    second_restriction = nothing,
    u_nodal::Union{Nothing,Vector{Vec{3,Float64}}} = nothing,
)
    order = Ferrite.getorder(Ferrite.geometric_interpolation(getcells(mesh, 1)))
    ipc = LagrangeCollection{order}()
    qrc = Thunderbolt.FacetQuadratureRuleCollection(max(2order - 1, 2))

    # Two passes: the centroid first, then the covariance about it.
    pts = Tuple{Vec{3,Float64},Float64}[]
    surface_subdomain = mesh.surface_subdomains[domain_name]
    for (element_type, facets) in surface_subdomain.data
        gip = geometric_interpolation(element_type)
        refshape = Ferrite.getrefshape(element_type)
        ip = getinterpolation(ipc, refshape)
        qr = getquadraturerule(qrc, refshape)
        cv = FacetValues(qr, ip, gip)
        for cell in FacetIterator(mesh, facets)
            if second_restriction !== nothing
                cellid(cell.cc) ∈ second_restriction || continue
            end
            Ferrite.reinit!(cv, cell)
            coords = getcoordinates(cell)
            for qp in QuadratureIterator(cv)
                x = spatial_coordinate(cv, qp, coords)
                if u_nodal !== nothing
                    x += spatial_coordinate(cv, qp, u_nodal[getnodes(cell)])
                end
                push!(pts, (x, getdetJdV(cv, qp)))
            end
        end
    end
    isempty(pts) && throw(ArgumentError("Facetset \"$domain_name\" is empty; cannot fit a basal plane."))

    A = sum(w for (_, w) in pts)
    centroid = sum(x * w for (x, w) in pts) / A

    C = zero(SymmetricTensor{2,3,Float64})
    for (x, w) in pts
        r = x - centroid
        C += symmetric(r ⊗ r) * w
    end
    C = C / A

    E = eigen(C)
    λ = eigvals(E)          # ascending; the smallest is out-of-plane
    V = eigvecs(E)
    normal = Vec{3}(V[:, 1])
    rms_residual = sqrt(max(λ[1], 0.0))

    return normal, centroid, rms_residual
end

angle_between(a, b) = rad2deg(acos(clamp(a ⋅ b, -1.0, 1.0)))

"""
    compute_long_axis(mesh; axis_from, apex_nodeset, base_facetset, volumetric_domain_name)

Long axis of an LV geometry, anchored at the annotated apex and pointing towards the base.

Three estimators are computed and reported; `axis_from` selects which one becomes `axis`. See
[`LongAxisInfo`](@ref) for what each is worth. The default `:basal_plane` fits a plane to the basal
surface and takes its normal through the apex, which is the best-conditioned choice for meshes whose
base is clipped flat.

This replaces the assumption that the long axis is the global z-axis. That assumption is not merely
approximate — a raw `EllipsoidalLVMesh` has its apex at *maximum* z while the `.vtu` meshes have it
at *minimum* z, so any code taking a bounding-box extremum as "the apex" is wrong on one of the two.
"""
function compute_long_axis(
    mesh::SimpleMesh{3};
    axis_from::Symbol = :basal_plane,
    apex_nodeset = "Apex",
    base_facetset = "Base",
    second_restriction = nothing,
    volumetric_domain_name = first(mesh.volumetric_subdomains.keys),
)
    axis_from ∈ (:basal_plane, :apex_base, :inertia) ||
        throw(ArgumentError("axis_from must be one of :basal_plane, :apex_base, :inertia; got :$axis_from"))

    apexnodes = getnodeset(mesh, apex_nodeset)
    isempty(apexnodes) && throw(ArgumentError("Nodeset \"$apex_nodeset\" is empty; cannot derive a long axis."))
    apex = sum(get_node_coordinate(mesh.grid.nodes[n]) for n in apexnodes) / length(apexnodes)

    # Estimator 1 — apex to area-weighted base centroid. Signed by construction.
    base_normal_raw, base_center, base_rms_residual =
        fit_basal_plane(mesh; domain_name = base_facetset, second_restriction)
    d = base_center - apex
    chamber_length = norm(d)
    chamber_length ≈ 0.0 && throw(ArgumentError("Apex and base centroid coincide; cannot derive a long axis."))
    apex_base_axis = d / chamber_length

    # Estimator 2 — normal of that same fitted plane, oriented away from the apex.
    base_normal = sign(base_normal_raw ⋅ apex_base_axis) * base_normal_raw

    # Estimator 3 — covariance symmetry axis. Unsigned; only meaningful if well conditioned.
    inertia_axis, inertia_conditioning = compute_principal_axis(mesh; domain_name = volumetric_domain_name)
    inertia_axis = sign(inertia_axis ⋅ apex_base_axis) * inertia_axis

    axis = axis_from === :basal_plane ? base_normal :
           axis_from === :apex_base   ? apex_base_axis :
                                        inertia_axis

    return LongAxisInfo(
        axis, axis_from, apex, base_center, chamber_length,
        base_normal,    angle_between(base_normal,    axis), base_rms_residual,
        apex_base_axis, angle_between(apex_base_axis, axis),
        inertia_axis,   angle_between(inertia_axis,   axis), inertia_conditioning,
    )
end