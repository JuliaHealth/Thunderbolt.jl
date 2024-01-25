using Thunderbolt, UnPack

import Ferrite: get_grid, find_field

# TODO refactor this one into the framework code and put a nice abstraction layer around it
struct StandardMechanicalIOPostProcessor{IO, CVC, CSC}
    io::IO
    cvc::CVC
    csc::CSC
end

function (postproc::StandardMechanicalIOPostProcessor)(t, problem, solver_cache)
    @unpack dh = problem
    grid = get_grid(dh)
    @unpack io, cvc = postproc

    # Compute some elementwise measures
    E_ff = zeros(getncells(grid))
    E_cc = zeros(getncells(grid))
    E_ll = zeros(getncells(grid))
    E_rr = zeros(getncells(grid))

    Jdata = zeros(getncells(grid))

    frefdata = zero(Vector{Ferrite.Vec{3}}(undef, getncells(grid)))
    srefdata = zero(Vector{Ferrite.Vec{3}}(undef, getncells(grid)))
    fdata = zero(Vector{Ferrite.Vec{3}}(undef, getncells(grid)))
    sdata = zero(Vector{Ferrite.Vec{3}}(undef, getncells(grid)))
    helixangledata = zero(Vector{Float64}(undef, getncells(grid)))
    helixanglerefdata = zero(Vector{Float64}(undef, getncells(grid)))

    # Compute some elementwise measures
    for sdh ∈ dh.subdofhandlers
        field_idx = find_field(sdh, :displacement)
        field_idx === nothing && continue 
        for cell ∈ CellIterator(sdh)
            cv = Thunderbolt.getcellvalues(cvc, getcells(grid, cellid(cell)))

            reinit!(cv, cell)
            global_dofs = celldofs(cell)
            field_dofs  = dof_range(sdh, field_idx)
            uₑ = solver_cache.uₙ[global_dofs] # element dofs

            E_ff_cell = 0.0
            E_cc_cell = 0.0
            E_rr_cell = 0.0
            E_ll_cell = 0.0

            Jdata_cell = 0.0

            frefdata_cell = Ferrite.Vec{3}((0.0, 0.0, 0.0))
            srefdata_cell = Ferrite.Vec{3}((0.0, 0.0, 0.0))
            fdata_cell = Ferrite.Vec{3}((0.0, 0.0, 0.0))
            sdata_cell = Ferrite.Vec{3}((0.0, 0.0, 0.0))
            helixangle_cell = 0.0
            helixangleref_cell = 0.0

            nqp = getnquadpoints(cv)
            for qp in QuadratureIterator(cv)
                dΩ = getdetJdV(cv, qp)

                # Compute deformation gradient F
                ∇u = function_gradient(cv, qp, uₑ)
                F = one(∇u) + ∇u

                C = tdot(F)
                E = (C-one(C))/2.0
                f₀,s₀,n₀ = evaluate_coefficient(problem.constitutive_model.microstructure_model, cell, qp, time)

                E_ff_cell += f₀ ⋅ E ⋅ f₀

                f₀_current = F⋅f₀
                f₀_current /= norm(f₀_current)

                s₀_current = F⋅s₀
                s₀_current /= norm(s₀_current)

                coords = getcoordinates(cell)
                x_global = spatial_coordinate(cv, qp, coords)

                # v_longitudinal = function_gradient(cv_cs, qp, coordinate_system.u_apicobasal[celldofs(cell)])
                # v_radial = function_gradient(cv_cs, qp, coordinate_system.u_transmural[celldofs(cell)])
                # v_circimferential = v_longitudinal × v_radial
                # @TODO compute properly via coordinate system
                v_longitudinal = Ferrite.Vec{3}((0.0, 0.0, 1.0))
                v_radial = Ferrite.Vec{3}((x_global[1],x_global[2],0.0))
                v_radial /= norm(v_radial)
                v_circimferential = v_longitudinal × v_radial # Ferrite.Vec{3}((x_global[2],x_global[1],0.0))
                v_circimferential /= norm(v_circimferential)
                #
                E_ll_cell += v_longitudinal ⋅ E ⋅ v_longitudinal
                E_rr_cell += v_radial ⋅ E ⋅ v_radial
                E_cc_cell += v_circimferential ⋅ E ⋅ v_circimferential

                Jdata_cell += det(F)

                frefdata_cell += f₀
                srefdata_cell += s₀

                fdata_cell += f₀_current
                sdata_cell += s₀_current

                helixangle_cell += acos(clamp(f₀_current ⋅ v_circimferential, -1.0, 1.0)) * sign((v_circimferential × f₀_current) ⋅ v_radial)
                helixangleref_cell += acos(clamp(f₀ ⋅ v_circimferential, -1.0, 1.0)) * sign((v_circimferential × f₀) ⋅ v_radial)
            end

            E_ff[Ferrite.cellid(cell)] = E_ff_cell / nqp
            E_cc[Ferrite.cellid(cell)] = E_cc_cell / nqp
            E_rr[Ferrite.cellid(cell)] = E_rr_cell / nqp
            E_ll[Ferrite.cellid(cell)] = E_ll_cell / nqp
            Jdata[Ferrite.cellid(cell)] = Jdata_cell / nqp
            frefdata[Ferrite.cellid(cell)] = frefdata_cell / nqp
            frefdata[Ferrite.cellid(cell)] /= norm(frefdata[Ferrite.cellid(cell)])
            srefdata[Ferrite.cellid(cell)] = srefdata_cell / nqp
            srefdata[Ferrite.cellid(cell)] /= norm(srefdata[Ferrite.cellid(cell)])
            fdata[Ferrite.cellid(cell)] = fdata_cell / nqp
            fdata[Ferrite.cellid(cell)] /= norm(fdata[Ferrite.cellid(cell)])
            sdata[Ferrite.cellid(cell)] = sdata_cell / nqp
            sdata[Ferrite.cellid(cell)] /= norm(sdata[Ferrite.cellid(cell)])
            helixanglerefdata[Ferrite.cellid(cell)] = helixangleref_cell / nqp
            helixangledata[Ferrite.cellid(cell)] = helixangle_cell / nqp
        end
    end

    # Save the solution
    Thunderbolt.store_timestep!(io, t, dh.grid)
    Thunderbolt.store_timestep_field!(io, t, dh, solver_cache.uₙ, :displacement)
    # TODo replace by "dump coefficient" function
    # Thunderbolt.store_timestep_field!(io, t, coordinate_system.dh, coordinate_system.u_transmural, "transmural")
    # Thunderbolt.store_timestep_field!(io, t, coordinate_system.dh, coordinate_system.u_apicobasal, "apicobasal")
    Thunderbolt.store_timestep_celldata!(io, t, hcat(frefdata...),"Reference Fiber Data")
    Thunderbolt.store_timestep_celldata!(io, t, hcat(fdata...),"Current Fiber Data")
    Thunderbolt.store_timestep_celldata!(io, t, hcat(srefdata...),"Reference Sheet Data")
    Thunderbolt.store_timestep_celldata!(io, t, hcat(sdata...),"Current Sheet Data")
    Thunderbolt.store_timestep_celldata!(io, t, E_ff,"E_ff")
    Thunderbolt.store_timestep_celldata!(io, t, E_cc,"E_cc")
    Thunderbolt.store_timestep_celldata!(io, t, E_rr,"E_rr")
    Thunderbolt.store_timestep_celldata!(io, t, E_ll,"E_ll")
    Thunderbolt.store_timestep_celldata!(io, t, Jdata,"J")
    Thunderbolt.store_timestep_celldata!(io, t, rad2deg.(helixangledata),"Helix Angle")
    Thunderbolt.store_timestep_celldata!(io, t, rad2deg.(helixanglerefdata),"Helix Angle (End Diastole)")
    Thunderbolt.finalize_timestep!(io, t)

    # min_vol = min(min_vol, calculate_volume_deformed_mesh(uₙ,dh,cv));
    # max_vol = max(max_vol, calculate_volume_deformed_mesh(uₙ,dh,cv));
end


function solve_test_ring(name_base, constitutive_model, grid, cs, face_models, ip_mech::Thunderbolt.VectorInterpolationCollection, qr_collection::QuadratureRuleCollection, Δt = 100.0, T = 1000.0)
    io = ParaViewWriter(name_base);
    # io = JLD2Writer(name_base);

    problem = semidiscretize(
        StructuralModel(constitutive_model, face_models),
        FiniteElementDiscretization(
            Dict(:displacement => ip_mech),
            [Dirichlet(:displacement, getfaceset(grid, "Myocardium"), (x,t) -> [0.0], [3])]
        ),
        grid
    )

    # Postprocessor
    cv_post = CellValueCollection(qr_collection, ip_mech)
    standard_postproc = StandardMechanicalIOPostProcessor(io, cv_post, CoordinateSystemCoefficient(cs))

    # Create sparse matrix and residual vector
    solver = LoadDrivenSolver(NewtonRaphsonSolver(;max_iter=100))

    Thunderbolt.solve(
        problem,
        solver,
        Δt, 
        (0.0, T),
        default_initializer,
        standard_postproc
    )
end


"""
In 'A transmurally heterogeneous orthotropic activation model for ventricular contraction and its numerical validation' it is suggested that uniform activtaion is fine.

TODO citation.

TODO add an example with a calcium profile compute via cell model and Purkinje activation
"""
calcium_profile_function(x,t) = t/1000.0 < 0.5 ? (1-x.transmural*0.7)*2.0*t/1000.0 : (2.0-2.0*t/1000.0)*(1-x.transmural*0.7)

for (name, order, ring_grid) ∈ [
    ("Linear-Ring", 1, Thunderbolt.generate_ring_mesh(40,8,8)),
    ("Quadratic-Ring", 2, Thunderbolt.generate_quadratic_ring_mesh(20,4,4))
]

intorder = 2*order
qr_collection = QuadratureRuleCollection(intorder-1)

ip_fsn = LagrangeCollection{order}()^3
ip_u = LagrangeCollection{order}()^3

ring_cs = compute_midmyocardial_section_coordinate_system(ring_grid)
solve_test_ring(name,
    ActiveStressModel(
        Guccione1991PassiveModel(),
        PiersantiActiveStress(;Tmax=10.0),
        PelceSunLangeveld1995Model(;calcium_field=AnalyticalCoefficient(
            calcium_profile_function,
            CoordinateSystemCoefficient(ring_cs)
        )),
        create_simple_microstructure_model(ring_cs, ip_fsn,
            endo_helix_angle = deg2rad(0.0),
            epi_helix_angle = deg2rad(0.0),
            endo_transversal_angle = 0.0,
            epi_transversal_angle = 0.0,
            sheetlet_pseudo_angle = deg2rad(0)
        )
    ), ring_grid, ring_cs,
    [NormalSpringBC(0.01, "Epicardium")],
    ip_u, qr_collection,
    100.0
)

end
