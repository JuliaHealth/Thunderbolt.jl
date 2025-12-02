# NOTE This example is work in progress. Please consult it at a later time again.
using Thunderbolt, UnPack
using LinearSolve
import Thunderbolt: OS

import Ferrite: get_grid, find_field

dt₀ = 10.0
tspan = (0.0, 1000.0)
dtvis = 25.0

"""
In 'A transmurally heterogeneous orthotropic activation model for ventricular contraction and its numerical validation' it is suggested that uniform activtaion is fine.

TODO citation.

TODO add an example with a calcium profile compute via cell model and Purkinje activation
"""
calcium_profile_function(x,t) = t/1000.0 < 0.5 ? (1-x.transmural*0.7)*2.0*t/1000.0 : (2.0-2.0*t/1000.0)*(1-x.transmural*0.7)

# for (name, order, ring_grid) ∈ [
#     ("Debug-Ring", 1, Thunderbolt.generate_ring_mesh(16,4,4)),
#     # ("Linear-Ring", 1, Thunderbolt.generate_ring_mesh(40,8,8)),
#     # ("Quadratic-Ring", 2, Thunderbolt.generate_quadratic_ring_mesh(20,4,4))
# ]

name = "ring-test"
order = 1
ring_grid = Thunderbolt.generate_ring_mesh(16,4,4)

qr_collection = QuadratureRuleCollection(2*order-1)

ip_fsn = LagrangeCollection{1}()^3
ip_mech = LagrangeCollection{order}()^3

ring_cs = compute_midmyocardial_section_coordinate_system(ring_grid)

constitutive_model = ActiveStressModel(
    Guccione1991PassiveModel(),
    # PiersantiActiveStress(;Tmax=10.0),
    Guccione1993ActiveModel(;Tmax=100.0),
    PelceSunLangeveld1995Model(;calcium_field=AnalyticalCoefficient(
        calcium_profile_function,
        CoordinateSystemCoefficient(ring_cs)
    )),
    create_simple_microstructure_model(ring_cs, ip_fsn,
        endo_helix_angle = deg2rad(-60.0),
        epi_helix_angle = deg2rad(60.0),
        endo_transversal_angle = 0.0,
        epi_transversal_angle = 0.0,
        sheetlet_pseudo_angle = deg2rad(0)
    )
)

quasistaticform = semidiscretize(
    StructuralModel(:displacement, constitutive_model, ()),
    FiniteElementDiscretization(
        Dict(:displacement => ip_mech),
        [
            Dirichlet(:displacement, getnodeset(ring_grid, "MyocardialAnchor1"), (x,t) -> (0.0, 0.0, 0.0), [1,2,3]),
            Dirichlet(:displacement, getnodeset(ring_grid, "MyocardialAnchor2"), (x,t) -> (0.0, 0.0), [2,3]),
            Dirichlet(:displacement, getnodeset(ring_grid, "MyocardialAnchor3"), (x,t) -> (0.0,), [3]),
            Dirichlet(:displacement, getnodeset(ring_grid, "MyocardialAnchor4"), (x,t) -> (0.0,), [3])
        ]
    ),
    ring_grid
)

problem = QuasiStaticProblem(quasistaticform, tspan)
timestepper = HomotopyPathSolver(
    NewtonRaphsonSolver(
        max_iter=10,
        inner_solver=LinearSolve.UMFPACKFactorization(),
    )
)

integrator = OS.init(problem, timestepper, dt=dt₀, verbose=true, adaptive=true, dtmax=25.0)

io = ParaViewWriter(name);

using Thunderbolt.TimerOutputs
TimerOutputs.enable_debug_timings(Thunderbolt)
TimerOutputs.reset_timer!()
for (u, t) in OS.TimeChoiceIterator(integrator, tspan[1]:dtvis:tspan[2])
    @info t,norm(u)

    @unpack dh = problem.f
    grid = get_grid(dh)
    cvc = CellValueCollection(qr_collection, ip_mech)

    # Compute some elementwise measures
    E_ff = zeros(getncells(grid))
    E_cc = zeros(getncells(grid))
    E_ll = zeros(getncells(grid))
    E_rr = zeros(getncells(grid))

    Jdata = zeros(getncells(grid))

    frefdata = zeros(Ferrite.Vec{3,Float64}, getncells(grid))
    srefdata = zeros(Ferrite.Vec{3,Float64}, getncells(grid))
    fdata = zeros(Ferrite.Vec{3,Float64}, getncells(grid))
    sdata = zeros(Ferrite.Vec{3,Float64}, getncells(grid))
    helixangledata = zeros(Float64, getncells(grid))
    helixanglerefdata = zeros(Float64, getncells(grid))

    # Compute some elementwise measures
    for sdh ∈ dh.subdofhandlers
        field_idx = find_field(sdh, :displacement)
        field_idx === nothing && continue
        qr = getquadraturerule(qr_collection, sdh)
        # fsncache = Thunderbolt.setup_coefficient_cache(problem.f.constitutive_model.microstructure_model, qr, sdh)
        cv = Thunderbolt.getcellvalues(cvc, sdh)
        for cell ∈ CellIterator(sdh)
            Thunderbolt.reinit!(cv, cell)
            global_dofs = celldofs(cell)
            field_dofs  = dof_range(sdh, field_idx)
            uₑ = u[global_dofs] # element dofs

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
                # coeff = evaluate_coefficient(fsncache, cell, qp, time)
                # f₀,s₀,n₀ = coeff.f, coeff.s, coeff.n
                # E_ff_cell += f₀ ⋅ E ⋅ f₀

                # f₀_current = F⋅f₀
                # f₀_current /= norm(f₀_current)

                # s₀_current = F⋅s₀
                # s₀_current /= norm(s₀_current)

                # coords = getcoordinates(cell)
                # x_global = spatial_coordinate(cv, qp, coords)

                # # v_longitudinal = function_gradient(cv_cs, qp, coordinate_system.u_apicobasal[celldofs(cell)])
                # # v_radial = function_gradient(cv_cs, qp, coordinate_system.u_transmural[celldofs(cell)])
                # # v_circimferential = v_radial × v_longitudinal
                # # @TODO compute properly via coordinate system
                # v_longitudinal = Ferrite.Vec{3}((0.0, 0.0, 1.0))
                # v_radial = Ferrite.Vec{3}((x_global[1],x_global[2],0.0))
                # v_radial /= norm(v_radial)
                # v_circimferential = v_radial × v_longitudinal # Ferrite.Vec{3}((x_global[2],x_global[1],0.0))
                # v_circimferential /= norm(v_circimferential)
                # #
                # E_ll_cell += v_longitudinal ⋅ E ⋅ v_longitudinal
                # E_rr_cell += v_radial ⋅ E ⋅ v_radial
                # E_cc_cell += v_circimferential ⋅ E ⋅ v_circimferential

                # Jdata_cell += det(F)

                # frefdata_cell += f₀
                # srefdata_cell += s₀

                # fdata_cell += f₀_current
                # sdata_cell += s₀_current

                # helixangle_cell += acos(clamp(f₀_current ⋅ v_circimferential, -1.0, 1.0)) * sign((v_circimferential × f₀_current) ⋅ v_radial)
                # helixangleref_cell += acos(clamp(f₀ ⋅ v_circimferential, -1.0, 1.0)) * sign((v_circimferential × f₀) ⋅ v_radial)
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
    Thunderbolt.store_timestep!(io, t, dh.grid) do file
        Thunderbolt.store_timestep_field!(io, t, dh, u, :displacement)
        # TODo replace by "dump coefficient" function
        Thunderbolt.store_timestep_celldata!(io, t, hcat(fdata...),"Current Fiber Data")
        Thunderbolt.store_timestep_celldata!(io, t, hcat(sdata...),"Current Sheet Data")
        Thunderbolt.store_timestep_celldata!(io, t, E_ff,"E_ff")
        Thunderbolt.store_timestep_celldata!(io, t, E_cc,"E_cc")
        Thunderbolt.store_timestep_celldata!(io, t, E_rr,"E_rr")
        Thunderbolt.store_timestep_celldata!(io, t, E_ll,"E_ll")
        Thunderbolt.store_timestep_celldata!(io, t, Jdata,"J")
        Thunderbolt.store_timestep_celldata!(io, t, rad2deg.(helixangledata),"Helix Angle")
        Thunderbolt.store_timestep_celldata!(io, t, rad2deg.(helixanglerefdata),"Helix Angle (End Diastole)")
    end
end
TimerOutputs.print_timer()
TimerOutputs.disable_debug_timings(Thunderbolt)
