# # [Mechanics Tutorial 3: Coupling with Lumped Blood Circuits](@id mechanics-tutorial_3d0dcoupling)
# ![Pressure Volume Loop](3d0d-pv-loop.gif)
#
# This tutorial shows how to couple 3d chamber models with 0d fluid models.
#
# ## Introduction
#
# In this tutorial we will reproduce a simplified version of the model presented by [RegSalAfrFedDedQar:2022:cem](@citet).
#
# !!! warning
#     The API for 3D-0D coupling is work in progress and is hence subject to potential breaking changes.
#
# ## Commented Program
# We start by loading Thunderbolt and LinearSolve to use a custom direct solver of our choice.
using Thunderbolt, LinearSolve
# Furthermore we will use CirculatorySystemModels to define the blood circuit model.
using CirculatorySystemModels, DynamicQuantities
# Finally, we try to approach a valid initial state by solving a simpler model first.
using ModelingToolkit, OrdinaryDiffEqTsit5, OrdinaryDiffEqOperatorSplitting

using ModelingToolkit: t_nounits as t, D_nounits as D

function ϕRSAFDQ2022(t, tc, tr, TC, TR, THB)
    c11 = 0 ≤ mod(t - tc, THB)
    c12 = mod(t - tc, THB) ≤ TC

    c21 = 0 ≤ mod(t - tr, THB)
    c22 = mod(t - tr, THB) ≤ TR

    return c11*c12 * 0.5(1-cos(π/TC*mod(t-tc,THB))) + c21*c22*0.5(1+cos(π/TR*mod(t-tr,THB)))
end

# We start by defining a MTK component to couple the circuit model with Thunderbolt.
@mtkmodel PressureCouplingChamber begin
    @components begin
        in  = Pin()
        out = Pin()
    end
    @parameters begin
        p3D = 0.0, [description = "Pressure of the associated 3D chamber"]
    end
    @variables begin
        V(t), [description = "Volume"]
        p(t), [description = "Pressure"]
    end
    @equations begin
        p ~ p3D
        D(V) ~ in.q + out.q
        p ~ in.p
        p ~ out.p
    end
end

# [RegSalAfrFedDedQar:2022:cem](@citet) use a leaky diode for the heart valves, which is not part of CirculatorySystemModels.
@mtkmodel LeakyResistorDiode begin
    @extend OnePort()
    @parameters begin
        Rₘᵢₙ
        Rₘₐₓ
    end
    @equations begin
        q ~ -(Δp / Rₘᵢₙ * (Δp < 0) + Δp / Rₘₐₓ * (Δp ≥ 0))
    end
end

@mtkmodel RSAFDQ2022Chamber begin
    @components begin
        in  = Pin()
        out = Pin()
    end
    @parameters begin
        Epass, [description = "Passive elastance"]
        Eactmax, [description = "Active elastance"]
        V0, [description = "Dead volume"]
        tC
        TC
        TR
        τ, [description = "Total beat time"]
        pₑₓ
    end
    @variables begin
        V(t), [description = "Volume"]
        p(t), [description = "Pressure"]
        E(t), [description = "Elastance"]
    end
    @equations begin
        E ~ Epass + Eactmax * ϕRSAFDQ2022(t, tC, tC+TC, TC, TR, τ)
        p ~ pₑₓ + E * (V - V0)
        D(V) ~ in.q + out.q
        p ~ in.p
        out.p ~ in.p
    end
end

@mtkmodel RSAFDQ2022CircuitModularLV begin
    @structural_parameters begin
        τ   = 800.0#ustrip(0.8u"s" |> us"ms") #, [unit = u"ms", description = "Contraction cycle length"]
    end
    # These are the converted parameters from the paper [RegSalAfrFedDedQar:2022:cem](@cite) Appendix A
    @parameters begin
        ## Systemic Circuit
        Rsysₐᵣ  = 106.65789473684211#ustrip(0.800u"mmHg*s/mL" |> us"kPa*ms/mL")
        Csysₐᵣ  = 9.000740192450037#ustrip(1.2u"mL/mmHg" |> us"mL/kPa")
        Lsysₐᵣ  = 666.6118421052632#ustrip(5e-3u"mmHg*s^2/mL" |> us"kPa*ms^2/mL")
        Rsysᵥₑₙ = 34.66381578947368#ustrip(0.260u"mmHg*s/mL" |> us"kPa*ms/mL")
        Csysᵥₑₙ = 1200.0986923266717#ustrip(160.0u"mL/mmHg" |> us"mL/kPa")
        Lsysᵥₑₙ = 66.66118421052632#ustrip(5e-4u"mmHg*s^2/mL" |> us"kPa*ms^2/mL")
        ## Pulmonary Circuit
        Rpulₐᵣ  = 21.66488486842105#ustrip(0.1625u"mmHg*s/mL" |> us"kPa*ms/mL")
        Cpulₐᵣ  = 75.00616827041698#ustrip(10.00u"mL/mmHg" |> us"mL/kPa")
        Lpulₐᵣ  = 66.66118421052632#ustrip(5e-4u"mmHg*s^2/mL" |> us"kPa*ms^2/mL")
        Rpulᵥₑₙ = 21.66488486842105#ustrip(0.1625u"mmHg*s/mL" |> us"kPa*ms/mL")
        Cpulᵥₑₙ = 120.00986923266716#ustrip(16.00u"mL/mmHg" |> us"mL/kPa")
        Lpulᵥₑₙ = 66.66118421052632#ustrip(5e-4u"mmHg*s^2/mL" |> us"kPa*ms^2/mL")
        ## Valves
        Rₘᵢₙ = 1.0#ustrip(0.0075u"mmHg*s/mL" |> us"kPa*ms/mL")
        Rₘₐₓ = 9.999177631578946e6#ustrip(75000.0u"mmHg*s/mL" |> us"kPa*ms/mL")
        ## Left Atrium
        Epassₗₐ   = 0.011999013157894737#ustrip(0.09u"mmHg/mL" |> us"kPa/mL")
        Eactmaxₗₐ = 0.009332565789473684#ustrip(0.07u"mmHg/mL" |> us"kPa/mL")
        V0ₗₐ = 4.0#ustrip(4.0u"mL" |> us"mL")
        tCₗₐ = 600.0#ustrip(0.6u"s" |> us"ms")
        TCₗₐ = 104.0#ustrip(0.104u"s" |> us"ms")
        TRₗₐ = 680.0#ustrip(0.68u"s" |> us"ms")
        ## Right Atrium
        Epassᵣₐ   = 0.009332565789473684#ustrip(0.07u"mmHg/mL" |> us"kPa/mL")
        Eactmaxᵣₐ = 0.007999342105263157#ustrip(0.06u"mmHg/mL" |> us"kPa/mL")
        V0ᵣₐ = 4.0#ustrip(4.0u"mL" |> us"mL")
        TRᵣₐ = 560.0#ustrip(0.56u"s" |> us"ms")
        tCᵣₐ = 64.0#ustrip(0.064u"s" |> us"ms")
        TCᵣₐ = 640.0#ustrip(0.64u"s" |> us"ms")
        ## Right Ventricle
        Epassᵣᵥ   = 0.0066661184210526315#ustrip(0.05u"mmHg/mL" |> us"kPa/mL")
        Eactmaxᵣᵥ = 0.07332730263157895#ustrip(0.55u"mmHg/mL" |> us"kPa/mL")
        V0ᵣᵥ = 10.0#ustrip(10.0u"mL" |> us"mL")
        tCᵣᵥ = 0.0#ustrip(0.0u"s" |> us"ms")
        TCᵣᵥ = 272.0#ustrip(0.272u"s" |> us"ms")
        TRᵣᵥ = 120.0#ustrip(0.12u"s" |> us"ms")
        ## External pressure
        pₑₓ = 0.0#ustrip(0.0u"mmHg" |> us"kPa")
    end
    @components begin
        LV = PressureCouplingChamber()
        LA = RSAFDQ2022Chamber(Epass=Epassₗₐ, Eactmax=Eactmaxₗₐ, V0=V0ₗₐ, tC=tCₗₐ, TC=TCₗₐ, TR=TRₗₐ, τ, pₑₓ)
        RV = RSAFDQ2022Chamber(Epass=Epassᵣᵥ, Eactmax=Eactmaxᵣᵥ, V0=V0ᵣᵥ, tC=tCᵣᵥ, TC=TCᵣᵥ, TR=TRᵣᵥ, τ, pₑₓ)
        RA = RSAFDQ2022Chamber(Epass=Epassᵣₐ, Eactmax=Eactmaxᵣₐ, V0=V0ᵣₐ, tC=tCᵣₐ, TC=TCᵣₐ, TR=TRᵣₐ, τ, pₑₓ)

        MV = LeakyResistorDiode(Rₘᵢₙ, Rₘₐₓ) # Mitral
        AV = LeakyResistorDiode(Rₘᵢₙ, Rₘₐₓ) # Aortic
        TV = LeakyResistorDiode(Rₘᵢₙ, Rₘₐₓ) # Triscupid
        PV = LeakyResistorDiode(Rₘᵢₙ, Rₘₐₓ) # Pulmonary

        # Systemic Circuit
        SYSₐᵣ  = CRL(R = Rsysₐᵣ , L = Lsysₐᵣ , C = Csysₐᵣ)  # Arterial
        SYSᵥₑₙ = CRL(R = Rsysᵥₑₙ, L = Lsysᵥₑₙ, C = Csysᵥₑₙ) # Venous
        # Pulmonary Circuit
        PULₐᵣ  = CRL(R = Rpulₐᵣ , L = Lpulₐᵣ , C = Cpulₐᵣ)  # Arterial
        PULᵥₑₙ = CRL(R = Rpulᵥₑₙ, L = Lpulᵥₑₙ, C = Cpulᵥₑₙ) # Venous
    end

    @equations begin
        connect(LV.out, AV.in)
        connect(AV.out, SYSₐᵣ.in)
        connect(SYSₐᵣ.out, SYSᵥₑₙ.in)
        connect(SYSᵥₑₙ.out, RA.in)
        connect(RA.out, TV.in)
        connect(TV.out, RV.in)
        connect(RV.out, PV.in)
        connect(PV.out, PULₐᵣ.in)
        connect(PULₐᵣ.out, PULᵥₑₙ.in)
        connect(PULᵥₑₙ.out, LA.in)
        connect(LA.out, MV.in)
        connect(MV.out, LV.in)
    end
end


## Compose the whole ODE system
@named circ_model = RSAFDQ2022CircuitModularLV()
circ_sys = mtkcompile(circ_model)

## Precomputed initial guess
u0fluid = [
    circ_sys.LV.V => 94.6 #mL
    circ_sys.LA.V => 51.81
    circ_sys.RV.V => 109.9 # mL
    circ_sys.RA.V => 45.6 # mL
    circ_sys.SYSₐᵣ.C.V => 79.47
    circ_sys.SYSₐᵣ.L.q => 0.0
    circ_sys.SYSᵥₑₙ.C.V => 3341.88
    circ_sys.SYSᵥₑₙ.L.q => 0.0602
    circ_sys.PULₐᵣ.C.V => 277.89
    circ_sys.PULₐᵣ.L.q => 0.056
    circ_sys.PULᵥₑₙ.C.V => 298.75
    circ_sys.PULᵥₑₙ.L.q => 0.068
]

# We now generate the mechanical subproblem as in the [first tutorial](@ref mechanics-tutorial_simple-active-stress)
scaling_factor = 3.4;
# !!! warning
#     Tuning parameter until all bugs are fixed in this tutorial :)
mesh = generate_ideal_lv_mesh(8,2,5;
    inner_radius = scaling_factor*0.7,
    outer_radius = scaling_factor*1.0,
    longitudinal_upper = 0.4,
    apex_inner = scaling_factor* 1.3,
    apex_outer = scaling_factor*1.5
)
mesh = Thunderbolt.hexahedralize(mesh)
# !!! todo
#     The 3D0D coupling does not yet support multiple subdomains.

coordinate_system = compute_lv_coordinate_system(mesh)
microstructure    = create_microstructure_model(
    coordinate_system,
    LagrangeCollection{1}()^3,
    ODB25LTMicrostructureParameters(),
);
passive_material_model = Guccione1991PassiveModel()
active_material_model  = Guccione1993ActiveModel()
function calcium_profile_function(x::LVCoordinate,t_global)
    linear_interpolation(t,y1,y2,t1,t2) = y1 + (t-t1) * (y2-y1)/(t2-t1)
    ca_peak(x)                          = 1.0
    t = t_global % 800.0
    if 0 ≤ t ≤ 120.0
        return linear_interpolation(t,        0.0, ca_peak(x),   0.0, 120.0)
    elseif t ≤ 272.0
        return linear_interpolation(t, ca_peak(x),        0.0, 120.0, 272.0)
    else
        return 0.0
    end
end
calcium_field = AnalyticalCoefficient(
    calcium_profile_function,
    coordinate_system,
)
sarcomere_model = CaDrivenInternalSarcomereModel(ConstantStretchModel(), calcium_field)
active_stress_model = ActiveStressModel(
    passive_material_model,
    active_material_model,
    sarcomere_model,
    microstructure,
)
weak_boundary_conditions = (RobinBC(1.0, "Epicardium"),NormalSpringBC(100.0, "Base"))
solid_model = QuasiStaticModel(:displacement, active_stress_model, weak_boundary_conditions);

# The solid model is now couple with the circuit model by adding a Lagrange multipliers constraining the 3D chamber volume to match the chamber volume in the 0D model.
p3D = circ_sys.LV.p3D
V0D = circ_sys.LV.V
fluid_model = MTKLumpedCicuitModel(circ_sys, u0fluid, [p3D])
coupler = LumpedFluidSolidCoupler(
    [
        ChamberVolumeCoupling(
            "Endocardium",
            RSAFDQ2022SurrogateVolume(),
            V0D,
            p3D,
        )
    ],
    :displacement,
)
coupled_model = RSAFDQ2022Model(solid_model, fluid_model, coupler);
# !!! todo
#     Once we figure out a nicer way to do this we should add more detailed docs here.

# Now we semidiscretize the model spatially as usual with finite elements and annotate the model with a stable split.
spatial_discretization_method = FiniteElementDiscretization(
    Dict(:displacement => LagrangeCollection{1}()^3),
    [
        Dirichlet(:displacement, getfacetset(mesh, "Base"), (x,t) -> [0.0], [3]),
        Dirichlet(:displacement, getnodeset(mesh, "MyocardialAnchor1"), (x,t) -> (0.0, 0.0, 0.0), [1,2,3]),
        Dirichlet(:displacement, getnodeset(mesh, "MyocardialAnchor2"), (x,t) -> (0.0, 0.0), [2,3]),
        Dirichlet(:displacement, getnodeset(mesh, "MyocardialAnchor3"), (x,t) -> (0.0,), [3]),
        Dirichlet(:displacement, getnodeset(mesh, "MyocardialAnchor4"), (x,t) -> (0.0,), [3])
    ],
)
splitform = semidiscretize(
    RSAFDQ2022Split(coupled_model),
    spatial_discretization_method,
    mesh,
)

dt₀ = 5.0
dtvis = 10.0
tspan = (0.0, 3*800.0)
# This speeds up the CI # hide
tspan = (0.0, 100.0)    # hide

# The remaining code is very similar to how we use SciML solvers.
chamber_solver = HomotopyPathSolver(
    NewtonRaphsonSolver(;
        max_iter=10,
        tol=1e-2,
        inner_solver=SchurComplementLinearSolver(
            LinearSolve.UMFPACKFactorization()
        )
    )
)
blood_circuit_solver = Tsit5()
# timestepper = LieTrotterGodunov((chamber_solver, blood_circuit_solver))
timestepper = OrdinaryDiffEqOperatorSplitting.PalindromicPairLieTrotterGodunov((chamber_solver, blood_circuit_solver))

u₀ = zeros(solution_size(splitform))
u₀solid_view = @view  u₀[OS.get_solution_indices(splitform, 1)]
u₀fluid_view = @view  u₀[OS.get_solution_indices(splitform, 2)]
for (sym, val) in u0fluid
    u₀fluid_view[ModelingToolkit.variable_index(circ_sys, sym)] = val
end

OrdinaryDiffEqOperatorSplitting.recursive_null_parameters(::ModelingToolkit.System) = Thunderbolt.DiffEqBase.NullParameters()

problem = OperatorSplittingProblem(splitform, u₀, tspan)
integrator = init(problem, timestepper, dt=dt₀, verbose=true, adaptive=true; dtmax=10.0);

## f2 = Figure()
## axs = [
##     Axis(f2[1, 1], title="LV"),
##     Axis(f2[1, 2], title="RV"),
##     Axis(f2[2, 1], title="LA"),
##     Axis(f2[2, 2], title="RA")
## ]

## vlv = Observable(Float64[])
## plv = Observable(Float64[])

## vrv = Observable(Float64[])
## prv = Observable(Float64[])

## vla = Observable(Float64[])
## pla = Observable(Float64[])

## vra = Observable(Float64[])
## pra = Observable(Float64[])

## lines!(axs[1], vlv, plv)
## lines!(axs[2], vrv, prv)
## lines!(axs[3], vla, pla)
## lines!(axs[4], vra, pra)
## for i in 1:4
##     xlims!(axs[1], 0.0, 180.0)
##     ylims!(axs[1], 0.0, 180.0)
## end
## display(f2)
# !!! todo
#     recover online visualization of the pressure volume loop

# !!! todo
#     The post-processing API is not yet finished.
#     Please revisit the tutorial later to see how to post-process the simulation online.
#     Right now the solution is just exported into VTK, such that users can visualize the solution in e.g. ParaView.

# Now we can finally solve the coupled problem in time.
io = ParaViewWriter("CM03_3d0d-coupling");
for (u, t) in TimeChoiceIterator(integrator, tspan[1]:dtvis:tspan[2])
    chamber_function = OS.get_operator(splitform, 1)
    (; dh) = chamber_function.structural_function
    store_timestep!(io, t, dh.grid)
    Thunderbolt.store_timestep_field!(io, t, dh, u[1:ndofs(dh)], :displacement) # TODO allow views
    Thunderbolt.finalize_timestep!(io, t)

    ## if t > 0.0
    ##     lv = chamber_function.tying_info.chambers[1]
    ##     append!(vlv.val, lv.V⁰ᴰval)
    ##     append!(plv.val, u[lv.pressure_dof_index_global])
    ##     notify(vlv)
    ##     notify(plv)
    ## end
    ## TODO plot other chambers
end
# !!! tip
#     If you want to see more details of the solution process launch Julia with Thunderbolt as debug module:
#     ```
#     JULIA_DEBUG=Thunderbolt julia --project --threads=auto my_simulation_runner.jl
#     ```

#md # ## References
#md # ```@bibliography
#md # Pages = ["cm03_3d0d-coupling.md"]
#md # Canonical = false
#md # ```

#md # ## [Plain program](@id mechanics-tutorial_3d0dcoupling-plain-program)
#md #
#md # Here follows a version of the program without any comments.
#md # The file is also available here: [`cm03_3d0d-coupling.jl`](cm03_3d0d-coupling.jl).
#md #
#md # ```julia
#md # @__CODE__
#md # ```
