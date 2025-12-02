# Taken from https://github.com/TS-CUBED/CirculatorySystemModels.jl/blob/3095ac17c9df8d51e52f9b115b0a9d0833e8be1f/test/runtests.jl
using CirculatorySystemModels.OrdinaryDiffEq,ModelingToolkit, CirculatorySystemModels
using GLMakie

τ = 1.0
Eshift=0.0
Ev=Inf
#### LV chamber parameters #### Checked
v0_lv = 5.0
p0_lv = 1.0
Emin_lv = 0.1
Emax_lv = 2.5
τes_lv = 0.3
τed_lv = 0.45
Eshift_lv = 0.0
#### RV Chamber parameters #### Checked
v0_rv = 10.0
p0_rv = 1.0
Emin_rv = 0.1
Emax_rv = 1.15
τes_rv = 0.3
τed_rv = 0.45
Eshift_rv = 0.0
### LA Atrium Parameters #### Checked
v0_la = 4.0
p0_la = 1.0
Emin_la = 0.15
Emax_la = 0.25
τpwb_la = 0.92
τpww_la = 0.09
τes_la = τpww_la/2
τed_la = τpww_la
Eshift_la = τpwb_la
### RA Atrium parameters #### Checked
v0_ra = 4.0
p0_ra = 1.0
Emin_ra = 0.15
Emax_ra = 0.25
τpwb_ra = 0.92
τpww_ra = 0.09
τes_ra = τpww_ra/2
τed_ra = τpww_ra
Eshift_ra = τpwb_ra
#### Valve parameters #### Checked
CQ_AV = 350.0
CQ_MV = 400.0
CQ_TV = 400.0
CQ_PV = 350.0
## Systemic Aortic Sinus #### Checked
Csas = 0.08
Rsas = 0.003
Lsas = 6.2e-5
pt0sas = 100.0
qt0sas = 0.0
## Systemic Artery #### Checked
Csat = 1.6
Rsat = 0.05
Lsat = 0.0017
pt0sat = 100.0
qt0sat = 0.0
## Systemic Arteriole #### Checked
Rsar = 0.5
## Systemic Capillary #### Checked 
Rscp = 0.52
## Systemic Vein #### Checked
Csvn = 20.5
Rsvn = 0.075
pt0svn = 0.0
qt0svn = 0.0
## Pulmonary Aortic Sinus #### Checked
Cpas = 0.18
Rpas = 0.002
Lpas = 5.2e-5
pt0pas = 30.0
qt0pas = 0.0
## Pulmonary Artery #### Checked
Cpat = 3.8
Rpat = 0.01
Lpat = 0.0017
pt0pat = 30.0
qt0pat = 0.0
## Pulmonary Arteriole #### Checked
Rpar = 0.05
## Pulmonary Capillary #### Checked
Rpcp = 0.25
## Pulmonary Vein #### Checked
Cpvn = 20.5
Rpvn = 0.006           # this was 0.006 originally and in the paper, seems to be wrong in the paper!
# CHANGED THIS IN THE CELLML MODEL AS WELL TO MATCH THE PAPER!!!!!
pt0pvn = 0.0
qt0pvn = 0.0
## KG diaphragm ## Not in cellML model
# left heart #
Kst_la = 2.5
Kst_lv = 20.0
Kf_sav = 0.0004
Ke_sav = 9000.0
M_sav = 0.0004
A_sav = 0.00047
# right heart # 
Kst_ra = 2.5
Kst_rv = 20.0
Kf_pav = 0.0004
Ke_pav = 9000.0
M_pav = 0.0004
A_pav = 0.00047
#
#### Diff valve params #### not in cellML model
Kp_av = 5500.0 # *  57.29578 # Shi Paper has values in radians!
Kf_av = 50.0
Kf_mv = 50.0
Kp_mv = 5500.0 # *  57.29578 
Kf_tv = 50.0
Kp_tv = 5500.0 # *  57.29578
Kf_pv = 50.0
Kp_pv = 5500.0 #*  57.29578
Kb_av = 2.0
Kv_av = 7.0
Kb_mv = 2.0
Kv_mv = 3.5
Kb_tv = 2.0
Kv_tv = 3.5
Kb_pv = 2.0
Kv_pv = 3.5
θmax_av = 75.0 * pi / 180
θmax_mv = 75.0 * pi / 180
θmin_av = 5.0 * pi / 180
θmin_mv = 5.0 * pi / 180
θmax_pv = 75.0 * pi / 180
θmax_tv = 75.0 * pi / 180
θmin_pv = 5.0 * pi / 180
θmin_tv = 5.0 * pi / 180

## pressure force and frictional force is the same for all 4 valves 

# Initial conditions #### Checked against cellML model

LV_Vt0 = 500
RV_Vt0 = 500
LA_Vt0 = 20
RA_Vt0 = 20

## Start Modelling
@variables t

## Ventricles
@named LV = ShiChamber(V₀=v0_lv, p₀=p0_lv, Eₘᵢₙ=Emin_lv, Eₘₐₓ=Emax_lv, τ=τ, τₑₛ=τes_lv, τₑₚ=τed_lv, Eshift=0.0)
# The atrium can be defined either as a ShiChamber with changed timing parameters, or as defined in the paper
@named LA = ShiChamber(V₀=v0_la, p₀=p0_la, Eₘᵢₙ=Emin_la, Eₘₐₓ=Emax_la, τ=τ, τₑₛ=τpww_la / 2, τₑₚ=τpww_la, Eshift=τpwb_la)
@named RV = ShiChamber(V₀=v0_rv, p₀=p0_rv, Eₘᵢₙ=Emin_rv, Eₘₐₓ=Emax_rv, τ=τ, τₑₛ=τes_rv, τₑₚ=τed_rv, Eshift=0.0)
# The atrium can be defined either as a ShiChamber with changed timing parameters, or as defined in the paper
@named RA = ShiChamber(V₀=v0_ra, p₀ = p0_ra, Eₘᵢₙ=Emin_ra, Eₘₐₓ=Emax_ra, τ=τ, τₑₛ=τpww_ra/2, τₑₚ =τpww_ra, Eshift=τpwb_ra)

## Valves as simple valves
@named AV = OrificeValve(CQ=CQ_AV)
@named MV = OrificeValve(CQ=CQ_MV)
@named TV = OrificeValve(CQ=CQ_TV)
@named PV = OrificeValve(CQ=CQ_PV)

####### Systemic Loop #######
# Systemic Aortic Sinus ##
@named SAS = CRL(C=Csas, R=Rsas, L=Lsas)
# Systemic Artery ##
@named SAT = CRL(C=Csat, R=Rsat, L=Lsat)
# Systemic Arteriole ##
@named SAR = Resistor(R=Rsar)
# Systemic Capillary ##
@named SCP = Resistor(R=Rscp)
# Systemic Vein ##
@named SVN = CR(R=Rsvn, C=Csvn)

####### Pulmonary Loop #######
# Pulmonary Aortic Sinus ##
@named PAS = CRL(C=Cpas, R=Rpas, L=Lpas)
# Pulmonary Artery ##
@named PAT = CRL(C=Cpat, R=Rpat, L=Lpat)
# Pulmonary Arteriole ##
@named PAR = Resistor(R=Rpar)
# Pulmonary Capillary ##
@named PCP = Resistor(R=Rpcp)
# Pulmonary Vein ##
@named PVN = CR(R=Rpvn, C=Cpvn)

##
circ_eqs = [
    connect(LV.out, AV.in)
    connect(AV.out, SAS.in)
    connect(SAS.out, SAT.in)
    connect(SAT.out, SAR.in)
    connect(SAR.out, SCP.in)
    connect(SCP.out, SVN.in)
    connect(SVN.out, RA.in)
    connect(RA.out, TV.in)
    connect(TV.out, RV.in)
    connect(RV.out, PV.in)
    connect(PV.out, PAS.in)
    connect(PAS.out, PAT.in)
    connect(PAT.out, PAR.in)
    connect(PAR.out, PCP.in)
    connect(PCP.out, PVN.in)
    connect(PVN.out, LA.in)
    connect(LA.out, MV.in)
    connect(MV.out, LV.in)
]

## Compose the whole ODE system
@named _circ_model = ODESystem(circ_eqs, t)
@named circ_model = compose(_circ_model,
    [LV, RV, LA, RA, AV, MV, PV, TV, SAS, SAT, SAR, SCP, SVN, PAS, PAT, PAR, PCP, PVN])

## And simplify it
circ_sys = structural_simplify(circ_model)

## Setup ODE
u0 = [
    LV.V => LV_Vt0
    LV.p => (LV_Vt0 - v0_lv) * Emin_lv + p0_lv
    RV.V => RV_Vt0
    RV.p => (RV_Vt0 - v0_rv) * Emin_rv + p0_rv
    LA.V => LA_Vt0
    RA.V => RA_Vt0
    SAS.C.p => pt0sas
    SAS.C.V => pt0sas * Csas
    SAS.L.q => qt0sas
    SAT.C.p => pt0sat
    SAT.C.V => pt0sat * Csat
    SAT.L.q => qt0sat
    SVN.C.p => pt0svn
    SVN.C.V => pt0svn * Csvn
    PAS.C.p => pt0pas
    PAS.C.V => pt0pas * Cpas
    PAS.L.q => qt0pas
    PAT.C.p => pt0pat
    PAT.C.V => pt0pat * Cpat
    PAT.L.q => qt0pat
    PVN.C.p => pt0pvn
    PVN.C.V => pt0pvn * Cpvn
]

prob = ODAEProblem(circ_sys, u0, (0.0, 20.0))
##
@time ShiSimpleSolV = solve(prob, Tsit5(), reltol=1e-9, abstol=1e-12, saveat=19:0.01:20)

f1 = Figure()
axs1 = [
    Axis(f1[1, 1], title="Volume"),
    Axis(f1[1, 2], title="Pressure")
]

lines!(axs1[1], ShiSimpleSolV.t,ShiSimpleSolV[LV.V])
lines!(axs1[1], ShiSimpleSolV.t,ShiSimpleSolV[RV.V])
lines!(axs1[1], ShiSimpleSolV.t,ShiSimpleSolV[LA.V])
lines!(axs1[1], ShiSimpleSolV.t,ShiSimpleSolV[RA.V])

lines!(axs1[2], ShiSimpleSolV.t,ShiSimpleSolV[LV.p])
lines!(axs1[2], ShiSimpleSolV.t,ShiSimpleSolV[RV.p])
lines!(axs1[2], ShiSimpleSolV.t,ShiSimpleSolV[LA.p])
lines!(axs1[2], ShiSimpleSolV.t,ShiSimpleSolV[RA.p])

f2 = Figure()
axs2 = [
    Axis(f2[1, 1], title="LV"),
    Axis(f2[1, 2], title="RV"),
    Axis(f2[2, 1], title="LA"),
    Axis(f2[2, 2], title="RA")
]

lines!(axs2[1], ShiSimpleSolV[LV.p],ShiSimpleSolV[LV.V])
lines!(axs2[2], ShiSimpleSolV[RV.p],ShiSimpleSolV[RV.V])
lines!(axs2[3], ShiSimpleSolV[LA.p],ShiSimpleSolV[LA.V])
lines!(axs2[4], ShiSimpleSolV[RA.p],ShiSimpleSolV[RA.V])
