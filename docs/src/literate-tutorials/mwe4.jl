using CirculatorySystemModels, DynamicQuantities
using ModelingToolkit, OrdinaryDiffEqTsit5, OrdinaryDiffEqOperatorSplitting
using ModelingToolkit: t_nounits as t, D_nounits as D

function ϕRSAFDQ2022(t, tc, tr, TC, TR, THB)
    # c11 = 0 ≤ mod(t - tc, THB)
    # c12 = mod(t - tc, THB) ≤ TC
    
    # c21 = 0 ≤ mod(t - tr, THB)
    # c22 = mod(t - tr, THB) ≤ TR
    
    # return c11*c12 * 0.5(1-cos(π/TC*mod(t-tc,THB))) + c21*c22*0.5(1+cos(π/TR*mod(t-tr,THB)))

    c11 = 0 ≤ (t - tc)
    c12 = (t - tc) ≤ TC
    
    c21 = 0 ≤ (t - tr)
    c22 = (t - tr) ≤ TR
    
    return c11*c12 * 0.5(1-cos(π/TC*(t-tc))) + c21*c22*0.5(1+cos(π/TR*(t-tr)))
end

@mtkmodel LeakyResistorDiode begin
    @extend OnePort()
    @parameters begin
        Rₘᵢₙ
        Rₘₐₓ
    end
    @equations begin
            q ~ - (Δp / Rₘᵢₙ * (Δp < 0) + Δp / Rₘₐₓ * (Δp ≥ 0))
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
        0 ~ in.p - out.p
        p ~ in.p
    end
end

@mtkmodel RSAFDQ2022Periphery begin
    @components begin
        in  = Pin()
        out = Pin()
    end
    @parameters begin
        R
        C
        L
    end
    @variables begin
        q(t), [description = "Flow"]
        p(t), [description = "Pressure"]
    end
    @equations begin
        D(p)  ~ (in.q - q) / C
        D(q)  ~ - R/L   * ( q  + (out.p - p)/R)
        p ~ in.p
        q ~ out.q
    end
end

@mtkmodel RSAFDQ2022CircuitModular begin
    @structural_parameters begin
        τ   = 0.8 #, [unit = u"s", description = "Contraction cycle length"]
    end
    @parameters begin
        ## Systemic Circuit
        Rsysₐᵣ  = 0.8
        Csysₐᵣ  = 1.2
        Lsysₐᵣ  = 5e-3
        Rsysᵥₑₙ = 0.26
        Csysᵥₑₙ = 60.0
        Lsysᵥₑₙ = 5e-4
        ## Pulmonary Circuit
        Rpulₐᵣ  = 0.1625
        Cpulₐᵣ  = 10.0
        Lpulₐᵣ  = 5e-4
        Rpulᵥₑₙ = 0.1625
        Cpulᵥₑₙ = 16.0
        Lpulᵥₑₙ = 5e-4
        ## Valves
        Rₘᵢₙ = 0.0075
        Rₘₐₓ = 75000.0
        ## Left Atrium
        Epassₗₐ = 0.09
        Eactmaxₗₐ = 0.07
        V0ₗₐ = 4.0
        tCₗₐ = 0.6
        TCₗₐ = 0.104
        TRₗₐ = 0.68
        ## Right Atrium
        Epassᵣₐ = 0.07
        Eactmaxᵣₐ = 0.06
        V0ᵣₐ = 4.0
        TRᵣₐ = 0.56
        tCᵣₐ = 0.064
        TCᵣₐ = 0.64
        ## Left Ventricle
        Epassₗᵥ = 0.08
        Eactmaxₗᵥ = 2.75
        V0ₗᵥ = 5.0
        tCₗᵥ = 0.0
        TCₗᵥ = 0.272
        TRₗᵥ = 0.12
        ## Right Ventricle
        Epassᵣᵥ = 0.05
        Eactmaxᵣᵥ = 0.55
        V0ᵣᵥ = 10.0
        tCᵣᵥ = 0.0
        TCᵣᵥ = 0.272
        TRᵣᵥ = 0.12
        ## External pressure
        pₑₓ = 0.0
    end
    @components begin
        LV = RSAFDQ2022Chamber(Epass=Epassₗᵥ, Eactmax=Eactmaxₗᵥ, V0=V0ₗᵥ, tC=tCₗᵥ, TC=TCₗᵥ, TR=TRₗᵥ, τ, pₑₓ)
        LA = RSAFDQ2022Chamber(Epass=Epassₗₐ, Eactmax=Eactmaxₗₐ, V0=V0ₗₐ, tC=tCₗₐ, TC=TCₗₐ, TR=TRₗₐ, τ, pₑₓ)
        RV = RSAFDQ2022Chamber(Epass=Epassᵣᵥ, Eactmax=Eactmaxᵣᵥ, V0=V0ᵣᵥ, tC=tCᵣᵥ, TC=TCᵣᵥ, TR=TRᵣᵥ, τ, pₑₓ)
        RA = RSAFDQ2022Chamber(Epass=Epassᵣₐ, Eactmax=Eactmaxᵣₐ, V0=V0ᵣₐ, tC=tCᵣₐ, TC=TCᵣₐ, TR=TRᵣₐ, τ, pₑₓ)

        MV = LeakyResistorDiode(Rₘᵢₙ, Rₘₐₓ) # Mitral
        AV = LeakyResistorDiode(Rₘᵢₙ, Rₘₐₓ) # Aortic
        TV = LeakyResistorDiode(Rₘᵢₙ, Rₘₐₓ) # Triscupid
        PV = LeakyResistorDiode(Rₘᵢₙ, Rₘₐₓ) # Pulmonary

        SYSₐᵣ = RSAFDQ2022Periphery(R = Rsysₐᵣ, L = Lsysₐᵣ, C = Csysₐᵣ)
        SYSᵥₑₙ = RSAFDQ2022Periphery(R = Rsysᵥₑₙ, L = Lsysᵥₑₙ, C = Csysᵥₑₙ)
        PULₐᵣ = RSAFDQ2022Periphery(R = Rpulₐᵣ, L = Lpulₐᵣ, C = Cpulₐᵣ)
        PULᵥₑₙ = RSAFDQ2022Periphery(R = Rpulᵥₑₙ, L = Lpulᵥₑₙ, C = Cpulᵥₑₙ)
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

@named circ_model2 = RSAFDQ2022CircuitModular()
circ_sys2 = mtkcompile(circ_model2)
u02 = [
    circ_sys2.LA.V => 65.0
    circ_sys2.LV.V => 120.0
    circ_sys2.RA.V => 65.0
    circ_sys2.RV.V => 145.0
    circ_sys2.SYSₐᵣ.q => 0.0
    circ_sys2.SYSᵥₑₙ.q => 0.0
    circ_sys2.PULₐᵣ.q => 0.0
    circ_sys2.PULᵥₑₙ.q => 0.0
    circ_sys2.SYSₐᵣ.p => 80.0
    circ_sys2.SYSᵥₑₙ.p => 30.0
    circ_sys2.PULₐᵣ.p => 35.0
    circ_sys2.PULᵥₑₙ.p => 24.0
]
prob2 = OrdinaryDiffEqTsit5.ODEProblem(circ_sys2, u02, (0.0, 20.0))
sol2 = solve(prob2, Tsit5())
