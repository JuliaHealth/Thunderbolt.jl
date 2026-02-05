
module MTKModels

import ModelingToolkit as MTK
# import ModelingToolkit: t_no_units as t
using ModelingToolkit
using SciCompDSL: @mtkmodel

@independent_variables t

function Φ_RSAFDQ2022MTK(t, tc, tr, TC, TR, THB)
    tnow1 = mod(t - tc, THB)
    c11 = 0 ≤ mod(t - tc, THB)
    c12 = mod(t - tc, THB) ≤ TC

    tnow2 = mod(t - tr, THB)
    c21 = 0 ≤ mod(t - tr, THB)
    c22 = mod(t - tr, THB) ≤ TR

    return c11 * c12 * 0.5(1-cos(π/TC * tnow1)) + !(c11*c12)*c21*c22*0.5(1+cos(π/TR * tnow2))
end
elastance_RSAFDQ2022MTK(t, Epass, Emax, tC, tR, TC, TR, THB) =
    Epass + Emax*Φ_RSAFDQ2022MTK(t, tC, tR, TC, TR, THB)

@mtkmodel RSAFDQ2022CircuitMTK begin
    # Set these to false for external inputs
    @structural_parameters begin
        lv_pressure_given = true
        rv_pressure_given = true
        la_pressure_given = true
        ra_pressure_given = true
    end
    # These are the converted parameters from the paper [RegSalAfrFedDedQar:2022:cem](@cite) Appendix A
    @parameters begin
        if !lv_pressure_given
            external_input_lv_p = 0.0
        end
        if !rv_pressure_given
            external_input_rv_p = 0.0
        end
        if !la_pressure_given
            external_input_la_p = 0.0
        end
        if !ra_pressure_given
            external_input_ra_p = 0.0
        end
        ## Systemic Circuit
        Rsysₐᵣ = 106.65789473684211#ustrip(0.800u"mmHg*s/mL" |> us"kPa*ms/mL")
        Csysₐᵣ = 9.000740192450037#ustrip(1.2u"mL/mmHg" |> us"mL/kPa")
        Lsysₐᵣ = 666.6118421052632#ustrip(5e-3u"mmHg*s^2/mL" |> us"kPa*ms^2/mL")
        Rsysᵥₑₙ = 34.66381578947368#ustrip(0.260u"mmHg*s/mL" |> us"kPa*ms/mL")
        Csysᵥₑₙ = 1200.0986923266717#ustrip(160.0u"mL/mmHg" |> us"mL/kPa")
        Lsysᵥₑₙ = 66.66118421052632#ustrip(5e-4u"mmHg*s^2/mL" |> us"kPa*ms^2/mL")
        ## Pulmonary Circuit
        Rpulₐᵣ = 21.66488486842105#ustrip(0.1625u"mmHg*s/mL" |> us"kPa*ms/mL")
        Cpulₐᵣ = 75.00616827041698#ustrip(10.00u"mL/mmHg" |> us"mL/kPa")
        Lpulₐᵣ = 66.66118421052632#ustrip(5e-4u"mmHg*s^2/mL" |> us"kPa*ms^2/mL")
        Rpulᵥₑₙ = 21.66488486842105#ustrip(0.1625u"mmHg*s/mL" |> us"kPa*ms/mL")
        Cpulᵥₑₙ = 120.00986923266716#ustrip(16.00u"mL/mmHg" |> us"mL/kPa")
        Lpulᵥₑₙ = 66.66118421052632#ustrip(5e-4u"mmHg*s^2/mL" |> us"kPa*ms^2/mL")
        ## Valves
        Rₘᵢₙ = 1.0#ustrip(0.0075u"mmHg*s/mL" |> us"kPa*ms/mL")
        Rₘₐₓ = 9.999177631578946e6#ustrip(75000.0u"mmHg*s/mL" |> us"kPa*ms/mL")
        ## Left Atrium
        if la_pressure_given
            Epassₗₐ   = 0.011999013157894737#ustrip(0.09u"mmHg/mL" |> us"kPa/mL")
            Eactmaxₗₐ = 0.009332565789473684#ustrip(0.07u"mmHg/mL" |> us"kPa/mL")
            V0ₗₐ      = 4.0#ustrip(4.0u"mL" |> us"mL")
            tCₗₐ      = 600.0#ustrip(0.6u"s" |> us"ms")
            TCₗₐ      = 104.0#ustrip(0.104u"s" |> us"ms")
            TRₗₐ      = 680.0#ustrip(0.68u"s" |> us"ms")
        end
        ## Right Atrium
        if ra_pressure_given
            Epassᵣₐ   = 0.009332565789473684#ustrip(0.07u"mmHg/mL" |> us"kPa/mL")
            Eactmaxᵣₐ = 0.007999342105263157#ustrip(0.06u"mmHg/mL" |> us"kPa/mL")
            V0ᵣₐ      = 4.0#ustrip(4.0u"mL" |> us"mL")
            TRᵣₐ      = 560.0#ustrip(0.56u"s" |> us"ms")
            tCᵣₐ      = 64.0#ustrip(0.064u"s" |> us"ms")
            TCᵣₐ      = 640.0#ustrip(0.64u"s" |> us"ms")
        end
        ## Right Ventricle
        if rv_pressure_given
            Epassᵣᵥ   = 0.0066661184210526315#ustrip(0.05u"mmHg/mL" |> us"kPa/mL")
            Eactmaxᵣᵥ = 0.07332730263157895#ustrip(0.55u"mmHg/mL" |> us"kPa/mL")
            V0ᵣᵥ      = 10.0#ustrip(10.0u"mL" |> us"mL")
            tCᵣᵥ      = 0.0#ustrip(0.0u"s" |> us"ms")
            TCᵣᵥ      = 272.0#ustrip(0.272u"s" |> us"ms")
            TRᵣᵥ      = 120.0#ustrip(0.12u"s" |> us"ms")
        end
        ## Left Ventricle
        if lv_pressure_given
            Epassₗᵥ = 0.01066578947368421 # ustrip(0.08u"mmHg/mL" |> us"kPa/mL")
            Eactmaxₗᵥ = 0.3666365131578947 # ustrip(2.75u"mmHg/mL" |> us"kPa/mL")
            V0ₗᵥ = 5.0#ustrip(10.0u"mL" |> us"mL")
            tCₗᵥ = 0.0#ustrip(0.0u"s" |> us"ms")
            TCₗᵥ = 340.0#ustrip(0.340u"s" |> us"ms")
            TRₗᵥ = 170.0#ustrip(0.170u"s" |> us"ms")
        end
        ## External pressure
        pₑₓ = 0.0#ustrip(0.0u"mmHg" |> us"kPa")
        τ = 800.0#ustrip(0.8u"s" |> us"ms") #, [unit = u"ms", description = "Contraction cycle length"]
    end
    @variables begin
        Vₗₐ(t) = 65.0, [irreducible=true]
        Vₗᵥ(t) = 120.0, [irreducible=true]
        Vᵣₐ(t) = 65.0, [irreducible=true]
        Vᵣᵥ(t) = 145.0, [irreducible=true]
        psysₐᵣ(t) = 10.66
        psysᵥₑₙ(t) = 4.0
        ppulₐᵣ(t) = 4.67
        ppulᵥₑₙ(t) = 3.2
        Qsysₐᵣ(t) = 0.0
        Qsysᵥₑₙ(t) = 0.0
        Qpulₐᵣ(t) = 0.0
        Qpulᵥₑₙ(t) = 0.0
        if lv_pressure_given
            Eₗᵥ(t)
        end
        pₗᵥ(t)
        if rv_pressure_given
            Eᵣᵥ(t)
        end
        pᵣᵥ(t)
        if ra_pressure_given
            Eᵣₐ(t)
        end
        pᵣₐ(t)
        if la_pressure_given
            Eₗₐ(t)
        end
        pₗₐ(t)
        Qₘᵥ(t)
        Qₐᵥ(t)
        Qₚᵥ(t)
        Qₜᵥ(t)
        Qsysᵥₑₙsysₐᵣ(t)
        Qᵣₐsysᵥₑₙ(t)
        Qpulᵥₑₙpulₐᵣ(t)
        Qₗₐpulᵥₑₙ(t)
    end
    @equations begin
        # Valves
        Qₘᵥ ~ (pₗₐ - pₗᵥ) / ifelse(pₗₐ > pₗᵥ, Rₘᵢₙ, Rₘₐₓ) #Qᵢ(pₗₐ, pₗᵥ)
        Qₐᵥ ~ (pₗᵥ - psysₐᵣ) / ifelse(pₗᵥ > psysₐᵣ, Rₘᵢₙ, Rₘₐₓ)#Qᵢ(pₗᵥ, psysₐᵣ)
        Qₜᵥ ~ (pᵣₐ - pᵣᵥ) / ifelse(pᵣₐ > pᵣᵥ, Rₘᵢₙ, Rₘₐₓ)#Qᵢ(pᵣₐ, pᵣᵥ)
        Qₚᵥ ~ (pᵣᵥ - ppulₐᵣ) / ifelse(pᵣᵥ > ppulₐᵣ, Rₘᵢₙ, Rₘₐₓ)#Qᵢ(pᵣᵥ, ppulₐᵣ)
        # LV
        if !lv_pressure_given
            pₗᵥ ~ external_input_lv_p
        else
            Eₗᵥ ~ elastance_RSAFDQ2022MTK(t, Epassₗᵥ, Eactmaxₗᵥ, tCₗᵥ, tCₗᵥ + TCₗᵥ, TCₗᵥ, TRₗᵥ, τ)
            pₗᵥ ~ Eₗᵥ*(Vₗᵥ - V0ₗᵥ)
        end
        MTK.D(Vₗᵥ) ~ Qₘᵥ - Qₐᵥ
        # RV
        if !rv_pressure_given
            pᵣᵥ ~ external_input_rv_p
        else
            Eᵣᵥ ~ elastance_RSAFDQ2022MTK(t, Epassᵣᵥ, Eactmaxᵣᵥ, tCᵣᵥ, tCᵣᵥ + TCᵣᵥ, TCᵣᵥ, TRᵣᵥ, τ)
            pᵣᵥ ~ Eᵣᵥ*(Vᵣᵥ - V0ᵣᵥ)
        end
        MTK.D(Vᵣᵥ) ~ Qₜᵥ - Qₚᵥ
        # LA
        if !la_pressure_given
            pₗₐ ~ external_input_la_p
        else
            Eₗₐ ~ elastance_RSAFDQ2022MTK(t, Epassₗₐ, Eactmaxₗₐ, tCₗₐ, tCₗₐ + TCₗₐ, TCₗₐ, TRₗₐ, τ)
            pₗₐ ~ Eₗₐ*(Vₗₐ - V0ₗₐ)
        end
        MTK.D(Vₗₐ) ~ Qpulᵥₑₙ - Qₘᵥ
        # RA
        if !ra_pressure_given
            pᵣₐ ~ external_input_ra_p
        else
            Eᵣₐ ~ elastance_RSAFDQ2022MTK(t, Epassᵣₐ, Eactmaxᵣₐ, tCᵣₐ, tCᵣₐ + TCᵣₐ, TCᵣₐ, TRᵣₐ, τ)
            pᵣₐ ~ Eᵣₐ*(Vᵣₐ - V0ᵣₐ)
        end
        MTK.D(Vᵣₐ) ~ Qsysᵥₑₙ - Qₜᵥ
        # Pressure and flow
        MTK.D(psysₐᵣ) ~ (Qₐᵥ - Qsysₐᵣ) / Csysₐᵣ
        MTK.D(psysᵥₑₙ) ~ (Qsysₐᵣ - Qsysᵥₑₙ) / Csysᵥₑₙ
        MTK.D(ppulₐᵣ) ~ (Qₚᵥ - Qpulₐᵣ) / Cpulₐᵣ
        MTK.D(ppulᵥₑₙ) ~ (Qpulₐᵣ - Qpulᵥₑₙ) / Cpulᵥₑₙ
        Qsysᵥₑₙsysₐᵣ ~ (psysᵥₑₙ - psysₐᵣ) / Rsysₐᵣ
        MTK.D(Qsysₐᵣ) ~ - Rsysₐᵣ/Lsysₐᵣ * (Qsysₐᵣ + Qsysᵥₑₙsysₐᵣ) # sys ar
        Qᵣₐsysᵥₑₙ ~ (pᵣₐ - psysᵥₑₙ) / Rsysᵥₑₙ
        MTK.D(Qsysᵥₑₙ) ~ - Rsysᵥₑₙ/Lsysᵥₑₙ * (Qsysᵥₑₙ + Qᵣₐsysᵥₑₙ) # sys ven
        Qpulᵥₑₙpulₐᵣ ~ (ppulᵥₑₙ - ppulₐᵣ) / Rpulₐᵣ
        MTK.D(Qpulₐᵣ) ~ - Rpulₐᵣ/Lpulₐᵣ * (Qpulₐᵣ + Qpulᵥₑₙpulₐᵣ) # pul ar
        Qₗₐpulᵥₑₙ ~ (pₗₐ - ppulᵥₑₙ) / Rpulᵥₑₙ
        MTK.D(Qpulᵥₑₙ) ~ - Rpulᵥₑₙ/Lpulᵥₑₙ * (Qpulᵥₑₙ + Qₗₐpulᵥₑₙ) # sys ar
    end
end

end
