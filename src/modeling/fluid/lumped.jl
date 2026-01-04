abstract type AbstractLumpedCirculatoryModel end

function solution_size(model::AbstractLumpedCirculatoryModel)
    return num_states(model)
end

# TODO SciMLParameter interface
struct LumpedCirculatoryModelFunction{M, T}
    m::M
    p::Vector{T}
end

Base.setindex!(m::LumpedCirculatoryModelFunction{<:Any, T}, val::T, i::Int) where {T} = m.p[i] = val

#FIXME
OS.recursive_null_parameters(f::LumpedCirculatoryModelFunction) = SciMLBase.NullParameters()

function ODEFunction(model::AbstractLumpedCirculatoryModel)
    return LumpedCirculatoryModelFunction(model, zeros(num_unknown_pressures(model)))
end

function DiffEqBase.ODEProblem(f::LumpedCirculatoryModelFunction, u0, tspan, p_; kwargs...)
    fun = DiffEqBase.ODEFunction((du, u, p, t) -> lumped_driver!(du, u, t, p.p, p.m))
    prob = DiffEqBase.ODEProblem(fun, u0, tspan, f; kwargs...)
    return prob
end

"""
    DummyLumpedCircuitModel(volume_fun)

Lock the volume at a certain value.
"""
struct DummyLumpedCircuitModel{F} <: AbstractLumpedCirculatoryModel
    volume_fun::F
end

get_variable_symbol_index(model::DummyLumpedCircuitModel, symbol::Symbol) = 1

num_states(::DummyLumpedCircuitModel) = 1
num_unknown_pressures(::DummyLumpedCircuitModel) = 1

function default_initial_condition!(u, model::DummyLumpedCircuitModel)
    u[1] = model.volume_fun(0.0)
end

function (model::DummyLumpedCircuitModel)(du, u, p, t)
    du[1] = model.volume_fun(t)-u[1]
end

"""
    ΦRSAFDQ2022(t,tC,tR,TC,TR,THB)

Activation transient from the paper [RegSalAfrFedDedQar:2022:cem](@citet).

     t  = time
   THB  = time for a full heart beat
[tC,TC] = contraction period
[tR,TR] = relaxation period
"""
function Φ_RSAFDQ2022(t, tC, tR, TC, TR, THB)
    tnow = mod(t - tC, THB)
    if 0 ≤ tnow < TC
        return 1/2 * (1-cos(π/TC * tnow))
    end
    tnow = mod(t - tR, THB)
    if 0 ≤ tnow < TR
        return 1/2 * (1+cos(π/TR * tnow))
    end
    return 0.0
end

elastance_RSAFDQ2022(t, Epass, Emax, tC, tR, TC, TR, THB) =
    Epass + Emax*Φ_RSAFDQ2022(t, tC, tR, TC, TR, THB)


"""
    RSAFDQ2022LumpedCicuitModel

A lumped (0D) circulatory model for LV simulations as presented in [RegSalAfrFedDedQar:2022:cem](@citet).
"""
Base.@kwdef struct RSAFDQ2022LumpedCicuitModel{
    T1, # kPa ms mL^-1
    T2, # mL kPa^-1
    T3, # mL
    T4, # ms
    T5, # kPa ms^2 mL^-1
    T6, # kPa mL^-1
    T7, # kPa
} <: AbstractLumpedCirculatoryModel
    lv_pressure_given::Bool = true
    rv_pressure_given::Bool = true
    la_pressure_given::Bool = true
    ra_pressure_given::Bool = true
    ## Systemic Circuit
    Rsysₐᵣ::T1 = 106.6578947368421 #ustrip(0.800u"mmHg*s/mL" |> us"kPa*ms/mL")
    Csysₐᵣ::T2 = 9.000740192450037 #ustrip(1.2u"mL/mmHg" |> us"mL/kPa")
    Lsysₐᵣ::T5 = 666.6118421052632 #ustrip(5e-3u"mmHg*s^2/mL" |> us"kPa*ms^2/mL")
    Rsysᵥₑₙ::T1 = 34.66381578947368 #ustrip(0.260u"mmHg*s/mL" |> us"kPa*ms/mL")
    Csysᵥₑₙ::T2 = 1200.098692326671 #ustrip(160.0u"mL/mmHg" |> us"mL/kPa")
    Lsysᵥₑₙ::T5 = 66.66118421052632 #ustrip(5e-4u"mmHg*s^2/mL" |> us"kPa*ms^2/mL")
    ## Pulmonary Circuit
    Rpulₐᵣ::T1 = 21.66488486842105 #ustrip(0.1625u"mmHg*s/mL" |> us"kPa*ms/mL")
    Cpulₐᵣ::T2 = 75.00616827041698 #ustrip(10.00u"mL/mmHg" |> us"mL/kPa")
    Lpulₐᵣ::T5 = 66.66118421052632 #ustrip(5e-4u"mmHg*s^2/mL" |> us"kPa*ms^2/mL")
    Rpulᵥₑₙ::T1 = 21.66488486842105 #ustrip(0.1625u"mmHg*s/mL" |> us"kPa*ms/mL")
    Cpulᵥₑₙ::T2 = 120.0098692326671 #ustrip(16.00u"mL/mmHg" |> us"mL/kPa")
    Lpulᵥₑₙ::T5 = 66.66118421052632 #ustrip(5e-4u"mmHg*s^2/mL" |> us"kPa*ms^2/mL")
    ## Valves
    Rₘᵢₙ::T1 = 1.0     #ustrip(0.0075u"mmHg*s/mL" |> us"kPa*ms/mL")
    Rₘₐₓ::T1 = 9.999e6 #ustrip(75000.0u"mmHg*s/mL" |> us"kPa*ms/mL")
    ## Left Atrium
    Epassₗₐ::T6   = 0.011999013157894737#ustrip(0.09u"mmHg/mL" |> us"kPa/mL")
    Eactmaxₗₐ::T6 = 0.009332565789473684#ustrip(0.07u"mmHg/mL" |> us"kPa/mL")
    V0ₗₐ::T3      = 4.0#ustrip(4.0u"mL" |> us"mL")
    tCₗₐ::T4      = 600.0#ustrip(0.6u"s" |> us"ms")
    TCₗₐ::T4      = 104.0#ustrip(0.104u"s" |> us"ms")
    TRₗₐ::T4      = 680.0#ustrip(0.68u"s" |> us"ms")
    ## Right Atrium
    Epassᵣₐ::T6   = 0.009332565789473684#ustrip(0.07u"mmHg/mL" |> us"kPa/mL")
    Eactmaxᵣₐ::T6 = 0.007999342105263157#ustrip(0.06u"mmHg/mL" |> us"kPa/mL")
    V0ᵣₐ::T3      = 4.0#ustrip(4.0u"mL" |> us"mL")
    TRᵣₐ::T4      = 560.0#ustrip(0.56u"s" |> us"ms")
    tCᵣₐ::T4      = 64.0#ustrip(0.064u"s" |> us"ms")
    TCᵣₐ::T4      = 640.0#ustrip(0.64u"s" |> us"ms")
    ## Right Ventricle
    Epassᵣᵥ::T6   = 0.0066661184210526315#ustrip(0.05u"mmHg/mL" |> us"kPa/mL")
    Eactmaxᵣᵥ::T6 = 0.07332730263157895#ustrip(0.55u"mmHg/mL" |> us"kPa/mL")
    V0ᵣᵥ::T3      = 10.0#ustrip(10.0u"mL" |> us"mL")
    tCᵣᵥ::T4      = 0.0#ustrip(0.0u"s" |> us"ms")
    TCᵣᵥ::T4      = 272.0#ustrip(0.272u"s" |> us"ms")
    TRᵣᵥ::T4      = 120.0#ustrip(0.12u"s" |> us"ms")
    ## Left Ventricle
    Epassₗᵥ::T6 = 0.01066578947368421 # ustrip(0.08u"mmHg/mL" |> us"kPa/mL")
    Eactmaxₗᵥ::T6 = 0.3666365131578947 # ustrip(2.75u"mmHg/mL" |> us"kPa/mL")
    V0ₗᵥ::T3 = 5.0#ustrip(10.0u"mL" |> us"mL")
    tCₗᵥ::T4 = 0.0#ustrip(0.0u"s" |> us"ms")
    TCₗᵥ::T4 = 340.0#ustrip(0.340u"s" |> us"ms")
    TRₗᵥ::T4 = 170.0#ustrip(0.170u"s" |> us"ms")
    ## External pressure
    pₑₓ::T7 = 0.0#ustrip(0.0u"mmHg" |> us"kPa")
    ## Global
    THB::T4 = 800.0 # 75 beats per minute
end

num_states(::RSAFDQ2022LumpedCicuitModel) = 12
num_unknown_pressures(model::RSAFDQ2022LumpedCicuitModel) =
    Int(!model.lv_pressure_given) +
    Int(!model.rv_pressure_given) +
    Int(!model.la_pressure_given) +
    Int(!model.ra_pressure_given)
function get_variable_symbol_index(model::RSAFDQ2022LumpedCicuitModel, symbol::Symbol)
    @unpack lv_pressure_given, la_pressure_given, ra_pressure_given, rv_pressure_given = model

    # Try to query index
    symbol == :Vₗₐ && return 1
    symbol == :Vₗᵥ && return 2
    symbol == :Vᵣₐ && return 3
    symbol == :Vᵣᵥ && return 4

    # Diagnostics
    valid_symbols = Set{Symbol}()
    push!(valid_symbols, :Vₗₐ)
    push!(valid_symbols, :Vₗᵥ)
    push!(valid_symbols, :Vᵣₐ)
    push!(valid_symbols, :Vᵣᵥ)
    @error "Variable named '$symbol' not found. The following symbols are defined and accessible: $valid_symbols."
end

function get_parameter_symbol_index(model::RSAFDQ2022LumpedCicuitModel, symbol::Symbol)
    @unpack lv_pressure_given, la_pressure_given, ra_pressure_given, rv_pressure_given = model

    # Try to query index
    symbol == :pₗₐ && return lumped_circuit_relative_la_pressure_index(model)
    symbol == :pₗᵥ && return lumped_circuit_relative_lv_pressure_index(model)
    symbol == :pᵣₐ && return lumped_circuit_relative_ra_pressure_index(model)
    symbol == :pᵣᵥ && return lumped_circuit_relative_rv_pressure_index(model)

    # Diagnostics
    valid_symbols = Set{Symbol}([])
    la_pressure_given && push!(valid_symbols, :pₗₐ)
    lv_pressure_given && push!(valid_symbols, :pₗᵥ)
    ra_pressure_given && push!(valid_symbols, :pᵣₐ)
    rv_pressure_given && push!(valid_symbols, :pᵣᵥ)

    @error "Variable named '$symbol' not found for model $model. The following symbols are defined and accessible: $valid_symbols."
end

function default_initial_condition!(u, model::RSAFDQ2022LumpedCicuitModel)
    # obtain via pre-pacing in isolation
    u .= [65.0, 120.0, 65.0, 145.0, 10.66, 4.0, 4.67, 3.2, 0.0, 0.0, 0.0, 0.0]
end

function lumped_circuit_relative_lv_pressure_index(model::RSAFDQ2022LumpedCicuitModel)
    model.lv_pressure_given &&
        @error "Trying to query extenal LV pressure index, but LV pressure is not an external input!"
    return 1
end
function lumped_circuit_relative_rv_pressure_index(model::RSAFDQ2022LumpedCicuitModel)
    model.rv_pressure_given &&
        @error "Trying to query extenal RV pressure index, but RV pressure is not an external input!"
    i = 1
    if model.lv_pressure_given
        i+=1
    end
    return i
end
function lumped_circuit_relative_la_pressure_index(model::RSAFDQ2022LumpedCicuitModel)
    model.la_pressure_given &&
        @error "Trying to query extenal LA pressure index, but LA pressure is not an external input!"
    i = 1
    if model.lv_pressure_given
        i+=1
    end
    if model.rv_pressure_given
        i+=1
    end
    return i
end
function lumped_circuit_relative_ra_pressure_index(model::RSAFDQ2022LumpedCicuitModel)
    model.la_pressure_given &&
        @error "Trying to query extenal RA pressure index, but RA pressure is not an external input!"
    i = 1
    if model.lv_pressure_given
        i+=1
    end
    if model.rv_pressure_given
        i+=1
    end
    if model.la_pressure_given
        i+=1
    end
    return i
end

function lumped_driver!(
    du,
    u,
    t,
    external_input::AbstractVector,
    model::RSAFDQ2022LumpedCicuitModel,
)
    # Evaluate the right hand side of equation system (6) in the paper
    # V = volume
    # p = pressure
    # Q = flow rates
    # E = elastance
    # [x]v = ventricle [x]
    Vₗₐ, Vₗᵥ, Vᵣₐ, Vᵣᵥ, psysₐᵣ, psysᵥₑₙ, ppulₐᵣ, ppulᵥₑₙ, Qsysₐᵣ, Qsysᵥₑₙ, Qpulₐᵣ, Qpulᵥₑₙ = u

    # Note tR = tC+TC
    @inline Eₗᵥ(p::RSAFDQ2022LumpedCicuitModel, t) = elastance_RSAFDQ2022(
        t,
        p.Epassₗᵥ,
        p.Eactmaxₗᵥ,
        p.tCₗᵥ,
        p.tCₗᵥ + p.TCₗᵥ,
        p.TCₗᵥ,
        p.TRₗᵥ,
        p.THB,
    )
    @inline Eᵣᵥ(p::RSAFDQ2022LumpedCicuitModel, t) = elastance_RSAFDQ2022(
        t,
        p.Epassᵣᵥ,
        p.Eactmaxᵣᵥ,
        p.tCᵣᵥ,
        p.tCᵣᵥ + p.TCᵣᵥ,
        p.TCᵣᵥ,
        p.TRᵣᵥ,
        p.THB,
    )
    @inline Eₗₐ(p::RSAFDQ2022LumpedCicuitModel, t) = elastance_RSAFDQ2022(
        t,
        p.Epassₗₐ,
        p.Eactmaxₗₐ,
        p.tCₗₐ,
        p.tCₗₐ + p.TCₗₐ,
        p.TCₗₐ,
        p.TRₗₐ,
        p.THB,
    )
    @inline Eᵣₐ(p::RSAFDQ2022LumpedCicuitModel, t) = elastance_RSAFDQ2022(
        t,
        p.Epassᵣₐ,
        p.Eactmaxᵣₐ,
        p.tCᵣₐ,
        p.tCᵣₐ + p.TCᵣₐ,
        p.TCᵣₐ,
        p.TRᵣₐ,
        p.THB,
    )

    (; V0ₗₐ, V0ᵣₐ, V0ᵣᵥ, V0ₗᵥ) = model
    (; Rsysₐᵣ, Rpulₐᵣ, Rsysᵥₑₙ, Rpulᵥₑₙ) = model
    (; Csysₐᵣ, Cpulₐᵣ, Csysᵥₑₙ, Cpulᵥₑₙ) = model
    (; Lsysₐᵣ, Lpulₐᵣ, Lsysᵥₑₙ, Lpulᵥₑₙ) = model

    #pₑₓ = 0.0 # External pressure created by organs
    pₗᵥ =
        model.lv_pressure_given ? Eₗᵥ(model, t)*(Vₗᵥ - V0ₗᵥ) :
        external_input[lumped_circuit_relative_lv_pressure_index(model)]
    pᵣᵥ =
        model.rv_pressure_given ? Eᵣᵥ(model, t)*(Vᵣᵥ - V0ᵣᵥ) :
        external_input[lumped_circuit_relative_rv_pressure_index(model)]
    pₗₐ =
        model.la_pressure_given ? Eₗₐ(model, t)*(Vₗₐ - V0ₗₐ) :
        external_input[lumped_circuit_relative_la_pressure_index(model)]
    pᵣₐ =
        model.ra_pressure_given ? Eᵣₐ(model, t)*(Vᵣₐ - V0ᵣₐ) :
        external_input[lumped_circuit_relative_ra_pressure_index(model)]

    @inline Rᵢ(p₁, p₂, p) = p₁ > p₂ ? p.Rₘᵢₙ : p.Rₘₐₓ # Resistance
    @inline Qᵢ(p₁, p₂, model) = (p₁ - p₂) / Rᵢ(p₁, p₂, model)
    Qₘᵥ = Qᵢ(pₗₐ, pₗᵥ, model)
    Qₐᵥ = Qᵢ(pₗᵥ, psysₐᵣ, model)
    Qₜᵥ = Qᵢ(pᵣₐ, pᵣᵥ, model)
    Qₚᵥ = Qᵢ(pᵣᵥ, ppulₐᵣ, model)

    # change in volume
    du[1] = Qpulᵥₑₙ - Qₘᵥ # LA
    du[2] = Qₘᵥ - Qₐᵥ # LV
    du[3] = Qsysᵥₑₙ - Qₜᵥ # RA
    du[4] = Qₜᵥ - Qₚᵥ # RV

    # Pressure change
    du[5] = (Qₐᵥ - Qsysₐᵣ) / Csysₐᵣ # sys ar
    du[6] = (Qsysₐᵣ - Qsysᵥₑₙ) / Csysᵥₑₙ # sys ven
    du[7] = (Qₚᵥ - Qpulₐᵣ) / Cpulₐᵣ # pul ar
    du[8] = (Qpulₐᵣ - Qpulᵥₑₙ) / Cpulᵥₑₙ # pul ven

    # Flows
    Q9     = (psysᵥₑₙ - psysₐᵣ) / Rsysₐᵣ
    du[9]  = - Rsysₐᵣ/Lsysₐᵣ * (Qsysₐᵣ + Q9) # sys ar
    Q10    = (pᵣₐ - psysᵥₑₙ) / Rsysᵥₑₙ
    du[10] = - Rsysᵥₑₙ/Lsysᵥₑₙ * (Qsysᵥₑₙ + Q10) # sys ven
    Q11    = (ppulᵥₑₙ - ppulₐᵣ) / Rpulₐᵣ
    du[11] = - Rpulₐᵣ/Lpulₐᵣ * (Qpulₐᵣ + Q11) # pul ar
    Q12    = (pₗₐ - ppulᵥₑₙ) / Rpulᵥₑₙ
    du[12] = - Rpulᵥₑₙ/Lpulᵥₑₙ * (Qpulᵥₑₙ + Q12) # sys ar
end
