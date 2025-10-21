abstract type AbstractLumpedCirculatoryModel end

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
function Φ_RSAFDQ2022(t,tC,tR,TC,TR,THB)
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

elastance_RSAFDQ2022(t,Epass,Emax,tC,tR,TC,TR,THB) = Epass + Emax*Φ_RSAFDQ2022(t,tC,tR,TC,TR,THB)


"""
    RSAFDQ2022LumpedCicuitModel

A lumped (0D) circulatory model for LV simulations as presented in [RegSalAfrFedDedQar:2022:cem](@citet).
"""
Base.@kwdef struct RSAFDQ2022LumpedCicuitModel{
    T1, # mmHg ms mL^-1
    T2, # mL mmHg^-1
    T3, # mL
    T4, # ms
    T5, # mmHg ms^2 mL^-1
    T6, # mmHg mL mL^-1
} <: AbstractLumpedCirculatoryModel
    lv_pressure_given::Bool = false
    rv_pressure_given::Bool = true
    la_pressure_given::Bool = true
    ra_pressure_given::Bool = true
    #
    Rsysₐᵣ::T1  = T1(0.8*1e3)
    Rpulₐᵣ::T1  = T1(0.1625*1e3)
    Rsysᵥₑₙ::T1 = T1(0.26*1e3)
    Rpulᵥₑₙ::T1 = T1(0.1625*1e3)
    #
    Csysₐᵣ::T2  = T2(1.2)
    Cpulₐᵣ::T2  = T2(10.0)
    Csysᵥₑₙ::T2 = T2(60.0)
    Cpulᵥₑₙ::T2 = T2(16.0)
    #
    Lsysₐᵣ::T5  = T5(5e-3*1e6)
    Lpulₐᵣ::T5  = T5(5e-4*1e6)
    Lsysᵥₑₙ::T5 = T5(5e-4*1e6)
    Lpulᵥₑₙ::T5 = T5(5e-4*1e6)
    # Valve stuff
    Rmin::T1 = T1(0.0075*1e3)
    Rmax::T1 = T1(75000.0*1e3)
    # Passive elastance
    Epassₗₐ::T6 = T6(0.09)
    Epassᵣₐ::T6 = T6(0.07)
    Epassᵣᵥ::T6 = T6(0.05)
    # Active elastance
    Eactmaxₗₐ::T6 = T6(0.07)
    Eactmaxᵣₐ::T6 = T6(0.06)
    Eactmaxᵣᵥ::T6 = T6(0.55)
    # "Initial volume"
    V0ₗₐ::T3 = T3(4.0)
    V0ᵣₐ::T3 = T3(4.0)
    V0ᵣᵥ::T3 = T3(10.0)
    # Event timings
    tCₗₐ::T4 = T4(0.6*1e3)
    TCₗₐ::T4 = T4(0.104*1e3)
    TRₗₐ::T4 = T4(0.68*1e3)
    TRᵣₐ::T4 = T4(0.56*1e3)
    tCᵣₐ::T4 = T4(0.064*1e3)
    TCᵣₐ::T4 = T4(0.64*1e3)
    tCᵣᵥ::T4 = T4(0.0*1e3)
    TCᵣᵥ::T4 = T4(0.272*1e3)
    TRᵣᵥ::T4 = T4(0.12*1e3)
    THB::T4 = T4(0.8*1e3) # 75 beats per minute
end

num_states(::RSAFDQ2022LumpedCicuitModel) = 12
num_unknown_pressures(model::RSAFDQ2022LumpedCicuitModel) = Int(!model.lv_pressure_given) + Int(!model.rv_pressure_given) + Int(!model.la_pressure_given) + Int(!model.ra_pressure_given)
function get_variable_symbol_index(model::RSAFDQ2022LumpedCicuitModel, symbol::Symbol)
    @unpack lv_pressure_given, la_pressure_given, ra_pressure_given, rv_pressure_given = model

    # Try to query index
    symbol == :Vₗₐ && return lumped_circuit_relative_la_pressure_index(model)
    symbol == :Vₗᵥ && return lumped_circuit_relative_lv_pressure_index(model)
    symbol == :Vᵣₐ && return lumped_circuit_relative_ra_pressure_index(model)
    symbol == :Vᵣᵥ && return lumped_circuit_relative_rv_pressure_index(model)

    # Diagnostics
    valid_symbols = Set{Symbol}()
    la_pressure_given && push!(valid_symbols, :Vₗₐ)
    lv_pressure_given && push!(valid_symbols, :Vₗᵥ)
    ra_pressure_given && push!(valid_symbols, :Vᵣₐ)
    rv_pressure_given && push!(valid_symbols, :Vᵣᵥ)
    @error "Variable named '$symbol' not found. The following symbols are defined and accessible: $valid_symbols."
end

function default_initial_condition!(u, model::RSAFDQ2022LumpedCicuitModel)
    # obtain via pre-pacing in isolation
    u .= [31.78263930696728, 20.619293500582675, 76.99868985499684, 28.062792020353495, 5.3259599006276925, 13.308990108813674, 1.848880514855276, 3.6948263599349302, -9.974721253140004, -17.12404226311947, -11.360818019572653, -19.32908606755043]
end

function lumped_circuit_relative_lv_pressure_index(model::RSAFDQ2022LumpedCicuitModel)
    model.lv_pressure_given && @error "Trying to query extenal LV pressure index, but LV pressure is not an external input!"
    return 1
end
function lumped_circuit_relative_rv_pressure_index(model::RSAFDQ2022LumpedCicuitModel)
    model.rv_pressure_given && @error "Trying to query extenal RV pressure index, but RV pressure is not an external input!"
    i = 1
    if model.lv_pressure_given
        i+=1
    end
    return i
end
function lumped_circuit_relative_la_pressure_index(model::RSAFDQ2022LumpedCicuitModel)
    model.la_pressure_given && @error "Trying to query extenal LA pressure index, but LA pressure is not an external input!"
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
    model.la_pressure_given && @error "Trying to query extenal RA pressure index, but RA pressure is not an external input!"
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

function lumped_driver!(du, u, t, external_input::AbstractVector, model::RSAFDQ2022LumpedCicuitModel)
    # Evaluate the right hand side of equation system (6) in the paper
    # V = volume
    # p = pressure
    # Q = flow rates
    # E = elastance
    # [x]v = ventricle [x]
    Vₗₐ, Vₗᵥ, Vᵣₐ, Vᵣᵥ, psysₐᵣ, psysᵥₑₙ, ppulₐᵣ, ppulᵥₑₙ, Qsysₐᵣ, Qsysᵥₑₙ, Qpulₐᵣ, Qpulᵥₑₙ = u

    @unpack Rsysₐᵣ, Rpulₐᵣ, Rsysᵥₑₙ, Rpulᵥₑₙ = model
    @unpack Csysₐᵣ, Cpulₐᵣ, Csysᵥₑₙ, Cpulᵥₑₙ = model
    @unpack Lsysₐᵣ, Lpulₐᵣ, Lsysᵥₑₙ, Lpulᵥₑₙ = model
    @unpack Rmin, Rmax = model
    # @unpack Epassₗₐ, Epassᵣₐ, Epassᵣᵥ = model
    # @unpack Eactmaxₗₐ, Eactmaxᵣₐ, Eactmaxᵣᵥ = model
    # @unpack Eₗₐ, Eᵣₐ, Eᵣᵥ = model
    # Note tR = tC+TC
    @inline Eₗᵥ(p::RSAFDQ2022LumpedCicuitModel,t) = elastance_RSAFDQ2022(t, p.Epassᵣₐ, 10.0*p.Eactmaxᵣₐ, p.tCᵣₐ, p.tCᵣₐ + p.TCᵣₐ, p.TCᵣₐ, p.TRᵣₐ, p.THB)
    @inline Eᵣᵥ(p::RSAFDQ2022LumpedCicuitModel,t) = elastance_RSAFDQ2022(t, p.Epassᵣᵥ, p.Eactmaxᵣᵥ, p.tCᵣᵥ, p.tCᵣᵥ + p.TCᵣᵥ, p.TCᵣᵥ, p.TRᵣᵥ, p.THB)
    @inline Eₗₐ(p::RSAFDQ2022LumpedCicuitModel,t) = elastance_RSAFDQ2022(t, p.Epassₗₐ, p.Eactmaxₗₐ, p.tCₗₐ, p.tCₗₐ + p.TCₗₐ, p.TCₗₐ, p.TRₗₐ, p.THB)
    @inline Eᵣₐ(p::RSAFDQ2022LumpedCicuitModel,t) = elastance_RSAFDQ2022(t, p.Epassᵣₐ, p.Eactmaxᵣₐ, p.tCᵣₐ, p.tCᵣₐ + p.TCᵣₐ, p.TCᵣₐ, p.TRᵣₐ, p.THB)

    @unpack V0ₗₐ, V0ᵣₐ, V0ᵣᵥ = model
    @unpack tCₗₐ, TCₗₐ, TRₗₐ, TRᵣₐ, tCᵣₐ, TCᵣₐ, tCᵣᵥ, TCᵣᵥ, TRᵣᵥ = model

    #pₑₓ = 0.0 # External pressure created by organs
    pₗᵥ = model.lv_pressure_given ? Eₗᵥ(model, t)*(Vₗᵥ - 5.0)  : external_input[lumped_circuit_relative_lv_pressure_index(model)]
    pᵣᵥ = model.rv_pressure_given ? Eᵣᵥ(model, t)*(Vᵣᵥ - V0ᵣᵥ) : external_input[lumped_circuit_relative_rv_pressure_index(model)]
    pₗₐ = model.la_pressure_given ? Eₗₐ(model, t)*(Vₗₐ - V0ₗₐ) : external_input[lumped_circuit_relative_la_pressure_index(model)]
    pᵣₐ = model.ra_pressure_given ? Eᵣₐ(model, t)*(Vᵣₐ - V0ᵣₐ) : external_input[lumped_circuit_relative_ra_pressure_index(model)]

    @inline Rᵢ(p₁, p₂, p) = p₁ < p₂ ? p.Rmin : p.Rmax # Resistance
    @inline Qᵢ(p₁, p₂, model) = (p₁ - p₂) / Rᵢ(p₁, p₂, model)
    # @inline Qᵢ(p₁, p₂, model) = max(p₁ - p₂, 0.0)
    Qₘᵥ = Qᵢ(pₗₐ, pₗᵥ, model)
    Qₐᵥ = Qᵢ(pₗᵥ, psysₐᵣ, model)
    Qₜᵥ = Qᵢ(pᵣₐ, pᵣᵥ, model)
    Qₚᵥ = Qᵢ(pᵣᵥ, ppulₐᵣ, model)
    # @show t, Qpulᵥₑₙ, Qₘᵥ, Qₐᵥ
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
    Q9 = (psysᵥₑₙ - psysₐᵣ) / Rsysₐᵣ
    du[9]  = - Rsysₐᵣ/Lsysₐᵣ* (Qsysₐᵣ + Q9) # sys ar
    Q10 = (pᵣₐ - psysᵥₑₙ) / Rsysᵥₑₙ
    du[10] = - Rsysᵥₑₙ/Lsysᵥₑₙ * (Qsysᵥₑₙ + Q10) # sys ven
    Q11 = (ppulᵥₑₙ - ppulₐᵣ) / Rpulₐᵣ
    du[11] = - Rpulₐᵣ/Lpulₐᵣ * (Qpulₐᵣ + Q11) # pul ar
    Q12 = (pₗₐ - ppulᵥₑₙ) / Rpulᵥₑₙ
    du[12] = - Rpulᵥₑₙ/Lpulᵥₑₙ * (Qpulᵥₑₙ + Q12) # sys ar
end
