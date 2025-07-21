using Catalyst, OrdinaryDiffEq

struct PiezoWrapper{ModelType,PiezoModelType,I4VectorType<:AbstractVector} <: Thunderbolt.AbstractIonicModel
    model::ModelType
    piezo_model::PiezoModelType
    I4::I4VectorType
end

function PiezoWrapper(model, piezo_model, npoints)
    I4 = zeros(npoints)
    return PiezoWrapper(model, piezo_model, I4)
end

Thunderbolt.num_states(pw::PiezoWrapper) = Thunderbolt.num_states(pw.model) + Thunderbolt.num_states(pw.piezo_model)

function Thunderbolt.default_initial_state(ionic_model::PiezoWrapper)
    u0 = zeros(Thunderbolt.num_states(ionic_model))
    model_init = Thunderbolt.default_initial_state(ionic_model.model)
    piezo_init = Thunderbolt.default_initial_state(ionic_model.piezo_model)
    u0[1:length(model_init)] .= model_init
    u0[length(model_init)+1:end] .= piezo_init
    return u0
end

@inline function Thunderbolt._pointwise_step_inner_kernel!(piezo_wrapper::F, i::I, t::T, Δt::T, cache::C) where {F<:PiezoWrapper,C<:Thunderbolt.ForwardEulerCellSolverCache,T<:Real,I<:Integer}
    u_local = @view cache.uₙmat[i, :]
    du_local = @view cache.dumat[i, :]
    x = Thunderbolt.getcoordinate(cache, i)
    I4 = piezo_wrapper.I4[i]
    xIstretch = 0.0
    scaling = 0.05
    V = get_V(du_local, u_local, piezo_wrapper.model)
    stretch = sqrt(I4) - 1
    P = stretch * 250.0
    prob = piezo_wrapper.piezo_model.problem[i]
    parameters = parameters(prob.f.sys)
    parameters[:V] = V
    parameters[:P] = P
    piezo_wrapper.piezo_model.problem[i] = remake(prob; p=parameters, u0=prob.u)
    prob = piezo_wrapper.piezo_model.problem[i]
    integrator = init(prob, Euler(); dt=Δt)
    step!(integrator)
    POpen = integrator(Δt)
    piezo_wrapper.piezo_model.problem
    # TODO: don't assume P(O) is always at first index.
    EK = get_EK(du_local, u_local, piezo_wrapper.model)
    gPz1K = 0.93
    gPz1Na = 0.28
    gPz1Ca = 1.0
    xIstretch += scaling * gPz1K * POpen * max(0.0, stretch) * (V - EK)
    # Calculating stretch induced current for Na
    ENa = get_ENa(du_local, u_local, piezo_wrapper.model)
    xIstretch += scaling * gPz1Na * POpen * max(0.0, stretch) * (V - ENa)
    # Calculating stretch induced current for Ca
    ECa = get_ECa(du_local, u_local, piezo_wrapper.model)
    xIstretch += scaling * gPz1Ca * POpen * max(0.0, stretch) * (V - ECa)
    # TODO get Cₘ
    Thunderbolt.cell_rhs!(du_local, u_local, x, t, piezo_wrapper.model)
    du_local -= xIstretch
    @inbounds for j in 1:length(u_local)
        u_local[j] += Δt * du_local[j]
    end

    return true
end

@inline function Thunderbolt._pointwise_step_inner_kernel!(piezo_wrapper::F, i::I, t::T, Δt::T, cache::C) where {F<:PiezoWrapper,C<:Thunderbolt.AdaptiveForwardEulerSubstepperCache,T<:Real,I<:Integer}
    nstates_inner = Thunderbolt.num_states(piezo_wrapper.model)
    u_local = @view cache.uₙmat[i, :]
    u_local_piezo = @view u_local[nstates_inner+1:end]
    u_local_inner = @view u_local[1:nstates_inner]
    du_local = @view cache.dumat[i, :]
    du_local_piezo = @view du_local[nstates_inner+1:end]
    du_local_inner = @view du_local[1:nstates_inner]
    x = Thunderbolt.getcoordinate(cache, i)
    I4 = piezo_wrapper.I4[i]
    xIstretch = 0.0
    scaling = 0.18
    V = get_V(du_local_inner, u_local_inner, piezo_wrapper.model)
    stretch = sqrt(I4) - 1
    P = max(0.0, stretch) * 250.0
    Thunderbolt.cell_rhs!(du_local_piezo, u_local_piezo, x, t, NodalOgiePiezo1(piezo_wrapper.piezo_model, P, V))
    # TODO: don't assume P(O) is always at first index.
    EK = get_EK(du_local_inner, u_local_inner, piezo_wrapper.model)
    gPz1K = 0.93
    gPz1Na = 0.28
    gPz1Ca = 1.0
    POpen = u_local_piezo[1]


    xIstretch += scaling * gPz1K * POpen * max(0.0, stretch) * (V - EK)
    # Calculating stretch induced current for Na
    ENa = get_ENa(du_local_inner, u_local_inner, piezo_wrapper.model)
    xIstretch += scaling * gPz1Na * POpen * max(0.0, stretch) * (V - ENa)
    # Calculating stretch induced current for Ca
    ECa = get_ECa(du_local_inner, u_local_inner, piezo_wrapper.model)
    xIstretch += scaling * gPz1Ca * POpen * max(0.0, stretch) * (V - ECa)

    φₘidx = Thunderbolt.transmembranepotential_index(piezo_wrapper)
    # TODO get Cₘ
    Thunderbolt.cell_rhs!(du_local_inner, u_local_inner, x, t, piezo_wrapper.model)
    du_local[1] -= xIstretch
    if abs(du_local[φₘidx]) < cache.reaction_threshold
        for j in 1:length(u_local)
            u_local[j] += Δt * du_local[j]
        end
    else
        Δtₛ = Δt / cache.substeps
        for j in 1:length(u_local)
            u_local[j] += Δtₛ * du_local[j]
        end

        for substep ∈ 2:cache.substeps
            tₛ = t + substep * Δtₛ
            #TODO Cₘ
            I4 = piezo_wrapper.I4[i]
            xIstretch = 0.0
            V = get_V(du_local_inner, u_local_inner, piezo_wrapper.model)
            stretch = sqrt(I4) - 1
            P = max(0.0, stretch * 250.0)
            Thunderbolt.cell_rhs!(du_local_piezo, u_local_piezo, x, tₛ, NodalOgiePiezo1(piezo_wrapper.piezo_model, P, V))
            POpen = u_local_piezo[1]
            EK = get_EK(du_local_inner, u_local_inner, piezo_wrapper.model)
            xIstretch += scaling * gPz1K * POpen * max(0.0, stretch) * (V - EK)
            # Calculating stretch induced current for Na
            ENa = get_ENa(du_local_inner, u_local_inner, piezo_wrapper.model)
            xIstretch += scaling * gPz1Na * POpen * max(0.0, stretch) * (V - ENa)
            # Calculating stretch induced current for Ca
            ECa = get_ECa(du_local_inner, u_local_inner, piezo_wrapper.model)
            xIstretch += scaling * gPz1Ca * POpen * max(0.0, stretch) * (V - ECa)
            Thunderbolt.cell_rhs!(du_local_inner, u_local_inner, x, tₛ, piezo_wrapper.model)
            du_local[1] -= xIstretch

            for j in 1:length(u_local)
                u_local[j] += Δtₛ * du_local[j]
            end
        end
    end
    return true
end

Thunderbolt.transmembranepotential_index(pw::PiezoWrapper) = Thunderbolt.transmembranepotential_index(pw.model)
