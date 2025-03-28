# This file contains the main infrastucture for coupled problems

struct EmptyTyingCache end

get_tying_dofs(::EmptyTyingCache, u) = nothing

assemble_tying!(Jₑ, residualₑ, uₑ, uₜ, cell,::EmptyTyingCache, t) = nothing
assemble_tying!(Jₑ, uₑ, uₜ, cell,::EmptyTyingCache, t) = nothing

setup_tying_cache(::Nothing, qr, sdh::SubDofHandler) = EmptyTyingCache()

function setup_tying_cache(tying_models::Union{<:AbstractVector,<:Tuple}, qr, sdh::SubDofHandler)
    length(tying_models) == 0 && return EmptyTyingCache()
    return ntuple(i->setup_tying_cache(tying_models[i], qr, sdh), length(tying_models))
end
