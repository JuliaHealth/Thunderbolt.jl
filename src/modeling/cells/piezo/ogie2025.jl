eyring(a, b, V) = a * exp(b * V)

Base.@kwdef struct OgiePiezo1{T<:Real}
    r1 ::T = 0.02634342   
    r3 ::T = 0.0008000712 
    r4 ::T = 0.008945307  
    r5 ::T = 1.3529515e-5 
    r6 ::T = 6.206403e-5  
    r7 ::T = 6.603723e-6  
    r8 ::T = 0.00079357607
    ce1::T = -2.3753126   
    ce3::T = -5.6010337   
    ce4::T = 0.70629007   
    cm4::T = -12.101605   
    ce5::T = 6.46389      
    ce6::T = -1.2173947   
    ce7::T = 8.739162     
    ce8::T = -3.364573    
end
struct NodalOgiePiezo1{T<:Real}
    parameters::OgiePiezo1{T}
    P::Float64
    V::Float64
end
Thunderbolt.num_states(::OgiePiezo1) = 4

function Thunderbolt.default_initial_state(::OgiePiezo1)
    return [0.0, 0.3, 0.1, 0.6]
end

function Thunderbolt.cell_rhs!(du::TD, u::TU, x::TX, t::TT, cell_parameters::TP) where {T,TD<:AbstractVector{T},TU,TX,TT,TP<:NodalOgiePiezo1}
    params = cell_parameters.parameters
    P = cell_parameters.P
    V = cell_parameters.V
    O_I1 = eyring(params.r1, params.ce1, V / 140.)
    I1_O = eyring(params.r1, params.ce1, V / 140.) *
           eyring(params.r3, params.ce3, V / 140.) *
           eyring(params.r5, params.ce5, V / 140.) /
           eyring(params.r6, params.ce6, V / 140.) /
           eyring(params.r4, params.ce4, V / 140.) / eyring(1.0, params.cm4, P / 70.)
    I1_C = eyring(params.r3, params.ce3, V / 140.)
    C_I1 = eyring(params.r4, params.ce4, V / 140.) * eyring(1.0, params.cm4, P / 70.)
    C_O = eyring(params.r5, params.ce5, V / 140.)
    O_C = eyring(params.r6, params.ce6, V / 140.)
    I2_O = eyring(params.r7, params.ce7, V / 140. * P / 70.)
    O_I2 = eyring(params.r8, params.ce8, V / 140.)
    
    O = u[1]
    C = u[2]
    I1 = u[3]
    I2 = u[4]

    du[1] = -O_I1 * O - O_I2 * O - O_C * O + C_O * C + I1_O * I1 + I2_O * I2
    du[2] = -C_I1 * C - C_O * C + O_C * O + I1_C * I1
    du[3] = -I1_C * I1 - I1_O * I1 + O_I1 * O + C_I1 * C
    du[4] = -I2_O * I2 + O_I2 * O
end
