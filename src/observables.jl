abstract type Observable end

fermi(ω::C, kBT) where {C} =
    iszero(kBT) ? ifelse(real(ω) <= 0, C(1), C(0)) : C(1/(exp(ω/kBT) + 1))

############################################################################################
# josephson
#    Equilibrium (static) Josephson current given by
#       Iᵢ = (e/h) Re ∫dω f(ω)Tr[(GʳΣʳᵢ-ΣʳᵢGʳ)τz]
#   josephson(g::GreenFunctionSlice, ωmax; kBT = 0, path = ...)[i] -> Iᵢ in units of e/h
#region

struct Josephson{T<:AbstractFloat,G<:GreenFunction,O<:NamedTuple} <: Observable
    g::G
    ωmax::T
    kBT::T
    contactind::Int          # contact index
    normalsize::Int         # number of orbitals per site in normal Hamiltonian
    path::FunctionWrapper{Tuple{Complex{T},Complex{T}},Tuple{T}}
    opts::O
end

#region ## Constructors ##

josephson(g::GreenFunction, contactind::Integer ,ωmax::Real; kw...) =
    josephson(g, contactind, ωmax + 0.0im; kw...)

function josephson(g::GreenFunction, contactind::Integer, ωmax::Complex{T}; kBT = 0.0, path = x -> (abs(x)*(1-abs(x)), sign(x) - 2x), kw...) where {T}
    normalsize = normal_size(hamiltonian(g))
    ωmax´ = real(ωmax)
    kBT´ = T(kBT)
    function path´(realω)
        η = imag(ωmax)
        imz, imz´ = path((realω+η*im)/ωmax´)
        ω = realω + im * (η + imz * ωmax´)
        dzdω = 1 + im * imz´
        return ω, dzdω
    end
    pathwrap = FunctionWrapper{Tuple{Complex{T},Complex{T}},Tuple{T}}(path´)
    return Josephson(g, ωmax´, kBT´, contactind, normalsize, pathwrap, NamedTuple(kw))
end

normal_size(h::AbstractHamiltonian) = normal_size(blockstructure(h))

function normal_size(b::OrbitalBlockStructure)
    n = first(blocksizes(b))
    iseven(n) && allequal(blocksizes(b)) ||
        argerror("A Nambu Hamiltonian must have an even and uniform number of orbitals per site, got $(blocksizes(b)).")
    return n ÷ 2
end

#endregion

#region ## API ##

temperature(J::Josephson) = J.kBT

maxenergy(J::Josephson) = J.ωmax

contact(J::Josephson) = J.contactind

options(J::Josephson) = J.opts


function (J::Josephson{T})(; params...) where {T}
    ωmin = -J.ωmax
    ωmax = ifelse(iszero(J.kBT), zero(J.ωmax), J.ωmax)
    Iᵢ, _ = quadgk(ω -> josephson_integrand(ω, J; params...), ωmin, ωmax; atol = sqrt(eps(T)), J.opts...)
    return Iᵢ
end

function josephson_integrand(ω, J; params...)
    complexω, dzdω = J.path(ω)
    gω = call!(J.g, complexω; params...)
    trace = josephson_trace(gω, J.contactind, J.normalsize)
    f = fermi(ω, J.kBT)
    return real(f * trace * dzdω)
end

# Do Tr[tmp*τz], where tmp = gr * Σi - Σi * gr
function josephson_trace(gω, contactind, normal)
    Σi = selfenergy(gω, contactind)
    gr = gω[contactind, contactind]
    tmp = gr * Σi
    mul!(tmp, Σi, gr, -1, 1)
    trace = zero(eltype(tmp))
    for i in axes(tmp, 2)
        c = ifelse(iseven(fld1(i, normal)), -1, 1)
        trace += c * tmp[i, i]
    end
    return trace
end

#endregion

#endregion


############################################################################################
# conductance
#    Zero temperature G = dI/dV in units of e^2/h for normal systems or NS systems
#       G = 
#region

#endregion