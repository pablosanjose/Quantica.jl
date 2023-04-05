abstract type Observable end

fermi(ω::C, kBT) where {C} =
    iszero(kBT) ? ifelse(real(ω) <= 0, C(1), C(0)) : C(1/(exp(ω/kBT) + 1))

############################################################################################
# josephson
#    Equilibrium (static) Josephson current given by
#       Iᵢ = (e/h) Re ∫dω f(ω)Tr[(GʳΣʳᵢ-ΣʳᵢGʳ)τz]
#   josephson(g::GreenFunctionSlice, ωmax; kBT = 0, path = ...)[i] -> Iᵢ in units of e/h
#region

struct Josephson{T<:AbstractFloat,G<:GreenFunction{T},O<:NamedTuple} <: Observable
    g::G
    ωmax::T
    kBT::T
    contactind::Int          # contact index
    normalsize::Int         # number of orbitals per site in normal Hamiltonian
    path::FunctionWrapper{Tuple{Complex{T},Complex{T}},Tuple{T}}
    opts::O
    Σ::Matrix{Complex{T}}
    points::Vector{Tuple{T,T}}
end

#region ## Constructors ##

josephson(g::GreenFunction, contactind::Integer ,ωmax::Real; kw...) =
    josephson(g, contactind, ωmax + 0.0im; kw...)

function josephson(g::GreenFunction, contactind::Integer, ωmax::Complex{T}; kBT = 0.0, path = x -> (x*(1-x), 1-2x), kw...) where {T}
    normalsize = normal_size(hamiltonian(g))
    realωmax = abs(real(ωmax))
    kBT´ = T(kBT)
    function path´(realω)
        η = imag(ωmax)
        imz, imz´ = path(abs(realω)/realωmax)
        imz´ *= sign(realω)
        ω = realω + im * (η + imz * realωmax)
        dzdω = 1 + im * imz´
        return ω, dzdω
    end
    pathwrap = FunctionWrapper{Tuple{Complex{T},Complex{T}},Tuple{T}}(path´)
    Σ = similar_contactΣ(g)
    points = Tuple{T,T}[]
    return Josephson(g, realωmax, kBT´, contactind, normalsize, pathwrap, NamedTuple(kw), Σ, points)
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
    empty!(J.points)
    Iᵢ, err = quadgk(ω -> josephson_integrand(ω, J; params...), ωmin, ωmax; atol = sqrt(eps(T)), J.opts...)
    return Iᵢ
end

function josephson_integrand(ω, J; params...)
    complexω, dzdω = J.path(ω)
    gω = call!(J.g, complexω; params...)
    trace = josephson_trace(gω, J)
    f = fermi(ω, J.kBT)
    integ = real(f * trace * dzdω)
    push!(J.points, (ω, integ))
    return integ
end

# Do Tr[tmp*τz], where tmp = gr * Σi - Σi * gr
function josephson_trace(gω, J)
    gr = gω[J.contactind, J.contactind]
    Σi = selfenergy!(J.Σ, gω, J.contactind)
    tmp = gr * Σi
    mul!(tmp, Σi, gr, -1, 1)
    trace = zero(eltype(tmp))
    for i in axes(tmp, 2)
        c = ifelse(iseven(fld1(i, J.normalsize)), -1, 1)
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