abstract type Observable end

fermi(ω::C, kBT) where {C} =
    iszero(kBT) ? ifelse(real(ω) <= 0, C(1), C(0)) : C(1/(exp(ω/kBT) + 1))

############################################################################################
# josephson
#    Equilibrium (static) Josephson current given by
#       Iᵢ = (e/h) Re ∫dω f(ω)Tr[(GʳΣʳᵢ-ΣʳᵢGʳ)τz]
#   josephson(g::GreenFunctionSlice, ωmax; kBT = 0, path = ...)[i] -> Iᵢ in units of e/h
#region

struct Josephson{T<:AbstractFloat,P<:Union{Missing,Vector{T}},G<:GreenFunction{T},O<:NamedTuple} <: Observable
    g::G
    ωmax::T
    kBT::T
    contactind::Int          # contact index
    path::FunctionWrapper{Tuple{Complex{T},Complex{T}},Tuple{T}}
    opts::O
    points::Vector{Tuple{T,T}}
    phaseshifts::P
    traces::Vector{Complex{T}}
    tauz::Vector{Int}           # precomputed diagonal of tauz
    Σ::Matrix{Complex{T}}       # preallocated workspace
    gΣΣg::Matrix{Complex{T}}    # preallocated workspace
    Σ´::Matrix{Complex{T}}      # preallocated workspace
    g´::Matrix{Complex{T}}      # preallocated workspace
    den::Matrix{Complex{T}}     # preallocated workspace
    cisτz::Vector{Complex{T}}   # preallocated workspace
end

#region ## Constructors ##

josephson(g::GreenFunction, ωmax::Real, contactind::Integer = 1; kw...) =
    josephson(g, ωmax + 0.0im, contactind; kw...)

function josephson(g::GreenFunction, ωmax::Complex{T};
    contact = 1, kBT = 0.0, path = x -> (x*(1-x), 1-2x), phases = missing, kw...) where {T}
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
    normalsize = normal_size(hamiltonian(g))
    tauz = tauz_diag.(axes(Σ, 1), normalsize)
    points = Tuple{T,T}[]
    phases´, traces = sanitize_phases_traces(phases, T)
    return Josephson(g, realωmax, kBT´, contact, pathwrap, NamedTuple(kw), points, phases´,
        traces, tauz, Σ, similar(Σ), similar(Σ), similar(Σ), similar(Σ), similar(tauz, Complex{T}))
end

normal_size(h::AbstractHamiltonian) = normal_size(blockstructure(h))

function normal_size(b::OrbitalBlockStructure)
    n = first(blocksizes(b))
    iseven(n) && allequal(blocksizes(b)) ||
        argerror("A Nambu Hamiltonian must have an even and uniform number of orbitals per site, got $(blocksizes(b)).")
    return n ÷ 2
end

tauz_diag(i, normalsize) = ifelse(iseven(fld1(i, normalsize)), -1, 1)

sanitize_phases_traces(::Missing, ::Type{T}) where {T} = missing, Complex{T}[]
sanitize_phases_traces(phases::Vector, ::Type{T}) where {T} =
    phases, similar(phases, Complex{T})
sanitize_phases_traces(phases::Integer, ::Type{T}) where {T} =
    sanitize_phases_traces(range(T(0), T(π), length = phases), T)
sanitize_phases_traces(phases, T) = sanitize_phases_traces(Vector(phases), T)

#endregion

#region ## API ##

temperature(J::Josephson) = J.kBT

maxenergy(J::Josephson) = J.ωmax

contact(J::Josephson) = J.contactind

options(J::Josephson) = J.opts

phaseshifts(J::Josephson) = J.phaseshifts

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
    traces = josephson_traces(J, gω)
    f = fermi(ω, J.kBT)
    integrand = real((f * dzdω) * traces)
    push!(J.points, (ω, first(integrand)))
    return integrand
end

function josephson_traces(J, gω)
    gr = gω[J.contactind, J.contactind]
    Σi = selfenergy!(J.Σ, gω, J.contactind)
    return josephson_traces!(J, gr, Σi)
end

josephson_traces!(J::Josephson{<:Any,Missing}, gr, Σi) = josephson_one_trace!(J, gr, Σi)

function josephson_traces!(J, gr, Σi)
    for (i, phaseshift) in enumerate(J.phaseshifts)
        gr´, Σi´ = apply_phaseshift!(J, gr, Σi, phaseshift)
        J.traces[i] = josephson_one_trace!(J, gr´, Σi´)
    end
    return J.traces
end

# Tr[(gr * Σi - Σi * gr) * τz]
function josephson_one_trace!(J, gr, Σi)
    gΣΣg = J.gΣΣg
    mul!(gΣΣg, gr, Σi)
    mul!(gΣΣg, Σi, gr, -1, 1)
    trace = zero(eltype(gΣΣg))
    for i in axes(gΣΣg, 2)
        trace += gΣΣg[i, i] * J.tauz[i]
    end
    return trace
end

# Σi´ = U Σi U' and gr´ = (gr₀⁻¹ - Σi´)⁻¹ = (1+gr*(Σi-Σi´))⁻¹gr
function apply_phaseshift!(J, gr, Σi, phaseshift)
    Σi´ = J.Σ´
    U = J.cisτz
    phasehalf = phaseshift/2
    @. U = cis(phasehalf * J.tauz)
    @. Σi´ = U * Σi * U'       # Σi´ = U Σi U'

    den = J.den
    one!(den)
    tmp = J.g´
    @. tmp = Σi - Σi´
    mul!(den, gr, tmp, 1, 1)            # den = 1-gr * (Σi - Σi´)
    gr´ = ldiv!(J.g´, lu!(den), gr)     # gr´ = (1+gr*(Σi-Σi´))⁻¹gr

    return gr´, Σi´
end

#endregion

#endregion


############################################################################################
# conductance
#    Zero temperature G = dI/dV in units of e^2/h for normal systems or NS systems
#       G = 
#region

#endregion