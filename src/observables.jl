############################################################################################
# Observables - common tools
#region

abstract type Observable end

fermi(ω::C, kBT = 0) where {C} =
    iszero(kBT) ? ifelse(real(ω) <= 0, C(1), C(0)) : C(1/(exp(ω/kBT) + 1))

normal_size(h::AbstractHamiltonian) = normal_size(blockstructure(h))

function normal_size(b::OrbitalBlockStructure)
    n = first(blocksizes(b))
    iseven(n) && allequal(blocksizes(b)) ||
        argerror("A Nambu Hamiltonian must have an even and uniform number of orbitals per site, got $(blocksizes(b)).")
    return n ÷ 2
end

trace_tau(g, ::Missing) = tr(g)

function trace_tau(g, tau)
    trace = zero(eltype(g))
    for i in axes(g, 2)
        trace += g[i, i] * tau[i]
    end
    return trace
end

mul_tau!(g, ::Missing) = g
mul_tau!(::Missing, g) = g

mul_tau!(g, tau::Vector) = (g .*= tau')
mul_tau!(tau::Vector, g) = (g .*= tau)

tauz_diag(i, normalsize) = ifelse(iseven(fld1(i, normalsize)), -1, 1)
taue_diag(i, normalsize) = ifelse(iseven(fld1(i, normalsize)), 0, 1)

#endregion

############################################################################################
# integrate - integrates a function f along a complex path ωcomplex(ω::Real), connecting ωi
#   The path is piecewise linear in the form of a sawtooth
#region

function integrate(f, ωs...; imshift = 0, kw...)
    ωs´ = sawtooth(imshift, ωs)
    intf = first(quadgk(f, ωs´...; kw...))
    return intf
end

function integrate!(f!, result, ωs...; imshift = 0, kw...)
    ωs´ = sawtooth(imshift, ωs)
    intf = first(quadgk!(f!, result, ωs´...; kw...))
    return intf
end

sawtooth(s, ωs) = _sawtooth(s, (), ωs...)
_sawtooth(s, ::Tuple{}, ω1, ωs...) = _sawtooth(s, (ω1 + im * s,), ωs...)
_sawtooth(s, ωs´, ω, ωs...) = _sawtooth(s,
    (ωs´..., 0.5 * (real(last(ωs´)) + ω) + im * (s + 0.5*(ω - real(last(ωs´)))), ω + im * s), ωs...)
_sawtooth(s, ωs´) = ωs´

#endregion

############################################################################################
# conductance
#   Zero temperature Gᵢⱼ = dIᵢ/dVⱼ in units of e^2/h for normal systems
#       Gᵢⱼ =  e^2/h × Tr{[δᵢⱼi(Gʳ-Gᵃ)Γⁱ-GʳΓⁱGᵃΓʲ]}
#   at ω = eV. For Nambu systems we have instead
#       Gᵢⱼ =  e^2/h × Tr{[δᵢⱼi(Gʳ-Gᵃ)Γⁱτₑ-GʳΓⁱτzGᵃΓʲτₑ]}
#   where τₑ = [1 0; 0 0] and τz = [1 0; 0 -1] in Nambu space, and again ω = eV.
#   Usage: G = conductance(g::GreenFunction, i, j; nambu = false) -> G(ω; params...)
#region

struct Conductance{T,C,G<:GreenFunction} <: Observable
    g::G
    i::Int                        # contact index for Iᵢ
    j::Int                        # contact index for Vⱼ
    τezdiag::Tuple{C,C}           # diagonal of τₑ and τz, or (missing, missing)
    Γ::Matrix{Complex{T}}         # prealloc workspace for selfenergy! (over all contacts)
    GrΓi::Matrix{Complex{T}}      # prealloc workspace GʳⱼᵢΓⁱ
    GaΓj::Matrix{Complex{T}}      # prealloc workspace GᵃᵢⱼΓʲ
    GΓGΓ::Matrix{Complex{T}}      # prealloc workspace GʳⱼᵢΓⁱGᵃᵢⱼΓʲ
end

#region ## Constructors ##

function conductance(g::GreenFunction{T}, i = 1, j = i; nambu = false) where {T}
    Γ = similar_contactΣ(g)
    ni = flatsize(blockstructure(g), i)
    nj = flatsize(blockstructure(g), j)
    if nambu
        normalsize = normal_size(hamiltonian(g))
        τezdiag = (taue_diag.(1:nj, normalsize), tauz_diag.(1:ni, normalsize))
    else
        τezdiag = (missing, missing)
    end
    GrΓi = Matrix{Complex{T}}(undef, nj, ni)
    GaΓj = Matrix{Complex{T}}(undef, ni, nj)
    GΓGΓ = Matrix{Complex{T}}(undef, nj, nj)
    return Conductance(g, i, j, τezdiag, Γ, GrΓi, GaΓj, GΓGΓ)
end

tau_vectors(charge, len) = [charge[mod1(i, length(charge))] for i in 1:len]

#endregion

#region ## API ##

currentcontact(G) = G.i

biascontact(G) = G.j

(G::Conductance{T})(ω::Real; params...) where {T} = G(ω + im*sqrt(eps(T)); params...)

function (G::Conductance)(ω::Complex; params...)
    τe, τz = G.τezdiag
    gω = call!(G.g, ω; params...)
    gʳⱼᵢ = gω[G.j, G.i]
    gᵃᵢⱼ = gʳⱼᵢ'
    Γi = selfenergy!(G.Γ, gω, G.i; onlyΓ = true)
    mul!(G.GrΓi, gʳⱼᵢ, Γi)
    Γj = G.i == G.j ? Γi : selfenergy!(G.Γ, gω, G.j; onlyΓ = true)
    mul!(G.GaΓj, gᵃᵢⱼ, Γj)
    mul_tau!(G.GrΓi, τz)
    mul!(G.GΓGΓ, G.GrΓi, G.GaΓj)
    # the -Tr{GʳΓⁱτzGᵃΓʲτₑ} term
    cond = - real(trace_tau(G.GΓGΓ, τe))
    if G.i == G.j
        # add the Tr(i(Gʳ-Gᵃ)Γⁱτₑ) term
        gmg = gʳⱼᵢ
        gmg .-= gᵃᵢⱼ
        iGmGΓ = mul!(G.GΓGΓ, gmg, Γi, im, 0)
        cond += real(trace_tau(iGmGΓ, τe))
    end
    return cond
end

#endregion

#endregion

############################################################################################
# josephson
#    Equilibrium (static) Josephson current given by
#       Iᵢ = (e/h) Re ∫dω f(ω)Tr[(GʳΣʳᵢ-ΣʳᵢGʳ)τz]
#    J = josephson(g::GreenFunction, ωmax; contact = i, kBT = 0, path = x -> (y, y'), phases, kw...)
#    J(; params...) -> Iᵢ in units of e/h, or [Iᵢ(ϕⱼ) for ϕⱼ in phases] if phases is an
#       integer (from 0 to π) or a collection of ϕ's
#    Keywords kw are passed to quadgk for the integral
#    A phase ϕ can be applied by gauging it away from the lead and into its coupling:
#       Σʳᵢ(ϕ) = UᵩΣʳᵢUᵩ' and Gʳ(ϕ) = [1+Gʳ(Σʳᵢ-Σʳᵢ(ϕ))]⁻¹Gʳ, where Uᵩ = exp(iϕτz/2).
#region

struct Josephson{T<:AbstractFloat,P<:Union{Missing,Vector{T}},G<:GreenFunction{T},O<:NamedTuple} <: Observable
    g::G
    ωmax::T
    imshift::T
    kBT::T
    contactind::Int          # contact index
    opts::O
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

josephson(g::GreenFunction{T}, ωmax::Real; kw...) where {T} =
    josephson(g, ωmax + im * sqrt(eps(T)); kw...)

function josephson(g::GreenFunction, ωmax::Complex{T};
    contact = 1, kBT = 0.0, phases = missing, kw...) where {T}
    realωmax = abs(real(ωmax))
    imshift = imag(ωmax)
    kBT´ = T(kBT)
    Σ = similar_contactΣ(g)
    normalsize = normal_size(hamiltonian(g))
    tauz = tauz_diag.(axes(Σ, 1), normalsize)
    phases´, traces = sanitize_phases_traces(phases, T)
    return Josephson(g, realωmax, imshift, kBT´, contact, NamedTuple(kw), phases´,
        traces, tauz, Σ, similar(Σ), similar(Σ), similar(Σ), similar(Σ), similar(tauz, Complex{T}))
end

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

function (J::Josephson{T,Missing})(; params...) where {T}
    ωmax = J.ωmax
    imshift = J.imshift
    fω = ω -> josephson_integrand(ω, J; params...)
    Iᵢ = iszero(J.kBT) ?
        integrate(fω, -ωmax, zero(ωmax); atol = sqrt(eps(T)), imshift, J.opts...) :
        integrate(fω, -ωmax, zero(ωmax), ωmax; atol = sqrt(eps(T)), imshift, J.opts...)
    return real(Iᵢ)
end

function (J::Josephson{T,<:AbstractVector})(; params...) where {T}
    ωmax = J.ωmax
    imshift = J.imshift
    result = zero(J.traces)
    fω! = (integrand, ω) -> (integrand .= josephson_integrand(ω, J; params...))
    Iᵢ = iszero(J.kBT) ?
        integrate!(fω!, result, -ωmax, zero(ωmax); atol = sqrt(eps(T)), imshift, J.opts...) :
        integrate!(fω!, result, -ωmax, zero(ωmax), ωmax; atol = sqrt(eps(T)), imshift, J.opts...)
    return real(Iᵢ)
end

function josephson_integrand(ω, J; params...)
    gω = call!(J.g, ω; params...)
    f = fermi(ω, J.kBT)
    traces = josephson_traces(J, gω, f)
    return traces
end

function josephson_traces(J, gω, f)
    gr = gω[J.contactind, J.contactind]
    Σi = selfenergy!(J.Σ, gω, J.contactind)
    return josephson_traces!(J, gr, Σi, f)
end

josephson_traces!(J::Josephson{<:Any,Missing}, gr, Σi) = josephson_one_trace!(J, gr, Σi, f)

function josephson_traces!(J, gr, Σi, f)
    for (i, phaseshift) in enumerate(J.phaseshifts)
        gr´, Σi´ = apply_phaseshift!(J, gr, Σi, phaseshift)
        J.traces[i] = josephson_one_trace!(J, gr´, Σi´, f)
    end
    return J.traces
end

# Tr[(gr * Σi - Σi * gr) * τz] * f(ω)
function josephson_one_trace!(J, gr, Σi, f)
    gΣΣg = J.gΣΣg
    mul!(gΣΣg, gr, Σi)
    mul!(gΣΣg, Σi, gr, -1, 1)
    trace = f * trace_tau(gΣΣg, J.tauz)
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