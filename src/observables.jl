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
# Integrator - integrates a function f along a complex path ωcomplex(ω::Real), connecting ωi
#   The path is piecewise linear in the form of a sawtooth with a given ± slope
#region

struct Integrator{I,T,P,O<:NamedTuple,F} <: Observable
    integrand::I    # call!(integrand, ω::Complex; params...)::Union{Number,Array{Number}}
    points::P       # a tuple of complex points that form the sawtooth integration path
    result::T       # can be missing (for scalar integrand) or a mutable type (nonscalar)
    opts::O         # kwargs for quadgk
    post::F         # function to apply to the integrand at the end of integration
end

#region ## Constructor ##

function Integrator(result, f, pts::NTuple{<:Any,Number}; imshift = missing, post = identity, slope = 1, opts...)
    imshift´ = imshift === missing ?
        sqrt(eps(promote_type(typeof.(float.(real.(pts)))...))) : float(imshift)
    pts´ = iszero(slope) ? pts .+ (im * imshift´) : sawtooth(imshift´, slope, pts)
    opts´ = NamedTuple(opts)
    return Integrator(f, pts´, result, opts´, post)
end

Integrator(f, pts::NTuple{<:Any,Number}; kw...) = Integrator(missing, f, pts; kw...)

sawtooth(is, sl, ωs) = _sawtooth(is, sl, (), ωs...)
_sawtooth(is, sl, ::Tuple{}, ω1, ωs...) = _sawtooth(is, sl, (ω1 + im * is,), ωs...)
_sawtooth(is, sl, ωs´, ωn, ωs...) = _sawtooth(is, sl,
    (ωs´..., 0.5 * (real(last(ωs´)) + ωn) + im * (is + sl * 0.5*(ωn - real(last(ωs´)))), ωn + im * is), ωs...)
_sawtooth(is, sl, ωs´) = ωs´

#endregion

#region ## API ##

integrand(I::Integrator) = I.integrand

points(I::Integrator) = I.points

options(I::Integrator) = I.opts

## call! ##
# scalar version
function call!(I::Integrator{<:Any,Missing}; params...)
    fx = x -> call!(I.integrand, x; params...)
    result, err = quadgk(fx, I.points...; I.opts...)
    result´ = I.post(result)
    return result´
end

# nonscalar version
function call!(I::Integrator; params...)
    fx! = (y, x) -> (y .= call!(I.integrand, x; params...))
    result, err = quadgk!(fx!, I.result, I.points...; I.opts...)
    result´ = I.post.(result)
    return result´
end

(I::Integrator)(; params...) = copy(call!(I; params...))

#endregion
#endregion

############################################################################################
# josephson
#   The equilibrium (static) Josephson current, in units of qe/h, *from* lead i is given by
#       Iᵢ = Re ∫dω J(ω; params...), where J(ω; params...) = (qe/h) × 2f(ω)Tr[(ΣʳᵢGʳ - GʳΣʳᵢ)τz]
#   J = JosephsonDensity(g::GreenFunction; contact = i, kBT = 0, phases)
#   J(ω; params...) -> scalar or vector [J(ϕⱼ) for ϕⱼ in phases] if `phases` is an
#       integer (num phases from 0 to π) or a collection of ϕ's
#   A phase ϕ can be applied by gauging it away from the lead and into its coupling:
#       Σʳᵢ(ϕ) = UᵩΣʳᵢUᵩ' and Gʳ(ϕ) = [1+Gʳ(Σʳᵢ-Σʳᵢ(ϕ))]⁻¹Gʳ, where Uᵩ = exp(iϕτz/2).
#   I = josephson(Integrator(J, (-ωmax, 0, ωmax); post = real, opts...)
#   Keywords opts are passed to quadgk for the integral
#region

struct JosephsonDensity{T<:AbstractFloat,P<:Union{Missing,AbstractArray},G<:GreenFunction{T}} <: Observable
    g::G
    kBT::T
    contactind::Int             # contact index
    tauz::Vector{Int}           # precomputed diagonal of tauz
    phaseshifts::P              # missing or collection of phase shifts to apply
    traces::P                   # preallocated workspace
    Σ::Matrix{Complex{T}}       # preallocated workspace
    ΣggΣ::Matrix{Complex{T}}    # preallocated workspace
    Σ´::Matrix{Complex{T}}      # preallocated workspace
    g´::Matrix{Complex{T}}      # preallocated workspace
    den::Matrix{Complex{T}}     # preallocated workspace
    cisτz::Vector{Complex{T}}   # preallocated workspace
end

#region ## Constructors ##

function josephson(g::GreenFunction{T}, ωmax; contact = 1, kBT = 0.0, phases = missing, imshift = missing, opts...) where {T}
    kBT´ = T(kBT)
    Σ = similar_contactΣ(g)
    normalsize = normal_size(hamiltonian(g))
    tauz = tauz_diag.(axes(Σ, 1), normalsize)
    phases´, traces = sanitize_phases_traces(phases, T)
    jd = JosephsonDensity(g, kBT´, contact, tauz, phases´,
        traces, Σ, similar(Σ), similar(Σ), similar(Σ), similar(Σ), similar(tauz, Complex{T}))
    integrator = iszero(kBT) ?
        Integrator(traces, jd, (-ωmax, 0); imshift, slope = 1, post = real, opts...) :
        Integrator(traces, jd, (-ωmax, 0, ωmax); imshift, slope = 1, post = real, opts...)
    return integrator
end

sanitize_phases_traces(::Missing, ::Type{T}) where {T} = missing, missing
sanitize_phases_traces(phases::Integer, ::Type{T}) where {T} =
    sanitize_phases_traces(range(0, π, length = phases), T)

function sanitize_phases_traces(phases, ::Type{T}) where {T}
    phases´ = Complex{T}.(phases)
    traces = similar(phases´)
    return phases´, traces
end

#endregion

#region ## API ##

temperature(J::JosephsonDensity) = J.kBT

contact(J::JosephsonDensity) = J.contactind

phaseshifts(I::Integrator{<:JosephsonDensity}) = phaseshifts(integrand(I))
phaseshifts(J::JosephsonDensity) = real.(J.phaseshifts)

function call!(J::JosephsonDensity, ω; params...)
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

josephson_traces!(J::JosephsonDensity{<:Any,Missing}, gr, Σi, f) = josephson_one_trace!(J, gr, Σi, f)

function josephson_traces!(J, gr, Σi, f)
    for (i, phaseshift) in enumerate(J.phaseshifts)
        gr´, Σi´ = apply_phaseshift!(J, gr, Σi, phaseshift)
        J.traces[i] = josephson_one_trace!(J, gr´, Σi´, f)
    end
    return J.traces
end

# 2 f(ω) Tr[(Σi * gr - gr * Σi) * τz]
function josephson_one_trace!(J, gr, Σi, f)
    ΣggΣ = J.ΣggΣ
    mul!(ΣggΣ, Σi, gr)
    mul!(ΣggΣ, gr, Σi, -1, 1)
    trace = 2 * f * trace_tau(ΣggΣ, J.tauz)
    return trace
end

# Σi´ = U Σi U' and gr´ = (gr₀⁻¹ - Σi´)⁻¹ = (1+gr*(Σi-Σi´))⁻¹gr
function apply_phaseshift!(J, gr, Σi, phaseshift)
    Σi´ = J.Σ´
    U = J.cisτz
    phasehalf = phaseshift/2
    @. U = cis(-phasehalf * J.tauz)
    @. Σi´ = U * Σi * U'

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
# conductance(gs::GreenFunctionSlice; nambu -> false) -> G(ω; params...)::Real
#   For gs = g[i::Int, j::Int = i] -> we get zero temperature Gᵢⱼ = dIᵢ/dVⱼ in units of e^2/h
#   where i, j are contact indices
#       Gᵢⱼ =  e^2/h × Tr{[δᵢⱼi(Gʳ-Gᵃ)Γⁱ-GʳΓⁱGᵃΓʲ]}         (nambu = false)
#       Gᵢⱼ =  e^2/h × Tr{[δᵢⱼi(Gʳ-Gᵃ)Γⁱτₑ-GʳΓⁱτzGᵃΓʲτₑ]}   (nambu = true)
#   and where τₑ = [1 0; 0 0] and τz = [1 0; 0 -1] in Nambu space, and ω = eV.
#region

struct Conductance{T,E,L,C,G<:GreenFunction{T,E,L}} <: Observable
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

function conductance(gs::GreenFunctionSlice{T}; nambu = false) where {T}
    i = slicerows(gs)
    j = slicecols(gs)
    g = parent(gs)
    ni = flatsize(blockstructure(g), i)
    nj = flatsize(blockstructure(g), j)
    Γ = similar_contactΣ(g)
    if nambu
        nsize = normal_size(hamiltonian(g))
        τezdiag = (taue_diag.(1:nj, nsize), tauz_diag.(1:ni, nsize))
    else
        τezdiag = (missing, missing)
    end
    GrΓi = Matrix{Complex{T}}(undef, nj, ni)
    GaΓj = Matrix{Complex{T}}(undef, ni, nj)
    GΓGΓ = Matrix{Complex{T}}(undef, nj, nj)
    return Conductance(g, i, j, τezdiag, Γ, GrΓi, GaΓj, GΓGΓ)
end

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
    mul_tau!(G.GrΓi, τz)                        # no-op if τz is missing
    mul!(G.GΓGΓ, G.GrΓi, G.GaΓj)
    # the -Tr{GʳΓⁱτzGᵃΓʲτₑ} term
    cond = - real(trace_tau(G.GΓGΓ, τe))        # simple trace if τe is missing
    if G.i == G.j
        # add the Tr(i(Gʳ-Gᵃ)Γⁱτₑ) term
        gmg = gʳⱼᵢ
        gmg .-= gᵃᵢⱼ
        iGmGΓ = mul!(G.GΓGΓ, gmg, Γi, im, 0)
        cond += real(trace_tau(iGmGΓ, τe))      # simple trace if τe is missing
    end
    return cond
end

#endregion
#endregion

############################################################################################
# ldos: local spectral density
#   d = ldos(::GreenSolution; kernel = I)      -> d[sites...]::Vector
#   d = ldos(::GreenFunctionSlice; kernel = I) -> d(ω; params...)::Vector
#   Here ldos is given as Tr(ρᵢᵢ * kernel) where ρᵢᵢ is the spectral function at site i
#region

struct LocalSpectralDensitySolution{T,E,L,G<:GreenSolution{T,E,L},K}
    gω::G
    kernel::K                      # should return a float when applied to gω[cellsite(n,i)]
end

struct LocalSpectralDensitySlice{T,E,L,G<:GreenFunction{T,E,L},K}
    g::G
    kernel::K                      # should return a float when applied to gω[cellsite(n,i)]
    latslice::LatticeSlice{T,E,L}
    diagonal::Vector{T}
end

#region ## Constructors ##

ldos(gω::GreenSolution; kernel = I) = LocalSpectralDensitySolution(gω, kernel)

function ldos(gs::GreenFunctionSlice{T}; kernel = I) where {T}
    slicerows(gs) === slicecols(gs) ||
        argerror("Cannot take ldos of a GreenFunctionSlice with rows !== cols")
    g = parent(gs)
    lat = lattice(g)
    latslice = lat[slicerows(gs)]
    diagonal = Vector{T}(undef, length(latslice))
    return LocalSpectralDensitySlice(g, kernel, latslice, diagonal)
end

#endregion

#region ## API ##

greenfunction(d::LocalSpectralDensitySlice) = d.g

Base.getindex(d::LocalSpectralDensitySolution; kw...) = d[getindex(lattice(d.gω); kw...)]

function Base.getindex(d::LocalSpectralDensitySolution{T}, l::LatticeSlice) where {T}
    v = T[]
    foreach(scell -> append_ldos!(v, scell, d.gω, d.kernel), subcells(l))
    return v
end

Base.getindex(d::LocalSpectralDensitySolution{T}, scell::CellSites) where {T} =
    append_ldos!(T[], scell, d.gω, d.kernel)

function call!(d::LocalSpectralDensitySlice, ω; params...)
    gω = call!(greenfunction(d), ω; params...)
    empty!(d.diagonal)
    for sc in subcells(d.latslice)
        append_ldos!(d.diagonal, sc, gω, d.kernel)
    end
    return d.diagonal
end

(d::LocalSpectralDensitySlice)(ω; params...) = copy(call!(d, ω; params...))

function append_ldos!(v, sc, gω, kernel)
    gcell = gω[sc]
    for i in siteindices(sc)
        rng = flatrange(hamiltonian(gω), i)
        gi = view(gcell, rng, rng)
        ldos = kernel === I ? -imag(tr(gi))/π : -imag(tr(gi * kernel))/π
        push!(v, ldos)
    end
    return v
end

#endregion
#endregion

############################################################################################
# current: current density Jᵢⱼ(ω) as a function of a charge operator
#   d = current(::GreenSolution; charge)      -> d[sites...]::SparseMatrixCSC{SVector{E,T}}
#   d = current(::GreenFunctionSlice; charge) -> d(ω; params...)::SparseMatrixCSC{SVector{E,T}}
#   Computes the zero-temperature equilibrium current density matrix Jᵢⱼ from site j to site i
#       Jᵢⱼ(ω) = (2/h) rᵢⱼ Re Tr[(Hᵢⱼgʳⱼᵢ - gʳᵢⱼHⱼᵢ)Q]
#   Here charge = Q, where Q is usually qe*I for normal, and qe*τz/2 for Nambu systems
#   We use a default charge = -I, corresponding to normal currents densities in units of e/h
#region

struct CurrentDensitySolution{T,E,L,G<:GreenSolution{T,E,L},K}
    gω::G
    charge::K                      # should return a float when applied to gʳᵢⱼHᵢⱼ
    cache::GreenSolutionCache{T,L,G}
end

struct CurrentDensitySlice{T,E,L,G<:GreenFunction{T,E,L},K}
    g::G
    charge::K                      # should return a float when applied to gʳᵢⱼHᵢⱼ
    latslice::LatticeSlice{T,E,L}
end

#region ## Constructors ##

current(gω::GreenSolution; charge = -I) =
    CurrentDensitySolution(gω, charge, GreenSolutionCache(gω))

function current(gs::GreenFunctionSlice; charge = -I)
    slicerows(gs) === slicecols(gs) ||
        argerror("Cannot currently take ldos of a GreenFunctionSlice with rows !== cols")
    g = parent(gs)
    latslice = lattice(g)[slicerows(gs)]
    return CurrentDensitySlice(g, charge, latslice)
end

#endregion

#region ## API ##

Base.getindex(d::CurrentDensitySolution; kw...) = d[getindex(lattice(d.gω); kw...)]

Base.getindex(d::CurrentDensitySolution, ls::LatticeSlice) = current_matrix(d.gω, ls, d)

Base.getindex(d::CurrentDensitySolution, scell::CellSites) = d[lattice(hamiltonian(d.gω))[scell]]

# no call! support here
function (d::CurrentDensitySlice)(ω; params...)
    gω = call!(d.g, ω; params...)
    ls = d.latslice
    cu = current(gω; charge = d.charge)
    return cu[ls]
end

function current_matrix(gω, ls, d)
    h = hamiltonian(parent(gω))
    current = h[ls, (hij, cij) -> apply_charge_current(hij, cij, d)]
    return current
end

function apply_charge_current(hij::B, (ci, cj), d::CurrentDensitySolution{T,E}) where {T,E,B}
    ni, i = cell(ci), siteindex(ci)
    nj, j = cell(cj), siteindex(cj)
    ni == nj && i == j && return zero(SVector{E,T})
    gji = mask_block(B, d.cache[cj, ci])
    gij = mask_block(B, d.cache[ci, cj])
    hji = hij'
    Iij = 2 * real(tr((hij * gji - gij * hji) * d.charge))
    lat = lattice(d.gω)
    ri = site(lat, i, ni)
    rj = site(lat, j, nj)
    Jij = (ri - rj) * Iij
    return Jij
end

#endregion

