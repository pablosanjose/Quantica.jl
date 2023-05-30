############################################################################################
# Observables - common tools
#region

abstract type IndexableObservable end  # any type that can be sliced into (current and ldos)

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

# if inds isa Integer, select contact cellsites
sanitize_latslice(i::Integer, g::GreenFunction) = latslice(selfenergies(contacts(g), i))
sanitize_latslice(sites, g) = lattice(g)[sites]

#endregion

############################################################################################
# Integrator - integrates a function f along a complex path ωcomplex(ω::Real), connecting ωi
#   The path is piecewise linear in the form of a sawtooth with a given ± slope
#region

struct Integrator{I,T,P,O<:NamedTuple,F}
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

struct JosephsonDensity{T<:AbstractFloat,P<:Union{Missing,AbstractArray},G<:GreenFunction{T}}
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
    Σ = similar_contactΣ(g, contact)
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
# ldos: local spectral density
#   d = ldos(::GreenSolution; kernel = I)      -> d[sites...]::Vector
#   d = ldos(::GreenSlice; kernel = I) -> d(ω; params...)::Vector
#   Here ldos is given as Tr(ρᵢᵢ * kernel) where ρᵢᵢ is the spectral function at site i
#   Here is the generic fallback that uses G. Any more specialized methods need to be added
#   to each GreenSolver
#region

struct LocalSpectralDensitySolution{T,E,L,G<:GreenSolution{T,E,L},K} <: IndexableObservable
    gω::G
    kernel::K                      # should return a float when applied to gω[cellsite(n,i)]
end

struct LocalSpectralDensitySlice{T,E,L,G<:GreenSlice{T,E,L},K}
    gs::G
    kernel::K                      # should return a float when applied to gω[cellsite(n,i)]
end

#region ## Constructors ##

ldos(gω::GreenSolution; kernel = I) = LocalSpectralDensitySolution(gω, kernel)

function ldos(gs::GreenSlice{T}; kernel = I) where {T}
    slicerows(gs) === slicecols(gs) ||
        argerror("Cannot take ldos of a GreenSlice with rows !== cols")
    return LocalSpectralDensitySlice(gs, kernel)
end

#endregion

#region ## API ##

greenfunction(d::LocalSpectralDensitySlice) = d.g

kernel(d::Union{LocalSpectralDensitySolution,LocalSpectralDensitySlice}) = d.kernel

Base.getindex(d::LocalSpectralDensitySolution; kw...) = d[getindex(lattice(d.gω); kw...)]

function Base.getindex(d::LocalSpectralDensitySolution{T}, l::LatticeSlice) where {T}
    v = T[]
    foreach(scell -> append_ldos!(v, scell, d.gω, d.kernel), subcells(l))
    return v
end

Base.getindex(d::LocalSpectralDensitySolution{T}, sites::Union{CellSites,Colon,Integer}) where {T} =
    append_ldos!(T[], sites, d.gω, d.kernel)

function call!(d::LocalSpectralDensitySlice{T}, ω; params...) where {T}
    sites = slicerows(d.gs)
    gω = call!(parent(d.gs), ω; params...)
    return append_ldos!(T[], sites, gω, d.kernel)
end

(d::LocalSpectralDensitySlice)(ω; params...) = copy(call!(d, ω; params...))

function append_ldos!(v, cs::CellSites, gω, kernel)
    gcell = gω[cs]
    bs = blockstructure(hamiltonian(gω))
    blocks = block_ranges(cs, bs)
    for rng in blocks
        gblock = view(gcell, rng, rng)
        ldos = ldos_kernel(gblock, kernel)
        push!(v, ldos)
    end
    return v
end

function append_ldos!(v, cind::Union{Colon,Integer}, gω, kernel)
    gcontact = view(gω, :, :)
    cbs = blockstructure(gω)
    blocks = block_ranges(cind, cbs)
    for rng in blocks
        gblock = view(gcontact, rng, rng)
        ldos = ldos_kernel(gblock, kernel)
        push!(v, ldos)
    end
    return v
end

ldos_kernel(g, kernel::UniformScaling) = - kernel.λ * imag(tr(g)) / π
ldos_kernel(g, kernel) = -imag(tr(g * kernel)) / π

#endregion
#endregion

############################################################################################
# conductance(gs::GreenSlice; nambu = false) -> G(ω; params...)::Real
#   For gs = g[i::Int, j::Int = i] -> we get zero temperature Gᵢⱼ = dIᵢ/dVⱼ in units of e^2/h
#   where i, j are contact indices
#       Gᵢⱼ =  e^2/h × Tr{[δᵢⱼi(Gʳ-Gᵃ)Γⁱ-GʳΓⁱGᵃΓʲ]}         (nambu = false)
#       Gᵢⱼ =  e^2/h × Tr{[δᵢⱼi(Gʳ-Gᵃ)Γⁱτₑ-GʳΓⁱτzGᵃΓʲτₑ]}   (nambu = true)
#   and where τₑ = [1 0; 0 0] and τz = [1 0; 0 -1] in Nambu space, and ω = eV.
#region

struct ConductanceSlice{T,E,L,C,G<:GreenFunction{T,E,L}}
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

function conductance(gs::GreenSlice{T}; nambu = false) where {T}
    i = slicerows(gs)
    j = slicecols(gs)
    check_contact_slice(i)
    check_contact_slice(j)
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
    return ConductanceSlice(g, i, j, τezdiag, Γ, GrΓi, GaΓj, GΓGΓ)
end

check_contact_slice(i) = i isa Integer ||
    argerror("Please use an Integer Green slice `g[i::Integer, j::Integer = i]` to compute the conductance `dIᵢ/dVⱼ` between contacts `i,j`")

#endregion

#region ## API ##

currentcontact(G) = G.i

biascontact(G) = G.j

function (G::ConductanceSlice)(ω; params...)
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
# current: current density Jᵢⱼ(ω) as a function of a charge operator
#   d = current(::GreenSolution[, dir]; charge)      -> d[sites...]::SparseMatrixCSC{SVector{E,T}}
#   d = current(::GreenSlice[, dir]; charge) -> d(ω; params...)::SparseMatrixCSC{SVector{E,T}}
#   Computes the zero-temperature equilibrium current density Jᵢⱼ from site j to site i
#       Jᵢⱼ(ω) = (2/h) rᵢⱼ Re Tr[(Hᵢⱼgʳⱼᵢ - gʳᵢⱼHⱼᵢ)Q]
#   Here charge = Q, where Q is usually qe*I for normal, and qe*τz/2 for Nambu systems
#   `dir` projects Jᵢⱼ along a certain direction, or takes the norm if missing
#   We use a default charge = -I, corresponding to normal currents densities in units of e/h
#region

struct CurrentDensitySolution{T,E,L,G<:GreenSolution{T,E,L},K,V<:Union{Missing,SVector}} <: IndexableObservable
    gω::G
    charge::K                               # should return a float when traced with gʳᵢⱼHᵢⱼ
    cache::GreenSolutionCache{T,L,G}        # memoizes g[sites]
    direction::V
end

struct CurrentDensitySlice{T,E,L,G<:GreenFunction{T,E,L},K,V<:Union{Missing,SVector}}
    g::G
    charge::K                               # should return a float when traced with gʳᵢⱼHᵢⱼ
    latslice::LatticeSlice{T,E,L}
    direction::V
end

#region ## Constructors ##

current(gω::GreenSolution; direction = missing, charge = -I) =
    CurrentDensitySolution(gω, charge, GreenSolutionCache(gω), sanitize_direction(direction, gω))

function current(gs::GreenSlice; direction = missing, charge = -I)
    slicerows(gs) === slicecols(gs) ||
        argerror("Cannot currently take ldos of a GreenSlice with rows !== cols")
    g = parent(gs)
    latslice = sanitize_latslice(slicerows(gs), g)
    return CurrentDensitySlice(g, charge, latslice, sanitize_direction(direction, g))
end

sanitize_direction(dir, ::GreenSolution{<:Any,E}) where {E} = _sanitize_direction(dir, Val(E))
sanitize_direction(dir, ::GreenFunction{<:Any,E}) where {E} = _sanitize_direction(dir, Val(E))
_sanitize_direction(::Missing, ::Val) = missing
_sanitize_direction(dir::Integer, ::Val{E}) where {E} = unitvector(dir, NTuple{E,Int})
_sanitize_direction(dir::SVector{E}, ::Val{E}) where {E} = dir
_sanitize_direction(dir::NTuple{E}, ::Val{E}) where {E} = SVector(dir)
_sanitize_direction(_, ::Val{E}) where {E} =
    argerror("Current direction should be an Integer or a NTuple/SVector of embedding dimension $E")

#endregion

#region ## API ##

charge(d::Union{CurrentDensitySolution,CurrentDensitySlice}) = d.charge

direction(d::Union{CurrentDensitySolution,CurrentDensitySlice}) = d.direction

Base.getindex(d::CurrentDensitySolution; kw...) = d[getindex(lattice(d.gω); kw...)]
Base.getindex(d::CurrentDensitySolution, ls::LatticeSlice) = current_matrix(d.gω, ls, d)
Base.getindex(d::CurrentDensitySolution, scell::CellSites) = d[lattice(hamiltonian(d.gω))[scell]]
Base.getindex(d::CurrentDensitySolution, i::Union{Integer,Colon}) = d[latslice(parent(d.gω), i)]

# no call! support here
function (d::CurrentDensitySlice)(ω; params...)
    gω = call!(d.g, ω; params...)
    ls = d.latslice
    cu = current(gω; charge = d.charge)
    return cu[ls]
end

function current_matrix(gω, ls, d)
    h = hamiltonian(parent(gω))
    current = h[ls, (hij, cij) -> maybe_project(apply_charge_current(hij, cij, d), d.direction)]
    return current
end

function apply_charge_current(hij_block::B, (ci, cj), d::CurrentDensitySolution{T,E}) where {T,E,B}
    ni, i = cell(ci), siteindex(ci)
    nj, j = cell(cj), siteindex(cj)
    ni == nj && i == j && return zero(SVector{E,T})
    gji = unblock(mask_block(B, d.cache[cj, ci]))
    gij = unblock(mask_block(B, d.cache[ci, cj]))
    hij = unblock(hij_block)
    hji = hij'
    hgghQ = (hij * gji - gij * hji) * d.charge
    # safeguard in case (hij * gji - gij * hji) isa Number and d.charge isa UniformScaling
    Iij = 2 * real(maybe_trace(hgghQ))
    lat = lattice(d.gω)
    ri = site(lat, i, ni)
    rj = site(lat, j, nj)
    Jij = (ri - rj) * Iij
    return Jij
end

maybe_project(J, ::Missing) = norm(J)
maybe_project(J, dir) = dot(dir, J)

maybe_trace(m::UniformScaling) = m.λ
maybe_trace(m) = tr(m)

#endregion

