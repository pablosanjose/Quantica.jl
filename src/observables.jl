############################################################################################
# Observables - common tools
#region

abstract type IndexableObservable end  # any type that can be sliced into (current and ldos)

fermi(ω::C, β = Inf; atol = sqrt(eps(real(C)))) where {C} =
    isinf(β) ? ifelse(abs(ω) < atol, C(0.5), ifelse(real(ω) <= 0, C(1), C(0))) : C(1/(exp(β * ω) + 1))

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

check_contact_slice(gs) = (rows(gs) isa Integer && cols(gs) isa Integer) ||
    argerror("Please use a Green slice of the form `g[i::Integer, j::Integer]` or `g[i::Integer]`")

check_same_contact_slice(gs) = (rows(gs) isa Integer && cols(gs) === rows(gs)) ||
    argerror("Please use a Green slice of the form `g[i::Integer]`")

check_different_contact_slice(gs) = (rows(gs) isa Integer && cols(gs) != rows(gs)) ||
    argerror("Please use a Green slice of the form `g[i::Integer, j::Integer] with `i ≠ j`")

check_nodiag_axes(gs::GreenSlice) = check_nodiag_axes(rows(gs)), check_nodiag_axes(rows(gs))
check_nodiag_axes(::DiagIndices) = argerror("Diagonal indexing not yet supported for this function")
check_nodiag_axes(_) = nothing

#endregion

############################################################################################
# Operators (wrapped AbstractHamiltonians representing observables other than energy)
#region

function current(h::AbstractHamiltonian; charge = -I, direction = 1)
    current = parametric(h,
        @onsite!(o -> zero(o)),
        @hopping!((t, r, dr) -> im*dr[direction]*charge*t))
    return Operator(current)
end

#endregion

############################################################################################
# ldos: local spectral density
#   d = ldos(::GreenSolution; kernel = missing)      -> d[sites...]::Vector
#   d = ldos(::GreenSlice; kernel = missing)         -> d(ω; params...)::Vector
#   Here ldos is given as Tr(ρᵢᵢ * kernel) where ρᵢᵢ is the spectral function at site i
#   Here is the generic fallback that uses G. Any more specialized methods need to be added
#   to each GreenSolver
#region

struct LocalSpectralDensitySolution{T,E,L,G<:GreenSolution{T,E,L},K} <: IndexableObservable
    gω::G
    kernel::K
end

struct LocalSpectralDensitySlice{T,E,L,G<:GreenSlice{T,E,L},K}
    gs::G
    kernel::K   # also inside gs
end

#region ## Constructors ##

ldos(gω::GreenSolution; kernel = missing) = LocalSpectralDensitySolution(gω, kernel)

function ldos(gs::GreenSlice{T}; kernel = missing) where {T}
    rows(gs) === cols(gs) ||
        argerror("Cannot take ldos of a GreenSlice with rows !== cols")
    g = parent(gs)
    i = ensure_diag_axes(rows(gs), kernel)
    gs´ = GreenSlice(g, i, i, T)  # forces output eltype to be T<:Real
    return LocalSpectralDensitySlice(gs´, kernel)
end

ensure_diag_axes(inds, kernel) = diagonal(inds; kernel)
ensure_diag_axes(inds::DiagIndices, kernel) = diagonal(parent(inds); kernel)
ensure_diag_axes(::SparseIndices, _) =
    argerror("Unexpected sparse indices in GreenSlice")

#endregion

#region ## API ##

greenfunction(d::LocalSpectralDensitySlice) = d.g

kernel(d::Union{LocalSpectralDensitySolution,LocalSpectralDensitySlice}) = d.kernel

Base.getindex(d::LocalSpectralDensitySolution; selectors...) =
    d[siteselector(; selectors...)]

function Base.getindex(d::LocalSpectralDensitySolution{T}, i) where {T}
    di = diagonal(i, kernel = d.kernel)
    gs = GreenSlice(parent(d.gω), di, di, T)  # get GreenSlice with real output
    output = getindex!(gs, d.gω; post = x -> -imag(x)/π)
    return diag(output)
end

(d::LocalSpectralDensitySlice)(ω; params...) = copy(call!(d, ω; params...))

# fallback through LocalSpectralDensitySolution - overload to allow a more efficient path
function call!(d::LocalSpectralDensitySlice{T}, ω; params...) where {T}
    output = call!(d.gs, ω; post = x -> -imag(x)/π, params...)
    return diag(output)
end

#endregion
#endregion

############################################################################################
# current: current density Jᵢⱼ(ω) as a function of a charge operator
#   d = current(::GreenSolution[, dir]; charge) -> d[sites...]::SparseMatrixCSC{SVector{E,T}}
#   d = current(::GreenSlice[, dir]; charge) -> d(ω; params...)::SparseMatrixCSC{SVector{E,T}}
#   Computes the zero-temperature equilibrium current density Jᵢⱼ from site j to site i
#       Jᵢⱼ(ω) = (2/h) rᵢⱼ Re Tr[(Hᵢⱼgʳⱼᵢ - gʳᵢⱼHⱼᵢ)Q]
#   Here charge = Q, where Q is usually qe*I for normal, and qe*τz/2 for Nambu systems
#   `dir` projects Jᵢⱼ along a certain direction, or takes the norm if missing
#   We use a default charge = -I, corresponding to normal currents densities in units of e/h
#region

## TODO: this could probably be refactored using sparse indexing
## removing GreenSolutionCache and unflat_sparse_slice in the process

struct CurrentDensitySolution{T,E,L,G<:GreenSolution{T,E,L},K,V<:Union{Missing,SVector}} <: IndexableObservable
    gω::G
    charge::K                               # should return a float when traced with gʳᵢⱼHᵢⱼ
    cache::GreenSolutionCache{T,L,G}        # memoizes g[sites]
    direction::V
end

struct CurrentDensitySlice{T,E,L,G<:GreenFunction{T,E,L},K,V<:Union{Missing,SVector}}
    g::G
    charge::K                               # should return a float when traced with gʳᵢⱼHᵢⱼ
    orbslice::OrbitalSliceGrouped{T,E,L}
    direction::V
end

#region ## Constructors ##

current(gω::GreenSolution; direction = missing, charge = -I) =
    CurrentDensitySolution(gω, charge, GreenSolutionCache(gω), sanitize_direction(direction, gω))

function current(gs::GreenSlice; direction = missing, charge = -I)
    rows(gs) === cols(gs) ||
        argerror("Cannot currently take ldos of a GreenSlice with rows !== cols")
    g = parent(gs)
    orbslice = orbrows(gs)
    return CurrentDensitySlice(g, charge, orbslice, sanitize_direction(direction, g))
end

sanitize_direction(dir, ::GreenSolution{<:Any,E}) where {E} = _sanitize_direction(dir, Val(E))
sanitize_direction(dir, ::GreenFunction{<:Any,E}) where {E} = _sanitize_direction(dir, Val(E))
_sanitize_direction(::Missing, ::Val) = missing
_sanitize_direction(dir::Integer, ::Val{E}) where {E} = unitvector(dir, SVector{E,Int})
_sanitize_direction(dir::SVector{E}, ::Val{E}) where {E} = dir
_sanitize_direction(dir::NTuple{E}, ::Val{E}) where {E} = SVector(dir)
_sanitize_direction(_, ::Val{E}) where {E} =
    argerror("Current direction should be an Integer or a NTuple/SVector of embedding dimension $E")

#endregion

#region ## API ##

charge(d::Union{CurrentDensitySolution,CurrentDensitySlice}) = d.charge

direction(d::Union{CurrentDensitySolution,CurrentDensitySlice}) = d.direction

Base.getindex(d::CurrentDensitySolution; kw...) = d[getindex(lattice(d.gω); kw...)]
Base.getindex(d::CurrentDensitySolution, scell::CellSites) = d[lattice(hamiltonian(d.gω))[scell]]
Base.getindex(d::CurrentDensitySolution, i::Union{Integer,Colon}) = d[latslice(parent(d.gω), i)]
Base.getindex(d::CurrentDensitySolution, ls::LatticeSlice) = current_matrix(d.gω, ls, d)

# no call! support here
function (d::CurrentDensitySlice)(ω; params...)
    gω = call!(d.g, ω; params...)
    ls = d.orbslice
    cu = current(gω; charge = d.charge)
    return cu[ls]
end

function current_matrix(gω, ls, d)
    h = hamiltonian(parent(gω))
    # see slices.jl for unflat_getindex
    current = unflat_sparse_slice(h, ls, ls,
        (hij, cij) -> maybe_project(apply_charge_current(hij, cij, d), d.direction))
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
#endregion

############################################################################################
# conductance(gs::GreenSlice; nambu = false) -> G(ω; params...)::Real
#   For gs = g[i::Int, j::Int = i] -> we get zero temperature Gᵢⱼ = dIᵢ/dVⱼ in units of e^2/h
#   where i, j are contact indices
#       Gᵢⱼ =  e^2/h × Tr{[δᵢⱼi(Gʳ-Gᵃ)Γⁱ-GʳΓⁱGᵃΓʲ]}         (nambu = false)
#       Gᵢⱼ =  e^2/h × Tr{[δᵢⱼi(Gʳ-Gᵃ)Γⁱτₑ-GʳΓⁱτzGᵃΓʲτₑ]}   (nambu = true)
#   and where τₑ = [1 0; 0 0] and τz = [1 0; 0 -1] in Nambu space, and ω = eV.
#region

struct Conductance{T,E,L,C,G<:GreenFunction{T,E,L}}
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
    check_contact_slice(gs)
    i = rows(gs)
    j = cols(gs)
    g = parent(gs)
    ni = norbitals(contactorbitals(g), i)
    nj = norbitals(contactorbitals(g), j)
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

function (G::Conductance)(ω; params...)
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
# tramsission(gs::GreenSlice) -> normal Tᵢⱼ = Tr{GʳΓⁱGᵃΓʲ} from contact j to i ≠ j
#region

struct Transmission{G<:Conductance}
    conductance::G
end

#region ## Constructors ##

function transmission(gs::GreenSlice)
    check_different_contact_slice(gs)
    return Transmission(conductance(gs; nambu = false))
end

#endregion

#region ## API ##

Base.parent(t::Transmission) = t.conductance

function (T::Transmission)(ω; params...)
    G = T.conductance
    gω = call!(G.g, ω; params...)
    gʳⱼᵢ = gω[G.j, G.i]
    gᵃᵢⱼ = gʳⱼᵢ'
    Γi = selfenergy!(G.Γ, gω, G.i; onlyΓ = true)
    mul!(G.GrΓi, gʳⱼᵢ, Γi)
    Γj = G.i == G.j ? Γi : selfenergy!(G.Γ, gω, G.j; onlyΓ = true)
    mul!(G.GaΓj, gᵃᵢⱼ, Γj)
    mul!(G.GΓGΓ, G.GrΓi, G.GaΓj)
    t = real(tr(G.GΓGΓ))
    return t
end

#endregion
#endregion

############################################################################################
# densitymatrix: equilibrium (static) ρ::DensityMatrix
#   ρ = densitymatrix(g::GreenSlice, ωpoints; opts...)
#   ρ(mu, kBT = 0; params...) gives the DensityMatrix that is solved with an integral over
#   a polygonal path connecting (ωpoints...) in the complex plane,
#       ρ(mu, kBT; params...) = -(1/π) Im ∫dω f(ω) g(ω; params...)
#   ρ = densitymatrix(g::GreenSlice; opts...) uses a GreenSolver-specific algorithm
#   Keywords opts are passed to QuadGK.quadgk for the integral or the algorithm used
#region

# produces integrand_transform!(gs(ω´; omegamap(ω´)..., params...) * f(ω´-mu))
# with ω´ = path_transform(ω)
struct DensityMatrixIntegrand{T,GF<:Function,P<:AbstractIntegrationPath,PT}
    gsfunc::GF         # (ω, symmetrize) -> gs(ω; symmetrize, omegamap(ω)..., params...)
    mu::T
    kBT::T
    path::P            # AbstractIntegrationPath object
    pts::PT            # ω-points that define specific integration path, derived from `path`
end

# Default solver (integration in complex plane)
struct DensityMatrixIntegratorSolver{I}
    ifunc::I
end

struct DensityMatrix{S,G<:GreenSlice}
    solver::S
    gs::G
end

#region ## densitymatrix API ##

# redirects to specialized method
densitymatrix(gs::GreenSlice; kw...) =
    densitymatrix(solver(parent(gs)), gs::GreenSlice; kw...)

# generic fallback if no specialized method exists
densitymatrix(s::AppliedGreenSolver, gs::GreenSlice; kw...) =
    argerror("Dedicated `densitymatrix` algorithm not implemented for $(nameof(typeof(s))). Use generic one instead.")

# default integrator solver
densitymatrix(gs::GreenSlice, ωmax::Real; kw...) =
    densitymatrix(gs::GreenSlice, Paths.radial(ωmax, π/4); kw...)

densitymatrix(gs::GreenSlice, ωs::NTuple{<:Any,Real}; kw...) =
    densitymatrix(gs::GreenSlice, Paths.sawtooth(ωs); kw...)

function densitymatrix(gs::GreenSlice{T}, path::AbstractIntegrationPath; omegamap = Returns((;)), atol = 1e-7, opts...) where {T}
    result = copy(call!_output(gs))
    post = post_transform_rho(path, gs)
    opts´ = (; post, atol, opts...)
    function ifunc(mu, kBT; params...)
        pts = points(path, mu, kBT; params...)
        realpts = realpoints(path, pts)
        gsfunc(ω, symmetrize) = call!(gs, ω; symmetrize, omegamap(ω)..., params...)
        ρd = DensityMatrixIntegrand(gsfunc, T(mu), T(kBT), path, pts)
        return Integrator(result, ρd, realpts; opts´...)
    end
    return DensityMatrix(DensityMatrixIntegratorSolver(ifunc), gs)
end

# we need to add the arc path segment from -∞ to ∞ * p.cisinf
function post_transform_rho(p::RadialPath, gs)
    arc = gs((p.angle/π)*I)
    function post(x)
        x .+= arc
        return x
    end
    return post
end

post_transform_rho(::AbstractIntegrationPath, _) = identity

#endregion

#region ## call API ##

# generic fallback (for other solvers)
(ρ::DensityMatrix)(mu = 0, kBT = 0; params...) =
    ρ.solver(mu, kBT; params...)
# special case for integrator solver
(ρ::DensityMatrix{<:DensityMatrixIntegratorSolver})(mu = 0, kBT = 0; params...) =
    ρ.solver(mu, kBT; params...)()

(s::DensityMatrixIntegratorSolver)(mu, kBT; params...) =
    s.ifunc(mu, kBT; params...);

(ρi::DensityMatrixIntegrand)(x) = copy(call!(ρi, x))

function call!(ρi::DensityMatrixIntegrand, x)
    ω = point(x, ρi.path, ρi.pts)
    j = jacobian(x, ρi.path, ρi.pts)
    f = fermi(chopsmall(ω - ρi.mu), inv(ρi.kBT))
    symmetrize = -j*f/(2π*im)
    output = ρi.gsfunc(ω, symmetrize)
    return output
end

#endregion

#region ## API ##

integrand(ρ::DensityMatrix{<:DensityMatrixIntegratorSolver}, mu = 0.0, kBT = 0.0; params...) =
    integrand(ρ.solver(mu, kBT; params...))

points(ρ::DensityMatrix{<:DensityMatrixIntegratorSolver}, mu = 0.0, kBT = 0.0; params...) =
    points(integrand(ρ, mu, kBT; params...))
points(ρ::DensityMatrixIntegrand) = ρ.pts

point(x, ρi::DensityMatrixIntegrand) = point(x, ρi.path, ρi.pts)

temperature(D::DensityMatrixIntegrand) = D.kBT

chemicalpotential(D::DensityMatrixIntegrand) = D.mu

Base.parent(ρ::DensityMatrix) = ρ.gs

call!_output(ρ::DensityMatrix) = call!_output(ρ.gs)

solver(ρ::DensityMatrix) = ρ.solver

#endregion

#endregion
#endregion


############################################################################################
# josephson
#   The equilibrium (static) Josephson current, in units of qe/h, *from* lead i is given by
#       Iᵢ = Re ∫dω J(ω; params...), where J(ω; params...) = (qe/h) × 2f(ω)Tr[(ΣʳᵢGʳ - GʳΣʳᵢ)τz]
#   J = josephson(g::GreenSlice, ωmax; contact = i, kBT = 0, phases)
#   J(ω; params...) -> scalar or vector [J(ϕⱼ) for ϕⱼ in phases] if `phases` is an
#       integer (num phases from 0 to π) or a collection of ϕ's
#   A phase ϕ can be applied by gauging it away from the lead and into its coupling:
#       Σʳᵢ(ϕ) = UᵩΣʳᵢUᵩ' and Gʳ(ϕ) = [1+Gʳ(Σʳᵢ-Σʳᵢ(ϕ))]⁻¹Gʳ, where Uᵩ = exp(iϕτz/2).
#   I = josephson(Integrator(J, (-ωmax, 0, ωmax); post = real, opts...)
#   Keywords opts are passed to quadgk for the integral
#region

struct JosephsonIntegrand{T<:AbstractFloat,P<:Union{Missing,AbstractArray},GF<:Function,PA,PT}
    gfunc::GF                   # ω -> g(ω; params...)
    kBT::T
    contactind::Int             # contact index
    tauz::Vector{Int}           # precomputed diagonal of tauz
    phaseshifts::P              # missing or collection of phase shifts to apply
    path::PA                    # AbstractIntegrationPath
    pts::PT                     # points in actual integration path, derived from `path`
    traces::P                   # preallocated workspace
    Σ::Matrix{Complex{T}}       # preallocated workspace, full self-energy
    ΣggΣ::Matrix{Complex{T}}    # preallocated workspace
    Σ´::Matrix{Complex{T}}      # preallocated workspace
    g´::Matrix{Complex{T}}      # preallocated workspace
    den::Matrix{Complex{T}}     # preallocated workspace
    cisτz::Vector{Complex{T}}   # preallocated workspace
end

struct Josephson{S,G<:GreenSlice}
    solver::S
    gs::G  # currently unused, but parallels DensityMatrix (used for OrbitalSliceArray conv)
end

# default solver (integration in complex plane)
struct JosephsonIntegratorSolver{I}
    ifunc::I
end

#region ## josephson API ##

josephson(gs::GreenSlice, ωmax::Real; kw...) =
    josephson(gs, Paths.radial(ωmax, π/4); kw...)

josephson(gs::GreenSlice, ωs::NTuple{<:Any,Real}; kw...) =
    josephson(gs, Paths.sawtooth(ωs); kw...)

function josephson(gs::GreenSlice{T}, path::AbstractIntegrationPath; omegamap = Returns((;)), phases = missing, atol = 1e-7, opts...) where {T}
    check_nodiag_axes(gs)
    check_same_contact_slice(gs)
    contact = rows(gs)
    g = parent(gs)
    Σfull = similar_contactΣ(g)
    Σ = similar_contactΣ(g, contact)
    normalsize = normal_size(hamiltonian(g))
    tauz = tauz_diag.(axes(Σ, 1), normalsize)
    phases´, traces = sanitize_phases_traces(phases, T)
    opts´ = (; post = real, atol, opts...)
    function ifunc(kBT; params...)
        pts = points(path, 0, kBT; params...)
        realpts = realpoints(path, pts)
        gfunc(ω) = call!(g, ω; omegamap(ω)..., params...)
        jd = JosephsonIntegrand(gfunc, T(kBT), contact, tauz, phases´, path, pts,
            traces, Σfull, Σ, similar(Σ), similar(Σ), similar(Σ), similar(tauz, Complex{T}))
        return Integrator(traces, jd, realpts; opts´...)
    end
    return Josephson(JosephsonIntegratorSolver(ifunc), gs)
end

sanitize_phases_traces(::Missing, ::Type{T}) where {T} = missing, missing
sanitize_phases_traces(phases::Integer, ::Type{T}) where {T} =
    sanitize_phases_traces(range(0, 2π, length = phases), T)

function sanitize_phases_traces(phases, ::Type{T}) where {T}
    phases´ = Complex{T}.(phases)
    traces = similar(phases´)
    return phases´, traces
end

#endregion

#region ## call API ##

# generic fallback (for other solvers)
(j::Josephson)(kBT = 0; params...) = j.solver(kBT; params...)
# special case for integrator solver (so we can access integrand etc before integrating)
(j::Josephson{<:JosephsonIntegratorSolver})(kBT = 0; params...) =
    j.solver(kBT; params...)()

(s::JosephsonIntegratorSolver)(kBT; params...) = s.ifunc(kBT; params...)

(J::JosephsonIntegrand)(x) = copy(call!(J, x))

function call!(Ji::JosephsonIntegrand, x)
    ω = point(x, Ji.path, Ji.pts)
    gω = Ji.gfunc(ω)
    f = fermi(ω, inv(Ji.kBT))
    traces = josephson_traces(Ji, gω, f)
    traces = mul_scalar_or_array!(traces, jacobian(x, Ji.path, Ji.pts))
    return traces
end

#endregion

#region ## API ##

integrand(J::Josephson{<:JosephsonIntegratorSolver}, kBT = 0.0; params...) =
    integrand(J.solver(kBT; params...))

points(J::Josephson{<:JosephsonIntegratorSolver}, kBT = 0.0; params...) =
    points(integrand(J, kBT; params...))
points(J::JosephsonIntegrand) = J.pts

point(x, Ji::JosephsonIntegrand) = point(x, Ji.path, Ji.pts)

temperature(J::JosephsonIntegrand) = J.kBT

contact(J::JosephsonIntegrand) = J.contactind

phaseshifts(I::Integrator{<:JosephsonIntegrand}) = phaseshifts(integrand(I))
phaseshifts(J::JosephsonIntegrand) = real.(J.phaseshifts)

numphaseshifts(J::JosephsonIntegrand) = numphaseshifts(J.phaseshifts)
numphaseshifts(::Missing) = 0
numphaseshifts(phaseshifts) = length(phaseshifts)

function josephson_traces(J, gω, f)
    gr = view(gω, J.contactind, J.contactind)
    Σi = selfenergy!(J.Σ, gω, J.contactind)
    return josephson_traces!(J, gr, Σi, f)
end

josephson_traces!(J::JosephsonIntegrand{<:Any,Missing}, gr, Σi, f) =
    josephson_one_trace!(J, gr, Σi, f)

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
# gap(::Hamiltonian, µ = 0, ...) -> minimum gap in the bands around µ
#region

function gaps(h::Hamiltonian{T,<:Any,1}, µ = 0, ϕstore = missing; atol = eps(T), nev = 1, kw...) where {T}
    g = greenfunction(h, GS.Schur())
    λs = schur_eigvals(g, µ)
    cϕs = λs .= -im .* log.(λs) # saves one allocation
    # remove duplicates within tolerance
    sort!(cϕs, by = real)
    runs = Runs(cϕs, (x, y) -> isapprox(real(x), real(y); atol))
    cϕs = [cϕs[first(rng)] for rng in runs]
    rϕs = real.(cϕs)
    iϕs = chopsmall.(abs.(imag.(cϕs)), atol)
    ϕstore === missing || copy!(ϕstore, cϕs)
    solver = ES.ShiftInvert(ES.ArnoldiMethod(; nev, kw...), µ)
    n = flatsize(h)
    Δs = [iszero(iϕ) || rank(h(rϕ)-µ*I; tol=atol) < n ?
        zero(T) :
        abs(minimum(x->abs(x-µ), energies(spectrum(h, rϕ; solver)))-µ) for (rϕ, iϕ) in zip(rϕs, iϕs)]
    return Δs
end

gap(h::AbstractHamiltonian{<:Any,<:Any,1}, args...; kw...) =
    minimum(gaps(h, args...; kw...))

#endregion
