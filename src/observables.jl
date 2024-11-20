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
# Operators
#region

function current(h::AbstractHamiltonian; charge = -I, direction = 1)
    current = parametric(h,
        @onsite!(o -> zero(o)),
        @hopping!((t, r, dr) -> im*dr[direction]*charge*t))
    return Operator(current)
end

#endregion

############################################################################################
# Integrator - integrates a function f along a complex path ωcomplex(ω::Real), connecting ωi
#   The path is piecewise linear in the form of a triangular sawtooth with a given ± slope
#region

struct Integrator{I,T,P,O<:NamedTuple,C,F}
    integrand::I    # call!(integrand, ω::Complex; params...)::Union{Number,Array{Number}}
    points::P       # a collection of points that form the triangular sawtooth integration path
    result::T       # can be missing (for scalar integrand) or a mutable type (nonscalar)
    quadgk_opts::O  # kwargs for quadgk
    callback::C     # callback to call at each integration step (callback(ω, i(ω)))
    post::F         # function to apply to the integrand at the end of integration
end

#region ## Constructor ##

function Integrator(result, f, pts; imshift = missing, post = identity, slope = 0, callback = Returns(nothing), quadgk_opts...)
    imshift´ = imshift === missing ?
        sqrt(eps(promote_type(typeof.(float.(real.(pts)))...))) : float(imshift)
    sanitize_integration_points(pts)
    pts´ = apply_complex_shifts(pts, imshift´, slope)
    quadgk_opts´ = NamedTuple(quadgk_opts)
    return Integrator(f, pts´, result, quadgk_opts´, callback, post)
end

Integrator(f, pts; kw...) = Integrator(missing, f, pts; kw...)

sanitize_integration_points(pts::Vector{<:Real}) = unique!(sort!(pts))
sanitize_integration_points(pts::Vector) = unique!(pts)
sanitize_integration_points(pts::AbstractRange) = sort(pts)

# fallback
function sanitize_integration_points(pts)
    if promote_type(typeof.(pts)...) <: Real
        issorted(pts) || argerror("Real integrated points should in general be sorted, got $pts")
    else
        allunique(pts) || argerror("Complex integrated points should in general be all unique, got $pts")
    end
    return pts
end

# If all points are real, apply sawtooth with slope
apply_complex_shifts(pts::NTuple{<:Any,Real}, imshift, slope) =
    iszero(slope) ? pts .+ (im * imshift) : triangular_sawtooth(imshift, slope, pts)
apply_complex_shifts(pts::AbstractVector{<:Real}, imshift, slope) =
    iszero(slope) ? (pts .+ (im * imshift)) : triangular_sawtooth(imshift, slope, pts)

# Otherwise do not modify
apply_complex_shifts(pts, imshift, slope) = pts

triangular_sawtooth(is, sl, ωs::Tuple) = _triangular_sawtooth(is, sl, (), ωs...)
_triangular_sawtooth(is, sl, ::Tuple{}, ω1, ωs...) = _triangular_sawtooth(is, sl, (ω1 + im * is,), ωs...)
_triangular_sawtooth(is, sl, ωs´, ωn, ωs...) = _triangular_sawtooth(is, sl,
    (ωs´..., 0.5 * (real(last(ωs´)) + ωn) + im * (is + sl * 0.5*(ωn - real(last(ωs´)))), ωn + im * is), ωs...)
_triangular_sawtooth(is, sl, ωs´) = ωs´

function triangular_sawtooth(imshift, slope, pts::Vector{T}) where {T<:Real}
    pts´ = pts .+ (im*imshift)
    for i in 2:length(pts)
        mid = 0.5 * (pts[i-1] + pts[i]) + im * (imshift + slope * 0.5 * (pts[i] - pts[i-1]))
        push!(pts´, mid)
    end
    sort!(pts´, by = real)
    return pts´
end

triangular_sawtooth(imshift, slope, pts) = triangular_sawtooth(imshift, slope, [pts...])

#endregion

#region ## API ##

integrand(I::Integrator) = I.integrand

path(I::Integrator) = I.points

options(I::Integrator) = I.quadgk_opts

## call! ##
# scalar version
function call!(I::Integrator{<:Any,Missing}; params...)
    fx = x -> begin
        y = call!(I.integrand, x; params...)  # should be a scalar
        I.callback(x, y)
        return y
    end
    result, err = quadgk(fx, I.points...; I.quadgk_opts...)
    result´ = I.post(result)
    return result´
end

# nonscalar version
function call!(I::Integrator; params...)
    fx! = (y, x) -> begin
        y .= serialize(call!(I.integrand, x; params...))
        I.callback(x, y)
        return y
    end
    result, err = quadgk!(fx!, serialize(I.result), I.points...; I.quadgk_opts...)
    # note: post-processing is not element-wise & can be in-place
    result´ = unsafe_deserialize(I.result, I.post(result))
    return result´
end

(I::Integrator)(; params...) = copy(call!(I; params...))

#endregion
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

# this produces gs(ω; params...) * f(ω-mu). Use post = gf_to_rho! after integration
struct DensityMatrixIntegrand{G<:GreenSlice,T,O}
    gs::G
    mu::T
    kBT::T
    omegamap::O
end

# Default solver (integration in complex plane)
struct DensityMatrixIntegratorSolver{I}
    ifunc::I
end

struct DensityMatrix{S,G<:GreenSlice}
    solver::S
    gs::G
end

#region ## Constructors ##

# generic fallback (for other solvers)
(ρ::DensityMatrix)(mu = 0, kBT = 0; params...) =
    ρ.solver(mu, kBT; params...)
# special case for integrator solver
(ρ::DensityMatrix{<:DensityMatrixIntegratorSolver})(mu = 0, kBT = 0, override_path = missing; params...) =
    ρ.solver(mu, kBT, override_path)(; params...)

(s::DensityMatrixIntegratorSolver)(mu, kBT, override_path = missing) =
    s.ifunc(mu, kBT, override_path);

# redirects to specialized method
densitymatrix(gs::GreenSlice; kw...) =
    densitymatrix(solver(parent(gs)), gs::GreenSlice; kw...)

# generic fallback
densitymatrix(s::AppliedGreenSolver, gs::GreenSlice; kw...) =
    argerror("Dedicated `densitymatrix` algorithm not implemented for $(nameof(typeof(s))). Use generic one instead.")

# default integrator solver
densitymatrix(gs::GreenSlice, ωmax::Number; opts...) = densitymatrix(gs, (-ωmax, ωmax); opts...)

function densitymatrix(gs::GreenSlice{T}, ωpoints; omegamap = Returns((;)), imshift = missing, atol = 1e-7, opts...) where {T}
    # check_nodiag_axes(gs)
    result = copy(call!_output(gs))
    opts´ = (; imshift, slope = 1, post = gf_to_rho!, atol, opts...)
    ωpoints_vec = collect(promote_type(T, typeof.(ωpoints)...), ωpoints)
    function ifunc(mu, kBT, override_path)
        ωpoints´ = override_path!(override_path, ωpoints_vec, ωpoints)
        ρd = DensityMatrixIntegrand(gs, T(mu), T(kBT), omegamap)
        pts = maybe_insert_mu!(ωpoints_vec, ωpoints´, mu, kBT)
        return Integrator(result, ρd, pts; opts´...)
    end
    return DensityMatrix(DensityMatrixIntegratorSolver(ifunc), gs)
end

# If all pts are real, maybe_insert_mu! inserts mu and orders. pts can be any container.
maybe_insert_mu!(pts´, pts, mu, kBT) =
    maybe_insert_mu!(pts´, pts, promote_type(typeof.(pts)...), mu, kBT)

maybe_insert_mu!(pts´, pts, _, mu, kBT) = pts
function maybe_insert_mu!(pts´, pts, ::Type{<:Real}, mu, kBT)
    # union spitting handles this type instability
    if (iszero(kBT) && maximum(pts) <= mu) || any(≈(mu), pts)
        return pts
    else
        return maybe_insert_mu!(copyto!(resize!(pts´, length(pts)), pts), mu, kBT)
    end
end

function maybe_insert_mu!(pts::AbstractVector{<:Real}, mu, kBT)
    sort!(push!(pts, mu))
    # If kBT = 0 it we filter out pts <= mu
    # it's unclear whether this is useful. It allocates less, but I don't see any speedup.
    iszero(kBT) && filter!(<=(mu), pts)
    return pts
end

maybe_insert_mu!(pts, mu, kBT) = pts

override_path!(::Missing, ptsvec, pts) = pts
override_path!(::Missing, ptsvec::Vector{<:Complex}, pts) = pts
override_path!(f::Function, ptsvec::Vector{<:Complex}, pts) = f.(pts)

function override_path!(pts´, ptsvec::Vector{<:Complex}, pts)
    resize!(ptsvec, length(pts))
    return pts´
end

override_path!(override_path, ptsvec, pts) =
    argerror("Override of real ωpoints not supported, use complex ωpoints upon construction")

#endregion

#region ## API ##

(gf::DensityMatrixIntegrand)(ω; params...) = copy(call!(gf, ω; params...))

function call!(gf::DensityMatrixIntegrand, ω; params...)
    output = call!(gf.gs, ω; gf.omegamap(ω)..., params...)
    serialize(output) .*= fermi(ω - gf.mu, inv(gf.kBT))
    return output
end

function gf_to_rho!(x::AbstractMatrix)
    x .= x .- x'
    x .*= -1/(2π*im)
    return x
end

# For diagonal indexing
function gf_to_rho!(x::AbstractVector)
    x .= x .- conj.(x)
    x .*= -1/(2π*im)
    return x
end

integrand(ρ::DensityMatrix{<:DensityMatrixIntegratorSolver}, mu = 0.0, kBT = 0.0) = integrand(ρ.solver(mu, kBT))

path(ρ::DensityMatrix{<:DensityMatrixIntegratorSolver}, mu = 0.0, kBT = 0.0) = path(ρ.solver(mu, kBT))

temperature(D::DensityMatrixIntegrand) = D.kBT

chemicalpotential(D::DensityMatrixIntegrand) = D.mu

Base.parent(ρ::DensityMatrix) = ρ.gs

call!_output(ρ::DensityMatrix) = call!_output(ρ.gs)

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

struct JosephsonIntegrand{T<:AbstractFloat,P<:Union{Missing,AbstractArray},O,G<:GreenFunction{T}}
    g::G
    kBT::T
    contactind::Int             # contact index
    tauz::Vector{Int}           # precomputed diagonal of tauz
    phaseshifts::P              # missing or collection of phase shifts to apply
    omegamap::O                 # function that maps ω to parameters
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

#region ## Constructors ##

# generic fallback (for other solvers)
(j::Josephson)(kBT = 0; params...) = j.solver(kBT; params...)
# special case for integrator solver
(j::Josephson{<:JosephsonIntegratorSolver})(kBT = 0, override_path = missing; params...) =
    j.solver(kBT, override_path)(; params...)

(s::JosephsonIntegratorSolver)(kBT, override_path = missing) = s.ifunc(kBT, override_path)

josephson(gs::GreenSlice{T}, ωmax::Number; kw...) where {T} =
    josephson(gs, (-ωmax, ωmax); kw...)

function josephson(gs::GreenSlice{T}, ωpoints; omegamap = Returns((;)), phases = missing, imshift = missing, atol = 1e-7, opts...) where {T}
    check_nodiag_axes(gs)
    check_same_contact_slice(gs)
    contact = rows(gs)
    g = parent(gs)
    Σfull = similar_contactΣ(g)
    Σ = similar_contactΣ(g, contact)
    normalsize = normal_size(hamiltonian(g))
    tauz = tauz_diag.(axes(Σ, 1), normalsize)
    phases´, traces = sanitize_phases_traces(phases, T)
    opts´ = (; imshift, slope = 1, post = real, atol, opts...)
    ωpoints_vec = collect(promote_type(T, typeof.(ωpoints)...), ωpoints)
    function ifunc(kBT, override_path)
        ωpoints´ = override_path!(override_path, ωpoints_vec, ωpoints)
        jd = JosephsonIntegrand(g, T(kBT), contact, tauz, phases´, omegamap,
            traces, Σfull, Σ, similar(Σ), similar(Σ), similar(Σ), similar(tauz, Complex{T}))
        pts = maybe_insert_mu!(ωpoints_vec, ωpoints´, zero(T), kBT)
        return Integrator(traces, jd, pts; opts´...)
    end
    return Josephson(JosephsonIntegratorSolver(ifunc), gs)
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

integrand(J::Josephson{<:JosephsonIntegratorSolver}, kBT = 0.0) = integrand(J.solver(kBT))

path(J::Josephson{<:JosephsonIntegratorSolver}, kBT = 0.0) = path(J.solver(kBT))

temperature(J::JosephsonIntegrand) = J.kBT

contact(J::JosephsonIntegrand) = J.contactind

phaseshifts(I::Integrator{<:JosephsonIntegrand}) = phaseshifts(integrand(I))
phaseshifts(J::JosephsonIntegrand) = real.(J.phaseshifts)

numphaseshifts(J::JosephsonIntegrand) = numphaseshifts(J.phaseshifts)
numphaseshifts(::Missing) = 0
numphaseshifts(phaseshifts) = length(phaseshifts)

(J::JosephsonIntegrand)(ω; params...) = copy(call!(J, ω; params...))

function call!(J::JosephsonIntegrand, ω; params...)
    gω = call!(J.g, ω; J.omegamap(ω)..., params...)
    f = fermi(ω, inv(J.kBT))
    traces = josephson_traces(J, gω, f)
    return traces
end

function josephson_traces(J, gω, f)
    gr = view(gω, J.contactind, J.contactind)
    Σi = selfenergy!(J.Σ, gω, J.contactind)
    return josephson_traces!(J, gr, Σi, f)
end

josephson_traces!(J::JosephsonIntegrand{<:Any,Missing}, gr, Σi, f) = josephson_one_trace!(J, gr, Σi, f)

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

function gaps(h::Hamiltonian{T,<:Any,1}, µ = 0, ϕstore = missing; atol = eps(T)) where {T}
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
    solver = ES.ShiftInvert(ES.ArnoldiMethod(nev = 1), µ)
    n = flatsize(h)
    Δs = [iszero(iϕ) || rank(h(rϕ)-µ*I; tol=atol) < n ? zero(T) : abs(first(energies(spectrum(h, rϕ; solver)))-µ) for (rϕ, iϕ) in zip(rϕs, iϕs)]
    return Δs
end

gap(h::AbstractHamiltonian{<:Any,<:Any,1}, args...; kw...) =
    minimum(gaps(h, args...; kw...))

#endregion
