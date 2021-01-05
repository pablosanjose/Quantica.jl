#######################################################################
# Green's function
#######################################################################
abstract type AbstractGreensSolver end

struct GreensFunction{S<:AbstractGreensSolver,L,B<:NTuple{L,Union{Int,Missing}},H<:Hamiltonian}
    solver::S
    h::H
    boundaries::B
end

"""
    greens(h::Hamiltonian, solveobject; boundaries::NTuple{L,Integer} = missing)

Construct the Green's function `g::GreensFunction` of `L`-dimensional Hamiltonian `h` using
the provided `solveobject`. Currently valid `solveobject`s are

- the `Bandstructure` of `h` (for an unbounded `h` or an `Hamiltonian{<:Superlattice}}`)
- the `Spectrum` of `h` (for a bounded `h`)
- `SingleShot1D()` (for 1D Hamiltonians, to use the single-shot generalized eigenvalue approach)

If a `boundaries = (n₁, n₂, ...)` is provided, a reflecting boundary is assumed for each
non-missing `nᵢ` perpendicular to Bravais vector `i` at a cell distance `nᵢ` from the
origin.

    h |> greens(h -> solveobject(h), args...)

Curried form equivalent to the above, giving `greens(h, solveobject(h), args...)`.

    g(ω, cells::Pair = missing)

From a constructed `g::GreensFunction`, obtain the retarded Green's function matrix at
frequency `ω` between unit cells `src` and `dst` by calling `g(ω, src => dst)`, where `src,
dst` are `::NTuple{L,Int}` or `SVector{L,Int}`. If `cells` is missing, `src` and `dst` are
assumed to be zero vectors. See also `greens!` to use a preallocated matrix.

# Examples

```jldoctest
julia> g = LatticePresets.square() |> hamiltonian(hopping(-1)) |> greens(bandstructure(resolution = 17))
GreensFunction{Bandstructure}: Green's function from a 2D bandstructure
  Matrix size    : 1 × 1
  Element type   : scalar (Complex{Float64})
  Band simplices : 512

julia> g(0.2)
1×1 Array{Complex{Float64},2}:
 6.663377810046025 - 24.472789025006396im

julia> m = similarmatrix(g); g(m, 0.2)
1×1 Array{Complex{Float64},2}:
 6.663377810046025 - 24.472789025006396im
```

# See also
    `greens!`
"""
greens(h::Hamiltonian{<:Any,L}, solverobject; boundaries = filltuple(missing, Val(L))) where {L} =
    GreensFunction(greensolver(solverobject, h), h, boundaries)
greens(solver::Function, args...; kw...) = h -> greens(solver(h), h, args...; kw...)

# fallback
greensolver(s::AbstractGreensSolver) = s

# call API
(g::GreensFunction)(ω::Number, cells = missing) = greens!(similarmatrix(g), g, ω, cells)

similarmatrix(g::GreensFunction, type = Matrix{blocktype(g.h)}) = similarmatrix(g.h, type)

greens!(matrix, g, ω, cells) = greens!(matrix, g, ω, sanitize_cells(cells, g))

sanitize_cells(::Missing, ::GreensFunction{S,L}) where {S,L} =
    zero(SVector{L,Int}) => zero(SVector{L,Int})
sanitize_cells((cell0, cell1)::Pair{Integer,Integer}, ::GreensFunction{S,1}) where {S} =
    (cell0,) => (cell1,)
sanitize_cells(cells::Pair{NTuple{L,Integer},NTuple{L,Integer}}, ::GreensFunction{S,L}) where {S,L} =
    cells
sanitize_cells(cells, g::GreensFunction{S,L}) where {S,L} =
    throw(ArgumentError("Cells should be of the form `cᵢ => cⱼ`, with each `c` an `NTuple{$L,Integer}`"))

const SVectorPair{L} = Pair{SVector{L,Int},SVector{L,Int}}

#######################################################################
# SingleShot1DGreensSolver
#######################################################################
struct SingleShot1D end

struct SingleShot1DGreensSolver{T,O<:OrbitalStructure} <: AbstractGreensSolver
    A::Matrix{T}
    B::Matrix{T}
    Acopy::Matrix{T}
    Bcopy::Matrix{T}
    maxdn::Int
    orbstruct::O
    H0block::CartesianIndices{2,Tuple{UnitRange{Int64}, UnitRange{Int64}}}
    H0diag::Vector{T}
end
#=
Precomputes A = [0 I 0...; 0 0 I; ...; -V₂ -V₁... ω-H₀ -V₋₁] and B = [I 0 0...; 0 I 0...;...; ... 0 0 V₋₂]
(form for two Harmonics {V₁,V₂})
These define the eigenvalue problem A*φ = λ B*φ, with λ = exp(i kx a0) and φ = [φ₀, φ₁, φ₂...φₘ],
where φₙ = λ φₙ₋₁ = λⁿ φ₀, and m = max
Since we need at least half the eigenpairs, we use LAPACK, and hence dense matrices.
=#
function SingleShot1DGreensSolver(h::Hamiltonian)
    latdim(h) == 1 || throw(ArgumentError("Cannot use a SingleShot1D Green function solver with an $(latdim(h))-dimensional Hamiltonian"))
    maxdn = maximum(har -> abs(first(har.dn)), h.harmonics)
    H = unitcell(h, (maxdn,))
    dimh = flatsize(H, 1)
    T = complex(blockeltype(H))
    A = zeros(T, 2dimh, 2dimh)
    B = zeros(T, 2dimh, 2dimh)
    @show H
    H0, V´, V = H[(0,)], H[(-1,)], H[(1,)]
    orbs = H.orbstruct
    block1, block2 = 1:dimh, dimh+1:2dimh
    copy!(view(A, block1, block2), I(dimh))
    _add!(view(A, block2, block1), v, orbs, -1)
    _add!(view(A, block2, block2), h0, orbs, -1)
    copy!(view(B, block1, block1), I(dimh))
    _add!(view(B, block2, block2), v´, orbs, 1)
    H0block = CartesianIndices((block2, block2))
    H0diag = [-A[i, j] for (i, j) in zip(block2, block2)]
    return SingleShot1DGreensSolver(A, B, copy(A), copy(B), maxdn, orbs, H0block, H0diag)
end


function (gs::SingleShot1DGreensSolver{T})(ω) where {T}
    A = copy!(gs.Acopy, gs.A)
    B = copy!(gs.Bcopy, gs.B)
    iη = im * sqrt(eps(real(T)))
    for (row, col, h0) in zip(gs.rngrow, gs.rngcol, gs.H0diag)
        A[row, col] = ω + iη - h0
    end
    λs, χs = eigen(A, B; sortby = abs)  # not eigen! because we need the ω-H0 block later
    return select_retarded(λs, χs, gs)
end

function select_retarded(λs, χs, gs)
    dimH = length(gs.H0diag)
    ret  = 1:length(λs)÷2
    adv  = length(λs)÷2 + 1:length(λs)
    λR   = view(λs, ret)
    λA   = view(λs, adv)
    φR   = view(χs, 1:dimh, ret)
    φA   = view(χs, 1:dimh, adv)
    φλR  = view(χs, dimh+1:2dimh, ret)
    @show φR * Diagonal(λR) * inv(φR)
    @show φA * Diagonal(inv.(λA)) * inv(φA)
    iG0  = view(gs.Acopy, gs.H0block)
    tmp  = view(gs.Bcopy, 1:dimh, ret)
    return λR, φR, φλR, iG0, tmp
end

function greensolver(::SingleShot1D, h)
    latdim(h) == 1 || throw(ArgumentError("Cannot use a SingleShot1D Green function solver with an $L-dimensional Hamiltonian"))
    return SingleShot1DGreensSolver(h)
end

# SingleShot1DGreensSolver provides the pieces to compute `GV = φR λR φR⁻¹` and from there `G = G_00 = (ω0 - H0 - V´GV)⁻¹` within the first `unit supercell` of a semi-infinite chain
# In an infinite chain we have instead `G_NN = (ω0 - H0 - V'GV - VGV')⁻¹`, for any N, where `VGV'= (V'G'V)'`. Here `GV` and `G'V` involve the retarded and advanced G sectors, respectively
# The retarded/advanced sectors are classified by the eigenmode velocity (positive, negative) if they are propagating, or abs(λ) (< 0, > 0) if they are evanescent
# The velocity is given by vᵢ = im * φᵢ'(V'λᵢ-Vλᵢ')φᵢ / φᵢ'φᵢ
# Unit supercells different from the first. Semi-infinite G_N0 = G_{N-1,0}VG = (GV)^N G.
# Each unit is an maxdn × maxdn block matrix of the actual `g` we want
function greens!(matrix, g::GreensFunction{<:SingleShot1DGreensSolver,1}, ω, (src, dst)::SVectorPair{1})
    λR, φR, φλR, iG0, tmp = g.solver(ω)
    N = mod(abs(first(dst - src)), g.maxdn)
    if !iszero(N)
        λR .^= N
        φλR = rmul!(φλR, Diagonal(λR))
    end
    iG0φR = mul!(tmp, iG0, φR)
    G = (iG0φR' \ φλR')'
    copy!(matrix, G)
    return matrix
end

#######################################################################
# BandGreensSolver
#######################################################################
struct SimplexData{D,E,T,C<:SMatrix,DD,SA<:SubArray}
    ε0::T
    εmin::T
    εmax::T
    k0::SVector{D,T}
    Δks::SMatrix{D,D,T,DD}     # k - k0 = Δks * z
    volume::T
    zvelocity::SVector{D,T}
    edgecoeffs::NTuple{E,Tuple{T,C}} # s*det(Λ)/w.w and Λc for each edge
    dωzs::NTuple{E,NTuple{2,SVector{D,T}}}
    defaultdη::SVector{D,T}
    φ0::SA
    φs::NTuple{D,SA}
end

struct BandGreensSolver{P<:SimplexData,E,H<:Hamiltonian} <: AbstractGreensSolver
    simplexdata::Vector{P}
    indsedges::NTuple{E,Tuple{Int,Int}} # all distinct pairs of 1:V, where V=D+1=num verts
    h::H
end

function Base.show(io::IO, g::GreensFunction{<:BandGreensSolver})
    print(io, summary(g), "\n",
"  Matrix size    : $(size(g.solver.h, 1)) × $(size(g.h, 2))
  Element type   : $(displayelements(g.solver.h))
  Band simplices : $(length(g.solver.simplexdata))")
end

Base.summary(g::GreensFunction{<:BandGreensSolver}) =
    "GreensFunction{Bandstructure}: Green's function from a $(latdim(g.solver.h))D bandstructure"

function greensolver(b::Bandstructure{D}, h) where {D}
    indsedges = tuplepairs(Val(D))
    v = [SimplexData(simplex, band, indsedges) for band in bands(b) for simplex in band.simplices]
    return BandGreensSolver(v,  indsedges, h)
end

edges_per_simplex(L) = binomial(L,2)

function SimplexData(simplex::NTuple{V}, band, indsedges) where {V}
    D = V - 1
    vs = ntuple(i -> vertices(band)[simplex[i]], Val(V))
    ks = ntuple(i -> SVector(Base.front(Tuple(vs[i]))), Val(V))
    εs = ntuple(i -> last(vs[i]), Val(V))
    εmin, εmax = extrema(εs)
    ε0 = first(εs)
    k0 = first(ks)
    Δks = hcat(tuple_minus_first(ks)...)
    zvelocity = SVector(tuple_minus_first(εs))
    volume = abs(det(Δks))
    edgecoeffs = edgecoeff.(indsedges, Ref(zvelocity))
    dωzs = sectionpoint.(indsedges, Ref(zvelocity))
    defaultdη = dummydη(zvelocity)
    φ0 = vertexstate(first(simplex), band)
    φs = vertexstate.(Base.tail(simplex), Ref(band))
    return SimplexData(ε0, εmin, εmax, k0, Δks, volume, zvelocity, edgecoeffs, dωzs, defaultdη, φ0, φs)
end

function edgecoeff(indsedge, zvelocity::SVector{D}) where {D}
    basis = edgebasis(indsedge, Val(D))
    othervecs = Base.tail(basis)
    edgevec = first(basis)
    cutvecs = (v -> dot(zvelocity, edgevec) * v - dot(zvelocity, v) * edgevec).(othervecs)
    Λc = hcat(cutvecs...)
    Λ = hcat(zvelocity, Λc)
    s = sign(det(hcat(basis...)))
    coeff = s * (det(Λ)/dot(zvelocity, zvelocity))
    return coeff, Λc
end

function edgebasis(indsedge, ::Val{D}) where {D}
    inds = ntuple(identity, Val(D+1))
    swappedinds = tupleswapfront(inds, indsedge) # places the two edge vertindices first
    zverts = (i->unitvector(SVector{D,Int}, i-1)).(swappedinds)
    basis = (z -> z - first(zverts)).(Base.tail(zverts)) # first of basis is edge vector
    return basis
end

function sectionpoint((i, j), zvelocity::SVector{D,T}) where {D,T}
    z0, z1 = unitvector(SVector{D,Int}, i-1), unitvector(SVector{D,Int}, j-1)
    z10 = z1 - z0
    # avoid numerical cancellation errors due to zvelocity perpendicular to edge
    d = chop(dot(zvelocity, z10), maximum(abs.(zvelocity)))
    dzdω = z10 / d
    dz0 = z0 - z10 * dot(zvelocity, z0) / d
    return dzdω, dz0   # The section z is dω * dzdω + dz0
end

# A vector, not parallel to zvelocity, and with all nonzero components and none equal
function dummydη(zvelocity::SVector{D,T}) where {D,T}
    (D == 1 || iszero(zvelocity)) && return SVector(ntuple(i->T(i), Val(D)))
    rng = MersenneTwister(0)
    while true
        dη = rand(rng, SVector{D,T})
        isparallel = dot(dη, zvelocity)^2 ≈ dot(zvelocity, zvelocity) * dot(dη, dη)
        isvalid = allunique(dη) && !isparallel
        isvalid && return dη
    end
    throw(error("Unexpected error finding dummy dη"))
end

function vertexstate(ind, band)
    ϕind = 1 + band.dimstates*(ind - 1)
    state = view(band.states, ϕind:(ϕind+band.dimstates-1))
    return state
end

## Call API

function greens!(matrix, g::GreensFunction{<:BandGreensSolver,L}, ω::Number, (src, dst)::SVectorPair{L}) where {L}
    fill!(matrix, zero(eltype(matrix)))
    dn = dst - src
    for simplexdata in g.solver.simplexdata
        g0, gjs = green_simplex(ω, dn, simplexdata, g.solver.indsedges)
        addsimplex!(matrix, g0, gjs, simplexdata)
    end
    return matrix
end

function green_simplex(ω, dn, data::SimplexData{L}, indsedges) where {L}
    dη = data.Δks' * dn
    phase = cis(dot(dn, data.k0))
    dω = ω - data.ε0
    gz = simplexterm.(dω, Ref(dη), Ref(data), data.edgecoeffs, data.dωzs, indsedges)
    g0z, gjz = first.(gz), last.(gz)
    g0 = im^(L-1) * phase * sum(g0z)
    gj = -im^L * phase * sum(gjz)
    return g0, gj
end

function simplexterm(dω, dη::SVector{D,T}, data, coeffs, (dzdω, dz0), (i, j)) where {D,T}
    bailout = Complex(zero(T)), Complex.(zero(dη))
    z = dω * dzdω + dz0
    # Edges with divergent sections do not contribute
    all(isfinite, z) || return bailout
    z0 = unitvector(SVector{D,T},i-1)
    z1 = unitvector(SVector{D,T},j-1)
    coeff, Λc = coeffs
    # If dη is zero (DOS) use a precomputed (non-problematic) simplex-constant vector
    dη´ = iszero(dη) ? data.defaultdη : dη
    d = dot(dη´, z)
    d0 = dot(dη´, z0)
    d1 = dot(dη´, z1)
    # Skip if singularity in formula
    (d ≈ d0 || d ≈ d1) && return bailout
    s = sign(dot(dη´, dzdω))
    coeff0 = coeff / prod(Λc' * dη´)
    coeffj = isempty(Λc) ? zero(dη) : (Λc ./ ((dη´)' * Λc)) * sumvec(Λc)
    params = s, d, d0, d1
    zs = z, z0, z1
    g0z = iszero(dη) ? g0z_asymptotic(D, coeff0, params) : g0z_general(coeff0, params)
    gjz = iszero(dη) ? gjz_asymptotic(D, g0z, coeffj, coeff0, zs, params) :
                       gjz_general(g0z, coeffj, coeff0, zs, params)
    return g0z, gjz
end

sumvec(::SMatrix{N,M,T}) where {N,M,T} = SVector(ntuple(_->one(T),Val(M)))

g0z_general(coeff0, (s, d, d0, d1)) =
    coeff0 * cis(d) * ((cosint_c(-s*(d0-d)) + im*sinint(d0-d)) - (cosint_c(-s*(d1-d)) + im*sinint(d1-d)))

gjz_general(g0z, coeffj, coeff0, (z, z0, z1), (s, d, d0, d1)) =
    g0z * (im * z - coeffj) + coeff0 * ((z0-z) * cis(d0) / (d0-d) - (z1-z) * cis(d1) / (d1-d))

g0z_asymptotic(D, coeff0, (s, d, d0, d1)) =
    coeff0 * (cosint_a(-s*(d0-d)) - cosint_a(-s*(d1-d))) * (im*d)^(D-1)/factorial(D-1)

function gjz_asymptotic(D, g0z, coeffj, coeff0, (z, z0, z1), (s, d, d0, d1))
    g0z´ = g0z
    for n in 1:(D-1)
        g0z´ += coeff0 * im^n * (im*d)^(D-1-n)/factorial(D-1-n) *
                ((d0-d)^n - (d1-d)^n)/(n*factorial(n))
    end
    gjz = g0z´ * (im * z - im * coeffj * d / D) +
        coeff0 * ((z0-z) * (im*d0)^D / (d0-d) - (z1-z) * (im*d1)^D / (d1-d)) / factorial(D)
    return gjz
end

cosint_c(x::Real) = ifelse(iszero(abs(x)), zero(x), cosint(abs(x))) + im*pi*(x<=0)

cosint_a(x::Real) = ifelse(iszero(abs(x)), zero(x), log(abs(x))) + im*pi*(x<=0)

function addsimplex!(matrix, g0, gjs, simplexdata)
    φ0 = simplexdata.φ0
    φs = simplexdata.φs
    vol = simplexdata.volume
    for c in CartesianIndices(matrix)
        (row, col) = Tuple(c)
        x = g0 * (φ0[row] * φ0[col]')
        for (φ, gj) in zip(φs, gjs)
            x += (φ[row]*φ[col]' - φ0[row]*φ0[col]') * gj
        end
        matrix[row, col] += vol * x
    end
    return matrix
end