#######################################################################
# Green's function
#######################################################################
abstract type GreenSolver end

struct GreensFunction{S<:GreenSolver,H}
    h::H
    solver::S
end

"""
    greens(h::Hamiltonian, solveobject)

Construct the Green's function `g::GreensFunction` of `h` using the provided `solveobject`.
Currently valid `solveobject`s are

- the `Bandstructure` of `h` (for an unbounded `h` or an `Hamiltonian{<:Superlattice}}`)
- the `Spectrum` of `h` (for a bounded `h`)

    h |> greens(h -> solveobject(h))

Curried form equivalent to the above, giving `greens(h, solveobject(h))` (see
example below).

    g([m,] ω, cells::Pair = missing)

From a constructed `g::GreensFunction`, obtain the retarded Green's function
matrix at frequency `ω` between unit cells `src` and `dst` by calling `g(ω, src
=> dst)`, where `src, dst` are `::NTuple{L,Int}` or `SVector{L,Int}`. If
`cells` is missing, `src` and `dst` are assumed to be zero vectors. For
performance, one can use a preallocated matrix `m` (e.g. `m =
similarmatrix(h)`) by calling `g(m, ω, cells)`.

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

"""
greens(h, solver) = GreensFunction(h, greensolver(solver))
greens(solver::Function) = h -> greens(h, solver(h))

# Needed to make similarmatrix work with GreensFunction
matrixtype(g::GreensFunction) = Matrix{eltype(g.h)}
Base.parent(g::GreensFunction) = g.h
optimize!(g::GreensFunction) = g

#######################################################################
# BandGreenSolver
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
    φ0::SA
    φs::NTuple{D,SA}
end

struct BandGreenSolver{P<:SimplexData,E} <: GreenSolver
    simplexdata::Vector{P}
    indsedges::NTuple{E,Tuple{Int,Int}} # all distinct pairs of 1:V, where V=D+1=num verts
end

function Base.show(io::IO, g::GreensFunction{<:BandGreenSolver})
    print(io, summary(g), "\n",
"  Matrix size    : $(size(g.h, 1)) × $(size(g.h, 2))
  Element type   : $(displayelements(g.h))
  Band simplices : $(length(g.solver.simplexdata))")
end

Base.summary(g::GreensFunction{<:BandGreenSolver}) =
    "GreensFunction{Bandstructure}: Green's function from a $(latdim(g.h))D bandstructure"

function greensolver(b::Bandstructure{D}) where {D}
    V = D + 1
    indsedges = tuplepairs(Val(D+1)) # not inferred for D>2
    v = [SimplexData(simplex, band, indsedges) for band in bands(b) for simplex in band.simplices]
    return BandGreenSolver(v,  indsedges)
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
    Δks = hcat(tuple_diff_first(ks)...)
    zvelocity = SVector(tuple_diff_first(εs))
    volume = abs(det(Δks))
    edgecoeffs = edgecoeff.(indsedges, Ref(zvelocity))
    dωzs = sectionpoint.(indsedges, Ref(zvelocity))
    φ0 = vertexstate(first(simplex), band)
    φs = vertexstate.(Base.tail(simplex), Ref(band))
    return SimplexData(ε0, εmin, εmax, k0, Δks, volume, zvelocity, edgecoeffs, dωzs, φ0, φs)
end

# Base.tail(t) .- first(t) but avoiding rounding errors in difference
tuple_diff_first(t::Tuple{T,Vararg{T,D}}) where {D,T} =
    ntuple(i -> ifelse(t[i+1] ≈ t[1], zero(T), t[i+1] - t[1]), Val(D))

function edgecoeff(indsedge, zvelocity::SVector{D}) where {D}
    basis = edgebasis(indsedge, Val(D))
    othervecs = Base.tail(basis)
    edgevec = first(basis)
    cutvecs = (v -> dot(zvelocity, edgevec) * v - dot(zvelocity, v) * edgevec).(othervecs)
    Λc = hcat(cutvecs...)
    Λ = hcat(Λc, zvelocity)
    s = sign(det(hcat(othervecs..., edgevec)))
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

function vertexstate(ind, band)
    ϕind = 1 + band.dimstates*(ind - 1)
    state = view(band.states, ϕind:(ϕind+band.dimstates-1))
    return state
end

## Call API

(g::GreensFunction{<:BandGreenSolver})(ω::Number, cells = missing) = g(similarmatrix(g), ω, cells)

function (g::GreensFunction{<:BandGreenSolver})(matrix::AbstractMatrix, ω::Number, cells = missing)
    fill!(matrix, zero(eltype(matrix)))
    cells´ = sanitize_dn(cells, g.h)
    for simplexdata in g.solver.simplexdata
        g0, gjs = green_simplex(ω, cells´, simplexdata, g.solver.indsedges)
        addsimplex!(matrix, g0, gjs, simplexdata)
    end
    return matrix
end

sanitize_dn((src, dst)::Pair, ::Hamiltonian{LA,L}) where {LA,L} =
    SVector{L}(dst) - SVector{L}(src)

sanitize_dn(::Missing, ::Hamiltonian{LA,L}) where {LA,L} =
    zero(SVector{L,Int})

function green_simplex(ω, dn, data::SimplexData{L}, indsedges) where {L}
    dη = data.Δks' * dn
    phase = cis(dot(dn, data.k0))
    dω = ω - data.ε0
    gz = simplexterm.(dω, Ref(dη), data.edgecoeffs, data.dωzs, indsedges)
    g0z, gjz = first.(gz), last.(gz)
    return im^(L-1) * phase * sum(g0z), -im^L * phase * sum(gjz)
end

function simplexterm(dω, dη::SVector{D,T}, coeffs, (dzdω, dz0), (i, j)) where {D,T}
    z = dω * dzdω + dz0
    z0 = unitvector(SVector{D,T},i-1)
    z1 = unitvector(SVector{D,T},j-1)
    coeff, Λc = coeffs
    # If dη is zero (DOS) use any non-problematic but simplex-constant vector
    dη´ = iszero(dη) ? Λc[:,1]*sign(Λc[1,1]) : dη
    d = dot(dη´, z)
    d0 = dot(dη´, z0)
    d1 = dot(dη´, z1)
    # Edges with divergent sections do not contribute
    (!isfinite(d) || d0 ≈ d || d1 ≈ d) && return Complex(zero(T)), Complex.(zero(dη))
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

cosint_c(x::Real) = cosint(abs(x)) + im*pi*(x<=0)

cosint_a(x::Real) = log(abs(x)) + im*pi*(x<=0)

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