############################################################################################
# selector constructors
#region

siteselector(; region = missing, sublats = missing, indices = missing) =
    SiteSelector(region, sublats, indices)
siteselector(s::SiteSelector; region = s.region, sublats = s.sublats, indices = s.indices) =
    SiteSelector(region, sublats, indices)
siteselector(lat::Lattice; kw...) = resolve(siteselector(; kw...), lat)

hopselector(; region = missing, sublats = missing, cells = missing, range = missing, indices = missing) =
    HopSelector(region, sublats, cells, range, indices)
hopselector(s::HopSelector; region = s.region, sublats = s.sublats, cells = s.cells, range = s.range, indices = s.indices) =
    HopSelector(region, sublats, cells, range, indices)
hopselector(lat::Lattice; kw...) = resolve(hopselector(; kw...), lat)

not(i) = Not(i)
not(i...) = Not(i)

#endregion

############################################################################################
# resolve
#region

function resolve(s::SiteSelector, lat::Lattice)
    s = SiteSelector(s.region, resolve_sublats(s.sublats, lat), s.indices)
    return ResolvedSelector(s, lat)
end

resolve_sublats(::Missing, lat) = missing # must be resolved to iterate over sublats
resolve_sublats(n::Not, lat) = Not(resolve_sublats(n.i, lat))
resolve_sublats(s, lat) = resolve_sublat_name.(s, Ref(lat))

function resolve_sublat_name(name::Union{Symbol,Integer}, lat)
    i = findfirst(isequal(name), lat.unitcell.names)
    return i === nothing ? 0 : i
end

resolve_sublat_name(s, lat) =
    throw(ErrorException( "Unexpected format $s for `sublats`, see `onsite` for supported options"))

#endregion

#######################################################################
# NeighborRange
#region

struct NeighborRange
    n::Int
end

"""
    nrange(n::Int)

Create a `NeighborRange` that represents a hopping range to distances corresponding to the
n-th nearest neighbors in a given lattice. Such distance is obtained by finding the n-th
closest pairs of sites in a lattice, irrespective of their sublattice.

    nrange(n::Int, lat::Lattice)

Obtain the actual nth-nearest-neighbot distance between sites in lattice `lat`.

# See also
    `hopping`
"""
nrange(n::Int) = NeighborRange(n)

function nrange(n, lat::Lattice{E,L}) where {E,L}
    sites = allsitepositions(lat)
    T = eltype(first(sites))
    dns = BoxIterator(zero(SVector{L,Int}))
    br = bravais(lat)
    # 640 is a heuristic cutoff for kdtree vs brute-force search
    if length(sites) <= 128
        dists = fill(T(Inf), n)
        for dn in dns
            iszero(dn) || ispositive(dn) || continue
            for (i, ri) in enumerate(sites), (j, rj) in enumerate(sites)
                j <= i && iszero(dn) && continue
                r = ri - rj + br * dn
                _update_dists!(dists, r'r)
            end
            isfinite(last(dists)) || acceptcell!(dns, dn)
        end
        dist = sqrt(last(dists))
    else
        tree = KDTree(sites)
        dist = T(Inf)
        for dn in dns
            iszero(dn) || ispositive(dn) || continue
            for r0 in sites
                r = r0 + br * dn
                dist = min(dist, _nrange(n, tree, r, nsites(lat)))
            end
            isfinite(dist) || acceptcell!(dns, dn)
        end
    end
    return dist
end

function _update_dists!(dists, dist::Real)
    len = length(dists)
    for (n, d) in enumerate(dists)
        isapprox(dist, d) && break
        if dist < d
            dists[n+1:len] .= dists[n:len-1]
            dists[n] = dist
            break
        end
    end
    return dists
end

function _nrange(n, tree, r::AbstractVector{T}, nmax) where {T}
    for m in n:nmax
        _, dists = knn(tree, r, 1 + m, true)
        popfirst!(dists)
        unique_sorted_approx!(dists)
        length(dists) == n && return maximum(dists)
    end
    return T(Inf)
end

#endregion

############################################################################################
# Base.in
#region

using Quantica.RegionPresets: Region

# are sites at (i,j) and (dni, dnj) or (dn, 0) selected?
@inline function Base.in(i::Integer, rs::ResolvedSelector{<:SiteSelector, LA}) where {E,L,LA<:Lattice{<:Any,E,L}}
    dn0 = zero(SVector{L,Int})
    return ((i, i), (dn0, dn0)) in rs
end

@inline Base.in(((i, j), (dni, dnj))::Tuple{Tuple,Tuple}, rs::ResolvedSelector{<:SiteSelector}) =
    isonsite((i, j), (dni, dnj)) && (i, dni) in rs

Base.in((i, dni)::Tuple{Integer,SVector}, rs::ResolvedSelector{<:SiteSelector}) =
    isinindices(i, rs.selector.indices) &&
    isinregion(i, dni, rs.selector.region, rs.lattice) &&
    isinsublats(sitesublat(i, rs.lattice), rs.selector.sublats)

isonsite((i, j), (dni, dnj)) = i == j && dni == dnj

isinregion(i::Int, ::Missing, lat) = true
isinregion(i::Int, dn, ::Missing, lat) = true
isinregion(i::Int, region::Union{Function,Region}, lat) =
    region(sites(lat)[i])
isinregion(i::Int, dn, region::Union{Function,Region}, lat) =
    region(sites(lat)[i] + bravais_matrix(lat) * dn)
isinregion(is::Tuple{Int,Int}, dns, ::Missing, lat) = true

function isinregion((row, col)::Tuple{Int,Int}, (dnrow, dncol), region::Union{Function,Region}, lat)
    br = bravais(lat)
    r, dr = _rdr(sites(lat)[col] + br * dncol, sites(lat)[row] + br * dnrow)
    return region(r, dr)
end

_rdr(r1, r2) = (0.5 * (r1 + r2), r2 - r1)

# Sublats are resolved, so they are equivalent to indices
isinsublats(i, j) = isinindices(i,j)

# Here we can have (1, 2:3)
isinindices(i::Integer, n::Not) = !isinindices(i, parent(n))
isinindices(i::Integer, ::Missing) = true
isinindices(i::Integer, j::Integer) = i == j
isinindices(i::Integer, r::NTuple{N,Integer}) where {N} = i in r
isinindices(i::Integer, inds::Tuple) = any(is -> isinindices(i, is), inds)
isinindices(i::Integer, r) = i in r
# Here we cover ((1,2) .=> (3,4), 1=>2) and ((1,2) => (3,4), 1=>2)
isinindices(is::Pair, n::Not) = !isinindices(is, parent(n))
isinindices(is::Pair, ::Missing) = true
# Here is => js could be (1,2) => (3,4) or 1:2 => 3:4, not simply 1 => 3
isinindices((j, i)::Pair, (js, is)::Pair) = isinindices(i, is) && isinindices(j, js)
# Here we support ((1,2) .=> (3,4), 3=>4) or ((1,2) .=> 3:4, 3=>4)
isinindices(pair::Pair, pairs) = any(p -> isinindices(pair, p), pairs)


#endregion