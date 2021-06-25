using Quantica.RegionPresets: Region

#######################################################################
# NeighborRange
#######################################################################
struct NeighborRange
    n::Int
end

"""
    nrange(n::Int)

Create a `NeighborRange` that represents a hopping range to distances corresponding to the
n-th nearest neighbors in a given lattice. Such distance is obtained by finding the n-th
closest pairs of sites in a lattice, irrespective of their sublattice.

    nrange(n::Int, lat::AbstractLattice)

Obtain the actual nth-nearest-neighbot distance between sites in lattice `lat`.

# See also
    `hopping`
"""
nrange(n::Int) = NeighborRange(n)

function nrange(n, lat::AbstractLattice{E,L}) where {E,L}
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


#######################################################################
# Onsite/Hopping selectors
#######################################################################
abstract type Selector end

struct SiteSelector{S,I,M} <: Selector
    region::M
    sublats::S  # NTuple{N,NameType} (unresolved) or Vector{Int} (resolved on a lattice)
    indices::I  # Once resolved, this should be an Union{Integer,Not} container
end

struct HopSelector{S,I,D,T,M} <: Selector
    region::M
    sublats::S  # NTuple{N,Pair{NameType,NameType}} (unres) or Vector{Pair{Int,Int}} (res)
    dns::D
    range::T
    indices::I # Once resolved, this should be a Pair{Integer,Integer} container
end

struct ResolvedSelector{S<:Selector,L<:AbstractLattice} <: Selector
    selector::S
    lattice::L
end

struct Not{T} # Symbolizes an excluded elements
    i::T
end

"""
    not(i)

Wrapper indicating the negation or complement of `i`, typically used to encode excluded site
indices. See `siteselector` and `hopselector` for applications.

"""
not(i) = Not(i)
not(i...) = Not(i)

"""
    siteselector(; region = missing, sublats = missing, indices = missing)

Return a `SiteSelector` object that can be used to select sites in a lattice contained
within the specified region and sublattices. Only sites with index `i`, at position `r` and
belonging to a sublattice with name `s::NameType` will be selected if

    `region(r) && s in sublats && i in indices`

Any missing `region`, `sublat` or `indices` will not be used to constraint the selection.

The keyword `sublats` allows the following formats:

    sublats = :A                    # Sites on sublat :A only
    sublats = (:A,)                 # Same as above
    sublats = (:A, :B)              # Sites on sublat :A and :B

The keyword `indices` accepts a single integer, or a collection thereof. If several
collections are given, they are flattened into a single one. Possible combinations:

    indices = 1                     # Site 1 only
    indices = (1, )                 # Same as above
    indices = (1, 2, 3)             # Sites 1, 2 or 3
    indices = [1, 2, 3]             # Same as above
    indices = 1:3                   # Same as above
    indices = (1:3, 7, 8)           # Sites 1, 2, 3, 7 or 8

Additionally, indices or sublattices can be wrapped in `not` to exclude them (see `not`):

    sublats = not(:A)               # Any sublat different from :A
    sublats = not(:A, :B)           # Any sublat different from :A and :B
    indices = not(8)                # Any site index different from 8
    indices = not(1, 3:4)           # Any site index different from 1, 3 or 4
    indices = (not(3:4), 4:6)       # Any site different from 3 and 4, *or* equal to 4, 5 or 6
"""
siteselector(; region = missing, sublats = missing, indices = missing) =
    SiteSelector(region, sublats, indices)
siteselector(s::SiteSelector; region = s.region, sublats = s.sublats, indices = s.indices) =
    SiteSelector(region, sublats, indices)

"""
    hopselector(; range = missing, dn = missing, sublats = missing, indices = missing, region = missing)

Return a `HopSelector` object that can be used to select hops between two sites in a
lattice. Only hops between two sites, with indices `ipair = src => dst`, at positions `r₁ =
r - dr/2` and `r₂ = r + dr`, belonging to unit cells at integer distance `dn´` and to
sublattices `s₁` and `s₂` will be selected if:

    `region(r, dr) && s in sublats && dn´ in dn && norm(dr) <= range && ipair in indices`

If any of these is `missing` it will not be used to constraint the selection.

The keyword `range` admits the following possibilities

    max_range                   # i.e. `norm(dr) <= max_range`
    (min_range, max_range)      # i.e. `min_range <= norm(dr) <= max_range`

Both `max_range` and `min_range` can be a `Real` or a `NeighborRange` created with
`nrange(n)`. The latter represents the distance of `n`-th nearest neighbors.

The keyword `dn` can be a `Tuple`/`Vector`/`SVector` of `Int`s, or a tuple thereof.

The keyword `sublats` allows the following formats:

    sublats = :A => :B                  # Hopping from :A to :B sublattices, but not from :B to :A
    sublats = (:A => :B,)               # Same as above
    sublats = (:A => :B, :C => :D)      # Hopping from :A to :B or :C to :D
    sublats = (:A, :C) .=> (:B, :D)     # Broadcasted pairs, same as above
    sublats = (:A, :C) => (:B, :D)      # Direct product, (:A=>:B, :A=:D, :C=>:B, :C=>D)

The keyword `indices` accepts a single `src => dest` pair or a collection thereof. Any `src
== dest` will be neglected. Possible combinations:

    indices = 1 => 2                    # Hopping from site 1 to 2, but not from 2 to 1
    indices = (1 => 2, 2 => 1)          # Hoppings from 1 to 2 or from 2 to 1
    indices = [1 => 2, 2 => 1]          # Same as above
    indices = [(1, 2) .=> (2, 1)]       # Broadcasted pairs, same as above
    indices = [1:10 => 20:25, 3 => 30]  # Direct product, any hopping from sites 1:10 to sites 20:25, or from 3 to 30

Additionally, indices or sublattices can be wrapped in `not` to exclude them (see `not`):

    sublats = not(:A => :B, :B => :A)   # Any sublat pairs different from :A => :B or :B => :A
    sublats = not(:A) => :B             # Any sublat pair s1 => s2 with s1 different from :A and s2 equal to :B
    indices = not(8 => 9)               # Any site indices different from 8 => 9
    indices = 1 => not(3:4)             # Any site pair 1 => s with s different from 3, 4

"""
hopselector(; region = missing, sublats = missing, dn = missing, range = missing, indices = missing) =
    HopSelector(region, sublats, sanitize_dn(dn), sanitize_range(range), indices)
hopselector(s::HopSelector; region = s.region, sublats = s.sublats, dn = s.dns, range = s.range, indices = s.indices) =
    HopSelector(region, sublats, sanitize_dn(dn), sanitize_range(range), indices)

ensurenametype((s1, s2)::Pair) = nametype(s1) => nametype(s2)

sanitize_dn(dn::Missing) = missing
sanitize_dn(dn::Tuple{Vararg{Number,N}}) where {N} = (_sanitize_dn(dn),)
sanitize_dn(dn) = (_sanitize_dn(dn),)
sanitize_dn(dn::Tuple) = _sanitize_dn.(dn)
_sanitize_dn(dn::Tuple{Vararg{Number,N}}) where {N} = SVector{N,Int}(dn)
_sanitize_dn(dn::SVector{N}) where {N} = SVector{N,Int}(dn)
_sanitize_dn(dn::Vector) = SVector{length(dn),Int}(dn)

sanitize_range(::Missing) = missing
sanitize_range(r) = _shift_eps(r, 1)
sanitize_range(r::NTuple{2,Any}) = (_shift_eps(first(r), -1), _shift_eps(last(r), 1))

_shift_eps(r::Real, m) = ifelse(isfinite(r), float(r) + m * sqrt(eps(float(r))), float(r))
_shift_eps(r, m) = r

# API

function resolve(s::SiteSelector, lat::AbstractLattice)
    s = SiteSelector(s.region, resolve_sublats(s.sublats, lat), s.indices)
    return ResolvedSelector(s, lat)
end

function resolve(s::HopSelector, lat::AbstractLattice)
    s = HopSelector(s.region, resolve_sublat_pairs(s.sublats, lat), check_dn_dims(s.dns, lat), resolve_range(s.range, lat), s.indices)
    return ResolvedSelector(s, lat)
end

resolve_sublats(::Missing, lat) = missing # must be resolved to iterate over sublats
resolve_sublats(n::Not, lat) = Not(resolve_sublats(n.i, lat))
resolve_sublats(s, lat) = resolve_sublat_name.(s, Ref(lat))

resolve_range(r::Tuple, lat) = sanitize_range(_resolve_range.(r, Ref(lat)))
resolve_range(r, lat) = sanitize_range(_resolve_range(r, lat))
_resolve_range(r::NeighborRange, lat) = nrange(r.n, lat)
_resolve_range(r, lat) = r

function resolve_sublat_name(name::Union{NameType,Integer}, lat)
    i = findfirst(isequal(name), lat.unitcell.names)
    return i === nothing ? 0 : i
end

resolve_sublat_name(s, lat) =
    throw(ErrorException( "Unexpected format $s for `sublats`, see `onsite` for supported options"))

resolve_sublat_pairs(::Missing, lat) = missing
resolve_sublat_pairs(n::Not, lat) = Not(resolve_sublat_pairs(n.i, lat))
resolve_sublat_pairs(s::Tuple, lat) = resolve_sublat_pairs.(s, Ref(lat))
resolve_sublat_pairs(s::Vector, lat) = resolve_sublat_pairs.(s, Ref(lat))
resolve_sublat_pairs((src, dst)::Pair, lat) = _resolve_sublat_pairs(src, lat) => _resolve_sublat_pairs(dst, lat)
_resolve_sublat_pairs(n::Not, lat) = Not(_resolve_sublat_pairs(n.i, lat))
_resolve_sublat_pairs(p, lat) = resolve_sublat_name.(p, Ref(lat))

resolve_sublat_pairs(s, lat) =
    throw(ErrorException( "Unexpected format $s for `sublats`, see `hopping` for supported options"))

check_dn_dims(dns::Missing, lat::AbstractLattice{E,L}) where {E,L} = dns
check_dn_dims(dns::Tuple{Vararg{SVector{L,Int}}}, lat::AbstractLattice{E,L}) where {E,L} = dns
check_dn_dims(dns, lat::AbstractLattice{E,L}) where {E,L} =
    throw(DimensionMismatch("Specified cell distance `dn` does not match lattice dimension $L"))

# are sites at (i,j) and (dni, dnj) or (dn, 0) selected?
@inline function Base.in(i::Integer, rs::ResolvedSelector{<:SiteSelector, LA}) where {E,L,LA<:AbstractLattice{E,L}}
    dn0 = zero(SVector{L,Int})
    return ((i, i), (dn0, dn0)) in rs
end

@inline Base.in(((i, j), (dni, dnj))::Tuple{Tuple,Tuple}, rs::ResolvedSelector{<:SiteSelector}) =
    isonsite((i, j), (dni, dnj)) && (i, dni) in rs

Base.in((i, dni)::Tuple{Integer,SVector}, rs::ResolvedSelector{<:SiteSelector}) =
    isinindices(i, rs.selector.indices) &&
    isinregion(i, dni, rs.selector.region, rs.lattice) &&
    isinsublats(sublat_site(i, rs.lattice), rs.selector.sublats)

Base.in((j, i)::Pair{<:Integer,<:Integer}, rs::ResolvedSelector{<:HopSelector}) = (i, j) in rs

function Base.in(is::Tuple{Integer,Integer}, rs::ResolvedSelector{<:HopSelector, LA}) where {E,L,LA<:AbstractLattice{E,L}}
    dn0 = zero(SVector{L,Int})
    return (is, (dn0, dn0)) in rs
end

Base.in((inds, dns), rs::ResolvedSelector{<:HopSelector}) =
    !isonsite(inds, dns) && isinindices(indstopair(inds), rs.selector.indices) &&
    isinregion(inds, dns, rs.selector.region, rs.lattice) && isindns(dns, rs.selector.dns) &&
    isinrange(inds, dns, rs.selector.range, rs.lattice) &&
    isinsublats(indstopair(sublat_site.(inds, Ref(rs.lattice))), rs.selector.sublats)

isonsite((i, j), (dni, dnj)) = i == j && dni == dnj

isinregion(i::Int, ::Missing, lat) = true
isinregion(i::Int, dn, ::Missing, lat) = true

isinregion(i::Int, region::Union{Function,Region}, lat) =
    region(allsitepositions(lat)[i])
isinregion(i::Int, dn, region::Union{Function,Region}, lat) =
    region(allsitepositions(lat)[i] + bravais(lat) * dn)

isinregion(is::Tuple{Int,Int}, dns, ::Missing, lat) = true
function isinregion((row, col)::Tuple{Int,Int}, (dnrow, dncol), region::Union{Function,Region}, lat)
    br = bravais(lat)
    r, dr = _rdr(allsitepositions(lat)[col] + br * dncol, allsitepositions(lat)[row] + br * dnrow)
    return region(r, dr)
end

isindns((dnrow, dncol)::Tuple{SVector,SVector}, dns) = isindns(dnrow - dncol, dns)
isindns(dn::SVector{L,Int}, dns::Tuple{Vararg{SVector{L,Int}}}) where {L} = dn in dns
isindns(dn::SVector, dns::Missing) = true
isindns(dn, dns) =
    throw(ArgumentError("Cell distance dn in selector is incompatible with Lattice."))

isinrange(inds, dns, ::Missing, lat) = true
isinrange((row, col), (dnrow, dncol), range, lat) =
    _isinrange(allsitepositions(lat)[col] - allsitepositions(lat)[row] + bravais(lat) * (dncol - dnrow), range)
_isinrange(p, rmax::Real) = p'p <= rmax^2
_isinrange(p, (rmin, rmax)::Tuple{Real,Real}) =  rmin^2 <= p'p <= rmax^2

is_below_min_range(inds, dns, rsel::ResolvedSelector) =
    is_below_min_range(inds, dns, rsel.selector.range, rsel.lattice)
is_below_min_range((i, j), (dni, dnj), (rmin, rmax)::Tuple, lat) =
    norm(siteposition(i, dni, lat) - siteposition(j, dnj, lat)) < rmin
is_below_min_range(inds, dn, range, lat) = false

# Sublats are resolved, so they are equivalent to indices
isinsublats(i, j) = isinindices(i,j)

# Here we can have (1, 2:3)
isinindices(i::Integer, n::Not) = !isinindices(i, n.i)
isinindices(i::Integer, ::Missing) = true
isinindices(i::Integer, j::Integer) = i == j
isinindices(i::Integer, r::NTuple{N,Integer}) where {N} = i in r
isinindices(i::Integer, inds::Tuple) = any(is -> isinindices(i, is), inds)
isinindices(i::Integer, r) = i in r
# Here we cover ((1,2) .=> (3,4), 1=>2) and ((1,2) => (3,4), 1=>2)
isinindices(is::Pair, n::Not) = !isinindices(is, n.i)
isinindices(is::Pair, ::Missing) = true
# Here is => js could be (1,2) => (3,4) or 1:2 => 3:4, not simply 1 => 3
isinindices((j, i)::Pair, (js, is)::Pair) = isinindices(i, is) && isinindices(j, js)
# Here we support ((1,2) .=> (3,4), 3=>4) or ((1,2) .=> 3:4, 3=>4)
isinindices(pair::Pair, pairs) = any(p -> isinindices(pair, p), pairs)

# merge non-missing fields of s´ into s
merge_non_missing(s::SiteSelector, s´::SiteSelector) =
    SiteSelector(merge_non_missing.(
        (s.region,  s.sublats, s.indices),
        (s´.region, s´.sublats, s´.indices))...)
merge_non_missing(s::HopSelector, s´::HopSelector) =
    HopSelector(merge_non_missing.(
        (s.region,  s.sublats,  s.dns,  s.range, s.indices),
        (s´.region, s´.sublats, s´.dns, s´.range, s´.indices))...)
merge_non_missing(o, o´::Missing) = o
merge_non_missing(o, o´) = o´

Base.adjoint(s::SiteSelector) = s
function Base.adjoint(s::HopSelector)
    # is_unconstrained_selector(s) &&
    #     @warn("Taking the adjoint of an unconstrained hopping is likely unintended")
    region´ = _adjoint(s.region)
    sublats´ = _adjoint.(s.sublats)
    dns´ = _adjoint.(s.dns)
    range = s.range
    indices´ = _adjoint.(s.indices)
    return HopSelector(region´, sublats´, dns´, range, indices´)
end

_adjoint(::Missing) = missing
_adjoint(f::Function) = (r, dr) -> f(r, -dr)
_adjoint(t::Pair) = reverse(t)
_adjoint(t::Tuple) = _adjoint.(t)
_adjoint(t::AbstractVector) = _adjoint.(t)
_adjoint(t::SVector) = -t

# is_unconstrained_selector(s::HopSelector{Missing,Missing,Missing}) = true
# is_unconstrained_selector(s::HopSelector) = false

#######################################################################
# site, sublat, dn generators
#######################################################################

sitepositions(lat::AbstractLattice, s::SiteSelector) =
    sitepositions(resolve(s, lat))
sitepositions(rs::ResolvedSelector{<:SiteSelector}) =
    (s for (i, s) in enumerate(allsitepositions(rs.lattice)) if i in rs)

siteindices(lat::AbstractLattice, s::SiteSelector) =
    siteindices(resolve(s, lat))
siteindices(rs::ResolvedSelector{<:SiteSelector}) =
    (i for i in siteindex_candidates(rs) if i in rs)
siteindices(rs::ResolvedSelector{<:SiteSelector}, sublat::Int) =
    (i for i in siteindex_candidates(rs, sublat) if i in rs)
siteindices(rs::ResolvedSelector{<:SiteSelector}, sublat, dn) =
    (i for i in siteindex_candidates(rs, sublat) if (i, dn) in rs)
siteindices(rs::ResolvedSelector{<:SiteSelector}, dn::SVector) =
    (i for i in siteindex_candidates(rs) if (i, dn) in rs)

# Given a sublattice, which site indices should be checked by selector?
siteindex_candidates(rs) = eachindex(allsitepositions(rs.lattice))
siteindex_candidates(rs, sublat) =
    _siteindex_candidates(rs.selector.indices, siterange(rs.lattice, sublat))
# indices can be missing, 1, 2:3, (1,2,3) or (1, 2:3)
# we also support (1, (2,3)) and [1, 2, 3], useful for source_candidates below
_siteindex_candidates(::Missing, sr) = sr
# Better not exclude candidates with not, since that can lead to collecting a huge range
_siteindex_candidates(::Not, sr) = sr
_siteindex_candidates(i::Integer, sr) = ifelse(i in sr, (i,), ())
_siteindex_candidates(inds::AbstractUnitRange, sr) = intersect(inds, sr)
_siteindex_candidates(inds::NTuple{N,Integer}, sr) where {N} = filter(in(sr), inds)
_siteindex_candidates(inds::AbstractVector{<:Integer}, sr) = filter(in(sr), inds)
_siteindex_candidates(inds, sr) = Iterators.flatten(_siteindex_candidates.(inds, Ref(sr)))

source_candidates(rs::ResolvedSelector{<:HopSelector}, sublat) =
    _source_candidates(rs.selector.indices, siterange(rs.lattice, sublat))
_source_candidates(::Missing, sr) = sr
_source_candidates(::Not, sr) = sr
_source_candidates(inds, sr) = _siteindex_candidates(_recursivefirst(inds), sr)

_recursivefirst(p::Pair) = first(p)
_recursivefirst(p) = _recursivefirst.(p)

sublats(rs::ResolvedSelector{<:SiteSelector{Missing}}) = sublats(rs.lattice)

function sublats(rs::ResolvedSelector{<:SiteSelector})
    subs = sublats(rs.lattice)
    return (s for s in subs if isinsublats(s, rs.selector.sublats))
end

function sublats(rs::ResolvedSelector{<:HopSelector})
    subs = sublats(rs.lattice)
    return (s => d for s in subs for d in subs if isinsublats(s => d, rs.selector.sublats))
end

dniter(rs::ResolvedSelector{S,LA}) where {S,E,L,LA<:Lattice{E,L}} =
    dniter(rs.selector.dns, Val(L))
dniter(dns::Missing, ::Val{L}) where {L} = BoxIterator(zero(SVector{L,Int}))
dniter(dns, ::Val) = dns

#######################################################################
# Tightbinding types
#######################################################################
abstract type TightbindingModelTerm end
abstract type AbstractOnsiteTerm <: TightbindingModelTerm end
abstract type AbstractHoppingTerm <: TightbindingModelTerm end

struct TightbindingModel{N,T<:Tuple{Vararg{TightbindingModelTerm,N}}}
    terms::T
end

struct OnsiteTerm{F,S<:Union{SiteSelector, ResolvedSelector{<:SiteSelector}},C} <: AbstractOnsiteTerm
    o::F
    selector::S
    coefficient::C
end

struct HoppingTerm{F,S<:Union{HopSelector, ResolvedSelector{<:HopSelector}},C} <: AbstractHoppingTerm
    t::F
    selector::S
    coefficient::C
end

#######################################################################
# TightbindingModel API
#######################################################################
terms(t::TightbindingModel) = t.terms

TightbindingModel(ts::TightbindingModelTerm...) = TightbindingModel(ts)

# (m::TightbindingModel)(r, dr) = sum(t -> t(r, dr), m.terms)  # this does not filter by selector, so it's wrong

# External API #

Base.:*(x::Number, m::TightbindingModel) = TightbindingModel(x .* m.terms)
Base.:*(m::TightbindingModel, x::Number) = x * m
Base.:-(m::TightbindingModel) = TightbindingModel((-1) .* m.terms)

Base.:+(m::TightbindingModel, t::TightbindingModel) = TightbindingModel((m.terms..., t.terms...))
Base.:-(m::TightbindingModel, t::TightbindingModel) = m + (-t)

function Base.show(io::IO, m::TightbindingModel{N}) where {N}
    ioindent = IOContext(io, :indent => "  ")
    print(io, "TightbindingModel{$N}: model with $N terms", "\n")
    foreach(t -> print(ioindent, t, "\n"), m.terms)
end

#######################################################################
# TightbindingModelTerm API
#######################################################################
OnsiteTerm(t::OnsiteTerm, os::SiteSelector) =
    OnsiteTerm(t.o, os, t.coefficient)

(o::OnsiteTerm{<:Function})(r,dr) = o.coefficient * o.o(r)
(o::OnsiteTerm)(r,dr) = o.coefficient * o.o

HoppingTerm(t::HoppingTerm, os::HopSelector) =
    HoppingTerm(t.t, os, t.coefficient)

(h::HoppingTerm{<:Function})(r, dr) = h.coefficient * h.t(r, dr)
(h::HoppingTerm)(r, dr) = h.coefficient * h.t

sublats(t::TightbindingModelTerm, lat) = resolve_sublats(t.selector, lat)

sublats(m::TightbindingModel) = (t -> t.selector.sublats).(terms(m))

resolve(t::OnsiteTerm, lat::AbstractLattice) = OnsiteTerm(t.o, resolve(t.selector, lat), t.coefficient)
resolve(t::HoppingTerm, lat::AbstractLattice) = HoppingTerm(t.t, resolve(t.selector, lat), t.coefficient)
resolve(m::TightbindingModel, lat::AbstractLattice) = TightbindingModel(resolve.(m.terms, Ref(lat)))

displayparameter(::Type{<:Function}) = "Function"
displayparameter(::Type{T}) where {T} = "$T"

displayrange(r::Real) = round(r, digits = 6)
displayrange(::Missing) = "any"
displayrange(nr::NeighborRange) = "NeighborRange($(nr.n))"
displayrange(rs::Tuple) = "($(displayrange(first(rs))), $(displayrange(last(rs))))"

function Base.show(io::IO, o::OnsiteTerm{F,<:SiteSelector}) where {F}
    i = get(io, :indent, "")
    print(io,
"$(i)OnsiteTerm{$(displayparameter(F))}:
$(i)  Sublattices      : $(o.selector.sublats === missing ? "any" : o.selector.sublats)
$(i)  Coefficient      : $(o.coefficient)")
end

function Base.show(io::IO, h::HoppingTerm{F,<:HopSelector}) where {F}
    i = get(io, :indent, "")
    print(io,
"$(i)HoppingTerm{$(displayparameter(F))}:
$(i)  Sublattice pairs : $(h.selector.sublats === missing ? "any" : h.selector.sublats)
$(i)  dn cell distance : $(h.selector.dns === missing ? "any" : h.selector.dns)
$(i)  Hopping range    : $(displayrange(h.selector.range))
$(i)  Coefficient      : $(h.coefficient)")
end

# External API #
"""
    onsite(o; region = missing, sublats = missing)

Create an `TightbindingModel` with a single `OnsiteTerm` that applies an onsite energy `o`
to a `Lattice` when creating a `Hamiltonian` with `hamiltonian`.

The onsite energy `o` can be a number, an `SMatrix`, a `UniformScaling` (e.g. `3*I`) or a
function of the form `r -> ...` for a position-dependent onsite energy.

The dimension of `o::AbstractMatrix` must match the orbital dimension of applicable
sublattices (see also `orbitals` option for `hamiltonian`). If `o::UniformScaling` it will
be converted to an identity matrix of the appropriate size when applied to multiorbital
sublattices. Similarly, if `o::SMatrix` it will be truncated or padded to the appropriate
size.

    onsite(model::TightbindingModel; kw...)

Return a `TightbindingModel` with only the onsite terms of `model`. Any non-missing `kw` is
applied to all such terms.

# Keyword arguments

Keywords are the same as for `siteselector`. Only sites at position `r` in sublattice with
name `s::NameType` will be selected if `region(r) && s in sublats` is true. Any missing
`region` or `sublat` will not be used to constraint the selection.

The keyword `sublats` allows the following formats:

    sublats = :A           # Onsite on sublat :A only
    sublats = (:A,)        # Same as above
    sublats = (:A, :B)     # Onsite on sublat :A and :B

# Combining models

`OnsiteTerm`s and `HoppingTerm`s created with `onsite` or `hopping` can added or substracted
together or be multiplied by scalars to build more complicated `TightbindingModel`s, e.g.
`onsite(1) - 3 * hopping(2)`

# Examples

```jldoctest
julia> model = onsite(1, sublats = (:A,:B)) - 2 * hopping(2, sublats = :A=>:A)
TightbindingModel{2}: model with 2 terms
  OnsiteTerm{Int64}:
    Sublattices      : (:A, :B)
    Coefficient      : 1
  HoppingTerm{Int64}:
    Sublattice pairs : :A => :A
    dn cell distance : any
    Hopping range    : NeighborRange(1)
    Coefficient      : -2

julia> newmodel = onsite(model; sublats = :A) + hopping(model)
TightbindingModel{2}: model with 2 terms
  OnsiteTerm{Int64}:
    Sublattices      : A
    Coefficient      : 1
  HoppingTerm{Int64}:
    Sublattice pairs : :A => :A
    dn cell distance : any
    Hopping range    : NeighborRange(1)
    Coefficient      : -2

julia> LatticePresets.honeycomb() |> hamiltonian(onsite(r -> @SMatrix[1 2; 3 4]), orbitals = Val(2))
Hamiltonian{<:Lattice} : Hamiltonian on a 2D Lattice in 2D space
  Bloch harmonics  : 1 (SparseMatrixCSC, sparse)
  Harmonic size    : 2 × 2
  Orbitals         : ((:a, :a), (:a, :a))
  Element type     : 2 × 2 blocks (ComplexF64)
  Onsites          : 2
  Hoppings         : 0
  Coordination     : 0.0
```

# See also
    `hopping`
"""
onsite(o; kw...) = onsite(o, siteselector(; kw...))

onsite(o, selector::Selector) = TightbindingModel(OnsiteTerm(o, selector, 1))

onsite(m::TightbindingModel, selector::Selector) =
    TightbindingModel(_onlyonsites(selector, m.terms...))

_onlyonsites(s, t::OnsiteTerm, args...) =
    (OnsiteTerm(t, merge_non_missing(t.selector, s)), _onlyonsites(s, args...)...)
_onlyonsites(s, t::HoppingTerm, args...) = (_onlyonsites(s, args...)...,)
_onlyonsites(s) = ()

"""
    hopping(t; range = nrange(1), dn = missing, sublats = missing, indices = missing, region = missing, plusadjoint = false)

Create an `TightbindingModel` with a single `HoppingTerm` that applies a hopping `t` to a
`Lattice` when creating a `Hamiltonian` with `hamiltonian`.

The hopping amplitude `t` can be a number, an `SMatrix`, a `UniformScaling` (e.g. `3*I`) or
a function of the form `(r, dr) -> ...` for a position-dependent hopping (`r` is the bond
center, and `dr` the bond vector). If `sublats` is specified as a sublattice name pair, or
tuple thereof, `hopping` is only applied between sublattices with said names.

The dimension of `t::AbstractMatrix` must match the orbital dimension of applicable
sublattices (see also `orbitals` option for `hamiltonian`). If `t::UniformScaling` it will
be converted to a (possibly rectangular) identity matrix of the appropriate size when
applied to multiorbital sublattices. Similarly, if `t::SMatrix` it will be truncated or
padded to the appropriate size.

`OnsiteTerm`s and `HoppingTerm`s created with `onsite` or `hopping` can be added or
substracted together to build more complicated `TightbindingModel`s.

    hopping(model::TightbindingModel; kw...)

Return a `TightbindingModel` with only the hopping terms of `model`. Any non-missing `kw` is
applied to all such terms.

# Keyword arguments

Most keywords are the same as for `hopselector`. Only hoppings between two sites at
positions `r₁ = r - dr/2` and `r₂ = r + dr`, belonging to unit cells at integer distance
`dn´` and to sublattices `s₁` and `s₂` will be selected if: `region(r, dr) && s in sublats
&& dn´ in dn && norm(dr) <= range`. If any of these is `missing` it will not be used to
constraint the selection.

The keyword `range` admits the following possibilities

    max_range                   # i.e. `norm(dr) <= max_range`
    (min_range, max_range)      # i.e. `min_range <= norm(dr) <= max_range`

Both `max_range` and `min_range` can be a `Real` or a `NeighborRange` created with
`nrange(n)`. The latter represents the distance of `n`-th nearest neighbors. Note that the
`range` default for `hopping` (unlike for the more general `hopselector`) is `nrange(1)`,
i.e. first-nearest-neighbors.

The keyword `dn` can be a `Tuple`/`Vector`/`SVector` of `Int`s, or a tuple thereof. The
keyword `sublats` allows the following formats:

    sublats = :A => :B                 # Hopping from :A to :B sublattices
    sublats = (:A => :B,)              # Same as above
    sublats = (:A => :B, :C => :D)     # Hopping from :A to :B or :C to :D
    sublats = (:A, :C) .=> (:B, :D)    # Broadcasted pairs, same as above
    sublats = (:A, :C) => (:B, :D)     # Direct product, (:A=>:B, :A=:D, :C=>:B, :C=>D)

The keyword `plusadjoint` produces a model with the input hopping term plus its adjoint.
Note that this substitution is made before multiplying by any coefficient, so that
`im*hopping(..., plusadjoint = true) == im*(hopping(...) + hopping(...)')`.

# Combining models

`OnsiteTerm`s and `HoppingTerm`s created with `onsite` or `hopping` can added or substracted
together or be multiplied by scalars to build more complicated `TightbindingModel`s, e.g.
`onsite(1) - 3 * hopping(2)`

# Examples

```jldoctest
julia> model = 3 * onsite(1) - hopping(2, dn = ((1,2), (0,0)), sublats = :A=>:B)
TightbindingModel{2}: model with 2 terms
  OnsiteTerm{Int64}:
    Sublattices      : any
    Coefficient      : 3
  HoppingTerm{Int64}:
    Sublattice pairs : :A => :B
    dn cell distance : ([1, 2], [0, 0])
    Hopping range    : NeighborRange(1)
    Coefficient      : -1

julia> newmodel = onsite(model) + hopping(model, range = 2)
TightbindingModel{2}: model with 2 terms
  OnsiteTerm{Int64}:
    Sublattices      : any
    Coefficient      : 3
  HoppingTerm{Int64}:
    Sublattice pairs : :A => :B
    dn cell distance : ([1, 2], [0, 0])
    Hopping range    : 2.0
    Coefficient      : -1

julia> LatticePresets.honeycomb() |> hamiltonian(hopping((r,dr) -> cos(r[1]), sublats = (:A,:B) => (:A,:B)))
Hamiltonian{<:Lattice} : Hamiltonian on a 2D Lattice in 2D space
  Bloch harmonics  : 5 (SparseMatrixCSC, sparse)
  Harmonic size    : 2 × 2
  Orbitals         : ((:a,), (:a,))
  Element type     : scalar (ComplexF64)
  Onsites          : 0
  Hoppings         : 6
  Coordination     : 3.0
```

# See also
    `onsite`, `nrange`
"""
function hopping(t; plusadjoint = false, range = nrange(1), kw...)
    hop = hopping(t, hopselector(; range = range, kw...))
    return plusadjoint ? hop + hop' : hop
end
hopping(t, selector) = TightbindingModel(HoppingTerm(t, selector, 1))

hopping(m::TightbindingModel, selector::Selector) =
    TightbindingModel(_onlyhoppings(selector, m.terms...))

_onlyhoppings(s, t::OnsiteTerm, args...) = (_onlyhoppings(s, args...)...,)
_onlyhoppings(s, t::HoppingTerm, args...) =
    (HoppingTerm(t, merge_non_missing(t.selector, s)), _onlyhoppings(s, args...)...)
_onlyhoppings(s) = ()

Base.:*(x::Number, o::OnsiteTerm) =
    OnsiteTerm(o.o, o.selector, x * o.coefficient)
Base.:*(x::Number, t::HoppingTerm) = HoppingTerm(t.t, t.selector, x * t.coefficient)
Base.:*(t::TightbindingModelTerm, x::Number) = x * t
Base.:-(t::TightbindingModelTerm) = (-1) * t

Base.adjoint(t::TightbindingModel) = TightbindingModel(adjoint.(terms(t)))
Base.adjoint(t::OnsiteTerm{<:Function}) = OnsiteTerm(r -> t.o(r)', t.selector, t.coefficient')
Base.adjoint(t::OnsiteTerm) = OnsiteTerm(t.o', t.selector, t.coefficient')
Base.adjoint(t::HoppingTerm{<:Function}) = HoppingTerm((r, dr) -> t.t(r, -dr)', t.selector', t.coefficient')
Base.adjoint(t::HoppingTerm) = HoppingTerm(t.t', t.selector', t.coefficient')

#######################################################################
# offdiagonal
#######################################################################
"""
    offdiagonal(model, lat, nsublats::NTuple{N,Int})

Build a restricted version of `model` that applies only to off-diagonal blocks formed by
sublattice groups of size `nsublats`.
"""
offdiagonal(m::TightbindingModel, lat, nsublats) =
    TightbindingModel(offdiagonal.(m.terms, Ref(lat), Ref(nsublats)))

offdiagonal(o::OnsiteTerm, lat, nsublats) =
    throw(ArgumentError("No onsite terms allowed in off-diagonal coupling"))

function offdiagonal(t::HoppingTerm, lat, nsublats)
    rs = resolve(t.selector, lat)
    s = collect(sublats(rs))
    sr = sublatranges(nsublats...)
    filter!(spair -> findblock(first(spair), sr) != findblock(last(spair), sr), s)
    rs´ = ResolvedSelector(hopselector(rs.selector, sublats = s), lat)
    return HoppingTerm(t.t, rs´, t.coefficient)
end

sublatranges(i::Int, is::Int...) = _sublatranges((1:i,), is...)
_sublatranges(rs::Tuple, i::Int, is...) = _sublatranges((rs..., last(last(rs)) + 1: last(last(rs)) + i), is...)
_sublatranges(rs::Tuple) = rs

findblock(s, sr) = findfirst(r -> s in r, sr)

#######################################################################
# @onsite! and @hopping!
#######################################################################
abstract type AbstractModifier end
abstract type ElementModifier{N,S,F} <: AbstractModifier end

struct ParametricFunction{N,F,P<:Val}
    f::F
    params::P
end

ParametricFunction{N}(f::F, p::P) where {N,F,P} = ParametricFunction{N,F,P}(f, p)

(pf::ParametricFunction)(args...; kw...) = pf.f(args...; kw...)

struct OnsiteModifier{N,S<:Selector,F<:ParametricFunction{N}} <: ElementModifier{N,S,F}
    f::F
    selector::S
end

struct HoppingModifier{N,S<:Selector,F<:ParametricFunction{N}} <: ElementModifier{N,S,F}
    f::F
    selector::S
end

const UniformModifier = ElementModifier{1}
const UniformHoppingModifier = HoppingModifier{1}
const UniformOnsiteModifier = OnsiteModifier{1}

"""
    parameters(p::AbstractModifier...)

Return the parameter names for one or several `AbstractModifier` created with `@onsite!`,
`@hopping!` or `@block!`.
"""
parameters(ms::AbstractModifier...) = mergetuples(_parameters.(ms)...)
_parameters(m::AbstractModifier) = _parameters(m.f)
_parameters(pf::ParametricFunction) = pf.params


"""
    @onsite!(args -> body; kw...)

Create an `ElementModifier <: AbstractModifier`, to be used with `parametric`, that applies
`f = args -> body` to onsite energies specified by `kw` (see `onsite` for details on
possible `kw`s). The form of `args -> body` may be `(o; params...) -> ...` or `(o, r;
params...) -> ...` if the modification is position (`r`) dependent. Keyword arguments
`params` are optional, and include any parameters that `body` depends on that the user may
want to tune.

Note: unlike `onsite` and `hopping`, `ElementModifier`s cannot be combined (i.e. you cannot
do `@onsite!(...) + @hopping!(...)`). `ElementModifier`s are not model terms but
transformations of an existing Hamiltonian that are meant to be applied sequentially (the
order of application usually matters).

# See also
    `@hopping!`, `@block!`, `parametric`
"""
macro onsite!(kw, f)
    f, N, params = get_f_N_params(f, "Only @onsite!(args -> body; kw...) syntax supported. Mind the `;`.")
    return esc(:(Quantica.OnsiteModifier(Quantica.ParametricFunction{$N}($f, $(Val(params))), Quantica.siteselector($kw))))
end

macro onsite!(f)
    f, N, params = get_f_N_params(f, "Only @onsite!(args -> body; kw...) syntax supported.  Mind the `;`.")
    return esc(:(Quantica.OnsiteModifier(Quantica.ParametricFunction{$N}($f, $(Val(params))), Quantica.siteselector())))
end

"""
    @hopping!(args -> body; kw...)

Create an `ElementModifier <: AbstractModifier`, to be used with `parametric`, that applies `f = args -> body`
to hoppings energies specified by `kw` (see `hopping` for details on possible `kw`s). The
form of `args -> body` may be `(t; params...) -> ...` or `(t, r, dr; params...) -> ...` if
the modification is position (`r`, `dr`) dependent. Keyword arguments `params` are optional,
and include any parameters that `body` depends on that the user may want to tune.

Note: unlike `onsite` and `hopping`, `ElementModifier`s cannot be combined (i.e. you cannot
do `@onsite!(...) + @hopping!(...)`). `ElementModifier`s are not model terms but
transformations of an existing Hamiltonian that are meant to be applied sequentially (the
order of application usually matters).

# See also
    `@onsite!`, `@block!`, `parametric`
"""
macro hopping!(kw, f)
    f, N, params = get_f_N_params(f, "Only @hopping!(args -> body; kw...) syntax supported. Mind the `;`.")
    return esc(:(Quantica.HoppingModifier(Quantica.ParametricFunction{$N}($f, $(Val(params))), Quantica.hopselector($kw))))
end

macro hopping!(f)
    f, N, params = get_f_N_params(f, "Only @hopping!(args -> body; kw...) syntax supported. Mind the `;`.")
    return esc(:(Quantica.HoppingModifier(Quantica.ParametricFunction{$N}($f, $(Val(params))), Quantica.hopselector())))
end

# Extracts normalized f, number of arguments and kwarg names from an anonymous function f
function get_f_N_params(f, msg)
    (f isa Expr && f.head == :->) || throw(ArgumentError(msg))
    d = ExprTools.splitdef(f)
    kwargs = convert(Vector{Any}, get!(d, :kwargs, []))
    d[:kwargs] = kwargs  # in case it wasn't Vector{Any} originally
    if isempty(kwargs)
        params = ()
        push!(kwargs, :(_...))  # normalization : append _... to kwargs
    else
        params = get_kwname.(kwargs) |> Tuple
        if !isempty(params) && last(params) == :...
            params = Base.front(params)  # drop _... kwarg from params
        else
            push!(kwargs, :(_...))  # normalization : append _... to kwargs
        end
    end
    N = haskey(d, :args) ? length(d[:args]) : 0
    f´ = ExprTools.combinedef(d)
    return f´, N, params
end

get_kwname(x::Symbol) = x
get_kwname(x::Expr) = x.head === :kw ? x.args[1] : x.head  # x.head == :...

resolve(s::ResolvedSelector, lat) = s

resolve(o::OnsiteModifier, lat) = OnsiteModifier(o.f, resolve(o.selector, lat))

resolve(h::HoppingModifier, lat) = HoppingModifier(h.f, resolve(h.selector, lat))

resolve(t::Tuple, lat) = _resolve(lat, t...)
_resolve(lat, t, ts...) = (resolve(t, lat), _resolve(lat, ts...)...)
_resolve(lat) = ()

# Intended for resolved ElementModifier{N} only. The N is the number of arguments accepted.
@inline (o!::UniformOnsiteModifier)(o, r; kw...) = o!(o; kw...)
@inline (o!::UniformOnsiteModifier)(o, r, dr; kw...) = o!(o; kw...)
@inline (o!::UniformOnsiteModifier)(o; kw...) = o!.f(o; kw...)
@inline (o!::OnsiteModifier)(o, r, dr; kw...) = o!(o, r; kw...)
@inline (o!::OnsiteModifier)(o, r; kw...) = o!.f(o, r; kw...)

@inline (h!::UniformHoppingModifier)(t, r, dr; kw...) = h!(t; kw...)
@inline (h!::UniformHoppingModifier)(t; kw...) = h!.f(t; kw...)
@inline (h!::HoppingModifier)(t, r, dr; kw...) = h!.f(t, r, dr; kw...)

#######################################################################
# @block!
#######################################################################
struct BlockModifier{V,C<:Union{Missing,NTuple{<:Any,SVector}},F<:ParametricFunction{1}} <: AbstractModifier
    f::F
    dns::C
    rows::V
    cols::V
    function BlockModifier{V,C,F}(f, dns, rows, cols) where {V,C<:Union{Missing,NTuple{<:Any,SVector}},F<:ParametricFunction{1}}
        new(f, dns, rows, cols)
    end
end

BlockModifier(f, dns, sites) = BlockModifier(f, dns, sites, sites)

function BlockModifier(f::F, dns, rows, cols) where F<:ParametricFunction{1}
    dns´ = sanitize_dn(dns)
    rows´ = collect(rows)  # rows, cols from siteindices will be generators
    cols´ = collect(cols)  # we need collections of known length to use them in `view`
    V = typeof(rows´)
    C = typeof(dns´)
    return BlockModifier{V,C,F}(f, dns´, rows´, cols´)
end

"""
    @block!((block; params...) -> modified_block, sites; dn = missing)
    @block!((block; params...) -> modified_block, (rows, cols); dn = missing)

Create an `BlockModifier <: AbstractModifier`, to be used with `parametric`, that applies `f
= (block; ...) -> ...` to a block `h[dn][sites, sites]` or `h[dn][rows, cols]` of
hamiltonian `h`. Keyword arguments `params` are optional, and include any parameters that
`modified_block` depends on that the user may want to tune. If the keyword `dn = missing`,
the `dn` in `h[dn]` will be restricted to `dn = (0...)`. Otherwise the specified `dn`'s will
be modified (e.g. when `dn = (1,0)` or `dn = ((1,0), (-1,0))`).

Upon construction of a `ParametricHamiltonian` with a `@block!` modifier, a check is
performed that the whole block specified by `(rows, cols)` is stored in the sparse
Hamiltonian harmonics. If it is not, any non-zero element in `modified_block` will fail to
be applied to the harmonic in question, so a warning is issued. The warning can be ignored
if the user knows that all non-zero elements in `modified_block` are indeed stored in the
harmonic, either as finite matrix elements or structural zeros.

Special care should be taken when using `@block!` on Hamiltonians with different number of
orbitals in different sublattices. To avoid type-instabities in this case, the internal
representation of Hamiltonian harmonics uses a uniform `eltype` that is an `SMatrix{N,N}`
with `N` the maximum number of orbitals among the different sublattices (padded with zeros
in sublattices with less than `N` orbitals). The matrix `modified_block` should have this
same uniform `eltype`.

# See also
    `@onsite!`, `@hopping!`, `parametric`
"""
macro block!(kw, f, rows, cols...)
    f, N, params = get_f_N_params(f, "Only @block!(args -> body, inds...; dn = ...) syntax supported. Mind the `;`.")
    N == 1 || throw(ArgumentError("The function passed to `@block!` should be single-argument, with optional keywords."))
    return esc(:(Quantica.BlockModifier(Quantica.ParametricFunction{1}($f, $(Val(params))), Quantica.sanitize_dn(($kw,)[:dn]), $rows, $(cols...))))
end

macro block!(f, rows, cols...)
    f, N, params = get_f_N_params(f, "Only @block!(args -> body, inds...; dn = ...) syntax supported. Mind the `;`.")
    N == 1 || throw(ArgumentError("The function passed to `@block!` should be single-argument, with optional keywords."))
    return esc(:(Quantica.BlockModifier(Quantica.ParametricFunction{1}($f, $(Val(params))), missing, $rows, $(cols...))))
end

@inline (b!::BlockModifier)(h; kw...) = b!.f(h; kw...)

resolve(e::BlockModifier, lat) = e

Base.in(dn, t::BlockModifier) = t.dns === missing ? iszero(dn) : dn in t.dns