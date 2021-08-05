############################################################################################
# selector constructors
#region

siteselector(; region = missing, sublats = missing, indices = missing) =
    UnresolvedSiteSelector(region, sublats, indices)
siteselector(s::UnresolvedSiteSelector; region = s.region, sublats = s.sublats, indices = s.indices) =
    UnresolvedSiteSelector(region, sublats, indices)
siteselector(lat::Lattice; kw...) = resolve(siteselector(; kw...), lat)

hopselector(; region = missing, sublats = missing, cells = missing, range = missing, indices = missing) =
    UnresolvedHopSelector(region, sublats, cells, range, indices)
hopselector(s::HopSelector; region = s.region, sublats = s.sublats, cells = s.cells, range = s.range, indices = s.indices) =
    UnresolvedHopSelector(region, sublats, cells, range, indices)
hopselector(lat::Lattice; kw...) = resolve(hopselector(; kw...), lat)

not(i) = Not(i)
not(i...) = Not(i)

#endregion

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


############################################################################################
# resolve
#region

function resolve(s::SiteSelector, lat::AbstractLattice)
    region  = resolve_Function(s.region)
    indices = resolve_siteinds(s.indices, lat)
    sublats = resolve_sublats(s.sublats, lat)
    return ResolvedSiteSelector(region, indices, sublats, lat)
end

resolve_Function(::Missing) = Returns(true)
resolve_Function(f::Function) = f

resolve_siteinds(inds, lat) = sanitize_


function resolve(s::HopSelector, lat::AbstractLattice)
    s = HopSelector(s.region, resolve_sublat_pairs(s.sublats, lat), check_dn_dims(s.dns, lat), resolve_range(s.range, lat), s.indices)
    return ResolvedSelector(s, lat)
end

resolve_sublats(::Missing, lat) = Int[]
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

#endregion