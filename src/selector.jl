############################################################################################
# selector constructors
#region

siteselector(; region = missing, sublats = missing, indices = missing) =
    SiteSelector(region, sublats, indices)
siteselector(s::SiteSelector; region = s.region, sublats = s.sublats, indices = s.indices) =
    SiteSelector(region, sublats, indices)
siteselector(lat::Lattice; kw...) = applied(siteselector(; kw...), lat)

hopselector(; region = missing, sublats = missing, indices = missing, cells = missing, range = nrange(1)) =
    HopSelector(region, sublats, indices, cells, range)
hopselector(s::HopSelector; region = s.region, sublats = s.sublats, indices = s.indices, cells = s.cells, range = s.range) =
    HopSelector(region, sublats, indices, cells, range)
hopselector(lat::Lattice; kw...) = applied(hopselector(; kw...), lat)

nrange(n::Int) = NeighborRange(n)

applied(s::SiteSelector, l::Lattice) = Applied(s, l)

function applied(s::HopSelector, l::Lattice)
    s´ = hopselector(s; range = applyrange(s.range, l))
    return Applied(s´, l)
end

applyrange(r, lat) = padrange(_applyrange(r, lat))
applyrange((rmin, rmax)::Tuple{Any,Any}, lat) =
    padrange(_applyrange(rmin, lat), _applyrange(rmax, lat))
_applyrange(r::NeighborRange, lat) = nrange(parent(r), lat)
_applyrange(r, lat) = r

padrange(r) = padrange(r, 1)
padrange((rmin, rmax)::Tuple{Any,Any}) = (padrange(rmin, -1), padrange(rmax, 1))
padrange(r::Real, m) = ifelse(isfinite(r), float(r) + m * sqrt(eps(float(r))), float(r))
padrange(r, m) = r

#endregion

############################################################################################
# Base.in
#region

Base.in(i, s::Applied{<:Selector}) = applied_in(i, s.src, s.dst)
Base.in((j, i)::Pair, s::Applied{<:Selector}) = applied_in((i, j), s.src, s.dst) # reversed

applied_in(((i, j), (celli, cellj))::Tuple{Tuple,Tuple}, sel::SiteSelector, lat) =
    isonsite((i, j), (celli, cellj)) && applied_in((i, celli), sel, lat)

applied_in((i, celli), sel::SiteSelector, lat) =
    recursive_in(i, sel.indices) &&
    recursive_in(site(lat, i, celli), sel.region) &&
    recursive_in(sitesublatname(lat, i), sel.sublats)

function applied_in(is::Tuple{Int,Int}, sel::HopSelector, lat)
    cell0 = zero(celltype(lat))
    return applied_in((is, (cell0, cell0)), sel, lat)
end

applied_in(((i, j), (dni, dnj)), sel::HopSelector, lat) =
    !isonsite((i, j), (dni, dnj)) &&
    recursive_in(j => i, sel.indices) &&
    recursive_in(Tuple(dni - dnj), sel.dcells) &&
    recursive_in(sitesublatname(lat, j) => sitesublatname(lat, i), sel.sublats) &&
    isinposition(rdr(site(lat, i, dni), site(lat, j, dnj)), sel.region, sel.range)

isonsite((i, j), (dni, dnj)) = i == j && dni == dnj

isinposition((r, dr), region, range) = isinrange(dr, range) && recursive_in((r, dr), region)

isinrange(dr, rmax::Real) = dr'dr <= rmax^2
isinrange(dr, (rmin, rmax)::Tuple{Real,Real}) =  rmin^2 <= dr'dr <= rmax^2

recursive_in(i, ::Missing) = true
recursive_in(i, dn::Tuple{Int,Int}) = i == dn
recursive_in(i, name::Symbol) = i == name
recursive_in(i, idx::Number) = i == idx
recursive_in(i, r::AbstractRange) = i in r
recursive_in(i, f::Function) = f(i)
recursive_in((r, dr)::Tuple{SVector,SVector}, region::Function) = region(r, dr)
recursive_in((i, j)::Pair, (is, js)::Pair) = recursive_in(i, is) && recursive_in(j, js)
recursive_in(i, cs) = any(is -> recursive_in(i, is), cs)

#endregion

############################################################################################
# nrange
#region

function nrange(n, lat::Lattice)
    latsites = sites(lat)
    T = numbertype(lat)
    dns = BoxIterator(zero(eltype(latsites)))
    br = bravais_mat(lat)
    # 128 is a heuristic cutoff for kdtree vs brute-force search
    if length(latsites) <= 128
        dists = fill(T(Inf), n)
        for dn in dns
            iszero(dn) || ispositive(dn) || continue
            for (i, ri) in enumerate(latsites), (j, rj) in enumerate(latsites)
                j <= i && iszero(dn) && continue
                r = ri - rj + br * dn
                _update_dists!(dists, r'r)
            end
            isfinite(last(dists)) || acceptcell!(dns, dn)
        end
        dist = sqrt(last(dists))
    else
        tree = KDTree(latsites)
        dist = T(Inf)
        for dn in dns
            iszero(dn) || ispositive(dn) || continue
            for r0 in latsites
                r = r0 + br * dn
                dist = min(dist, _nrange(n, tree, r, nsites(lat)))
            end
            isfinite(dist) || acceptcell!(dns, dn)
        end
    end
    return dist
end

function _update_dists!(dists, dist)
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


function unique_sorted_approx!(v::AbstractVector{T}) where {T}
    i = 1
    xprev = first(v)
    for j in 2:length(v)
        if v[j] ≈ xprev
            xprev = v[j]
        else
            i += 1
            xprev = v[i] = v[j]
        end
    end
    resize!(v, i)
    return v
end

#endregion


# ############################################################################################
# # resolve
# #region

# function resolve(s::SiteSelector, lat::Lattice)
#     s = SiteSelector(s.region, resolve_sublats(s.sublats, lat), s.indices)
#     return ResolvedSelector(s, lat)
# end

# resolve_sublats(::Missing, lat) = missing # must be resolved to iterate over sublats
# resolve_sublats(n::Not, lat) = Not(resolve_sublats(parent(n), lat))
# resolve_sublats(s, lat) = resolve_sublat_name.(s, Ref(lat))

# function resolve_sublat_name(name::Union{Symbol,Integer}, lat)
#     i = findfirst(isequal(name), lat.unitcell.names)
#     return i === nothing ? 0 : i
# end

# resolve_sublat_name(s, lat) =
#     throw(ErrorException( "Unexpected format $s for `sublats`, see `onsite` for supported options"))

# function resolve(s::HopSelector, lat::Lattice)
#     s = HopSelector(s.region,
#                     resolve_sublat_pairs(s.sublats, lat),
#                     check_dn_dims(s.dns, lat),
#                     resolve_range(s.range, lat),
#                     s.indices)
#     return ResolvedSelector(s, lat)
# end

# resolve_sublats(::Missing, lat) = missing
# resolve_sublats(n::Not, lat) = Not(resolve_sublats(parent(n), lat))
# resolve_sublats(s, lat) = resolve_sublat_name.(s, Ref(lat))

# resolve_range(r::Tuple, lat) = padrange(_resolve_range.(r, Ref(lat)))
# resolve_range(r, lat) = padrange(_resolve_range(r, lat))
# _resolve_range(r::NeighborRange, lat) = nrange(parent(r), lat)
# _resolve_range(r, lat) = r

# padrange(::Missing) = missing
# padrange(r) = shift_eps(r, 1)
# padrange(r::NTuple{2,Any}) = (shift_eps(first(r), -1), shift_eps(last(r), 1))

# shift_eps(r::Real, m) = ifelse(isfinite(r), float(r) + m * sqrt(eps(float(r))), float(r))
# shift_eps(r, m) = r

# function resolve_sublat_name(name::Union{Symbol,Integer}, lat)
#     i = findfirst(isequal(name), lat.unitcell.names)
#     return i === nothing ? 0 : i
# end

# resolve_sublat_name(s, lat) =
#     throw(ErrorException( "Unexpected format $s for `sublats`, see `siteselector` for supported options"))

# resolve_sublat_pairs(::Missing, lat) = missing
# resolve_sublat_pairs(n::Not, lat) = Not(resolve_sublat_pairs(n.i, lat))
# resolve_sublat_pairs(s::Tuple, lat) = resolve_sublat_pairs.(s, Ref(lat))
# resolve_sublat_pairs(s::Vector, lat) = resolve_sublat_pairs.(s, Ref(lat))
# resolve_sublat_pairs((src, dst)::Pair, lat) = _resolve_sublat_pairs(src, lat) => _resolve_sublat_pairs(dst, lat)
# _resolve_sublat_pairs(n::Not, lat) = Not(_resolve_sublat_pairs(n.i, lat))
# _resolve_sublat_pairs(p, lat) = resolve_sublat_name.(p, Ref(lat))

# resolve_sublat_pairs(s, lat) =
#     throw(ErrorException( "Unexpected format $s for `sublats`, see `hopselector` for supported options"))

# check_dn_dims(dns::Missing, lat::Lattice{<:Any,<:Any,L}) where {L} = dns
# check_dn_dims(dns::Tuple{Vararg{SVector{L,Int}}}, lat::Lattice{<:Any,<:Any,L}) where {L} = dns
# check_dn_dims(dns, lat::Lattice{<:Any,<:Any,L}) where {L} =
#     throw(DimensionMismatch("Specified cell distance `dn` does not match lattice dimension $L"))
# #endregion
