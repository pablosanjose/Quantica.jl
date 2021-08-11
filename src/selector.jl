############################################################################################
# selector constructors
#region

siteselector(; region = missing, sublats = missing, indices = missing) =
    SiteSelector(region, sublats, indices)
siteselector(s::SiteSelector; region = s.region, sublats = s.sublats, indices = s.indices) =
    SiteSelector(region, sublats, indices)
siteselector(lat::Lattice; kw...) =
    appliedon(siteselector(; kw...), lat)

hopselector(; region = missing, sublats = missing, indices = missing, cells = missing, range = nrange(1)) =
    HopSelector(region, sublats, indices, cells, range)
hopselector(s::HopSelector; region = s.region, sublats = s.sublats, indices = s.indices, cells = s.cells, range = s.range) =
    HopSelector(region, sublats, indices, cells, range)
hopselector(lat::Lattice; kw...) =
    appliedon(hopselector(; kw...), lat)

nrange(n::Int) = NeighborRange(n)

function apply(s::HopSelector, l::Lattice)
    s.dcells === missing && (s.range === missing || !isfinite(maximum(s.range))) &&
        throw(ErrorException("Tried to apply an infinite-range HopSelector on an unbounded lattice"))
    return hopselector(s; range = sanitize_minmaxrange(s.range, l))
end

sanitize_minmaxrange(r, lat) = sanitize_minmaxrange((zero(numbertype(lat)), r), lat)
sanitize_minmaxrange((rmin, rmax)::Tuple{Any,Any}, lat) =
    padrange(applyrange(rmin, lat), applyrange(rmax, lat))

applyrange(r::NeighborRange, lat) = nrange(parent(r), lat)
applyrange(r::Real, lat) = r

padrange(r) = padrange(r, 1)
padrange((rmin, rmax)::Tuple{Any,Any}) = (padrange(rmin, -1), padrange(rmax, 1))
padrange(r::Real, m) = isfinite(r) ? float(r) + m * sqrt(eps(float(r))) : missing
# rmax::Missing needed for type-stable hop_targets with infinite range

#endregion

############################################################################################
# Base.in
#region

 # tuple reverse respect to pair
Base.in((j, i)::Pair, s::AppliedOn{<:Selector}) = in_applied((i, j), source(s), target(s))
Base.in(is, s::AppliedOn{<:Selector}) = in_applied(is, source(s), target(s))

in_applied(i::Int, sel::SiteSelector, lat) = in_applied((i, site(lat, i)), sel, lat)

# in_applied((i, celli), sel::SiteSelector, lat) = in_applied((i, site(lat, i, celli)), sel, lat)
# # This indirection allows to reuse computation of r = site(lat, i, celli)
in_applied((i, r)::Tuple{Int,SVector{E,T}}, sel::SiteSelector, lat::Lattice{T,E}) where {T,E} =
    in_recursive(i, sel.indices) &&
    in_recursive(r, sel.region) &&
    in_recursive(sitesublatname(lat, i), sel.sublats)

in_applied(((i, j), (dni, dnj)), sel::HopSelector, lat) =
    !isonsite((i, j), (dni, dnj)) &&
    in_recursive(j => i, sel.indices) &&
    in_recursive(Tuple(dni - dnj), sel.dcells) &&
    in_recursive(sitesublatname(lat, j) => sitesublatname(lat, i), sel.sublats) &&
    isinposition(rdr(site(lat, i, dni), site(lat, j, dnj)), sel.region, sel.range)

isonsite((i, j), (dni, dnj)) = i == j && dni == dnj

isinposition((r, dr), region, range) = isinrange(dr, range) && in_recursive((r, dr), region)

isinrange(dr, (rmin, rmax)::Tuple{Real,Real}) =  rmin^2 <= dr'dr <= rmax^2

in_recursive(i, ::Missing) = true
in_recursive(i, dn::Tuple{Int,Int}) = i == dn
in_recursive(i, name::Symbol) = i == name
in_recursive(i, idx::Number) = i == idx
in_recursive(i, r::AbstractRange) = i in r
in_recursive(i, f::Function) = f(i)
in_recursive((r, dr)::Tuple{SVector,SVector}, region::Function) = region(r, dr)
in_recursive((i, j)::Pair, (is, js)::Pair) = in_recursive(i, is) && in_recursive(j, js)
in_recursive(i, cs) = any(is -> in_recursive(i, is), cs)

#endregion

############################################################################################
# foreach_site, foreach_cell, foreach_hop
#region

function foreach_site(f, latsel::AppliedOn{<:SiteSelector}, cell = zerocell(target(latsel)))
    sel, lat = source(latsel), target(latsel)
    for s in sublats(lat)
        in_recursive(sublatname(lat, s), sel.sublats) || continue
        is, check_is = candidates(sel.indices, siterange(lat, s))
        for i in is
            check_is && in_recursive(i, sel.indices) || continue
            r = site(lat, i, cell)
            in_recursive(r, sel.region) && f(s, i, r)
        end
    end
    return nothing
end

# f(dn, iter_dn) is a function of cell distance dn and cell iterator iter_dn
function foreach_cell(f, latsel::AppliedOn{<:HopSelector})
    sel, lat = source(latsel), target(latsel)
    iter_dn, check_dn = candidates(sel.dcells, BoxIterator(zerocell(lat)))
    for dn in iter_dn
        check_dn && in_recursive(Tuple(dn), sel.dcells) || continue
        f(dn, iter_dn)
    end
    return nothing
end

function foreach_hop!(f, iter_dni, latsel::AppliedOn{<:HopSelector}, kdtrees, dni = zerocell(target(latsel)))
    sel, lat = source(latsel), target(latsel)
    rmin, rmax = sel.range
    dnj = zero(dni)
    found = false
    for si in sublats(lat), sj in sublats(lat)
        in_recursive(sublatname(lat, sj) => sublatname(lat, si), sel.sublats) || continue
        js = source_candidates(sel.indices, siterange(lat, sj))
        for j in js
            check_js &&
            rj = site(lat, j)
            is = target_candidates(rj, sj, rmax, lat, kdtrees)
            for i in is
                !isonsite((i, j), (dni, dnj)) && in_recursive(j => i, sel.indices) || continue
                r, dr = rdr(site(lat, j, dnj), site(lat, i, dni))
                # Make sure we don't stop searching cells until we reach minimum range
                norm(dr) <= rmin && (found = true)
                if isinposition((r, dr), sel.region, sel.range)
                    found = true
                    f(s, i, r)
                end
            end
        end
    end
    found && acceptcell!(iter_dni, dni)
    return nothing
end

# checks whether selection is a known container of the correct eltype(default). If it is,
# returns selection, needs_check = false. Otherwise, returns default, needs_check = true.
candidates(selection::Missing, default) = default, false
candidates(selection, default) = candidates(s, default, eltype(default))
candidates(selection::NTuple{<:Any,T}, default, ::Type{T}) where {T} = selection, false
candidates(selection::T, default, ::Type{T}) where {N,T} = (selection,), false
candidates(selection, default, T) = default, true

source_candidates(selection, default) =
    vcat_or_default(take_element(selection, first), default)
target_candidates(selection, default) =
    vcat_or_default(take_element(selection, last),  default)

take_element(selection::Pair, element) = (element(selection),)
take_element(selection::NTuple{<:Any,Pair}, element) = element.(selection)
take_element(selection, element) = missing

vcat_or_default(::Missing, default) = default
vcat_or_default(elements, default) = vcat(elements...)

# Although range can be (rmin, rmax) we return all targets within rmax.
# Those below rmin get filtered later
function target_candidates(rj, sj, rmax::Real, lat, kdtrees)
    if !isassigned(kdtrees, sj)
        sitepos = sites(lat, sj)
        (kdtrees[s1] = KDTree(sitepos))
    end
    targetlist = inrange(kdtrees[s1], rj, rmax)
    targetlist .+= offsets(lat)[s1]
    return targetlist
end

target_candidates(rj, sj, ::Missing, lat, kdtrees) = siterange(lat, sj)

#     rsel = resolve(term.selector, lat)
#     L > 0 && checkinfinite(rsel)
#     allpos = allsitepositions(lat)
#     for (s2, s1) in sublats(rsel)  # Each is a Pair s2 => s1
#         dns = dniter(rsel)
#         for dn in dns
#             keepgoing = false
#             ijv = builder[dn]
#             for j in source_candidates(rsel, s2)
#                 sitej = allpos[j]
#                 rsource = sitej - bravais(lat) * dn
#                 is = targets(builder, rsel.selector.range, rsource, s1)
#                 for i in is
#                     # Make sure we don't stop searching until we reach minimum range
#                     is_below_min_range((i, j), (dn, zero(dn)), rsel) && (keepgoing = true)
#                     ((i, j), (dn, zero(dn))) in rsel || continue
#                     keepgoing = true
#                     rtarget = allsitepositions(lat)[i]
#                     r, dr = _rdr(rsource, rtarget)
#                     v = to_blocktype(term(r, dr), eltype(builder), builder.orbs[s1], builder.orbs[s2])
#                     push!(ijv, (i, j, v))
#                 end
#             end
#             keepgoing && acceptcell!(dns, dn)
#         end
#     end
#     return nothing
# end

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

############################################################################################
# Lattice - Selector generators
#region

siteisr(lat::Lattice; kw...) = siteisr(siteselector(lat; kw...))

function siteisr(as::AppliedOn{<:SiteSelector})
    R = eltype(sites(target(as)))
    gen = TypedGenerator{Tuple{Int,Int,R}}(
        ((i, s, r) for (i, s, r) in siteisr_candidates(as) if (i, r) in as))
    return gen
end

siteisr_candidates(as) = siteisr_candidates(as, as.indices, as.sublats)