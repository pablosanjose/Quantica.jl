############################################################################################
# selector constructors
#region

siteselector(; region = missing, sublats = missing, indices = missing) =
    SiteSelector(region, sublats, indices)
siteselector(s::SiteSelector; region = s.region, sublats = s.sublats, indices = s.indices) =
    SiteSelector(region, sublats, indices)
siteselector(lat::Lattice; kw...) =
    appliedon(siteselector(; kw...), lat)

hopselector(; region = missing, sublats = missing, indices = missing, cells = missing, range = neighbors(1)) =
    HopSelector(region, sublats, indices, cells, range)
hopselector(s::HopSelector; region = s.region, sublats = s.sublats, indices = s.indices, cells = s.dcells, range = s.range) =
    HopSelector(region, sublats, indices, cells, range)
hopselector(lat::Lattice; kw...) =
    appliedon(hopselector(; kw...), lat)

neighbors(n::Int) = Neighbors(n)

#endregion

############################################################################################
# Base.in constructors
#region

function Base.in((i, r)::Tuple{Int,SVector{E,T}}, sel::AppliedSiteSelector{T,E}) where {T,E}
    lat = lattice(sel)
    name = sitesublatname(lat, i)
    return inregion(r, sel) &&
           insublats(name, sel)
end

function Base.in(((j, i), (nj, ni))::Tuple{Pair,Pair}, sel::AppliedHopSelector)
    dcell = nj - ni
    ri, rj = site(lat, i, dni), site(lat, j, dnj)
    r, dr = rdr(rj => ri)
    return ((j, i), (r, dr), dcell) in sel
end

function Base.in(((j, i), (r, dr), dcell)::Tuple{Pair,Tuple,SVector}, sel::AppliedHopSelector)
    lat = lattice(sel)
    namei, namej = sitesublatname(lat, i), sitesublatname(lat, j)
    return !isonsite((j, i), dcell) &&
            indcell(dcell, sel) &&
            insublats(namej => namei, sel) &&
            iswithinrange(dr, sel) &&
            inregion((r, dr), sel)
end

isonsite((j, i), dn) = ifelse(i == j && iszero(dn), true, false)

#endregion

############################################################################################
# foreach_site, foreach_cell, foreach_hop
#region

function foreach_site(f, sel::AppliedSiteSelector, cell = zerocell(lattice(sel)))
    lat = lattice(sel)
    for s in sublats(lat)
        insublats(sublatname(lat, s), sel) || continue
        is = siterange(lat, s)
        for i in is
            r = site(lat, i, cell)
            inregion(r, sel) && f(s, i, r)
        end
    end
    return nothing
end

function foreach_cell(f, sel::AppliedHopSelector)
    lat = lattice(sel)
    dcells_list = dcells(sel)
    if isempty(dcells_list) # no dcells specified
        dcells_iter = BoxIterator(zerocell(lat))
        for dn in dcells_iter
            !indcells(dn, sel) && continue
            f(dn, dcells_iter)
        end
    else
        for dn in dcells_list
            f(dn, missing)
        end
    end
    return nothing
end

function foreach_hop!(f, sel::AppliedHopSelector, iter_ni, kdtrees, ni = zerocell(lattice(sel)))
    lat = lattice(sel)
    _, rmax = sel.range
    # source cell at origin
    nj = zero(ni)
    found = false
    for si in sublats(lat), sj in sublats(lat)
        insublats(sublatname(lat, sj) => sublatname(lat, si), sel) || continue
        js = siterange(lat, sj)
        for j in js
            is = inrange_targets(site(lat, j, nj - ni), lat, si, rmax, kdtrees)
            for i in is
                !isonsite((j, i), nj - ni) || continue
                r, dr = rdr(site(lat, j, nj) => site(lat, i, ni))
                # Make sure we don't stop searching cells until we reach minimum range
                isbelowrange(dr, sel) && (found = true)
                if iswithinrange(dr, sel) && inregion((r, dr), sel)
                    found = true
                    f((si, sj), (i, j), (r, dr))
                end
            end
        end
    end
    found && acceptcell!(iter_ni, ni)
    return nothing
end

# Although range can be (rmin, rmax) we return all targets within rmax.
# Those below rmin get filtered later
function inrange_targets(rsource, lat, si, rmax, kdtrees)
    if isfinite(rmax)
        if !isassigned(kdtrees, si)
            sitepos = sites(lat, si)
            kdtrees[si] = KDTree(sitepos)
        end
        targetlist = inrange(kdtrees[si], rsource, rmax)
        targetlist .+= offsets(lat)[si]
    else
        # need collect for type-stability
        targetlist = collect(siterange(lat, si))
    end
    return targetlist
end

#endregion

############################################################################################
# neighbors
#region

function nrange(n, lat::Lattice{T}) where {T}
    latsites = sites(lat)
    dns = BoxIterator(zerocell(lat))
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

function _nrange(n, tree, r::AbstractVector, nmax)
    for m in n:nmax
        _, dists = knn(tree, r, 1 + m, true)
        popfirst!(dists)
        unique_sorted_approx!(dists)
        length(dists) == n && return maximum(dists)
    end
    return convert(eltype(r), Inf)
end

function unique_sorted_approx!(v::AbstractVector)
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

function ispositive(ndist)
    result = false
    for i in ndist
        i == 0 || (result = i > 0; break)
    end
    return result
end

#endregion