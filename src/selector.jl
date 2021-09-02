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
            indcells(dcell, sel) &&
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