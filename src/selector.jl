############################################################################################
# selector constructors
#region

siteselector(; region = missing, sublats = missing, cells = missing) =
    SiteSelector(region, sublats, cells)
siteselector(s::SiteSelector; region = s.region, sublats = s.sublats, cells = s.cells) =
    SiteSelector(region, sublats, cells)

hopselector(; region = missing, sublats = missing, dcells = missing, range = neighbors(1)) =
    HopSelector(region, sublats, dcells, range)
hopselector(s::HopSelector; region = s.region, sublats = s.sublats, dcells = s.dcells, range = s.range) =
    HopSelector(region, sublats, dcells, range)

neighbors(n::Int) = Neighbors(n)
neighbors(n::Int, lat::Lattice) = nrange(n, lat)

#endregion

############################################################################################
# Base.in constructors
#region

function Base.in((i, r, n)::Tuple{Int,SVector{E,T},SVector{L,Int}}, sel::AppliedSiteSelector{T,E,L}) where {T,E,L}
    return incells(n, sel) && (i, r) in sel
end

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

# foreach_cell(f,...) should be called with a boolean function f that returns whether the
# cell should be mark as accepted when BoxIterated
function foreach_cell(f, sel::AppliedSiteSelector)
    lat = lattice(sel)
    cells_list = cells(sel)
    if isempty(cells_list) # no cells specified
        iter = BoxIterator(zerocell(lat))
        for cell in iter
            f(cell) && accepcell!(iter, cell)
        end
    else
        for cell in cells_list
            f(cell)
        end
    end
    return nothing
end


function foreach_cell(f, sel::AppliedHopSelector)
    lat = lattice(sel)
    dcells_list = dcells(sel)
    if isempty(dcells_list) # no dcells specified
        iter = BoxIterator(zerocell(lat))
        for dn in iter
            f(dn) && acceptcell!(iter, dn)
        end
    else
        for dn in dcells_list
            f(dn)
        end
    end
    return nothing
end

function foreach_site(f, sel::AppliedSiteSelector, cell)
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

function foreach_hop(f, sel::AppliedHopSelector, kdtrees, ni = zerocell(lattice(sel)))
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
    return found
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