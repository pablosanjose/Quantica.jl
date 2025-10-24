############################################################################################
# selector constructors
#region

siteselector(s::NamedTuple) = siteselector(; s...)
siteselector(; region = missing, sublats = missing, cells = missing) =
    SiteSelector(region, sublats, cells)
siteselector(s::SiteSelector; region = s.region, sublats = s.sublats, cells = s.cells) =
    SiteSelector(region, sublats, cells)

hopselector(h::NamedTuple) = hopselector(; h...)
hopselector(; region = missing, sublats = missing, dcells = missing, range = neighbors(1), includeonsite = false) =
    HopSelector(region, sublats, dcells, range, includeonsite)
hopselector(s::HopSelector; region = s.region, sublats = s.sublats, dcells = s.dcells, range = s.range, includeonsite = s.includeonsite) =
    HopSelector(region, sublats, dcells, range, includeonsite, s.adjoint)

neighbors(n::Int) = Neighbors(n)
neighbors(n::Int, lat::Lattice) = nrange(n, lat)

#endregion

############################################################################################
# Base.in constructors
#region

function Base.in((s, r)::Tuple{Int,SVector{E,T}}, sel::AppliedSiteSelector{T,E}) where {T,E}
    return !isnull(sel) && inregion(r, sel) &&
           insublats(s, sel)
end

function Base.in((s, r, cell)::Tuple{Int,SVector{E,T},SVector{L,Int}}, sel::AppliedSiteSelector{T,E,L}) where {T,E,L}
    return !isnull(sel) && incells(cell, sel) &&
           inregion(r, sel) &&
           insublats(s, sel)
end

## Cannot add this, as it is ambiguous for L == E
# function Base.in((i, cell)::Tuple{Int,SVector{L,Int}}, sel::AppliedSiteSelector{T,E,L}) where {T,E,L}
#     lat = lattice(sel)
#     r = site(lat, i, cell)
#     return (i, r, cell) in sel
# end

## We therefore also skip this for consistency
# function Base.in(((j, i), (nj, ni))::Tuple{Pair,Pair}, sel::AppliedHopSelector)
#     lat = lattice(sel)
#     ri, rj = site(lat, i, ni), site(lat, j, nj)
#     r, dr = rdr(rj => ri)
#     dcell = ni - nj
#     return ((j, i), (r, dr), dcell) in sel
# end

function Base.in(((j, i), (sj, si), (r, dr), dcell)::Tuple{Pair, Pair,Tuple,SVector}, sel::AppliedHopSelector)
    return !isnull(sel) && (includeonsite(sel) || !isonsite((j, i), dcell)) &&
            indcells(dcell, sel) &&
            insublats(sj => si, sel) &&
            iswithinrange(dr, sel) &&
            inregion((r, dr), sel)
end

isonsite((j, i), dn) = ifelse(i == j && iszero(dn), true, false)
isonsite(dr) = iszero(dr)

#endregion

############################################################################################
# foreach_site, foreach_cell, foreach_hop
#region

# foreach_cell(f,...) should be called with a boolean function f that returns whether the
# cell should be mark as accepted when BoxIterated
function foreach_cell(f, sel::AppliedSiteSelector)
    isnull(sel) && return nothing
    lat = lattice(sel)
    cells_list = cells(sel)
    if isempty(cells_list)      # no cells specified
        iter = BoxIterator(zerocell(lat))
        keepgoing = true        # will search until we find at least one
        for cell in iter
            if !incells(cell, sel)
                keepgoing && acceptcell!(iter, cell)
                continue
            end
            found = f(cell)
            if found || keepgoing
                acceptcell!(iter, cell)
                found && (keepgoing = false)
            end
        end
    else
        for cell in cells_list
            f(cell)
        end
    end
    return nothing
end

# This method is essentially identical to the one above. TODO: perhaps refactor to avoid code duplication
function foreach_cell(f, sel::AppliedHopSelector)
    isnull(sel) && return nothing
    lat = lattice(sel)
    dcells_list = dcells(sel)
    if isempty(dcells_list) # no dcells specified
        iter = BoxIterator(zerocell(lat))
        keepgoing = true        # will search until we find at least one
        for dcell in iter
            if !indcells(dcell, sel)
                keepgoing && acceptcell!(iter, dcell)
                continue
            end
            found = f(dcell)
            if found || keepgoing
                acceptcell!(iter, dcell)
                found && (keepgoing = false)
            end
        end
    else
        for dcell in dcells_list
            f(dcell)
        end
    end
    return nothing
end

function foreach_site(f, sel::AppliedSiteSelector, cell::SVector)
    isnull(sel) && return nothing
    lat = lattice(sel)
    for s in sublats(lat)
        insublats(s, sel) || continue
        is = siterange(lat, s)
        for i in is
            r = site(lat, i, cell)
            inregion(r, sel) && f(s, i, r)
        end
    end
    return nothing
end

function foreach_hop(f, sel::AppliedHopSelector, kdtrees::Vector{<:KDTree}, ni::SVector = zerocell(lattice(sel)))
    isnull(sel) && return nothing
    lat = lattice(sel)
    _, rmax = sel.range
    # source cell at origin
    nj = zero(ni)
    found = false
    for si in sublats(lat), sj in sublats(lat)
        insublats(sj => si, sel) || continue
        js = siterange(lat, sj)
        for j in js
            is = inrange_targets(site(lat, j, nj - ni), lat, si, rmax, kdtrees)
            for i in is
                !includeonsite(sel) && isonsite((j, i), ni - nj) && continue
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


## Unused
# function foreach_site(f, sel::AppliedSiteSelector, ls::LatticeSlice)
#     lat = parent(ls)
#     islice = 0
#     for scell in subcells(ls)
#         n = cell(scell)
#         for i in siteindices(scell)
#             r = site(lat, i, n)
#             islice += 1
#             if (i, r, n) in sel
#                 f(i, r, n, islice)
#             end
#         end
#     end
#     return nothing
# end

# function foreach_hop(f, sel::AppliedHopSelector, ls::LatticeSlice, kdtree::KDTree)
#     lat = lattice(sel)
#     _, rmax = sel.range
#     found = false
#     isfiniterange = isfinite(rmax)
#     jslice = 0
#     for scellj in subcells(ls)
#         nj = cell(scellj)
#         for j in siteindices(scellj)
#             jslice += 1
#             rj = site(lat, j, nj)
#             if isfiniterange
#                 targetlist = inrange(kdtree, rj, rmax)
#                 for islice in targetlist
#                     ni, i = ls[islice]
#                     ri = site(lat, i, ni)
#                     r, dr = rdr(rj => ri)
#                     dcell = ni - nj
#                     if (j => i, (r, dr), dcell) in sel
#                         found = true
#                         f((i, j), (r, dr), (ni, nj), (islice, jslice))
#                     end
#                 end
#             else
#                 islice = 0
#                 for scelli in subcells(ls)
#                     ni = cell(scelli)
#                     dcell = ni - nj
#                     for i in siteindices(scelli)
#                         islice += 1
#                         isonsite((j, i), dcell) && continue
#                         ri = site(lat, i, ni)
#                         r, dr = rdr(rj => ri)
#                         if (j => i, (r, dr), dcell) in sel
#                             found = true
#                             f((i, j), (r, dr), (ni, nj), (islice, jslice))
#                         end
#                     end
#                 end
#             end
#         end
#     end
#     return found
# end

#endregion
