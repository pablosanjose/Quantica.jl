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
hopselector(s::HopSelector; region = s.region, sublats = s.sublats, indices = s.indices, cells = s.dcells, range = s.range) =
    HopSelector(region, sublats, indices, cells, range)
hopselector(lat::Lattice; kw...) =
    appliedon(hopselector(; kw...), lat)

nrange(n::Int) = NeighborRange(n)

#endregion

############################################################################################
# selector apply
#region

apply(s::SiteSelector, ::Lattice{T,E}) where {T,E} = AppliedSiteSelector{T,E}(
        r -> in_recursive(r, s.region),
        n -> in_recursive(n, s.sublats),
        i -> in_recursive(i, s.indices)
        )

function apply(s::HopSelector, l::Lattice{T,E,L}) where {T,E,L}
    rmin, rmax = sanitize_minmaxrange(s.range, l)
    L > 0 && s.dcells === missing && rmax === missing &&
        throw(ErrorException("Tried to apply an infinite-range HopSelector on an unbounded lattice"))
    return AppliedHopSelector{T,E,L}(
        (r, dr) -> in_recursive((r, dr), s.region),
        npair   -> in_recursive(npair, s.sublats),
        ipair   -> in_recursive(ipair, s.indices),
        dn      -> in_recursive(Tuple(dn), s.dcells),
        (rmin, rmax)
        )
end

sanitize_minmaxrange(r, lat) = sanitize_minmaxrange((zero(numbertype(lat)), r), lat)
sanitize_minmaxrange((rmin, rmax)::Tuple{Any,Any}, lat) =
    padrange(applyrange(rmin, lat), -1), padrange(applyrange(rmax, lat), 1)

applyrange(r::NeighborRange, lat) = nrange(parent(r), lat)
applyrange(r::Real, lat) = r

padrange(r::Real, m) = isfinite(r) ? float(r) + m * sqrt(eps(float(r))) : float(r)

@inline in_recursive(i, ::Missing) = true
@inline in_recursive((r, dr)::Tuple{SVector,SVector}, region::Function) = ifelse(region(r, dr), true, false)
@inline in_recursive((i, j)::Pair, (is, js)::Pair) = ifelse(in_recursive(i, is) && in_recursive(j, js), true, false)
# This if-elseif helps the compile infer that in_recursive always returns a Bool (it bails after too much dispath)
@inline function in_recursive(i, x)
    result = if x isa Tuple{Int,Int}
            i === x
        elseif x isa Symbol
            i === x
        elseif x isa Number
            i === x
        elseif x isa AbstractRange
            ifelse(i in x, true, false)
        elseif x isa Function
            ifelse(x(i), true, false)
        else
            ifelse(any(is -> in_recursive(i, is), x), true, false)
        end
    return result
end

############################################################################################
# foreach_site, foreach_cell, foreach_hop
#region

function foreach_site(f, latsel::AppliedOn{<:AppliedSiteSelector}, cell = zerocell(target(latsel)))
    sel, lat = source(latsel), target(latsel)
    for s in sublats(lat)
        insublats(sublatname(lat, s), sel) || continue
        is, check_is = candidates(sel.indices, siterange(lat, s))
        for i in is
            check_is && !inindices(i, sel) && continue
            r = site(lat, i, cell)
            inregion(r, sel) && f(s, i, r)
        end
    end
    return nothing
end

function foreach_cell(f, latsel::AppliedOn{<:AppliedHopSelector})
    sel, lat = source(latsel), target(latsel)
    iter_dn, check_dn = candidates(sel.dcells, BoxIterator(zerocell(lat)))
    for dn in iter_dn
        check_dn && !indcells(dn, sel) && continue
        f(dn, iter_dn)
    end
    return nothing
end

function foreach_hop!(f, iter_dni, latsel::AppliedOn{<:AppliedHopSelector}, kdtrees, dni = zerocell(target(latsel)))
    sel, lat = source(latsel), target(latsel)
    _, rmax = sel.range
    dnj = zero(dni)
    found = false
    for si in sublats(lat), sj in sublats(lat)
        insublats(sublatname(lat, sj) => sublatname(lat, si), sel) || continue
        js = source_candidates(sel.indices, () -> siterange(lat, sj))
        for j in js
            is = target_candidates(sel.indices,
                 () -> inrange_targets(site(lat, j, dnj - dni), si, rmax, lat, kdtrees))
            for i in is
                !isonsite((i, j), (dni, dnj)) && inindices(j => i, sel) || continue
                r, dr = rdr(site(lat, j, dnj) => site(lat, i, dni))
                # Make sure we don't stop searching cells until we reach minimum range
                isbelowrange(dr, sel) && (found = true)
                if iswithinrange(dr, sel) && inregion((r, dr), sel)
                    found = true
                    f((si, sj), (i, j), (r, dr))
                end
            end
        end
    end
    found && acceptcell!(iter_dni, dni)
    return nothing
end

# candidates: Build an container of indices, cells, etc... that is guaranteed to include all
# selection, otherwise return `default` (which includes all)
# We check whether selection is a known container of the correct eltype(default). If it is,
# returns selection, needs_check = false. Otherwise, returns default, needs_check = true.
candidates(selection::Missing, default) = default, false
candidates(selection, default) = candidates(selection, default, eltype(default))
candidates(selection::NTuple{<:Any,T}, default, ::Type{T}) where {T} = selection, false
candidates(selection::T, default, ::Type{T}) where {N,T} = (selection,), false
candidates(selection, default, T) = default, true

source_candidates(selection, default_func) =
    vcat_or_default(take_element(selection, first), default_func)
target_candidates(selection, default_func) =
    vcat_or_default(take_element(selection, last), default_func)

take_element(selection::Pair, element) = (element(selection),)
take_element(selection::NTuple{<:Any,Pair}, element) = element.(selection)
take_element(selection, element) = missing

vcat_or_default(::Missing, default_func) = default_func()
vcat_or_default(elements,  default_func) = vcat(elements...)

# Although range can be (rmin, rmax) we return all targets within rmax.
# Those below rmin get filtered later
function inrange_targets(rsource, si, rmax, lat, kdtrees)
    if !isassigned(kdtrees, si)
        sitepos = sites(lat, si)
        kdtrees[si] = KDTree(sitepos)
    end
    targetlist = inrange(kdtrees[si], rsource, rmax)
    targetlist .+= offsets(lat)[si]
    return targetlist
end

#endregion

############################################################################################
# nrange
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