
############################################################################################
# selector apply
#region

apply(s::SiteSelector, lat::Lattice{T,E,L}) where {T,E,L} = AppliedSiteSelector{T,E,L}(
        lat,
        r -> in_recursive(r, s.region),
        n -> in_recursive(n, s.sublats),
        i -> in_recursive(i, s.indices)
        )

function apply(s::HopSelector, lat::Lattice{T,E,L}) where {T,E,L}
    rmin, rmax = sanitize_minmaxrange(s.range, lat)
    L > 0 && s.dcells === missing && rmax === missing &&
        throw(ErrorException("Tried to apply an infinite-range HopSelector on an unbounded lattice"))
    return AppliedHopSelector{T,E,L}(
        lat,
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

in_recursive(i, ::Missing) = true
in_recursive((r, dr)::Tuple{SVector,SVector}, region::Function) = ifelse(region(r, dr), true, false)
in_recursive((i, j)::Pair, (is, js)::Pair) = ifelse(in_recursive(i, is) && in_recursive(j, js), true, false)
in_recursive(i, dn::NTuple{<:Any,Int}) = i === dn
in_recursive(i, j::Number) = i === j
in_recursive(n, name::Symbol) = n === name
in_recursive(i, rng::AbstractRange) = ifelse(i in rng, true, false)
in_recursive(r, region::Function) = ifelse(region(r), true, false)
in_recursive(i, tup) = ifelse(any(is -> in_recursive(i, is), tup), true, false)


#endregion

############################################################################################
# model apply
#region

function apply(t::OnsiteTerm, (lat, os)::Tuple{Lattice{T,E,L},OrbitalStructure{O}}) where {T,E,L,O}
    aons = (r, orbs) -> sanitize_block(blocktype(os), t(r), (orbs, orbs))
    asel = apply(selector(t), lat)
    return AppliedOnsiteTerm{T,E,L,O}(aons, asel)
end

function apply(t::HoppingTerm, (lat, os)::Tuple{Lattice{T,E,L},OrbitalStructure{O}}) where {T,E,L,O}
    ahop = (r, dr, orbs) -> sanitize_block(blocktype(os), t(r, dr), orbs)
    asel = apply(selector(t), lat)
    return AppliedHoppingTerm{T,E,L,O}(ahop, asel)
end

apply(m::TightbindingModel, latos) = TightbindingModel(apply.(terms(m), Ref(latos)))

#endregion