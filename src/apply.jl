
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

#endregion

############################################################################################
# model apply
#region

function apply(t::OnsiteTerm, (lat, os)::Tuple{Lattice{T,E},OrbitalStructure{O}}) where {T,E,O}
    asel = apply(selector(t), lat)
    aons = (r, orbs) -> sanitize_block(blocktype(os), t(r), (orbs, orbs))
    return AppliedOnsiteTerm{T,E,O}(aons, asel)
end

function apply(t::HoppingTerm, (lat, os)::Tuple{Lattice{T,E,L},OrbitalStructure{O}}) where {T,E,L,O}
    asel = apply(selector(t), lat)
    ahop = (r, dr, orbs) -> sanitize_block(blocktype(os), t(r, dr), orbs)
    return AppliedHoppingTerm{T,E,L,O}(ahop, asel)
end

apply(m::TightbindingModel, latos) = TightbindingModel(apply.(terms(m), Ref(latos)))

#endregion