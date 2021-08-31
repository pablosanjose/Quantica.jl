
############################################################################################
# apply selector
#region

function apply(s::SiteSelector, lat::Lattice{T,E,L}) where {T,E,L}
    region = r -> region_apply(r, s.region)
    sublats = Symbol[]
    recursive_push!(sublats, s.sublats)
    return AppliedSiteSelector{T,E,L}(lat, region, sublats)
end

function apply(s::HopSelector, lat::Lattice{T,E,L}) where {T,E,L}
    rmin, rmax = sanitize_minmaxrange(s.range, lat)
    L > 0 && s.dcells === missing && rmax === missing &&
        throw(ErrorException("Tried to apply an infinite-range HopSelector on an unbounded lattice"))
    region = (r, dr) -> region_apply((r, dr), s.region)
    sublats = Pair{Symbol,Symbol}[]
    recursive_push!(sublats, s.sublats)
    dcells = SVector{L,Int}[]
    recursive_push!(dcells, s.dcells)
    return AppliedHopSelector{T,E,L}(lat, region, sublats, dcells, (rmin, rmax))
end

sanitize_minmaxrange(r, lat) = sanitize_minmaxrange((zero(numbertype(lat)), r), lat)
sanitize_minmaxrange((rmin, rmax)::Tuple{Any,Any}, lat) =
    padrange(applyrange(rmin, lat), -1), padrange(applyrange(rmax, lat), 1)

applyrange(r::Neighbors, lat) = nrange(parent(r), lat)
applyrange(r::Real, lat) = r

padrange(r::Real, m) = isfinite(r) ? float(r) + m * sqrt(eps(float(r))) : float(r)

region_apply(r, ::Missing) = true
region_apply((r, dr)::Tuple{SVector,SVector}, region::Function) = ifelse(region(r, dr), true, false)
region_apply(r::SVector, region::Function) = ifelse(region(r), true, false)

recursive_push!(v::Vector, ::Missing) = v
recursive_push!(v::Vector{T}, x::T) where {T} = push!(v, x)
recursive_push!(v::Vector{S}, x::NTuple{<:Any,Int}) where {S<:SVector,T} = push!(v, S(x))
recursive_push!(v::Vector{Pair{T,T}}, x::T) where {T} = push!(v, x => x)
recursive_push!(v::Vector{Pair{T,T}}, (x, y)::Tuple{T,T}) where {T} = push!(v, x => y)
recursive_push!(v::Vector{Pair{T,T}}, (x, y)::Pair) where {T} = push!(v, Iterators.product(x, y))
recursive_push!(v::Vector, xs)= foreach(x -> recursive_push!(v, x), xs)

#endregion

############################################################################################
# apply model terms
#region

function apply(t::OnsiteTerm, (lat, os)::Tuple{Lattice{T,E,L},OrbitalStructure{O}}) where {T,E,L,O}
    f = (r, orbs) -> sanitize_block(O, t(r), (orbs, orbs))
    asel = apply(selector(t), lat)
    return AppliedOnsiteTerm{T,E,L,O}(f, asel)   # f gets wrapped in a FunctionWrapper
end

function apply(t::HoppingTerm, (lat, os)::Tuple{Lattice{T,E,L},OrbitalStructure{O}}) where {T,E,L,O}
    f = (r, dr, orbs) -> sanitize_block(O, t(r, dr), orbs)
    asel = apply(selector(t), lat)
    return AppliedHoppingTerm{T,E,L,O}(f, asel)  # f gets wrapped in a FunctionWrapper
end

apply(m::TightbindingModel, latos) = TightbindingModel(apply.(terms(m), Ref(latos)))

#endregion

############################################################################################
# apply parametric modifiers
#region

function apply(m::OnsiteModifier, h::Hamiltonian)
    f = parametric_function(m)
    asel = apply(selector(m), lattice(h))
    ptrs = pointers(h, asel)
    return AppliedOnsiteModifier(f, ptrs)
end

function apply(m::HoppingModifier, h::Hamiltonian)
    f = parametric_function(m)
    asel = apply(selector(m), lattice(h))
    ptrs = pointers(h, asel)
    return AppliedHoppingModifier(f, ptrs)
end

function pointers(h::Hamiltonian{T,E}, s::AppliedSiteSelector{T,E}) where {T,E}
    ptr_r = Tuple{Int,SVector{E,T},Int}[]
    lat = lattice(h)
    har0 = first(harmonics(h))
    mh = matrix(har0)
    rows = rowvals(mh)
    norbs = norbitals(h)
    for scol in sublats(lat), col in siterange(lat, scol), p in nzrange(mh, col)
        row = rows[p]
        col == row || continue
        r = site(lat, row)
        if (row, r) in s
            n = norbs[scol]
            push!(ptr_r, (p, r, n))
        end
    end
    return ptr_r
end

function pointers(h::Hamiltonian{T,E}, s::AppliedHopSelector{T,E}) where {T,E}
    hars = harmonics(h)
    ptr_r_dr = [Tuple{Int,SVector{E,T},SVector{E,T},Tuple{Int,Int}}[] for _ in hars]
    lat = lattice(h)
    os = orbitalstructure(h)
    dn0 = zerocell(lat)
    norbs = norbitals(h)
    for (har, ptr_r_dr) in zip(hars, ptrs)
        mh = matrix(har)
        rows = rowvals(mh)
        for scol in sublats(lat), col in siterange(lat, scol), p in nzrange(mh, col)
            row = rows[p]
            dn = dcell(har)
            r, dr = rdr(site(lat, col, dn0) => site(lat, row, dn))
            if (col => row, (r, dr), dn) in s
                ncol = norbs[scol]
                srow = site_to_sublat(row, os)
                nrow = norbs[srow]
                push!(ptr_r_dr, (p, r, dr, (nrow, ncol)))
            end
        end
    end
    return ptrs
end

#endregion