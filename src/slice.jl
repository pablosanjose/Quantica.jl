
############################################################################################
# indexing Lattice and LatticeSlice - returns a LatticeSlice
#region

Base.getindex(lat::Lattice; kw...) = lat[siteselector(; kw...)]

Base.getindex(lat::Lattice, ss::SiteSelector) = lat[apply(ss, lat)]

function Base.getindex(lat::Lattice, as::AppliedSiteSelector)
    latslice = LatticeSlice(lat)
    sinds = Int[]
    foreach_cell(as) do cell
        isempty(sinds) || (sinds = Int[])
        cs = CellSites(cell, sinds)
        foreach_site(as, cell) do s, i, r
            push!(cs, i)
        end
        if isempty(cs)
            return false
        else
            push!(latslice, cs)
            return true
        end
    end
    return latslice
end

Base.getindex(l::Lattice, c::CellSites) = LatticeSlice(l, [c])

Base.getindex(ls::LatticeSlice; kw...) = getindex(ls, siteselector(; kw...))

Base.getindex(ls::LatticeSlice, ss::SiteSelector) = getindex(ls, apply(ss, parent(ls)))

function Base.getindex(l::LatticeSlice{<:Any,<:Any,L}, i::Integer) where {L}
    offset = 0
    for scell in subcells(l)
        ninds = length(siteindices(scell))
        if ninds + offset < i
            offset += ninds
        else
            return cell(scell), siteindices(scell)[i-offset]
        end
    end
    @boundscheck(boundserror(l, i))
end

# indexlist is populated with latslice indices of selected sites
function Base.getindex(latslice::LatticeSlice, as::AppliedSiteSelector)
    lat = parent(latslice)
    latslice´ = LatticeSlice(lat)
    sinds = Int[]
    j = 0
    for subcell in subcells(latslice)
        dn = cell(subcell)
        cs = CellSites(sinds, dn)
        for i in siteindices(subcell)
            j += 1
            r = site(lat, i, dn)
            if (i, r, dn) in as
                push!(cs, i)
            end
        end
        if !isempty(cs)
            push!(latslice´, cs)
            sinds = Int[]  #start new site list
        end
    end
    return latslice´
end

#endregion

############################################################################################
# indexing OrbitalSlice
#region

# create new OrbitalSlice only with the i'th orbitals, where i in inds
function Base.getindex(s::OrbitalSlice{L}, inds::Vector) where {L}
    sort!(inds)
    scs = CellOrbitals{L}[]
    orbinds = Int[]
    cellind = 1
    i = 1
    celloffset = 0
    while i <= length(inds)
        sc = subcells(s, cellind)
        ind = inds[i] - celloffset
        if ind <= norbs(sc)
            push!(orbinds, orbindices(sc)[ind])
            i += 1
        else
            if !isempty(orbinds)
                push!(scs, CellOrbitals(cell(sc), orbinds))
                orbinds = Int[]
            end
            cellind += 1
            celloffset += norbs(sc)
        end
    end
    isempty(orbinds) || push!(scs, CellOrbitals(cell(subcells(s, cellind)), orbinds))
    return OrbitalSlice(scs)
end


#endregion

############################################################################################
# findsubcell(c, ::LatticeSlice) and findsite(i, ::CellSites)
#region

findsubcell(c, l::LatticeSlice{<:Any,<:Any,L}) where {L} =
    findsubcell(SVector{L,Int}(c), l)

# returns (subcell, siteoffset), or nothing if not found
function findsubcell(cell´::SVector, l::LatticeSlice)
    offset = 0
    for sc in subcells(l)
        if cell´ == cell(sc)
            return sc, offset
            return nothing  # since cells are unique
        else
            offset += nsites(sc)
        end
    end
    return nothing
end

findsite(i::Integer, s::CellSites) = findfirst(==(i), siteindices(s))

#endregion

############################################################################################
# combine, combine! and intersect!
#region

combine(ls::LatticeSlice, lss::LatticeSlice...) =
    combine!(LatticeSlice(parent(ls)), ls, lss...)

## unused
# combine(os::OrbitalSlice{L}, oss::OrbitalSlice{L}...) where {L} =
#     combine!(OrbitalSlice{L}(), os, oss...)

# combine!(os::OrbitalSlice, oss::OrbitalSlice...) =
#     OrbitalSlice(combine_subcells!(subcells(os), subcells.(oss)...))

function combine!(ls0::S, lss::S...) where {L,S<:LatticeSlice{<:Any,<:Any,L}}
    lat = parent(ls0)
    all(l -> l === lat, parent.(lss)) ||
        argerror("Cannot combine LatticeBlocks of different lattices")
    isempty(lss) || combine_subcells!(subcells(ls0), subcells.(lss)...)
    return ls0
end

combine_subcells(scs::Vector{S}...) where {L,S<:CellSites{L}} = combine_subcells!(S[], scs...)

function combine_subcells!(sc0::Vector{S}, scs::Vector{S}...) where {L, S<:CellSites{L}}
    allcellinds = Tuple{SVector{L,Int},Int}[]
    for scells in (sc0, scs...), scell in scells, ind in siteindices(scell)
        push!(allcellinds, (cell(scell), ind))
    end
    sort!(allcellinds)
    unique!(allcellinds)

    currentcell = first(first(allcellinds))
    scell = CellSites(currentcell)
    scells = sc0
    empty!(scells)
    push!(scells, scell)
    for (c, i) in allcellinds
        if c == currentcell
            push!(siteindices(scell), i)
        else
            scell = CellSites(c)
            push!(siteindices(scell), i)
            push!(scells, scell)
            currentcell = c
        end
    end
    return sc0
end

# Unused?
# function Base.intersect!(ls::L, ls´::L) where {L<:LatticeSlice}
#     for subcell in subcells(ls)
#         found = false
#         for subcell´ in subcells(ls´)
#             if cell(subcell) == cell(subcell´)
#                 intersect!(siteindices(subcell), siteindices(subcell´))
#                 found = true
#                 break
#             end
#         end
#         found || empty!(subcell)
#     end
#     deleteif!(isempty, subcells(ls))
#     return ls
# end

#endregion

############################################################################################
# convert LatticeSlice to Lattice/OrbitalSlice
#    build a 0D Lattice or an OrbitalSlice using the sites in LatticeSlice
#region

function lattice(ls::LatticeSlice{T,E}, store = missing) where {T,E}
    lat = parent(ls)
    _empty!(store)
    sls = [sublat(collect(sublatsites(ls, s, store)); name = sublatname(lat, s))
           for s in sublats(lat)]
    return lattice(sls)
end

# positions of sites in a given sublattice. Pushes selected slice indices into store
function sublatsites(l::LatticeSlice, s::Integer, store = missing)
    n = 0
    gen = ((_store!(store, n); site(l.lat, i, cell(subcell)))
        for subcell in subcells(l) for i in siteindices(subcell)
        if (n += 1; i in siterange(l.lat, s)))
    return gen
end

orbslice(x, h::AbstractHamiltonian, store...) = orbslice(x, blockstructure(h), store...)
orbslice(sc::CellSites, bs::OrbitalBlockStructure, store...) = _orbslice((sc,), bs, store...)
orbslice(ls::LatticeSlice, bs::OrbitalBlockStructure, store...) =
    _orbslice(subcells(ls), bs, store...)

# stores site and subcell offsets in siteoffsets, subcelloffsets if not misssing
function _orbslice(subcells, bs::OrbitalBlockStructure, siteoffsets = missing, subcelloffsets = missing)
    _empty!(siteoffsets)
    _store!(siteoffsets, 0)
    _empty!(subcelloffsets)
    _store!(subcelloffsets, 0)
    orbscells = [CellOrbitals(cell(sc), Int[]) for sc in subcells]
    offsetall = 0
    for (oc, sc) in zip(orbscells, subcells)
        for i in siteindices(sc)
            irng = flatrange(bs, i)
            append!(orbindices(oc), irng)
            offsetall += length(irng)
            _store!(siteoffsets, offsetall)
        end
        _store!(subcelloffsets, offsetall)
    end
    return OrbitalSlice(orbscells)
end

_empty!(::Missing) = missing
_empty!(v) = empty!(v)

_store!(::Missing, _) = missing
_store!(v, n) = push!(v, n)

#endregion