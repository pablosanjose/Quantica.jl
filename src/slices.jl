
############################################################################################
# slice of Lattice and LatticeSlice - returns a LatticeSlice
#region

Base.getindex(lat::Lattice; kw...) = lat[siteselector(; kw...)]

Base.getindex(lat::Lattice, ss::SiteSelector) = lat[apply(ss, lat)]

Base.getindex(lat::Lattice, ::UnboundedSiteSelector) = lat[siteselector(; cells = zerocell(lat))]

function Base.getindex(lat::Lattice, as::AppliedSiteSelector)
    latslice = LatticeSlice(lat)
    sinds = Int[]
    foreach_cell(as) do cell
        isempty(sinds) || (sinds = Int[])
        cs = CellSites(cell, sinds)
        foreach_site(as, cell) do s, i, r
            push!(siteindices(cs), i)
        end
        if isempty(cs)
            return false
        else
            push!(subcells(latslice), cs)
            return true
        end
    end
    return latslice
end

Base.getindex(l::Lattice, c::CellSites) = LatticeSlice(l, [apply(c, l)])

Base.getindex(ls::LatticeSlice; kw...) = getindex(ls, siteselector(; kw...))

Base.getindex(ls::LatticeSlice, ss::SiteSelector) = getindex(ls, apply(ss, parent(ls)))

# return cell, siteindex of the i-th site of LatticeSlice
function Base.getindex(l::LatticeSlice, i::Integer)
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
    return zerocell(lattice(l)), 0
end

# indexlist is populated with latslice indices of selected sites
function Base.getindex(latslice::LatticeSlice, as::AppliedSiteSelector)
    lat = parent(latslice)
    latslice´ = LatticeSlice(lat)
    sinds = Int[]
    j = 0
    for subcell in subcells(latslice)
        dn = cell(subcell)
        cs = CellSites(dn, sinds)
        for i in siteindices(subcell)
            j += 1
            r = site(lat, i, dn)
            if (i, r, dn) in as
                push!(siteindices(cs), i)
            end
        end
        if !isempty(cs)
            push!(subcells(latslice´), cs)
            sinds = Int[]  #start new site list
        end
    end
    return latslice´
end

#endregion

############################################################################################
# slice of Hamiltonian h[latslice] - returns a SparseMatrix{B,Int}
#   Elements::B can be transformed by `post(hij, (ci, cj))` using h[latslice; post]
#   Here ci and cj are single-site CellSite for h
#   ParametricHamiltonian deliberately not supported, as the output is not updatable
#region

## disabled this method because h[] is too similar to h[()], and becomes confusing
# Base.getindex(h::Hamiltonian; post = (hij, cij) -> hij, kw...) = h[getindex(lattice(h); kw...), post]

function Base.getindex(h::Hamiltonian, ls::LatticeSlice, post = (hij, cij) -> hij)
    @assert lattice(h) === lattice(ls)
    cszero = zerocellsites(h, 1)
    B = typeof(post(zero(blocktype(h)), (cszero, cszero)))
    ncols = nrows = length(ls)
    builder = CSC{B}(ncols)
    hars = harmonics(h)
    for colcs in cellsites(ls)
        colcell = cell(colcs)
        colsite = siteindices(colcs)
        for har in hars
            rowcell = colcell + dcell(har)
            s = findsubcell(rowcell, ls)
            s === nothing && continue
            (ind, rowoffset) = s
            rowsubcell = subcells(ls, ind)
            rowsubcellinds = siteindices(rowsubcell)
            # rowsubcellinds are the site indices in original unitcell for subcell = rowcell
            # rowoffset is the latslice site offset for the rowsubcellinds sites
            hmat = unflat(matrix(har))
            hrows = rowvals(hmat)
            for ptr in nzrange(hmat, colsite)
                hrow = hrows[ptr]
                for (irow, rowsubcellind) in enumerate(rowsubcellinds)
                    if hrow == rowsubcellind
                        rowcs = cellsite(rowcell, hrow)
                        hij, cij = nonzeros(hmat)[ptr], (rowcs, colcs)
                        # hrow is the original unitcell site index for row.
                        # We need the latslice site index lsrow
                        lsrow = rowoffset + irow
                        pushtocolumn!(builder, lsrow, post(hij, cij))
                    end
                end
            end
        end
        finalizecolumn!(builder)
    end
    return sparse(builder, nrows, ncols)
end

#endregion

############################################################################################
# findsubcell(c, ::LatticeSlice) and Base.in
#region

# returns (subcellindex, siteoffset), or nothing if not found
function findsubcell(cell´::SVector, l::LatticeSlice)
    offset = 0
    for (i, sc) in enumerate(subcells(l))
        if cell´ == cell(sc)
            return i, offset  # since cells are unique
        else
            offset += nsites(sc)
        end
    end
    return nothing
end

## Unused
# findsite(i::Integer, s::CellSites) = findfirst(==(i), siteindices(s))
# function findsite((i, dn)::Tuple{Int,SVector}, ls::LatticeSlice)
#     s = findsubcell(dn, ls)
#     s === nothing && return nothing
#     j = findsite(i, subcells(ls, first(s)))
#     j === nothing && return nothing
#     return last(s) + j
# end

Base.in(idn::Tuple{Int,SVector}, ls::LatticeSlice) = findsite(idn, ls) !== nothing

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
    isempty(lss) || all(isempty, lss) || combine_subcells!(subcells(ls0), subcells.(lss)...)
    return ls0
end

## unused
# combine_subcells(scs::Vector{S}...) where {L,S<:CellSites{L}} = combine_subcells!(S[], scs...)

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
# grow and growdiff
#   LatticeSlice generated by hoppings in h and not contained in seed latslice
#region

function grow(ls::LatticeSlice, h::AbstractHamiltonian)
    parent(ls) === lattice(h) ||
        argerror("Tried to grow a LatticeSlice with a Hamiltonian defined on a different Lattice")
    ls´ = LatticeSlice(parent(ls))
    for sc in subcells(ls)
        c = cell(sc)
        for har in harmonics(h)
            mat = matrix(har)
            c´ = c + dcell(har)
            s = findsubcell(c´, ls´)
            if s === nothing
                sc´ = CellSites(c´)
                push!(subcells(ls´), sc´)
            else
                sc´ = subcells(ls´, first(s))
            end
            for col in siteindices(sc), ptr in nzrange(mat, col)
                row = rowvals(mat)[ptr]
                push!(siteindices(sc´), row)
            end
        end
    end
    for sc in subcells(ls´)
        unique!(sort!(siteindices(sc)))
    end
    return ls´
end

growdiff(ls::LatticeSlice, h::AbstractHamiltonian) = setdiff!(grow(ls, h), ls)

function Base.setdiff!(ls::LatticeSlice, ls0::LatticeSlice)
    for sc in subcells(ls)
        s = findsubcell(cell(sc), ls0)
        s === nothing && continue
        sc0 = subcells(ls0, first(s))
        setdiff!(siteindices(sc), siteindices(sc0))
    end
    deleteif!(isempty, subcells(ls))
    return ls
end

#endregion

############################################################################################
# convert LatticeSlice to a 0D Lattice
#    build a 0D Lattice using the sites in LatticeSlice
#region

function lattice0D(ls::LatticeSlice{T,E}, store = missing) where {T,E}
    lat = parent(ls)
    missing_or_empty!(store)
    sls = [sublat(collect(sublatsites(ls, s, store)); name = sublatname(lat, s))
           for s in sublats(lat)]
    return lattice(sls)
end

# positions of sites in a given sublattice. Pushes selected slice indices into store
function sublatsites(l::LatticeSlice, s::Integer, store = missing)
    n = 0
    gen = ((missing_or_push!(store, n); site(l.lat, i, cell(subcell)))
        for subcell in subcells(l) for i in siteindices(subcell)
        if (n += 1; i in siterange(l.lat, s)))
    return gen
end

missing_or_empty!(::Missing) = missing
missing_or_empty!(v) = empty!(v)

missing_or_push!(::Missing, _) = missing
missing_or_push!(v, n) = push!(v, n)

#endregion

############################################################################################
# convert a LatticeSlice to an OrbitalSlice and CellSites to CellOrbs
#region

orbslice(x, g::GreenSolution) = orbslice(x, hamiltonian(g))
orbslice(x, h::AbstractHamiltonian) = orbslice(x, blockstructure(h))
orbslice(sc::CellSites, bs::OrbitalBlockStructure) = _orbslice((sc,), bs)
orbslice(scs::Vector{<:CellSites}, bs::OrbitalBlockStructure) = _orbslice(scs, bs)
orbslice(ls::LatticeSlice, bs::OrbitalBlockStructure) = _orbslice(subcells(ls), bs)

function _orbslice(subcells, bs::OrbitalBlockStructure)
    subcells = [cellorbs(sc, bs) for sc in subcells]
    return OrbitalSlice(subcells)
end

cellorbs(sc::CellSites, g::GreenSolution) = cellorbs(sc, blockstructure(hamiltonian(g)))
cellorbs(sc::CellSites, h::AbstractHamiltonian) = cellorbs(sc, blockstructure(h))

function cellorbs(sc::CellSites, bs::OrbitalBlockStructure)
    orbinds = Int[]
    orbrngs = UnitRange{Int}[]
    offset = 0
    for i in siteindices(sc)
        irng = flatrange(bs, i)
        len = length(irng)
        append!(orbinds, irng)
        push!(orbrngs, offset+1:offset+len)
        offset += len
    end
    return CellOrbitals(cell(sc), orbinds, orbrngs)
end

#endregion
