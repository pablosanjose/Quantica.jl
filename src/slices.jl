
############################################################################################
# slice of Lattice and LatticeSlice - returns a LatticeSlice
#region

Base.getindex(lat::Lattice; kw...) = lat[siteselector(; kw...)]

Base.getindex(lat::Lattice, ls::LatticeSlice) = ls

Base.getindex(lat::Lattice, ss::SiteSelector) = lat[apply(ss, lat)]

Base.getindex(lat::Lattice, ::UnboundedSiteSelector) = lat[siteselector(; cells = zerocell(lat))]

function Base.getindex(lat::Lattice, as::AppliedSiteSelector)
    L = latdim(lat)
    csites = CellSites{L,Vector{Int}}[]
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
            push!(csites, cs)
            return true
        end
    end
    cellsdict = index(cell, csites)
    return LatticeSlice(lat, cellsdict)
end

Base.getindex(l::Lattice, c::CellIndices) = LatticeSlice(l, [apply(c, l)])

Base.getindex(ls::LatticeSlice; kw...) = getindex(ls, siteselector(; kw...))

Base.getindex(ls::LatticeSlice, ss::SiteSelector) = getindex(ls, apply(ss, parent(ls)))

# return cell, siteindex of the i-th site of LatticeSlice
function Base.getindex(l::LatticeSlice, i::Integer)
    offset = 0
    for scell in cellsdict(l)
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
    for subcell in cellsdict(latslice)
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
            push!(cellsdict(latslice´), cs)
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
    # @assert lattice(h) === lattice(ls)   # TODO: fails upon plotting a current density (see tutorial)
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
            rowsubcell = findsubcell(rowcell, ls)
            rowsubcell === nothing && continue
            rowoffset = offsets(ls, rowcell)
            rowsubcellinds = siteindices(rowsubcell)
            # rowsubcellinds are the site indices in original unitcell for subcell = rowcell
            # rowoffset is the latslice site offset for the rowsubcellinds sites
            hmat = unflat(matrix(har))
            hrows = rowvals(hmat)
            for ptr in nzrange(hmat, colsite)
                hrow = hrows[ptr]
                for (irow, rowsubcellind) in enumerate(rowsubcellinds)
                    if hrow == rowsubcellind
                        rowcs = CellSite(rowcell, hrow)
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
# combine, combine! and intersect!
#region

function combine(ls::S, lss::S...) where {S<:LatticeSlice}
    lat = parent(ls)
    all(l -> l === lat, parent.(lss)) ||
        argerror("Cannot combine LatticeSlices of different lattices")
    sc = combine(cellsdict.((ls, lss...))...)
    return LatticeSlice(lat, sc)
end

combine(d) = d

combine(d1::D, d2::D, ds::D...) where {D<:CellIndicesDict} =  mergewith(combine_subcells, d1, d2, ds...)

combine_subcells(c::C, cs::C...) where {C<:CellSites} =
    CellSites(cell(c), union(siteindices(c), siteindices.(cs)...))

combine_subcells(c::C, cs::C...) where {C<:CellOrbitals} =
    CellOrbitals(cell(c), union(orbindices(c), orbindices.(cs)...))

function combine_subcells(c::C, cs::C...) where {C<:CellOrbitalsGrouped}
    groups´ = merge(orbgroups(c), orbgroups.(cs)...)
    indices´ = union(orbindices(c), orbindices.(cs)...)
    return CellIndices(cell(c), indices´, OrbitalLikeGrouped(groups´))
end

#endregion

############################################################################################
# grow and growdiff
#   LatticeSlice generated by hoppings in h and not contained in seed latslice
#region

function grow(ls::LatticeSlice, h::AbstractHamiltonian)
    checksamelattice(ls, h)
    cdict = grow(cellsdict(ls), h)
    return LatticeSlice(parent(ls), cdict)
end

function growdiff(ls::LatticeSlice, h::AbstractHamiltonian)
    checksamelattice(ls, h)
    cdict = cellsdict(ls)
    cdict´ = grow(cdict, h)
    setdiff!(cdict´, cdict)
    return LatticeSlice(parent(ls), cdict´)
end

checksamelattice(ls, h) = parent(ls) === lattice(h) ||
    argerror("Tried to grow a LatticeSlice with a Hamiltonian defined on a different Lattice")

function grow(css::CellSitesDict{L}, h::AbstractHamiltonian) where {L}
    css´ = CellSitesDict{L}()
    for cs in css
        c = cell(cs)
        for har in harmonics(h)
            c´ = c + dcell(har)
            s = findsubcell(c´, css´)
            if s === nothing
                cs´ = CellSites(c´)
                insert!(css´, c´, cs´)
            else
                cs´ = s
            end
            mat = unflat(har)
            for col in siteindices(cs), ptr in nzrange(mat, col)
                row = rowvals(mat)[ptr]
                push!(siteindices(cs´), row)
            end
        end
    end
    for cs in css´
        unique!(sort!(siteindices(cs)))
    end
    return index(cell, css´)
end

function Base.setdiff!(cdict::CellSitesDict, cdict0::CellSitesDict)
    for cs in cdict
        cs0 = findsubcell(cell(cs), cdict0)
        cs0 === nothing && continue
        setdiff!(siteindices(cs), siteindices(cs0))
    end
    deleteif!(isempty, cdict)
    return cdict
end

function deleteif!(test, d::Dictionary)
    for (key, val) in pairs(d)
        test(val) && delete!(d, key)
    end
    return d
end

#endregion

############################################################################################
# convert SiteSlice to a 0D Lattice
#    build a 0D Lattice using the sites in LatticeSlice
#region

function lattice0D(ls::LatticeSlice, store = missing)
    lat = parent(ls)
    missing_or_empty!(store)
    sls = [sublat(collect(sublatsites(ls, s, store)); name = sublatname(lat, s))
           for s in sublats(lat)]
    return lattice(sls)
end

# positions of sites in a given sublattice. Pushes selected slice indices into store
function sublatsites(l::LatticeSlice, s::Integer, store = missing)
    n = 0
    gen = ((missing_or_push!(store, n); site(lattice(l), i, cell(subcell)))
        for subcell in cellsdict(l) for i in siteindices(subcell)
        if (n += 1; i in siterange(lattice(l), s)))
    return gen
end

missing_or_empty!(::Missing) = missing
missing_or_empty!(v) = empty!(v)

missing_or_push!(::Missing, _) = missing
missing_or_push!(v, n) = push!(v, n)

#endregion

############################################################################################
# reordered_site_orbitals
#   convert a list of site indices for an OrbitalSliceGrouped or CellOrbitalsGroupedDict
#   to a list of their of orbital indices
#region

function reordered_site_orbitals(siteinds, orbs::Union{OrbitalSliceGrouped,CellOrbitalsGroupedDict})
    rngs = collect(orbranges(orbs)) # consecutive ranges of orbitals for each site in orbs
    orbinds = Int[]                      # will store the reotdered orbital indices for each site
    for siteind in siteinds
        append!(orbinds, rngs[siteind])
    end
    return orbinds
end

#endregion

############################################################################################
# sites_to_orbs: convert sites to orbitals, preserving site groups if possible
#   conversion rules:
#       - SiteSlice to an OrbitalSliceGrouped
#       - CellSitesDict to a CellOrbitalsGroupedDict
#       - CellSites to a CellOrbitalsGrouped
#       - Union{Integer,Colon} to a CellOrbitalsGrouped
#   no-op rules (site-rouping of ungrouped Orbitals is not possible in general):
#       - AnyOrbitalSlice
#       - AnyCellOrbitalsDict
#       - AnyCellOrbitals
# sites_to_orbs_flat: converts sites to orbitals, without site groups
#   conversion rules:
#       - SiteSlice to an OrbitalSlice
#region

## no-ops

sites_to_orbs(s::AnyOrbitalSlice, _) = s
sites_to_orbs(c::AnyCellOrbitalsDict, _) = c
sites_to_orbs(c::AnyCellOrbitals, _) = c

## convert SiteSlice -> OrbitalSliceGrouped/OrbitalSlice
Contacts
sites_to_orbs(s::SiteSelector, g) = sites_to_orbs(lattice(g)[s], g)
sites_to_orbs(kw::NamedTuple, g) = sites_to_orbs(getindex(lattice(g); kw...), g)
sites_to_orbs(i::Integer, g) = orbslice(selfenergies(contacts(g), i))
sites_to_orbs(l::SiteSlice, g) =
    OrbitalSliceGrouped(lattice(l), sites_to_orbs(cellsdict(l), blockstructure(g)))

sites_to_orbs_flat(l::SiteSlice, g) =
    OrbitalSlice(lattice(l), sites_to_orbs_flat(cellsdict(l), blockstructure(g)))

## convert CellSitesDict to CellOrbitalsGroupedDict/CellOrbitalsDict

sites_to_orbs(c::CellSitesDict, g) = sites_to_orbs(c, blockstructure(g))

function sites_to_orbs(cellsdict::CellSitesDict{L}, os::OrbitalBlockStructure) where {L}
    # inference fails if cellsdict is empty, so we need to specify eltype
    co = CellOrbitalsGrouped{L,Vector{Int}}[sites_to_orbs(cellsites, os) for cellsites in cellsdict]
    return CellOrbitalsGroupedDict(co)
end

sites_to_orbs_flat(c::CellSitesDict, g) = sites_to_orbs_flat(c, blockstructure(g))

function sites_to_orbs_flat(cellsdict::CellSitesDict{L}, os::OrbitalBlockStructure) where {L}
    # inference fails if cellsdict is empty, so we need to specify eltype
    co = CellOrbitals{L,Vector{Int}}[sites_to_orbs_flat(cellsites, os) for cellsites in cellsdict]
    return CellOrbitalsDict(co)
end

## convert CellSites -> CellOrbitalsGrouped

sites_to_orbs(c::CellSites, g) = sites_to_orbs(c, blockstructure(g))

function sites_to_orbs(cs::CellSites, os::OrbitalBlockStructure)
    sites = siteindices(cs)
    orbinds, groups = orbinds_and_groups(sites, os)
    return CellIndices(cell(cs), orbinds, OrbitalLikeGrouped(dictionary(groups)))
end

function sites_to_orbs_flat(cs::CellSites, os::OrbitalBlockStructure)
    sites = siteindices(cs)
    orbinds = orbinds_only(sites, os)
    return CellIndices(cell(cs), orbinds, OrbitalLike())
end

function orbinds_and_groups(sites, os)
    orbinds = Int[]
    groups = Pair{Int,UnitRange{Int}}[]
    for i in sites
        rng = flatrange(os, i)
        append!(orbinds, rng)
        push!(groups, i => rng) # site index => orb range
    end
    return orbinds, groups
end

function orbinds_and_groups(i::Integer, os)
    orbinds = flatrange(os, i)  # a UnitRange
    groups = (i => orbinds,)
    return orbinds, groups
end

function orbinds_and_groups(::Colon, os)
    orbinds = flatrange(os, :)  # a UnitRange
    groups = Pair{Int,UnitRange{Int}}[]
    for i in siterange(os)
        push!(groups, i => flatrange(os, i))
    end
    return orbinds, groups
end

function orbinds_and_groups(sites::AbstractUnitRange, os)
    orbinds = flatrange(os, sites)  # a UnitRange
    groups = Pair{Int,UnitRange{Int}}[]
    for i in sites
        push!(groups, i => flatrange(os, i))
    end
    return orbinds, groups
end

function orbinds_only(sites, os)
    orbinds = Int[]
    for i in sites
        rng = flatrange(os, i)
        append!(orbinds, rng)
    end
    return orbinds
end

#endregion
