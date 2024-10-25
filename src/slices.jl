############################################################################################
# slice of Lattice and LatticeSlice - returns a LatticeSlice
#region

Base.getindex(lat::Lattice; kw...) = lat[siteselector(; kw...)]

Base.getindex(lat::Lattice, kw::NamedTuple) = lat[siteselector(; kw...)]

Base.getindex(lat::Lattice, ls::LatticeSlice) = ls

Base.getindex(lat::Lattice, ss::SiteSelector) = lat[apply(ss, lat)]

# Special case for unbounded selector
Base.getindex(lat::Lattice, ::SiteSelectorAll) = lat[siteselector(; cells = zerocell(lat))]

function Base.getindex(lat::Lattice, as::AppliedSiteSelector)
    L = latdim(lat)
    csites = CellSites{L,Vector{Int}}[]
    if !isnull(as)
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
    end
    cellsdict = CellSitesDict{L}(cell.(csites), csites)
    return LatticeSlice(lat, cellsdict)
end

Base.getindex(l::Lattice, c::AnyCellSites) =
    LatticeSlice(l, [apply(sanitize_cellindices(c, l), l)])

Base.getindex(ls::LatticeSlice; kw...) = getindex(ls, siteselector(; kw...))

Base.getindex(ls::LatticeSlice, kw::NamedTuple) = getindex(ls, siteselector(; kw...))

Base.getindex(ls::LatticeSlice, ss::SiteSelector, args...) =
    getindex(ls, apply(ss, ls), args...)

# return cell, siteindex of the i-th site of LatticeSlice
Base.@propagate_inbounds function Base.getindex(l::LatticeSlice, i::Integer)
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

function Base.getindex(latslice::SiteSlice, as::AppliedSiteSelector)
    lat = parent(latslice)
    L = latdim(lat)
    cellsdict´ = CellSitesDict{L}()
    isnull(as) && return SiteSlice(lat, cellsdict´)
    sinds = Int[]
    for subcell in cellsdict(latslice)
        dn = cell(subcell)
        subcell´ = CellSites(dn, sinds)
        for i in siteindices(subcell)
            r = site(lat, i, dn)
            if (i, r, dn) in as
                push!(siteindices(subcell´), i)
            end
        end
        if !isempty(subcell´)
            set!(cellsdict´, cell(subcell´), subcell´)
            sinds = Int[]  # start new site list
        end
    end
    return SiteSlice(lat, cellsdict´)
end

function Base.getindex(latslice::OrbitalSliceGrouped, as::AppliedSiteSelector)
    lat = parent(latslice)
    L = latdim(lat)
    cellsdict´ = CellOrbitalsGroupedDict{L}()
    isnull(as) && return OrbitalSliceGrouped(lat, cellsdict´)
    oinds = Int[]
    ogroups = Dictionary{Int,UnitRange{Int}}()
    for subcell in cellsdict(latslice)
        dn = cell(subcell)
        subcell´ = CellOrbitalsGrouped(dn, oinds, ogroups)
        orbs´ = orbindices(subcell´)
        for (i, orbrng) in pairs(orbgroups(subcell))
            r = site(lat, i, dn)
            if (i, r, dn) in as
                append!(orbs´, orbrng)
                set!(ogroups, i, orbrng)
            end
        end
        if !isempty(subcell´)
            set!(cellsdict´, cell(subcell´), subcell´)
            # start new orb list
            oinds = Int[]
            ogroups = Dictionary{Int,UnitRange{Int}}()
        end
    end
    return OrbitalSliceGrouped(lat, cellsdict´)
end

# function Base.getindex(latslice::OrbitalSliceGrouped, cs::CellSites)
#     lat = parent(latslice)
#     cs´ = apply(cs, lat)
#     dn = cell(cs´)
#     cd = cellsdict(latslice)
#     groups = haskey(cd, dn) ? orbgroups(cd[dn]) : argerror("cell not found in lattice slice")
#     groups´ = Dictionary{Int,UnitRange{Int}}()
#     orbs´ = Int[]
#     for i in siteindices(cs´)
#         if haskey(groups, i)
#             orbrange = groups[i]
#             append!(orbs´, orbrange)
#             set!(groups´, i, orbrange)
#         else
#             argerror("cellsite not found in lattice slice")
#         end
#     end
#     cellsorbs´ = CellOrbitalsGrouped(dn, orbs´, groups´)
#     cellsdict´ = cellinds_to_dict(cellsorbs´)
#     return OrbitalSliceGrouped(lat, cellsdict´)
# end

#endregion

############################################################################################
# unflat_sparse_slice: slice of Hamiltonian h[latslice] - returns a SparseMatrix{B,Int}
#   This is a more efficient slice builder than mortar, but does not flatten B
#   Elements::B can be transformed by `post(hij, (ci, cj))` with `h[latslice, latslice, post]`
#   Here ci and cj are single-site `CellSite` for h
#   ParametricHamiltonian deliberately not supported, as the output is not updatable
#region

function unflat_sparse_slice(h::Hamiltonian, lsrows::LS, lscols::LS = lsrows, post = (hij, cij) -> hij) where {LS<:LatticeSlice}
    # @assert lattice(h) === lattice(ls)   # TODO: fails upon plotting a current density (see tutorial)
    cszero = zerocellsites(h, 1)           # like `sites(1)`, but with explicit cell
    B = typeof(post(zero(blocktype(h)), (cszero, cszero)))
    nrows, ncols = nsites(lsrows), nsites(lscols)
    builder = CSC{B}(ncols)
    hars = harmonics(h)
    for colcs in cellsites(lscols)
        colcell = cell(colcs)
        colsite = siteindices(colcs)
        for har in hars
            rowcell = colcell + dcell(har)
            rowsubcell = findsubcell(rowcell, lsrows)
            rowsubcell === nothing && continue
            rowoffset = offsets(lsrows, rowcell)
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
# ham_to_orbslices: build OrbitalSliceGrouped for source and destination cells of h
#region

function ham_to_orbslices(h::Hamiltonian)
    lat = lattice(h)
    cssrc = CellSites(zerocell(lat), :)
    osrc = sites_to_orbs(LatticeSlice(lat, [cssrc]), h)
    csdst = [CellSites(dcell(har), :) for har in harmonics(h)]
    odst = sites_to_orbs(LatticeSlice(lat, csdst), h)
    return odst, osrc
end

ham_to_orbslices(h::ParametricHamiltonian) = ham_to_orbslices(parent(h))

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

combine(d1::D, d2::D, ds::D...) where {D<:CellIndicesDict} =
    mergewith(combine_subcells, d1, d2, ds...)

combine_subcells(c::C, cs::C...) where {C<:CellSites} =
    CellSites(cell(c), union(siteindices(c), siteindices.(cs)...))

combine_subcells(c::C, cs::C...) where {C<:CellOrbitals} =
    CellOrbitals(cell(c), union(orbindices(c), orbindices.(cs)...))

function combine_subcells(c::C, cs::C...) where {C<:CellOrbitalsGrouped}
    groups´ = merge(orbgroups(c), orbgroups.(cs)...)
    indices´ = union(orbindices(c), orbindices.(cs)...)
    return CellOrbitalsGrouped(cell(c), indices´, groups´)
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

function grow(css::Union{CellSitesDict{L},CellOrbitalsGroupedDict{L}}, h::AbstractHamiltonian) where {L}
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
    return CellSitesDict{L}(cell.(css´), css´)
end

function Base.setdiff!(cdict::CellSitesDict, cdict0::Union{CellSitesDict,CellOrbitalsGroupedDict})
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
# sites_to_orbs_nogroups: converts sites to orbitals, without site groups
#region

#region ## sites_to_orbs

## no-ops

sites_to_orbs(s::AnyOrbitalSlice, _) = s
sites_to_orbs(c::AnyCellOrbitalsDict, _) = c
sites_to_orbs(c::AnyCellOrbitals, _) = c
sites_to_orbs(c::SparseIndices{<:Any,<:Hamiltonian}, _) = c

# unused
# sites_to_orbs_nogroups(cs::CellOrbitals, _) =  cs

## DiagIndices -> DiagIndices

function sites_to_orbs(d::DiagIndices, g)
    ker = kernel(d)
    inds = sites_to_orbs(parent(d), g)
    return SparseIndices(inds, ker)
end

## PairIndices -> OrbitalPairIndices

# creates a Hamiltonian with same blocktype as g, or complex if kernel::Missing
# this may be used as an intermediate to build sparse versions of g[i,j]
function sites_to_orbs(d::PairIndices{<:HopSelector}, g)
    hg = hamiltonian(g)
    ker = kernel(d)
    bs = maybe_scalarize(blockstructure(hg), ker)
    b = IJVBuilder(lattice(hg), bs)
    hopsel = parent(d)
    h = hamiltonian!(b, hopping(I, hopsel))
    return SparseIndices(h, ker)   # OrbitalPairIndices
end

## convert SiteSlice -> OrbitalSliceGrouped

sites_to_orbs(kw::NamedTuple, g) = sites_to_orbs(siteselector(; kw...), g)
sites_to_orbs(s::SiteSelector, g) = sites_to_orbs(lattice(g)[s], g)
sites_to_orbs(i::Union{Colon,Integer}, g) =
    orbslice(contacts(g), i)
sites_to_orbs(l::SiteSlice, g) =
    OrbitalSliceGrouped(lattice(l), sites_to_orbs(cellsdict(l), blockstructure(g)))
sites_to_orbs(l::SiteSlice, s::OrbitalSliceGrouped) = s[l]

## convert CellSitesDict to CellOrbitalsGroupedDict

sites_to_orbs(c::CellSitesDict, g) = sites_to_orbs(c, blockstructure(g))

function sites_to_orbs(cellsdict::CellSitesDict{L}, os::OrbitalBlockStructure) where {L}
    # inference fails if cellsdict is empty, so we need to specify eltype
    co = CellOrbitalsGrouped{L,Vector{Int}}[sites_to_orbs(cellsites, os) for cellsites in cellsdict]
    return cellinds_to_dict(co)
end

## convert CellSites -> CellOrbitalsGrouped

sites_to_orbs(c::CellSites, g, ker...) =
    sites_to_orbs(sanitize_cellindices(c, g), blockstructure(g), ker...)

function sites_to_orbs(cs::CellSites, os::OrbitalBlockStructure, ker...)
    os´ = maybe_scalarize(os, ker...)
    sites = siteindices(cs)
    groups = _groups(sites, os´) # sites, orbranges
    orbinds = _orbinds(sites, groups, os´)
    return CellOrbitalsGrouped(cell(cs), orbinds, Dictionary(groups...))
end

#endregion

#region ## sites_to_orbs_nogroups

## convert SiteSlice -> OrbitalSlice

sites_to_orbs_nogroups(l::SiteSlice, g) =
    OrbitalSlice(lattice(l), sites_to_orbs_nogroups(cellsdict(l), blockstructure(g)))

## convert CellSitesDict to CellOrbitalsDict

sites_to_orbs_nogroups(c::CellSitesDict, g) = sites_to_orbs_nogroups(c, blockstructure(g))

function sites_to_orbs_nogroups(cellsdict::CellSitesDict{L}, os::OrbitalBlockStructure) where {L}
    # inference fails if cellsdict is empty, so we need to specify eltype
    co = CellOrbitals{L,Vector{Int}}[sites_to_orbs_nogroups(cellsites, os) for cellsites in cellsdict]
    return cellinds_to_dict(co)
end

## convert CellSites -> CellOrbitals (no groups)

sites_to_orbs_nogroups(cs::CellSites, g) =
    sites_to_orbs_nogroups(sanitize_cellindices(cs, g), blockstructure(g))

function sites_to_orbs_nogroups(cs::CellSites, os::OrbitalBlockStructure)
    sites = siteindices(cs)
    orbinds = _orbinds(sites, os)
    return CellOrbitals(cell(cs), orbinds)
end

## convert CellOrbitalsGrouped -> CellOrbitals (no groups)

# unused
# sites_to_orbs_nogroups(cs::CellOrbitalsGrouped, _) = CellOrbitals(cell(cs), orbindices(cs))

#endregion

#region ## CORE FUNCTIONS

_groups(i::Integer, os) = [i], [flatrange(os, i)]
_groups(::Colon, os) = _groups(siterange(os), os)

function _groups(sites, os)
    siteinds = Int[]
    orbranges = UnitRange{Int}[]
    sizehint!(siteinds, length(sites))
    sizehint!(orbranges, length(sites))
    for site in sites
        rng = flatrange(os, site)
        push!(siteinds, site)
        push!(orbranges, rng)
    end
    return siteinds, orbranges
end

_orbinds(sites::Union{Integer,Colon,AbstractUnitRange}, _, os) = flatrange(os, sites)
_orbinds(sites::Union{Integer,Colon,AbstractUnitRange}, os) = flatrange(os, sites)

function _orbinds(_, (sites, orbrngs), os)  # reuse precomputed groups
    orbinds = Int[]
    sizehint!(orbinds, length(sites))
    for rng in orbrngs
        append!(orbinds, rng)
    end
    return orbinds
end

function _orbinds(sites, os)
    orbinds = Int[]
    sizehint!(orbinds, length(sites))
    for i in sites
        rng = flatrange(os, i)
        append!(orbinds, rng)
    end
    return orbinds
end

#endregion
#endregion

############################################################################################
# Utilities
#region

missing_or_empty!(::Missing) = missing
missing_or_empty!(v) = empty!(v)

missing_or_push!(::Missing, _) = missing
missing_or_push!(v, n) = push!(v, n)

#endregion
