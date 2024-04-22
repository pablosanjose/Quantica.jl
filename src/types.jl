############################################################################################
# Lattice  -  see lattice.jl for methods
#region

struct Sublat{T<:AbstractFloat,E}
    sites::Vector{SVector{E,T}}
    name::Symbol
end

struct Unitcell{T<:AbstractFloat,E}
    sites::Vector{SVector{E,T}}
    names::Vector{Symbol}
    offsets::Vector{Int}        # Linear site number offsets for each sublat
    function Unitcell{T,E}(sites, names, offsets) where {T<:AbstractFloat,E}
        names´ = uniquenames!(sanitize_Vector_of_Symbols(names))
        length(names´) == length(offsets) - 1 ||
            argerror("Incorrect number of sublattice names, got $(length(names´)), expected $(length(offsets) - 1)")
        return new(sites, names´, offsets)
    end
end

struct Bravais{T,E,L}
    matrix::Matrix{T}
    function Bravais{T,E,L}(matrix) where {T,E,L}
        (E, L) == size(matrix) || internalerror("Bravais: unexpected matrix size $((E,L)) != $(size(matrix))")
        L > E &&
            throw(DimensionMismatch("Number of Bravais vectors ($L) cannot be greater than embedding dimension $E"))
        return new(matrix)
    end
end

struct Lattice{T<:AbstractFloat,E,L}
    bravais::Bravais{T,E,L}
    unitcell::Unitcell{T,E}
    nranges::Vector{Tuple{Int,T}}  # [(nth_neighbor, min_nth_neighbor_distance)...]
end

#region ## Constructors ##

Bravais(::Type{T}, E, m) where {T} = Bravais(T, Val(E), m)
Bravais(::Type{T}, ::Val{E}, m::Tuple{}) where {T,E} =
    Bravais{T,E,0}(sanitize_Matrix(T, E, ()))
Bravais(::Type{T}, ::Val{E}, m::NTuple{<:Any,Number}) where {T,E} =
    Bravais{T,E,1}(sanitize_Matrix(T, E, (m,)))
Bravais(::Type{T}, ::Val{E}, m::NTuple{L,Any}) where {T,E,L} =
    Bravais{T,E,L}(sanitize_Matrix(T, E, m))
Bravais(::Type{T}, ::Val{E}, m::SMatrix{<:Any,L}) where {T,E,L} =
    Bravais{T,E,L}(sanitize_Matrix(T, E, m))
Bravais(::Type{T}, ::Val{E}, m::AbstractMatrix) where {T,E} =
    Bravais{T,E,size(m,2)}(sanitize_Matrix(T, E, m))
Bravais(::Type{T}, ::Val{E}, m::AbstractVector) where {T,E} =
    Bravais{T,E,1}(sanitize_Matrix(T, E, hcat(m)))

Unitcell(sites::Vector{SVector{E,T}}, names, offsets) where {E,T} =
    Unitcell{T,E}(sites, names, offsets)

function uniquenames!(names::Vector{Symbol})
    allnames = Symbol[:_]
    for (i, name) in enumerate(names)
        if name in allnames
            names[i] = uniquename(allnames, name, i)
            @warn "Renamed repeated sublattice :$name to :$(names[i])"
        end
        push!(allnames, names[i])
    end
    return names
end

function uniquename(allnames, name, i)
    newname = Symbol(Char(64+i)) # Lexicographic, starting from Char(65) = 'A'
    newname = newname in allnames ? uniquename(allnames, name, i + 1) : newname
    return newname
end


#endregion

#region ## API ##

bravais(l::Lattice) = l.bravais

unitcell(l::Lattice) = l.unitcell

nranges(l::Lattice) = l.nranges

bravais_vectors(l::Lattice) = bravais_vectors(l.bravais)
bravais_vectors(b::Bravais) = eachcol(b.matrix)

bravais_matrix(l::Lattice) = bravais_matrix(l.bravais)
bravais_matrix(b::Bravais{T,E,L}) where {T,E,L} =
    convert(SMatrix{E,L,T}, ntuple(i -> b.matrix[i], Val(E*L)))

matrix(b::Bravais) = b.matrix

sublatnames(l::Lattice) = sublatnames(l.unitcell)
sublatnames(u::Unitcell) = u.names
sublatname(l::Lattice, s) = sublatname(l.unitcell, s)
sublatname(u::Unitcell, s) = u.names[s]
sublatname(s::Sublat) = s.name

sublatindex(l::Lattice, name::Symbol) = sublatindex(l.unitcell, name)

function sublatindex(u::Unitcell, name::Symbol)
    i = findfirst(==(name), sublatnames(u))
    i === nothing && boundserror(u, string(name))
    return i
end

nsublats(l::Lattice) = nsublats(l.unitcell)
nsublats(u::Unitcell) = length(u.names)

sublats(l::Lattice) = sublats(l.unitcell)
sublats(u::Unitcell) = 1:nsublats(u)

nsites(s::Sublat) = length(s.sites)
nsites(lat::Lattice, sublat...) = nsites(lat.unitcell, sublat...)
nsites(u::Unitcell) = length(u.sites)
nsites(u::Unitcell, sublat) = sublatlengths(u)[sublat]

sites(s::Sublat) = s.sites
sites(l::Lattice, sublat...) = sites(l.unitcell, sublat...)
sites(u::Unitcell) = u.sites
sites(u::Unitcell, sublat) = view(u.sites, siterange(u, sublat))
sites(u::Unitcell, ::Missing) = sites(u)            # to work with QuanticaMakieExt
sites(l::Lattice, name::Symbol) = sites(unitcell(l), name)
sites(u::Unitcell, name::Symbol) = sites(u, sublatindex(u, name))

site(l::Lattice, i) = sites(l)[i]
site(l::Lattice, i, dn) = site(l, i) + bravais_matrix(l) * dn

siterange(l::Lattice, sublat...) = siterange(l.unitcell, sublat...)
siterange(u::Unitcell, sublat::Integer) = (1+u.offsets[sublat]):u.offsets[sublat+1]
siterange(u::Unitcell, name::Symbol) = siterange(u, sublatindex(u, name))
siterange(u::Unitcell) = 1:last(u.offsets)

sitesublat(lat::Lattice, siteidx) = sitesublat(lat.unitcell.offsets, siteidx)

function sitesublat(offsets, siteidx)
    l = length(offsets)
    for s in 2:l
        @inbounds offsets[s] + 1 > siteidx && return s - 1
    end
    return l
end

sitesublatname(lat, i) = sublatname(lat, sitesublat(lat, i))

sitesublatiter(l::Lattice) = sitesublatiter(l.unitcell)
sitesublatiter(u::Unitcell) = ((i, s) for s in sublats(u) for i in siterange(u, s))

offsets(l::Lattice) = offsets(l.unitcell)
offsets(u::Unitcell) = u.offsets

sublatlengths(lat::Lattice) = sublatlengths(lat.unitcell)
sublatlengths(u::Unitcell) = diff(u.offsets)

embdim(::Sublat{<:Any,E}) where {E} = E
embdim(::Lattice{<:Any,E}) where {E} = E

latdim(::Lattice{<:Any,<:Any,L}) where {L} = L

numbertype(::Sublat{T}) where {T} = T
numbertype(::Lattice{T}) where {T} = T

zerocell(::Bravais{<:Any,<:Any,L}) where {L} = zero(SVector{L,Int})
zerocell(::Lattice{<:Any,<:Any,L}) where {L} = zero(SVector{L,Int})
zerocellsites(l::Lattice, i) = cellsites(zerocell(l), i)

Base.length(l::Lattice) = nsites(l)

Base.copy(l::Lattice) = deepcopy(l)

Base.:(==)(l::Lattice, l´::Lattice) = l.bravais == l´.bravais && l.unitcell == l´.unitcell
Base.:(==)(b::Bravais, b´::Bravais) = b.matrix == b´.matrix
# we do not demand equal names for unitcells to be equal
Base.:(==)(u::Unitcell, u´::Unitcell) = u.sites == u´.sites && u.offsets == u´.offsets

#endregion
#endregion

############################################################################################
# Selectors  -  see selectors.jl for methods
#region

struct SiteSelector{F,S,C}
    region::F
    sublats::S
    cells::C
end

const SiteSelectorAll = SiteSelector{Missing,Missing,Missing}

struct AppliedSiteSelector{T,E,L}
    lat::Lattice{T,E,L}
    region::FunctionWrapper{Bool,Tuple{SVector{E,T}}}
    sublats::Vector{Int}
    cells::Vector{SVector{L,Int}}
    isnull::Bool    # if isnull, the selector selects nothing, regardless of other fields
end

struct HopSelector{F,S,D,R}
    region::F
    sublats::S
    dcells::D
    range::R
    adjoint::Bool  # make apply take the "adjoint" of the selector
end

struct AppliedHopSelector{T,E,L}
    lat::Lattice{T,E,L}
    region::FunctionWrapper{Bool,Tuple{SVector{E,T},SVector{E,T}}}
    sublats::Vector{Pair{Int,Int}}
    dcells::Vector{SVector{L,Int}}
    range::Tuple{T,T}
    isnull::Bool    # if isnull, the selector selects nothing, regardless of other fields
end

struct Neighbors
    n::Int
end

#region ## Constructors ##

HopSelector(re, su, dc, ra) = HopSelector(re, su, dc, ra, false)

#endregion

#region ## API ##

Base.Int(n::Neighbors) = n.n

region(s::Union{SiteSelector,HopSelector}) = s.region

cells(s::SiteSelector) = s.cells

lattice(ap::AppliedSiteSelector) = ap.lat
lattice(ap::AppliedHopSelector) = ap.lat

cells(ap::AppliedSiteSelector) = ap.cells
dcells(ap::AppliedHopSelector) = ap.dcells

# if isempty(s.dcells) or isempty(s.sublats), none were specified, so we must accept any
inregion(r, s::AppliedSiteSelector) = s.region(r)
inregion((r, dr), s::AppliedHopSelector) = s.region(r, dr)

insublats(n, s::AppliedSiteSelector) = isempty(s.sublats) || n in s.sublats
insublats(npair::Pair, s::AppliedHopSelector) = isempty(s.sublats) || npair in s.sublats

incells(cell, s::AppliedSiteSelector) = isempty(s.cells) || cell in s.cells
indcells(dcell, s::AppliedHopSelector) = isempty(s.dcells) || dcell in s.dcells

iswithinrange(dr, s::AppliedHopSelector) = iswithinrange(dr, s.range)
iswithinrange(dr, (rmin, rmax)::Tuple{Real,Real}) =  ifelse(sign(rmin)*rmin^2 <= dr'dr <= rmax^2, true, false)

isbelowrange(dr, s::AppliedHopSelector) = isbelowrange(dr, s.range)
isbelowrange(dr, (rmin, rmax)::Tuple{Real,Real}) =  ifelse(dr'dr < rmin^2, true, false)

isnull(s::AppliedSiteSelector) = s.isnull
isnull(s::AppliedHopSelector) = s.isnull

Base.adjoint(s::HopSelector) = HopSelector(s.region, s.sublats, s.dcells, s.range, !s.adjoint)

Base.NamedTuple(s::SiteSelector) =
    (; region = s.region, sublats = s.sublats, cells = s.cells)
Base.NamedTuple(s::HopSelector) =
    (; region = s.region, sublats = s.sublats, dcells = s.dcells, range = s.range)

#endregion
#endregion

############################################################################################
# MatrixElementTypes
#region

struct SMatrixView{N,M,T,NM}
    s::SMatrix{N,M,T,NM}
    SMatrixView{N,M,T,NM}(mat) where {N,M,T,NM} = new(sanitize_SMatrix(SMatrix{N,M,T}, mat))
end

const MatrixElementType{T} = Union{
    Complex{T},
    SMatrix{N,N,Complex{T}} where {N},
    SMatrixView{N,N,Complex{T}} where {N}}

const MatrixElementUniformType{T} = Union{
    Complex{T},
    SMatrix{N,N,Complex{T}} where {N}}

const MatrixElementNonscalarType{T,N} = Union{
    SMatrix{N,N,Complex{T}},
    SMatrixView{N,N,Complex{T}}}

#region ## Constructors ##

SMatrixView(mat::SMatrix{N,M,T,NM}) where {N,M,T,NM} = SMatrixView{N,M,T,NM}(mat)

SMatrixView{N,M}(mat::AbstractMatrix{T}) where {N,M,T} = SMatrixView{N,M,T}(mat)

SMatrixView{N,M,T}(mat) where {N,M,T} = SMatrixView{N,M,T,N*M}(mat)

SMatrixView(::Type{SMatrix{N,M,T,NM}}) where {N,M,T,NM} = SMatrixView{N,M,T,NM}

#endregion

#region ## API ##

unblock(s::SMatrixView) = s.s
unblock(s) = s

Base.parent(s::SMatrixView) = s.s

Base.view(s::SMatrixView, i...) = view(s.s, i...)

Base.getindex(s::SMatrixView, i...) = getindex(s.s, i...)

Base.zero(::Type{SMatrixView{N,M,T,NM}}) where {N,M,T,NM} = SMatrixView(zero(SMatrix{N,M,T,NM}))
Base.zero(::S) where {S<:SMatrixView} = zero(S)

Base.adjoint(s::SMatrixView) = SMatrixView(s.s')

Base.:+(s::SMatrixView...) = SMatrixView(+(parent.(s)...))
Base.:-(s::SMatrixView) = SMatrixView(-parent.(s))
Base.:*(s::SMatrixView, x::Number) = SMatrixView(parent(s) * x)
Base.:*(x::Number, s::SMatrixView) = SMatrixView(x * parent(s))

Base.:(==)(s::SMatrixView, s´::SMatrixView) = parent(s) == parent(s´)

#endregion
#endregion

############################################################################################
# LatticeSlice - see slices.jl for methods
#   Encodes subsets of sites (or orbitals) of a lattice in different cells. Produced e.g. by
#   lat[siteselector]. No ordering is guaranteed, but cells and sites must both be unique
#region

struct SiteLike end

struct SiteLikePos{T,E,B<:MatrixElementType{T}}
    r::SVector{E,T}
    blocktype::Type{B}
end

struct OrbitalLike end

struct OrbitalLikeGrouped
    groups::Dictionary{Int,UnitRange{Int}}  # site => range of inds in parent
                                            # (non-selected sites are not present)
end

struct CellIndices{L,I,G<:Union{SiteLike,SiteLikePos,OrbitalLike,OrbitalLikeGrouped}}
    cell::SVector{L,Int}
    inds::I    # can be anything: Int, Colon, Vector{Int}, etc.
    type::G    # can be SiteLike, OrbitalLike or OrbitalLikeGrouped
end

const CellSites{L,I} = CellIndices{L,I,SiteLike}
const CellSite{L} = CellIndices{L,Int,SiteLike}
const CellSitePos{T,E,L,B} = CellIndices{L,Int,SiteLikePos{T,E,B}} # for non-spatial models
const AnyCellSite = Union{CellSite,CellSitePos}
const AnyCellSites = Union{CellSites,CellSitePos}

const CellOrbitals{L,I} = CellIndices{L,I,OrbitalLike}
const CellOrbitalsGrouped{L,I} = CellIndices{L,I,OrbitalLikeGrouped}
const CellOrbital{L} = CellIndices{L,Int,OrbitalLike}
const AnyCellOrbitals = Union{CellOrbital,CellOrbitals,CellOrbitalsGrouped}

const CellIndicesDict{L,C<:CellIndices{L}} = Dictionary{SVector{L,Int},C}
const CellSitesDict{L} = Dictionary{SVector{L,Int},CellSites{L,Vector{Int}}}
const CellOrbitalsDict{L} = Dictionary{SVector{L,Int},CellOrbitals{L,Vector{Int}}}
const CellOrbitalsGroupedDict{L} = Dictionary{SVector{L,Int},CellOrbitalsGrouped{L,Vector{Int}}}
const AnyCellOrbitalsDict = Union{CellOrbitalsDict,CellOrbitalsGroupedDict}

struct LatticeSlice{T,E,L,C<:CellIndices{L}}
    lat::Lattice{T,E,L}
    cellsdict::CellIndicesDict{L,C}
    offsets::Dictionary{SVector{L,Int},Int}    # offset from number of indices in each cell
    siteindsdict::Dictionary{CellSite{L},UnitRange{Int}}    # index range in slice per site
    function LatticeSlice{T,E,L,C}(lat, cellsdict) where {T,E,L,C}
        all(cs -> allunique(cs.inds), cellsdict) || argerror("CellSlice cells must be unique")
        offsetsvec = lengths_to_offsets(length, cellsdict)
        pop!(offsetsvec)    # remove last entry, so there is one offset per cellsdict value
        offsets = Dictionary(cell.(cellsdict), offsetsvec)
        siteindsdict = siteindexdict(cellsdict)
        return new(lat, cellsdict, offsets, siteindsdict)
    end
    function LatticeSlice{T,E,L,C}(lat, cellsdict, offsets, siteindsdict) where {T,E,L,C}
        return new(lat, cellsdict, offsets, siteindsdict)
    end
end

const SiteSlice{T,E,L} = LatticeSlice{T,E,L,CellSites{L,Vector{Int}}}
const OrbitalSlice{T,E,L} = LatticeSlice{T,E,L,CellOrbitals{L,Vector{Int}}}
const OrbitalSliceGrouped{T,E,L} = LatticeSlice{T,E,L,CellOrbitalsGrouped{L,Vector{Int}}}
const AnyOrbitalSlice = Union{OrbitalSlice,OrbitalSliceGrouped}

#region ## Constructors ##

CellSite(cell, ind::Int) = CellIndices(sanitize_SVector(Int, cell), ind, SiteLike())
CellSite(c::CellSitePos) = CellSite(c.cell, c.inds)
CellSites(cell, inds = Int[]) = CellIndices(sanitize_SVector(Int, cell), sanitize_cellindices(inds), SiteLike())
# exported lowercase constructor for general inds
cellsites(cell, inds) = CellSites(cell, inds)
# no check for unique inds
unsafe_cellsites(cell, inds) = CellIndices(cell, inds, SiteLike())

CellOrbitals(cell, inds = Int[]) =
    CellIndices(sanitize_SVector(Int, cell), sanitize_cellindices(inds), OrbitalLike())
CellOrbital(cell, ind::Int) =
    CellIndices(sanitize_SVector(Int, cell), ind, OrbitalLike())

CellOrbitalsGrouped(cell, inds, groups::Dictionary) =
    CellIndices(sanitize_SVector(Int, cell), sanitize_cellindices(inds), OrbitalLikeGrouped(groups))

# B <: MatrixElementType{T}
CellSitePos(cell, ind, r, B) = CellIndices(sanitize_SVector(Int, cell), ind, SiteLikePos(r, B))

# LatticeSlice from an AbstractVector of CellIndices
LatticeSlice(lat::Lattice, cs::AbstractVector{<:CellIndices}) =
    LatticeSlice(lat, cellinds_to_dict(cs))

# CellIndices to Dictionary(cell=>cellind)
cellinds_to_dict(cs::AbstractVector{C}) where {L,C<:CellIndices{L}} =
    CellIndicesDict{L,C}(cell.(cs), cs)
cellinds_to_dict(cells::AbstractVector{SVector{L,Int}}, cs::AbstractVector{C}) where {L,C<:CellIndices{L}} =
    CellIndicesDict{L,C}(cells, cs)
cellinds_to_dict(cs::CellIndices{L}) where {L} = cellinds_to_dict(SVector(cs))
# don't allow single-cellsites in dictionaries (it polutes the LatticeSlice type diversity)
cellinds_to_dict(cs::AbstractVector{C}) where {L,C<:CellSite{L}} =
    cellinds_to_dict(CellSites{L,Vector{Int}}.(cs))

# make OrbitalSliceGrouped scalar (i.e. groups are siteindex => siteindex:siteindex)
scalarize(s::OrbitalSliceGrouped) = OrbitalSliceGrouped(s.lat, scalarize(s.cellsdict))
scalarize(d::CellOrbitalsGroupedDict) = scalarize.(d)
scalarize(c::CellOrbitalsGrouped) =
    CellOrbitalsGrouped(c.cell, collect(keys(c.type.groups)),
        dictionary(i => i:i for i in keys(c.type.groups)))

# outer constructor for LatticeSlice
LatticeSlice(lat::Lattice{T,E,L}, cellsdict::CellIndicesDict{L,C}) where {T,E,L,C} =
    LatticeSlice{T,E,L,C}(lat, cellsdict)
SiteSlice(lat::Lattice{T,E,L}, cellsdict::CellIndicesDict{L,C}) where {T,E,L,C<:CellSites} =
    LatticeSlice{T,E,L,C}(lat, cellsdict)
OrbitalSlice(lat::Lattice{T,E,L}, cellsdict::CellIndicesDict{L,C}) where {T,E,L,C<:CellOrbitals} =
    LatticeSlice{T,E,L,C}(lat, cellsdict)
OrbitalSliceGrouped(lat::Lattice{T,E,L}, cellsdict::CellIndicesDict{L,C}) where {T,E,L,C<:CellOrbitalsGrouped} =
    LatticeSlice{T,E,L,C}(lat, cellsdict)

# ranges of inds, where inds are orbital indices for orbgroupeddict, site indices otherwise
siteindexdict(cs::CellOrbitalsGroupedDict) = Dictionary(cellsites(cs), orbranges(cs))
# empty siteindexdict when sites are not known
siteindexdict(::CellOrbitalsDict{L}) where {L} = Dictionary{CellSite{L}, UnitRange{Int}}()
# For CellSitesDict, site indices are used
siteindexdict(cs::CellSitesDict) = dictionary(c => i:i for (i, c) in enumerate(cellsites(cs)))

unsafe_replace_lattice(l::LatticeSlice{T,E,L,C}, lat::Lattice{T,E´,L}) where {T,E,E´,L,C} =
    LatticeSlice{T,E´,L,C}(lat, cellsdict(l), offsets(l), siteindexdict(l))

#endregion

#region ## API ##

lattice(ls::LatticeSlice) = ls.lat

siteindices(s::AnyCellSites) = s.inds
siteindices(s::CellOrbitalsGrouped) = keys(s.type.groups)
orbindices(s::AnyCellOrbitals) = s.inds
siteindex(s::AnyCellSite) = s.inds
orbindex(s::CellOrbital) = s.inds

indexcollection(l::LatticeSlice, c::CellSitePos) = siteindexdict(l)[CellSite(c)]
indexcollection(l::LatticeSlice, c::CellSite) = siteindexdict(l)[c]
indexcollection(l::LatticeSlice, cs::CellSites) =
    [j for i in siteindices(cs) for j in indexcollection(l, CellSite(cell(cs), i))]
indexcollection(lold::LatticeSlice, lnew::LatticeSlice) =
    [j for s in cellsites(lnew) for j in indexcollection(lold, s)]

siteindexdict(l::LatticeSlice) = l.siteindsdict
siteindexdict(l::OrbitalSlice) = missing        # in this case, cellsites are not known

cellsdict(l::LatticeSlice) = l.cellsdict
cellsdict(l::LatticeSlice, cell) = l.cellsdict[cell]

offsets(l::LatticeSlice) = l.offsets
offsets(l::LatticeSlice, i) = l.offsets[i]

ncells(x::LatticeSlice) = ncells(cellsdict(x))
ncells(x::CellIndicesDict) = length(x)
ncells(x::CellIndices) = 1

cell(s::CellIndices) = s.cell
cells(l::LatticeSlice) = keys(l.cellsdict)

nsites(s::LatticeSlice, cell...) = nsites(cellsdict(s, cell...))
nsites(s::CellIndicesDict) = isempty(s) ? 0 : sum(nsites, s)
nsites(c::CellIndices) = length(siteindices(c))

norbitals(s::AnyOrbitalSlice, cell...) = norbitals(cellsdict(s, cell...))
norbitals(s::CellIndicesDict) = isempty(s) ? 0 : sum(norbitals, s)
norbitals(c::AnyCellOrbitals) = length(orbindices(c))

findsubcell(cell::SVector, d::CellIndicesDict) = get(d, cell, nothing)
findsubcell(cell::SVector, l::LatticeSlice) = findsubcell(cell, cellsdict(l))

boundingbox(l::LatticeSlice) = boundingbox(keys(cellsdict(l)))

# interface for non-spatial models (cell already defined for CellIndices)
pos(s::CellSitePos) = s.type.r
ind(s::CellSitePos) = s.inds

# iterators

sites(l::LatticeSlice) =
    (site(l.lat, i, cell(subcell)) for subcell in cellsdict(l) for i in siteindices(subcell))

# single-site (CellSite) iterator
cellsites(cs::Union{SiteSlice,OrbitalSliceGrouped}) = cellsites(cs.cellsdict)
cellsites(cdict::CellIndicesDict) =
    (CellSite(cell(cinds), i) for cinds in cdict for i in siteindices(cinds))

# single-orbital (CellOrbital) iterator
cellorbs(co::Union{OrbitalSlice,OrbitalSliceGrouped}) = cellorbs(co.cellsdict)
cellorbs(codict::Union{CellOrbitalsDict,CellOrbitalsGroupedDict}) =
    (CellOrbital(cell(corbs), i) for corbs in codict for i in orbindices(corbs))

# orbitals in unit cell for each site
orbgroups(s::CellOrbitalsGrouped) = s.type.groups
orbgroups(s::CellOrbitalsGrouped, i::Integer) = s.type.groups[i]
orbgroups(d::CellOrbitalsGroupedDict) = (orbrng for corbs in d for orbrng in orbgroups(corbs))
orbgroups(cs::OrbitalSliceGrouped) = orbgroups(cellsdict(cs))

# set of consecutive orbital ranges for each site in slice
orbranges(cs::OrbitalSliceGrouped) = orbranges(cellsdict(cs))
orbranges(cs::Union{CellOrbitalsGrouped, CellOrbitalsGroupedDict}) = Iterators.accumulate(
    (rng, rng´) -> maximum(rng)+1:maximum(rng)+length(rng´), orbgroups(cs), init = 0:0)

Base.isempty(s::LatticeSlice) = isempty(s.cellsdict)
Base.isempty(s::CellIndices) = isempty(s.inds)

Base.length(l::LatticeSlice) = nsites(l)
Base.length(c::CellIndices) = length(c.inds)

Base.parent(ls::LatticeSlice) = ls.lat

Base.copy(ls::LatticeSlice) = LatticeSlice(ls.lat, copy(ls.cellsdict))

#endregion
#endregion

############################################################################################
# Models  -  see models.jl for methods
#region

abstract type AbstractModel end
abstract type AbstractModelTerm end

# wrapper of a function f(x1, ... xN; kw...) with N arguments and the kwargs in params
struct ParametricFunction{N,F}
    f::F
    params::Vector{Symbol}
end

## Non-parametric ##

struct OnsiteTerm{F,S<:SiteSelector,T<:Number} <: AbstractModelTerm
    f::F
    selector::S
    coefficient::T
end

# specialized for a given lattice *and* hamiltonian - for hamiltonian building
struct AppliedOnsiteTerm{T,E,L,B} <: AbstractModelTerm
    f::FunctionWrapper{B,Tuple{SVector{E,T},Int}}  # o(r, sublat_orbitals)
    selector::AppliedSiteSelector{T,E,L}
end

struct HoppingTerm{F,S<:HopSelector,T<:Number} <: AbstractModelTerm
    f::F
    selector::S
    coefficient::T
end

# specialized for a given lattice *and* hamiltonian - for hamiltonian building
struct AppliedHoppingTerm{T,E,L,B} <: AbstractModelTerm
    f::FunctionWrapper{B,Tuple{SVector{E,T},SVector{E,T},Tuple{Int,Int}}}  # t(r, dr, (orbs1, orbs2))
    selector::AppliedHopSelector{T,E,L}
end

const AbstractTightbindingTerm = Union{OnsiteTerm, AppliedOnsiteTerm,
                                       HoppingTerm, AppliedHoppingTerm}

struct TightbindingModel{T<:NTuple{<:Any,AbstractTightbindingTerm}} <: AbstractModel
    terms::T  # Collection of `AbstractTightbindingTerm`s
end

## Parametric ##

# We fuse applied and non-applied versions, since these only apply the selector, not f
struct ParametricOnsiteTerm{N,S<:Union{SiteSelector,AppliedSiteSelector},F<:ParametricFunction{N},T<:Number} <: AbstractModelTerm
    f::F
    selector::S
    coefficient::T
    spatial::Bool   # If true, f is a function of position r. Otherwise it takes a single CellSite
end

struct ParametricHoppingTerm{N,S<:Union{HopSelector,AppliedHopSelector},F<:ParametricFunction{N},T<:Number} <: AbstractModelTerm
    f::F
    selector::S
    coefficient::T
    spatial::Bool   # If true, f is a function of positions r, dr. Otherwise it takes two CellSite's
end

const AbstractParametricTerm{N} = Union{ParametricOnsiteTerm{N},ParametricHoppingTerm{N}}
const AppliedParametricTerm{N} = Union{ParametricOnsiteTerm{N,<:AppliedSiteSelector},
                                       ParametricHoppingTerm{N,<:AppliedHopSelector}}

struct ParametricModel{T<:NTuple{<:Any,AbstractParametricTerm},M<:TightbindingModel} <: AbstractModel
    npmodel::M  # non-parametric model to use as base
    terms::T    # Collection of `AbstractParametricTerm`s
end

## BlockModels ##

struct InterblockModel{M<:AbstractModel,N}
    model::M
    block::NTuple{N,UnitRange{Int}}  # May be two or more ranges
end

struct IntrablockModel{M<:AbstractModel}
    model::M
    block::UnitRange{Int}
end

const AbstractBlockModel{M} = Union{InterblockModel{M},IntrablockModel{M}}

#region ## Constructors ##

ParametricFunction{N}(f::F, params = Symbol[]) where {N,F} =
    ParametricFunction{N,F}(f, params)

TightbindingModel(ts::AbstractTightbindingTerm...) = TightbindingModel(ts)
ParametricModel(ts::AbstractParametricTerm...) = ParametricModel(TightbindingModel(), ts)
ParametricModel(m::TightbindingModel) = ParametricModel(m, ())
ParametricModel(m::ParametricModel) = m

#endregion

#region ## API ##

(f::ParametricFunction)(args...; kw...) = f.f(args...; kw...)

nonparametric(m::TightbindingModel) = m
nonparametric(m::ParametricModel) = m.npmodel

terms(t::AbstractModel) = t.terms

allterms(t::TightbindingModel) = t.terms
allterms(t::ParametricModel) = (terms(nonparametric(t))..., t.terms...)

selector(t::AbstractModelTerm) = t.selector

functor(t::AbstractModelTerm) = t.f

parameters(t::AbstractParametricTerm) = t.f.params

coefficient(t::OnsiteTerm) = t.coefficient
coefficient(t::HoppingTerm) = t.coefficient
coefficient(t::AbstractParametricTerm) = t.coefficient

narguments(t::OnsiteTerm{<:Function}) = 1
narguments(t::HoppingTerm{<:Function}) = 2
narguments(t::OnsiteTerm) = 0
narguments(t::HoppingTerm) = 0
narguments(t::AbstractParametricTerm) = narguments(t.f)
narguments(::ParametricFunction{N}) where {N} = N

Base.parent(m::InterblockModel) = m.model

block(m::InterblockModel) = m.block

is_spatial(t::AbstractParametricTerm) = t.spatial
is_spatial(t) = true

## call API##

(term::OnsiteTerm{<:Function})(r) = term.coefficient * term.f(r)
(term::OnsiteTerm)(r) = term.coefficient * term.f

(term::AppliedOnsiteTerm)(r, orbs) = term.f(r, orbs)

(term::HoppingTerm{<:Function})(r, dr) = term.coefficient * term.f(r, dr)
(term::HoppingTerm)(r, dr) = term.coefficient * term.f

(term::AppliedHoppingTerm)(r, dr, orbs) = term.f(r, dr, orbs)

(term::AbstractParametricTerm{0})(args...; kw...) = term.coefficient * term.f.f(; kw...)
(term::AbstractParametricTerm{1})(x, args...; kw...) = term.coefficient * term.f.f(x; kw...)
(term::AbstractParametricTerm{2})(x, y, args...; kw...) = term.coefficient * term.f.f(x, y; kw...)
(term::AbstractParametricTerm{3})(x, y, z, args...; kw...) = term.coefficient * term.f.f(x, y, z; kw...)

# We need these for SelfEnergyModelSolver, which uses a ParametricModel. We return a
# ParametricOnsiteTerm, not an OnsiteTerm because the latter is tied to a Hamiltonian at its
# orbital structure, not only to a site selection
function (t::ParametricOnsiteTerm{N})(; kw...) where {N}
    f = ParametricFunction{N}((args...) -> t.f(args...; kw...)) # no params
    return ParametricOnsiteTerm(f, t.selector, t.coefficient, t.spatial)
end

function (t::ParametricHoppingTerm{N})(; kw...) where {N}
    f = ParametricFunction{N}((args...) -> t.f(args...; kw...)) # no params
    return ParametricHoppingTerm(f, t.selector, t.coefficient, t.spatial)
end

## Model term algebra

Base.:*(x::Number, m::TightbindingModel) = TightbindingModel(x .* terms(m))
Base.:*(x::Number, m::ParametricModel) = ParametricModel(x * nonparametric(m), x .* terms(m))
Base.:*(m::AbstractModel, x::Number) = x * m
Base.:-(m::AbstractModel) = (-1) * m

Base.:+(m::TightbindingModel, m´::TightbindingModel) =
    TightbindingModel((terms(m)..., terms(m´)...))
Base.:+(m::ParametricModel, m´::ParametricModel) =
    ParametricModel(nonparametric(m) + nonparametric(m´), (terms(m)..., terms(m´)...))
Base.:+(m::TightbindingModel, m´::ParametricModel) =
    ParametricModel(m + nonparametric(m´), terms(m´))
Base.:+(m::ParametricModel, m´::TightbindingModel) = m´ + m
Base.:-(m::AbstractModel, m´::AbstractModel) = m + (-m´)

Base.:*(x::Number, o::OnsiteTerm) = OnsiteTerm(o.f, o.selector, x * o.coefficient)
Base.:*(x::Number, t::HoppingTerm) = HoppingTerm(t.f, t.selector, x * t.coefficient)
Base.:*(x::Number, o::ParametricOnsiteTerm) =
    ParametricOnsiteTerm(o.f, o.selector, x * o.coefficient, o.spatial)
Base.:*(x::Number, t::ParametricHoppingTerm) =
    ParametricHoppingTerm(t.f, t.selector, x * t.coefficient, t.spatial)

Base.adjoint(m::TightbindingModel) = TightbindingModel(adjoint.(terms(m))...)
Base.adjoint(m::ParametricModel) = ParametricModel(adjoint.(terms(m))...)
Base.adjoint(t::OnsiteTerm{<:Function}) = OnsiteTerm(r -> t.f(r)', t.selector, t.coefficient')
Base.adjoint(t::OnsiteTerm) = OnsiteTerm(t.f', t.selector, t.coefficient')
Base.adjoint(t::HoppingTerm{<:Function}) = HoppingTerm((r, dr) -> t.f(r, -dr)', t.selector', t.coefficient')
Base.adjoint(t::HoppingTerm) = HoppingTerm(t.f', t.selector', t.coefficient')

function Base.adjoint(o::ParametricOnsiteTerm{N}) where {N}
    f = ParametricFunction{N}((args...; kw...) -> o.f(args...; kw...)', o.f.params)
    return ParametricOnsiteTerm(f, o.selector, o.coefficient', o.spatial)
end

function Base.adjoint(t::ParametricHoppingTerm{N}) where {N}
    f = ParametricFunction{N}((args...; kw...) -> t.f(args...; kw...)', t.f.params)
    return ParametricHoppingTerm(f, t.selector, t.coefficient', t.spatial)
end

#endregion
#endregion

############################################################################################
# Model Modifiers  -  see models.jl for methods
#region

abstract type AbstractModifier end

struct OnsiteModifier{N,S<:SiteSelector,F<:ParametricFunction{N}} <: AbstractModifier
    f::F
    selector::S
    spatial::Bool
end

struct AppliedOnsiteModifier{B,N,R<:SVector,F<:ParametricFunction{N},S<:SiteSelector,P<:CellSitePos} <: AbstractModifier
    parentselector::S        # unapplied selector, needed to grow a ParametricHamiltonian
    blocktype::Type{B}  # These are needed to cast the modification to the sublat block type
    f::F
    ptrs::Vector{Tuple{Int,R,P,Int}}
    # [(ptr, r, si, norbs)...] for each selected site, dn = 0 harmonic
    spatial::Bool   # If true, f is a function of position r. Otherwise it takes a single CellSite
end

struct HoppingModifier{N,S<:HopSelector,F<:ParametricFunction{N}} <: AbstractModifier
    f::F
    selector::S
    spatial::Bool  # If true, f is a function of positions r, dr. Otherwise it takes two CellSite's
end

struct AppliedHoppingModifier{B,N,R<:SVector,F<:ParametricFunction{N},S<:HopSelector,P<:CellSitePos} <: AbstractModifier
    parentselector::S        # unapplied selector, needed to grow a ParametricHamiltonian
    blocktype::Type{B}  # These are needed to cast the modification to the sublat block type
    f::F
    ptrs::Vector{Vector{Tuple{Int,R,R,P,P,Tuple{Int,Int}}}}
    # [[(ptr, r, dr, si, sj, (norbs, norbs´)), ...], ...] for each selected hop on each harmonic
    spatial::Bool  # If true, f is a function of positions r, dr. Otherwise it takes two CellSite's
end

const Modifier = Union{OnsiteModifier,HoppingModifier}
const AppliedModifier = Union{AppliedOnsiteModifier,AppliedHoppingModifier}

#region ## Constructors ##

AppliedOnsiteModifier(m::AppliedOnsiteModifier, ptrs) =
    AppliedOnsiteModifier(m.parentselector, m.blocktype, m.f, ptrs, m.spatial)

AppliedHoppingModifier(m::AppliedHoppingModifier, ptrs) =
    AppliedHoppingModifier(m.parentselector, m.blocktype, m.f, ptrs, m.spatial)

#endregion

#region ## API ##

selector(m::Modifier) = m.selector
selector(m::AppliedModifier) = m.parentselector

parameters(m::AbstractModifier) = m.f.params

parametric_function(m::AbstractModifier) = m.f

pointers(m::AppliedModifier) = m.ptrs

blocktype(m::AppliedModifier) = m.blocktype

is_spatial(m::AbstractModifier) = m.spatial

narguments(m::AbstractModifier) = narguments(m.f)

@inline (m::AppliedOnsiteModifier{B,1})(o, r, orbs; kw...) where {B} =
    mask_block(B, m.f.f(o; kw...), (orbs, orbs))
@inline (m::AppliedOnsiteModifier{B,2})(o, r, orbs; kw...) where {B} =
    mask_block(B, m.f.f(o, r; kw...), (orbs, orbs))

@inline (m::AppliedHoppingModifier{B,1})(t, r, dr, orborb; kw...) where {B} =
    mask_block(B, m.f.f(t; kw...), orborb)
@inline (m::AppliedHoppingModifier{B,3})(t, r, dr, orborb; kw...) where {B} =
    mask_block(B, m.f.f(t, r, dr; kw...), orborb)

Base.similar(m::A) where {A <: AppliedModifier} = A(m.blocktype, m.f, similar(m.ptrs, 0), m.spatial)

Base.parent(m::AppliedOnsiteModifier) = OnsiteModifier(m.f, m.parentselector, m.spatial)
Base.parent(m::AppliedHoppingModifier) = HoppingModifier(m.f, m.parentselector, m.spatial)

#endregion
#endregion

############################################################################################
# OrbitalBlockStructure
#    Block structure for Hamiltonians, sorted by sublattices
#region

struct OrbitalBlockStructure{B}
    blocksizes::Vector{Int}       # number of orbitals per site in each sublattice
    subsizes::Vector{Int}         # number of blocks (sites) in each sublattice
    function OrbitalBlockStructure{B}(blocksizes, subsizes) where {B}
        subsizes´ = Quantica.sanitize_Vector_of_Type(Int, subsizes)
        # This checks also that they are of equal length
        blocksizes´ = Quantica.sanitize_Vector_of_Type(Int, length(subsizes´), blocksizes)
        return new(blocksizes´, subsizes´)
    end
end

#region ## Constructors ##

@inline function OrbitalBlockStructure(T, orbitals, subsizes)
    orbitals´ = sanitize_orbitals(orbitals) # <:Union{Val,distinct_collection}
    B = blocktype(T, orbitals´)
    return OrbitalBlockStructure{B}(orbitals´, subsizes)
end

# Useful as a minimal constructor when no orbital information is known
OrbitalBlockStructure{B}(hsize::Int) where {B} = OrbitalBlockStructure{B}(Val(1), [hsize])

blocktype(::Type{T}, m::Val{1}) where {T} = Complex{T}
blocktype(::Type{T}, m::Val{N}) where {T,N} = SMatrix{N,N,Complex{T},N*N}
blocktype(T::Type, distinct_norbs) = maybe_SMatrixView(blocktype(T, val_maximum(distinct_norbs)))
maybe_SMatrixView(C::Type{<:Complex}) = C
maybe_SMatrixView(S::Type{<:SMatrix}) = SMatrixView(S)

val_maximum(n::Int) = Val(n)
val_maximum(ns) = Val(maximum(argval.(ns)))

argval(::Val{N}) where {N} = N
argval(n::Int) = n

#endregion

#region ## API ##

blocktype(::OrbitalBlockStructure{B}) where {B} = B

blockeltype(::OrbitalBlockStructure{<:MatrixElementType{T}}) where {T} = Complex{T}

blocksizes(b::OrbitalBlockStructure) = b.blocksizes

subsizes(b::OrbitalBlockStructure) = b.subsizes

flatsize(b::OrbitalBlockStructure) = blocksizes(b)' * subsizes(b)
flatsize(b::OrbitalBlockStructure{B}, ls::SiteSlice) where {B<:Union{Complex,SMatrix}} =
    length(ls) * blocksize(b)
flatsize(b::OrbitalBlockStructure, ls::SiteSlice) = sum(cs -> flatsize(b, cs), cellsites(ls); init = 0)
flatsize(b::OrbitalBlockStructure, cs::AnyCellSite) = blocksize(b, siteindex(cs))

unflatsize(b::OrbitalBlockStructure) = sum(subsizes(b))

blocksize(b::OrbitalBlockStructure, iunflat, junflat) = (blocksize(b, iunflat), blocksize(b, junflat))

blocksize(b::OrbitalBlockStructure{<:SMatrixView}, iunflat) = length(flatrange(b, iunflat))

blocksize(b::OrbitalBlockStructure{B}, iunflat...) where {N,B<:SMatrix{N}} = N

blocksize(b::OrbitalBlockStructure{B}, iunflat...) where {B<:Number} = 1

siterange(b::OrbitalBlockStructure) = 1:unflatsize(b)

orbrange(b::OrbitalBlockStructure) = 1:flatsize(b)

function sublatorbrange(b::OrbitalBlockStructure, sind::Integer)
    bss = blocksizes(b)
    sss = subsizes(b)
    offset = sind == 1 ? 0 : sum(i -> bss[i] * sss[i], 1:sind-1)
    rng = offset + 1:offset + bss[sind] * sss[sind]
    return rng
end

# Basic relation: iflat - 1 == (iunflat - soffset - 1) * b + soffset´
function flatrange(b::OrbitalBlockStructure{<:SMatrixView}, iunflat::Integer)
    checkinrange(iunflat, b)
    soffset  = 0
    soffset´ = 0
    @inbounds for (s, b) in zip(b.subsizes, b.blocksizes)
        if soffset + s >= iunflat
            offset = muladd(iunflat - soffset - 1, b, soffset´)
            return offset+1:offset+b
        end
        soffset  += s
        soffset´ += b * s
    end
    @boundscheck(blockbounds_error())
    return 1:0
end

flatrange(b::OrbitalBlockStructure{<:SMatrix{N}}, iunflat::Integer) where {N} =
    (checkinrange(iunflat, b); ((iunflat - 1) * N + 1 : iunflat * N))

flatrange(b::OrbitalBlockStructure{<:Number}, iunflat::Integer) =
    (checkinrange(iunflat, b); iunflat:iunflat)

flatrange(o::OrbitalBlockStructure, ::Colon) = orbrange(o)

function flatrange(o::OrbitalBlockStructure, runflat::AbstractUnitRange)
    imin, imax = first(runflat), last(runflat)
    checkinrange(imin, o)
    checkinrange(imax, o)
    orng = first(flatrange(o, imin)) : last(flatrange(o, imax))
    return orng
end

checkinrange(siteind::Integer, b::OrbitalBlockStructure) =
    @boundscheck(1 <= siteind <= flatsize(b) ||
        argerror("Requested site $siteind our of range [1, $(flatsize(b))]"))

flatindex(b::OrbitalBlockStructure, i) = first(flatrange(b, i))

function unflatindex_and_blocksize(o::OrbitalBlockStructure{<:SMatrixView}, iflat::Integer)
    soffset  = 0
    soffset´ = 0
    @boundscheck(iflat < 0 && blockbounds_error())
    @inbounds for (s, b) in zip(o.subsizes, o.blocksizes)
        if soffset´ + b * s >= iflat
            iunflat = (iflat - soffset´ - 1) ÷ b + soffset + 1
            return iunflat, b
        end
        soffset  += s
        soffset´ += b * s
    end
    @boundscheck(blockbounds_error())
end

@noinline blockbounds_error() = throw(BoundsError())

unflatindex_and_blocksize(::OrbitalBlockStructure{B}, iflat::Integer) where {N,B<:SMatrix{N}} =
    (iflat - 1)÷N + 1, N
unflatindex_and_blocksize(::OrbitalBlockStructure{<:Number}, iflat::Integer) = (iflat, 1)

Base.copy(b::OrbitalBlockStructure{B}) where {B} =
    OrbitalBlockStructure{B}(copy(blocksizes(b)), copy(subsizes(b)))

Base.:(==)(b::OrbitalBlockStructure{B}, b´::OrbitalBlockStructure{B}) where {B} =
    b.blocksizes == b´.blocksizes && b.subsizes == b´.subsizes

#endregion
#endregion

############################################################################################
## Special matrices  -  see specialmatrices.jl for methods
#region

  ############################################################################################
  # HybridSparseMatrix
  #    Internal Matrix type for Bloch harmonics in Hamiltonians
  #    Wraps site-block + flat versions of the same SparseMatrixCSC
  #region

struct HybridSparseMatrix{T,B<:MatrixElementType{T}} <: SparseArrays.AbstractSparseMatrixCSC{B,Int}
    blockstruct::OrbitalBlockStructure{B}
    unflat::SparseMatrixCSC{B,Int}
    flat::SparseMatrixCSC{Complex{T},Int}
    # 0 = in sync, 1 = flat needs sync, -1 = unflat needs sync, 2 = none initialized
    sync_state::Base.RefValue{Int}
    # Number of stored nonzeros in unflat and flat - to guard against tampering
    ufnnz::Vector{Int}
end

#region ## Constructors ##

HybridSparseMatrix(bs, unflat, flat, sync_state) =
    HybridSparseMatrix(bs, unflat, flat, sync_state, [nnz(unflat), nnz(flat)])

HybridSparseMatrix(b::OrbitalBlockStructure{Complex{T}}, flat::SparseMatrixCSC{Complex{T},Int}) where {T} =
    HybridSparseMatrix(b, flat, flat, Ref(0))  # aliasing

function HybridSparseMatrix(b::OrbitalBlockStructure{B}, unflat::SparseMatrixCSC{B,Int}) where {T,B<:MatrixElementNonscalarType{T}}
    m = HybridSparseMatrix(b, unflat, flat(b, unflat), Ref(0))
    needs_flat_sync!(m)
    return m
end

function HybridSparseMatrix(b::OrbitalBlockStructure{B}, flat::SparseMatrixCSC{Complex{T},Int}) where {T,B<:MatrixElementNonscalarType{T}}
    m = HybridSparseMatrix(b, unflat(b, flat), flat, Ref(0))
    needs_unflat_sync!(m)
    return m
end

#endregion

#region ## API ##

blockstructure(s::HybridSparseMatrix) = s.blockstruct

unflat_unsafe(s::HybridSparseMatrix) = s.unflat

flat_unsafe(s::HybridSparseMatrix) = s.flat

syncstate(s::HybridSparseMatrix) = s.sync_state

check_integrity(s::HybridSparseMatrix) =
    (nnz(s.unflat) == s.ufnnz[1] && nnz(s.flat) == s.ufnnz[2]) ||
    argerror("The AbstractHamiltonian seems to have been modified externally and has become corrupted")

update_nnz!(s::HybridSparseMatrix) = (s.ufnnz .= (nnz(s.unflat), nnz(s.flat)))

# are flat === unflat? Only for scalar eltype
isaliased(::HybridSparseMatrix{<:Any,<:Complex}) = true
isaliased(::HybridSparseMatrix) = false

Base.size(h::HybridSparseMatrix, i::Integer...) = size(unflat_unsafe(h), i...)

flatsize(h::HybridSparseMatrix, args...) = flatsize(blockstructure(h), args...)

SparseArrays.getcolptr(s::HybridSparseMatrix) = getcolptr(s.unflat)
SparseArrays.rowvals(s::HybridSparseMatrix) = rowvals(s.unflat)
SparseArrays.nonzeros(s::HybridSparseMatrix) = nonzeros(s.unflat)

#endregion
#endregion

  ############################################################################################
  # SparseMatrixView
  #    View of a SparseMatrixCSC that can produce a proper (non-view) SparseMatrixCSC
  #    of size possibly larger than the view (padded with zeros)
  #region

struct SparseMatrixView{C,V<:SubArray}
    matview::V
    mat::SparseMatrixCSC{C,Int}
    ptrs::Vector{Int}           # ptrs of parent(matview) that affect mat
end

#region ## Constructor ##

function SparseMatrixView(matview::SubArray{C,<:Any,<:SparseMatrixCSC}, dims = missing) where {C}
    matparent = parent(matview)
    viewrows, viewcols = matview.indices
    rows = rowvals(matparent)
    ptrs = Int[]
    for col in viewcols, ptr in nzrange(matparent, col)
        rows[ptr] in viewrows && push!(ptrs, ptr)
    end
    if dims === missing || dims == size(matview)
        mat = sparse(matview)
    else
        nr, nc = dims
        nrv, ncv = size(matview)
        nr >= nrv && nc >= ncv ||
            argerror("SparseMatrixView dims cannot be smaller than size(view) = $((nrv, ncv))")
        # it's important to preserve structural zeros in mat, which sparse(matview) does
        mat = [sparse(matview) spzeros(C, nrv, nc - ncv);
               spzeros(C, nr - nrv, ncv) spzeros(C, nr - nrv, nc - ncv)]
    end
    return SparseMatrixView(matview, mat, ptrs)
end

#endregion

#region ## API ##

matrix(s::SparseMatrixView) = s.mat

function update!(s::SparseMatrixView)
    nzs = nonzeros(s.mat)
    nzs´ = nonzeros(parent(s.matview))
    for (i, ptr) in enumerate(s.ptrs)
        nzs[i] = nzs´[ptr]
    end
    return s
end

Base.size(s::SparseMatrixView, i...) = size(s.mat, i...)

minimal_callsafe_copy(s::SparseMatrixView) =
    SparseMatrixView(view(copy(parent(s.matview)), s.matview.indices...), copy(s.mat), s.ptrs)

minimal_callsafe_copy(s::SparseMatrixView, aliasham) =
    SparseMatrixView(view(call!_output(aliasham), s.matview.indices...), copy(s.mat), s.ptrs)


#endregion

#endregion

  ############################################################################################
  # BlockSparseMatrix and BlockMatrix
  #   MatrixBlock : Block within a parent matrix, at a given set of rows and cols
  #   BlockSparseMatrix : SparseMatrixCSC with added blocks that can be updated in place
  #   BlockMatrix : Matrix with added blocks that can be updated in place
  #region

abstract type AbstractBlockMatrix end

struct MatrixBlock{C<:Number,A<:AbstractMatrix,UR,UC,D<:Union{Missing,Matrix}}
    block::A
    rows::UR             # row indices in parent matrix for each row in block
    cols::UC             # col indices in parent matrix for each col in block
    coefficient::C       # coefficient to apply to block
    denseblock::D        # either missing or Matrix(block), to aid in ldiv
end

struct BlockSparseMatrix{C,N,M<:NTuple{N,MatrixBlock}} <: AbstractBlockMatrix
    mat::SparseMatrixCSC{C,Int}
    blocks::M
    ptrs::NTuple{N,Vector{Int}}    # nzvals indices for blocks
end

struct BlockMatrix{C,N,M<:NTuple{N,MatrixBlock}} <: AbstractBlockMatrix
    mat::Matrix{C}
    blocks::M
end

#region ## Constructors ##

function MatrixBlock(block::AbstractMatrix{C}, rows, cols, denseblock = missing) where {C}
    checkblockinds(block, rows, cols)
    return MatrixBlock(block, rows, cols, one(C), denseblock)
end

function MatrixBlock(block::SubArray, rows, cols, denseblock = missing)
    checkblockinds(block, rows, cols)
    m = simplify_matrixblock(block, rows, cols)
    return MatrixBlock(m.block, m.rows, m.cols, m.coefficient, denseblock)
end

BlockSparseMatrix(mblocks::MatrixBlock...) = BlockSparseMatrix(mblocks)

function BlockSparseMatrix(mblocks::NTuple{<:Any,MatrixBlock}, dims = missing)
    blocks = blockmat.(mblocks)
    C = promote_type(eltype.(blocks)...)
    I, J = Int[], Int[]
    foreach(b -> appendIJ!(I, J, b), mblocks)
    mat = dims === missing ? sparse(I, J, zero(C)) : sparse(I, J, zero(C), dims...)
    ptrs = getblockptrs.(mblocks, Ref(mat))
    return BlockSparseMatrix(mat, mblocks, ptrs)
end

function BlockMatrix(mblocks::MatrixBlock...)
    nrows = maxrows(mblocks)
    ncols = maxcols(mblocks)
    C = promote_type(eltype.(blocks)...)
    mat = zeros(C, nrows, ncols)
    return BlockMatrix(mat, blocks)
end

#endregion

#region ## API ##

blockmat(m::MatrixBlock) = m.block

blockrows(m::MatrixBlock) = m.rows

blockcols(m::MatrixBlock) = m.cols

coefficient(m::MatrixBlock) = m.coefficient

denseblockmat(m::MatrixBlock) = m.denseblock

pointers(m::BlockSparseMatrix) = m.ptrs

blocks(m::AbstractBlockMatrix) = m.blocks

matrix(b::AbstractBlockMatrix) = b.mat

maxrows(mblocks::NTuple{<:Any,MatrixBlock}) = maximum(b -> maximum(b.rows), mblocks; init = 0)
maxcols(mblocks::NTuple{<:Any,MatrixBlock}) = maximum(b -> maximum(b.cols), mblocks; init = 0)

Base.size(b::AbstractBlockMatrix, i...) = size(b.mat, i...)
Base.size(b::MatrixBlock, i...) = size(blockmat(b), i...)

Base.eltype(b::MatrixBlock) = eltype(blockmat(b))
Base.eltype(m::AbstractBlockMatrix) = eltype(matrix(m))

Base.:-(b::MatrixBlock) =
    MatrixBlock(blockmat(b), blockrows(b), blockcols(b), -coefficient(b), denseblockmat(b))

@noinline function checkblockinds(block, rows, cols)
    length.((rows, cols)) == size(block) && allunique(rows) &&
        (cols === rows || allunique(cols)) || internalerror("MatrixBlock: mismatched size")
    return nothing
end

isspzeros(b::MatrixBlock) = isspzeros(b.block)
isspzeros(b::SubArray) = isspzeros(parent(b))
isspzeros(b::SparseMatrixCSC) = iszero(nnz(b))
isspzeros(b::Tuple) = all(isspzeros, b)
isspzeros(b) = false

minimal_callsafe_copy(s::BlockSparseMatrix) =
    BlockSparseMatrix(copy(s.mat), minimal_callsafe_copy.(s.blocks), s.ptrs)

minimal_callsafe_copy(s::BlockMatrix) =
    BlockMatrix(copy(s.mat), minimal_callsafe_copy.(s.blocks))

minimal_callsafe_copy(m::MatrixBlock) =
    MatrixBlock(m.block, m.rows, m.cols, m.coefficient, copy_ifnotmissing(m.denseblock))

#endregion
#endregion

  ############################################################################################
  # InverseGreenBlockSparse
  #    BlockSparseMatrix representing G⁻¹ on a unitcell (+ possibly extended sites)
  #    with self-energies as blocks. It knows which indices correspond to which contacts
  #region

struct InverseGreenBlockSparse{C}
    mat::BlockSparseMatrix{C}
    nonextrng::UnitRange{Int}       # range of indices for non-extended sites
    unitcinds::Vector{Vector{Int}}  # orbital indices in parent unitcell of each contact
    unitcindsall::Vector{Int}       # merged, uniqued and sorted unitcinds
    source::Matrix{C}               # preallocation for ldiv! solve
end

#region ## API ##

matrix(s::InverseGreenBlockSparse) = matrix(s.mat)

orbrange(s::InverseGreenBlockSparse) = s.nonextrng

extrange(s::InverseGreenBlockSparse) = last(s.nonextrng + 1):size(mat, 1)

Base.size(s::InverseGreenBlockSparse, I::Integer...) = size(matrix(s), I...)
Base.axes(s::InverseGreenBlockSparse, I::Integer...) = axes(matrix(s), I...)

# updates only ω block and applies all blocks to BlockSparseMatrix
function update!(s::InverseGreenBlockSparse, ω)
    bsm = s.mat
    Imat = blockmat(first(blocks(bsm)))
    Imat.diag .= ω   # Imat should be <: Diagonal
    return update!(bsm)
end

#endregion
#endregion

#endregion top

############################################################################################
# Harmonic  -  see hamiltonian.jl for methods
#region

struct Harmonic{T,L,B}
    dn::SVector{L,Int}
    h::HybridSparseMatrix{T,B}
end

#region ## API ##

dcell(h::Harmonic) = h.dn

matrix(h::Harmonic) = h.h

flat(h::Harmonic) = flat(h.h)

unflat(h::Harmonic) = unflat(h.h)

flat_unsafe(h::Harmonic) = flat_unsafe(h.h)

unflat_unsafe(h::Harmonic) = unflat_unsafe(h.h)

Base.size(h::Harmonic, i...) = size(matrix(h), i...)

Base.isless(h::Harmonic, h´::Harmonic) = sum(abs2, dcell(h)) < sum(abs2, dcell(h´))

Base.zero(h::Harmonic{<:Any,<:Any,B}) where {B} = Harmonic(zero(dcell(h)), zero(matrix(h)))

Base.copy(h::Harmonic) = Harmonic(dcell(h), copy(matrix(h)))

Base.:(==)(h::Harmonic, h´::Harmonic) = h.dn == h´.dn && unflat(h.h) == unflat(h´.h)

Base.iszero(h::Harmonic) = iszero(flat(h))

#endregion
#endregion

############################################################################################
# UnflatInds, HybridInds  - getindex(::AbstractHamiltonian), see hamiltonian.jl for methods
#region

struct UnflatInds{T}
    inds::T
end

struct HybridInds{T}
    inds::T
end

unflat(i) = UnflatInds(i)
unflat(i, is...) = UnflatInds((i, is...))
unflat() = UnflatInds(())

hybrid(i) = HybridInds(i)
hybrid(i, is...) = HybridInds((i, is...))
hybrid() = HybridInds(())

Base.parent(u::UnflatInds) = u.inds
Base.parent(u::HybridInds) = u.inds

#endregion

############################################################################################
# Hamiltonian  -  see hamiltonian.jl for methods
#region

abstract type AbstractHamiltonian{T,E,L,B} end

const AbstractHamiltonian0D{T,E,B} = AbstractHamiltonian{T,E,0,B}
const AbstractHamiltonian1D{T,E,B} = AbstractHamiltonian{T,E,1,B}

struct Hamiltonian{T,E,L,B} <: AbstractHamiltonian{T,E,L,B}
    lattice::Lattice{T,E,L}
    blockstruct::OrbitalBlockStructure{B}
    harmonics::Vector{Harmonic{T,L,B}}
    bloch::HybridSparseMatrix{T,B}
    # Enforce sorted-dns-starting-from-zero invariant onto harmonics
    function Hamiltonian{T,E,L,B}(lattice, blockstruct, harmonics, bloch) where {T,E,L,B}
        n = nsites(lattice)
        all(har -> size(matrix(har)) == (n, n), harmonics) ||
            throw(DimensionMismatch("Harmonic $(size.(matrix.(harmonics), 1)) sizes don't match number of sites $n"))
        sort!(harmonics)
        (isempty(harmonics) || !iszero(dcell(first(harmonics)))) && pushfirst!(harmonics,
            Harmonic(zero(SVector{L,Int}), HybridSparseMatrix(blockstruct, spzeros(B, n, n))))
        return new(lattice, blockstruct, harmonics, bloch)
    end
end

#region ## API ##

## AbstractHamiltonian

latdim(h::AbstractHamiltonian) = latdim(lattice(h))

bravais_matrix(h::AbstractHamiltonian) = bravais_matrix(lattice(h))

norbitals(h::AbstractHamiltonian) = blocksizes(blockstructure(h))

blockeltype(::AbstractHamiltonian) = blockeltype(blockstructure(h))

blocktype(h::AbstractHamiltonian) = blocktype(blockstructure(h))

nsites(h::AbstractHamiltonian) = nsites(lattice(h))

flatsize(h::AbstractHamiltonian, args...) = flatsize(blockstructure(h), args...)

flatrange(h::AbstractHamiltonian, iunflat) = flatrange(blockstructure(h), iunflat)

flatrange(h::AbstractHamiltonian, name::Symbol) =
    sublatorbrange(blockstructure(h), sublatindex(lattice(h), name))

zerocell(h::AbstractHamiltonian) = zerocell(lattice(h))
zerocellsites(h::AbstractHamiltonian, i) = zerocellsites(lattice(h), i)

# OpenHamiltonian is not <: AbstractHamiltonian
ncontacts(h::AbstractHamiltonian) = 0

function harmonic_index(h::AbstractHamiltonian, dn)
    for (i, har) in enumerate(harmonics(h))
        dcell(har) == dn && return har, i
    end
    boundserror(harmonics(h), dn)
    return first(harmonics(h)), 1  # unreachable
end

# Unless params are given, it returns the Hamiltonian with defaults parameters
default_hamiltonian(h::AbstractHamiltonian; params...) = h(; params...)

## Hamiltonian

Hamiltonian(l::Lattice{T,E,L}, b::OrbitalBlockStructure{B}, h::Vector{Harmonic{T,L,B}}, bl) where {T,E,L,B} =
    Hamiltonian{T,E,L,B}(l, b, h, bl)

function Hamiltonian(l, b::OrbitalBlockStructure{B}, h) where {B}
    n = nsites(l)
    bloch = HybridSparseMatrix(b, spzeros(B, n, n))
    needs_initialization!(bloch)
    return Hamiltonian(l, b, h, bloch)
end

hamiltonian(h::Hamiltonian) = h

blockstructure(h::Hamiltonian) = h.blockstruct

lattice(h::Hamiltonian) = h.lattice

harmonics(h::Hamiltonian) = h.harmonics

bloch(h::Hamiltonian) = h.bloch

minimal_callsafe_copy(h::Hamiltonian) = Hamiltonian(
    lattice(h), blockstructure(h), copy.(harmonics(h)), copy_matrices(bloch(h)))

function flat_sync!(h::Hamiltonian)
    for har in harmonics(h)
        harmat = matrix(har)
        needs_flat_sync(harmat) && flat_sync!(harmat)
    end
    return h
end

Base.size(h::Hamiltonian, i...) = size(bloch(h), i...)
Base.axes(h::Hamiltonian, i...) = axes(bloch(h), i...)
Base.iszero(h::Hamiltonian) = all(iszero, harmonics(h))

Base.copy(h::Hamiltonian) = Hamiltonian(
    copy(lattice(h)), copy(blockstructure(h)), copy.(harmonics(h)), copy(bloch(h)))

copy_lattice(h::Hamiltonian) = Hamiltonian(
    copy(lattice(h)), blockstructure(h), harmonics(h), bloch(h))

function LinearAlgebra.ishermitian(h::Hamiltonian)
    for hh in h.harmonics
        isassigned(h, -hh.dn) || return false
        flat(hh.h) ≈ h[-hh.dn]' || return false
    end
    return true
end

function Base.:(==)(h::Hamiltonian, h´::Hamiltonian)
    hs = sort(h.harmonics, by = har -> har.dn)
    hs´ = sort(h.harmonics, by = har -> har.dn)
    equalharmonics = length(hs) == length(hs´) && all(splat(==), zip(hs, hs´))
    return h.lattice == h´.lattice && h.blockstruct == h´.blockstruct && equalharmonics
end

#endregion
#endregion

############################################################################################
# ParametricHamiltonian  -  see hamiltonian.jl for methods
#region

struct ParametricHamiltonian{T,E,L,B,M<:NTuple{<:Any,AppliedModifier}} <: AbstractHamiltonian{T,E,L,B}
    hparent::Hamiltonian{T,E,L,B}
    h::Hamiltonian{T,E,L,B}        # To be modified upon application of parameters
    modifiers::M                   # Tuple of AppliedModifier's. Cannot FunctionWrapper them
                                   # because they involve kwargs
    allptrs::Vector{Vector{Int}}   # allptrs are all modified ptrs in each harmonic (needed for reset!)
    allparams::Vector{Symbol}
end

#region ## API ##

# Note: this gives the *modified* hamiltonian, not parent
hamiltonian(h::ParametricHamiltonian) = h.h

bloch(h::ParametricHamiltonian) = h.h.bloch

parameters(h::ParametricHamiltonian) = h.allparams

modifiers(h::ParametricHamiltonian) = h.modifiers
modifiers(h::Hamiltonian) = ()

pointers(h::ParametricHamiltonian) = h.allptrs

# refers to hparent [not h.h, which is only used as the return of call!(ph, ω; ...)]
harmonics(h::ParametricHamiltonian) = harmonics(h.hparent)

blockstructure(h::ParametricHamiltonian) = blockstructure(parent(h))

blocktype(h::ParametricHamiltonian) = blocktype(parent(h))

lattice(h::ParametricHamiltonian) = lattice(hamiltonian(h))

minimal_callsafe_copy(p::ParametricHamiltonian) = ParametricHamiltonian(
    p.hparent, minimal_callsafe_copy(p.h), p.modifiers, p.allptrs, p.allparams)

Base.parent(h::ParametricHamiltonian) = h.hparent

Base.size(h::ParametricHamiltonian, i...) = size(parent(h), i...)

Base.copy(p::ParametricHamiltonian) = ParametricHamiltonian(
    copy(p.hparent), copy(p.h), p.modifiers, copy.(p.allptrs), copy(p.allparams))

LinearAlgebra.ishermitian(h::ParametricHamiltonian) =
    argerror("`ishermitian(::ParametricHamiltonian)` not supported, as the result can depend on the values of parameters.")

copy_lattice(p::ParametricHamiltonian) = ParametricHamiltonian(
    copy_lattice(p.hparent), p.h, p.modifiers, p.allptrs, p.allparams)

#endregion
#endregion

############################################################################################
# Mesh  -  see mesh.jl for methods
#region

abstract type AbstractMesh{V,S} end

struct Mesh{V,S} <: AbstractMesh{V,S}    # S-1 is the manifold dimension
    verts::Vector{V}
    neighs::Vector{Vector{Int}}          # all neighbors neighs[i][j] of vertex i
    simps::Vector{NTuple{S,Int}}         # list of simplices, each a group of S neighboring
                                         # vertex indices
end


#region ## Constructors ##

function Mesh{S}(verts, neighs) where {S}
    simps  = build_cliques(neighs, Val(S))
    return Mesh(verts, neighs, simps)
end

#endregion

#region ## API ##

dim(::AbstractMesh{<:Any,S}) where {S} = S - 1

coordinates(s::AbstractMesh) = (coordinates(v) for v in vertices(s))
coordinates(s::AbstractMesh, i::Int) = coordinates(vertices(s, i))

nvertices(m::Mesh) = length(m.verts)

vertices(m::Mesh) = m.verts
vertices(m::Mesh, i) = m.verts[i]

neighbors(m::Mesh) = m.neighs
neighbors(m::Mesh, i::Int) = m.neighs[i]

neighbors_forward(m::Mesh, i::Int) = Iterators.filter(>(i), m.neighs[i])
neighbors_forward(v::Vector, i::Int) = Iterators.filter(>(i), v[i])

simplices(m::Mesh) = m.simps
simplices(m::Mesh, i::Int) = m.simps[i]

Base.copy(m::Mesh) = Mesh(copy(m.verts), copy.(m.neighs), copy(m.simps))

#endregion
#endregion

############################################################################################
# Spectrum and Bandstructure - see solvers/eigensolvers.jl for solver backends <: AbstractEigenSolver
#                    -  see bands.jl for methods
# AppliedEigenSolver - wraps a solver vs ϕs, with a mapping and transform, into a FunctionWrapper
#region

abstract type AbstractEigenSolver end

const EigenComplex{T} = Eigen{Complex{T},Complex{T},Matrix{Complex{T}},Vector{Complex{T}}}

const MatrixView{C} = SubArray{C,2,Matrix{C},Tuple{Base.Slice{Base.OneTo{Int}}, UnitRange{Int}}, true}

struct Spectrum{T<:AbstractFloat,B}
    eigen::EigenComplex{T}
    blockstruct::OrbitalBlockStructure{B}
end

struct AppliedEigenSolver{T<:AbstractFloat,L}
    solver::FunctionWrapper{EigenComplex{T},Tuple{SVector{L,T}}}
end

struct BandVertex{T<:AbstractFloat,E}
    coordinates::SVector{E,Complex{T}}       # SVector(momentum..., energy)
    states::MatrixView{Complex{T}}
end

# Subband: an AbstractMesh with manifold_dimension (S-1) = embedding_dimension (E) - 1
# and with interval search trees to allow slicing
# CAUTION: "embedding" dimension E here refers to the Mesh object (unrelated to Lattice's E)
#   unlike a Subband, a general Mesh can have S≠E, like a 1D curve (S=2) in  3D (E=3) space.
struct Subband{T,E} <: AbstractMesh{BandVertex{T,E},E}  # we restrict S == E
    mesh::Mesh{BandVertex{T,E},E}
    trees::NTuple{E,IntervalTree{T,IntervalValue{T,Int}}} # for interval searches
    projstates::Dict{Tuple{Int,Int},Matrix{Complex{T}}} # (simpind, vind) => projected_states
end

struct Bandstructure{T,E,L,B} # E = L+1
    subbands::Vector{Subband{T,E}}
    solvers::Vector{AppliedEigenSolver{T,L}}  # one per Julia thread
    blockstruct::OrbitalBlockStructure{B}
end

#region ## Constructors ##

Spectrum(eigen::Eigen, h::AbstractHamiltonian) = Spectrum(eigen, blockstructure(h))
Spectrum(eigen::Eigen, h, ::Missing) = Spectrum(eigen, h)

function Spectrum(ss::Vector{Subband{<:Any,1}}, os::OrbitalBlockStructure)
    ϵs = [energy(only(vertices(s))) for s in ss]
    ψs = stack(hcat, (states(only(vertices(s))) for s in ss))
    eigen = Eigen(ϵs, ψs)
    return Spectrum(eigen, os)
end

BandVertex(ke::SVector{N}, s::MatrixView{Complex{T}}) where {N,T} =
    BandVertex(SVector{N,Complex{T}}(ke), s)
BandVertex(ke, s::Matrix) = BandVertex(ke, view(s, :, 1:size(s, 2)))
BandVertex(k, e, s::Matrix) = BandVertex(k, e, view(s, :, 1:size(s, 2)))
BandVertex(k, e, s::SubArray) = BandVertex(vcat(k, e), s)

Subband(verts::Vector{<:BandVertex{<:Any,E}}, neighs::Vector) where {E} =
    Subband(Mesh{E}(verts, neighs))

function Subband(mesh::Mesh)
    verts, simps = vertices(mesh), simplices(mesh)
    orient_simplices!(simps, verts)                             # see mesh.jl
    trees = subband_trees(verts, simps)
    return Subband(mesh, trees)
end

function Subband(mesh::Mesh{<:BandVertex{T}}, trees) where {T}
    projs = Dict{Tuple{Int,Int},Matrix{Complex{T}}}()
    return Subband(mesh, trees, projs)
end

function subband_trees(verts::Vector{BandVertex{T,E}}, simps) where {T,E}
    trees = ntuple(Val(E)) do i
        list = [IntervalValue(shrinkright(extrema(j->coordinates(verts[j])[i], s))..., n)
                     for (n, s) in enumerate(simps)]
        sort!(list)
        return IntervalTree{T,IntervalValue{T,Int}}(list)
    end
    return trees
end

# Interval is closed, we want semiclosed on the left -> exclude the upper limit
shrinkright((x, y)) = (x, prevfloat(y))

#endregion

#region ## API ##

(s::AppliedEigenSolver{T,0})() where {T} = s.solver(SVector{0,T}())
(s::AppliedEigenSolver{T,L})(φs::SVector{L}) where {T,L} = s.solver(sanitize_SVector(SVector{L,T}, φs))
(s::AppliedEigenSolver{T,L})(φs::NTuple{L,Any}) where {T,L} = s.solver(sanitize_SVector(SVector{L,T}, φs))
(s::AppliedEigenSolver{T,L})(φs...) where {T,L} =
    throw(ArgumentError("AppliedEigenSolver call requires $L parameters/Bloch phases, received $φs"))

energies(s::Spectrum) = s.eigen.values

states(s::Spectrum) = s.eigen.vectors

blockstructure(s::Spectrum) = s.blockstruct
blockstructure(b::Bandstructure) = b.blockstruct

solvers(b::Bandstructure) = b.solvers

subbands(b::Bandstructure) = b.subbands

subbands(b::Bandstructure, i...) = getindex(b.subbands, i...)

nsubbands(b::Bandstructure) = nsubbands(subbands(b))
nsubbands(b::Vector{<:Subband}) = length(b)

nvertices(b::Bandstructure) = nvertices(subbands(b))
nvertices(b::Vector{<:Subband}) = sum(s->length(vertices(s)), b; init = 0)

nedges(b::Bandstructure) = nedges(subbands(b))
nedges(b::Vector{<:Subband}) = sum(s -> sum(length, neighbors(s)), b; init = 0) ÷ 2

nsimplices(b::Bandstructure) = nsimplices(subbands(b))
nsimplices(b::Vector{<:Subband}) = sum(s->length(simplices(s)), b)

# vertex coordinates can be complex, interally, but always appear real through the API
coordinates(s::SVector) = real(s)
coordinates(v::BandVertex) = real(v.coordinates)

energy(v::BandVertex) = last(v.coordinates)

base_coordinates(v::BandVertex) = SVector(Base.front(Tuple(coordinates(v))))

states(v::BandVertex) = v.states

degeneracy(v::BandVertex) = size(v.states, 2)

parentrows(v::BandVertex) = first(parentindices(v.states))
parentcols(v::BandVertex) = last(parentindices(v.states))

vertices(s::Subband, i...) = vertices(s.mesh, i...)

neighbors(s::Subband, i...) = neighbors(s.mesh, i...)

neighbors_forward(s::Subband, i) = neighbors_forward(s.mesh, i)

simplices(s::Subband, i...) = simplices(s.mesh, i...)

energy(s::Subband, vind::Int) = energy(vertices(s, vind))

energies(s::Subband, vinds::NTuple{D´,Int}) where {D´} =
    SVector{D´}(energy.(Ref(s), vinds))

base_coordinates(s::Subband, vind::Int) = base_coordinates(vertices(s, vind))
base_coordinates(s::Subband, vinds::NTuple{D,Int}) where {D} =
    reduce(hcat, base_coordinates.(Ref(s), vinds))

trees(s::Subband) = s.trees
trees(s::Subband, i::Int) = s.trees[i]

projected_states(s::Subband) = s.projstates

# last argument: saxes = ((dim₁, x₁), (dim₂, x₂)...)
function foreach_simplex(f, s::Subband, ((dim, k), xs...))
    for interval in intersect(trees(s, dim), (k, k))
        interval_in_slice!(interval, s, xs...) || continue
        sind = value(interval)
        f(sind)
    end
    return nothing
end

interval_in_slice!(interval, s, (dim, k), xs...) =
    interval in intersect(trees(s, dim), (k, k)) && interval_in_slice!(interval, s, xs...)
interval_in_slice!(interval, s) = true

mesh(s::Subband) = s.mesh
mesh(m::Mesh) = m

meshes(b::Bandstructure) = (mesh(s) for s in subbands(b))
meshes(s::Subband) = (mesh(s),)
meshes(s::Mesh) = (s,)
meshes(xs) = (mesh(x) for x in xs)

embdim(::AbstractMesh{<:SVector{E}}) where {E} = E

embdim(::AbstractMesh{<:BandVertex{<:Any,E}}) where {E} = E

meshdim(::AbstractMesh{<:Any,S}) where {S} = S

dims(m::AbstractMesh) = (embdim(m), meshdim(m))

Base.size(s::Spectrum, i...) = size(s.eigen.vectors, i...)

Base.isempty(s::Subband) = isempty(simplices(s))

Base.length(b::Bandstructure) = length(b.subbands)

#endregion
#endregion

############################################################################################
# SelfEnergySolvers - see solvers/selfenergy.jl for self-energy solvers
#region

abstract type AbstractSelfEnergySolver end
abstract type RegularSelfEnergySolver <: AbstractSelfEnergySolver end
abstract type ExtendedSelfEnergySolver <: AbstractSelfEnergySolver end

#endregion

############################################################################################
# Green solvers - see solvers/greensolvers.jl
#region

# Generic system-independent directives for solvers, e.g. GS.Schur()
abstract type AbstractGreenSolver end

# Application to a given OpenHamiltonian, but still independent from (ω; params...)
# It should support call!(::AppliedGreenSolver, ω; params...) -> GreenSolution
abstract type AppliedGreenSolver end

# Solver with fixed ω and params that can compute G[subcell, subcell´] or G[cell, cell´]
# It should also be able to return contact G with G[]
abstract type GreenSlicer{C<:Complex} end   # C is the eltype of the slice

#endregion

############################################################################################
# SelfEnergy - see solvers/selfenergy.jl
#   Wraps an AbstractSelfEnergySolver and a SiteSlice
#     -It produces Σs(ω)::AbstractMatrix defined over a SiteSlice
#     -If solver::ExtendedSelfEnergySolver -> 3 AbstractMatrix blocks over latslice+extended
#   AbstractSelfEnergySolvers can be associated with methods of attach(h, sargs...; kw...)
#   To associate such a method we add a SelfEnergy constructor that will be used by attach
#     - SelfEnergy(h::AbstractHamiltonian, sargs...; kw...) -> SelfEnergy
#region

struct SelfEnergy{T,E,L,S<:AbstractSelfEnergySolver}
    solver::S                                # returns AbstractMatrix block(s) over latslice
    orbslice::OrbitalSliceGrouped{T,E,L}     # sites on each unitcell with a selfenergy
    function SelfEnergy{T,E,L,S}(solver, orbslice) where {T,E,L,S<:AbstractSelfEnergySolver}
        isempty(orbslice) && argerror("Cannot create a self-energy over an empty LatticeSlice")
        return new(solver, orbslice)
    end
end

#region ## Constructors ##

SelfEnergy(solver::S, orbslice::OrbitalSliceGrouped{T,E,L}) where {T,E,L,S<:AbstractSelfEnergySolver} =
    SelfEnergy{T,E,L,S}(solver, orbslice)

#endregion

#region ## API ##

orbslice(Σ::SelfEnergy) = Σ.orbslice

solver(Σ::SelfEnergy) = Σ.solver

has_selfenergy(s::SelfEnergy) = has_selfenergy(solver(s))
has_selfenergy(s::AbstractSelfEnergySolver) = true
# see nothing.jl for override for the case of SelfEnergyEmptySolver

call!(Σ::SelfEnergy, ω; params...) = call!(Σ.solver, ω; params...)

call!_output(Σ::SelfEnergy) = call!_output(solver(Σ))

(Σ::SelfEnergy)(; params...) = SelfEnergy(Σ.solver(; params...), Σ.orbslice)

minimal_callsafe_copy(Σ::SelfEnergy) =
    SelfEnergy(minimal_callsafe_copy(Σ.solver), Σ.orbslice)

#endregion
#endregion

############################################################################################
# OpenHamiltonian
#    A collector of selfenergies `attach`ed to an AbstractHamiltonian
#region

struct OpenHamiltonian{T,E,L,H<:AbstractHamiltonian{T,E,L},S<:NTuple{<:Any,SelfEnergy}}
    h::H
    selfenergies::S
end

#region ## Constructors ##

OpenHamiltonian(h::AbstractHamiltonian) = OpenHamiltonian(h, ())

#endregion

#region ## API ##

selfenergies(oh::OpenHamiltonian) = oh.selfenergies

hamiltonian(oh::OpenHamiltonian) = oh.h

default_hamiltonian(oh::OpenHamiltonian) = default_hamiltonian(oh.h)

lattice(oh::OpenHamiltonian) = lattice(oh.h)

zerocell(h::OpenHamiltonian) = zerocell(parent(h))

ncontacts(h::OpenHamiltonian) = length(selfenergies(h))

attach(Σ::SelfEnergy) = oh -> attach(oh, Σ)
attach(args...; kw...) = oh -> attach(oh, args...; kw...)
attach(oh::OpenHamiltonian, args...; kw...) = attach(oh, SelfEnergy(oh.h, args...; kw...))
attach(oh::OpenHamiltonian, Σ::SelfEnergy) = OpenHamiltonian(oh.h, (oh.selfenergies..., Σ))
attach(h::AbstractHamiltonian, args...; kw...) = attach(h, SelfEnergy(h, args...; kw...))
attach(h::AbstractHamiltonian, Σ::SelfEnergy) = OpenHamiltonian(h, (Σ,))

# fallback for SelfEnergy constructor
SelfEnergy(h::AbstractHamiltonian, args...; kw...) = argerror("Unknown attach/SelfEnergy systax")

minimal_callsafe_copy(oh::OpenHamiltonian) =
    OpenHamiltonian(minimal_callsafe_copy(oh.h), minimal_callsafe_copy.(oh.selfenergies))

Base.size(oh::OpenHamiltonian, i...) = size(oh.h, i...)

Base.parent(oh::OpenHamiltonian) = oh.h

boundingbox(oh::OpenHamiltonian) =
    boundingbox(tupleflatten(boundingbox.(orbslice.(selfenergies(oh)))...))

#endregion
#endregion

############################################################################################
# Contacts - see greenfunction.jl
#    Collection of selfenergies supplemented with a ContactOrbitals
#    ContactOrbitals includes orbslice = flat merged Σlatslices + block info
#    Supports call!(c, ω; params...) -> (Σs::MatrixBlock...) over orbslice
#region

struct ContactOrbitals{L}
    orbsdict::CellOrbitalsGroupedDict{L}    # non-extended orbital indices for merged contact
    corbsdict::Vector{CellOrbitalsGroupedDict{L}}  # same for each contact (alias of Σ's)
    contactinds::Vector{Vector{Int}}        # orbital indices in orbdict for each contact
    offsets::Vector{Int}                    # orbsdict offset from number of orbs per cell
end

struct Contacts{L,N,S<:NTuple{N,SelfEnergy},O<:OrbitalSliceGrouped}
    selfenergies::S                         # one per contact, produce flat AbstractMatrices
    orbitals::ContactOrbitals{L}            # needed to extract site/subcell/contact blocks
    orbslice::O
end

const EmptyContacts{L} = Contacts{L,0,Tuple{}}

#region ## Constructors ##

ContactOrbitals{L}() where {L} =
    ContactOrbitals(CellOrbitalsGroupedDict{L}(), CellOrbitalsGroupedDict{L}[], Vector{Int}[], Int[])

function Contacts(oh::OpenHamiltonian)
    Σs = selfenergies(oh)
    Σorbslices = orbslice.(Σs)
    h = hamiltonian(oh)
    orbitals = ContactOrbitals(h, Σorbslices...)  # see greenfunction.jl for constructor
    oslice = OrbitalSliceGrouped(lattice(oh), cellsdict(orbitals))
    return Contacts(Σs, orbitals, oslice)
end

#endregion

#region ## API ##

cellsdict(c::ContactOrbitals) = c.orbsdict
cellsdict(c::ContactOrbitals, i::Integer) = c.corbsdict[i]

contactinds(c::ContactOrbitals) = c.contactinds
contactinds(c::ContactOrbitals, i) = c.contactinds[i]

ncontacts(c::Contacts) = ncontacts(c.orbitals)
ncontacts(c::ContactOrbitals) = length(c.corbsdict)

offsets(c::Contacts) = offsets(c.orbitals)
offsets(c::ContactOrbitals) = c.offsets

selfenergies(c::Contacts) = c.selfenergies
selfenergies(c::Contacts, i::Integer) = check_contact_index(i, c) && c.selfenergies[i]

has_selfenergy(c::Contacts) = any(has_selfenergy, selfenergies(c))

# c::Union{Contacts,ContactOrbitals} here
check_contact_index(i, c) = 1 <= i <= ncontacts(c) ||
    argerror("Cannot get contact $i, there are $(ncontacts(c)) contacts")

# for checks in contact construction
check_contact_slice(s::LatticeSlice) =
    isempty(s) && argerror("No contact sites found in selection")

contactorbitals(c::Contacts) = c.orbitals

function contact_orbslice(h; sites...)
    contactslice = getindex(lattice(h); sites...)
    check_contact_slice(contactslice)  # in case it is empty
    lsparent = sites_to_orbs(contactslice, h)
    return lsparent
end

orbslice(c::Contacts) = c.orbslice
orbslice(c::Contacts, ::Colon) = c.orbslice
orbslice(c::Contacts, i::Integer) = orbslice(selfenergies(c, i))

cellsdict(c::Contacts, i...) = cellsdict(c.orbitals, i...)

contactinds(c::Contacts, i...) = contactinds(c.orbitals, i...)

norbitals(c::ContactOrbitals) = norbitals(c.orbsdict)
norbitals(c::ContactOrbitals, i) = norbitals(c.corbsdict[i])

orbgroups(c::ContactOrbitals) = orbgroups(c.orbsdict)
orbgroups(c::ContactOrbitals, i) = orbgroups(c.corbsdict[i])

orbranges(c::ContactOrbitals) = orbranges(c.orbsdict)
orbranges(c::ContactOrbitals, i) = orbranges(c.corbsdict[i])

boundingbox(c::Contacts) = boundingbox(orbslice(c))

Base.isempty(c::Contacts) = isempty(selfenergies(c))

# Base.length(c::Contacts) = length(selfenergies(c)) # unused

minimal_callsafe_copy(c::Contacts) =
    Contacts(minimal_callsafe_copy.(c.selfenergies), c.orbitals, c.orbslice)

#endregion

#endregion

############################################################################################
# Green - see greenfunction.jl and solvers/green
#  General strategy:
#  -Contacts: Σs::Tuple(Selfenergy...) + contactBS -> call! produces MatrixBlocks for Σs
#      -SelfEnergy: latslice + solver -> call! produces flat matrices (one or three)
#  -GreenFunction: ham + c::Contacts + an AppliedGreenSolver = apply(GS.AbsSolver, ham, c)
#      -call!(::GreenFunction, ω; params...) -> call! ham + Contacts, returns GreenSolution
#      -AppliedGreenSolver: usually wraps the SelfEnergy flat matrices. call! -> GreenSlicer
#  -GreenSolution: ham, slicer, Σblocks, contactBS
#      -GreenSlicer: implements getindex to build g(ω)[rows, cols]
#region

struct GreenFunction{T,E,L,S<:AppliedGreenSolver,H<:AbstractHamiltonian{T,E,L},C<:Contacts}
    parent::H
    solver::S
    contacts::C
end

# Obtained with gω = call!(g::GreenFunction, ω; params...) or g(ω; params...)
# Allows gω[i, j] for i,j integer Σs indices ("contacts")
# Allows gω[cell, cell´] using T-matrix, with cell::Union{SVector,CellSites}
# Allows also view(gω, ...)
struct GreenSolution{T,E,L,S<:GreenSlicer,G<:GreenFunction{T,E,L},Σs}
    parent::G
    slicer::S       # gives G(ω; p...)[i,j] for i,j::AppliedGreenIndex
    contactΣs::Σs   # Tuple of selfenergies Σ(ω)::MatrixBlock or NTuple{3,MatrixBlock}, one per contact
    contactorbs::ContactOrbitals{L}
end

struct DiagIndices{K,V}   # represents Green indices to return only diagonal elements
    inds::V
    kernel::K
end

struct GreenIndices{I,R}
    inds::I
    orbinds::R  # orbinds = sites_to_orbs(inds)
end

# Obtained with gs = g[; siteselection...]
# Alows call!(gs, ω; params...) or gs(ω; params...)
#   required to do e.g. h |> attach(g´[sites´], couplingmodel; sites...)
struct GreenSlice{T,E,L,G<:GreenFunction{T,E,L},R<:GreenIndices,C<:GreenIndices}
    parent::G
    rows::R
    cols::C
end

#region ## Constuctors ##

function GreenSlice(parent, rows, cols)
    if rows isa DiagIndices || cols isa DiagIndices
        rows === cols || argerror("Diagonal indices should be identical for rows and columns")
    end
    rows´ = greenindices(rows, parent)
    cols´ = cols === rows ? rows´ : greenindices(cols, parent)
    return GreenSlice(parent, rows´, cols´)
end

# see sites_to_orbs in slices.jl
greenindices(inds, g) = GreenIndices(inds, sites_to_orbs(inds, g))
greenindices(g::GreenSlice) = g.rows, g.cols

#endregion

#region ## API ##

diagonal(inds; kernel = missing) = DiagIndices(inds, kernel)                      # exported
diagonal(; kernel = missing, kw...) = DiagIndices(siteselector(; kw...), kernel)  # exported

Base.parent(i::DiagIndices) = i.inds

kernel(i::DiagIndices) = i.kernel

# returns the Hamiltonian field
hamiltonian(g::Union{GreenFunction,GreenSolution,GreenSlice}) = hamiltonian(g.parent)

# Like the above, but it may not be === the field (it can be a copy with parameters applied)
# needed for qplot(g(; params...))
default_hamiltonian(g::GreenFunction) = default_hamiltonian(parent(g))  # default params

lattice(g::Union{GreenFunction,GreenSolution,GreenSlice}) = lattice(g.parent)

latslice(g::GreenFunction, i) = orbslice(g.contacts, i)

# function latslice(g::GreenFunction, ls::LatticeSlice)
#     lattice(g) === lattice(ls) || internalerror("latslice: parent lattice mismatch")
#     return ls
# end
# latslice(g::GreenFunction, is::SiteSelector) = lattice(g)[is]
# latslice(g::GreenFunction; kw...) = latslice(g, siteselector(; kw...))

zerocell(g::Union{GreenFunction,GreenSolution,GreenSlice}) = zerocell(lattice(g))

solver(g::GreenFunction) = g.solver

contacts(g::GreenFunction) = g.contacts
contacts(g::Union{GreenSolution,GreenSlice}) = contacts(parent(g))

ncontacts(g::GreenFunction) = ncontacts(g.contacts)
ncontacts(g::Union{GreenSolution,GreenSlice}) = ncontacts(parent(g))

slicer(g::GreenSolution) = g.slicer

selfenergies(g::GreenFunction) = selfenergies(contacts(g))
selfenergies(g::GreenSolution) = g.contactΣs

has_selfenergy(g::Union{GreenFunction,GreenSlice,GreenSolution}) =
    has_selfenergy(contacts(g))

contactorbitals(g::GreenFunction) = contactorbitals(g.contacts)
contactorbitals(g::GreenSolution) = g.contactorbs
contactorbitals(g::GreenSlice) = contactorbitals(parent(g))

blockstructure(g::GreenFunction) = blockstructure(hamiltonian(g))
blockstructure(g::GreenSolution) = blockstructure(hamiltonian(g))
blockstructure(g::GreenSlice) = blockstructure(parent(g))

norbitals(g::GreenFunction) = norbitals(g.parent)
norbitals(g::GreenSlice) = norbitals(g.parent.parent)

contactinds(g::GreenFunction, i...) = contactinds(contacts(g), i...)
contactinds(g::Union{GreenSolution,GreenSlice}, i...) = contactinds(contactorbitals(g), i...)

greenfunction(g::GreenSlice) = g.parent

rows(g::GreenSlice) = g.rows.inds

cols(g::GreenSlice) = g.cols.inds

orbrows(g::GreenSlice) = g.rows.orbinds

orbcols(g::GreenSlice) = g.cols.orbinds

Base.axes(g::GreenSlice) = (orbrows(g), orbcols(g))

# ifelse(rows && cols are contacts, (rows, cols), (orbrows, orbcols))
# I.e: if rows, cols are contact indices retrieve them instead of orbslices.
orbinds_or_contactinds(g) = orbinds_or_contactinds(rows(g), cols(g), orbrows(g), orbcols(g))
orbinds_or_contactinds(r::Union{Colon,Integer}, c::Union{Colon,Integer}, _, _) = (r, c)
orbinds_or_contactinds(_, _, or, oc) = (or, oc)

Base.parent(g::GreenFunction) = g.parent
Base.parent(g::GreenSolution) = g.parent
Base.parent(g::GreenSlice) = g.parent

Base.size(g::GreenFunction, i...) = size(g.parent, i...)
Base.size(g::GreenSolution, i...) = size(g.parent, i...)

flatsize(g::GreenFunction, i...) = flatsize(g.parent, i...)
flatsize(g::GreenSolution, i...) = flatsize(g.parent, i...)

function similar_Matrix(gs::GreenSlice{T}) where {T}
    m = norbitals(orbrows(gs))
    n = norbitals(orbcols(gs))
    return Matrix{Complex{T}}(undef, m, n)
end

boundaries(g::GreenFunction) = boundaries(solver(g))
# fallback (for solvers without boundaries, or for OpenHamiltonian)
boundaries(_) = ()

boundingbox(g::GreenFunction) = isempty(contacts(g)) ?
    (zerocell(lattice(g)), zerocell(lattice(g))) : boundingbox(contacts(g))

copy_lattice(g::GreenFunction) = GreenFunction(copy_lattice(g.parent), g.solver, g.contacts)
copy_lattice(g::GreenSolution) = GreenSolution(
    copy_lattice(g.parent), g.slicer, g.contactΣs, g.contactbs)
copy_lattice(g::GreenSlice) = GreenSlice(
    copy_lattice(g.parent), g.rows, g.cols)

function minimal_callsafe_copy(g::GreenFunction)
    parent´ = minimal_callsafe_copy(g.parent)
    contacts´ = minimal_callsafe_copy(g.contacts)
    solver´ = minimal_callsafe_copy(g.solver, parent´, contacts´)
    return GreenFunction(parent´, solver´, contacts´)
end

function minimal_callsafe_copy(g::GreenSolution)
    parentg´ = minimal_callsafe_copy(g.parent)
    parentham = hamiltonian(parentg´)
    parentcontacts = contacts(parentg´)
    slicer´ = minimal_callsafe_copy(g.slicer, parentham, parentcontacts)
    g´ = GreenSolution(parentg´, slicer´, g.contactΣs, g.contactorbs)
    return g´
end

minimal_callsafe_copy(g::GreenSlice) =
    GreenSlice(minimal_callsafe_copy(g.parent), g.rows, g.cols)

Base.:(==)(g::GreenFunction, g´::GreenFunction) = function_not_defined("==")
Base.:(==)(g::GreenSolution, g´::GreenSolution) = function_not_defined("==")
Base.:(==)(g::GreenSlice, g´::GreenSlice) = function_not_defined("==")

#endregion
#endregion

############################################################################################
# Operator - Hamiltonian-like operator representing observables other than a Hamiltonian
#   It works as a wrapper of an AbstractHamiltonian, see observables.jl for constructors
# VectorOperator - like the above for a collection of AbstractHamiltonians
# BarebonesOperator - same thing but with arbitrary element type and no support for call!
#region

struct Operator{H<:AbstractHamiltonian}
    h::H
end

struct BarebonesHarmonic{L,B}
    dn::SVector{L,Int}
    mat::SparseMatrixCSC{B,Int}
end

struct BarebonesOperator{L,B}
    harmonics::Dictionary{SVector{L,Int},BarebonesHarmonic{L,B}}
end

#region ## Constructors ##

BarebonesOperator(harmonics::Vector) =
    BarebonesOperator(index(dcell, BarebonesHarmonic.(harmonics)))

BarebonesHarmonic(har) = BarebonesHarmonic(dcell(har), sparse(har))

#endregion

#region ## API ##

hamiltonian(o::Operator) = o.h

harmonics(o::BarebonesOperator) = o.harmonics

matrix(h::BarebonesHarmonic) = h.mat

dcell(h::BarebonesHarmonic) = h.dn

(o::Operator)(φ...; kw...) = o.h(φ...; kw...)

call!(o::Operator, φ...; kw...) = call!(o.h, φ...; kw...)

Base.getindex(o::Operator, i...) = getindex(o.h, i...)

Base.eltype(::BarebonesOperator{<:Any,B}) where {B} = B

Base.size(o::BarebonesOperator, is...) = size(matrix(first(harmonics(o))), is...)

Base.getindex(o::BarebonesOperator{L}, dn) where {L} =
    getindex(o, sanitize_SVector(SVector{L,Int}, dn))

Base.getindex(o::BarebonesOperator{L}, dn::SVector{L,Int}) where {L} =
    matrix(harmonics(o)[dn])

# Unlike o[dn][i, j], o[si::AnyCellSites, sj::AnyCellSites] returns a zero if !haskey(dn)
function Base.getindex(o::BarebonesOperator, i::AnyCellSites, j::AnyCellSites = i)
    dn = cell(j) - cell(i)
    si, sj = siteindices(i), siteindices(j)
    if haskey(harmonics(o), dn)
        x = o[dn][si, sj]
    else
        checkbounds(Bool, matrix(first(harmonics(o))), si, sj)
        x = zero(eltype(o))
    end
    return x
end


SparseArrays.nnz(h::BarebonesOperator) = sum(har -> nnz(matrix(har)), harmonics(h))

#endregion

############################################################################################
# OrbitalSliceArray: array type over orbital slice - see specialmatrices.jl
#   orbaxes is a tuple of OrbitalSlice's or Missing's, one per dimension
#   if missing, the dimension does not span orbitals but something else
#region

struct OrbitalSliceArray{C,N,M<:AbstractArray{C,N},A<:NTuple{N,Union{OrbitalSliceGrouped,Missing}}} <: AbstractArray{C,N}
    parent::M
    orbaxes::A
end

const OrbitalSliceVector{C,V,A} = OrbitalSliceArray{C,1,V,A}
const OrbitalSliceMatrix{C,M,A} = OrbitalSliceArray{C,2,M,A}

OrbitalSliceVector(v::AbstractVector, axes) = OrbitalSliceArray(v, axes)
OrbitalSliceMatrix(m::AbstractMatrix, axes) = OrbitalSliceArray(m, axes)

orbaxes(a::OrbitalSliceArray) = a.orbaxes

Base.parent(a::OrbitalSliceArray) = a.parent

#endregion
