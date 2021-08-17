############################################################################################
# Lattice
#region

struct Sublat{T<:AbstractFloat,E}
    sites::Vector{SVector{E,T}}
    name::Symbol
end

struct Unitcell{T<:AbstractFloat,E}
    sites::Vector{SVector{E,T}}
    names::Vector{Symbol}
    offsets::Vector{Int}        # Linear site number offsets for each sublat
end

struct Bravais{T,E,L}
    matrix::Matrix{T}
    function Bravais{T,E,L}(matrix) where {T,E,L}
        (E, L) == size(matrix) || throw(ErrorException("Internal error: unexpected matrix size $((E,L)) != $(size(matrix))"))
        L > E &&
            throw(DimensionMismatch("Number $L of Bravais vectors cannot be greater than embedding dimension $E"))
        return new(matrix)
    end
end

struct Lattice{T<:AbstractFloat,E,L}
    bravais::Bravais{T,E,L}
    unitcell::Unitcell{T,E}
end

#region internal API

unitcell(l::Lattice) = l.unitcell

bravais_vecs(l::Lattice) = bravais_vecs(l.bravais)
bravais_vecs(b::Bravais) = eachcol(b.matrix)

bravais_mat(l::Lattice) = bravais_mat(l.bravais)
bravais_mat(b::Bravais{T,E,L}) where {T,E,L} =
    convert(SMatrix{E,L,T}, ntuple(i -> b.matrix[i], Val(E*L)))

sublatnames(l::Lattice) = l.unitcell.names
sublatname(l::Lattice, s) = sublatname(l.unitcell, s)
sublatname(u::Unitcell, s) = u.names[s]
sublatname(s::Sublat) = s.name

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
sites(u::Unitcell, sublat) = view(u.sites, u.offsets[sublat]+1:u.offsets[sublat+1])

site(l::Lattice, i) = sites(l)[i]
site(l::Lattice, i, dn) = site(l, i) + bravais_mat(l) * dn

siterange(l::Lattice, sublat) = siterange(l.unitcell, sublat)
siterange(u::Unitcell, sublat) = (1+u.offsets[sublat]):u.offsets[sublat+1]

sitesublat(lat::Lattice, siteidx, ) = sitesublat(lat.unitcell.offsets, siteidx)
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

zerocell(::Lattice{<:Any,<:Any,L}) where {L} = zero(SVector{L,Int})

#endregion

#endregion

############################################################################################
# Selectors
#region

abstract type Selector end

struct SiteSelector{M,S,I} <: Selector
    region::M
    sublats::S
    indices::I
end

struct HopSelector{M,S,I,D,T} <: Selector
    region::M
    sublats::S
    indices::I
    dcells::D
    range::T
end

struct BlockSelector{V<:Union{Missing,Vector}} <: Selector
    cells::V
    rows::Vector{Int}
    cols::Vector{Int}
end

struct NeighborRange
    n::Int
end

struct AppliedOn{S,D}
    src::S
    dst::D
end

#region internal API

appliedon(s, d) = AppliedOn(apply(s, d), d)

apply(s, d) = s  # fallback for no action on s

source(a::AppliedOn) = a.src

target(a::AppliedOn) = a.dst

Base.parent(n::NeighborRange) = n.n

#endregion

#endregion

############################################################################################
# Model
#region
abstract type TightbindingModelTerm end

struct TightbindingModel{T}
    terms::T  # Collection of `TightbindingModelTerm`s
end

# These need to be concrete as they are involved in hot construction loops
struct OnsiteTerm{F,S<:Selector,T<:Number} <: TightbindingModelTerm
    o::F
    selector::S
    coefficient::T
end

struct HoppingTerm{F,S<:Selector,T<:Number} <: TightbindingModelTerm
    t::F
    selector::S
    coefficient::T
end

#region internal API

terms(t::TightbindingModel) = t.terms

selector(t::TightbindingModelTerm) = t.selector

#endregion
#endregion

############################################################################################
# OrbitalStructure
#region

struct OrbitalStructure{O<:Union{Number,SMatrix}}
    blocktype::Type{O}    # Hamiltonian's blocktype
    norbitals::Vector{Int}
    offsets::Vector{Int}
    flatoffsets::Vector{Int}
end

#region internal API

norbitals(o::OrbitalStructure) = o.norbitals

orbtype(o::OrbitalStructure{O}) where {O<:Number} = O
orbtype(o::OrbitalStructure{O}) where {N,T,O<:SMatrix{N,N,T}} = SVector{N,T}

blocktype(o::OrbitalStructure) = o.blocktype

offsets(o::OrbitalStructure) = o.offsets

flatoffsets(o::OrbitalStructure) = o.flatoffsets

#endregion
#endregion

############################################################################################
# Hamiltonian
#region

struct HamiltonianHarmonic{L,O}
    dn::SVector{L,Int}
    h::SparseMatrixCSC{O,Int}
end

struct Hamiltonian{T,E,L,O}
    lattice::Lattice{T,E,L}
    orbstruct::OrbitalStructure{O}
    harmonics::Vector{HamiltonianHarmonic{L,O}}
    # Enforce sorted-dns-starting-from-zero invariant onto harmonics
    function Hamiltonian{T,E,L,O}(lattice, orbstruct, harmonics) where {T,E,L,O}
        n = nsites(lattice)
        all(har -> size(matrix(har)) == (n, n), harmonics) ||
            throw(DimensionMismatch("Harmonic sizes don't match number of sites $n"))
        length(harmonics) > 0 && iszero(dcell(first(harmonics))) || pushfirst!(harmonics,
            HamiltonianHarmonic(zero(SVector{L,Int}), sparse(Int[], Int[], O[], n, n)))
        sort!(harmonics)
        return new(lattice, orbstruct, harmonics)
    end
end

Hamiltonian(l::Lattice{T,E,L}, o::OrbitalStructure{O}, h::Vector{HamiltonianHarmonic{L,O}},) where {T,E,L,O} =
    Hamiltonian{T,E,L,O}(l, o, h)

#region internal API

matrix(h::HamiltonianHarmonic) = h.h

dcell(h::HamiltonianHarmonic) = h.dn

orbitalstructure(h::Hamiltonian) = h.orbstruct

lattice(h::Hamiltonian) = h.lattice

harmonics(h::Hamiltonian) = h.harmonics

orbtype(h::Hamiltonian) = orbtype(orbitalstructure(h))

blocktype(h::Hamiltonian) = blocktype(orbitalstructure(h))

norbitals(h::Hamiltonian) = norbitals(orbitalstructure(h))

Base.size(h::Hamiltonian, i...) = size(matrix(first(harmonics(h))), i...)

Base.isless(h::HamiltonianHarmonic, h´::HamiltonianHarmonic) = sum(abs2, dcell(h)) < sum(abs2, dcell(h´))

#endregion
#endregion

############################################################################################
# CSC Hamiltonian builder
#region

# struct UnfinalizedSparseMatrixCSC{Tv,Ti} <: AbstractSparseMatrix{Tv,Ti}
#     m::Int
#     n::Int
#     colptr::Vector{Ti}
#     rowval::Vector{Ti}
#     nzval::Vector{Tv}
# end

# function SparseMatrixBuilder{Tv}(m, n, nnzguess = missing) where {Tv}
#     colptr = [1]
#     rowval = Int[]
#     nzval = Tv[]
#     matrix = UnfinalizedSparseMatrixCSC(m, n, colptr, rowval, nzval)
#     builder = SparseMatrixBuilder(matrix, 1, 0, CoSort(rowval, nzval))
#     nnzguess === missing || sizehint!(builder, nnzguess)
#     return builder
# end

# # Unspecified size constructor
# SparseMatrixBuilder{Tv}() where Tv = SparseMatrixBuilder{Tv}(0, 0)

# function SparseMatrixBuilder(s::SparseMatrixCSC{Tv,Int}) where {Tv}
#     colptr = getcolptr(s)
#     rowval = rowvals(s)
#     nzval = nonzeros(s)
#     nnzguess = length(nzval)
#     resize!(rowval, 0)
#     resize!(nzval, 0)
#     resize!(colptr, 1)
#     colptr[1] = 1
#     builder = SparseMatrixBuilder(s, 1, 0, CoSort(rowval, nzval))
#     sizehint!(builder, nnzguess)
#     return builder
# end

# SparseArrays.getcolptr(s::UnfinalizedSparseMatrixCSC) = s.colptr
# SparseArrays.nonzeros(s::UnfinalizedSparseMatrixCSC) = s.nzval
# SparseArrays.rowvals(s::UnfinalizedSparseMatrixCSC) = s.rowval
# Base.size(s::UnfinalizedSparseMatrixCSC) = (s.m, s.n)
# Base.size(s::UnfinalizedSparseMatrixCSC, k) = size(s)[k]

# function Base.sizehint!(s::SparseMatrixBuilder, n)
#     sizehint!(getcolptr(s.matrix), n + 1)
#     sizehint!(nonzeros(s.matrix), n)
#     sizehint!(rowvals(s.matrix), n)
#     return s
# end

# function pushtocolumn!(s::SparseMatrixBuilder, row::Int, x, skipdupcheck::Bool = true)
#     nrows = size(s.matrix, 1)
#     nrows == 0 || 1 <= row <= size(s.matrix, 1) || throw(ArgumentError("tried adding a row $row out of bounds ($(size(s.matrix, 1)))"))
#     if skipdupcheck || !isintail(row, rowvals(s.matrix), getcolptr(s.matrix)[s.colcounter])
#         push!(rowvals(s.matrix), row)
#         push!(nonzeros(s.matrix), x)
#         s.rowvalcounter += 1
#     end
#     return s
# end

# function isintail(element, container, start::Int)
#     for i in start:length(container)
#         container[i] == element && return true
#     end
#     return false
# end

# function finalizecolumn!(s::SparseMatrixBuilder, sortcol::Bool = true)
#     size(s.matrix, 2) > 0 && s.colcounter > size(s.matrix, 2) && throw(DimensionMismatch("Pushed too many columns to matrix"))
#     if sortcol
#         s.cosorter.offset = getcolptr(s.matrix)[s.colcounter] - 1
#         sort!(s.cosorter)
#         isgrowing(s.cosorter) || throw(error("Internal error: repeated rows"))
#     end
#     s.colcounter += 1
#     push!(getcolptr(s.matrix), s.rowvalcounter + 1)
#     return nothing
# end

# function finalizecolumn!(s::SparseMatrixBuilder, ncols::Int)
#     for _ in 1:ncols
#         finalizecolumn!(s)
#     end
#     return nothing
# end

# function SparseArrays.sparse(s::SparseMatrixBuilder{<:Any,<:SparseMatrixCSC})
#     completecolptr!(getcolptr(s.matrix), size(s.matrix, 2), s.rowvalcounter)
#     return s.matrix
# end

# function completecolptr!(colptr, cols, lastrowptr)
#     colcounter = length(colptr)
#     if colcounter < cols + 1
#         resize!(colptr, cols + 1)
#         for col in (colcounter + 1):(cols + 1)
#             colptr[col] = lastrowptr + 1
#         end
#     end
#     return colptr
# end

# function SparseArrays.sparse(s::SparseMatrixBuilder{<:Any,<:UnfinalizedSparseMatrixCSC})
#     m, n = size(s.matrix)
#     rowval = rowvals(s.matrix)
#     colptr = getcolptr(s.matrix)
#     nzval = nonzeros(s.matrix)
#     if m != 0 && n != 0
#         completecolptr!(colptr, n, s.rowvalcounter)
#     else # determine size of matrix after the fact
#         m, n = isempty(rowval) ? 0 : maximum(rowval), s.colcounter - 1
#     end
#     return SparseMatrixCSC(m, n, colptr, rowval, nzval)
# end

#endregion

# #######################################################################
# # Modifiers
# #######################################################################
# struct ParametricFunction{N,F}
#     f::F
#     params::Vector{Symbol}
# end

# struct Modifier{N,S<:Selector,F}
#     f::ParametricFunction{N,F}
#     selector::S
# end

# const ElementModifier{N,S<:ElementSelector,F} = Modifier{N,S,F}
# const HopModifier{N,S<:Union{HopSelector,ResolvedHopSelector},F} = Modifier{N,S,F}
# const SiteModifier{N,S<:Union{SiteSelector,ResolvedSiteSelector},F} = Modifier{N,S,F}
# const BlockModifier{N,S<:BlockSelector,F} = Modifier{N,S,F}
# const UniformModifier = ElementModifier{1}
# const UniformHopModifier = HopModifier{1}
# const UniformSiteModifier = SiteModifier{1}