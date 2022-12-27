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
        return new(sites, names´, offsets)
    end
end

struct Bravais{T,E,L}
    matrix::Matrix{T}
    function Bravais{T,E,L}(matrix) where {T,E,L}
        (E, L) == size(matrix) || internalerror("Bravais: unexpected matrix size $((E,L)) != $(size(matrix))")
        L > E &&
            throw(DimensionMismatch("Number $L of Bravais vectors cannot be greater than embedding dimension $E"))
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
Bravais(::Type{T}, ::Val{E}, m::NTuple{E´,Number}) where {T,E,E´} =
    Bravais{T,E,1}(sanitize_Matrix(T, E, (m,)))
Bravais(::Type{T}, ::Val{E}, m::NTuple{L,Any}) where {T,E,L} =
    Bravais{T,E,L}(sanitize_Matrix(T, E, m))
Bravais(::Type{T}, ::Val{E}, m::SMatrix{E,L}) where {T,E,L} =
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
        push!(allnames, name)
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

sublatindex(u::Unitcell, name::Symbol) = findfirst(==(name), sublatnames(u))

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
sites(u::Unitcell, ::Nothing) = view(u.sites, 1:0)  # to work with sublatindex
sites(l::Lattice, name::Symbol) = sites(unitcell(l), name)
sites(u::Unitcell, name::Symbol) = sites(u, sublatindex(u, name))

site(l::Lattice, i) = sites(l)[i]
site(l::Lattice, i, dn) = site(l, i) + bravais_matrix(l) * dn

siterange(l::Lattice, sublat) = siterange(l.unitcell, sublat)
siterange(u::Unitcell, sublat) = (1+u.offsets[sublat]):u.offsets[sublat+1]

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

zerocell(::Lattice{<:Any,<:Any,L}) where {L} = zero(SVector{L,Int})

Base.copy(l::Lattice) = deepcopy(l)

#endregion
#endregion

############################################################################################
# Selectors  -  see selector.jl for methods
#region

struct SiteSelector{F,S,C}
    region::F
    sublats::S
    cells::C
end

struct AppliedSiteSelector{T,E,L}
    lat::Lattice{T,E,L}
    region::FunctionWrapper{Bool,Tuple{SVector{E,T}}}
    sublats::Vector{Symbol}
    cells::Vector{SVector{L,Int}}
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
    sublats::Vector{Pair{Symbol,Symbol}}
    dcells::Vector{SVector{L,Int}}
    range::Tuple{T,T}
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

Base.adjoint(s::SiteSelector) = s

Base.adjoint(s::HopSelector) = HopSelector(s.region, s.sublats, s.dcells, s.range, !s.adjoint)

#endregion
#endregion

############################################################################################
# LatticeBlock - see lattice.jl for methods
#   Encodes subsets of sites of a lattice in different cells. Produced by lat[siteselector]
#region

struct Subcell{L}
    inds::Vector{Int}
    cell::SVector{L,Int}
end

struct LatticeBlock{T,E,L}
    lat::Lattice{T,E,L}
    subcells::Vector{Subcell{L}}
end

#region ## Constructors ##

Subcell(cell) = Subcell(Int[], cell)

LatticeBlock(lat::Lattice{<:Any,<:Any,L}) where {L} = LatticeBlock(lat, Subcell{L}[])

#endregion

#region ## API ##

siteindices(s::Subcell) = s.inds

cell(s::Subcell) = s.cell

subcells(l::LatticeBlock) = l.subcells

cells(l::LatticeBlock) = (s.cell for s in l.subcells)

nsites(l::LatticeBlock) = isempty(l) ? 0 : sum(nsites, subcells(l))
nsites(s::Subcell) = length(s.inds)

boundingbox(l::LatticeBlock) = boundingbox(cell(c) for c in subcells(l))

Base.parent(lb::LatticeBlock) = lb.lat

Base.isempty(s::LatticeBlock) = isempty(s.subcells)
Base.isempty(s::Subcell) = isempty(s.inds)

Base.empty!(s::Subcell) = empty!(s.inds)

Base.push!(s::Subcell, i::Int) = push!(s.inds, i)
Base.push!(lb::LatticeBlock, s::Subcell) = push!(lb.subcells, s)

function Base.intersect!(lb::L, lb´::L) where {L<:LatticeBlock}
    for subcell in subcells(lb)
        found = false
        for subcell´ in subcells(lb´)
            if cell(subcell) == cell(subcell´)
                intersect!(siteindices(subcell), siteindices(subcell´))
                found = true
                break
            end
        end
        found || empty!(subcell)
    end
    deleteif!(isempty, subcells(lb))
    return lb
end

#endregion
#endregion

############################################################################################
# Model Terms  -  see model.jl for methods
#region

# Terms #

struct TightbindingModel{T}
    terms::T  # Collection of `TightbindingModelTerm`s
end

struct OnsiteTerm{F,S<:SiteSelector,T<:Number}
    o::F
    selector::S
    coefficient::T
end

struct AppliedOnsiteTerm{T,E,L,B}
    o::FunctionWrapper{B,Tuple{SVector{E,T},Int}}  # o(r, sublat_orbitals)
    selector::AppliedSiteSelector{T,E,L}
end

struct HoppingTerm{F,S<:HopSelector,T<:Number}
    t::F
    selector::S
    coefficient::T
end

struct AppliedHoppingTerm{T,E,L,B}
    t::FunctionWrapper{B,Tuple{SVector{E,T},SVector{E,T},Tuple{Int,Int}}}  # t(r, dr, (orbs1, orbs2))
    selector::AppliedHopSelector{T,E,L}
end

const TightbindingModelTerm = Union{OnsiteTerm,HoppingTerm,AppliedOnsiteTerm,AppliedHoppingTerm}

#region ## Constructors ##

TightbindingModel(ts::TightbindingModelTerm...) = TightbindingModel(ts)

OnsiteTerm(t::OnsiteTerm, os::SiteSelector) = OnsiteTerm(t.o, os, t.coefficient)

HoppingTerm(t::HoppingTerm, os::HopSelector) = HoppingTerm(t.t, os, t.coefficient)

#endregion

#region ## API ##

terms(t::TightbindingModel) = t.terms

selector(t::TightbindingModelTerm) = t.selector

(term::OnsiteTerm{<:Function})(r) = term.coefficient * term.o(r)
(term::OnsiteTerm)(r) = term.coefficient * term.o

(term::AppliedOnsiteTerm)(r, orbs) = term.o(r, orbs)

(term::HoppingTerm{<:Function})(r, dr) = term.coefficient * term.t(r, dr)
(term::HoppingTerm)(r, dr) = term.coefficient * term.t

(term::AppliedHoppingTerm)(r, dr, orbs) = term.t(r, dr, orbs)

# Model term algebra

Base.:*(x::Number, m::TightbindingModel) = TightbindingModel(x .* terms(m))
Base.:*(m::TightbindingModel, x::Number) = x * m
Base.:-(m::TightbindingModel) = (-1) * m

Base.:+(m::TightbindingModel, m´::TightbindingModel) = TightbindingModel((terms(m)..., terms(m´)...))
Base.:-(m::TightbindingModel, m´::TightbindingModel) = m + (-m´)

Base.:*(x::Number, o::OnsiteTerm) = OnsiteTerm(o.o, o.selector, x * o.coefficient)
Base.:*(x::Number, t::HoppingTerm) = HoppingTerm(t.t, t.selector, x * t.coefficient)

Base.adjoint(m::TightbindingModel) = TightbindingModel(adjoint.(terms(m)))
Base.adjoint(t::OnsiteTerm{<:Function}) = OnsiteTerm(r -> t.o(r)', t.selector, t.coefficient')
Base.adjoint(t::OnsiteTerm) = OnsiteTerm(t.o', t.selector, t.coefficient')
Base.adjoint(t::HoppingTerm{<:Function}) = HoppingTerm((r, dr) -> t.t(r, -dr)', t.selector', t.coefficient')
Base.adjoint(t::HoppingTerm) = HoppingTerm(t.t', t.selector', t.coefficient')

#endregion
#endregion

# ############################################################################################
# # OrbitalStructure  -  see hamiltonian.jl for methods
# #region

# struct OrbitalStructure{B<:Union{Number,SMatrix}}
#     blocktype::Type{B}    # Hamiltonian's blocktype
#     norbitals::Vector{Int}
#     offsets::Vector{Int}  # index offset for each sublattice (== offsets(::Lattice))
# end

# #region ## Constructors ##

# # norbs is a collection of number of orbitals, one per sublattice (or a single one for all)
# # B type instability when calling from `hamiltonian` is removed by @inline (const prop)
# @inline function OrbitalStructure(lat::Lattice, norbs, T = numbertype(lat))
#     B = blocktype(T, norbs)
#     return OrbitalStructure{B}(lat, norbs)
# end

# function OrbitalStructure{B}(lat::Lattice, norbs) where {B}
#     norbs´ = sanitize_Vector_of_Type(Int, nsublats(lat), norbs)
#     offsets´ = offsets(lat)
#     return OrbitalStructure{B}(B, norbs´, offsets´)
# end

# # blocktype(T::Type, norbs) = blocktype(T, val_maximum(norbs))
# # blocktype(::Type{T}, m::Val{1}) where {T} = Complex{T}
# # blocktype(::Type{T}, m::Val{N}) where {T,N} = SMatrix{N,N,Complex{T},N*N}

# # val_maximum(n::Int) = Val(n)
# # val_maximum(ns::Tuple) = Val(maximum(argval.(ns)))

# # argval(::Val{N}) where {N} = N
# # argval(n::Int) = n

# #endregion

# #region ## API ##

# norbitals(o::OrbitalStructure) = o.norbitals

# orbtype(::OrbitalStructure{B}) where {B} = orbtype(B)
# orbtype(::Type{B}) where {B<:Number} = B
# orbtype(::Type{B}) where {N,T,B<:SMatrix{N,N,T}} = SVector{N,T}

# blocktype(o::OrbitalStructure) = o.blocktype

# offsets(o::OrbitalStructure) = o.offsets

# nsites(o::OrbitalStructure) = last(offsets(o))

# nsublats(o::OrbitalStructure) = length(norbitals(o))

# sublats(o::OrbitalStructure) = 1:nsublats(o)

# siterange(o::OrbitalStructure, sublat) = (1+o.offsets[sublat]):o.offsets[sublat+1]

# # Equality does not need equal T
# Base.:(==)(o1::OrbitalStructure, o2::OrbitalStructure) =
#     o1.norbs == o2.norbs && o1.offsets == o2.offsets

# #endregion
# #endregion

############################################################################################
# Hamiltonian builders
#region

abstract type AbstractHamiltonianBuilder{T,E,L,B} end

abstract type AbstractBuilderHarmonic{L,B} end

struct IJVHarmonic{L,B} <: AbstractBuilderHarmonic{L,B}
    dn::SVector{L,Int}
    collector::IJV{B}
end

mutable struct CSCHarmonic{L,B} <: AbstractBuilderHarmonic{L,B}
    dn::SVector{L,Int}
    collector::CSC{B}
end

struct IJVBuilder{T,E,L,B} <: AbstractHamiltonianBuilder{T,E,L,B}
    lat::Lattice{T,E,L}
    blockstruct::BlockStructure{B}
    harmonics::Vector{IJVHarmonic{L,B}}
    kdtrees::Vector{KDTree{SVector{E,T},Euclidean,T}}
end

struct CSCBuilder{T,E,L,B} <: AbstractHamiltonianBuilder{T,E,L,B}
    lat::Lattice{T,E,L}
    blockstruct::BlockStructure{B}
    harmonics::Vector{CSCHarmonic{L,B}}
end

## Constructors ##

function IJVBuilder(lat::Lattice{T,E,L}, blockstruct::BlockStructure{B}) where {E,L,T,B}
    harmonics = IJVHarmonic{L,B}[]
    kdtrees = Vector{KDTree{SVector{E,T},Euclidean,T}}(undef, nsublats(lat))
    return IJVBuilder(lat, blockstruct, harmonics, kdtrees)
end

function CSCBuilder(lat::Lattice{<:Any,<:Any,L}, blockstruct::BlockStructure{B}) where {L,B}
    harmonics = CSCHarmonic{L,B}[]
    return CSCBuilder(lat, blockstruct, harmonics)
end

empty_harmonic(b::CSCBuilder{<:Any,<:Any,L,B}, dn) where {L,B} =
    CSCHarmonic{L,B}(dn, CSC{B}(nsites(b.lat)))

empty_harmonic(::IJVBuilder{<:Any,<:Any,L,B}, dn) where {L,B} =
    IJVHarmonic{L,B}(dn, IJV{B}())

## API ##

collector(har::AbstractBuilderHarmonic) = har.collector  # for IJVHarmonic and CSCHarmonic

dcell(har::AbstractBuilderHarmonic) = har.dn

kdtrees(b::IJVBuilder) = b.kdtrees

Base.filter!(f::Function, b::IJVBuilder) =
    foreach(bh -> filter!(f, bh.collector), b.harmonics)

finalizecolumn!(b::CSCBuilder, x...) =
    foreach(har -> finalizecolumn!(collector(har), x...), b.harmonics)

Base.isempty(h::IJVHarmonic) = isempty(collector(h))
Base.isempty(s::CSCHarmonic) = isempty(collector(s))

lattice(b::AbstractHamiltonianBuilder) = b.lat

blockstructure(b::AbstractHamiltonianBuilder) = b.blockstruct

harmonics(b::AbstractHamiltonianBuilder) = b.harmonics

function Base.getindex(b::AbstractHamiltonianBuilder{<:Any,<:Any,L}, dn::SVector{L,Int}) where {L}
    hars = b.harmonics
    for har in hars
        dcell(har) == dn && return collector(har)
    end
    har = empty_harmonic(b, dn)
    push!(hars, har)
    return collector(har)
end

function SparseArrays.sparse(builder::AbstractHamiltonianBuilder{T,<:Any,L,B}) where {T,L,B}
    HT = Harmonic{T,L,B}
    b = blockstructure(builder)
    n = nsites(lattice(builder))
    hars = HT[sparse(b, har, n, n) for har in harmonics(builder) if !isempty(har)]
    return hars
end

function SparseArrays.sparse(b::BlockStructure{B}, har::AbstractBuilderHarmonic{L,B}, m::Integer, n::Integer) where {L,B}
    s = sparse(collector(har), m, n)
    return Harmonic(dcell(har), HybridSparseMatrixCSC(b, s))
end

#endregion

############################################################################################
# Harmonic  -  see hamiltonian.jl for methods
#region

struct Harmonic{T,L,B}
    dn::SVector{L,Int}
    h::HybridSparseMatrixCSC{T,B}
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

Base.zero(h::Harmonic{<:Any,<:Any,B}) where B = Harmonic(zero(dcell(h)), zero(matrix(h)))

Base.copy(h::Harmonic) = Harmonic(dcell(h), copy(matrix(h)))

#endregion
#endregion

############################################################################################
# Hamiltonian  -  see hamiltonian.jl for methods
#region

abstract type AbstractHamiltonian{T,E,L,B} end

struct Hamiltonian{T,E,L,B} <: AbstractHamiltonian{T,E,L,B}
    lattice::Lattice{T,E,L}
    blockstruct::BlockStructure{B}
    harmonics::Vector{Harmonic{T,L,B}}
    bloch::HybridSparseMatrixCSC{T,B}
    # Enforce sorted-dns-starting-from-zero invariant onto harmonics
    function Hamiltonian{T,E,L,B}(lattice, blockstruct, harmonics, bloch) where {T,E,L,B}
        n = nsites(lattice)
        all(har -> size(matrix(har)) == (n, n), harmonics) ||
            throw(DimensionMismatch("Harmonic $(size.(matrix.(harmonics), 1)) sizes don't match number of sites $n"))
        sort!(harmonics)
        (isempty(harmonics) || !iszero(dcell(first(harmonics)))) && pushfirst!(harmonics,
            Harmonic(zero(SVector{L,Int}), HybridSparseMatrixCSC(blockstruct, spzeros(B, n, n))))
        return new(lattice, blockstruct, harmonics, bloch)
    end
end

#region ## API ##

## AbstractHamiltonian

norbitals(h::AbstractHamiltonian) = blocksizes(blockstructure(h))

blockeltype(::AbstractHamiltonian) = blockeltype(blockstructure(h))

blocktype(h::AbstractHamiltonian) = blocktype(blockstructure(h))

# see sparsetools.jl
flatrange(h::AbstractHamiltonian, iunflat) = flatrange(blockstructure(h), iunflat)

## Hamiltonian

Hamiltonian(l::Lattice{T,E,L}, b::BlockStructure{B}, h::Vector{Harmonic{T,L,B}}, bl) where {T,E,L,B} =
    Hamiltonian{T,E,L,B}(l, b, h, bl)

function Hamiltonian(l, b::BlockStructure{B}, h) where {B}
    n = nsites(l)
    bloch = HybridSparseMatrixCSC(b, spzeros(B, n, n))
    needs_initialization!(bloch)
    return Hamiltonian(l, b, h, bloch)
end

hamiltonian(h::Hamiltonian) = h

blockstructure(h::Hamiltonian) = h.blockstruct

lattice(h::Hamiltonian) = h.lattice

harmonics(h::Hamiltonian) = h.harmonics

bloch(h::Hamiltonian) = h.bloch

Base.size(h::Hamiltonian, i...) = size(first(harmonics(h)), i...)

Base.copy(h::Hamiltonian) = Hamiltonian(
    copy(lattice(h)), copy(blockstructure(h)), copy.(harmonics(h)), copy(bloch(h)))

function LinearAlgebra.ishermitian(h::Hamiltonian)
    for hh in h.harmonics
        isassigned(h, -hh.dn) || return false
        hh.h ≈ h[-hh.dn]' || return false
    end
    return true
end

#endregion
#endregion

############################################################################################
# Model Modifiers  -  see model.jl for methods
#region

# wrapper of a function f(x1, ... xN; kw...) with N arguments and the kwargs in params
struct ParametricFunction{N,F}
    f::F
    params::Vector{Symbol}
end

abstract type AbstractModifier end

struct OnsiteModifier{N,S<:SiteSelector,F<:ParametricFunction{N}} <: AbstractModifier
    f::F
    selector::S
end

struct AppliedOnsiteModifier{B,N,R<:SVector,F<:ParametricFunction{N}} <: AbstractModifier
    blocktype::Type{B}
    f::F
    ptrs::Vector{Tuple{Int,R,Int}}
    # [(ptr, r, norbs)...] for each selected site, dn = 0 harmonic
end

struct HoppingModifier{N,S<:HopSelector,F<:ParametricFunction{N}} <: AbstractModifier
    f::F
    selector::S
end

struct AppliedHoppingModifier{B,N,R<:SVector,F<:ParametricFunction{N}} <: AbstractModifier
    blocktype::Type{B}
    f::F
    ptrs::Vector{Vector{Tuple{Int,R,R,Tuple{Int,Int}}}}
    # [[(ptr, r, dr, (norbs, norbs´)), ...], ...] for each selected hop on each harmonic
end

const Modifier = Union{OnsiteModifier,HoppingModifier}
const AppliedModifier = Union{AppliedOnsiteModifier,AppliedHoppingModifier}

#region ## Constructors ##

ParametricFunction{N}(f::F, params) where {N,F} = ParametricFunction{N,F}(f, params)

#endregion

#region ## API ##

selector(m::Modifier) = m.selector

parameters(m::Union{Modifier,AppliedModifier}) = m.f.params

parametric_function(m::Union{Modifier,AppliedModifier}) = m.f

pointers(m::AppliedModifier) = m.ptrs

blocktype(m::AppliedModifier) = m.blocktype

(m::AppliedOnsiteModifier{B,1})(o, r, orbs; kw...) where {B} =
    mask_block(B, m.f.f(o; kw...), (orbs, orbs))
(m::AppliedOnsiteModifier{B,2})(o, r, orbs; kw...) where {B} =
    mask_block(B, m.f.f(o, r; kw...), (orbs, orbs))

(m::AppliedHoppingModifier{B,1})(t, r, dr, orborb; kw...) where {B} =
    mask_block(B, m.f.f(t; kw...), orborb)
(m::AppliedHoppingModifier{B,3})(t, r, dr, orborb; kw...) where {B} =
    mask_block(B, m.f.f(t, r, dr; kw...), orborb)

Base.similar(m::A) where {A <: AppliedModifier} = A(m.blocktype, m.f, similar(m.ptrs, 0))

function emptyptrs!(m::AppliedHoppingModifier{<:Any,<:Any,R}, n) where {R}
    resize!(m.ptrs, 0)
    foreach(_ -> push!(m.ptrs, Tuple{Int,R,R,Tuple{Int,Int}}[]), 1:n)
    return m
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

Base.parent(h::ParametricHamiltonian) = h.hparent

hamiltonian(h::ParametricHamiltonian) = h.h

bloch(h::ParametricHamiltonian) = h.h.bloch

parameters(h::ParametricHamiltonian) = h.allparams

modifiers(h::ParametricHamiltonian) = h.modifiers

pointers(h::ParametricHamiltonian) = h.allptrs

harmonics(h::ParametricHamiltonian) = harmonics(parent(h))

blockstructure(h::ParametricHamiltonian) = blockstructure(parent(h))

blocktype(h::ParametricHamiltonian) = blocktype(parent(h))

lattice(h::ParametricHamiltonian) = lattice(parent(h))

Base.size(h::ParametricHamiltonian, i...) = size(parent(h), i...)

Base.copy(p::ParametricHamiltonian) = ParametricHamiltonian(
    copy(p.hparent), copy(p.h), p.modifiers, deepcopy(p.allptrs), copy(p.allparams))

#endregion
#endregion

############################################################################################
# Mesh  -  see mesh.jl for methods
#region

abstract type AbstractMesh{V,S} end

struct Mesh{V,S} <: AbstractMesh{V,S}
    verts::Vector{V}
    neighs::Vector{Vector{Int}}          # all neighbors neighs[i][j] of vertex i
    simps::Vector{NTuple{S,Int}}         # list of simplices, each one a group of neighboring vertex indices
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

vertices(m::Mesh) = m.verts
vertices(m::Mesh, i) = m.verts[i]

neighbors(m::Mesh) = m.neighs
neighbors(m::Mesh, i::Int) = m.neighs[i]

neighbors_forward(m::Mesh, i::Int) = Iterators.filter(>(i), m.neighs[i])
neighbors_forward(v::Vector, i::Int) = Iterators.filter(>(i), v[i])

simplices(m::Mesh) = m.simps
simplices(m::Mesh, i::Int) = m.simps[i]

Base.copy(m::Mesh) = Mesh(copy(m.verts), deepcopy(m.neighs), copy(m.simps))

#endregion
#endregion

############################################################################################
# Spectrum  -  see solvers/eigensolvers.jl for solver backends <: AbstractEigenSolver
#           -  see spectrum.jl for public API
#region

abstract type AbstractEigenSolver end

struct Spectrum{T,B}
    eigen::Eigen{Complex{T},Complex{T},Matrix{Complex{T}},Vector{Complex{T}}}
    blockstruct::BlockStructure{B}
end

#region ## Constructors ##

Spectrum(eigen::Eigen, h::AbstractHamiltonian) = Spectrum(eigen, blockstructure(h))
Spectrum(eigen::Eigen, h, ::Missing) = Spectrum(eigen, h)

function Spectrum(eigen::Eigen, h, transform)
    s = Spectrum(eigen, h)
    map!(transform, energies(s))
    return s
end

#endregion

#region ## API ##

energies(s::Spectrum) = s.eigen.values

states(s::Spectrum) = s.eigen.vectors

Base.size(s::Spectrum, i...) = size(s.eigen.vectors, i...)

#endregion
#endregion

############################################################################################
# SpectrumSolver - reuses bloch matrix when applying to many Bloch phases, see spectrum.jl
#region

struct SpectrumSolver{T,L,B}
    solver::FunctionWrapper{Spectrum{T,B},Tuple{SVector{L,T}}}
end

#region ## API ##

(s::SpectrumSolver{T,L})(φs::Vararg{<:Any,L}) where {T,L} = s.solver(sanitize_SVector(SVector{L,T}, φs))
(s::SpectrumSolver{T,L})(φs::SVector{L}) where {T,L} = s.solver(sanitize_SVector(SVector{L,T}, φs))
(s::SpectrumSolver{T,L})(φs::NTuple{L,Any}) where {T,L} = s.solver(sanitize_SVector(SVector{L,T}, φs))
(s::SpectrumSolver{T,L})(φs...) where {T,L} =
    throw(ArgumentError("SpectrumSolver call requires $L parameters/Bloch phases, received $φs"))

#endregion
#endregion

############################################################################################
# Bands and friends -  see spectrum.jl for methods
#region

const MatrixView{C} = SubArray{C,2,Matrix{C},Tuple{Base.Slice{Base.OneTo{Int}}, UnitRange{Int}}, true}

struct BandVertex{T<:AbstractFloat,E}
    coordinates::SVector{E,T}
    states::MatrixView{Complex{T}}
end

# Subband is a type of AbstractMesh with manifold dimension = embedding dimension - 1
# and with interval search trees to allow slicing
struct Subband{T,E} <: AbstractMesh{BandVertex{T,E},E}  # we restrict S == E
    mesh::Mesh{BandVertex{T,E},E}
    trees::NTuple{E,IntervalTree{T,IntervalValue{T,Int}}}
end

struct Bands{T,E,L,B} # E = L+1
    subbands::Vector{Subband{T,E}}
    solvers::Vector{SpectrumSolver{T,L,B}}  # one per Julia thread
end

#region ## Constructors ##

BandVertex(x, s::Matrix) = BandVertex(x, view(s, :, 1:size(s, 2)))
BandVertex(m, e, s::Matrix) = BandVertex(m, e, view(s, :, 1:size(s, 2)))
BandVertex(m, e, s::SubArray) = BandVertex(vcat(m, e), s)

Subband(verts::Vector{<:BandVertex{<:Any,E}}, neighs) where {E} =
    Subband(Mesh{E}(verts, neighs))

function Subband(mesh::Mesh{<:BandVertex{T,E}}) where {T,E}
    verts, simps = vertices(mesh), simplices(mesh)
    order_simplices!(simps, verts)
    trees = ntuple(Val(E)) do i
        list = [IntervalValue(shrinkright(extrema(j->coordinates(verts[j])[i], s))..., n)
                     for (n, s) in enumerate(simps)]
        sort!(list)
        return IntervalTree{T,IntervalValue{T,Int}}(list)
    end
    return Subband(mesh, trees)
end

# Interval is closed, we want semiclosed on the left -> exclude the upper limit
shrinkright((x, y)) = (x, prevfloat(y))

#endregion

#region ## API ##
# BandVertex #

coordinates(s::SVector) = s
coordinates(v::BandVertex) = v.coordinates

energy(v::BandVertex) = last(v.coordinates)

base_coordinates(v::BandVertex) = SVector(Base.front(Tuple(v.coordinates)))

states(v::BandVertex) = v.states

degeneracy(v::BandVertex) = size(v.states, 2)

parentrows(v::BandVertex) = first(parentindices(v.states))
parentcols(v::BandVertex) = last(parentindices(v.states))

embdim(::AbstractMesh{<:SVector{E}}) where {E} = E

embdim(::AbstractMesh{<:BandVertex{<:Any,E}}) where {E} = E

# Subband #

vertices(s::Subband, i...) = vertices(s.mesh, i...)

neighbors(s::Subband, i...) = neighbors(s.mesh, i...)

neighbors_forward(s::Subband, i) = neighbors_forward(s.mesh, i)

simplices(s::Subband, i...) = simplices(s.mesh, i...)

trees(s::Subband) = s.trees
trees(s::Subband, i::Int) = s.trees[i]

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

Base.isempty(s::Subband) = isempty(simplices(s))

# Band #

basemesh(b::Bands) = b.basemesh

subbands(b::Bands) = b.subbands

subbands(b::Bands, i...) = getindex(b.subbands, i...)

#endregion
#endregion

############################################################################################
# Green solvers - see solvers/greensolvers.jl
#region

abstract type AbstractGreenSolver end
abstract type AppliedGreenSolver end
abstract type AppliedDirectGreenSolver <: AppliedGreenSolver end
abstract type AppliedInverseGreenSolver <: AppliedGreenSolver end

# API for s::AppliedGreenSolver
#   - call!(s; params...) -> AppliedGreenSolver
#   - call!(s, ω; params...) -> AbstractMatrix
# From these, we define the out-of-place call syntax:

(s::AppliedGreenSolver)(; params...) = copy_callsafe(call!(s; params...))
(s::AppliedGreenSolver)(ω; params...) = copy(call!(s, ω; params...))

# Note: copy_callsafe is required to avoid aliased and/or wrong m1 and m2 when doing e.g.
# gb = green(h)[...]; g1 = gb(; params1...); g2 = gb(; params2...); m1, m2 = g1(ω), g2(ω)

#endregion

############################################################################################
# Green - see green.jl
#region

# abstract type AbstractGreen end

# # g::Green represents G in the whole lattice. It must be made "concrete" to be used, i.e.
# # converting it to a GreenBlock with g[; site_selections...]
# struct Green{T,L,S<:AbstractGreenSolver,H<:AbstractHamiltonian{T,<:Any,L}} <: AbstractGreen
#     h::H
#     boundaries::NTuple{L,T}  # We use floats for boundaries to allow Inf or large values
#     solver::S
# end

# # produces G on latblocks with call!(g, ω; params...) or g(ω; params...)
# # Partial application: call!(g; params...) or g(; params...) produces a GreenBlock
# struct GreenBlock{S<:AppliedDirectGreenSolver,
#                   L<:NTuple{<:Any,<:LatticeBlock},
#                   P} <: AbstractGreen
#     # solver returns an AbstractMatrix with its elements matching latblocks sites
#     solver::S
#     latblocks::L
#     parent::P
# end

# # produces G⁻¹ on latblocks with call!(g, ω; params...) or g(ω; params...)
# # Partial application: call!(g; params...) or g(; params...) produces a GreenBlock
# struct GreenBlockInverse{S<:AppliedInverseGreenSolver,
#                          L<:NTuple{<:Any,<:LatticeBlock},
#                          P} <: AbstractGreen
#     # solver returns an AbstractMatrix with its first elements matching latblocks sites
#     # optionally it could have more trailing entries corresponding to auxiliary orbitals
#     solver::S
#     latblocks::L
#     parent::P
#     auxiliaries::Int    # number of auxiliary orbitals
# end

#region ## API ##

# hamiltonian(g::Green) = g.h

# lattice(g::Green) = lattice(g.h)

# boundaries(g::Green) = g.boundaries

# solver(g::Green) = g.solver
# solver(g::GreenBlockInverse) = g.solver

# latblocks(g::GreenBlock) = g.latblocks
# latblocks(g::GreenBlockInverse) = g.latblocks

# Base.parent(g::Green) = g.h
# Base.parent(g::GreenBlock) = g.parent
# Base.parent(g::GreenBlockInverse) = g.parent

# Base.copy(g::AbstractGreen) = deepcopy(g)

#endregion
#endregion