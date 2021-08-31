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

bravais(l::Lattice) = l.bravais

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

struct SiteSelector{F,S,I}
    region::F
    sublats::S
    indices::I
end

struct AppliedSiteSelector{T,E,L}
    lat::Lattice{T,E,L}
    region::FunctionWrapper{Bool,Tuple{SVector{E,T}}}
    sublats::Vector{Symbol}
end

struct HopSelector{F,S,I,D,R}
    region::F
    sublats::S
    indices::I
    dcells::D
    range::R
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

#region internal API

Base.parent(n::Neighbors) = n.n

lattice(ap::AppliedSiteSelector) = ap.lat
lattice(ap::AppliedHopSelector) = ap.lat

dcells(ap::AppliedHopSelector) = ap.dcells

# if isempty(s.dcells) or isempty(s.sublats), none were specified, so we must accept any
inregion(r, s::AppliedSiteSelector) = s.region(r)
inregion((r, dr), s::AppliedHopSelector) = s.region(r, dr)

insublats(n, s::AppliedSiteSelector) = isempty(s.sublats) || n in s.sublats
insublats(npair::Pair, s::AppliedHopSelector) = isempty(s.sublats) || npair in s.sublats
indcells(dcell, s::AppliedHopSelector) = isempty(s.dcells) || dcell in s.dcells

iswithinrange(dr, s::AppliedHopSelector) = iswithinrange(dr, s.range)
iswithinrange(dr, (rmin, rmax)::Tuple{Real,Real}) =  ifelse(rmin^2 <= dr'dr <= rmax^2, true, false)

isbelowrange(dr, s::AppliedHopSelector) = isbelowrange(dr, s.range)
isbelowrange(dr, (rmin, rmax)::Tuple{Real,Real}) =  ifelse(dr'dr < rmin^2, true, false)

#endregion

#endregion

############################################################################################
# Model Terms
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

struct AppliedOnsiteTerm{T,E,L,O}
    o::FunctionWrapper{O,Tuple{SVector{E,T},Int}}  # o(r, sublat_orbitals)
    selector::AppliedSiteSelector{T,E,L}
end

struct HoppingTerm{F,S<:HopSelector,T<:Number}
    t::F
    selector::S
    coefficient::T
end

struct AppliedHoppingTerm{T,E,L,O}
    t::FunctionWrapper{O,Tuple{SVector{E,T},SVector{E,T},Tuple{Int,Int}}}  # t(r, dr, (orbs1, orbs2))
    selector::AppliedHopSelector{T,E,L}
end

const TightbindingModelTerm = Union{OnsiteTerm,HoppingTerm,AppliedOnsiteTerm,AppliedHoppingTerm}

#region Term internal API

terms(t::TightbindingModel) = t.terms

selector(t::TightbindingModelTerm) = t.selector

(term::OnsiteTerm{<:Function})(r) = term.coefficient * term.o(r)
(term::OnsiteTerm)(r) = term.coefficient * term.o

(term::AppliedOnsiteTerm)(r, orbs) = term.o(r, orbs)

(term::HoppingTerm{<:Function})(r, dr) = term.coefficient * term.t(r, dr)
(term::HoppingTerm)(r, dr) = term.coefficient * term.t

(term::AppliedHoppingTerm)(r, dr, orbs) = term.t(r, dr, orbs)

#endregion
#endregion

############################################################################################
# Model Modifiers
#region

# wrapper of a function f(x1, ... xN; kw...) with N arguments and the kwargs in params
struct ParametricFunction{N,F}
    f::F
    params::Vector{Symbol}
end

ParametricFunction{N}(f::F, params) where {N,F} = ParametricFunction{N,F}(f, params)

struct OnsiteModifier{N,S<:SiteSelector,F<:ParametricFunction{N}}
    f::F
    selector::S
end

struct PartiallyAppliedOnsiteModifier{N,T,E,F<:ParametricFunction{N}}
    f::F
    ptrs::Vector{Tuple{Int,SVector{E,T},Int}}
    # [(ptr, r, norbs)...] for each selected site, dn = 0 harmonic
end

struct AppliedOnsiteModifier{T,E,L,O}
    f::FunctionWrapper{O,Tuple{O,SVector{E,T},Int}}
    # o(old_o, r, norbs)
    ptrs::Vector{Tuple{Int,SVector{E,T},Int}}
    # [(ptr, r, norbs)...] for each selected site, dn = 0 harmonic
end

struct HoppingModifier{N,S<:HopSelector,F<:ParametricFunction{N}}
    f::F
    selector::S
end

struct PartiallyAppliedHoppingModifier{N,T,E,F<:ParametricFunction{N}}
    f::F
    ptrs::Vector{Vector{Tuple{Int,SVector{E,T},SVector{E,T},Tuple{Int,Int}}}}
    # [[(ptr, r, dr, (norbs, norbs´)), ...], ...] for each selected hop on each harmonic
end

struct AppliedHoppingModifier{T,E,L,O}
    f::FunctionWrapper{O,Tuple{O,SVector{E,T},SVector{E,T},Tuple{Int,Int}}}
    # t(old_t, r, dr, (orbs1, orbs2))
    ptrs::Vector{Vector{Tuple{Int,SVector{E,T},SVector{E,T},Tuple{Int,Int}}}}
    # [[(ptr, r, dr, (norbs, norbs´)), ...], ...] for each selected hop on each harmonic
end

const Modifier = Union{OnsiteModifier,HoppingModifier}
const PartiallyAppliedModifier = Union{PartiallyAppliedOnsiteModifier,PartiallyAppliedHoppingModifier}
const AppliedModifier = Union{AppliedOnsiteModifier,AppliedHoppingModifier}

#region Modifier internal API

selector(m::Modifier) = m.selector

parameters(m::Union{Modifier,PartiallyAppliedModifier}) = m.f.params

parametric_function(m::Union{Modifier,PartiallyAppliedModifier,AppliedModifier}) = m.f

pointers(m::Union{PartiallyAppliedModifier,AppliedModifier}) = m.ptrs

(m::PartiallyAppliedOnsiteModifier{1})(o, r; kw...) = m.f.f(o; kw...)
(m::PartiallyAppliedOnsiteModifier{2})(o, r; kw...) = m.f.f(o, r; kw...)

(m::AppliedOnsiteModifier)(o, r, orbs) = m.f(o, r, orbs)

(m::PartiallyAppliedHoppingModifier{1})(t, r, dr; kw...) = m.f.f(t; kw...)
(m::PartiallyAppliedHoppingModifier{3})(t, r, dr; kw...) = m.f.f(t, r, dr; kw...)

(m::AppliedHoppingModifier)(t, r, dr, orbs) = m.f(t, r, dr, orbs)

#endregion
#endregion

############################################################################################
# OrbitalStructure
#region

struct OrbitalStructure{O<:Union{Number,SMatrix}}
    blocktype::Type{O}    # Hamiltonian's blocktype
    norbitals::Vector{Int}
    offsets::Vector{Int}
end

#region internal API

norbitals(o::OrbitalStructure) = o.norbitals

orbtype(::OrbitalStructure{O}) where {O<:Number} = O
orbtype(::OrbitalStructure{O}) where {N,T,O<:SMatrix{N,N,T}} = SVector{N,T}

blocktype(o::OrbitalStructure) = o.blocktype

offsets(o::OrbitalStructure) = o.offsets

nsites(o::OrbitalStructure) = last(offsets(o))

#endregion
#endregion

############################################################################################
# Hamiltonian
#region

# Any Hamiltonian that can be passed to `flatten` and `bloch`
abstract type AbstractHamiltonian{T,E,L,O} end

struct HamiltonianHarmonic{L,O}
    dn::SVector{L,Int}
    h::SparseMatrixCSC{O,Int}
end

struct Hamiltonian{T,E,L,O} <: AbstractHamiltonian{T,E,L,O}
    lattice::Lattice{T,E,L}
    orbstruct::OrbitalStructure{O}
    harmonics::Vector{HamiltonianHarmonic{L,O}}
    # Enforce sorted-dns-starting-from-zero invariant onto harmonics
    function Hamiltonian{T,E,L,O}(lattice, orbstruct, harmonics) where {T,E,L,O}
        n = nsites(lattice)
        all(har -> size(matrix(har)) == (n, n), harmonics) ||
            throw(DimensionMismatch("Harmonic $(size.(matrix.(harmonics), 1)) sizes don't match number of sites $n"))
        sort!(harmonics)
        length(harmonics) > 0 && iszero(dcell(first(harmonics))) || pushfirst!(harmonics,
            HamiltonianHarmonic(zero(SVector{L,Int}), sparse(Int[], Int[], O[], n, n)))
        return new(lattice, orbstruct, harmonics)
    end
end

Hamiltonian(l::Lattice{T,E,L}, o::OrbitalStructure{O}, h::Vector{HamiltonianHarmonic{L,O}},) where {T,E,L,O} =
    Hamiltonian{T,E,L,O}(l, o, h)

#region internal API

hamiltonian(h::Hamiltonian) = h

matrix(h::HamiltonianHarmonic) = h.h

dcell(h::HamiltonianHarmonic) = h.dn

orbitalstructure(h::Hamiltonian) = h.orbstruct

lattice(h::Hamiltonian) = h.lattice

harmonics(h::Hamiltonian) = h.harmonics

orbtype(h::Hamiltonian) = orbtype(orbitalstructure(h))

blocktype(h::Hamiltonian) = blocktype(orbitalstructure(h))

norbitals(h::Hamiltonian) = norbitals(orbitalstructure(h))

Base.size(h::HamiltonianHarmonic, i...) = size(matrix(h), i...)
Base.size(h::Hamiltonian, i...) = size(first(harmonics(h)), i...)

Base.isless(h::HamiltonianHarmonic, h´::HamiltonianHarmonic) = sum(abs2, dcell(h)) < sum(abs2, dcell(h´))

copy_harmonics(h::Hamiltonian) = Hamiltonian(lattice(h), orbitalstructure(h), deepcopy(harmonics(h)))

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
# Parametric
#region

struct ParametricHamiltonian{T,E,L,O,M<:NTuple{<:Any,PartiallyAppliedModifier}} <: AbstractHamiltonian{T,E,L,O}
    hparent::Hamiltonian{T,E,L,O}
    h::Hamiltonian{T,E,L,O}
    modifiers::M                   # Tuple of PartiallyAppliedModifier's (unwrapped until kwargs are known)
    allptrs::Vector{Vector{Int}}   # allptrs are all modified ptrs in each harmonic (needed for reset!)
    allparams::Vector{Symbol}
end

Base.parent(h::ParametricHamiltonian) = h.hparent

hamiltonian(h::ParametricHamiltonian) = h.h

parameters(h::ParametricHamiltonian) = h.allparams

modifiers(h::ParametricHamiltonian) = h.modifiers

pointers(h::ParametricHamiltonian) = h.allptrs

harmonics(h::ParametricHamiltonian) = harmonics(parent(h))

orbitalstructure(h::ParametricHamiltonian) = orbitalstructure(parent(h))

blocktype(h::ParametricHamiltonian) = blocktype(parent(h))

lattice(h::ParametricHamiltonian) = lattice(parent(h))

Base.size(h::ParametricHamiltonian, i...) = size(parent(h), i...)

#endregion

############################################################################################
# Flat
#region

abstract type AbstractFlatHamiltonian{T,E,L,O} <: AbstractHamiltonian{T,E,L,O} end

struct FlatHamiltonian{T,E,L,O<:Number,H<:Hamiltonian{T,E,L,<:SMatrix}} <: AbstractFlatHamiltonian{T,E,L,O}
    h::H
    flatorbstruct::OrbitalStructure{O}
end

orbitalstructure(h::AbstractFlatHamiltonian) = h.flatorbstruct

unflatten(h::AbstractFlatHamiltonian) = parent(h)

lattice(h::AbstractFlatHamiltonian) = lattice(parent(h))

harmonics(h::AbstractFlatHamiltonian) = harmonics(parent(h))

orbtype(h::AbstractFlatHamiltonian) = orbtype(orbitalstructure(h))

blocktype(h::AbstractFlatHamiltonian) = blocktype(orbitalstructure(h))

norbitals(h::AbstractFlatHamiltonian) = norbitals(orbitalstructure(h))

Base.size(h::AbstractFlatHamiltonian) = nsites(orbitalstructure(h)), nsites(orbitalstructure(h))
Base.size(h::AbstractFlatHamiltonian, i) = i <= 0 ? throw(BoundsError()) : ifelse(1 <= i <= 2, nsites(orbitalstructure(h)), 1)

Base.parent(h::AbstractFlatHamiltonian) = h.h

#endregion

############################################################################################
# Bloch
#region

struct Bloch{L,O,O´,H<:AbstractHamiltonian{<:Any,<:Any,L,O´}}
    h::H
    output::SparseMatrixCSC{O,Int}  # output has same structure as merged harmonics(h)
end                                 # or its flattened version if O != O´

Base.parent(b::Bloch) = b.h

#endregion