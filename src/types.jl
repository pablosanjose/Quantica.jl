#######################################################################
# Lattice
#######################################################################
struct Sublat{T,E}
    sites::Vector{SVector{E,T}}
    name::Symbol
end

struct Unitcell{T,E}
    sites::Vector{SVector{E,T}}
    names::Vector{Symbol}
    offsets::Vector{Int}        # Linear site number offsets for each sublat
end

struct Bravais{T,E,L}
    vectors::NTuple{L,SVector{E,T}}
    function Bravais{T,E,L}(vectors) where {T,E,L}
        L > E &&
            throw(DimensionMismatch("Number $L of Bravais vectors cannot be greater than embedding dimension $E"))
        return new(vectors)
    end
end
# outer constructor
Bravais(vectors::NTuple{L,SVector{E,T}}) where {T,E,L} = Bravais{T,E,L}(vectors)

mutable struct Lattice{T,E,L}
    bravais::Bravais{T,E,L}
    unitcell::Unitcell{T,E}
end

#region internal API

unitcell(l::Lattice) = l.unitcell

bravais_vectors(b::Bravais) = b.vectors
bravais_vectors(l::Lattice) = bravais_vectors(l.bravais)

bravais_matrix(b::Bravais) = hcat(b.vectors...)
bravais_matrix(l::Lattice) = bravais_matrix(l.bravais)

nsublats(l::Lattice) = nsublats(l.unitcell)
nsublats(u::Unitcell) = length(u.names)

nsites(s::Sublat) = length(s.sites)
nsites(lat::Lattice, sublat...) = nsites(lat.unitcell, sublat...)
nsites(u::Unitcell) = length(u.sites)
nsites(u::Unitcell, sublat) = sublatlengths(u)[sublat]

sites(l::Lattice, sublat...) = sites(l.unitcell, sublat...)
sites(u::Unitcell) = u.sites
sites(u::Unitcell, sublat) = view(u.sites, u.offsets[i]+1, u.offsets[i+1])
sites(s::Sublat) = s.sites

names(l::Lattice) = names(l.unitcell)
names(u::Unitcell) = u.names
name(s::Sublat) = s.name

offsets(u::Unitcell) = u.offsets

sublatlengths(lat::Lattice) = sublatlengths(lat.unitcell)
sublatlengths(u::Unitcell) = diff(u.offsets)

#endregion

#######################################################################
# Selectors
#######################################################################
abstract type Selector end
abstract type UnresolvedSelector <: Selector end
abstract type ResolvedSelector <: Selector  end

struct NeighborRange
    n::Int
end

struct Not{T} # Symbolizes excluded elements
    i::T
end

struct UnresolvedSiteSelector <: UnresolvedSelector
    region
    sublats  # Collection of Symbol
    indices
end

struct UnresolvedHopSelector <: UnresolvedSelector
    region
    sublats  # Collection of Pair{Symbol,Symbol}
    dcells
    range
    indices
end

struct ResolvedSiteSelector{T,E,L,F} <: ResolvedSelector
    region::F
    sublats::Vector{Int}
    indices::Vector{Int}  # negative index -i is equivalent to Not(i)
    lattice::Lattice{T,E,L}
end

struct ResolvedHopSelector{T,E,L,F} <: ResolvedSelector
    region::F
    sublats::Vector{Pair{Int,Int}}
    indices::Vector{Pair{Int,Int}}  # negative index -i is equivalent to Not(i)
    dcells::Vector{SVector{L,Int}}
    range::T
    lattice::Lattice{T,E,L}
end

const SiteSelector = Union{UnresolvedSiteSelector,ResolvedSiteSelector}
const HopSelector = Union{UnresolvedHopSelector,ResolvedHopSelector}
const ElementSelector = Union{SiteSelector,HopSelector}

struct BlockSelector{V<:Union{Missing,Vector}} <: Selector
    dcells::V
    rows::Vector{Int}
    cols::Vector{Int}
end

#######################################################################
# Model
#######################################################################
abstract type TightbindingModelTerm end
abstract type AbstractOnsiteTerm <: TightbindingModelTerm end
abstract type AbstractHoppingTerm <: TightbindingModelTerm end

struct TightbindingModel
    terms  # Collection of `TightbindingModelTerm`s
end

# These need to be concrete as they are involved in hot construction loops
struct OnsiteTerm{F,S,T} <: AbstractOnsiteTerm
    o::F
    selector::S
    coefficient::T
end

struct HoppingTerm{F,S,T} <: AbstractHoppingTerm
    t::F
    selector::S
    coefficient::T
end

#######################################################################
# Modifiers
#######################################################################
struct ParametricFunction{N,F}
    f::F
    params::Vector{Symbol}
end

struct Modifier{N,S<:Selector,F}
    f::ParametricFunction{N,F}
    selector::S
end

const ElementModifier{N,S<:ElementSelector,F} = Modifier{N,S,F}
const HopModifier{N,S<:Union{HopSelector,ResolvedHopSelector},F} = Modifier{N,S,F}
const SiteModifier{N,S<:Union{SiteSelector,ResolvedSiteSelector},F} = Modifier{N,S,F}
const BlockModifier{N,S<:BlockSelector,F} = Modifier{N,S,F}
const UniformModifier = ElementModifier{1}
const UniformHopModifier = HopModifier{1}
const UniformSiteModifier = SiteModifier{1}