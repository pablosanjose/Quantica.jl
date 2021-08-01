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
end

abstract type AbstractLattice{T<:AbstractFloat,E,L} end

mutable struct Lattice{T,E,L} <: AbstractLattice{T,E,L}
    bravais::Bravais{T,E,L}
    unitcell::Unitcell{T,E}
end

struct Supercell{L,L´,O<:OffsetArray} # L,L´ are lattice/superlattice dims
    vectors::NTuple{L´,SVector{L,Int}}
    sites::UnitRange{Int}
    cells::CartesianIndices{L,NTuple{L,UnitRange{Int}}}
    mask::O  # Dimensions of O is L + 1
end

struct Superlattice{T,E,L,S<:Supercell} <: AbstractLattice{T,E,L}
    bravais::Bravais{T,E,L}
    unitcell::Unitcell{T,E}
    supercell::S
end

#region Constructors
sublat(sites::Vector{<:SVector}; name = :_, kw...) =
    Sublat(sites, Symbol(name))
sublat(sites...; kw...) =
    sublat(sanitize_Vector_of_SVectors(sites); kw...)

#endregion

#region internal API

bravais(b::Bravais) = hcat(b.vectors...)
bravais(l::AbstractLattice) = bravais(l.bravais)

nsublats(l::AbstractLattice) = nsublats(l.unitcell)
nsublats(u::Unitcell) = length(u.names)

nsites(s::Sublat) = length(s.sites)
nsites(lat::Lattice, sublat...) = nsites(lat.unitcell, sublat...)
nsites(u::Unitcell) = length(u.sites)
nsites(u::Unitcell, sublat) = sublatlengths(u)[sublat]
nsites(slat::Superlattice, sublats...) = nsites(slat.supercel, sublats...)
nsites(s::Supercell) = sum(s.mask)
nsites(s::Supercell{L}, sublat) where {L} = sum(view(s.mask, sublat, ntuple(Returns(:), Val(L))...))

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