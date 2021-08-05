#######################################################################
# Lattice
#######################################################################
struct Sublat{T,E}
    sites::Vector{SVector{E,T}}
    name::Symbol
end

struct Unitcell{T<:AbstractFloat,E}
    sites::Vector{SVector{E,T}}
    names::Vector{Symbol}
    offsets::Vector{Int}        # Linear site number offsets for each sublat
end

struct Bravais{T,E,L}
    matrix::Matrix
    function Bravais{T,E,L}(matrix) where {T,E,L}
        (E, L) == size(matrix) || throw(ErrorException("Internal error: unexpected matrix size $((E,L)) != $(size(matrix))"))
        L > E &&
            throw(DimensionMismatch("Number $L of Bravais vectors cannot be greater than embedding dimension $E"))
        return new(matrix)
    end
end

mutable struct Lattice{T<:AbstractFloat,E,L}
    bravais::Bravais{T,E,L}
    unitcell::Unitcell{T,E}
end

#region internal API

unitcell(l::Lattice) = l.unitcell

bravais_vectors(b::Bravais) = eachcol(b.matrix)
bravais_vectors(l::Lattice) = bravais_vectors(l.bravais)

bravais_matrix(b::Bravais{T,E,L}) where {T,E,L} = convert(SMatrix{E,L,T}, b.matrix)
bravais_matrix(l::Lattice) = bravais_matrix(l.bravais)

names(l::Lattice) = names(l.unitcell)
names(u::Unitcell) = u.names
name(s::Sublat) = s.name

nsublats(l::Lattice) = nsublats(l.unitcell)
nsublats(u::Unitcell) = length(u.names)

sublats(l::Lattice) = sublats(l.unitcell)
sublats(u::Unitcell) = 1:nsublats(u)

nsites(s::Sublat) = length(s.sites)
nsites(lat::Lattice, sublat...) = nsites(lat.unitcell, sublat...)
nsites(u::Unitcell) = length(u.sites)
nsites(u::Unitcell, sublat) = sublatlengths(u)[sublat]

sites(l::Lattice, sublat...) = sites(l.unitcell, sublat...)
sites(u::Unitcell) = u.sites
sites(u::Unitcell, sublat) = view(u.sites, u.offsets[sublat]+1, u.offsets[sublat+1])
sites(s::Sublat) = s.sites

siterange(l::Lattice, sublat) = siterange(l.unitcell, sublat)
siterange(u::Unitcell, sublat) = (1+u.offsets[sublat]):u.offsets[sublat+1]

siteindsublats(l::Lattice) = siteindsublats(l.unitcell)
siteindsublats(u::Unitcell) = TypedGenerator{Tuple{Int,Int}}(
    ((i, s) for s in sublats(u) for i in siterange(u, s)), nsites(u))

offsets(u::Unitcell) = u.offsets

sublatlengths(lat::Lattice) = sublatlengths(lat.unitcell)
sublatlengths(u::Unitcell) = diff(u.offsets)

valdim(::Sublat{<:Any,E}) where {E} = Val(E)
valdim(::Lattice{<:Any,E}) where {E} = Val(E)

latdim(::Lattice{<:Any,<:Any,L}) where {L} = L

numbertype(::Sublat{T}) where {T} = T
numbertype(::Lattice{T}) where {T} = T

sitesublat(siteidx, lat::Lattice) = sublat_site(siteidx, lat.unitcell.offsets)

function sitesublat(siteidx, offsets)
    l = length(offsets)
    for s in 2:l
        @inbounds offsets[s] + 1 > siteidx && return s - 1
    end
    return l
end

#endregion

#######################################################################
# Selectors
#region

abstract type Selector end

struct SiteSelector{S,I,M} <: Selector
    region::M
    sublats::S  # NTuple{N,Symbol} (unresolved) or Vector{Int} (resolved on a lattice)
    indices::I  # Once resolved, this should be an Union{Integer,Not} container
end

struct HopSelector{S,I,D,T,M} <: Selector
    region::M
    sublats::S  # NTuple{N,Pair{Symbol,Symbol}} (unres) or Vector{Pair{Int,Int}} (res)
    dns::D
    range::T
    indices::I  # Once resolved, this should be a Pair{Int,Int} container
end

struct BlockSelector{V<:Union{Missing,Vector}} <: Selector
    cells::V
    rows::Vector{Int}
    cols::Vector{Int}
end

struct ResolvedSelector{S<:Selector,L<:Lattice}
    selector::S
    lattice::L
end

struct Not{T} # Symbolizes an excluded elements
    i::T
end

#region internal API

Base.parent(n::Not) = n.i

#endregion
#endregion

# #######################################################################
# # Model
# #######################################################################
# abstract type TightbindingModelTerm end
# abstract type AbstractOnsiteTerm <: TightbindingModelTerm end
# abstract type AbstractHoppingTerm <: TightbindingModelTerm end

# struct TightbindingModel
#     terms  # Collection of `TightbindingModelTerm`s
# end

# # These need to be concrete as they are involved in hot construction loops
# struct OnsiteTerm{F,S,T} <: AbstractOnsiteTerm
#     o::F
#     selector::S
#     coefficient::T
# end

# struct HoppingTerm{F,S,T} <: AbstractHoppingTerm
#     t::F
#     selector::S
#     coefficient::T
# end

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