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

bravais_vecs(b::Bravais) = eachcol(b.matrix)
bravais_vecs(l::Lattice) = bravais_vecs(l.bravais)

bravais_mat(b::Bravais{T,E,L}) where {T,E,L} = convert(SMatrix{E,L,T}, b.matrix)
bravais_mat(l::Lattice) = bravais_mat(l.bravais)

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

sites(l::Lattice, sublat...) = sites(l.unitcell, sublat...)
sites(u::Unitcell) = u.sites
sites(u::Unitcell, sublat) = view(u.sites, u.offsets[sublat]+1, u.offsets[sublat+1])
sites(s::Sublat) = s.sites

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

sitesubiter(l::Lattice) = sitesubiter(l.unitcell)
sitesubiter(u::Unitcell) = TypedGenerator{Tuple{Int,Int}}(
    ((i, s) for s in sublats(u) for i in siterange(u, s)), nsites(u))

offsets(u::Unitcell) = u.offsets

sublatlengths(lat::Lattice) = sublatlengths(lat.unitcell)
sublatlengths(u::Unitcell) = diff(u.offsets)

valdim(::Sublat{<:Any,E}) where {E} = Val(E)
valdim(::Lattice{<:Any,E}) where {E} = Val(E)

latdim(::Lattice{<:Any,<:Any,L}) where {L} = L

numbertype(::Sublat{T}) where {T} = T
numbertype(::Lattice{T}) where {T} = T

celltype(::Lattice{<:Any,<:Any,L}) where {L} = SVector{L,Int}

#endregion

#######################################################################
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
    dns::D
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

struct Applied{S,D}
    src::S
    dst::D
end

#region internal API

Base.parent(n::NeighborRange) = n.n

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