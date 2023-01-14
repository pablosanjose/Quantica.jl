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
# LatticeSlice - see lattice.jl for methods
#   Encodes subsets of sites of a lattice in different cells. Produced by lat[siteselector]
#   No ordering of cells or sites indices is guaranteed, but both must be unique
#region

struct Subcell{L}
    inds::Vector{Int}
    cell::SVector{L,Int}
end

struct LatticeSlice{T,E,L}
    lat::Lattice{T,E,L}
    subcells::Vector{Subcell{L}}
end

#region ## Constructors ##

Subcell(cell) = Subcell(Int[], cell)

LatticeSlice(lat::Lattice{<:Any,<:Any,L}) where {L} = LatticeSlice(lat, Subcell{L}[])

#endregion

#region ## API ##

siteindices(s::Subcell) = s.inds

cell(s::Subcell) = s.cell

subcells(l::LatticeSlice) = l.subcells

cells(l::LatticeSlice) = (s.cell for s in l.subcells)

nsites(l::LatticeSlice) = isempty(l) ? 0 : sum(nsites, subcells(l))
nsites(s::Subcell) = length(s.inds)

boundingbox(l::LatticeSlice) = boundingbox(cell(c) for c in subcells(l))

sites(l::LatticeSlice) =
    (site(l.lat, i, cell(subcell)) for subcell in subcells(l) for i in siteindices(subcell))

Base.parent(ls::LatticeSlice) = ls.lat

Base.isempty(s::LatticeSlice) = isempty(s.subcells)
Base.isempty(s::Subcell) = isempty(s.inds)

Base.empty!(s::Subcell) = empty!(s.inds)

Base.push!(s::Subcell, i::Int) = push!(s.inds, i)
Base.push!(ls::LatticeSlice, s::Subcell) = push!(ls.subcells, s)

Base.copy(s::Subcell) = Subcell(copy(s.inds), s.cell)

# Unused?
# function Base.intersect!(ls::L, ls´::L) where {L<:LatticeSlice}
#     for subcell in subcells(ls)
#         found = false
#         for subcell´ in subcells(ls´)
#             if cell(subcell) == cell(subcell´)
#                 intersect!(siteindices(subcell), siteindices(subcell´))
#                 found = true
#                 break
#             end
#         end
#         found || empty!(subcell)
#     end
#     deleteif!(isempty, subcells(ls))
#     return ls
# end

# function subcellind(l::LatticeSlice, iunflat)
#     counter = 0
#     for (nc, scell) in enumerate(subcells(l))
#         ns = nsites(scell)
#         if counter + ns < iunflat
#             counter += ns
#         else
#             return (nc, iunflat - counter)
#         end
#     end
#     @boundscheck(throw(BoundsError(l, iunflat)))
# end

function Base.getindex(l::LatticeSlice{<:Any,<:Any,L}, i::Integer) where {L}
    offset = 0
    for scell in subcells(l)
        ninds = length(siteindices(scell))
        if ninds + offset < i
            offset += ninds
        else
            return cell(scell), siteindices(scell)[i-offset]
        end
    end
    @boundscheck(throw(BoundsError(l, i)))
end

Base.length(l::LatticeSlice) = nsites(l)
Base.length(s::Subcell) = nsites(s)

#endregion
#endregion

############################################################################################
# Models  -  see model.jl for methods
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
end

struct ParametricHoppingTerm{N,S<:Union{HopSelector,AppliedHopSelector},F<:ParametricFunction{N},T<:Number} <: AbstractModelTerm
    f::F
    selector::S
    coefficient::T
end

const AbstractParametricTerm{N} = Union{ParametricOnsiteTerm{N},ParametricHoppingTerm{N}}

struct ParametricModel{T<:NTuple{<:Any,AbstractParametricTerm}} <: AbstractModel
    terms::T  # Collection of `AbstractParametricTerm`s
end

const AppliedParametricTerm{N} = Union{ParametricOnsiteTerm{N,<:AppliedSiteSelector},
                                       ParametricHoppingTerm{N,<:AppliedHopSelector}}
const AppliedParametricModel = ParametricModel{<:NTuple{<:Any,AppliedParametricTerm}}

#region ## Constructors ##

ParametricFunction{N}(f::F, params = Symbol[]) where {N,F} =
    ParametricFunction{N,F}(f, params)

TightbindingModel(ts::AbstractTightbindingTerm...) = TightbindingModel(ts)
ParametricModel(ts::AbstractParametricTerm...) = ParametricModel(ts)

OnsiteTerm(t::OnsiteTerm, os::SiteSelector) = OnsiteTerm(t.f, os, t.coefficient)
ParametricOnsiteTerm(t::ParametricOnsiteTerm, os::SiteSelector) =
    ParametricOnsiteTerm(t.f, os, t.coefficient)

HoppingTerm(t::HoppingTerm, os::HopSelector) = HoppingTerm(t.f, os, t.coefficient)
ParametricHoppingTerm(t::ParametricHoppingTerm, os::HopSelector) =
    ParametricHoppingTerm(t.f, os, t.coefficient)

#endregion

#region ## API ##

terms(t::AbstractModel) = t.terms

selector(t::AbstractModelTerm) = t.selector

functor(t::AbstractModelTerm) = t.f

parameters(t::AbstractParametricTerm) = t.f.params

coefficient(t::OnsiteTerm) = t.coefficient
coefficient(t::HoppingTerm) = t.coefficient
coefficient(t::AbstractParametricTerm) = t.coefficient

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

# We need these for SelfEnergyModel, which uses a ParametricModel. We return a
# ParametricOnsiteTerm, not an OnsiteTerm because the latter is tied to a Hamiltonian at its
# orbital structure, not only to a site selection
function (t::ParametricOnsiteTerm{N})(; kw...) where {N}
    f = ParametricFunction{N}((args...) -> t.f(args...; kw...)) # no params
    return ParametricOnsiteTerm(f, t.selector, t.coefficient)
end

function (t::ParametricHoppingTerm{N})(; kw...) where {N}
    f = ParametricFunction{N}((args...) -> t.f(args...; kw...)) # no params
    return ParametricHoppingTerm(f, t.selector, t.coefficient)
end

# Model term algebra

Base.:*(x::Number, m::TightbindingModel) = TightbindingModel(x .* terms(m))
Base.:*(x::Number, m::ParametricModel) = ParametricModel(x .* terms(m))
Base.:*(m::AbstractModel, x::Number) = x * m
Base.:-(m::AbstractModel) = (-1) * m

Base.:+(m::TightbindingModel, m´::TightbindingModel) = TightbindingModel((terms(m)..., terms(m´)...))
Base.:+(m::AbstractModel, m´::AbstractModel) = ParametricModel((terms(m)..., terms(m´)...))
Base.:-(m::AbstractModel, m´::AbstractModel) = m + (-m´)

Base.:*(x::Number, o::OnsiteTerm) = OnsiteTerm(o.f, o.selector, x * o.coefficient)
Base.:*(x::Number, t::HoppingTerm) = HoppingTerm(t.f, t.selector, x * t.coefficient)
Base.:*(x::Number, o::ParametricOnsiteTerm) =
    ParametricOnsiteTerm(o.f, o.selector, x * o.coefficient)
Base.:*(x::Number, t::ParametricHoppingTerm) =
    ParametricHoppingTerm(t.f, t.selector, x * t.coefficient)

Base.adjoint(m::TightbindingModel) = TightbindingModel(adjoint.(terms(m)))
Base.adjoint(m::ParametricModel) = ParametricModel(adjoint.(terms(m)))
Base.adjoint(t::OnsiteTerm{<:Function}) = OnsiteTerm(r -> t.f(r)', t.selector, t.coefficient')
Base.adjoint(t::OnsiteTerm) = OnsiteTerm(t.f', t.selector, t.coefficient')
Base.adjoint(t::HoppingTerm{<:Function}) = HoppingTerm((r, dr) -> t.f(r, -dr)', t.selector', t.coefficient')
Base.adjoint(t::HoppingTerm) = HoppingTerm(t.f', t.selector', t.coefficient')

function Base.adjoint(o::ParametricOnsiteTerm{N}) where {N}
    f = ParametricFunction{N}((args...; kw...) -> o.f(args...; kw...)', o.f.params)
    return ParametricOnsiteTerm(f, o.selector, o.coefficient')
end

function Base.adjoint(t::ParametricHoppingTerm{N}) where {N}
    f = ParametricFunction{N}((args...; kw...) -> t.f(args...; kw...)', t.f.params)
    return ParametricHoppingTerm(f, t.selector, t.coefficient')
end

#endregion
#endregion

############################################################################################
# Model Modifiers  -  see model.jl for methods
#region

abstract type AbstractModifier end

struct OnsiteModifier{N,S<:SiteSelector,F<:ParametricFunction{N}} <: AbstractModifier
    f::F
    selector::S
end

struct AppliedOnsiteModifier{B,N,R<:SVector,F<:ParametricFunction{N}} <: AbstractModifier
    blocktype::Type{B}  # These are needed to cast the modification to the sublat block type
    f::F
    ptrs::Vector{Tuple{Int,R,Int}}
    # [(ptr, r, norbs)...] for each selected site, dn = 0 harmonic
end

struct HoppingModifier{N,S<:HopSelector,F<:ParametricFunction{N}} <: AbstractModifier
    f::F
    selector::S
end

struct AppliedHoppingModifier{B,N,R<:SVector,F<:ParametricFunction{N}} <: AbstractModifier
    blocktype::Type{B}  # These are needed to cast the modification to the sublat block type
    f::F
    ptrs::Vector{Vector{Tuple{Int,R,R,Tuple{Int,Int}}}}
    # [[(ptr, r, dr, (norbs, norbs´)), ...], ...] for each selected hop on each harmonic
end

const Modifier = Union{OnsiteModifier,HoppingModifier}
const AppliedModifier = Union{AppliedOnsiteModifier,AppliedHoppingModifier}

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
# Harmonic  -  see hamiltonian.jl for methods
#region

struct Harmonic{T,L,B}
    dn::SVector{L,Int}
    h::HybridSparseBlochMatrix{T,B}
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

const AbstractHamiltonian0D{T,E,B} = AbstractHamiltonian{T,E,0,B}
const AbstractHamiltonian1D{T,E,B} = AbstractHamiltonian{T,E,1,B}

struct Hamiltonian{T,E,L,B} <: AbstractHamiltonian{T,E,L,B}
    lattice::Lattice{T,E,L}
    blockstruct::SublatBlockStructure{B}
    harmonics::Vector{Harmonic{T,L,B}}
    bloch::HybridSparseBlochMatrix{T,B}
    # Enforce sorted-dns-starting-from-zero invariant onto harmonics
    function Hamiltonian{T,E,L,B}(lattice, blockstruct, harmonics, bloch) where {T,E,L,B}
        n = nsites(lattice)
        all(har -> size(matrix(har)) == (n, n), harmonics) ||
            throw(DimensionMismatch("Harmonic $(size.(matrix.(harmonics), 1)) sizes don't match number of sites $n"))
        sort!(harmonics)
        (isempty(harmonics) || !iszero(dcell(first(harmonics)))) && pushfirst!(harmonics,
            Harmonic(zero(SVector{L,Int}), HybridSparseBlochMatrix(blockstruct, spzeros(B, n, n))))
        return new(lattice, blockstruct, harmonics, bloch)
    end
end

#region ## API ##

## AbstractHamiltonian

norbitals(h::AbstractHamiltonian) = blocksizes(blockstructure(h))

blockeltype(::AbstractHamiltonian) = blockeltype(blockstructure(h))

blocktype(h::AbstractHamiltonian) = blocktype(blockstructure(h))

flatsize(h::AbstractHamiltonian) = flatsize(blockstructure(h))

# see specialmatrices.jl
flatrange(h::AbstractHamiltonian, iunflat) = flatrange(blockstructure(h), iunflat)

## Hamiltonian

Hamiltonian(l::Lattice{T,E,L}, b::SublatBlockStructure{B}, h::Vector{Harmonic{T,L,B}}, bl) where {T,E,L,B} =
    Hamiltonian{T,E,L,B}(l, b, h, bl)

function Hamiltonian(l, b::SublatBlockStructure{B}, h) where {B}
    n = nsites(l)
    bloch = HybridSparseBlochMatrix(b, spzeros(B, n, n))
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
    blockstruct::SublatBlockStructure{B}
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
# Bands -  see spectrum.jl for methods
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
# SelfEnergy solvers - see selfenergy.jl for self-energy solvers
#   Any new s::AbstractSelfEnergySolver is associated to some forms of attach(g, sargs...)
#   For each such form we must add a SelfEnergy constructor that will be used by attach
#     - SelfEnergy(h::AbstractHamiltonian, sargs...; siteselect...) -> SelfEnergy
#   This wraps the s::AbstractSelfEnergySolver that should support the call! API
#     - call!(s::RegularSelfEnergySolver, ω; params...) -> Σreg::MatrixBlock
#     - call!(s::ExtendedSelfEnergySolver, ω; params...) -> (Σᵣᵣ, Vᵣₑ, gₑₑ⁻¹, Vₑᵣ) MatrixBlock's
#         With the extended case, the equivalent Σreg reads Σreg = Σᵣᵣ + VᵣₑgₑₑVₑᵣ
#     - call!_output(s::AbstractSelfEnergySolver) -> object returned by call!(s, ω; params...)
#region

abstract type AbstractSelfEnergySolver end
abstract type RegularSelfEnergySolver <: AbstractSelfEnergySolver end
abstract type ExtendedSelfEnergySolver <: AbstractSelfEnergySolver end

# Support for call!(s::AbstractSelfEnergySolver; params...) -> AbstractSelfEnergySolver

struct WrappedRegularSelfEnergySolver{F} <: RegularSelfEnergySolver
    f::F
end

struct WrappedExtendedSelfEnergySolver{F} <: ExtendedSelfEnergySolver
    f::F
end

call!(s::WrappedRegularSelfEnergySolver, ω; params...) = s.f(ω)
call!(s::WrappedExtendedSelfEnergySolver, ω; params...) = s.f(ω)

call!(s::RegularSelfEnergySolver; params...) =
    WrappedRegularSelfEnergySolver(ω -> call!(s, ω; params...))
call!(s::ExtendedSelfEnergySolver; params...) =
    WrappedExtendedSelfEnergySolver(ω -> call!(s, ω; params...))
call!(s::WrappedRegularSelfEnergySolver; params...) = s
call!(s::WrappedExtendedSelfEnergySolver; params...) = s

#endregion

############################################################################################
# SelfEnergy - see selfenergy.jl
#   Contacts is a collection of SelfEnergy's finalized to be wrapped into a GreenFunction
#region

struct SelfEnergy{T,E,L,S<:AbstractSelfEnergySolver}
    solver::S                                        # returns MatrixBlock over latslice
    latslice::LatticeSlice{T,E,L}
end

#region ## API ##

latslice(c::SelfEnergy) = c.latslice

solver(c::SelfEnergy) = c.solver

call!(Σ::SelfEnergy; params...) = SelfEnergy(call!(Σ.solver; params...), Σ.latslice)
call!(Σ::SelfEnergy, ω; params...) = call!(Σ.solver, ω; params...) # returns a MatrixBlock

#endregion
#endregion

############################################################################################
# OpenHamiltonian
#region

struct OpenHamiltonian{T,E,L,H<:AbstractHamiltonian{T,E,L},S<:NTuple{<:Any,SelfEnergy{T,E,L}}}
    h::H
    selfenergies::S
end

#region ## API ##

hamiltonian(oh::OpenHamiltonian) = oh.h

selfenergies(oh::OpenHamiltonian) = oh.selfenergies

attach(Σ::SelfEnergy) = oh -> attach(oh, Σ)
attach(args...; kw...) = oh -> attach(oh, args...; kw...)
attach(oh::OpenHamiltonian, args...; kw...) = attach(oh, SelfEnergy(oh.h, args...; kw...))
attach(oh::OpenHamiltonian, Σ::SelfEnergy) = OpenHamiltonian(oh.h, (oh.selfenergies..., Σ))
attach(h::AbstractHamiltonian, args...; kw...) = attach(h, SelfEnergy(h, args...; kw...))
attach(h::AbstractHamiltonian, Σ::SelfEnergy) = OpenHamiltonian(h, (Σ,))

#endregion
#endregion

############################################################################################
# Contacts
#region

struct Contacts{T,E,L,S<:NTuple{<:Any,SelfEnergy}}
    selfenergies::S                       # used to produce MatrixBlock's
    latsliceall::LatticeSlice{T,E,L}      # merged latslice for all self-energies
    blockstruct::MultiBlockStructure{L}   # needed to build HybridMatrix'es over latsliceall
end

#region ## Constructors ##

function Contacts(oh::OpenHamiltonian)
    Σs = selfenergies(oh)
    Σlatslices = latslice.(Σs)
    latsliceall = merge(Σlatslices...)
    bs = MultiBlockStructure(latsliceall, hamiltonian(oh))
    return Contacts(Σs, latsliceall, bs)
end

#endregion

#region ## API ##

latslice(c::Contacts) = c.latsliceall

selfenergies(c::Contacts) = c.selfenergies

blockstruct(c::Contacts) = c.blockstruct

call!(c::Contacts, ω; params...) = call!.(c.selfenergies, Ref(ω); params...)
call!(c::Contacts; params...) =
    Contacts(call!.(c.selfenergies; params...), c.mergedlatslice, c.blockstruct)

#endregion

#endregion

############################################################################################
# Green solvers - see solvers/greensolvers.jl
#   Any new S::AbstractGreenSolver must implement
#     - apply(s, h::OpenHamiltonian) -> AppliedGreenSolver
#   Any new s::AppliedGreenSolver must implement
#     - call!(s; params...) -> AppliedGreenSolver
#     - call!(s, h, contacts, ω; params...) -> GreenMatrix
#     This GreenMatrix provides in particular:
#        - GreenMatrixSolver to compute e.g. G[cell,cell´]::HybridMatrix for any cells
#        - g::HybridMatrix = G(ω; params...) over merged contact latslice
#        - linewidth operatod Γᵢ::Matrix for each contact
#   Any new s::GreenMatrixSolver must implement
#      - s[] -> G::HybridMatrix over contact sites
#      - s[cell, cell´] G::HybridMatrix between unitcells cell <= cell´
#      - s[subcell, subcell´] G::HybridMatrix between subcell <= subcell´
#region

# Generic system-independent directives for solvers, e.g. GS.Schur()
abstract type AbstractGreenSolver end

# Application to a given system, but still independent from ω, system params or Σs
abstract type AppliedGreenSolver end

# Solver with fixed ω and params that can compute G[subcell, subcell´] or G[cell, cell´]
# It should also be able to return contact G with G[]
abstract type GreenMatrixSolver end

#endregion

############################################################################################
# Green - see greenfunction.jl
#region

struct GreenFunction{T,E,L,S<:AppliedGreenSolver,H<:AbstractHamiltonian{T,E,L},C<:Contacts}
    parent::H
    solver::S
    contacts::C
end

# Obtained with gω = call!(g::GreenFunction, ω; params...) or g(ω; params...)
# Allows gω[i, j] -> HybridMatrix for i,j integer Σs indices ("contacts")
# Allows gω[cell, cell´] -> HybridMatrix using T-matrix, with cell::Union{SVector,Subcell}
# Allows also view(gω, ...)
struct GreenMatrix{T,E,L,D<:GreenMatrixSolver,M<:NTuple{<:Any,AbstractMatrix}}
    solver::D                        # computes G(ω; p...)[cell,cell´] for any cell/subcell
    g::HybridMatrix{Complex{T},L}    # hybrid matrix G(ω; p...)[i,i´] on latslice sites i
    Γs::M                            # linewidths Γ=i(Σ-Σ⁺) for each contact
    latslice::LatticeSlice{T,E,L}    # === contacts.mergedlatslice from parent GreenFunction
end

# Obtained with gs = g[; siteselection...]
# Alows call!(gs, ω; params...) -> View{HybridMatrix} or gs(ω; params...) -> HybridMatrix
#   required to do h |> attach(g´[sites´], couplingmodel; sites...)
struct GreenFunctionSlice{T,E,L,G<:GreenFunction{T,E,L}}
    g::G
    latslice::LatticeSlice{T,E,L}
end

#region ## Constructors ##

GreenFunction(oh::OpenHamiltonian, as::AppliedGreenSolver) =
    GreenFunction(hamiltonian(oh), as, Contacts(oh))

#endregion

#region ## API ##

hamiltonian(g::GreenFunction) = g.parent

solver(g::GreenFunction) = g.solver

contacts(g::GreenFunction) = g.contacts

Base.parent(g::GreenFunction) = g.parent

Base.getindex(g::GreenMatrix, c::Subcell, c´::Subcell) =
    g.solver(cell(c), cell(c´), siteindices(c), siteindices(c´))

Base.getindex(g::GreenMatrix{<:Any,<:Any,L}, c::NTuple{L,Int}, c´::NTuple{L,Int}) where {L} =
    g.solver(SVector(c), SVector(c´))

Base.getindex(g::GreenMatrix) = g.solver()

#endregion
#endregion