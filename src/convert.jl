# In v0.7+ the idea is that `convert`` is a shortcut to a "safe" subset of the constructors
# for a type, that guarantees the resulting type. The constructor is the central machinery
# to instantiate types and convert between instances. (For parametric types, unless overridden,
# you get an implicit internal constructor without parameters, so no need to define that externally)

Base.convert(::Type{T}, l::T) where T<:SMatrixView = l
Base.convert(::Type{T}, l::SMatrixView) where T<:SMatrixView = T(parent(l))

Base.convert(::Type{T}, l::T) where T<:Sublat = l
Base.convert(::Type{T}, l::Sublat) where T<:Sublat = T(l)

Base.convert(::Type{T}, l::T) where T<:CellSites = l
Base.convert(::Type{T}, l::CellSites) where T<:CellSites = T(l)

Base.convert(::Type{T}, l::T) where T<:AbstractHamiltonian = l
Base.convert(::Type{T}, l::AbstractHamiltonian) where T<:AbstractHamiltonian = T(l)

Base.convert(::Type{T}, l::T) where T<:Mesh = l
Base.convert(::Type{T}, l::Mesh) where T<:Mesh = T(l)

# Constructors for conversion
Sublat{T,E}(s::Sublat, name = s.name) where {T<:AbstractFloat,E} =
    Sublat{T,E}([sanitize_SVector(SVector{E,T}, site) for site in sites(s)], name)

CellSites{L,V}(c::CellSites) where {L,V} =
    CellIndices(convert(SVector{L,Int}, cell(c)), convert(V, siteindices(c)), SiteLike())

CellSites{L,V}(c::CellSite) where {L,V<:Vector} =
    CellIndices(convert(SVector{L,Int}, cell(c)), convert(V, [siteindices(c)]), SiteLike())

function Hamiltonian{E}(h::Hamiltonian) where {E}
    lat = lattice(h)
    lat´ = lattice(lat, dim = Val(E))
    bs = blockstructure(h)
    hs = harmonics(h)
    b = bloch(h)
    return Hamiltonian(lat´, bs, hs, b)
end

function ParametricHamiltonian{E}(ph::ParametricHamiltonian) where {E}
    hparent = Hamiltonian{E}(parent(ph))
    h = Hamiltonian{E}(hamiltonian(ph))
    ms = modifiers(ph)
    ptrs = pointers(ph)
    pars = parameters(ph)
    return ParametricHamiltonian(hparent, h, ms, ptrs, pars)
end

Mesh{S}(m::Mesh) where {S} =
    Mesh(convert(Vector{S}, vertices(m)), neighbors(m), simplices(m))

# We need this to promote different sublats into common dimensionality and type to combine
# into a lattice
Base.promote(ss::Sublat{T,E}...) where {T,E} = ss

function Base.promote_rule(::Type{Sublat{T1,E1}}, ::Type{Sublat{T2,E2}}) where {T1,T2,E1,E2}
    E = max(E1, E2)
    T = float(promote_type(T1, T2))
    return Sublat{T,E}
end

# Block type promotion, for use in e.g. combine

const HBtype{B} = AbstractHamiltonian{<:Any,<:Any,<:Any,B}
const SMatrixOrView{N,M,T,NM} = Union{SMatrix{N,M,T,NM}, SMatrixView{N,M,T,NM}}

Base.promote_rule(::Type{<:HBtype{S}}, ::Type{<:HBtype{S}}) where {S} = HBtype{S}

Base.promote_rule(::Type{<:HBtype{C}}, ::Type{<:HBtype{C´}}) where {C<:Complex,C´<:Complex} =
    HBtype{promote_type(C,C´)}

Base.promote_rule(::Type{<:HBtype{C´}}, ::Type{<:HBtype{S}}) where {C´<:Complex,C,N,NN,S<:SMatrixOrView{N,N,C,NN}} =
    HBtype{SMatrixView{N,N,promote_type(C,C´),NN}}

function Base.promote_rule(::Type{<:HBtype{S1}}, ::Type{<:HBtype{S2}}) where {N1,C1,S1<:SMatrixOrView{N1,N1,C1},N2,C2,S2<:SMatrixOrView{N2,N2,C2}}
    N = max(N1,N2)
    C = promote_type(C1, C2)
    return HBtype{SMatrixView{N,N,C,N*N}}
end
