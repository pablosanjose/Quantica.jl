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

# Constructors for conversion
Sublat{T,E}(s::Sublat, name = s.name) where {T<:AbstractFloat,E} =
    Sublat{T,E}([sanitize_SVector(SVector{E,T}, site) for site in sites(s)], name)

CellSites{L,V}(c::CellSites) where {L,V} =
    CellSites{L,V}(convert(SVector{L,Int}, cell(c)), convert(V, siteindices(c)))

function Hamiltonian{T,E}(h::Hamiltonian) where {T,E}
    lat = lattice(h)
    lat´ = lattice(lat, dim = Val(E), type = T)
    bs = blockstructure(h)
    hs = harmonics(h)
    b = bloch(h)
    return Hamiltonian(lat´, bs, hs, b)
end

function ParametricHamiltonian{T,E}(ph::ParametricHamiltonian) where {T,E}
    hparent = Hamiltonian{T,E}(parent(ph))
    h = Hamiltonian{T,E}(hamiltonian(ph))
    ms = modifiers(ph)
    ptrs = pointers(ph)
    pars = parameters(ph)
    return ParametricHamiltonian(hparent, h, ms, ptrs, pars)
end

# We need this to promote different sublats into common dimensionality and type to combine
# into a lattice
Base.promote(ss::Sublat{T,E}...) where {T,E} = ss

function Base.promote_rule(::Type{Sublat{T1,E1}}, ::Type{Sublat{T2,E2}}) where {T1,T2,E1,E2}
    E = max(E1, E2)
    T = float(promote_type(T1, T2))
    return Sublat{T,E}
end
