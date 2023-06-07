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

# Constructors for conversion
Sublat{T,E}(s::Sublat, name = s.name) where {T<:AbstractFloat,E} =
    Sublat{T,E}([sanitize_SVector(SVector{E,T}, site) for site in sites(s)], name)

CellSites{L,V}(c::CellSites) where {L,V} =
    CellSites{L,V}(convert(SVector{L,Int}, cell(c)), convert(V, siteindices(c)))

# We need this to promote different sublats into common dimensionality and type to combine
# into a lattice
Base.promote(ss::Sublat{T,E}...) where {T,E} = ss

function Base.promote_rule(::Type{Sublat{T1,E1}}, ::Type{Sublat{T2,E2}}) where {T1,T2,E1,E2}
    E = max(E1, E2)
    T = float(promote_type(T1, T2))
    return Sublat{T,E}
end

function Base.promote_rule(::Type{AbstractHamiltonian{T1,E1,L,B1}}, ::Type{AbstractHamiltonian{T2,E2,L,B2}}) where {T1,E1,B1,T2,E2,B2,L}
    T = float(promote_type(T1, T2))
    E = max(E1, E2)
    B = promote_block(B1, B2)
    return AbstractHamiltonian{T,E,L,B}
end

promote_block(T::Type{<:Number}, T´::Type{<:Number}) = promote_type(T, T´)
promote_block(T´::Type{<:Number}, ::Type{S}) where {N,M,T,S<:SMatrix{N,M,T}} =
    SMatrix{N,M,promote_type(T, T´),N*M}
promote_block(::Type{S}, T´::Type{<:Number}) where {N,M,T,S<:SMatrix{N,M,T}} =
    promote_block(T´, S)

function promote_block(::Type{S1}, ::Type{S2}) where {N1,M1,T1,S1<:SMatrix{N1,M1,T1},N2,M2,T2,S2<:SMatrix{N2,M2,T2}}
    N = max(N1,N2)
    M = max(M1, M2)
    T = promote_type(T1, T2)
    return SMatrix{N,M,T,N*M}
end