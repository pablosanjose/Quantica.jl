# In v0.7+ the idea is that `convert`` is a shortcut to a "safe" subset of the constructors
# for a type, that guarantees the resulting type. The constructor is the central machinery
# to instantiate types and convert between instances. (For parametric types, unless overridden,
# you get an implicit internal constructor without parameters, so no need to define that externally)

Base.convert(::Type{T}, l::T) where T<:Sublat = l
Base.convert(::Type{T}, l::Sublat) where T<:Sublat = T(l)

# Base.convert(::Type{T}, l::T) where T<:Bravais = l
# Base.convert(::Type{T}, l::Bravais) where T<:Bravais = T(l)

# Constructors for conversion
Sublat{T,E}(s::Sublat, name = s.name) where {T,E} =
    Sublat([sanitize_SVector(SVector{E,T}, site) for site in s.sites], name)

# We need this to promote different sublats into common dimensionality and type to combine
# into a lattice
Base.promote(ss::Sublat{T,E}...) where {T,E} = ss

function Base.promote_rule(::Type{Sublat{T1,E1}}, ::Type{Sublat{T2,E2}}) where {T1,T2,E1,E2}
    E´ = max(E1, E2)
    T´ = float(promote_type(T1, T2))
    return Sublat{T´,E´}
end