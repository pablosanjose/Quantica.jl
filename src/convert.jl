# In v0.7+ the idea is that `convert`` is a shortcut to a "safe" subset of the constructors
# for a type, that guarantees the resulting type. The constructor is the central machinery
# to instantiate types and convert between instances. (For parametric types, unless overridden,
# you get an implicit internal constructor without parameters, so no need to define that externally)

Base.convert(::Type{T}, l::T) where T<:Tuple{Vararg{Sublat}} = l
Base.convert(::Type{Tuple{Vararg{Sublat{E,T}}}}, l::Tuple{Vararg{Sublat}}) where {E,T} = Sublat{E,T}.(l)

Base.convert(::Type{T}, l::T) where T<:Sublat = l
Base.convert(::Type{T}, l::Sublat) where T<:Sublat = T(l)

Base.convert(::Type{T}, l::T) where T<:Bravais = l
Base.convert(::Type{T}, l::Bravais) where T<:Bravais = T(l)

# Constructors for conversion

Sublat{E,T}(s::Sublat, name = s.name) where {E,T} =
    Sublat([padright(site, zero(T), Val(E)) for site in s.sites], name)

# We need this to promote different sublats into common dimensionality and type to combine
# into a lattice, while neglecting orbital dimension
Base.promote(ss::Sublat{E,T}...) where {E,T} = ss
Base.promote_rule(::Type{Sublat{E1,T1}}, ::Type{Sublat{E2,T2}}) where {E1,E2,T1,T2} =
    Sublat{max(E1, E2), promote_type(T1, T2)}

Bravais{E,L,T}(b::Bravais) where {E,L,T} =
    Bravais(padtotype(b.matrix, SMatrix{E,L,T}), padright(b.semibounded, Val(L)))

# Promotion

# promote_model(model::Model, sys::System{E,L,T,Tv}, systems...) where {E,L,T,Tv} = promote_model(Tv, model, systems...)
# promote_model(::Type{Tv}, model::Model, sys::System{E,L,T,Tv2}, systems...) where {Tv,E,L,T,Tv2} = promote_model(promote_type(Tv, Tv2), model, systems...)
# promote_model(::Type{Tv}, model::Model) where {Tv} = convert(Model{Tv}, model)