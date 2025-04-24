############################################################################################
# Serializer  -  see serializer.jl
#   A Serializer is used to contruct an iterator that encodes Hamiltonian matrix elements in
#   a way that can be turned into a Vector and back into a Hamiltonian
#   encoder = s -> vec and its inverse decoder = vec -> s translate between s::B and some
#   iterable, typically some AbstractVector. B<:SMatrixView is translated to its parent.
#   Both can also be tuples (s->vec, (s, s´)->vec) and viceversa if the translation
#   of hoppings requires both the hopping s and its conjugate s´.
#region

struct Serializer{T,S,E,D} <: Modifier
    type::Type{T}
    parameter::Symbol             # parameter for ParametricHamiltonians, :stream by default
    selectors::S                  # unapplied selectors, needed to rebuild ptrs if h changes
    encoder::E
    decoder::D
end

struct AppliedSerializer{T,H<:AbstractHamiltonian,S<:Serializer{T},P<:Dictionary} <: AppliedModifier
    parent::S
    h::H
    ptrs::P     # [dn => [(ptr, ptr´, serialrng)...]...] or [dn => [(ptr, serialrng)...]...]
    len::Int
end

#region ## Constructors ##

serializer(T::Type; kw...) = serializer(T, siteselector(), hopselector(); kw...)

function serializer(T::Type, sel::Selector, selectors...; encoder = identity, decoder = identity, parameter = :stream)
    check_encoder_decoder_tuples(encoder, decoder)
    s = Serializer(T, parameter, (sel, selectors...), encoder, decoder)
    return s
end

serializer(h::AbstractHamiltonian{T}, args...; kw...) where {T} =
    serializer(Complex{T}, h, args...; kw...)

serializer(T::Type, h::AbstractHamiltonian, args...; kw...) =
    apply(serializer(T, args...; kw...), h)

check_encoder_decoder_tuples(encoder::Function, decoder::Function) = nothing
check_encoder_decoder_tuples(encoder::Tuple{Function,Function}, decoder::Tuple{Function,Function}) =
    nothing
check_encoder_decoder_tuples(_, _) = argerror("encoder and decoder must be both functions or both tuples of functions")

#endregion

#region ## API ##

call!(s::AppliedSerializer; kw...) = AppliedSerializer(s.parent, call!(s.h; kw...), s.ptrs, s.len)

(s::AppliedSerializer)(; kw...) = AppliedSerializer(s.parent, s.h(; kw...), s.ptrs, s.len)

hamiltonian(s::AppliedSerializer) = parametric(s.h, s)

# assume s has been applied to h or a copy of h
maybe_relink_serializer(s::AppliedSerializer, h) =
    s.h === h ? s : AppliedSerializer(s.parent, h, s.ptrs, s.len)
maybe_relink_serializer(m::AppliedModifier, _) = m

parent_hamiltonian(s::AppliedSerializer) = s.h

pointers(s::AppliedSerializer) = s.ptrs

Base.length(s::AppliedSerializer) = s.len

Base.eltype(s::AppliedSerializer) = eltype(s.parent)
Base.eltype(::Serializer{T}) where {T} = T

Base.parent(s::AppliedSerializer) = s.parent

selectors(s::Serializer) = s.selectors
selectors(s::AppliedSerializer) = selectors(s.parent)

parameter_names(s::Serializer) = (s.parameter,)
parameter_names(s::AppliedSerializer) = parameter_names(s.parent)

encoder(s::Serializer) = s.encoder
encoder(s::AppliedSerializer) = encoder(s.parent)

decoder(s::Serializer) = s.decoder
decoder(s::AppliedSerializer) = decoder(s.parent)


#endregion

#endregion

############################################################################################
# AppliedSerializer API
#   support for serialization of Hamiltonians and ParametricHamiltonians
#region

## applymodifier! API

function applymodifiers!(h, m::AppliedSerializer; kw...)
    pname = only(parameter_names(m))
    nkw = NamedTuple(kw)
    # this should override hamiltonian(s), which should be aliased with h
    haskey(nkw, pname) && deserialize!(m, nkw[pname])
    return h
end

function _merge_pointers!(p, sm::AppliedSerializer)
    for (pn, ps) in zip(p, pointers(sm)), p in ps
        push!(pn, Base.front(p)...)
    end
    return p
end

## serialize and serialize!

Base.@propagate_inbounds function serialize!(v, s::AppliedSerializer; kw...)
    @boundscheck check_serializer_length(s, v)
    h0 = parent_hamiltonian(s)
    h = call!(h0; kw...)
    enc = encoder(s)
    i = 0
    for (har, ptrs) in zip(harmonics(h), pointers(s)), p in ptrs
        for vi in serialize_core(h, har, p, enc)
            i += 1
            v[i] = vi
        end
    end
    return v
end

serialize(s::AppliedSerializer{T}; kw...) where {T} =
    serialize!(Vector{T}(undef, length(s)), s; kw...)

# encoder::Function
function serialize_core(h, har, (ptr, rng), encoder::Function)
    v = encoder(maybe_parent(nonzeros(unflat(har))[ptr]))
    return v
end

# encoder::Tuple{Function,Function} = (f1, f2)
# encode onsites with f1, hoppings and their adjoints with f2(s1, s2)
function serialize_core(h, har, (ptr, ptr´, rng), (enc1, enc2)::NTuple{2,Function})
    dn = dcell(har)
    mat = unflat(har)
    pv = maybe_parent(nonzeros(mat)[ptr])
    if ptr == ptr´ && iszero(dn)
        v = enc1(pv)
    else
        mat´ = h[unflat(-dn)]
        pv´ = maybe_parent(nonzeros(mat´)[ptr´])
        v = enc2(pv, pv´)
    end
    return v
end

serialize_core(args...) = argerror("Unknown encoder type. Use `encoder = s->vec` or `encoder = (s->vec, (s1, s2)->vec)`")

maybe_parent(s::SMatrixView) = parent(s)
maybe_parent(s) = s

check_serializer_length(s, v) = length(v) == length(s) ||
        throw(DimensionMismatch("Vector length $(length(v)) does not match serializer length $(length(s))"))

## deserialize!

deserialize(s::AppliedSerializer, v; kw...) = deserialize!(s(; kw...), v)

deserialize!(s::AppliedSerializer, v; kw...) = deserialize!(call!(s; kw...), v)

Base.@propagate_inbounds function deserialize!(s::AppliedSerializer{<:Any,<:Hamiltonian}, v)
    @boundscheck check_serializer_length(s, v)
    h = parent_hamiltonian(s)
    dec = decoder(s)
    for (har, ptrs) in zip(harmonics(h), pointers(s))
        isempty(ptrs) && continue
        for p in ptrs
            deserialize_core!(h, har, v, p, dec)
        end
        trigger_flat_sync!(har, h, dec)
    end
    return h
end

# decoder::Function : ignore ptr´, decode only ptr
function deserialize_core!(h, har, v, (ptr, rng), decoder::Function)
    mat = unflat(har)
    s = decoder(view(v, rng))
    maybe_unwrap_and_assign(nonzeros(mat), ptr, s)
    return mat
end

# decoder::Tuple{Function,Function} = (f1, f2)
# decode onsites with f1, hoppings and their adjoints with f2
@inline function deserialize_core!(h, har, v, (ptr, ptr´, rng), (dec1, dec2)::NTuple{2,Function})
    mat = unflat(har)
    dn = dcell(har)
    nzs = nonzeros(mat)
    if ptr == ptr´ && iszero(dn)
        s = dec1(view(v, rng))
        maybe_unwrap_and_assign(nzs, ptr, s)
    else
        s1, s2 = dec2(view(v, rng))
        nzs´ = nonzeros(h[unflat(-dn)])
        maybe_unwrap_and_assign(nzs, ptr, s1)
        maybe_unwrap_and_assign(nzs´, ptr´, s2)
    end
    return mat
end

deserialize_core!(args...) =
    argerror("Unknown decoder type. Use `decoder = vec->s` or `decoder = (vec->s, vec->(s1, s2))`")

maybe_unwrap_and_assign(nzs::AbstractVector{<:Number}, ptr, s::Number) = (nzs[ptr] = s)
maybe_unwrap_and_assign(nzs::AbstractVector{<:Number}, ptr, s) = (nzs[ptr] = only(s))
maybe_unwrap_and_assign(nzs::AbstractVector{S}, ptr, s) where {S} = (nzs[ptr] = convert(S, s))

trigger_flat_sync!(har, h, decoder::Function) = needs_flat_sync!(matrix(har))

function trigger_flat_sync!(har, h, decoder::Tuple)
    dn = dcell(har)
    needs_flat_sync!(h[hybrid(dn)])
    needs_flat_sync!(h[hybrid(-dn)])
    return har
end

## check

function check(s::AppliedSerializer{T}; params...) where {T}
    h = parent_hamiltonian(s)
    h(; params...) == deserialize(s, serialize(s; params...); params...) ||
        argerror("decoder/encoder pair do not seem to be correct inverse of each other")
    return nothing
end

#endregion


############################################################################################
# serialize and deserialize for AbstractOrbitalArray
#   extract underlying data and reconstruct an object using such data
#region

serialize(a::AbstractOrbitalArray) = serialize_array(parent(a))
serialize(a::AbstractArray) = serialize_array(a)
serialize(::Type{T}, a::AbstractArray) where {T} = reinterpret(T, serialize(a))

serialize_array(a::Array) = a
serialize_array(a::Diagonal{<:Any,<:Vector}) = a.diag
serialize_array(a::SparseMatrixCSC) = nonzeros(a)

deserialize(a::AbstractArray{T}, v::AbstractArray) where {T} =
    deserialize(a, reinterpret(T, v))
deserialize(a::AbstractArray{T}, v::AbstractArray{T}) where {T} =
    (check_serializer_size(a, v); unsafe_deserialize(a, v))

unsafe_deserialize(a::AbstractOrbitalArray, v::AbstractArray) =
    similar(a, deserialize_array(parent(a), v), orbaxes(a))
unsafe_deserialize(a::AbstractArray, v::AbstractArray) =
    deserialize_array(a, v)

deserialize_array(::AbstractArray{<:Any,N}, v::AbstractArray{<:Any,N}) where {N} = v
deserialize_array(::Diagonal, v::AbstractVector) =
    Diagonal(convert(Vector, v))
deserialize_array(a::SparseMatrixCSC, v::AbstractVector) =
    SparseMatrixCSC(a.m, a.n, a.colptr, a.rowval, convert(Vector, v))

check_serializer_size(a, v) =
    size(serialize(a)) == size(v) ||
        argerror("Wrong size of serialized array, expected length $(size(serialize(a))), got $(size(v))")

#endregion
