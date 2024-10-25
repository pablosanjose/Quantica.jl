############################################################################################
# AppliedSerializer
#   support for serialization of Hamiltonians and ParametricHamiltonians
#region

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
# serialize and deserialize for OrbitalSliceArray
#   extract underlying data and reconstruct an object using such data
#   no check of correct array dimensionality is performed for performance
#region

serialize(a::OrbitalSliceArray) = serialize_array(parent(a))
serialize(a::AbstractArray) = serialize_array(a)
serialize(::Type{T}, a::AbstractArray{T}) where {T} = serialize(a)
serialize(::Type{T}, a::AbstractArray) where {T} = reinterpret(T, serialize(a))

serialize_array(a::Array) = a
serialize_array(a::Diagonal{<:Any,<:Vector}) = a.diag
serialize_array(a::SparseMatrixCSC) = nonzeros(a)

deserialize(a, v) = (check_serializer_length(a, v); _deserialize(a, v))

_deserialize(a::OrbitalSliceArray{T}, v::AbstractArray{T}) where {T} =
    OrbitalSliceArray(deserialize_array(parent(a), v), orbaxes(a))
_deserialize(a::AbstractArray{T}, v::AbstractArray{T}) where {T} = deserialize_array(a, v)
_deserialize(a::AbstractArray{T}, v::AbstractArray) where {T} = deserialize(a, reinterpret(T, v))

deserialize_array(::AbstractArray{<:Any,N}, v::AbstractArray{<:Any,N}) where {N} = v
deserialize_array(::Diagonal, v::AbstractVector) =
    Diagonal(convert(Vector, v))
deserialize_array(a::SparseMatrixCSC, v::AbstractVector) =
    SparseMatrixCSC(a.m, a.n, a.colptr, a.rowval, convert(Vector, v))

check_serializer_length(a::AbstractArray{T}, v::AbstractArray{T}) where {T} =
    length(serialize(a)) == length(v) || argerror("Wrong length of serialized array")

check_serializer_length(a::AbstractArray, v::AbstractArray{T}) where {T} =
    length(serialize(T, a)) == length(v) || argerror("Wrong length of serialized array")

#endregion
