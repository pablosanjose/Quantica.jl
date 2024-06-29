
############################################################################################
# Serializer
#   support for serialization of Hamiltonians and ParametricHamiltonians
#region

## serializer_pointers

serializer_pointers(h::ParametricHamiltonian, encoder, selectors...) =
    serializer_pointers(hamiltonian(h), encoder, selectors...)

function serializer_pointers(h::Hamiltonian{<:Any,<:Any,L}, encoder, selectors...) where {L}
    E = serializer_pointer_type(encoder)
    skip_reverse = encoder isa Tuple
    d = Dictionary{SVector{L,Int},Vector{E}}()
    for har in harmonics(h)
        dn = dcell(har)
        if skip_reverse && haskey(d, -dn)
            ptrs = E[]
        else
            ptrs = push_and_merge_pointers!(E[], h, har, selectors...)
        end
        insert!(d, dn, ptrs)
    end
    return d
end

serializer_pointer_type(::Function) = Tuple{Int,UnitRange{Int}}
serializer_pointer_type(::Tuple{Function,Function}) = Tuple{Int,Int,UnitRange{Int}}

function push_and_merge_pointers!(ptrs, h, har, sel::Selector, selectors...)
    asel = apply(sel, lattice(h))
    push_pointers!(ptrs, h, har, asel)
    return push_and_merge_pointers!(ptrs, h, har, selectors...)
end

push_and_merge_pointers!(ptrs, h, har) = unique!(sort!(ptrs))

# gets called by push_pointers! in apply.jl
function push_pointer!(ptrs::Vector{Tuple{Int,Int,UnitRange{Int}}}, (p, _...), h, har, (row, col))
    dn = dcell(har)
    if row == col && iszero(dn)
        p´ = p
    elseif row > col && iszero(dn)
        return ptrs     # we don't want duplicates
    else
        mat´ = h[unflat(-dn)]
        p´ = sparse_pointer(mat´, (col, row)) # adjoint element
    end
     # we leave the serial range empty (initialized later), as it depends on the encoder
    push!(ptrs, (p, p´, 1:0))
    return ptrs
end

push_pointer!(ptrs::Vector{Tuple{Int,UnitRange{Int}}}, (p, _...), _...) = push!(ptrs, (p, 1:0))

## update_serial_ranges!

function update_serial_ranges!(s::Serializer)
    h = hamiltonian(s)
    enc = encoder(s)
    offset = 0
    for (har, ptrs) in zip(harmonics(h), pointers(s)), (i, p) in enumerate(ptrs)
        len = length(serialize_core(h, har, p, enc))
        ptrs[i] = (Base.front(p)..., offset+1:offset+len)
        offset += len
    end
    return Serializer(s, offset)
end

## serialize and serialize!

function serialize!(v, s::Serializer; kw...)
    @boundscheck check_serializer_length(s, v)
    h = call!(s.h; kw...)
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

serialize(s::Serializer{T}; kw...) where {T} =
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

deserialize(s::Serializer, v; kw...) = deserialize!(s(; kw...), v)

deserialize!(s::Serializer, v; kw...) = deserialize!(call!(s; kw...), v)

function deserialize!(s::Serializer{<:Any,<:Hamiltonian}, v)
    @boundscheck check_serializer_length(s, v)
    h = hamiltonian(s)
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

function check(s::Serializer{T}; params...) where {T}
    h = hamiltonian(s)
    h(; params...) == deserialize(s, serialize(s; params...); params...) ||
        argerror("decoder/encoder pair do not seem to be correct inverse of each other")
    return nothing
end

#endregion
