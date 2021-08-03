sanitize_Vector_of_Symbols(names) = Symbol[convert(Symbol, name) for name in names]

sanitize_Vector_of_SVectors(vs) =
    eltype(vs) <: Number ? [sanitize_SVector(vs)] : [promote(sanitize_SVector.(vs)...)...]

sanitize_Tuple_of_SVectors(vs::Tuple) = sanitize_SVector.(vs)
sanitize_Tuple_of_SVectors(m::SMatrix{<:Any,L}) where {L} = ntuple(i -> m[:, i], Val(L))
sanitize_Tuple_of_SVectors(S, vs::Tuple) = sanitize_SVector.(S, vs)
sanitize_Tuple_of_SVectors(S, m::SMatrix{<:Any,L}) where {L} =
    ntuple(i -> sanitize_SVector(S, m[i, :]), Val(L))

sanitize_SVector(::Tuple{}) = SVector{0,Float64}()
sanitize_SVector(x::Number) = SVector{1}(x)
sanitize_SVector(v) = convert(SVector, v)

sanitize_SVector(::Type{SVector{N,T}}, v::SVector{N}) where {N,T} = convert(SVector{N,T}, v)
sanitize_SVector(::Type{SVector{N,T}}, v) where {N,T} =
    SVector(ntuple(i-> i > length(v) ? zero(T) : convert(T, v[i]), Val(N)))

sanitize_SMatrix(x) = hcat(sanitize_Tuple_of_SVectors(x)...)
sanitize_SMatrix(::Type{T}, x) where {T<:Number} = convert.(T, sanitize_SMatrix(x))
# sanitize_SMatrix(::Type{S}, x) where {L,S<:SMatrix{L}} = 