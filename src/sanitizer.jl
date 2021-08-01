sanitize_Vector_of_SVectors(v::Number) = [sanitize_SVector(v)]
sanitize_Vector_of_SVectors(v::NTuple{<:Any,Number}) = [sanitize_SVector(v)]
sanitize_Vector_of_SVectors(v::AbstractVector{<:Number}) = [sanitize_SVector(v)]
sanitize_Vector_of_SVectors(vs) = [promote(sanitize_SVector.(vs)...)...]

sanitize_SVector(::Tuple{}) = SVector{0,Float64}
sanitize_SVector(x::Number) = SVector{1}(x)
sanitize_SVector(v) = convert(SVector, v)
sanitize_SVector(T, v) = sanitize_eltype(T, sanitize_SVector(v))

sanitize_eltype(T::Type, v::SVector{N}) where {N} = convert(SVector{N,T}, v)
sanitize_eltype(T::Type, v) = convert.(T, v)
