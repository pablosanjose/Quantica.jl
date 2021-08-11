############################################################################################
# Non-numerical sanitizers
#region

sanitize_Vector_of_Symbols(names) = Symbol[convert(Symbol, name) for name in names]

#endregion

############################################################################################
# Array sanitizers (padding with zeros if necessary)
#region

sanitize_Vector_of_Type(::Type{T}, len, x::T´) where {T,T´<:Union{T,Val}} = fill(val_convert(T, x), len)

function sanitize_Vector_of_Type(::Type{T}, len, xs) where {T}
    xs´ = T[val_convert(T, x) for x in xs]
    length(xs´) == len || throw(ArgumentError("Received a collection with $(length(xs´)) elements, should have $len."))
    return xs´
end

val_convert(T, ::Val{N}) where {N} = convert(T, N)
val_convert(T, x) = convert(T, x)

sanitize_Vector_of_SVectors(vs) =
    eltype(vs) <: Number ? [sanitize_SVector(vs)] : [promote(sanitize_SVector.(vs)...)...]
sanitize_Vector_of_SVectors(S::Type, vs) =
    eltype(vs) <: Number ? [sanitize_SVector(S, vs)] : [promote(sanitize_SVector.(S, vs)...)...]

sanitize_SVector(::Tuple{}) = SVector{0,Float64}()
sanitize_SVector(x::Number) = SVector{1}(x)
sanitize_SVector(v) = convert(SVector, v)
sanitize_SVector(::Type{T}, v) where {T<:Number} = convert.(T, sanitize_SVector(v))
sanitize_SVector(::Type{SVector{N,T}}, v::SVector{N}) where {N,T} = convert(SVector{N,T}, v)
sanitize_SVector(::Type{SVector{N,T}}, v) where {N,T} =
    SVector(ntuple(i -> i > length(v) ? zero(T) : convert(T, v[i]), Val(N)))

function sanitize_SMatrix(::Type{SMatrix{E,L,T}}, x, (rows, cols) = (E, L)) where {T<:Number,E,L}
    t = ntuple(Val(E*L)) do l
        j, i = fldmod1(l, E)
        j > max(cols, length(x)) || i > max(rows, length(x[j])) ? zero(T) : T(x[j][i])
    end
    return SMatrix{E,L,T}(t)
end

function sanitize_SMatrix(::Type{SMatrix{E,L,T}}, s::SMatrix, (rows, cols) = (E, L)) where {T<:Number,E,L}
    c = 0
    t = ntuple(Val(E*L)) do l
        j, i = fldmod1(l, E)
        checkbounds(Bool, s, i, j) && i <= rows && j <= cols ? (c += 1; convert(T, s[c])) : zero(T)
    end
    return SMatrix{E,L,T}(t)
end

function sanitize_Matrix(::Type{T}, E, cols...) where {T}
    m = zeros(T, E, length(cols))
    for (j, col) in enumerate(cols), i in 1:E
        @inbounds m[i,j] = col[i]
    end
    return m
end

function sanitize_Matrix(::Type{T}, E, m::AbstractMatrix) where {T}
    m´ = zeros(T, E, size(m, 2))
    m´[axes(m)...] .= convert.(T, m)
    return m´
end

#endregion

############################################################################################
# Block sanitizers
#region

sanitize_block(S::Type{<:Number}, s, _) = convert(S, first(s))
sanitize_block(S::Type{<:SMatrix}, s, size) = sanitize_SMatrix(S, s, size)

#endregion
