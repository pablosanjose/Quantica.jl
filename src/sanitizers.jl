############################################################################################
# Non-numerical sanitizers
#region

sanitize_Vector_of_Symbols(name::Symbol) = [name]
sanitize_Vector_of_Symbols(names) = Symbol[convert(Symbol, name) for name in names]

sanitize_orbitals(o::Val) = o

sanitize_orbitals(o::Int) = Val(o)

sanitize_orbitals(o) = allequal(o) ? Val(first(o)) : o

#endregion

############################################################################################
# Array sanitizers (padding with zeros if necessary)
#region

sanitize_Vector_of_Type(::Type{T}, len, x::T´) where {T,T´<:Union{T,Val}} = fill(val_convert(T, x), len)

function sanitize_Vector_of_Type(::Type{T}, len, xs) where {T}
    xs´ = sanitize_Vector_of_Type(T, xs)
    length(xs´) == len || throw(ArgumentError("Received a collection with $(length(xs´)) elements, should have $len."))
    return xs´
end

function sanitize_Vector_of_Type(::Type{T}, xs) where {T}
    xs´ = T[val_convert(T, x) for x in xs]
    return xs´
end

val_convert(T, ::Val{N}) where {N} = convert(T, N)
val_convert(T, x) = convert(T, x)

sanitize_Vector_of_SVectors(::Type{T}, ::Tuple{}) where {T} = T[]
sanitize_Vector_of_SVectors(::Type{T}, vs) where {T} =
    eltype(vs) <: Number ? [sanitize_SVector(T, vs)] : [sanitize_SVector.(T, vs)...]

sanitize_SVector(::Tuple{}) = SVector{0,Float64}()
sanitize_SVector(x::Number) = SVector{1}(x)
sanitize_SVector(v) = convert(SVector{length(v)}, v)
sanitize_SVector(::Type{T}, v::SVector{<:Any,T}) where {T<:Number} = v
sanitize_SVector(::Type{T}, v) where {T<:Number} =
    isempty(v) ? SVector{0,T}() : convert.(T, sanitize_SVector(v))
sanitize_SVector(::Type{SVector{N,T}}, v::SVector{N}) where {N,T} = convert(SVector{N,T}, v)
sanitize_SVector(::Type{SVector{N,T}}, v) where {N,T} =
    SVector{N,T}(ntuple(i -> i > length(v) ? zero(T) : convert(T, v[i]), Val(N)))

function sanitize_SMatrix(::Type{S}, x, (rows, cols) = (E, L)) where {T<:Number,E,L,S<:SMatrix{E,L,T}}
    t = ntuple(Val(E*L)) do l
        j, i = fldmod1(l, E)
        i <= max(rows, length(x[j])) && j <= max(cols, length(x)) ? T(x[j][i]) : zero(T)
    end
    return SMatrix{E,L,T}(t)
end

function sanitize_SMatrix(::Type{S}, s::AbstractMatrix, (rows, cols) = (E, L)) where {T<:Number,E,L,S<:SMatrix{E,L,T}}
    t = ntuple(Val(E*L)) do l
        j, i = fldmod1(l, E)
        i <= rows && j <= cols && checkbounds(Bool, s, i, j) ? convert(T, s[i,j]) : zero(T)
    end
    return SMatrix{E,L,T}(t)
end

function sanitize_Matrix(::Type{T}, E, cols::Tuple) where {T}
    m = zeros(T, E, length(cols))
    for (j, col) in enumerate(cols), i in 1:E
        if i <= length(col)
            m[i, j] = col[i]
        end
    end
    return m
end

function sanitize_Matrix(::Type{T}, E, m::AbstractMatrix) where {T}
    m´ = zeros(T, E, size(m, 2))
    axs = intersect.(axes(m), axes(m´))
    m´[axs...] .= convert.(T, view(m, axs...))
    return m´
end

#endregion

# ############################################################################################
# # Block sanitizers
# #region

# sanitize_block(S::Type{<:Number}, s, _) = convert(S, first(s))
# sanitize_block(S::Type{<:SMatrix}, s::SMatrix, size) = sanitize_SMatrix(S, s, size)
# sanitize_block(::Type{S}, s::Number, size) where {S<:SMatrix} = sanitize_SMatrix(S, S(s*I), size)
# sanitize_block(::Type{S}, s::UniformScaling, size) where {S<:SMatrix} =
#     sanitize_SMatrix(S, S(s), size)

# #endregion

############################################################################################
# Supercell sanitizers
#region

sanitize_supercell(::Val{L}, ::Tuple{}) where {L} = SMatrix{L,0,Int}()
sanitize_supercell(::Val{L}, ns::NTuple{L´,NTuple{L,Int}}) where {L,L´} =
    sanitize_SMatrix(SMatrix{L,L´,Int}, ns)
sanitize_supercell(::Val{L}, n::Tuple{Int}) where {L} =
    SMatrix{L,L,Int}(first(n) * I)
sanitize_supercell(::Val{L}, ns::NTuple{L,Int}) where {L} =
    SMatrix{L,L,Int}(Diagonal(SVector(ns)))
sanitize_supercell(::Val{L}, s::Tuple{SMatrix{<:Any,<:Any,Int}}) where {L} = only(s)
sanitize_supercell(::Val{L}, v) where {L} =
    throw(ArgumentError("Improper supercell specification $v for an $L lattice dimensions, see `supercell`"))

#endregion

############################################################################################
# Eigen sanitizers
#region

sanitize_eigen(ε, Ψ) = Eigen(sorteigs!(ε, Ψ)...)
sanitize_eigen(ε::AbstractVector, Ψs::AbstractVector{<:AbstractVector}) =
    sanitize_eigen(ε, hcat(Ψs...))
sanitize_eigen(ε::AbstractVector{<:Real}, Ψ::AbstractMatrix) =
    sanitize_eigen(complex.(ε), Ψ)

function sorteigs!(ϵ::AbstractVector, ψ::AbstractMatrix)
    p = Vector{Int}(undef, length(ϵ))
    p´ = similar(p)
    sortperm!(p, ϵ, by = real, alg = Base.DEFAULT_UNSTABLE)
    Base.permute!!(ϵ, copy!(p´, p))
    Base.permutecols!!(ψ, copy!(p´, p))
    return ϵ, ψ
end

#endregion
