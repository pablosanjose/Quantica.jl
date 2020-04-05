toSMatrix() = SMatrix{0,0,Float64}()
toSMatrix(ss::NTuple{N,Number}...) where {N} = toSMatrix(SVector{N}.(ss)...)
toSMatrix(ss::SVector{N}...) where {N} = hcat(ss...)
toSMatrix(::Type{T}, ss...) where {T} = _toSMatrix(T, toSMatrix(ss...))
_toSMatrix(::Type{T}, s::SMatrix{N,M}) where {N,M,T} = convert(SMatrix{N,M,T}, s)
# Dynamic dispatch
toSMatrix(ss::AbstractVector...) = toSMatrix(Tuple.(ss)...)
toSMatrix(s::AbstractMatrix) = SMatrix{size(s,1), size(s,2)}(s)

toSVector(::Tuple{}) = SVector{0,Float64}()
toSVectors(vs...) = [promote(toSVector.(vs)...)...]
toSVector(v::SVector) = v
toSVector(v::NTuple{N,Number}) where {N} = SVector(v)
toSVector(x::Number) = SVector{1}(x)
toSVector(::Type{T}, v) where {T} = T.(toSVector(v))
toSVector(::Type{T}, ::Tuple{}) where {T} = SVector{0,T}()
# Dynamic dispatch
toSVector(v::AbstractVector) = SVector(Tuple(v))

ensuretuple(s::Tuple) = s
ensuretuple(s) = (s,)

# ensureSMatrix(f::Function) = f
# ensureSMatrix(m::T) where T<:Number = SMatrix{1,1,T,1}(m)
# ensureSMatrix(m::SMatrix) = m
# ensureSMatrix(m::Array) =
#     throw(ErrorException("Write all model terms using scalars or @SMatrix[matrix]"))

_rdr(r1, r2) = (0.5 * (r1 + r2), r2 - r1)

# zerotuple(::Type{T}, ::Val{L}) where {T,L} = ntuple(_ -> zero(T), Val(L))

function padright!(v::Vector, x, n::Integer)
    n0 = length(v)
    resize!(v, max(n, n0))
    for i in (n0 + 1):n
        v[i] = x
    end
    return v
end

padright(sv::StaticVector{E,T}, x::T, ::Val{E}) where {E,T} = sv
padright(sv::StaticVector{E1,T1}, x::T2, ::Val{E2}) where {E1,T1,E2,T2} =
    (T = promote_type(T1,T2); SVector{E2, T}(ntuple(i -> i > E1 ? x : convert(T, sv[i]), Val(E2))))
padright(sv::StaticVector{E,T}, ::Val{E2}) where {E,T,E2} = padright(sv, zero(T), Val(E2))
padright(sv::StaticVector{E,T}, ::Val{E}) where {E,T} = sv
padright(t::NTuple{N´,<:Any}, x, ::Val{N}) where {N´,N} = ntuple(i -> i > N´ ? x : t[i], Val(N))
padright(t::NTuple{N´,<:Any}, ::Val{N}) where {N´,N} = ntuple(i -> i > N´ ? 0 : t[i], Val(N))

filltuple(x, ::Val{L}) where {L} = ntuple(_ -> x, Val(L))

@inline padtotype(s::SMatrix{E,L}, st::Type{S}) where {E,L,E2,L2,S<:SMatrix{E2,L2}} =
    S(SMatrix{E2,E}(I) * s * SMatrix{L,L2}(I))
@inline padtotype(s::UniformScaling, st::Type{S}) where {S<:SMatrix} = S(s)
@inline padtotype(s::Number, ::Type{S}) where {S<:SMatrix} = S(s*I)
@inline padtotype(s::Number, ::Type{T}) where {T<:Number} = T(s)
@inline padtotype(s::AbstractArray, ::Type{T}) where {T<:Number} = T(first(s))
@inline padtotype(s::UniformScaling, ::Type{T}) where {T<:Number} = T(s.λ)

## Work around BUG: -SVector{0,Int}() isa SVector{0,Union{}}
negative(s::SVector{L,<:Number}) where {L} = -s
negative(s::SVector{0,<:Number}) = s

empty_sparse(::Type{M}, n, m) where {M} = sparse(Int[], Int[], M[], n, m)

display_as_tuple(v, prefix = "") = isempty(v) ? "()" :
    string("(", prefix, join(v, string(", ", prefix)), ")")

displayvectors(mat::SMatrix{E,L,<:AbstractFloat}; kw...) where {E,L} =
    ntuple(l -> round.(Tuple(mat[:,l]); kw...), Val(L))
displayvectors(mat::SMatrix{E,L,<:Integer}; kw...) where {E,L} =
    ntuple(l -> Tuple(mat[:,l]), Val(L))

# pseudoinverse of s times an integer n, so that it is an integer matrix (for accuracy)
pinvmultiple(s::SMatrix{L,0}) where {L} = (SMatrix{0,0,Int}(), 0)
function pinvmultiple(s::SMatrix{L,L´}) where {L,L´}
    L < L´ && throw(DimensionMismatch("Supercell dimensions $(L´) cannot exceed lattice dimensions $L"))
    qrfact = qr(s)
    n = det(qrfact.R)
    # Cannot check det(s) ≈ 0 because s can be non-square
    abs(n) ≈ 0 && throw(ErrorException("Supercell appears to be singular"))
    pinverse = inv(qrfact.R) * qrfact.Q'
    return round.(Int, n * inv(qrfact.R) * qrfact.Q'), round(Int, n)
end

@inline tuplejoin() = ()
@inline tuplejoin(x) = x
@inline tuplejoin(x, y) = (x..., y...)
@inline tuplejoin(x, y, z...) = (x..., tuplejoin(y, z...)...)

function isgrowing(vs::AbstractVector, i0 = 1)
    i0 > length(vs) && return true
    vprev = vs[i0]
    for i in i0 + 1:length(vs)
        v = vs[i]
        v <= vprev && return false
        vprev = v
    end
    return true
end

function ispositive(ndist)
    result = false
    for i in ndist
        i == 0 || (result = i > 0; break)
    end
    return result
end

isnonnegative(ndist) = iszero(ndist) || ispositive(ndist)

############################################################################################
######## _copy! and _add! #  Revise after #33589 is merged #################################
############################################################################################

_copy!(dest, src) = copy!(dest, src)
_copy!(dst::DenseMatrix{<:Number}, src::SparseMatrixCSC{<:Number}) = _fast_sparse_copy!(dst, src)
_copy!(dst::DenseMatrix{<:SMatrix{N,N}}, src::SparseMatrixCSC{<:SMatrix{N,N}}) where {N} = _fast_sparse_copy!(dst, src)
# _copy!(dst::AbstractMatrix{<:Number}, src::AbstractMatrix{<:SMatrix}) = _flatten_muladd!(dst, src)

_add!(dest, src, α) = _plain_muladd(dest, src, α)
_add!(dst::DenseMatrix{<:Number}, src::SparseMatrixCSC{<:Number}, α = 1) = _fast_sparse_muladd!(dst, src, α)
_add!(dst::DenseMatrix{<:SMatrix{N,N}}, src::SparseMatrixCSC{<:SMatrix{N,N}}, α = I) where {N} = _fast_sparse_muladd!(dst, src, α)
# _add!(dst::AbstractMatrix{<:Number}, src::AbstractMatrix{<:SMatrix}, α = I) = _flatten_muladd!(dst, src, α)

# Using broadcast .+= instead allocates unnecesarily
function _plain_muladd(dst, src, α)
    @boundscheck checkbounds(dst, axes(src)...)
    for i in eachindex(src)
        @inbounds dst[i] += α * src[i]
    end
    return dst
end

function _fast_sparse_copy!(dst::DenseMatrix{T}, src::SparseMatrixCSC) where {T}
    @boundscheck checkbounds(dst, axes(src)...)
    fill!(dst, zero(eltype(src)))
    for col in 1:size(src, 1)
        for p in nzrange(src, col)
            @inbounds dst[rowvals(src)[p], col] = nonzeros(src)[p]
        end
    end
    return dst
end

# Only needed for dense <- sparse (#33589), copy!(sparse, sparse) is fine
function _fast_sparse_muladd!(dst::DenseMatrix{T}, src::SparseMatrixCSC, α = I) where {T}
    @boundscheck checkbounds(dst, axes(src)...)
    for col in 1:size(src, 1)
        for p in nzrange(src, col)
            @inbounds dst[rowvals(src)[p], col] += α * nonzeros(src)[p]
        end
    end
    return dst
end

rclamp(r1::UnitRange, r2::UnitRange) = clamp(minimum(r1), extrema(r2)...):clamp(maximum(r1), extrema(r2)...)

# function _flatten_muladd!(dst::DenseMatrix{T}, src::SparseMatrixCSC{S}, α = zero(T)) where {T<:Number,N,S<:SMatrix{N,N}}
#     checkflattenaxes(dst, src)
#     iszero(α) ? fill!(dst, zero(eltype(src))) : (α != 1 && α != I && rmul!(dst, α))
#     for col in 1:size(src, 1)
#         for p in nzrange(src, col)
#             coffset = CartesianIndex(((rowvals(src)[p] - 1) * N, (col - 1) * N))
#             smatrix = nonzeros(src)[p]
#             for i in CartesianIndices((1:N, 1:N))
#                 @inbounds dst[coffset + i] += smatrix[i]
#             end
#         end
#     end
#     return dst
# end

# function _flatten_muladd!(dst::SparseMatrixCSC{T}, src::SparseMatrixCSC{S}, α = zero(T)) where {T<:Number,N,S<:SMatrix{N,N}}
#     rdst = rowvals(dst)
#     ndst = nonzeros(dst)
#     cdst = getcolptr(dst)
#     cdst[1] = 1

#     iszero(α) ? fill!(ndst, zero(eltype(S))) : (α != 1 && α != I && rmul!(ndst, α))
#     copy_structure!(dst, src)

#     p´ = col´ = 0
#     for col in 1:size(src, 2), j in 1:N
#         col´ += 1
#         for p in nzrange(src, col)
#             nz = nonzeros(src)[p]
#             row´ = (rowvals(src)[p] - 1) * N
#             for i in 1:N
#                 row´ += 1
#                 p´ += 1
#                 rdst[p´] = row´
#                 ndst[p´] += nz[i, j]
#             end
#         end
#         cdst[col´ + 1] = p´ + 1
#     end
#     return dst
# end

# function _flatten_copy!(dst::SparseMatrixCSC{T}, src::SparseMatrixCSC{S}) where {T<:Number,N,S<:SMatrix{N,N}}
#     checkflattenaxes(dst, src)
#     l = length(nonzeros(src))
#     l´ = N * N * l
#     rdst = resize!(rowvals(dst), l´)
#     ndst = resize!(nonzeros(dst), l´)
#     cdst = resize!(getcolptr(dst), size(src, 2) * N + 1)
#     cdst[1] = 1

#     fill!(ndst, zero(eltype(S)))

#     p´ = col´ = 0
#     for col in 1:size(src, 2), j in 1:N
#         col´ += 1
#         for p in nzrange(src, col)
#             row´ = (rowvals(src)[p] - 1) * N
#             nz = nonzeros(src)[p]
#             for i in 1:N
#                 row´ += 1
#                 p´ += 1
#                 rdst[p´] = row´
#                 ndst[p´] = nz[i, j]
#             end
#         end
#         cdst[col´ + 1] = p´ + 1
#     end
#     return dst
# end

# function _flatten_muladd!(dst::DenseMatrix{<:Number}, src::DenseMatrix{S}, α = zero(T)) where {T<:Number,N,S<:SMatrix{N,N}}
#     checkflattenaxes(dst, src)
#     iszero(α) ? fill!(dst, zero(eltype(S))) : (α != 1 && α != I && rmul!(dst, α))
#     c = CartesianIndices(src)
#     i0 = first(c)
#     for i in c
#         smatrix = src[i]
#         ioffset = (i - i0) * N
#         for j in 1:N, i in 1:N
#             @inbounds dst[ioffset + CartesianIndex(i, j)] += smatrix[i, j]
#         end
#     end
#     return dst
# end

# function checkflattenaxes(dst::AbstractMatrix{<:Number}, src::AbstractMatrix{S}) where {T,N,S<:SMatrix{N,N,T}}
#     Base.require_one_based_indexing(src)
#     axes(dst) == (1:(N * size(src, 1)), 1:(N * size(src, 2))) ||
#         throw(ArgumentError( "arrays must have the same axes (after flattening) for copy!"))
# end

# checkflattenaxes(dst::AbstractMatrix{<:SMatrix}, src::AbstractMatrix{<:Number}) =
#     throw(ArgumentError("unflattening not supported"))

# function checkflattenaxes(dst::AbstractMatrix, src::AbstractMatrix)
#     axes(dst) == axes(src) ||
#         throw(ArgumentError( "arrays must have the same axes for copy!"))
# end

# function copy_structure!(dst::SparseMatrixCSC{T}, src::SparseMatrixCSC) where {T}
#     @boundscheck checkflattenaxes(dst, src)
#     if !samestructure(dst, src)
#         N = blocksize(eltype(src))
#         rdst = rowvals(dst)
#         ndst = nonzeros(dst)
#         for col in 1:size(src, 2), p in nzrange(src, col)
#             row´ = (rowvals(src)[p] - 1) * N + 1
#             col´ = (col - 1) * N + 1
#             isstored(dst, row´, col´) || _sparse_insert!(dst, row´, col´, N)
#         end
#     end
#     return dst
# end

# samestructure(a, b) =
#     blocksize(eltype(a)) == blocksize(eltype(b)) && rowvals(a) == rowvals(b) && getcolptr(a) == getcolptr(b)


# function _sparse_insert!(s::SparseMatrixCSC{T}, row, col, N) where {T}
#     if N != 1         # Block insert
#         v = view(s, (row):(row + N - 1), (col):(col + N - 1))
#         v .= one(T)   # necessary to create new stored entries
#         v .= zero(T)  # here they are not removed, see #17404
#     else
#         s[row, col] = one(T)
#         s[row, col] = zero(T)
#     end
#     return s
# end

# # Taken fron Julialang/#33821
# function isstored(A::SparseMatrixCSC, i::Integer, j::Integer)
#     @boundscheck checkbounds(A, i, j)
#     rows = rowvals(A)
#     for istored in nzrange(A, j) # could do binary search if the row indices are sorted?
#         i == rows[istored] && return true
#     end
#     return false
# end

# blocksize(s::Type{<:SMatrix{N,N}}) where {N} = N
# blocksize(s::Type{<:Number}) = 1

############################################################################################

function pushapproxruns!(runs::AbstractVector{<:UnitRange}, list::AbstractVector{T},
                         offset = 0, degtol = sqrt(eps(real(T)))) where {T}
    len = length(list)
    len < 2 && return runs
    rmin = rmax = 1
    prev = list[1]
    @inbounds for i in 2:len
        next = list[i]
        if abs(next - prev) < degtol
            rmax = i
        else
            rmin < rmax && push!(runs, (offset + rmin):(offset + rmax))
            rmin = rmax = i
        end
        prev = next
    end
    rmin < rmax && push!(runs, (offset + rmin):(offset + rmax))
    return runs
end

function hasapproxruns(list::AbstractVector{T}, degtol = sqrt(eps(real(T)))) where {T}
    for i in 2:length(list)
        abs(list[i] - list[i-1]) < degtol && return true
    end
    return false
end

eltypevec(::AbstractMatrix{T}) where {T<:Number} = T
eltypevec(::AbstractMatrix{S}) where {N,T<:Number,S<:SMatrix{N,N,T}} = SVector{N,T}

# pinverse(s::SMatrix) = (qrfact = qr(s); return inv(qrfact.R) * qrfact.Q')

# padrightbottom(m::Matrix{T}, im, jm) where {T} = padrightbottom(m, zero(T), im, jm)

# function padrightbottom(m::Matrix{T}, zeroT::T, im, jm) where T
#     i0, j0 = size(m)
#     [i <= i0 && j<= j0 ? m[i,j] : zeroT for i in 1:im, j in 1:jm]
# end


tuplesort((a,b)::Tuple{<:Number,<:Number}) = a > b ? (b, a) : (a, b)
tuplesort(t::Tuple) = t
tuplesort(::Missing) = missing

# collectfirst(s::T, ss...) where {T} = _collectfirst((s,), ss...)
# _collectfirst(ts::NTuple{N,T}, s::T, ss...) where {N,T} = _collectfirst((ts..., s), ss...)
# _collectfirst(ts::Tuple, ss...) = (ts, ss)
# _collectfirst(ts::NTuple{N,System}, s::System, ss...) where {N} = _collectfirst((ts..., s), ss...)
# collectfirsttolast(ss...) = tuplejoin(reverse(collectfirst(ss...))...)



# allorderedpairs(v) = [(i, j) for i in v, j in v if i >= j]

# Like copyto! but with potentially different tensor orders (adapted from Base.copyto!)
function copyslice!(dest::AbstractArray{T1,N1}, Rdest::CartesianIndices{N1},
                    src::AbstractArray{T2,N2}, Rsrc::CartesianIndices{N2}, by = identity) where {T1,T2,N1,N2}
    isempty(Rdest) && return dest
    if length(Rdest) != length(Rsrc)
        throw(ArgumentError("source and destination must have same length (got $(length(Rsrc)) and $(length(Rdest)))"))
    end
    checkbounds(dest, first(Rdest))
    checkbounds(dest, last(Rdest))
    checkbounds(src, first(Rsrc))
    checkbounds(src, last(Rsrc))
    src′ = Base.unalias(dest, src)
    @inbounds for (Is, Id) in zip(Rsrc, Rdest)
        @inbounds dest[Id] = by(src′[Is])
    end
    return dest
end

function appendslice!(dest::AbstractArray, src::AbstractArray{T,N}, Rsrc::CartesianIndices{N}) where {T,N}
    checkbounds(src, first(Rsrc))
    checkbounds(src, last(Rsrc))
    Rdest = (length(dest) + 1):(length(dest) + length(Rsrc))
    resize!(dest, last(Rdest))
    src′ = Base.unalias(dest, src)
    @inbounds for (Is, Id) in zip(Rsrc, Rdest)
        @inbounds dest[Id] = src′[Is]
    end
    return dest
end

######################################################################
# Permutations (taken from Combinatorics.jl)
######################################################################

struct Permutations{T}
    a::T
    t::Int
end

Base.eltype(::Type{Permutations{T}}) where {T} = Vector{eltype(T)}

Base.length(p::Permutations) = (0 <= p.t <= length(p.a)) ? factorial(length(p.a), length(p.a)-p.t) : 0

"""
    permutations(a)
Generate all permutations of an indexable object `a` in lexicographic order. Because the number of permutations
can be very large, this function returns an iterator object.
Use `collect(permutations(a))` to get an array of all permutations.
"""
permutations(a) = Permutations(a, length(a))

"""
    permutations(a, t)
Generate all size `t` permutations of an indexable object `a`.
"""
function permutations(a, t::Integer)
    if t < 0
        t = length(a) + 1
    end
    Permutations(a, t)
end

function Base.iterate(p::Permutations, s = collect(1:length(p.a)))
    (!isempty(s) && max(s[1], p.t) > length(p.a) || (isempty(s) && p.t > 0)) && return
    nextpermutation(p.a, p.t ,s)
end

function nextpermutation(m, t, state)
    perm = [m[state[i]] for i in 1:t]
    n = length(state)
    if t <= 0
        return(perm, [n+1])
    end
    s = copy(state)
    if t < n
        j = t + 1
        while j <= n &&  s[t] >= s[j]; j+=1; end
    end
    if t < n && j <= n
        s[t], s[j] = s[j], s[t]
    else
        if t < n
            reverse!(s, t+1)
        end
        i = t - 1
        while i>=1 && s[i] >= s[i+1]; i -= 1; end
        if i > 0
            j = n
            while j>i && s[i] >= s[j]; j -= 1; end
            s[i], s[j] = s[j], s[i]
            reverse!(s, i+1)
        else
            s[1] = n+1
        end
    end
    return (perm, s)
end

# Taken from Combinatorics.jl
# TODO: This should really live in Base, otherwise it's type piracy
"""
    factorial(n, k)

Compute ``n!/k!``.
"""
function Base.factorial(n::T, k::T) where T<:Integer
    if k < 0 || n < 0 || k > n
        throw(DomainError((n, k), "n and k must be nonnegative with k ≤ n"))
    end
    f = one(T)
    while n > k
        f = Base.checked_mul(f, n)
        n -= 1
    end
    return f
end

Base.factorial(n::Integer, k::Integer) = factorial(promote(n, k)...)
