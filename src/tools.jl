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

@inline iszero_or_empty(s::SVector{0}) = true
@inline iszero_or_empty(s::SVector) = iszero(s)
@inline iszero_or_empty(s) = isempty(s) || iszero(s)

ensuretuple(s::Tuple) = s
ensuretuple(s) = (s,)

tupletopair(s::Tuple) = Pair(s...)

tuplemost(t::NTuple{N,Any}) where {N} = ntuple(i -> t[i], Val(N-1))

filltuple(x, ::Val{L}) where {L} = ntuple(_ -> x, Val(L))

@inline tuplejoin() = ()
@inline tuplejoin(x) = x
@inline tuplejoin(x, y) = (x..., y...)
@inline tuplejoin(x, y, z...) = (x..., tuplejoin(y, z...)...)

tupleproduct(p1, p2) = tupleproduct(ensuretuple(p1), ensuretuple(p2))
tupleproduct(p1::NTuple{M,Any}, p2::NTuple{N,Any}) where {M,N} =
    ntuple(i -> (p1[1+fld(i-1, N)], p2[1+mod(i-1, N)]), Val(M * N))

mergetuples(ts::Tuple...) = keys(merge(tonamedtuple.(ts)...))
tonamedtuple(ts::NTuple{N,Any}) where {N} = NamedTuple{ts}(filltuple(0,Val(N)))


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

# Pad element type to a "larger" type
@inline padtotype(s::SMatrix{E,L}, ::Type{S}) where {E,L,E2,L2,S<:SMatrix{E2,L2}} =
    S(SMatrix{E2,E}(I) * s * SMatrix{L,L2}(I))
@inline padtotype(x::Number, ::Type{S}) where {E,L,S<:SMatrix{E,L}} =
    S(x * (SMatrix{E,1}(I) * SMatrix{1,L}(I)))
@inline padtotype(x::Number, ::Type{T}) where {T<:Number} = T(x)
@inline padtotype(u::UniformScaling, ::Type{T}) where {T<:Number} = T(u.λ)
@inline padtotype(u::UniformScaling, ::Type{S}) where {S<:SMatrix} = S(u)

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

# isnonnegative(ndist) = iszero(ndist) || ispositive(ndist)

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

tuplesort((a,b)::Tuple{<:Number,<:Number}) = a > b ? (b, a) : (a, b)
tuplesort(t::Tuple) = t
tuplesort(::Missing) = missing

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

# ######################################################################
# # SparseMatrixIJV
# ######################################################################

# struct SparseMatrixIJV{Tv,Ti<:Integer} <: AbstractSparseMatrix{Tv,Ti}
#     I::Vector{Ti}
#     J::Vector{Ti}
#     V::Vector{Tv}
#     m::Ti
#     n::Ti
#     klasttouch::Vector{Ti}
#     csrrowptr::Vector{Ti}
#     csrcolval::Vector{Ti}
#     csrnzval::Vector{Tv}
#     csccolptr::Vector{Ti}
#     cscrowval::Vector{Ti}
#     cscnzval::Vector{Tv}
# end

# SparseMatrixIJV{Tv}(m::Ti, n::Ti) where {Tv,Ti} = SparseMatrixIJV{Tv,Ti}(m,n)

# function SparseMatrixIJV{Tv,Ti}(m::Integer, n::Integer; hintnnz = 0) where {Tv,Ti}
#     I = Ti[]
#     J = Ti[]
#     V = Tv[]
#     klasttouch = Vector{Ti}(undef, n)
#     csrrowptr = Vector{Ti}(undef, m + 1)
#     csrcolval = Vector{Ti}()
#     csrnzval = Vector{Tv}()
#     csccolptr = Vector{Ti}(undef, n + 1)
#     cscrowval = Vector{Ti}()
#     cscnzval = Vector{Tv}()

#     if hintnnz > 0
#         sizehint!(I, hintnnz)
#         sizehint!(J, hintnnz)
#         sizehint!(V, hintnnz)
#         sizehint!(csrcolval, hintnnz)
#         sizehint!(csrnzval, hintnnz)
#         sizehint!(cscrowval, hintnnz)
#         sizehint!(cscnzval, hintnnz)
#     end

#     return SparseMatrixIJV{Tv,Ti}(I, J, V, m, n, klasttouch, csrrowptr, csrcolval, csrnzval,
#                                                              csccolptr, cscrowval, cscnzval)
# end

# Base.summary(::SparseMatrixIJV{Tv,Ti}) where {Tv,Ti} =
#     "SparseMatrixIJV{$Tv,$Ti} : Sparse matrix builder using the IJV format"

# function Base.show(io::IO, ::MIME"text/plain", s::SparseMatrixIJV)
#     i = get(io, :indent, "")
#     print(io, i, summary(s), "\n", "$i  Nonzero elements : $(length(s.I))")
# end

# function Base.push!(s::SparseMatrixIJV, (i, j, v))
#     push!(s.I, i)
#     push!(s.J, j)
#     push!(s.V, v)
#     return s
# end

# function SparseArrays.sparse(s::SparseMatrixIJV)
#     numnz = length(s.I)
#     resize!(s.csrcolval, numnz)
#     resize!(s.csrnzval,  numnz)
#     resize!(s.cscrowval, numnz)
#     resize!(s.cscnzval,  numnz)
#     return SparseArrays.sparse!(s.I, s.J, s.V, s.m, s.n, +, s.klasttouch,
#         s.csrrowptr, s.csrcolval, s.csrnzval, s.csccolptr, s.cscrowval, s.cscnzval)
# end

# Base.size(s::SparseMatrixIJV) = (s.m, s.n)

############################################################################################
######## fast sparse copy #  Revise after #33589 is merged #################################
############################################################################################

# Using broadcast .+= instead allocates unnecesarily
function _plain_muladd(dst, src, α)
    @boundscheck checkbounds(dst, axes(src)...)
    for i in eachindex(src)
        @inbounds dst[i] += α * src[i]
    end
    return dst
end

# Only needed for dense <- sparse (#33589), copy!(sparse, sparse) is fine
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

function _fast_sparse_muladd!(dst::AbstractMatrix{T}, src::SparseMatrixCSC, α = I) where {T}
    @boundscheck checkbounds(dst, axes(src)...)
    for col in 1:size(src, 1)
        for p in nzrange(src, col)
            @inbounds dst[rowvals(src)[p], col] += α * nonzeros(src)[p]
        end
    end
    return dst
end

rclamp(r1::UnitRange, r2::UnitRange) = clamp(minimum(r1), extrema(r2)...):clamp(maximum(r1), extrema(r2)...)