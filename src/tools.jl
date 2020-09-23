toSMatrix() = SMatrix{0,0,Float64}()
toSMatrix(s) = toSMatrix(tuple(s))
toSMatrix(ss::NTuple{M,NTuple{N,Number}}) where {N,M} = toSMatrix(SVector{N}.(ss))
toSMatrix(ss::NTuple{M,SVector{N}}) where {N,M} = hcat(ss...)

toSMatrix(::Type{T}, ss) where {T<:Number} = _toSMatrix(T, toSMatrix(ss))
_toSMatrix(::Type{T}, s::SMatrix{N,M}) where {N,M,T} = convert(SMatrix{N,M,T}, s)
# Dynamic dispatch
toSMatrix(s::AbstractMatrix) = SMatrix{size(s,1), size(s,2)}(s)
toSMatrix(s::AbstractVector) = toSMatrix(Tuple(s))

toSVector(::Tuple{}) = SVector{0,Float64}()
toSVector(v::SVector) = v
toSVector(v::NTuple{N,Any}) where {N} = SVector(v)
toSVector(x::Number) = SVector{1}(x)
toSVector(::Type{T}, v) where {T} = T.(toSVector(v))
toSVector(::Type{T}, ::Tuple{}) where {T} = SVector{0,T}()
# Dynamic dispatch
toSVector(v::AbstractVector) = SVector(Tuple(v))

unitvector(::Type{SVector{L,T}}, i) where {L,T} =
    SVector{L,T}(ntuple(j -> j == i ? one(T) : zero(T), Val(L)))

ensuretuple(s::Tuple) = s
ensuretuple(s) = (s,)

indstopair(s::Tuple) = Pair(last(s), first(s))

tuplemost(t::NTuple{N,Any}) where {N} = ntuple(i -> t[i], Val(N-1))

filltuple(x, ::Val{L}) where {L} = ntuple(_ -> x, Val(L))
filltuple(x, ::NTuple{N,Any}) where {N} = ntuple(_ -> x, Val(N))

@inline tuplejoin() = ()
@inline tuplejoin(x) = x
@inline tuplejoin(x, y) = (x..., y...)
@inline tuplejoin(x, y, z...) = (x..., tuplejoin(y, z...)...)

tuplesplice(s::NTuple{N,T}, ind, el) where {N,T} = ntuple(i -> i === ind ? T(el) : s[i], Val(N))

tupleproduct(p1, p2) = tupleproduct(ensuretuple(p1), ensuretuple(p2))
tupleproduct(p1::NTuple{M,Any}, p2::NTuple{N,Any}) where {M,N} =
    ntuple(i -> (p1[1+fld(i-1, N)], p2[1+mod(i-1, N)]), Val(M * N))

tupleswapfront(tup::NTuple{L}, (i, j)) where {L} =
    i < j ? swap(swap(tup, i => 1), j => 2) : swap(swap(tup, j => 2), i => 1)

swap(tup::NTuple{L}, (i, i´)) where {L} =
    ntuple(l -> tup[ifelse(l == i´, i, ifelse(l == i, i´, l))], Val(L))

tuplepairs(::Val{V}) where {V} = tuplepairs((), ntuple(identity, Val(V)))
tuplepairs(c::Tuple, ::Tuple{}) = c

function tuplepairs(c::Tuple, r::NTuple{V}) where {V}
    t = Base.tail(r)
    c´ = (c..., tuple.(first(r), t)...)
    return tuplepairs(c´, t)
end

mergetuples(ts...) = keys(merge(tonamedtuple.(ts)...))
tonamedtuple(ts::Val{T}) where {T} = NamedTuple{T}(filltuple(0,T))

function deletemultiple_nocheck(dn::SVector{N}, axes::NTuple{M,Int}) where {N,M}
    ind = first(axes)
    dn´ = deleteat(dn, ind)
    taxes = Base.tail(axes)
    axes´ = taxes .- (taxes .> ind)
    return deletemultiple_nocheck(dn´, axes´)
end
deletemultiple_nocheck(dn::SVector, axes::Tuple{}) = dn

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
padright(t::NTuple{N´,Any}, x, ::Val{N}) where {N´,N} = ntuple(i -> i > N´ ? x : t[i], Val(N))
padright(t::NTuple{N´,Any}, ::Val{N}) where {N´,N} = ntuple(i -> i > N´ ? 0 : t[i], Val(N))

# Pad element type to a "larger" type
@inline padtotype(s::SMatrix{E,L}, ::Type{S}) where {E,L,E2,L2,S<:SMatrix{E2,L2}} =
    S(SMatrix{E2,E}(I) * s * SMatrix{L,L2}(I))
@inline padtotype(s::StaticVector, ::Type{S}) where {N,T,S<:SVector{N,T}} =
    padright(T.(s), Val(N))
@inline padtotype(x::Number, ::Type{S}) where {E,L,S<:SMatrix{E,L}} =
    S(x * (SMatrix{E,1}(I) * SMatrix{1,L}(I)))
@inline padtotype(s::Number, ::Type{S}) where {N,T,S<:SVector{N,T}} =
    padright(SA[T(s)], Val(N))
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

# pseudoinverse of supercell s times an integer n, so that it is an integer matrix (for accuracy)
pinvmultiple(s::SMatrix{L,0}) where {L} = (SMatrix{0,0,Int}(), 0)
function pinvmultiple(s::SMatrix{L,L´}) where {L,L´}
    L < L´ && throw(DimensionMismatch("Supercell dimensions $(L´) cannot exceed lattice dimensions $L"))
    qrfact = qr(s)
    # Cannot check det(s) ≈ 0 because s can be non-square
    det(qrfact.R) ≈ 0 && throw(ErrorException("Supercell appears to be singular"))
    pinverse = inv(qrfact.R) * qrfact.Q'
    n = round.(Int, det(s's))
    npinverse = round.(Int, n * pinverse)
    return npinverse, n
end

pinverse(::SMatrix{E,0,T}) where {E,T} = SMatrix{0,E,T}() # BUG: workaround StaticArrays bug SMatrix{E,0,T}()'

function pinverse(m::SMatrix)
    qrm = qr(m)
    return inv(qrm.R) * qrm.Q'
end

_blockdiag(s1::SMatrix{E1,L1,T1}, s2::SMatrix{E2,L2,T2}) where {E1,L1,T1,E2,L2,T2} = hcat(
    ntuple(j->vcat(s1[:,j], zero(SVector{E2,T2})), Val(L1))...,
    ntuple(j->vcat(zero(SVector{E1,T1}), s2[:,j]), Val(L2))...)

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

chop(x::T, x0 = one(T)) where {T<:Real} = ifelse(abs(x) < √eps(T(x0)), zero(T), x)
chop(x::C, x0 = one(R)) where {R<:Real,C<:Complex{R}} = chop(real(x), x0) + im*chop(imag(x), x0)

function unique_sorted_approx!(v::AbstractVector{T}) where {T}
    i = 1
    xprev = first(v)
    for j in 2:length(v)
        if v[j] ≈ xprev
            xprev = v[j]
        else
            i += 1
            xprev = v[i] = v[j]
        end
    end
    resize!(v, i)
    return v
end

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

# Like copyto! but with potentially different tensor orders (adapted from Base.copyto!, see #33588)
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
# convert a matrix/number block to a matrix/inlinematrix string
######################################################################

_isreal(x) = all(o -> imag(o) ≈ 0, x)
_isimag(x) = all(o -> real(o) ≈ 0, x)

matrixstring(row, x) = string("Onsite[$row] : ", _matrixstring(x))
matrixstring(row, col, x) = string("Hopping[$row, $col] : ", _matrixstring(x))

matrixstring_inline(row, x) = string("Onsite[$row] : ", _matrixstring_inline(x))
matrixstring_inline(row, col, x) = string("Hopping[$row, $col] : ", _matrixstring_inline(x))

_matrixstring(x::Number) = numberstring(x)
_matrixstring_inline(x::Number) = numberstring(x)
function _matrixstring(s::SMatrix)
    ss = repr("text/plain", s)
    pos = findfirst(isequal('\n'), ss)
    return pos === nothing ? ss : ss[pos:end]
end

function _matrixstring_inline(s::SMatrix{N}) where {N}
    stxt = numberstring.(transpose(s))
    stxt´ = vcat(stxt, SMatrix{1,N}(ntuple(_->";", Val(N))))
    return string("[", stxt´[1:end-1]..., "]")
end

numberstring(x) = _isreal(x) ? string(" ", real(x)) : _isimag(x) ? string(" ", imag(x), "im") : string(" ", x)

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

############################################################################################
######## fast sparse copy #  Revise after #33589 is merged #################################
############################################################################################

# Using broadcast .+= instead allocates unnecesarily
function _plain_muladd!(dst, src, α)
    @boundscheck checkbounds(dst, axes(src)...)
    for i in eachindex(src)
        @inbounds dst[i] += α * src[i]
    end
    return dst
end

# Only needed for dense <- sparse (#33589), copy!(sparse, sparse) is fine in v1.5+
function _fast_sparse_copy!(dst::AbstractMatrix{T}, src::SparseMatrixCSC) where {T}
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

rclamp(r1::UnitRange, r2::UnitRange) = isempty(r1) ? r1 : clamp(minimum(r1), extrema(r2)...):clamp(maximum(r1), extrema(r2)...)

iclamp(minmax, r::Missing) = minmax
iclamp((x1, x2), (xmin, xmax)) = (max(x1, xmin), min(x2, xmax))