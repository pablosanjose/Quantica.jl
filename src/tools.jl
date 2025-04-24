############################################################################################
# Misc tools
#region

rdr((r1, r2)::Pair) = (0.5 * (r1 + r2), r2 - r1)

@inline tuplejoin() = ()
@inline tuplejoin(x) = x
@inline tuplejoin(x, y) = (x..., y...)
@inline tuplejoin(x, y, z...) = (x..., tuplejoin(y, z...)...)

@inline tupleflatten() = ()
@inline tupleflatten(x::Tuple) = x
@inline tupleflatten(x0, xs...) = tupleflatten((x0,), xs...)
@inline tupleflatten(x0::Tuple, x1, xs...) = tupleflatten((x0..., x1), xs...)
@inline tupleflatten(x0::Tuple, x1::Tuple, xs...) = tupleflatten((x0..., x1...), xs...)

tuplefill(x, ::Val{N}) where {N} = ntuple(Returns(x), Val(N))

padtuple(t, x, N) = ntuple(i -> i <= length(t) ? t[i] : x, N)

@inline tupletake(x, ::Val{N}) where {N} = ntuple(i -> x[i], Val(N))

@inline function smatrixtake(s::SMatrix, ::Val{N}) where {N}
    is = SVector{N,Int}(1:N)
    return s[is, is]
end

# find indices in range 1:L that are not in inds, assuming inds is sorted and unique
inds_complement(val, inds::SVector) = SVector(inds_complement(val, Tuple(inds)))
inds_complement(::Val{L}, inds::NTuple{L´,Integer}) where {L,L´} =
    _inds_complement(ntuple(zero, Val(L-L´)), 1, 1, inds...)

function _inds_complement(v::NTuple{N}, vn, rn, i, inds...) where {N}
    if i != rn
        # static analog of v[vn] = rn
        v´ = ntuple(i -> ifelse(i == vn, rn, v[i]), Val(N))
        return _inds_complement(v´, vn+1, rn+1, inds...)
    else
        return _inds_complement(v, vn, rn+1, i, inds...)
    end
end
_inds_complement(v, vn, rn) = v

# repeated elements in an integer array
onlyrepeated(v) = onlyrepeated!(copy(v))
function onlyrepeated!(v)
    sort!(v)
    i´ = 0
    for (i, rng) in enumerate(equalruns(v))
        if length(rng) > 1
            i´ += 1
            v[i´] = v[first(rng)]
        end
    end
    resize!(v, i´)
    return v
end

@noinline internalerror(func::String) =
    throw(ErrorException("Internal error in $func. Please file a bug report at https://github.com/pablosanjose/Quantica.jl/issues"))

@noinline argerror(msg) = throw(ArgumentError(msg))

@noinline boundserror(m, i) = throw(BoundsError(m, i))

@noinline checkmatrixsize(::UniformScaling, s) = nothing
@noinline checkmatrixsize(val, s) = (size(val, 1), size(val, 2)) == s ||
    throw(ArgumentError("Expected a block or matrix of size $s, got size $((size(val, 1), size(val, 2)))"))

@noinline function_not_defined(name) = argerror("Function $name not defined for the requested types")

unitvector(i, ::Type{SVector{L,T}}) where {L,T} =
    SVector{L,T}(ntuple(j -> j == i ? one(T) : zero(T), Val(L)))

function padright(pt, ::Val{L}) where {L}
    T = eltype(pt)
    L´ = length(pt)
    return SVector(ntuple(i -> i > L´ ? zero(T) : pt[i], Val(L)))
end

function boundingbox(positions)
    isempty(positions) && argerror("Cannot find bounding box of an empty collection")
    posmin = posmax = first(positions)
    for pos in positions
        posmin = min.(posmin, pos)
        posmax = max.(posmax, pos)
    end
    return (posmin, posmax)
end

copy_ifnotmissing(::Missing) = missing
copy_ifnotmissing(d) = copy(d)

typename(::T) where {T} = nameof(T)

chopsmall(x::T, atol = sqrt(eps(real(T)))) where {T<:Real} =
    ifelse(abs(x) < atol, zero(T), x)
chopsmall(x::C, atol = sqrt(eps(real(C)))) where {C<:Complex} =
    chopsmall(real(x), atol) + im*chopsmall(imag(x), atol)
chopsmall(xs, atol) = chopsmall.(xs, Ref(atol))
chopsmall(xs) = chopsmall.(xs)
chopsmall(xs::UniformScaling, atol...) = I * chopsmall(xs.λ, atol...)

mul_scalar_or_array!(x::Number, factor) = factor * x
mul_scalar_or_array!(x::Tuple, factor) = factor .* x
mul_scalar_or_array!(x::AbstractArray, factor) = (x .*= factor; x)

# Flattens matrix of Matrix{<:Number} into a matrix of Number's
function mortar(ms::AbstractMatrix{M}) where {C<:Number,M<:Matrix{C}}
    isempty(ms) && return convert(Matrix{C}, ms)
    mrows = size.(ms, 1)
    mcols = size.(ms, 2)
    allequal(eachrow(mcols)) && allequal(eachcol(mrows)) ||
        internalerror("mortar: inconsistent rows or columns")
    roff = lengths_to_offsets(view(mrows, :, 1))
    coff = lengths_to_offsets(view(mcols, 1, :))
    mat = zeros(C, last(roff), last(coff))
    for c in CartesianIndices(ms)
        src = ms[c]
        i, j = Tuple(c)
        Rdst = CartesianIndices((roff[i]+1:roff[i+1], coff[j]+1:coff[j+1]))
        Rsrc = CartesianIndices(src)
        copyto!(mat, Rdst, src, Rsrc)
    end
    return mat
end

#equivalent with SparseMatrixCSC
function mortar(ms::AbstractMatrix{M}) where {C<:Number,M<:SparseMatrixCSC{C}}
    isempty(ms) && return convert(SparseMatrixCSC{C,Int}, ms)
    mrows = size.(ms, 1)
    mcols = size.(ms, 2)
    allequal(eachrow(mcols)) && allequal(eachcol(mrows)) ||
        internalerror("mortar: inconsistent rows or columns")
    totalrows = sum(i -> size(ms[i, 1], 1), axes(ms, 1))
    totalcols = sum(i -> size(ms[1, i], 2), axes(ms, 2))
    totalnnz = sum(nnz, ms)
    b = CSC{C}(totalcols, totalnnz)
    for mj in axes(ms, 2), col in axes(ms[1, mj], 2)
        rowoffset = 0
        for mi in axes(ms, 1)
            m = ms[mi, mj]
            for ptr in nzrange(m, col)
                row = rowoffset + rowvals(m)[ptr]
                val = nonzeros(m)[ptr]
                pushtocolumn!(b, row, val)
            end
            rowoffset += size(m, 1)
        end
        finalizecolumn!(b)
    end
    return sparse(b, totalrows, totalcols)
end

# # faspath for generators or other collections
mortar(ms) = length(ms) == 1 ? only(ms) : mortar(collect(ms))

# equivalent to mat = I[:, cols]. Useful for Green function source
# no check of mat size vs cols is done
function one!(mat::AbstractArray{T}, cols = axes(mat, 2)) where {T}
    fill!(mat, zero(T))
    for (col, row) in enumerate(cols)
        mat[row, col] = one(T)
    end
    return mat
end

one!(mat::AbstractArray, ::Colon) = one!(mat)

lengths_to_offsets(v::NTuple{<:Any,Integer}) = (0, cumsum(v)...)
lengths_to_offsets(v) = prepend!(cumsum(v), 0)
lengths_to_offsets(f::Function, v) = prepend!(accumulate((i,j) -> i + f(j), v; init = 0), 0)

# remove elements of xs after and including the first for which f(x) is true
function cut_tail!(f, xs)
    i = 1
    for outer i in eachindex(xs)
        f(xs[i]) && break
    end
    resize!(xs, i-1)
    return xs
end

# fast tr(A*B)
trace_prod(A, B) = (check_sizes(A,B); unsafe_trace_prod(A, B))

unsafe_trace_prod(A::AbstractMatrix, B::AbstractMatrix) = sum(splat(*), zip(transpose(A), B))
unsafe_trace_prod(A::Number, B::AbstractMatrix) = A*tr(B)
unsafe_trace_prod(A::AbstractMatrix, B::Number) = unsafe_trace_prod(B, A)
unsafe_trace_prod(A::UniformScaling, B::AbstractMatrix) = A.λ*tr(B)
unsafe_trace_prod(A::AbstractMatrix, B::UniformScaling) = unsafe_trace_prod(B, A)
unsafe_trace_prod(A::Diagonal, B::Diagonal) = sum(i -> A[i] * B[i], axes(A,1))
unsafe_trace_prod(A::Diagonal, B::AbstractMatrix) = sum(i -> A[i] * B[i, i], axes(A,1))
unsafe_trace_prod(A::AbstractMatrix, B::Diagonal) = unsafe_trace_prod(B, A)
unsafe_trace_prod(A::Diagonal, B::Number) = only(A) * B
unsafe_trace_prod(A::Number, B::Diagonal) = unsafe_trace_prod(B, A)
unsafe_trace_prod(A::Union{SMatrix,UniformScaling,Number}, B::Union{SMatrix,UniformScaling,Number}) =
     tr(A*B)

check_sizes(A::AbstractMatrix,B::AbstractMatrix) = size(A,2) == size(B,1) ||
    throw(DimensionMismatch("A has dimensions $(size(A)) but B has dimensions $(size(B))"))
check_sizes(_, _) = nothing

# Taken from Base julia, now deprecated there
function permute!!(a, p::AbstractVector{<:Integer})
    Base.require_one_based_indexing(a, p)
    count = 0
    start = 0
    while count < length(a)
        ptr = start = findnext(!iszero, p, start+1)::Int
        temp = a[start]
        next = p[start]
        count += 1
        while next != start
            a[ptr] = a[next]
            p[ptr] = 0
            ptr = next
            next = p[next]
            count += 1
        end
        a[ptr] = temp
        p[ptr] = 0
    end
    a
end

# like permute!! applied to each row of a, in-place in a (overwriting p).
function permutecols!!(a::AbstractMatrix, p::AbstractVector{<:Integer})
    Base.require_one_based_indexing(a, p)
    count = 0
    start = 0
    while count < length(p)
        ptr = start = findnext(!iszero, p, start+1)::Int
        next = p[start]
        count += 1
        while next != start
            swapcols!(a, ptr, next)
            p[ptr] = 0
            ptr = next
            next = p[next]
            count += 1
        end
        p[ptr] = 0
    end
    a
end

Base.@propagate_inbounds function swapcols!(a::AbstractMatrix, i, j)
    i == j && return
    cols = axes(a,2)
    @boundscheck i in cols || throw(BoundsError(a, (:,i)))
    @boundscheck j in cols || throw(BoundsError(a, (:,j)))
    for k in axes(a,1)
        @inbounds a[k,i],a[k,j] = a[k,j],a[k,i]
    end
end

is_square(a::AbstractMatrix) = size(a, 1) == size(a, 2)
is_square(a) = false

maybe_broadcast!(::typeof(identity), x) = x
maybe_broadcast!(f, x) = (x .= f.(x); x)

#endregion

############################################################################################
# SparseMatrixCSC tools
# all merged_* functions assume matching structure of sparse matrices
#region

function sparse_pointer(mat::AbstractSparseMatrixCSC, (row, col))
    rows = rowvals(mat)
    for ptr in nzrange(mat, col)
        rows[ptr] == row && return ptr
    end
    argerror("Element ($row, $col) not stored in sparse matrix")
end

function nnzdiag(b::AbstractSparseMatrixCSC)
    count = 0
    rowptrs = rowvals(b)
    for col in 1:size(b, 2)
        for ptr in nzrange(b, col)
            rowptrs[ptr] == col && (count += 1; break)
        end
    end
    return count
end

stored_rows(m::AbstractSparseMatrixCSC) = unique!(sort!(copy(rowvals(m))))

function stored_cols(m::AbstractSparseMatrixCSC)
    cols = Int[]
    for col in axes(m, 2)
        isempty(nzrange(m, col)) || push!(cols, col)
    end
    return cols
end

# merge several sparse matrices onto the first using only structural zeros
function merge_sparse(mats, ::Type{B} = eltype(first(mats))) where {B}
    mat0 = first(mats)
    nrows, ncols = size(mat0)
    nrows == ncols || throw(ArgumentError("Internal error: matrix not square"))
    nnzguess = sum(mat -> nnz(mat), mats)
    collector = CSC{B}(ncols, nnzguess)
    for col in 1:ncols
        for (n, mat) in enumerate(mats)
            vals = nonzeros(mat)
            rows = rowvals(mat)
            for p in nzrange(mat, col)
                val = zero(B)
                row = rows[p]
                pushtocolumn!(collector, row, val, false) # skips repeated rows
            end
        end
        finalizecolumn!(collector)
    end
    matrix = sparse(collector, ncols, ncols)
    return matrix
end

function merged_mul!(C::SparseMatrixCSC{<:Number}, A::HybridSparseMatrix, b::Number, α = 1, β = 0)
    bs = blockstructure(A)
    if needs_flat_sync(A)
        merged_mul!(C, bs, unflat(A), b, α, β)
    else
        merged_mul!(C, bs, flat(A), b, α, β)
    end
    return C
end

merged_mul!(C::SparseMatrixCSC{<:Number}, ::OrbitalBlockStructure, A::SparseMatrixCSC{B}, b::Number, α = 1, β = 0) where {B<:Complex} =
    merged_flat_mul!(C, A, b, α, β)

function merged_flat_mul!(C::SparseMatrixCSC{<:Number}, A::SparseMatrixCSC{<:Number}, b::Number, α = 1, β = 0)
    nzA = nonzeros(A)
    nzC = nonzeros(C)
    αb = α * b
    if length(nzA) == length(nzC)  # assume identical structure (C has merged structure)
        @. nzC = muladd(αb, nzA, β * nzC)
    else
        # A has less elements than C, but C includes all of A
        for col in axes(A, 2), p in nzrange(A, col)
            row = rowvals(A)[p]
            for p´ in nzrange(C, col)
                row´ = rowvals(C)[p´]
                if row == row´
                    nzC[p´] = muladd(αb, nzA[p], β * nzC[p´])
                    break
                end
            end
        end
    end
    return C
end

function merged_mul!(C::SparseMatrixCSC{<:Number}, bs::OrbitalBlockStructure{B}, A::SparseMatrixCSC{B}, b::Number, α = 1, β = 0) where {B<:MatrixElementNonscalarType}
    colsA = axes(A, 2)
    rowsA = rowvals(A)
    valsA = nonzeros(A)
    rowsC = rowvals(C)
    valsC = nonzeros(C)
    αb = α * b
    colC = 1
    for colA in colsA
        N = blocksize(bs, colA)
        for colN in 1:N
            ptrsA, ptrsC = nzrange(A, colA), nzrange(C, colC)
            ptrA, ptrC = first(ptrsA), first(ptrsC)
            while ptrA in ptrsA && ptrC in ptrsC
                rowA, rowC = rowsA[ptrA], rowsC[ptrC]
                rngflat = flatrange(bs, rowA)
                rowAflat, N´ = first(rngflat), length(rngflat)
                if rowAflat == rowC
                    valA = valsA[ptrA]
                    for rowN in 1:N´
                        valsC[ptrC] = muladd(αb, valA[rowN, colN], β * valsC[ptrC])
                        ptrC += 1
                    end
                elseif rowAflat > rowC
                    ptrC += N´
                else
                    ptrA += 1
                end
            end
            colC += 1
        end
    end
    return C
end

#endregion
