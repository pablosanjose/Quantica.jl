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

merge_parameters!(p, m, ms...) = merge_parameters!(append!(p, parameters(m)), ms...)
merge_parameters!(p) = unique!(sort!(p))

typename(::T) where {T} = nameof(T)

chop(x::T) where {T<:Real} = ifelse(abs2(x) < eps(real(T)), zero(T), x)
chop(x::Complex) = chop(real(x)) + im*chop(imag(x))
chop(xs) = chop.(xs)

# Flattens matrix of Matrix{<:Number} into a matrix of Number's
function mortar(ms::AbstractMatrix{M}) where {C<:Number,M<:AbstractMatrix{C}}
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
# faspath for generators or other collections
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


# function get_or_push!(by, x, xs)
#     for x´ in xs
#         by(x) == by(x´) && return x´
#     end
#     push!(xs, x)
#     return x
# end

# # split runs of consecutive elements of itr such that by(itr[i]) == by(itr[i+1]),
# # applying post to each. Assume no type widening. Each run is reduced with reduce(by, run)
# splitruns(itr; kw...) = splitruns(identity, itr; kw...)

# function splitruns(post, itr; by = identity, reduce = missing)
#     i0, itr´ = Iterators.peel(itr)
#     run = [post(i0)]
#     rrun = reduce === missing ? run : reduce(by(i0), run)
#     runs = [rrun]
#     for i in itr´
#         p = post(i)
#         byi, byi0 = by(i), by(i0)
#         if byi == byi0
#             push!(run, p)
#         else
#             run = [p]
#             rrun = reduce === missing ? run : reduce(byi, run)
#             push!(runs, rrun)
#         end
#         i0 = i
#     end
#     return runs
# end

#endregion

############################################################################################
# SparseMatrixCSC tools
# all merged_* functions assume matching structure of sparse matrices
#region

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

function merged_mul!(C::SparseMatrixCSC{<:Number}, ::OrbitalBlockStructure, A::SparseMatrixCSC{B}, b::Number, α = 1, β = 0) where {B<:Complex}
    nzA = nonzeros(A)
    nzC = nonzeros(C)
    αb = α * b
    if length(nzA) == length(nzC)  # assume idential structure (C has merged structure)
        @. nzC = muladd(αb, nzA, β * nzC)
    else
        # A has less elements than C
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

############################################################################################
# Dynamic package loader
#   This is in global Quantica scope to avoid name collisions
#   We also `import` instead of `using` to avoid collisions between different backends
#region

function ensureloaded(package::Symbol)
    if !isdefined(Quantica, package)
        @warn("Required package $package not loaded. Loading...")
        eval(:(import $package))
    end
    return nothing
end

#endregion
