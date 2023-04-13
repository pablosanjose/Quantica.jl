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

padtuple(t, x, N) = ntuple(i -> i <= length(t) ? t[i] : x, N)

@noinline internalerror(func::String) =
    throw(ErrorException("Internal error in $func. Please file a bug report at https://github.com/pablosanjose/Quantica.jl/issues"))

@noinline argerror(msg) = throw(ArgumentError(msg))

@noinline boundserror(m, i) = throw(BoundsError(m, i))

@noinline checkmatrixsize(::UniformScaling, s) = nothing
@noinline checkmatrixsize(val, s) = (size(val, 1), size(val, 2)) == s ||
    throw(ArgumentError("Expected an block or matrix of size $s, got size $((size(val, 1), size(val, 2)))"))

function boundingbox(positions)
    isempty(positions) && argerror("Cannot find bounding box of an empty collection")
    posmin = posmax = first(positions)
    for pos in positions
        posmin = min.(posmin, pos)
        posmax = max.(posmax, pos)
    end
    return (posmin, posmax)
end

deleteif!(test, v::AbstractVector) = deleteat!(v, (i for (i, x) in enumerate(v) if test(x)))

copy_ifnotmissing(::Missing) = missing
copy_ifnotmissing(d) = copy(d)

merge_parameters!(p, m, ms...) = merge_parameters!(append!(p, parameters(m)), ms...)
merge_parameters!(p) = unique!(sort!(p))

typename(::T) where {T} = nameof(T)

chop(x::T) where {T<:Number} = ifelse(abs2(x) < eps(real(T)), zero(T), x)

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

# function get_or_push!(by, x, xs)
#     for x´ in xs
#         by(x) == by(x´) && return x´
#     end
#     push!(xs, x)
#     return x
# end

#endregion

############################################################################################
# SparseMatrixCSC tools
# all merged_* functions assume matching structure of sparse matrices
#region

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

function merged_mul!(C::SparseMatrixCSC{<:Number}, A::HybridSparseBlochMatrix, b::Number, α = 1, β = 0)
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

# symmetrize!/antisymmetrize! without changing sparse structure
function symmetrize!(s::SparseMatrixCSC, factor = 1)
    for col in axes(s, 2), ptr in nzrange(s, col)
        row = rowvals(s)[ptr]
        nonzeros(s)[ptr] += factor * s[col, row]
    end
    return s
end

#endregion

############################################################################################
# Dynamic package loader
#   This is in global Quantica scope to avoid name collisions
#   We also `import` instead of `using` to avoid collisions between several backends
#region

function ensureloaded(package::Symbol)
    if !isdefined(Quantica, package)
        @warn("Required package $package not loaded. Loading...")
        eval(:(import $package))
    end
    return nothing
end

#endregion
