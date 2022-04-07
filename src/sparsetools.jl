############################################################################################
# MatrixElementType
#region

struct SMatrixView{N,M,T}
    s::SMatrix{N,M,T}
    SMatrixView{N,M,T}(s) where {N,M,T} = new(s)
end

SMatrixView(s::SMatrix{N,M,T}) where {N,M,T} = SMatrixView{N,M,T}(s)

SMatrixView{N,M}(s) where {N,M} = SMatrixView(SMatrix{N,M}(s))

Base.parent(s::SMatrixView) = s.s

Base.view(s::SMatrixView, i::Integer...) = view(s.s, i...)

const MatrixElementType{T} = Union{
    Complex{T},
    SMatrix{N,N,Complex{T}} where {N},
    SMatrixView{N,N,Complex{T}} where {N}}

const MatrixElementUniformType{T} = Union{
    Complex{T},
    SMatrix{N,N,Complex{T}} where {N}}

const MatrixElementNonscalarType{T} = Union{
    SMatrix{N,N,Complex{T}} where {N},
    SMatrixView{N,N,Complex{T}} where {N}}

#endregion

############################################################################################
# BlockStructure
#region

# struct BlockStructure{B<:MatrixElementType}
struct BlockStructure{B}
    blocksizes::Vector{Int}       # block sizes for each sublattice
    subsizes::Vector{Int}         # number of blocks (sites) in each sublattice
    function BlockStructure{B}(blocksizes, subsizes) where {B}
        subsizes´ = Quantica.sanitize_Vector_of_Type(Int, subsizes)
        # This checks also that they are of equal length
        blocksizes´ = Quantica.sanitize_Vector_of_Type(Int, length(subsizes´), blocksizes)
        return new(blocksizes´, subsizes´)
    end
end

## Constructors ##

@inline function BlockStructure(T, blocksizes, subsizes)
    B = blocktype(T, blocksizes)
    return BlockStructure{B}(blocksizes, subsizes)
end

blocktype(T::Type, norbs) = blocktype(T, val_maximum(norbs))
blocktype(::Type{T}, m::Val{1}) where {T} = Complex{T}
blocktype(::Type{T}, m::Val{N}) where {T,N} = SMatrix{N,N,Complex{T},N*N}

val_maximum(n::Int) = Val(n)
val_maximum(ns) = Val(maximum(argval.(ns)))

argval(::Val{N}) where {N} = N
argval(n::Int) = n

## API ##

blocksizes(b::BlockStructure) = b.blocksizes

subsizes(b::BlockStructure) = b.subsizes

blocksize(b::BlockStructure, i, j) = (blocksize(b, i), blocksize(b, j))

blocksize(b::BlockStructure, i) = length(flatrange(b, i))

flatsize(b::BlockStructure) = blocksizes(b)' * subsizes(b)

unflatsize(b::BlockStructure) = sum(subsizes(b))

flatindex(b::BlockStructure, i) = first(flatrange(b, i))

# Basic relation: iflat - 1 == (iunflat - soffset - 1) * b + soffset´
function flatrange(b::BlockStructure, iflat::Integer)
    soffset  = 0
    soffset´ = 0
    @boundscheck(iflat < 0 && blockbounds_error())
    @inbounds for (s, b) in zip(b.subsizes, b.blocksizes)
        if soffset + s >= iflat
            offset = muladd(i - soffset - 1, b, soffset´)
            return offset+1:offset+b
        end
        soffset  += s
        soffset´ += b * s
    end
    @boundscheck(blockbounds_error())
end

function unflatindex(b::BlockStructure, iflat::Integer)
    soffset  = 0
    soffset´ = 0
    @boundscheck(i < 0 && blockbounds_error())
    @inbounds for (s, b) in zip(b.subsizes, b.blocksizes)
        if soffset + s >= iflat
            iunflat = (iflat - soffset´ - 1) ÷ b + soffset + 1
            return iunflat, b
        end
        soffset  += s
        soffset´ += b * s
    end
    @boundscheck(blockbounds_error())
end

@noinline blockbounds_error() = throw(BoundsError())

#endregion

############################################################################################
# HybridSparseMatrixCSC - wraps blocked + flat versions of the same SparseMatrixCSC
#region

struct HybridSparseMatrixCSC{T,B<:MatrixElementType{T}} <: SparseArrays.AbstractSparseMatrixCSC{B,Int}
    blockstruct::BlockStructure{B}
    unflat::SparseMatrixCSC{B,Int}
    flat::SparseMatrixCSC{Complex{T},Int}
    needs_sync::Ref{Int}  # 0 = in sync, 1 = flat needs sync, -1 = unflat need sync
end

## Constructors ##

HybridSparseMatrixCSC(b::BlockStructure{Complex{T}}, flat::SparseMatrixCSC{Complex{T},Int}) where {T} =
    HybridSparseMatrixCSC(b, flat, flat, Ref(0))  # aliasing

function HybridSparseMatrixCSC(b::BlockStructure{B}, unflat::SparseMatrixCSC{B,Int}) where {T,B<:MatrixElementNonscalarType{T}}
    m = HybridSparseMatrixCSC(b, unflat, flat(b, unflat), Ref(0))
    needs_flat_sync!(m)
    return m
end

function HybridSparseMatrixCSC(b::BlockStructure{B}, flat::SparseMatrixCSC{Complex{T},Int}) where {T,B<:MatrixElementNonscalarType{T}}
    unflat = unflat(b, flat)
    m = HybridSparseMatrixCSC(b, unflat(b, flat), flat, Ref(0))
    needs_unflat_sync!(m)
    return m
end

## API ##

blockstructure(s::HybridSparseMatrixCSC) = s.blockstruct

function unflat(s::HybridSparseMatrixCSC)
    needs_unflat_sync(s) && unflat_sync!(s)
    return s.unflat
end

function flat(s::HybridSparseMatrixCSC)
    needs_flat_sync(s) && flat_sync!(s)
    return s.flat
end

# Sync states
needs_no_sync!(s::HybridSparseMatrixCSC)     = (s.needs_sync[] = 0)
needs_no_sync(s::HybridSparseMatrixCSC)      = (s.needs_sync[] == 0)
needs_flat_sync!(s::HybridSparseMatrixCSC)   = (s.needs_sync[] = 1)
needs_flat_sync(s::HybridSparseMatrixCSC)    = (s.needs_sync[] == 1)
needs_unflat_sync!(s::HybridSparseMatrixCSC) = (s.needs_sync[] = -1)
needs_unflat_sync(s::HybridSparseMatrixCSC)  = (s.needs_sync[] == -1)

needs_no_sync(s::HybridSparseMatrixCSC{T,Complex{T}}) where {T}     = true
needs_flat_sync(s::HybridSparseMatrixCSC{T,Complex{T}}) where {T}   = false
needs_unflat_sync(s::HybridSparseMatrixCSC{T,Complex{T}}) where {T} = false

# Base.show

Base.show(io::IO, m::MIME"text/plain", s::HybridSparseMatrixCSC) = 
    show(io, m, unflat(s))

#endregion

############################################################################################
# HybridSparseMatrixCSC indexing
#region

Base.getindex(b::HybridSparseMatrixCSC{<:Any,<:SMatrixView}, i::Integer, j::Integer) =
    view(parent(unflat(b)[i, j]), flatrange(b, i), flatrange(b, j))

Base.getindex(b::HybridSparseMatrixCSC, i::Integer, j::Integer) = unflat(b)[i, j]

# only allowed for elements that are already stored
function Base.setindex!(b::HybridSparseMatrixCSC{<:Any,S}, val::AbstractVecOrMat, i::Integer, j::Integer) where {S<:SMatrixView}
    @boundscheck(checkstored(unflat(b), i, j))
    val´ = mask_block(val, S, blocksize(blockstructure(b), i, j))
    unflat(b)[i, j] = val´
    needs_flat_sync!(b)
    return val´
end

function Base.setindex!(b::HybridSparseMatrixCSC, val::AbstractVecOrMat, i::Integer, j::Integer)
    @boundscheck(checkstored(unflat(b), i, j))
    unflat(b)[i, j] = val
    needs_flat_sync!(b)
    return val
end

function mask_block(val::SMatrix{R,C}, ::Type{S}, (nrows, ncols) = size(val)) where {R,C,N,T,S<:SMatrixView{N,N,T}}
    (R, C) == (nrows, ncols) || blocksize_error((R, C), (nrows, ncols))
    return SMatrixView(SMatrix{N,N,T}(SMatrix{N,R}(I) * val * SMatrix{C,N}(I)))
end

function mask_block(val, ::Type{S}, (nrows, ncols) = size(val)) where {N,T,S<:SMatrixView{N,N,T}}
    size(val) == (nrows, ncols) || blocksize_error(size(val), (nrows, ncols))
    t = ntuple(Val(N*N)) do i
        n, m = mod1(i, N), fld1(i, N)
        @inbounds n > nrows || m > ncols ? zero(T) : T(a[n,m])
    end
    return SMatrixView(SMatrix{N,N,T}(t))
end

checkstored(mat, i, j) = i in view(rowvals(mat), nzrange(mat, j)) ||
    throw(ArgumentError("Adding new structural elements is not allowed"))

@noinline blocksize_error(s1, s2) =
    throw(ArgumentError("Expected an element of size $s2, got size $s1"))

#endregion

############################################################################################
# HybridSparseMatrixCSC flat/unflat conversion
#region

# Uniform case
function flat(b::BlockStructure{B}, unflat::SparseMatrixCSC{B´}) where {N,C,B<:SMatrix{N,N,C},B´<:SMatrix{N,N}}
    nnzguess = nnz(unflat) * N * N
    builder = CSC{C}(flatsize(b), nnzguess)
    nzs = nonzeros(unflat)
    rows = rowvals(unflat)
    cols = 1:unflatsize(b)
    for col in cols, bcol in 1:N
        for ptr in nzrange(unflat, col)
            firstrow´ = (rows[ptr] - 1) * N + 1
            vals = nzs[ptr][:,bcol]
            appendtocolumn!(builder, firstrow´, vals)
        end
        finalizecolumn!(builder, false)  # no need to sort column
    end
    n = flatsize(b)
    return sparse(builder, n, n)
end

# Non-uniform case
function flat(b::BlockStructure{B}, unflat::SparseMatrixCSC{B´}) where {N,C,B<:SMatrixView{N,N,C},B´<:SMatrixView{N,N}}
    nnzguess = nnz(unflat) * N * N
    builder = CSC{C}(flatsize(b), nnzguess)
    nzs = nonzeros(unflat)
    rows = rowvals(unflat)
    cols = 1:unflatsize(b)
    for col in cols, bcol in 1:blocksize(b, col)
        for ptr in nzrange(unflat, col)
            row = rows[ptr]
            firstrow´ = flatindex(b, row)
            vals = view(parent(nzs[ptr]), 1:blocksize(b, row), bcol)
            appendtocolumn!(builder, firstrow´, vals)
        end
        finalizecolumn!(builder, false)  # no need to sort column
    end
    n = flatsize(b)
    return sparse(builder, n, n)
end

# Uniform case
function unflat(b::BlockStructure{B}, flat::SparseMatrixCSC{<:Number}) where {N,B<:SMatrix{N,N}}
    @boundscheck(checkblocks(b, flat))
    nnzguess = nnz(flat) ÷ (N * N)
    builder = CSC{C}(unflatsize(b), nnzguess)
    rows = rowvals(flat)
    cols = 1:N:flatsize(b)
    for firstcol in cols
        ptrs = nzrange(flat, firstcol)
        ptrs´ = first(ptrs):N:last(ptrs)
        for ptr in ptrs´
            firstrow = rows[ptr]
            row´ = (firstrow - 1) ÷ N + 1
            vals = B(view(flat, firstrow:firstrow+N-1, firstcol:firstcol+N-1))
            pushtocolumn!(builder, row´, vals)
        end
        finalizecolumn!(builder, false)  # no need to sort column
    end
    n = unflatsize(b)
    return sparse(builder, n, n)
end

# Non-uniform case
function unflat(b::BlockStructure{B}, flat::SparseMatrixCSC{<:Number}) where {N,B<:SMatrixView{N,N}}
    @boundscheck(checkblocks(b, flat))
    nnzguess = nnz(flat) ÷ (N * N)
    unflatcols = unflatsize(b)
    builder = CSC{B}(unflatcols, nnzguess)
    rowsflat = rowvals(flat)
    col = 1
    for ucol in 1:unflatcols
        colrng = flatrange(b, col)
        firstcol = first(colrng)
        bcol = length(colrng)
        ptrs = rnzrange(flat, firstcol)
        ptr = first(ptrs)
        while ptr in ptrs
            firstrow = rowsflat[ptr]
            urow, brow = unflatindex(b, firstrow)
            val = mask_block(B, view(flat, firstrow:firstrow+brow-1, firstcol:firstcol+bcol-1))
            pushtocolumn!(builder, urow, val)
            ptr += N
        end
        finalizecolumn!(builder, false)  # no need to sort column
    end
    n = unflatsize(b)
    return sparse(builder, n, n)
end

checkblocks(b, flat) = nothing ## TODO: must check that all structural elements come in blocks

#endregion

############################################################################################
# HybridSparseMatrixCSC syncing
#region

# Uniform case
function flat_sync!(s::HybridSparseMatrixCSC{<:Any,S}) where {N,S<:SMatrix{N,N}}
    flat, unflat = s.flat, s.unflat
    cols = axes(unflat, 2)
    nzflat, nzunflat = nonzeros(flat), nonzeros(unflat)
    ptr´ = 1
    for col in cols, bcol in 1:N, ptr in nzrange(unflat, col)
        nz = nzunflat[ptr]
        for brow in 1:N
            nzflat[ptr´] = nz[brow, bcol]
            ptr´ += 1
        end
    end
    return s
end

# function merge_sparse(mats::Vector{<:SparseMatrixCSC{B}}) where {B}
#     mat0 = first(mats)
#     nrows, ncols = size(mat0)
#     nrows == ncols || internalerror("merge_sparse: matrix not square")
#     nnzguess = sum(mat -> nnz(mat), mats)
#     collector = CSC{B}(ncols, nnzguess)
#     for col in 1:ncols
#         for (n, mat) in enumerate(mats)
#             vals = nonzeros(mat)
#             rows = rowvals(mat)
#             for p in nzrange(mat, col)
#                 val = n == 1 ? vals[p] : zero(B)
#                 row = rows[p]
#                 pushtocolumn!(collector, row, val, false) # skips repeated rows
#             end
#         end
#         finalizecolumn!(collector)
#     end
#     matrix = sparse(collector, ncols)
#     return matrix
# end

#endregion

# ############################################################################################
# # BlockSparseMatrix
# #region

# const BlockView{T} = SubArray{T, 2, SparseMatrixCSC{T, Int}, Tuple{UnitRange{Int}, UnitRange{Int}}, false}

# struct BlockRange
#     rows::UnitRange{Int}
#     cols::UnitRange{Int}
# end

# Base.zero(::Type{BlockRange}) = BlockRange(1:0, 1:0)

# struct BlockStructure{T}
#     eltype::Type{T}
#     blocksizes::Vector{Int}       # block sizes for each sublattice
#     subsizes::Vector{Int}         # number of blocks (sites) in each sublattice
#     isscalar::Bool
# end

# struct BlockSparseMatrix{T} <: SparseArrays.AbstractSparseMatrixCSC{T,Int}
#     blockstruct::BlockStructure{T}
#     flat::SparseMatrixCSC{T,Int}
#     blocks::SparseMatrixCSC{BlockRange,Int}
#     function BlockSparseMatrix{T}(b, flat, blocks) where {T}
#         size(flat, 1) == size(flat, 2) == b.blocksizes' * b.subsizes ||
#             throw(ArgumentError("Wrong matrix size, blocksize, or subsize"))
#         checkblocks(flat, b)
#         return new(b, flat, blocks)
#     end
# end

# ## Constructors ##

# BlockStructure(t, b, s) = BlockStructure(t, b, s, all(==(1), b))

# BlockStructure(b::BlockStructure{T}, subsizes::Vector{Int}) where {T} =
#     BlockStructure(b.eltype, b.blocksizes, subsizes, b.isscalar)

# # empty constructor
# function BlockSparseMatrix(b::BlockStructure{T}) where {T}
#     n, n´ = flatsquaresize(b), squaresize(b)
#     return BlockSparseMatrix(b, spzeros(T, n, n), spzeros(BlockRange, n´, n´))
# end

# BlockSparseMatrix(b::BlockStructure{T}, flat::AbstractMatrix{T}, blocks) where {T} =
#     BlockSparseMatrix{T}(b, flat, blocks)

# ## TODO: should check block consistency in inner sparse structure
# checkblocks(flat, blockstruct) = nothing

# ## API ##

# blockstructure(b::BlockSparseMatrix) = b.blockstruct

# flat(b::BlockSparseMatrix) = b.flat

# blocks(b::BlockSparseMatrix) = b.blocks

# isscalar(b::BlockSparseMatrix) = isscalar(blockstructure(b))
# isscalar(b::BlockStructure) = b.isscalar

# Base.eltype(::BlockSparseMatrix{T}) where {T} = BlockView{T}

# blocksizes(m::BlockSparseMatrix) = blocksizes(m.blockstruct)
# blocksizes(b::BlockStructure) = b.blocksizes

# subsizes(m::BlockSparseMatrix) = m.blockstruct.subsizes

# flatsquaresize(m::BlockSparseMatrix) = flatsquaresize(blockstructure(m))
# flatsquaresize(b::BlockStructure) = b.blocksizes' * b.subsizes

# squaresize(m::BlockSparseMatrix) = squaresize(blockstructure(m))
# squaresize(b::BlockStructure) = sum(b.subsizes)

# Base.eltype(::BlockStructure{T}) where {T} = T

# Base.size(m::BlockSparseMatrix, i) = ifelse(i <= 2, squaresize(m) , 1)

# Base.size(m::BlockSparseMatrix) = let s = squaresize(m); return (s, s); end

# Base.:(==)(b1::BlockStructure, b2::BlockStructure) =
#     eltype(b1) == eltype(b2) && b1.blocksizes == b2.blocksizes && b1.subsizes == b2.subsizes

# function SparseArrays.spzeros(b::BlockStructure{T}) where {T}
#     n, m = flatsquaresize(b), squaresize(b)
#     return BlockSparseMatrix(b, spzeros(T, n, n), spzeros(BlockRange, m, m))
# end

# SparseArrays.nnz(b::BlockSparseMatrix) = nnz(blocks(b))

# function nnzdiag(m::BlockSparseMatrix)
#     b = blocks(m)
#     count = 0
#     rowptrs = rowvals(b)
#     for col in 1:size(b, 2)
#         for ptr in nzrange(b, col)
#             rowptrs[ptr] == col && (count += 1; break)
#         end
#     end
#     return count
# end

# SparseArrays.rowvals(m::BlockSparseMatrix) = rowvals(blocks(m))

# SparseArrays.nzrange(m::BlockSparseMatrix, col::Integer) = nzrange(blocks(m), col)

# ## Errors ##

# @noinline blockbounds_error() = throw(BoundsError())

# @noinline scalar_error(irng, jrng) =
#     throw(ArgumentError("Trying to assign a scalar element to a $((length(irng), length(jrng)))-sized block"))

# @noinline blocksize_error(v, val) =
#     throw(ArgumentError("Trying to assign a $(size(val))-sized element to a $(size(v))-sized block"))

# #endregion

# ############################################################################################
# # BlockSparseMatrix indexing
# #region

# Base.getindex(b::BlockSparseMatrix, i::Integer, j::Integer) =
#     view(flat(b), flatrange(b, i), flatrange(b, j))

# function Base.setindex!(b::BlockSparseMatrix, val::AbstractVecOrMat, i::Integer, j::Integer)
#     v = b[i, j]
#     size(v) == size(val) || blocksize_error(v, val)
#     return v .= val
# end

# function Base.setindex!(b::BlockSparseMatrix, val::Number, i::Integer, j::Integer)
#     irng, jrng = flatrange(b, i), flatrange(b, j)
#     length(irng) == length(jrng) == 1 || scalar_error(irng, jrng)
#     return flat(b)[first(irng), first(jrng)] = val
# end

# function flatrange(b::BlockStructure, i::Integer)
#     soffset  = 0
#     soffset´ = 0
#     @boundscheck(i < 0 && blockbounds_error())
#     @inbounds for (s, b) in zip(b.subsizes, b.blocksizes)
#         if soffset + s >= i
#             offset = muladd(i - soffset - 1, b, soffset´)
#             return offset+1:offset+b
#         end
#         soffset  += s
#         soffset´ += b * s
#     end
#     @boundscheck(blockbounds_error())
# end

# flatrange(m::BlockSparseMatrix, i::Integer) = flatrange(blockstructure(m), i)

# flatindex(m::BlockSparseMatrix, i) = first(flatrange(m, i))

# #region

# ############################################################################################
# # IJV Builder
# #region

# struct IJV{B}
#     blockstruct::BlockStructure{B}
#     i::Vector{Int}
#     j::Vector{Int}
#     v::Vector{B}
#     iblock::Vector{Int}            # Will only be populated if !isscalar(blockstruct)
#     jblock::Vector{Int}            # as otherwise they will not be used
#     vblock::Vector{BlockRange}
# end

# ## Constructors ##

# IJV(b::BlockStructure{B}) where {B} = IJV(b, Int[], Int[], B[], Int[], Int[], BlockRange[])

# ## API ##

# blockstructure(ijv::IJV) = ijv.blockstruct

# isscalar(ijv::IJV) = isscalar(blockstructure(ijv))

# function Base.push!(ijv::IJV, (iblock, jblock, vblock)::Tuple{Int,Int,Any})
#     b = blockstructure(ijv)
#     irng, jrng = flatrange(b, iblock), flatrange(b, jblock)
#     checkblocksize(vblock, irng, jrng)
#     @inbounds for (j´, j) in enumerate(jrng), (i´, i) in enumerate(irng)
#         push!(ijv.i, i)
#         push!(ijv.j, j)
#         push_val!(ijv, i´, j´, vblock)
#     end
#     if !isscalar(b)
#         push!(ijv.iblock, iblock)
#         push!(ijv.jblock, jblock)
#         push!(ijv.vblock, BlockRange(irng, jrng))
#     end
#     return ijv
# end

# checkblocksize(vb, is, js) = size(vb) == (length(is), length(js)) ||
#     throw(ArgumentError("Block size should be $((length(is), length(js)))"))
# checkblocksize(vb::UniformScaling, is, js) = nothing
# checkblocksize(vb::Number, is, js) = checkblocksize(SMatrix{1,1}(vb), is, js)

# push_val!(ijv, i´, j´, vb::AbstractMatrix) = push!(ijv.v, vb[i´, j´])
# push_val!(ijv, i´, j´, vb::Number) = push!(ijv.v, vb)
# push_val!(ijv, i´, j´, vb::UniformScaling) = push!(ijv.v, ifelse(i´ == j´, vb.λ, zero(vb.λ)))

# function Base.sizehint!(ijv::IJV, n)
#     sizehint!(ijv.iblock, n)
#     sizehint!(ijv.jblock, n)
#     sizehint!(ijv.vblock, n)
#     bmax = maximum(blocksizes(blockstructure(ijv)))^2
#     sizehint!(ijv.i, n * bmax)
#     sizehint!(ijv.j, n * bmax)
#     sizehint!(ijv.v, n * bmax)
#     return ijv
# end

# function Base.filter!(f::Function, ijv::IJV)
#     ind = 0
#     for (i, j, v, ib, jb, vb) in zip(ijv.i, ijv.j, ijv.v, ijv.iblock, ijv.jblock, ijv.vblock)
#         if f(i, j, v)
#             ind += 1
#             ijv.i[ind] = i
#             ijv.j[ind] = j
#             ijv.v[ind] = v
#             if !isscalar(b)
#                 ijv.iblock[ind] = ib
#                 ijv.jblock[ind] = jb
#                 ijv.vblock[ind] = vb
#             end
#         end
#     end
#     resize!(ijv.i, ind)
#     resize!(ijv.j, ind)
#     resize!(ijv.v, ind)
#     if !isscalar(b)
#         resize!(ijv.iblock, ind)
#         resize!(ijv.jblock, ind)
#         resize!(ijv.vblock, ind)
#     end
#     return ijv
# end

# Base.isempty(h::IJV) = length(h.i) == 0

# function SparseArrays.sparse(c::IJV)
#     b = blockstructure(c)
#     n, n´ = flatsquaresize(b), squaresize(b)
#     flat = sparse(c.i, c.j, c.v, n, n)
#     if isscalar(b)
#         # Optimization: alias flat in the scalar case, skip sparse call
#         zerorng = fill(zero(BlockRange), length(flat.nzval))
#         blocks = SparseMatrixCSC(flat.m, flat.n, flat.colptr, flat.rowval, zerorng)
#     else
#         blocks = sparse(c.iblock, c.jblock, c.vblock, n´, n´, (br, _) -> br)
#     end
#     return BlockSparseMatrix(blockstructure(c), flat, blocks)
# end

# SparseArrays.findnz(b::BlockSparseMatrix) = IJV(blockstructure(b), findnz(flat(b))...)

# Base.iszero(b::BlockSparseMatrix) = iszero(flat(b))

# #endregion



# ############################################################################################
# # Matrix transformations [involves OrbitalStructure's]
# # all merged_* functions assume matching structure of sparse matrices
# #region

# # merge several sparse matrices onto the first using only structural zeros
# function merge_sparse(mats::Vector{<:SparseMatrixCSC{B}}) where {B}
#     mat0 = first(mats)
#     nrows, ncols = size(mat0)
#     nrows == ncols || internalerror("merge_sparse: matrix not square")
#     nnzguess = sum(mat -> nnz(mat), mats)
#     collector = CSC{B}(ncols, nnzguess)
#     for col in 1:ncols
#         for (n, mat) in enumerate(mats)
#             vals = nonzeros(mat)
#             rows = rowvals(mat)
#             for p in nzrange(mat, col)
#                 val = n == 1 ? vals[p] : zero(B)
#                 row = rows[p]
#                 pushtocolumn!(collector, row, val, false) # skips repeated rows
#             end
#         end
#         finalizecolumn!(collector)
#     end
#     matrix = sparse(collector, ncols)
#     return matrix
# end

# function merged_mul!(C::SparseMatrixCSC, A::SparseMatrixCSC, b::Number, α = 1, β = 0)
#     nzA = nonzeros(A)
#     nzC = nonzeros(C)
#     if length(nzA) == length(nzC)  # assume idential structure (C has merged structure)
#         @. nzC = β * nzC + α * b * nzA
#     else
#         for col in axes(A, 2), p in nzrange(A, col)
#             row = rowvals(A)[p]
#             for p´ in nzrange(C, col)
#                 row´ = rowvals(C)[p´]
#                 if row == row´
#                     nzC[p´] = β * nzC[p´] + α * b * nzA[p]
#                     break
#                 end
#             end
#         end
#     end
#     return C
# end

# function merge_sparse(mats::Vector{BlockSparseMatrix{B}}) where {B}
#     flat´ = merge_sparse(flat.(mats))
#     blocks´ = merge_sparse(blocks.(mats))
#     bs = blockstructure(first(mats))
#     return BlockSparseMatrix{B}(bs, flat´, blocks´)
# end

# #endregion