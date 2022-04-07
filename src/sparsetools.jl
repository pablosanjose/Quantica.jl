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
    builder = CSC{B}(unflatsize(b), nnzguess)
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