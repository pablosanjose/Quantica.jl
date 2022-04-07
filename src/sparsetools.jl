############################################################################################
# MatrixElementType
#region

struct SMatrixView{N,M,T,NM}
    s::SMatrix{N,M,T,NM}
    SMatrixView{N,M,T,NM}(s) where {N,M,T,NM} = new(convert(SMatrix{N,M,T,NM}, s))
end

SMatrixView(s::SMatrix{N,M,T,NM}) where {N,M,T,NM} = SMatrixView{N,M,T,NM}(s)

SMatrixView(::Type{<:SMatrix{N,M,T,NM}}) where {N,M,T,NM} = SMatrixView{N,M,T,NM}

SMatrixView{N,M}(s) where {N,M} = SMatrixView(SMatrix{N,M}(s))

Base.parent(s::SMatrixView) = s.s

Base.view(s::SMatrixView, i::Integer...) = view(s.s, i...)

Base.zero(::Type{SMatrixView{N,M,T,NM}}) where {N,M,T,NM} = zero(SMatrix{N,M,T,NM})

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

blocktype(T::Type, norbs) = SMatrixView(blocktype(T, val_maximum(norbs)))
blocktype(::Type{T}, m::Val{1}) where {T} = Complex{T}
blocktype(::Type{T}, m::Val{N}) where {T,N} = SMatrix{N,N,Complex{T},N*N}

val_maximum(n::Int) = Val(n)
val_maximum(ns) = Val(maximum(argval.(ns)))

argval(::Val{N}) where {N} = N
argval(n::Int) = n

## API ##

blocktype(::BlockStructure{B}) where {B} = B

blocksizes(b::BlockStructure) = b.blocksizes

subsizes(b::BlockStructure) = b.subsizes

flatsize(b::BlockStructure) = blocksizes(b)' * subsizes(b)

unflatsize(b::BlockStructure) = sum(subsizes(b))

blocksize(b::BlockStructure, iflat, jflat) = (blocksize(b, iflat), blocksize(b, jflat))

blocksize(b::BlockStructure, iflat) = length(flatrange(b, iflat))

function blocksize_unflat(b::BlockStructure, iunflat)
    soffset  = 0
    @boundscheck(iunflat < 0 && blockbounds_error())
    @inbounds for (s, b) in zip(b.subsizes, b.blocksizes)
        soffset + s >= iunflat && return b
        soffset += s
    end
    @boundscheck(blockbounds_error())
end

# Basic relation: iflat - 1 == (iunflat - soffset - 1) * b + soffset´
function flatrange(b::BlockStructure, iflat::Integer)
    soffset  = 0
    soffset´ = 0
    @boundscheck(iflat < 0 && blockbounds_error())
    @inbounds for (s, b) in zip(b.subsizes, b.blocksizes)
        if soffset + s >= iflat
            offset = muladd(iflat - soffset - 1, b, soffset´)
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

flatindex(b::BlockStructure, i) = first(flatrange(b, i))

@noinline blockbounds_error() = throw(BoundsError())

#endregion

############################################################################################
# HybridSparseMatrixCSC - wraps blocked + flat versions of the same SparseMatrixCSC
#region

struct HybridSparseMatrixCSC{T,B<:MatrixElementType{T}} <: SparseArrays.AbstractSparseMatrixCSC{B,Int}
    blockstruct::BlockStructure{B}
    unflat::SparseMatrixCSC{B,Int}
    flat::SparseMatrixCSC{Complex{T},Int}
    sync_state::Ref{Int}  # 0 = in sync, 1 = flat needs sync, -1 = unflat needs sync, 2 = none initialized
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

## Show ##

Base.show(io::IO, m::MIME"text/plain", s::HybridSparseMatrixCSC) = 
    show(io, m, unflat(s))

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
needs_no_sync!(s::HybridSparseMatrixCSC)     = (s.sync_state[] = 0)
needs_no_sync(s::HybridSparseMatrixCSC)      = (s.sync_state[] == 0)
needs_flat_sync!(s::HybridSparseMatrixCSC)   = (s.sync_state[] = 1)
needs_flat_sync(s::HybridSparseMatrixCSC)    = (s.sync_state[] == 1)
needs_unflat_sync!(s::HybridSparseMatrixCSC) = (s.sync_state[] = -1)
needs_unflat_sync(s::HybridSparseMatrixCSC)  = (s.sync_state[] == -1)
needs_initialization!(s::HybridSparseMatrixCSC) = (s.sync_state[] = 2)
needs_initialization(s::HybridSparseMatrixCSC) = (s.sync_state[] == 2)

needs_no_sync(s::HybridSparseMatrixCSC{T,Complex{T}}) where {T}     = true
needs_flat_sync(s::HybridSparseMatrixCSC{T,Complex{T}}) where {T}   = false
needs_unflat_sync(s::HybridSparseMatrixCSC{T,Complex{T}}) where {T} = false

function Base.copy!(h::HybridSparseMatrixCSC{T,B}, h´::HybridSparseMatrixCSC{T,B}) where {T,B}
    copy!(blockstructure(h), blockstructure(h´))
    copy!(h.unflat, h´.unflat)
    copy!(h.flat, h´.flat)
    h.sync_state[] = h´.sync_state[]
    return h
end

SparseArrays.nnz(b::HybridSparseMatrixCSC) = nnz(unflat(b))

function nnzdiag(m::HybridSparseMatrixCSC)
    b = unflat(m)
    count = 0
    rowptrs = rowvals(b)
    for col in 1:size(b, 2)
        for ptr in nzrange(b, col)
            rowptrs[ptr] == col && (count += 1; break)
        end
    end
    return count
end

Base.size(h::HybridSparseMatrixCSC, i::Integer...) = size(h.unflat, i...)

#endregion

############################################################################################
# HybridSparseMatrixCSC indexing
#region

Base.getindex(b::HybridSparseMatrixCSC{<:Any,<:SMatrixView}, i::Integer, j::Integer) =
    view(parent(unflat(b)[i, j]), flatrange(b, i), flatrange(b, j))

Base.getindex(b::HybridSparseMatrixCSC, i::Integer, j::Integer) = unflat(b)[i, j]

# only allowed for elements that are already stored
function Base.setindex!(b::HybridSparseMatrixCSC{<:Any,B}, val::AbstractVecOrMat, i::Integer, j::Integer) where {B<:SMatrixView}
    @boundscheck(checkstored(unflat(b), i, j))
    val´ = mask_block(B, val, blocksize(blockstructure(b), i, j))
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

function mask_block(::Type{B}, val, (nrows, ncols) = size(val)) where {N,T,B<:SMatrix{N,N,T}}
    # This check includes the case where val is a scalar
    val isa UniformScaling || (size(val, 1), size(val, 2)) == (nrows, ncols) == (N, N) ||
        blocksize_error(size(val), (N, N))
    return B(val)
end

mask_block(::Type{B}, val::Number, (nrows, ncols) = (1, 1)) where {B<:Complex} =
    convert(B, val)

function mask_block(::Type{B}, val::SMatrix{R,C}, (nrows, ncols) = size(val)) where {R,C,N,T,B<:SMatrixView{N,N,T}}
    (R, C) == (nrows, ncols) || blocksize_error((R, C), (nrows, ncols))
    return SMatrixView(SMatrix{N,R}(I) * val * SMatrix{C,N}(I))
end

function mask_block(::Type{B}, val, (nrows, ncols) = size(val)) where {N,T,B<:SMatrixView{N,N,T}}
    val isa UniformScaling || (size(val, 1), size(val, 2)) == (nrows, ncols) ||
        blocksize_error(size(val), (nrows, ncols))
    t = ntuple(Val(N*N)) do i
        n, m = mod1(i, N), fld1(i, N)
        @inbounds n > nrows || m > ncols ? zero(T) : T(val[n,m])
    end
    return SMatrixView(SMatrix{N,N,T}(t))
end

mask_block(t, val, s = size(val)) = blocksize_error(size(val), s)

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

############################################################################################
# SparseMatrix transformations
# all merged_* functions assume matching structure of sparse matrices
#region

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