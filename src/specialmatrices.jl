
############################################################################################
################################# HybridSparseBlochMatrix ##################################
#region ####################################################################################

############################################################################################
# SMatrixView
#   eltype that signals to HybridSparseBlochMatrix that a variable-size view must be returned
#   of its elements, because the number of orbitals is not uniform
#region

struct SMatrixView{N,M,T,NM}
    s::SMatrix{N,M,T,NM}
    SMatrixView{N,M,T,NM}(s) where {N,M,T,NM} = new(convert(SMatrix{N,M,T,NM}, s))
end

SMatrixView(s::SMatrix{N,M,T,NM}) where {N,M,T,NM} = SMatrixView{N,M,T,NM}(s)

SMatrixView(::Type{<:SMatrix{N,M,T,NM}}) where {N,M,T,NM} = SMatrixView{N,M,T,NM}

SMatrixView{N,M}(s) where {N,M} = SMatrixView(SMatrix{N,M}(s))

Base.parent(s::SMatrixView) = s.s

Base.view(s::SMatrixView, i...) = view(s.s, i...)

Base.zero(::Type{SMatrixView{N,M,T,NM}}) where {N,M,T,NM} = zero(SMatrix{N,M,T,NM})

# for generic code as e.g. flat/unflat or merged_flatten_mul!
Base.getindex(s::SMatrixView, i::Integer...) = s.s[i...]

#endregion

############################################################################################
# MatrixElementType & friends
#region

const MatrixElementType{T} = Union{
    Complex{T},
    SMatrix{N,N,Complex{T}} where {N},
    SMatrixView{N,N,Complex{T}} where {N}}

const MatrixElementUniformType{T} = Union{
    Complex{T},
    SMatrix{N,N,Complex{T}} where {N}}

const MatrixElementNonscalarType{T,N} = Union{
    SMatrix{N,N,Complex{T}},
    SMatrixView{N,N,Complex{T}}}

#endregion

############################################################################################
# SublatBlockStructure
#region

struct SublatBlockStructure{B}
    blocksizes::Vector{Int}       # block sizes (number of site orbitals) in each sublattice
    subsizes::Vector{Int}         # number of blocks (sites) in each sublattice
    function SublatBlockStructure{B}(blocksizes, subsizes) where {B}
        subsizes´ = Quantica.sanitize_Vector_of_Type(Int, subsizes)
        # This checks also that they are of equal length
        blocksizes´ = Quantica.sanitize_Vector_of_Type(Int, length(subsizes´), blocksizes)
        return new(blocksizes´, subsizes´)
    end
end

## Constructors ##

@inline function SublatBlockStructure(T, blocksizes, subsizes)
    B = blocktype(T, blocksizes)
    return SublatBlockStructure{B}(blocksizes, subsizes)
end

blocktype(T::Type, norbs) = SMatrixView(blocktype(T, val_maximum(norbs)))
blocktype(::Type{T}, m::Val{1}) where {T} = Complex{T}
blocktype(::Type{T}, m::Val{N}) where {T,N} = SMatrix{N,N,Complex{T},N*N}
# blocktype(::Type{T}, N::Int) where {T} = blocktype(T, Val(N))

val_maximum(n::Int) = Val(n)
val_maximum(ns) = Val(maximum(argval.(ns)))

argval(::Val{N}) where {N} = N
argval(n::Int) = n

## API ##

blocktype(::SublatBlockStructure{B}) where {B} = B

blockeltype(::SublatBlockStructure{<:MatrixElementType{T}}) where {T} = Complex{T}

blocksizes(b::SublatBlockStructure) = b.blocksizes

subsizes(b::SublatBlockStructure) = b.subsizes

flatsize(b::SublatBlockStructure) = blocksizes(b)' * subsizes(b)

unflatsize(b::SublatBlockStructure) = sum(subsizes(b))

blocksize(b::SublatBlockStructure, iunflat, junflat) = (blocksize(b, iunflat), blocksize(b, junflat))

blocksize(b::SublatBlockStructure{<:SMatrixView}, iunflat) = length(flatrange(b, iunflat))

blocksize(b::SublatBlockStructure{B}, iunflat) where {N,B<:SMatrix{N}} = N

blocksize(b::SublatBlockStructure{B}, iunflat) where {B<:Number} = 1

# Basic relation: iflat - 1 == (iunflat - soffset - 1) * b + soffset´
function flatrange(b::SublatBlockStructure{<:SMatrixView}, iunflat::Integer)
    soffset  = 0
    soffset´ = 0
    @boundscheck(iunflat < 0 && blockbounds_error())
    @inbounds for (s, b) in zip(b.subsizes, b.blocksizes)
        if soffset + s >= iunflat
            offset = muladd(iunflat - soffset - 1, b, soffset´)
            return offset+1:offset+b
        end
        soffset  += s
        soffset´ += b * s
    end
    @boundscheck(blockbounds_error())
end

flatrange(b::SublatBlockStructure{<:SMatrix{N}}, iunflat::Integer) where {N} =
    (iunflat - 1) * N + 1 : iunflat * N
flatrange(b::SublatBlockStructure{<:Number}, iunflat::Integer) = iunflat:inflat

flatindex(b::SublatBlockStructure, i) = first(flatrange(b, i))

function unflatindex(b::SublatBlockStructure{<:SMatrixView}, iflat::Integer)
    soffset  = 0
    soffset´ = 0
    @boundscheck(iflat < 0 && blockbounds_error())
    @inbounds for (s, b) in zip(b.subsizes, b.blocksizes)
        if soffset´ + b * s >= iflat
            iunflat = (iflat - soffset´ - 1) ÷ b + soffset + 1
            return iunflat, b
        end
        soffset  += s
        soffset´ += b * s
    end
    @boundscheck(blockbounds_error())
end

unflatindex(b::SublatBlockStructure{B}, iflat::Integer) where {N,B<:SMatrix{N}} =
    (iflat - 1)÷N + 1, N
unflatindex(b::SublatBlockStructure{<:Number}, iflat::Integer) = iflat, 1

Base.copy(b::SublatBlockStructure{B}) where {B} =
    SublatBlockStructure{B}(copy(blocksizes(b)), copy(subsizes(b)))

@noinline blockbounds_error() = throw(BoundsError())

#endregion

############################################################################################
# HybridSparseBlochMatrix - wraps site-block + flat versions of the same SparseMatrixCSC
#region

struct HybridSparseBlochMatrix{T,B<:MatrixElementType{T}} <: SparseArrays.AbstractSparseMatrixCSC{B,Int}
    blockstruct::SublatBlockStructure{B}
    unflat::SparseMatrixCSC{B,Int}
    flat::SparseMatrixCSC{Complex{T},Int}
    sync_state::Ref{Int}  # 0 = in sync, 1 = flat needs sync, -1 = unflat needs sync, 2 = none initialized
end

## Constructors ##

HybridSparseBlochMatrix(b::SublatBlockStructure{Complex{T}}, flat::SparseMatrixCSC{Complex{T},Int}) where {T} =
    HybridSparseBlochMatrix(b, flat, flat, Ref(0))  # aliasing

function HybridSparseBlochMatrix(b::SublatBlockStructure{B}, unflat::SparseMatrixCSC{B,Int}) where {T,B<:MatrixElementNonscalarType{T}}
    m = HybridSparseBlochMatrix(b, unflat, flat(b, unflat), Ref(0))
    needs_flat_sync!(m)
    return m
end

function HybridSparseBlochMatrix(b::SublatBlockStructure{B}, flat::SparseMatrixCSC{Complex{T},Int}) where {T,B<:MatrixElementNonscalarType{T}}
    m = HybridSparseBlochMatrix(b, unflat(b, flat), flat, Ref(0))
    needs_unflat_sync!(m)
    return m
end

## Show ##

Base.show(io::IO, m::MIME"text/plain", s::HybridSparseBlochMatrix) =
    show(io, m, unflat(s))

## API ##

blockstructure(s::HybridSparseBlochMatrix) = s.blockstruct

unflat_unsafe(s::HybridSparseBlochMatrix) = s.unflat

flat_unsafe(s::HybridSparseBlochMatrix) = s.flat

# are flat === unflat? Only for scalar eltype
isaliased(::HybridSparseBlochMatrix{<:Any,<:Complex}) = true
isaliased(::HybridSparseBlochMatrix) = false

function unflat(s::HybridSparseBlochMatrix)
    needs_unflat_sync(s) && unflat_sync!(s)
    return s.unflat
end

function flat(s::HybridSparseBlochMatrix)
    needs_flat_sync(s) && flat_sync!(s)
    return s.flat
end

# Sync states
needs_no_sync!(s::HybridSparseBlochMatrix)     = (s.sync_state[] = 0)
needs_flat_sync!(s::HybridSparseBlochMatrix)   = (s.sync_state[] = 1)
needs_unflat_sync!(s::HybridSparseBlochMatrix) = (s.sync_state[] = -1)
needs_initialization!(s::HybridSparseBlochMatrix) = (s.sync_state[] = 2)

needs_no_sync(s::HybridSparseBlochMatrix)      = (s.sync_state[] == 0)
needs_flat_sync(s::HybridSparseBlochMatrix)    = (s.sync_state[] == 1)
needs_unflat_sync(s::HybridSparseBlochMatrix)  = (s.sync_state[] == -1)
needs_initialization(s::HybridSparseBlochMatrix) = (s.sync_state[] == 2)

needs_no_sync!(s::HybridSparseBlochMatrix{<:Any,<:Complex})     = (s.sync_state[] = 0)
needs_flat_sync!(s::HybridSparseBlochMatrix{<:Any,<:Complex})   = (s.sync_state[] = 0)
needs_unflat_sync!(s::HybridSparseBlochMatrix{<:Any,<:Complex}) = (s.sync_state[] = 0)

needs_no_sync(s::HybridSparseBlochMatrix{<:Any,<:Complex})      = true
needs_flat_sync(s::HybridSparseBlochMatrix{<:Any,<:Complex})    = false
needs_unflat_sync(s::HybridSparseBlochMatrix{<:Any,<:Complex})  = false

function Base.copy!(h::HybridSparseBlochMatrix{T,B}, h´::HybridSparseBlochMatrix{T,B}) where {T,B}
    copy!(blockstructure(h), blockstructure(h´))
    copy!(h.unflat, h´.unflat)
    isaliased(h´) || copy!(h.flat, h´.flat)
    h.sync_state[] = h´.sync_state[]
    return h
end

function Base.copy(h::HybridSparseBlochMatrix)
    b = copy(blockstructure(h))
    u = copy(h.unflat)
    f = isaliased(h) ? u : copy(h.flat)
    s = Ref(h.sync_state[])
    return HybridSparseBlochMatrix(b, u, f, s)
end

function copy_matrices(h::HybridSparseBlochMatrix)
    b = blockstructure(h)
    u = copy(h.unflat)
    f = isaliased(h) ? u : copy(h.flat)
    s = Ref(h.sync_state[])
    return HybridSparseBlochMatrix(b, u, f, s)
end

SparseArrays.nnz(b::HybridSparseBlochMatrix) = nnz(unflat(b))

function nnzdiag(m::HybridSparseBlochMatrix)
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

Base.size(h::HybridSparseBlochMatrix, i::Integer...) = size(h.unflat, i...)

flatsize(h::HybridSparseBlochMatrix) = flatsize(h.blockstruct)

#endregion

############################################################################################
# HybridSparseBlochMatrix indexing
#region

Base.getindex(b::HybridSparseBlochMatrix{<:Any,<:SMatrixView}, i::Integer, j::Integer) =
    view(parent(unflat(b)[i, j]), flatrange(b, i), flatrange(b, j))

Base.getindex(b::HybridSparseBlochMatrix, i::Integer, j::Integer) = unflat(b)[i, j]

# only allowed for elements that are already stored
function Base.setindex!(b::HybridSparseBlochMatrix{<:Any,B}, val::AbstractVecOrMat, i::Integer, j::Integer) where {B<:SMatrixView}
    @boundscheck(checkstored(unflat(b), i, j))
    val´ = mask_block(B, val, blocksize(blockstructure(b), i, j))
    unflat(b)[i, j] = val´
    needs_flat_sync!(b)
    return val´
end

function Base.setindex!(b::HybridSparseBlochMatrix, val::AbstractVecOrMat, i::Integer, j::Integer)
    @boundscheck(checkstored(unflat(b), i, j))
    unflat(b)[i, j] = val
    needs_flat_sync!(b)
    return val
end

mask_block(::Type{B}, val::UniformScaling, args...) where {N,B<:MatrixElementNonscalarType{<:Any,N}} =
    mask_block(B, SMatrix{N,N}(val))

mask_block(::Type{B}, val::UniformScaling, args...) where {B<:Number} =
    mask_block(B, convert(B, val.λ))

function mask_block(B, val, size)
    @boundscheck(checkblocksize(val, size)) # tools.jl
    return mask_block(B, val)
end

@inline mask_block(::Type{B}, val) where {N,B<:SMatrix{N,N}} = B(val)

@inline mask_block(::Type{B}, val::Number) where {B<:Complex} = convert(B, val)

@inline mask_block(::Type{B}, val::SMatrix{R,C}) where {R,C,N,T,B<:SMatrixView{N,N,T}} =
    SMatrixView(SMatrix{N,R}(I) * val * SMatrix{C,N}(I))

function mask_block(::Type{B}, val) where {N,T,B<:SMatrixView{N,N,T}}
    (nrows, ncols) = size(val)
    s = ntuple(Val(N*N)) do i
        n, m = mod1(i, N), fld1(i, N)
        @inbounds n > nrows || m > ncols ? zero(T) : T(val[n,m])
    end
    return SMatrixView(SMatrix{N,N,T}(s))
end

mask_block(t, val) = throw(ArgumentError("Unexpected block size"))

checkstored(mat, i, j) = i in view(rowvals(mat), nzrange(mat, j)) ||
    throw(ArgumentError("Adding new structural elements is not allowed"))

#endregion

############################################################################################
# HybridSparseBlochMatrix flat/unflat conversion
#region

function flat(b::SublatBlockStructure{B}, unflat::SparseMatrixCSC{B´}) where {N,T,B<:MatrixElementNonscalarType{T,N},B´<:MatrixElementNonscalarType{T,N}}
    nnzguess = nnz(unflat) * N * N
    builder = CSC{Complex{T}}(flatsize(b), nnzguess)
    nzs = nonzeros(unflat)
    rows = rowvals(unflat)
    cols = 1:unflatsize(b)
    for col in cols, bcol in 1:blocksize(b, col)
        for ptr in nzrange(unflat, col)
            row = rows[ptr]
            firstrow´ = flatindex(b, row)
            vals = view(nzs[ptr], 1:blocksize(b, row), bcol)
            appendtocolumn!(builder, firstrow´, vals)
        end
        finalizecolumn!(builder, false)  # no need to sort column
    end
    n = flatsize(b)
    return sparse(builder, n, n)
end

function unflat(b::SublatBlockStructure{B}, flat::SparseMatrixCSC{<:Number}) where {N,B<:MatrixElementNonscalarType{<:Any,N}}
    @boundscheck(checkblocks(b, flat)) # tools.jl
    nnzguess = nnz(flat) ÷ (N * N)
    ncols = unflatsize(b)
    builder = CSC{B}(ncols, nnzguess)
    rowsflat = rowvals(flat)
    for ucol in 1:ncols
        colrng = flatrange(b, ucol)
        fcol = first(colrng)
        Ncol = length(colrng)
        ptrs = nzrange(flat, fcol)
        ptr = first(ptrs)
        while ptr in ptrs
            frow = rowsflat[ptr]
            urow, Nrow = unflatindex(b, frow)
            valview = view(flat, frow:frow+Nrow-1, fcol:fcol+Ncol-1)
            val = mask_block(B, valview)
            pushtocolumn!(builder, urow, val)
            ptr += Nrow
        end
        finalizecolumn!(builder, false)  # no need to sort column
    end
    n = unflatsize(b)
    return sparse(builder, n, n)
end

checkblocks(b, flat) = nothing ## TODO: must check that all structural elements come in blocks

#endregion

############################################################################################
# HybridSparseBlochMatrix syncing
#region

# Uniform case
function flat_sync!(s::HybridSparseBlochMatrix{<:Any,S}) where {N,S<:SMatrix{N,N}}
    checkinitialized(s)
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
    needs_no_sync!(s)
    return s
end

checkinitialized(s) =
    needs_initialization(s) && internalerror("sync!: Tried to sync uninitialized matrix")

## TODO
flat_sync!(s::HybridSparseBlochMatrix{<:Any,S}) where {S<:SMatrixView} =
    internalerror("flat_sync!: not yet implemented method for non-uniform orbitals")

## TODO
unflat_sync!(s) = internalerror("unflat_sync!: method not yet implemented")

#endregion

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

function merged_mul!(C::SparseMatrixCSC{<:Number}, A::HybridSparseBlochMatrix, b::Number, α = 1, β = 0)
    bs = blockstructure(A)
    if needs_flat_sync(A)
        merged_mul!(C, bs, unflat(A), b, α, β)
    else
        merged_mul!(C, bs, flat(A), b, α, β)
    end
    return C
end

function merged_mul!(C::SparseMatrixCSC{<:Number}, ::SublatBlockStructure, A::SparseMatrixCSC{B}, b::Number, α = 1, β = 0) where {B<:Complex}
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

function merged_mul!(C::SparseMatrixCSC{<:Number}, bs::SublatBlockStructure{B}, A::SparseMatrixCSC{B}, b::Number, α = 1, β = 0) where {B<:MatrixElementNonscalarType}
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
# SparseMatrix injection and pointers
#region

# Build a new sparse matrix mat´ with same structure as mat plus the diagonal
# return also: (1) ptrs to mat´ for each nonzero in mat, (2) diagonal ptrs in mat´
function store_diagonal_ptrs(mat::SparseMatrixCSC{T}) where {T}
    # like mat + I, but avoiding accidental cancellations
    mat´ = mat + Diagonal(iszero.(diag(s)))
    pmat, pdiag = Int[], Int[]
    rows, rows´ = rowvals(mat), rowvals(mat´)
    for col in axes(mat´, 2)
        ptrs = nzrange(mat, col)
        ptrs´ = nzrange(mat´, col)
        p, p´ = first(ptrs), first(ptrs´)
        while p´ in ptrs´
            row´ = rows´[p´]
            row´ == col && push!(pdiag, p´)
            if p in ptrs && row´ == rows[p]
                push!(pmat, p´)
                p += 1
            end
            p´ += 1
        end
    end
    return mat´, (pmat, pdiag)
end

#endregion

#endregion top

############################################################################################
###################################### HybridMatrix ########################################
#region ####################################################################################

############################################################################################
# HybridMatrix - see green.jl for constructors
#   Flat dense matrix endowed with subcell, site (orbital) and contact block structures
#region

struct MultiBlockStructure{L}
    cells::Vector{SVector{L,Int}}    # cells corresponding to for each subcell block
    subcelloffsets::Vector{Int}      # block offsets for each subcell
    siteoffsets::Vector{Int}         # block offsets for each site (for multiorbital sites)
    contactinds::Vector{Vector{Int}} # parent indices for each Σ contact
end

struct HybridMatrix{C,L,A<:AbstractMatrix{C}} <: AbstractMatrix{C}
    parent::A
    blockstruct::MultiBlockStructure{L}
end

#region ## API ##

blockstructure(m::HybridMatrix) = m.blockstruct

cells(m::HybridMatrix) = cells(m.blockstruct)
cells(m::MultiBlockStructure) = m.cells

siterange(m::HybridMatrix, iunflat) = siterange(m.blockstruct, iunflat)
siterange(m::MultiBlockStructure, iunflat) = m.siteoffsets[iunflat]+1:m.siteoffsets[iunflat+1]

subcellrange(m::HybridMatrix, si) = subcellrange(m.blockstruct, si)
subcellrange(m::MultiBlockStructure, si::Integer) = m.subcelloffsets[si]+1:m.subcelloffsets[si+1]
subcellrange(m::MultiBlockStructure, cell::SVector) = subcellrange(m, subcellindex(m, cell))

function subcellindex(m::MultiBlockStructure, cell::SVector)
    for (i, cell´) in enumerate(m.cells)
        cell === cell´ && return i
    end
    @boundscheck(throw(BoundsError(m, cell)))
end

flat(m::HybridMatrix) = m.parent

flatsize(m::HybridMatrix) = flatsize(m.blockstruct)
flatsize(m::MultiBlockStructure) = last(m.subcelloffsets)

unflatsize(m::HybridMatrix) = unflatsize(m.blockstruct)
unflatsize(m::MultiBlockStructure) = length(m.siteoffsets) - 1

contactinds(m::HybridMatrix) = m.contactinds
contactinds(m::HybridMatrix, i) = m.contactinds[i]

Base.view(m::HybridMatrix, i::Integer, j::Integer) =
    view(m.parent, siterange(m, i), siterange(m, j))

Base.view(m::HybridMatrix, cell::SVector{<:Any,Int}, cell´::SVector{<:Any,Int}) =
    view(m.parent, subcellrange(m, cell), subcellrange(m, cell´))

Base.view(m::HybridMatrix, cell::NTuple{<:Any,Int}, cell´::NTuple{<:Any,Int}) =
    view(m, SVector(cell), SVector(cell´))

Base.size(m::HybridMatrix) = (unflatsize(m), unflatsize(m))

function Base.size(m::HybridMatrix, i::Integer)
    s = if i<1
        @boundscheck(throw(BoundsError(m, i)))
    elseif i<=2
        unflatsize(m)
    else
        1
    end
    return s
end

Base.getindex(m::HybridMatrix, i...) = copy(view(m, i...))

Base.setindex!(m::HybridMatrix, val, i...) = (view(m, i...) .= val)

function Base.setindex!(m::HybridMatrix, val::UniformScaling, i...)
    v = view(m, i...)
    λ = val.λ
    for c in CartesianIndices(v)
        (i, j) = Tuple(c)
        @inbounds v[c] = λ * (i == j)
    end
    return v
end

#endregion
#endregion
#endregion top

############################################################################################
#################################### BlockSparseMatrix #####################################
#region ####################################################################################

############################################################################################
# BlockSparseMatrix
#   Flat sparse matrix that can be efficiently updated using block matrices `blocks`
#region

struct MatrixBlock{C<:Number, A<:AbstractMatrix{C},U}
    block::A
    rows::U             # row indices in parent matrix
    cols::U             # col indices in parent matrix
    coefficient::C      # coefficient to apply to block
end

struct BlockSparseMatrix{C,N,M<:NTuple{N,MatrixBlock}}
    mat::SparseMatrixCSC{C,Int}
    blocks::M
    ptrs::NTuple{N,Vector{Int}}    # nzvals indices for blocks
end

#region ## Constructors ##

MatrixBlock(block::AbstractMatrix{C}, rows, cols) where {C} =
    MatrixBlock(block, rows, cols, one(C))

function BlockSparseMatrix(mblocks::MatrixBlock...)
    blocks = blockmat.(mblocks)
    C = promote_type(eltype.(blocks)...)
    # I = Iterators.flatten(blockrows.(mblocks)) |> collect
    # J = Iterators.flatten(blockcols.(mblocks)) |> collect
    I, J = Int[], Int[]
    foreach(b -> appendIJ!(I, J, b), mblocks)
    mat = sparse(I, J, zero(C))
    ptrs = getblockptrs.(mblocks, Ref(mat))
    return BlockSparseMatrix(mat, mblocks, ptrs)
end

function appendIJ!(I, J, b::MatrixBlock{<:Any,<:AbstractSparseMatrixCSC})
    for col in axes(b.block, 2), ptr in nzrange(b.block, col)
        push!(I, b.rows[rowvals(b.block)[ptr]])
        push!(J, b.cols[col])
    end
    return I, J
end

function appendIJ!(I, J, b::MatrixBlock{<:Any,<:StridedMatrix})
    for c in CartesianIndices(b.block)
        row, col = Tuple(c)
        push!(I, b.rows[row])
        push!(J, b.cols[col])
    end
    return I, J
end

function appendIJ!(I, J, b::MatrixBlock{<:Any,<:Diagonal})
    for col in axes(b.block, 2)
        row = col
        push!(I, b.rows[row])
        push!(J, b.cols[col])
    end
    return I, J
end

getblockptrs(mblock, mat) = getblockptrs(mblock.block, mblock.rows, mblock.cols, mat)

function getblockptrs(block::AbstractSparseMatrixCSC, is, js, mat)
    checkblocksize(block, is, js)
    ptrs = Int[]
    for col in axes(block, 2)
        colmat = js[col]
        p = 0
        for ptr in nzrange(block, col)
            p+=1
            row = rowvals(block)[ptr]
            rowmat = is[row]
            for ptrmat in nzrange(mat, colmat)
                rowvals(mat)[ptrmat] == rowmat && (push!(ptrs, ptrmat); break)
            end
        end
    end

    nnz(block) == length(ptrs) ||
        argerror("Sparse matrix does not contain structural block")
    return ptrs
end

function getblockptrs(block::StridedMatrix, is, js, mat)
    checkblocksize(block, is, js)
    ptrs = Int[]
    for c in CartesianIndices(block)
        row, col = Tuple(c)
        rowmat, colmat = is[row], js[col]
        for ptrmat in nzrange(mat, colmat)
            rowvals(mat)[ptrmat] == rowmat && (push!(ptrs, ptrmat); break)
        end
    end
    length(block) == length(ptrs) ||
        argerror("Sparse matrix does not contain structural block")
    return ptrs
end

function getblockptrs(block::Diagonal, is, js, mat)
    checkblocksize(block, is, js)
    ptrs = Int[]
    for (col, colmat) in enumerate(js)
        col > size(block, 1) && break
        rowmat = is[col]
        for ptrmat in nzrange(mat, colmat)
            rowvals(mat)[ptrmat] == rowmat && (push!(ptrs, ptrmat); break)
        end
    end
    min(size(block)...) == length(ptrs) ||
        argerror("Sparse matrix does not contain structural block")
    return ptrs
end

checkblocksize(block, is, js) =
    (length(is), length(js)) == size(block) || argerror("Block indices size mismatch")
#endregion

#region ## API ##

SparseArrays.sparse(b::BlockSparseMatrix) = b.mat

blockmat(m::MatrixBlock) = m.block

blockrows(m::MatrixBlock) = m.rows

blockcols(m::MatrixBlock) = m.cols

coefficient(m::MatrixBlock) = m.coefficient

Base.size(m::MatrixBlock, i...) = size(m.block, i...)

blocks(m::BlockSparseMatrix) = m.blocks

function update!(m::BlockSparseMatrix)
    fill!(nonzeros(m.mat), 0)
    addblocks!(m)
    return m
end

function addblocks!(m::BlockSparseMatrix)
    for (mblock, ptrs) in zip(m.blocks, m.ptrs)
        mat = blockmat(mblock)
        coef = coefficient(mblock)
        for (x, ptr) in zip(stored(mat), ptrs)
            nonzeros(m.mat)[ptr] += coef * x
        end
    end
    return m
end

stored(block::AbstractSparseMatrixCSC) = nonzeros(block)
stored(block::StridedMatrix) = block
stored(block::Diagonal) = block.diag

Base.eltype(m::BlockSparseMatrix) = eltype(m.mat)
Base.eltype(m::MatrixBlock) = eltype(m.block)

Base.:-(b::MatrixBlock) = MatrixBlock(b.block, b.rows, b.cols, -b.coefficient)

#endregion
#endregion
#endregion top
