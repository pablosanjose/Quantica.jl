############################################################################################
# Functionality for various matrix structures in Quantica.jl - see spectialmatrixtypes.jl
############################################################################################

############################################################################################
## HybridSparseBlochMatrix
#region

############################################################################################
# HybridSparseBlochMatrix API
#region

function unflat(s::HybridSparseBlochMatrix)
    needs_unflat_sync(s) && unflat_sync!(s)
    return unflat_unsafe(s)
end

function flat(s::HybridSparseBlochMatrix)
    needs_flat_sync(s) && flat_sync!(s)
    return flat_unsafe(s)
end

# Sync states
needs_no_sync!(s::HybridSparseBlochMatrix)     = (syncstate(s)[] = 0)
needs_flat_sync!(s::HybridSparseBlochMatrix)   = (syncstate(s)[] = 1)
needs_unflat_sync!(s::HybridSparseBlochMatrix) = (syncstate(s)[] = -1)
needs_initialization!(s::HybridSparseBlochMatrix) = (syncstate(s)[] = 2)

needs_no_sync(s::HybridSparseBlochMatrix)      = (syncstate(s)[] == 0)
needs_flat_sync(s::HybridSparseBlochMatrix)    = (syncstate(s)[] == 1)
needs_unflat_sync(s::HybridSparseBlochMatrix)  = (syncstate(s)[] == -1)
needs_initialization(s::HybridSparseBlochMatrix) = (syncstate(s)[] == 2)

needs_no_sync!(s::HybridSparseBlochMatrix{<:Any,<:Complex})     = (syncstate(s)[] = 0)
needs_flat_sync!(s::HybridSparseBlochMatrix{<:Any,<:Complex})   = (syncstate(s)[] = 0)
needs_unflat_sync!(s::HybridSparseBlochMatrix{<:Any,<:Complex}) = (syncstate(s)[] = 0)

needs_no_sync(s::HybridSparseBlochMatrix{<:Any,<:Complex})      = true
needs_flat_sync(s::HybridSparseBlochMatrix{<:Any,<:Complex})    = false
needs_unflat_sync(s::HybridSparseBlochMatrix{<:Any,<:Complex})  = false

function Base.copy!(h::HybridSparseBlochMatrix{T,B}, h´::HybridSparseBlochMatrix{T,B}) where {T,B}
    copy!(blockstructure(h), blockstructure(h´))
    copy!(unflat_unsafe(h), unflat_unsafe(h´))
    isaliased(h´) || copy!(flat_unsafe(h), flat_unsafe(h´))
    syncstate(h)[] = syncstate(h´)[]
    return h
end

function Base.copy(h::HybridSparseBlochMatrix)
    b = copy(blockstructure(h))
    u = copy(unflat_unsafe(h))
    f = isaliased(h) ? u : copy(flat_unsafe(h))
    s = Ref(syncstate(h)[])
    return HybridSparseBlochMatrix(b, u, f, s)
end

function copy_matrices(h::HybridSparseBlochMatrix)
    b = blockstructure(h)
    u = copy(unflat_unsafe(h))
    f = isaliased(h) ? u : copy(flat_unsafe(h))
    s = Ref(syncstate(h)[])
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

Base.size(h::HybridSparseBlochMatrix, i::Integer...) = size(unflat_unsafe(h), i...)

flatsize(h::HybridSparseBlochMatrix) = flatsize(blockstructure(h))

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

function flat(b::OrbitalBlockStructure{B}, unflat::SparseMatrixCSC{B´}) where {N,T,B<:MatrixElementNonscalarType{T,N},B´<:MatrixElementNonscalarType{T,N}}
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

function unflat(b::OrbitalBlockStructure{B}, flat::SparseMatrixCSC{<:Number}) where {N,B<:MatrixElementNonscalarType{<:Any,N}}
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
    flat, unflat = flat_unsafe(s), unflat_unsafe(s)
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
## MatrixBlock API
#region

# Try to revert subarray to parent through a simple reordering of rows and cols
# This is possible if rows and cols is a permutation of the parent axes.
function simplify_matrixblock!(block::SubArray, rows, cols)
    viewrows, viewcols = block.indices
    if isperm(viewrows) && (viewrows === viewcols || isperm(viewcols))
        invpermute!(rows, viewrows)
        # aliasing checking
        if cols === rows
            viewcols === viewrows || invpermute!(copy(cols), viewcols)
        else
            invpermute!(cols, viewcols)
        end
        return MatrixBlock(parent(block), rows, cols)
    else # cannot simplify
        return MatrixBlock(sparse(block), rows, cols)
    end
end

function appendIJ!(I, J, b::MatrixBlock{<:Any,<:AbstractSparseMatrixCSC})
    for col in axes(blockmat(b), 2), ptr in nzrange(blockmat(b), col)
        push!(I, blockrows(b)[rowvals(blockmat(b))[ptr]])
        push!(J, blockcols(b)[col])
    end
    return I, J
end

function appendIJ!(I, J, b::MatrixBlock{<:Any,<:Diagonal})
    for col in axes(blockmat(b), 2)
        row = col
        push!(I, blockrows(b)[row])
        push!(J, blockcols(b)[col])
    end
    return I, J
end

function appendIJ!(I, J, b::MatrixBlock{<:Any,<:StridedArray})
    for c in CartesianIndices(blockmat(b))
        row, col = Tuple(c)
        push!(I, blockrows(b)[row])
        push!(J, blockcols(b)[col])
    end
    return I, J
end

function linewidth(Σ::MatrixBlock)
    Σmat = blockmat(Σ)
    Γ = Σmat - Σmat'
    Γ .*= im
    return Γ
end

Base.size(b::MatrixBlock, i...) = size(blockmat(b), i...)

Base.eltype(b::MatrixBlock) = eltype(blockmat(b))

Base.:-(b::MatrixBlock) =
    MatrixBlock(blockmat(b), blockrows(b), blockcols(b), -coefficient(b))

#endregion top

############################################################################################
## BlockSparseMatrix
#region

############################################################################################
# BlockSparseMatrix getblockptrs
#region

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

############################################################################################
# BlockSparseMatrix API
#region

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

Base.eltype(m::BlockSparseMatrix) = eltype(sparse(m))

#endregion
#endregion top
