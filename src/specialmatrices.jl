############################################################################################
# Functionality for various matrix structures in Quantica.jl
############################################################################################

############################################################################################
## HybridSparseBlochMatrix
#region

############################################################################################
# HybridSparseBlochMatrix - flat/unflat
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

# flat/unflat conversion
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
    @boundscheck(checkblocks(b, flat))
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
# HybridSparseBlochMatrix - copying
#region

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
    @boundscheck(checkmatrixsize(val, size)) # tools.jl
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

#endregion top

############################################################################################
## BlockSparseMatrix
#region

############################################################################################
# MatrixBlock simplify
#   Revert subarray to parent through a simple reordering of rows and cols
#   This is possible if rows and cols is a permutation of the parent axes
#   but we only do a weak check of this (parent size == view size) for performance reasons
#region

simplify_matrixblock(block::SubArray, rows, cols) =
    simplify_matrixblock(parent(block), block.indices..., rows, cols)

function simplify_matrixblock(mat::AbstractMatrix, viewrows, viewcols, rows, cols)
    if size(mat) != (length(viewrows), length(viewcols))
        internalerror("simplify_matrixblock: received a SubArray that is not a permutation")
    elseif cols === rows
        rows´ = cols´ = simplify_indices(viewrows, rows)
    else
        rows´ = simplify_indices(viewrows, rows)
        cols´ = simplify_indices(viewcols, cols)
    end
    return MatrixBlock(mat, rows´, cols´)
end

simplify_indices(viewinds, inds) =
    invpermute!(convert(Vector{Int}, inds), viewinds)

#endregion

############################################################################################
# appendIJ! for MatrixBlocks
#   Useful to build a BlockSparseMatrix from a set of MatrixBlocks
#region

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

#endregion

############################################################################################
# BlockSparseMatrix getblockptrs
#region

getblockptrs(mblock, mat) = getblockptrs(mblock.block, mblock.rows, mblock.cols, mat)

function getblockptrs(block::AbstractSparseMatrixCSC, is, js, mat)
    checkblockinds(block, is, js)
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
    checkblockinds(block, is, js)
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
    checkblockinds(block, is, js)
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

#endregion

############################################################################################
# BlockSparseMatrix/BlockMatrix update!
#region

function update!(m::BlockSparseMatrix)
    nzs = nonzeros(matrix(m))
    fill!(nzs, 0)
    # Since blocks(m) is an inhomogeneous tuple, we cannot do a type-stable loop
    sparse_addblocks!(nzs, 1, pointers(m), blocks(m)...)
    return m
end

function sparse_addblocks!(nzs, i, ps, mblock, bs...)
    ptrs = ps[i]
    bmat = blockmat(mblock)
    coef = coefficient(mblock)
    for (x, ptr) in zip(stored(bmat), ptrs)
        nzs[ptr] += coef * x
    end
    return sparse_addblocks!(nzs, i+1, ps, bs...)
end

sparse_addblocks!(nzs, _, _) = nzs

stored(block::AbstractSparseMatrixCSC) = nonzeros(block)
stored(block::StridedMatrix) = block
stored(block::Diagonal) = block.diag

function update!(m::BlockMatrix)
    mat = matrix(m)
    fill!(mat, 0)
    # Since blocks(m) is an inhomogeneous tuple, we cannot do a type-stable loop
    dense_addblocks!(mat, blocks(m)...)
    return m
end

function dense_addblocks!(mat, mblock, bs...)
    bmat = blockmat(mblock)
    coef = coefficient(mblock)
    vmat = view(mat, blockrows(mblock), blockcols(mblock))
    vmat .+= coef .* bmat
    return dense_addblocks!(mat, bs...)
end

dense_addblocks!(mat) = mat

#endregion
#endregion top

############################################################################################
## InverseGreenBlockSparse
#region

# inverse_green from 0D AbstractHamiltonian + contacts
function inverse_green(h::AbstractHamiltonian{T,<:Any,0}, contacts) where {T}
    Σs = selfenergies(contacts)
    hdim = flatsize(h)
    haxis = 1:hdim
    ωblock = MatrixBlock((zero(Complex{T}) * I)(hdim), haxis, haxis)
    hblock = MatrixBlock(call!_output(h), haxis, haxis)
    extoffset = hdim
    # these are indices of contact orbitals within the merged orbital slice
    unitcinds = unit_contact_inds(contacts)
    # holds all non-extended orbital indices
    unitcindsall = unique!(sort!(reduce(vcat, unitcinds)))
    checkcontactindices(unitcindsall, hdim)
    solvers = solver.(Σs)
    blocks = selfenergyblocks(extoffset, unitcinds, 1, (ωblock, -hblock), solvers...)
    mat = BlockSparseMatrix(blocks...)
    source = zeros(Complex{T}, size(mat, 2), length(unitcindsall))
    nonextrng = 1:flatsize(h)
    return InverseGreenBlockSparse(mat, nonextrng, unitcinds, unitcindsall, source)
end

# switch from contactinds (relative to merged contact orbslice) to unitcinds (relative
# to parent unitcell)
function unit_contact_inds(contacts)
    orbindsall = orbindices(only(subcells(orbslice(contacts))))
    unitcinds = [orbindsall[cinds] for cinds in contactinds(contacts)]
    return unitcinds
end

checkcontactindices(allcontactinds, hdim) = maximum(allcontactinds) <= hdim ||
    internalerror("InverseGreenBlockSparse: unexpected contact indices beyond Hamiltonian dimension")

#endregion