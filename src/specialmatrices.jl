############################################################################################
# Functionality for various matrix structures in Quantica.jl
############################################################################################

############################################################################################
## HybridSparseMatrix
#region

############################################################################################
# HybridSparseMatrix - flat/unflat
#region

function unflat(s::HybridSparseMatrix)
    check_integrity(s)
    needs_unflat_sync(s) && unflat_sync!(s)
    return unflat_unsafe(s)
end

function flat(s::HybridSparseMatrix)
    check_integrity(s)
    needs_flat_sync(s) && flat_sync!(s)
    return flat_unsafe(s)
end

# Sync states
needs_no_sync!(s::HybridSparseMatrix)     = (syncstate(s)[] = 0)
needs_flat_sync!(s::HybridSparseMatrix)   = (syncstate(s)[] = 1)
needs_unflat_sync!(s::HybridSparseMatrix) = (syncstate(s)[] = -1)
needs_initialization!(s::HybridSparseMatrix) = (syncstate(s)[] = 2)

needs_no_sync(s::HybridSparseMatrix)      = (syncstate(s)[] == 0)
needs_flat_sync(s::HybridSparseMatrix)    = (syncstate(s)[] == 1)
needs_unflat_sync(s::HybridSparseMatrix)  = (syncstate(s)[] == -1)
needs_initialization(s::HybridSparseMatrix) = (syncstate(s)[] == 2)

needs_no_sync!(s::HybridSparseMatrix{<:Any,<:Complex})     = (syncstate(s)[] = 0)
needs_flat_sync!(s::HybridSparseMatrix{<:Any,<:Complex})   = (syncstate(s)[] = 0)
needs_unflat_sync!(s::HybridSparseMatrix{<:Any,<:Complex}) = (syncstate(s)[] = 0)

needs_no_sync(s::HybridSparseMatrix{<:Any,<:Complex})      = true
needs_flat_sync(s::HybridSparseMatrix{<:Any,<:Complex})    = false
needs_unflat_sync(s::HybridSparseMatrix{<:Any,<:Complex})  = false

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
            urow, Nrow = unflatindex_and_blocksize(b, frow)
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
# HybridSparseMatrix - copying
#region

function Base.copy!(h::HybridSparseMatrix{T,B}, h´::HybridSparseMatrix{T,B}) where {T,B}
    copy!(blockstructure(h), blockstructure(h´))
    copy!(unflat_unsafe(h), unflat_unsafe(h´))
    isaliased(h´) || copy!(flat_unsafe(h), flat_unsafe(h´))
    syncstate(h)[] = syncstate(h´)[]
    return h
end

function Base.copy(h::HybridSparseMatrix)
    b = copy(blockstructure(h))
    u = copy(unflat_unsafe(h))
    f = isaliased(h) ? u : copy(flat_unsafe(h))
    s = Ref(syncstate(h)[])
    return HybridSparseMatrix(b, u, f, s)
end

function copy_matrices(h::HybridSparseMatrix)
    b = blockstructure(h)
    u = copy(unflat_unsafe(h))
    f = isaliased(h) ? u : copy(flat_unsafe(h))
    s = Ref(syncstate(h)[])
    return HybridSparseMatrix(b, u, f, s)
end

#endregion

############################################################################################
# HybridSparseMatrix indexing
#region

Base.getindex(b::HybridSparseMatrix{<:Any,<:SMatrixView}, i::Integer, j::Integer) =
    view(parent(unflat(b)[i, j]), flatrange(b, i), flatrange(b, j))

Base.getindex(b::HybridSparseMatrix, i::Integer, j::Integer) = unflat(b)[i, j]

# only allowed for elements that are already stored
function Base.setindex!(b::HybridSparseMatrix{<:Any,B}, val::AbstractVecOrMat, i::Integer, j::Integer) where {B<:SMatrixView}
    @boundscheck(checkstored(unflat(b), i, j))
    val´ = mask_block(B, val, blocksize(blockstructure(b), i, j))
    unflat(b)[i, j] = val´
    needs_flat_sync!(b)
    return val´
end

function Base.setindex!(b::HybridSparseMatrix, val::AbstractVecOrMat, i::Integer, j::Integer)
    @boundscheck(checkstored(unflat(b), i, j))
    unflat(b)[i, j] = val
    needs_flat_sync!(b)
    return val
end

checkstored(mat, i, j) = i in view(rowvals(mat), nzrange(mat, j)) ||
    throw(ArgumentError("Adding new structural elements is not allowed"))

#endregion

############################################################################################
# mask_block
#   converts input to a specific block type B (with or without size check)
#region

# in case the first argument is a value, not a type
@inline mask_block(::B, val) where {B} = mask_block(B, val)

@inline mask_block(::Type{B}, val::UniformScaling, size = (N, N)) where {T,N,B<:MatrixElementNonscalarType{T,N}} =
    mask_block(B, sanitize_SMatrix(SMatrix{N,N,T}, SMatrix{N,N}(val), size))

@inline mask_block(::Type{B}, val::UniformScaling, size...) where {B<:Number} =
    convert(B, val.λ)

@inline mask_block(::Type{B}, val, size...) where {B<:Number} = convert(B, only(val)) # conversion not needed?

@inline function mask_block(B, val, size)
    @boundscheck(checkmatrixsize(val, size)) # tools.jl
    return mask_block(B, val)
end

@inline mask_block(::Type{B}, val) where {N,B<:SMatrix{N,N}} = B(val)

@inline mask_block(::Type{B}, val::SMatrix{R,C}) where {R,C,N,T,B<:SMatrixView{N,N,T}} =
    SMatrixView(SMatrix{N,R}(I) * val * SMatrix{C,N}(I))

@inline function mask_block(::Type{B}, val) where {N,T,B<:SMatrixView{N,N,T}}
    (nrows, ncols) = size_or_1x1(val)
    s = ntuple(Val(N*N)) do i
        n, m = mod1(i, N), fld1(i, N)
        @inbounds n > nrows || m > ncols ? zero(T) : T(val[n,m])
    end
    return SMatrixView(SMatrix{N,N,T}(s))
end

mask_block(t::Type, val) = argerror("Unexpected block size")

size_or_1x1(::Number) = (1, 1)
size_or_1x1(val) = size(val)

#endregion

############################################################################################
# HybridSparseMatrix syncing
#region

checkinitialized(s) =
    needs_initialization(s) && internalerror("sync!: Tried to sync uninitialized matrix")

# Uniform case
function flat_sync!(s::HybridSparseMatrix{<:Any,S}) where {N,S<:SMatrix{N,N}}
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

# Non-uniform case
function flat_sync!(s::HybridSparseMatrix{<:Any,S}) where {N,S<:SMatrixView{N,N}}
    checkinitialized(s)
    flat, unflat = flat_unsafe(s), unflat_unsafe(s)
    cols, rows, rowsflat = axes(unflat, 2), rowvals(unflat), rowvals(flat)
    nzflat, nzunflat = nonzeros(flat), nonzeros(unflat)
    bs = blockstructure(s)
    for col in cols
        colrng = flatrange(bs, col)
        for (j, colflat) in enumerate(colrng)
            ptrsflat = nzrange(flat, colflat)
            ptrflat = first(ptrsflat)
            for ptr in nzrange(unflat, col)
                row = rows[ptr]
                val = nzunflat[ptr]
                rowrng = flatrange(bs, row)
                for (i, rowflat) in enumerate(rowrng)
                    rowsflat[ptrflat] == rowflat && ptrflat in ptrsflat ||
                        internalerror("flat_sync!: unexpected structural mismatch")
                    nzflat[ptrflat] = val[i,j]
                    ptrflat += 1
                end
            end
        end
    end
    needs_no_sync!(s)
    return s
end

# Uniform and non-uniform cases
function unflat_sync!(s::HybridSparseMatrix{<:Any,S}) where {S<:Union{SMatrix,SMatrixView}}
    checkinitialized(s)
    flat, unflat = flat_unsafe(s), unflat_unsafe(s)
    cols, rows = axes(unflat, 2), rowvals(unflat)
    nzunflat = nonzeros(unflat)
    bs = blockstructure(s)
    for col in cols
        colrng = flatrange(bs, col)
        for ptr in nzrange(unflat, col)
            row = rows[ptr]
            rowrng = flatrange(bs, row)
            val = view(flat, colrng, rowrng)
            nzunflat[ptr] = S(val)
        end
    end
    needs_no_sync!(s)
    return s
end

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
    hdim = flatsize(h)
    haxis = 1:hdim
    ωblock = MatrixBlock((zero(Complex{T}) * I)(hdim), haxis, haxis)
    hblock = MatrixBlock(call!_output(h), haxis, haxis)
    mat, unitcinds, unitcindsall = inverse_green_mat((ωblock, -hblock), hdim, contacts)
    source = zeros(Complex{T}, size(mat, 2), length(unitcindsall))
    nonextrng = 1:flatsize(h)
    return InverseGreenBlockSparse(mat, nonextrng, unitcinds, unitcindsall, source)
end

# case without contacts
function inverse_green_mat(blocks, _, ::Contacts{<:Any,0})
    mat = BlockSparseMatrix(blocks...)
    unitcinds = Vector{Int}[]
    unitcindsall = Int[]
    return mat, unitcinds, unitcindsall
end

function inverse_green_mat(blocks, hdim, contacts)
    Σs = selfenergies(contacts)
    extoffset = hdim
    unitcinds = [orbindices(only(cellsdict(contacts, i))) for i in 1:ncontacts(contacts)]
    unitcindsall = orbindices(only(cellsdict(contacts)))
    checkcontactindices(unitcindsall, hdim)
    solvers = solver.(Σs)
    Σblocks = selfenergyblocks(extoffset, unitcinds, 1, (), solvers...)
    g⁻¹blocks = maybe_switch_sign.(Σblocks)
    blocks´ = (blocks..., g⁻¹blocks...)
    # we need to flatten extended blocks, that come as NTuple{3}'s
    mat = BlockSparseMatrix(tupleflatten(blocks´...)...)
    return mat, unitcinds, unitcindsall
end

# matrix blocks of g⁻¹ have negative signs for Σreg, positive for Σext = (V', g⁻¹´, V)
maybe_switch_sign(Σ::MatrixBlock) = -Σ
maybe_switch_sign(Vg⁻¹V::NTuple{3,MatrixBlock}) = Vg⁻¹V

checkcontactindices(allcontactinds, hdim) = maximum(allcontactinds) <= hdim ||
    internalerror("InverseGreenBlockSparse: unexpected contact indices beyond Hamiltonian dimension")

#endregion

############################################################################################
## OrbitalSliceArray
#region

# AbstractArray interface
Base.size(a::OrbitalSliceArray) = size(parent(a))
Base.iterate(a::OrbitalSliceArray, i...) = iterate(parent(a), i...)
Base.length(a::OrbitalSliceArray) = length(parent(a))
Base.IndexStyle(::Type{T}) where {M,T<:OrbitalSliceArray{<:Any,<:Any,M}} = IndexStyle(M)
Base.similar(a::OrbitalSliceArray) = OrbitalSliceArray(similar(parent(a)), orbaxes(a))
Base.similar(a::OrbitalSliceArray, t::Type) = OrbitalSliceArray(similar(parent(a), t), orbaxes(a))
# doesn't make sense to keep orbaxes in similar with different dimensions.
Base.similar(a::OrbitalSliceArray, dims::Tuple) = similar(parent(a), dims)
Base.copy(a::OrbitalSliceArray) = OrbitalSliceArray(copy(parent(a)), orbaxes(a))
Base.@propagate_inbounds Base.getindex(a::OrbitalSliceArray, i::Int) =
    getindex(parent(a), i)
Base.@propagate_inbounds Base.getindex(a::OrbitalSliceArray, I::Vararg{Int, N}) where {N} =
    getindex(parent(a), I...)
Base.@propagate_inbounds Base.setindex!(a::OrbitalSliceArray, v, i::Int) = setindex!(parent(a), v, i)
Base.@propagate_inbounds Base.setindex!(a::OrbitalSliceArray, v, I::Vararg{Int, N}) where {N} = setindex!(parent(a), v, I...)

# Additional indexing over sites
Base.getindex(a::OrbitalSliceMatrix; sites...) = getindex(a, siteselector(; sites...))
Base.getindex(a::OrbitalSliceMatrix, i::NamedTuple, j::NamedTuple = i) =
    getindex(a, siteselector(i), siteselector(j))
Base.getindex(a::OrbitalSliceMatrix, i::NamedTuple, j::SiteSelector) =
    getindex(a, siteselector(i), j)
Base.getindex(a::OrbitalSliceMatrix, i::SiteSelector, j::NamedTuple) =
    getindex(a, i, siteselector(j))

# SiteSelector: return a new OrbitalSliceMatrix
function Base.getindex(a::OrbitalSliceMatrix, i::SiteSelector, j::SiteSelector = i)
    rowslice, colslice = orbaxes(a)
    rowslice´, colslice´ = rowslice[i], colslice[j]
    rows = collect(indexcollection(rowslice, rowslice´))
    cols = i === j && rowslice === colslice ? rows : indexcollection(colslice, colslice´)
    m = parent(a)[rows, cols]
    return OrbitalSliceMatrix(m, (rowslice´, colslice´))
end

# CellSites: return an unwrapped Matrix or a view thereof (non-allocating)
Base.getindex(a::OrbitalSliceMatrix, i::AnyCellSites, j::AnyCellSites = i) =
    copy(view(a, i, j))

Base.getindex(a::OrbitalSliceMatrix, i::C, j::C = i) where {B,C<:CellSitePos{<:Any,<:Any,<:Any,B}} =
    sanitize_block(B, view(a, i, j))

function Base.view(a::OrbitalSliceMatrix, i::AnyCellSites, j::AnyCellSites = i)
    rowslice, colslice = orbaxes(a)
    i´, j´ = apply(i, lattice(rowslice)), apply(j, lattice(colslice))
    rows = indexcollection(rowslice, i´)
    cols = j === i && rowslice === colslice ? rows : indexcollection(colslice, j´)
    return view(parent(a), rows, cols)
end

## broadcasting

# following the manual: https://docs.julialang.org/en/v1/manual/interfaces/#man-interface-array

Broadcast.BroadcastStyle(::Type{<:OrbitalSliceArray}) = Broadcast.ArrayStyle{OrbitalSliceArray}()

Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{OrbitalSliceArray}}, ::Type{ElType}) where {ElType} =
    OrbitalSliceArray(similar(Array{ElType}, axes(bc)), orbaxes(find_osa(bc)))

find_osa(bc::Base.Broadcast.Broadcasted) = find_osa(bc.args)
find_osa(args::Tuple) = find_osa(find_osa(args[1]), Base.tail(args))
find_osa(x) = x
find_osa(::Tuple{}) = nothing
find_osa(a::OrbitalSliceArray, rest) = a
find_osa(::Any, rest) = find_osa(rest)

# taken from https://github.com/JuliaArrays/OffsetArrays.jl/blob/756e839563c88faa4ebe4ff971286747863aaff0/src/OffsetArrays.jl#L469

Base.dataids(A::OrbitalSliceArray) = Base.dataids(parent(A))
Broadcast.broadcast_unalias(dest::OrbitalSliceArray, src::OrbitalSliceArray) =
    parent(dest) === parent(src) ? src : Broadcast.unalias(dest, src)

## conversion

maybe_OrbitalSliceArray(i) = x -> maybe_OrbitalSliceArray(x, i)

maybe_OrbitalSliceArray(x::AbstractVector, i) = maybe_OrbitalSliceVector(x, i)
maybe_OrbitalSliceArray(x::AbstractVector, (i,_)::Tuple) = maybe_OrbitalSliceVector(x, i)
maybe_OrbitalSliceArray(x::AbstractMatrix, i) = maybe_OrbitalSliceMatrix(x, i)

maybe_OrbitalSliceVector(x, i::DiagIndices{Missing,<:OrbitalSliceGrouped}) =
    maybe_OrbitalSliceVector(x, parent(i))
maybe_OrbitalSliceVector(x, i::DiagIndices{<:Any,<:OrbitalSliceGrouped}) =
    maybe_OrbitalSliceVector(x, scalarize(parent(i)))
maybe_OrbitalSliceVector(x, i::OrbitalSliceGrouped) = OrbitalSliceVector(x, (i,))

# fallback
maybe_OrbitalSliceVector(x, i) = x

maybe_OrbitalSliceMatrix(x, i::OrbitalSliceGrouped) =
    maybe_OrbitalSliceMatrix(x, (i, i))
maybe_OrbitalSliceMatrix(x, (i, j)::Tuple{OrbitalSliceGrouped,OrbitalSliceGrouped}) =
    OrbitalSliceMatrix(x, (i, j))

# fallback
maybe_OrbitalSliceMatrix(x, i) = x

#endregion
