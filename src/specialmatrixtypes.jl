############################################################################################
# Functionality for various matrix structures in Quantica.jl - see spectialmatrices.jl
############################################################################################

############################################################################################
## HybridSparseBlochMatrix
#    Internal Matrix type for Bloch harmonics in Hamiltonians
#region

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
#   Block structure for Hamiltonians, sorted by sublattices
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

#region ## Constructors ##

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

#endregion

#region ## API ##

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
#endregion

############################################################################################
# HybridSparseBlochMatrix
#    wraps site-block + flat versions of the same SparseMatrixCSC
#region

struct HybridSparseBlochMatrix{T,B<:MatrixElementType{T}} <: SparseArrays.AbstractSparseMatrixCSC{B,Int}
    blockstruct::SublatBlockStructure{B}
    unflat::SparseMatrixCSC{B,Int}
    flat::SparseMatrixCSC{Complex{T},Int}
    sync_state::Ref{Int}  # 0 = in sync, 1 = flat needs sync, -1 = unflat needs sync, 2 = none initialized
end

#region ## Constructors ##

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

#endregion

#region ## API ##

blockstructure(s::HybridSparseBlochMatrix) = s.blockstruct

unflat_unsafe(s::HybridSparseBlochMatrix) = s.unflat

flat_unsafe(s::HybridSparseBlochMatrix) = s.flat

syncstate(s::HybridSparseBlochMatrix) = s.sync_state

# are flat === unflat? Only for scalar eltype
isaliased(::HybridSparseBlochMatrix{<:Any,<:Complex}) = true
isaliased(::HybridSparseBlochMatrix) = false

#endregion

#endregion

#endregion top

############################################################################################
## MatrixBlock
#   A Block within a parent matrix, at a given set of rows and cols
#region

struct MatrixBlock{C<:Number, A<:AbstractMatrix{C},U}
    block::A
    rows::U             # row indices in parent matrix
    cols::U             # col indices in parent matrix
    coefficient::C      # coefficient to apply to block
end

#region ## Constructors ##

MatrixBlock(block::AbstractMatrix{C}, rows, cols) where {C} =
    MatrixBlock(block, rows, cols, one(C))

#endregion

#region ## API ##

blockmat(m::MatrixBlock) = m.block

blockrows(m::MatrixBlock) = m.rows

blockcols(m::MatrixBlock) = m.cols

coefficient(m::MatrixBlock) = m.coefficient

#endregion

#endregion

############################################################################################
## BlockSparseMatrix
#   Flat sparse matrix that can be efficiently updated using block matrices `blocks`
#region

struct BlockSparseMatrix{C,N,M<:NTuple{N,MatrixBlock}}
    mat::SparseMatrixCSC{C,Int}
    blocks::M
    ptrs::NTuple{N,Vector{Int}}    # nzvals indices for blocks
end

#region ## Constructors ##

function BlockSparseMatrix(mblocks::MatrixBlock...)
    blocks = blockmat.(mblocks)
    C = promote_type(eltype.(blocks)...)
    I, J = Int[], Int[]
    foreach(b -> appendIJ!(I, J, b), mblocks)
    mat = sparse(I, J, zero(C))
    ptrs = getblockptrs.(mblocks, Ref(mat))
    return BlockSparseMatrix(mat, mblocks, ptrs)
end

#endregion

#region ## API ##

blocks(m::BlockSparseMatrix) = m.blocks

SparseArrays.sparse(b::BlockSparseMatrix) = b.mat

#endregion
#endregion
