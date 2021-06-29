#######################################################################
# Slice
#######################################################################
struct Slice{H<:Union{Hamiltonian,ParametricHamiltonian},A<:AbstractSparseMatrix,A´<:AbstractSparseMatrix,X,X´}
    h::H
    fullmat::A
    fullmatflat::A´
    axes::X
    axesflat::X´
end

(s::Slice{<:ParametricHamiltonian})(; params...) =
    Slice(s.h(; params...), s.fullmat, s.fullmatflat, s.axes, s.axesflat)

function Base.show(io::IO, m::MIME"text/plain", s::Slice)
    i = get(io, :indent, "")
    print(io, i, "Slice of ")
    show(io, m, parent(s))
    print(io, i, "\n",
"$i  Sliced size      : $(size(s))
$i  Flat size        : $(sizeflat(s))")
end

orbitalstructure(s::Slice) = orbitalstructure(parent(s))

matrixtype(s::Slice) = matrixtype(parent(s))

blocktype(s::Slice) = blocktype(parent(s))

blockeltype(s::Slice) = blockeltype(parent(s))

axesflat(s::Slice) = s.axesflat

axesflat(s::Slice, n) = axesflat(s)[n]

Base.parent(s::Slice) = s.h

Base.axes(s::Slice) = s.axes

Base.axes(s::Slice, n) = axes(s)[n]

Base.size(s::Slice, n...) = size(view(s.fullmat, axes(s)...), n...)

sizeflat(s::Slice, n...) = size(view(s.fullmatflat, axesflat(s)...), n...)

# bloch of slices

# This is faster than bloch!(similarmatrix(s), parent(s), args...), because
# copy!(sparse, view(sparse,...)) is slower than copy(view(sparse, ...))
function bloch(s::Slice, args...)
    h = parent(s)
    m = bloch(h, args...)
    T = eltype(m)
    axes = axes_for_eltype(s, T)
    m´ = copy(view(m, axes...))
    return m´
end

function bloch!(matrix::AbstractMatrix{T}, s::Slice, args...) where {T}
    bfull = bloch!(fullmat_for_eltype(s, T), s.h, args...)
    axes = axes_for_eltype(s, T)
    fast_sparse_copy!(matrix, bfull, axes)
    return matrix
end

# fastpath for dense arrays
similarmatrix(s::Slice, ::Type{S}) where {T<:SMatrix,S<:StridedMatrix{T}} = S(undef, size(s)...)
similarmatrix(s::Slice, ::Type{S}) where {T<:Number, S<:StridedMatrix{T}} = S(undef, sizeflat(s)...)
similarmatrix(s::Slice, ::Type{<:StridedMatrix}) = Matrix{blocktype(s)}(undef, size(s)...)

# in any other case (sparse), we need to add structural zeros
function similarmatrix(s::Slice, args...)
    mat = similarmatrix(s.h, args...)
    axes = axes_for_eltype(s, eltype(mat))
    v = view(mat, axes...)
    mat´ = copy(v)
    return mat´
end

axes_for_eltype(s, ::Type{<:SMatrix}) = s.axes
axes_for_eltype(s, ::Type{<:Number}) = s.axesflat

fullmat_for_eltype(s, ::Type{<:SMatrix}) = s.fullmat
fullmat_for_eltype(s, ::Type{<:Number}) = s.fullmatflat

# indexing into a slice
Base.getindex(s::Slice{<:Hamiltonian}, dn...) = parent(s)[dn...][axes(s)...]

# indexing of Hamiltonian and ParametricHamiltonian
function Base.getindex(ph::Union{Hamiltonian,ParametricHamiltonian}, is, js)
    h = parent(ph)
    fullmatflat, fullmat = similarmatrix_flat_unflat(h)
    rows = all_unflat_indices(h, is)
    cols = all_unflat_indices(h, js)
    flatrows = all_flat_indices(h, is)
    flatcols = all_flat_indices(h, js)
    axes = (rows, cols)
    axesflat = (flatrows, flatcols)
    return Slice(ph, fullmat, fullmatflat, axes, axesflat)
end

# axes and axesflat need not be vectors of indices. Can be Colon or other containers.
all_unflat_indices(h, i::Integer) = all_unflat_indices(h, i:i)
all_unflat_indices(h, is::Colon) = is
all_unflat_indices(h, is::AbstractRange) = is
all_unflat_indices(h, is::SiteSelector) = collect(siteindices(h.lattice, is))
all_unflat_indices(h, is) = siteindices(h.lattice, is)

all_flat_indices(::Hamiltonian{<:Any,<:Any, <:Number}, is) = is
all_flat_indices(h::Hamiltonian{<:Any,<:Any, <:Number}, is::SiteSelector) = collect(siteindices(h.lattice, is))
all_flat_indices(h::Hamiltonian{<:Any,<:Any, <:Number}, i::Integer) = all_flat_indices(h, i:i)
all_flat_indices(h::Hamiltonian{<:Any,<:Any, <:SMatrix}, i::Integer) = all_flat_indices(h, i:i)
all_flat_indices(::Hamiltonian{<:Any,<:Any, <:SMatrix}, is::Colon) = is

function all_flat_indices(h::Hamiltonian{<:Any,<:Any, <:SMatrix}, is)
    o = orbitalstructure(h)
    is´ = Int[]
    for i in siteindices(h.lattice, is)
        append!(is´, flatindices(i, o))
    end
    return is´
end

function all_flat_indices(h::Hamiltonian{<:Any,<:Any, <:SMatrix}, is::AbstractRange)
    o = orbitalstructure(h)
    is´ = intersect(is, 1:nsites(h))
    is == is´ || throw(ArgumentError("Indices $is are out of range $(1:nsites(h))"))
    istart = flatoffset_site(first(is´), o) + 1
    oend, N = flatoffsetorbs_site(last(is´), o)
    iend = oend+N
    return istart:iend
end

similarmatrix_flat_unflat(h::Hamiltonian{<:Any,<:Any, <:SMatrix}) =
    similarmatrix(h, flatten), similarmatrix(h)

function similarmatrix_flat_unflat(h::Hamiltonian{<:Any,<:Any, <:Number})
    mat = similarmatrix(h)
    return mat, mat
end