#######################################################################
# BoxIterator
#region

"""
    BoxIterator(seed::SVector{N,Int}; maxiterations = TOOMANYITERS)

Cartesian iterator `iter` over `SVector{N,Int}`s (`cell`s) that starts at `seed` and
grows outwards in the form of a box of increasing sides (not necesarily equal) until
it encompasses a certain N-dimensional region. To signal that a cell is in the desired
region the user calls `acceptcell!(iter, cell)`.
"""
struct BoxIterator{N}
    seed::SVector{N,Int}
    maxiter::Int
    dimdir::MVector{2,Int}
    nmoves::MVector{N,Bool}
    pmoves::MVector{N,Bool}
    npos::MVector{N,Int}
    ppos::MVector{N,Int}
end

const TOOMANYITERS = 10^8

Base.IteratorSize(::Type{BoxIterator}) = Base.SizeUnknown()

Base.IteratorEltype(::Type{BoxIterator}) = Base.HasEltype()

Base.eltype(::Type{BoxIterator{N}}) where {N} = SVector{N,Int}

Base.CartesianIndices(b::BoxIterator) =
    CartesianIndices(UnitRange.(Tuple(b.npos), Tuple(b.ppos)))

function BoxIterator(seed::SVector{N}; maxiterations = TOOMANYITERS) where {N}
    BoxIterator(seed, maxiterations, MVector(1, 2),
        ones(MVector{N,Bool}), ones(MVector{N,Bool}), MVector{N,Int}(seed), MVector{N,Int}(seed))
end

function iteratorreset!(b::BoxIterator{N}) where {N}
    b.dimdir[1] = 1
    b.dimdir[2] = 2
    b.nmoves .= ones(MVector{N,Bool})
    b.pmoves .= ones(MVector{N,Bool})
    b.npos   .= MVector(b.seed)
    b.ppos   .= MVector(b.seed)
    return nothing
end

struct BoxIteratorState{N}
    range::CartesianIndices{N, NTuple{N,UnitRange{Int}}}
    rangestate::CartesianIndex{N}
    iteration::Int
end

Base.iterate(b::BoxIterator{0}) = (SVector{0,Int}(), nothing)
Base.iterate(b::BoxIterator{0}, state) = nothing

function Base.iterate(b::BoxIterator{N}) where {N}
    range = CartesianIndices(ntuple(i -> b.seed[i]:b.seed[i], Val(N)))
    itrange = iterate(range)
    if itrange === nothing
        return nothing
    else
        (cell, rangestate) = itrange
        return (SVector(Tuple(cell)), BoxIteratorState(range, rangestate, 1))
    end
end

function Base.iterate(b::BoxIterator, s::BoxIteratorState)
    itrange = iterate(s.range, s.rangestate)
    facedone = itrange === nothing
    if facedone
        alldone = !any(b.pmoves) && !any(b.nmoves) || checkmaxiter(b, s)
        if alldone  # Last shells in all directions were empty, trim from boundingboxcorners
            b.npos .+= 1
            b.ppos .-= 1
            return nothing
        else
            newrange = nextface!(b)
            # newrange === nothing && return nothing
            itrange = iterate(newrange)
            # itrange === nothing && return nothing
            (cell, rangestate) = itrange
            return (SVector(Tuple(cell)), BoxIteratorState(newrange, rangestate, s.iteration + 1))
        end
    else
        (cell, rangestate) = itrange
        return (SVector(Tuple(cell)), BoxIteratorState(s.range, rangestate, s.iteration + 1))
    end
end

@noinline function checkmaxiter(b::BoxIterator, s::BoxIteratorState)
    exceeded = isless(b.maxiter, s.iteration)
    exceeded && @warn("Region seems unbounded after $(b.maxiter) iterations")
    return exceeded
end


function nextface!(b::BoxIterator{N}) where {N}
    @inbounds for i in 1:2N
        nextdimdir!(b)
        newdim, newdir = Tuple(b.dimdir)
        if newdir == 1
            if b.nmoves[newdim]
                b.npos[newdim] -= 1
                b.nmoves[newdim] = false
                return newrangeneg(b, newdim)
            end
        else
            if b.pmoves[newdim]
                b.ppos[newdim] += 1
                b.pmoves[newdim] = false
                return newrangepos(b, newdim)
            end
        end
    end
    return nothing
end

function nextdimdir!(b::BoxIterator{N}) where {N}
    dim, dir = Tuple(b.dimdir)
    if dim < N
        dim += 1
    else
        dim = 1
        dir = ifelse(dir == 1, 2, 1)
    end
    b.dimdir[1] = dim
    b.dimdir[2] = dir
    return nothing
end

@inline function newrangeneg(b::BoxIterator{N}, dim) where {N}
    return CartesianIndices(ntuple(
        i -> b.npos[i]:(i == dim ? b.npos[i] : b.ppos[i]),
        Val(N)))
end

@inline function newrangepos(b::BoxIterator{N}, dim) where {N}
    return CartesianIndices(ntuple(
        i -> (i == dim ? b.ppos[i] : b.npos[i]):b.ppos[i],
        Val(N)))
end

function acceptcell!(b::BoxIterator{N}, cell) where {N}
    dim, dir = Tuple(b.dimdir)
    if dir == 1
        @inbounds for i in 1:N
            (cell[i] == b.ppos[i]) && (b.pmoves[i] = true)
            (i == dim || cell[i] == b.npos[i]) && (b.nmoves[i] = true)
        end
    else
        @inbounds for i in 1:N
            (i == dim || cell[i] == b.ppos[i]) && (b.pmoves[i] = true)
            (cell[i] == b.npos[i]) && (b.nmoves[i] = true)
        end
    end
    return nothing
end

# Fallback for non-BoxIterators
acceptcell!(b, cell) = nothing

#endregion

#######################################################################
# CoSort
#region

struct CoSortTup{T,T´}
    x::T
    y::T´
end

mutable struct CoSort{T,T´} <: AbstractVector{CoSortTup{T,T´}}
    sortvector::Vector{T}
    covector::Vector{T´}
    offset::Int
    function CoSort{T,T´}(sortvector, covector, offset) where {T,T´}
        length(covector) >= length(sortvector) ? new(sortvector, covector, offset) :
            throw(DimensionMismatch("Coarray length should exceed sorting array"))
    end
end

CoSort(sortvector::Vector{T}, covector::Vector{T´}) where {T,T´} =
    CoSort{T,T´}(sortvector, covector, 0)

function cosort!(s, c)
    cs = CoSort(s, c)
    sort!(cs)
    return cs.sortvector, cs.covector
end

Base.size(c::CoSort) = (size(c.sortvector, 1) - c.offset,)

Base.getindex(c::CoSort, i) =
    CoSortTup(getindex(c.sortvector, i + c.offset), getindex(c.covector, i + c.offset))

Base.setindex!(c::CoSort, t::CoSortTup, i) =
    (setindex!(c.sortvector, t.x, i + c.offset); setindex!(c.covector, t.y, i + c.offset); c)

Base.isless(a::CoSortTup, b::CoSortTup) = isless(a.x, b.x)

Base.Sort.defalg(v::C) where {T<:Union{Number, Missing}, C<:CoSort{T}} = Base.DEFAULT_UNSTABLE

isgrowing(c::CoSort) = isgrowing(c.sortvector, c.offset + 1)

function isgrowing(vs::AbstractVector, i0 = 1)
    i0 > length(vs) && return true
    vprev = vs[i0]
    for i in i0 + 1:length(vs)
        v = vs[i]
        v <= vprev && return false
        vprev = v
    end
    return true
end

#endregion

#######################################################################
# Combinations -- gratefully borrowed from Combinatorics.jl
#region

struct Combinations
    n::Int
    t::Int
end

@inline function Base.iterate(c::Combinations, s = [min(c.t - 1, i) for i in 1:c.t])
    if c.t == 0 # special case to generate 1 result for t==0
        isempty(s) && return (s, [1])
        return
    end
    # for i in c.t:-1:1
    for ii in 1:c.t
        i = c.t + 1 - ii
        s[i] += 1
        if s[i] > (c.n - (c.t - i))
            continue
        end
        for j in i+1:c.t
            s[j] = s[j-1] + 1
        end
        break
    end
    s[1] > c.n - c.t + 1 && return
    (s, s)
end

Base.length(c::Combinations) = binomial(c.n, c.t)

Base.eltype(::Type{Combinations}) = Vector{Int}

Base.IteratorSize(::Type{Combinations}) = Base.HasLength()

Base.IteratorEltype(::Type{Combinations}) = Base.HasEltype()

#######################################################################
# Runs
#region

# iteration yields ranges of subsequent xs elements such that istogether on them gives true
struct Runs{T,F}
    xs::Vector{T}
    istogether::F
end

equalruns(xs) = Runs(xs, ==)
approxruns(xs::Vector{T}, atol = sqrt(eps(real(T)))) where {T<:Number} = Runs(xs, (x, y) -> isapprox(x, y; atol))

function last_in_run(xs::Vector{T}, i, istogether) where {T}
    xi = xs[i]
    for j in i:length(xs)
        (j == length(xs) || !istogether(xs[j+1], xi)) && return j
    end
    return i
end

function Base.iterate(s::Runs, j = 1)
    j > length(s.xs) && return nothing
    lst = last_in_run(s.xs, j, s.istogether)
    return j:lst, lst + 1
end

Base.IteratorSize(::Runs) = Base.SizeUnknown()

Base.IteratorEltype(::Runs) = Base.HasEltype()

Base.eltype(::Runs) = UnitRange{Int}

#endregion


# #######################################################################
# # MarchingSimplices
# #######################################################################
# struct MarchingSimplices{D,D´}
#     cinds::CartesianIndices{D´,NTuple{D´,Base.OneTo{Int}}}
#     simps::Vector{NTuple{D´,CartesianIndex{D}}}  # simplices verts in unit cuboid
#     unitperms::Vector{NTuple{D,Int}}             # unit vector order of simplices
# end

# # c is a CartesianIndices over all vertices
# function marchingsimplices(c::CartesianIndices{D}) where {D}
#     unitperms = marchingsimplices_unitperms(Val(D))
#     edgeperms = map(edge -> unitvector.(CartesianIndex{D}, edge), unitperms)
#     simps = marchingsimplices_verts.(edgeperms)
#     srange = eachindex(simps)
#     refverts = ntuple(i -> Base.OneTo(c.indices[i][1:end-1]), Val(D))
#     cinds = CartesianIndices((refverts..., srange))
#     return MarchingSimplices(cinds, simps, unitperms)
# end

# # indices of unit vectors in a unit cuboid, taken in any order
# marchingsimplices_unitperms(::Val{D}) where {D} = permutations(ntuple(identity, Val(D)))

# # simplex vertices as the sum of unit vectors in any permutation
# marchingsimplices_verts(edges::NTuple{D}) where {D} =
#     (zero(CartesianIndex{D}), edges...) |> cumsum |> orientsimp

# function orientsimp(sverts)
#     volume = det(hcat(SVector.(Tuple.(Base.tail(sverts)))...))
#     sverts´ = ifelse(volume >= 0, sverts, switchlast(sverts))
#     return sverts´
# end

# switchlast(s::NTuple{N,T}) where {N,T} = ntuple(i -> i < N - 1 ? s[i] : s[2N - i - 1] , Val(N))

# function Base.iterate(m::MarchingSimplices, s...)
#     it = iterate(m.cinds, s...)
#     it === nothing && return nothing
#     c, s´ = it
#     t = Tuple(c)
#     nsimp = last(t)
#     refvert = CartesianIndex(Base.front(t))
#     verts = Ref(refvert) .+ m.simps[nsimp]
#     return verts, s´
# end

# Base.IteratorSize(::MarchingSimplices{D}) where {D} = Base.HasShape{D}()

# Base.axes(m::MarchingSimplices, i...) = axes(m.cinds, i...)

# Base.size(m::MarchingSimplices, i...) = size(m.cinds, i...)

# Base.CartesianIndices(m::MarchingSimplices) = m.cinds

# Base.IteratorEltype(::MarchingSimplices) = Base.HasEltype()

# Base.eltype(::MarchingSimplices{D,D´}) where {D,D´} = NTuple{D´,CartesianIndex{D}}

# Base.length(m::MarchingSimplices) = prod(length.(axes(m)))


# permutations(ss::NTuple) = permutations!(typeof(ss)[], ss, ())

# function permutations!(p, s1, s2)
#     for (i, s) in enumerate(s1)
#         permutations!(p, delete(s1, i), (s2..., s))
#     end
#     return p
# end

# permutations!(p, ::Tuple{}, s2) = push!(p, s2)

# delete(t::NTuple{N,Any}, i) where {N} = ntuple(j -> j < i ? t[j] : t[j+1], Val(N-1))


# unitvector(::Type{SVector{L,T}}, i) where {L,T} = SVector{L,T}(unitvector(NTuple{L,T}, i))
# unitvector(::Type{CartesianIndex{L}}, i) where {L} = CartesianIndex(unitvector(NTuple{L,Int}, i))
# unitvector(::Type{NTuple{L,T}}, i) where {L,T} =ntuple(j -> j == i ? one(T) : zero(T), Val(L))


# #######################################################################
# # SparseMatrixBuilder
# #######################################################################

# mutable struct SparseMatrixBuilder{T,S<:AbstractSparseMatrix{T}}
#     matrix::S
#     colcounter::Int
#     rowvalcounter::Int
#     cosorter::CoSort{Int,T,Vector{Int},Vector{T}}
# end

# struct UnfinalizedSparseMatrixCSC{Tv,Ti} <: AbstractSparseMatrix{Tv,Ti}
#     m::Int
#     n::Int
#     colptr::Vector{Ti}
#     rowval::Vector{Ti}
#     nzval::Vector{Tv}
# end

# function SparseMatrixBuilder{Tv}(m, n, nnzguess = missing) where {Tv}
#     colptr = [1]
#     rowval = Int[]
#     nzval = Tv[]
#     matrix = UnfinalizedSparseMatrixCSC(m, n, colptr, rowval, nzval)
#     builder = SparseMatrixBuilder(matrix, 1, 0, CoSort(rowval, nzval))
#     nnzguess === missing || sizehint!(builder, nnzguess)
#     return builder
# end

# # Unspecified size constructor
# SparseMatrixBuilder{Tv}() where Tv = SparseMatrixBuilder{Tv}(0, 0)

# function SparseMatrixBuilder(s::SparseMatrixCSC{Tv,Int}) where {Tv}
#     colptr = getcolptr(s)
#     rowval = rowvals(s)
#     nzval = nonzeros(s)
#     nnzguess = length(nzval)
#     resize!(rowval, 0)
#     resize!(nzval, 0)
#     resize!(colptr, 1)
#     colptr[1] = 1
#     builder = SparseMatrixBuilder(s, 1, 0, CoSort(rowval, nzval))
#     sizehint!(builder, nnzguess)
#     return builder
# end

# SparseArrays.getcolptr(s::UnfinalizedSparseMatrixCSC) = s.colptr
# SparseArrays.nonzeros(s::UnfinalizedSparseMatrixCSC) = s.nzval
# SparseArrays.rowvals(s::UnfinalizedSparseMatrixCSC) = s.rowval
# Base.size(s::UnfinalizedSparseMatrixCSC) = (s.m, s.n)
# Base.size(s::UnfinalizedSparseMatrixCSC, k) = size(s)[k]

# function Base.sizehint!(s::SparseMatrixBuilder, n)
#     sizehint!(getcolptr(s.matrix), n + 1)
#     sizehint!(nonzeros(s.matrix), n)
#     sizehint!(rowvals(s.matrix), n)
#     return s
# end

# function pushtocolumn!(s::SparseMatrixBuilder, row::Int, x, skipdupcheck::Bool = true)
#     nrows = size(s.matrix, 1)
#     nrows == 0 || 1 <= row <= size(s.matrix, 1) || throw(ArgumentError("tried adding a row $row out of bounds ($(size(s.matrix, 1)))"))
#     if skipdupcheck || !isintail(row, rowvals(s.matrix), getcolptr(s.matrix)[s.colcounter])
#         push!(rowvals(s.matrix), row)
#         push!(nonzeros(s.matrix), x)
#         s.rowvalcounter += 1
#     end
#     return s
# end

# function isintail(element, container, start::Int)
#     for i in start:length(container)
#         container[i] == element && return true
#     end
#     return false
# end

# function finalizecolumn!(s::SparseMatrixBuilder, sortcol::Bool = true)
#     size(s.matrix, 2) > 0 && s.colcounter > size(s.matrix, 2) && throw(DimensionMismatch("Pushed too many columns to matrix"))
#     if sortcol
#         s.cosorter.offset = getcolptr(s.matrix)[s.colcounter] - 1
#         sort!(s.cosorter)
#         isgrowing(s.cosorter) || throw(error("Internal error: repeated rows"))
#     end
#     s.colcounter += 1
#     push!(getcolptr(s.matrix), s.rowvalcounter + 1)
#     return nothing
# end

# function finalizecolumn!(s::SparseMatrixBuilder, ncols::Int)
#     for _ in 1:ncols
#         finalizecolumn!(s)
#     end
#     return nothing
# end

# function SparseArrays.sparse(s::SparseMatrixBuilder{<:Any,<:SparseMatrixCSC})
#     completecolptr!(getcolptr(s.matrix), size(s.matrix, 2), s.rowvalcounter)
#     return s.matrix
# end

# function completecolptr!(colptr, cols, lastrowptr)
#     colcounter = length(colptr)
#     if colcounter < cols + 1
#         resize!(colptr, cols + 1)
#         for col in (colcounter + 1):(cols + 1)
#             colptr[col] = lastrowptr + 1
#         end
#     end
#     return colptr
# end

# function SparseArrays.sparse(s::SparseMatrixBuilder{<:Any,<:UnfinalizedSparseMatrixCSC})
#     m, n = size(s.matrix)
#     rowval = rowvals(s.matrix)
#     colptr = getcolptr(s.matrix)
#     nzval = nonzeros(s.matrix)
#     if m != 0 && n != 0
#         completecolptr!(colptr, n, s.rowvalcounter)
#     else # determine size of matrix after the fact
#         m, n = isempty(rowval) ? 0 : maximum(rowval), s.colcounter - 1
#     end
#     return SparseMatrixCSC(m, n, colptr, rowval, nzval)
# end

# #######################################################################
# # SparseMatrixReader
# #######################################################################

# struct SparseMatrixReader{Tv,Ti}
#     matrix::SparseMatrixCSC{Tv,Ti}
# end

# Base.IteratorSize(::SparseMatrixReader) = Base.HasLength()

# Base.IteratorEltype(::SparseMatrixReader) = Base.HasEltype()

# Base.eltype(::SparseMatrixReader{Tv,Ti}) where {Tv,Ti} = Tuple{Ti,Ti,Tv}

# Base.length(s::SparseMatrixReader) = nnz(s.matrix)

# function Base.iterate(s::SparseMatrixReader, state = (1, 1))
#     (ptr, col) = state
#     ptr > length(s) && return nothing
#     @inbounds while ptr > s.matrix.colptr[col + 1] - 1
#          col += 1
#     end
#     return (s.matrix.rowval[ptr], col, s.matrix.nzval[ptr], ptr), (ptr + 1, col)
# end

# enumerate_sparse(s::SparseMatrixCSC) = SparseMatrixReader(s)

# #######################################################################
# # SiteSublats
# #######################################################################

# struct SiteSublats{G,U}
#     g::G
#     u::U
# end

# sitesublats(u) = SiteSublats(((i, s) for s in sublats(u) for i in siterange(u, s)), u)

# Base.iterate(s::SiteSublats, x...) = iterate(s.g, x...)

# Base.IteratorSize(::SiteSublats) = Base.HasLength()

# Base.IteratorEltype(::SiteSublats) = Base.HasEltype()

# Base.eltype(::SiteSublats) = Tuple{Int,Int}

# Base.length(s::SiteSublats) = nsites(s.u)

# #######################################################################
# # Runs
# #######################################################################
# # iteration yields ranges of subsequent xs elements such that istogether on them gives true
# struct Runs{T,F}
#     xs::Vector{T}
#     istogether::F
# end

# equalruns(xs) = Runs(xs, ==)
# approxruns(xs::Vector{T}) where {T<:Number} = Runs(xs, (x, y) -> isapprox(x, y; atol = sqrt(eps(real(T)))))

# function last_in_run(xs::Vector{T}, i, istogether) where {T}
#     xi = xs[i]
#     for j in i:length(xs)
#         (j == length(xs) || !istogether(xs[j+1], xi)) && return j
#     end
#     return i
# end

# # Base.iterate(s::Runs) = iterate(s, 1)
# function Base.iterate(s::Runs, j = 1)
#     j > length(s.xs) && return nothing
#     lst = last_in_run(s.xs, j, s.istogether)
#     return j:lst, lst + 1
# end

# Base.IteratorSize(::Runs) = Base.SizeUnknown()

# Base.IteratorEltype(::Runs) = Base.HasEltype()

# Base.eltype(::Runs) = UnitRange{Int}

# #######################################################################
# # MarchingNeighbors
# #######################################################################
# struct MarchingNeighbors{D,G}
#     n::CartesianIndex{D}
#     gen::G
#     len::Int
# end

# # c is a CartesianRange over vertices, and n is a CartesianIndex from which to find neighbors
# function marchingneighbors(c, n)
#     rp = max(n, first(c)):min(n + oneunit(n), last(c))
#     rm = max(n - oneunit(n), first(c)):min(n, last(c))
#     gen = Iterators.filter(!isequal(n), Iterators.flatten((rp, rm)))
#     len = length(rp) + length(rm) - 2
#     return MarchingNeighbors(n, gen, len)
# end

# function marchingneighbors_forward(c, n)
#     rp = max(n, first(c)):min(n + oneunit(n), last(c))
#     gen = Iterators.filter(!isequal(n), rp)
#     len = length(rp) - 2
#     return MarchingNeighbors(n, gen, len)
# end

# Base.iterate(m::MarchingNeighbors, s...) = iterate(m.gen, s...)

# Base.IteratorSize(::MarchingNeighbors) = Base.HasLength()

# Base.IteratorEltype(::MarchingNeighbors) = Base.HasEltype()

# Base.eltype(::MarchingNeighbors{D}) where {D} = CartesianIndex{D}

# Base.length(m::MarchingNeighbors) = m.len

# #######################################################################
# # MarchingSimplices
# #######################################################################
# struct MarchingSimplices{D,D´}
#     cinds::CartesianIndices{D´,NTuple{D´,Base.OneTo{Int}}}
#     simps::Vector{NTuple{D´,CartesianIndex{D}}}  # simplices verts in unit cuboid
#     unitperms::Vector{NTuple{D,Int}}             # unit vector order of simplices
# end

# # c is a CartesianIndices over all vertices
# function marchingsimplices(c::CartesianIndices{D}) where {D}
#     unitperms = marchingsimplices_unitperms(Val(D))
#     edgeperms = map(edge -> unitvector.(CartesianIndex{D}, edge), unitperms)
#     simps = marchingsimplices_verts.(edgeperms)
#     srange = eachindex(simps)
#     refverts = ntuple(i -> Base.OneTo(c.indices[i][1:end-1]), Val(D))
#     cinds = CartesianIndices((refverts..., srange))
#     return MarchingSimplices(cinds, simps, unitperms)
# end

# # indices of unit vectors in a unit cuboid, taken in any order
# marchingsimplices_unitperms(::Val{D}) where {D} = permutations(ntuple(identity, Val(D)))

# # simplex vertices as the sum of unit vectors in any permutation
# marchingsimplices_verts(edges::NTuple{D}) where {D} =
#     (zero(CartesianIndex{D}), edges...) |> cumsum |> orientsimp

# function orientsimp(sverts)
#     volume = det(hcat(SVector.(Tuple.(Base.tail(sverts)))...))
#     sverts´ = ifelse(volume >= 0, sverts, switchlast(sverts))
#     return sverts´
# end

# switchlast(s::NTuple{N,T}) where {N,T} = ntuple(i -> i < N - 1 ? s[i] : s[2N - i - 1] , Val(N))

# function Base.iterate(m::MarchingSimplices, s...)
#     it = iterate(m.cinds, s...)
#     it === nothing && return nothing
#     c, s´ = it
#     t = Tuple(c)
#     nsimp = last(t)
#     refvert = CartesianIndex(Base.front(t))
#     verts = Ref(refvert) .+ m.simps[nsimp]
#     return verts, s´
# end

# Base.IteratorSize(::MarchingSimplices{D}) where {D} = Base.HasShape{D}()

# Base.axes(m::MarchingSimplices, i...) = axes(m.cinds, i...)

# Base.size(m::MarchingSimplices, i...) = size(m.cinds, i...)

# Base.CartesianIndices(m::MarchingSimplices) = m.cinds

# Base.IteratorEltype(::MarchingSimplices) = Base.HasEltype()

# Base.eltype(::MarchingSimplices{D,D´}) where {D,D´} = NTuple{D´,CartesianIndex{D}}

# Base.length(m::MarchingSimplices) = prod(length.(axes(m)))