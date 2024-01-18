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

eldim(::BoxIterator{N}) where {N} = N

function BoxIterator(seed::SVector{N}; maxiterations = TOOMANYITERS) where {N}
    BoxIterator(seed, maxiterations, MVector(1, 2),
        ones(MVector{N,Bool}), ones(MVector{N,Bool}), MVector{N,Int}(seed), MVector{N,Int}(seed))
end

struct BoxIteratorState{N}
    range::CartesianIndices{N, NTuple{N,UnitRange{Int}}}
    rangestate::CartesianIndex{N}
    iteration::Int
end

Base.iterate(b::BoxIterator{0}) = (SVector{0,Int}(), nothing)
Base.iterate(b::BoxIterator{0}, state) = nothing

function Base.iterate(b::BoxIterator)
    N = eldim(b)
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


function nextface!(b::BoxIterator)
    N = eldim(b)
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

function nextdimdir!(b::BoxIterator)
    N = eldim(b)
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

@inline function newrangeneg(b::BoxIterator, dim)
    N = eldim(b)
    return CartesianIndices(ntuple(
        i -> b.npos[i]:(i == dim ? b.npos[i] : b.ppos[i]),
        Val(N)))
end

@inline function newrangepos(b::BoxIterator, dim)
    N = eldim(b)
    return CartesianIndices(ntuple(
        i -> (i == dim ? b.ppos[i] : b.npos[i]):b.ppos[i],
        Val(N)))
end

function acceptcell!(b::BoxIterator, cell)
    N = eldim(b)
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

# iteration yields ranges of subsequent xs elements such that istogether of consecutive pairs gives true
struct Runs{T,F}
    xs::Vector{T}
    istogether::F
end

equalruns(xs) = Runs(xs, ==)
approxruns(xs::Vector{T}, atol = sqrt(eps(real(T)))) where {T<:Number} = Runs(xs, (x, y) -> isapprox(x, y; atol))

function Base.iterate(s::Runs, frst = 1)
    xs = s.xs
    frst > length(xs) && return nothing
    # find first element in run
    for frst´ in frst:length(xs)
        xj´ = xs[frst´]
        if s.istogether(xj´, xj´)
            frst = frst´
            break
        elseif frst´ == length(xs)
            return nothing
        end
    end
    # find last element in run, which is at least xs[frst]
    lst = frst
    for lst´ in frst+1:length(xs)
        if !s.istogether(xs[lst´-1], xs[lst´])
            lst = lst´-1
            break
        else
            lst = lst´
        end
    end
    return frst:lst, lst + 1
end

Base.IteratorSize(::Runs) = Base.SizeUnknown()

Base.IteratorEltype(::Runs) = Base.HasEltype()

Base.eltype(::Runs) = UnitRange{Int}

#endregion
