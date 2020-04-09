######################################################################
# Mesh
#######################################################################

abstract type AbstractMesh{D} end

struct Mesh{D,T,V<:AbstractArray{SVector{D,T}}} <: AbstractMesh{D}   # D is dimension of parameter space
    vertices::V                         # Iterable vertex container with SVector{D,T} eltype
    adjmat::SparseMatrixCSC{Bool,Int}   # Undirected graph: both dest > src and dest < src
end

# const Mesh{D,T} = Mesh{D,T,Vector{SVector{D,T}},Vector{Tuple{Int,Vararg{Int,D}}}}

# Mesh{D,T}() where {D,T} = Mesh(SVector{D,T}[], sparse(Int[], Int[], Bool[]), NTuple{D+1,Int}[])

function Base.show(io::IO, mesh::Mesh{D}) where {D}
    i = get(io, :indent, "")
    print(io,
"$(i)Mesh{$D}: mesh of a $D-dimensional manifold
$i  Vertices   : $(nvertices(mesh))
$i  Edges      : $(nedges(mesh))")
end

nvertices(m::Mesh) = length(m.vertices)

nedges(m::Mesh) = div(nnz(m.adjmat), 2)

nsimplices(m::Mesh) = length(simplices(m))

vertices(m::Mesh) = m.vertices

edges(m::Mesh, src) = nzrange(m.adjmat, src)

edgedest(m::Mesh, edge) = rowvals(m.adjmat)[edge]

edgevertices(m::Mesh) =
    ((vsrc, m.vertices[edgedest(m, edge)]) for (i, vsrc) in enumerate(m.vertices) for edge in edges(m, i))

function minmax_edge_length(m::Mesh{D,T}) where {D,T<:Real}
    minlen2 = typemax(T)
    maxlen2 = zero(T)
    verts = vertices(m)
    for src in eachindex(verts), edge in edges(m, src)
        dest = edgedest(m, edge)
        dest > src || continue # Need only directed graph
        vec = verts[dest] - verts[src]
        norm2 = vec' * vec
        norm2 < minlen2 && (minlen2 = norm2)
        norm2 > maxlen2 && (maxlen2 = norm2)
    end
    return sqrt(minlen2), sqrt(maxlen2)
end

######################################################################
# Compute N-simplices (N = number of vertices)
######################################################################
function simplices(mesh::Mesh{D}, ::Val{N} = Val(D+1)) where {D,N}
    N > 0 || throw(ArgumentError("Need a positive number of vertices for simplices"))
    N == 1 && return Tuple.(1:nvertices(mesh))
    simps = NTuple{N,Int}[]
    buffer = (NTuple{N,Int}[], NTuple{N,Int}[], Int[])
    for src in eachindex(vertices(mesh))
        append!(simps, _simplices(buffer, mesh, src))
    end
    N > 2 && alignnormals!(simps, vertices(mesh))
    return simps
end

# Add (greater) neighbors to last vertex of partials that are also neighbors of scr, till N
function _simplices(buffer::Tuple{P,P,V}, mesh, src) where {N,P<:AbstractArray{<:NTuple{N}},V}
    partials, partials´, srcneighs = buffer
    resize!(srcneighs, 0)
    resize!(partials, 0)
    for edge in edges(mesh, src)
        srcneigh = edgedest(mesh, edge)
        srcneigh > src || continue # Directed graph, to avoid simplex duplicates
        push!(srcneighs, srcneigh)
        push!(partials, padright((src, srcneigh), 0, Val(N)))
    end
    for pass in 3:N
        resize!(partials´, 0)
        for partial in partials
            nextsrc = partial[pass - 1]
            for edge in edges(mesh, nextsrc)
                dest = edgedest(mesh, edge)
                dest > nextsrc || continue # If not directed, no need to check
                dest in srcneighs && push!(partials´, modifyat(partial, pass, dest))
            end
        end
        partials, partials´ = partials´, partials
    end
    return partials
end

modifyat(s::NTuple{N,T}, ind, el) where {N,T} = ntuple(i -> i === ind ? el : s[i], Val(N))

function alignnormals!(simplices, vertices)
    for (i, s) in enumerate(simplices)
        volume = elementvolume(vertices, s)
        volume < 0 && (simplices[i] = switchlast(s))
    end
    return simplices
end

# Project N-1 edges onto (N-1)-dimensional vectors to have a deterministic volume
elementvolume(verts, s::NTuple{N,Int}) where {N} =
    elementvolume(hcat(ntuple(i -> padright(SVector(verts[s[i+1]] - verts[s[1]]), Val(N-1)), Val(N-1))...))
elementvolume(mat::SMatrix{N,N}) where {N} = det(mat)

switchlast(s::NTuple{N,T}) where {N,T} = ntuple(i -> i < N - 1 ? s[i] : s[2N - i - 1] , Val(N))

######################################################################
# Special meshes
######################################################################
"""
    marchingmesh(ranges::Vararg{AbstractRange,L}; axes = 1.0 * I)

Creates a L-dimensional marching-tetrahedra `Mesh` over a parallelepiped with axes given by
the columns of `axes`. The points along axis `i` are given by `ranges[i]`.

# External links

- Marching tetrahedra (https://en.wikipedia.org/wiki/Marching_tetrahedra) in Wikipedia
"""
marchingmesh(ranges::Vararg{AbstractRange,L}; axes = 1.0 * I) where {L} =
    _marchingmesh(ranges, SMatrix{L,L}(axes))

marchingmesh(; kw...) = throw(ArgumentError("Need a finite number of ranges to build a mesh"))

function _marchingmesh(ranges::NTuple{D,AbstractRange}, axes::SMatrix{D,D}) where {D}
    npoints = length.(ranges)
    cs = CartesianIndices(ntuple(n -> 1:npoints[n], Val(D)))
    ls = LinearIndices(cs)
    csinner = CartesianIndices(ntuple(n -> 1:npoints[n]-1, Val(D)))

    # edge vectors for marching tetrahedra in D-dimensions (skip zero vector [first])
    uedges = [c for c in CartesianIndices(ntuple(_ -> 0:1, Val(D)))][2:end]
    # tetrahedra built from the D unit-length uvecs added in any permutation
    perms = permutations(
            ntuple(i -> CartesianIndex(ntuple(j -> i == j ? 1 : 0, Val(D))), Val(D)))
    utets = [cumsum(pushfirst!(perm, zero(CartesianIndex{D}))) for perm in perms]

    # We don't use generators because their non-inferreble eltype causes problems later
    verts = [axes * SVector(getindex.(ranges, Tuple(c))) for c in cs]

    s = SparseMatrixBuilder{Bool}(length(cs), length(cs))
    for c in cs
        for u in uedges
            dest = c + u    # dest > src
            dest in cs && pushtocolumn!(s, ls[dest], true)
            dest = c - u    # dest < src
            dest in cs && pushtocolumn!(s, ls[dest], true)
        end
        finalizecolumn!(s)
    end
    adjmat = sparse(s)

    return Mesh(verts, adjmat)
end
