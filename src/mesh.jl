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

function minmax_edge(m::Mesh{D,T}) where {D,T<:Real}
    minlen2 = typemax(T)
    maxlen2 = zero(T)
    verts = vertices(m)
    minedge = zero(first(verts))
    maxedge = zero(first(verts))
    for src in eachindex(verts), edge in edges(m, src)
        dest = edgedest(m, edge)
        dest > src || continue # Need only directed graph
        vec = verts[dest] - verts[src]
        norm2 = vec' * vec
        norm2 < minlen2 && (minlen2 = norm2; minedge = vec)
        norm2 > maxlen2 && (maxlen2 = norm2; maxedge = vec)
    end
    return minedge, maxedge
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
# Mesh specifications
######################################################################
"""
    MeshSpec

Parent type of mesh specifications, which are currently `MarchingMeshSpec` (constructed with
`marchingmesh`) and `LinearMeshSpec` (constructed with `linearmesh`).

# See also
    `marchingmesh`, `linearmesh`, `buildmesh`
"""
abstract type MeshSpec{L} end

Base.show(io::IO, spec::MeshSpec{L}) where {L} =
    print(io, "MeshSpec{$L} : specifications for building a $(L)D mesh.")

#fallback
buildlift(::MeshSpec, bravais) = missing


#######################################################################
# MarchingMeshSpec
#######################################################################
struct MarchingMeshSpec{L,R,T,M<:NTuple{L,Tuple{Number,Number}}} <: MeshSpec{L}
    minmaxaxes::M
    axes::SMatrix{L,L,T}
    points::R
end

"""
    marchingmesh(minmaxaxes...; axes = 1.0 * I, points = 13)

Create a `MeshSpec` for a L-dimensional marching-tetrahedra `Mesh` over a parallelepiped
with axes given by the columns of `axes`. The points along axis `i` are distributed between
`first(minmaxaxes[i])` and `last(minmaxaxes[i])`. The number of points on each axis is given
by `points`, or `points[i]` if several are given.

# Examples

```jldoctest
julia> buildmesh(marchingmesh((-π, π), (0,2π); points = 25))
Mesh{2}: mesh of a 2-dimensional manifold
  Vertices   : 625
  Edges      : 1776

julia> buildmesh(marchingmesh((-π, π), (0,2π); points = (10,10)))
Mesh{2}: mesh of a 2-dimensional manifold
  Vertices   : 100
  Edges      : 261
```

# See also
    `linearmesh`, `buildmesh`

# External links
- Marching tetrahedra (https://en.wikipedia.org/wiki/Marching_tetrahedra) in Wikipedia
"""
marchingmesh(minmaxaxes::Vararg{Tuple,L}; axes = 1.0 * I, points = 13) where {L} =
    MarchingMeshSpec(minmaxaxes, SMatrix{L,L}(axes), points)

marchingmesh(; kw...) = throw(ArgumentError("Need a finite number of ranges to define a marching mesh"))

function buildmesh(m::MarchingMeshSpec{D}, bravais = missing) where {D}
    ranges = ((b, r)->range(b...; length = r)).(m.minmaxaxes, m.points)
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

    # We don't use generators because their non-inferreble eltype causes problems elsewhere
    verts = [m.axes * SVector(getindex.(ranges, Tuple(c))) for c in cs]

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

buildlift(::MarchingMeshSpec{L}, bravais::SMatrix{E,L}) where {E,L} = missing
buildlift(::MarchingMeshSpec{L´}, bravais::SMatrix{E,L}) where {E,L,L´} =
    (pt...) -> padright(pt, Val(L))

#######################################################################
# LinearMeshSpec
#######################################################################
struct LinearMeshSpec{N,L,T,R} <: MeshSpec{1}
    vertices::SVector{N,SVector{L,T}}
    samelength::Bool
    closed::Bool
    points::R
end

"""
    linearmesh(nodes...; points = 13, samelength = false, closed = false)

Create a `MeshSpec` for a one-dimensional `Mesh` connecting the `nodes` with straight
segments, each containing a number `points` of points (endpoints included). If a different
number of points for each of the `N` segments is required, use `points::NTuple{N,Int}`.
If `samelength` each segment has equal length in mesh coordinates. If `closed` the last node
is connected to the first node.

# Examples

```jldoctest
julia> buildmesh(linearmesh(:Γ, :K, :M, :Γ; points = (101, 30, 30)))
Mesh{1}: mesh of a 1-dimensional manifold
  Vertices   : 159
  Edges      : 158
```

# See also
    `marchingmesh`, `buildmesh`
"""
linearmesh(nodes...; points = 13, samelength::Bool = false, closed::Bool = false) =
    LinearMeshSpec(sanitize_BZpts(nodes), samelength, closed, points)

function sanitize_BZpts(pts)
    pts´ = parse_BZpoint.(pts)
    dim = maximum(length.(pts´))
    pts´´ = SVector(padright.(pts´, Val(dim)))
    return pts´´
end

parse_BZpoint(p::SVector) = float.(p)
parse_BZpoint(p::Tuple) = SVector(float.(p))
function parse_BZpoint(p::Symbol)
    pt = get(BZpoints, p, missing)
    pt === missing && throw(ArgumentError("Unknown Brillouin zone point $p, use one of $(keys(BZpoints))"))
    return SVector(float.(pt))
end
const BZpoints =
    ( Γ  = (0,)
    , X  = (pi,)
    , Y  = (0, pi)
    , Z  = (0, 0, pi)
    , K  = (2pi/3, -2pi/3)
    , Kp = (4pi/3, 2pi/3)
    , M  = (pi, 0)
    )

linearmesh_nodes(l, br) = cumsum(SVector(0, segment_lengths(l, br)...))

function segment_lengths(s::LinearMeshSpec{N,LS,TS}, br::SMatrix{E,LB,TB}) where {TS,TB,N,E,LS,LB}
    T = promote_type(TS, TB)
    verts = padright.(s.vertices, Val(LB))
    dϕs = ntuple(i -> verts[i + 1] - verts[i], Val(N-1))
    if s.samelength
        ls = filltuple(T(1/(N-1)), Val(N-1))
    else
        ibr = pinverse(br)'
        ls = (dϕ -> norm(ibr * dϕ)).(dϕs)
        ls = ls ./ sum(ls)
    end
    return ls
end

function idx_to_node(s, br)
    nodes = SVector.(linearmesh_nodes(s, br))
    nmax = length(nodes)
    nodefunc = nvec -> begin
        n = only(nvec)
        node = if n >= nmax
            nodes[nmax]
        else
            nc = max(n, 1)
            i = Int(floor(nc))
            nodes[i] + rem(nc, 1) * (nodes[i+1] - nodes[i])
        end
        return node
    end
    return nodefunc
end

buildmesh(s::LinearMeshSpec{N,L,T}) where {N,L,T} = buildmesh(s, SMatrix{L,L,T}(I))

function buildmesh(s::LinearMeshSpec{N}, br) where {N}
    ranges = ((i, r) -> range(i, i+1, length = r)).(ntuple(identity, Val(N-1)), s.points)
    verts = SVector.(first(ranges))
    for r in Base.tail(ranges)
        pop!(verts)
        append!(verts, SVector.(r))
    end
    s.closed && pop!(verts)
    nv = length(verts)
    nodefunc = idx_to_node(s, br)
    verts .= nodefunc.(verts)
    adjmat = sparse(vcat(1:nv-1, 2:nv), vcat(2:nv, 1:nv-1), true, nv, nv)
    s.closed && (adjmat[end, 1] = adjmat[1, end] = true)
    return Mesh(verts, adjmat)
end

function buildlift(s::LinearMeshSpec, br::SMatrix{E,L}) where {E,L}
    ls = segment_lengths(s, br)
    nodes = linearmesh_nodes(s, br)
    verts = padright.(s.vertices, Val(L))
    l = sum(ls)
    liftfunc = x -> begin
        xc = clamp(only(x), 0, l)
        for (i, node) in enumerate(nodes)
            if node > xc
                p = verts[i-1] + (xc - nodes[i-1])/ls[i-1] * (verts[i]-verts[i-1])
                return p
            end
        end
        return last(verts)
    end
    return liftfunc
end