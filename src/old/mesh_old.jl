######################################################################
# CuboidMesh
######################################################################
struct CuboidMesh{D,T,D´}
    ticks::NTuple{D,Vector{T}}
    simpitr::MarchingSimplices{D,D´}
end

function Base.show(io::IO, mesh::CuboidMesh{D}) where {D}
    i = get(io, :indent, "")
    print(io,
"$(i)CuboidMesh{$D}: a mesh of a $(D)D parameter cuboid
$i  Ranges     : $(extrema.(mesh.ticks))
$i  Axes ticks : $(length.(mesh.ticks))
$i  Simplices  : $(size(mesh.simpitr)) -> $(length(mesh.simpitr))")
end

"""
    cuboid(ticks...; subticks = 13)

Create a `CuboidMesh` of L-dimensional marching-tetrahedra over a cuboid aligned with the
Cartesian axes. The dimension `L` is given by the number of `ticks`, each of the form `(x₁,
x₂,...)`. The interval between `xⱼ` and `xⱼ₊₁` ticks in axis `i` are further subdivided to
have a number of subticks including endpoints. The number is `subticks` if `subticks` is an
`Integer`, `subticks[i]` if `subticks = (s₁, s₂,...)` or `subticks[i][j]` if `subticks =
((s₁₁, s₁₂,...), (s₂₁, s₂₂,...), ...)`.

# Examples

```jldoctest
julia> cuboid((-π, π), (0, 2π); subticks = 25)
CuboidMesh{2}: a mesh of a 2D parameter cuboid
  Ranges     : ((-3.141592653589793, 3.141592653589793), (0.0, 6.283185307179586))
  Axes ticks : (25, 25)
  Simplices  : (24, 24, 2) -> 1152

julia> cuboid((-π, π), (0, 2π); subticks = (10, 10))
CuboidMesh{2}: a mesh of a 2D parameter cuboid
  Ranges     : ((-3.141592653589793, 3.141592653589793), (0.0, 6.283185307179586))
  Axes ticks : (10, 10)
  Simplices  : (9, 9, 2) -> 162
```

# External links
- Marching tetrahedra (https://en.wikipedia.org/wiki/Marching_tetrahedra) in Wikipedia
"""
cuboid(ticks::Vararg{Tuple,L}; subticks = 13) where {L} = _cuboid(sanitize_subticks(subticks, ticks), ticks...)

sanitize_subticks(st::NTuple{L,Any}, t::NTuple{L,Any}) where {L} = _sanitize_subticks.(st, t)
sanitize_subticks(st, t) = _sanitize_subticks.(Ref(st), t)
_sanitize_subticks(st::Number, t::Tuple{Vararg{Number}}) = Base.tail(_sanitize_subticks.(st, t))
_sanitize_subticks(st::Number, ::Number) = Int(st)
_sanitize_subticks(st::Tuple{Vararg{Number,L´}}, ::Tuple{Vararg{Number,L}}) where {L,L´} = L´ == L - 1 ? st :
    throw(ArgumentError("Malformed `subticks`. The number of subticks for each axis should be one less than the number of ticks for that axis"))

function _cuboid(subticks, axesticks::Vararg{Tuple,L}) where {L}
    allticks = ntuple(Val(L)) do i
        allaxisticks = typeof(1.0)[]  # We want the machine's float type, without committing to Float64
        axisticks = axesticks[i]
        nticks = length(axisticks)
        foreach(1:nticks-1) do j
            append!(allaxisticks, range(axisticks[j], axisticks[j+1], length = subticks[i][j]))
            j == nticks-1 || pop!(allaxisticks)
        end
        allaxisticks
    end
    simpitr = marchingsimplices(CartesianIndices(eachindex.(allticks)))
    return CuboidMesh(allticks, simpitr)
end

Base.eachindex(mesh::CuboidMesh) = CartesianIndices(eachindex.(mesh.ticks))

Base.getindex(mesh::CuboidMesh, n::CartesianIndex) = SVector(getindex.(mesh.ticks, Tuple(n)))
Base.getindex(mesh::CuboidMesh, i...) = mesh[eachindex(mesh)[i...]]

Base.size(mesh::CuboidMesh, i...) = size(eachindex(mesh), i...)

Base.length(mesh::CuboidMesh) = prod(length.(mesh.ticks))

vertices(mesh::CuboidMesh) = (SVector(v) for v in Iterators.product(mesh.ticks...))

nvertices(mesh::CuboidMesh) = length(vertices(mesh))

neighbors(mesh::CuboidMesh, n::CartesianIndex) = marchingneighbors(eachindex(mesh), n)

neighbors_forward(mesh::CuboidMesh, n::CartesianIndex) = marchingneighbors_forward(eachindex(mesh), n)

# function neighbors(mesh::CuboidMesh, j::Int)
#     c = eachindex(mesh)
#     l = LinearIndices(c)
#     return (l[i] for i in neighbors(mesh, c[j]))
# end

marchingsimplices(m::CuboidMesh) = m.simpitr

unitperms(m::CuboidMesh) = m.simpitr.unitperms

unitsimplex(m::CuboidMesh, i) = m.simpitr.simps[i]