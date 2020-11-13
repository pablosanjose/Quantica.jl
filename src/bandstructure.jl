#######################################################################
# Spectrum
#######################################################################
struct Subspace{C,T,S<:AbstractMatrix{C}}
    energy::T
    basis::S
end

function Base.show(io::IO, s::Subspace{C,T}) where {C,T}
    i = get(io, :indent, "")
    print(io,
"$(i)Subspace{$C,$T}: eigenenergy subspace
$i  Energy       : $(s.energy)
$i  Degeneracy   : $(degeneracy(s))")
end

"""
    degeneracy(s::Subspace)

Return the degeneracy of a given energy subspace. It is equal to `size(s.basis, 2)`.

# See also
    `spectrum`, `bandstructure`
"""
degeneracy(s::Subspace) = size(s.basis, 2)

collect_subspaces(ϵs, ψs, ::Type{T}) where {T} = _collect_subspaces(ϵs, ψs, typeof(Subspace(zero(T), view(ψs, :, 1:1))))
_collect_subspaces(ϵs, ψs, ::Type{SS}) where {C,T,SS<:Subspace{C,T}} =
    SS[Subspace(_convert_energy(ϵs, rng, T), view(ψs, :, rng)) for rng in approxruns(ϵs)]

_convert_energy(ϵs, rng, ::Type{T}) where {T<:Real} = mean(i -> T(real(ϵs[i])), rng)
_convert_energy(ϵs, rng, ::Type{T}) where {T<:Complex} = mean(i -> T(ϵs[i]), rng)

Base.iterate(s::Subspace) = s.energy, Val(:basis)
Base.iterate(s::Subspace, ::Val{:basis}) = s.basis, Val(:done)
Base.iterate(::Subspace, ::Val{:done}) = nothing
Base.IteratorSize(::Subspace) = Base.HasLength()
Base.first(s::Subspace) = s.energy
Base.last(s::Subspace) = s.basis
Base.length(s::Subspace) = 2

struct Spectrum{C,T,E<:AbstractVector{T},A<:AbstractMatrix{C}}
    energies::E
    states::A
    subs::Vector{UnitRange{Int}}
    subs´::Vector{UnitRange{Int}}
end

Spectrum(energies, states, subs) = Spectrum(energies, states, subs, copy(subs))

function Base.show(io::IO, s::Spectrum{C,T}) where {C,T}
    i = get(io, :indent, "")
    print(io,
"$(i)Spectrum{$C,$T}: spectrum of a 0D Hamiltonian
$i  Orbital type : $C
$i  Energy type  : $T
$i  Energy range : $(extrema(real, s.energies))
$i  Eigenpairs   : $(length(s.energies))
$i  Subspaces    : $(length(s.subs))")
end

"""
    spectrum(h; method = LinearAlgebraPackage(), transform = missing)

Compute the spectrum of a 0D Hamiltonian `h` (or alternatively of the bounded unit cell of a
finite dimensional `h`) using one of the following `method`s

    method                    diagonalization function
    --------------------------------------------------------------
    LinearAlgebraPackage()     LinearAlgebra.eigen!
    ArpackPackage()            Arpack.eigs (must be `using Arpack`)

The option `transform = ε -> f(ε)` allows to transform eigenvalues by `f` in the returned
spectrum (useful for performing shifts or other postprocessing).

# Indexing

The eigenenergies `εv::Vector` and eigenstates `ψm::Matrix` in a `s::Spectrum` object can be
accessed via destructuring, `εv, ψm = sp`, or `εv, ψm = Tuple(sp)`, or `εv = first(sp) =
sp.energies, ψm = last(sp) = sp.states`. Any degenerate energies appear repeated in `εv`.
Alternatively, one can access one or more complete `sub::Subspace`s (eigenenergy together
with its eigenstates, including all degenerates) via the indexing syntax,

    s[1]                   : first `Subspace`
    s[2:4]                 : subspaces 2, 3 and 4
    s[[2,5,6]]             : subspaces 2, 5 and 6
    s[around = 0.2]        : single subspace with energy closest to 0.2
    s[around = (0.2, 10)]  : the ten subspaces with energies closest to 0.2

The eigenenergy `ε` and subspace basis `ψs` of a `sub::Subspace` can be obtained via
destructuring, `ε, ψs = sub`, or `ε = first(sub) = sub.energy, ψs = last(sub) = sub.basis`.
For performance reasons `ψs` is a `SubArray` view of the appropriate columns of `ψm`, not an
independent copy.

# See also
    `bandstructure`, `diagonalizer`
"""
function spectrum(h; method = LinearAlgebraPackage(), transform = missing)
    diag = diagonalizer(h; method = method)
    (ϵk, ψk) = diag(())
    subs = collect(approxruns(ϵk))
    s = Spectrum(ϵk, ψk, subs)
    transform === missing || transform!(transform, s)
    return s
end

Base.Tuple(s::Spectrum) = (s.energies, s.states)

"""
    transform!(f::Function, s::Spectrum)

Transform the energies of `s` by applying `f` to them in place.
"""
transform!(f, s::Spectrum) = (map!(f, s.energies, s.energies); s)

Base.iterate(s::Spectrum) = s.energies, Val(:states)
Base.iterate(s::Spectrum, ::Val{:states}) = s.states, Val(:done)
Base.iterate(::Spectrum, ::Val{:done}) = nothing
Base.first(s::Spectrum) = s.energies
Base.last(s::Spectrum) = s.states

_subspace(s::Spectrum, rngs) = _subspace.(Ref(s), rngs)

function _subspace(s::Spectrum, rng::AbstractUnitRange)
    ε = mean(j -> s.energies[j], rng)
    ψs = view(s.states, :, rng)
    Subspace(ε, ψs)
end

Base.getindex(s::Spectrum, i::Int) = _subspace(s, s.subs[i])
Base.getindex(s::Spectrum, is::Union{AbstractUnitRange,AbstractVector}) = getindex.(Ref(s), is)
Base.getindex(s::Spectrum; around) = get_around(s, around)

get_around(s::Spectrum, ε0::Number) = get_around(s, ε0, 1)
get_around(s::Spectrum, (ε0, n)::Tuple) = get_around(s, ε0, 1:n)

function get_around(s::Spectrum, ε0::Number, which)
    copy!(s.subs´, s.subs)
    rngs = partialsort!(s.subs´, which, by = rng -> abs(s.energies[first(rng)] - ε0))
    return _subspace(s, rngs)
end

######################################################################
# BandMesh
######################################################################
struct BandMesh{D´,T<:Number}  # D´ is dimension of BaseMesh space plus one (energy)
    verts::Vector{SVector{D´,T}}        # vertices of the band mesh
    degs::Vector{Int}                   # Vertex degeneracies
    adjmat::SparseMatrixCSC{Bool,Int}   # Undirected adjacency graph: both dest > src and dest < src
    sinds::Vector{NTuple{D´,Int}}       # indices of verts D´-tuples that form simplices
end

function Base.show(io::IO, mesh::BandMesh{D}) where {D}
    i = get(io, :indent, "")
    print(io,
"$(i)BandMesh{$D}: mesh of a $(D)D band in $(D-1)D parameter space
$i  Vertices   : $(nvertices(mesh))
$i  Edges      : $(nedges(mesh))
$i  Simplices  : $(nsimplices(mesh))")
end

nvertices(m::BandMesh) = length(m.verts)

nedges(m::BandMesh) = div(nnz(m.adjmat), 2)

nsimplices(m::BandMesh) = length(m.sinds)

vertices(m::BandMesh) = m.verts

edges(adjmat, src) = nzrange(adjmat, src)

edgedest(adjmat, edge) = rowvals(adjmat)[edge]

edgevertices(m::BandMesh) =
    ((vsrc, m.verts[edgedest(m.adjmat, edge)]) for (i, vsrc) in enumerate(m.verts) for edge in edges(m.adjmat, i))

degeneracy(m::BandMesh, i) = m.degs[i]

transform!(f::Function, m::BandMesh) = (map!(f, vertices(m), vertices(m)); m)

#######################################################################
# Bandstructure
#######################################################################
struct SimplexIndexer{D,T}
    basemesh::CuboidMesh{D,T}
    sptrs::Array{UnitRange{Int},D}  # range of indices of sverts and svecs for each simplex CartesianIndex in base mesh
end

struct Bandstructure{D,C,T,S<:AbstractMatrix{C},E,D´,B<:BandMesh{D´,T},M<:Diagonalizer}   # D is dimension of base mesh, D´ = D+1
    bands::Vector{B}                                # band meshes (vertices + adjacencies)
    sverts::Vector{NTuple{D´,SVector{D´,T}}}        # (base-coords..., energy) of each simplex vertex (groupings of bands.verts)
    sbases::Vector{NTuple{D´,S}}                    # basis on each simplex vertex, possibly degenerate
    sprojs::Vector{NTuple{D´,Matrix{E}}}            # projection of basis on each simplex vertex to interpolate
    indexers::Vector{SimplexIndexer{D,T}}           # provides ranges of simplices above corresponding to a given base-mesh minicuboid
    diag::M                                         # diagonalizer that can be used to add additional base-meshes for refinement
end

function Bandstructure(bands, sverts, sbases::Vector{NTuple{D´,S}}, indexers, diag) where {D´,C,S<:AbstractMatrix{C}}
    E = eltype(C)
    sprojs = Vector{NTuple{D´,Matrix{E}}}(undef, length(sbases))
    return Bandstructure(bands, sverts, sbases, sprojs, indexers, diag)
end

function Base.show(io::IO, bs::Bandstructure)
    i = get(io, :indent, "")
    ioindent = IOContext(io, :indent => string(i, "  "))
    print(io, i, summary(bs), "\n",
"$i  Bands         : $(length(bs.bands))
$i  Vertices      : $(nvertices(bs))
$i  Edges         : $(nedges(bs))
$i  Simplices     : $(nsimplices(bs))")
end

Base.summary(::Bandstructure{D}) where {D} =
    "Bandstructure{$D}: bands of a $(D)D Hamiltonian"

# API #
"""
    bandstructure(h::Hamiltonian; subticks = 13, kw...)

Compute `bandstructure(h, cuboid((-π,π)...; subticks = subticks); kw...)` using a base mesh
(of type `CuboidMesh`) over `h`'s full Brillouin zone with the specified `subticks` along
each [-π,π] reciprocal axis.

    bandstructure(h::Hamiltonian, nodes...; subticks = 13, kw...)

Create a linecut of a bandstructure of `h` along a polygonal line connecting two or more
`nodes`. Each node is either a `Tuple` or `SVector` of Bloch phases, or a symbolic name for
a Brillouin zone point (`:Γ`,`:K`, `:K´`, `:M`, `:X`, `:Y` or `:Z`). Each segment in the
polygon has the specified number of `subticks`. Different `subticks` per segments can be
specified with `subticks = (p1, p2...)`.

    bandstructure(h::Hamiltonian, mesh::CuboidMesh; mapping = missing, kw...)

Compute the bandstructure `bandstructure(h, mesh; kw...)` of Bloch Hamiltonian `bloch(h,
ϕ)`, with `ϕ = v` taken on each vertex `v` of the base `mesh` (or `ϕ = mapping(v...)` if a
`mapping` function is provided).

    bandstructure(ph::ParametricHamiltonian, ...; kw...)

Compute the bandstructure of a `ph`. Unless all parameters have default values, a `mapping`
is required between mesh vertices and Bloch/parameters for `ph`, see details on `mapping`
below.

    bandstructure(matrixf::Function, mesh::CuboidMesh; kw...)

Compute the bandstructure of the Hamiltonian matrix `m = matrixf(ϕ)`, with `ϕ` evaluated on
the vertices `v` of the `mesh`. Note that `ϕ` in `matrixf(ϕ)` is an unsplatted container.
Hence, i.e. `matrixf(x) = ...` or `matrixf(x, y) = ...` will not work. Use `matrixf((x,)) =
...`, `matrixf((x, y)) = ...` or matrixf(s::SVector) = ...` instead.

    h |> bandstructure([mesh,]; kw...)

Curried form of the above equivalent to `bandstructure(h[, mesh]; kw...)`.

# Options

The default options are

    (mapping = missing, minoverlap = 0.3, method = LinearAlgebraPackage(), transform = missing, showprogress = true)

`mapping`: when not `missing`, `mapping = v -> p` is a function that map base mesh vertices
`v` to Bloch phases and/or parameters `p`. The structure of `p` is whatever is accepted by
`bloch(h, p, ...)` (see `bloch`). For `h::Hamiltonian`, `p = ϕs::Union{Tuple,SVector}` are
Bloch phases. For `h::ParametricHamiltonian`, `p = (ϕs..., (; ps))` or `p = (ϕs, (; ps))`
combine Bloch phases `ϕs` and keyword parameters `ps` of `ph`. This allows to compute a
bandstructure along a cut in the Brillouin zone/parameter space of `ph`, see examples below.

The option `minoverlap` determines the minimum overlap between eigenstates to connect
them into a common subband.

`method`: it is chosen automatically if unspecified, and can be one of the following

    method                     diagonalization function
    --------------------------------------------------------------
    LinearAlgebraPackage()     LinearAlgebra.eigen!
    ArpackPackage()            Arpack.eigs (must be `using Arpack`)

Options passed to the `method` will be forwarded to the diagonalization function. For example,
`method = ArpackPackage(nev = 8, sigma = 1im)` will use `Arpack.eigs(matrix; nev = 8,
sigma = 1im)` to compute the bandstructure.

`transform`: the option `transform = ε -> fε(ε)` allows to transform eigenvalues by `fε` in
the returned bandstructure (useful for performing shifts or other postprocessing). We can
also do `transform -> (fφ, fε)` to transform also mesh vertices with fφ. Additionally,
`transform -> isometric` or `transform -> (isometric, fε)` will transform mesh vertices into
momenta, assuming they represent Bloch phases. This works both in full bandstructures and
linecuts.

`showprogress`: indicate whether progress bars are displayed during the calculation

# Indexing

The bands in a `bs::Bandstructure` object can be accessed with `bands`, while the indexing
syntax `bs[(φs...)]` gives access to one or more `sub::Subspace` objects, contructed by
linear interpolation of each band at base-mesh coordinates `φs`.

    bs[(φs...), 1]                  : first interpolated subspaces at base mesh coordinates `φs`, ordered by energy
    bs[(φs...), 1:3]                : interpolated subspaces 1 to 3 at base mesh coordinates `φs`, ordered by energy
    bs[(φs...)]                     : interpolated subspaces at base mesh coordinates `φs` in any band
    bs[(φs...), around = 0.2]       : the single interpolated subspaces at `φs` with energies closest to 0.2
    bs[(φs...), around = (0.2, 10)] : the ten interpolated subspaces at `φs` with energies closest to 0.2

The eigenenergy `ε` and subspace basis `ψs` of a `sub::Subspace` can themselves be obtained
via destructuring, `ε, ψs = sub`, or `ε = first(sub), ψs = last(sub)`.

# Examples
```jldoctest
julia> h = LatticePresets.honeycomb() |> hamiltonian(hopping(-1)) |> unitcell(3);

julia> bandstructure(h; subticks = 25, method = LinearAlgebraPackage())
Bandstructure{2}: collection of 2D bands
  Bands        : 8
  Element type : scalar (Complex{Float64})
  BandMesh{2}: mesh of a 2-dimensional manifold
    Vertices   : 625
    Edges      : 1776

julia> bandstructure(h, :Γ, :X, :Y, :Γ; subticks = (10,15,10))
Bandstructure{2}: collection of 1D bands
  Bands        : 18
  Element type : scalar (Complex{Float64})
  BandMesh{1}: mesh of a 1-dimensional manifold
    Vertices   : 33
    Edges      : 32

julia> bandstructure(h, mesh((0, 2π); subticks = 13); mapping = φ -> (φ, 0))
       # Equivalent to bandstructure(h, :Γ, :X; subticks = 13)
Bandstructure{2}: collection of 1D bands
  Bands        : 18
  Element type : scalar (Complex{Float64})
  BandMesh{1}: mesh of a 1-dimensional manifold
    Vertices   : 11
    Edges      : 10

julia> ph = parametric(h, @hopping!((t; α) -> t * α));

julia> bandstructure(ph, mesh((0, 2π); subticks = 13); mapping = φ -> (φ, 0, (; α = 2φ)))
Bandstructure{2}: collection of 1D bands
  Bands        : 18
  Element type : scalar (Complex{Float64})
  BandMesh{1}: mesh of a 1-dimensional manifold
    Vertices   : 11
    Edges      : 10
```

# See also
    `cuboid`, `diagonalizer`, `bloch`, `parametric`
"""
bandstructure

nvertices(bs::Bandstructure) = isempty(bands(bs)) ? 0 : sum(nvertices, bands(bs))

nedges(bs::Bandstructure) = isempty(bands(bs)) ? 0 : sum(nedges, bands(bs))

nsimplices(bs::Bandstructure) = isempty(bands(bs)) ? 0 : sum(nsimplices, bands(bs))

nbands(bs::Bandstructure) = length(bands(bs))

"""
    bands(bs::Bandstructure[, i])

Return a `bands::Vector{BandMesh}` of all the bands in `bs`, or `bands[i]` if `i` is given.

# See also
    `bandstructure`
"""
bands(bs::Bandstructure) = bs.bands
bands(bs::Bandstructure, i) = bs.bands[i]

"""
    vertices(bs::Bandstructure, i)

Return the vertices `(k..., ϵ)` of the i-th band in `bs`, in the form of a
`Vector{SVector{L+1}}`, where `L` is the lattice dimension.
"""
vertices(bs::Bandstructure, i) = vertices(bands(bs, i))

"""
    transform!(f::Function, b::Bandstructure)

Transform the energies of all bands in `b` by applying `f` to them in place.

    transform!((fk, fε), b::Bandstructure)

Transform Bloch phases and energies of all bands in `b` by applying `fk` and `fε` to them in
place, respectively. If any of them is `missing`, it will be ignored.

"""
transform!(f, bs::Bandstructure) = transform!(sanitize_transform(f), bs)

function transform!((fk, fε)::Tuple{Function,Function}, bs::Bandstructure)
    for band in bands(bs)
        vs = vertices(band)
        for (i, v) in enumerate(vs)
            vs[i] = SVector((fk(SVector(Base.front(Tuple(v))))..., fε(last(v))))
        end
        alignnormals!(band.sinds, vs)
    end
    return bs
end

#######################################################################
# isometric and Brillouin zone points
#######################################################################
function isometric(h::Hamiltonian)
    r = qr(bravais(h)).R
    r = r * sign(r[1,1])
    ibr = inv(r')
    return ϕs -> ibr * ϕs
end

isometric(h::Hamiltonian{<:Any,L}, nodes) where {L} = _isometric(h, parsenode.(nodes, Val(L)))

_isometric(h, pts::Tuple) = _isometric(h, [pts...])

function _isometric(h, pts::Vector)
    br = bravais(h)
    pts´ = map(p -> br * p, pts)
    pathlength = pushfirst!(cumsum(norm.(diff(pts))), 0.0)
    isometric = piecewise_mapping(pathlength)
    return isometric
end

nodeindices(nodes::NTuple{N,Any}) where {N} = ntuple(i -> i-1, Val(N))

piecewise_mapping(nodes, ::Val{N}) where {N} = piecewise_mapping(parsenode.(nodes, Val(N)))

function piecewise_mapping(pts)
    N = length(pts) # could be a Tuple or a different container
    mapping = x -> begin
        x´ = clamp(only(x), 0, N-1)
        i = min(floor(Int, x´), N-2) + 1
        p = pts[i] + (x´ - i + 1) * (pts[i+1] - pts[i])
        return p
    end
    return mapping
end

parsenode(pt::SVector, ::Val{L}) where {L} = padright(pt, Val(L))
parsenode(pt::Tuple, val) = parsenode(SVector(float.(pt)), val)

function parsenode(node::Symbol, val)
    pt = get(BZpoints, node, missing)
    pt === missing && throw(ArgumentError("Unknown Brillouin zone point $pt, use one of $(keys(BZpoints))"))
    pt´ = parsenode(pt, val)
    return pt´
end

const BZpoints =
    ( Γ  = (0,)
    , X  = (pi,)
    , Y  = (0, pi)
    , Z  = (0, 0, pi)
    , K  = (2pi/3, -2pi/3)
    , K´ = (4pi/3, 2pi/3)
    , M  = (pi, 0)
    )

#######################################################################
# bandstructure building
#######################################################################
function bandstructure(h::Hamiltonian{<:Any, L}; subticks = 13, kw...) where {L}
    base = cuboid(filltuple((-π, π), Val(L))...; subticks = subticks)
    return bandstructure(h, base; kw...)
end

function bandstructure(h::Hamiltonian{<:Any,L}, node1, node2, nodes...; subticks = 13, transform = missing, kw...) where {L}
    allnodes = (node1, node2, nodes...)
    mapping´ = piecewise_mapping(allnodes, Val(L))
    base = cuboid(nodeindices(allnodes); subticks = subticks)
    transform´ = sanitize_transform(transform, h, allnodes)
    return bandstructure(h, base; mapping = mapping´, transform = transform´, kw...)
end

function bandstructure(h::Union{Hamiltonian,ParametricHamiltonian}, basemesh::CuboidMesh;
                       transform = missing, showprogress = true, kw...)
    diag = diagonalizer(h; kw...)
    b = bandstructure(diag, basemesh, showprogress)
    if transform !== missing
        transform´ = sanitize_transform(transform, h)
        transform!(transform´, b)
    end
    return b
end

function bandstructure(matrixf::Function, basemesh::CuboidMesh;
                       mapping = missing, transform = missing, showprogress = true)
    matrixf´ = wrapmapping(mapping, matrixf)
    dimh = size(samplematrix(matrixf´, basemesh), 1)
    diag = diagonalizer(matrixf´, dimh)
    b = bandstructure(diag, basemesh, showprogress)
    transform === missing || transform!(transform, b)
    return b
end
@inline map_phiparams(mapping::Missing, basevertex) = sanitize_phiparams(basevertex)
@inline map_phiparams(mapping::Function, basevertex) = sanitize_phiparams(mapping(basevertex...))

wrapmapping(mapping::Missing, matrixf::Function) = matrixf
wrapmapping(mapping::Function, matrixf::Function) = basevertex -> matrixf(toSVector(mapping(basevertex...)))

sanitize_transform(::Missing, args...) = (identity, identity)
sanitize_transform(f::Function, args...) = (identity, f)
sanitize_transform(f::typeof(isometric), args...) = (isometric(args...), identity)
sanitize_transform((_,f)::Tuple{typeof(isometric),Function}, args...) = (isometric(args...), f)
sanitize_transform(fs::Tuple{Function,Function}, args...) = fs
sanitize_transform((_,f)::Tuple{Missing,Function}, args...) = (identity, f)
sanitize_transform((f,_)::Tuple{Function,Missing}, args...) = (f, identity)

samplematrix(matrixf, basemesh) = matrixf(Tuple(first(vertices(basemesh))))

function bandstructure(diag::Diagonalizer, basemesh::CuboidMesh, showprogress)
    # Step 1/2 - Diagonalising:
    subspaces = bandstructure_diagonalize(diag, basemesh, showprogress)
    # Step 2/2 - Knitting bands:
    bands, cuboidinds, linearinds = bandstructure_knit(diag, basemesh, subspaces, showprogress)

    sverts, sbases, sptrs = bandstructure_collect(subspaces, bands, cuboidinds)

    indexers = [SimplexIndexer(basemesh, sptrs)]

    return Bandstructure(bands, sverts, sbases, indexers, diag)
end

#######################################################################
# bandstructure_diagonalize
#######################################################################
function bandstructure_diagonalize(diag, basemesh::CuboidMesh, showprogress = false)
    prog = Progress(length(basemesh), "Step 1/2 - Diagonalising: ")
    subspaces = [build_subspaces(diag, vertex, showprogress, prog) for vertex in vertices(basemesh)]
    return subspaces
end

function build_subspaces(diag::Diagonalizer, vertex::SVector{E,T}, showprog, prog) where {E,T}
    (ϵs, ψs) = diag(Tuple(vertex))
    subspaces = collect_subspaces(ϵs, ψs, T)
    showprog && ProgressMeter.next!(prog; showvalues = ())
    return subspaces
end

#######################################################################
# bandstructure_knit
#######################################################################
struct BandLinearIndex
    bandidx::Int
    vertidx::Int
end

Base.zero(::BandLinearIndex) = zero(BandLinearIndex)
Base.zero(::Type{BandLinearIndex}) = BandLinearIndex(0, 0)

Base.iszero(b::BandLinearIndex) = iszero(b.bandidx)

struct BandCuboidIndex{D}
    baseidx::CartesianIndex{D}
    colidx::Int
end

Base.Tuple(ci::BandCuboidIndex) = (Tuple(ci.baseidx)..., ci.colidx)

function bandstructure_knit(diag, basemesh::CuboidMesh{D,T}, subspaces::Array{Vector{S},D}, showprog = false) where {D,T,C,S<:Subspace{C}}
    nverts = sum(length, subspaces)
    prog = Progress(nverts, "Step 2/2 - Knitting bands: ")

    bands = BandMesh{D+1,T}[]
    pending = Tuple{BandCuboidIndex{D},BandCuboidIndex{D}}[]   # pairs of neighboring vertex indices src::IT, dst::IT
    linearinds = [zeros(BandLinearIndex, length(ss)) for ss in subspaces] # 0 == unclassified, > 0 vertex index
    cuboidinds = BandCuboidIndex{D}[]                          # cuboid indices of processed vertices
    I = Int[]; J = Int[]                                       # To build adjacency matrices
    P = real(eltype(C))                                        # type of projections between states
    maxsubs = maximum(length, subspaces)
    projinds = Vector{Tuple{P,Int}}(undef, maxsubs)            # Reusable list of projections for sorting

    bandidx = 0
    while true
        bandidx += 1
        seedidx = next_unprocessed(linearinds, subspaces)
        seedidx === nothing && break
        resize!(pending, 1)
        resize!(I, 0)
        resize!(J, 0)
        pending[1] = (seedidx, seedidx) # source CartesianIndex for band search, with no originating vertex
        bandmesh = knit_band(bandidx, basemesh, subspaces, diag.minoverlap, pending, cuboidinds, linearinds, I, J, projinds, showprog, prog)
        iszero(nsimplices(bandmesh)) || push!(bands, bandmesh)
    end

    return bands, cuboidinds, linearinds
end

function next_unprocessed(linearinds, subspaces)
    ci = CartesianIndices(linearinds)
    @inbounds for (n, vs) in enumerate(linearinds), i in eachindex(subspaces[n])
        iszero(vs[i]) && return BandCuboidIndex(ci[n], i)
    end
    return nothing
end

function knit_band(bandidx, basemesh::CuboidMesh{D,T}, subspaces, minoverlap, pending, cuboidinds, linearinds, I, J, projinds, showprog, prog) where {D,T}
    verts = SVector{D+1,T}[]
    degs = Int[]
    vertcounter = 0
    while !isempty(pending)
        src, dst = pop!(pending)
        n, i     = dst.baseidx, dst.colidx
        n0, i0   = src.baseidx, src.colidx
        # process dst only if unclassified (otherwise simply link)
        if !iszero(linearinds[n][i])
            append_adjacent!(I, J, linearinds[n0][i0], linearinds[n][i])
            continue
        end

        vert = vcat(vertex(basemesh, n), SVector(subspaces[n][i].energy))
        push!(verts, vert)
        push!(degs, degeneracy(subspaces[n][i]))
        push!(cuboidinds, dst)
        vertcounter += 1
        linearinds[n][i] = BandLinearIndex(bandidx, vertcounter)
        src == dst || append_adjacent!(I, J, linearinds[n0][i0], linearinds[n][i])
        showprog && ProgressMeter.next!(prog; showvalues = ())

        subdst = subspaces[n][i]
        deg = degeneracy(subdst)
        for n´ in neighbors(basemesh, n)
            deg == 1 && n´ == n0 && continue  # Only if deg == 1 is no-backstep justified (think deg at BZ boundary)
            sorted_valid_projections!(projinds, subspaces[n´], subdst, minoverlap, bandidx, linearinds[n´])
            cumdeg´ = 0
            for (p, i´) in projinds
                i´ == 0 && break
                push!(pending, (dst, BandCuboidIndex(n´, i´)))
                cumdeg´ += degeneracy(subspaces[n´][i´])
                cumdeg´ >= deg && break # links on each column n´ = cumulated deg at most equal to deg links
            end
        end
    end

    adjmat = sparse(I, J, true)

    sinds = band_simplices(verts, adjmat)

    return BandMesh(verts, degs, adjmat, sinds)
end

function append_adjacent!(I, J, msrc, mdst)
    append!(I, (mdst.vertidx, msrc.vertidx))
    append!(J, (msrc.vertidx, mdst.vertidx))
    return nothing
end

function sorted_valid_projections!(projinds, subs::Vector{<:Subspace}, sub0::Subspace{C}, minoverlap, bandidx, linearindscol) where {C} 
    nsubs = length(subs)
    realzero = zero(real(eltype(C)))
    complexzero = zero(eltype(C))
    fill!(projinds, (realzero, 0))
    for (j, sub) in enumerate(subs)
        bandidx´ = linearindscol[j].bandidx
        bandidx´ == 0 || bandidx´ == bandidx || continue
        p = proj_squared(sub.basis, sub0.basis, realzero, complexzero)
        p > minoverlap && (projinds[j] = (p, j))
    end
    sort!(projinds, rev = true, alg = Base.DEFAULT_UNSTABLE)
    return projinds
end

# non-allocating version of `sum(abs2, ψ' * ψ0)`
function proj_squared(ψ, ψ0, realzero, complexzero)
    size(ψ, 1) == size(ψ0, 1) || throw(error("Internal error: eigenstates of different sizes"))
    p = realzero
    # project the smaller-dim subspace onto the larger-dim subspace
    for j0 in axes(ψ0, 2), j in axes(ψ, 2)
        p0 = complexzero
        @simd for i0 in axes(ψ0, 1)
            @inbounds p0 += dot(ψ[i0,j], ψ0[i0,j0])
        end
        p += abs2(p0)
    end
    return p
end

######################################################################
# Simplices
######################################################################
function band_simplices(vertices::Vector{SVector{D´,T}}, adjmat)  where {D´,T}
    D´ > 0 || throw(ArgumentError("Need a positive number of simplex vertices"))
    nverts = length(vertices)
    D´ == 1 && return Tuple.(1:nverts)
    sinds = NTuple{D´,Int}[]
    if nverts >= D´
        buffer = (NTuple{D´,Int}[], NTuple{D´,Int}[])
        for srcind in eachindex(vertices)
            newsinds = vertex_simplices!(buffer, adjmat, srcind)
            D´ > 2 && alignnormals!(newsinds, vertices)
            append!(sinds, newsinds)
        end
    end
    return sinds
end

# Add (greater) neighbors to last vertex of partials that are also neighbors of all members of partial, till N
function vertex_simplices!(buffer::Tuple{P,P}, adjmat, srcind) where {D´,P<:AbstractArray{<:NTuple{D´}}}
    partials, partials´ = buffer
    resize!(partials, 0)
    push!(partials, padright((srcind,), Val(D´)))
    for pass in 2:D´
        resize!(partials´, 0)
        for partial in partials
            nextsrc = partial[pass - 1]
            for edge in edges(adjmat, nextsrc), neigh in edgedest(adjmat, edge)
                valid = neigh > nextsrc && isconnected(neigh, partial, adjmat)
                valid || continue
                newinds = tuplesplice(partial, pass, neigh)
                push!(partials´, newinds)
            end
        end
        partials, partials´ = partials´, partials
    end
    return partials
end

# equivalent to all(n -> n in neighbors(adjmat, neigh), partial)
function isconnected(neigh, partial, adjmat)
    connected = all(partial) do ind
        ind == 0 && return true
        for edge in edges(adjmat, neigh), neigh´ in edgedest(adjmat, edge)
            ind == neigh´ && return true
        end
        return false
    end
    return connected
end

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
# bandstructure_collect
######################################################################
function bandstructure_collect(subspaces::Array{Vector{Subspace{C,T,S}},D}, bands, cuboidinds) where {C,T,S,D}
    nsimps = isempty(bands) ? 0 : sum(nsimplices, bands)
    sverts = Vector{NTuple{D+1,SVector{D+1,T}}}(undef, nsimps)
    sbases = Vector{NTuple{D+1,S}}(undef, nsimps)
    sptrs = fill(1:0, size(subspaces) .- 1)                  # assuming non-periodic basemesh
    s0inds = Vector{CartesianIndex{D}}(undef, nsimps)    # base cuboid index for reference vertex in simplex, for sorting

    scounter = 0
    ioffset = 0
    for band in bands
        for s in band.sinds
            scounter += 1
            let ioffset = ioffset  # circumvent boxing, JuliaLang/#15276
                baseinds = (i -> cuboidinds[ioffset + i].baseidx).(s)
                pbase = sortperm(SVector(baseinds))
                s0inds[scounter] = baseinds[first(pbase)] # equivalent to minimum(baseinds)
                sverts[scounter] = ntuple(i -> band.verts[s[pbase[i]]], Val(D+1))
                sbases[scounter] = ntuple(Val(D+1)) do i
                    c = cuboidinds[ioffset + s[pbase[i]]]
                    subspaces[c.baseidx][c.colidx].basis
                end
            end
        end
        ioffset += nvertices(band)
    end

    psimps = sortperm(s0inds; alg = Base.DEFAULT_UNSTABLE)
    permute!(s0inds, psimps)
    permute!(sverts, psimps)
    permute!(sbases, psimps)

    for rng in equalruns(s0inds)
        sptrs[s0inds[first(rng)]] = rng
    end

    return sverts, sbases, sptrs
end

#######################################################################
# Bandstructure indexing
#######################################################################
Base.getindex(bs::Bandstructure, ϕs::Tuple; around = missing) = interpolate_bandstructure(bs, ϕs, around)

function interpolate_bandstructure(bs::Bandstructure{D,C,T}, ϕs, around) where {D,C,T}
    D == length(ϕs) || throw(ArgumentError("Bandstructure needs a NTuple{$D} of base coordinates for interpolation"))
    found = false
    inds = filltuple(0, Val(D))
    indexer = first(bs.indexers)
    for outer indexer in bs.indexers
        inds = find_basemesh_interval.(ϕs, indexer.basemesh.ticks)
        found = !any(iszero, inds)
        found && break
    end
    found || throw(ArgumentError("Cannot interpolate $ϕs within any of the bandstructure's base meshes"))
    rng = indexer.sptrs[CartesianIndex(inds)]
    subs = Subspace{C,T,Matrix{C}}[]
    # Since we have sorted simplices to canonical base vertex order in bandstructure_collect
    # we can avoid unncecessary simplices (and double matches with ϕs at a simplex boundary)
    # by demanding equal vertices
    simplexbase = find_basemesh_simplex(ϕs, bs, rng)
    if simplexbase !== nothing
        basevertices, dϕinds = simplexbase
        for i in rng
            sverts = bs.sverts[i]
            basevertices´ = frontSVector.(sverts)
            basevertices´ == basevertices || continue
            sbases = bs.sbases[i]
            sprojs = get_or_add_projection_basis!(bs, i)
            push_interpolated_subspace!(subs, dϕinds, sverts, sbases, sprojs)
        end
    end
    subs´ = filter_around!(subs, around)
    return subs´
end

function find_basemesh_interval(ϕ, ticks)
    @inbounds for m = 1:length(ticks)-1
        ticks[m] <= ϕ < ticks[m+1] && return m
    end
    ϕ ≈ last(ticks) && return length(ticks) - 1
    return 0
end

function find_basemesh_simplex(ϕs, bs, rng)
    for i in rng
        basevertices = frontSVector.(bs.sverts[i])
        dϕinds = normalized_base_inds(ϕs, basevertices)
        insimplex(dϕinds) && return basevertices, dϕinds
    end
    return nothing
end

function normalized_base_inds(ϕs, baseverts)
    dbase = tuple_minus_first(baseverts)
    smat = hcat(dbase...)
    dϕvec = SVector(ϕs) - first(baseverts)
    dϕinds = inv(smat) * dϕvec
    return dϕinds
end

insimplex(dϕinds) = sum(dϕinds) <= 1 && all(i -> 0 <= i <= 1, dϕinds)

add_projection_bases!(bs) = foreach(i -> get_or_add_projection_basis!(bs, i), eachindex(bs.sprojs))

function get_or_add_projection_basis!(bs, i)
    isassigned(bs.sprojs, i) && return bs.sprojs[i]
    bases = bs.sbases[i]
    firstbasis = first(bases)
    projbasis = firstbasis
    for basis in Base.tail(bases)
        size(basis, 2) < size(projbasis, 2) && (projbasis = basis)
    end
    sprojs = projection_basis.(bases, Ref(projbasis))
    bs.sprojs[i] = sprojs
    return sprojs
end

# Projects src onto dst, and finds an orthonormal basis of the projection subspace
function projection_basis(dst, src)
    t = dst' * src  # n´ x n matrix, so that dst * t spans the projection subspace
    src === dst && return t  # assumes orthonormal src == dst
    n = size(src, 2)
    n´ = size(dst, 2)
    @inbounds for j in 1:n
        col = view(t, :, j)
        for j´ in 1:j-1
            col´ = view(t, :, j´)
            r = dot(col´, col)/dot(col´, col´)
            col .-= r .* col´
        end
        normalize!(col)
    end
    return t
end

function push_interpolated_subspace!(subs, dϕinds, verts, bases, projs)
    ϵs = last.(verts)
    dϵs = SVector(tuple_minus_first(ϵs))
    ϵ0 = first(ϵs)
    energy = ϵ0 + dot(dϕinds, dϵs)
    basis = interpolate_subspace_basis(dϕinds, bases, projs)
    push!(subs, Subspace(energy, basis))
    return nothing
end

function interpolate_subspace_basis(dϕinds, bases, projs)
    ibasis = first(bases) * first(projs)
    ibasis .*= (1 - sum(dϕinds))
    for i in 2:length(bases)
        mul!(ibasis, bases[i], projs[i], dϕinds[i-1], 1)
    end
    return ibasis
end

filter_around!(ss, ::Missing) = sort!(ss; by = s -> s.energy, alg = Base.DEFAULT_UNSTABLE)
filter_around!(ss, ε0::Number) = filter_around!(ss, ε0, 1)

function filter_around!(ss, (ε0, n)::Tuple)
    0 <= n <= length(ss) || throw(ArgumentError("Cannot retrieve more than $(length(ss)) subspaces"))
    return filter_around!(ss, ε0, 1:n)
end

filter_around!(ss::Vector{<:Subspace}, ε0, which) = partialsort!(ss, which; by = s -> abs(s.energy - ε0))