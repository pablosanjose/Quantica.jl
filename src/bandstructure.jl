#######################################################################
# Subspace
#######################################################################
struct Subspace{D,C,T,S<:AbstractMatrix{C},O<:OrbitalStructure}
    energy::T
    basis::S
    orbstruct::O
    basevert::SVector{D,T}
end
Subspace(h::Hamiltonian, args...) = Subspace(h.orbstruct, args...)
Subspace(::Missing, energy, basis, basevert...) =
    Subspace(OrbitalStructure(eltype(basis), size(basis, 1)), energy, basis, basevert...)
Subspace(o::OrbitalStructure, energy, basis, basevert...) =
    Subspace(energy, unflatten_orbitals_or_reinterpret(basis, o), o, basevert...)
Subspace(energy::T, basis, orbstruct) where {T} = Subspace(energy, basis, orbstruct, SVector{0,T}())
Subspace(energy::T, basis, orbstruct, basevert) where {T} = Subspace(energy, basis, orbstruct, SVector(T.(basevert)))

function Base.show(io::IO, s::Subspace{D,C,T}) where {D,C,T}
    i = get(io, :indent, "")
    print(io,
"$(i)Subspace{$D}: eigenenergy subspace on a $(D)D manifold
$i  Energy       : $(s.energy)
$i  Degeneracy   : $(degeneracy(s))
$i  Bloch/params : $(s.basevert)
$i  Basis eltype : $C")
end

"""
    degeneracy(s::Subspace)

Return the degeneracy of a given energy subspace. It is equal to `size(s.basis, 2)`.

# See also
    `spectrum`, `bandstructure`
"""
degeneracy(s::Subspace) = degeneracy(s.basis)
degeneracy(m::AbstractMatrix) = isempty(m) ? 1 : size(m, 2)  # To support sentinel empty projs

orbitalstructure(s::Subspace) = s.orbstruct

flatten(s::Subspace) =
    Subspace(s.energy, flatten(s.basis, s.orbstruct), s.orbstruct, s.basevert)

unflatten(s::Subspace, o::OrbitalStructure) = Subspace(s.energy, unflatten_orbitals(s.basis, o), o, s.basevert)

# destructuring
Base.iterate(s::Subspace) = s.energy, Val(:basis)
Base.iterate(s::Subspace, ::Val{:basis}) = s.basis, Val(:done)
Base.iterate(::Subspace, ::Val{:done}) = nothing
Base.IteratorSize(::Subspace) = Base.HasLength()
Base.first(s::Subspace) = s.energy
Base.last(s::Subspace) = s.basis
Base.length(s::Subspace) = 2

Base.copy(s::Subspace) = deepcopy(s)

#######################################################################
# Spectrum
#######################################################################
struct Spectrum{D,C,T,S<:AbstractMatrix{C},E<:AbstractVector{T},M<:Diagonalizer}
    energies::E
    states::S
    diag::M
    basevert::SVector{D,T}
    subs::Vector{UnitRange{Int}}
end

Spectrum(energies::AbstractVector{T}, states, diag) where {T} = Spectrum(energies, states, diag, zero(SVector{0,real(T)}))

function Spectrum(energies, states, diag, basevert::SVector{<:Any,T}) where {T}
    energies´ = maybereal(energies, T)
    subs = collect(approxruns(energies´))
    for rng in subs
        orthonormalize!(view(states, :, rng))
    end
    return Spectrum(energies´, states, diag, basevert, subs)
end

maybereal(energies, ::Type{T}) where {T<:Real} = T.(real.(energies))
maybereal(energies, ::Type{T}) where {T<:Complex} = T.(energies)

function Base.show(io::IO, s::Spectrum{D,C,T}) where {D,C,T}
    i = get(io, :indent, "")
    print(io,
"$(i)Spectrum{$D,$C,$T}: spectrum of a $(D)D Hamiltonian
$i  Orbital type : $C
$i  Energy type  : $T
$i  Energy range : $(extrema(real, s.energies))
$i  Bloch/params : $(Tuple(s.basevert))
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
    (ϵk, ψk) = diag((), NoUnflatten())
    s = Spectrum(ϵk, ψk, diag)
    transform === missing || transform!(transform, s)
    return s
end

"""
    transform!(f::Function, s::Spectrum)

Transform the energies of `s` by applying `f` to them in place.
"""
transform!(f, s::Spectrum) = (map!(f, s.energies, s.energies); s)

# destructuring
Base.first(s::Spectrum) = s.energies
Base.last(s::Spectrum) = s.states
Base.iterate(s::Spectrum) = first(s), Val(:states)
Base.iterate(s::Spectrum, ::Val{:states}) = last(s), Val(:done)
Base.iterate(::Spectrum, ::Val{:done}) = nothing
Base.Tuple(s::Spectrum) = (first(s), last(s))

Base.getindex(s::Spectrum, i::Int) = subspace(s, s.subs[i])
Base.getindex(s::Spectrum, is::Union{AbstractUnitRange,AbstractVector}) = getindex.(Ref(s), is)
Base.getindex(s::Spectrum; around) = get_around(s, around)

get_around(s::Spectrum, ε0::Number) = get_around(s, ε0, 1)
get_around(s::Spectrum, (ε0, n)::Tuple) = get_around(s, ε0, 1:n)

function get_around(s::Spectrum, ε0::Number, which)
    rngs = partialsort(s.subs, which, by = rng -> abs(s.energies[first(rng)] - ε0))
    return subspace(s, rngs)
end

subspace(s::Spectrum, rngs) = subspace.(Ref(s), rngs)

function subspace(s::Spectrum, rng::AbstractUnitRange)
    ε = mean(j -> s.energies[j], rng)
    ψs = view(s.states, :, rng)
    return Subspace(s.diag.orbstruct, ε, ψs)
end

nsubspaces(s::Spectrum) = length(s.subs)

basis_slice_type(s::Spectrum) = typeof(view(s.states, :, 1:0))
basis_block_type(s::Spectrum) = typeof(view(s.states, 1:0, 1:0))

######################################################################
# Band
######################################################################
struct Band{D,C,T,S<:AbstractMatrix{C},S´<:AbstractMatrix,D´}  # D´ is dimension of BaseMesh space plus one (energy)
    basemesh::CuboidMesh{D,T,D´}
    verts::Vector{SVector{D´,T}}
    vbases::Vector{S}
    vptrs::Array{UnitRange{Int},D}
    adjmat::SparseMatrixCSC{Bool,Int}
    simps::Vector{NTuple{D´,Int}}
    sbases::Vector{NTuple{D´,S´}}
    sptrs::Array{UnitRange{Int},D´}
end

Band(basemesh::CuboidMesh{D,T}, verts::Vector{SVector{D´,T´}}, args...) where {D,D´,T,T´} =
    Band(CuboidMesh{D,T´}(basemesh), verts, args...)

function Base.show(io::IO, b::Band{D}) where {D}
    i = get(io, :indent, "")
    print(io,
"$(i)Band{$D}: a band over a $(D)D parameter cuboid
$i  Vertices   : $(nvertices(b))
$i  Edges      : $(nedges(b))
$i  Simplices  : $(nsimplices(b))")
end

nvertices(m::Band) = length(m.verts)

nedges(m::Band) = div(nnz(m.adjmat), 2)

nsimplices(m::Band) = length(m.simps)

vertices(m::Band) = m.verts

edges(adjmat, src) = nzrange(adjmat, src)

edgedest(adjmat, edge) = rowvals(adjmat)[edge]

edgevertices(b::Band) =
    ((vsrc, b.verts[edgedest(b.adjmat, edge)]) for (i, vsrc) in enumerate(b.verts) for edge in edges(b.adjmat, i))

degeneracy(m::Band, i) = degeneracy(m.vbases[i])

transform!(f::Function, m::Band) = (map!(f, vertices(m), vertices(m)); m)

Base.getindex(b::Band, i::Int) = b.verts[i], b.vbases[i]

#######################################################################
# Bandstructure
#######################################################################
struct Bandstructure{D,C,T,B<:Band{D,C,T},M<:Diagonalizer}   # D is dimension of base mesh, D´ = D+1
    bands::Vector{B}  # band meshes (vertices + adjacencies)
    diag::M           # diagonalizer that can be used to add additional base-meshes for refinement
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

nvertices(bs::Bandstructure) = isempty(bands(bs)) ? 0 : sum(nvertices, bands(bs))

nedges(bs::Bandstructure) = isempty(bands(bs)) ? 0 : sum(nedges, bands(bs))

nsimplices(bs::Bandstructure) = isempty(bands(bs)) ? 0 : sum(nsimplices, bands(bs))

nbands(bs::Bandstructure) = length(bands(bs))

"""
    bands(bs::Bandstructure[, i])

Return a `bands::Vector{Band}` of all the bands in `bs`, or `bands[i]` if `i` is given.

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
transform!(f, bs::Bandstructure) = transform!(sanitize_band_transform(f), bs)

function transform!((fk, fε)::Tuple{Function,Function}, bs::Bandstructure)
    for band in bands(bs)
        vs = vertices(band)
        for (i, v) in enumerate(vs)
            vs[i] = SVector((fk(SVector(Base.front(Tuple(v))))..., fε(last(v))))
        end
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
# bandstructure API
#######################################################################
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

    (mapping = missing, method = LinearAlgebraPackage(), transform = missing, splitbands = true, showprogress = true)

`mapping`: when not `missing`, `mapping = v -> p` is a function that map base mesh vertices
`v` to Bloch phases and/or parameters `p`. The structure of `p` is whatever is accepted by
`bloch(h, p, ...)` (see `bloch`). For `h::Hamiltonian`, `p = ϕs::Union{Tuple,SVector}` are
Bloch phases. For `h::ParametricHamiltonian`, `p = (ϕs..., (; ps))` or `p = (ϕs, (; ps))`
combine Bloch phases `ϕs` and keyword parameters `ps` of `ph`. This allows to compute a
bandstructure along a cut in the Brillouin zone/parameter space of `ph`, see examples below.

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

`splitbands`: split all bands into disconnected subbands. See also `splitbands!`

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
  Band{2}: mesh of a 2-dimensional manifold
    Vertices   : 625
    Edges      : 1776

julia> bandstructure(h, :Γ, :X, :Y, :Γ; subticks = (10,15,10))
Bandstructure{2}: collection of 1D bands
  Bands        : 18
  Element type : scalar (Complex{Float64})
  Band{1}: mesh of a 1-dimensional manifold
    Vertices   : 33
    Edges      : 32

julia> bandstructure(h, mesh((0, 2π); subticks = 13); mapping = φ -> (φ, 0))
       # Equivalent to bandstructure(h, :Γ, :X; subticks = 13)
Bandstructure{2}: collection of 1D bands
  Bands        : 18
  Element type : scalar (Complex{Float64})
  Band{1}: mesh of a 1-dimensional manifold
    Vertices   : 11
    Edges      : 10

julia> ph = parametric(h, @hopping!((t; α) -> t * α));

julia> bandstructure(ph, mesh((0, 2π); subticks = 13); mapping = φ -> (φ, 0, (; α = 2φ)))
Bandstructure{2}: collection of 1D bands
  Bands        : 18
  Element type : scalar (Complex{Float64})
  Band{1}: mesh of a 1-dimensional manifold
    Vertices   : 11
    Edges      : 10
```

# See also
    `cuboid`, `diagonalizer`, `bloch`, `parametric`, `splitbands!`
"""
bandstructure(args...; kw...) = h -> bandstructure(h, args...; kw...)

function bandstructure(h::Hamiltonian{<:Any, L}; subticks = 13, kw...) where {L}
    L == 0 && throw(ArgumentError("Hamiltonian is 0D, use `spectrum` instead of `bandstructure`"))
    base = cuboid(filltuple((-π, π), Val(L))...; subticks = subticks)
    return bandstructure(h, base; kw...)
end

function bandstructure(h::Hamiltonian{<:Any,L}, node1, node2, nodes...; subticks = 13, transform = missing, kw...) where {L}
    allnodes = (node1, node2, nodes...)
    mapping´ = piecewise_mapping(allnodes, Val(L))
    base = cuboid(nodeindices(allnodes); subticks = subticks)
    transform´ = sanitize_band_transform(transform, h, allnodes)
    return bandstructure(h, base; mapping = mapping´, transform = transform´, kw...)
end

function bandstructure(h::Union{Hamiltonian,ParametricHamiltonian}, basemesh::CuboidMesh;
                       transform = missing, splitbands = true, showprogress = true, kw...)
    diag = diagonalizer(h; kw...)
    b = bandstructure(diag, basemesh, splitbands, showprogress)
    transform === missing || transform!(sanitize_band_transform(transform, h), b)
    return b
end

function bandstructure(matrixf::Function, basemesh::CuboidMesh;
                       mapping = missing, transform = missing, splitbands = true, showprogress = true)
    matrixf´ = wrapmapping(mapping, matrixf)
    dimh = size(samplematrix(matrixf´, basemesh), 1)
    diag = diagonalizer(matrixf´, dimh)
    b = bandstructure(diag, basemesh, splitbands, showprogress)
    transform === missing || transform!(transform, b)
    return b
end

@inline map_phiparams(mapping::Missing, basevertex) = sanitize_phiparams(basevertex)
@inline map_phiparams(mapping::Function, basevertex) = sanitize_phiparams(mapping(basevertex...))

wrapmapping(mapping::Missing, matrixf::Function) = matrixf
wrapmapping(mapping::Function, matrixf::Function) = basevertex -> matrixf(toSVector(mapping(basevertex...)))

sanitize_band_transform(::Missing, args...) = (identity, identity)
sanitize_band_transform(f::Function, args...) = (identity, f)
sanitize_band_transform(f::typeof(isometric), args...) = (isometric(args...), identity)
sanitize_band_transform((_,f)::Tuple{typeof(isometric),Function}, args...) = (isometric(args...), f)
sanitize_band_transform(fs::Tuple{Function,Function}, args...) = fs
sanitize_band_transform((_,f)::Tuple{Missing,Function}, args...) = (identity, f)
sanitize_band_transform((f,_)::Tuple{Function,Missing}, args...) = (f, identity)

samplematrix(matrixf, basemesh) = matrixf(Tuple(first(vertices(basemesh))))

#######################################################################
# bandstructure building
#######################################################################
function bandstructure(diag::Diagonalizer, basemesh::CuboidMesh, split, showprogress)
    # Step 1/2 - Diagonalise:
    spectra = bandstructure_diagonalize(diag, basemesh, showprogress)
    # Step 2/2 - Knit bands:
    bands = bandstructure_knit(basemesh, spectra, showprogress)
    bs = Bandstructure(bands, diag)
    split && (bs = splitbands(bs))
    return bs
end

## Diagonalize bands step
function bandstructure_diagonalize(diag, basemesh::CuboidMesh, showprogress)
    prog = Progress(length(basemesh), "Step 1/2 - Diagonalising: ")
    spectra = [build_spectrum(diag, vertex, showprogress, prog) for vertex in vertices(basemesh)]
    return spectra
end

function build_spectrum(diag, vertex, showprog, prog)
    (ϵs, ψs) = diag(Tuple(vertex), NoUnflatten())
    spectrum = Spectrum(ϵs, ψs, diag, vertex)
    showprog && ProgressMeter.next!(prog)
    return spectrum
end

## Knit bands step
function bandstructure_knit(basemesh::CuboidMesh{D}, spectra::AbstractArray{SP}, showprog) where {D,C,T,SP<:Spectrum{D,C,T}}
    simpitr = marchingsimplices(basemesh) # D+1 - dimensional iterator over simplices

    S       = basis_slice_type(first(spectra))
    verts   = SVector{D+1,T}[]
    vbases  = S[]
    vptrs   = Array{UnitRange{Int}}(undef, size(vertices(basemesh)))
    simps   = NTuple{D+1,Int}[]
    sbases  = NTuple{D+1,Matrix{C}}[]
    sptrs   = Array{UnitRange{Int}}(undef, size(simpitr))

    cbase = eachindex(basemesh)
    lbase = LinearIndices(cbase)

    # Collect vertices
    for csrc in cbase
        len = length(verts)
        push_verts!((verts, vbases), spectra[csrc])
        vptrs[csrc] = len+1:length(verts)
    end

    # Store subspace projections in vertex adjacency matrix
    prog = Progress(length(cbase), "Step 2/2 - Knitting bands: ")
    S´ = basis_block_type(first(spectra))
    I, J, V = Int[], Int[], S´[]
    for csrc in cbase
        for cdst in neighbors_forward(basemesh, csrc)
            push_adjs!((I, J, V), spectra[csrc], spectra[cdst], vptrs[csrc], vptrs[cdst])
        end
        showprog && ProgressMeter.next!(prog)
    end
    n = length(verts)
    adjprojs = sparse(I, J, V, n, n)
    adjmat = sparsealiasbool(adjprojs)

    # Build simplices from stable cycles around base simplices
    buffers = NTuple{D+1,Int}[], NTuple{D+1,Int}[]
    emptybases = filltuple(fill(zero(C), 0, 0), Val(D+1))  # sentinel bases for deg == 1 simplices
    for (csimp, vs) in zip(CartesianIndices(simpitr), simpitr)  # vs isa NTuple{D+1,CartesianIndex{D}}
        len = length(simps)
        ranges = getindex.(Ref(vptrs), vs)
        push_simps!((simps, sbases, verts), buffers, ranges, adjprojs, emptybases)
        sptrs[csimp] = len+1:length(simps)
    end

    bands = [Band(basemesh, verts, vbases, vptrs, adjmat, simps, sbases, sptrs)]
    return bands
end

function push_verts!((verts, vbases), spectrum)
    for rng in spectrum.subs
        vert = SVector(Tuple(spectrum.basevert)..., spectrum.energies[first(rng)])
        basis = view(spectrum.states, :, rng)
        push!(verts, vert)
        push!(vbases, basis)
    end
    return nothing
end

function push_adjs!((I, J, V), ssrc, sdst, rsrc, rdst)
    ψdst = sdst.states
    ψsrc = ssrc.states
    proj = ψdst' * ψsrc
    proj´ = copy(proj')
    for (is, rs) in enumerate(ssrc.subs)
        srcdim = length(rs)
        for (id, rd) in enumerate(sdst.subs)
            crange = CartesianIndices((rd, rs))
            crange´ = CartesianIndices((rs, rd))
            rank = rankproj(proj, crange)
            if !iszero(rank)
                srcdim -= rank
                srcdim < 0 && @warn("Unexpected band connectivity between $(ssrc.basevert) and $(sdst.basevert). Rank $rank in $(size(crange)) projector.")
                append!(I, (rdst[id], rsrc[is]))
                append!(J, (rsrc[is], rdst[id]))
                append!(V, (view(proj, crange), view(proj´, crange´)))
            end
        end
    end
    return nothing
end

# equivalent to r = round(Int, tr(m'm)) over crange, but if r > 0, must compute and count singular values
function rankproj(m, crange)
    r = round(Int, sum(c -> abs2(m[c]), crange))
    (iszero(r) || minimum(size(crange)) == 1) && return r
    s = svdvals(view(m, crange))
    r = count(s -> abs2(s) >= 0.5, s)
    return r
end

sparsealiasbool(s) = SparseMatrixCSC(s.m, s.n, s.colptr, s.rowval, fill(true, length(s.nzval)))

function push_simps!((simps, sbases, verts), buffers, ranges, adjprojs, emptybases)
    cycles = build_cycles!(buffers, ranges, adjprojs)
    for cycle in cycles
        done = false
        projs = getindex.(Ref(adjprojs), shiftleft(cycle), cycle)
        degs = degeneracy.(projs)
        if all(isequal(1), degs)  # optimization with sentinel emptybases
            push!(simps, cycle)
            push!(sbases, emptybases)
            continue
        end
        d, start = findmin(degs) # minimum degeneracy of source
        projs = shiftleft(projs, start - 1)
        bases = shiftright(accumulate(orthomul, projs; init = I(d)))
        while !done
            if rankbasis(first(bases)) < d
                basis_start = discard_zero_cols(first(bases))
                d = rankbasis(basis_start)
                d == 0 && break
                bases = shiftright(accumulate(orthomul, projs; init = basis_start))
            else
                done = true
            end
        end
        if done
            push!(simps, cycle)
            bases = shiftright(bases, start - 1)
            push!(sbases, bases)
        end
    end
    return nothing
end

# build cycles of elements in ranges that are connected by adjprojs
function build_cycles!((partials, partials´), ranges::NTuple{D´}, adjprojs) where {D´}
    resize!(partials, length(first(ranges)))
    for (n, i) in enumerate(first(ranges))
        partials[n] = padright((i,), Val(D´))
    end
    rows = rowvals(adjprojs)
    for n in 1:D´
        empty!(partials´)
        for partial in partials
            if n == D´
                seed = first(partial)
                nextrng = seed:seed
            else
                nextrng = ranges[n+1]
            end
            src = partial[n]
            for ptr in nzrange(adjprojs, src)
                dst = rows[ptr]
                dst in nextrng || continue
                partial´ = n == D´ ? partial : tuplesplice(partial, n+1, dst)
                push!(partials´, partial´)
            end
        end
        partials, partials´ = partials´, partials
    end
    return partials
end

function discard_zero_cols(mat)
    ncols = rankbasis(mat)
    mat´ = similar(mat, size(mat, 1), ncols)
    j = 0
    for col in eachcol(mat)
        if !iszero(col)
            j += 1
            mat´[:, j] .= col
        end
    end
    return mat´
end

# cheap-rank for number of non-zero columns
rankbasis(m) = count(!iszero, eachcol(m))

# The minimum projected norm2 per src state (m column) is 0.5/dim(src) by default.
# 0.5 due to adjacency, and 1/dim(src) because the projection eigenstate has a minimum
# 1/√din(src) component over some src basis vector
function orthomul(m1, m2)
    m = m2 * m1
    threshold = 0.5 / size(m, 2)
    return orthonormalize!(m, threshold)
end

#######################################################################
# splitbands
#######################################################################
"""
    splitbands(bs::Bandstructure)

Splits the bands in `bs` into disconnected subbands that share no vertices. See also `splitbands` option in `bandstructure`.

# See also
    `bandstructure`
"""
splitbands(b::Bandstructure) = Bandstructure(splitbands(b.bands), b.diag)

function splitbands(bands::Vector{<:Band})
    bands´ = similar(bands, 0)
    splitbands!.(Ref(bands´), bands)
    return bands´
end

function splitbands!(bands, band)
    bs, is, nbands = subgraphs(band.adjmat)
    nbands == 1 && return push!(bands, band)
    adjmats = split_adjmats(band.adjmat, bs, is, nbands)
    verts, vbases, vptrs = split_list(band.verts, band.vbases, band.vptrs, bs, is, nbands)
    simps, sbases, sptrs = split_list(band.simps, band.sbases, band.sptrs, bs, is, nbands)
    for n in 1:nbands
        newband = Band(band.basemesh, verts[n], vbases[n], vptrs[n], adjmats[n], simps[n], sbases[n], sptrs[n])
        nsimplices(newband) > 0 && push!(bands, newband)
    end
    return bands
end

struct BandIndex
    n::Int
end

Base.:*(_, b::BandIndex) = b
Base.:*(b::BandIndex, _) = b
Base.:+(b1::BandIndex, b2::BandIndex) = BandIndex(min(b1.n, b2.n))
Base.zero(b::Type{BandIndex}) = BandIndex(typemax(Int))

# for each row or column i in a, compute its subgraph bs[i] and the column/row index within that subgraph, is[i]
function subgraphs(a)
    bs = BandIndex.(1:size(a, 2))
    bs´ = copy(bs)
    total = sum(i -> i.n, bs)
    done = false
    while !done
        mul!(bs´, a, bs, true, true)
        bs, bs´ = bs´, bs
        total´ = sum(i -> i.n, bs)
        done = total == total´
        total = total´
    end
    dict = Dict(1 => 1)
    m = 1
    for b in bs
        if !haskey(dict, b.n)
            m += 1
            dict[b.n] = m
        end
    end
    is = similar(bs, Int)
    ngraphs = m
    counters = fill(0, ngraphs)
    for (i, b) in enumerate(bs)
        n = dict[b.n]
        bs[i] = BandIndex(n)
        counters[n] += 1
        is[i] = counters[n]
    end
    return bs, is, ngraphs
end

function split_adjmats(adjmat, bs, is, nbands)
    I0, J0, _ = findnz(adjmat)
    Is = [Int[] for _ in 1:nbands]
    Js = copy.(Is)
    for (i, j) in zip(I0, J0)
        n = bs[i].n
        n == bs[j].n || throw(ArgumentError("Unexpected error in subgraphs function"))
        push!(Is[n], is[i])
        push!(Js[n], is[j])
    end
    nverts = [count(isequal(BandIndex(n)), bs) for n in 1:nbands]
    adjmats = [sparse(Is[n], Js[n], true, nverts[n], nverts[n]) for n in 1:nbands]
    return adjmats
end

function split_list(vs, vbs, vptrs, bs, is, nbands)
    vs´ = [similar(vs, 0) for _ in 1:nbands]
    vbs´ = [similar(vbs, 0) for _ in 1:nbands]
    vptrs´ = [similar(vptrs) for _ in 1:nbands]
    counters = fill(0, nbands)
    counters´ = copy(counters)
    for (iptr, rng) in enumerate(vptrs)
        for i in rng
            n, v = get_vert_or_simp(vs[i], i, bs, is)
            push!(vs´[n], v)
            push!(vbs´[n], vbs[i])
        end
        counters´ = length.(vs´)
        for n in 1:nbands
            vptrs´[n][iptr] = counters[n]+1:counters´[n]
        end
        counters, counters´ = counters´, counters
    end
    return vs´, vbs´, vptrs´
end

# return band index and vertex/simplex
get_vert_or_simp(v::SVector, i, bs, is) = bs[i].n, v
get_vert_or_simp(s::NTuple, i, bs, is) = bs[first(s)].n, getindex.(Ref(is), s)

#######################################################################
# Bandstructure indexing
#######################################################################
Base.getindex(bs::Bandstructure, ϕs::Tuple; around = missing) = interpolate_bandstructure(bs, ϕs, around)

function interpolate_bandstructure(bs::Bandstructure, ϕs, around)
    subs = subspace_type(bs, ϕs)[]
    foreach(band -> interpolate_band!(subs, band, ϕs, bs.diag.orbstruct), bs.bands)
    return filter_around!(subs, around)
end

function subspace_type(bs::Bandstructure, ϕs)
    (isempty(bs.bands) || isempty(first(bs.bands).vbases)) && throw(ArgumentError("Cannot index into empty bandstructure"))
    band = first(bs.bands)
    ε0 = last(first(band.verts))
    b0 = first(band.vbases)
    m0 = Matrix{eltype(b0)}(undef, size(b0, 1), 0)
    return typeof(Subspace(bs.diag.orbstruct, ε0, m0, ϕs))
end

function interpolate_band!(subs, band::Band{D}, ϕs, orbstruct) where {D}
    D == length(ϕs) || throw(ArgumentError("Bandstructure needs a NTuple{$D} of base coordinates for interpolation"))
    s = find_basemesh_simplex(promote(ϕs...), band.basemesh)
    s === nothing && return subs
    (ϕs, inds, dϕinds) = s
    for i in band.sptrs[inds...]
        si = band.simps[i]
        ε0, εs = firsttail(last.(getindex.(Ref(band.verts), si)))
        b0, bs = firsttail(getindex.(Ref(band.vbases), si))
        p0, ps = firsttail(band.sbases[i])
        energy = ε0 * (1 - sum(dϕinds)) + sum(εs .* dϕinds)
        basis = isempty(p0) ? copy(b0) : b0 * p0
        lmul!(1 - sum(dϕinds), basis)
        if isempty(p0)  # deg == 1 sentinel optimization
            for j in 1:D
                basis .+= bs[j] .* (sentinel_phase(bs[j], b0) * dϕinds[j])
            end
        else
            foreach(i -> mul!(basis, bs[i], ps[i], dϕinds[i], true), 1:D)
        end
        # h is necessary here to perhaps unflatten basis
        push!(subs, Subspace(orbstruct, energy, basis, ϕs))
    end
    return subs
end

sentinel_phase(bj::AbstractVector{<:Real}, b0::AbstractVector{<:Real}) = 1
sentinel_phase(bj, b0) = cis(angle(dot(bj, b0)))

# The simplex sought has unitperm such that reverse(dϕs[unitperm]) is sorted
function find_basemesh_simplex(ϕs::NTuple{D}, basemesh) where {D}
    baseinds = find_basemesh_interval.(ϕs, basemesh.ticks)
    any(iszero, baseinds) && return nothing
    vertex0 = getindex.(basemesh.ticks, baseinds)
    vertex1 = getindex.(basemesh.ticks, baseinds .+ 1)
    dϕs = (ϕs .- vertex0) ./ (vertex1 .- vertex0)
    dϕperm = Tuple(sortperm(SVector(dϕs), rev = true))
    baseperms = unitperms(basemesh)
    i = findfirst(isequal(dϕperm), baseperms)
    i === nothing && return nothing
    sverts = SVector.(Tuple.(Base.tail(unitsimplex(basemesh, i))))
    smat = hcat(sverts...)
    dϕinds = inv(smat) * SVector(dϕs)
    inds = (baseinds..., i)
    return ϕs, inds, dϕinds
end

function find_basemesh_interval(ϕ, ticks)
    @inbounds for m = 1:length(ticks)-1
        ticks[m] <= ϕ < ticks[m+1] && return m
    end
    ϕ ≈ last(ticks) && return length(ticks) - 1
    return 0
end

filter_around!(ss, ::Missing) = sort!(ss; by = s -> s.energy, alg = Base.DEFAULT_UNSTABLE)
filter_around!(ss, ε0::Number) = filter_around!(ss, ε0, 1)

function filter_around!(ss, (ε0, n)::Tuple)
    0 <= n <= length(ss) || throw(ArgumentError("Cannot retrieve more than $(length(ss)) subspaces"))
    return filter_around!(ss, ε0, 1:n)
end

filter_around!(ss::Vector{<:Subspace}, ε0, which) = partialsort!(ss, which; by = s -> abs(s.energy - ε0))

#######################################################################
# Bandstructure minima, maxima, gap, gapedge
#######################################################################
"""
    minima(b::Bandstructure{1}; refinesteps = 0)

For a 1D bandstructure `b`, compute a vector of `Vector{Tuple{T,T}}`s (one per band),
containing pairs `(φ, ε)` of Bloch phase and energy where the band, as sampled, has a local
minimum. The minima will be further refined by a number `refinesteps` of bisections steps.
Only band vertices with one neighbors on each side will be considered as potential local
minimum.

# See also:
    `maxima`, `gapedge`, `gap`
"""
minima(b::Bandstructure{1,<:Any,T}; kw...) where {T} =
    Vector{Tuple{real(T),real(T)}}[band_extrema(band, minimum_criterion, b.diag; kw...) for band in b.bands]

minimum_criterion(ε, εs...) = all(>=(ε), εs)

"""
    maxima(b::Bandstructure{1}; refinesteps = 0)

For a 1D bandstructure `b`, compute a vector of `Vector{Tuple{T,T}}`s (one per band),
containing pairs `(φ, ε)` of Bloch phase and energy where the band, as sampled, has a local
maximum. The maxima will be further refined by a number `refinesteps` of bisections steps.
Only band vertices with one neighbors on each side will be considered as potential local
maximum.

# See also:
    `minima`, `gapedge`, `gap`
"""
maxima(b::Bandstructure{1,<:Any,T}; kw...) where {T} =
    Vector{Tuple{real(T),real(T)}}[band_extrema(band, maximum_criterion, b.diag; kw...) for band in b.bands]

maximum_criterion(ε, εs...) = all(<=(ε), εs)

function band_extrema(b::Band{1}, criterion, diag; refinesteps = 0)
    found, neighbors = findall_with_neighbors(criterion, b)
    found´ = refine_bisection.(found, neighbors, refinesteps, criterion, Ref(diag))
    unique!(x -> round.(chop.(x), digits = 8), found´)
    return found´
end

function findall_with_neighbors(criterion::Function, b::Band{D,<:Any,T´}) where {D,T´}
    vertices = b.verts
    T = real(T´)
    found = NTuple{2,T}[]
    neighs = Tuple{NTuple{2,T},NTuple{2,T}}[]
    for (i, vertex) in enumerate(vertices)
        ns = neighbors(b, i)
        if length(ns) == max_neighbors(D)
            (φ0, ε0) = real.(vertex)
            i1, i2 = ns
            (φ1, ε1) = real.(vertices[i1])
            (φ2, ε2) = real.(vertices[i2])
            φ1 < φ0 < φ2 || φ2 < φ0 < φ1 || continue
            if criterion(ε0, ε1, ε2)
                push!(found, (φ0,  ε0))
                push!(neighs, ((φ1, ε1), (φ2, ε2)))
            end
        end
    end
    return found, neighs
end

neighbors(b::Band, i::Int) = (rowvals(b.adjmat)[ptr] for ptr in nzrange(b.adjmat, i))

max_neighbors(D::Int) = 2^D # sum(m->binomial(D+1, m), 1:D), where D = 1 for 1D bands

function refine_bisection(found, neighs, steps, criterion, diag)
    φ0, ε0 = found
    (φ1, ε1), (φ2, ε2) = neighs
    if φ1 > φ2
        φ1, ε1, φ2, ε2 = φ2, ε2, φ1, ε1
    end
    φ1 < φ0 < φ2 || return φ0, ε0
    for _ in 1:steps
        a, b = SA[(φ1-φ0)^2 φ1-φ0; (φ2-φ0)^2 φ2-φ0] \ SA[ε1-ε0; ε2-ε0]
        dφ = -b / 2a  # corresponds to estimated dε = -b^2/4a
        if iszero(dφ) || isnan(dφ) || ifelse(dφ > 0, dφ/(φ2 - φ0), -dφ/(φ0 - φ1)) > 1
            break
        end
        εs, _ = diag(φ0 + dφ)
        ε0´ = select_closest(real.(εs), criterion, ε0, ε1, ε2)
        if dφ > 0
            φ1, φ0, φ2 = φ0, φ0 + dφ, φ2
            ε1, ε0, ε2 = ε0, ε0´, ε2
        else
            φ1, φ0, φ2 = φ1, φ0 + dφ, φ0
            ε1, ε0, ε2 = ε1, ε0´, ε0
        end
    end
    return chop(φ0), ε0
end

select_closest(εs, criterion, ε0, εi...) =
    last(findmin(ε -> ifelse(criterion(ε, ε0, εi...), abs(ε - ε0), Inf), εs))

"""
    gapedge(b::Bandstructure{1}, ε₀; refinesteps = 0)

For a 1D bandstructure `b`, compute two tuples, `(φ₊, ε₊)` and `(φ₋, ε₋)`, of band points
closest in energy to `ε₀`, from above and below, respectively. If ε₀ is inside a band or
outside the global bandwidth, `φ₊` and `φ₋` will be `missing`. See `minima` or `maxima` for
details about `refinesteps`.

    gapedge(b::Bandstructure{1}, ε₀, +; kw...)
    gapedge(b::Bandstructure{1}, ε₀, -; kw...)

Compute only `(φ₊, ε₊)` or `(φ₋, ε₋)`, respectively.

# See also:
    `gap`, `minima`, `maxima`
"""
gapedge(b::Bandstructure{1}, ε0; kw...) = gapedge(b, ε0, +; kw...), gapedge(b, ε0, -; kw...)

function gapedge(b::Bandstructure{1,<:Any,T}, ε0, ::typeof(+); kw...) where {T}
    isinband(b, ε0) && return (missing, zero(T))
    minbands = Iterators.flatten(filter!.(φε -> last(φε) > ε0, minima(b; kw...)))
    isempty(minbands) && return (missing, T(Inf))
    (φ₊, ε₊) = findmin(last, minbands) |> last
    return (φ₊, ε₊)
end

function gapedge(b::Bandstructure{1,<:Any,T}, ε0, ::typeof(-); kw...) where {T}
    isinband(b, ε0) && return (missing, zero(T))
    maxbands = Iterators.flatten(filter!.(φε -> last(φε) < ε0, maxima(b; kw...)))
    isempty(maxbands) && return (missing, T(-Inf))
    (φ₋, ε₋) = findmax(last, maxbands) |> last
    return (φ₋, ε₋)
end

"""
    isinband(b::Bandstructure, ε)
    isinband(b::Band, ε)

Returns true if `ε` is contained within a band, false otherwise.

# See also:
    `gap`, `minima`, `maxima`
"""
isinband(b::Bandstructure, ε) = any(band -> isinband(band, ε), b.bands)
isinband(b::Band, ε) = maximum(last, b.verts) > ε > minimum(last, b.verts)

"""
    gap(b::Bandstructure{1}, ε₀; refinesteps = 0)

Compute the gap if a 1D bandstructure `b` around ε₀, if any.

# See also:
    `gapedge`, `minima`, `maxima`
"""
function gap(b::Bandstructure{1}, ε0; kw...)
    (_, ε₊), (_, ε₋) = gapedge(b, ε0; kw...)
    return ε₊ - ε₋
end