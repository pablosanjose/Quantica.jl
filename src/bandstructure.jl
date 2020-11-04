#######################################################################
# Spectrum
#######################################################################
struct Spectrum{E,T,A<:AbstractMatrix{T}}
    energies::E
    states::A
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

The energies and eigenstates in the resulting `s::Spectrum` object can be accessed with
`energies(s)` and `states(s)`

# See also
    `energies`, `states`, `bandstructure`

"""
function spectrum(h; method = LinearAlgebraPackage(), transform = missing)
    matrix = similarmatrix(h, method_matrixtype(method, h))
    bloch!(matrix, h)
    (ϵk, ψk) = diagonalize(matrix, method)
    s = Spectrum(ϵk, ψk)
    transform === missing || transform!(transform, s)
    return s
end

"""
    energies(s::Spectrum)

Return the energies of `s` as a `Vector`

# See also
    `spectrum`, `states`
"""
energies(s::Spectrum) = s.energies

"""
    states(s::Spectrum)

Return the states of `s` as the columns of a `Matrix`

# See also
    `spectrum`, `energies`
"""
states(s::Spectrum) = s.states

"""
    transform!(f::Function, s::Spectrum)

Transform the energies of `s` by applying `f` to them in place.
"""
transform!(f, s::Spectrum) = (map!(f, s.energies, s.energies); s)

######################################################################
# BandMesh
######################################################################
struct BandMesh{D´,T<:Number}  # D´ is dimension of BaseMesh space plus one (energy)
    verts::Vector{SVector{D´,T}}
    adjmat::SparseMatrixCSC{Bool,Int}   # Undirected graph: both dest > src and dest < src
    simpinds::Vector{NTuple{D´,Int}}
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

nsimplices(m::BandMesh) = length(m.simpinds)

vertices(m::BandMesh) = m.verts

edges(adjmat, src) = nzrange(adjmat, src)

# neighbors(adjmat::SparseMatrixCSC, src::Int) = view(rowvals(adjmat), nzrange(adjmat, src))

edgedest(adjmat, edge) = rowvals(adjmat)[edge]

edgevertices(m::BandMesh) =
    ((vsrc, m.verts[edgedest(m.adjmat, edge)]) for (i, vsrc) in enumerate(m.verts) for edge in edges(m.adjmat, i))

transform!(f::Function, m::BandMesh) = (map!(f, vertices(m), vertices(m)); m)

#######################################################################
# Bandstructure
#######################################################################
struct Simplices{D´,T,S<:SubArray,D}
    sverts::Vector{NTuple{D´,SVector{D´,T}}}
    sstates::Vector{NTuple{D´,S}}
    sptrs::Array{UnitRange{Int},D}  # range of indices of sverts and svecs for each simplex CartesianIndex in base mesh
end

struct Bandstructure{D,T,M<:CuboidMesh{D},D´,B<:BandMesh{D´,T},S<:Simplices{D´,T}}   # D is dimension of base mesh, D´ = D+1
    base::M
    bands::Vector{B}
    simplices::S
end

function Base.show(io::IO, bs::Bandstructure{D,M}) where {D,M}
    i = get(io, :indent, "")
    ioindent = IOContext(io, :indent => string(i, "  "))
    print(io, i, summary(bs), "\n",
"$i  Bands         : $(length(bs.bands))
$i  Element type  : $(displayelements(M))
$i  Vertices      : $(nvertices(bs))
$i  Edges         : $(nedges(bs))
$i  Simplices     : $(nsimplices(bs))")
end

Base.summary(::Bandstructure{D,M}) where {D,M} =
    "Bandstructure{$D}: bands of a $(D)D Hamiltonian"

# API #

nvertices(bs::Bandstructure) = sum(nvertices, bands(bs))

nedges(bs::Bandstructure) = sum(nedges, bands(bs))

nsimplices(bs::Bandstructure) = sum(nsimplices, bands(bs))

nbands(bs::Bandstructure) = length(bands(bs))

"""
    bands(bs::Bandstructure)

Return a vector of all the `Band`s in `bs`.
"""
bands(bs::Bandstructure) = bs.bands

"""
    vertices(bs::Bandstructure, i)

Return the vertices `(k..., ϵ)` of the i-th band in `bs`, in the form of a
`Vector{SVector{L+1}}`, where `L` is the lattice dimension.
"""
vertices(bs::Bandstructure, i) = vertices(bands(bs)[i])

"""
    energies(b::Bandstructure)

Return the sorted unique energies of `b` as a `Vector`

# See also
    `bandstructure`, `states`
"""
energies(bs::Bandstructure) = unique!(sort!([last(v) for b in bands(bs) for v in vertices(b)]))

"""
    states(bs::Bandstructure, i)

Return the states of each vertex of the i-th band in `bs`, in the form of a `Matrix` of size
`(nψ, nk)`, where `nψ` is the length of each state vector, and `nk` the number of vertices.
"""
states(bs::Bandstructure, i) = states(bands(bs)[i])

# states(b::Band) = reshape(b.statess, b.dimstates, :)

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
        alignnormals!(band.simpinds, vs)
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
# bandstructure
#######################################################################
"""
    bandstructure(h::Hamiltonian; subticks = 13, kw...)

Compute `bandstructure(h, mesh((-π,π)...; subticks = subticks); kw...)` using a mesh over `h`'s
full Brillouin zone with the specified `subticks` along each [-π,π] reciprocal axis.

    bandstructure(h::Hamiltonian, nodes...; subticks = 13, kw...)

Create a linecut of a bandstructure of `h` along a polygonal line connecting two or more
`nodes`. Each node is either a `Tuple` or `SVector` of Bloch phases, or a symbolic name for
a Brillouin zone point (`:Γ`,`:K`, `:K´`, `:M`, `:X`, `:Y` or `:Z`). Each segment in the
polygon has the specified number of `subticks`. Different `subticks` per segments can be
specified with `subticks = (p1, p2...)`.

    bandstructure(h::Hamiltonian, mesh::BandMesh; mapping = missing, kw...)

Compute the bandstructure `bandstructure(h, mesh; kw...)` of Bloch Hamiltonian `bloch(h,
ϕ)`, with `ϕ = v` taken on each vertex `v` of `mesh` (or `ϕ = mapping(v...)` if a `mapping`
function is provided).

    bandstructure(ph::ParametricHamiltonian, ...; kw...)

Compute the bandstructure of a `ph`. Unless all parameters have default values, a `mapping`
is required between mesh vertices and Bloch/parameters for `ph`, see details on `mapping`
below.

    bandstructure(matrixf::Function, mesh::BandMesh; kw...)

Compute the bandstructure of the Hamiltonian matrix `m = matrixf(ϕ)`, with `ϕ` evaluated on
the vertices `v` of the `mesh`. Note that `ϕ` in `matrixf(ϕ)` is an unsplatted container.
Hence, i.e. `matrixf(x) = ...` or `matrixf(x, y) = ...` will not work. Use `matrixf((x,)) =
...`, `matrixf((x, y)) = ...` or matrixf(s::SVector) = ...` instead.

    h |> bandstructure([mesh,]; kw...)

Curried form of the above equivalent to `bandstructure(h, [mesh]; kw...)`.

# Options

The default options are

    (mapping = missing, minoverlap = 0.3, method = LinearAlgebraPackage(), transform = missing, showprogress = true)

`mapping`: when not `missing`, `mapping = v -> p` is a function that map mesh vertices `v`
to Bloch phases and/or parameters `p`. The structure of `p` is whatever is accepted by
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
    `mesh`, `bloch`, `parametric`
"""
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
                       method = LinearAlgebraPackage(), minoverlap = 0.3, mapping = missing, transform = missing, showprogress = true)
    # ishermitian(h) || throw(ArgumentError("Hamiltonian must be hermitian"))
    matrix = similarmatrix(h, method_matrixtype(method, h))
    diag = diagonalizer(matrix, method, minoverlap)
    matrixf(vertex) = bloch!(matrix, h, map_phiparams(mapping, vertex))
    b = bandstructure(matrixf, basemesh, diag, showprogress)
    if transform !== missing
        transform´ = sanitize_transform(transform, h)
        transform!(transform´, b)
    end
    return b
end

function bandstructure(matrixf::Function, basemesh::CuboidMesh;
                       method = LinearAlgebraPackage(),  minoverlap = 0.3, mapping = missing, transform = missing, showprogress = true)
    matrixf´ = wrapmapping(mapping, matrixf)
    matrix = samplematrix(matrixf´, basemesh)
    diag = diagonalizer(matrix, method, minoverlap)
    b = bandstructure(matrixf´, basemesh, diag, showprogress)
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

function bandstructure(matrixf::Function, basemesh::CuboidMesh, diago::Diagonalizer, showprogress)
    # Step 1/3 - Diagonalising:
    subspaces, nverts = bandstructure_diagonalize(matrixf, basemesh, diago, showprogress)
    # Step 2/3 - Knitting bands:
    bands, cuboidinds, linearinds = bandstructure_knit(basemesh, diago, subspaces, nverts, showprogress)
    # Step 3/3 - Collecting simplices:
    simplices = bandstructure_collect(subspaces, bands, cuboidinds, showprogress)

    return Bandstructure(basemesh, bands, simplices)
end

#######################################################################
# bandstructure_diagonalize
#######################################################################
struct Subspace{C,T,S<:SubArray{C}}
    energy::T
    states::S
end

degeneracy(s::Subspace) = size(s.states, 2)

function bandstructure_diagonalize(matrixf::Function, basemesh::CuboidMesh{D,T}, diago::Diagonalizer{M,S}, showprogress = false) where {D,T,M,C,S<:SubArray{C}}
    prog = Progress(length(basemesh), "Step 1/3 - Diagonalising: ")
    subspaces = Array{Vector{Subspace{C,T,S}},D}(undef, size(basemesh)...)
    nverts = 0
    for n in eachindex(basemesh)
        matrix = matrixf(Tuple(vertex(basemesh, n)))
        # (ϵk, ψk) = diagonalize(Hermitian(matrix), d)  ## This is faster (!)
        (ϵs, ψs) = diagonalize(matrix, diago.method)
        subspaces[n] = collect_subspaces(ϵs, ψs, Subspace{C,T,S})
        nverts += length(subspaces[n])
        showprogress && ProgressMeter.next!(prog; showvalues = ())
    end
    return subspaces, nverts
end

collect_subspaces(ϵs, ψs, ::Type{SS}) where {SS} =
    SS[Subspace(ϵs[first(rng)], view(ψs, :, rng)) for rng in approxruns(ϵs)]

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

function bandstructure_knit(basemesh::CuboidMesh{D,T}, diago::Diagonalizer{M,S}, subspaces, nverts, showprog = false) where {D,T,M,S}
    prog = Progress(nverts, "Step 2/3 - Knitting bands: ")

    bands = BandMesh{D+1,T}[]
    pending = Tuple{BandCuboidIndex{D},BandCuboidIndex{D}}[]   # pairs of neighboring vertex indices src::IT, dst::IT
    linearinds = [zeros(BandLinearIndex, length(ss)) for ss in subspaces] # 0 == unclassified, > 0 vertex index
    cuboidinds = BandCuboidIndex{D}[]                          # cuboid indices of processed vertices
    I = Int[]; J = Int[]                                       # To build adjacency matrices
    P = real(eltype(eltype(S)))                                # type of projections between states
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
        bandmesh = knit_band(bandidx, basemesh, subspaces, diago.minoverlap, pending, cuboidinds, linearinds, I, J, projinds, showprog, prog)
        push!(bands, bandmesh)
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
        push!(cuboidinds, dst)
        vertcounter += 1
        linearinds[n][i] = BandLinearIndex(bandidx, vertcounter)
        src == dst || append_adjacent!(I, J, linearinds[n0][i0], linearinds[n][i])
        showprog && ProgressMeter.next!(prog; showvalues = ())

        subdst = subspaces[n][i]
        deg = degeneracy(subdst)
        found = false
        for n´ in neighbors(basemesh, n)
            deg == 1 && n´ == n0 && continue  # Only if deg == 1 is this justified (think deg at BZ boundary)
            sorted_valid_projections!(projinds, subspaces[n´], subdst, minoverlap, bandidx, linearinds[n´])
            cumdeg´ = 0
            for (p, i´) in projinds
                i´ == 0 && break
                push!(pending, (dst, BandCuboidIndex(n´, i´)))
                cumdeg´ += degeneracy(subspaces[n´][i´])
                cumdeg´ >= deg && break # links on each column n´ = cumulated deg at most equal to deg links
                found = true
            end
        end
    end

    adjmat = sparse(I, J, true)

    simpinds = band_simplices(verts, adjmat)

    return BandMesh(verts, adjmat, simpinds)
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
        p = proj(sub.states, sub0.states, realzero, complexzero)
        p > minoverlap && (projinds[j] = (p, j))
    end
    sort!(projinds, rev = true, alg = Base.DEFAULT_UNSTABLE)
    return projinds
end

# non-allocating version of `sum(abs2, ψ' * ψ0)`
function proj(ψ, ψ0, realzero, complexzero)
    size(ψ, 1) == size(ψ0, 1) || throw(error("Internal error: eigenstates of different sizes"))
    p = realzero
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
    simpinds = NTuple{D´,Int}[]
    if nverts >= D´
        buffer = (NTuple{D´,Int}[], NTuple{D´,Int}[])
        for srcind in eachindex(vertices)
            newsimps = vertex_simplices!(buffer, adjmat, srcind)
            D´ > 2 && alignnormals!(newsimps, vertices)
            append!(simpinds, newsimps)
        end
    end
    return simpinds
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
function bandstructure_collect(subspaces::Array{Vector{Subspace{C,T,S}},D}, bands, cuboidinds, showprog) where {C,T,S,D}
    nsimplices = sum(band -> length(band.simpinds), bands)
    prog = Progress(nsimplices, "Step 3/3 - Collecting simplices: ")

    sverts = Vector{NTuple{D+1,SVector{D+1,T}}}(undef, nsimplices)
    sstates = Vector{NTuple{D+1,S}}(undef, nsimplices)
    sptrs = fill(1:0, size(subspaces) .- 1)                    # assuming non-periodic basemesh
    s0inds = Vector{CartesianIndex{D}}(undef, nsimplices)    # base cuboid index for reference vertex in simplex, for sorting

    scounter = 0
    ioffset = 0
    for band in bands
        for s in band.simpinds
            scounter += 1
            let ioffset = ioffset  # circumvent boxing, JuliaLang/#15276
                s0inds[scounter] = minimum(i -> cuboidinds[ioffset + i].baseidx, s)
                sverts[scounter] = ntuple(i -> band.verts[s[i]], Val(D+1))
                sstates[scounter] = ntuple(Val(D+1)) do i
                    c = cuboidinds[ioffset + s[i]]
                    subspaces[c.baseidx][c.colidx].states
                end
            end
            showprog && ProgressMeter.next!(prog; showvalues = ())
        end
        ioffset += nvertices(band)
    end

    p = sortperm(s0inds; alg = Base.DEFAULT_UNSTABLE)
    permute!(s0inds, p)
    permute!(sverts, p)
    permute!(sstates, p)

    for rng in equalruns(s0inds)
        sptrs[s0inds[first(rng)]] = rng
    end

    return Simplices(sverts, sstates, sptrs)
end