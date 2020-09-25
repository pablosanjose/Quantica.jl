#######################################################################
# Spectrum
#######################################################################
struct Spectrum{E,T,A<:AbstractMatrix{T}}
    energies::E
    states::A
end

"""
    spectrum(h; method = defaultmethod(h), transform = missing)

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
function spectrum(h; method = defaultmethod(h), transform = missing)
    matrix = similarmatrix(h, method)
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

#######################################################################
# Bandstructure
#######################################################################
struct Band{M,A<:AbstractVector{M},MD<:Mesh,S<:AbstractArray}
    mesh::MD        # Mesh with missing vertices removed
    simplices::S    # Tuples of indices of mesh vertices that define mesh simplices
    states::A       # Must be resizeable container to build & refine band
    dimstates::Int  # Needed to extract the state at a given vertex from vector `states`
end

function Band(mesh::Mesh{D}, states::AbstractVector{M}, dimstates::Int) where {M,D}
    simps = simplices(mesh, Val(D))
    return Band(mesh, simps, states, dimstates)
end

struct Bandstructure{D,M,B<:Band{M},MD<:Mesh{D}}   # D is dimension of parameter space
    bands::Vector{B}
    kmesh::MD
end

function Base.show(io::IO, b::Bandstructure{D,M}) where {D,M}
    i = get(io, :indent, "")
    ioindent = IOContext(io, :indent => string(i, "  "))
    print(io, i, summary(b), "\n",
"$i  Bands        : $(length(b.bands))
$i  Element type : $(displayelements(M))")
    print(ioindent, "\n", b.kmesh)
end

Base.summary(b::Bandstructure{D,M}) where {D,M} =
    "Bandstructure{$D}: collection of $(D)D bands"

# API #
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

vertices(b::Band) = vertices(b.mesh)

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

states(b::Band) = reshape(b.states, b.dimstates, :)

"""
    transform!(f::Function, b::Bandstructure)

Transform the energies of all bands in `b` by applying `f` to them in place.
"""
function transform!(f, bs::Bandstructure)
    for band in bands(bs)
        vs = vertices(band)
        for (i, v) in enumerate(vs)
            vs[i] = SVector((tuplemost(Tuple(v))..., f(last(v))))
        end
    end
    return bs
end

#######################################################################
# bandstructure
#######################################################################
"""
    bandstructure(h::Hamiltonian; points = 13, kw...)

Compute the bandstructure of `h` on a mesh over `h`'s full Brillouin zone, with `points`
points along each axis, spanning the interval [-π,π] along each reciprocal axis.

    bandstructure(h::Hamiltonian, spec::MeshSpec; lift = missing, kw...)

Call `bandstructure(h, mesh; lift = lift, kw...)` with `mesh = buildmesh(spec, h)` and `lift
= buildlift(spec, h)` if not provided. See `MeshSpec` for available mesh specs. If the `lift
= missing` and the dimensions of the mesh do not match the Hamiltonian's, a `lift` function
is used that lifts the mesh onto the dimensions `h` by appending vertex coordinates with
zeros.

    bandstructure(h::Hamiltonian, mesh::Mesh; lift = missing, kw...)

Compute the bandstructure `bandstructure(h, mesh; kw...)` of Bloch Hamiltonian `bloch(h,
ϕ)`, with `ϕ = v` taken on each vertex `v` of `mesh` (or `ϕ = lift(v...)` if a `lift`
function is provided).

    bandstructure(ph::ParametricHamiltonian, ...; kw...)

Compute the bandstructure of a `ph` with `i` parameters (see `parameters(ph)`), where `mesh`
is interpreted as a discretization of parameter space ⊗ Brillouin zone, so that each vertex
reads `v = (p₁,..., pᵢ, ϕ₁,..., ϕⱼ)`, with `p` the values assigned to `parameters(ph)` and
`ϕᵢ` the Bloch phases.

    bandstructure(matrixf::Function, mesh::Mesh; kw...)

Compute the bandstructure of the Hamiltonian matrix `m = matrixf(ϕ)`, with `ϕ` evaluated on
the vertices `v` of the `mesh`. Note that `ϕ` in `matrixf(ϕ)` is an unsplatted container.
Hence, i.e. `matrixf(x) = ...` or `matrixf(x, y) = ...` will not work, use `matrixf((x,)) =
...` or `matrixf((x, y)) = ...` instead.

    h |> bandstructure([mesh,]; kw...)

Curried form of the above equivalent to `bandstructure(h, [mesh]; kw...)`.

# Options

The default options are

    (lift = missing, minoverlap = 0, method = defaultmethod(h), transform = missing)

`lift`: when not `missing`, `lift` is a function `lift = (vs...) -> ϕ`, where `vs` are the
coordinates of a mesh vertex and `ϕ` are Bloch phases if sampling a `h::Hamiltonian`, or
`(paramsⱼ..., ϕᵢ...)` if sampling a `ph::ParametricHamiltonian`, and `params` are values for
`parameters(ph)`. It represents a mapping from a mesh and a Brillouin/parameter space. This
allows to compute a bandstructure along a cut in the Brillouin zone/parameter space, see
below for examples.

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

`transform`: the option `transform = ε -> f(ε)` allows to transform eigenvalues by `f` in the returned
bandstructure (useful for performing shifts or other postprocessing).

# Examples
```jldoctest
julia> h = LatticePresets.honeycomb() |> hamiltonian(hopping(-1, range = 1/√3)) |> unitcell(3);

julia> bandstructure(h; points = 25, method = LinearAlgebraPackage())
Bandstructure{2}: collection of 2D bands
  Bands        : 8
  Element type : scalar (Complex{Float64})
  Mesh{2}: mesh of a 2-dimensional manifold
    Vertices   : 625
    Edges      : 1776

julia> bandstructure(h, linearmesh(:Γ, :X, :Y, :Γ))
Bandstructure{1}: collection of 1D bands
  Bands        : 17
  Element type : scalar (Complex{Float64})
  Mesh{1}: mesh of a 1-dimensional manifold
    Vertices   : 37
    Edges      : 36

julia> bandstructure(h, marchingmesh((0, 2π); points = 25); lift = φ -> (φ, 0))
       # Equivalent to bandstructure(h, linearmesh(:Γ, :X; points = 11))
Bandstructure{1}: collection of 1D bands
  Bands        : 18
  Element type : scalar (Complex{Float64})
  Mesh{1}: mesh of a 1-dimensional manifold
    Vertices   : 25
    Edges      : 24
```

# See also
    `marchingmesh`, `linearmesh`
"""
function bandstructure(h::Hamiltonian; points = 13, kw...)
    meshspec = marchingmesh(filltuple((-π, π), Val(latdim(h)))...; points = points)
    return bandstructure(h, meshspec; kw...)
end

function bandstructure(h::Union{Hamiltonian,ParametricHamiltonian}, spec::MeshSpec; lift = missing, kw...)
    mesh = buildmesh(spec, h)
    lift´ = lift === missing ? buildlift(spec, h) : lift
    return bandstructure(h, mesh; lift = lift´, kw...)
end

function bandstructure(h::Union{Hamiltonian,ParametricHamiltonian}, mesh::Mesh;
                       method = defaultmethod(h), lift = missing, minoverlap = 0, transform = missing)
    # ishermitian(h) || throw(ArgumentError("Hamiltonian must be hermitian"))
    matrix = similarmatrix(h, method)
    codiag = codiagonalizer(h, matrix, mesh, lift)
    diag = diagonalizer(method, codiag, minoverlap)
    matrixf(ϕs) = bloch!(matrix, h, applylift(lift, ϕs))
    b = _bandstructure(matrixf, matrix, mesh, diag)
    transform === missing || transform!(transform, b)
    return b
end

function bandstructure(matrixf::Function, mesh::Mesh;
                       method = missing, lift = missing, minoverlap = 0, transform = missing)
    matrixf´ = _wraplift(matrixf, lift)
    matrix = _samplematrix(matrixf´, mesh)
    method´ = method === missing ? defaultmethod(matrix) : method
    codiag = codiagonalizer(matrixf´, matrix, mesh, missing)
    diag = diagonalizer(method´, codiag, minoverlap)
    b = _bandstructure(matrixf´, matrix, mesh, diag)
    transform === missing || transform!(transform, b)
    return b
end

_samplematrix(matrixf, mesh) = matrixf(Tuple(first(vertices(mesh))))

_wraplift(matrixf, lift::Missing) = matrixf
_wraplift(matrixf, lift) = ϕs -> matrixf(applylift(lift, ϕs))

@inline applylift(lift::Missing, ϕs) = toSVector(ϕs)

@inline applylift(lift::Function, ϕs) = toSVector(lift(ϕs...))

function _bandstructure(matrixf::Function, matrix´::AbstractMatrix{M}, mesh::MD, d::Diagonalizer) where {M,D,T,MD<:Mesh{D,T}}
    nϵ = 0                           # Temporary, to be reassigned
    ϵks = Matrix{T}(undef, 0, 0)     # Temporary, to be reassigned
    ψks = Array{M,3}(undef, 0, 0, 0) # Temporary, to be reassigned

    lenψ = size(matrix´, 1)
    nk = nvertices(mesh)
    # function to apply to eigenvalues when building bands (depends on momenta type)
    by = _maybereal(T)

    p = Progress(nk, "Step 1/2 - Diagonalising: ")
    for (n, ϕs) in enumerate(vertices(mesh))
        matrix = matrixf(Tuple(ϕs))
        # (ϵk, ψk) = diagonalize(Hermitian(matrix), d)  ## This is faster (!)
        (ϵk, ψk) = diagonalize(matrix, d.method)
        resolve_degeneracies!(ϵk, ψk, ϕs, d.codiag)
        if n == 1  # With first vertex can now know the number of eigenvalues... Reassign
            nϵ = size(ϵk, 1)
            ϵks = Matrix{T}(undef, nϵ, nk)
            ψks = Array{M,3}(undef, lenψ, nϵ, nk)
        end
        copyslice!(ϵks, CartesianIndices((1:nϵ, n:n)),
                   ϵk,  CartesianIndices((1:nϵ,)), by)
        copyslice!(ψks, CartesianIndices((1:lenψ, 1:nϵ, n:n)),
                   ψk,  CartesianIndices((1:lenψ, 1:nϵ)))
        ProgressMeter.next!(p; showvalues = ())
    end

    p = Progress(nϵ * nk, "Step 2/2 - Connecting bands: ")
    pcounter = 0
    bands = Band{M,Vector{M},Mesh{D+1,T,Vector{SVector{D+1,T}}},Vector{NTuple{D+1,Int}}}[]
    vertindices = zeros(Int, nϵ, nk) # 0 == unclassified, -1 == different band, > 0 vertex index
    pending = Tuple{Int,CartesianIndex{2}}[] # (originating vertex index, (ϵ, k))
    dests = Int[]; srcs = Int[]       # To build adjacency matrices
    sizehint!(pending, nk)
    while true
        src = findfirst(iszero, vertindices)
        src === nothing && break
        resize!(pending, 1)
        resize!(dests, 0)
        resize!(srcs, 0)
        pending[1] = (0, src) # source CartesianIndex for band search, with no originating vertex
        band = extractband(mesh, ϵks, ψks, vertindices, d.minoverlap, pending, dests, srcs)
        nverts = nvertices(band.mesh)
        nverts > D && push!(bands, band) # avoid bands with no simplices
        pcounter += nverts
        ProgressMeter.update!(p, pcounter; showvalues = ())
    end
    return Bandstructure(bands, mesh)
end

_maybereal(::Type{<:Complex}) = identity
_maybereal(::Type{<:Real}) = real

function extractband(kmesh::Mesh{D,T}, ϵks::AbstractArray{T}, ψks::AbstractArray{M}, vertindices, minoverlap, pending, dests, srcs) where {D,T,M}
    lenψ, nϵ, nk = size(ψks)
    kverts = vertices(kmesh)
    states = eltype(ψks)[]
    sizehint!(states, nk * lenψ)
    verts = SVector{D+1,T}[]
    lenverts = 0
    sizehint!(verts, nk)
    adjmat = SparseMatrixBuilder{Bool}()
    srcidx = 0  # represents the index of the last added vertex (used to search for the nexts)
    while !isempty(pending)
        origin, src = pop!(pending) # origin is the vertex index that originated this src, 0 if none (first)
        ϵ, k = Tuple(src) # src == CartesianIndex(ϵ::Int, k::Int)
        vertex = vcat(kverts[k], SVector(ϵks[src]))
        push!(verts, vertex)
        srcidx += 1 # Always equals length(verts)
        vertindices[ϵ, k] = srcidx
        appendslice!(states, ψks, CartesianIndices((1:lenψ, ϵ:ϵ, k:k)))
        if origin != 0
            append!(dests, (origin, srcidx))
            append!(srcs, (srcidx, origin))
        end
        added_vertices = 0
        for edgek in edges(kmesh, k)
            k´ = edgedest(kmesh, edgek)
            proj, ϵ´ = findmostparallel(ψks, k´, ϵ, k)
            # if unclassified and sufficiently parallel add it to pending list
            if proj >= minoverlap && iszero(vertindices[ϵ´, k´])
                push!(pending, (srcidx, CartesianIndex(ϵ´, k´)))
                added_vertices += 1
            end
        end
        # In 1D we avoid backsteps, to keep nicely continuous bands
        D == 1 && added_vertices == 0 && break
    end
    for (i, vi) in enumerate(vertindices)
        @inbounds vi > 0 && (vertindices[i] = -1) # mark as classified in a different band
    end
    adjmat = sparse(dests, srcs, true)
    mesh = Mesh(verts, adjmat)
    return Band(mesh, states, lenψ)
end

function findmostparallel(ψks::Array{M,3}, destk, srcb, srck) where {M}
    T = real(eltype(M))
    dimh, nϵ, nk = size(ψks)
    maxproj = zero(T)
    destb = 0
    @inbounds for nb in 1:nϵ
        proj = zero(M)
        for i in 1:dimh
            proj += ψks[i, nb, destk]' * ψks[i, srcb, srck]
        end
        absproj = T(abs(tr(proj)))
        if maxproj <= absproj  # must happen at least once
            destb = nb
            maxproj = absproj
        end
    end
    return maxproj, destb
end