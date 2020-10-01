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
        alignnormals!(band.simplices, vs)
    end
    return bs
end

#######################################################################
# bandstructure
#######################################################################
"""
    bandstructure(h::Hamiltonian; points = 13, kw...)

Compute `bandstructure(h, mesh((-π,π)...; points = points); kw...)` using a mesh over `h`'s
full Brillouin zone with the specified `points` along each [-π,π] reciprocal axis.

    bandstructure(h::Hamiltonian, nodes...; points = 13, kw...)

Create a linecut of a bandstructure of `h` along a polygonal line connecting two or more
`nodes`. Each node is either a `Tuple` or `SVector` of Bloch phases, or a symbolic name for
a Brillouin zone point (`:Γ`,`:K`, `:K´`, `:M`, `:X`, `:Y` or `:Z`). Each segment in the
polygon has the specified number of `points`. Different `points` per segments can be
specified with `points = (p1, p2...)`.

    bandstructure(h::Hamiltonian, mesh::Mesh; mapping = missing, kw...)

Compute the bandstructure `bandstructure(h, mesh; kw...)` of Bloch Hamiltonian `bloch(h,
ϕ)`, with `ϕ = v` taken on each vertex `v` of `mesh` (or `ϕ = mapping(v...)` if a `mapping`
function is provided).

    bandstructure(ph::ParametricHamiltonian, ...; kw...)

Compute the bandstructure of a `ph`. Unless all parameters have default values, a `mapping`
is required between mesh vertices and Bloch/parameters for `ph`, see details on `mapping`
below.

    bandstructure(matrixf::Function, mesh::Mesh; kw...)

Compute the bandstructure of the Hamiltonian matrix `m = matrixf(ϕ)`, with `ϕ` evaluated on
the vertices `v` of the `mesh`. Note that `ϕ` in `matrixf(ϕ)` is an unsplatted container.
Hence, i.e. `matrixf(x) = ...` or `matrixf(x, y) = ...` will not work. Use `matrixf((x,)) =
...`, `matrixf((x, y)) = ...` or matrixf(s::SVector) = ...` instead.

    h |> bandstructure([mesh,]; kw...)

Curried form of the above equivalent to `bandstructure(h, [mesh]; kw...)`.

# Options

The default options are

    (mapping = missing, minoverlap = 0, method = LinearAlgebraPackage(), transform = missing)

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

# Examples
```jldoctest
julia> h = LatticePresets.honeycomb() |> hamiltonian(hopping(-1)) |> unitcell(3);

julia> bandstructure(h; points = 25, method = LinearAlgebraPackage())
Bandstructure{2}: collection of 2D bands
  Bands        : 8
  Element type : scalar (Complex{Float64})
  Mesh{2}: mesh of a 2-dimensional manifold
    Vertices   : 625
    Edges      : 1776

julia> bandstructure(h, :Γ, :X, :Y, :Γ; points = (10,15,10))
Bandstructure{1}: collection of 1D bands
  Bands        : 18
  Element type : scalar (Complex{Float64})
  Mesh{1}: mesh of a 1-dimensional manifold
    Vertices   : 33
    Edges      : 32

julia> bandstructure(h, mesh((0, 2π); points = 11); mapping = φ -> (φ, 0))
       # Equivalent to bandstructure(h, :Γ, :X; points = 11)
Bandstructure{1}: collection of 1D bands
  Bands        : 18
  Element type : scalar (Complex{Float64})
  Mesh{1}: mesh of a 1-dimensional manifold
    Vertices   : 11
    Edges      : 10

julia> ph = parametric(h, @hopping!((t; α) -> t * α));

julia> bandstructure(ph, mesh((0, 2π); points = 11); mapping = φ -> (φ, 0, (; α = 2φ)))
Bandstructure{1}: collection of 1D bands
  Bands        : 18
  Element type : scalar (Complex{Float64})
  Mesh{1}: mesh of a 1-dimensional manifold
    Vertices   : 11
    Edges      : 10
```

# See also
    `mesh`, `bloch`, `parametric`
"""
function bandstructure(h::Hamiltonian{<:Any, L}; points = 13, kw...) where {L}
    m = mesh(filltuple((-π, π), Val(L))...; points = points)
    return bandstructure(h, m; kw...)
end

function bandstructure(h::Hamiltonian{<:Any,L}, node1, node2, nodes...; points = 13, transform = missing, kw...) where {L}
    allnodes = (node1, node2, nodes...)
    mapping´ = piecewise_mapping(allnodes, Val(L))
    transform´ = sanitize_transform(transform, h, allnodes)
    return bandstructure(h, piecewise_mesh(allnodes, points); mapping = mapping´, transform = transform´, kw...)
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

function bandstructure(h::Union{Hamiltonian,ParametricHamiltonian}, mesh::Mesh;
                       method = LinearAlgebraPackage(), minoverlap = 0, mapping = missing, transform = missing)
    # ishermitian(h) || throw(ArgumentError("Hamiltonian must be hermitian"))
    matrix = similarmatrix(h, method_matrixtype(method, h))
    diag = diagonalizer(h, matrix, mesh, method, minoverlap, mapping)
    matrixf(ϕsmesh) = bloch!(matrix, h, map_phiparams(mapping, ϕsmesh))
    b = _bandstructure(matrixf, matrix, mesh, diag)
    if transform !== missing
        transform´ = sanitize_transform(transform, h)
        transform!(transform´, b)
    end
    return b
end

function bandstructure(matrixf::Function, mesh::Mesh;
                       method = LinearAlgebraPackage(),  minoverlap = 0, mapping = missing, transform = missing)
    matrixf´ = wrapmapping(mapping, matrixf)
    matrix = _samplematrix(matrixf´, mesh)
    diag = diagonalizer(matrixf´, matrix, mesh, method, minoverlap, missing)
    b = _bandstructure(matrixf´, matrix, mesh, diag)
    transform === missing || transform!(transform, b)
    return b
end
@inline map_phiparams(mapping::Missing, ϕsmesh) = sanitize_phiparams(ϕsmesh)
@inline map_phiparams(mapping::Function, ϕsmesh) = sanitize_phiparams(mapping(ϕsmesh...))

wrapmapping(mapping::Missing, matrixf::Function) = matrixf
wrapmapping(mapping::Function, matrixf::Function) = ϕsmesh -> matrixf(toSVector(mapping(ϕsmesh...)))

sanitize_transform(::Missing, args...) = (identity, identity)
sanitize_transform(f::Function, args...) = (identity, f)
sanitize_transform(f::typeof(isometric), args...) = (isometric(args...), identity)
sanitize_transform((_,f)::Tuple{typeof(isometric),Function}, args...) = (isometric(args...), f)
sanitize_transform(fs::Tuple{Function,Function}, args...) = fs
sanitize_transform((_,f)::Tuple{Missing,Function}, args...) = (identity, f)
sanitize_transform((f,_)::Tuple{Function,Missing}, args...) = (f, identity)

_samplematrix(matrixf, mesh) = matrixf(Tuple(first(vertices(mesh))))

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
        band = extract_band(mesh, ϵks, ψks, vertindices, d.minoverlap, pending, dests, srcs)
        nverts = nvertices(band.mesh)
        nverts > D && push!(bands, band) # avoid bands with no simplices
        pcounter += nverts
        ProgressMeter.update!(p, pcounter; showvalues = ())
    end
    return Bandstructure(bands, mesh)
end

_maybereal(::Type{<:Complex}) = identity
_maybereal(::Type{<:Real}) = real

#######################################################################
# extract_band
#######################################################################

function extract_band(kmesh::Mesh{D,T}, ϵks::AbstractArray{T}, ψks::AbstractArray{M}, vertindices, minoverlap, pending, dests, srcs) where {D,T,M}
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
        srcidx = vertindices[src]
        if srcidx != 0
            append_adjacent!(dests, srcs, origin, srcidx)
            continue
        end
        ϵ, k = Tuple(src) # src == CartesianIndex(ϵ::Int, k::Int)
        vertex = vcat(kverts[k], SVector(ϵks[src]))
        push!(verts, vertex)
        srcidx = length(verts)
        vertindices[ϵ, k] = srcidx
        append_slice!(states, ψks, CartesianIndices((1:lenψ, ϵ:ϵ, k:k)))
        append_adjacent!(dests, srcs, origin, srcidx)
        added_vertices = 0
        for edgek in edges(kmesh, k)
            k´ = edgedest(kmesh, edgek)
            proj, ϵ´ = findmostparallel(ψks, k´, ϵ, k)
            # if unclassified and sufficiently parallel add it to pending list
            if proj >= minoverlap && !iszero(ϵ´) && iszero(vertindices[ϵ´, k´])
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

function append_adjacent!(dests, srcs, origin, srcidx)
    if origin != 0 && srcidx != 0
        append!(dests, (origin, srcidx))
        append!(srcs, (srcidx, origin))
    end
    return nothing
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

#######################################################################
# resolve_degeneracies
#######################################################################
# Tries to make states continuous at crossings. Here, ϵ needs to be sorted
function resolve_degeneracies!(ϵ, ψ, ϕs, codiag)
    issorted(ϵ, by = real) || sorteigs!(codiag.perm, ϵ, ψ)
    hasapproxruns(ϵ, codiag.degtol) || return ϵ, ψ
    ranges, ranges´ = codiag.rangesA, codiag.rangesB
    resize!(ranges, 0)
    pushapproxruns!(ranges, ϵ, 0, codiag.degtol) # 0 is an offset
    for n in codiag.matrixindices
        v = codiag.comatrix(ϕs, n)
        resize!(ranges´, 0)
        for (i, r) in enumerate(ranges)
            subspace = view(ψ, :, r)
            vsubspace = subspace' * v * subspace
            veigen = eigen!(Hermitian(vsubspace))
            if hasapproxruns(veigen.values, codiag.degtol)
                roffset = minimum(r) - 1 # Range offset within the ϵ vector
                pushapproxruns!(ranges´, veigen.values, roffset, codiag.degtol)
            end
            subspace .= subspace * veigen.vectors
        end
        ranges, ranges´ = ranges´, ranges
        isempty(ranges) && break
    end
    return ψ
end

# Could perhaps be better/faster using a generalized CoSort
function sorteigs!(perm, ϵ::Vector, ψ::Matrix)
    resize!(perm, length(ϵ))
    p = sortperm!(perm, ϵ, by = real)
    # permute!(ϵ, p)
    sort!(ϵ, by = real)
    Base.permutecols!!(ψ, p)
    return ϵ, ψ
end
