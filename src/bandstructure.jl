#######################################################################
# Spectrum
#######################################################################
struct Spectrum{E,T,A<:AbstractMatrix{T}}
    energies::E
    states::A
end

"""
    spectrum(h; method = defaultmethod(h))

Compute the spectrum of a 0D Hamiltonian `h` (or alternatively of the bounded unit cell of a
finite dimensional `h`) using one of the following `method`s

    method                    diagonalization function
    --------------------------------------------------------------
    LinearAlgebraPackage()     LinearAlgebra.eigen!
    ArpackPackage()            Arpack.eigs (must be `using Arpack`)

The energies and eigenstates in the resulting `s::Spectrum` object can be accessed with
`energies(s)` and `states(s)`

# See also
    `energies`, `states`, `bandstructure`

"""
function spectrum(h; method = defaultmethod(h))
    matrix = similarmatrix(h, method)
    bloch!(matrix, h)
    (ϵk, ψk) = diagonalize(matrix, method)
    return Spectrum(ϵk, ψk)
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
    ioindent = IOContext(io, :indent => string("  "))
    print(io,
"Bandstructure: bands for a $(D)D hamiltonian
  Bands        : $(length(b.bands))
  Element type : $(displayelements(M))")
    print(ioindent, "\n", b.kmesh)
end

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
    states(bs::Bandstructure, i)

Return the states of each vertex of the i-th band in `bs`, in the form of a `Matrix` of size
`(nψ, nk)`, where `nψ` is the length of each state vector, and `nk` the number of vertices.
"""
states(bs::Bandstructure, i) = states(bands(bs)[i])

states(b::Band) = reshape(b.states, b.dimstates)

#######################################################################
# bandstructure
#######################################################################
"""
    bandstructure(h::Hamiltonian, mesh::Mesh; minprojection = 0.5, method = defaultmethod(h))

Compute the bandstructure of Bloch Hamiltonian `bloch(h, ϕs)`, with `ϕs` evaluated on the
vertices of `mesh`. It is assumed that `h` is hermitian.

    bandstructure(matrixf::Function, mesh::Mesh; minprojection = 0.5, method = defaultmethod(h))

Compute the bandstructure of the Hamiltonian matrix `matrixf(ϕs...)`, with `ϕs`
evaluated on the vertices of `mesh`. It is assumed that `h` is hermitian.

The option `minprojection` determines the minimum projection between eigenstates to connect
them into a common subband. The option `method` is chosen automatically if unspecified, and
can be one of the following

    method                    diagonalization function
    --------------------------------------------------------------
    LinearAlgebraPackage()     LinearAlgebra.eigen!
    ArpackPackage()            Arpack.eigs (must be `using Arpack`)

Options passed to the `method` will be forwarded to the diagonalization function. For example,
`method = ArpackPackage(nev = 8, sigma = 1im)` will use `Arpack.eigs(matrix; nev = 8,
sigma = 1im)` to compute the bandstructure.

    bandstructure(h::Hamiltonian; npoints = 13, shift = missing, kw...)

Same as above with a  uniform `mesh = marchingmesh(; npoints = npoints, shift = shift)`
of marching tetrahedra (generalized to the lattice dimensions of the Hamiltonian). Note that
`npoints` denotes the number of points along each Bloch axis, including endpoints (can be
a tuple for axis-dependent points). If `npoints` is a single Int, it is first converted to a
homogeneous `NTuple{L,Int}` (same npoints for all Bloch axes).

    bandstructure(matrixf::Function; npoints, shift = missing, kw...)

Same as above, but where `npoints` must be specified (the dimension of the mesh cannot be
inferred from `matrixf`). If `matrixf(ϕs...)` has `L` arguments (i.e. `ϕs` are vertices of a
`Mesh{L}`), then use `npoints::NTuple{L,Int}` (or optionally `npoints::Int` for `L==1`).

# Example
```
julia> h = LatticePresets.honeycomb() |> unitcell(3) |> hamiltonian(hopping(-1, range = 1/√3));

julia> bandstructure(h; npoints = 25, method = LinearAlgebraPackage())
Bandstructure: bands for a 2D hamiltonian
  Bands        : 8
  Element type : scalar (Complex{Float64})
  Mesh{2}: mesh of a 2-dimensional manifold
    Vertices   : 625
    Edges      : 3552
```

# See also
    marchingmesh
"""
function bandstructure(h::Hamiltonian{<:Lattice,L}; npoints = 13, shift = missing, kw...) where {L}
    checkfinitedim(h)
    npoints´ = _npoints(npoints, Val(L))
    mesh = marchingmesh(npoints´...; shift = shift)
    return bandstructure(h, mesh; kw...)
end

_npoints(npoints::Int, ::Val{L}) where {L} = filltuple(npoints, Val(L))
_npoints(npoints::NTuple{L,Int}, ::Val{L}) where {L} = npoints

function bandstructure(h::Hamiltonian{<:Lattice,L,M}, mesh::Mesh{L};
                       method = defaultmethod(h), minprojection = 0.5) where {L,L´,M}
    # ishermitian(h) || throw(ArgumentError("Hamiltonian must be hermitian"))
    matrix = similarmatrix(h, method)
    d = Diagonalizer(method, codiagonalizer(h, matrix, mesh), minprojection)
    matrixf(φs...) = bloch!(matrix, h, φs)
    return _bandstructure(matrixf, matrix, mesh, d)
end

function bandstructure(matrixf::Function; npoints = _need_npoints_error(), shift = missing, kw...)
    mesh = marchingmesh(; npoints = npoints, shift = shift)
    return bandstructure(matrixf, mesh; kw...)
end

function bandstructure(matrixf::Function, mesh::Mesh;
                       matrix = _samplematrix(matrixf, mesh),
                       method = defaultmethod(matrix), minprojection = 0.5)
    d = Diagonalizer(method, codiagonalizer(matrixf, matrix, mesh), minprojection)
    return _bandstructure(matrixf, matrix, mesh, d)
end

_need_npoints_error() =
    throw(ArgumentError("Please specify `npoints::NTuple{L,Int}`, where `L` is the number of matrix function arguments"))

_samplematrix(matrixf, mesh) = matrixf(Tuple(first(vertices(mesh)))...)

function _bandstructure(matrixf::Function, matrix´::AbstractMatrix{M}, mesh::MD, d::Diagonalizer) where {M,D,T,MD<:Mesh{D,T}}
    nϵ = 0                           # Temporary, to be reassigned
    ϵks = Matrix{T}(undef, 0, 0)     # Temporary, to be reassigned
    ψks = Array{M,3}(undef, 0, 0, 0) # Temporary, to be reassigned

    dimh = size(matrix´, 1)
    nk = nvertices(mesh)
    # function to apply to eigenvalues when building bands (depends on momenta type)
    by = _maybereal(T)

    p = Progress(nk, "Step 1/2 - Diagonalising: ")
    for (n, ϕs) in enumerate(vertices(mesh))
        matrix = matrixf(Tuple(ϕs)...)
        # (ϵk, ψk) = diagonalize(Hermitian(matrix), d)  ## This is faster (!)
        (ϵk, ψk) = diagonalize(matrix, d.method)
        resolve_degeneracies!(ϵk, ψk, ϕs, matrix, d.codiag)
        if n == 1  # With first vertex can now know the number of eigenvalues... Reassign
            nϵ = size(ϵk, 1)
            ϵks = Matrix{T}(undef, nϵ, nk)
            ψks = Array{M,3}(undef, dimh, nϵ, nk)
        end
        copyslice!(ϵks, CartesianIndices((1:nϵ, n:n)),
                   ϵk,  CartesianIndices((1:nϵ,)), by)
        copyslice!(ψks, CartesianIndices((1:dimh, 1:nϵ, n:n)),
                   ψk,  CartesianIndices((1:dimh, 1:nϵ)), identity)
        ProgressMeter.next!(p; showvalues = ())
    end

    p = Progress(nϵ * nk, "Step 2/2 - Connecting bands: ")
    pcounter = 0
    bands = Band{M,Vector{M},Mesh{D+1,T,Vector{SVector{D+1,T}}},Vector{NTuple{D+1,Int}}}[]
    vertindices = zeros(Int, nϵ, nk) # 0 == unclassified, -1 == different band, > 0 vertex index
    pending = CartesianIndex{2}[]
    sizehint!(pending, nk)
    while true
        src = findfirst(iszero, vertindices)
        src === nothing && break
        resize!(pending, 1)
        pending[1] = src # source CartesianIndex for band search
        band = extractband(mesh, pending, ϵks, ψks, vertindices, d.minprojection)
        nverts = nvertices(band.mesh)
        nverts > D && push!(bands, band) # avoid bands with no simplices
        pcounter += nverts
        ProgressMeter.update!(p, pcounter; showvalues = ())
    end
    return Bandstructure(bands, mesh)
end

_maybereal(::Type{<:Complex}) = identity
_maybereal(::Type{<:Real}) = real

function extractband(kmesh::Mesh{D,T}, pending, ϵks::AbstractArray{T}, ψks::AbstractArray{M}, vertindices, minprojection) where {D,T,M}
    dimh, nϵ, nk = size(ψks)
    kverts = vertices(kmesh)
    states = eltype(ψks)[]
    sizehint!(states, nk * dimh)
    verts = SVector{D+1,T}[]
    sizehint!(verts, nk)
    adjmat = SparseMatrixBuilder{Bool}()
    vertindices[first(pending)] = 1 # pending starts with a single vertex
    for c in pending
        ϵ, k = Tuple(c) # c == CartesianIndex(ϵ::Int, k::Int)
        vertex = vcat(kverts[k], SVector(ϵks[c]))
        push!(verts, vertex)
        appendslice!(states, ψks, CartesianIndices((1:dimh, ϵ:ϵ, k:k)))
        for edgek in edges(kmesh, k)
            k´ = edgedest(kmesh, edgek)
            proj, ϵ´ = findmostparallel(ψks, k´, ϵ, k)
            if proj >= minprojection
                if iszero(vertindices[ϵ´, k´]) # unclassified
                    push!(pending, CartesianIndex(ϵ´, k´))
                    vertindices[ϵ´, k´] = length(pending) # this is clever!
                end
                indexk´ = vertindices[ϵ´, k´]
                indexk´ > 0 && pushtocolumn!(adjmat, indexk´, true)
            end
        end
        finalizecolumn!(adjmat)
    end
    for (i, vi) in enumerate(vertindices)
        @inbounds vi > 0 && (vertindices[i] = -1) # mark as classified in a different band
    end
    mesh = Mesh(verts, sparse(adjmat))
    return Band(mesh, states, dimh)
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