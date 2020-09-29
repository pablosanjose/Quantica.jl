#######################################################################
# Hamiltonian
#######################################################################
struct HamiltonianHarmonic{L,M,A<:Union{AbstractMatrix{M},SparseMatrixBuilder{M}}}
    dn::SVector{L,Int}
    h::A
end

HamiltonianHarmonic{L,M,A}(dn::SVector{L,Int}, n::Int, m::Int) where {L,M,A<:SparseMatrixCSC{M}} =
    HamiltonianHarmonic(dn, sparse(Int[], Int[], M[], n, m))

HamiltonianHarmonic{L,M,A}(dn::SVector{L,Int}, n::Int, m::Int) where {L,M,A<:Matrix{M}} =
    HamiltonianHarmonic(dn, zeros(M, n, m))

struct Hamiltonian{LA<:AbstractLattice,L,M,A<:AbstractMatrix,B<:AbstractMatrix,
                   H<:HamiltonianHarmonic{L,M,A},
                   O<:Tuple{Vararg{Tuple{Vararg{NameType}}}}} # <: AbstractMatrix{M}
    lattice::LA
    harmonics::Vector{H}
    orbitals::O
    blochmatrix::B
end

function Hamiltonian(lat, hs::Vector{H}, orbs, n::Int, m::Int, blochtype = missing) where {L,M,H<:HamiltonianHarmonic{L,M}}
    sort!(hs, by = h -> abs.(h.dn))
    if isempty(hs) || !iszero(first(hs).dn)
        pushfirst!(hs, H(zero(SVector{L,Int}), empty_sparse(M, n, m)))
    end
    blochmatrix = similarmatrix(blochtype, hs, orbs, lat)
    return Hamiltonian(lat, hs, orbs, blochmatrix)
end

Base.show(io::IO, ham::Hamiltonian) = show(io, MIME("text/plain"), ham)
function Base.show(io::IO, ::MIME"text/plain", ham::Hamiltonian)
    i = get(io, :indent, "")
    print(io, i, summary(ham), "\n",
"$i  Orbitals         : $(displayorbitals(ham))
$i  Site eltype      : $(displayelements(ham))
$i  Bloch eltype     : $(displayelements(ham.blochmatrix))
$i  Bloch matrix     : $(displaymatrixtype(typeof(ham.blochmatrix)))
$i  Bloch harmonics  : $(length(ham.harmonics))
$i  Harmonic size    : $((n -> "$n × $n")(nsites(ham)))
$i  Onsites          : $(nonsites(ham))
$i  Hoppings         : $(nhoppings(ham))
$i  Coordination     : $(nhoppings(ham) / nsites(ham))")
    ioindent = IOContext(io, :indent => string("  "))
    issuperlattice(ham.lattice) && print(ioindent, "\n", ham.lattice.supercell)
end

Base.summary(h::Hamiltonian{LA}) where {E,L,LA<:Lattice{E,L}} =
    "Hamiltonian{<:Lattice} : Hamiltonian on a $(L)D Lattice in $(E)D space"

Base.summary(::Hamiltonian{LA}) where {E,L,T,L´,LA<:Superlattice{E,L,T,L´}} =
    "Hamiltonian{<:Superlattice} : $(L)D Hamiltonian on a $(L´)D Superlattice in $(E)D space"

Base.eltype(::Hamiltonian{<:Any,<:Any,M}) where {M} = M

Base.isequal(h1::HamiltonianHarmonic, h2::HamiltonianHarmonic) =
    h1.dn == h2.dn && h1.h == h2.h

displaymatrixtype(::Type{<:SparseMatrixCSC}) = "SparseMatrixCSC (sparse)"
displaymatrixtype(::Type{<:Array}) = "Matrix (dense)"
displaymatrixtype(A::Type{<:AbstractArray}) = string(A)
displayelements(h::Hamiltonian) = displayelements(blocktype(h))
displayelements(m::AbstractMatrix) = displayelements(eltype(m))
displayelements(::Type{S}) where {N,T,S<:SMatrix{N,N,T}} = "$N × $N blocks ($T)"
displayelements(::Type{T}) where {T} = "scalar ($T)"

displayorbitals(h::Hamiltonian) =
    replace(replace(string(h.orbitals), "Symbol(\"" => ":"), "\")" => "")

#######################################################################
# flatten
#######################################################################
"""
    flatten(h::Hamiltonian)

Flatten a multiorbital Hamiltonian `h` into one with a single orbital per site. The
associated lattice is flattened also, so that there is one site per orbital for each initial
site (all at the same position). Note that in the case of sparse Hamiltonians, zeros in
hopping/onsite matrices are preserved as structural zeros upon flattening.

    h |> flatten()

Curried form equivalent to `flatten(h)` or `h |> flatten` (included for consistency with
the rest of the API).

# Examples

```jldoctest
julia> h = LatticePresets.honeycomb() |>
           hamiltonian(hopping(@SMatrix[1; 2], range = 1/√3, sublats = :A =>:B),
           orbitals = (Val(1), Val(2)))
Hamiltonian{<:Lattice} : Hamiltonian on a 2D Lattice in 2D space
  Bloch harmonics  : 3 (SparseMatrixCSC, sparse)
  Harmonic size    : 2 × 2
  Orbitals         : ((:a,), (:a, :a))
  Site eltype      : 2 × 2 blocks (Complex{Float64})
  Onsites          : 0
  Hoppings         : 3
  Coordination     : 1.5

julia> flatten(h)
Hamiltonian{<:Lattice} : Hamiltonian on a 2D Lattice in 2D space
  Bloch harmonics  : 3 (SparseMatrixCSC, sparse)
  Harmonic size    : 3 × 3
  Orbitals         : ((:flat,), (:flat,))
  Site eltype      : scalar (Complex{Float64})
  Onsites          : 0
  Hoppings         : 6
  Coordination     : 2.0
```
"""
flatten() = h -> flatten(h)

function flatten(h::Hamiltonian)
    all(isequal(1), norbitals(h)) && return copy(h)
    harmonics´ = [flatten(har, h.orbitals, h.lattice) for har in h.harmonics]
    lattice´ = flatten(h.lattice, h.orbitals)
    orbitals´ = (_ -> (:flat, )).(h.orbitals)
    n = nsites(lattice´)
    return Hamiltonian(lattice´, harmonics´, orbitals´, n, n, matrixtype(harmonics´))
end

flatten(h::HamiltonianHarmonic, orbs, lat) = HamiltonianHarmonic(h.dn, flatten(h.h, orbs, lat))

function flatten(src::SparseMatrixCSC{<:SMatrix{N,N,T}}, orbs, lat, ::Type{T´} = T) where {N,T,T´}
    norbs = length.(orbs)
    offsets´ = flatoffsets(lat.unitcell.offsets, norbs)
    dim´ = last(offsets´)

    builder = SparseMatrixBuilder{T´}(dim´, dim´, nnz(src) * N * N)

    for col in 1:size(src, 2)
        scol = sublat(lat, col)
        for j in 1:norbs[scol]
            for p in nzrange(src, col)
                row = rowvals(src)[p]
                srow = sublat(lat, row)
                rowoffset´ = flatoffset(row, lat, norbs, offsets´)
                val = nonzeros(src)[p]
                for i in 1:norbs[srow]
                    pushtocolumn!(builder, rowoffset´ + i, val[i, j])
                end
            end
            finalizecolumn!(builder, false)
        end
    end
    matrix = sparse(builder)
    return matrix
end

function flatten(src::DenseMatrix{<:SMatrix{N,N,T}}, orbs, lat, ::Type{T´} = T) where {N,T,T´}
    norbs = length.(orbs)
    offsets´ = flatoffsets(lat.unitcell.offsets, norbs)
    dim´ = last(offsets´)
    matrix = similar(src, T´, dim´, dim´)

    for col in 1:size(src, 2), row in 1:size(src, 1)
        srow, scol = sublat(lat, row), sublat(lat, col)
        nrow, ncol = norbs[srow], norbs[scol]
        val = src[row, col]
        rowoffset´ = flatoffset(row, lat, norbs, offsets´)
        coloffset´ = flatoffset(col, lat, norbs, offsets´)
        for j in 1:ncol, i in 1:nrow
            matrix[rowoffset´ + i, coloffset´ + j] = val[i, j]
        end
    end
    return matrix
end

flatten(src::AbstractMatrix{<:Number}, x...) = src

function flatten(lat::Lattice, orbs)
    length(orbs) == nsublats(lat) || throw(ArgumentError("Msmatch between sublattices and orbitals"))
    unitcell´ = flatten(lat.unitcell, orbs)
    bravais´ = lat.bravais
    lat´ = Lattice(bravais´, unitcell´)
end

function flatten(unitcell::Unitcell, orbs::NTuple{S,Any}) where {S}
    norbs = length.(orbs)
    offsets´ = [flatoffsets(unitcell.offsets, norbs)...]
    ns´ = last(offsets´)
    sites´ = similar(unitcell.sites, ns´)
    i = 1
    for sl in 1:S, site in sitepositions(unitcell, sl), rep in 1:norbs[sl]
        sites´[i] = site
        i += 1
    end
    names´ = unitcell.names
    unitcell´ = Unitcell(sites´, names´, offsets´)
    return unitcell´
end

#######################################################################
# similarmatrix
#######################################################################

similarmatrix(dest_type, hs, orbs, lat) = _similarmatrix(dest_type, matrixtype(hs), hs, orbs, lat)

_similarmatrix(::Missing, src_type, hs, orbs, lat) =
    similar_merged(hs)
_similarmatrix(::Type{A}, ::Type{A´}, hs, orbs, lat) where {T<:Number,A<:AbstractSparseMatrix{T},T´<:Number,A´<:AbstractSparseMatrix{T´}} =
    similar_merged(hs, T)
_similarmatrix(::Type{A}, ::Type{A´}, hs, orbs, lat) where {N,T<:SMatrix{N,N},A<:AbstractSparseMatrix{T},T´<:SMatrix{N,N},A´<:AbstractSparseMatrix{T´}} =
    similar_merged(hs, T)
_similarmatrix(::Type{A}, ::Type{A´}, hs, orbs, lat) where {N,T<:Number,A<:AbstractSparseMatrix{T},T´<:SMatrix{N,N},A´<:AbstractSparseMatrix{T´}} =
    flatten(similar_merged(hs), orbs, lat, T)
_similarmatrix(::Type{A}, ::Type{A´}, hs, orbs, lat) where {T<:Number,A<:Matrix{T},T´<:Number,A´<:AbstractMatrix{T´}} =
    similar(A, size(first(hs).h))
_similarmatrix(::Type{A}, ::Type{A´}, hs, orbs, lat) where {N,T<:SMatrix{N,N},A<:Matrix{T},T´<:SMatrix{N,N},A´<:AbstractMatrix{T´}} =
    similar(A, size(first(hs).h))
_similarmatrix(::Type{A}, ::Type{A´}, hs, orbs, lat) where {N,T<:Number,A<:Matrix{T},T´<:SMatrix{N,N},A´<:AbstractMatrix{T´}} =
    similar(A, _flatsize(lat, orbs))

_similarmatrix(::typeof(flatten), ::Type{A´}, hs, orbs, lat) where {N,T,S<:SMatrix{N,N,T},A´<:AbstractSparseMatrix{S}} =
    _similarmatrix(AbstractSparseMatrix{T}, A´, hs, orbs, lat)
_similarmatrix(::typeof(flatten), ::Type{A´}, hs, orbs, lat) where {N,T,S<:SMatrix{N,N,T},A´<:DenseMatrix{S}} =
    _similarmatrix(Matrix{T}, A´, hs, orbs, lat)
_similarmatrix(::typeof(flatten), ::Type{A´}, hs, orbs, lat) where {T<:Number,A´<:AbstractArray{T}} =
    _similarmatrix(A´, A´, hs, orbs, lat)
_similarmatrix(::Type{SparseMatrixCSC}, ::Type{A´}, hs, orbs, lat) where {T´,A´<:AbstractMatrix{T´}} =
    _similarmatrix(SparseMatrixCSC{T´}, A´, hs, orbs, lat)
_similarmatrix(::Type{Matrix}, ::Type{A´}, hs, orbs, lat) where {T´,A´<:AbstractMatrix{T´}} =
    _similarmatrix(Matrix{T´}, A´, hs, orbs, lat)

_similarmatrix(dest_type, src_type, hs, orbs, lat) = throw(ArgumentError("Unexpected `blochtype` ($src_type => $dest_type)"))

# ensure we have stored entries for all harmonics in the sparse case
similar_merged(hs::AbstractVector{H}, ::Type{M´} = M) where {M´,M,H<:HamiltonianHarmonic{<:Any,M,<:AbstractSparseMatrix}} =
    _similar_merged_sparse(hs, M´)
similar_merged(hs::AbstractVector{H}, ::Type{M´} = M) where {M´,M,H<:HamiltonianHarmonic{<:Any,M,<:DenseMatrix}} =
    similar(first(hs).h, M´)

function _similar_merged_sparse(hs::Vector{<:HamiltonianHarmonic{L,M}},::Type{M´} = M) where {L,M,M´}
    h0 = first(hs)
    n, m = size(h0.h)
    iszero(h0.dn) || throw(ArgumentError("First Hamiltonian harmonic is not the fundamental"))
    nh = length(hs)
    builder = SparseMatrixBuilder{M´}(n, m)
    for col in 1:m
        for i in eachindex(hs)
            h = hs[i].h
            for p in nzrange(h, col)
                v = i == 1 ? nonzeros(h)[p] : zero(M)
                row = rowvals(h)[p]
                pushtocolumn!(builder, row, v, false) # skips repeated rows
            end
        end
        pushtocolumn!(builder, col, zero(M), false) # if not present already, add structural zeros to diagonal
        finalizecolumn!(builder)
    end
    ho = sparse(builder)
    return ho
end
# IDEA: could sum and subtract all harmonics instead: sum(h->h.h, hs)
# Tested, it is slower

#######################################################################
# Hamiltonian API
#######################################################################

latdim(h::Hamiltonian) = last(dims(h.lattice))

matrixtype(::AbstractVector{H}) where {A,H<:HamiltonianHarmonic{<:Any,<:Any,A}} = A
matrixtype(h::Hamiltonian) = matrixtype(h.harmonics)

orbtype(::Hamiltonian{<:Any,<:Any,M}) where {M} = eltype(M)

# find SMatrix type that can hold all matrix elements between lattice sites
blocktype(orbs, type::Type{Tv}) where {Tv} =
    _blocktype(orbitaltype(orbs, Tv))
_blocktype(::Type{S}) where {N,Tv,S<:SVector{N,Tv}} = SMatrix{N,N,Tv,N*N}
_blocktype(::Type{S}) where {S<:Number} = S

blocktype(h::Hamiltonian{LA,L,M}) where {LA,L,M} = M

promote_blocktype(hs::Hamiltonian...) = promote_blocktype(blocktype.(hs)...)
promote_blocktype(s1::Type, s2::Type, ss::Type...) =
    promote_blocktype(promote_blocktype(s1, s2), ss...)
promote_blocktype(::Type{SMatrix{N1,N1,T1,NN1}}, ::Type{SMatrix{N2,N2,T2,NN2}}) where {N1,NN1,T1,N2,NN2,T2} =
    SMatrix{max(N1, N2), max(N1, N2), promote_type(T1, T2), max(NN1,NN2)}
promote_blocktype(T1::Type{<:Number}, T2::Type{<:Number}) = promote_type(T1, T2)
promote_blocktype(T::Type) = T

blockdim(h::Hamiltonian) = blockdim(blocktype(h))
blockdim(::Type{S}) where {N,S<:SMatrix{N,N}} = N
blockdim(::Type{T}) where {T<:Number} = 1

# find SVector type that can hold all orbital amplitudes in any lattice sites
orbitaltype(orbs, type::Type{Tv}) where {Tv} =
    _orbitaltype(SVector{1,Tv}, orbs...)
_orbitaltype(::Type{S}, ::NTuple{D,NameType}, os...) where {N,Tv,D,S<:SVector{N,Tv}} =
    (M = max(N,D); _orbitaltype(SVector{M,Tv}, os...))
_orbitaltype(t::Type{SVector{N,Tv}}) where {N,Tv} = t
_orbitaltype(t::Type{SVector{1,Tv}}) where {Tv} = Tv

orbitaltype(h::Hamiltonian{LA,L,M}) where {N,T,LA,L,M<:SMatrix{N,N,T}} = SVector{N,T}
orbitaltype(h::Hamiltonian{LA,L,M}) where {LA,L,M<:Number} = M

function nhoppings(ham::Hamiltonian)
    count = 0
    for h in ham.harmonics
        count += iszero(h.dn) ? (_nnz(h.h) - _nnzdiag(h.h)) : _nnz(h.h)
    end
    return count
end

function nonsites(ham::Hamiltonian)
    count = 0
    for h in ham.harmonics
        iszero(h.dn) && (count += _nnzdiag(h.h))
    end
    return count
end

_nnz(h::AbstractSparseMatrix) = count(!iszero, nonzeros(h)) # Does not include stored zeros
_nnz(h::DenseMatrix) = count(!iszero, h)

function _nnzdiag(s::SparseMatrixCSC)
    count = 0
    rowptrs = rowvals(s)
    nz = nonzeros(s)
    for col in 1:size(s,2)
        for ptr in nzrange(s, col)
            rowptrs[ptr] == col && (count += !iszero(nz[ptr]); break)
        end
    end
    return count
end
_nnzdiag(s::Matrix) = count(!iszero, s[i,i] for i in 1:minimum(size(s)))

Base.isequal(h1::Hamiltonian, h2::Hamiltonian) =
    isequal(h1.lattice, h2.lattice) && isequal(h1.harmonics, h2.harmonics) &&
    isequal(h1.orbitals, h2.orbitals)

SparseArrays.issparse(h::Hamiltonian{LA,L,M,A}) where {LA,L,M,A<:AbstractSparseMatrix} = true
SparseArrays.issparse(h::Hamiltonian{LA,L,M,A}) where {LA,L,M,A} = false

Base.parent(h::Hamiltonian) = h

# Dual numbers #

DualNumbers.Dual(h::Hamiltonian) = Hamiltonian(h.lattice, Dual.(h.harmonics), h.orbitals, dualmatrix(h.blochmatrix))

DualNumbers.Dual(h::HamiltonianHarmonic) = HamiltonianHarmonic(h.dn, dualmatrix(h.h))

# Iterators #

function nonzero_indices(h::Hamiltonian, rowrange = 1:size(h, 1), colrange = 1:size(h, 2))
    rowrange´ = rclamp(rowrange, 1:size(h, 1))
    colrange´ = rclamp(colrange, 1:size(h, 2))
    gen = ((har.dn, rowvals(har.h)[ptr], col)
                for har in h.harmonics
                for col in colrange´
                for ptr in nzrange_inrows(har.h, col, rowrange´)
                if !iszero(nonzeros(har.h)[ptr]))
    return gen
end

function nonzero_indices(har::HamiltonianHarmonic, rowrange = 1:size(h, 1), colrange = 1:size(h, 2))
    rowrange´ = rclamp(rowrange, 1:size(har, 1))
    colrange´ = rclamp(colrange, 1:size(har, 2))
    gen = ((rowvals(har.h)[ptr], col)
                for col in colrange´
                for ptr in nzrange_inrows(har.h, col, rowrange´)
                if !iszero(nonzeros(har.h)[ptr]))
    return gen
end

function nzrange_inrows(h, col, rowrange)
    ptrs = nzrange(h, col)
    rows = rowvals(h)
    ptrmin = first(ptrs)
    ptrmax = last(ptrs)

    for p in ptrs
        rows[p] in rowrange && break
        ptrmin = p + 1
    end

    if ptrmin < ptrmax
        for p in ptrmax:-1:ptrmin
            ptrmax = p
            rows[p] in rowrange && break
        end
    end

    return ptrmin:ptrmax
end

# External API #
"""
    hamiltonian(lat, model; orbitals = Val(1), orbtype = Complex{Tlat}, blochtype = flatten)

Create a `Hamiltonian` by applying `model::TighbindingModel` to the lattice `lat` (see
`hopping` and `onsite` for details on building tightbinding models).

# Keywords

The number of orbitals on each sublattice can be specified by the keyword `orbitals`
(otherwise all sublattices have one orbital by default).

The type `B` of Bloch matrices (produced from the Hamiltonian with `bloch`/`bloch!`) can be
specified by `blochtype = B`. Quantica tries to choose an adequate `B<:blochtype` for
typical `blochtype`s. The special syntax `blochtype = flatten` (default) is equivalent to
`blochtype = SparseMatrixCSC{T}`, where `T<:Number` is the Hamiltonian's numeric type, both
in the single-orbital and multiorbital Hamiltonians. The default Bloch matrix is thus always
flat (a numeric eltype is necessary for many diagonalization libraries). In contrast,
`blochtype = SparseMatrixCSC` is equivalent, but with `T<:SMatrix` in the multiorbital case,
preserving the orbital structure.

The following, and obvious combinations, are possible formats for the `orbitals` keyword:

    orbitals = :up               # all sublattices have 1 orbital named :up
    orbitals = Val(1)            # same as above, with automatic names (default)
    orbitals = ((:a, :b),)       # first sublattice has 2 orbital named :a, :b, the rest have one
    orbitals = (:a, :b, 3)       # all sublattices have 3 orbitals named :a and :b and :3
    orbitals = ((:a, :b), (:c,)) # first sublattice has 2 orbitals, second has one
    orbitals = ((:a, :b), :c)    # same as above
    orbitals = (Val(2), Val(1))  # same as above, with automatic names
    orbitals = (:A => (:a, :b), :D => :c) # sublattice :A has two orbitals, :D and rest have one
    orbitals = :D => Val(4)      # sublattice :D has four orbitals, rest have one

The matrix sizes of tightbinding `model` must match the orbitals specified. Internally, we
define a block size `N = max(num_orbitals)`. If `N = 1` (all sublattices with one orbital)
the the Hamiltonian element type is `orbtype`. Otherwise it is `SMatrix{N,N,orbtype}` blocks,
padded with the necessary zeros as required. Keyword `orbtype` is `Complex{Tlat}` by default,
where `Tlat` is the number type of `lat`.

    lat |> hamiltonian(model; kw...)

Curried form of `hamiltonian` equivalent to `hamiltonian(lat, model[, funcmodel]; kw...)`.

    hamiltonian(h::Hamiltonian; orbtype = missing, blochtype = missing)

Build a new Hamiltonian with different `orbtype` and/or `blochtype`, otherwise identical to h.

# Indexing

Indexing into a Hamiltonian `h` works as follows. Access the `HamiltonianHarmonic` matrix at
a given `dn::NTuple{L,Int}` with `h[dn]`. Assign `v` into element `(i,j)` of said matrix
with `h[dn][i,j] = v` or `h[dn, i, j] = v`. Broadcasting with vectors of indices `is` and
`js` is supported, `h[dn][is, js] = v_matrix`.

To add an empty harmonic with a given `dn::NTuple{L,Int}`, do `push!(h, dn)`. To delete it,
do `deleteat!(h, dn)`.

# Examples

```jldoctest
julia> h = hamiltonian(LatticePresets.honeycomb(), hopping(@SMatrix[1 2; 3 4], range = 1/√3), orbitals = Val(2))
Hamiltonian{<:Lattice} : Hamiltonian on a 2D Lattice in 2D space
  Bloch harmonics  : 5 (SparseMatrixCSC, sparse)
  Harmonic size    : 2 × 2
  Orbitals         : ((:a, :a), (:a, :a))
  Site eltype      : 2 × 2 blocks (Complex{Float64})
  Onsites          : 0
  Hoppings         : 6
  Coordination     : 3.0

julia> push!(h, (3,3)) # Adding a new Hamiltonian harmonic (if not already present)
Hamiltonian{<:Lattice} : Hamiltonian on a 2D Lattice in 2D space
  Bloch harmonics  : 6 (SparseMatrixCSC, sparse)
  Harmonic size    : 2 × 2
  Orbitals         : ((:a, :a), (:a, :a))
  Site eltype      : 2 × 2 blocks (Complex{Float64})
  Onsites          : 0
  Hoppings         : 6
  Coordination     : 3.0

julia> h[(3,3)][1,1] = @SMatrix[1 2; 2 1]; h[(3,3)] # element assignment
2×2 SparseMatrixCSC{StaticArrays.SArray{Tuple{2,2},Complex{Float64},2,4},Int64} with 1 stored entry:
  [1, 1]  =  [1.0+0.0im 2.0+0.0im; 2.0+0.0im 1.0+0.0im]

julia> h[(3,3)][[1,2],[1,2]] .= Ref(@SMatrix[1 2; 2 1])
2×2 view(::SparseMatrixCSC{StaticArrays.SArray{Tuple{2,2},Complex{Float64},2,4},Int64}, [1, 2], [1, 2]) with eltype StaticArrays.SArray{Tuple{2,2},Complex{Float64},2,4}:
 [1.0+0.0im 2.0+0.0im; 2.0+0.0im 1.0+0.0im]  [1.0+0.0im 2.0+0.0im; 2.0+0.0im 1.0+0.0im]
 [1.0+0.0im 2.0+0.0im; 2.0+0.0im 1.0+0.0im]  [1.0+0.0im 2.0+0.0im; 2.0+0.0im 1.0+0.0im]
```

# See also:
    `onsite`, `hopping`, `bloch`, `bloch!`
"""
hamiltonian(lat::AbstractLattice, ts...; orbitals = missing, kw...) =
    _hamiltonian(lat, sanitize_orbs(orbitals, lat.unitcell.names), ts...; kw...)
_hamiltonian(lat::AbstractLattice, orbs; kw...) = _hamiltonian(lat, orbs, TightbindingModel(); kw...)
_hamiltonian(lat::AbstractLattice, orbs, m::TightbindingModel; orbtype::Type = Complex{coordtype(lat)}, blochtype = missing) =
    hamiltonian_sparse(blocktype(orbs, orbtype), lat, orbs, m, blochtype)

hamiltonian(t::TightbindingModel...; kw...) = lat -> hamiltonian(lat, t...; kw...)

sanitize_orbs(o::Union{Val,NameType,Integer}, names::NTuple{N}) where {N} =
    ntuple(n -> sanitize_orbs(o), Val(N))
sanitize_orbs(o::NTuple{M,Union{NameType,Integer}}, names::NTuple{N}) where {M,N} =
    (ont = nametype.(o); ntuple(n -> ont, Val(N)))
sanitize_orbs(o::Missing, names) = sanitize_orbs((:a,), names)
sanitize_orbs(o::Pair, names) = sanitize_orbs((o,), names)
sanitize_orbs(os::NTuple{M,Pair}, names::NTuple{N}) where {N,M} =
    ntuple(Val(N)) do n
        for m in 1:M
            first(os[m]) == names[n] && return sanitize_orbs(os[m])
        end
        return (:a,)
    end
sanitize_orbs(os::NTuple{M,Any}, names::NTuple{N}) where {N,M} =
    ntuple(n -> n > M ? (:a,) : sanitize_orbs(os[n]), Val(N))

sanitize_orbs(p::Pair) = sanitize_orbs(last(p))
sanitize_orbs(o::Integer) = (nametype(o),)
sanitize_orbs(o::NameType) = (o,)
sanitize_orbs(o::Val{N}) where {N} = ntuple(_ -> :a, Val(N))
sanitize_orbs(o::NTuple{N,Union{Integer,NameType}}) where {N} = nametype.(o)
sanitize_orbs(p) = throw(ArgumentError("Wrong format for orbitals, see `hamiltonian`"))

function hamiltonian(h::Hamiltonian; orbtype = missing, blochtype = missing)
    harmonics´ = convert_harmonics(h, orbtype)
    blochmatrix´ = convert_blochmatrix(h, harmonics´, blochtype)
    return Hamiltonian(h.lattice, harmonics´, h.orbitals, blochmatrix´)
end

convert_harmonics(h::Hamiltonian, ::Missing) = h.harmonics
convert_harmonics(h::Hamiltonian, orbtype´) =
    orbtype(h) <: orbtype´ ? h.harmonics : convert_harmonics.(h.harmonics, orbtype´)
convert_harmonics(h::HamiltonianHarmonic{L,M}, ::Type{T´}) where {L,T,N,M<:SMatrix{N,N,T},T´} =
    HamiltonianHarmonic(h.dn, SMatrix{N,N,T´}.(h.h))
convert_harmonics(h::HamiltonianHarmonic{L,T}, ::Type{T´}) where {L,T<:Number,T´<:Number} =
    HamiltonianHarmonic(h.dn, T´.(h.h))

convert_blochmatrix(h, blochtype) = convert_blochmatrix(h, h.harmonics, blochtype)
convert_blochmatrix(h, harmonics´, ::Missing) = h.blochmatrix
convert_blochmatrix(h, harmonics´, blochtype::Type{T}) where {T} =
    typeof(h.blochmatrix) <: T ? h.blochmatrix : similarmatrix(blochtype, harmonics´, h.orbitals, h.lattice)
convert_blochmatrix(h, harmonics´, blochtype::typeof(flatten)) =
    eltype(h.blochmatrix) <: Number ? h.blochmatrix : similarmatrix(blochtype, harmonics´, h.orbitals, h.lattice)

blochtype!(h::Hamiltonian, blochtype) =
    blochtype!(h, convert_blochmatrix(h, blochtype))
blochtype!(h::Hamiltonian, blochmatrix::AbstractMatrix) =
    Hamiltonian(h.lattice, h.harmonics, h.orbitals, blochmatrix)

Base.Matrix(h::Hamiltonian) = Hamiltonian(h.lattice, Matrix.(h.harmonics), h.orbitals, Matrix(h.blochmatrix))
Base.Matrix(h::HamiltonianHarmonic) = HamiltonianHarmonic(h.dn, Matrix(h.h))

Base.copy(h::Hamiltonian) = Hamiltonian(copy(h.lattice), copy.(h.harmonics), h.orbitals, copy(h.blochmatrix))
Base.copy(h::HamiltonianHarmonic) = HamiltonianHarmonic(h.dn, copy(h.h))
copy_harmonics(h::Hamiltonian) = Hamiltonian(h.lattice, copy.(h.harmonics), h.orbitals, h.blochmatrix)

Base.size(h::Hamiltonian, n) = size(first(h.harmonics).h, n)
Base.size(h::Hamiltonian) = size(first(h.harmonics).h)
Base.size(h::HamiltonianHarmonic, n) = size(h.h, n)
Base.size(h::HamiltonianHarmonic) = size(h.h)

flatsize(h::Hamiltonian, n) = first(flatsize(h)) # h is always square
flatsize(h::Hamiltonian) = _flatsize(h.lattice, h.orbitals)
function _flatsize(lat, orbs)
    n = sum(sublatlengths(lat) .* length.(orbs))
    return (n, n)
end

function LinearAlgebra.ishermitian(h::Hamiltonian)
    for hh in h.harmonics
        isassigned(h, -hh.dn) || return false
        hh.h == h[-hh.dn]' || return false
    end
    return true
end

bravais(h::Hamiltonian) = bravais(h.lattice)

nsites(h::Hamiltonian) = isempty(h.harmonics) ? 0 : nsites(first(h.harmonics))
nsites(h::HamiltonianHarmonic) = size(h.h, 1)

nsublats(h::Hamiltonian) = nsublats(h.lattice)

norbitals(h::Hamiltonian) = length.(h.orbitals)

# External API #

"""
    dims(lat_or_ham) -> (E, L)

Return a tuple `(E, L)` of the embedding `E` and lattice dimensions `L` of `AbstractLattice`
or `Hamiltonian` `lat_or_ham`
"""

dims(lat::AbstractLattice{E,L}) where {E,L} = E, L
dims(h::Hamiltonian) = dims(h.lattice)

"""
    sitepositions(lat::AbstractLattice; kw...)
    sitepositions(h::Hamiltonian; kw...)

Build a generator of the positions of sites in the lattice unitcell. Only sites specified
by `siteselector(kw...)` are selected, see `siteselector` for details.

"""
sitepositions(lat::AbstractLattice; kw...) = sitepositions(lat, siteselector(;kw...))
sitepositions(h::Hamiltonian; kw...) = sitepositions(h.lattice, siteselector(;kw...))

"""
    siteindices(lat::AbstractLattice; kw...)
    siteindices(lat::Hamiltonian; kw...)

Build a generator of the unique indices of sites in the lattice unitcell. Only sites
specified by `siteselector(kw...)` are selected, see `siteselector` for details.

"""
siteindices(lat::AbstractLattice; kw...) = siteindices(lat, siteselector(;kw...))
siteindices(h::Hamiltonian; kw...) = siteindices(h.lattice, siteselector(;kw...))

"""
    transform!(f::Function, h::Hamiltonian)

Transform the site positions of the Hamiltonian's lattice in place without modifying the
Hamiltonian harmonics.
"""
function transform!(f, h::Hamiltonian)
    transform!(f, h.lattice)
    return h
end

# Indexing #
Base.push!(h::Hamiltonian{<:Any,L}, dn::NTuple{L,Int}) where {L} = push!(h, SVector(dn...))
Base.push!(h::Hamiltonian{<:Any,L}, dn::Vararg{Int,L}) where {L} = push!(h, SVector(dn...))
function Base.push!(h::Hamiltonian{<:Any,L}, dn::SVector{L,Int}) where {L}
    get_or_push!(h.harmonics, dn, size(h))
    return h
end

function get_or_push!(harmonics::Vector{H}, dn::SVector{L,Int}, dims) where {L,M,A,H<:HamiltonianHarmonic{L,M,A}}
    for hh in harmonics
        hh.dn == dn && return hh
    end
    hh = HamiltonianHarmonic{L,M,A}(dn, dims...)
    push!(harmonics, hh)
    return hh
end

Base.getindex(h::Hamiltonian, dn::NTuple) = getindex(h, SVector(dn))
@inline function Base.getindex(h::Hamiltonian{<:Any,L}, dn::SVector{L,Int}) where {L}
    nh = findfirst(hh -> hh.dn == dn, h.harmonics)
    nh === nothing && throw(BoundsError(h, dn))
    return h.harmonics[nh].h
end
Base.getindex(h::Hamiltonian, dn::NTuple, i::Vararg{Int}) = h[dn][i...]
Base.getindex(h::Hamiltonian{LA, L}, i::Vararg{Int}) where {LA,L} = h[zero(SVector{L,Int})][i...]

Base.deleteat!(h::Hamiltonian{<:Any,L}, dn::Vararg{Int,L}) where {L} =
    deleteat!(h, toSVector(dn))
Base.deleteat!(h::Hamiltonian{<:Any,L}, dn::NTuple{L,Int}) where {L} =
    deleteat!(h, toSVector(dn))
function Base.deleteat!(h::Hamiltonian{<:Any,L}, dn::SVector{L,Int}) where {L}
    nh = findfirst(hh -> hh.dn == SVector(dn...), h.harmonics)
    nh === nothing || deleteat!(h.harmonics, nh)
    return h
end

Base.isassigned(h::Hamiltonian, dn::Vararg{Int}) = isassigned(h, SVector(dn))
Base.isassigned(h::Hamiltonian, dn::NTuple) = isassigned(h, SVector(dn))
Base.isassigned(h::Hamiltonian{<:Any,L}, dn::SVector{L,Int}) where {L} =
    findfirst(hh -> hh.dn == dn, h.harmonics) != nothing

## Boolean masking
"""
    &(h1::Hamiltonian{<:Superlattice}, h2::Hamiltonian{<:Superlattice})

Construct a new `Hamiltonian{<:Superlattice}` using an `and` boolean mask, i.e. with a
supercell that contains cells that are both in the supercell of `h1` and `h2`

    &(s1::Superlattice, s2::Superlattice}

Equivalent of the above for `Superlattice`s

# See also:
    `|`, `xor`
"""
(Base.:&)(s1::Hamiltonian{<:Superlattice}, s2::Hamiltonian{<:Superlattice}) =
    boolean_mask_hamiltonian(Base.:&, s1, s2)

"""
    |(h1::Hamiltonian{<:Superlattice}, h2::Hamiltonian{<:Superlattice})

Construct a new `Hamiltonian{<:Superlattice}` using an `or` boolean mask, i.e. with a
supercell that contains cells that are either in the supercell of `h1` or `h2`

    |(s1::Superlattice, s2::Superlattice}

Equivalent of the above for `Superlattice`s

# See also:
    `&`, `xor`
"""
(Base.:|)(s1::Hamiltonian{<:Superlattice}, s2::Hamiltonian{<:Superlattice}) =
    boolean_mask_hamiltonian(Base.:|, s1, s2)

"""
    xor(h1::Hamiltonian{<:Superlattice}, h2::Hamiltonian{<:Superlattice})

Construct a new `Hamiltonian{<:Superlattice}` using a `xor` boolean mask, i.e. with a
supercell that contains cells that are either in the supercell of `h1` or `h2` but not in
both

    xor(s1::Superlattice, s2::Superlattice}

Equivalent of the above for `Superlattice`s

# See also:
    `&`, `|`
"""
(Base.xor)(s1::Hamiltonian{<:Superlattice}, s2::Hamiltonian{<:Superlattice}) =
    boolean_mask_hamiltonian(Base.xor, s1, s2)

function boolean_mask_hamiltonian(f, s1::Hamiltonian{<:Superlattice}, s2::Hamiltonian{<:Superlattice})
    check_compatible_hsuper(s1, s2)
    return Hamiltonian(f(s1.lattice, s2.lattice), copy(s1.harmonics), s1.orbitals, s1.blochmatrix)
end

function check_compatible_hsuper(s1, s2)
    compatible = isequal(s1.harmonics, s2.harmonics) && isequal(s1.orbitals, s2.orbitals)
    compatible || throw(ArgumentError("Hamiltonians are incompatible for boolean masking"))
    return nothing
end

#######################################################################
# auxiliary types
#######################################################################
struct IJV{L,M}
    dn::SVector{L,Int}
    i::Vector{Int}
    j::Vector{Int}
    v::Vector{M}
end

struct IJVBuilder{L,M,E,T,O,LA<:AbstractLattice{E,L,T}}
    lat::LA
    orbs::O
    ijvs::Vector{IJV{L,M}}
    kdtrees::Vector{KDTree{SVector{E,T},Euclidean,T}}
end

IJV{L,M}(dn::SVector{L} = zero(SVector{L,Int})) where {L,M} =
    IJV(dn, Int[], Int[], M[])

function IJVBuilder(lat::AbstractLattice{E,L,T}, orbs, ijvs::Vector{IJV{L,M}}) where {E,L,T,M}
    kdtrees = Vector{KDTree{SVector{E,T},Euclidean,T}}(undef, nsublats(lat))
    return IJVBuilder(lat, orbs, ijvs, kdtrees)
end

IJVBuilder(lat::AbstractLattice{E,L}, orbs, ::Type{M}) where {E,L,M} =
    IJVBuilder(lat, orbs, IJV{L,M}[])

function IJVBuilder(lat::AbstractLattice{E,L}, orbs, hs::Hamiltonian...) where {E,L}
    M = promote_blocktype(hs...)
    ijvs = IJV{L,M}[]
    builder = IJVBuilder(lat, orbs, ijvs)
    offset = 0
    for h in hs
        for har in h.harmonics
            ijv = builder[har.dn]
            push_block!(ijv, har, offset)
        end
        offset += size(h, 1)
    end
    return builder
end

Base.eltype(b::IJVBuilder{L,M}) where {L,M} = M

function Base.getindex(b::IJVBuilder{L,M}, dn::SVector{L2,Int}) where {L,L2,M}
    L == L2 || throw(error("Tried to apply an $L2-dimensional model to an $L-dimensional lattice"))
    for e in b.ijvs
        e.dn == dn && return e
    end
    e = IJV{L,M}(dn)
    push!(b.ijvs, e)
    return e
end

Base.length(h::IJV) = length(h.i)
Base.isempty(h::IJV) = length(h) == 0
Base.copy(h::IJV) = IJV(h.dn, copy(h.i), copy(h.j), copy(h.v))

function Base.resize!(h::IJV, n)
    resize!(h.i, n)
    resize!(h.j, n)
    resize!(h.v, n)
    return h
end

Base.push!(ijv::IJV, (i, j, v)::Tuple) = (push!(ijv.i, i); push!(ijv.j, j); push!(ijv.v, v))

function push_block!(ijv::IJV{L,M}, h::HamiltonianHarmonic, offset) where {L,M}
    I, J, V = findnz(h.h)
    for (i, j, v) in zip(I, J, V)
        push!(ijv, (i + offset, j + offset, padtotype(v, M)))
    end
    return ijv
end

#######################################################################
# hamiltonian_sparse
#######################################################################
function hamiltonian_sparse(Mtype, lat, orbs, model, blochtype)
    builder = IJVBuilder(lat, orbs, Mtype)
    return hamiltonian_sparse!(builder, lat, orbs, model, blochtype)
end

function hamiltonian_sparse!(builder::IJVBuilder{L,M}, lat::AbstractLattice{E,L}, orbs, model, blochtype) where {E,L,M}
    applyterms!(builder, terms(model)...)
    n = nsites(lat)
    HT = HamiltonianHarmonic{L,M,SparseMatrixCSC{M,Int}}
    harmonics = HT[HT(e.dn, sparse(e.i, e.j, e.v, n, n)) for e in builder.ijvs if !isempty(e)]
    return Hamiltonian(lat, harmonics, orbs, n, n, blochtype)
end

applyterms!(builder, terms...) = foreach(term -> applyterm!(builder, term), terms)

applyterm!(builder::IJVBuilder, term::Union{OnsiteTerm, HoppingTerm}) =
    applyterm!(builder, term)

function applyterm!(builder::IJVBuilder{L}, term::OnsiteTerm) where {L}
    lat = builder.lat
    dn0 = zero(SVector{L,Int})
    ijv = builder[dn0]
    allpos = allsitepositions(lat)
    rsel = resolve(term.selector, lat)
    for s in sublats(rsel), i in siteindices(rsel, s)
        r = allpos[i]
        v = toeltype(term(r, r), eltype(builder), builder.orbs[s], builder.orbs[s])
        push!(ijv, (i, i, v))
    end
    return nothing
end

function applyterm!(builder::IJVBuilder{L}, term::HoppingTerm) where {L}
    lat = builder.lat
    rsel = resolve(term.selector, lat)
    L > 0 && checkinfinite(rsel)
    allpos = allsitepositions(lat)
    for (s2, s1) in sublats(rsel)  # Each is a Pair s2 => s1
        dns = dniter(rsel)
        for dn in dns
            keepgoing = false
            ijv = builder[dn]
            for j in source_candidates(rsel, s2)
                sitej = allpos[j]
                rsource = sitej - bravais(lat) * dn
                is = targets(builder, rsel.selector.range, rsource, s1)
                for i in is
                    # Make sure we don't stop searching until we reach minimum range
                    is_below_min_range((i, j), (dn, zero(dn)), rsel) && (keepgoing = true)
                    ((i, j), (dn, zero(dn))) in rsel || continue
                    keepgoing = true
                    rtarget = allsitepositions(lat)[i]
                    r, dr = _rdr(rsource, rtarget)
                    v = toeltype(term(r, dr), eltype(builder), builder.orbs[s1], builder.orbs[s2])
                    push!(ijv, (i, j, v))
                end
            end
            keepgoing && acceptcell!(dns, dn)
        end
    end
    return nothing
end

# For use in Hamiltonian building
toeltype(t::Number, ::Type{T}, t1::NTuple{1}, t2::NTuple{1}) where {T<:Number} = T(t)
toeltype(t::Number, ::Type{S}, t1::NTuple{1}, t2::NTuple{1}) where {S<:SMatrix} =
    padtotype(t, S)
toeltype(t::SMatrix{N1,N2}, ::Type{S}, t1::NTuple{N1}, t2::NTuple{N2}) where {N1,N2,S<:SMatrix} =
    padtotype(t, S)

toeltype(u::UniformScaling, ::Type{T}, t1::NTuple{1}, t2::NTuple{1}) where {T<:Number} = T(u.λ)
toeltype(u::UniformScaling, ::Type{S}, t1::NTuple{N1}, t2::NTuple{N2}) where {N1,N2,S<:SMatrix} =
    padtotype(SMatrix{N1,N2}(u), S)

# For use in ket building
toeltype(t::Number, ::Type{T}, t1::NTuple{1}) where {T<:Number} = T(t)
toeltype(t::Number, ::Type{S}, t1::NTuple{1}) where {S<:SVector} = padtotype(t, S)
toeltype(t::SVector{N}, ::Type{S}, t1::NTuple{N}) where {N,S<:SVector} = padtotype(t, S)
toeltype(t::SMatrix{N}, ::Type{S}, t1::NTuple{N}) where {N,S<:SMatrix} = padtotype(t, S)

# Fallback to catch mismatched or undesired block types
toeltype(t::Array, x...) = throw(ArgumentError("Array input in model, please use StaticArrays instead (e.g. SA[1 0; 0 1] instead of [1 0; 0 1])"))
toeltype(t, x...) = throw(DimensionMismatch("Dimension mismatch between model and Hamiltonian. Does the `orbitals` kwarg in your `hamiltonian` match your model?"))

# Although range can be (rmin, rmax) we return all targets within rmax.
# Those below rmin get filtered later by `in rsel`
function targets(builder, range, rsource, s1)
    rmax = maximum(range)
    !isfinite(rmax) && return targets(builder, missing, rsource, s1)
    if !isassigned(builder.kdtrees, s1)
        sitepos = sitepositions(builder.lat.unitcell, s1)
        (builder.kdtrees[s1] = KDTree(sitepos))
    end
    targetlist = inrange(builder.kdtrees[s1], rsource, rmax)
    targetlist .+= builder.lat.unitcell.offsets[s1]
    return targetlist
end

targets(builder, range::Missing, rsource, s1) = siterange(builder.lat, s1)

checkinfinite(rs) =
    rs.selector.dns === missing && (rs.selector.range === missing || !isfinite(maximum(rs.selector.range))) &&
    throw(ErrorException("Tried to implement an infinite-range hopping on an unbounded lattice"))

#######################################################################
# Matrix(::KetModel, ::Hamiltonian), and Vector
#######################################################################
"""
  Vector(km::KetModel, h::Hamiltonian)

Construct a `Vector` representation of `km` applied to Hamiltonian `h`.
"""
Base.Vector(km::KetModel, h::Hamiltonian) = vec(Matrix(km, h))

"""
  Matrix(km::KetModel, h::Hamiltonian)
  Matrix(kms::NTuple{N,KetModel}, h::Hamiltonian)
  Matrix(kms::AbstractMatrix, h::Hamiltonian)
  Matrix(kms::StochasticTraceKets, h::Hamiltonian)

Construct an `M×N` `Matrix` representation of the `N` kets `kms` applied to `M×M`
Hamiltonian `h`. If `kms::StochasticTraceKets` for `n` random kets (constructed with
`randomkets(n)`), a normalization `1/√n` required for stochastic traces is included.
"""
Base.Matrix(km::KetModel, h::Hamiltonian) = Matrix((km,), h)

function Base.Matrix(km::AbstractMatrix, h::Hamiltonian)
    check_compatible_kets(km, h)
    kmat = Matrix(km)
    return kmat
end

# kmodels should be a Union{NTuple{N,KetModel},StochasticTraceKets}
function Base.Matrix(kmodels, h::Hamiltonian)
    kmodels´ = resolve_tuple(kmodels, h.lattice)
    allpos = allsitepositions(h.lattice)
    T = guess_eltype(kmodels´, allpos, h)
    orbs = h.orbitals
    kmat = [generate_amplitude(km, i, allpos[i], T, orbs[s]) for (i, s) in sitesublats(h.lattice), km in kmodels´]
    check_compatible_kets(kmat, h)
    maybe_normalize!(kmat, kmodels)
    return kmat
end

resolve_tuple(ks::NTuple{N,KetModel}, lat) where {N} = resolve.(ks, Ref(lat))
resolve_tuple(ks::StochasticTraceKets, lat) = resolve(ks, lat)

function guess_eltype(kms, allpos, h)
    km = first(kms)
    term = first(km.model.terms)
    rsel = term.selector
    s = first(sublats(rsel))
    i = first(siteindices(rsel, s))
    r = allpos[i]
    t = term(r, r)
    z = zero(orbitaltype(h))
    T = _guess_eltype(km.maporbitals, t, z)
    return T
end

_guess_eltype(::Val{true}, t::Number, z) = typeof(t * z)
_guess_eltype(::Val{false}, t::Number, z) = typeof(t * z)
_guess_eltype(::Val{false}, t::SVector{<:Any,T}, z::SVector) where {T} = typeof(zero(T) * z)
_guess_eltype(::Val{false}, t::SMatrix{M,N,T}, z::SVector{M2,T2}) where {M,N,M2,T,T2} = typeof(SMatrix{M2,N}(zero(T) * zero(T2) * I))

function maybe_normalize!(kmat, kms::StochasticTraceKets)
    kms.ketmodel.normalized && normalize_columns!(kmat)
    kmat .*= sqrt(1/size(kmat,2))
    return kmat
end

function maybe_normalize!(kmat, kms::Tuple{KetModel})
    for (i, km) in enumerate(kms)
        km.normalized && normalize_columns!(kmat, i)
    end
    return kmat
end

check_compatible_kets(kmat::AbstractMatrix, h::Hamiltonian) =
    comp_eltypes(h, kmat) && size(kmat, 1) == size(h, 2) ||
        throw(ArgumentError("ket vector or matrix is incompatible with Hamiltonian"))

comp_eltypes(h::Hamiltonian, k::AbstractMatrix) = comp_eltypes(blocktype(h), eltype(k))
comp_eltypes(::Type{<:Number}, ::Type{<:Number}) = true
comp_eltypes(::Type{<:Number}, ::Type{<:SMatrix{1}}) = true
comp_eltypes(::Type{<:SMatrix{N,M}}, ::Type{<:SVector{M}}) where {N,M} = true
comp_eltypes(::Type{<:SMatrix{N,M}}, ::Type{<:SMatrix{M}}) where {N,M}  = true
comp_eltypes(t1, t2) = false

### generate_amplitude (asssumes resolved selectors) ###

function generate_amplitude(ketmodel::KetModel, i, r, T, orbs)
    amplitude = sum(ketmodel.model.terms) do term
        i in term.selector ? maybe_maporbitals(ketmodel.maporbitals, T, orbs, term, r) : zero(T)
    end
    return amplitude
end

function maybe_maporbitals(::Val{false}, T, orbs, term, r)
    return toeltype(term(r, r), T, orbs)
end

function maybe_maporbitals(::Val{true}, T, orbs::NTuple{N}, term, r) where {N}
    x = SVector{N}(ntuple(_ -> Number(term(r, r)), Val(N)))
    return toeltype(x, T, orbs)
end

#######################################################################
# unitcell/supercell for Hamiltonians
#######################################################################
function supercell(ham::Hamiltonian, args...; kw...)
    slat = supercell(ham.lattice, args...; kw...)
    return Hamiltonian(slat, ham.harmonics, ham.orbitals, ham.blochmatrix)
end

# Fast-path unitcell(ham), does not allocate a new blochmatrix
function unitcell(ham::Hamiltonian{<:Lattice})
    iszero(ham.harmonics[1].dn) || throw(error("Unexpected error: first harmonic is not the fundamental"))
    return Hamiltonian(ham.lattice, [ham.harmonics[1]], ham.orbitals, ham.blochmatrix)
end

function unitcell(ham::Hamiltonian{<:Lattice}, args...; modifiers = (), kw...)
    sham = supercell(ham, args...; kw...)
    return unitcell(sham; modifiers = modifiers)
end

function unitcell(ham::Hamiltonian{LA,L}; modifiers = ()) where {E,L,T,L´,LA<:Superlattice{E,L,T,L´}}
    lat = ham.lattice
    sc = lat.supercell
    isc = inv_supercell(bravais(lat), sc.matrix)
    modifiers´ = resolve.(ensuretuple(modifiers), Ref(lat))
    mapping = OffsetArray{Int}(undef, sc.sites, sc.cells.indices...) # store supersite indices newi
    mapping .= 0
    foreach_supersite((s, oldi, olddn, newi) -> mapping[oldi, Tuple(olddn)...] = newi, lat)
    dim = nsites(sc)
    B = blocktype(ham)
    S = typeof(SparseMatrixBuilder{B}(dim, dim))
    harmonic_builders = HamiltonianHarmonic{L´,B,S}[]
    foreach_supersite(lat) do s, source_i, source_dn, newcol
        for oldh in ham.harmonics
            rows = rowvals(oldh.h)
            vals = nonzeros(oldh.h)
            target_dn = source_dn + oldh.dn
            super_dn = new_dn(target_dn, isc)
            wrapped_dn = wrap_dn(target_dn, super_dn, sc.matrix)
            newh = get_or_push!(harmonic_builders, super_dn, dim, newcol)
            for p in nzrange(oldh.h, source_i)
                target_i = rows[p]
                # check: wrapped_dn could exit bounding box along non-periodic direction
                checkbounds(Bool, mapping, target_i, Tuple(wrapped_dn)...) || continue
                newrow = mapping[target_i, Tuple(wrapped_dn)...]
                val = applymodifiers(vals[p], lat, (source_i, target_i), (source_dn, target_dn), modifiers´...)
                iszero(newrow) || pushtocolumn!(newh.h, newrow, val)
            end
        end
        foreach(h -> finalizecolumn!(h.h), harmonic_builders)
    end
    harmonics = [HamiltonianHarmonic(h.dn, sparse(h.h)) for h in harmonic_builders]
    unitlat = unitcell(lat)
    n = nsites(unitlat)
    orbs = ham.orbitals
    blochtype = typeof(ham.blochmatrix)
    return Hamiltonian(unitlat, harmonics, orbs, n, n, blochtype)
end

function get_or_push!(hs::Vector{<:HamiltonianHarmonic{L,B,<:SparseMatrixBuilder}}, dn, dim, currentcol) where {L,B}
    for h in hs
        h.dn == dn && return h
    end
    newh = HamiltonianHarmonic(dn, SparseMatrixBuilder{B}(dim, dim))
    currentcol > 1 && finalizecolumn!(newh.h, currentcol - 1) # for columns that have been already processed
    push!(hs, newh)
    return newh
end

inv_supercell(br, sc::SMatrix{L,L´}) where {L,L´} = inv(extended_supercell(br, sc))[SVector{L´}(1:L´), :]

new_dn(oldn, isc) = floor.(Int, isc * oldn)

wrap_dn(olddn::SVector, newdn::SVector, supercell::SMatrix) = olddn - supercell * newdn

applymodifiers(val, lat, inds, dns) = val

function applymodifiers(val, lat, inds, dns, m::UniformModifier, ms...)
    selected = (inds, dns) in m.selector
    val´ = selected ? m.f(val) : val
    return applymodifiers(val´, lat, inds, dns, ms...)
end

function applymodifiers(val, lat, (row, col), (dnrow, dncol), m::OnsiteModifier, ms...)
    selected = ((row, col), (dnrow, dncol)) in m.selector
    if selected
        r = allsitepositions(lat)[col] + bravais(lat) * dncol
        val´ = selected ? m(val, r) : val
    else
        val´ = val
    end
    return applymodifiers(val´, lat, (row, col), (dnrow, dncol), ms...)
end

function applymodifiers(val, lat, (row, col), (dnrow, dncol), m::HoppingModifier, ms...)
    selected = ((row, col), (dnrow, dncol)) in m.selector
    if selected
        br = bravais(lat)
        r, dr = _rdr(allsitepositions(lat)[col] + br * dncol, allsitepositions(lat)[row] + br * dnrow)
        val´ = selected ? m(val, r, dr) : val
    else
        val´ = val
    end
    return applymodifiers(val´, lat, (row, col), (dnrow, dncol), ms...)
end

#######################################################################
# wrap
#######################################################################
"""
    wrap(h::Hamiltonian, axes; phases = missing)

Build a new Hamiltonian from `h` reducing its dimensions from `L` to `L - length(axes)` by
wrapping the specified Bravais `axes` into a loop. `axes` can be an integer ∈ 1:L or a tuple
of such integers. If `phases` are given (with `length(axes) == length(phases)`), the wrapped
hoppings at a cell distance `dn` along `axes` will be multiplied by a factor
`cis(-dot(phases, dn))`. This is useful, for example, to represent a flux Φ through a loop,
using a single `axes = 1` and `phases = 2π * Φ/Φ₀`.

    wrap(h::Hamiltonian; kw...)

Wrap all axes of `h`, yielding a compactified zero-dimensional Hamiltonian.

    h |> wrap(axes; kw...)

Curried form equivalent to `wrap(h, axes; kw...)`.

# Examples

```jldoctest
julia> LatticePresets.honeycomb() |> hamiltonian(hopping(1, range = 1/√3)) |>
       unitcell((1,-1), (10, 10)) |> wrap(2)
Hamiltonian{<:Lattice} : Hamiltonian on a 1D Lattice in 2D space
  Bloch harmonics  : 3 (SparseMatrixCSC, sparse)
  Harmonic size    : 40 × 40
  Orbitals         : ((:a,), (:a,))
  Site eltype      : scalar (Complex{Float64})
  Onsites          : 0
  Hoppings         : 120
  Coordination     : 3.0
```
"""
wrap(h::Hamiltonian, axis::Int; kw...) = wrap(h, (axis,); kw...)

wrap(h::Hamiltonian{<:Lattice,L}; kw...) where {L} = wrap(h, ntuple(identity, Val(L)); kw...)

function wrap(h::Hamiltonian{<:Lattice,L}, axes::NTuple{N,Int}; phases = missing) where {L,N}
    all(axis -> 1 <= axis <= L, axes) && allunique(axes) || throw(ArgumentError("wrap axes should be unique and between 1 and the lattice dimension $L"))
    lattice´ = _wrap(h.lattice, axes)
    phases´ = (phases === missing) ? filltuple(0, Val(N)) : phases
    harmonics´ = _wrap(h.harmonics, axes, phases´, size(h))
    return Hamiltonian(lattice´, harmonics´, h.orbitals, h.blochmatrix)
end

wrap(axes::Union{Integer,Tuple}; kw...) = h -> wrap(h, axes; kw...)

_wrap(lat::Lattice, axes) = Lattice(_wrap(lat.bravais, axes), lat.unitcell)

function _wrap(br::Bravais{E,L}, axes) where {E,L}
    mask = deletemultiple_nocheck(SVector{L}(1:L), axes)
    return Bravais(br.matrix[:, mask])
end

function _wrap(harmonics::Vector{HamiltonianHarmonic{L,M,A}}, axes::NTuple{N,Int}, phases::NTuple{N,Number}, sizeh) where {L,M,A,N}
    harmonics´ = HamiltonianHarmonic{L-N,M,A}[]
    for har in harmonics
        dn = har.dn
        dn´ = deletemultiple_nocheck(dn, axes)
        phase = -sum(phases .* dn[SVector(axes)])
        newh = get_or_push!(harmonics´, dn´, sizeh)
        # map!(+, newh, newh, factor * har.h) # TODO: activate after resolving #37375
        newh.h .+= cis(phase) .* har.h
    end
    return harmonics´
end

#######################################################################
# combine
#######################################################################
"""
    combine(hams::Hamiltonian...; coupling = missing)

Build a new Hamiltonian `h` that combines all `hams` as diagonal blocks, and applies
`coupling::Model`, if provided, to build the off-diagonal couplings. Note that the diagonal
blocks are not modified by the coupling model.
"""
combine(hams::Hamiltonian...; coupling = missing) = _combine(coupling, hams...)

_combine(::Missing, hams...) = _combine(TightbindingModel(), hams...)

function _combine(model::TightbindingModel, hams::Hamiltonian...)
    lat = combine((h -> h.lattice).(hams)...)
    orbs = tuplejoin((h -> h.orbitals).(hams)...)
    builder = IJVBuilder(lat, orbs, hams...)
    model´ = offdiagonal(model, lat, nsublats.(hams))
    ham = hamiltonian_sparse!(builder, lat, orbs, model´)
    return ham
end

#######################################################################
# Bloch routines
#######################################################################

"""
    bloch(h::Hamiltonian{<:Lattice}, ϕs)

Build the Bloch Hamiltonian matrix of `h`, for Bloch phases `ϕs = (ϕ₁, ϕ₂,...)` (or an
`SVector(ϕs...)`). In terms of Bloch wavevector `k`, `ϕs = k * bravais(h)`, it is defined as
`H(ϕs) = ∑exp(-im * ϕs' * dn) h_dn` where `h_dn` are Bloch harmonics connecting unit cells
at a distance `dR = bravais(h) * dn`. The resulting matrix has a type matching the
`blochtype` option specified upon creating `h` (e.g. can be flattened, with numerical
eltype, despite `h` being a multiorbital system).

    bloch(h::Hamiltonian{<:Lattice})

Build the intra-cell Hamiltonian matrix of `h`, without adding any Bloch harmonics.

    bloch(h::Hamiltonian{<:Lattice}, ϕs, axis::Int)

A nonzero `axis` produces the derivative of the Bloch matrix respect to `ϕs[axis]` (i.e. the
velocity operator along this axis), `∂H(ϕs) = ∑ -im * dn[axis] * exp(-im * ϕs' * dn) h_dn`

    bloch(h::Hamiltonian{<:Lattice}, ϕs::NTuple{L,Real}, dnfunc::Function)

Generalization that applies a prefactor `dnfunc(dn) * exp(im * ϕs' * dn)` to the `dn`
harmonic.

    h |> bloch(ϕs, ...)

Curried forms of `bloch`, equivalent to `bloch(h, ϕs, ...)`

# Notes

`bloch` allocates a new, independent matrix on each call (no-aliasing guarantee). For a
non-allocating version of `bloch`, see `bloch!`.

# Examples

```jldoctest
julia> h = LatticePresets.honeycomb() |> hamiltonian(onsite(1) + hopping(2)) |> bloch((0, 0))
2×2 SparseMatrixCSC{Complex{Float64},Int64} with 4 stored entries:
  [1, 1]  =  13.0+0.0im
  [2, 1]  =  6.0+0.0im
  [1, 2]  =  6.0+0.0im
  [2, 2]  =  13.0+0.0im
```

# See also:
    `bloch!`, `hamiltonian`
"""
bloch(ϕs, axis = 0) = h -> bloch(h, ϕs, axis)
bloch(h::Hamiltonian, args...) = copy(bloch!(h, args...))

"""
    bloch!(h::Hamiltonian, ϕs, [axis])

In-place version of `bloch`, without the no-aliasing guarantee. This should be
used for performance, but care should be taken about aliasing issues (e.g. two
`Hamiltonian`s may produce a matrix that shares the same memory). Therefore, the output of
`bloch!` is meant to be used just after being generated.

# Examples

```jldoctest
julia> h = LatticePresets.honeycomb() |> hamiltonian(hopping(2I), orbitals = (Val(2), Val(1)));

julia> bloch!(h, (0, 0))
2×2 SparseMatrixCSC{StaticArrays.SArray{Tuple{2,2},Complex{Float64},2,4},Int64} with 4 stored entries:
  [1, 1]  =  [12.0+0.0im 0.0+0.0im; 0.0+0.0im 12.0+0.0im]
  [2, 1]  =  [6.0+0.0im 0.0+0.0im; 0.0+0.0im 0.0+0.0im]
  [1, 2]  =  [6.0+0.0im 0.0+0.0im; 0.0+0.0im 0.0+0.0im]
  [2, 2]  =  [12.0+0.0im 0.0+0.0im; 0.0+0.0im 0.0+0.0im]
```

# See also:
    `bloch`, `hamiltonian`
"""
bloch!(h::Hamiltonian, ϕs = (), axis = 0) = _bloch!(h, toSVector(ϕs), axis)

function _bloch!(h::Hamiltonian{<:Lattice,L,M}, ϕs, axis::Number) where {L,M}
    rawmatrix = parent(h.blochmatrix)
    if iszero(axis)
        _copy!(rawmatrix, first(h.harmonics).h, h) # faster copy!(dense, sparse) specialization
        add_harmonics!(rawmatrix, h, ϕs, dn -> 1)
    else
        fill!(rawmatrix, zero(M)) # There is no guarantee of same structure
        add_harmonics!(rawmatrix, h, ϕs, dn -> -im * dn[axis])
    end
    return h.blochmatrix
end

function _bloch!(h::Hamiltonian{<:Lattice,L,M}, ϕs, dnfunc::Function) where {L,M}
    prefactor0 = dnfunc(zero(ϕs))
    rawmatrix = parent(h.blochmatrix)
    if iszero(prefactor0)
        fill!(rawmatrix, zero(eltype(rawmatrix)))
    else
        _copy!(rawmatrix, first(h.harmonics).h, h)
        rmul!(rawmatrix, prefactor0)
    end
    add_harmonics!(rawmatrix, h, ϕs, dnfunc)
    return h.blochmatrix
end

add_harmonics!(zerobloch, h::Hamiltonian{<:Lattice}, ϕs::SVector{0}, _) = zerobloch

function add_harmonics!(zerobloch, h::Hamiltonian{<:Lattice,L}, ϕs::SVector{L}, dnfunc) where {L}
    ϕs´ = ϕs'
    for ns in 2:length(h.harmonics)
        hh = h.harmonics[ns]
        hhmatrix = hh.h
        prefactor = dnfunc(hh.dn)
        iszero(prefactor) && continue
        ephi = prefactor * cis(-ϕs´ * hh.dn)
        _add!(zerobloch, hhmatrix, h, ephi)
    end
    return zerobloch
end

############################################################################################
######## _copy! and _add! call specialized methods in tools.jl #############################
############################################################################################

_copy!(dest, src, h) = copy!(dest, src)
_copy!(dst::AbstractMatrix{<:Number}, src::SparseMatrixCSC{<:Number}, h) = _fast_sparse_copy!(dst, src)
_copy!(dst::DenseMatrix{<:Number}, src::SparseMatrixCSC{<:Number}, h) = _fast_sparse_copy!(dst, src)
_copy!(dst::DenseMatrix{<:SMatrix{N,N}}, src::SparseMatrixCSC{<:SMatrix{N,N}}, h) where {N} = _fast_sparse_copy!(dst, src)
_copy!(dst::AbstractMatrix{<:Number}, src::SparseMatrixCSC{<:SMatrix}, h) = flatten_sparse_copy!(dst, src, h)
_copy!(dst::DenseMatrix{<:Number}, src::DenseMatrix{<:SMatrix}, h) = flatten_dense_copy!(dst, src, h)

_add!(dest, src, h, α) = _plain_muladd!(dest, src, α)
_add!(dst::AbstractMatrix{<:Number}, src::SparseMatrixCSC{<:Number}, h, α = 1) = _fast_sparse_muladd!(dst, src, α)
_add!(dst::AbstractMatrix{<:SMatrix{N,N}}, src::SparseMatrixCSC{<:SMatrix{N,N}}, h, α = I) where {N} = _fast_sparse_muladd!(dst, src, α)
_add!(dst::AbstractMatrix{<:Number}, src::SparseMatrixCSC{<:SMatrix}, h, α = I) = flatten_sparse_muladd!(dst, src, h, α)
_add!(dst::DenseMatrix{<:Number}, src::DenseMatrix{<:SMatrix}, h, α = I) = flatten_dense_muladd!(dst, src, h, α)

function flatten_sparse_copy!(dst, src, h)
    fill!(dst, zero(eltype(dst)))
    norbs = length.(h.orbitals)
    offsets = h.lattice.unitcell.offsets
    offsets´ = flatoffsets(offsets, norbs)
    coloffset = 0
    for s´ in sublats(h.lattice)
        N´ = norbs[s´]
        for col in siterange(h.lattice, s´)
            for p in nzrange(src, col)
                val = nonzeros(src)[p]
                row = rowvals(src)[p]
                rowoffset, M´ = flatoffsetorbs(row, h.lattice, norbs, offsets´)
                for j in 1:N´, i in 1:M´
                    dst[i + rowoffset, j + coloffset] = val[i, j]
                end
            end
            coloffset += N´
        end
    end
    return dst
end

function flatten_sparse_muladd!(dst, src, h, α = I)
    norbs = length.(h.orbitals)
    offsets = h.lattice.unitcell.offsets
    offsets´ = flatoffsets(offsets, norbs)
    coloffset = 0
    for s´ in sublats(h.lattice)
        N´ = norbs[s´]
        for col in siterange(h.lattice, s´)
            for p in nzrange(src, col)
                val = α * nonzeros(src)[p]
                row = rowvals(src)[p]
                rowoffset, M´ = flatoffsetorbs(row, h.lattice, norbs, offsets´)
                for j in 1:N´, i in 1:M´
                    dst[i + rowoffset, j + coloffset] += val[i, j]
                end
            end
            coloffset += N´
        end
    end
    return dst
end

function flatten_dense_muladd!(dst, src, h, α = I)
    norbs = length.(h.orbitals)
    offsets = h.lattice.unitcell.offsets
    offsets´ = flatoffsets(offsets, norbs)
    coloffset = 0
    for s´ in sublats(h.lattice)
        N´ = norbs[s´]
        for col in siterange(h.lattice, s´)
            rowoffset = 0
            for s in sublats(h.lattice)
                M´ = norbs[s]
                for row in siterange(h.lattice, s)
                    val = α * src[row, col]
                    for j in 1:N´, i in 1:M´
                        dst[i + rowoffset, j + coloffset] += val[i, j]
                    end
                    rowoffset += M´
                end
            end
            coloffset += N´
        end
    end
    return dst
end

function flatten_dense_copy!(dst, src, h)
    fill!(dst, zero(eltype(dst)))
    return flatten_dense_muladd!(dst, src, h, I)
end

# sublat offsets after flattening (without padding zeros)
flatoffsets(offsets, norbs) = _flatoffsets((0,), offsets, norbs...)
_flatoffsets(offsets´::NTuple{N,Any}, offsets, n, ns...) where {N} =
    _flatoffsets((offsets´..., offsets´[end] + n * (offsets[N+1] - offsets[N])), offsets, ns...)
_flatoffsets(offsets´, offsets) = offsets´

# offset of site i after flattening
@inline flatoffset(args...) = first(flatoffsetorbs(args...))

function flatoffsetorbs(i, lat, norbs, offsets´)
    s = sublat(lat, i)
    N = norbs[s]
    offset = lat.unitcell.offsets[s]
    Δi = i - offset
    i´ = offsets´[s] + (Δi - 1) * N
    return i´, N
end