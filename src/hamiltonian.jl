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

struct Hamiltonian{LA<:AbstractLattice,L,M,A<:AbstractMatrix,
                   H<:HamiltonianHarmonic{L,M,A},
                   O<:Tuple{Vararg{Tuple{Vararg{NameType}}}}} <: AbstractMatrix{M}
    lattice::LA
    harmonics::Vector{H}
    orbitals::O
end

function Hamiltonian(lat, hs::Vector{H}, orbs, n::Int, m::Int) where {L,M,H<:HamiltonianHarmonic{L,M}}
    sort!(hs, by = h -> abs.(h.dn))
    if isempty(hs) || !iszero(first(hs).dn)
        pushfirst!(hs, H(zero(SVector{L,Int}), empty_sparse(M, n, m)))
    end
    return Hamiltonian(lat, hs, orbs)
end

Base.show(io::IO, ham::Hamiltonian) = show(io, MIME("text/plain"), ham)
function Base.show(io::IO, ::MIME"text/plain", ham::Hamiltonian)
    i = get(io, :indent, "")
    print(io, i, summary(ham), "\n",
"$i  Bloch harmonics  : $(length(ham.harmonics)) ($(displaymatrixtype(ham)))
$i  Harmonic size    : $((n -> "$n × $n")(nsites(ham)))
$i  Orbitals         : $(displayorbitals(ham))
$i  Element type     : $(displayelements(ham))
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

displaymatrixtype(h::Hamiltonian) = displaymatrixtype(matrixtype(h))
displaymatrixtype(::Type{<:SparseMatrixCSC}) = "SparseMatrixCSC, sparse"
displaymatrixtype(::Type{<:Array}) = "Matrix, dense"
displaymatrixtype(A::Type{<:AbstractArray}) = string(A)
displayelements(h::Hamiltonian) = displayelements(blocktype(h))
displayelements(::Type{S}) where {N,T,S<:SMatrix{N,N,T}} = "$N × $N blocks ($T)"
displayelements(::Type{T}) where {T} = "scalar ($T)"
displayorbitals(h::Hamiltonian) =
    replace(replace(string(h.orbitals), "Symbol(\"" => ":"), "\")" => "")

SparseArrays.issparse(h::Hamiltonian{LA,L,M,A}) where {LA,L,M,A<:AbstractSparseMatrix} = true
SparseArrays.issparse(h::Hamiltonian{LA,L,M,A}) where {LA,L,M,A} = false

# Internal API #

latdim(h::Hamiltonian{LA}) where {E,L,LA<:AbstractLattice{E,L}} = L

matrixtype(::Hamiltonian{LA,L,M,A}) where {LA,L,M,A} = A
realtype(::Hamiltonian{<:Any,<:Any,M}) where {M} = real(eltype(M))

# find SVector type that can hold all orbital amplitudes in any lattice sites
orbitaltype(orbs, type::Type{Tv} = Complex{T}) where {T,Tv} =
    _orbitaltype(SVector{1,Tv}, orbs...)
_orbitaltype(::Type{S}, ::NTuple{D,NameType}, os...) where {N,Tv,D,S<:SVector{N,Tv}} =
    (M = max(N,D); _orbitaltype(SVector{M,Tv}, os...))
_orbitaltype(t::Type{SVector{N,Tv}}) where {N,Tv} = t
_orbitaltype(t::Type{SVector{1,Tv}}) where {Tv} = Tv

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

checkfinitedim(h::Hamiltonian{LA,L}) where {LA,L} =
    L == 0 && throw(ArgumentError("A finite-dimensional Hamiltonian is required, not zero-dimensional"))

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

_nnz(h::AbstractSparseMatrix) = nnz(h)
_nnz(h::DenseMatrix) = count(!iszero, h)

function _nnzdiag(s::SparseMatrixCSC)
    count = 0
    rowptrs = rowvals(s)
    for col in 1:size(s,2)
        for ptr in nzrange(s, col)
            rowptrs[ptr] == col && (count += 1; break)
        end
    end
    return count
end
_nnzdiag(s::Matrix) = count(!iszero, s[i,i] for i in 1:minimum(size(s)))

# Iteration tools #

struct EachIndexNonzeros{H}
    h::H
    rowrange::UnitRange{Int}
    colrange::UnitRange{Int}
end

eachindex_nz(h, rowrange = 1:size(h, 1), colrange = 1:size(h, 2)) =
    EachIndexNonzeros(h, rclamp(rowrange, 1:size(h, 1)), rclamp(colrange, 1:size(h, 2)))

function firststate(itr::EachIndexNonzeros{<:Hamiltonian}, nhar)
    m = itr.h.harmonics[nhar].h
    row, col = nextnonzero_row_col(m, itr)
    return (row, col, nhar)
end

function nextnonzero_row_col(m::DenseMatrix, itr, col = first(itr.colrange))
    for col´ in col:last(itr.colrange), row in itr.rowrange
        iszero(m[row, col´]) || return (row, col´)
    end
    # (0, 0) is sentinel for "no non-zero row for col´ >= col
    return (0, 0)
end

function nextnonzero_row_col(m::AbstractSparseMatrix, itr, col = first(itr.colrange))
    rows = rowvals(m)
    for col´ in col:last(itr.colrange)
        ptridx = findfirst(p -> isvalidrowcol(rows[p], col´, m, itr), nzrange(m, col´))
        ptridx === nothing || return (ptridx, col´)
    end
    # (0, 0) is sentinel for "no non-zero row for col´ >= col
    return (0, 0)
end

isvalidrowcol(row, col, m, itr) = row in itr.rowrange && !iszero(m[row, col])

function Base.iterate(itr::EachIndexNonzeros{<:Hamiltonian}, (ptridx, col, nhar) = firststate(itr, 1))
    nhar > length(itr.h.harmonics) && return nothing
    har = itr.h.harmonics[nhar]
    i = _iterate(har.h, itr, ptridx, col)
    if i === nothing
        nhar´ = nhar + 1
        return nhar´ > length(itr.h.harmonics) ? nothing : iterate(itr, firststate(itr, nhar´))
    else
        ((row, col), (ptridx, col)) = i
        return (row, col, har.dn), (ptridx, col, nhar)
    end
end

firststate(itr::EachIndexNonzeros{<:HamiltonianHarmonic}) = nextnonzero_row_col(itr.h.h, itr)

function Base.iterate(itr::EachIndexNonzeros{<:HamiltonianHarmonic}, (row, col) = firststate(itr))
    _iterate(itr.h.h, itr, row, col)
end

# Returns nothing or ((row, col), (nextrow, nextcol)), where row and nextrow can be a ptridx
function _iterate(m::AbstractSparseMatrix, itr, ptridx, col)
    col in itr.colrange || return nothing  # will also return nothing if col == 0 (sentinel)
    ptrs = nzrange(m, col)
    rows = rowvals(m)
    if ptridx <= length(ptrs)
        row = rows[ptrs[ptridx]]
        row in itr.rowrange && return (row, col), (ptridx + 1, col)
    end
    ptridx´, col´ = nextnonzero_row_col(m, itr, col + 1)
    return _iterate(m, itr, ptridx´, col´)
end

function _iterate(m::DenseMatrix, itr, row, col)
    col in itr.colrange || return nothing
    for row´ in row:last(itr.rowrange)
        iszero(m[row´, col]) || return (row´, col), (row´ + 1, col)
    end
    row´, col´ = nextnonzero_row_col(m, itr, col + 1)
    return _iterate(m, itr, row´, col´)
end

Base.IteratorSize(::EachIndexNonzeros) = Base.SizeUnknown()
Base.IteratorEltype(::EachIndexNonzeros) = Base.HasEltype()
Base.eltype(s::EachIndexNonzeros{<:Hamiltonian}) = Tuple{Int, Int, typeof(first(s.h.harmonics).dn)}
Base.eltype(s::EachIndexNonzeros{<:HamiltonianHarmonic}) = Tuple{Int, Int}

# stored_indices(h::Hamiltonian) = ((har.dn, rowvals(har.h)[ptr], col) for har in h.harmonics
#                                   for col in 1:size(har.h, 2) for ptr in nzrange(har.h, col))

# External API #
"""
    hamiltonian(lat[, model]; orbitals, type)

Create a `Hamiltonian` by additively applying `model::TighbindingModel` to the lattice `lat`
(see `hopping` and `onsite` for details on building tightbinding models).

The number of orbitals on each sublattice can be specified by the keyword `orbitals`
(otherwise all sublattices have one orbital by default). The following, and obvious
combinations, are possible formats for the `orbitals` keyword:

    orbitals = :a                # all sublattices have 1 orbital named :a
    orbitals = (:a,)             # same as above
    orbitals = (:a, :b, 3)       # all sublattices have 3 orbitals named :a and :b and :3
    orbitals = ((:a, :b), (:c,)) # first sublattice has 2 orbitals, second has one
    orbitals = ((:a, :b), :c)    # same as above
    orbitals = (Val(2), Val(1))  # same as above, with automatic names
    orbitals = (:A => (:a, :b), :D => :c) # sublattice :A has two orbitals, :D and rest have one
    orbitals = :D => Val(4)      # sublattice :D has four orbitals, rest have one

The matrix sizes of tightbinding `model` must match the orbitals specified. Internally, we
define a block size `N = max(num_orbitals)`. If `N = 1` (all sublattices with one orbital)
the the Hamiltonian element type is `type`. Otherwise it is `SMatrix{N,N,type}` blocks,
padded with the necessary zeros as required. Keyword `type` is `Complex{T}` by default,
where `T` is the number type of `lat`.

    h(ϕ₁, ϕ₂, ...)
    h((ϕ₁, ϕ₂, ...))

Build the Bloch Hamiltonian matrix `bloch(h, (ϕ₁, ϕ₂, ...))` of a `h::Hamiltonian` on an
`L`D lattice. (See also `bloch!` for a non-allocating version of `bloch`.)

    lat |> hamiltonian(model[, funcmodel]; kw...)

Functional `hamiltonian` form equivalent to `hamiltonian(lat, model[, funcmodel]; kw...)`.

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
Hamiltonian{<:Lattice} : 2D Hamiltonian on a 2D Lattice in 2D space
  Bloch harmonics  : 5 (SparseMatrixCSC, sparse)
  Harmonic size    : 2 × 2
  Orbitals         : ((:a, :a), (:a, :a))
  Element type     : 2 × 2 blocks (Complex{Float64})
  Onsites          : 0
  Hoppings         : 6
  Coordination     : 3.0

julia> push!(h, (3,3)) # Adding a new Hamiltonian harmonic (if not already present)
Hamiltonian{<:Lattice} : 2D Hamiltonian on a 2D Lattice in 2D space
  Bloch harmonics  : 6 (SparseMatrixCSC, sparse)
  Harmonic size    : 2 × 2
  Orbitals         : ((:a, :a), (:a, :a))
  Element type     : 2 × 2 blocks (Complex{Float64})
  Onsites          : 0
  Hoppings         : 6
  Coordination     : 3.0

julia> h[3,3][1,1] = @SMatrix[1 2; 2 1]; h[3,3] # element assignment
2×2 SparseArrays.SparseMatrixCSC{StaticArrays.SArray{Tuple{2,2},Complex{Float64},2,4},Int64} with 1 stored entry:
  [1, 1]  =  [1.0+0.0im 2.0+0.0im; 2.0+0.0im 1.0+0.0im]

julia> h[3,3][[1,2],[1,2]] .= rand(SMatrix{2,2,Float64}, 2, 2) # Broadcast assignment
2×2 view(::SparseArrays.SparseMatrixCSC{StaticArrays.SArray{Tuple{2,2},Complex{Float64},2,4},Int64}, [1, 2], [1, 2]) with eltype StaticArrays.SArray{Tuple{2,2},Complex{Float64},2,4}:
 [0.271152+0.0im 0.921417+0.0im; 0.138212+0.0im 0.525911+0.0im]  [0.444284+0.0im 0.280035+0.0im; 0.565106+0.0im 0.121869+0.0im]
 [0.201126+0.0im 0.912446+0.0im; 0.372099+0.0im 0.931358+0.0im]  [0.883422+0.0im 0.874016+0.0im; 0.296095+0.0im 0.995861+0.0im]

julia> hopfunc(;k = 0) = hopping(k); hamiltonian(LatticePresets.square(), onsite(1) + hopping(2), hopfunc) # Parametric Hamiltonian
Parametric Hamiltonian{<:Lattice} : 2D Hamiltonian on a 2D Lattice in 2D space
  Bloch harmonics  : 5 (SparseMatrixCSC, sparse)
  Harmonic size    : 1 × 1
  Orbitals         : ((:a,),)
  Elements         : scalars (Complex{Float64})
  Onsites          : 1
  Hoppings         : 4
  Coordination     : 4.0

```
"""
hamiltonian(lat, ts...; orbitals = missing, kw...) =
    _hamiltonian(lat, sanitize_orbs(orbitals, lat.unitcell.names), ts...; kw...)
_hamiltonian(lat::AbstractLattice, orbs; kw...) = _hamiltonian(lat, orbs, TightbindingModel(); kw...)
_hamiltonian(lat::AbstractLattice, orbs, f::Function; kw...) = _hamiltonian(lat, orbs, TightbindingModel(), f; kw...)
_hamiltonian(lat::AbstractLattice, orbs, m::TightbindingModel; type::Type = Complex{numbertype(lat)}, kw...) =
    hamiltonian_sparse(blocktype(orbs, type), lat, orbs, m; kw...)

hamiltonian(t::TightbindingModel...; kw...) =
    z -> hamiltonian(z, t...; kw...)
hamiltonian(f::Function, t::TightbindingModel...; kw...) =
    z -> hamiltonian(z, f, t...; kw...)
hamiltonian(h::Hamiltonian) =
    z -> hamiltonian(z, h)

(h::Hamiltonian)(phases...) = bloch(h, phases...)

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

Base.Matrix(h::Hamiltonian) = Hamiltonian(h.lattice, Matrix.(h.harmonics), h.orbitals)
Base.Matrix(h::HamiltonianHarmonic) = HamiltonianHarmonic(h.dn, Matrix(h.h))

Base.copy(h::Hamiltonian) = Hamiltonian(copy(h.lattice), copy.(h.harmonics), h.orbitals)
Base.copy(h::HamiltonianHarmonic) = HamiltonianHarmonic(h.dn, copy(h.h))

Base.size(h::Hamiltonian, n) = size(first(h.harmonics).h, n)
Base.size(h::Hamiltonian) = size(first(h.harmonics).h)
Base.size(h::HamiltonianHarmonic, n) = size(h.h, n)
Base.size(h::HamiltonianHarmonic) = size(h.h)

function LinearAlgebra.ishermitian(h::Hamiltonian)
    for hh in h.harmonics
        isnonnegative(hh.dn) || continue
        isassigned(h, -hh.dn) || return false
        hh.h == h[-hh.dn]' || return false
    end
    return true
end

bravais(h::Hamiltonian) = bravais(h.lattice)

issemibounded(h::Hamiltonian) = issemibounded(h.lattice)

nsites(h::Hamiltonian) = isempty(h.harmonics) ? 0 : nsites(first(h.harmonics))
nsites(h::HamiltonianHarmonic) = size(h.h, 1)

nsublats(h::Hamiltonian) = nsublats(h.lattice)

norbitals(h::Hamiltonian) = length.(h.orbitals)

# External API #

"""
    transform!(h::Hamiltonian, f::Function)

Transform the site positions of the Hamiltonian's lattice in place without modifying the
Hamiltonian harmonics.
"""
function transform!(h::Hamiltonian, f::Function)
    transform!(h.lattice, f)
    return h
end

# Indexing #

Base.push!(h::Hamiltonian{<:Any,L}, dn::NTuple{L,Int}) where {L} = push!(h, SVector(dn...))
Base.push!(h::Hamiltonian{<:Any,L}, dn::Vararg{Int,L}) where {L} = push!(h, SVector(dn...))
function Base.push!(h::Hamiltonian{<:Any,L,M,A}, dn::SVector{L,Int}) where {L,M,A}
    for hh in h.harmonics
        hh.dn == dn && return hh
    end
    hh = HamiltonianHarmonic{L,M,A}(dn, size(h)...)
    push!(h.harmonics, hh)
    return h
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
function hamiltonian_sparse(Mtype, lat, orbs, model)
    builder = IJVBuilder(lat, orbs, Mtype)
    return hamiltonian_sparse!(builder, lat, orbs, model)
end

function hamiltonian_sparse!(builder::IJVBuilder{L,M}, lat::AbstractLattice{E,L}, orbs, model) where {E,L,M}
    checkmodelorbs(model, orbs, lat)
    applyterms!(builder, terms(model)...)
    n = nsites(lat)
    HT = HamiltonianHarmonic{L,M,SparseMatrixCSC{M,Int}}
    harmonics = HT[HT(e.dn, sparse(e.i, e.j, e.v, n, n)) for e in builder.ijvs if !isempty(e)]
    return Hamiltonian(lat, harmonics, orbs, n, n)
end


applyterms!(builder, terms...) = foreach(term -> applyterm!(builder, term), terms)

applyterm!(builder::IJVBuilder, term::Union{OnsiteTerm, HoppingTerm}) =
    applyterm!(builder, term, sublats(term, builder.lat))

function applyterm!(builder::IJVBuilder{L,M}, term::OnsiteTerm, termsublats) where {L,M}
    selector = term.selector
    lat = builder.lat
    for s in termsublats
        is = siterange(lat, s)
        dn0 = zero(SVector{L,Int})
        ijv = builder[dn0]
        offset = lat.unitcell.offsets[s]
        for i in is
            isinregion(i, dn0, selector.region, lat) || continue
            r = lat.unitcell.sites[i]
            vs = orbsized(term(r,r), builder.orbs[s])
            v = padtotype(vs, M)
            term.forcehermitian ? push!(ijv, (i, i, 0.5 * (v + v'))) : push!(ijv, (i, i, v))
        end
    end
    return nothing
end

function applyterm!(builder::IJVBuilder{L,M}, term::HoppingTerm, termsublats) where {L,M}
    selector = term.selector
    checkinfinite(selector)
    lat = builder.lat
    for (s1, s2) in termsublats
        is, js = siterange(lat, s1), siterange(lat, s2)
        dns = dniter(selector.dns, Val(L))
        for dn in dns
            addadjoint = term.forcehermitian
            foundlink = false
            ijv = builder[dn]
            addadjoint && (ijvc = builder[negative(dn)])
            for j in js
                sitej = lat.unitcell.sites[j]
                rsource = sitej - lat.bravais.matrix * dn
                itargets = targets(builder, selector.range, rsource, s1)
                for i in itargets
                    isselfhopping((i, j), (s1, s2), dn) && continue
                    isinregion((i, j), (dn, zero(dn)), selector.region, lat) || continue
                    foundlink = true
                    rtarget = lat.unitcell.sites[i]
                    r, dr = _rdr(rsource, rtarget)
                    vs = orbsized(term(r, dr), builder.orbs[s1], builder.orbs[s2])
                    v = padtotype(vs, M)
                    if addadjoint
                        v *= redundancyfactor(dn, (s1, s2), selector)
                        push!(ijv, (i, j, v))
                        push!(ijvc, (j, i, v'))
                    else
                        push!(ijv, (i, j, v))
                    end
                end
            end
            foundlink && acceptcell!(dns, dn)
        end
    end
    return nothing
end

orbsized(m, orbs) = orbsized(m, orbs, orbs)
orbsized(m, o1::NTuple{D1}, o2::NTuple{D2}) where {D1,D2} = padtotype(m, SMatrix{D1,D2})

dniter(dns::Missing, ::Val{L}) where {L} = BoxIterator(zero(SVector{L,Int}))
dniter(dns, ::Val) = dns

function targets(builder, range::Real, rsource, s1)
    if !isassigned(builder.kdtrees, s1)
        sites = view(builder.lat.unitcell.sites, siterange(builder.lat, s1))
        (builder.kdtrees[s1] = KDTree(sites))
    end
    targets = inrange(builder.kdtrees[s1], rsource, range)
    targets .+= builder.lat.unitcell.offsets[s1]
    return targets
end

targets(builder, range::Missing, rsource, s1) = eachindex(builder.lat.sublats[s1].sites)

checkinfinite(selector) =
    selector.dns === missing && (selector.range === missing || !isfinite(selector.range)) &&
    throw(ErrorException("Tried to implement an infinite-range hopping on an unbounded lattice"))

isselfhopping((i, j), (s1, s2), dn) = i == j && s1 == s2 && iszero(dn)

# Avoid double-counting hoppings when adding adjoint
redundancyfactor(dn, ss, selector) =
    isnotredundant(dn, selector) || isnotredundant(ss, selector) ? 1.0 : 0.5
isnotredundant(dn::SVector, selector) = selector.dns !== missing && !iszero(dn)
isnotredundant((s1, s2)::Tuple{Int,Int}, selector) = selector.sublats !== missing && s1 != s2

#######################################################################
# unitcell/supercell for Hamiltonians
#######################################################################
function supercell(ham::Hamiltonian, args...; kw...)
    slat = supercell(ham.lattice, args...; kw...)
    return Hamiltonian(slat, ham.harmonics, ham.orbitals)
end

function unitcell(ham::Hamiltonian{<:Lattice}, args...; modifiers = (), kw...)
    sham = supercell(ham, args...; kw...)
    return unitcell(sham; modifiers = modifiers)
end

function unitcell(ham::Hamiltonian{LA,L}; modifiers = ()) where {E,L,T,L´,LA<:Superlattice{E,L,T,L´}}
    lat = ham.lattice
    sc = lat.supercell
    modifiers´ = resolve.(ensuretuple(modifiers), Ref(lat))
    mapping = OffsetArray{Int}(undef, sc.sites, sc.cells.indices...) # store supersite indices newi
    mapping .= 0
    foreach_supersite((s, oldi, olddn, newi) -> mapping[oldi, Tuple(olddn)...] = newi, lat)
    dim = nsites(sc)
    B = blocktype(ham)
    S = typeof(SparseMatrixBuilder{B}(dim, dim))
    harmonic_builders = HamiltonianHarmonic{L´,B,S}[]
    pinvint = pinvmultiple(sc.matrix)
    foreach_supersite(lat) do s, source_i, source_dn, newcol
        for oldh in ham.harmonics
            rows = rowvals(oldh.h)
            vals = nonzeros(oldh.h)
            target_dn = source_dn + oldh.dn
            super_dn = new_dn(target_dn, pinvint)
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
    orbs = ham.orbitals
    return Hamiltonian(unitlat, harmonics, orbs)
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

wrap_dn(olddn::SVector, newdn::SVector, supercell::SMatrix) = olddn - supercell * newdn

applymodifiers(val, lat, inds, dns) = val

function applymodifiers(val, lat, inds, dns, m::ElementModifier{Val{false}}, ms...)
    selected = m.selector(lat, inds, dns)
    val´ = selected ? m.f(val) : val
    return applymodifiers(val´, lat, inds, dns, ms...)
end

function applymodifiers(val, lat, (row, col), (dnrow, dncol), m::Onsite!{Val{true}}, ms...)
    selected = m.selector(lat, (row, col), (dnrow, dncol))
    if selected
        r = sites(lat)[col] + bravais(lat) * dncol
        val´ = selected ? m.f(val, r) : val
    else
        val´ = val
    end
    return applymodifiers(val´, lat, (row, col), (dnrow, dncol), ms...)
end

function applymodifiers(val, lat, (row, col), (dnrow, dncol), m::Hopping!{Val{true}}, ms...)
    selected = m.selector(lat, (row, col), (dnrow, dncol))
    if selected
        br = bravais(lat)
        r, dr = _rdr(sites(lat)[col] + br * dncol, sites(lat)[row] + br * dnrow)
        val´ = selected ? m.f(val, r, dr) : val
    else
        val´ = val
    end
    return applymodifiers(val´, lat, (row, col), (dnrow, dncol), ms...)
end

#######################################################################
# wrap
#######################################################################
"""
    wrap(h::Hamiltonian, axis::Int; factor = 1)

Build a new Hamiltonian wherein the Bravais `axis` is wrapped into a loop. If a `factor` is
given, the wrapped hoppings will be multiplied by said factor. This is useful to represent a
flux Φ through the loop, if `factor = exp(im * 2π * Φ/Φ₀)`.

    h |> wrap(axis; kw...)

Functional form equivalent to `wrap(h, axis; kw...)`.

# Examples
```
julia> LatticePresets.honeycomb() |> hamiltonian(hopping(1, range = 1/√3)) |> unitcell((1,-1), (10, 10)) |> wrap(2)
Hamiltonian{<:Lattice} : 1D Hamiltonian on a 1D Lattice in 2D space
  Bloch harmonics  : 3 (SparseMatrixCSC, sparse)
  Harmonic size    : 40 × 40
  Orbitals         : ((:a,), (:a,))
  Element type     : scalar (Complex{Float64})
  Onsites          : 0
  Hoppings         : 120
  Coordination     : 3.0
```
"""
function wrap(h::Hamiltonian{<:Lattice,L}, axis; factor = 1) where {L}
    1 <= axis <= L || throw(ArgumentError("wrap axis should be between 1 and the lattice dimension $L"))
    lattice´ = _wrap(h.lattice, axis)
    harmonics´ = _wrap(h.harmonics, axis, factor)
    return Hamiltonian(lattice´, harmonics´, h.orbitals)
end

wrap(axis; kw...) = h -> wrap(h, axis; kw...)

_wrap(lat::Lattice, axis) = Lattice(_wrap(lat.bravais, axis), lat.unitcell)

function _wrap(br::Bravais{E,L}, axis) where {E,L}
    mask = deleteat(SVector{L}(1:L), axis)
    return Bravais(br.matrix[:, mask], br.semibounded[mask])
end

function _wrap(harmonics::Vector{HamiltonianHarmonic{L,M,A}}, axis, factor) where {L,M,A}
    harmonics´ = HamiltonianHarmonic{L-1,M,A}[]
    for har in harmonics
        dn = har.dn
        dn´ = deleteat(dn, axis)
        factor´ = iszero(dn[axis]) ? 1 : factor
        add_or_push!(harmonics´, dn´, har.h, factor´)
    end
    return harmonics´
end

function add_or_push!(hs::Vector{<:HamiltonianHarmonic}, dn, matrix::AbstractMatrix, factor)
    for h in hs
        if h.dn == dn
            h.h .+= factor .* matrix
            return h
        end
    end
    newh = HamiltonianHarmonic(dn, matrix)
    push!(hs, newh)
    return newh
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
struct SupercellBloch{L,T,H<:Hamiltonian{<:Superlattice}}
    hamiltonian::H
    phases::SVector{L,T}
    axis::Int
end
SupercellBloch(h, ϕs) = SupercellBloch(h, ϕs, 0)

Base.summary(h::SupercellBloch{L,T}) where {L,T} =
    "SupercellBloch{$L)}: Bloch Hamiltonian matrix lazily defined on an $(L)D supercell"

function Base.show(io::IO, sb::SupercellBloch)
    ioindent = IOContext(io, :indent => string("  "))
    print(io, summary(sb), "
  Phases          : $(Tuple(sb.phases))
  Axis            : $(iszero(sb.axis) ? "none" : sb.axis)\n")
    print(ioindent, sb.hamiltonian.lattice.supercell)
end

"""
    bloch!(matrix, h::Hamiltonian, ...)

In-place version of `bloch`. Overwrite `matrix` with the Bloch Hamiltonian matrix of `h` for
the specified Bloch phases `ϕs` (see `bloch` for definition and API).  A conventient way to
obtain a `matrix` is to use `similarmatrix(h)`, which will return an `AbstractMatrix` of the
same type as the Hamiltonian's. Note, however, that matrix need not be of the same type
(e.g. it can be dense, while `h` is sparse).

# Examples
```
julia> h = LatticePresets.honeycomb() |> hamiltonian(onsite(1), hopping(2));

julia> bloch!(similarmatrix(h), h, (.2,.3), 2)  # velocity operator along `bravais(h)[2]`
2×2 SparseArrays.SparseMatrixCSC{Complex{Float64},Int64} with 4 stored entries:
  [1, 1]  =  1.99001-0.199667im
  [2, 1]  =  1.96013-0.397339im
  [1, 2]  =  1.96013+0.397339im
  [2, 2]  =  1.99001-0.199667im
```

# See also:
    bloch, optimize!, similarmatrix
"""
bloch!(matrix, h, ϕs::Number...) = _bloch!(matrix, h, toSVector(ϕs), 0)
bloch!(matrix, h, ϕs::Tuple, axis = 0) = _bloch!(matrix, h, toSVector(ϕs), axis)
bloch!(matrix, h, ϕs::SVector, axis = 0) = _bloch!(matrix, h, ϕs, axis)
bloch!(matrix, ϕs::Number...) = h -> bloch!(matrix, h, ϕs...)
bloch!(matrix, ϕs::Tuple, axis = 0) = h -> bloch!(matrix, h, ϕs, axis)

function _bloch!(matrix::AbstractMatrix, h::Hamiltonian{<:Lattice,L,M}, ϕs, axis::Number) where {L,M}
    rawmatrix = parent(matrix)
    if iszero(axis)
        _copy!(rawmatrix, first(h.harmonics).h) # faster copy!(dense, sparse) specialization
        add_harmonics!(rawmatrix, h, ϕs, dn -> 1)
    else
        fill!(rawmatrix, zero(M)) # There is no guarantee of same structure
        add_harmonics!(rawmatrix, h, ϕs, dn -> -im * dn[axis])
    end
    return matrix
end

function _bloch!(matrix::AbstractMatrix, h::Hamiltonian{<:Lattice,L,M}, ϕs, dnfunc::Function) where {L,M}
    prefactor0 = dnfunc(zero(ϕs))
    rawmatrix = parent(matrix)
    if iszero(prefactor0)
        fill!(rawmatrix, zero(M))
    else
        _copy!(rawmatrix, first(h.harmonics).h)
        rmul!(rawmatrix, prefactor0)
    end
    add_harmonics!(rawmatrix, h, ϕs, dnfunc)
    return matrix
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
        _add!(zerobloch, hhmatrix, ephi)
    end
    return zerobloch
end


"""
    bloch(h::Hamiltonian{<:Lattice}, ϕs::Real...)

Build the Bloch Hamiltonian matrix of `h`, for the specified Bloch phases `ϕs`. In terms of
Bloch wavevector `k`, `ϕs = k * bravais(h)`, it is defined as Overwrite `matrix` with the
Bloch Hamiltonian matrix of `h`, for the specified Bloch phases `ϕs`, defined as
`H(ϕs) = ∑exp(-im * ϕs' * dn) h_dn` where `h_dn` are Bloch harmonics connecting unit cells
at a distance `dR = bravais(h) * dn`.

    bloch(h::Hamiltonian{<:Lattice})

Build the intra-cell Hamiltonian matrix of `h`, without adding any Bloch harmonics.

    bloch(h::Hamiltonian{<:Lattice}, ϕs::NTuple{L,Real}[, axis::Int = 0])

A nonzero `axis` produces the derivative of the Bloch matrix respect to `ϕs[axis]` (i.e. the
velocity operator along this axis), `∂H(ϕs) = ∑ -im * dn[axis] * exp(-im * ϕs' * dn) h_dn`

    bloch(matrix, h::Hamiltonian{<:Lattice}, ϕs::NTuple{L,Real}, dnfunc::Function)

Generalization that applies a prefactor `dnfunc(dn) * exp(im * ϕs' * dn)` to the `dn`
harmonic.

    h |> bloch(ϕs...)
    h(ϕs...)

Functional forms of `bloch`, equivalent to `bloch(h, ϕs...)`

    bloch(h::Hamiltonian{<:Superlattice}, ϕs::Number...)
    bloch(h::Hamiltonian{<:Superlattice}, ϕs::Tuple[, axis = 0])

Build a `SupercellBloch` object that lazily implements the Bloch Hamiltonian in the
`Superlattice` without actually building the matrix (e.g. for matrix-free diagonalization).

# Notes

`bloch` allocates a new matrix on each call. For a non-allocating version of `bloch`, see
`bloch!`.

# Examples
```
julia> h = LatticePresets.honeycomb() |> hamiltonian(onsite(1), hopping(2)) |> bloch(.2,.3)
2×2 SparseArrays.SparseMatrixCSC{Complex{Float64},Int64} with 4 stored entries:
  [1, 1]  =  1.99001-0.199667im
  [2, 1]  =  1.96013-0.397339im
  [1, 2]  =  1.96013+0.397339im
  [2, 2]  =  1.99001-0.199667im
```

# See also:
    bloch!, optimize!, similarmatrix
"""
bloch(ϕs::Number...) = h -> bloch(h, ϕs...)
bloch(ϕs::Tuple, axis = 0) = h -> bloch(h, ϕs, axis)
bloch(h::Hamiltonian{<:Lattice}, args...) = bloch!(similarmatrix(h), h, args...)
bloch(h::Hamiltonian{<:Superlattice}, ϕs::Number...) =
    SupercellBloch(h, toSVector(ϕs))
bloch(h::Hamiltonian{<:Superlattice}, ϕs::Tuple, axis = 0) =
    SupercellBloch(h, toSVector(ϕs), axis)

"""
    similarmatrix(h::Hamiltonian; optimize = true)

Create an uninitialized matrix of the same type of the Hamiltonian's matrix, calling
`optimize!(h)` first if `optimize = true` to produce an optimal work matrix in the sparse
case.
"""
function similarmatrix(h::Hamiltonian; optimize = true)
    optimize && optimize!(h)
    sm = size(h)
    T = eltype(h)
    matrix = similar(h.harmonics[1].h, T, sm[1], sm[2])
    return matrix
end

"""
    optimize!(h::Hamiltonian)

Prepare a sparse Hamiltonian `h` to increase the performance of subsequent calls to
`bloch(h, ϕs...)` and `bloch!(matrix, h, ϕs...)` by minimizing memory reshufflings. It also
adds missing structural zeros to the diagonal to enable shifts by `α*I` (for
shift-and-invert methods).

No optimization will be performed on non-sparse Hamiltonians, or those defined on
`Superlattice`s, for which Bloch Hamiltonians are lazily evaluated.

Note that when calling `similarmatrix(h)` on a sparse `h`, `optimize!` is called first.

# See also:
    bloch, bloch!
"""
function optimize!(ham::Hamiltonian{<:Lattice,L,M,A}) where {L,M,A<:SparseMatrixCSC}
    h0 = first(ham.harmonics)
    n, m = size(h0.h)
    iszero(h0.dn) || throw(ArgumentError("First Hamiltonian harmonic is not the fundamental"))
    nh = length(ham.harmonics)
    builder = SparseMatrixBuilder{M}(n, m)
    for col in 1:m
        for i in eachindex(ham.harmonics)
            h = ham.harmonics[i].h
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
    copy!(h0.h, ho) # Inject new structural zeros into zero harmonics
    return ham
end
# IDEA: could sum and subtract all harmonics instead
# Tested, it is slower

function optimize!(ham::Hamiltonian{<:Lattice,L,M,A}) where {L,M,A<:AbstractMatrix}
    # @warn "Hamiltonian is not sparse. Nothing changed."
    return ham
end

function optimize!(ham::Hamiltonian{<:Superlattice})
    # @warn "Hamiltonian is defined on a Superlattice. Nothing changed."
    return ham
end


"""
    flatten(h::Hamiltonian)

Flatten a multiorbital Hamiltonian `h` into one with a single orbital per site. The
associated lattice is flattened also, so that there is one site per orbital for each initial
site (all at the same position). Note that in the case of sparse Hamiltonians, zeros in
hopping/onsite matrices are preserved as structural zeros upon flattening.

    h |> flatten()

Functional form equivalent to `flatten(h)` of `h |> flatten` (included for consistency with
the rest of the API).

# Examples
```
julia> h = LatticePresets.honeycomb() |> hamiltonian(hopping(@SMatrix[1 2], range = 1/√3, sublats = (:A,:B)), orbitals = (Val(1), Val(2)))
Hamiltonian{<:Lattice} : 2D Hamiltonian on a 2D Lattice in 2D space
  Bloch harmonics  : 5 (SparseMatrixCSC, sparse)
  Harmonic size    : 2 × 2
  Orbitals         : ((:a,), (:a, :a))
  Element type     : 2 × 2 blocks (Complex{Float64})
  Onsites          : 0
  Hoppings         : 6
  Coordination     : 3.0

julia> flatten(h)
Hamiltonian{<:Lattice} : 2D Hamiltonian on a 2D Lattice in 2D space
  Bloch harmonics  : 5 (SparseMatrixCSC, sparse)
  Harmonic size    : 3 × 3
  Orbitals         : ((:flat,), (:flat,))
  Element type     : scalar (Complex{Float64})
  Onsites          : 0
  Hoppings         : 12
  Coordination     : 4.0
```
"""
flatten() = h -> flatten(h)

function flatten(h::Hamiltonian)
    all(isequal(1), norbitals(h)) && return copy(h)
    harmonics´ = [flatten(har, h.orbitals, h.lattice) for har in h.harmonics]
    lattice´ = flatten(h.lattice, h.orbitals)
    orbitals´ = (_ -> (:flat, )).(h.orbitals)
    return Hamiltonian(lattice´, harmonics´, orbitals´)
end

flatten(h::HamiltonianHarmonic, orbs, lat) =
    HamiltonianHarmonic(h.dn, _flatten(h.h, length.(orbs), lat))

function _flatten(src::SparseMatrixCSC{<:SMatrix{N,N,T}}, norbs::NTuple{S,<:Any}, lat) where {N,T,S}
    offsets´ = flattenoffsets(lat.unitcell.offsets, norbs)
    dim´ = last(offsets´)

    builder = SparseMatrixBuilder{T}(dim´, dim´, nnz(src) * N * N)

    for col in 1:size(src, 2)
        scol = sublat(lat, col)
        for j in 1:norbs[scol]
            for p in nzrange(src, col)
                row = rowvals(src)[p]
                srow = sublat(lat, row)
                rowoffset´ = flattenind(row, lat, norbs, offsets´) - 1
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

function _flatten(src::DenseMatrix{<:SMatrix{N,N,T}}, norbs::NTuple{S,<:Any}, lat) where {N,T,S}
    offsets´ = flattenoffsets(lat.unitcell.offsets, norbs)
    dim´ = last(offsets´)
    matrix = similar(src, T, dim´, dim´)

    for col in 1:size(src, 2), row in 1:size(src, 1)
        srow, scol = sublat(lat, row), sublat(lat, col)
        nrow, ncol = norbs[srow], norbs[scol]
        val = src[row, col]
        rowoffset´ = flattenind(row, lat, norbs, offsets´) - 1
        coloffset´ = flattenind(col, lat, norbs, offsets´) - 1
        for j in 1:ncol, i in 1:nrow
            matrix[rowoffset´ + i, coloffset´ + j] = val[i, j]
        end
    end
    return matrix
end

# sublat offsets after flattening (without padding zeros)
function flattenoffsets(offsets, norbs)
    offsets´ = similar(offsets)
    i = 0
    for s in eachindex(norbs)
        offsets´[s] = i
        ns = offsets[s + 1] - offsets[s]
        no = norbs[s]
        i += ns * no
    end
    offsets´[end] = i
    return offsets´
end

# index of first orbital of site i after flattening
function flattenind(i, lat, norbs, offsets´)
    s = sublat(lat, i)
    offset = lat.unitcell.offsets[s]
    Δi = i - offset
    i´ = offsets´[s] + (Δi - 1) * norbs[s] + 1
    return i´
end

function flatten(lat::Lattice, orbs)
    length(orbs) == nsublats(lat) || throw(ArgumentError("Msmatch between sublattices and orbitals"))
    unitcell´ = flatten(lat.unitcell, length.(orbs))
    bravais´ = lat.bravais
    lat´ = Lattice(bravais´, unitcell´)
end

function flatten(unitcell::Unitcell, norbs::NTuple{S,Int}) where {S}
    offsets´ = flattenoffsets(unitcell.offsets, norbs)
    ns´ = last(offsets´)
    sites´ = similar(unitcell.sites, ns´)
    i = 1
    for sl in 1:S, site in sites(unitcell, sl), rep in 1:norbs[sl]
        sites´[i] = site
        i += 1
    end
    names´ = unitcell.names
    unitcell´ = Unitcell(sites´, names´, offsets´)
    return unitcell´
end