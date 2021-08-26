#######################################################################
# Lattice
#######################################################################
"""
    sublat(sites...; name::$(NameType))
    sublat(sites::Vector{<:SVector}; name::$(NameType))

Create a `Sublat{E,T,D}` that adds a sublattice, of name `name`, with sites at positions
`sites` in `E` dimensional space. Sites can be entered as tuples or `SVectors`.

# Examples

```jldoctest
julia> sublat((0.0, 0), (1, 1), (1, -1), name = :A)
Sublat{2,Float64} : sublattice of Float64-typed sites in 2D space
  Sites    : 3
  Name     : :A
```
"""
sublat

"""
    bravais_mat(lat::Lattice)
    bravais_mat(h::Hamiltonian)

Obtain the Bravais matrix of lattice `lat` or Hamiltonian `h`

# Examples

```jldoctest
julia> lat = lattice(sublat((0,0)), bravais = ((1.0, 2), (3, 4)));

julia> bravais_mat(lat)
2×2 SMatrix{2, 2, Float64, 4} with indices SOneTo(2)×SOneTo(2):
 1.0  3.0
 2.0  4.0
```

# See also
    `lattice`
"""
bravais

"""
    lattice(sublats::Sublat...; bravais = (), dim, type, names)

Create a `Lattice{dim,L,type}` with `L` Bravais vectors `bravais` and sublattices `sublats`
converted to a common  `dim`-dimensional embedding space and type `type`.

The keyword `bravais` indicates one or more Bravais vectors in the form of tuples or other
iterables. It can also be an `AbstractMatrix` of dimension `E×L`. The default `bravais = ()`
corresponds to a bounded lattice with no Bravais vectors.

A keyword `names` can be used to rename `sublats`. Given names can be replaced to ensure
that all sublattice names are unique.

    lattice(lat::Lattice; bravais = missing, dim = missing, type = missing, names = missing)

Create a new lattice by applying any non-missing `kw` to `lat`. For performance, allocations
will be avoided if possible (depends on `kw`), so the result can share memory of `lat`. To
avoid that, do `lattice(copy(lat); kw...)`.

See also `LatticePresets` for built-in lattices.

# Examples

```jldoctest
julia> lattice(sublat((0, 0)), sublat((0, Float32(1))); bravais = (1, 0), dim = Val(3))
Lattice{3,1,Float32} : 1D lattice in 3D space
  Bravais vectors : ((1.0f0, 0.0f0, 0.0f0),)
  Sublattices     : 2
    Names         : (:A, :B)
    Sites         : (1, 1) --> 2 total per unit cell

julia> LatticePresets.honeycomb(names = (:C, :D))
Lattice{2,2,Float64} : 2D lattice in 2D space
  Bravais vectors : ((0.5, 0.866025), (-0.5, 0.866025))
  Sublattices     : 2
    Names         : (:C, :D)
    Sites         : (1, 1) --> 2 total per unit cell

julia> LatticePresets.cubic(bravais = ((1, 0), (0, 2)))
Lattice{3,2,Float64} : 2D lattice in 3D space
  Bravais vectors : ((1.0, 0.0, 0.0), (0.0, 2.0, 0.0))
  Sublattices     : 1
    Names         : (:A)
    Sites         : (1) --> 1 total per unit cell
```

# See also
    `LatticePresets`, `bravais`, `sublat`, `supercell`, `intracell`
"""
lattice

"""
    x |> transform!(f::Function)

Curried version of `transform!`, equivalent to `transform!(f, x)`

    transform!(f::Function, lat::Lattice)

Transform the site positions of `lat` by applying `f` to them in place.
"""
transform!

"""
    combine(lats::Lattice...)

If all `lats` have compatible Bravais vectors, combine them into a single lattice.
Sublattice names are renamed to be unique if necessary.
"""
combine

"""
    supercell(lat::Lattice{E,L}, v::NTuple{L,Integer}...; seed = missing, kw...)
    supercell(lat::Lattice{E,L}, uc::SMatrix{L,L´,Int}; seed = missing, kw...)

Generates a `Lattice` from an `L`-dimensional lattice `lat` and a larger unit cell, such
that its Bravais vectors are `br´= br * uc`. Here `uc::SMatrix{L,L´,Int}` is the integer
supercell matrix, with the `L´` vectors `v`s as columns. If no `v` are given, the new
lattice will be bounded.

Only sites selected by `siteselector(; kw...)` will be included in the supercell (see
`siteselector` for details on the available keywords `kw`). The search for included sites
will start from point `seed::Union{Tuple,SVector}`, or the origin if `seed = missing`. If no
keyword `region` is given in `kw`, a Bravais unit cell perpendicular to the `v` axes will be
selected for the `L-L´` non-periodic directions.

    supercell(lattice::Lattice{E,L}, factor::Integer; kw...)

Calls `supercell` with a uniformly scaled `uc = SMatrix{L,L}(factor * I)`

    supercell(lattice::Lattice{E,L}, factors::Integer...; kw...)

Calls `supercell` with different scaling along each Bravais vector (diagonal supercell
with factors along the diagonal)

    supercell(h::Hamiltonian, v...; mincoordination, modifiers = (), kw...)

Transforms the `Lattice` of `h` to have a larger unit cell, while expanding the Hamiltonian
accordingly.

A nonzero `mincoordination` indicates a minimum number of nonzero hopping neighbors required
for sites to be included in the resulting unit cell. Sites with inferior coordination will
be removed recursively, until all remaining satisfy `mincoordination`.

The `modifiers` (a tuple of `ElementModifier`s, either `@onsite!` or `@hopping!` with no
free parameters) will be applied to onsite and hoppings as the hamiltonian is expanded. See
`@onsite!` and `@hopping!` for details.

Note: for performance reasons, in sparse hamiltonians only the stored onsites and hoppings
will be transformed by `ElementModifier`s, so you might want to add zero onsites or hoppings
when building `h` to have a modifier applied to them later.

    lat_or_h |> supercell(v...; kw...)

Curried syntax, equivalent to `supercell(lat_or_h, v...; kw...)`

# Examples

```jldoctest
julia> supercell(LatticePresets.honeycomb(), region = RegionPresets.circle(300))
Lattice{2,0,Float64} : 0D lattice in 2D space
  Bravais vectors : ()
  Sublattices     : 2
    Names         : (:A, :B)
    Sites         : (326483, 326483) --> 652966 total per unit cell

julia> supercell(LatticePresets.triangular(), (1,1), (1, -1))
Lattice{2,2,Float64} : 2D lattice in 2D space
  Bravais vectors : ((0.0, 1.732051), (1.0, 0.0))
  Sublattices     : 1
    Names         : (:A)
    Sites         : (2) --> 2 total per unit cell

julia> LatticePresets.square() |> supercell(3)
Lattice{2,2,Float64} : 2D lattice in 2D space
  Bravais vectors : ((3.0, 0.0), (0.0, 3.0))
  Sublattices     : 1
    Names         : (:A)
    Sites         : (9) --> 9 total per unit cell

julia> supercell(LatticePresets.square(), 3) |> supercell
Lattice{2,2,Float64} : 2D lattice in 2D space
  Bravais vectors : ((3.0, 0.0), (0.0, 3.0))
  Sublattices     : 1
    Names         : (:A)
    Sites         : (9) --> 9 total per unit cell
```

# See also
    `supercell`, `siteselector`
"""
supercell


"""
    siteselector(; region = missing, sublats = missing, indices = missing)

Return a `SiteSelector` object that can be used to select sites in a lattice contained
within the specified region and sublattices. Only sites with index `i`, at position `r` and
belonging to a sublattice with name `s::NameType` will be selected if

    `region(r) && s in sublats && i in indices`

Any missing `region`, `sublat` or `indices` will not be used to constraint the selection.

The keyword `sublats` allows the following formats:

    sublats = :A                    # Sites on sublat :A only
    sublats = (:A,)                 # Same as above
    sublats = (:A, :B)              # Sites on sublat :A and :B

The keyword `indices` accepts a single integer, or a collection thereof. If several
collections are given, they are flattened into a single one. Possible combinations:

    indices = 1                     # Site 1 only
    indices = (1, )                 # Same as above
    indices = (1, 2, 3)             # Sites 1, 2 or 3
    indices = [1, 2, 3]             # Same as above
    indices = 1:3                   # Same as above
    indices = (1:3, 7, 8)           # Sites 1, 2, 3, 7 or 8

Additionally, indices or sublattices can be wrapped in `not` to exclude them (see `not`):

    sublats = not(:A)               # Any sublat different from :A
    sublats = not(:A, :B)           # Any sublat different from :A and :B
    indices = not(8)                # Any site index different from 8
    indices = not(1, 3:4)           # Any site index different from 1, 3 or 4
    indices = (not(3:4), 4:6)       # Any site different from 3 and 4, *or* equal to 4, 5 or 6
"""
siteselector

"""
    hopselector(; range = missing, dn = missing, sublats = missing, indices = missing, region = missing)

Return a `HopSelector` object that can be used to select hops between two sites in a
lattice. Only hops between two sites, with indices `ipair = src => dst`, at positions `r₁ =
r - dr/2` and `r₂ = r + dr`, belonging to unit cells at integer distance `dn´` and to
sublattices `s₁` and `s₂` will be selected if:

    `region(r, dr) && s in sublats && dn´ in dn && norm(dr) <= range && ipair in indices`

If any of these is `missing` it will not be used to constraint the selection.

The keyword `range` admits the following possibilities

    max_range                   # i.e. `norm(dr) <= max_range`
    (min_range, max_range)      # i.e. `min_range <= norm(dr) <= max_range`

Both `max_range` and `min_range` can be a `Real` or a `NeighborRange` created with
`neighbors(n)`. The latter represents the distance of `n`-th nearest neighbors.

The keyword `dn` can be a `Tuple`/`Vector`/`SVector` of `Int`s, or a tuple thereof.

The keyword `sublats` allows the following formats:

    sublats = :A => :B                  # Hopping from :A to :B sublattices, but not from :B to :A
    sublats = (:A => :B,)               # Same as above
    sublats = (:A => :B, :C => :D)      # Hopping from :A to :B or :C to :D
    sublats = (:A, :C) .=> (:B, :D)     # Broadcasted pairs, same as above
    sublats = (:A, :C) => (:B, :D)      # Direct product, (:A=>:B, :A=:D, :C=>:B, :C=>D)

The keyword `indices` accepts a single `src => dest` pair or a collection thereof. Any `src
== dest` will be neglected. Possible combinations:

    indices = 1 => 2                    # Hopping from site 1 to 2, but not from 2 to 1
    indices = (1 => 2, 2 => 1)          # Hoppings from 1 to 2 or from 2 to 1
    indices = [1 => 2, 2 => 1]          # Same as above
    indices = [(1, 2) .=> (2, 1)]       # Broadcasted pairs, same as above
    indices = [1:10 => 20:25, 3 => 30]  # Direct product, any hopping from sites 1:10 to sites 20:25, or from 3 to 30

Additionally, indices or sublattices can be wrapped in `not` to exclude them (see `not`):

    sublats = not(:A => :B, :B => :A)   # Any sublat pairs different from :A => :B or :B => :A
    sublats = not(:A) => :B             # Any sublat pair s1 => s2 with s1 different from :A and s2 equal to :B
    indices = not(8 => 9)               # Any site indices different from 8 => 9
    indices = 1 => not(3:4)             # Any site pair 1 => s with s different from 3, 4

"""
hopselector

"""
    neighbors(n::Int)

Create a `NeighborRange` that represents a hopping range to distances corresponding to the
n-th nearest neighbors in a given lattice. Such distance is obtained by finding the n-th
closest pairs of sites in a lattice, irrespective of their sublattice.

    neighbors(n::Int, lat::Lattice)

Obtain the actual nth-nearest-neighbot distance between sites in lattice `lat`.

# See also
    `hopping`
"""
neighbors

"""
    hamiltonian(lat, model; orbitals, orbtype)

Create a `Hamiltonian` by applying `model::TighbindingModel` to the lattice `lat` (see
`hopping` and `onsite` for details on building tightbinding models).

    lat |> hamiltonian(model; kw...)

Curried form of `hamiltonian` equivalent to `hamiltonian(lat, model; kw...)`.

# Keywords

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
the Hamiltonian element type is `orbtype`. Otherwise it is `SMatrix{N,N,orbtype}` blocks,
padded with the necessary zeros as required. Keyword `orbtype` is `Complex{T}` by default,
where `T` is the number type of `lat`.

# Indexing

Indexing into a Hamiltonian `h` works as follows. Access the `HamiltonianHarmonic` matrix at
a given `dn::NTuple{L,Int}` with `h[dn]`. The special `h[]` syntax stands for `h[(0...)]`
for the zero-harmonic. Assign `v` into element `(i,j)` of said matrix with `h[dn][i,j] = v`.
Broadcasting with vectors of indices `is` and `js` is supported, `h[dn][is, js] = v_matrix`.

A slicing syntax `h[rows, cols]` (without specifying `dn`) is also available, that creates a
special `hs::Slice{<:Hamiltonian}` object that represents a slice of the Hamiltonian matrix
restricted to `rows` and `cols`. Here, `rows` and `cols` are collections of site indices, or
alternatively `SiteSelector`s (see `siteselector`s for details). If `rows::Integer` or
`cols::Integer`, they will be converted to a single-element range (to preserve always a
matrix-like slice, unlike for `AbstractArray` indexing). Slices support `bloch` and `bloch!`
to produce the corresponding matrices, and can also be indexed as `hs[dn::Tuple]` that
produces the equivalent to `h[dn][rows, cols]`.

To add an empty harmonic with a given `dn::NTuple{L,Int}`, do `push!(h, dn)`. To delete it,
do `deleteat!(h, dn)`.

# Examples

```jldoctest
julia> h = hamiltonian(LatticePresets.honeycomb(), hopping(@SMatrix[1 2; 3 4], range = 1/√3), orbitals = Val(2))
Hamiltonian{<:Lattice} : Hamiltonian on a 2D Lattice in 2D space
  Bloch harmonics  : 5 (SparseMatrixCSC, sparse)
  Harmonic size    : 2 × 2
  Orbitals         : ((:a, :a), (:a, :a))
  Element type     : 2 × 2 blocks (ComplexF64)
  Onsites          : 0
  Hoppings         : 6
  Coordination     : 3.0

julia> push!(h, (3,3)) # Adding a new Hamiltonian harmonic (if not already present)
Hamiltonian{<:Lattice} : Hamiltonian on a 2D Lattice in 2D space
  Bloch harmonics  : 6 (SparseMatrixCSC, sparse)
  Harmonic size    : 2 × 2
  Orbitals         : ((:a, :a), (:a, :a))
  Element type     : 2 × 2 blocks (ComplexF64)
  Onsites          : 0
  Hoppings         : 6
  Coordination     : 3.0

julia> h[(3,3)][1,1] = @SMatrix[1 2; 2 1]; h[(3,3)] # element assignment
2×2 SparseMatrixCSC{SMatrix{2, 2, ComplexF64, 4}, Int64} with 1 stored entry:
 [1.0+0.0im 2.0+0.0im; 2.0+0.0im 1.0+0.0im]                      ⋅                     
                     ⋅                                           ⋅                     

julia> h[(3,3)][[1,2],[1,2]] .= Ref(@SMatrix[1 2; 2 1])
2×2 SparseMatrixCSC{SMatrix{2, 2, ComplexF64, 4}, Int64} with 4 stored entries:
 [1.0+0.0im 2.0+0.0im; 2.0+0.0im 1.0+0.0im]  [1.0+0.0im 2.0+0.0im; 2.0+0.0im 1.0+0.0im]
 [1.0+0.0im 2.0+0.0im; 2.0+0.0im 1.0+0.0im]  [1.0+0.0im 2.0+0.0im; 2.0+0.0im 1.0+0.0im]

 julia> h = unitcell(h); h[]
2×2 SparseMatrixCSC{SMatrix{2, 2, ComplexF64, 4}, Int64} with 2 stored entries:
                     ⋅                       [1.0+0.0im 2.0+0.0im; 3.0+0.0im 4.0+0.0im]
 [1.0+0.0im 2.0+0.0im; 3.0+0.0im 4.0+0.0im]                      ⋅                     

```

# See also
    `onsite`, `hopping`, `bloch`, `bloch!`
"""
hamiltonian