#######################################################################
# Lattice
#######################################################################
"""
`LatticePresets` is a Quantica submodule containing severeal pre-defined lattices. The
alias `LP` can be used in place of `LatticePresets`. Supported lattices are

    LP.linear(; a0 = 1, kw...)      # linear lattice in 1D
    LP.square(; a0 = 1, kw...)      # square lattice in 2D
    LP.triangular(; a0 = 1, kw...)  # triangular lattice in 2D
    LP.honeycomb(; a0 = 1, kw...)   # honeycomb lattice in 2D
    LP.cubic(; a0 = 1, kw...)       # cubic lattice in 3D
    LP.fcc(; a0 = 1, kw...)         # face-centered-cubic lattice in 3D
    LP.bcc(; a0 = 1, kw...)         # body-centered-cubic lattice in 3D
    LP.hcp(; a0 = 1, kw...)         # hexagonal-closed-packed lattice in 3D

In all cases `a0` denotes the lattice constant, and `kw...` are extra keywords forwarded to
`lattice`.

# Examples

```jldoctest
julia> LatticePresets.honeycomb(names = (:C, :D))
Lattice{Float64,2,2} : 2D lattice in 2D space
  Bravais vectors : [[0.5, 0.866025], [-0.5, 0.866025]]
  Sublattices     : 2
    Names         : (:C, :D)
    Sites         : (1, 1) --> 2 total per unit cell

julia> LatticePresets.cubic(bravais = ((1, 0), (0, 2)))
Lattice{Float64,3,2} : 2D lattice in 3D space
  Bravais vectors : [[1.0, 0.0, 1.0], [0.0, 2.0, 1.0]]
  Sublattices     : 1
    Names         : (:A)
    Sites         : (1) --> 1 total per unit cell
```

# See also
    `RegionPresets`, `HamiltonianPresets`
"""
LatticePresets

"""
`RegionPresets` is a Quantica submodule containing several pre-defined regions of type
`Region{E}`, where `E` is the space dimension. The alias `RP` can be used in place of
`RegionPresets`. Supported regions are

    RP.circle(radius = 10, center = (0, 0))                         # 2D
    RP.ellipse((rx, ry) = (10, 15), center = (0, 0))                # 2D
    RP.square(side = 10, center = (0, 0))                           # 2D
    RP.rectangle((sx, sy) = (10, 15), center = (0, 0))              # 2D
    RP.sphere(radius = 10, center = (0, 0, 0))                      # 3D
    RP.spheroid((rx, ry, rz) = (10, 15, 20), center = (0, 0, 0))    # 3D
    RP.cube(side = 10, center = (0, 0, 0))                          # 3D
    RP.cuboid((sx, sy, sz) = (10, 15, 20), center = (0, 0, 0))      # 3D


Calling a `f::Region{E}` object on a `r::Tuple` or `r::SVector` as in `f(r)` or `f(r...)`
returns `true` or `false` if `r` is inside the region or not. Note that only the first `E`
coordinates of `r` will be checked. Arbitrary boolean functions can also be wrapped in
`Region{E}`, e.g. `f = Region{2}(r -> r[1]^2 < r[2])`.

# Examples

```jldoctest
julia> RegionPresets.circle(10)(20, 0, 0)
false

julia> RegionPresets.circle(10)(0, 0, 20)
true
```

# See also
    `LatticePresets`, `HamiltonianPresets`
"""
RegionPresets

"""
    sublat(sites...; name::Symbol)
    sublat(sites::Vector; name::Symbol)

Create a `Sublat{E,T}` that adds a sublattice, of name `name`, with sites at positions
`sites` in `E` dimensional space. Sites positions can be entered as tuples or `SVectors`.

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
    bravais_matrix(lat::Lattice)
    bravais_matrix(h::Hamiltonian)

Obtain the Bravais matrix of lattice `lat` or Hamiltonian `h`

# Examples

```jldoctest
julia> lat = lattice(sublat((0,0)), bravais = ((1.0, 2), (3, 4)));

julia> bravais_matrix(lat)
2×2 SMatrix{2, 2, Float64, 4} with indices SOneTo(2)×SOneTo(2):
 1.0  3.0
 2.0  4.0

```

# See also
    `lattice`
"""
bravais_matrix

"""
    lattice(sublats::Sublat...; bravais = (), dim, type, names)

Create a `Lattice{T,E,L}` from sublattices `sublats`, where `L` is the number of Bravais
vectors `bravais`, `T = type` is the `AbstractFloat` type of spatial site coordinates, and
`E = dim` is the spatial embedding dimension.

The keyword `bravais` indicates one or more Bravais vectors in the form of tuples or other
iterables. It can also be an `AbstractMatrix` of dimension `E×L`. The default `bravais = ()`
corresponds to a bounded lattice with no Bravais vectors.

A keyword `names` can be used to rename `sublats`. Given `sublats` names will be replaced if
necessary by `:A`, `:B` etc. to ensure that all sublattice names are unique.

    lattice(lat::Lattice; bravais = missing, dim = missing, type = missing, names = missing)

Create a new lattice by applying any non-missing `kw` to `lat`.

    lat[kw...]

Indexing into a lattice `lat` with keywords returns a generator of `(i, r)`, the index `i`
and positions `r` of sites in `lat`, filtered by `siteselector(; kw...)`. See `siteselector`
for details on possible `kw`.

See also `LatticePresets` for built-in lattices.

# Examples

```jldoctest
julia> lat = lattice(sublat((0, 0)), sublat((0, 1)); bravais = (1, 0), type = Float32, dim = 3, names = (:up, :down))
Lattice{Float32,3,1} : 1D lattice in 3D space
  Bravais vectors : Vector{Float32}[[1.0, 0.0, 0.0]]
  Sublattices     : 2
    Names         : (:up, :down)
    Sites         : (1, 1) --> 2 total per unit cell

julia> lattice(lat; type = Float64, names = (:A, :B))
Lattice{Float64,3,1} : 1D lattice in 3D space
  Bravais vectors : [[1.0, 0.0, 0.0]]
  Sublattices     : 2
    Names         : (:A, :B)
    Sites         : (1, 1) --> 2 total per unit cell

```

# See also
    `LatticePresets`, `bravais`, `sublat`, `supercell`, `siteindices`, `sitepositions`
"""
lattice

"""
    sitepositions(l::Lattice; kw...)

Returns a vector with site positions in lattice `l`, filtered by `siteselector(; kw...)`.
Equivalent to `last.(l[kw...])`. See `siteselector` for details on possible `kw`.

# Examples
```jldoctest
julia> sitepositions(LatticePresets.honeycomb(); sublats = :A)
1-element Vector{SVector{2, Float64}}:
 [0.0, -0.2886751345948129]
```

"""
sitepositions

"""
    sitepositions(l::Lattice; kw...)

Returns a vector with site indices in lattice `l`, filtered by `siteselector(; kw...)`.
Equivalent to `first.(l[kw...])`. See `siteselector` for details on possible `kw`.

# Examples
```jldoctest
julia> siteindices(LatticePresets.honeycomb(); sublats = :B)
1-element Vector{Int64}:
 2
```

"""
siteindices

"""
    transform(lat::Lattice, f::Function)

Build a new lattice transforming each site positions `r` into `f(r)`.

    transform(h::Hamiltonian, ms::Modifier...)

Build a new Hamiltonian by applying `Modifier`s `ms` onto `h`, see `@onsite!`, `@hopping!`
for details on building `Modifier`s.

    x |> transform(f::Function)

Curried version of `transform`, equivalent to `transform(f, x)`

Note: `Quantica.transform!` is also available for in-place transforms. Use with care, as
aliasing (i.e. several objects sharing the modified one) can produce unexpected results.

# See also
    `translate`

"""
transform

"""
    translate(lat::Lattice, δr)

Build a new lattice translating each site positions from `r` to `r + δr`.

    x |> translate(δr)

Curried version of `translate`, equivalent to `translate(x, δr)`

Note: `Quantica.translate!` is also available for in-place translations. Use with care, as
aliasing (i.e. several objects sharing the modified one) can produce unexpected results.

# See also
    `transform`

"""
translate

# """
#     combine(lats::Lattice...)

# If all `lats` have compatible Bravais vectors, combine them into a single lattice.
# Sublattice names are renamed to be unique if necessary.
# """
# combine

"""
    supercell(lat::Lattice{E,L}, v::NTuple{L,Integer}...; seed = missing, kw...)
    supercell(lat::Lattice{E,L}, uc::SMatrix{L,L´,Int}; seed = missing, kw...)

Generates a `Lattice` from an `L`-dimensional lattice `lat` with a larger unit cell, such
that its Bravais vectors are `br´= br * uc`. Here `uc::SMatrix{L,L´,Int}` is the integer
supercell matrix, with the `L´` vectors `v`s as its columns. If no `v` are given, the new
lattice will have no Bravais vectors (i.e. it will be bounded, with its shape determined by
keywords `kw...`). Likewise, if `L´ < L`, the resulting lattice will be bounded along `L´ -
L` directions, as dictated by `kw...`.

Only sites selected by `siteselector(; kw...)` will be included in the supercell (see
`siteselector` for details on the available keywords `kw`). The search for included sites
will start from point `seed::Union{Tuple,SVector}`, or the origin if `seed = missing`. If no
keyword `region` is given in `kw`, a single Bravais unit cell perpendicular to the `v` axes
will be selected along the `L-L´` bounded directions.

    supercell(lattice::Lattice{E,L}, factor::Integer; kw...)

Calls `supercell` with a uniformly scaled `uc = SMatrix{L,L}(factor * I)`

    supercell(lattice::Lattice{E,L}, factors::Integer...; kw...)

Calls `supercell` with different scaling along each Bravais vector (diagonal supercell `uc`
with `factors` along the diagonal)

    supercell(h::Hamiltonian, v...; mincoordination = 0, kw...)

Transforms the `Lattice` of `h` to have a larger unit cell, while expanding the Hamiltonian
accordingly.

A nonzero `mincoordination` indicates a minimum number of nonzero hopping neighbors required
for sites to be included in the resulting unit cell. Sites with less coordination will be
removed recursively, until all remaining sites satisfy `mincoordination`.

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
    siteselector(; region = missing, sublats = missing)

Return a `SiteSelector` object that can be used to select sites in a lattice contained
within the specified region and sublattices. Sites at position `r` and belonging to a
sublattice with name `s::Symbol` will be selected only if

    `region(r) && s in sublats`

Any missing `region`, `sublat` or `indices` will not be used to constraint the selection.

The constructor `siteselector(; kw...)` is not meant to be called by the end user. Instead,
the kwargs `kw` are input into different functions that allow filtering sites, which
themselves call `siteselector` internally as needed. Some of these functions are

    - getindex(l::Lattice; kw...) : return site indices and positions filtered by `kw` (also `l[kw...]`)
    - onsite(...; kw...)          : onsite model term to be applied to sites specified by `kw`
    - @onsite!(...; kw...)        : onsite modifier to be applied to sites specified by `kw`

# Examples

```jldoctest
julia> lat = LP.honeycomb() |> supercell(region = RegionPresets.circle(100))
Lattice{Float64,2,0} : 0D lattice in 2D space
  Bravais vectors : []
  Sublattices     : 2
    Names         : (:A, :B)
    Sites         : (36281, 36281) --> 72562 total per unit cell

julia> lat[] == sites(lat)
true

julia> lat[sublats = :A, region = RP.circle(50)] |> length
9062
```

# See also
    `hopselector`, `lattice`, `onsite`, `@onsite!`
"""
siteselector

"""
    hopselector(; range = neighbors(1), dcells = missing, sublats = missing, region = missing)

Return a `HopSelector` object that can be used to select hops between two sites in a
lattice. Hops between two sites at positions `r₁ = r - dr/2` and `r₂ = r + dr`, belonging to
unit cells at integer distance `dcell` and to sublattices with names `s₁::Symbol` and
`s₂::Symbol` will be selected if:

    `region(r, dr) && (s₁ => s₂ in sublats) && (dcell in dcells) && (norm(dr) <= range)`

If any of these is `missing` it will not be used to constraint the selection.

The keyword `range` admits the following possibilities

    max_range                   # i.e. `norm(dr) <= max_range`
    (min_range, max_range)      # i.e. `min_range <= norm(dr) <= max_range`

Both `max_range` and `min_range` can be a `Real` or a `Neighbors` object created with
`neighbors(n)`. The latter represents the distance of the `n`-th nearest neighbors in
lattice `lat` (see `neighbors`).

The keyword `dcells` can be a `Tuple`/`SVector` of `Int`s, or a collection of them.

The keyword `sublats` allows various forms, including:

    sublats = :A                          # Hops from :A to :A
    sublats = :A => :B                    # Hops from :A to :B sublattices, but not from :B to :A
    sublats = (:A => :B,)                 # Same as above
    sublats = (:A => :B, :C => :D)        # Hopping from :A to :B or :C to :D
    sublats = (:A, :C) .=> (:B, :D)       # Broadcasted pairs, same as above
    sublats = (:A, :C) => (:B, :D)        # Direct product, (:A=>:B, :A=:D, :C=>:B, :C=>D)
    sublats = (spec₁, spec₂, ...)         # Hops matching any of the `spec`'s with any form as above

The constructor `hopselector(; kw...)` is not meant to be called by the end user. Instead,
the kwargs `kw` are input into different functions that allow filtering pairs of sites,
which themselves call `hopselector` internally as needed. Some of these functions are

    - hopping(...; kw...)   : hopping model term to be applied to site pairs specified by `kw`
    - @onsite!(...; kw...)  : hopping modifier to be applied to site pairs specified by `kw`

# Examples

```jldoctest
julia> lat = LP.honeycomb() |> hamiltonian(hopping(1, range = neighbors(2), sublats = (:A, :B) .=> (:A, :B)))
Hamiltonian{Float64,2,2}: Hamiltonian on a 2D Lattice in 2D space
  Bloch harmonics  : 7
  Harmonic size    : 2 × 2
  Orbitals         : [1, 1]
  Element type     : scalar (ComplexF64)
  Onsites          : 0
  Hoppings         : 12
  Coordination     : 6.0

julia> lat = LP.honeycomb() |> hamiltonian(hopping(1, range = neighbors(2), sublats = (:A, :B) => (:A, :B)))
Hamiltonian{Float64,2,2}: Hamiltonian on a 2D Lattice in 2D space
  Bloch harmonics  : 7
  Harmonic size    : 2 × 2
  Orbitals         : [1, 1]
  Element type     : scalar (ComplexF64)
  Onsites          : 0
  Hoppings         : 18
  Coordination     : 9.0
```

# See also
    `siteselector`, `hopping`, `@hopping!`

"""
hopselector

"""
    neighbors(n::Int)

Create a `Neighbors(n)` object that represents a hopping range to distances corresponding to
the n-th nearest neighbors in a given lattice, irrespective of their sublattice. Neighbors
at equal distance do not count towards `n`.

    neighbors(n::Int, lat::Lattice)

Obtain the actual nth-nearest-neighbot distance between sites in lattice `lat`.

# See also
    `hopping`
"""
neighbors

"""
    hamiltonian(lat::Lattice{T}, model; orbitals = 1, type = T)

Create a `Hamiltonian` with a given number of `orbitals` per sublattice of type
`Complex{type}` by applying `model::TighbindingModel` to the lattice `lat` (see `hopping`
and `onsite` for details on building tightbinding models).

    lat |> hamiltonian(model; kw...)

Curried form of `hamiltonian` equivalent to `hamiltonian(lat, model; kw...)`.

# Orbitals

Each matrix element in the Hamiltonian corresponds to a pair of sites, each of which may
have one or more orbitals. Sites in the same sublattice have an equal number of orbitals
given by `orbitals`. If `orbitals = (n₁, n₂, ...)` is a collection of integers, one per
sublattice, sites in each sublattice will contain the corresponding number `nᵢ` of orbitals.
For type stability, the matrix elements of Hamiltonians are stored as blocks of equal size
`N = max(orbitals)`. If `N = 1` (all sublattices with one orbital) the Hamiltonian matrix
element type is `Complex{type}`. Otherwise it is `SMatrix{N,N,Complex{type}}`, with each
block padded with the necessary zeros as required. Keyword `type` is `T` by default, where
`T <: AbstractFloat` is the number type of `lat`.

# Indexing

Indexing into a Hamiltonian `h` works as follows. Access the `Harmonic` matrix at a given
unit cell distance `dn::NTuple{L,Int}` with `h[dn]`. The special `h[]` syntax stands for
`h[(0...)]` for the zero-harmonic. Assign `v` into element `(i,j)` of said matrix with
`h[dn][i,j] = v`. Broadcasting with vectors of indices `is` and `js` is supported,
`h[dn][is, js] .= v_matrix`.

To add an empty harmonic with a given `dn::NTuple{L,Int}`, do `push!(h, dn)`. To delete it,
do `deleteat!(h, dn)`.

# Examples

```jldoctest
julia> h = hamiltonian(LatticePresets.honeycomb(), hopping(SA[1 2; 2 4], range = 1/√3), orbitals = 2)
Hamiltonian{Float64,2,2}: Hamiltonian on a 2D Lattice in 2D space
  Bloch harmonics  : 5
  Harmonic size    : 2 × 2
  Orbitals         : [2, 2]
  Element type     : 2 × 2 blocks (ComplexF64)
  Onsites          : 0
  Hoppings         : 6
  Coordination     : 3.0

julia> push!(h, (3,3)) # Adding a new Hamiltonian harmonic (if not already present)
Hamiltonian{Float64,2,2}: Hamiltonian on a 2D Lattice in 2D space
  Bloch harmonics  : 6
  Harmonic size    : 2 × 2
  Orbitals         : [2, 2]
  Element type     : 2 × 2 blocks (ComplexF64)
  Onsites          : 0
  Hoppings         : 6
  Coordination     : 3.0

julia> h[(3,3)][1,1] = @SMatrix[1 2; 2 1]; h[(3,3)] # element assignment
2×2 SparseMatrixCSC{SMatrix{2, 2, ComplexF64, 4}, Int64} with 1 stored entry:
 [1.0+0.0im 2.0+0.0im; 2.0+0.0im 1.0+0.0im]                      ⋅                     
                     ⋅                                           ⋅                     

julia> h[(3,3)][[1,2],[1,2]] .= Ref(SA[1 2; 2 1])  # multiple element assignment
2×2 SparseMatrixCSC{SMatrix{2, 2, ComplexF64, 4}, Int64} with 4 stored entries:
 [1.0+0.0im 2.0+0.0im; 2.0+0.0im 1.0+0.0im]  [1.0+0.0im 2.0+0.0im; 2.0+0.0im 1.0+0.0im]
 [1.0+0.0im 2.0+0.0im; 2.0+0.0im 1.0+0.0im]  [1.0+0.0im 2.0+0.0im; 2.0+0.0im 1.0+0.0im]

julia> h[]                                        # inspect matrix of zero harmonic
2×2 SparseMatrixCSC{SMatrix{2, 2, ComplexF64, 4}, Int64} with 2 stored entries:
                     ⋅                       [1.0+0.0im 2.0+0.0im; 2.0+0.0im 4.0+0.0im]
 [1.0+0.0im 2.0+0.0im; 2.0+0.0im 4.0+0.0im]                      ⋅                     

```

# See also
    `onsite`, `hopping`, `bloch`, `bloch!`
"""
hamiltonian