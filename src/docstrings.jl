"""
`LatticePresets` is a Quantica submodule containing several pre-defined lattices. The
alias `LP` can be used in place of `LatticePresets`. Currently supported lattices are

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
  Bravais vectors : [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
  Sublattices     : 1
    Names         : (:A,)
    Sites         : (1,) --> 1 total per unit cell
```

# See also
    `RegionPresets`, `HamiltonianPresets`, `ExternalPresets`
"""
LatticePresets

"""
`HamiltonianPresets` is a Quantica submodule containing several pre-defined Hamiltonians.
The alias `HP` can be used in place of `HamiltonianPresets`. Currently supported
hamiltonians are

    HP.graphene(; kw...)
    HP.twisted_bilayer_graphene(; kw...)

For details on the keyword arguments `kw` see the corresponding docstring

```jldoctest
julia> HamiltonianPresets.twisted_bilayer_graphene(twistindices = (30, 1))
Hamiltonian{Float64,3,2}: Hamiltonian on a 2D Lattice in 3D space
  Bloch harmonics  : 7
  Harmonic size    : 11164 × 11164
  Orbitals         : [1, 1, 1, 1]
  Element type     : scalar (ComplexF64)
  Onsites          : 0
  Hoppings         : 315684
  Coordination     : 28.27696
```

# See also
    `LatticePresets`, `RegionPresets`, `ExternalPresets`

"""
HamiltonianPresets

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

Calling a `f::Region{E}` object on a `r::Tuple` or `r::SVector` with `f(r)` or `f(r...)`
returns `true` or `false` if `r` is inside the region or not. Note that only the first `E`
coordinates of `r` will be checked. Arbitrary boolean functions can also be wrapped in
`Region{E}` to create custom regions, e.g. `f = Region{2}(r -> r[1]^2 < r[2])`.

Boolean combinations of `Regions` are supported using `&`, `|`, `xor` and `!` operators,
such as `annulus = RP.circle(10) & !RP.circle(5)`.

# Examples

```jldoctest
julia> RegionPresets.circle(10)(20, 0, 0)
false

julia> RegionPresets.circle(10)(0, 0, 20)
true
```

# See also
    `LatticePresets`, `HamiltonianPresets`, `ExternalPresets`
"""
RegionPresets

"""
`ExternalPresets` is a Quantica submodule containing utilities to import objects from
external applications The alias `EP` can be used in place of `ExternalPresets`. Currently
supported importers are

    EP.wannier90(args...; kw...)

For details on the arguments `args` and keyword arguments `kw` see the docstring for the
corresponding function.

# See also
    `LatticePresets`, `RegionPresets`, `HamiltonianPresets`

"""
ExternalPresets

"""
    sublat(sites...; name::Symbol = :A)
    sublat(sites::AbstractVector; name::Symbol = :A)

Create a `Sublat{E,T}` that adds a sublattice, of name `name`, with sites at positions
`sites` in `E` dimensional space. Sites positions can be entered as `Tuple`s or `SVector`s.

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
    bravais_matrix(h::AbstractHamiltonian)

Return the Bravais matrix of lattice `lat` or AbstractHamiltonian `h`, with Bravais vectors
as its columns.

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
    lattice(sublats::AbstractVector{<:Sublat}; bravais = (), dim, type, names)

Create a `Lattice{T,E,L}` from sublattices `sublats`, where `L` is the number of Bravais
vectors given by `bravais`, `T = type` is the `AbstractFloat` type of spatial site
coordinates, and `dim = E` is the spatial embedding dimension.

    lattice(lat::Lattice; bravais = missing, dim = missing, type = missing, names = missing)

Create a new lattice by applying any non-missing keywords to `lat`.

    lattice(x)

Return the parent lattice of object `x`, of type e.g. `LatticeSlice`, `Hamiltonian`, etc.

## Keywords

- `bravais`: a collection of one or more Bravais vectors of type NTuple{E} or SVector{E}. It can also be an `AbstractMatrix` of dimension `E×L`. The default `bravais = ()` corresponds to a bounded lattice with no Bravais vectors.
- `names`: a collection of Symbols. Can be used to rename `sublats`. Any repeated names will be replaced if necessary by `:A`, `:B` etc. to ensure that all sublattice names are unique.

## Indexing

    lat[kw...]
    lat[siteselector(; kw...)]

Indexing into a lattice `lat` with keywords returns `LatticeSlice` representing a finite
collection of sites selected by `siteselector(; kw...)`. See `siteselector` for details on
possible `kw`, and `sites` to obtain site positions.

    lat[]

Special case equivalent to `lat[cells = (0,...)]` that returns a `LatticeSlice` of the
zero-th unitcell.

    lat[i::CellSites]

With an `i` of type `CellSites` contructed with `sites([cell,] indices)`, return a
`LatticeSlice` of the corresponding sites.

# Examples

```jldoctest
julia> lat = lattice(sublat((0, 0)), sublat((0, 1)); bravais = (1, 0), type = Float32, dim = 3, names = (:up, :down))
Lattice{Float32,3,1} : 1D lattice in 3D space
  Bravais vectors : Vector{Float32}[[1.0, 0.0, 0.0]]
  Sublattices     : 2
    Names         : (:up, :down)
    Sites         : (1, 1) --> 2 total per unit cell

julia> lattice(lat; type = Float64, names = (:A, :B), dim = 2)
Lattice{Float64,2,1} : 1D lattice in 2D space
  Bravais vectors : [[1.0, 0.0]]
  Sublattices     : 2
    Names         : (:A, :B)
    Sites         : (1, 1) --> 2 total per unit cell
```

# See also
    `LatticePresets`, `sublat`, `sites`, `supercell`
"""
lattice

"""
    sites(lat::Lattice[, sublat])

Return a collection of site positions in the unit cell of lattice `lat`. If a
`sublat::Symbol` or `sublat::Int` is specified, only sites for the specified sublattice are
returned.

    sites(ls::LatticeSlice)

Return a collection of positions of a LatticeSlice, generally obtained by indexing a
lattice `lat[sel...]` with some `siteselector` keywords `sel`. See also `lattice`.

Note: the returned collections can be of different types (vectors, generators, views...)

    sites(cell_index, site_indices)
    sites(site_indices)

Construct a simple selector of sites, of type `CellSites`, with given `site_indices` in a
given cell at `cell_index`. Here, `site_indices` can be an index, a collection of integers
or `:` (for all sites), and `cell_index` should be a collection of `L` integers, where `L`
is the lattice dimension. If omitted, `cell_index` defaults to the zero-th cell `(0,...)`.

`CellSites` produced with `sites` can be used to index `Lattice`s, `AbstractHamiltonian`s,
`GreenFunction`s, `GreenSlice`s, `OrbitalSliceArray`s, etc. Note that selecting sites based
on cell and site indices requires finding the indices beforehand, which can be done e.g.
through plotting the system with `qplot`. This is lower level and potentially more fragile
than using `siteselector`s, as indices are chosen freely by Quantica in an unspecified way,
but it does have a smaller overhead.

# Examples
```jldoctest
julia> sites(LatticePresets.honeycomb(), :A)
1-element view(::Vector{SVector{2, Float64}}, 1:1) with eltype SVector{2, Float64}:
 [0.0, -0.2886751345948129]
```

# See also
    `lattice`, `siteselector`
"""
sites

"""
    position(b::ExternalPresets.WannierBuilder)

Returns the position operator in the Wannier basis. It is given as a `r::BarebonesOperator`
object, which can be indexed as `r[s, s´]` to obtain matrix elements `⟨s|R|s´⟩` of the
position operator `R` (a vector). Here `s` and `s´` represent site indices, constructed
with `sites(cell, inds)`. To obtain the matrix between cells separated by
`dn::SVector{L,Int}`, do `r[dn]`. The latter will throw an error if the `dn` harmonic is not
present.

# See also
    `current`, `sites`
"""
position

"""
    supercell(lat::Lattice{E,L}, v::NTuple{L,Integer}...; seed = missing, kw...)
    supercell(lat::Lattice{E,L}, uc::SMatrix{L,L´,Int}; seed = missing, kw...)

Generate a new `Lattice` from an `L`-dimensional lattice `lat` with a larger unit cell, such
that its Bravais vectors are `br´= br * uc`. Here `uc::SMatrix{L,L´,Int}` is the integer
supercell matrix, with the `L´` vectors `v`s as its columns. If no `v` are given, the new
lattice will have no Bravais vectors (i.e. it will be bounded, with its shape determined by
keywords `kw...`). Likewise, if `L´ < L`, the resulting lattice will be bounded along `L´ -
L` directions, as dictated by `kw...`.

Only sites selected by `siteselector(; kw...)` will be included in the supercell (see
`siteselector` for details on the available keywords `kw`). If no
keyword `region` is given in `kw`, a single Bravais unit cell perpendicular to the `v` axes
will be selected along the `L-L´` bounded directions.

    supercell(lattice::Lattice{E,L}, factors::Integer...; seed = missing, kw...)

Call `supercell` with different scaling along each Bravais vector, so that supercell matrix
`uc` is `Diagonal(factors)`. If a single `factor` is given, `uc = SMatrix{L,L}(factor * I)`

    supercell(h::Hamiltonian, v...; mincoordination = 0, seed = missing, kw...)

Transform the `Lattice` of `h` to have a larger unit cell, while expanding the Hamiltonian
accordingly.

## Keywords

- `seed::NTuple{L,Integer}`: starting cell index to perform search of included sites. By default `seed = missing`, which makes search start from the zero-th cell.
- `mincoordination::Integer`: minimum number of nonzero hopping neighbors required for sites to be included in the supercell. Sites with less coordination will be removed recursively, until all remaining sites satisfy `mincoordination`.

## Currying

    lat_or_h |> supercell(v...; kw...)

Curried syntax, equivalent to `supercell(lat_or_h, v...; kw...)`

# Examples

```jldoctest
julia> LatticePresets.square() |> supercell((1, 1), region = r -> 0 < r[1] < 5)
Lattice{Float64,2,1} : 1D lattice in 2D space
  Bravais vectors : [[1.0, 1.0]]
  Sublattices     : 1
    Names         : (:A,)
    Sites         : (8,) --> 8 total per unit cell

julia> LatticePresets.honeycomb() |> supercell(3)
Lattice{Float64,2,2} : 2D lattice in 2D space
  Bravais vectors : [[1.5, 2.598076], [-1.5, 2.598076]]
  Sublattices     : 2
    Names         : (:A, :B)
    Sites         : (9, 9) --> 18 total per unit cell
```

# See also
    `supercell`, `siteselector`
"""
supercell

"""
    reverse(lat_or_h::Union{Lattice,AbstractHamiltonian})

Build a new lattice or hamiltonian with the orientation of all Bravais vectors and harmonics
reversed.

# See also
    `reverse!`, `transform`
"""
Base.reverse

"""
    reverse!(lat_or_h::Union{Lattice,AbstractHamiltonian})

In-place version of `reverse`, inverts all Bravais vectors and harmonics of `lat_or_h`.

# See also
    `reverse`, `transform`
"""
Base.reverse!

"""
    transform(lat_or_h::Union{Lattice,AbstractHamiltonian}, f::Function)

Build a new lattice or hamiltonian transforming each site positions `r` into `f(r)`.

## Currying

    x |> transform(f::Function)

Curried version of `transform`, equivalent to `transform(f, x)`

Note: Unexported `Quantica.transform!` is also available for in-place transforms. Use with
care, as aliasing (i.e. several objects sharing the modified one) can produce unexpected
results.

# Examples

```jldoctest
julia> LatticePresets.square() |> transform(r -> 3r)
Lattice{Float64,2,2} : 2D lattice in 2D space
  Bravais vectors : [[3.0, 0.0], [0.0, 3.0]]
  Sublattices     : 1
    Names         : (:A,)
    Sites         : (1,) --> 1 total per unit cell
```

# See also
    `translate`, `reverse`, `reverse!`
"""
transform

"""
    translate(lat::Lattice, δr)

Build a new lattice translating each site positions from `r` to `r + δr`, where `δr` can be
a `NTuple` or an `SVector` in embedding space.

## Currying

    x |> translate(δr)

Curried version of `translate`, equivalent to `translate(x, δr)`

Note: Unexported `Quantica.translate!` is also available for in-place translations. Use with
care, as aliasing (i.e. several objects sharing the modified one) can produce unexpected
results.

# Examples

```jldoctest
julia> LatticePresets.square() |> translate((3,3)) |> sites
1-element Vector{SVector{2, Float64}}:
 [3.0, 3.0]

```

# See also
    `transform`, `reverse`, `reverse!`

"""
translate

"""
    combine(lats::Lattice...)

If all `lats` have compatible Bravais vectors, combine them into a single lattice.
If necessary, sublattice names are renamed to remain unique.

    combine(hams::Hamiltonians...; coupling = TighbindingModel())

Combine a collection `hams` of Hamiltonians into one by combining their corresponding
lattices, and optionally by adding a coupling between them, given by the hopping terms in
`coupling`.

Note that the `coupling` model will be applied to the combined lattice (which may have
renamed sublattices to avoid name collissions). However, only hopping terms between
different `hams` blocks will be applied.

## Limitations

Currently, `combine` only works with `Lattice{T}` `AbstractHamiltonians{T}` with the same
`T`. Furthermore, if any of the `hams` is a `ParametricHamiltonian` or `coupling` is a
`ParametricModel`, the sublattice names of all `hams` must be distinct. This ensures that
parametric models, which get applied through `Modifiers` after construction of the
`ParametricHamiltonian`, are not applied to the wrong sublattice, since sublattice names
could be renamed by `combine` if they are not unique. Therefore, be sure to choose unique
sublattice names upon construction for all the `hams` to be combined (see `lattice`).

# Examples
```jldoctest
julia> # Building Bernal-stacked bilayer graphene

julia> hbot = HP.graphene(a0 = 1, dim = 3, names = (:A,:B));

julia> htop = translate(HP.graphene(a0 = 1, dim = 3, names = (:C,:D)), (0, 1/√3, 1/√3));

julia> h2 = combine(hbot, htop; coupling = hopping(1, sublats = :B => :C) |> plusadjoint)
Hamiltonian{Float64,3,2}: Hamiltonian on a 2D Lattice in 3D space
  Bloch harmonics  : 5
  Harmonic size    : 4 × 4
  Orbitals         : [1, 1, 1, 1]
  Element type     : scalar (ComplexF64)
  Onsites          : 0
  Hoppings         : 14
  Coordination     : 3.5
```

# See also
    `hopping`
"""
combine

"""
    siteselector(; region = missing, sublats = missing, cells = missing)

Return a `SiteSelector` object that can be used to select a finite set of sites in a
lattice. Sites at position `r::SVector{E}`, belonging to a cell of index `n::SVector{L,Int}`
and to a sublattice with name `s::Symbol` will be selected only if

    `region(r) && s in sublats && n in cells`

Any missing `region`, `sublat` or `cells` will not be used to constraint the selection.

## Generalization

While `sublats` and `cells` are usually collections of `Symbol`s and `SVector`s,
respectively, they also admit other possibilities:

- If either `cells` or `sublats` are a single cell or sublattice, they will be treated as single-element collections
- If `sublat` is a collection of `Integer`s, it will refer to sublattice numbers.
- If `cells` is an `i::Integer`, it will be converted to an `SVector{1}`
- If `cells` is a collection, each element will be converted to an `SVector`.
- If `cells` is a boolean function, `n in cells` will be the result of `cells(n)`

## Usage

Although the constructor `siteselector(; kw...)` is exported, the end user does not usually
need to call it directly. Instead, the keywords `kw` are input into different functions that
allow filtering sites, which themselves call `siteselector` internally as needed. Some of
these functions are

- getindex(lat::Lattice; kw...) : return a LatticeSlice with sites specified by `kw` (also `lat[kw...]`)
- supercell(lat::Lattice; kw...) : returns a bounded lattice with the sites specified by `kw`
- onsite(...; kw...) : onsite model term to be applied to sites specified by `kw`
- @onsite!(...; kw...) : onsite modifier to be applied to sites specified by `kw`

# See also
    `hopselector`, `lattice`, `supercell`, `onsite`, `@onsite`, `@onsite!`
"""
siteselector

"""
    hopselector(; range = neighbors(1), dcells = missing, sublats = missing, region = missing)

Return a `HopSelector` object that can be used to select a finite set of hops between sites
in a lattice. Hops between two sites at positions `r₁ = r - dr/2` and `r₂ = r + dr`,
belonging to unit cells with a cell distance `dn::SVector{L,Int}` and to a sublattices with
names `s₁::Symbol` and `s₂::Symbol` will be selected only if

    `region(r, dr) && (s₁ => s₂ in sublats) && (dcell in dcells) && (norm(dr) <= range)`

If any of these is `missing` it will not be used to constraint the selection.

## Generalization

While `range` is usually a `Real`, and `sublats` and `dcells` are usually collections of
`Pair{Symbol}`s and `SVector`s, respectively, they also admit other possibilities:

    sublats = :A                       # Hops from :A to :A
    sublats = :A => :B                 # Hops from :A to :B sublattices, but not from :B to :A
    sublats = (:A => :B,)              # Same as above
    sublats = (:A => :B, :C => :D)     # Hopping from :A to :B or :C to :D
    sublats = (:A, :C) .=> (:B, :D)    # Broadcasted pairs, same as above
    sublats = (:A, :C) => (:B, :D)     # Direct product, (:A=>:B, :A=>:D, :C=>:B, :C=>D)
    sublats = 1 => 2                   # Hops from 1st to 2nd sublat. All the above patterns also admit Ints
    sublats = (spec₁, spec₂, ...)      # Hops matching any of the specs with any of the above forms

    dcells  = dn::SVector{L,Integer}   # Hops between cells separated by `dn`
    dcells  = dn::NTuple{L,Integer}    # Hops between cells separated by `SVector(dn)`
    dcells  = f::Function              # Hops between cells separated by `dn` such that `f(dn) == true`

    range   = neighbors(n)             # Hops within the `n`-th nearest neighbor distance in the lattice
    range   = (min, max)               # Hops at distance inside the `[min, max]` closed interval (bounds can also be `neighbors(n)`)

## Usage

Although the constructor `hopselector(; kw...)` is exported, the end user does not usually
need to call it directly. Instead, the keywords `kw` are input into different functions that
allow filtering hops, which themselves call `hopselector` internally as needed. Some of
these functions are

    - hopping(...; kw...)   : hopping model term to be applied to site pairs specified by `kw`
    - @hopping(...; kw...)  : parametric hopping model term to be applied to site pairs specified by `kw`
    - @hopping!(...; kw...) : hopping modifier to be applied to site pairs specified by `kw`

# Examples

```jldoctest
julia> h = LP.honeycomb() |> hamiltonian(hopping(1, range = neighbors(2), sublats = (:A, :B) .=> (:A, :B)))
Hamiltonian{Float64,2,2}: Hamiltonian on a 2D Lattice in 2D space
  Bloch harmonics  : 7
  Harmonic size    : 2 × 2
  Orbitals         : [1, 1]
  Element type     : scalar (ComplexF64)
  Onsites          : 0
  Hoppings         : 12
  Coordination     : 6.0

julia> h = LP.honeycomb() |> hamiltonian(hopping(1, range = (neighbors(2), neighbors(3)), sublats = (:A, :B) => (:A, :B)))
Hamiltonian{Float64,2,2}: Hamiltonian on a 2D Lattice in 2D space
  Bloch harmonics  : 9
  Harmonic size    : 2 × 2
  Orbitals         : [1, 1]
  Element type     : scalar (ComplexF64)
  Onsites          : 0
  Hoppings         : 18
  Coordination     : 9.0
```

# See also
    `siteselector`, `lattice`, `hopping`, `@hopping`, `@hopping!`

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
    hamiltonian(lat::Lattice, model; orbitals = 1)

Create an `AbstractHamiltonian` (i.e. an `Hamiltonian` or `ParametricHamiltonian`) by
applying `model` to the lattice `lat` (see `onsite`, `@onsite`, `hopping` and `@hopping` for
details on building parametric and non-parametric tight-binding models).

    hamiltonian(lat::Lattice, model, modifiers...; orbitals = 1)

Create a `ParametricHamiltonian` where all onsite and hopping terms in `model` can be
parametrically modified through the provided parametric `modifiers` (see `@onsite!` and
`@hopping!` for details on defining modifiers).

    hamiltonian(h::AbstractHamiltonian, modifier, modifiers...)

Add modifiers to an existing `AbstractHamiltonian`.

    hamiltonian(h::ParametricHamiltonian)

Return the base (non-parametric) `Hamiltonian` of `h`, with all modifiers and parametric
model terms removed (see `@onsite`, `@hopping`, `@onsite!`, `@hopping!`).

## Keywords

- `orbitals`: number of orbitals per sublattice. If an `Integer` (or a `Val{<:Integer}` for type-stability), all sublattices will have the same number of orbitals. A collection of values indicates the orbitals on each sublattice.

## Currying

    lat |> hamiltonian(model[, modifiers...]; kw...)

Curried form of `hamiltonian` equivalent to `hamiltonian(lat, model, modifiers...; kw...)`.

    lat |> model

Alternative and less general curried form equivalent to `hamiltonian(lat, model)`.

    h |> modifier

Alternative and less general curried form equivalent to `hamiltonian(h, modifier)`.

## Indexing

    h[dn::SVector{L,Int}]
    h[dn::NTuple{L,Int}]

Return the Bloch harmonic of an `h::AbstractHamiltonian` in the form of a `SparseMatrixCSC`
with complex scalar `eltype`. This matrix is "flat", in the sense that it contains matrix
elements between individual orbitals, not sites. This distinction is only relevant for
multiorbital Hamiltonians. To access the non-flattened matrix use `h[unflat(dn)]` (see
also `unflat`).

    h[()]

Special syntax equivalent to `h[(0...)]`, which access the fundamental Bloch harmonic.

    h[i::CellSites, j::CellSites = i]

With `i` and `j` of type `CellSites` and constructed with `sites([cell,] indices)`, return a
`SparseMatrixCSC` block of `h` between the sites with the corresponding `indices` and in the
given `cell`s. Alternatively, one can also use `view(h, i, j = i)`, which should be
non-allocating for `AbstractHamiltonian`s with uniform number of orbitals.

    h[srow::SiteSelector, scol::SiteSelector = srow]
    h[kwrow::NamedTuple, kwcol::NamedTuple = kwrow]

Return an `OrbitalSliceMatrix` of `h` between row and column sites selected by `srow` and
`scol`, or by `siteselector(; kwrow...)` and `siteselector(; kwcol...)`

Note: `CellSites` and `SiteSelector`s can be mixed when indexing, in which case the matrix
block will be returned as a `SparseMatrixCSC`, instead of an `OrbitalSliceMatrix`.

## Call syntax

    ph(; params...)

Return a `h::Hamiltonian` from a `ph::ParametricHamiltonian` by applying specific values to
its parameters `params`. If `ph` is a non-parametric `Hamiltonian` instead, this is a no-op.

    h(φs; params...)

Return the flat, sparse Bloch matrix of `h::AbstractHamiltonian` at Bloch phases `φs`, with
applied parameters `params` if `h` is a `ParametricHamiltonian`. The Bloch matrix is defined
as

        H = ∑_dn exp(-im φs⋅dn) H_dn

where `H_dn = h[dn]` is the `dn` flat Bloch harmonic of `h`, and `φs[i] = k⋅aᵢ` in terms of
the wavevector `k` and the Bravais vectors `aᵢ`.

# Examples

```jldoctest
julia> h = hamiltonian(LP.honeycomb(), hopping(SA[0 1; 1 0], range = 1/√3), orbitals = 2)
Hamiltonian{Float64,2,2}: Hamiltonian on a 2D Lattice in 2D space
  Bloch harmonics  : 5
  Harmonic size    : 2 × 2
  Orbitals         : [2, 2]
  Element type     : 2 × 2 blocks (ComplexF64)
  Onsites          : 0
  Hoppings         : 6
  Coordination     : 3.0

julia> h((0,0))
4×4 SparseArrays.SparseMatrixCSC{ComplexF64, Int64} with 8 stored entries:
     ⋅          ⋅      0.0+0.0im  3.0+0.0im
     ⋅          ⋅      3.0+0.0im  0.0+0.0im
 0.0+0.0im  3.0+0.0im      ⋅          ⋅
 3.0+0.0im  0.0+0.0im      ⋅          ⋅

julia> h[sites(1), sites(2)]
2×2 SparseArrays.SparseMatrixCSC{ComplexF64, Int64} with 4 stored entries:
 0.0+0.0im  1.0+0.0im
 1.0+0.0im  0.0+0.0im

julia> ph = h |> @hopping!((t; p = 3) -> p*t); ph[region = RP.square(1)]
4×4 OrbitalSliceMatrix{ComplexF64,SparseMatrixCSC}:
 0.0+0.0im  0.0+0.0im  0.0+0.0im  3.0+0.0im
 0.0+0.0im  0.0+0.0im  3.0+0.0im  0.0+0.0im
 0.0+0.0im  3.0+0.0im  0.0+0.0im  0.0+0.0im
 3.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im
```

# See also
    `lattice`, `onsite`, `hopping`, `@onsite`, `@hopping`, `@onsite!`, `@hopping!`, `ishermitian`, `OrbitalSliceMatrix`
"""
hamiltonian

"""
    ishermitian(h::Hamiltonian)

Check whether `h` is Hermitian. This is not supported for `h::ParametricHamiltonian`, as the
result can depend of the specific values of its parameters.

"""
ishermitian

"""
    onsite(o; sites...)
    onsite(r -> o(r); sites...)

Build a `TighbindingModel` representing a uniform or a position-dependent onsite potential,
respectively, on sites selected by `siteselector(; sites...)` (see `siteselector` for
details).

Site positions are `r::SVector{E}`, where `E` is the embedding dimension of the lattice. The
onsite potential `o` can be a `Number` (for single-orbital sites), a `UniformScaling` (e.g.
`2I`) or an `AbstractMatrix` (use `SMatrix` for performance) of dimensions matching the
number of orbitals in the selected sites. Models may be applied to a lattice `lat` to
produce a `Hamiltonian` with `hamiltonian(lat, model; ...)`, see `hamiltonian`. Position
dependent models are forced to preserve the periodicity of the lattice.

    onsite(m::{TighbindingModel,ParametricModel}; sites...)

Convert `m` into a new model with just onsite terms acting on `sites`.

## Model algebra

Models can be combined using `+`, `-` and `*`, or conjugated with `'`, e.g. `onsite(1) - 2 *
hopping(1)'`.

# Examples
```jldoctest
julia> model = onsite(r -> norm(r) * SA[0 1; 1 0]; sublats = :A) - hopping(I; range = 2)
TightbindingModel: model with 2 terms
  OnsiteTerm{Function}:
    Region            : any
    Sublattices       : A
    Cells             : any
    Coefficient       : 1
  HoppingTerm{LinearAlgebra.UniformScaling{Bool}}:
    Region            : any
    Sublattice pairs  : any
    Cell distances    : any
    Hopping range     : 2.0
    Reverse hops      : false
    Coefficient       : -1

julia> LP.cubic() |> supercell(4) |> hamiltonian(model, orbitals = 2)
Hamiltonian{Float64,3,3}: Hamiltonian on a 3D Lattice in 3D space
  Bloch harmonics  : 27
  Harmonic size    : 64 × 64
  Orbitals         : [2]
  Element type     : 2 × 2 blocks (ComplexF64)
  Onsites          : 64
  Hoppings         : 2048
  Coordination     : 32.0
```

# See also
    `hopping`, `@onsite`, `@hopping`, `@onsite!`, `@hopping!`, `hamiltonian`
"""
onsite

"""
    hopping(t; hops...)
    hopping((r, dr) -> t(r, dr); hops...)

Build a `TighbindingModel` representing a uniform or a position-dependent hopping amplitude,
respectively, on hops selected by `hopselector(; hops...)` (see `hopselector` for details).

Hops from a site at position `r₁` to another at `r₂` are described using the hop center `r =
(r₁ + r₂)/2` and the hop vector `dr = r₂ - r₁`. Hopping amplitudes `t` can be a `Number`
(for hops between single-orbital sites), a `UniformScaling` (e.g. `2I`) or an
`AbstractMatrix` (use `SMatrix` for performance) of dimensions matching the number of
orbitals in the selected sites. Models may be applied to a lattice `lat` to produce an
`Hamiltonian` with `hamiltonian(lat, model; ...)`, see `hamiltonian`. Position dependent
models are forced to preserve the periodicity of the lattice.

    hopping(m::Union{TighbindingModel,ParametricModel}; hops...)

Convert `m` into a new model with just hopping terms acting on `hops`.

## Model algebra

Models can be combined using `+`, `-` and `*`, or conjugated with `'`, e.g. `onsite(1) - 2 *
hopping(1)'`.

# Examples
```jldoctest
julia> model = hopping((r, dr) -> cis(dot(SA[r[2], -r[1]], dr)); dcells = (0,0)) + onsite(r -> rand())
TightbindingModel: model with 2 terms
  HoppingTerm{Function}:
    Region            : any
    Sublattice pairs  : any
    Cell distances    : (0, 0)
    Hopping range     : Neighbors(1)
    Reverse hops      : false
    Coefficient       : 1
  OnsiteTerm{Function}:
    Region            : any
    Sublattices       : any
    Cells             : any
    Coefficient       : 1

julia> LP.honeycomb() |> supercell(2) |> hamiltonian(model)
Hamiltonian{Float64,2,2}: Hamiltonian on a 2D Lattice in 2D space
  Bloch harmonics  : 1
  Harmonic size    : 8 × 8
  Orbitals         : [1, 1]
  Element type     : scalar (ComplexF64)
  Onsites          : 8
  Hoppings         : 16
  Coordination     : 2.0
```

# See also
    `onsite`, `@onsite`, `@hopping`, `@onsite!`, `@hopping!`, `hamiltonian`, `plusadjoint`
"""
hopping

"""
    @onsite((; params...) -> o(; params...); sites...)
    @onsite((r; params...) -> o(r; params...); sites...)

Build a `ParametricModel` representing a uniform or a position-dependent onsite potential,
respectively, on sites selected by `siteselector(; sites...)` (see `siteselector` for
details).

Site positions are `r::SVector{E}`, where `E` is the embedding dimension of the lattice. The
onsite potential `o` can be a `Number` (for single-orbital sites), a `UniformScaling` (e.g.
`2I`) or an `AbstractMatrix` (use `SMatrix` for performance) of dimensions matching the
number of orbitals in the selected sites. Parametric models may be applied to a lattice
`lat` to produce a `ParametricHamiltonian` with `hamiltonian(lat, model; ...)`, see
`hamiltonian`. Position dependent models are forced to preserve the periodicity of the
lattice.

The difference between regular and parametric tight-binding models (see `onsite` and
`hopping`) is that parametric models may depend on arbitrary parameters, specified by the
`params` keyword arguments. These are inherited by `h::ParametricHamiltonian`, which can
then be evaluated very efficiently for different parameter values by callling `h(;
params...)`, to obtain a regular `Hamiltonian` without reconstructing it from scratch.

    @onsite((ω; params...) -> Σᵢᵢ(ω; params...); sites...)
    @onsite((ω, r; params...) -> Σᵢᵢ(ω, r; params...); sites...)

Special form of a parametric onsite potential meant to model a self-energy (see `attach`).

    @onsite((i; params...) --> ...; sites...)
    @onsite((ω, i; params...) --> ...; sites...)

The `-->` syntax allows to treat the argument `i` as a site index, instead of a position. In
fact, the type of `i` is `CellSitePos`, so they can be used to index `OrbitalSliceArray`s
(see doctrings for details). The functions `pos(i)`, `cell(i)` and `ind(i)` yield the
position, cell and site index of the site. This syntax is useful to implement models that
depend on observables (in the form of `OrbitalSliceArray`s), like in self-consistent mean
field calculations.

## Model algebra

Parametric models can be combined using `+`, `-` and `*`, or conjugated with `'`, e.g.
`@onsite((; o=1) -> o) - 2 * hopping(1)'`. The combined parametric models can share
parameters.

# Examples
```jldoctest
julia> model = @onsite((r; dμ = 0) -> (r[1] + dμ) * I; sublats = :A) + @onsite((; dμ = 0) -> - dμ * I; sublats = :B)
ParametricModel: model with 2 terms
  ParametricOnsiteTerm{ParametricFunction{1}}
    Region            : any
    Sublattices       : A
    Cells             : any
    Coefficient       : 1
    Argument type     : spatial
    Parameters        : [:dμ]
  ParametricOnsiteTerm{ParametricFunction{0}}
    Region            : any
    Sublattices       : B
    Cells             : any
    Coefficient       : 1
    Argument type     : spatial
    Parameters        : [:dμ]

julia> LP.honeycomb() |> supercell(2) |> hamiltonian(model, orbitals = 2)
ParametricHamiltonian{Float64,2,2}: Parametric Hamiltonian on a 2D Lattice in 2D space
  Bloch harmonics  : 1
  Harmonic size    : 8 × 8
  Orbitals         : [2, 2]
  Element type     : 2 × 2 blocks (ComplexF64)
  Onsites          : 8
  Hoppings         : 0
  Coordination     : 0.0
  Parameters       : [:dμ]
```

# See also
    `onsite`, `hopping`, `@hopping`, `@onsite!`, `@hopping!`, `attach`, `hamiltonian`, `OrbitalSliceArray`
"""
macro onsite end

"""
    @hopping((; params...) -> t(; params...); hops...)
    @hopping((r, dr; params...) -> t(r; params...); hops...)

Build a `ParametricModel` representing a uniform or a position-dependent hopping amplitude,
respectively, on hops selected by `hopselector(; hops...)` (see `hopselector` for details).

Hops from a site at position `r₁` to another at `r₂` are described using the hop center `r =
(r₁ + r₂)/2` and the hop vector `dr = r₂ - r₁`. Hopping amplitudes `t` can be a `Number`
(for hops between single-orbital sites), a `UniformScaling` (e.g. `2I`) or an
`AbstractMatrix` (use `SMatrix` for performance) of dimensions matching the number of site
orbitals in the selected sites. Parametric models may be applied to a lattice `lat` to
produce a `ParametricHamiltonian` with `hamiltonian(lat, model; ...)`, see `hamiltonian`.
Position dependent models are forced to preserve the periodicity of the lattice.

The difference between regular and parametric tight-binding models (see `onsite` and
`hopping`) is that parametric models may depend on arbitrary parameters, specified by the
`params` keyword arguments. These are inherited by `h::ParametricHamiltonian`, which can
then be evaluated very efficiently for different parameter values by callling `h(;
params...)`, to obtain a regular `Hamiltonian` without reconstructing it from scratch.

    @hopping((ω; params...) -> Σᵢⱼ(ω; params...); hops...)
    @hopping((ω, r, dr; params...) -> Σᵢⱼ(ω, r, dr; params...); hops...)

Special form of a parametric hopping amplitude meant to model a self-energy (see `attach`).

    @hopping((i, j; params...) --> ...; sites...)
    @hopping((ω, i, j; params...) --> ...; sites...)

The `-->` syntax allows to treat the arguments `i, j` as a site indices, instead of a
positions. Here `i` is the destination (row) and `j` the source (column) site. In fact, the
type of `i` and `j` is `CellSitePos`, so they can be used to index `OrbitalSliceArray`s (see
doctrings for details). The functions `pos(i)`, `cell(i)` and `ind(i)` yield the position,
cell and site index of the site. This syntax is useful to implement models that depend on
observables (in the form of `OrbitalSliceArray`s), like in self-consistent mean field
calculations.

## Model algebra

Parametric models can be combined using `+`, `-` and `*`, or conjugated with `'`, e.g.
`@onsite((; o=1) -> o) - 2 * hopping(1)'`. The combined parametric models can share
parameters.

# Examples
```jldoctest
julia> model = @hopping((r, dr; t = 1, A = Returns(SA[0,0])) -> t * cis(-dr' * A(r)))
ParametricModel: model with 1 term
  ParametricHoppingTerm{ParametricFunction{2}}
    Region            : any
    Sublattice pairs  : any
    Cell distances    : any
    Hopping range     : Neighbors(1)
    Reverse hops      : false
    Coefficient       : 1
    Argument type     : spatial
    Parameters        : [:t, :A]

julia> LP.honeycomb() |> supercell(2) |> hamiltonian(model)
ParametricHamiltonian{Float64,2,2}: Parametric Hamiltonian on a 2D Lattice in 2D space
  Bloch harmonics  : 5
  Harmonic size    : 8 × 8
  Orbitals         : [1, 1]
  Element type     : scalar (ComplexF64)
  Onsites          : 0
  Hoppings         : 24
  Coordination     : 3.0
  Parameters       : [:A, :t]
```

# See also
    `onsite`, `hopping`, `@onsite`, `@onsite!`, `@hopping!`, `attach`, `hamiltonian`, `OrbitalSliceArray`
"""
macro hopping end

"""
    @onsite!((o; params...) -> o´(o; params...); sites...)
    @onsite!((o, r; params...) -> o´(o, r; params...); sites...)

Build a uniform or position-dependent onsite term modifier, respectively, acting on sites
selected by `siteselector(; sites...)` (see `siteselector` for details).

Site positions are `r::SVector{E}`, where `E` is the embedding dimension of the lattice. The
original onsite potential is `o`, and the modified potential is `o´`, which is a function of
`o` and possibly `r`. It may optionally also depend on parameters, enconded in `params`.

Modifiers are meant to be applied to an `h:AbstractHamiltonian` to obtain a
`ParametricHamiltonian` (with `hamiltonian(h, modifiers...)` or `hamiltonian(lat, model,
modifiers...)`, see `hamiltonian`). Modifiers will affect only pre-existing model terms. In
particular, if no onsite model has been applied to a specific site, its onsite potential
will be zero, and will not be modified by any `@onsite!` modifier. Conversely, if an onsite
model has been applied, `@onsite!` may modify the onsite potential even if it is zero. The
same applies to `@hopping!`.

    @onsite((o, i; params...) --> ...; sites...)

The `-->` syntax allows to treat the argument `i` as a site index, instead of a position. In
fact, the type of `i` is `CellSitePos`, so they can be used to index `OrbitalSliceArray`s
(see doctrings for details). The functions `pos(i)`, `cell(i)` and `ind(i)` yield the
position, cell and site index of the site. This syntax is useful to implement models that
depend on observables (in the form of `OrbitalSliceArray`s), like in self-consistent mean
field calculations.

# Examples
```jldoctest
julia> model = onsite(0); disorder = @onsite!((o; W = 0) -> o + W * rand())
OnsiteModifier{ParametricFunction{1}}:
  Region            : any
  Sublattices       : any
  Cells             : any
  Argument type     : spatial
  Parameters        : [:W]

julia> LP.honeycomb() |> hamiltonian(model) |> supercell(10) |> hamiltonian(disorder)
ParametricHamiltonian{Float64,2,2}: Parametric Hamiltonian on a 2D Lattice in 2D space
  Bloch harmonics  : 1
  Harmonic size    : 200 × 200
  Orbitals         : [1, 1]
  Element type     : scalar (ComplexF64)
  Onsites          : 200
  Hoppings         : 0
  Coordination     : 0.0
  Parameters       : [:W]
```

# See also
    `onsite`, `hopping`, `@onsite`, `@hopping`, `@hopping!`, `hamiltonian`, `OrbitalSliceArray`
"""
macro onsite! end

"""
    @hopping!((t; params...) -> t´(t; params...); hops...)
    @hopping!((t, r, dr; params...) -> t´(t, r, dr; params...); hops...)

Build a uniform or position-dependent hopping term modifier, respectively, acting on hops
selected by `hopselector(; hops...)` (see `hopselector` for details).

Hops from a site at position `r₁` to another at `r₂` are described using the hop center `r =
(r₁ + r₂)/2` and the hop vector `dr = r₂ - r₁`. The original hopping amplitude is `t`, and
the modified hopping is `t´`, which is a function of `t` and possibly `r, dr`. It may
optionally also depend on parameters, enconded in `params`.

Modifiers are meant to be applied to an `h:AbstractHamiltonian` to obtain a
`ParametricHamiltonian` (with `hamiltonian(h, modifiers...)` or `hamiltonian(lat, model,
modifiers...)`, see `hamiltonian`). Modifiers will affect only pre-existing model terms. In
particular, if no onsite model has been applied to a specific site, its onsite potential
will be zero, and will not be modified by any `@onsite!` modifier. Conversely, if an onsite
model has been applied, `@onsite!` may modify the onsite potential even if it is zero. The
same applies to `@hopping!`.

    @hopping!((t, i, j; params...) --> ...; sites...)

The `-->` syntax allows to treat the arguments `i, j` as a site indices, instead of a
positions. Here `i` is the destination (row) and `j` the source (column) site. In fact, the
type of `i` and `j` is `CellSitePos`, so they can be used to index `OrbitalSliceArray`s (see
doctrings for details). The functions `pos(i)`, `cell(i)` and `ind(i)` yield the position,
cell and site index of the site. This syntax is useful to implement models that depend on
observables (in the form of `OrbitalSliceArray`s), like in self-consistent mean field
calculations.

# Examples
```jldoctest
julia> model = hopping(1); peierls = @hopping!((t, r, dr; A = r -> SA[0,0]) -> t * cis(-dr' * A(r)))
HoppingModifier{ParametricFunction{3}}:
  Region            : any
  Sublattice pairs  : any
  Cell distances    : any
  Hopping range     : Inf
  Reverse hops      : false
  Argument type     : spatial
  Parameters        : [:A]

julia> LP.honeycomb() |> hamiltonian(model) |> supercell(10) |> hamiltonian(peierls)
ParametricHamiltonian{Float64,2,2}: Parametric Hamiltonian on a 2D Lattice in 2D space
  Bloch harmonics  : 5
  Harmonic size    : 200 × 200
  Orbitals         : [1, 1]
  Element type     : scalar (ComplexF64)
  Onsites          : 0
  Hoppings         : 600
  Coordination     : 3.0
  Parameters       : [:A]
```

# See also
    `onsite`, `hopping`, `@onsite`, `@hopping`, `@onsite!`, `hamiltonian`, `OrbitalSliceArray`
"""
macro hopping! end

"""
    plusadjoint(t::Model)

Returns a model `t + t'`. This is a convenience function analogous to the `+ h.c.` notation.

# Example
```jldoctest
julia> model = hopping(im, sublats = :A => :B) |> plusadjoint
TightbindingModel: model with 2 terms
  HoppingTerm{Complex{Bool}}:
    Region            : any
    Sublattice pairs  : :A => :B
    Cell distances    : any
    Hopping range     : Neighbors(1)
    Reverse hops      : false
    Coefficient       : 1
  HoppingTerm{Complex{Int64}}:
    Region            : any
    Sublattice pairs  : :A => :B
    Cell distances    : any
    Hopping range     : Neighbors(1)
    Reverse hops      : true
    Coefficient       : 1

julia> h = hamiltonian(LP.honeycomb(), model)
Hamiltonian{Float64,2,2}: Hamiltonian on a 2D Lattice in 2D space
  Bloch harmonics  : 5
  Harmonic size    : 2 × 2
  Orbitals         : [1, 1]
  Element type     : scalar (ComplexF64)
  Onsites          : 0
  Hoppings         : 6
  Coordination     : 3.0

julia> h((0,0))
2×2 SparseArrays.SparseMatrixCSC{ComplexF64, Int64} with 2 stored entries:
     ⋅      0.0-3.0im
 0.0+3.0im      ⋅
```

"""
plusadjoint

"""

    torus(h::AbstractHamiltonian, (ϕ₁, ϕ₂,...))

For an `h` of lattice dimension `L` and a set of `L` Bloch phases `ϕ = (ϕ₁, ϕ₂,...)`,
contruct a new `h´::AbstractHamiltonian` on a bounded torus, i.e. with all Bravais
vectors eliminated by stitching the lattice onto itself along the corresponding Bravais
vector. Intercell hoppings along stitched directions will pick up a Bloch phase
`exp(-iϕ⋅dn)`.

If a number `L´` of phases `ϕᵢ` are `:` instead of numbers, the corresponding Bravais
vectors will not be stitched, and the resulting `h´` will have a finite lattice dimension
`L´`.

## Currying

    h |> torus((ϕ₁, ϕ₂,...))

Currying syntax equivalent to `torus(h, (ϕ₁, ϕ₂,...))`.

# Examples

```jldoctest
julia> h2D = HP.graphene(); h1D = torus(h2D, (:, 0.2))
Hamiltonian{Float64,2,1}: Hamiltonian on a 1D Lattice in 2D space
  Bloch harmonics  : 3
  Harmonic size    : 2 × 2
  Orbitals         : [1, 1]
  Element type     : scalar (ComplexF64)
  Onsites          : 0
  Hoppings         : 4
  Coordination     : 2.0

julia> h2D((0.3, 0.2)) ≈ h1D(0.3)
true
```

# See also
    `hamiltonian`, `supercell`
"""
torus

"""
    unflat(dn)

Construct an `u::Unflat` object wrapping some indices `dn`. This object is meant to be used
to index into a `h::AbstractHamiltonian` as `h[u]`, which returns an non-flattened version
of the Bloch harmonic `h[dn]`. Each element in the matrix `h[u]` is an `SMatrix` block
representing onsite or hoppings between whole sites, in contrast to `h[dn]` where they are
scalars representing single orbitals. This is only relevant for multi-orbital Hamiltonians
`h`.

    unflat()

Equivalent to `unflat(())`

# Examples

```
julia> h = HP.graphene(orbitals = 2); h[unflat(0,0)]
2×2 SparseArrays.SparseMatrixCSC{SMatrix{2, 2, ComplexF64, 4}, Int64} with 2 stored entries:
                     ⋅                       [2.7+0.0im 0.0+0.0im; 0.0+0.0im 2.7+0.0im]
 [2.7+0.0im 0.0+0.0im; 0.0+0.0im 2.7+0.0im]                      ⋅
```
"""
unflat

"""
`EigenSolvers` is a Quantica submodule containing support for several pre-defined
eigensolver extensions. The alias `ES` can be used in place of `EigenSolvers`. Currently
supported solvers are

    ES.LinearAlgebra(; kw...)       # Uses `eigen(mat; kw...)` from the `LinearAlgebra` package
    ES.Arpack(; kw...)              # Uses `eigs(mat; kw...)` from the `Arpack` package (WARNING: Arpack is not thread-safe)
    ES.KrylovKit(params...; kw...)  # Uses `eigsolve(mat, params...; kw...)` from the `KrylovKit` package
    ES.ArnoldiMethod(; kw...)       # Uses `partialschur(mat; kw...)` from the `ArnoldiMethod` package.

To use each of these solvers the corresponding package must be loaded with e.g. `using
ArnoldiMethod`. The exception is the default `ES.LinearAlgebra` which is a direct
Quantica dependency and does not require loading.

Additionally, to compute interior eigenvalues, we can use a shift-invert method around
energy `ϵ0` (uses `LinearMaps` and a `LinearAlgebra.lu` factorization), combined with any
solver `s` from the list above:

    ES.ShiftInvert(s, ϵ0)           # Perform a lu-based shift-invert with solver `s`

The `ShiftInvert` solver extension requires doing `using LinearMaps`.

# Examples

```
julia> using LinearMaps, ArnoldiMethod  # loads required extensions

julia> h = HP.graphene(t0 = 1) |> supercell(10);

julia> spectrum(h, (0,0); solver = ES.ShiftInvert(ES.ArnoldiMethod(nev = 4), 0.0)) |> energies
4-element Vector{ComplexF64}:
 -0.3819660112501042 + 2.407681231060336e-16im
 -0.6180339887498942 - 2.7336317916863215e-16im
  0.6180339887498937 - 1.7243387890744497e-16im
  0.3819660112501042 - 1.083582785131051e-16im
```

# See also
    `spectrum`, `bands`
"""
EigenSolvers

"""
    spectrum(h::AbstractHamiltonian, ϕs; solver = EigenSolvers.LinearAlgebra(), transform = missing, params...)

Compute the `Spectrum` of the Bloch matrix `h(ϕs; params...)` using the specified
eigensolver, with `transform` applied to the resulting eigenenergies, if not `missing`.
Eigenpairs are sorted by the real part of their energy. See `EigenSolvers` for available
solvers and their options.

    spectrum(h::AbstractHamiltonian; kw...)

For a 0D `h`, equivalent to `spectrum(h, (); kw...)`

    spectrum(m::AbstractMatrix; solver = EigenSolvers.LinearAlgebra()], transform = missing)

Compute the `Spectrum` of matrix `m` using `solver` and `transform`.

    spectrum(b::Bandstructure, ϕs)

Compute the `Spectrum` corresponding to slicing the bandstructure `b` at point `ϕs` of its
base mesh (see `bands` for details).

## Indexing and destructuring

Eigenenergies `ϵs::Tuple` and eigenstates `ψs::Matrix` can be extracted from a spectrum `sp`
using any of the following

    ϵs, ψs = sp
    ϵs = first(sp)
    ϵs = energies(sp)
    ψs = last(sp)
    ψs = states(sp)

In addition, one can extract the `n` eigenpairs closest (in real energy) to a given energy
`ϵ₀` with

    ϵs, ψs = sp[1:n, around = ϵ₀]

More generally, `sp[inds, around = ϵ₀]` will take the eigenpairs at position given by `inds`
after sorting by increasing distance to `ϵ₀`, or the closest eigenpair in `inds` is missing.
If `around` is omitted, the ordering in `sp` is used.

# Examples

```jldoctest
julia> h = HP.graphene(t0 = 1); spectrum(h, (0,0))
Spectrum{Float64,ComplexF64} :
Energies:
2-element Vector{ComplexF64}:
 -2.9999999999999982 + 0.0im
  2.9999999999999982 + 0.0im
States:
2×2 Matrix{ComplexF64}:
 -0.707107+0.0im  0.707107+0.0im
  0.707107+0.0im  0.707107+0.0im
```

# See also
    `EigenSolvers`, `bands`
"""
spectrum

"""
    energies(sp::Spectrum)

Returns the energies in `sp` as a vector of Numbers (not necessarily real). Equivalent to `first(sp)`.

# See also
    `spectrum`, `bands`
"""
energies

"""
    states(sp::Spectrum)

Returns the eigenstates in `sp` as columns of a matrix. Equivalent to `last(sp)`.

# See also
    `spectrum`, `bands`
"""
states

"""
    bands(h::AbstractHamiltonian, xcolᵢ...; kw...)

Construct a `Bandstructure` object, which contains in particular a collection of
continuously connected `Subband`s of `h`, obtained by diagonalizing the matrix `h(ϕs;
params...)` on an `M`-dimensional mesh of points `(x₁, x₂, ..., xₘ)`, where each `xᵢ` takes
values in the collection `xcolᵢ`. The mapping between points in the mesh points and values
of `(ϕs; params...)` is defined by keyword `mapping` (`identity` by default, see Keywords).
Diagonalization is multithreaded and will use all available Julia threads (start session
with `julia -t N` to have `N` threads).

    bands(f::Function, xcolᵢ...; kw...)

Like the above using `f(ϕs)::AbstractMatrix` in place of `h(ϕs; params...)`, and returning a
`Vector{<:Subband}` instead of a `Bandstructure` object. This is provided as a lower level
driver without the added slicing functionality of a full `Bandstructure` object, see below.

    bands(h::AbstractHamiltonian; kw...)

Equivalent to `bands(h::AbstractHamiltonian, xcolᵢ...; kw...)` with a default `xcolᵢ =
subdiv(-π, π, 49)`.

## Keywords

- `solver`: eigensolver to use for each diagonalization (see `Eigensolvers`). Default: `ES.LinearAlgebra()`
- `mapping`: a function of the form `(x, y, ...) -> ϕs` or `(x, y, ...) -> ftuple(ϕs...; params...)` that translates points `(x, y, ...)` in the mesh to Bloch phases `ϕs` or phase+parameter FrankenTuples `ftuple(ϕs...; params...)`. See also linecuts below. Default: `identity`
- `transform`: function to apply to each eigenvalue after diagonalization. Default: `identity`
- `degtol::Real`: maximum distance between to nearby eigenvalue so that they are classified as degenerate. Default: `sqrt(eps)`
- `split::Bool`: whether to split bands into disconnected subbands. Default: `true`
- `projectors::Bool`: whether to compute interpolating subspaces in each simplex (for use as GreenSolver). Default: `true`
- `warn::Bool`: whether to emit warning when band dislocations are encountered. Default: `true`
- `showprogress::Bool`: whether to show or not a progress bar. Default: `true`
- `defects`: (experimental) a collection of extra points to add to the mesh, typically the location of topological band defects such as Dirac points, so that interpolation avoids creating dislocation defects in the bands. You need to also increase `patches` to repair the subband dislocations using the added defect vertices. Default: `()`
- `patches::Integer`: (experimental) if a dislocation is encountered, attempt to patch it by searching for the defect recursively to a given order, or using the provided `defects` (preferred). Default: `0`

## Currying

    h |> bands(xcolᵢ...; kw...)

Curried form of `bands` equivalent to `bands(h, xcolᵢ...; kw...)`

## Band linecuts

To do a linecut of a bandstructure along a polygonal path in the `L`-dimensional Brillouin
zone, mapping a set of 1D points `xs` to a set of `nodes`, with `pts` interpolation points
in each segment, one can use the following convenient syntax

    bands(h, subdiv(xs, pts); mapping = (xs => nodes))

Here `nodes` can be a collection of `SVector{L}` or of named Brillouin zone points from the
list (`:Γ`,`:K`, `:K´`, `:M`, `:X`, `:Y`, `:Z`). If `mapping = nodes`, then `xs` defaults to
`0:length(nodes)-1`. See also `subdiv` for its alternative methods.

## Indexing and slicing

    b[i]

Extract `i`-th subband from `b::Bandstructure`. `i` can also be a `Vector`, an
`AbstractRange` or any other argument accepted by `getindex(subbands::Vector, i)`

    b[slice::Tuple]

Compute a section of `b::Bandstructure` with a "plane" defined by `slice = (ϕ₁, ϕ₂,..., ϕₗ[,
ϵ])`, where each `ϕᵢ` or `ϵ` can be a real number (representing a fixed momentum or energy)
or a `:` (unconstrained along that dimension). For bands of an `L`-dimensional lattice,
`slice` will be padded to an `L+1`-long tuple with `:` if necessary. The result is a
collection of of sliced `Subband`s.

# Examples

```
julia> phis = range(0, 2pi, length = 50); h = LP.honeycomb() |> hamiltonian(@hopping((; t = 1) -> t));

julia> bands(h(t = 1), phis, phis)
Bandstructure{Float64,3,2}: 3D Bandstructure over a 2-dimensional parameter space of type Float64
  Subbands  : 1
  Vertices  : 5000
  Edges     : 14602
  Simplices : 9588

julia> bands(h, phis, phis; mapping = (x, y) -> ftuple(0, x; t = y/2π))
Bandstructure{Float64,3,2}: 3D Bandstructure over a 2-dimensional parameter space of type Float64
  Subbands  : 1
  Vertices  : 4950
  Edges     : 14553
  Simplices : 9604

julia> bands(h(t = 1), subdiv((0, 2, 3), (20, 30)); mapping = (0, 2, 3) => (:Γ, :M, :K))
Bandstructure{Float64,2,1}: 2D Bandstructure over a 1-dimensional parameter space of type Float64
  Subbands  : 1
  Vertices  : 97
  Edges     : 96
  Simplices : 96
```

# See also
    `spectrum`, `subdiv`
"""
bands

"""
    subdiv((x₁, x₂, ..., xₙ), (p₁, p₂, ..., pₙ₋₁))

Build a vector of values between `x₁` and `xₙ` containing all `xᵢ` such that in each
interval `[xᵢ, xᵢ₊₁]` there are `pᵢ` equally space values.

    subdiv((x₁, x₂, ..., xₙ), p)

Same as above with all `pᵢ = p`

    subdiv(x₁, x₂, p)

Equivalent to `subdiv((x₁, x₂), p) == collect(range(x₁, x₂, length = p))`
"""
subdiv

"""
    attach(h::AbstractHamiltonian, args..; sites...)
    attach(h::OpenHamiltonian, args...; sites...)

Build an `h´::OpenHamiltonian` by attaching (adding) a `Σ::SelfEnergy` to a finite number of
sites in `h` specified by `siteselector(; sites...)`. This also defines a "contact" on said
sites that can be referred to (with index `i::Integer` for the i-th attached contact) when
slicing Green functions later. Self-energies are taken into account when building the Green
function `g(ω) = (ω - h´ - Σ(ω))⁻¹` of the resulting `h´`, see `greenfunction`.

## Self-energy forms

The different forms of `args` yield different types of self-energies `Σ`. Currently
supported forms are:

    attach(h, gs::GreenSlice, coupling::AbstractModel; transform = missing, sites...)

Adds a generic self-energy `Σ(ω) = V´⋅gs(ω)⋅V` on `h`'s `sites`, where `V` and `V´` are
couplings (given by `coupling`) between said `sites` and the `LatticeSlice` in `gs` (after
applying `transform` to the latter). Allowed forms of `gs` include both `g[bath_sites...]`
and `g[contactind::Integer]` where `g` is any `GreenFunction`.

    attach(h, model::ParametricModel; sites...)

Add self-energy `Σᵢⱼ(ω)` defined by a `model` composed of parametric terms (`@onsite` and
`@hopping`) with `ω` as first argument, as in e.g. `@onsite((ω, r) -> Σᵢᵢ(ω, r))` and
`@hopping((ω, r, dr) -> Σᵢⱼ(ω, r, dr))`

    attach(h, nothing; sites...)

Add a `nothing` contact with a null self-energy `Σᵢⱼ(ω) = 0` on selected sites, which in
effect simply amounts to labeling those sites with a contact number, but does not lead to
any dressing the Green function. This is useful for some `GreenFunction` solvers such as
`GS.KPM` (see `greenfunction`), which need to know the sites of interest beforehand (the
contact sites in this case).

    attach(h, g1D::GreenFunction; reverse = false, transform = identity, sites...)

Add a self-energy `Σ(ω) = h₋₁⋅g1D(ω)[surface]⋅h₁` corresponding to a semi-infinite 1D lead
(i.e. with a finite `boundary`, see `greenfunction`), where `h₁` and `h₋₁` are intercell
couplings, and `g1D` is the lead `GreenFunction`. The `g1D(ω)` is taken at the `suface`
unitcell, either adjacent to the `boundary` on its positive side (if `reverse = false`) or
on its negative side (if `reverse = true`). Note that `reverse` only flips the direction we
extend the lattice to form the lead, but does not flip the unit cell (may use `transform`
for that) or any contacts in the lead. The positions of the selected `sites` in `h` must
match, modulo an arbitrary displacement, those of the left or right unit cell surface of the
lead (i.e. sites coupled to the adjacent unit cells), after applying `transform` to the
latter. If they don't match, use the `attach` syntax below.

Advanced: If the `g1D` does not have any self-energies, the produced self-energy is in fact
an `ExtendedSelfEnergy`, which is numerically more stable than a naive implementation of
`RegularSelfEnergy`'s, since `g1D(ω)[surface]` is never actually computed. Conversely, if
`g1D` has self-energies attached, a `RegularSelfEnergy` is produced.

    attach(h, g1D::GreenFunction, coupling::AbstractModel; reverse = false, transform = identity,  sites...)

Add a self-energy `Σ(ω) = V´⋅g1D(ω)[surface]⋅V` corresponding to a 1D lead (semi-infinite or
infinite), but with couplings `V` and `V´`, defined by `coupling`, between `sites` and the
`surface` lead unitcell (or the one with index zero if there is no boundary) . See also
Advanced note above.

## Currying

    h |> attach(args...; sites...)

Curried form equivalent to `attach(h, args...; sites...)`.

# Examples

```jldoctest
julia> # A graphene flake with two out-of-plane cubic-lattice leads

julia> g1D = LP.cubic(names = :C) |> hamiltonian(hopping(1)) |> supercell((0,0,1), region = RP.square(4)) |> greenfunction(GS.Schur(boundary = 0));

julia> coupling = hopping(1, range = 2);

julia> gdisk = HP.graphene(a0 = 1, dim = 3) |> supercell(region = RP.circle(10)) |> attach(g1D, coupling; region = RP.square(4)) |> attach(g1D, coupling; region = RP.square(4), reverse = true) |> greenfunction;
```

# See also
    `greenfunction`, `GreenSolvers`

"""
attach

"""
    greenfunction(h::Union{AbstractHamiltonian,OpenHamiltonian}, solver::AbstractGreenSolver)

Build a `g::GreenFunction` of Hamiltonian `h` using `solver`. See `GreenSolvers` for
available solvers. If `solver` is not provided, a default solver is chosen automatically
based on the type of `h`.

## Currying

    h |> greenfunction(solver)

Curried form equivalent to `greenfunction(h, solver)`.

## Partial evaluation

`GreenFunction`s allow independent, partial evaluation of their positions (producing a
`GreenSlice`) and energy/parameters (producing a `GreenSolution`). Depending on the solver,
this may avoid repeating calculations unnecesarily when sweeping over either of these with
the other fixed.

    g[ss]
    g[siteselector(; ss...)]

Build a `gs::GreenSlice` that represents a Green function at arbitrary energy and parameter
values, but at specific sites on the lattice defined by `siteselector(; ss...)`, with
`ss::NamedTuple` (see `siteselector`).

    g[contact_index::Integer]

Build a `GreenSlice` equivalent to `g[contact_sites...]`, where `contact_sites...`
correspond to sites in contact number `contact_index` (must have `1<= contact_index <=
number_of_contacts`). See `attach` for details on attaching contacts to a Hamiltonian.

    g[:]

Build a `GreenSlice` over all contacts.

    g[dst, src]

Build a `gs::GreenSlice` between sites specified by `src` and `dst`, which can take any of
the forms above. Therefore, all the previous single-index slice forms correspond to a
diagonal block `g[i, i]`.

    g[diagonal(i; kernel = missing)]

Build a diagonal `gs::GreenSlice` over sites specified by `i`. If `kernel = missing` the
diagonal entries are `g[o, o]` for orbitals `o` in sites encoded in `i`. If `kernel` is a
matrix, the diagonal elements are `tr(g[site, site] * kernel)` over each site `i`. Note that
if there are several orbitals per site, `g[site, site]` may have different sizes (i.e.
number of orbitals vs number of sites). Upon evaluating `gs(ω)`, the result is a `Diagonal`
matrix wrapped in an `OrbitalSliceMatrix`, and spans full unit cells.See also `diagonal`.

    g[sitepairs(; kernel = missing, hops...)]

Like the above but for a selection of site pairs selected by `hopselector(; hops...)`. Upon
evaluating `gs(ω)`, the result is a `SparseMatrixCSC` wrapped in an `OrbitalSliceMatrix`,
and spans full unit cells. See also `sitepairs`.

    g(ω; params...)

Build a `gω::GreenSolution` that represents a retarded Green function at arbitrary points on
the lattice, but at fixed energy `ω` and system parameter values `param`. If `ω` is complex,
the retarded or advanced Green function is returned, depending on `sign(imag(ω))`. If `ω` is
`Real`, a small, positive imaginary part is automatically added internally to produce the
retarded `g`.

    gω[i]
    gω[i, j]
    gs(ω; params...)

For any `gω::GreenSolution` or `gs::GreenSlice`, build the Green function matrix fully
evaluated at fixed energy, parameters and positions. The matrix is a dense
`m::OrbitalSliceMatrix` with scalar element type, so that any orbital structure on each site
is flattened. Note that the resulting `m` can itself be indexed over collections of sites
with `m[i, j]`, where `i, j` are `siteselector(; ss...)` or `ss::NamedTuple`.

    view(gω, i::C, j::C == i)

For any `gω::GreenSolution` and `C<:Union{Colon,Integer}`, obtain a view (of type
`SubArray`, not `OrbitalSliceMatrix`) of the corresponding intra or inter-contact propagator
`gω[i, j]` with minimal allocations.

    g(; params...)

For any `g::Union{GreenFunction,GreenSlice}`, produce a new `GreenFunction` or `GreenSlice`
with all parameters fixed to `params` (or to their default values if not provided).

## Full evaluation

    gs(ω; params...)
    gω[sites...]

For `gs::GreenSlice` or `gω::GreenSolution`, return a fully evaluated `m::AbstractMatrix`.
If the selected site slice was defined using `sites`, the concrete type of `m` will be will
be a conventional `Matrix`-based type. Otherwise, it will be of type `OrbitalSliceMatrix`,
an `AbstractMatrix` type that supports both conventional indexing and indexing with `sites`
and `siteselectors`.

Advanced: in addition to the above, an unexported method `Quantica.call!(gs, ω; params...)`
is provided to reuse the output matrix `m` (preallocated inside `gs`). Use with caution, as
it may lead to unexpected aliasing

# Example
```jldoctest
julia> g = LP.honeycomb() |> hamiltonian(@hopping((; t = 1) -> t)) |> supercell(region = RP.circle(10)) |> greenfunction(GS.SparseLU())
GreenFunction{Float64,2,0}: Green function of a Hamiltonian{Float64,2,0}
  Solver          : AppliedSparseLUGreenSolver
  Contacts        : 0
  Contact solvers : ()
  Contact sizes   : ()
  ParametricHamiltonian{Float64,2,0}: Parametric Hamiltonian on a 0D Lattice in 2D space
    Bloch harmonics  : 1
    Harmonic size    : 726 × 726
    Orbitals         : [1, 1]
    Element type     : scalar (ComplexF64)
    Onsites          : 0
    Hoppings         : 2098
    Coordination     : 2.88981
    Parameters       : [:t]

julia> gω = g(0.1; t = 2)
GreenSolution{Float64,2,0}: Green function at arbitrary positions, but at a fixed energy

julia> ss = (; region = RP.circle(2), sublats = :B);

julia> gs = g[ss]
GreenSlice{Float64,2,0}: Green function at arbitrary energy, but at a fixed lattice positions

julia> gω[ss] == gs(0.1; t = 2)
true

julia> summary(gω[ss])
"14×14 OrbitalSliceMatrix{ComplexF64,Array}"
```

# See also
    `GreenSolvers`, `diagonal`, `sitepairs`, `ldos`, `conductance`, `current`, `josephson`
"""
greenfunction

"""

`GreenSolvers` is a Quantica submodule containing several pre-defined Green function
solvers. The alias `GS` can be used in place of `GS`. Currently supported solvers and their
possible keyword arguments are

- `GS.SparseLU()` : Direct inversion solver for 0D Hamiltonians using a `SparseArrays.lu(hmat)` factorization
- `GS.Spectrum(; spectrum_kw...)` : Diagonalization solver for 0D Hamiltonians using `spectrum(h; spectrum_kw...)`
    - `spectrum_kw...` : keyword arguments passed on to `spectrum`
    - This solver does not accept ParametricHamiltonians. Convert to Hamiltonian with `h(; params...)` first. Contact self-energies that depend on parameters are supported.
- `GS.Schur(; boundary = Inf)` : Solver for 1D Hamiltonians based on a deflated, generalized Schur factorization
    - `boundary` : 1D cell index of a boundary cell, or `Inf` for no boundaries. Equivalent to removing that specific cell from the lattice when computing the Green function.
- `GS.KPM(; order = 100, bandrange = missing, kernel = I)` : Kernel polynomial method solver for 0D Hamiltonians
    - `order` : order of the expansion in Chebyshev polynomials `Tₙ(h)` of the Hamiltonian `h` (lowest possible order is `n = 0`).
    - `bandrange` : a `(min_energy, max_energy)::Tuple` interval that encompasses the full band of the Hamiltonian. If `missing`, it is computed automatically, but `using ArnoldiMethod` is required first.
    - `kernel` : generalization that computes momenta as `μₙ = Tr[Tₙ(h)*kernel]`, so that the local density of states (see `ldos`) becomes the density of the `kernel` operator.
    - This solver does not allow arbitrary indexing of the resulting `g::GreenFunction`, only on contacts `g[contact_ind::Integer]`. If the system has none, we can add a dummy contact using `attach(h, nothing; sites...)`, see `attach`.
- `GS.Bands(bands_arguments; boundary = missing, bands_kw...)`: solver based on the integration of bandstructure simplices
    - `bands_arguments`: positional arguments passed on to `bands`
    - `bands_kw`: keyword arguments passed on to `bands`
    - `boundary`: either `missing` (no boundary), or `dir => cell_pos` (single boundary), where `dir::Integer` is the Bravais vector normal to the boundary, and `cell_pos::Integer` the value of cell indices `cells[dir]` that define the boundary (i.e. `cells[dir] <= cell_pos` are vaccum)
    - This solver only allows zero or one boundary. WARNING: if a boundary is used, the algorithm may become unstable for very fine band meshes.

"""
GreenSolvers

"""
    diagonal(i; kernel = missing)

Wrapper over site or orbital indices `i` (used to index into a `g::GreenFunction` or
`g::GreenSolution`) that represent purely diagonal entries. Here `i` can be any index
accepted in `g[i,i]`, e.g. `i::Integer` (contact index), `i::Colon` (merged contacts),
`i::SiteSelector` (selected sites), etc.

If `kernel = Q` (a matrix) instead of `missing`, each diagonal block for multiorbital site
`i` is replaced with `Tr(gᵢᵢQ)`.

For a `gω::GreenSolution`, `gω[diagonal(sel)] = diag(gω[sel, sel])`, although where possible
the former computation is done more efficiently internally.

    diagonal(; kernel = missing, sites...)

Equivalent to `diagonal(siteselector(; sites...); kernel)`

## Keywords

    - `kernel`: if missing, all orbitals in the diagonal `g[i, i]` are returned when indexing `g[diagonal(i)]`. Otherwise, `Tr(g[site, site]*kernel)` for each site included in `i` is returned.

# Example
```julia
julia> g = HP.graphene(orbitals = 2) |> attach(nothing, cells = (0,0)) |> greenfunction();

julia> g(1)[diagonal(:)]                            # g(ω = 1) diagonal on all contact orbitals
4×4 OrbitalSliceMatrix{ComplexF64,LinearAlgebra.Diagonal{ComplexF64, Vector{ComplexF64}}}:
 -0.10919-0.0839858im       0.0+0.0im             0.0+0.0im             0.0+0.0im
      0.0+0.0im        -0.10919-0.0839858im       0.0+0.0im             0.0+0.0im
      0.0+0.0im             0.0+0.0im        -0.10919-0.0839858im       0.0+0.0im
      0.0+0.0im             0.0+0.0im             0.0+0.0im        -0.10919-0.0839858im

julia> g(1)[diagonal(:, kernel = SA[1 0; 0 -1])]    # σz spin density of the above
2×2 OrbitalSliceMatrix{ComplexF64,LinearAlgebra.Diagonal{ComplexF64, Vector{ComplexF64}}}:
 5.61885e-12+1.38778e-17im           0.0+0.0im
         0.0+0.0im          -5.61882e-12+2.77556e-17im
```

# See also
    `sitepairs`, `greenfunction`, `ldos`, `densitymatrix`
"""
diagonal

"""

    sitepairs(s::HopSelector; kernel = missing)

Create a selection of site pairs `s::SparseIndices` used to sparsely index into a
`g::GreenFunction` or `g::GreenSolution`, as `g[s]`. Of the resulting `OrbitalSliceMatrix`
only the selected pairs of matrix elements will be computed, leaving the rest as zero
(sparse matrix). The sparse matrix spans the minimum number of complete unit cells to
include all site pairs

If `kernel = Q` (a matrix instead of `missing`), each of these site blocks `gᵢⱼ` will be
replaced by `Tr(kernel * gᵢⱼ)`.

    sitepairs(; kernel = missing, hops...)

Equivalent to `sitepairs(hopselector(; hops...); kernel)`

## Keywords

    - `kernel`: if missing, all orbitals blocks `gᵢⱼ = g[i, j]` between selected sites pairs (i,j) are returned when indexing `g[sitepairs(...)]`. Otherwise, `gᵢⱼ` is replaced by `Tr(gᵢⱼ*kernel)`.

# Example
```julia
julia> g = HP.graphene(orbitals = 2, a0 = 1) |> attach(nothing, cells = (0,0)) |> greenfunction();

julia> summary(g(1)[sitepairs(range = 1)])     # g(ω=1) site blocks between all sites in zero cell and all other sites at distance 1
"28×4 OrbitalSliceMatrix{ComplexF64,SparseArrays.SparseMatrixCSC{ComplexF64, Int64}}"

julia> summary(g(1)[sitepairs(range = 1, kernel = SA[1 0; 0 -1])])    # σz spin density of the above
"14×2 OrbitalSliceMatrix{ComplexF64,SparseArrays.SparseMatrixCSC{ComplexF64, Int64}}"
```

# See also
    `diagonal`, `hopselector`, `greenfunction`, `ldos`, `densitymatrix`
"""
sitepairs

"""
    ldos(gs::GreenSlice; kernel = missing)

Build `ρs::LocalSpectralDensitySlice`, a partially evaluated object representing the local
density of states `ρᵢ(ω)` at specific sites `i` but at arbitrary energy `ω`.

    ldos(gω::GreenSolution; kernel = missing)

Build `ρω::LocalSpectralDensitySolution`, as above, but for `ρᵢ(ω)` at a fixed `ω` and
arbitrary sites `i`. See also `greenfunction` for details on building a `GreenSlice` and
`GreenSolution`.

The local density of states is defined here as ``ρᵢ(ω) = -Tr(gᵢᵢ(ω))/π``, where `gᵢᵢ(ω)` is
the retarded Green function at a given site `i`.

## Keywords

- `kernel` : for multiorbital sites, `kernel` allows to compute a generalized `ldos` `ρᵢ(ω) = -imag(Tr(gᵢᵢ(ω) * kernel))/π`, where `gᵢᵢ(ω)` is the retarded Green function at site `i` and energy `ω`. If `kernel = missing`, the complete, orbital-resolved `ldos` is returned. Default: `missing`

## Full evaluation

    ρω[sites...]
    ρs(ω; params...)

Given a partially evaluated `ρω::LocalSpectralDensitySolution` or
`ρs::LocalSpectralDensitySlice`, build an `OrbitalSliceVector` `[ρ₁(ω), ρ₂(ω)...]` of fully
evaluated local densities of states. See `OrbitalSliceVector` for further details.

# Example
```
julia> g = HP.graphene(a0 = 1, t0 = 1) |> supercell(region = RP.circle(20)) |> attach(nothing, region = RP.circle(1)) |> greenfunction(GS.KPM(order = 300, bandrange = (-3.1, 3.1)))
GreenFunction{Float64,2,0}: Green function of a Hamiltonian{Float64,2,0}
  Solver          : AppliedKPMGreenSolver
  Contacts        : 1
  Contact solvers : (SelfEnergyEmptySolver,)
  Contact sizes   : (6,)
  Hamiltonian{Float64,2,0}: Hamiltonian on a 0D Lattice in 2D space
    Bloch harmonics  : 1
    Harmonic size    : 2898 × 2898
    Orbitals         : [1, 1]
    Element type     : scalar (ComplexF64)
    Onsites          : 0
    Hoppings         : 8522
    Coordination     : 2.94065

julia> ldos(g(0.2))[1]
6-element OrbitalSliceVector{Vector{Float64}}:
 0.036802204179316955
 0.034933055722650375
 0.03493305572265026
 0.03493305572265034
 0.03493305572265045
 0.036802204179317045

julia> ldos(g(0.2))[1] == -imag.(diag(g[diagonal(1)](0.2))) ./ π
true
```

# See also
    `greenfunction`, `diagonal`, `current`, `conductance`, `josephson`, `transmission`, `OrbitalSliceVector`

"""
ldos

"""
    current(h::AbstractHamiltonian; charge = -I, direction = 1)

Build an `Operator` object that behaves like a `ParametricHamiltonian` in regards to calls
and getindex, but whose matrix elements are hoppings ``im*(rⱼ-rᵢ)[direction]*charge*tⱼᵢ``,
where `tᵢⱼ` are the hoppings in `h`. This operator is equal to ``∂h/∂Aᵢ``, where `Aᵢ`
is a gauge field along `direction = i`.

    current(gs::GreenSlice; charge = -I, direction = missing)

Build `Js::CurrentDensitySlice`, a partially evaluated object representing the equilibrium
local current density `Jᵢⱼ(ω)` at arbitrary energy `ω` from site `j` to site `i`, both taken
from a specific lattice slice. The current is computed along a given `direction` (see
Keywords).

    current(gω::GreenSolution; charge = -I, direction = missing)

Build `Jω::CurrentDensitySolution`, as above, but for `Jᵢⱼ(ω)` at a fixed `ω` and arbitrary
sites `i, j`. See also `greenfunction` for details on building a `GreenSlice` and
`GreenSolution`.

The local current density is defined here as ``Jᵢⱼ(ω) = (2/h) rᵢⱼ Re Tr[(Hᵢⱼgⱼᵢ(ω) -
gᵢⱼ(ω)Hⱼᵢ) * charge]``, with the integrated local current given by ``Jᵢⱼ = ∫ f(ω) Jᵢⱼ(ω)
dω``. Here `Hᵢⱼ` is the hopping from site `j` at `rⱼ` to `i` at `rᵢ`, `rᵢⱼ = rᵢ - rⱼ`,
`charge` is the charge of carriers in orbital space (see Keywords), and `gᵢⱼ(ω)` is the
retarded Green function between said sites.

## Keywords

- `charge` : for multiorbital sites, `charge` can be a general matrix, which allows to compute arbitrary currents, such as spin currents.
- `direction`: as defined above, `Jᵢⱼ(ω)` is a vector. If `direction` is `missing` the norm `|Jᵢⱼ(ω)|` is returned. If it is an `u::Union{SVector,Tuple}`, `u⋅Jᵢⱼ(ω)` is returned. If an `n::Integer`, `Jᵢⱼ(ω)[n]` is returned.

## Full evaluation

    Jω[sites...]
    Js(ω; params...)

Given a partially evaluated `Jω::CurrentDensitySolution` or `ρs::CurrentDensitySlice`, build
a sparse matrix `Jᵢⱼ(ω)` along the specified `direction` of fully evaluated local current
densities.

Note: Evaluating the current density returns a `SparseMatrixCSC` currently, instead of a
`OrbitalSliceMatrix`, since the latter is designed for dense arrays.

# Example

```
julia> # A semi-infinite 1D lead with a magnetic field `B`

julia> g = LP.square() |> supercell((1,0), region = r->-2<r[2]<2) |> hamiltonian(@hopping((r, dr; B = 0.1) -> cis(B * dr' * SA[r[2],-r[1]]))) |> greenfunction(GS.Schur(boundary = 0));

julia> J = current(g[cells = SA[1]])
CurrentDensitySlice{Float64} : current density at a fixed location and arbitrary energy
  charge      : LinearAlgebra.UniformScaling{Int64}(-1)
  direction   : missing

julia> J(0.2; B = 0.1)
3×3 SparseArrays.SparseMatrixCSC{Float64, Int64} with 4 stored entries:
  ⋅         0.0290138   ⋅
 0.0290138   ⋅         0.0290138
  ⋅         0.0290138   ⋅

julia> J(0.2; B = 0.0)
3×3 SparseArrays.SparseMatrixCSC{Float64, Int64} with 4 stored entries:
  ⋅           7.77156e-16   ⋅
 7.77156e-16   ⋅           5.55112e-16
  ⋅           5.55112e-16   ⋅
```

# See also
    `greenfunction`, `ldos`, `conductance`, `josephson`, `transmission`

"""
current

"""
    conductance(gs::GreenSlice; nambu = false)

Given a slice `gs = g[i::Integer, j::Integer]` of a `g::GreenFunction`, build a partially
evaluated object `G::Conductance` representing the zero-temperature, linear,
differential conductance `Gᵢⱼ = dIᵢ/dVⱼ` between contacts `i` and `j` at arbitrary bias `ω =
eV` in units of `e^2/h`. `Gᵢⱼ` is given by

      Gᵢⱼ =  e^2/h × Tr{[im*δᵢⱼ(gʳ-gᵃ)Γⁱ-gʳΓⁱgᵃΓʲ]}         (nambu = false)
      Gᵢⱼ =  e^2/h × Tr{[im*δᵢⱼ(gʳ-gᵃ)Γⁱτₑ-gʳΓⁱτ₃gᵃΓʲτₑ]}   (nambu = true)

Here `gʳ = g(ω)` and `gᵃ = (gʳ)' = g(ω')` are the retarded and advanced Green function of
the system, and `Γⁱ = im * (Σⁱ - Σⁱ')` is the decay rate at contact `i`. For Nambu systems
(`nambu = true`), the matrices `τₑ=[I 0; 0 0]` and `τ₃ = [I 0; 0 -I]` ensure that charge
reversal in Andreev reflections is properly taken into account. For normal systems (`nambu =
false`), the total current at finite bias and temperatures is given by ``Iᵢ = e/h × ∫
dω ∑ⱼ [fᵢ(ω) - fⱼ(ω)] Gᵢⱼ(ω)``, where ``fᵢ(ω)`` is the Fermi distribution in lead `i`.

## Keywords

- `nambu` : whether to consider the Hamiltonian of the system is written in a Nambu basis, each site containing `N` electron orbitals followed by `N` hole orbitals.

## Full evaluation

    G(ω; params...)

Compute the conductance at the specified contacts.

# Examples
```jldoctest
julia> # A central system g0 with two 1D leads and transparent contacts

julia> glead = LP.square() |> hamiltonian(hopping(1)) |> supercell((1,0), region = r->-2<r[2]<2) |> greenfunction(GS.Schur(boundary = 0));

julia> g0 = LP.square() |> hamiltonian(hopping(1)) |> supercell(region = r->-2<r[2]<2 && r[1]≈0) |> attach(glead, reverse = true) |> attach(glead) |> greenfunction;

julia> G = conductance(g0[1])
Conductance{Float64}: Zero-temperature conductance dIᵢ/dVⱼ from contacts i,j, in units of e^2/h
  Current contact  : 1
  Bias contact     : 1

julia> G(0.2) ≈ 3
true
```

# See also
    `greenfunction`, `ldos`, `current`, `josephson`, `transmission`

"""
conductance

"""
    transmission(gs::GreenSlice)

Given a slice `gs = g[i::Integer, j::Integer]` of a `g::GreenFunction`, build a partially
evaluated object `T::Transmission` representing the normal transmission probability `Tᵢⱼ(ω)`
from contact `j` to `i` at energy `ω`. It can be written as ``Tᵢⱼ = Tr{gʳΓⁱgᵃΓʲ}``. Here `gʳ
= g(ω)` and `gᵃ = (gʳ)' = g(ω')` are the retarded and advanced Green function of the system,
and `Γⁱ = im * (Σⁱ - Σⁱ')` is the decay rate at contact `i`

## Full evaluation

    T(ω; params...)

Compute the transmission `Tᵢⱼ(ω)` at a given `ω` and for the specified `params` of `g`.

# Examples

```jldoctest
julia> # A central system g0 with two 1D leads and transparent contacts

julia> glead = LP.square() |> hamiltonian(hopping(1)) |> supercell((1,0), region = r->-2<r[2]<2) |> greenfunction(GS.Schur(boundary = 0));

julia> g0 = LP.square() |> hamiltonian(hopping(1)) |> supercell(region = r->-2<r[2]<2 && r[1]≈0) |> attach(glead, reverse = true) |> attach(glead) |> greenfunction;

julia> T = transmission(g0[2, 1])
Transmission: total transmission between two different contacts
  From contact  : 1
  To contact    : 2

julia> T(0.2) ≈ 3   # The difference from 3 is due to the automatic `im*sqrt(eps(Float64))` added to `ω`
false

julia> T(0.2 + 1e-10im) ≈ 3
true
```

# See also
    `greenfunction`, `conductance`, `ldos`, `current`, `josephson`

"""
transmission

"""
    densitymatrix(gs::GreenSlice; opts...)

Compute a `ρ::DensityMatrix` at thermal equilibrium on sites encoded in `gs`. The actual
matrix for given system parameters `params`, and for a given chemical potential `mu` and
temperature `kBT` is obtained by calling `ρ(mu = 0, kBT = 0; params...)`. The algorithm used is
specialized for the GreenSolver used, if available. In this case, `opts` are options for
said algorithm.

    densitymatrix(gs::GreenSlice, (ωmin, ωmax); opts..., quadgk_opts...)
    densitymatrix(gs::GreenSlice, ωpoints; opts..., quadgk_opts...)

As above, but using a generic algorithm that relies on numerical integration along a contour
in the complex plane, between points `(ωmin, ωmax)` (or along a polygonal path connecting
`ωpoints`, a collection of numbers), which should be chosen so as to encompass the full
system bandwidth. When the `ωpoints` are all real, an extra point is added at `ω = µ` to the
integration path, to better integrate the step in the Fermi function. Keywords `quadgk_opts`
are passed to the `QuadGK.quadgk` integration routine. See below for additiona `opts`.

    densitymatrix(gs::GreenSlice, ωmax::Number; opts...)

As above with `ωmin = -ωmax`.

## Full evaluation

    ρ(μ = 0, kBT = 0; params...)   # where ρ::DensityMatrix

Evaluate the density matrix at chemical potential `μ` and temperature `kBT` (in the same
units as the Hamiltonian) for the given `g` parameters `params`, if any. The result is given
as an `OrbitalSliceMatrix`, see its docstring for further details.

If the generic integration algorithm is used with complex `ωpoints`, the following form is
also available:

    ρ(μ = 0, kBT = 0, override_path = missing; params...)

Here `override` can be a collection of `ωpoints` that will replace the original ones
provided when defining `ρ`. Alternatively, it can be a function that will be applied to the
original `ωpoints`. This may be useful when the integration path must depend on `params`.

## Algorithms and keywords

The generic integration algorithm allows for the following `opts` (see also `josephson`):

- `omegamap`: a function `ω -> (; params...)` that translates `ω` at each point in the integration contour to a set of system parameters. Useful for `ParametricHamiltonians` which include terms `Σ(ω)` that depend on a parameter `ω` (one would then use `omegamap = ω -> (; ω)`). Default: `ω -> (;)`, i.e. no mapped parameters.
- `imshift`: a small imaginary shift to add to the integration contour if all its vertices `ωpoints` are real numbers. Default: `missing` which is equivalent to `sqrt(eps)` for the relevant number type.
- `slope`: if `ωpoints` are all real numbers (typically encompassing the system's bandwidth), the integration contour is transformed into a triangular sawtooth path these points. Between each pair of points the path increases and then decreases linearly with the given `slope`. Default: `1.0`.
- `post`: a function to apply to the result of the integration. Default: `identity`.
- `callback`: a function to be called as `callback(x, y)` at each point in the integration, where `x` is the contour point and `y` is the integrand evaluated at that point. Useful for inspection and debugging, e.g. `callback(x, y) = @show x`. Default: `Returns(nothing)`.
- `atol`: absolute integration tolerance. The default `1e-7` is chosen to avoid excessive integration times when the current is actually zero. Default `1e-7`.

The `quadgk_opts` are extra keyword arguments (other than `atol`) to pass on to the function `QuadGK.quadgk` that is used for the integration.

Currently, the following GreenSolvers implement dedicated densitymatrix algorithms:

- `GS.Schur`: based on numerical integration over Bloch phase. Boundaries and non-empty contacts are not currently supported. Assumes Hermitian Hamiltonian. No `opts`.
- `GS.Spectrum`: based on summation occupation-weigthed eigenvectors. No `opts`.
- `GS.KPM`: based on the Chebyshev expansion of the Fermi function. Currently only works for zero temperature and only supports `nothing` contacts (see `attach`). No `opts`.

# Example
```
julia> g = HP.graphene(a0 = 1) |> supercell(region = RP.circle(10)) |> greenfunction(GS.Spectrum());

julia> ρ = densitymatrix(g[region = RP.circle(0.5)])
DensityMatrix: density matrix on specified sites with solver of type DensityMatrixSpectrumSolver

julia> ρ()  # with mu = kBT = 0 by default
2×2 OrbitalSliceMatrix{ComplexF64,Matrix{ComplexF64}}:
       0.5+0.0im  -0.262865+0.0im
 -0.262865+0.0im        0.5+0.0im
```

"""
densitymatrix

"""
    josephson(gs::GreenSlice, ωpoints; opts..., quadgk_opts...)

For a `gs = g[i::Integer]` slice of the `g::GreenFunction` of a hybrid junction, build a
`J::Josephson` object representing the equilibrium (static) Josephson current `I_J` flowing
into `g` through contact `i`, integrated along a polygonal contour connecting `ωpoints` (a
collection of numbers) in the complex plane. When the `ωpoints` are all real, an extra point
is added at `ω = 0` to the integration path, to better integrate the step in the Fermi
function.

The result of `I_J` is given in units of `qe/h` (`q` is the dimensionless carrier charge).
`I_J` can be written as ``I_J = Re ∫ dω f(ω) j(ω)``, where ``j(ω) = (qe/h) × 2Tr[(ΣʳᵢGʳ -
GʳΣʳᵢ)τz]``. Here `f(ω)` is the Fermi function with `µ = 0`.

    josephson(gs::GreenSlice, ωmax::Real; kw...)

As above, but with `ωpoints = (-ωmax, ωmax)`.

## Full evaluation

    J(kBT = 0, override_path = missing; params...)   # where J::Josephson

Evaluate the current `I_J` at chemical potemtial `µ = 0` and temperature `kBT` (in the same
units as the Hamiltonian) for the given `g` parameters `params`, if any.

It's possible to use `override_path` to override a complex integration path at evaluation
time. In this case `override_path` can be a collection of `ωpoints` that will replace the
original ones provided when defining `J`. Alternatively, it can be a function that will be
applied to the original `ωpoints`. This may be useful when the integration path must depend
on `params`.

## Keywords

The generic integration algorithm allows for the following `opts` (see also `densitymatrix`):

- `omegamap`: a function `ω -> (; params...)` that translates `ω` at each point in the integration contour to a set of system parameters. Useful for `ParametricHamiltonians` which include terms `Σ(ω)` that depend on a parameter `ω` (one would then use `omegamap = ω -> (; ω)`). Default: `ω -> (;)`, i.e. no mapped parameters.
- `phases` : collection of superconducting phase biases to apply to the contact, so as to efficiently compute the full current-phase relation `[I_J(ϕ) for ϕ in phases]`. Note that each phase bias `ϕ` is applied by a `[cis(-ϕ/2)*I 0*I; 0*I cis(ϕ/2)*I]` rotation to the self energy, which is almost free. If `missing`, a single `I_J` is returned.
- `imshift`: a small imaginary shift to add to the integration contour if all its vertices `ωpoints` are real numbers. Default: `missing` which is equivalent to `sqrt(eps)` for the relevant number type.
- `slope`: if `ωpoints`, are all real numbers (typically encompassing the system's bandwidth), the integration contour is transformed into a triangular sawtooth path these points. Between each pair of points the path increases and then decreases linearly with the given `slope`. Default: `1.0`.
- `post`: function to apply to the result of `∫ dω f(ω) j(ω)` to obtain the result, `post = real` by default.
- `callback`: a function to be called as `callback(x, y)` at each point in the integration, where `x` is the contour point and `y` is the integrand at that point. Useful for inspection and debugging, e.g. `callback(x, y) = @show x`. Default: `Returns(nothing)`.
- `atol`: absolute integration tolerance. The default `1e-7` is chosen to avoid excessive integration times when the current is actually zero. Default `1e-7`.

The `quadgk_opts` are extra keyword arguments (other than `atol`) to pass on to the function `QuadGK.quadgk` that is used for the integration.

## Note on analyticity

A non-zero `slope` parameter (as is the default) moves the integration path into the
upper-half complex-ω plane for increased performance. For this to work it's necessary that
the Green function and it's attached self-energies all be analytic in the upper half-plane
of complex ω. (Technically things will work also with independent analyticity in the
upper-left and upper-right quarter-planes, since the path passes 0 by default). However, no
check of analyticity is performed, so it is up to the user to ensure that. If this is not
possible, consider using `slope = 0`, or choosing a set of `ωpoints` that avoids
non-analyticities and cuts.

# Examples

```
julia> glead = LP.square() |> hamiltonian(@onsite((; ω = 0) -> 0.0005 * SA[0 1; 1 0] + im*ω*I) + hopping(SA[1 0; 0 -1]), orbitals = 2) |> supercell((1,0), region = r->-2<r[2]<2) |> greenfunction(GS.Schur(boundary = 0));

julia> g0 = LP.square() |> hamiltonian(hopping(SA[1 0; 0 -1]), orbitals = 2) |> supercell(region = r->-2<r[2]<2 && r[1]≈0) |> attach(glead, reverse = true) |> attach(glead) |> greenfunction;

julia> J = josephson(g0[1], 4; omegamap = ω -> (;ω), phases = subdiv(0, pi, 10))
Josephson: equilibrium Josephson current at a specific contact using solver of type JosephsonIntegratorSolver

julia> J(0.0)
10-element Vector{Float64}:
 7.060440509787806e-18
 0.0008178484258721882
 0.0016108816082772972
 0.002355033150366814
 0.0030277117620820513
 0.003608482493380227
 0.004079679643085058
 0.004426918320990192
 0.004639358112465513
 2.2618383948099795e-12
```

# See also
    `greenfunction`,`ldos`, `current`, `conductance`, `transmission`
"""
josephson

"""
    Quantica.integrand(J::Josephson{<:JosephsonIntegratorSolver}, kBT = 0)

Return the complex integrand `d::JosephsonIntegrand` whose integral over frequency yields the
Josephson current, `J(kBT) = post(∫dω d(ω))`, with `post = real`. To evaluate the `d` for a
given `ω` and parameters, use `d(ω; params...)`, or `call!(d, ω; params...)` for its
mutating (non-allocating) version.

    Quantica.integrand(ρ::DensityMatrix{<:DensityMatrixIntegratorSolver}, mu = 0, kBT = 0)

Like above for the density matrix `ρ(mu, kBT)`, with `d::DensityMatrixIntegrand` and `post =
Quantica.gf_to_rho!` that computes `-(GF-GF')/(2π*im)` in place for a matrix `GF`.

"""
integrand

"""
    Quantica.path(O::Josephson, args...)
    Quantica.path(O::DensityMatrix, args...)

Return the vertices of the polygonal integration path used to compute `O(args...)`.
"""
path


"""
    OrbitalSliceArray <: AbstractArray

A type of `AbstractArray` defined over a set of orbitals (see also `orbaxes`). It wraps a
regular array that can be obtained with `parent(::OrbitalSliceArray)`, and supports all the
general AbstractArray interface. In addition, it also supports indexing using
`siteselector`s and `cellindices`. `OrbitalSliceVector` and `OrbitalSliceMatrix` are special
cases of `OrbitalSliceArray` of dimension 1 and 2 respectively.

This is the common output type produced by `GreenFunctions` and most observables.

Note that for `m::OrbitalSliceMatrix`, `mat[i]` is equivalent to `mat[i,i]`, and `mat[;
sel...]` is equivalent to `mat[(; sel...), (; sel...)]`.

# `siteselector` indexing

    mat[(; rowsites...), (; colsites...)]
    mat[rowsel::SiteSelector, colsel::SiteSelector]

If we index an `OrbitalSliceMatrix` with `s::NamedTuple` or a `siteselector(; s...)`, we
obtain a new `OrbitalSliceMatrix` over the orbitals of the selected sites.

# `sites` indexing

    mat[sites(cell_index, site_indices)]
    mat[sites(row_cell_index, row_site_indices), sites(col_cell_index, col_site_indices)]

If we index an `OrbitalSliceMatrix` with `sites`, we obtain an unwrapped `Matrix` over
the sites with `site_indices` within cell with `cell_index`. Here `site_indices` can be
an `Int`, a container of `Int`, or a `:` (for all sites in the unit cell). If any of the
specified sites are not already in `orbaxes(mat)`, indexing will throw an error.

Note that in this case we do not obtain a new `OrbitalSliceMatrix`. This behavior is
required for performance, as re-wrapping in a new `OrbitalSliceMatrix` requires recomputing
and allocating the new `orbaxes`.

    view(mat, rows::CellSites, cols::Cellsites = rows)

Like the above, but returns a view instead of a copy of the indexed orbital matrix.

Note: `diagonal` indexing is currently not supported by `OrbitalSliceArray`.

# Examples

```
julia> g = LP.linear() |> hamiltonian(hopping(SA[0 1; 1 0]) + onsite(I), orbitals = 2) |> supercell(4) |> greenfunction;

julia> mat = g(0.2)[region = r -> 2<=r[1]<=4]
6×6 OrbitalSliceMatrix{ComplexF64,Matrix{ComplexF64}}:
 -1.93554e-9-0.545545im          0.0-0.0im               0.0-0.0im              -0.5+0.218218im          0.4+0.37097im           0.0+0.0im
         0.0-0.0im       -1.93554e-9-0.545545im         -0.5+0.218218im          0.0-0.0im               0.0+0.0im               0.4+0.37097im
         0.0-0.0im              -0.5+0.218218im  -1.93554e-9-0.545545im          0.0-0.0im               0.0+0.0im              -0.5+0.218218im
        -0.5+0.218218im          0.0-0.0im               0.0-0.0im       -1.93554e-9-0.545545im         -0.5+0.218218im          0.0+0.0im
         0.4+0.37097im           0.0+0.0im               0.0+0.0im              -0.5+0.218218im  -1.93554e-9-0.545545im          0.0-0.0im
         0.0+0.0im               0.4+0.37097im          -0.5+0.218218im          0.0+0.0im               0.0-0.0im       -1.93554e-9-0.545545im

julia> mat[(; cells = SA[1]), (; cells = SA[0])]
2×4 OrbitalSliceMatrix{ComplexF64,Matrix{ComplexF64}}:
 0.4+0.37097im  0.0+0.0im       0.0+0.0im       -0.5+0.218218im
 0.0+0.0im      0.4+0.37097im  -0.5+0.218218im   0.0+0.0im

julia> mat[sites(SA[1], 1)]
2×2 Matrix{ComplexF64}:
 -1.93554e-9-0.545545im          0.0-0.0im
         0.0-0.0im       -1.93554e-9-0.545545im
```

# See also
    `siteselector`, `cellindices`, `orbaxes`
"""
OrbitalSliceArray, OrbitalSliceVector, OrbitalSliceMatrix


"""
    orbaxes(A::OrbitalSliceArray)

Return the orbital axes of `A`. This is a tuple of `OrbitalSliceGrouped` objects that can be
used e.g. to index another `OrbitalSliceArray` or to inspect the indices of each site with
`siteindexdict`.

# Examples

```jldoctest
julia> g = HP.graphene(orbitals = 2) |> supercell((1,-1)) |> greenfunction;

julia> d = ldos(g[cells = SA[0]])(2); summary(d)
"8-element OrbitalSliceVector{Float64,Array}"

julia> a = only(orbaxes(d))
OrbitalSliceGrouped{Float64,2,1} : collection of subcells of orbitals (grouped by sites) for a 1D lattice in 2D space
  Cells       : 1
  Cell range  : ([0], [0])
  Total sites : 4

julia> siteindexdict(a)
4-element Dictionaries.Dictionary{Quantica.CellIndices{1, Int64, Quantica.SiteLike}, UnitRange{Int64}}
 CellSites{1,Int64} : 1 site in cell zero
  Sites : 1 │ 1:2
 CellSites{1,Int64} : 1 site in cell zero
  Sites : 2 │ 3:4
 CellSites{1,Int64} : 1 site in cell zero
  Sites : 3 │ 5:6
 CellSites{1,Int64} : 1 site in cell zero
  Sites : 4 │ 7:8
```

# See also
    `siteindexdict`
"""
orbaxes

"""
    siteindexdict(axis::OrbitalSliceGrouped)

Return a dictionary of `CellSite`s representing single sites in an orbital axis of an
`OrbitalSliceArray`, typically obtained with `orbaxes`. See `orbaxes` for an example.

# See also
    `orbaxes`, `OrbitalSliceArray`
"""
siteindexdict

"""
    serializer(T::Type, selectors...; parameter = :stream, encoder = identity, decoder = identity)

Construct a abstract `s::Serializer` object defines the rules to serialize and deserialize an
AbstractHamiltonian, i.e. to translate the matrix elements selected by `selectors`
(`SiteSelectors` or `HopSelectors`) into a 1D array of scalars of type `T`.

    serializer(T::Type; kw...)

Equivalent to `serializer(T, siteselector(), hopselector(); kw...)`, i.e. with all onsites
and hoppings included in selection.

    serializer(T::Type, h::AbstractHamiltonian, selectors...; kw...)

Applies a `Serializer` object like those above, and applies it to `h::AbstractHamiltonian`
to produce an `AppliedSerializer{T,<:AbstractHamiltonian}`, that can be used to serialize
and deserialize `h`. See `serialize` and `deserialize` for further details.

    serializer(h::AbstractHamiltonian{T}, selectors...; kw...)

Equivalent to `serializer(Complex{T}, h, selectors...; kw...)`.

## Keywords

- `parameter`: the parameter name used to address the serialized vector after transforming an `AppliedSerializer` into a `ParametricHamiltonian`, see below. Default: `:stream`.
- `encoder`: a function `s -> vec` that translates a single matrix element `s` into an collection `vec` of scalars of type `T`. Also supported is `encoder = (s->vec, (s,s´)->vec´)`, which applies the second function to hoppings `hᵢⱼ = s` and their adjoint `hⱼᵢ = s´` to encode both in a single collection `vec´` (onsites `hᵢᵢ` are still encoded using the first single-argument function). This is useful for Hermitian Hamiltonians, where `hⱼᵢ` can be derived from `hᵢⱼ`. Default: `identity`.
- `decoder`: the inverse function `vec -> s` of the encoder. If `encoder` is a tuple, `decoder` should also be a tuple of the inverse functions of each encoder function. Default: `identity`.

Note: for an `h::AbstractHamiltonian` with a non-uniform number of orbitals, the matrix
element passed to the `encoder` should always be assumed to be a square `SMatrix` of a fixed
size that can fit all sites, padded with zeros if necessary (see "Element type" when
displaying it, and the output of `h[unflat(dn)]`). Likewise, the decoder should return a
square `SMatrix` of the same size, or any other container that can be converted to one. The
latter is also important in cases with a uniform orbital number greater than one (non-scalar
element type).

The user can check that the encoder and decoder are mutual inverses with `Quantica.check(s;
params...)` where `params` is any choice of Hamiltonian parameters. This essentially checks
that `h(; params) == deserialize(s, serialize(s; params...); params...)` holds.

## Call syntax and ParametricHamiltonians

    as(; params...)

Transforms `as::AppliedSerializer{T,<:ParametricHamiltonian}` into an
`as´::AppliedSerializer{T,<:Hamiltonian}` where the enclosed `h` is replaced by `h´ = h(;
params...)`. The in-place (aliasing) version of the above is `Quantica.call!(as;
params...)`. Note that the enclosed `AbstractHamiltonian`s can be retrieved with e.g.
`Quantica.parent_hamiltonian(as)`.

    hamiltonian(as::AppliedSerializer)

Builds a `ph::ParametricHamiltonian` by adding `as` as a parametric modifier (similar to
`@onsite!` or `@hopping!`) to the `h::AbstractHamiltonian` enclosed in `as`. As a result,
`ph` acquires a new parameter of the name given by the `parameter` keyword specified
originally (`:stream` by default, see above). This parameter takes a serialized stream (e.g.
the output of `serialize(as)`) and replaces the corresponding elements in `ph`.

    h |> s

For `s::Serializer` and `h::AbstractHamiltonian`, converts `s` into an
`as::AppliedSerializer` by applying it to `h` and then adds `as` as a parametric modifier to
`h` to produce a `ph::ParametricHamiltonian`, as above. Note that `h |> as` with
`as::AppliedSerializer` is not allowed, since `as` can only be a modifier of its enclosed
`AbstractHamiltonian`.

## Examples

```
julia> h1 = LP.linear() |> hopping((r, dr) -> im*dr[1]) - @onsite((r; U = 2) -> U);

julia> as = serializer(Float64, h1; encoder = s -> reim(s), decoder = v -> complex(v[1], v[2]))
AppliedSerializer : translator between a selection of of matrix elements of an AbstractHamiltonian and a collection of scalars
  Object            : ParametricHamiltonian
  Object parameters : [:U]
  Stream parameter  : :stream
  Output eltype     : Float64
  Encoder/Decoder   : Single
  Length            : 6

julia> v = serialize(as; U = 4)
6-element Vector{Float64}:
 -4.0
  0.0
 -0.0
 -1.0
  0.0
  1.0

julia> h2 = deserialize!(as, v);

julia> h2 == h1(U = 4)
true

julia> h3 = hamiltonian(as)
ParametricHamiltonian{Float64,1,1}: Parametric Hamiltonian on a 1D Lattice in 1D space
  Bloch harmonics  : 3
  Harmonic size    : 1 × 1
  Orbitals         : [1]
  Element type     : scalar (ComplexF64)
  Onsites          : 1
  Hoppings         : 2
  Coordination     : 2.0
  Parameters       : [:U, :stream]

julia> h3(stream = v, U = 5) == h1(U = 4)  # stream overwrites the U=5 onsite terms
true
```

## See also
    `serialize`, `serialize!`, `deserialize`, `deserialize!`, `siteselector`, `hopselector`
"""
serializer

"""
    serialize(as::AppliedSerializer{T}; params...)

Construct a `Vector{T}` that encodes a selection of matrix elements of `h(; params...)`
where `h = Quantica.parent_hamiltonian(as)` is the `AbstractHamiltonian` used to build the
`AppliedSerializer`, see `serializer`.

    serialize(m::OrbitalSliceArray)

Return an `Array` of the same eltype as `m` that contains all the stored matrix elements of
`m`. See `deserialize` for the inverse operation.

    serialize(T::Type, m::OrbitalSliceArray)

Reinterpret `serialize(m)` as a collection with eltype `T`

## See also
    `serializer`, `serialize!`, `deserialize`, `deserialize!`

"""
serialize

"""
    serialize!(v, as::AppliedSerializer; params...)

Fill `v` in-place with the output of `serialize(as; params...)`, see `serialize` for
details.

## See also
    `serialize`, `serialize!`, `deserialize`, `deserialize!`
"""
serialize!

"""
    deserialize(as::AppliedSerializer, v; params...)

Construct `h(; params...)`, where `h = Quantica.parametric_hamiltonian(as)` is the
`AbstractHamiltonian` enclosed in `as`, with the matrix elements enconded in `v =
serialize(s)` restored (i.e. overwritten). See `serialize` for details.

    deserialize(m::OrbitalSliceArray, v)

Reconstruct an `OrbitalSliceArray` with the same structure as `m` but with the matrix
elements enconded in `v`. This `v` is typically the result of a `serialize` call to a
another similar `m`, but the only requirement is that is has the correct size. If `v` has
the wrong eltype, it will be reintepreted to match the eltype of `m`.

## See also
    `serializer`, `serialize`, `serialize!`, `deserialize!`
"""
deserialize

"""
    deserialize!(as::AppliedSerializer, v; params...)

In-place version of `deserialize`. It returns `h´ = Quantica.call!(h; params...)` with
serialised elements `v` restored (i.e. overwritten). Here `h =
Quantica.parent_hamiltonian(s)` is the `AbstractHamiltonian` used to construct `as`. The
resulting `h´::Hamiltonian` is not an independent copy, but is aliased with `h`.

## Examples

```
julia> h = HP.graphene() |> supercell(2);

julia> s = serializer(h)
AppliedSerializer : translator between a selection of of matrix elements of an AbstractHamiltonian and a collection of scalars
  Object            : Hamiltonian
  Object parameters : none
  Stream parameter  : :stream
  Output eltype     : ComplexF64
  Encoder/Decoder   : Single
  Length            : 24

julia> h === deserialize!(s, serialize(s))
true

julia> h === deserialize(s, serialize(s))
false

julia> h == deserialize(s, serialize(s))
true
```

## See also
    `serializer`, `serialize`, `serialize!`, `deserialize`

"""
deserialize!

"""
    Quantica.gaps(h::Hamiltonian1D{T}, µ = 0; atol = eps(T), kw...)

Compute the energy gaps of a 1D Hamiltonian `h` at chemical potential `µ`. The result is a
`Vector{T}` of the local minima of the `|ϵ(ϕ) - µ|`, where `ϵ(ϕ)` is the energy band closest
to `µ` and `ϕ ∈ [-π,π]` is the Bloch phase. The `atol` parameter is the absolute tolerance
used to determine the local minima versus `ϕ`, which are computed using the `Schur` solver
for 1D Hamiltonians. The keywords `kw` are passed to the ArnoldiMethod `partialschur!`
eigensolver (`kw = (; nev = 1)` by default).

The `LinearMaps` and `ArnoldiMethod` packages must be loaded to enable this functionality.

# Examples

```
julia> using LinearMaps, ArnoldiMethod

julia> h = LP.linear() |> supercell(4) |> hopping(1) - @onsite((r; U = 2) -> ifelse(iseven(r[1]), U, -U));

julia> Quantica.gaps(h(U=2))
2-element Vector{Float64}:
 1.9999999999999996
 1.9999999999999991
```

# See also:
    `Quantica.gap`, `Quantica.decay_lengths`

"""
gaps

"""
    Quantica.gap(h::Hamiltonian1D{T}, µ = 0; atol = eps(T), kw...)

Compute the minimal gap around `µ`, see `Quantica.gaps`

# See also:
    `Quantica.gaps`, `Quantica.decay_lengths`

"""
gap

"""
    Quantica.decay_lengths(g::GreenFunctionSchurLead, µ = 0; reverse = false)
    Quantica.decay_lengths(h::AbstractHamiltonian1D, µ = 0; reverse = false)

Compute the decay lengths of evanescent modes of a 1D `AbstractHamiltonian` `h` or a 1D
`GreenFunction` `g` using the `GS.Schur` solver. The modes decaying towards positive
direction (relative to the Bravais vector) are used, unless `reverse = true`.

# Examples

```
julia> h = LP.linear() |> supercell(4) |> hopping(1) - @onsite((r; U = 2) -> ifelse(iseven(r[1]), U, -U));

julia> Quantica.decay_lengths(h(U=2))
1-element Vector{Float64}:
 0.28364816427662776
```

# See also:
    `Quantica.gaps`

"""
decay_lengths

"""
    meanfield(g::GreenFunction, args...; kw...)

Build a `M::MeanField` object that can be used to compute the Hartree-Fock-Bogoliubov mean
field `Φ` between selected sites interacting through a given charge-charge potential. The
density matrix used to build the mean field is obtained with
`densitymatrix(g[pair_selection], args...; kw...)`, see `densitymatrix` for details.

The mean field between site `i` and `j` is defined as `Φᵢⱼ = δᵢⱼ hartreeᵢ + fockᵢⱼ`, where

    hartreeᵢ = ν * Q * Σ_k v_H(r_i-r_k) * tr(ρ[k,k]*Q)
    fockᵢⱼ  = -v_F(r_i-r_j) * Q * ρ[i,j] * Q

Here `Q` is the charge operator, `v_H` and `v_F` are Hartree and Fock
interaction potentials, and `ρ` is the density matrix evaluated at specific chemical
potential and temperature. Also `ν = ifelse(nambu, 0.5, 1.0)`, and `v_F(0) = v_H(0) = U`,
where `U` is the onsite interaction.

## Keywords

- `potential`: charge-charge potential to use for both Hartree and Fock. Can be a number or a function of position. Default: `1`
- `hartree`: charge-charge potential `v_H` for the Hartree mean field. Can be a number or a function of position. Overrides `potential`. Default: `potential`
- `fock`: charge-charge potential `v_F` for the Fock mean field. Can be a number, a function of position or `nothing`. In the latter case all Fock terms (even onsite) will be dropped. Default: `hartree`
- `onsite`: charge-charge onsite potential. Overrides both Hartree and Fock potentials for onsite interactions. Default: `hartree(0)`
- `charge`: a number (in single-orbital systems) or a matrix (in multi-orbital systems) representing the charge operator on each site. Default: `I`
- `nambu::Bool`: specifies whether the model is defined in Nambu space. In such case, `charge` should also be in Nambu space, typically `SA[1 0; 0 -1]` or similar. Default: `false`
- `namburotation::Bool`: if `nambu == true` and spinful systems, specifies whether the spinor basis is `[c↑, c↓, c↓⁺, -c↑⁺]` (`namburotation = true`) or `[c↑, c↓, c↑⁺, c↓⁺]` (`namburotation = false`). Default: `false`
- `selector::NamedTuple`: a collection of `hopselector` directives that defines the pairs of sites (`pair_selection` above) that interact through the charge-charge potential. Default: `(; range = 0)` (i.e. onsite)

Any additional keywords `kw` are passed to the `densitymatrix` function used to compute the
mean field, see above

## Evaluation and Indexing

    M(µ = 0, kBT = 0; params...)    # where M::MeanField

Build an `Φ::CompressedOrbitalMatrix`, which is a special form of `OrbitalSliceMatrix` that
can be indexed at pairs of individual sites, e.g. `ϕ[sites(2), sites(1)]` to return an
`SMatrix`. This type of matrix is less flexible than `OrbitalSliceMatrix` but is fully
static, and can encode symmetries. Its features are implementation details and are bound to
change. The returned `Φ` is just meant to be used in non-spatial models, see Examples below.

# Examples

```jldoctest
julia> model = hopping(I) - @onsite((i; phi = zerofield) --> phi[i]);  # see zerofield docstring

julia> g = LP.honeycomb() |> hamiltonian(model, orbitals = 2) |> supercell((1,-1)) |> greenfunction;

julia> M = meanfield(g; selector = (; range = 1), charge = I, potential = 0.05)
MeanField{SMatrix{2, 2, ComplexF64, 4}} : builder of Hartree-Fock-Bogoliubov mean fields
  Charge type      : 2 × 2 blocks (ComplexF64)
  Hartree pairs    : 14
  Mean field pairs : 28
  Nambu            : false

julia> phi0 = M(0.2, 0.3);

julia> phi0[sites(1), sites(2)] |> Quantica.chopsmall
2×2 SMatrix{2, 2, ComplexF64, 4} with indices SOneTo(2)×SOneTo(2):
 0.00109527+0.0im         0.0+0.0im
        0.0+0.0im  0.00109527+0.0im

julia> phi0[sites(1)] |> Quantica.chopsmall
2×2 SMatrix{2, 2, ComplexF64, 4} with indices SOneTo(2)×SOneTo(2):
 0.296672+0.0im       0.0+0.0im
      0.0+0.0im  0.296672+0.0im

julia> phi1 = M(0.2, 0.3; phi = phi0);

julia> phi1[sites(1), sites(2)] |> Quantica.chopsmall
2×2 SMatrix{2, 2, ComplexF64, 4} with indices SOneTo(2)×SOneTo(2):
 0.00307712+0.0im         0.0+0.0im
        0.0+0.0im  0.00307712+0.0im
```

# See also
    `zerofield`, `densitymatrix`
"""
meanfield

"""
        zerofield

An sigleton of type `ZeroField` that represents a zero-valued field. It has the property
that it returns zero no matter how it is indexed (`zerofield[inds...] = 0.0 * I`), so it is
useful as a default value in a non-spatial model involving mean fields. See `meanfield` for
a usage example.

# See also
    `meanfield`

"""
zerofield
