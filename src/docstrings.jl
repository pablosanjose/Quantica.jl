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
    `RegionPresets`, `HamiltonianPresets`
"""
LatticePresets

"""
`HamiltonianPresets` is a Quantica submodule containing several pre-defined Hamiltonians.
The alias `HP` can be used in place of `LatticePresets`. Currently supported hamiltonians
are

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
    `LatticePresets`, `RegionPresets`

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
    `LatticePresets`, `HamiltonianPresets`
"""
RegionPresets

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

Indexing into a lattice `lat` with keywords returns `LatticeSlice` representing a finite
collection of sites selected by `siteselector(; kw...)`. See `siteselector` for details on
possible `kw`, and `sites` to obtain site positions.

    lat[]

Special case equivalent to `lat[cells = (0,...)]` that returns a `LatticeSlice` of the
zero-th unitcell.

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
    transform(lat_or_h::Union{Lattice,AbstractHamiltonian}, f::Function)

Build a new lattice or hamiltonian transforming each site positions `r` into `f(r)`.

## Currying

    x |> transform(f::Function)

Curried version of `transform`, equivalent to `transform(f, x)`

    Note: Unexported `Quantica.transform!` is also available for in-place transforms. Use with care, as aliasing (i.e. several objects sharing the modified one) can produce unexpected results.

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
    `translate`
"""
transform

"""
    translate(lat::Lattice, δr)

Build a new lattice translating each site positions from `r` to `r + δr`, where `δr` can be
a `NTuple` or an `SVector` in embedding space.

## Currying

    x |> translate(δr)

Curried version of `translate`, equivalent to `translate(x, δr)`

    Note: Unexported `Quantica.translate!` is also available for in-place translations. Use with care, as aliasing (i.e. several objects sharing the modified one) can produce unexpected results.

# Examples

```jldoctest
julia> LatticePresets.square() |> translate((3,3)) |> sites
1-element Vector{SVector{2, Float64}}:
 [3.0, 3.0]

```

# See also
    `transform`

"""
translate

"""
    combine(lats::Lattice...)

If all `lats` have compatible Bravais vectors, combine them into a single lattice.
If necessary, sublattice names are renamed to remain unique.

    combine(hams::AbstractHamiltonians...; coupling = TighbindingModel())

Combine a collection `hams` of hamiltonians into one by combining their corresponding
lattices, and optionally by adding a coupling between them, given by the hopping terms in
`coupling`.

    Note that the `coupling` model will be applied to the combined lattice (which may have renamed sublattices to avoid name collissions). However, only hopping terms between different `hams` blocks will be applied. 

# Examples
```jldoctest
julia> # Building Bernal-stacked bilayer graphene

julia> hbot = HP.graphene(a0 = 1, dim = 3); htop = translate(hbot, (0, 1/√3, 1/√3));

julia> h2 = combine(hbot, htop; coupling = hopping(1, sublats = :B => :C, plusadjoint = true))
┌ Warning: Renamed repeated sublattice :A to :C
└ @ Quantica ~/.julia/dev/Quantica/src/types.jl:60
┌ Warning: Renamed repeated sublattice :B to :D
└ @ Quantica ~/.julia/dev/Quantica/src/types.jl:60
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
- If `cells` is a collection of `NTuple`s, they will be converted to `SVector`s.
- If `cells` is a boolean function, `n in cells` will be the result of `cells(n)`
- If `cells` is an `Integer`, it will include all cells with `n[i] in 0:cells-1`

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

Create a `Hamiltonian` or `ParametricHamiltonian` by applying `model` to the lattice `lat`
(see `onsite`, `@onsite`, `hopping` and `@hopping` for details on building tight-binding
models).

    hamiltonian(lat::Lattice, model, modifiers...; orbitals = 1)
    hamiltonian(h::AbstractHamiltonian, modifiers...; orbitals = 1)

Create a `ParametricHamiltonian` where all onsite and hopping terms in `model` can be
parametrically modified through the provided `modifiers` (see `@onsite!` and `@hopping!` for
details on defining modifiers).

## Keywords

- `orbitals`: number of orbitals per sublattice. If an `Integer` (or a `Val{Integer}`), all sublattices will have the same number of orbitals. A collection of `Integers` indicates the orbitals on each sublattice.

## Currying

    lat |> hamiltonian(model[, modifiers...]; kw...)

Curried form of `hamiltonian` equivalent to `hamiltonian(lat, model, modifiers...; kw...)`.

## Indexing

    h[dn::SVector{L,Int}]
    h[dn::NTuple{L,Int}]

Return the Bloch harmonic of an `h::AbstractHamiltonian` in the form of a
`HybridSparseMatrix`. This special matrix type contains both an `unflat` sparse
representation of the harmonic with one site per element, and a `flat` representation with
one orbital per element. To obtain each of these, use `unflat(h[dn])` and `flat(h[dn])`.

    h[()]

Special syntax equivalent to `h[(0...)]`, which access the fundamental Bloch harmonic.

## Call syntax

    ph(; params...)

Return a `h::Hamiltonian` from a `ph::ParametricHamiltonian` by applying specific values to
its parameters `params`. If `ph` is a non-parametric `Hamiltonian` instead, this is a no-op.

    h(ϕ...; params...)

Return the flat, sparse Bloch matrix of `h::AbstractHamiltonian` at Bloch phases `ϕ` (with
applied parameters `params` if `h` is a `ParametricHamiltonian`), defined as `H = ∑_dn
e⁻ⁱᵠᵈⁿ H_dn`, where `H_dn = flat(h[dn])` is the `dn` flat Bloch harmonic of `h`

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

julia> h(0,0)
4×4 SparseArrays.SparseMatrixCSC{ComplexF64, Int64} with 8 stored entries:
     ⋅          ⋅      0.0+0.0im  3.0+0.0im
     ⋅          ⋅      3.0+0.0im  0.0+0.0im
 0.0+0.0im  3.0+0.0im      ⋅          ⋅    
 3.0+0.0im  0.0+0.0im      ⋅          ⋅    
```

# See also
    `lattice`, `onsite`, `hopping`, `@onsite`, `@hopping`, `@onsite!`, `@hopping!`, `ishermitian`
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

Build a tight-binding model representing a uniform or a position-dependent onsite
potential, respectively, on sites selected by `siteselector(; sites...)` (see `siteselector`
for details).

Site positions are `r::SVector{E}`, where `E` is the embedding dimension of the lattice. The
onsite potential `o` can be a `Number` (for single-orbital sites), a `UniformScaling` (e.g.
`2I`) or an `AbstractMatrix` (use `SMatrix` for performance) of dimensions matching the
number of orbitals in the selected sites. Models may be applied to a lattice `lat` to
produce a `Hamiltonian` with `hamiltonian(lat, model; ...)`, see `hamiltonian`. Position
dependent models are forced to preserve the periodicity of the lattice.

Models can be combined using `+`, `-` and `*`, e.g. `onsite(1) - 2 * hopping(1)`.

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
    hopping(t; plusadjoint = false, hops...)
    hopping((r, dr) -> t(r, dr); plusadjoint = false, hops...)

Build a tight-binding model representing a uniform or a position-dependent hopping
amplitude, respectively, on hops selected by `hopselector(; hops...)` (see `hopselector` for
details). If `plusadjoint = true`, the adjoint hopping term is added on the reverse `hops`.

Hops from a site at position `r₁` to another at `r₂` are described using the hop center `r =
(r₁ + r₂)/2` and the hop vector `dr = r₂ - r₁`. Hopping amplitudes `t` can be a `Number`
(for hops between single-orbital sites), a `UniformScaling` (e.g. `2I`) or an
`AbstractMatrix` (use `SMatrix` for performance) of dimensions matching the number of
orbitals in the selected sites. Models may be applied to a lattice `lat` to produce an
`Hamiltonian` with `hamiltonian(lat, model; ...)`, see `hamiltonian`. Position dependent
models are forced to preserve the periodicity of the lattice.

Models can be combined using `+`, `-` and `*`, e.g. `onsite(1) - 2 * hopping(1)`.

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
    `onsite`, `@onsite`, `@hopping`, `@onsite!`, `@hopping!`, `hamiltonian`
"""
hopping

"""
    @onsite((; params...) -> o(; params...); sites...)
    @onsite((r; params...) -> o(r; params...); sites...)

Build a parametric tight-binding model representing a uniform or a position-dependent
onsite potential, respectively, on sites selected by `siteselector(; sites...)` (see
`siteselector` for details).

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

Parametric models can be combined with other regular or parametric models using `+`, `-` and
`*`, e.g. `onsite(1) - 2 * hopping(1)`. The combined parametric models can share parameters.

    @onsite((ω; params...) -> Σᵢᵢ(ω; params...); sites...)
    @onsite((ω, r; params...) -> Σᵢᵢ(ω, r; params...); sites...)

Special form of a parametric onsite potential meant to model a self-energy (see `attach`).

# Examples
```jldoctest
julia> model = @onsite((r; dμ = 0) -> (r[1] + dμ) * I; sublats = :A) + @onsite((; dμ = 0) -> - dμ * I; sublats = :B)
ParametricModel: model with 2 terms
  ParametricOnsiteTerm{ParametricFunction{1}}
    Region            : any
    Sublattices       : A
    Cells             : any
    Coefficient       : 1
    Parameters        : [:dμ]
  ParametricOnsiteTerm{ParametricFunction{0}}
    Region            : any
    Sublattices       : B
    Cells             : any
    Coefficient       : 1
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
    `onsite`, `hopping`, `@hopping`, `@onsite!`, `@hopping!`, `attach`, `hamiltonian`
"""
macro onsite end

"""
    @hopping((; params...) -> t(; params...); hops...)
    @hopping((r, dr; params...) -> t(r; params...); hops...)

Build a parametric tight-binding model representing a uniform or a position-dependent
hopping amplitude, respectively, on hops selected by `hopselector(; hops...)` (see
`hopselector` for details).

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

Parametric models can be combined with other regular or parametric models using `+`, `-` and
`*`, e.g. `onsite(1) - 2 * hopping(1)`. The combined parametric models can share parameters.

    @hopping((ω; params...) -> Σᵢⱼ(ω; params...); hops...)
    @hopping((ω, r, dr; params...) -> Σᵢⱼ(ω, r, dr; params...); hops...)

Special form of a parametric hopping amplitude meant to model a self-energy (see `attach`).

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
    `onsite`, `hopping`, `@onsite`, `@onsite!`, `@hopping!`, `attach`, `hamiltonian`
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

# Examples
```jldoctest
julia> model = onsite(0); disorder = @onsite!((o; W = 0) -> o + W * rand())
OnsiteModifier{ParametricFunction{1}}:
  Region            : any
  Sublattices       : any
  Cells             : any
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
    `onsite`, `hopping`, `@onsite`, `@hopping`, `@hopping!`, `hamiltonian`
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

# Examples
```jldoctest
julia> model = hopping(1); peierls = @hopping!((t, r, dr; A = r -> SA[0,0]) -> t * cis(-dr' * A(r)))
OnsiteModifier{ParametricFunction{3}}:
  Region            : any
  Sublattice pairs  : any
  Cell distances    : any
  Hopping range     : Neighbors(1)
  Reverse hops      : false
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
    `onsite`, `hopping`, `@onsite`, `@hopping`, `@onsite!`, `hamiltonian`
"""
macro hopping! end

"""

    wrap(h::AbstractHamiltonian, (ϕ₁, ϕ₂,...))

For an `h` of lattice dimension `L` and a set of `L` Bloch phases `ϕ = (ϕ₁, ϕ₂,...)`,
contruct a new zero-dimensional `h´::AbstractHamiltonian` for all Bravais vectors have been
eliminated by wrapping the lattice onto itself along the corresponding Bravais vector.
Intercell hoppings along wrapped directions will pick up a Bloch phase `exp(-iϕ⋅dn)`.

If a number `L´` of phases `ϕᵢ` are `:` instead of numbers, the corresponding Bravais
vectors will not be wrapped, and the resulting `h´` will have a finite lattice dimension
`L´`.

# Examples

```jldoctest
julia> h2D = HP.graphene(); h1D = wrap(h2D, (:, 0.2))
Hamiltonian{Float64,2,1}: Hamiltonian on a 1D Lattice in 2D space
  Bloch harmonics  : 3
  Harmonic size    : 2 × 2
  Orbitals         : [1, 1]
  Element type     : scalar (ComplexF64)
  Onsites          : 0
  Hoppings         : 4
  Coordination     : 2.0

julia> h2D(0.3, 0.2) ≈ h1D(0.3)
true
```

# See also
    `hamiltonian`, `supercell`
"""
wrap

"""
    flat(m::HybridSparseMatrix)

Return a flat sparse version of `m`, with each element corresponding to a single orbital.
The argument `m` is a Bloch harmonic of an `h::AbstractHamiltonian`, obtained with the
syntax `h[dn]`, see `hamiltonian`.

# Examples

```jldoctest
julia> h = HP.graphene(orbitals = 2); flat(h[(0,0)])
4×4 SparseArrays.SparseMatrixCSC{ComplexF64, Int64} with 8 stored entries:
     ⋅          ⋅      2.7+0.0im  0.0+0.0im
     ⋅          ⋅      0.0+0.0im  2.7+0.0im
 2.7+0.0im  0.0+0.0im      ⋅          ⋅    
 0.0+0.0im  2.7+0.0im      ⋅          ⋅    
```
"""
flat

"""
    unflat(m::HybridSparseMatrix)

Return an unflat sparse version of `m`, with each element corresponding to a single site.
The argument `m` is a Bloch harmonic of an `h::AbstractHamiltonian`, obtained with the
syntax `h[dn]`, see `hamiltonian`.

# Examples

```jldoctest
julia> h = HP.graphene(orbitals = 2); unflat(h[(0,0)])
2×2 SparseArrays.SparseMatrixCSC{SMatrix{2, 2, ComplexF64, 4}, Int64} with 2 stored entries:
                     ⋅                       [2.7+0.0im 0.0+0.0im; 0.0+0.0im 2.7+0.0im]
 [2.7+0.0im 0.0+0.0im; 0.0+0.0im 2.7+0.0im]                      ⋅                     
```
"""
unflat

"""
`EigenSolvers` is a Quantica submodule containing several pre-defined eigensolvers. The
alias `ES` can be used in place of `EigenSolvers`. Currently supported solvers are

    ES.LinearAlgebra(; kw...)       # Uses `eigen(mat; kw...)` from the `LinearAlgebra` package
    ES.Arpack(; kw...)              # Uses `eigs(mat; kw...)` from the `Arpack` package
    ES.KrylovKit(params...; kw...)  # Uses `eigsolve(mat, params...; kw...)` from the `KrylovKit` package
    ES.ArnoldiMethod(; kw...)       # Uses `partialschur(mat; kw...)` from the `ArnoldiMethod` package

Additionally, to compute interior eigenvalues, we can use a shift-invert method around
energy `ϵ0` (uses `LinearMaps` and a `LinearSolve.lu` factorization), combined with any
solver `s` from the list above:

    ES.ShiftInvert(s, ϵ0)           # Perform a lu-based shift-invert with solver `s`

If the required packages are not already available, they will be automatically loaded when
calling these solvers.

# Examples

```jldoctest
julia> h = HP.graphene(t0 = 1) |> supercell(10);

julia> spectrum(h, (0,0), ES.ShiftInvert(ES.ArnoldiMethod(nev = 4), 0.0)) |> energies
4-element Vector{ComplexF64}:
 -0.38196601125010465 + 3.686368662666227e-16im
  -0.6180339887498938 + 6.015655020129746e-17im
   0.6180339887498927 + 2.6478518218421853e-16im
  0.38196601125010476 - 1.741261108320361e-16im
```

# See also
    `spectrum`, `bands`
"""
EigenSolvers

"""
    spectrum(h::AbstractHamiltonian, ϕs[, solver = EigenSolvers.LinearAlgebra()]; params...)

Computes the eigenspectrum of the Bloch matrix `h(ϕs; params...)` using the specified
eigensolver. See `EigenSolvers` for available solvers and their options.

## Indexing and destructuring

Eigenenergies `ϵs::Tuple` and eigenstates `ψs::Matrix` can be extracted from a spectrum `sp`
using either of the following

    ϵs, ψs = sp
    ϵs = first(sp)
    ϵs = energies(sp)
    ψs = last(sp)
    ψs = states(sp)

In addition, one can extract the `n` eigenpairs closest to a given energy `ϵ₀` with

    ϵs, ψs = sp[1:n, around = ϵ₀]

More generally, `sp[inds, around = ϵ₀]` will take the eigenpairs at position given by `inds`
after sorting by increasing distance to `ϵ₀`. If `around` is omitted, the ordering in `sp`
is used.

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

Construct continuously connected bands of `h` by diagonalizing the matrix `h(ϕs; params...)`
on an `M`-dimensional mesh of points `(x₁, x₂, ..., xₘ)` where each `xᵢ` takes values in the
collection `xcolᵢ`. The mapping between points in the mesh points and values of `(ϕs;
params...)` is defined by keyword `mapping` (`identity` by default, see Keywords).
Diagonalization is multithreaded and will use all available Julia threads (start session
with `julia -t N` to have `N` threads).

## Keywords

- `solver`: eigensolver to use for each diagonalization (see `Eigensolvers`). Default: `ES.LinearAlgebra()`
- `mapping`: a function of the form `(x, y, ...) -> ϕs` or `(x, y, ...) -> ftuple(ϕs...; params...)` that translates points `(x, y, ...)` in the mesh to Bloch phases `ϕs` or phase+parameter FrankenTuples `(ϕs; params...)`. See also linecuts below. Default: `identity`
- `transform`: function to apply to each eigenvalue after diagonalization. Default: `identity`
- `degtol::Real`: maximum distance between to nearby eigenvalue so that they are classified as degenerate. Default: `sqrt(eps)`
- `split::Bool`: whether to split bands into disconnected subbands. Default: `true`
- `defects`: a collection of extra points to add to the mesh, typically the location of topological band defects such as Dirac points, so that interpolation avoids creating dislocation defects in the bands.
- `patches::Integer`: (experimental) if a dislocation is encountered, attempt to patch it by searching for the defect recursively to a given order. Default: `0`
- `warn::Bool`: whether to emit warning when band dislocations are encountered
- `showprogress::Bool`: whether to show or not a progress bar. Default: `true`

## Band linecuts

To do a linecut of a bandstructure along a polygonal path in the `L`-dimensional Brillouin
zone, mapping a set of 1D points `xs` to a set of `nodes`, with `pts` interpolation points
in each segment, one can use the following convenient syntax

    bands(h, subdiv(xs, pts); mapping = (xs => nodes))

Here `nodes` can be a collection of `SVector{L}` or of named Brillouin zone points from the
list (`:Γ`,`:K`, `:K´`, `:M`, `:X`, `:Y`, `:Z`). If `mapping = nodes`, then `xs` defaults to
`0:length(nodes)-1`. See also `subdiv` for its alternative methods.

## Indexing

    b[i]

Extract `i`-th subband from `b::Bands`. `i` can also be a `Vector`, an `AbstractRange` or
any other argument accepted by `getindex(subbands::Vector, i)`

    b[slice::Tuple]

Compute a section of `b::Bands` with a "plane" defined by `slice = (ϕ₁, ϕ₂,..., ϕₗ, ϵ)`,
where each `ϕᵢ` or `ϵ` can be a real number (representing a fixed momentum or energy) or a
`:` (unconstrained along that dimension). For bands of an `L`-dimensional lattice, `slice`
must be a tuple of length `L+1`. The result is a collection of of sliced `Subband`s.

# Examples

```jldoctest
julia> phis = range(0, 2pi, length = 50); h = LP.honeycomb() |> hamiltonian(@hopping((; t = 1) -> t));

julia> bands(h(t = 1), phis, phis)
Bands{Float64,3,2}: 3D Bands over a 2-dimensional parameter space of type Float64
  Subbands  : 1
  Vertices  : 5000
  Edges     : 14602
  Simplices : 9588

julia> bands(h, phis, phis; mapping = (x, y) -> ftuple(0, x; t = y/2π))
Bands{Float64,3,2}: 3D Bands over a 2-dimensional parameter space of type Float64
  Subbands  : 1
  Vertices  : 4950
  Edges     : 14553
  Simplices : 9604

julia> bands(h(t = 1), subdiv((0, 2, 3), (20, 30)); mapping = (0, 2, 3) => (:Γ, :M, :K))
Bands{Float64,2,1}: 2D Bands over a 1-dimensional parameter space of type Float64
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

    attach(h, gs::GreenSlice, coupling::AbstractModel; sites...)

Adds a generic self-energy `Σ(ω) = V´⋅gs(ω)⋅V` on `h`'s `sites`, where `V` and `V´` are
couplings, given by `coupling`, between said `sites` and the `LatticeSlice` in `gs`. Allowed
forms of `gs` include both `g[bath_sites...]` and `g[contactind::Integer]`m where `g` is any
`GreenFunction`.

    attach(h, model::ParametricModel; sites...)

Add self-energy `Σᵢⱼ(ω)` defined by a `model` composed of parametric terms (`@onsite` and
`@hopping`) with `ω` as first argument, as in e.g. `@onsite((ω, r) -> Σᵢᵢ(ω, r))` and
`@hopping((ω, r, dr) -> Σᵢⱼ(ω, r, dr))`

    attach(h, nothing; sites...)

Add null self-energy `Σᵢⱼ(ω) = 0` on selected sites, which in effect simply amounts to
defining a contact on said sites, but does not lead to any dressing the Green function. This
is useful for some `GreenFunction` solvers such as `GS.KPM` (see `greenfunction`), which
need to know the sites of interest beforehand (the contact sites in this case).

    attach(h, g1D::GreenFunction; reverse = false, transform = identity, sites...)

Add a self-energy `Σ(ω) = h₋₁⋅g1D(ω)[surface]⋅h₁` corresponding to a semi-infinite 1D lead
(i.e. with a finite `boundary`, see `greenfunction`), where `h₁` and `h₋₁` are intercell
couplings, and `g1D` is the lead `GreenFunction`. The `g1D(ω)` is taken at the `suface`
unitcell, either adjacent to the `boundary` on its positive side (if `reverse = false`) or
on its negative side (if `reverse = true`). The selected `sites` in `h` must match,
geometrically, those of the lead unit cell after applying `transform` to the latter. If they
don't, use the `attach` syntax below.

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

julia> g1D = LP.cubic() |> hamiltonian(hopping(1)) |> supercell((0,0,1), region = RP.square(4)) |> greenfunction(GS.Schur(boundary = 0));

julia> coupling = hopping(1, range = 2);

julia> gdisk = HP.graphene(a0 = 1, dim = 3) |> supercell(region = RP.circle(10)) |> attach(g1D, coupling; region = RP.square(4)) |> attach(g1D, coupling; region = RP.square(4), reverse = true) |> greenfunction;

```

# See also
    `greenfunction`, `GreenSolvers`

"""
attach

"""
    greenfunction(h::Union{AbstractHamiltonian,OpenHamiltonian}, solver::GreenSolver)

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

    g[sites...]
    g[siteselector(; sites...)]

Build a `gs::GreenSlice` that represents a Green function at arbitrary energy and parameter
values, but at specific sites on the lattice defined by `siteselector(; sites...)`

    g[contact_index::Integer]

Build a `GreenSlice` equivalent to `g[contact_sites...]`, where `contact_sites...`
correspond to sites in contact number `contact_index` (must have `1<= contact_index <=
number_of_contacts`). See `attach` for details on attaching contacts to a Hamiltonian.

    g[:]

Build a `GreenSlice` over all contacts.

    g[dst, src]

Build a `gs::GreenSlice` between sites specified by `src` and `dst`, which can take any of
the forms above. Therefore, all the previous slice forms correspond to a diagonal block
`g[i, i]`.

    g(ω; params...)

Build a `gω::GreenSolution` that represents a Green function at arbitrary points on the
lattice, but at fixed energy `ω` and system parameter values `param`. For most solvers, it
is required to add to `ω` with a small imaginary part, either positive (for the retarded) or
negative (for the advanced Green function).

    gω[i]
    gω[i, j]
    gs(ω; params...)

For any `gω::GreenSolution` or `gs::GreenSlice`, build the Green function matrix fully
evaluated at fixed energy, parameters and positions. The matrix is dense and has scalar
elements, so that any orbital structure on each site is flattened.

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

julia> gω = g(0.1 + 0.0001im; t = 2)
GreenSolution{Float64,2,0}: Green function at arbitrary positions, but at fixed energy

julia> gs = g[region = RP.circle(2), sublats = :B]
GreenSlice{Float64,2,0}: Green function at arbitrary energy, but at fixed lattice positions

julia> gω[region = RP.circle(2), sublats = :B] == gs(0.1 + 0.0001im; t = 2)
true
```

# See also
    `GreenSolvers`, `ldos`, `conductance`, `current`, `josephson`
"""

"""

`GreenSolvers` is a Quantica submodule containing several pre-defined Green function
solvers. The alias `GS` can be used in place of `GS`. Currently supported solvers and their
possible keyword arguments are

- `GS.SparseLU()` : Direct inversion solver for 0D Hamiltonians using a `SparseArrays.lu(hmat; kw...)` factorization
- `GS.Schur(; boundary = Inf)` : Solver for 1D Hamiltonians based on a deflated, generalized Schur factorization
    - `boundary` : 1D cell index of a boundary cell, or `Inf` for no boundaries. Equivalent to removing that specific cell from the lattice when computing the Green function.
- `GS.KPM(; order = 100, bandrange = missing)` : Kernel polynomial method solver for 0D Hamiltonians
    - `order` : order of the Chebyshev expansion (lowest is zero)
    - `bandrange` : a `(min_energy, max_energy)::Tuple` interval that encompasses the full band of the Hamiltonian. If `missing`, it is computed automatically.
    - This solver does not allow arbitrary indexing of the resulting `g::GreenFunction`, only on contacts `g[contact_ind::Integer]`. If the system has none, we can add a dummy contact using `attach(h, nothing; sites...)`, see `attach`.

## TODO

Still in the TODO list is a bandstructure/spectrum-based solver, valid for Hamiltonians of any dimension with and without boundaries

"""
GreenSolvers

"""
    ldos(gs::GreenSlice; kernel = I)

Build `ρs::LocalSpectralDensitySlice`, a partially evaluated object representing the local
density of states `ρᵢ(ω)` at specific sites `i` but at arbitrary energy `ω`.

    ldos(gω::GreenSolution; kernel = I)

Build `ρω::LocalSpectralDensitySolution`, as above, but for `ρᵢ(ω)` at fixed `ω` and
arbitrary sites `i`. See also `greenfunction` for details on building a `GreenSlice` and
`GreenSolution`.

The local density of states is defined here as ``ρᵢ(ω) = -Tr(gᵢᵢ(ω))/π``, where `gᵢᵢ(ω)` is
the retarded Green function at a given site `i`. Therefore a small imaginary part should be
added to `ω` when using a `gω::GreenSolution` to obtain a correct retarded result with most
Green solvers.

## Keywords

- `kernel` : for multiorbital sites, `kernel` allows to compute a generalized `ldos` `ρᵢ(ω) = -Tr(gᵢᵢ(ω) * kernel)/π`, where `gᵢᵢ(ω)` is the retarded Green function at site `i` and energy `ω`.

## Full evaluation

    ρω[sites...]
    ρs(ω; params...)

Given a partially evaluated `ρω::LocalSpectralDensitySolution` or
`ρs::LocalSpectralDensitySlice`, build a vector `[ρ₁(ω), ρ₂(ω)...]` of fully evaluated local
densities of states. If `ω` above is real, a small positive imaginary part will be added
internally for the evaluation.

# Example
```jldoctest
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

julia> ldos(g(0.2))[1]  # The KPM solver doesn't require an imag(ω) > 0 broadening
6-element Vector{Float64}:
 0.037505015417307
 0.03583857530882366
 0.03583857530882353
 0.035838575308823506
 0.03583857530882361
 0.037505015417306886
```

"""
ldos

"""
    conductance(gs::GreenSlice; nambu = false)

Given a slice `gs = g[i::Integer, j::Integer]` of a `g::GreenFunction`, build a partially
evaluated object `G::ConductanceSlice` representing the zero-temperature, linear,
differential conductance `Gᵢⱼ = dIᵢ/dVⱼ` between contacts `i` and `j` at arbitrary bias `ω =
eV` in units of `e^2/h`. `Gᵢⱼ` is given by

      ``Gᵢⱼ =  e^2/h × Tr{[δᵢⱼi(gʳ-gᵃ)Γⁱ-gʳΓⁱgᵃΓʲ]}``         (nambu = false)
      ``Gᵢⱼ =  e^2/h × Tr{[δᵢⱼi(gʳ-gᵃ)Γⁱτₑ-gʳΓⁱτ₃gᵃΓʲτₑ]}``   (nambu = true)

Here `gʳ = g(ω)` and `gᵃ = (gʳ)' = g(ω')` are the retarded and advanced Green function of
the system, and `Γⁱ = (Σⁱ - Σⁱ') * im` is the decay rate at contact `i`. For Nambu systems
(`nambu = true`), the matrices `τₑ=[I 0; 0 0]` and `τ₃ = [I 0; 0 -I]` ensure that charge
reversal in Andreev reflections is properly taken into account. For normal systems (`nambu =
false`), the total current at finite bias and temperatures is given by ``Iᵢ = e/h × ∫
dω ∑ⱼ [fᵢ(ω) - fⱼ(ω)] Gᵢⱼ(ω)``, where ``fᵢ(ω)`` is the Fermi distribution in lead `i`.

## Keywords

- `nambu` : whether to consider the Hamiltonian of the system is written in a Nambu basis, each site containing `N` electron orbitals followed by `N` hole orbitals.

## Full evaluation

    G(ω; params...)

Compute the conductance between the specified contacts. If `ω` is real, a small positive
imaginary part will be added internally for the evaluation.

```jldoctest
julia> # A central system g0 with two 1D leads and transparent contacts

julia> gl = LP.square() |> hamiltonian(hopping(1)) |> supercell((1,0), region = r->-2<r[2]<2) |> greenfunction(GS.Schur(boundary = 0));

julia> g0 = LP.square() |> hamiltonian(hopping(1)) |> supercell(region = r->-2<r[2]<2 && r[1]≈0) |> attach(gl, reverse = true) |> attach(gl) |> greenfunction;

julia> G = conductance(g0[1])
ConductanceSlice{Float64}: Zero-temperature conductance dIᵢ/dVⱼ from contacts i,j, in units of e^2/h
  Current contact  : 1
  Bias contact     : 1

julia> G(0.2)
2.999999999999999
```

# See also
    `greenfunction`, `ldos`, `current`, `josephson`

"""
conductance

"""
    current(gs::GreenSlice; charge = -I, direction = missing)

Build `Js::CurrentDensitySlice`, a partially evaluated object representing the equilibrium
local current density `Jᵢⱼ(ω)` at arbitrary energy `ω` from site `j` to site `i`, both taken
from a specific lattice slice. The current is computed along a given `direction` (see
Keywords).

    current(gω::GreenSolution; charge = -I, direction = missing)

Build `Jω::CurrentDensitySolution`, as above, but for `Jᵢⱼ(ω)` at fixed `ω` and arbitrary
sites `i, j`. See also `greenfunction` for details on building a `GreenSlice` and
`GreenSolution`.

The local current density is defined here as ``Jᵢⱼ(ω) = (2/h) rᵢⱼ Re Tr[(Hᵢⱼgⱼᵢ(ω) -
gᵢⱼ(ω)Hⱼᵢ) * charge]``, with the integrated local current given by ``Jᵢⱼ = ∫ f(ω) Jᵢⱼ(ω)
dω``. Here `Hᵢⱼ` is the hopping from site `j` at `rⱼ` to `i` at `rᵢ`, `rᵢⱼ = rᵢ - rⱼ`,
`charge` is the charge of carriers in orbital space (see Keywords), and `gᵢⱼ(ω)` is the
retarded Green function between said sites. Therefore a small imaginary part should be added
to `ω` when using a `gω::GreenSolution` to obtain a correct retarded result with most Green
solvers.

## Keywords

- `charge` : for multiorbital sites, `charge` can be a general matrix, which allows to compute arbitrary currents, such as spin currents.
- `direction`: as defined above, `Jᵢⱼ(ω)` is a vector. If `direction` is `missing` the norm `|Jᵢⱼ(ω)|` is returned. If it is an `u::Union{SVector,Tuple}`, `u⋅Jᵢⱼ(ω)` is returned. If an `n::Integer`, `Jᵢⱼ(ω)[n]` is returned.

## Full evaluation

    Jω[sites...]
    Js(ω; params...)

Given a partially evaluated `Jω::CurrentDensitySolution` or `ρs::CurrentDensitySlice`, build
a sparse matrix `Jᵢⱼ(ω)` along the specified `direction` of fully evaluated local current
densities. If `ω` above is real, a small positive imaginary part will be added internally
for the evaluation.

# Example

```jldoctest
julia> # A semi-infinite 1D lead with a magnetic field `B`

julia> g = LP.square() |> hamiltonian(@hopping((r, dr; B = 0.1) -> cis(B * dr' * SA[r[2],-r[1]]))) |> supercell((1,0), region = r->-2<r[2]<2) |> greenfunction(GS.Schur(boundary = 0));

julia> J = current(g[cells = SA[1]]); J(0.2; B = 0.1)
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

"""
current
