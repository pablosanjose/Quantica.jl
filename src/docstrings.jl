#######################################################################
# Lattice
#######################################################################
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
zero-th unitcell

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

    sublats = :A                          # Hops from :A to :A
    sublats = :A => :B                    # Hops from :A to :B sublattices, but not from :B to :A
    sublats = (:A => :B,)                 # Same as above
    sublats = (:A => :B, :C => :D)        # Hopping from :A to :B or :C to :D
    sublats = (:A, :C) .=> (:B, :D)       # Broadcasted pairs, same as above
    sublats = (:A, :C) => (:B, :D)        # Direct product, (:A=>:B, :A=:D, :C=>:B, :C=>D)
    sublats = 1 => 2                      # Hops from first to second sublat. Similarly, all above patterns using Integers.
    sublats = (spec₁, spec₂, ...)         # Hops matching any of the `spec`'s with any form as above

    dcells  = dn::SVector{L,Integer}      # Hops between cells at distance `dn`
    dcells  = dn::NTuple{L,Integer}       # Hops between cells at distance `SVector(dn)`
    dcells  = f::Function                 # Hops between cells at distance `dn` such that `f(dn) == true`

    range   = neighbors(n)                # Hops within the `n`-th nearest neighbor distance in the lattice
    range   = (min_range, max_range)      # Hops at distance inside the `[min_range, max_range]` closed interval (bounds can also be `neighbors(n)`)


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
    hamiltonian(lat::Lattice{T}, model; orbitals = 1)

Create a `Hamiltonian` or `ParametricHamiltonian` by applying `model` to the lattice `lat`
(see `onsite`, `@onsite`, `hopping` and `@hopping` for details on building tight-binding
models).

    hamiltonian(lat::Lattice{T}, model, modifiers...; orbitals = 1)

Same as above, but returning always a `ParametricHamiltonian` where all onsite and hopping
terms in model can be parametrically modified through the provided `modifiers` (see
`@onsite!` and `@hopping!` for details on defining modifiers).

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
    `lattice`, `onsite`, `hopping`, `@onsite`, `@hopping`, `@onsite!`, `@hopping!`
"""
hamiltonian

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
    bands(h::AbstractHamiltonian, ϕs::AbstractRange...; kw...)
"""
bands