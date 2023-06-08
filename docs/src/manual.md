# Manual

Welcome to the Quantica.jl manual!

Here you will read about using Quantica.jl to build and compute properties of tight-binding models. This includes

- Defining general lattices in arbitrary dimensions
- Defining generic tight-binding models with arbitrary parameter dependences
- Building Hamiltonians of mono or multiorbital systems by combining lattices and models
- Computing bandstructures of Hamiltonians using a range of solvers
- Creating "open Hamiltonians" by attaching self-energies of different types to Hamiltonians, representing e.g. leads
- Computing Green functions of Hamiltonians or open Hamiltonians using a range of solvers
- Computing observables from Green functions, such as spectral densities, current densities, local and nonlocal conductances, Josephson currents, critical currents, transmission probabilities, etc.

!!! tip Full usage instructions on all Quantica functions can be obtained within the Julia REPL by querying its docstrings. For example, to obtained details on the `hamiltonian` function or on the available `LatticePresets`, just type `?hamiltonian` or `?LatticePresets`.

# Glossary

- `Sublat`: a sublattice, representing a number of identical sites within the unit cell of a bounded or unbounded lattice. Each site has a position in an `E`-dimensional space (`E` is called the embedding dimension). All sites in a given `Sublat` will be able to hold the same number of orbitals, and they can be thought of as identical atoms. Each `Sublat` in a `Lattice` can be given a unique name, by default `:A`, `:B`, etc.
- `Lattice`: a collection of `Sublat`s plus a collection of `L` Bravais vectors that define the periodicity of the lattice. A bounded lattice has `L=0`, and no Bravais vectors. A `Lattice` with `L > 0` can be understood as a periodic (unbounded) set of unit cells, each containing a set of sites, each of which belongs to a different sublattice.
- `SiteSelector`: a rule that defines a subset of sites in a `Lattice`
- `HopSelector`: a rule that defines a subset of site pairs in a `Lattice`
- `LatticeSlice`: a finite subset of sites in a `Lattice`, defined by their cell index (an `L`-dimensional integer vector) and their site index (an integer) within the unit cell. Can be obtained by combining a `Lattice` and a (bounded) `SiteSelector`.
- `AbstractModel`: either a `TightBindingModel` or a `ParametricModel`
  - `TightBindingModel`: a set of `HoppingTerm`s and `OnsiteTerm`s
  - `OnsiteTerm`: a rule that, applied to a single site, produces a scalar or a (square) matrix that represents the intra-site Hamiltonian elements (single or multi-orbital)
  - `HoppingTerm`: a rule that, applied to a pair of sites, produces a scalar or a matrix that represents the inter-site Hamiltonian elements (single or multi-orbital)
  - `ParametricOnsiteTerm` and `ParametricHoppingTerm`: like the above, but dependent on some parameters that can be adjusted.
- `AbstractHamiltonian`: either a `Hamiltonian` or a `ParametricHamiltonian`
  - `Hamiltonian`: a `Lattice` combined with a `TightBindingModel`, with a specification of the number of orbitals in each `Sublat` in the `Lattice`. It represents a tight-binding Hamiltonian sharing the same periodicity as the `Lattice` (it is translationally invariant under Bravais vector shifts).
  - `ParametricHamiltonian`: like the above, but using a `ParametricModel`, which makes it dependent on a set of parameters.
- `SelfEnergy`: an operator defined to act on a `LatticeSlice` of an `AbstractHamiltonian`.
- `OpenHamiltonian`: an `AbstractHamiltonian` combined with a set of `SelfEnergies`
- `GreenFunction`: an `OpenHamiltonian` combined with a `GreenSolver`, which is an algorithm that can compute the Green function at any energy between any subset of sites of the underlying lattice.
  - `GreenSlice`: a `GreenFunction` evaluated on a specific set of sites, but at an unspecified energy
  - `GreenSolution`: a `GreenFunction` evaluated at a specific energy, but on an unspecified set of sites

# Building Lattices

## Constructing a Lattice from scratch

Consider a lattice like graphene's. It has two sublattices, A and B, forming a honeycomb pattern in space. The position of the single A site inside the unitcell is `[0, -a0/√3]`, with the B site at `[0, a0/√3]`. The `i=1,2` Bravais vectors are `Aᵢ = [± cos(π/3), sin(π/3)]`. If we set the lattice constant `a0 = 1`, one way to build this lattice would be

```jldoctest
julia> A = (cos(π/3), sin(π/3)), (-cos(π/3), sin(π/3));

julia> sA = sublat((0, -1/√3), name = :A);

julia> sB = sublat((0,  1/√3), name = :B);

julia> lattice(sA, sB, bravais = A)
Lattice{Float64,2,2} : 2D lattice in 2D space
  Bravais vectors : [[0.5, 0.866025], [-0.5, 0.866025]]
  Sublattices     : 2
    Names         : (:A, :B)
    Sites         : (1, 1) --> 2 total per unit cell
```

!!! tip Note that we have used `Tuple`s, such as `(0, 1/√3)` instead of `Vector`s, like `[0, 1/√3]`. In Julia small-length `Tuple`s are much more efficient than `Vector`s, since their length is known and fixed at compile time. Static vectors (`SVector`) and matrices (`SMatrix`) are also available to Quantica, which are just as efficient as `Tuple`s. They be entered as `SA[0, 1/√3]` and `SA[1 0; 0 1]`, respectively. Always use `Tuple`, `SVector` and `SMatrix` in Quantica where possible.

If we don't plan to address the two sublattices individually, we could also fuse them into one with
```jldoctest
julia> lat = lattice(sublat((0, 1/√3), (0, -1/√3)), bravais = A)
Lattice{Float64,2,2} : 2D lattice in 2D space
  Bravais vectors : [[0.5, 0.866025], [-0.5, 0.866025]]
  Sublattices     : 1
    Names         : (:A,)
    Sites         : (2,) --> 2 total per unit cell
```

This lattice has type `Lattice{T,E,L}`, with `T = Float64` the numeric type of position coordinates, `E = 2` the dimension of embedding space, and `L = 2` the number of Bravais vectors (i.e. the lattice dimension). Both `T` and `E`, and even the `Sublat` names can be overridden when creating a lattice. One can also provide the Bravais vectors as a matrix, with each `Aᵢ` as a column

```jldoctest
julia> Amat = SA[-cos(π/3) cos(π/3); sin(π/3) sin(π/3)];

julia> lat´ = lattice(sA, sB, bravais = Amat, type = Float32, dim = 3, names = (:C, :D))
Lattice{Float32,3,2} : 2D lattice in 3D space
  Bravais vectors : Vector{Float32}[[-0.5, 0.866025, 0.0], [0.5, 0.866025, 0.0]]
  Sublattices     : 2
    Names         : (:C, :D)
    Sites         : (1, 1) --> 2 total per unit cell
```

!!! tip For the `dim` keyword above we can alternatively use `dim = Val(3)`, which is slightly more efficient, because the value is encoded as a type. This is a Julia thing (the concept of type stability), and can be ignored upon a first contact with Quantica.

One can also *convert* an existing lattice like the above to have a different type, embedding dimension, bravais vectors, `Sublat` names with

```jldoctest
julia> lat´´ = lattice(lat´, bravais = √3 * Amat, type = Float16, dim = 2, names = (:Boron, :Nitrogen))
Lattice{Float16,2,2} : 2D lattice in 2D space
  Bravais vectors : Vector{Float16}[[-0.866, 1.5], [0.866, 1.5]]
  Sublattices     : 2
    Names         : (:Boron, :Nitrogen)
    Sites         : (1, 1) --> 2 total per unit cell
```

A list of site positions in a lattice `lat` can be obtained with `sites(lat)`, or `sites(lat, sublat)` to restrict to a specific sublattice
```jldoctest
julia> sites(lat´´)
2-element Vector{SVector{2, Float16}}:
 [0.0, -0.5]
 [0.0, 0.5]

julia> sites(lat´´, :Nitrogen)
1-element view(::Vector{SVector{2, Float16}}, 2:2) with eltype SVector{2, Float16}:
 [0.0, 0.5]
```

Similarly, the Bravais matrix of a `lat` can be obtained with `bravais_matrix(lat)`.


## Lattice presets

We can also use a collection of pre-built lattices in different dimensions, which are defined in the submodule `LatticePresets`, also called `LP`. These presets currently include
- `LP.linear`: linear 1D lattice
- `LP.square`: square 2D lattice
- `LP.honeycomb`: square 2D lattice
- `LP.cubic`: cubic 3D lattice
- `LP.bcc`: body-centered cubic 3D lattice
- `LP.fcc`: face-centered cubic 3D lattice

One can modify any of these presets by passing a `bravais`, `type`, `dim`, `names` and also a new keyword `a0` for the lattice constant. The last lattice above can thus be also obtained with

```jldoctest
julia> lat´´ = LP.honeycomb(a0 = √3, type = Float16, names = (:Boron, :Nitrogen))
Lattice{Float16,2,2} : 2D lattice in 2D space
  Bravais vectors : Vector{Float16}[[0.866, 1.5], [-0.866, 1.5]]
  Sublattices     : 2
    Names         : (:Boron, :Nitrogen)
    Sites         : (1, 1) --> 2 total per unit cell
```

## SiteSelectors

A central concept in Quantica is a selector. There are two types of selectors, `SiteSelector`s and `HopSelectors`. `SiteSelector`s are a set of directives that can be applied to a lattice to select a subset of its sites. Similarly, `HopSelector`s can be used to select a number of site pairs, and will be used later to define tight-binding models and Hamiltonians.

Let us define a SiteSelector that picks all sites belonging to the `:B` sublattice of a given lattice within a circle of radius 10

```jldoctest
julia> s = siteselector(region = r -> norm(r) <= 10, sublats = :B)
SiteSelector: a rule that defines a finite collection of sites in a lattice
  Region            : Function
  Sublattices       : B
  Cells             : any
```

Note that this selector is defined independently of the lattice. To apply it to a lattice `lat` we do `lat[s]`, which results in a `LatticeSlice`
```jldoctest
julia> lat = LP.honeycomb(); lat[s]
LatticeSlice{Float64,2,2} : collection of subcells for a 2D lattice in 2D space
  Cells       : 363
  Cell range  : ([-11, -11], [11, 11])
  Total sites : 363
```
The `Cell range` above are the corners of a bounding box that contains all unit cell indices with at least one selected site.

!!! tip Collect the site positions of a `LatticeSlice` with `collect(sites(ls))`. If you do `sites(ls)` instead, you will get a lazy iterator that can be used to iterate efficiently among site positions without allocating them in memory

Apart from `region` and `sublats` we can also restrict the unitcells by their index. For example, to select all sites in unit cells within the above bounding box we can do
```jldoctest
julia> s´ = siteselector(cells = CartesianIndices((-11:11, -11:11)))
SiteSelector: a rule that defines a finite collection of sites in a lattice
  Region            : any
  Sublattices       : any
  Cells             : CartesianIndices((-11:11, -11:11))

julia> lat[s´]
LatticeSlice{Float64,2,2} : collection of subcells for a 2D lattice in 2D space
  Cells       : 529
  Cell range  : ([-11, -11], [11, 11])
  Total sites : 1058
```

We can also often omit constructing the `SiteSelector` altogether by passing the keywords directly
```jldoctest
julia> ls = lat[cells = n -> 0 <= n[1] <= 2 && abs(n[2]) < 3, sublats = :A]
LatticeSlice{Float64,2,2} : collection of subcells for a 2D lattice in 2D space
  Cells       : 15
  Cell range  : ([0, -2], [2, 2])
  Total sites : 15
```

Selectors are very expressive and powerful. Do check `siteselector` and `hopselector` docstrings for more details.

## Transforming lattices

To transform a lattice, so that site positions `r` become `f(r)` use `transform`
```jldoctest
julia> f(r) = SA[0 1; 1 0] * r
f (generic function with 1 method)

julia> rotated_honeycomb = transform(LP.honeycomb(a0 = √3), f)
Lattice{Float64,2,2} : 2D lattice in 2D space
  Bravais vectors : [[1.5, 0.866025], [1.5, -0.866025]]
  Sublattices     : 2
    Names         : (:A, :B)
    Sites         : (1, 1) --> 2 total per unit cell

julia> sites(rotated_honeycomb)
2-element Vector{SVector{2, Float64}}:
 [-0.5, 0.0]
 [0.5, 0.0]
```

To translate a lattice by a displacement vector `δr` use `translate` 
```jldoctest
julia> δr = SA[0, 1];

julia> sites(translate(rotated_honeycomb, δr))
2-element Vector{SVector{2, Float64}}:
 [-0.5, 1.0]
 [0.5, 1.0]
```

## Currying: chaining transformations with the `|>` operator

Many functions in Quantica have a "curried" version that allows them to be chained together using the pipe operator `|>`.

!!! tip The curried version of a function `f(x1, x2...)` is `f´ = x1 -> f(x2...)`, so that the curried form of `f(x1, x2...)` is `x2 |> f´(x2...)`, or `f´(x2...)(x1)`. This gives the first argument `x1` a privileged role. Users of object-oriented languages such as Python may find this use of the `|>` operator somewhat similar to the way the dot operator works there (i.e. `x1.f(x2...)`).

The last example above can then be written as
```jldoctest
julia> LP.honeycomb(a0 = √3) |> transform(f) |> translate(δr) |> sites
2-element Vector{SVector{2, Float64}}:
 [-0.5, 1.0]
 [0.5, 1.0]
```

This type of curried syntax is supported by most Quantica functions, and will be used extensively in this manual.

## Extending lattices with supercells

As a periodic structure, the choice of the unitcell in an unbounded lattice is to an extent arbitrary. Given a lattice `lat` we can obtain another with a unit cell 3 times larger with `supercell(lat, 3)`

```jldoctest
julia> lat = LP.honeycomb() |> supercell(3)
Lattice{Float64,2,2} : 2D lattice in 2D space
  Bravais vectors : [[1.5, 2.598076], [-1.5, 2.598076]]
  Sublattices     : 2
    Names         : (:A, :B)
    Sites         : (9, 9) --> 18 total per unit cell
```
More generally, given a lattice `lat` with Bravais matrix `Amat = bravais_matrix(lat)`, we can obtain a larger one with Bravais matrix `Amat´ = Amat * S`, where `S` is a square matrix of integers. In the example above, `S = SA[3 0; 0 3]`. The columns of `S` represent the coordinates of the new Bravais vectors in the basis of the old Bravais vectors. A more general example with e.g. `S = SA[3 1; -1 2]` can be written either in terms of `S` or of its columns

```jldoctest
julia> supercell(lat, SA[3 1; -1 2]) == supercell(lat, (3, -1), (1, 2))
true
```

We can also use `supercell` to reduce the number of Bravais vectors, and hence the lattice dimensionality. To construct a new lattice with a single Bravais vector `A₁´ = 3A₁ - A₂`, just omit the second one
```jldoctest
julia> supercell(lat, (3, -1))
Lattice{Float64,2,1} : 1D lattice in 2D space
  Bravais vectors : [[6.0, 5.196152]]
  Sublattices     : 2
    Names         : (:A, :B)
    Sites         : (27, 27) --> 54 total per unit cell
```

Its important to note that the lattice in the directions perpendicular to the new Bravais vector is bounded. With the syntax above, the new unitcell will be minimal. We may however define how many sites to include in the new unitcell by adding a `SiteSelector` directive to be applied in the non-periodic directions. For example, to create a 10-site wide, square-lattice nanoribbon we can do

```jldoctest
julia> LP.square() |> supercell((1,0), region = r -> 0 <= r[2] < 10)
Lattice{Float64,2,1} : 1D lattice in 2D space
  Bravais vectors : [[1.0, 0.0]]
  Sublattices     : 1
    Names         : (:A,)
    Sites         : (10,) --> 10 total per unit cell
```

!!! tip As discussed in the `SiteSelector` section, we don't build a `siteselector(region = ...)` object to then pass it to `supercell`: as shown above we instead pass the corresponding keywords directly to `supercell`, which takes care to build the selector internally.

## Visualizing lattices

To produce an interactive visualization of `Lattice`s or other Quantica object you need to load GLMakie or some other plotting backend from the Makie repository (doing `using GLMakie`, see also Installation). Then, a number of new plotting functions will become available. The main one is `qplot`

```julia
julia> using GLMakie

julia> lat = LP.bcc() |> supercell(6);

julia> qplot(lat, sitecolor = :orange)
```
![BCC lattice](assets/bcclat.png)