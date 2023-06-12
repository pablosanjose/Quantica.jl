# Manual

Welcome to the Quantica.jl manual!

Here you will learn how to use Quantica.jl to build and compute properties of tight-binding models. This includes

- Defining general lattices in arbitrary dimensions
- Defining generic tight-binding models with arbitrary parameter dependences
- Building Hamiltonians of mono or multiorbital systems by combining lattices and models
- Computing bandstructures of Hamiltonians using a range of solvers
- Creating "open Hamiltonians" by attaching self-energies of different types to Hamiltonians, representing e.g. leads
- Computing Green functions of Hamiltonians or open Hamiltonians using a range of solvers
- Computing observables from Green functions, such as spectral densities, current densities, local and nonlocal conductances, Josephson currents, critical currents, transmission probabilities, etc.

!!! tip "Check the docstrings"
    Full usage instructions on all Quantica functions can be found [here](@ref api) or within the Julia REPL by querying their docstrings. For example, to obtain details on the `hamiltonian` function or on the available `LatticePresets`, just type `?hamiltonian` or `?LatticePresets`.

## Glossary

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

## Lattices

### Constructing a Lattice from scratch

Consider a lattice like graphene's. It has two sublattices, A and B, forming a honeycomb pattern in space. The position of site A inside the unitcell is `[0, -a0/√3]`, with site B at `[0, a0/√3]`. The `i=1,2` Bravais vectors are `Aᵢ = [± cos(π/3), sin(π/3)]`. If we set the lattice constant to `a0 = 1`, one way to build this lattice in Quantica would be

```jldoctest
julia> A1, A2 = (cos(π/3), sin(π/3)), (-cos(π/3), sin(π/3));

julia> sA = sublat((0, -1/√3), name = :A);

julia> sB = sublat((0,  1/√3), name = :B);

julia> lattice(sA, sB, bravais = (A1, A2))
Lattice{Float64,2,2} : 2D lattice in 2D space
  Bravais vectors : [[0.5, 0.866025], [-0.5, 0.866025]]
  Sublattices     : 2
    Names         : (:A, :B)
    Sites         : (1, 1) --> 2 total per unit cell
```

!!! tip "Tuple, SVector and SMatrix"
    Note that we have used `Tuple`s, such as `(0, 1/√3)` instead of `Vector`s, like `[0, 1/√3]`. In Julia small-length `Tuple`s are much more efficient as containers than `Vector`s, since their length is known and fixed at compile time. Static vectors (`SVector`) and matrices (`SMatrix`) are also available to Quantica, which are just as efficient as `Tuple`s. They be entered as e.g. `SA[0, 1/√3]` and `SA[1 0; 0 1]`, respectively. For efficiency, always use `Tuple`, `SVector` and `SMatrix` in Quantica where possible.

If we don't plan to address the two sublattices individually, we could also fuse them into one, like
```jldoctest
julia> lat = lattice(sublat((0, 1/√3), (0, -1/√3)), bravais = (A1, A2))
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

!!! tip "Advanced: static `dim` with `Val`"
    For the `dim` keyword above we can alternatively use `dim = Val(3)`, which is slightly more efficient, because the value is encoded as a type. This is a Julia thing (the concept of type stability), and can be ignored upon a first contact with Quantica.

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


### Lattice presets

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

### Visualization

To produce an interactive visualization of `Lattice`s or other Quantica object you need to load GLMakie, CairoMakie or some other plotting backend from the Makie repository (i.e. do `using GLMakie`, see also Installation). Then, a number of new plotting functions will become available. The main one is `qplot`. A Lattice is represented, by default, as the sites in a unitcell plus the Bravais vectors.

```julia
julia> using GLMakie

julia> lat = LP.honeycomb()

julia> qplot(lat, hide = ())
```
```@raw html
<img src="../assets/honeycomb_lat.png" alt="Honeycomb lattice" width="250" />
```

`qplot` accepts a large number of keywords to customize your plot. In the case of lattice, most of these are passed over to the function `plotlattice`, specific to lattices and Hamiltonians. In the case above, `hide = ()` means "don't hide any element of the plot". See the `qplot` and `plotlattice` docstrings for details.

!!! tip "GLMakie vs CairoMakie"
    GLMakie is optimized for interactive GPU-accelerated, rasterized plots. If you need to export to PDF for publications or in a Jupyter notebook, use CairoMakie instead, which in general renders non-interactive, but vector-based plots.

### SiteSelectors

A central concept in Quantica is that of a "selector". There are two types of selectors, `SiteSelector`s and `HopSelectors`. `SiteSelector`s are a set of directives or rules that define a subset of its sites. The rules are defined through three keywords
- `region`: a boolean function of allowed site positions `r`.
- `sublats`: allowed sublattices of selected sites
- `cells`: allowed cell indices of selected sites

Similarly, `HopSelector`s can be used to select a number of site pairs, and will be used later to define hoppings in tight-binding models (see further below).

As an example, let us define a `SiteSelector` that picks all sites belonging to the `:B` sublattice of a given lattice within a circle of radius `10`
```jldoctest
julia> s = siteselector(region = r -> norm(r) <= 10, sublats = :B)
SiteSelector: a rule that defines a finite collection of sites in a lattice
  Region            : Function
  Sublattices       : B
  Cells             : any
```

Note that this selector is defined independently of the lattice. To apply it to a lattice `lat` we do `lat[s]`, which results in a `LatticeSlice` (i.e. a finite portion, or slice, of `lat`)
```jldoctest
julia> lat = LP.honeycomb(); lat[s]
LatticeSlice{Float64,2,2} : collection of subcells for a 2D lattice in 2D space
  Cells       : 363
  Cell range  : ([-11, -11], [11, 11])
  Total sites : 363
```
The `Cell range` above are the corners of a bounding box *in cell-index space* that contains all unit cell indices with at least one selected site. 

Let's plot it
```julia
julia> qplot(lat[s], hide = ())
```
```@raw html
<img src="../assets/latslice.png" alt="A LatticeSlice" width="400" />
```

!!! tip "qplot selector"
    The above `qplot(lat[s])` can also be written as `qplot(lat, selector = s)`, which will be useful when plotting `AbstractHamiltonians`.

!!! tip "Sites of a LatticeSlice"
    Collect the site positions of a `LatticeSlice` into a vector with `collect(sites(ls))`. If you do `sites(ls)` instead, you will get a lazy generator that can be used to iterate efficiently among site positions without allocating them in memory.

Apart from `region` and `sublats` we can also restrict the unitcells by their cell index. For example, to select all sites in unit cells within the above bounding box we can do
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

We can often omit constructing the `SiteSelector` altogether by using the keywords directly
```jldoctest
julia> ls = lat[cells = n -> 0 <= n[1] <= 2 && abs(n[2]) < 3, sublats = :A]
LatticeSlice{Float64,2,2} : collection of subcells for a 2D lattice in 2D space
  Cells       : 15
  Cell range  : ([0, -2], [2, 2])
  Total sites : 15
```

Selectors are very expressive and powerful. Do check `siteselector` and `hopselector` docstrings for more details.

### Transforming lattices

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

### Currying: chaining transformations with the `|>` operator

Many functions in Quantica have a "curried" version that allows them to be chained together using the pipe operator `|>`.

!!! note "Definition of currying"
    The curried version of a function `f(x1, x2...)` is `f´ = x1 -> f(x2...)`, so that the curried form of `f(x1, x2...)` is `x2 |> f´(x2...)`, or `f´(x2...)(x1)`. This gives the first argument `x1` a privileged role. Users of object-oriented languages such as Python may find this use of the `|>` operator somewhat similar to the way the dot operator works there (i.e. `x1.f(x2...)`).

The last example above can then be written as
```jldoctest
julia> LP.honeycomb(a0 = √3) |> transform(f) |> translate(δr) |> sites
2-element Vector{SVector{2, Float64}}:
 [-0.5, 1.0]
 [0.5, 1.0]
```

This type of curried syntax is supported by most Quantica functions, and will be used extensively in this manual.

### Extending lattices with supercells

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

Its important to note that the lattice in the directions perpendicular to the new Bravais vector is bounded. With the syntax above, the new unitcell will be minimal. We may however define how many sites to include in the new unitcell by adding a `SiteSelector` directive to be applied in the non-periodic directions. For example, to create a 10 * a0 wide, honeycomb nanoribbon we can do

```jldoctest
julia> lat = LP.honeycomb() |> supercell((1,-1), region = r -> -5 <= r[2] <= 5)
Lattice{Float64,2,1} : 1D lattice in 2D space
  Bravais vectors : [[1.0, 0.0]]
  Sublattices     : 2
    Names         : (:A, :B)
    Sites         : (12, 12) --> 24 total per unit cell

julia> qplot(lat[cells = -7:7])
```
```@raw html
<img src="../assets/nanoribbon_lat.png" alt="Honeycomb nanoribbon" height="250" />
```
!!! tip "No need to build selectors explicitly"
    Note that we we didn't build a `siteselector(region = ...)` object to pass it to `supercell`. Instead, as shown above, we passed the corresponding keywords directly to `supercell`, which then takes care to build the selector internally.

## Models

We now will see how to build a generic single-particle tight-binding model, with Hamiltonian

    ``H = \sum_{i\alpha j\beta} c_{i\alpha}^\dagger V_{\alpha\beta}(r_i, r_j)c_{j\alpha}``

Here, `α,β` are orbital indices in each site, `i,j` are site indices, and `rᵢ, rⱼ` are site positions. In Quantica we would write the above model as

```jldoctest
julia> model = onsite(r -> V(r, r)) + hopping((r, dr) -> V(r-dr/2, r+dr/2))
TightbindingModel: model with 2 terms
  OnsiteTerm{Function}:
    Region            : any
    Sublattices       : any
    Cells             : any
    Coefficient       : 1
  HoppingTerm{Function}:
    Region            : any
    Sublattice pairs  : any
    Cell distances    : any
    Hopping range     : Neighbors(1)
    Reverse hops      : false
    Coefficient       : 1
```
where `V(rᵢ, rⱼ)` is a function that returns a matrix ``V_{\alpha\beta}(r_i, r_j)`` (preferably an `SMatrix`) of the required orbital dimensionality.

Note that when writing models we distinguish between onsite (`rᵢ=rⱼ`) and hopping (`rᵢ≠rⱼ`) terms. For the former, `r` is the site position. For the latter we use a bond-center and bond-distance `(r, dr)` parametrization of `V`, so that `r₁, r₂ = r ∓ dr/2`

If the onsite  and hopping amplitudes do not depend on position, we can simply input them as constants
```jldoctest
julia> model = onsite(1) - 2*hopping(1)
TightbindingModel: model with 2 terms
  OnsiteTerm{Int64}:
    Region            : any
    Sublattices       : any
    Cells             : any
    Coefficient       : 1
  HoppingTerm{Int64}:
    Region            : any
    Sublattice pairs  : any
    Cell distances    : any
    Hopping range     : Neighbors(1)
    Reverse hops      : false
    Coefficient       : -2
```

!!! tip "Model term algebra"
    Note that we can combine model terms as in the above example by summing and subtracting them, and using constant coefficients.

### HopSelectors

By default `onsite` terms apply to any site in a Lattice, and `hopping` terms apply to any pair of sites within nearest-neighbor distance (see the `Hopping range: Neighbors(1)` above).

We can change this default by specifying a `SiteSelector` or `HopSelector` for each term. `SiteSelector`s where already introduced to create and slice Lattices. `HopSelectors` are very similar, but support slightly different keywords:
- `region`: to restrict according to bond center `r` and bond vector `dr`
- `sublats`: to restrict source and target sublattices
- `dcells`: to restrict the distance in cell index
- `range`: to restrict the distance in real space

As an example, a `HopSelector` that selects any two sites at a distance between `1.0` and the second-nearest neighbor distance, with the first belonging to sublattice `:B` and the second to sublattice `:A`, and their mean position inside a unit circle

```jldoctest
julia> hs = hopselector(range = (1.0, neighbors(2)), sublats = :B => :A, region = (r, dr) -> norm(r) < 1)
HopSelector: a rule that defines a finite collection of hops between sites in a lattice
  Region            : Function
  Sublattice pairs  : :B => :A
  Cell distances    : any
  Hopping range     : (1.0, Neighbors(2))
  Reverse hops      : false

julia> model = plusadjoint(hopping(1, hs)) - 2*onsite(1, sublats = :B)
TightbindingModel: model with 3 terms
  HoppingTerm{Int64}:
    Region            : Function
    Sublattice pairs  : :B => :A
    Cell distances    : any
    Hopping range     : (1.0, Neighbors(2))
    Reverse hops      : false
    Coefficient       : 1
  HoppingTerm{Int64}:
    Region            : Function
    Sublattice pairs  : :B => :A
    Cell distances    : any
    Hopping range     : (1.0, Neighbors(2))
    Reverse hops      : true
    Coefficient       : 1
  OnsiteTerm{Int64}:
    Region            : any
    Sublattices       : B
    Cells             : any
    Coefficient       : 1
```

`HopSelector`s and `SiteSelector`s can be used to restrict `onsite` and `hopping` terms as in the example above.

!!! tip "plusadjoint function"
    The convenience function `plusadjoint(term) = term + term'` adds the Hermitian conjugate of its argument (`term'`), equivalent to the `+ h.c.` notation often used in the literature.

!!! note "Index-agnostic modeling"
    The Quantica approach to defining tight-binding models does not rely on site indices (`i,j` above), since these are arbitrary, and may even be beyond the control of the user (for example after using `supercell`). Instead, we rely on physical properties of sites, such as position, distance or sublattice. In the future we might add an interface to also allow index-based modeling if there is demand for it, but we have yet to encounter an example where it is preferable.

### Parametric Models

The models introduced above are non-parametric, in the sense that they encode fixed, numerical Hamiltonian matrix elements. In actual problems, it is commonplace to have models that depend on a number of free parameters that will need to be adjusted during a calculation. For example, one may need to compute the phase diagram of a system as a function of a spin-orbit coupling or applied magnetic field. For these cases, we have `ParametricModel`s.

Parametric models are defined with
- `@onsite((; params...) -> ...; sites...)`
- `@onsite((r; params...) -> ...; sites...)`
- `@hopping((; params...) -> ...; hops...)`
- `@hopping((r, dr; params...) -> ...; hops...)`

where `params` enter as keyword arguments with (optional) default values. An example of a hopping model with a Peierls phase in the symmetric gauge
```jldoctest
julia> model_perierls = @hopping((r, dr; B = 0, t = 1) -> t * cis(-im * Bz/2 * SA[-r[2], r[1], 0]' * dr))
ParametricModel: model with 1 term
  ParametricHoppingTerm{ParametricFunction{2}}
    Region            : any
    Sublattice pairs  : any
    Cell distances    : any
    Hopping range     : Neighbors(1)
    Reverse hops      : false
    Coefficient       : 1
    Parameters        : [:B, :t]
```
Note that `B` and `t` are free parameters in the model.

One can linearly combine parametric and non-parametric models freely, omit argument default values, and use any of the functional argument forms described for `onsite` and `hopping` (but not the constant argument form)
```jldoctest
julia> model´ = 2 * (onsite(1) - 2 * @hopping((; t) -> t))
ParametricModel: model with 2 terms
  ParametricHoppingTerm{ParametricFunction{0}}
    Region            : any
    Sublattice pairs  : any
    Cell distances    : any
    Hopping range     : Neighbors(1)
    Reverse hops      : false
    Coefficient       : -4
    Parameters        : [:t]
  OnsiteTerm{Int64}:
    Region            : any
    Sublattices       : any
    Cells             : any
    Coefficient       : 2
```

### Modifiers

There is a third model-related functionality known as a `OnsiteModifier` and `HoppingModifier`. Given a model that defines a set of onsite and hopping amplitudes on a subset of sites and hops, one can define a parametric-dependent modification of a subset of said amplitudes. Modifiers are built with
- `@onsite!((o; params...) -> new_onsite; sites...)`
- `@onsite!((o, r; params...) -> new_onsite; sites...)`
- `@hopping((t; params...) -> new_hopping; hops...)`
- `@hopping((t, r, dr; params...) -> new_hopping; hops...)`

For example, the following modifier inserts a peierls phase on any non-zero hopping in a model
```jldoctest
julia> model_perierls! = @hopping!((t, r, dr; B = 0) -> t * cis(-Bz/2 * SA[-r[2], r[1], 0]' * dr))
HoppingModifier{ParametricFunction{3}}:
  Region            : any
  Sublattice pairs  : any
  Cell distances    : any
  Hopping range     : Inf
  Reverse hops      : false
  Parameters        : [:B]
```
The difference with `model_perierls` is that `model_perierls!` will never add any new hoppings. It will only modify a subset or all previously existing hoppings in a model. Modifiers are not models themselves, and cannot be combined with other models. They are instead meant to be applied sequentially after applying a model.

We now show how models and modifiers can be used in practice to construct Hamiltonians.

!!! note "Mind the `;`"
    While syntax like `onsite(2, sublats = :B)` and `onsite(2; sublats = :B)` are equivalent in Julia, due to the way keyword arguments are parsed, the same is not true for macro calls like `@onsite`, `@onsite!`, `@hopping` and `@hopping!`. These macros just emulate the function call syntax. But to work you must currently always use the `;` separator for keywords. Hence, something like `@onsite((; p) -> p; sublats = :B)` works, but `@onsite((; p) -> p, sublats = :B)` does not.

## Hamiltonians

We build a Hamiltonian by combining a lattice and a model, specifying the number of orbitals on each lattice if there is more than one. A spinful graphene model with nearest neighbor hopping `t0 = 2.7`
```jldoctest
julia> lat = LP.honeycomb(); model = hopping(2.7*I);

julia> h = hamiltonian(lat, model; orbitals = 2)
Hamiltonian{Float64,2,2}: Hamiltonian on a 2D Lattice in 2D space
  Bloch harmonics  : 5
  Harmonic size    : 2 × 2
  Orbitals         : [2, 2]
  Element type     : 2 × 2 blocks (ComplexF64)
  Onsites          : 0
  Hoppings         : 6
  Coordination     : 3.0
```

A crucial thing to remember when defining multi-orbital Hamiltonians as the above is that `onsite` and `hopping` amplitudes need to be matrices of the correct size. The symbol `I` in Julia represents the identity matrix of any size, which is convenient to define a spin-preserving hopping in the case above. An alternative would be to use `model = hopping(2.7*SA[1 0; 0 1])`.

!!! tip "Models with different number of orbitals per sublattice"
    Non-homogeneous multiorbital models are more advanced but are fully supported in Quantica. Just use `orbitals = (n₁, n₂,...)` to have `nᵢ` orbitals in sublattice `i`, and make sure your model is consistent with that. As in the case of the `dim` keyword in `lattice`, you can also use `Val(nᵢ)` for marginally faster construction.

### A more elaborate example: the Kane-Mele model

The Kane-Mele model for graphene describes intrinsic spin-orbit coupling (SOC), in the form of an imaginary second-nearest-neighbor hopping between same-sublattice sites, with a sign that alternates depending on hop direction `dr`. A possible implementation in Quantica would be
```jldoctest
SOC(dr) = 0.05 * ifelse(iseven(round(Int, atan(dr[2], dr[1])/(pi/3))), im, -im)

model =
  hopping(1, range = neighbors(1)) +
  hopping((r, dr) ->  SOC(dr); sublats = :A => :A, range = neighbors(2)) +
  hopping((r, dr) -> -SOC(dr); sublats = :B => :B, range = neighbors(2))

h = LatticePresets.honeycomb() |> hamiltonian(model)

qplot(h, inspector = true)
```

```@raw html
<img src="../assets/latticeKM.png" alt="Honeycomb lattice" width="350" />
```

The `inspector = true` keyword enables interactive tooltips in the visualization of `h` that allows to navigate each `onsite` and `hopping` amplitude graphically. Note that sites connected to the unit cell of `h` by some hopping are included, but are rendered with partial transparency by default.

### ParametricHamiltonians

If we use a `ParametricModel` instead of a simple `TightBindingModel` we will obtain a `ParametricHamiltonian` instead of a simple `Hamiltonian`, both of which are subtypes of the `AbstractHamiltonian` type
```jldoctest
julia> model_param = @hopping((; t = 2.7) -> t*I);

julia> h_param = hamiltonian(lat, model_param; orbitals = 2)
ParametricHamiltonian{Float64,2,2}: Parametric Hamiltonian on a 2D Lattice in 2D space
  Bloch harmonics  : 5
  Harmonic size    : 2 × 2
  Orbitals         : [2, 2]
  Element type     : 2 × 2 blocks (ComplexF64)
  Onsites          : 0
  Hoppings         : 6
  Coordination     : 3.0
  Parameters       : [:t]
```

We can also apply `Modifier`s by passing them as extra arguments to `hamiltonian`, which results again in a `ParametricHamiltonian` with the parametric modifiers applied
```jldoctest
julia> peierls! = @hopping!((t, r, dr; Bz = 0) -> t * cis(-Bz/2 * SA[-r[2], r[1]]' * dr));

julia> h_param_mod = hamiltonian(lat, model_param, peierls!; orbitals = 2)
ParametricHamiltonian{Float64,2,2}: Parametric Hamiltonian on a 2D Lattice in 2D space
  Bloch harmonics  : 5
  Harmonic size    : 2 × 2
  Orbitals         : [2, 2]
  Element type     : 2 × 2 blocks (ComplexF64)
  Onsites          : 0
  Hoppings         : 6
  Coordination     : 3.0
  Parameters       : [:Bz, :t]
```
Note that `SA[-r[2], r[1]]` above is a 2D `SVector`, because since the embedding dimension is `E = 2`, both `r` and `dr` are also 2D `SVector`s.

We can also apply modifiers to an already constructed `AbstractHamiltonian`. The following is equivalent to the above
```jldoctest
julia> h_param_mod = hamiltonian(h_param, peierls!);
```

!!! warning "Modifiers do not commute"
    We can add as many modifiers as we need by passing them as extra arguments to `hamiltonian`. Beware, however, that modifiers do not necessarily commute, in the sense that the result will in general depend on their order.

We can obtain a plain `Hamiltonian` from a `ParametricHamiltonian` by applying specific values to its parameters. To do so, simply use the call syntax with parameters as keyword arguments
```jldoctest
julia> h_param_mod(Bz = 0.1, t = 1)
Hamiltonian{Float64,2,2}: Hamiltonian on a 2D Lattice in 2D space
  Bloch harmonics  : 5
  Harmonic size    : 2 × 2
  Orbitals         : [2, 2]
  Element type     : 2 × 2 blocks (ComplexF64)
  Onsites          : 0
  Hoppings         : 6
  Coordination     : 3.0
```

### Obtaining actual matrices

For an L-dimensional AbstractHamiltonian `h` (i.e. defined on a Lattice with `L` Bravais vectors), the Hamiltonian matrix between any unit cell with cell index `n` and another unit cell at `n+dn` (here known as a Hamiltonian "harmonic") is given by `h[dn]`
```jldoctest
julia> h[(1,0)]
4×4 SparseArrays.SparseMatrixCSC{ComplexF64, Int64} with 4 stored entries:
     ⋅          ⋅      2.7+0.0im  0.0+0.0im
     ⋅          ⋅      0.0+0.0im  2.7+0.0im
     ⋅          ⋅          ⋅          ⋅
     ⋅          ⋅          ⋅          ⋅

julia> h[(0,0)]
4×4 SparseArrays.SparseMatrixCSC{ComplexF64, Int64} with 8 stored entries:
     ⋅          ⋅      2.7+0.0im  0.0+0.0im
     ⋅          ⋅      0.0+0.0im  2.7+0.0im
 2.7+0.0im  0.0+0.0im      ⋅          ⋅
 0.0+0.0im  2.7+0.0im      ⋅          ⋅
```

!!! tip "Cell distance indices"
    We can use `Tuple`s or `SVector`s for cell distance indices `dn`. An empty `Tuple` `dn = ()` will always return the main intra-unitcell harmonic: `h[()] = h[(0,0...)] = h[SA[0,0...]]`.

!!! note "Bounded Hamiltonians"
    If the Hamiltonian has a bounded lattice (i.e. it has `L=0` Bravais vectors), we will simply use an empty tuple to obtain its matrix `h[()]`. This is not in conflict with the above syntax.

Note that if `h` is a `ParametricHamiltonian`, such as `h_param` above, we will get zeros in place of the unspecified parametric terms, unless we actually first specify the values of the parameters
```jldoctest
julia> h_param[(0,0)] # Parameter t is not specified -> it is not applied
4×4 SparseArrays.SparseMatrixCSC{ComplexF64, Int64} with 8 stored entries:
     ⋅          ⋅      0.0+0.0im  0.0+0.0im
     ⋅          ⋅      0.0+0.0im  0.0+0.0im
 0.0+0.0im  0.0+0.0im      ⋅          ⋅
 0.0+0.0im  0.0+0.0im      ⋅          ⋅

julia> h_param(t=2)[(0,0)]
4×4 SparseArrays.SparseMatrixCSC{ComplexF64, Int64} with 8 stored entries:
     ⋅          ⋅      2.0+0.0im  0.0+0.0im
     ⋅          ⋅      0.0+0.0im  2.0+0.0im
 2.0+0.0im  0.0+0.0im      ⋅          ⋅
 0.0+0.0im  2.0+0.0im      ⋅          ⋅
```

!!! note "ParametricHamiltonian harmonics"
    The above behavior for unspecified parameters is not set in stone and may change in future versions. Another option would be to apply their default values (which may, however, not exist).

We are usually not interested in the harmonics `h[dn]` themselves, but rather in the Bloch matrix of a Hamiltonian
    `` H(\phi) = \sum_{dn} H_{dn} \exp(-i \phi * dn)``
where ``H_{dn}`` are the Hamiltonian harmonics, ``\phi = (\phi_1, \phi_2...) = (k\cdot A_1, k\cdot A_2...)`` are the Bloch phases, ``k`` is the Bloch wavevector and ``A_i`` are the Bravais vectors.

We obtain the Bloch matrix using the syntax `h(ϕ; params...)`
```jldoctest
julia> h((0,0))
4×4 SparseArrays.SparseMatrixCSC{ComplexF64, Int64} with 8 stored entries:
     ⋅          ⋅      8.1+0.0im  0.0+0.0im
     ⋅          ⋅      0.0+0.0im  8.1+0.0im
 8.1+0.0im  0.0+0.0im      ⋅          ⋅
 0.0+0.0im  8.1+0.0im      ⋅          ⋅

julia> h_param_mod((0.2, 0.3); B = 0.1)
4×4 SparseArrays.SparseMatrixCSC{ComplexF64, Int64} with 8 stored entries:
         ⋅                  ⋅          7.92559-1.33431im      0.0+0.0im
         ⋅                  ⋅              0.0+0.0im      7.92559-1.33431im
 7.92559+1.33431im      0.0+0.0im              ⋅                  ⋅
     0.0+0.0im      7.92559+1.33431im          ⋅                  ⋅
```

Note that unspecified parameters take their default values when using the call syntax (as per the standard Julia convention). Any unspecified parameter that does not have a default value will produce an `UndefKeywordError` error.

## Bandstructures

## GreenFunctions