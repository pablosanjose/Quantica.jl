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

# Building a Lattice

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
julia> lattice(sublat((0, 1/√3), (0, -1/√3)), bravais = A)
Lattice{Float64,2,2} : 2D lattice in 2D space
  Bravais vectors : [[0.5, 0.866025], [-0.5, 0.866025]]
  Sublattices     : 1
    Names         : (:A,)
    Sites         : (2,) --> 2 total per unit cell
```

This lattice has type `Lattice{T,E,L}`, with `T = Float64` the numeric type of position coordinates, `E = 2` the dimension of embedding space, and `L = 2` the number of Bravais vectors (i.e. the lattice dimension). Both `T` and `E`, and even the `Sublat` names can be overridden when creating a lattice. One can also provide the Bravais vectors as a matrix, with each `Aᵢ` as a column

```jldoctest
julia> Amat = SA[-cos(π/3) cos(π/3); sin(π/3) sin(π/3)];

julia> lat = lattice(sA, sB, bravais = Amat, type = Float32, dim = 3, names = (:C, :D))
Lattice{Float32,3,2} : 2D lattice in 3D space
  Bravais vectors : Vector{Float32}[[-0.5, 0.866025, 0.0], [0.5, 0.866025, 0.0]]
  Sublattices     : 2
    Names         : (:C, :D)
    Sites         : (1, 1) --> 2 total per unit cell
```

!!! tip For the `dim` keyword above we can alternatively use `dim = Val(3)`, which is slightly more efficient, because the value is encoded as a type. This is a Julia thing (the concept of type stability), and can be ignored upon a first contact with Quantica.

One can also *convert* an existing lattice like the above to have a different type, embedding dimension, bravais vectors, `Sublat` names with

```jldoctest
julia> lattice(lat, bravais = √3 * Amat, type = Float16, dim = 2, names = (:Boron, :Nitrogen))
Lattice{Float16,2,2} : 2D lattice in 2D space
  Bravais vectors : Vector{Float16}[[-0.866, 1.5], [0.866, 1.5]]
  Sublattices     : 2
    Names         : (:Boron, :Nitrogen)
    Sites         : (1, 1) --> 2 total per unit cell
```

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
julia> LP.honeycomb(a0 = √3, type = Float16, names = (:Boron, :Nitrogen))
Lattice{Float16,2,2} : 2D lattice in 2D space
  Bravais vectors : Vector{Float16}[[0.866, 1.5], [-0.866, 1.5]]
  Sublattices     : 2
    Names         : (:Boron, :Nitrogen)
    Sites         : (1, 1) --> 2 total per unit cell
```