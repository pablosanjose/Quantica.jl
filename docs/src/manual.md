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

