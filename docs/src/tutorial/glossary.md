# Glossary

This is a summary of the type of objects you will be studying.

- **`Sublat`**: a sublattice, representing a number of identical sites within the unit cell of a bounded or unbounded lattice. Each site has a position in an `E`-dimensional space (`E` is called the embedding dimension). All sites in a given `Sublat` will be able to hold the same number of orbitals, and they can be thought of as identical atoms. Each `Sublat` in a `Lattice` can be given a unique name, by default `:A`, `:B`, etc.
- **`Lattice`**: a collection of `Sublat`s plus a collection of `L` Bravais vectors that define the periodicity of the lattice. A bounded lattice has `L=0`, and no Bravais vectors. A `Lattice` with `L > 0` can be understood as a periodic (unbounded) collection of unit cells, each containing a set of sites, each of which belongs to a different sublattice.
- **`SiteSelector`**: a rule that defines a subset of sites in a `Lattice` (not necessarily restricted to a single unit cell)
- `HopSelector`: a rule that defines a subset of site pairs in a `Lattice` (not necessarily restricted to the same unit cell)
- **`LatticeSlice`**: a *finite* subset of sites in a `Lattice`, defined by their cell index (an `L`-dimensional integer vector, usually denoted by `n` or `cell`) and their site index within the unit cell (an integer). A `LatticeSlice` an be constructed by combining a `Lattice` and a (bounded) `SiteSelector`.
- **`AbstractModel`**: either a `TightBindingModel` or a `ParametricModel`
  - **`TightBindingModel`**: a set of `HoppingTerm`s and `OnsiteTerm`s
    - **`OnsiteTerm`**: a rule that, applied to a single site, produces a scalar or a (square) matrix that represents the intra-site Hamiltonian elements (single or multi-orbital)
    - **`HoppingTerm`**: a rule that, applied to a pair of sites, produces a scalar or a matrix that represents the inter-site Hamiltonian elements (single or multi-orbital)
  - **`ParametricModel`**: a set of `ParametricOnsiteTerm`s and `ParametricHoppingTerm`s
    - **`ParametricOnsiteTerm`**: an `OnsiteTerm` that depends on a set of free parameters that can be adjusted, and that may or may not have a default value
    - **`ParametricHoppingTerm`**: a `HoppingTerm` that depends on parameters, like `ParametricOnsiteTerm` above
- **`AbstractHamiltonian`**: either a `Hamiltonian` or a `ParametricHamiltonian`
  - **`Hamiltonian`**: a `Lattice` combined with a `TightBindingModel`.

    It also includes a specification of the number of orbitals in each `Sublat` in the `Lattice`. A `Hamiltonian` represents a tight-binding Hamiltonian sharing the same periodicity as the `Lattice` (it is translationally invariant under Bravais vector shifts).

  - `ParametricHamiltonian`: like the above, but using a `ParametricModel`, which makes it dependent on a set of free parameters that can be efficiently adjusted.

  An `h::AbstractHamiltonian` can be used to produce a Bloch matrix `h(ϕ; params...)` of the same size as the number of orbitals per unit cell, where `ϕ = [ϕᵢ...]` are Bloch phases and `params` are values for the free parameters, if any.
- **`Spectrum`**: the set of eigenpairs (eigenvalues and corresponding eigenvectors) of a Bloch matrix. It can be computed with a number of `EigenSolvers`.
- **`Bandstructure`**: a collection of spectra, evaluated over a discrete mesh (typically a discretization of the Brillouin zone), that is connected to its mesh neighbors into a linearly-interpolated approximation of the `AbstractHamiltonian`'s bandstructure.
- **`SelfEnergy`**: an operator `Σ(ω)` defined to act on a `LatticeSlice` of an `AbstractHamiltonian` that depends on energy `ω`.
- **`OpenHamiltonian`**: an `AbstractHamiltonian` combined with a set of `SelfEnergies`
- **`GreenFunction`**: an `OpenHamiltonian` combined with an `AbstractGreenSolver`, which is an algorithm that can in general compute the retarded or advanced Green function at any energy between any subset of sites of the underlying lattice.
  - **`GreenSlice`**: a `GreenFunction` evaluated on a specific set of sites, but at an unspecified energy
  - **`GreenSolution`**: a `GreenFunction` evaluated at a specific energy, but on an unspecified set of sites

- **`OrbitalSliceArray`**: an `AbstractArray` that can be indexed with a `SiteSelector`, in addition to the usual scalar indexing. Particular cases are `OrbitalSliceMatrix` and `OrbitalSliceVector`. This is the most common type obtained from `GreenFunction`s and observables obtained from them.
- **Observables**: Supported observables, obtained from Green functions using various algorithms, include **local density of states**, **density matrices**, **current densities**, **transmission probabilities**, **conductance** and **Josephson currents**
