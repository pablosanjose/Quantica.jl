# Tutorial

Welcome to the Quantica.jl tutorial!

Here you will learn how to use Quantica.jl to build and compute properties of tight-binding models. This includes

- Defining general **Lattices** in arbitrary dimensions
- Defining generic tight-binding **Models** with arbitrary parameter dependences
- Building **Hamiltonians** of mono- or multiorbital systems by combining Lattices and Models
- Computing **Bandstructures** of Hamiltonians
- Computing **GreenFunctions** of Hamiltonians or OpenHamiltonians (i.e. Hamiltonians with attached self-energies from other Hamiltonians, such as leads).
- Computing **Observables** from Green functions, such as spectral densities, current densities, local and nonlocal conductances, Josephson currents, critical currents, transmission probabilities, etc.

Check the menu on the left for shortcuts to the relevant sections.

!!! tip "Check the docstrings"
    Full usage instructions on all Quantica.jl functions can be found [here](@ref api) or within the Julia REPL by querying their docstrings. For example, to obtain details on the `hamiltonian` function or on the available `LatticePresets`, just type `?hamiltonian` or `?LatticePresets`.
