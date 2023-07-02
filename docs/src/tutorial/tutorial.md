# Tutorial

Welcome to the Quantica.jl tutorial!

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
