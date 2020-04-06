```@meta
CurrentModule = Quantica
```

# Quantica.jl

A Julia package for building generic tight-binding models and computing various spectral and transport properties.

!!! note "Important information"

    This package supersedes [Elsa.jl](https://github.com/pablosanjose/Elsa.jl/), which will soon be deprecated.

## Manual

```@contents
Pages = [
    "man/tutorial.md",
    "man/examples.md",
    "man/reference.md",
]
Depth = 1
```

## Current functionality

- Build arbitrary lattices (periodic or bounded in any dimension and with any unit cell)

- Define generic model Hamiltonians by applying a model onto a lattice

- Use models with arbitrary orbital structure, spatial dependence and coordination (e.g. normal/superconducting, spin-orbit coupling, etc.)

- Define parametric Hamiltonians that efficiently implement external parameters dependencies

- Efficiently compute the Bloch Hamiltonian matrix at arbitrary wave vector

- Compute the band structure or spectrum of a Hamiltonian.

- Use advanced meshing and co-diagonalization techniques to resolve degeneracies and extract subbands

- Use Order-N Kernel polynomial methods to compute spectral and transport properties efficiently

!!! tip "Funding"

    This work has been partly funded by the Spanish Ministry of Economy and Competitiveness under Grant Nos. FIS2015-65706-P, PCI2018-093026, and the CSIC Intramural Project 201760I086.

