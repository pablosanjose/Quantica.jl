# Quantica.jl

[Quantica.jl](https://github.com/pablosanjose/Quantica.jl/actions) is a Julia package for building generic tight-binding models and computing various spectral and transport properties.

## Manual

```@contents
Pages = [
    "tutorial.md",
    "examples.md",
    "reference.md",
]
Depth = 1
```

## Current functionality

- Build arbitrary lattices (periodic or bounded in any dimension and with any unit cell)

- Define generic model Hamiltonians by applying a model onto a lattice

- Use models with arbitrary orbital structure, spatial dependence and coordination (e.g. normal/superconducting, spin-orbit coupling, etc.)

- Define parametric Hamiltonians that efficiently implement external parameters dependencies

- Efficiently compute the Bloch Hamiltonian matrix at arbitrary wave vector

- Compute the band structure or spectrum of a Hamiltonian, using advanced meshing and co-diagonalization techniques to resolve degeneracies and extract subbands

- ...