```@meta
CurrentModule = Quantica
```

# Introduction

Quantica.jl is a Julia package designed to build generic tight-binding models and to compute several useful properties.

## Current functionality

- Build arbitrary `Lattice`s (periodic or bounded in any dimension and with any unit cell)

- Define generic model `Hamiltonian`s by applying a `Model` onto a `Lattice`

- Use `Model`s with arbitrary orbital structure, spatial dependence and coordination (e.g. normal/superconducting, spin-orbit coupling, etc.)

- Define `ParametricHamiltonian`s to efficiently apply external parameters in a Hamiltonian

- Efficiently compute the Bloch Hamiltonian matrix at arbitrary wavevector.

- Compute the bandstructure or spectrum of a Hamiltonian.

- Use sophisticated meshing and codiagonalization techniques to resolve degeneracies and optimally reconnect subbands.

- Use Order-N Kernel polynomial methods to compute spectral and transport properties efficiently

## Planned functionality

- Interpolated Green's functions
- Landauer and Kubo formalisms
- Self-consistent mean field calculations

## Design principles

- Present a simple, discoverable, expressive and well-documented API to the user

- Focus on performance, generality, modularity, composability and good coding practices

# Manual

```@contents
Pages = [
    "man/overview.md",
]
Depth = 1
```


```@index
```

```@autodocs
Modules = [Quantica]
```

!!! note

    This work has been partly funded by the Spanish Ministry of Economy and Competitiveness under Grant Nos. FIS2015-65706-P, PCI2018-093026, and the CSIC Intramural Project 201760I086. 
   
