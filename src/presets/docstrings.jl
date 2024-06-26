"""
    ExternalPresets.wannier90(filename::String; kw...)

Import a Wannier90 tight-binding file in the form of a `w::EP.WannierBuilder` object. It can
be used to obtain a `Hamiltonian` with `hamiltonian(w)`, and the matrix of the position
operator with `sites(w)`.

    ExternalPresets.wannier90(filename, model::AbstractModel; kw...)

Modify the `WannierBuilder` after import by adding `model` to it.

    push!(w::EP.WannierBuilder, modifier::AbstractModifier)
    w |> modifier

Applies a `modifier` to `w`.

## Keywords
- htol: skip matrix elements of the Hamiltonian smaller than this (in absolute value). Default: `1e-8`
- rtol: skip non-diagonal matrix elements of the position operator smaller than this (in absolute value). Default: `1e-8`
- dim: dimensionality of the embedding space for the Wannier orbitals, dropping trailing dimensions beyond `dim` if smaller than 3. Default: `3`
- latdim: dimensionality of the lattice, dropping trailing dimensions beyond `latdim` if smaller than 3. Should be `latdim <= dim`. Default: `dim`
- type: override the real number type of the imported system. Default: `Float64`

# Examples
```
julia> w = EP.wannier90("wannier_tb.dat", @onsite((; o) -> o);  htol = 1e-4, rtol = 1e-4, dim = 2, type = Float32)
WannierBuilder{Float32,2,2} : 2-dimensional Hamiltonian builder from Wannier90 input, with positions of type Float32 in 2D-dimensional space
  cells      : 151
  elements   : 6724
  modifiers  : 1

julia> h = hamiltonian(w)
ParametricHamiltonian{Float32,2,2}: Parametric Hamiltonian on a 2D Lattice in 2D space
  Bloch harmonics  : 151
  Harmonic size    : 10 × 10
  Orbitals         : [1]
  Element type     : scalar (ComplexF32)
  Onsites          : 10
  Hoppings         : 6704
  Coordination     : 670.4
  Parameters       : [:o]

julia> r = position(w)
BarebonesOperator{2}: a simple collection of 2D Bloch harmonics
  Bloch harmonics  : 151
  Harmonic size    : 10 × 10
  Element type     : SVector{2, ComplexF32}
  Nonzero elements : 7408

julia> r[sites(SA[0,0], 3), sites(SA[1,0],2)]
2-element SVector{2, ComplexF32} with indices SOneTo(2):
 -0.0016230071f0 - 0.00012927242f0im
   0.008038711f0 + 0.004102786f0im
```

# See also
    `hamiltonian`, `position`
"""
ExternalPresets.wannier90
