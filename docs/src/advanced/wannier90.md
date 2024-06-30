# Wannier90 imports

A common way to obtain quantitative tight-binding models of materials is to *Wannierize* density-functional-theory (DFT) bandstructures. In a nutshell, this procedure consists in obtaining a basis set of a subspace of some DFT bandstructure, subject to the condition that the obtained states are maximally localized. A popular implementation of this algorithm is (Wannier90)[https://wannier.org]. Among other things, this tool produces output files that encode a tight-binding Hamiltonian for the material and the matrix elements of the position operator in the maximally localized Wannier basis.

Quantica.jl includes a function that can import Wannier90 tight-binding files. By default these files are 3D systems
```julia
julia> w = ExternalPresets.wannier90("wannier_tb.dat")
WannierBuilder{Float64,3} : 3-dimensional Hamiltonian builder of type Float64 from Wannier90 input
  cells      : 755
  elements   : 36388
  modifiers  : 0
```
In this case, however, the model in the "wannier_tb.dat" file is a 2D MoS2 crystal. We can project out all out-of-plane matrix elements by specifying the dimension with `dim`. We can also drop any Hamiltonian matrix element smaller than, say `htol = 1e-5`, and any position matrix element of norm smaller than `rtol = 1e-4`. This greatly simplifies the problem
```julia
julia> w = ExternalPresets.wannier90("wannier_tb.dat"; dim = 2, htol = 1e-5, rtol = 1e-4)
WannierBuilder{Float64,2} : 2-dimensional Hamiltonian builder of type Float64 from Wannier90 input
  cells      : 151
  elements   : 7510
  modifiers  : 0
```
This object can then be converted into a Hamiltonian `h` or a position operator `r`
```julia
julia> h = hamiltonian(w)
Hamiltonian{Float64,2,2}: Hamiltonian on a 2D Lattice in 2D space
  Bloch harmonics  : 151
  Harmonic size    : 10 × 10
  Orbitals         : [1]
  Element type     : scalar (ComplexF64)
  Onsites          : 10
  Hoppings         : 7500
  Coordination     : 750.0

julia> r = position(w)
BarebonesOperator{2}: a simple collection of 2D Bloch harmonics
  Bloch harmonics  : 151
  Harmonic size    : 10 × 10
  Element type     : SVector{2, ComplexF64}
  Nonzero elements : 7408
```
Note that `r` is not of type `Hamiltonian`. The `BarebonesOperator` is a specially simple operator type that simply encodes a number of Bloch harmonics (matrices between different unit cells) of arbitrary element type. It supports only a subset of the funcitionality of `AbstractHamiltonian`s. In particular, it supports indexing:
```julia
julia> r[SA[0,0]]
10×10 SparseArrays.SparseMatrixCSC{SVector{2, ComplexF64}, Int64} with 50 stored entries:
 [-0.000563148+0.0im, 1.79768+0.0im]                  …            ⋅
           ⋅                                             [0.164126-2.15538e-5im, -0.000484848-0.0144407im]
           ⋅                                             [0.0195449-4.9251e-5im, 2.02798e-7+0.00140866im]
 [2.48859e-5-0.0185437im, -0.00534254-1.88085e-5im]                ⋅
           ⋅                                             [2.07772e-7-0.00769914im, 0.00831306+1.45056e-5im]
 [-0.00340134-1.02057e-5im, 1.89607e-5+0.00656423im]  …            ⋅
 [-0.000371236+0.0227337im, -0.101768+1.64659e-5im]                ⋅
           ⋅                                             [0.210672-5.77589e-5im, -0.000233323-0.00456068im]
 [0.164126-2.14909e-5im, -0.000483435-0.0144407im]                 ⋅
           ⋅                                             [0.000608652+0.0im, 2.12317+0.0im]

julia> r[sites(1), sites(4)]
2-element SVector{2, ComplexF64} with indices SOneTo(2):
  2.4885857e-5 + 0.018543702im
 -0.0053425408 + 1.8808481e-5im
```

It is possible to modify the imported Wannier90 models using the full Quantica.jl machinery. For example, we can add any `AbstractModel` to the Wannier90 model upon import just by passing it as a second argument
```julia
julia> w = EP.wannier90("wannier_tb.dat", @onsite((; Δ = 0) -> Δ); dim = 2)
WannierBuilder{Float64,2} : 2-dimensional Hamiltonian builder of type Float64 from Wannier90 input
  cells      : 151
  elements   : 7560
  modifiers  : 1

julia> h = hamiltonian(w)
ParametricHamiltonian{Float64,2,2}: Parametric Hamiltonian on a 2D Lattice in 2D space
  Bloch harmonics  : 151
  Harmonic size    : 10 × 10
  Orbitals         : [1]
  Element type     : scalar (ComplexF64)
  Onsites          : 10
  Hoppings         : 7540
  Coordination     : 754.0
  Parameters       : [:Δ]
```
Note that since we used a `ParametricModel` with a single parametric term, this introduced one `modifier`, since ParametricModels are simply an ordinary base model plus one modifier for each parametric term. As a result, `h` is now parametric.

!!! note "Adding models after import"
    Although the above is the recommended way to add a Quantica model to a Wannier90 model (i.e. explicitly at import time), one can also do the same with `Quantica.add!(EP.hbuilder(w), model)` to modify `w` in place after its creation. This employs internal functionality, so it is not recommended, as it could change without warning.

We can also use the following syntax apply one or more modifiers explicitly
```julia
julia> w´ = w |> @onsite!((o; k = 0) -> o + k)
WannierBuilder{Float64,2} : 2-dimensional Hamiltonian builder of type Float64 from Wannier90 input
  cells      : 151
  elements   : 7560
  modifiers  : 2
```

An interesting application of modifiers is the addition of an electric field that couples to the full `r` operator. In an strict tight-binding limit, we would add an electric field `E` simply as an onsite potential
```julia
julia> hE = h |> @onsite!((o, r; E = SA[0,0]) -> o + E'*r);
```
However, we actually have the full `r` operator now, which includes non-diagonal matrix elements. We can then incorporate the electric field term `E'*r` more precisely. We can do so using the `-->` syntax and the indexing functionality of the `r::BarebonesOperator` that we obtained from Wannier90
```julia
julia> hE = h |> @onsite!((o, i; E = SA[0,0]) --> o + E'*r[i,i]) |> @hopping!((t, i, j; E = SA[0,0]) --> t + E'*r[i,j]);
```

!!! note "Closures over non-constant objects"
    Note that the above creates a closure over `r`, which is not `const`. As a result this would incur a small performance and allocation cost when evaluating `hE(E=...)`. We can avoid it e.g. by defining `r` as a constant, `const r = sites(w)`.
