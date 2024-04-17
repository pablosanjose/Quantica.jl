# Non-spatial models and self-consistent mean-field problems

As briefly mentioned when discussing parametric models and modifiers, we have a special syntax that allows models to depend on sites directly, instead of on their spatial location. We call these non-spatial models. A simple example, with an onsite energy proportional to the site **index**
```julia
julia> model = @onsite((i; p = 1) --> ind(i) * p)
ParametricModel: model with 1 term
  ParametricOnsiteTerm{ParametricFunction{1}}
    Region            : any
    Sublattices       : any
    Cells             : any
    Coefficient       : 1
    Argument type     : non-spatial
    Parameters        : [:p]
```
or a modifier that changes a hopping between different cells
```julia
julia> modifier = @hopping!((t, i, j; dir = 1) --> (cell(i) - cell(j))[dir])
HoppingModifier{ParametricFunction{3}}:
  Region            : any
  Sublattice pairs  : any
  Cell distances    : any
  Hopping range     : Inf
  Reverse hops      : false
  Argument type     : non-spatial
  Parameters        : [:dir]
```

Note that we use the special syntax `-->` instead of `->`. This indicates that the positional arguments of the function, here called `i` and `j`, are no longer site positions as up to now. These `i, j` are non-spatial arguments, as noted by the `Argument type` property shown above. Instead of a position, a non-spatial argument `i` represent an individual site, whose index is `ind(i)`, its position is `pos(i)` and the cell it occupies on the lattice is `cell(i)`.

Technically `i` is of type `CellSitePos`, which is an internal type not meant for the end user to instantiate. One special property of this type, however, is that it can efficiently index `OrbitalSliceArray`s. We can use this to build a Hamiltonian that depends on an observable, such as a `densitymatrix`. A simple example of a four-site chain with onsite energies shifted by a potential proportional to the local density on each site:
```julia
julia> h = LP.linear() |> onsite(2) - hopping(1) |> supercell(4) |> supercell;

julia> h(SA[])
4×4 SparseArrays.SparseMatrixCSC{ComplexF64, Int64} with 10 stored entries:
  2.0+0.0im  -1.0+0.0im       ⋅           ⋅
 -1.0+0.0im   2.0+0.0im  -1.0+0.0im       ⋅
      ⋅      -1.0+0.0im   2.0+0.0im  -1.0+0.0im
      ⋅           ⋅      -1.0+0.0im   2.0+0.0im

julia> g = greenfunction(h, GS.Spectrum());

julia> ρ = densitymatrix(g[])(0.5, 0) ## density matrix at chemical potential `µ=0.5` and temperature `kBT = 0`  on all sites
4×4 OrbitalSliceMatrix{Matrix{ComplexF64}}:
 0.138197+0.0im  0.223607+0.0im  0.223607+0.0im  0.138197+0.0im
 0.223607+0.0im  0.361803+0.0im  0.361803+0.0im  0.223607+0.0im
 0.223607+0.0im  0.361803+0.0im  0.361803+0.0im  0.223607+0.0im
 0.138197+0.0im  0.223607+0.0im  0.223607+0.0im  0.138197+0.0im

julia> hρ = h |> @onsite!((o, i; U = 0, ρ) --> o + U * ρ[i])
ParametricHamiltonian{Float64,1,0}: Parametric Hamiltonian on a 0D Lattice in 1D space
  Bloch harmonics  : 1
  Harmonic size    : 4 × 4
  Orbitals         : [1]
  Element type     : scalar (ComplexF64)
  Onsites          : 4
  Hoppings         : 6
  Coordination     : 1.5
  Parameters       : [:U, :ρ]

julia> hρ(SA[]; U = 1, ρ = ρ0)
4×4 SparseArrays.SparseMatrixCSC{ComplexF64, Int64} with 10 stored entries:
 2.1382+0.0im    -1.0+0.0im         ⋅             ⋅
   -1.0+0.0im  2.3618+0.0im    -1.0+0.0im         ⋅
        ⋅        -1.0+0.0im  2.3618+0.0im    -1.0+0.0im
        ⋅             ⋅        -1.0+0.0im  2.1382+0.0im
```

Note the `ρ[i]` above. This indexes `ρ` at site `i`. For a multiorbital hamiltonian, this will be a matrix (the local density matrix on each site `i`). Here it is just a number, either ` 0.138197` (sites 1 and 4) or `0.361803` (sites 2 and 3).

The above provides the tools to implement self-consistent mean field. We just need to find a fixed point `ρf(ρ) = ρ` of the function `ρf` that produces the density matrix of the system.

In the following example we use the FixedPoint.jl package. It provides a simple utility `afps(f, x0)` that computes the solution of `f(x) = x` with initial condition `x0`. The package requires `x0` to be a (real) `AbstractArray`. Note that any other fixed-point search routine that work with `AbstractArray`s should also work.
```julia
julia> using FixedPoint

julia> ρf(hρ; µ = 0.5, kBT = 0, U = 0.1) = ρ -> densitymatrix(greenfunction(hρ(; U, ρ), GS.Spectrum())[])(µ, kBT)
ρf (generic function with 1 method)

julia> (ρsol, ρerror, iters) = @time afps(ρf(hρ; U = 0.4), real(ρ0), tol = 1e-8, ep = 0.95, vel = 0.0); @show iters, ρerror; ρsol
  0.000627 seconds (1.91 k allocations: 255.664 KiB)
(iters, ρerror) = (8, 2.0632459629688071e-10)
4×4 OrbitalSliceMatrix{Matrix{ComplexF64}}:
 0.145836+0.0im  0.227266+0.0im  0.227266+0.0im  0.145836+0.0im
 0.227266+0.0im  0.354164+0.0im  0.354164+0.0im  0.227266+0.0im
 0.227266+0.0im  0.354164+0.0im  0.354164+0.0im  0.227266+0.0im
 0.145836+0.0im  0.227266+0.0im  0.227266+0.0im  0.145836+0.0im
```

!!! note "Bring your own fixed-point solution!"
    Note that fixed-point calculations can be tricky, and the search algorithm can have a huge impact in convergence (if the problem converges at all!). For this reason, Quantica.jl does not provide built-in fixed-point routines, only the functionality to write functions such as `ρf` above.

!!! tip "Sparse mean fields"
    The method explained above to build a Hamiltonian supports all the `SiteSelector` and `HopSelector` functionality of conventional models. Therefore, although the density matrix computed above is dense, its application to the Hamiltonian is sparse: it only touches the onsite matrix elements. Likewise, we could for example use `@hopping!` with a finite `range` to apply a Fock mean field within a finite range. In the future we will support built-in Hartree-Fock model presets with adjustable sparsity.

# Wannier90 imports

A common way to obtain quantitative tight-binding models of materials is to *Wannierize* density-functional-theory (DFT) bandstructures. In a nutshell, this procedure consists in obtaining a basis set of a subspace of some DFT bandstructure, subject to the condition that the obtained states are maximally localized. A popular implementation of this algorithm is (Wannier90)[https://wannier.org]. Among other things, this tool produces output files that encode a tight-binding Hamiltonian for the material and the matrix elements of the position operator in the maximally localized Wannier basis.

Quantica.jl includes a function that can import Wannier90 tight-binding files. By default these files are 3D systems
```
julia> w = wannier90("wannier_tb.dat")
WannierBuilder{Float64,3} : 3-dimensional Hamiltonian builder of type Float64 from Wannier90 input
  cells      : 755
  elements   : 36388
  modifiers  : 0
```
In this case, however, the model in the "wannier_tb.dat" file is a 2D MoS2 crystal. We can project out all out-of-plane matrix elements by specifying the dimension with `dim`. We can also drop any Hamiltonian matrix element smaller than, say `htol = 1e-5`, and any position matrix element of norm smaller than `rtol = 1e-4`. This greatly simplifies the problem
```
julia> w = wannier90("wannier_tb.dat"; dim = 2, htol = 1e-5, rtol = 1e-4)
WannierBuilder{Float64,2} : 2-dimensional Hamiltonian builder of type Float64 from Wannier90 input
  cells      : 151
  elements   : 7510
  modifiers  : 0
```
This object can then be converted into a Hamiltonian `h` or a position operator `r`
```
julia> h = hamiltonian(w)
Hamiltonian{Float64,2,2}: Hamiltonian on a 2D Lattice in 2D space
  Bloch harmonics  : 151
  Harmonic size    : 10 × 10
  Orbitals         : [1]
  Element type     : scalar (ComplexF64)
  Onsites          : 10
  Hoppings         : 7500
  Coordination     : 750.0

julia> r = sites(w)
BarebonesOperator{2}: a simple collection of 2D Bloch harmonics
  Bloch harmonics  : 151
  Harmonic size    : 10 × 10
  Element type     : SVector{2, ComplexF64}
  Nonzero elements : 7408
```
Note that `r` is not of type `Hamiltonian`. The `BarebonesOperator` is a specially simple operator type that simply encodes a number of Bloch harmonics (matrices between different unit cells) of arbitrary element type. It supports only a subset of the funcitionality of `AbstractHamiltonian`s. In particular, it supports indexing:
```
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

julia> r[cellsites(SA[0,0], 1), cellsites(SA[0,0], 4)]
2-element SVector{2, ComplexF64} with indices SOneTo(2):
  2.4885857e-5 + 0.018543702im
 -0.0053425408 + 1.8808481e-5im
```

It is possible to modify the imported Wannier90 models using the full Quantica.jl machinery. For example, we can add any `AbstractModel` to the Wannier90 model upon import just by passing it as a second argument
```
julia> w = wannier90("wannier_tb.dat", @onsite((; Δ = 0) -> Δ); dim = 2)
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
    Although the above is the recommended way to add a Quantica model to a Wannier90 model (i.e. explicitly at import time), one can also do the same with `Quantica.add!(Quantica.hbuilder(w), model)` to modify `w` in place after its creation. This employs internal functionality, so it is not recommended, as it could change without warning.

We can also use the following syntax apply one or more modifiers explicitly
```
julia> w´ = w |> @onsite!((o; k = 0) -> o + k)
WannierBuilder{Float64,2} : 2-dimensional Hamiltonian builder of type Float64 from Wannier90 input
  cells      : 151
  elements   : 7560
  modifiers  : 2
```

An interesting application of modifiers is the addition of an electric field that couples to the full `r` operator. In an strict tight-binding limit, we would add an electric field `E` simply as an onsite potential
```
julia> hE = h |> @onsite!((o, r; E = SA[0,0]) -> o + E'*r);
```
However, we actually have the full `r` operator now, which includes non-diagonal matrix elements. We can then incorporate the electric field term `E'*r` more precisely. We can do so using the `-->` syntax and the indexing functionality of the `r::BarebonesOperator` that we obtained from Wannier90
```
julia> hE = h |> @onsite!((o, i; E = SA[0,0]) --> o + E'*r[i,i]) |> @hopping!((t, i, j; E = SA[0,0]) --> t + E'*r[i,j]);
```

!!! note "Closures over non-constant objects"
    Note that the above creates a closure over `r`, which is not `const`. As a result this would incur a small performance and allocation cost when evaluating `hE(E=...)`. We can avoid it e.g. by defining `r` as a constant, `const r = sites(w)`.
