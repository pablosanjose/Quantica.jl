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
