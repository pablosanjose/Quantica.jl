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

julia> ρ0 = densitymatrix(g[])(0.5, 0) ## density matrix at chemical potential `µ=0.5` and temperature `kBT = 0`  on all sites
4×4 OrbitalSliceMatrix{Matrix{ComplexF64}}:
 0.138197+0.0im  0.223607+0.0im  0.223607+0.0im  0.138197+0.0im
 0.223607+0.0im  0.361803+0.0im  0.361803+0.0im  0.223607+0.0im
 0.223607+0.0im  0.361803+0.0im  0.361803+0.0im  0.223607+0.0im
 0.138197+0.0im  0.223607+0.0im  0.223607+0.0im  0.138197+0.0im

julia> hρ = h |> @onsite!((o, i; U = 0, ρ = ρ0) --> o + U * ρ[i])
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

!!! tip "Sparse vs dense"
    The method explained above to build a Hamiltonian `hρ` using `-->` supports all the `SiteSelector` and `HopSelector` functionality of conventional models. Therefore, although the density matrix computed above is dense, its application to the Hamiltonian is sparse: it only touches the onsite matrix elements in this case. Likewise, we could for example use `@hopping!` with a finite `range` to apply a Fock mean field within a finite range.

The above provides the tools to implement self-consistent mean field problems in Quantica. The problem consists of finding a fixed point `f(h) = h` of the function
```julia
function f(h)
    g = greenfunction(h, GS.Spectrum())
    ρ = densitymatrix(g[])(0.5, 0)
    return hρ(; ρ)
end
```
that takes a Hamiltonian `h`, computes `ρ` with it, and returns a new Hamiltonian `h = hρ(ρ)`. Its fixed point corresponds to the self-consistent Hamiltonian, and the corresponding `ρ` is the self-consistent mean-field density matrix.

In the following we show how to approach such as problem. We will employ an external library to solve generic fixed-point problems, and show how to make it work with Quantica AbstractHamiltonians efficiently. Many generic fixed-point solver backends exist. Here we use the SIAMFANLEquations.jl package. It provides a simple utility `aasol(f, x0)` that computes the solution of `f(x) = x` with initial condition `x0` using Anderson acceleration. This is an example of how it works to compute the fixed point of function `f(x) = 1 + atan(x)`
```julia
julia> using SIAMFANLEquations

julia> f!(x, x0) = (x .= 1 .+ atan.(x0))

julia> m = 3; x0 =  rand(3); vstore = rand(3, 3m+3);  # order m, initial condition x0, and preallocated space vstore

julia> aasol(f!, x0, m, vstore).solution
3-element Vector{Float64}:
 2.132267725272934
 2.132267725272908
 2.132267725284556
```
The package requires as input an in-place version `f!` of the function `f`, and the preallocation of some storage space vstore (see the `aasol` documentation). The package, as [a few others](https://docs.sciml.ai/NonlinearSolve/stable/solvers/fixed_point_solvers/), also requires the variable `x` and the initial condition `x0` to be (real) `AbstractArray` (or a scalar, but we need the former for our use case), hence the broadcast dots above. In our case we will therefore need to translate back and forth from a Hamiltonian `h` to a real vector `x` to pass it to `aasol`.

This translation is achieved with Quantica's `serializer` funcionality. A `s::Serializer{T}` is an object that takes an `h::AbstractHamiltonian`, a selection of the sites and hoppings to be translated, and an `encoder`/`decoder` pair of functions to translate each element. This `s` can then be used to convert the specified elements of `h` into a vector of scalars of type `T` and back, possibly after applying some parameter values. Consider this self-explanatory example from the `serializer` docstring
```julia
julia> h = LP.linear() |> hopping((r, dr) -> im*dr[1]) - @onsite((r; U = 2) -> U);

julia> s = serializer(Float64, h; encoder = s -> (real(s), imag(s)), decoder = v -> complex(v[1], v[2]))
Serializer{Float64} : encoder/decoder of matrix elements into a collection of scalars
  Object            : ParametricHamiltonian
  Output eltype     : Float64
  Encoder/Decoder   : Single
  Length            : 6

julia> v = serialize(s; U = 4)
6-element Vector{Float64}:
 -4.0
  0.0
 -0.0
 -1.0
  0.0
  1.0

julia> h´ = deserialize!(s, v);

julia> h´ == h(U = 4)
true
```
The serializer functionality is designed with efficiency in mind. Using the in-place `serialize!`/`deserialize!` pair we can do the encode/decode round trip without allocations
```
julia> using BenchmarkTools

julia> v = Vector{Float64}(undef, length(s));

julia> @btime deserialize!($s, serialize!($v, $s));
  149.737 ns (0 allocations: 0 bytes)
```
It also allows powerful compression into relevant degrees of freedom through appropriate use of encoders/decoders, see the docstring.

With this serializer functionality we can build a version of the fixed-point function `f` that operates on real vectors. Let's return to our original Hamiltonian
```julia
julia> h = LP.linear() |> onsite(2) - hopping(1) |> supercell(4) |> supercell;

julia> ρ0 = densitymatrix(greenfunction(h, GS.Spectrum())[])(0.5, 0)

julia> hρ = h |> @onsite!((o, i; U = 0, ρ = ρ0) --> o + U * ρ[i]);

julia> s = serializer(Float64, hρ, siteselector(); encoder = real, decoder = identity)
Serializer{Float64} : encoder/decoder of matrix elements into a collection of scalars
  Object            : ParametricHamiltonian
  Output eltype     : Float64
  Encoder/Decoder   : Single
  Length            : 4

julia> function f!(x, x0, (s, params))
        h = deserialize!(s, x0)
        g = greenfunction(h, GS.Spectrum())
        ρ = densitymatrix(g[])(0.5, 0)
        serialize!(x, s; ρ = ρ, params...)
        return x
    end;
```
Then we can proceed as in the `f(x) = 1 + atan(x)` example
```julia
julia> m = 2; len = length(s); x0 = rand(len); vstore = rand(len, 3m+3);  # order m, initial condition x0, and preallocated space vstore

julia> x = aasol(f!, x0, m, vstore; pdata = (s, (; U = 0.3))).solution
4-element Vector{Float64}:
 2.04319819150754
 2.1068018084891236
 2.1068018084820492
 2.0431981915212862

julia> h´ = deserialize!(s, x)
Hamiltonian{Float64,1,0}: Hamiltonian on a 0D Lattice in 1D space
  Bloch harmonics  : 1
  Harmonic size    : 4 × 4
  Orbitals         : [1]
  Element type     : scalar (ComplexF64)
  Onsites          : 4
  Hoppings         : 6
  Coordination     : 1.5

julia> h´[()]
4×4 SparseArrays.SparseMatrixCSC{ComplexF64, Int64} with 10 stored entries:
 2.0432+0.0im    -1.0+0.0im         ⋅             ⋅
   -1.0+0.0im  2.1068+0.0im    -1.0+0.0im         ⋅
        ⋅        -1.0+0.0im  2.1068+0.0im    -1.0+0.0im
        ⋅             ⋅        -1.0+0.0im  2.0432+0.0im
```
Note that the content of `pdata` is passed by `aasol` as a third argument to `f!`. We use this to pass the serializer `s` and `U` parameter to use.


!!! note "Bring your own fixed-point solver!"
    Note that fixed-point calculations can be tricky, and the search algorithm can have a huge impact in convergence (if the problem converges at all!). For this reason, Quantica.jl does not provide built-in fixed-point routines, only the functionality to write functions such as `f` above. Numerous packages exist for fixed-point computations in julia. Check [NonlinearSolve.jl](https://github.com/SciML/NonlinearSolve.jl) for one prominent metapackage.
