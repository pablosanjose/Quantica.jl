
# Self-consistent mean-field problems

Here we show how to solve interacting-electron problems in Quantica, approximated at the mean field level. A mean field is a collection of onsite and hopping terms that are added to a given `h::AbstractHamiltonian`, that depend on the density matrix `ρ`. Since `ρ` itself depends on `h`, this defines a self-consistent problem.

If the mean field solution is dubbed `Φ`, the problem consists in finding a fixed point solution to the function `Φ = M(Φ)` for a certain function `M` that takes `Φ`, computes `h` with the added mean field onsite and hopping terms, computes the density matrix, and from that computes the new mean field `Φ`. To attack this problem we will employ non-spatial models and a new `meanfield` constructor.

Schematically the process is as follows:
- We start from an `AbstractHamiltonian` that includes a non-interacting model `model_0` and non-spatial model `model_1 + model_2` with a mean field parameter, e.g. `Φ`,
```julia
julia> model_0 = hopping(1); # a possible non-interacting model

julia> model_1 = @onsite((i; Φ = zerofield) --> Φ[i]);       # Onsite Hartree-Fock

julia> model_2 = @hopping((i, j; Φ = zerofield) -> Φ[i, j]); # Non-local Fock

julia> h = lat |> hamiltonian(model_0 + model_1 + model_2)
```
Here `model_1` corresponds to Hartree and onsite-Fock mean field terms, while `model_2` corresponds to inter-site Fock terms.
The default value `Φ = zerofield` is an singleton object that represents no interactions, so `model_1` and `model_2` vanish by default.
- We build the `GreenFunction` of `h` with `g = greenfunction(h, solver; kw...)` using the `GreenSolver` of choice
- We construct a `M::MeanField` object using `M = meanfield(g; potential = pot, other_options...)`

  Here `pot(r)` is the charge-charge interaction potential between electrons. We can also specify `hopselector` directives to define which sites interacts, adding e.g. `selector = (; range = 2)` to `other_options`, to make sites at distance `2` interacting. See `meanfield` docstring for further details.

- We evaluate this `M` with `Φ0 = M(µ, kBT; params...)`.

  This computes the density matrix at specific chemical potential `µ` and temperature `kBT`, and for specific parameters of `h` (possibly including `Φ`). Then it computes the appropriate Hartree and Fock terms, and stores them in the returned `Φ0::OrbitalSliceMatrix`, where `Φ0ᵢⱼ = δᵢⱼ hartreeᵢ + fockᵢⱼ`. In normal systems, these terms read

    $$\text{hartree}_i = Q \sum_k v_H(r_i-r_k) \text{tr}(\rho_{kk}Q)$$

    $$\text{fock}_{ij}  = -v_F(r_i-r_j) Q \rho_{ij} Q$$

  where `v_H` and `v_F` are Hartree and Fock charge-charge interaction potentials (by default equal to `pot`), and the charge operator is `Q` (equal to the identity by default, but can be changed to implement e.g. spin-spin interactions).

  When computing `Φ0` we don't specify `Φ` in `params`, so that `Φ0` is evaluated using the non-interacting model, hence its name.

- The self-consistent condition can be tackled naively by iteration-until-convergence,
```julia
Φ0 = M(µ, kBT; params...)
Φ1 = M(µ, KBT; Φ = Φ0, params...)
Φ2 = M(µ, KBT; Φ = Φ1, params...)
...
```
A converged solution `Φ`, if found, should satisfy the fixed-point condition

    Φ_sol ≈ M(µ, KBT; Φ = Φ_sol, params...)

Then, the self-consistent Hamiltonian is given by `h(; Φ = Φ_sol, params...)`.

The key problem is to actually find the fixed point of the `M` function. The naive iteration above is not optimal, and often does not converge. To do a better job we should use a dedicated fixed-point solver.

!!! note "Superconducting systems"
    Superconducting (Nambu) Hamiltonians obey the same equations for the Hartree and Fock mean fields, with a proper definition of `Q`, and an extra `1/2` coefficient in the Hartree trace, see the `meanfield` doctring.

!!! note "Interactions given in the form of a TightbindingModel"
    As explained in the `meanfield` docstring, we can also provide the interaction potential, both the `hartree` and the `fock` parts, as a non-parametric model, using the `onsite` and `hopping` functionality.

## Using an external fixed-point solver

We now show how to approach such a fixed-point problem. We will employ an external library to solve generic fixed-point problems, and show how to make it work with Quantica `MeanField` objects efficiently. Many generic fixed-point solver backends exist. Here we use the SIAMFANLEquations.jl package. It provides a simple utility `aasol(f, x0)` that computes the solution of `f(x) = x` with initial condition `x0` using Anderson acceleration. This is an example of how it works to compute the fixed point of function `f(x) = 1 + atan(x)`
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
The package requires as input an in-place version `f!` of the function `f`, and the preallocation of some storage space vstore (see the `aasol` documentation). The package, as [a few others](https://docs.sciml.ai/NonlinearSolve/stable/solvers/fixed_point_solvers/), also requires the variable `x` and the initial condition `x0` to be an `AbstractArray` (or a scalar, but we need the former for our use case), hence the broadcast dots above. In our case we will therefore need to translate back and forth from an `Φ::OrbitalSliceMatrix` to a real vector `x` to pass it to `aasol`. This translation is achieved using Quantica's `serialize`/`deserialize` funcionality.

## Using Serializers with fixed-point solvers

With the serializer functionality we can build a version of the fixed-point function `f` that operates on real vectors. Let's take a 1D Hamiltonian with a sawtooth potential, and build a Hartree mean field (note the `fock = nothing` keyword)
```julia
julia> using SIAMFANLEquations

julia> h = LP.linear() |> supercell(4) |> onsite(r->r[1]) - hopping(1) + @onsite((i; phi = zerofield) --> phi[i]);

julia> M = meanfield(greenfunction(h); onsite = 1, selector = (; range = 0), fock = nothing)
MeanField{ComplexF64} : builder of Hartree-Fock-Bogoliubov mean fields
  Charge type      : scalar (ComplexF64)
  Hartree pairs    : 4
  Mean field pairs : 4
  Nambu            : false

julia> Φ0 = M(0.0, 0.0);

julia> function f!(x, x0, (M, Φ0))
        Φ = M(0.0, 0.0; phi = deserialize(Φ0, x0))
        copy!(x, serialize(Φ))
        return x
    end;
```
Then we can proceed as in the `f(x) = 1 + atan(x)` example
```julia
julia> m = 2; x0 = serialize(Φ0); vstore = rand(length(x0), 3m+3);  # order m, initial condition x0, and preallocated space vstore

julia> x = aasol(f!, x0, m, vstore; pdata = (M, Φ0)).solution
4-element Vector{ComplexF64}:
  0.5658185030962436 + 0.0im
   0.306216109313951 + 0.0im
 0.06696362342872919 + 0.0im
 0.06100176416107613 + 0.0im

julia> h´ = h(; phi = deserialize(Φ0, x))
Hamiltonian{Float64,1,1}: Hamiltonian on a 1D Lattice in 1D space
  Bloch harmonics  : 3
  Harmonic size    : 4 × 4
  Orbitals         : [1]
  Element type     : scalar (ComplexF64)
  Onsites          : 4
  Hoppings         : 8
  Coordination     : 2.0

julia> h´[()]
4×4 SparseArrays.SparseMatrixCSC{ComplexF64, Int64} with 10 stored entries:
 0.565819+0.0im     -1.0+0.0im          ⋅            ⋅
     -1.0+0.0im  1.30622+0.0im     -1.0+0.0im        ⋅
          ⋅         -1.0+0.0im  2.06696+0.0im   -1.0+0.0im
          ⋅              ⋅         -1.0+0.0im  3.061+0.0im
```
Note that the content of `pdata` is passed by `aasol` as a third argument to `f!`. We use this to pass the serializer `s` and `U` parameter to use.

!!! note "Bring your own fixed-point solver!"
    Note that fixed-point calculations can be tricky, and the search algorithm can have a huge impact in convergence (if the problem converges at all!). For this reason, Quantica.jl does not provide built-in fixed-point routines, only the functionality to write functions such as `f` above. Numerous packages exist for fixed-point computations in julia. Check [NonlinearSolve.jl](https://github.com/SciML/NonlinearSolve.jl) for one prominent metapackage.

## GreenSolvers without support for ParametricHamiltonians

Some `GreenSolver`'s, like `GS.KPM`, do not support `ParametricHamiltonian`s. In such cases, the approach above will fail, since it will not be possible to build `g` before knowing `phi`. In such cases one would need to rebuild the `meanfield` object at each step of the fixed-point solver. This is one way to do it.

```julia
julia> using SIAMFANLEquations

julia> h = LP.linear() |> supercell(4) |> supercell |> onsite(1) - hopping(1) + @onsite((i; phi) --> phi[i]);

julia> M´(phi = zerofield) = meanfield(greenfunction(h(; phi), GS.Spectrum()); onsite = 1, selector = (; range = 0), fock = nothing)
M´ (generic function with 3 methods)

julia> Φ0 = M´()(0.0, 0.0);

julia> function f!(x, x0, (M´, Φ0))
        Φ = M´(deserialize(Φ0, x0))(0.0, 0.0)
        copy!(x, serialize(Φ))
        return x
           end;

julia> m = 2; x0 = serialize(Φ0); vstore = rand(length(x0), 3m+3);  # order m, initial condition x0, and preallocated space vstore

julia> x = aasol(f!, x0, m, vstore; pdata = (M´, Φ0)).solution
4-element Vector{ComplexF64}:
 0.15596283661234628 + 0.0im
 0.34403716338765444 + 0.0im
 0.34403716338765344 + 0.0im
 0.15596283661234572 + 0.0im
```
