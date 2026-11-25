# Non-allocating evaluation

Most Quantica objects are callable: `h(ϕs; params...)` returns a Bloch matrix, `gs(ω; params...)` a Green function matrix, `spectrum(h, ϕs; params...)` a `Spectrum`, and so on. Each such call returns a new, independent object. This is safe and convenient, but in tight loops (sweeps over Bloch phases or parameters, fixed-point iterations, etc.) the repeated allocations can become a significant overhead.

For these cases Quantica provides an unexported, in-place alternative, `Quantica.call!`, and a reusable eigensolver object, `Quantica.diagonalizer`.

## `Quantica.call!`

For several callable objects `x`, `Quantica.call!(x, args...; params...)` computes the same result as `x(args...; params...)`, but writes it into buffers preallocated inside `x`, and returns them without copying. The result is therefore *aliased* with `x`, and is overwritten by the next `call!` on `x`:
```julia
julia> h = LP.honeycomb() |> hamiltonian(hopping(1) + @onsite((; m = 0) -> m, sublats = :A) - @onsite((; m = 0) -> m, sublats = :B));

julia> m1 = Quantica.call!(h, (0, 0); m = 0.3); m2 = Quantica.call!(h, (π, 0); m = 0.3);

julia> m1 === m2  # the same matrix, now holding the Bloch matrix at (π, 0)
true

julia> h((0, 0); m = 0.3) === h((0, 0); m = 0.3)  # independent copies
false
```
In fact, `x(args...; params...)` is typically implemented as `copy(Quantica.call!(x, args...; params...))`. The main objects supporting `call!` are `AbstractHamiltonian`s, `GreenSlice`s, `ldos` slices, `Scattering` objects, `MeanField`s, `AppliedSerializer`s and the `diagonalizer` described below. See the `Quantica.call!` docstring for the full list.

!!! warning "Aliasing"
    Any result of `call!` should be consumed (or explicitly `copy`ed) before the next `call!` on the same object. Since `call!` mutates `x`, it is also not thread-safe: use a separate `x` per thread.

## Repeated diagonalizations with `Quantica.diagonalizer`

Each call to `spectrum(h, ϕs; params...)` sets up a new eigensolver. To diagonalize the same `h` many times, at different `ϕs` or `params`, build a `d = Quantica.diagonalizer(h; solver, transform)` once and reuse it,
```julia
julia> d = Quantica.diagonalizer(h)
AppliedEigenSolver{Float64}: diagonalizer for a 2D AbstractHamiltonian
  Solver          : LinearAlgebra
  Workspace       : allocation-free call! enabled
  Matrix size     : 2 × 2

julia> d((2π/3, -2π/3); m = 0.3) |> energies
2-element Vector{ComplexF64}:
 -0.3 + 0.0im
  0.3 + 0.0im
```
Here `d(ϕs; params...)` is equivalent to `spectrum(h, ϕs; solver, transform, params...)`, and returns an independent `Spectrum`. For a 0D `h`, use `d(; params...)`.

When `solver` supports it, `d` also holds a workspace (dense matrix, LAPACK workspaces from [FastLapackInterface.jl](https://github.com/DynareJulia/FastLapackInterface.jl) and output buffers), so that `Quantica.call!(d, ϕs; params...)` performs the diagonalization without any allocations. Currently this is the case for the default `EigenSolvers.LinearAlgebra()` without keyword arguments, and BLAS element types (`Float32` and `Float64`). Other solvers are supported too, but they allocate. As an example, this computes the minimum direct gap of a gapped graphene over a fine grid of Bloch phases, with no allocations inside the loop,
```julia
julia> function mingap(d, phis; params...)
           gap = Inf
           for ϕ1 in phis, ϕ2 in phis
               ϵs = energies(Quantica.call!(d, (ϕ1, ϕ2); params...))
               gap = min(gap, real(ϵs[2] - ϵs[1]))
           end
           return gap
       end;

julia> using BenchmarkTools

julia> @btime mingap($d, range(-π, π, 301); m = 0.3)
  27.926 ms (0 allocations: 0 bytes)
0.6
```
Note that each `Spectrum` returned by `call!(d, ...)` is overwritten by the next call, so it should be used (or copied) right away, as done above.

!!! tip "Bandstructures"
    `bands` already reuses one `Quantica.diagonalizer` per thread internally.
