# Observables

We are almost at our destination now. We have defined a `Lattice`, a `Model` for our system, we applied the `Model` to the `Lattice` to obtain a `Hamiltonian` or a `ParametricHamiltonian`, and finally, after possibly attaching some contacts to outside reservoirs and specifying a `GreenSolver`, we obtained a `GreenFunction`. It is now time to use the `GreenFunction` to obtain some observables of interest.

Currently, we have the following observables built into Quantica.jl (with more to come in the future)
    - `ldos`: computes the local density of states at specific energy and sites
    - `current`: computes the local current density along specific directions, and at specific energy and sites
    - `transmission`: computes the transmission probability between contacts
    - `conductance`: computes the differential conductance `dIᵢ/dVⱼ` between contacts `i` and `j`
    - `josephson`: computes the supercurrent and the current-phase relation through a given contact in a superconducting system

See the corresponding docstrings for full usage instructions. Here we will present some basic examples

## Local density of states (LDOS)

Let us compute the LDOS in a cavity like in the previous section. Instead of computing the Green function between a contact to an arbitrary point, we can construct an object `ρ = ldos(g(ω))` without any contacts. By using a small imaginary part in `ω`, we broaden the discrete spectrum, and obtain a finite LDOS. Then, we can pass `ρ` directly as a site shader to `qplot`
```julia
julia> h = LP.square() |> onsite(4) - hopping(1) |> supercell(region = r -> norm(r) < 40*(1+0.2*cos(5*atan(r[2],r[1]))));

julia> g = h|> greenfunction;

julia> ρ = ldos(g(0.1 + 0.001im))
LocalSpectralDensitySolution{Float64} : local density of states at fixed energy and arbitrary location
  kernel   : LinearAlgebra.UniformScaling{Bool}(true)

julia> qplot(h, hide = :hops, sitecolor = ρ, siteradius = ρ, minmaxsiteradius = (0, 2), sitecolormap = :balance)
```
```@raw html
<img src="../../assets/star_shape_ldos.png" alt="LDOS" width="500" class="center"/>
```

Note that `ρ[sites...]` produces a vector with the LDOS at sites defined by `siteselector(; sites...)` (`ρ[]` is the ldos over all sites). We can also define a `kernel` to be traced over orbitals to obtain the spectral density of site-local observables (see `diagonal` slicing in the preceding section).

## Current

A similar computation can be done to obtain the current density, using `J = current(g(ω), direction = missing)`. This time `J[sᵢ, sⱼ]` yields a sparse matrix of current densities along a given direction for each hopping (or the current norm if `direction = missing`). Passing `J` as a hopping shader yields the equilibrium current in a system. In the above example we can add a magnetic flux to make this current finite
```julia
julia> h = LP.square() |> supercell(region = r -> norm(r) < 40*(1+0.2*cos(5*atan(r[2],r[1])))) |> onsite(4) - @hopping((r, dr; B = 0.1) -> cis(B * dr[1] * r[2]));

julia> g = h |> greenfunction;

julia> J = current(g(0.1; B = 0.01))
CurrentDensitySolution{Float64} : current density at a fixed energy and arbitrary location
  charge      : LinearAlgebra.UniformScaling{Int64}(-1)
  direction   : missing

julia> qplot(h, siteradius = 0.08, sitecolor = :black, siteoutline = 0, hopradius = J, hopcolor = J, minmaxhopradius = (0, 2), hopcolormap = :balance, hopdarken = 0)
```
```@raw html
<img src="../../assets/star_shape_current.png" alt="Current density with magnetic flux" width="500" class="center"/>
```
!!! note "Remember to construct supercell before applying position-dependent fields"
    Note that we built the supercell before applying the model with the magnetic flux. Not doing so would make the gauge field be repeated in each unit cell when expanding the supercell. This was mentioned in the section on Hamiltonians, and is a common mistake when modeling systems with position dependent fields.

## Transmission

## Conductance

## Josephson
