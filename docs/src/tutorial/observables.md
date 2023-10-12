# Observables

We are almost at our destination now. We have defined a `Lattice`, a `Model` for our system, we applied the `Model` to the `Lattice` to obtain a `Hamiltonian` or a `ParametricHamiltonian`, and finally, after possibly attaching some contacts to outside reservoirs and specifying a `GreenSolver`, we obtained a `GreenFunction`. It is now time to use the `GreenFunction` to obtain some observables of interest.

Currently, we have the following observables built into Quantica.jl (with more to come in the future)
    - `ldos`: computes the local density of states at specific energy and sites
    - `current`: computes the local current density along specific directions, and at specific energy and sites
    - `transmission`: computes the total transmission between contacts
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
<img src="../../assets/star_shape_ldos.png" alt="LDOS" width="400" class="center"/>
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
<img src="../../assets/star_shape_current.png" alt="Current density with magnetic flux" width="400" class="center"/>
```

!!! note "Remember to construct supercell before applying position-dependent fields"
    Note that we built the supercell before applying the model with the magnetic flux. Not doing so would make the gauge field be repeated in each unit cell when expanding the supercell. This was mentioned in the section on Hamiltonians, and is a common mistake when modeling systems with position dependent fields.

## Transmission

The transmission `Tᵢⱼ` from contact `j` to contact `i` can be computed using `transmission`. This function accepts a `GreenSlice` between the contact. Let us recover the four-terminal setup of the preceding section, but let's make it bigger this time
```julia
julia> hcentral = LP.square() |> onsite(4) - hopping(1) |> supercell(region = RP.circle(100) | RP.rectangle((202, 50)) | RP.rectangle((50, 202)))

julia> glead = LP.square() |> onsite(4) - hopping(1) |> supercell((1, 0), region = r -> abs(r[2]) <= 50/2) |> greenfunction(GS.Schur(boundary = 0));

julia> Rot = r -> SA[0 -1; 1 0] * r;  # 90º rotation function

julia> g = hcentral |>
           attach(glead, region = r -> r[1] ==  101) |>
           attach(glead, region = r -> r[1] == -101, reverse = true) |>
           attach(glead, region = r -> r[2] ==  101, transform = Rot) |>
           attach(glead, region = r -> r[2] == -101, reverse = true, transform = Rot) |>
           greenfunction;

julia> gx1 = sum(abs2, g(0.04)[siteselector(), 1], dims = 2);

julia> qplot(hcentral, hide = :hops, siteoutline = 1, sitecolor = (i, r) -> gx1[i], siteradius = (i, r) -> gx1[i], minmaxsiteradius = (0, 2), sitecolormap = :balance)
```
```@raw html
<img src="../../assets/four_terminal_g_big.png" alt="Green function from right lead" width="400" class="center"/>
```

It's apparent from the plot that the transmission from right to left (`T₂₁` here) at this energy of `0.04` is larger than from right to top (`T₃₁`). Is this true in general? Let us compute the two transmissions as a function of energy. To show the progress of the calculation we can use a monitor package, such as `ProgressMeter`
```julia
julia> using ProgressMeter

julia> T₂₁ = transmission(g[2,1]); T₃₁ = transmission(g[3,1]); ωs = subdiv(0, 4, 200);

julia> T₂₁ω = @showprogress [T₂₁(ω) for ω in ωs]; T₃₁ω = @showprogress [T₃₁(ω) for ω in ωs];
Progress: 100%|██████████████████████████████████████████████████████████████| Time: 0:01:05
Progress: 100%|██████████████████████████████████████████████████████████████| Time: 0:01:04

julia> f = Figure(); a = Axis(f[1,1], xlabel = "ω/t", ylabel = "T(ω)"); lines!(a, ωs, T₂₁ω, label = L"T_{2,1}"); lines!(a, ωs, T₃₁ω, label = L"T_{3,1}"); axislegend("Transmission", position = :lt); f
```
```@raw html
<img src="../../assets/four_terminal_T.png" alt="Total transmission from right contact" width="400" class="center"/>
```

!!! note "Total transmission vs transmission probability"
    Note that `transmission` gives the total transmission, which is the sum of the transmission probability from each orbital in the source contact to any other orbital in the drain contact. As such it is not normalized to 1, but to the number of source orbitals. It also gives the local conductance from a given contact in units of $$e^2/h$$ according to the Landauer formula, $$G\_j = e^2/h \sum_i T_{ij}(eV)$$.

## Conductance

## Josephson
