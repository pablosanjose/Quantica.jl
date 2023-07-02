# GreenFunctions

Up to now we have seen how to define Lattices, Models, Hamiltonians and Bandstructures. Most problems require the computation of different physical observables for these objects, e.g. the local density of states or various transport coefficients. We reduce this general problem to the computation of the retarded Green function

``G^r_{ij}(\omega) = \langle i|(\omega-H-\Sigma(\omega))^{-1}|j\rangle``

where `i, j` are orbitals, `H` is the (possibly infinite) Hamiltonian matrix, and `Σ(ω)` is the self-energy coming from any coupling to other systems (typically described by their own `AbstractHamiltonian`).

We split the problem of computing `Gʳᵢⱼ(ω)` of a given `h::AbstractHamiltonian` into four steps:

1. Attach self-energies to `h` using the command `oh = attach(h, args...)`. This produces a new object `oh::OpenHamiltonian` with a number of `Contacts`, numbered `1` to `N`
2. Use `g = greenfunction(oh, solver)` to build a `g::GreenFunction` representing `Gʳ` (at arbitrary `ω` and `i,j`), where `oh::OpenHamiltonian` and `solver::GreenSolver` (see `GreenSolvers` below for available solvers)
3. Evaluate `gω = g(ω; params...)` at fixed energy `ω` and model parameters, which produces a `gω::GreenSolution`
4. Slice `gω[sᵢ, sⱼ]` or `gω[sᵢ] == gω[sᵢ, sᵢ]` to obtain `Gʳᵢⱼ(ω)` as a flat matrix, where `sᵢ, sⱼ` are either site selectors over sites spanning orbitals `i,j`, or integers denoting contacts, `1` to `N`.

!!! tip "GreenSlice vs. GreenSolution"
    The two last steps can be interchanged, by first obtaining a `gs::GreenSlice` with `gs = g[sᵢ, sⱼ]` and then obtaining the `Gʳᵢⱼ(ω)` matrix with `gs(ω; params...)`.

## A simple example

Here is a simple example of the Green function of a 1D lead with two sites per unit cell, a boundary at `cell = 0`, and with no attached self-energies for simplicity
```julia
julia> hlead = LP.square() |> supercell((1,0), region = r -> 0 <= r[2] < 2) |> hopping(1);

julia> glead = greenfunction(hlead, GreenSolvers.Schur(boundary = 0))
GreenFunction{Float64,2,1}: Green function of a Hamiltonian{Float64,2,1}
  Solver          : AppliedSchurGreenSolver
  Contacts        : 0
  Contact solvers : ()
  Contact sizes   : ()
  Hamiltonian{Float64,2,1}: Hamiltonian on a 1D Lattice in 2D space
    Bloch harmonics  : 3
    Harmonic size    : 2 × 2
    Orbitals         : [1]
    Element type     : scalar (ComplexF64)
    Onsites          : 0
    Hoppings         : 6
    Coordination     : 3.0

julia> gω = glead(0.2)  # we first fix energy to ω = 0.2
GreenSolution{Float64,2,1}: Green function at arbitrary positions, but at a fixed energy

julia> gω[cells = 1:2]  # we now ask for the Green function between orbitals in the first two unit cells to the righht of the boundary
4×4 Matrix{ComplexF64}:
   0.1-0.858258im    -0.5-0.0582576im  -0.48-0.113394im   -0.2+0.846606im
  -0.5-0.0582576im    0.1-0.858258im    -0.2+0.846606im  -0.48-0.113394im
 -0.48-0.113394im    -0.2+0.846606im   0.104-0.869285im   0.44+0.282715im
  -0.2+0.846606im   -0.48-0.113394im    0.44+0.282715im  0.104-0.869285im
```
Note that the result is a 4 x 4 matrix, because there are 2 orbitals (one per site) in each of the two unit cells. Note also that the Schur GreenSolver used here allows us to compute the Green function between distant cells with little overhead
```julia
julia> @time gω[cells = 1:2];
  0.000067 seconds (70 allocations: 6.844 KiB)

julia> @time gω[cells = (SA[10], SA[100000])];
  0.000098 seconds (229 allocations: 26.891 KiB)
```

## GreenSolvers

The currently implemented `GreenSolver`s (abbreviated as `GS`) are the following

- `GS.SparseLU()`

  For bounded (`L=0`) AbstractHamiltonians. Default for `L=0`.

  Uses a sparse `LU` factorization to compute the `⟨i|(ω - H - Σ(ω))⁻¹|j⟩` inverse.


- `GS.KPM(order = 100, bandrange = missing, kernel = I)`

  For bounded (`L=0`) Hamiltonians, and restricted to sites belonging to contacts (see the section on Contacts).

  It precomputes the Chebyshev momenta


- `GS.Schur(boundary = Inf)`

  For 1D (`L=1`) AbstractHamiltonians with only nearest-cell coupling. Default for `L=1`.

  Uses a deflating Generalized Schur (QZ) factorization of the generalized eigenvalue problem to compute the unit-cell self energies.
  The Dyson equation then yields the Green function between arbitrary unit cells, which is further dressed using a T-matrix approach if the lead has any attached self-energy.


!!! note "GS.Bands"
    In the near future we will also have `GS.Bands` as a general solver in lattice dimensions `L ∈ [1,3]`.

## Attaching Contacts

A self energy `Σ(ω)` acting of a finite set of sites of `h` (i.e. on a `LatticeSlice` of `lat = lattice(h)`) can be incorporated using the `attach` command. This defines a new Contact in `h`. The general syntax is `oh = attach(h, args...; sites...)`, where the `sites` directives define the Contact `LatticeSlice` (`lat[siteselector(; sites...)]`), and `args` can take a number of forms.

The supported `attach` forms are the following

- **Generic self-energy**

  `attach(h, gs::GreenSlice, coupling::AbstractModel; sites...)`

  This is the generic form of `attach`, which couples some sites `i` of a `g::Greenfunction` (defined by the slice `gs = g[i]`), to `sites` of `h` using a `coupling` model. This results in a self-energy `Σ(ω) = V´⋅gs(ω)⋅V` on `h` `sites`, where `V` and `V´` are couplings matrices given by `coupling`.


- **Dummy self-energy**

  `attach(h, nothing; sites...)`

  This form merely defines a new contact on the specified `sites`, but  adds no actual self-energy to it. It is meant as a way to refer to some sites of interest using the `g[i::Integer]` slicing syntax for `GreenFunction`s, where `i` is the contact index.


- **Model self-energy**

  `attach(h, model::AbstractModel; sites...)`

  This form defines a self-energy `Σᵢⱼ(ω)` in terms of `model`, which must be composed purely of parametric terms (`@onsite` and `@hopping`) that have `ω` as first argument, as in e.g. `@onsite((ω, r) -> Σᵢᵢ(ω, r))` or `@hopping((ω, r, dr) -> Σᵢⱼ(ω, r, dr))`. This is a modellistic approach, wherein the self-energy is not computed from the properties of another `AbstractHamiltonian`, but rather has an arbitrary form defined by the user.


- **Matched lead self-energy**

  `attach(h, glead::GreenFunction; reverse = false, transform = identity, sites...)`

  Here `glead` is a GreenFunction of a 1D lead, possibly with a boundary.

  With this syntax `sites` must select a number of sites in `h` whose position match (after applying `transform` to them and modulo an arbitrary displacement) the sites in the unit cell of `glead`. Then, the coupling between these and the first unit cell of `glead` on the positive side of the boundary will be the same as between `glead` unitcells, i.e. `V = hlead[(1,)]`, where `hlead = hamiltonian(glead)`.

  If `reverse == true`, the lead is reversed before being attached, so that h is coupled through `V = hlead[(-1,)]` to the first unitcell on the negative side of the boundary. If there is no boundary, the `cell = 0` unitcell of the `glead` is used.


- **Generic lead self-energy**

  `attach(h, glead::GreenFunction, model::AbstractModel; reverse = false, transform = identity, sites...)`

  The same as above, but without any restriction on `sites`. The coupling between these and the first unit cell of `glead` (transformed by `transform`) is constructed using `model::TightbindingModel`. The "first unit cell" is defined as above.


## A more advanced example

Let us define the classical example of a multiterminal mesoscopic junction. We choose a square lattice, and a circular central region of radius `10`, with four leads of width `5` coupled to it at right angles.

We first define a single lead `Greenfunction` and the central Hamiltonian
```julia
julia> glead = LP.square() |> hopping(1) |> supercell((1, 0), region = r -> abs(r[2]) <= 5/2) |> greenfunction(GS.Schur(boundary = 0));

julia> hcentral = LP.square() |> hopping(1) |> supercell(region = RP.circle(10) | RP.rectangle((22, 5)) | RP.rectangle((5, 22)))
```
The two rectangles overlayed on the circle above create the stubs where the leads will be attached:
```@raw html
<img src="../../assets/central.png" alt="Central region with stubs" width="250" class="center"/>
```

We now attach `glead` four times using the `Matched lead` syntax
```julia
julia> Rot = r -> SA[0 -1; 1 0] * r;  # 90º rotation function

julia> gcentral = hcentral |>
    attach(glead, region = r -> r[1] ==  11) |>
    attach(glead, region = r -> r[1] == -11, reverse = true) |>
    attach(glead, region = r -> r[2] ==  11, transform = Rot) |>
    attach(glead, region = r -> r[2] == -11, reverse = true, transform = Rot) |>
    greenfunction
OpenHamiltonian{Float64,2,0}: Hamiltonian with a set of open contacts
  Number of contacts : 4
  Contact solvers    : (:SelfEnergySchurSolver, :SelfEnergySchurSolver, :SelfEnergySchurSolver, :SelfEnergySchurSolver)
  Hamiltonian{Float64,2,0}: Hamiltonian on a 0D Lattice in 2D space
    Bloch harmonics  : 1
    Harmonic size    : 353 × 353
    Orbitals         : [1]
    Element type     : scalar (ComplexF64)
    Onsites          : 0
    Hoppings         : 1320
    Coordination     : 3.73938

julia> qplot(g, children = (; selector = siteselector(; cells = 1:5), sitecolor = :blue))
```
```@raw html
<img src="../../assets/multiterminal.png" alt="Multiterminal system" width="300" class="center"/>
```

Note that since we did not specify the `solver` in `greenfunction`, the `L=0` default `GS.SparseLU()` was taken.

!!! tip "The GreenFunction <-> AbstractHamiltonian relation"
    Its important un appreciate that a `g::GreenFunction` represents the retarded Green function between sites in a given `AbstractHamiltonian`, but not on sites of the coupled `AbstractHamiltonians` of its attached self-energies. Therefore, `gcentral` above cannot yield observables in the leads (blue sites above), only on the red sites. To obtain observables in a given lead, its `GreenFunction` must be constructed, with an attached self-energy coming from the central region plus the other three leads.
