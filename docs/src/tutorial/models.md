# Models

We now will show how to build a generic single-particle tight-binding model, with generic Hamiltonian

``H = \sum_{i\alpha j\beta} c_{i\alpha}^\dagger V_{\alpha\beta}(r_i, r_j)c_{j\alpha}``

Here, `α,β` are orbital indices in each site, `i,j` are site indices, and `rᵢ, rⱼ` are site positions. In Quantica.jl we would write the above model as

```julia
julia> model = onsite(r -> V(r, r)) + hopping((r, dr) -> V(r-dr/2, r+dr/2))
TightbindingModel: model with 2 terms
  OnsiteTerm{Function}:
    Region            : any
    Sublattices       : any
    Cells             : any
    Coefficient       : 1
  HoppingTerm{Function}:
    Region            : any
    Sublattice pairs  : any
    Cell distances    : any
    Hopping range     : Neighbors(1)
    Reverse hops      : false
    Coefficient       : 1
```
where `V(rᵢ, rⱼ)` is a function that returns a matrix (ideally an `SMatrix`) ``V_{\alpha\beta}(r_i, r_j)`` of the required orbital dimensionality.

Note that when writing models we distinguish between onsite (`rᵢ=rⱼ`) and hopping (`rᵢ≠rⱼ`) terms. For the former, `r` is the site position. For the latter we use a bond-center and bond-distance `(r, dr)` parametrization of `V`, so that `r₁, r₂ = r ∓ dr/2`

If the onsite and hopping amplitudes do not depend on position, we can simply use constants
```julia
julia> model = onsite(1) - 2*hopping(1)
TightbindingModel: model with 2 terms
  OnsiteTerm{Int64}:
    Region            : any
    Sublattices       : any
    Cells             : any
    Coefficient       : 1
  HoppingTerm{Int64}:
    Region            : any
    Sublattice pairs  : any
    Cell distances    : any
    Hopping range     : Neighbors(1)
    Reverse hops      : false
    Coefficient       : -2
```

!!! tip "Model term algebra"
    Note that we can combine model terms as in the above example by summing and subtracting them, and using constant coefficients.

## HopSelectors

By default `onsite` terms apply to any site in a Lattice, and `hopping` terms apply to any pair of sites within nearest-neighbor distance (see the `Hopping range: Neighbors(1)` above).

We can change this default by specifying a `SiteSelector` or `HopSelector` for each term. `SiteSelector`s where [already introduced](@ref siteselectors) to create `LatticeSlices`. `HopSelectors` are very similar, but support slightly different keywords:
- `region`: to restrict according to bond center `r` and bond vector `dr`
- `sublats`: to restrict source and target sublattices
- `dcells`: to restrict the distance in cell index
- `range`: to restrict the distance in real space

As an example, a `HopSelector` that selects any two sites at a distance between `1.0` and the second-nearest neighbor distance, with the first belonging to sublattice `:B` and the second to sublattice `:A`, and their bond center inside a unit circle
```julia
julia> hs = hopselector(range = (1.0, neighbors(2)), sublats = :B => :A, region = (r, dr) -> norm(r) < 1)
HopSelector: a rule that defines a finite collection of hops between sites in a lattice
  Region            : Function
  Sublattice pairs  : :B => :A
  Cell distances    : any
  Hopping range     : (1.0, Neighbors(2))
  Reverse hops      : false
```
We can now use this `HopSelector` to restrict the hoppings in a model, just as `SiteSelector`s can be used to restrict onsite terms
```julia
julia> model = plusadjoint(hopping(1, hs)) - 2*onsite(1, sublats = :B)
TightbindingModel: model with 3 terms
  HoppingTerm{Int64}:
    Region            : Function
    Sublattice pairs  : :B => :A
    Cell distances    : any
    Hopping range     : (1.0, Neighbors(2))
    Reverse hops      : false
    Coefficient       : 1
  HoppingTerm{Int64}:
    Region            : Function
    Sublattice pairs  : :B => :A
    Cell distances    : any
    Hopping range     : (1.0, Neighbors(2))
    Reverse hops      : true
    Coefficient       : 1
  OnsiteTerm{Int64}:
    Region            : any
    Sublattices       : B
    Cells             : any
    Coefficient       : 1
```
Note that we can pass the `Selector` itself as a second argument to `hopping` and `onsite`, or alternatively use selector keywords directly as in the `onsite` above.

!!! tip "plusadjoint function"
    The convenience function `plusadjoint(term) = term + term'` adds the Hermitian conjugate of its argument (`term'`), equivalent to the `+ h.c.` notation often used in the literature.

!!! note "Index-agnostic modeling"
    The Quantica.jl approach to defining tight-binding models does not rely explicitly on site indices (`i,j` above), since these are arbitrary, and may even be beyond the control of the user (for example after using `supercell`). Instead, we rely on physical properties of sites, such as position, distance or sublattice.

## Parametric Models

The models introduced above are non-parametric, in the sense that they encode fixed, numerical Hamiltonian matrix elements. In many problems, it is commonplace to have models that depend on a number of free parameters that will need to be adjusted during a calculation. For example, one may need to compute the phase diagram of a system as a function of a spin-orbit coupling or an applied magnetic field. For these cases, we have `ParametricModel`s.

Parametric models are defined with
- `@onsite((; params...) -> ...; sites...)`
- `@onsite((r; params...) -> ...; sites...)`
- `@hopping((; params...) -> ...; hops...)`
- `@hopping((r, dr; params...) -> ...; hops...)`

where the `params` keyword arguments define the free parameters, together with (optional) default values. Here is an example of a hopping model with a Peierls phase in the symmetric gauge, with the magnetic field `Bz` and the zero-field hopping `t` as free parameters
```julia
julia> model_perierls = @hopping((r, dr; Bz = 0, t = 1) -> t * cis(-im * Bz/2 * SA[-r[2], r[1], 0]' * dr))
ParametricModel: model with 1 term
  ParametricHoppingTerm{ParametricFunction{2}}
    Region            : any
    Sublattice pairs  : any
    Cell distances    : any
    Hopping range     : Neighbors(1)
    Reverse hops      : false
    Coefficient       : 1
    Argument type     : spatial
    Parameters        : [:Bz, :t]
```

One can linearly combine parametric and non-parametric models freely, omit parameter default values, and use any of the functional argument forms described for `onsite` and `hopping` (although not the constant argument form):
```julia
julia> model´ = 2 * (onsite(1) - 2 * @hopping((; t) -> t))
ParametricModel: model with 2 terms
  ParametricHoppingTerm{ParametricFunction{0}}
    Region            : any
    Sublattice pairs  : any
    Cell distances    : any
    Hopping range     : Neighbors(1)
    Reverse hops      : false
    Coefficient       : -4
    Argument type     : spatial
    Parameters        : [:t]
  OnsiteTerm{Int64}:
    Region            : any
    Sublattices       : any
    Cells             : any
    Coefficient       : 2
```

!!! tip "Non-spatial parametric models with -->"
    The `->` in the above parametric models `@onsite` and `@hopping`, but also in the modifiers below, can be changed to `-->`. This indicates that the function arguments are no longer treated as site or link positions `r` and `dr`, but as objects `i, j` representing destination and source sites. This allows to address sites directly instead of through their spatial location. See the Mean Field section for further details.


## Modifiers

There is a third model-related functionality known as `OnsiteModifier`s and `HoppingModifier`s. Given a model that defines a set of onsite and hopping amplitudes on a subset of sites and hops, one can define a parameter-dependent modification of a subset of said amplitudes. This is a useful way to introduce a new parameter dependence on an already defined model. Modifiers are built with
- `@onsite!((o; params...) -> new_onsite; sites...)`
- `@onsite!((o, r; params...) -> new_onsite; sites...)`
- `@hopping((t; params...) -> new_hopping; hops...)`
- `@hopping((t, r, dr; params...) -> new_hopping; hops...)`
where the first argument `o` and `t` is the unmodified value of the onsite or hopping amplitude, respectively. Here `sites` and `hops` are once more keyword arguments to restrict the modification with a `SiteSelector` or `HopSelector`.

For example, the following `HoppingModifier` inserts a Peierls phase on all the hopping in a given model
```julia
julia> model_perierls! = @hopping!((t, r, dr; B = 0) -> t * cis(-Bz/2 * SA[-r[2], r[1], 0]' * dr))
HoppingModifier{ParametricFunction{3}}:
  Region            : any
  Sublattice pairs  : any
  Cell distances    : any
  Hopping range     : Inf
  Reverse hops      : false
  Argument type     : spatial
  Parameters        : [:B]
```
The difference with `model_perierls` is that `model_perierls!` will never add any new hoppings. It will only modify previously existing hoppings in a model. Modifiers are not models themselves, and cannot be summed to other models. They are instead meant to be applied sequentially after applying a model.

In the next section we show how models and modifiers can be used in practice to construct Hamiltonians.

!!! note "Mind the `;`"
    While syntax like `onsite(2, sublats = :B)` and `onsite(2; sublats = :B)` are equivalent in Julia, due to the way keyword arguments are parsed, the same is not true for macro calls like `@onsite`, `@onsite!`, `@hopping` and `@hopping!`. These macros just emulate the function call syntax. But to work you must currently always use the `;` separator for keywords. Hence, something like `@onsite((; p) -> p; sublats = :B)` works, but `@onsite((; p) -> p, sublats = :B)` does not.
