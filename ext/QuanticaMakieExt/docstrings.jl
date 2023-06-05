"""
    qplot(object; figure = (;), axis = (;), fancyaxis = true, inspector = false, plotkw...)

Render a plot of `object` using the `Makie` package. Supported `object`s and associated
specialized plot recipes:

- `object::Lattice`             -> `plotlattice` (supports also `LatticeSlice`s)
- `object::Hamiltonian`         -> `plotlattice` (supports also `ParametricHamiltonian`)
- `object::GreenFunction`       -> `plotlattice` (see below)
- `object::Bandstructure`       -> `plotbands`   (supports also slices of `Bandstructure`)
- `object::Subband`             -> `plotbands`   (supports also collections of `Subbands`)

Supported `Makie` backends include `GLMakie`, `CairoMakie`, `WGLMakie`, etc. Instead of
`using Makie`, load a specific backend directly, e.g. `using GLMakie`.

    qplot(g::GreenFunction; children = (plotkw₁::NamedTuple, plotkw₂::NamedTuple, ...), kw...)

Render the parent Hamiltonian of `g`, along with a representation of each of `g`'s
self-energies. Specific `plotkwᵢ` keywords for each self-energy can be given in `children`,
cycling over them if necessary.

## Keywords

- `figure`: keywords to pass onto the plot `Figure` (e.g. `resolution` or `fontsize`, see `Makie.Figure`)
- `axis`: keywords to pass on to the plot axis (see `Makie.LScene`, `Makie.Axis3` or `Makie.Axis` for options)
- `fancyaxis::Bool`: for 3D plots, whether to use `Makie.LScene` (supports zoom+pan) instead of `Makie.Axis3`
- `inspector::Bool`: whether to enable interactive tooltips of plot elements
- `plotkw`: additional keywords to pass on to `plotlattice` or `plotbands`, see their docstring for details.

"""
qplot

"""
    qplot!(object; plotkw...)

Render `object` on the currently active scene using either `plotlattice!` (for lattice-based
objects) or `plotbands!` (for bands-based object), and passing `plotkw` keywords along. See
their respective docstrings for possible keywords.

"""
qplot!

"""
    plotlattice(lat::Lattice; kw...)

Render the lattice unit cell and its Bravais vectors.

    plotlattice(lat::LatticeSlice; kw...)

Render a finite subset of sites in a lattice and its Bravais vectors.

    plotlattice(h::AbstractHamiltonian; kw...)

Render the lattice on which `h` is defined, together with all hoppings in the form of
inter-site links.

## Keywords

- `flat = true`: whether to render sites and hops as flat circles and lines, or as 3D meshes
- `force_transparency = false`: whether to disable occussion of all non-transparent elements
- `shellopacity = 0.07`: opacity of elements surrounding the unitcell or the set of selected sites (dubbed "shell")
- `cellopacity = 0.03`: opacity of the unitcell's boundingbox
- `cellcolor = RGBAf(0,0,1)`: color of the unitcell's boundingbox
- `sitecolor = missing`: color of sites, as a index in `sitecolormap`, a named color, a `Makie.Colorant`, or as a site shader (see below). If `missing`, cycle through `sitecolormap`.
- `sitecolormap = :Spectral_9`: colormap to use for `sitecolor` (see options in https://tinyurl.com/cschemes)
- `siteopacity = missing`: opacity of sites, as a real between 0 and 1, or as a site shader (see below). If `missing`, obey `shellopacity`.
- `siteradius = 0.25`: radius of sites as a real in units of lattice dimensions, or as a site shader (see below)
- `minmaxsiteradius = (0.0, 0.5)`: if `sitedradius` is a shader, minimum and maximum site radius.
- `siteoutline = 2`: thickness of the outline around flat sites
- `siteoutlinedarken = 0.6`: darkening factor of the outline around flat sites
- `sitedarken = 0.0`: darkening factor for sites
- `hopcolor = missing`: color of hops, as a index in `hopcolormap`, a named color, a `Makie.Colorant`, or as a hop shader (see below).If `missing`, cycle through `sitecolormap`.
- `hopcolormap = :Spectral_9`: colormap to use for `hopcolor` (see options in https://tinyurl.com/cschemes)
- `hopopacity = missing`: opacity of hops, as a real between 0 and 1, or as a hop shader (see below)
- `hopradius = 0.03`: radius of hops as a real number in units of lattice dimensions, or as a hop shader (see below)
- `hoppixels = 6`: if `flat = true` fixed hop linewidth in pixels, or maximum pixel linewidth if `hopradius` is a shader.
- `minmaxhopradius = (0, 0.1)`: if `hopdradius` is a shader, minimum and maximum hop radius.
- `hopdarken = 0.85`: darkening factor for hops
- `selector = missing`: an optional `siteselector(; sites...)` to filter which sites are shown (see `siteselector`)
- `pixelscalesites = 2√2`: for `flat = true`, conversion factor between `minmaxsiteradius` (in lattice space) and actual site radius (in pixel space)
- `hide = (:cell,)`: collection of elements to hide, to choose from `(:hops, :sites, :hops, :bravais, :cell, :axes, :shell, :all)`

## Shaders

The element properties in the list above that accept site shaders may take either of these options
- `(i, r) -> Real`: a real function of site index `i::Int` and site position `r::SVector`.
- `LocalSpectralDensitySolution`: a generator of local density of states at a fixed energy (see `ldos`). It is evaluated at the site position.
- `CurrentDensitySolution`: a generator of local current density at a fixed energy (see `current`). It is taken as the sum of currents along all hops connected to the site.

Element properties marked as accepting hop shaders may take either of these options
- `((src, dst), (r, dr)) -> Real`: a real function of site indices `(src, dst)` and hop coordinates `(r, dr)` (see `hopselector` for definition of hop coordinates)
- `LocalSpectralDensitySolution`: a generator of local density of states at a fixed energy (see `ldos`). It is evaluated as the average between connected sites.
- `CurrentDensitySolution`: a generator of local current density at a fixed energy (see `current`). It is taken as the current along the hop.

"""
plotlattice

"""
    plotlattice!(object; kw...)

Render lattice-based `object` on currently active scene. See `plotlattice` for possible
keywords `kw`

"""
plotlattice!

"""
    plotbands(b::Bandstructure; kw...)
    plotbands(s::Subband; kw...)
    plotbands(ss::AbstractVector{<:Subbands}; kw...)

Render a `Bandstructure` object, a single subband (e.g. `s = b[1]`) or a collection of
subbands or subband slices (e.g. `ss = b[1:4]` or `ss = b[(:,0,:)]`).

## Keywords

- `color = missing`: color of subbands, as a index in `colormap`, a named color, a `Makie.Colorant`, or as a band shader (see below). If `missing`, cycle through `colormap`.
- `colormap = :Spectral_9`: colormap to use for `color` (see options in https://tinyurl.com/cschemes)
- `opacity = 1.0`: opacity of subbands, as a real between 0 and 1 or as a band shader (see below)
- `size = 3`: stroke thickness, in pixels, when plotting line-like subbands (only relevant to 2D plots). May also be a band shader (see below)
- `minmaxsize = (0, 6)`: if `size` is a shader, minimum and maximum stroke thickness, in pixels.
- `nodesizefactor = 4`: relative size of nodes respect to subbands.
- `nodedarken = 0.0`: darkening factor of nodes despect to subband color.
- `hide = ()`: collection of elements to hide, to choose from `(:nodes, :bands)`

## Shaders
The element properties in the list above that accept band shaders may take either of these options
- `(ψ, ϵ, ϕs) -> Real`: a real function of the eigenstates `ψ::Matrix` in the subband, eigenenergy `ϵ`, and Bloch phases `ϕs::SVector`.

"""
plotbands

"""
    plotbands!(object; kw...)

Render bands-based `object` on currently active scene. See `plotbands` for possible keywords
`kw`

"""
plotbands!