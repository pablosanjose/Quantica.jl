############################################################################################
# plotlattice recipe
#region

@recipe(PlotLattice) do scene
    Theme(
        flat = true,
        force_transparency = false,
        shellopacity = 0.07,
        cellopacity = 0.03,
        cellcolor = RGBAf(0,0,1),
        boundarycolor = RGBAf(1,0,0),
        boundaryopacity = 0.07,
        sitecolor = missing,         # accepts (i, r) -> float, IndexableObservable
        siteopacity = missing,       # accepts (i, r) -> float, IndexableObservable
        minmaxsiteradius = (0.0, 0.5),
        siteradius = 0.2,            # accepts (i, r) -> float, IndexableObservable
        siteradiusfactor = 1.0,      # independent multiplier to apply to siteradius
        siteoutline = 2,
        siteoutlinedarken = 0.6,
        sitedarken = 0.0,
        sitecolormap = :Spectral_9,
        hopcolor = missing,          # accepts ((i,j), (r,dr)) -> float, IndexableObservable
        hopopacity = missing,        # accepts ((i,j), (r,dr)) -> float, IndexableObservable
        minmaxhopradius = (0.0, 0.1),
        hopradius = 0.01,            # accepts ((i,j), (r,dr)) -> float, IndexableObservable
        hopdarken = 0.85,
        hopcolormap = :Spectral_9,
        hoppixels = 2,
        selector = missing,
        hide = :cell,               # :hops, :sites, :bravais, :cell, :axes, :shell, :boundary, :contacts, :all
        isAxis3 = false,            # for internal use, to fix marker scaling
        marker = :auto,
        children = missing
    )
end

#endregion

############################################################################################
# qplot
#region

function Quantica.qplot(h::PlotLatticeArgumentType;
    fancyaxis = true, axis = axis_defaults(h, fancyaxis), figure = user_default_figure, inspector = true, plotkw...)
    fig, ax = empty_fig_axis(h; fancyaxis, axis, figure)
    plotkw´ = (isAxis3 = ax isa Axis3, inspector, plotkw...)   # isAxis3 necessary to fix marker scaling
    plotlattice!(ax, h; plotkw´...)
    inspector && DataInspector(; default_inspector..., user_default_inspector...)
    return fig
end

Quantica.qplot!(x::PlotLatticeArgumentType; kw...) = plotlattice!(x; kw...)

#endregion

############################################################################################
# PlotLattice Primitives
#region

struct SitePrimitives{E}
    centers::Vector{Point{E,Float32}}
    indices::Vector{Int}
    hues::Vector{Float32}
    opacities::Vector{Float32}
    radii::Vector{Float32}
    tooltips::Vector{String}
    colors::Vector{RGBAf}
end

struct HoppingPrimitives{E}
    centers::Vector{Point{E,Float32}}
    vectors::Vector{Vec{E,Float32}}
    indices::Vector{Tuple{Int,Int}}
    hues::Vector{Float32}
    opacities::Vector{Float32}
    radii::Vector{Float32}
    tooltips::Vector{String}
    colors::Vector{RGBAf}
end

const ColorOrSymbol = Union{Symbol, Makie.Colorant}
const ColorsPerSublat = Union{NTuple{<:Any,ColorOrSymbol}, AbstractVector{<:ColorOrSymbol}}

#region ## Constructors ##

SitePrimitives{E}() where {E} =
    SitePrimitives(Point{E,Float32}[], Int[], Float32[], Float32[], Float32[], String[], RGBAf[])

HoppingPrimitives{E}() where {E} =
    HoppingPrimitives(Point{E,Float32}[], Vec{E,Float32}[], Tuple{Int,Int}[], Float32[], Float32[], Float32[], String[], RGBAf[])

function siteprimitives(ls, h, plot, is_shell)   # function barrier
    opts = (plot[:sitecolor][], plot[:siteopacity][], plot[:shellopacity][], plot[:siteradius][])
    opts´ = maybe_evaluate_observable.(opts, Ref(ls))
    return _siteprimitives(ls, h, opts´, is_shell)
end

function _siteprimitives(ls::LatticeSlice{<:Any,E}, h, opts, is_shell) where {E}
    sp = SitePrimitives{E}()
    lat = parent(ls)
    for (i´, cs) in enumerate(Quantica.cellsites(ls))
        ni = Quantica.cell(cs)
        i = Quantica.siteindex(cs)
        # in case opts contains some array over latslice (from an observable)
        opts´ = maybe_getindex.(opts, i´)
        push_siteprimitive!(sp, opts´, lat, i, ni, view(h, cs), is_shell)
    end
    return sp
end

function hoppingprimitives(ls, ls´, h, siteradii, plot)   # function barrier
    opts = (plot[:hopcolor][], plot[:hopopacity][], plot[:shellopacity][], plot[:hopradius][], plot[:flat][])
    opts´ = maybe_evaluate_observable.(opts, Ref(ls))
    return _hoppingprimitives(ls, ls´, h, opts´, siteradii)
end

function _hoppingprimitives(ls::LatticeSlice{<:Any,E}, ls´, h, opts, siteradii) where {E}
    hp = HoppingPrimitives{E}()
    hp´ = HoppingPrimitives{E}()
    lat = parent(ls)
    sdict = Quantica.siteindexdict(ls)
    sdict´ = Quantica.siteindexdict(ls´)
    for (j´, csj) in enumerate(Quantica.cellsites(ls))
        nj = Quantica.cell(csj)
        j = Quantica.siteindex(csj)
        siteradius = isempty(siteradii) ? Float32(0) : siteradii[j´]
        for har in harmonics(h)
            dn = Quantica.dcell(har)
            ni = nj + dn
            mat = Quantica.matrix(har)
            rows = rowvals(mat)
            for ptr in nzrange(mat, j)
                i = rows[ptr]
                Quantica.isonsite((i, j), dn) && continue
                csi = sites(ni, i)
                i´ = get(sdict, csi, nothing)
                if i´ === nothing   # dst is not in latslice
                    if haskey(sdict´, csi)  # dst is in latslice´
                        opts´ = maybe_getindex.(opts, j´)
                        push_hopprimitive!(hp´, opts´, lat, (i, j), (ni, nj), siteradius, view(h, csi, csj), false)
                    end
                else
                    opts´ = maybe_getindex.(opts, i´, j´)
                    push_hopprimitive!(hp, opts´, lat, (i, j), (ni, nj), siteradius, view(h, csi, csj), true)
                end
            end
        end
    end
    return hp, hp´
end

maybe_evaluate_observable(o::Quantica.IndexableObservable, ls) = o[ls]
maybe_evaluate_observable(x, ls) = x

maybe_getindex(v::AbstractVector{<:Number}, i) = v[i]
maybe_getindex(m::AbstractMatrix{<:Number}, i) = sum(view(m, i, :))
maybe_getindex(m::Quantica.AbstractSparseMatrixCSC{<:Number}, i) = sum(view(nonzeros(m), nzrange(m, i)))
maybe_getindex(v, i) = v
maybe_getindex(v::AbstractVector{<:Number}, i, j) = 0.5*(v[i] + v[j])
maybe_getindex(m::AbstractMatrix{<:Number}, i, j) = m[i, j]
maybe_getindex(v, i, j) = v

## push! ##

# sitecolor here could be a color, a symbol, a vector/tuple of either, a number, a function, or missing
function push_siteprimitive!(sp, (sitecolor, siteopacity, shellopacity, siteradius), lat, i, ni, matii, is_shell)
    r = Quantica.site(lat, i, ni)
    s = Quantica.sitesublat(lat, i)
    push!(sp.centers, r)
    push!(sp.indices, i)
    push_sitehue!(sp, sitecolor, i, r, s)
    push_siteopacity!(sp, siteopacity, shellopacity, i, r, is_shell)
    push_siteradius!(sp, siteradius, i, r)
    push_sitetooltip!(sp, i, r, matii)
    return sp
end

push_sitehue!(sp, ::Union{Missing,ColorsPerSublat}, i, r, s) = push!(sp.hues, s)
push_sitehue!(sp, sitecolor::Real, i, r, s) = push!(sp.hues, sitecolor)
push_sitehue!(sp, sitecolor::Function, i, r, s) = push!(sp.hues, sitecolor(i, r))
push_sitehue!(sp, ::Symbol, i, r, s) = push!(sp.hues, 0f0)
push_sitehue!(sp, ::Makie.Colorant, i, r, s) = push!(sp.hues, 0f0)
push_sitehue!(sp, sitecolor, i, r, s) = argerror("Unrecognized sitecolor")

push_siteopacity!(sp, ::Missing, shellopacity, i, r, is_shell) = push!(sp.opacities, is_shell ? 1.0 : shellopacity)
push_siteopacity!(sp, ::Missing, ::Missing, i, r, is_shell) = push!(sp.opacities, 1.0)
push_siteopacity!(sp, siteopacity::Real, shellopacity, i, r, is_shell) = push!(sp.opacities, is_shell ? siteopacity : shellopacity)
push_siteopacity!(sp, siteopacity::Function, shellopacity, i, r, is_shell) = push!(sp.opacities, is_shell ? siteopacity(i, r) : shellopacity)
push_siteopacity!(sp, siteopacity, shellopacity, i, r, is_shell) = argerror("Unrecognized siteradius")

push_siteradius!(sp, siteradius::Real, i, r) = push!(sp.radii, siteradius)
push_siteradius!(sp, siteradius::Function, i, r) = push!(sp.radii, siteradius(i, r))
push_siteradius!(sp, siteradius, i, r) = argerror("Unrecognized siteradius")

push_sitetooltip!(sp, i, r, mat) = push!(sp.tooltips, matrixstring(i, mat))
push_sitetooltip!(sp, i, r) = push!(sp.tooltips, positionstring(i, r))

# hopcolor here could be a color, a symbol, a vector/tuple of either, a number, a function, or missing
function push_hopprimitive!(hp, (hopcolor, hopopacity, shellopacity, hopradius, flat), lat, (i, j), (ni, nj), radius, matij, is_shell)
    src, dst = Quantica.site(lat, j, nj), Quantica.site(lat, i, ni)
    # If end site is opaque (not in outer shell), dst is midpoint, since the inverse hop will be plotted too
    # otherwise it is shifted by radius´ = radius minus hopradius correction if flat = false, and src also
    radius´ = flat ? radius : sqrt(max(0, radius^2 - hopradius^2))
    unitvec = normalize(dst - src)
    dst = is_shell ? (src + dst)/2 : dst - unitvec * radius´
    src = src + unitvec * radius´
    r, dr = (src + dst)/2, (dst - src)
    sj = Quantica.sitesublat(lat, j)
    push!(hp.centers, r)
    push!(hp.vectors, dr)
    push!(hp.indices, (i, j))
    push_hophue!(hp, hopcolor, (i, j), (r, dr), sj)
    push_hopopacity!(hp, hopopacity, shellopacity, (i, j), (r, dr), is_shell)
    push_hopradius!(hp, hopradius, (i, j), (r, dr))
    push_hoptooltip!(hp, (i, j), matij)
    return hp
end

push_hophue!(hp, ::Union{Missing,ColorsPerSublat}, ij, rdr, s) = push!(hp.hues, s)
push_hophue!(hp, hopcolor::Real, ij, rdr, s) = push!(hp.hues, hopcolor)
push_hophue!(hp, hopcolor::Function, ij, rdr, s) = push!(hp.hues, hopcolor(ij, rdr))
push_hophue!(hp, ::Symbol, ij, rdr, s) = push!(hp.hues, 0f0)
push_hophue!(hp, ::Makie.Colorant, ij, rdr, s) = push!(hp.hues, 0f0)
push_hophue!(hp, hopcolor, ij, rdr, s) = argerror("Unrecognized hopcolor")

push_hopopacity!(hp, ::Missing, shellopacity, ij, rdr, is_shell) = push!(hp.opacities, is_shell ? 1.0 : shellopacity)
push_hopopacity!(hp, ::Missing, ::Missing, ij, rdr, is_shell) = push!(hp.opacities, 1.0)
push_hopopacity!(hp, hopopacity::Real, shellopacity, ij, rdr, is_shell) = push!(hp.opacities, hopopacity)
push_hopopacity!(hp, hopopacity::Function, shellopacity, ij, rdr, is_shell) = push!(hp.opacities, hopopacity(ij, rdr))
push_hopopacity!(hp, hopopacity, shellopacity, ij, rdr, is_shell) = argerror("Unrecognized hopradius")

push_hopradius!(hp, hopradius::Real, ij, rdr) = push!(hp.radii, hopradius)
push_hopradius!(hp, hopradius::Function, ij, rdr) = push!(hp.radii, hopradius(ij, rdr))
push_hopradius!(hp, hopradius, ij, rdr) = argerror("Unrecognized hopradius")

push_hoptooltip!(hp, (i, j), mat) = push!(hp.tooltips, matrixstring(i, j, mat))

#endregion

#region ## API ##

embdim(p::SitePrimitives{E}) where {E} = E
embdim(p::HoppingPrimitives{E}) where {E} = E

## update_color! ##

update_colors!(p::Union{SitePrimitives,HoppingPrimitives}, plot) =
    update_colors!(p, plot, safeextrema(p.hues), safeextrema(p.opacities))

update_colors!(p::SitePrimitives, plot, extremahues, extremaops) =
    update_colors!(p, extremahues, extremaops, plot[:sitecolor][], plot[:siteopacity][],
        Makie.ColorSchemes.colorschemes[plot[:sitecolormap][]], plot[:sitedarken][])

update_colors!(p::HoppingPrimitives, plot, extremahues, extremaops) =
    update_colors!(p, extremahues, extremaops, plot[:hopcolor][], plot[:hopopacity][],
        Makie.ColorSchemes.colorschemes[plot[:hopcolormap][]], plot[:hopdarken][])

function update_colors!(p, extremahues, extremaops, pcolor, popacity, colormap, pdarken)
    resize!(p.colors, length(p.hues))
    for (i, (c, α)) in enumerate(zip(p.hues, p.opacities))
        p.colors[i] = transparent(
            darken(primitive_color(c, extremahues, colormap, pcolor), pdarken),
            primitite_opacity(α, extremaops, popacity))
    end
    return p
end

# color == missing means sublat color
primitive_color(colorindex, extrema, colormap, ::Missing) =
    RGBAf(colormap[mod1(round(Int, colorindex), length(colormap))])
primitive_color(colorindex, extrema, colormap, colorname::Symbol) =
    parse_color(colorname)
primitive_color(colorindex, extrema, colormap, pcolor::Makie.Colorant) =
    parse_color(pcolor)
primitive_color(colorindex, extrema, colormap, colors::ColorsPerSublat) =
    parse_color(colors[mod1(round(Int, colorindex), length(colors))])
primitive_color(colorindex, extrema, colormap, _) =
    parse_color(colormap[normalize_range(colorindex, extrema)])

parse_color(colorname::Symbol) = parse(RGBAf, colorname)
parse_color(color::Makie.Colorant) = convert(RGBAf, color)

# opacity::Function should be scaled
primitite_opacity(α, extrema, ::Function) = normalize_range(α, extrema)
# otherwise (opacity::Union{Missing,Real}) we leave it fixed
primitite_opacity(α, extrema, _) = α

## update_radii! ##

update_radii!(p, plot) = update_radii!(p, plot, safeextrema(p.radii))

function update_radii!(p::SitePrimitives, plot, extremarads)
    siteradius = plot[:siteradius][]
    minmaxsiteradius = plot[:minmaxsiteradius][]
    return update_radii!(p, extremarads, siteradius, minmaxsiteradius)
end

function update_radii!(p::HoppingPrimitives, plot, extremarads)
    hopradius = plot[:hopradius][]
    minmaxhopradius = plot[:minmaxhopradius][]
    return update_radii!(p, extremarads, hopradius, minmaxhopradius)
end

function update_radii!(p, extremarads, radius, minmaxradius)
    for (i, r) in enumerate(p.radii)
        p.radii[i] = primitive_radius(normalize_range(r, extremarads), radius, minmaxradius)
    end
    return p
end

primitive_radius(normr, radius::Number, minmaxradius) = radius
primitive_radius(normr, radius, (minr, maxr)) = minr + (maxr - minr) * normr

## primitive_scales ##

function primitive_scales(p::HoppingPrimitives, plot)
    hopradius = plot[:hopradius][]
    minmaxhopradius = plot[:minmaxhopradius][]
    scales = Vec3f[]
    for (r, v) in zip(p.radii, p.vectors)
        push!(scales, primitive_scale(r, v, hopradius, minmaxhopradius))
    end
    return scales
end

primitive_scale(normr, v, hopradius::Number, minmaxhopradius) =
    Vec3f(hopradius, hopradius, norm(v)/2)

function primitive_scale(normr, v, hopradius, (minr, maxr))
    hopradius´ = minr + (maxr - minr) * normr
    return Vec3f(hopradius´, hopradius´, norm(v)/2)
end

## primitive_segments ##

function primitive_segments(p::HoppingPrimitives{E}, plot) where {E}
    segments = Point{E,Float32}[]
    for (r, dr) in zip(p.centers, p.vectors)
        push!(segments, r - dr/2)
        push!(segments, r + dr/2)
    end
    return segments
end

## primitive_linewidths ##

function primitive_linewidths(p::HoppingPrimitives{E}, plot) where {E}
    hoppixels = plot[:hoppixels][]
    hopradius = plot[:hopradius][]
    linewidths = Float32[]
    for r in p.radii
        linewidth = primitive_linewidth(r, hopradius, hoppixels)
        append!(linewidths, (linewidth, linewidth))
    end
    return linewidths
end

primitive_linewidth(normr, hopradius::Number, hoppixels) = hoppixels
primitive_linewidth(normr, hopradius, hoppixels) = hoppixels * normr

#endregion

#endregion

############################################################################################
# PlotLattice for different arguments
#region

function Makie.plot!(plot::PlotLattice{Tuple{L}}) where {L<:Lattice}
    lat = to_value(plot[1])
    h = Quantica.hamiltonian(lat)
    return plotlattice!(plot, h; plot.attributes...)
end

function Makie.plot!(plot::PlotLattice{Tuple{L}}) where {L<:LatticeSlice}
    ls = to_value(plot[1])
    lat = Quantica.parent(ls)
    h = Quantica.hamiltonian(lat)
    return plotlattice!(plot, h, ls; plot.attributes...)
end

function Makie.plot!(plot::PlotLattice{Tuple{L}}) where {L<:ParametricHamiltonian}
    ph = to_value(plot[1])
    h = ph()
    return plotlattice!(plot, h; plot.attributes...)
end

function Makie.plot!(plot::PlotLattice{Tuple{H}}) where {H<:Hamiltonian}
    h = to_value(plot[1])
    lat = Quantica.lattice(h)
    sel = sanitize_selector(plot[:selector][], lat)
    latslice = lat[sel]
    return plotlattice!(plot, h, latslice; plot.attributes...)
end

# For E < 2 Hamiltonians, promote to 2D
function Makie.plot!(plot::PlotLattice{Tuple{H}}) where {H<:Hamiltonian{<:Any,1}}
    h = Hamiltonian{2}(to_value(plot[1]))
    return plotlattice!(plot, h; plot.attributes...)
end

function Makie.plot!(plot::PlotLattice{Tuple{H,S}}) where {H<:Hamiltonian{<:Any,1},S<:LatticeSlice}
    h = Hamiltonian{2}(to_value(plot[1]))
    lat = Quantica.lattice(h)
    l = Quantica.unsafe_replace_lattice(to_value(plot[2]), lat)
    return plotlattice!(plot, h, l; plot.attributes...)
end

function Makie.plot!(plot::PlotLattice{Tuple{H,S,S´}}) where {H<:Hamiltonian{<:Any,1},S<:LatticeSlice,S´<:LatticeSlice}
    h = Hamiltonian{2}(to_value(plot[1]))
    lat = Quantica.lattice(h)
    l  = Quantica.unsafe_replace_lattice(to_value(plot[2]), lat)
    l´ = Quantica.unsafe_replace_lattice(to_value(plot[3]), lat)
    return plotlattice!(plot, h, l, l´; plot.attributes...)
end

function Makie.plot!(plot::PlotLattice{Tuple{H,S}}) where {H<:Hamiltonian,S<:LatticeSlice}
    h = to_value(plot[1])
    latslice = to_value(plot[2])
    latslice´ = Quantica.growdiff(latslice, h)
    return plotlattice!(plot, h, latslice, latslice´; plot.attributes...)
end

function Makie.plot!(plot::PlotLattice{Tuple{H,S,S´}}) where {H<:Hamiltonian,S<:LatticeSlice,S´<:LatticeSlice}
    h = to_value(plot[1])
    latslice = to_value(plot[2])    # selected sites
    latslice´ = to_value(plot[3])   # shell around sites
    lat = Quantica.lattice(h)

    # if e.g. `plot[:sitecolor][] == :hopcolor`, replace with `plot[:hopcolor][]`
    resolve_cross_references!(plot)

    hidesites = ishidden((:sites, :all), plot)
    hidehops = ishidden((:hops, :hoppings, :links, :all), plot)
    hidebravais = ishidden((:bravais, :all), plot)
    hideshell = ishidden((:shell, :all), plot) || iszero(Quantica.latdim(h))

    # plot bravais axes
    if !hidebravais
        plotbravais!(plot, lat, latslice)
    end

    # collect sites
    if !hidesites
        sp  = siteprimitives(latslice, h, plot, true)      # latslice
        if hideshell
            update_colors!(sp, plot)
            update_radii!(sp, plot)
        else
            sp´ = siteprimitives(latslice´, h, plot, false)    # boundary shell around latslice
            joint_colors_radii_update!(sp, sp´, plot)
        end
    end

    # collect hops
    if !hidehops
        radii = hidesites ? () : sp.radii
        hp, hp´ = hoppingprimitives(latslice, latslice´, h, radii, plot)
        if hideshell
            update_colors!(hp, plot)
            update_radii!(hp, plot)
        else
            joint_colors_radii_update!(hp, hp´, plot)
        end
    end

    # plot hops
    if !hidehops
        hopopacity = plot[:hopopacity][]
        transparency = plot[:force_transparency][] || has_transparencies(hopopacity)
        if !plot[:flat][]
            plothops_shaded!(plot, hp, transparency)
            hideshell || plothops_shaded!(plot, hp´, true)
        else
            plothops_flat!(plot, hp, transparency)
            hideshell || plothops_flat!(plot, hp´, true)
        end
    end

    # plot sites
    if !hidesites
        siteopacity = plot[:siteopacity][]
        transparency = plot[:force_transparency][] || has_transparencies(siteopacity)
        if !plot[:flat][]
            plotsites_shaded!(plot, sp, transparency)
            hideshell || plotsites_shaded!(plot, sp´, true)
        else
            plotsites_flat!(plot, sp, transparency)
            hideshell || plotsites_flat!(plot, sp´, true)
        end
    end

    return plot
end

function Makie.plot!(plot::PlotLattice{Tuple{G}}) where {G<:Union{GreenFunction,OpenHamiltonian}}
    g = to_value(plot[1])
    gsel = haskey(plot, :selector) && plot[:selector][] !== missing ?
        plot[:selector][] : green_selector(g)
    h = default_hamiltonian(g)
    latslice = lattice(h)[gsel]
    latslice´ = Quantica.growdiff(latslice, h)
    L = Quantica.latdim(h)
    squaremarker = !plot[:flat][] ? Rect3f(Vec3f(-0.5), Vec3f(1)) : Rect2

    # plot boundary as cells
    if !ishidden((:boundary, :boundaries), plot)
        bsel = boundary_selector(g)
        blatslice = latslice[bsel]
        blatslice´ = latslice´[bsel]
        if L > 1
            bkws = (; plot.attributes..., hide = (:axes, :sites, :hops), cellcolor = :boundarycolor, cellopacity = :boundaryopacity)
            isempty(blatslice) || plotlattice!(plot, blatslice; bkws...)
            isempty(blatslice´) || plotlattice!(plot, blatslice´; bkws...)
        end
        hideB = Quantica.tupleflatten(:bravais, plot[:hide][])
        bkws´ = (; plot.attributes..., hide = hideB, sitecolor = :boundarycolor, siteopacity = :boundaryopacity,
            siteradiusfactor = sqrt(2), marker = squaremarker)
        isempty(blatslice) || plotlattice!(plot, blatslice; bkws´...)
        isempty(blatslice´) || plotlattice!(plot, blatslice´; bkws´...)

    end

    # plot cells
    plotlattice!(plot, h, latslice, latslice´; plot.attributes...)

    # plot contacts
    if !ishidden((:contact, :contacts), plot)
        Σkws = Iterators.cycle(parse_children(plot[:children]))
        Σs = Quantica.selfenergies(g)
        hideΣ = Quantica.tupleflatten(:bravais, plot[:hide][])
        for (Σ, Σkw) in zip(Σs, Σkws)
            Σplottables = Quantica.selfenergy_plottables(Σ)
            for Σp in Σplottables
                plottables, kws = get_plottables_and_kws(Σp)
                plotlattice!(plot, plottables...; plot.attributes..., hide = hideΣ, marker = squaremarker, kws..., Σkw...)
            end
        end
    end
    return plot
end

parse_children(::Missing) = (NamedTuple(),)
parse_children(p::Tuple) = p
parse_children(p::NamedTuple) = (p,)
parse_children(p::Attributes) = parse_children(NamedTuple(p))
parse_children(p::Observable) = parse_children(p[])

sanitize_selector(::Missing, lat) = Quantica.siteselector(; cells = Quantica.zerocell(lat))
sanitize_selector(s::Quantica.SiteSelector, lat) = s
sanitize_selector(s, lat) = argerror("Specify a site selector with `selector = siteselector(; kw...)`")

function joint_colors_radii_update!(p, p´, plot)
    hext, oext = jointextrema(p.hues, p´.hues), jointextrema(p.opacities, p´.opacities)
    update_colors!(p, plot, hext, oext)
    update_colors!(p´, plot, hext, oext)
    rext = jointextrema(p.radii, p´.radii)
    update_radii!(p, plot, rext)
    update_radii!(p´, plot, rext)
    return nothing
end

############################################################################################
# green_selector and boundary_selector: sites plotted for green functions
#region

function green_selector(g)
    mincell, maxcell = green_bounding_box(g)
    s = siteselector(cells = n -> all(mincell .<= n .<= maxcell))
    return s
end

boundary_selector(g) = siteselector(cells = n -> isboundarycell(n, g))

function green_bounding_box(g)
    L = Quantica.latdim(hamiltonian(g))
    return isempty(Quantica.selfenergies(g)) ?
        boundary_bounding_box(Val(L), Quantica.boundaries(g)...) :
        broaden_bounding_box(Quantica.boundingbox(g), Quantica.boundaries(g)...)
end

function broaden_bounding_box((mincell, maxcell)::Tuple{SVector{N},SVector{N}}, (dir, cell), bs...) where {N}
    isfinite(cell) || return (mincell, maxcell)
    mincell´ = SVector(ntuple(i -> i == dir ? min(mincell[i], cell + 1) : mincell[i], Val(N)))
    maxcell´ = SVector(ntuple(i -> i == dir ? max(maxcell[i], cell - 1) : maxcell[i], Val(N)))
    return broaden_bounding_box((mincell´, maxcell´), bs...)
end

broaden_bounding_box(mm::Tuple) = mm

boundary_bounding_box(::Val{0}) = (SVector{0,Int}(), SVector{0,Int}())

function boundary_bounding_box(::Val{L}, (dir, cell), bs...) where {L}
    cell´ = isfinite(cell) ? cell + 1 : 0
    m = SVector(ntuple(i -> i == dir ? cell´ : 0, Val(L)))
    return broaden_bounding_box((m, m), bs...)
end

isboundarycell(n, g) = any(((dir, cell),) -> n[dir] == cell, Quantica.boundaries(g))

get_plottables_and_kws(Σp::Tuple) = (Σp, NamedTuple())
get_plottables_and_kws(Σp::Quantica.FrankenTuple) = (Tuple(Σp), NamedTuple(Σp))
get_plottables_and_kws(Σp) = ((Σp,), NamedTuple())

#endregion

############################################################################################
# core plotting methods
#region

function plotsites_shaded!(plot::PlotLattice, sp::SitePrimitives, transparency)
    inspector_label = (self, i, r) -> sp.tooltips[i]
    sizefactor = plot[:siteradiusfactor][]
    if plot[:marker][] == :auto  # circle markers
        marker = (;)
    else
        marker = (; marker = plot[:marker][])
        sizefactor *= sqrt(2)
    end
    markersize = sizefactor * sp.radii
    meshscatter!(plot, sp.centers; color = sp.colors, markersize, marker...,
            space = :data,
            transparency,
            inspector_label)
    return plot
end

function plotsites_flat!(plot::PlotLattice, sp::SitePrimitives{E}, transparency) where {E}
    inspector_label = (self, i, r) -> sp.tooltips[i]
    sizefactor = ifelse(plot[:isAxis3][] && plot[:flat][], 0.5, sqrt(2)) * plot[:siteradiusfactor][]

    if plot[:marker][] == :auto  # default circular markers
        marker = (;)
    else
        marker = (; marker = plot[:marker][])
        sizefactor *= 0.5
    end
    markersize = 2*sp.radii*sizefactor
    scatter!(plot, sp.centers;
        markersize, marker...,
        color = sp.colors,
        markerspace = :data,
        strokewidth = plot[:siteoutline][],
        strokecolor = darken.(sp.colors, Ref(plot[:siteoutlinedarken][])),
        transparency,
        inspector_label)
    return plot
end

function plothops_shaded!(plot::PlotLattice, hp::HoppingPrimitives, transparency)
    inspector_label = (self, i, r) -> hp.tooltips[i]
    scales = primitive_scales(hp, plot)
    cyl = Cylinder(Point3f(0., 0., -1), Point3f(0., 0, 1), Float32(1))
    vectors = Quantica.sanitize_SVector.(SVector{3,Float32}, hp.vectors) # convert to 3D
    meshscatter!(plot, hp.centers; color = hp.colors,
        rotation = vectors, markersize = scales, marker = cyl,
        transparency,
        inspector_label)
    return plot
end

function plothops_flat!(plot::PlotLattice, hp::HoppingPrimitives, transparency)
    inspector_label = (self, i, r) -> (hp.tooltips[(i+1) ÷ 2])
    colors = hp.colors
    segments = primitive_segments(hp, plot)
    linewidths = primitive_linewidths(hp, plot)
    linesegments!(plot, segments;
        space = :data,
        color = colors,
        linewidth = linewidths,
        transparency,
        inspector_label)
    return plot
end

function plotbravais!(plot::PlotLattice, lat::Lattice{<:Any,E,L}, latslice) where {E,L}
    iszero(L) && return plot
    bravais = Quantica.bravais(lat)
    vs = Point{E,Float32}.(Quantica.bravais_vectors(bravais))
    vtot = sum(vs)
    r0 = Point{E,Float32}(Quantica.mean(Quantica.sites(lat))) - 0.5 * vtot
    if !ishidden(:axes, plot)
        for (v, color) in zip(vs, (:red, :green, :blue))
            arrows!(plot, [r0], [v]; color, inspectable = false)
        end
    end
    if !ishidden((:cell, :cells), plot) && L > 1
        cellcolor = parse(RGBAf, plot[:cellcolor][])
        colface = transparent(cellcolor, plot[:cellopacity][])
        coledge = transparent(cellcolor, 5 * plot[:cellopacity][])
        rect = Rect{L,Int}(Point{L,Int}(0), Point{L,Int}(1))
        mrect0 = GeometryBasics.mesh(rect, pointtype=Point{L,Float32}, facetype=QuadFace{Int})
        vertices0 = mrect0.position
        mat = Quantica.bravais_matrix(bravais)
        for sc in Quantica.cellsdict(latslice)
            cell = Quantica.cell(sc)
            mrect = GeometryBasics.mesh(rect, pointtype=Point{E,Float32}, facetype=QuadFace{Int})
            vertices = mrect.position
            vertices .= Ref(r0) .+ Ref(mat) .* (vertices0 .+ Ref(cell))
            mesh!(plot, mrect; color = colface, transparency = true, shading = NoShading, inspectable = false)
            wireframe!(plot, mrect; color = coledge, transparency = true, linewidth = 1, inspectable = false)
        end
    end

    return plot
end

#endregion

############################################################################################
# tooltip strings
#region

positionstring(i, r) = string("Site[$i] : ", vectorstring(r))

function vectorstring(r::SVector)
    rs = repr("text/plain", r)
    pos = findfirst(isequal('\n'), rs)
    return pos === nothing ? rs : rs[pos:end]
end

matrixstring(row, x) = string("Onsite[$row] : ", matrixstring(x))
matrixstring(row, col, x) = string("Hopping[$row, $col] : ", matrixstring(x))

function matrixstring(s::AbstractMatrix)
    ss = repr("text/plain", s)
    pos = findfirst(isequal('\n'), ss)
    return pos === nothing ? ss : ss[pos:end]
end

numberstring(x) = isreal(x) ? string(" ", real(x)) : isimag(x) ? string(" ", imag(x), "im") : string(" ", x)

isreal(x) = all(o -> imag(o) ≈ 0, x)
isimag(x) = all(o -> real(o) ≈ 0, x)

#endregion

############################################################################################
# convert_arguments
#region

Makie.convert_arguments(::PointBased, lat::Lattice, sublat = missing) =
    (Point.(Quantica.sites(lat, sublat)),)
Makie.convert_arguments(p::PointBased, lat::Lattice{<:Any,1}, sublat = missing) =
    Makie.convert_arguments(p, Quantica.lattice(lat, dim = Val(2)), sublat)
Makie.convert_arguments(p::PointBased, h::Union{AbstractHamiltonian,GreenFunction,GreenSolution,OpenHamiltonian}, sublat = missing) =
    Makie.convert_arguments(p, Quantica.lattice(h), sublat)
Makie.convert_arguments(p::Type{<:LineSegments}, g::Union{GreenFunction,GreenSolution,OpenHamiltonian}, sublat = missing) =
    Makie.convert_arguments(p, Quantica.hamiltonian(g), sublat)

function Makie.convert_arguments(::Type{<:LineSegments}, h::AbstractHamiltonian{<:Any,E}, sublat = missing) where {E}
    segments = Point{max(E,2),Float32}[]
    lat = Quantica.lattice(h)
    for har in harmonics(h)
        colrng = sublat === missing ? axes(h, 2) : siterange(lat, sublat)
        mat = Quantica.unflat(har)
        dn = Quantica.dcell(har)
        for col in colrng, ptr in nzrange(mat, col)
            row = rowvals(mat)[ptr]
            row == col && continue
            append!(segments, (site(lat, col, zero(dn)), site(lat, row, dn)))
        end
    end
    return (segments,)
end

#endregion
