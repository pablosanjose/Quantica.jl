
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
        sitecolor = missing,         # accepts (i, r) -> float, IndexableObservable
        siteopacity = missing,       # accepts (i, r) -> float, IndexableObservable
        minmaxsiteradius = (0.0, 0.5),
        siteradius = 0.25,           # accepts (i, r) -> float, IndexableObservable
        siteoutline = 2,
        siteoutlinedarken = 0.6,
        sitedarken = 0.0,
        sitecolormap = :Spectral_9,
        hopcolor = missing,          # accepts ((i,j), (r,dr)) -> float, IndexableObservable
        hopopacity = missing,        # accepts ((i,j), (r,dr)) -> float, IndexableObservable
        minmaxhopradius = (0.0, 0.1),
        hopradius = 0.03,            # accepts ((i,j), (r,dr)) -> float, IndexableObservable
        hopdarken = 0.85,
        hopcolormap = :Spectral_9,
        hoppixels = 6,
        pixelscalesites = 2√2,
        selector = missing,
        hide = :cell, # :hops, :sites, :bravais, :cell, :axes, :shell, :all
    )
end

#endregion

############################################################################################
# qplot
#region

const PlotLatticeArgumentType{E} = Union{Lattice{<:Any,E},LatticeSlice{<:Any,E},AbstractHamiltonian{<:Any,E}}

function Quantica.qplot(h::PlotLatticeArgumentType{3}; fancyaxis = true, axis = (;), figure = (;), inspector = false, plotkw...)
    fig, ax = empty_fig_axis_3D(plotlat_default_3D...; fancyaxis, axis, figure)
    fancyaxis ? plotlattice!(ax, h; plotkw...) :
                plotlattice!(ax, h; pixelscalesites = 1, plotkw...)  # Makie BUG workaround?
    inspector && DataInspector()
    return fig
end

function Quantica.qplot(h::PlotLatticeArgumentType; axis = (;), figure = (;), inspector = false, plotkw...)
    fig, ax = empty_fig_axis_2D(plotlat_default_2D...; axis, figure)
    plotlattice!(ax, h; plotkw...)
    inspector && DataInspector()
    return fig
end

function Quantica.qplot(g::GreenFunction; fancyaxis = true, axis = (;), figure = (;), inspector = false, children = missing, plotkw...)
    fig, ax = empty_fig_axis(g; fancyaxis, axis, figure)
    Σkws = Iterators.cycle(parse_children(children))
    Σs = Quantica.selfenergies(Quantica.contacts(g))
    for (Σ, childkw) in zip(Σs, Σkws)
        primitives = selfenergy_plottable(Quantica.solver(Σ), Quantica.plottables(Σ)...; childkw...)
        for (prim, primkw) in primitives
            plotlattice!(ax, prim; plotkw..., primkw..., childkw...)
        end
    end
    # Makie BUG: To allow inspector to show topmost tooltip, it should be transparent
    # if other layers (here the leads) are transparent
    plotlattice!(ax, parent(g); plotkw..., force_transparency = inspector && !isempty(Σs))
    inspector && DataInspector()
    return fig
end

Quantica.qplot!(x::Union{PlotLatticeArgumentType,GreenFunction}; kw...) =
    plotlattice!(x; kw...)

parse_children(::Missing) = (NamedTuple(),)
parse_children(p::Tuple) = p
parse_children(p::NamedTuple) = (p,)

empty_fig_axis(::GreenFunction{<:Any,3}; kw...) =
    empty_fig_axis_3D(plotlat_default_3D...; kw...)
empty_fig_axis(::GreenFunction; kw...) =
    empty_fig_axis_2D(plotlat_default_2D...; kw...)

const plotlat_default_figure = (; resolution = (1200, 1200), fontsize = 40)

const plotlat_default_axis3D = (;
    xlabel = "x", ylabel = "y", zlabel = "z",
    xticklabelcolor = :gray, yticklabelcolor = :gray, zticklabelcolor = :gray,
    xspinewidth = 0.2, yspinewidth = 0.2, zspinewidth = 0.2,
    xlabelrotation = 0, ylabelrotation = 0, zlabelrotation = 0,
    xticklabelsize = 30, yticklabelsize = 30, zticklabelsize = 30,
    xlabelsize = 40, ylabelsize = 40, zlabelsize = 40,
    xlabelfont = :italic, ylabelfont = :italic, zlabelfont = :italic,
    perspectiveness = 0.0, aspect = :data)

const plotlat_default_axis2D = (; autolimitaspect = 1)

const plotlat_default_lscene = (;)

const plotlat_default_2D =
    (plotlat_default_figure, plotlat_default_axis2D)
const plotlat_default_3D =
    (plotlat_default_figure, plotlat_default_axis3D, plotlat_default_lscene)

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

#region ## Constructors ##

SitePrimitives{E}() where {E} =
    SitePrimitives(Point{E,Float32}[], Int[], Float32[], Float32[], Float32[], String[], RGBAf[])

HoppingPrimitives{E}() where {E} =
    HoppingPrimitives(Point{E,Float32}[], Vec{E,Float32}[], Tuple{Int,Int}[], Float32[], Float32[], Float32[], String[], RGBAf[])

function siteprimitives(ls, h, plot, opacityflag)   # function barrier
    opts = (plot[:sitecolor][], plot[:siteopacity][], plot[:shellopacity][], plot[:siteradius][])
    opts´ = maybe_evaluate_observable.(opts, Ref(ls))
    return _siteprimitives(ls, h, opts´, opacityflag)
end

function _siteprimitives(ls::LatticeSlice{<:Any,E}, h, opts, opacityflag) where {E}
    sp = SitePrimitives{E}()
    mat = Quantica.matrix(first(harmonics(h)))
    lat = parent(ls)
    for (i´, cs) in enumerate(Quantica.cellsites(ls))
        ni = Quantica.cell(cs)
        i = Quantica.siteindex(cs)
        # in case opts contains some array over latslice (from an observable)
        opts´ = maybe_getindex.(opts, i´)
        push_siteprimitive!(sp, opts´, lat, i, ni, mat[i, i], opacityflag)
    end
    return sp
end

function hoppingprimitives(ls, h, siteradii, plot)   # function barrier
    opts = (plot[:hopcolor][], plot[:hopopacity][], plot[:shellopacity][], plot[:hopradius][])
    opts´ = maybe_evaluate_observable.(opts, Ref(ls))
    return _hoppingprimitives(ls, h, opts´, siteradii)
end

function _hoppingprimitives(ls::LatticeSlice{<:Any,E}, h, opts, siteradii) where {E}
    hp = HoppingPrimitives{E}()
    hp´ = HoppingPrimitives{E}()
    lat = parent(ls)
    sdict = sitedict(ls)
    for (j´, cs) in enumerate(Quantica.cellsites(ls))
        nj = Quantica.cell(cs)
        j = Quantica.siteindex(cs)
        siteradius = isempty(siteradii) ? Float32(0) : siteradii[j´]
        for har in harmonics(h)
            dn = Quantica.dcell(har)
            ni = nj + dn
            mat = Quantica.matrix(har)
            rows = rowvals(mat)
            for ptr in nzrange(mat, j)
                i = rows[ptr]
                Quantica.isonsite((i, j), dn) && continue
                i´ = get(sdict, (i, ni), nothing)
                if i´ === nothing
                    opts´ = maybe_getindex.(opts, j´)
                    push_hopprimitive!(hp´, opts´, lat, (i, j), (ni, nj), siteradius, mat[i, j], false)
                else
                    opts´ = maybe_getindex.(opts, i´, j´)
                    push_hopprimitive!(hp, opts´, lat, (i, j), (ni, nj), siteradius, mat[i, j], true)
                end
            end
        end
    end
    return hp, hp´
end

sitedict(ls::LatticeSlice) =
    Dict([(Quantica.siteindex(cs), Quantica.cell(cs)) => j for (j, cs) in enumerate(Quantica.cellsites(ls))])

maybe_evaluate_observable(o::Quantica.IndexableObservable, ls) = o[ls]
maybe_evaluate_observable(x, ls) = x

maybe_getindex(v::AbstractVector, i) = v[i]
maybe_getindex(m::AbstractMatrix, i) = sum(view(m, :, i))
maybe_getindex(m::Quantica.AbstractSparseMatrixCSC, i) = sum(view(nonzeros(m), nzrange(m, i)))
maybe_getindex(v, i) = v
maybe_getindex(v::AbstractVector, i, j) = 0.5*(v[i] + v[j])
maybe_getindex(m::AbstractMatrix, i, j) = m[i, j]
maybe_getindex(v, i, j) = v

## push! ##

function push_siteprimitive!(sp, (sitecolor, siteopacity, shellopacity, siteradius), lat, i, ni, matii, opacityflag)
    r = Quantica.site(lat, i, ni)
    s = Quantica.sitesublat(lat, i)
    push!(sp.centers, r)
    push!(sp.indices, i)
    push_sitehue!(sp, sitecolor, i, r, s)
    push_siteopacity!(sp, siteopacity, shellopacity, i, r, opacityflag)
    push_siteradius!(sp, siteradius, i, r)
    push_sitetooltip!(sp, i, r, matii)
    return sp
end

push_sitehue!(sp, ::Missing, i, r, s) = push!(sp.hues, s)
push_sitehue!(sp, sitecolor::Real, i, r, s) = push!(sp.hues, sitecolor)
push_sitehue!(sp, sitecolor::Function, i, r, s) = push!(sp.hues, sitecolor(i, r))
push_sitehue!(sp, ::Symbol, i, r, s) = push!(sp.hues, 0f0)
push_sitehue!(sp, ::Makie.Colorant, i, r, s) = push!(sp.hues, 0f0)
push_sitehue!(sp, sitecolor, i, r, s) = argerror("Unrecognized sitecolor")

push_siteopacity!(sp, ::Missing, bop, i, r, flag) = push!(sp.opacities, flag ? 1.0 : bop)
push_siteopacity!(sp, siteopacity::Real, bop, i, r, flag) = push!(sp.opacities, siteopacity)
push_siteopacity!(sp, siteopacity::Function, bop, i, r, flag) = push!(sp.opacities, siteopacity(i, r))
push_siteopacity!(sp, siteopacity, bop, i, r, flag) = argerror("Unrecognized siteradius")

push_siteradius!(sp, siteradius::Real, i, r) = push!(sp.radii, siteradius)
push_siteradius!(sp, siteradius::Function, i, r) = push!(sp.radii, siteradius(i, r))
push_siteradius!(sp, siteradius, i, r) = argerror("Unrecognized siteradius")

push_sitetooltip!(sp, i, r, mat) = push!(sp.tooltips, matrixstring(i, mat))
push_sitetooltip!(sp, i, r) = push!(sp.tooltips, positionstring(i, r))

function push_hopprimitive!(hp, (hopcolor, hopopacity, shellopacity, hopradius), lat, (i, j), (ni, nj), radius, matij, opacityflag)
    src, dst = Quantica.site(lat, j, nj), Quantica.site(lat, i, ni)
    opacityflag && (dst = (src + dst)/2)
    src += normalize(dst - src) * radius
    r, dr = (src + dst)/2, (dst - src)
    sj = Quantica.sitesublat(lat, j)
    push!(hp.centers, r)
    push!(hp.vectors, dr)
    push!(hp.indices, (i, j))
    push_hophue!(hp, hopcolor, (i, j), (r, dr), sj)
    push_hopopacity!(hp, hopopacity, shellopacity, (i, j), (r, dr), opacityflag)
    push_hopradius!(hp, hopradius, (i, j), (r, dr))
    push_hoptooltip!(hp, (i, j), matij)
    return hp
end

push_hophue!(hp, ::Missing, ij, rdr, s) = push!(hp.hues, s)
push_hophue!(hp, hopcolor::Real, ij, rdr, s) = push!(hp.hues, hopcolor)
push_hophue!(hp, hopcolor::Function, ij, rdr, s) = push!(hp.hues, hopcolor(ij, rdr))
push_hophue!(hp, ::Symbol, ij, rdr, s) = push!(hp.hues, 0f0)
push_hophue!(hp, ::Makie.Colorant, ij, rdr, s) = push!(hp.hues, 0f0)
push_hophue!(hp, hopcolor, ij, rdr, s) = argerror("Unrecognized hopcolor")

push_hopopacity!(hp, ::Missing, bop, ij, rdr, opacityflag) = push!(hp.opacities, opacityflag ? 1.0 : bop)
push_hopopacity!(hp, hopopacity::Real, bop, ij, rdr, opacityflag) = push!(hp.opacities, hopopacity)
push_hopopacity!(hp, hopopacity::Function, bop, ij, rdr, opacityflag) = push!(hp.opacities, hopopacity(ij, rdr))
push_hopopacity!(hp, hopopacity, bop, ij, rdr, opacityflag) = argerror("Unrecognized hopradius")

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
primitive_color(color, extrema, colormap, ::Missing) =
    RGBAf(colormap[mod1(round(Int, color), length(colormap))])
primitive_color(color, extrema, colormap, colorname::Symbol) =
    parse(RGBAf, colorname)
primitive_color(color, extrema, colormap, pcolor::Makie.Colorant) =
    convert(RGBAf, pcolor)
primitive_color(color, extrema, colormap, _) =
    RGBAf(colormap[normalize_range(color, extrema)])

# opacity == missing means reduced opacity in shell
primitite_opacity(α, extrema, ::Missing) = α
primitite_opacity(α, extrema, _) = normalize_range(α, extrema)

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
# PlotLattice for AbstractHamiltonian and Lattice
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

function Makie.plot!(plot::PlotLattice{Tuple{H}}) where {T,E,H<:Hamiltonian{T,E}}
    h = to_value(plot[1])
    lat = Quantica.lattice(h)
    sel = sanitize_selector(plot[:selector][], lat)
    latslice = lat[sel]
    return plotlattice!(plot, h, latslice; plot.attributes...)
end

# For E < 2 Hamiltonians, promote to 2D
function Makie.plot!(plot::PlotLattice{Tuple{H}}) where {T,H<:Hamiltonian{T,1}}
    h = Hamiltonian{2}(to_value(plot[1]))
    return plotlattice!(plot, h; plot.attributes...)
end

function Makie.plot!(plot::PlotLattice{Tuple{H,S}}) where {E,L,H<:Hamiltonian{<:Any,E,L},S<:LatticeSlice}
    h = to_value(plot[1])
    latslice = to_value(plot[2])
    lat = Quantica.lattice(h)
    latslice´ = Quantica.growdiff(latslice, h)

    hidesites = ishidden((:sites, :all), plot)
    hidehops = ishidden((:hops, :hoppings, :links, :all), plot)
    hidebravais = ishidden((:bravais, :all), plot)
    hideshell = ishidden((:shell, :all), plot) || iszero(L)

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
        hp, hp´ = hoppingprimitives(latslice, h, radii, plot)
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

function plotsites_shaded!(plot::PlotLattice, sp::SitePrimitives, transparency)
    inspector_label = (self, i, r) -> sp.tooltips[i]
    meshscatter!(plot, sp.centers; color = sp.colors, markersize = sp.radii,
            space = :data,
            transparency,
            inspector_label)
    return plot
end

function plotsites_flat!(plot::PlotLattice, sp::SitePrimitives, transparency)
    inspector_label = (self, i, r) -> sp.tooltips[i]
    scalefactor = plot[:pixelscalesites][]
    markersize = scalefactor ≈ 1 ? sp.radii : scalefactor * sp.radii
    scatter!(plot, sp.centers; color = sp.colors, markersize,
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
    cyl = Cylinder(Point3f(0., 0., -1.0), Point3f(0., 0, 1.0), Float32(1))
    meshscatter!(plot, hp.centers; color = hp.colors,
        rotations = hp.vectors, markersize = scales, marker = cyl,
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
        cellcolor = plot[:cellcolor][]
        colface = transparent(cellcolor, plot[:cellopacity][])
        coledge = transparent(cellcolor, 5 * plot[:cellopacity][])
        rect = Rect{L,Int}(Point{L,Int}(0), Point{L,Int}(1))
        mrect0 = GeometryBasics.mesh(rect, pointtype=Point{L,Float32}, facetype=QuadFace{Int})
        vertices0 = mrect0.position
        mat = Quantica.bravais_matrix(bravais)
        for sc in Quantica.subcells(latslice)
            cell = Quantica.cell(sc)
            mrect = GeometryBasics.mesh(rect, pointtype=Point{E,Float32}, facetype=QuadFace{Int})
            vertices = mrect.position
            vertices .= Ref(r0) .+ Ref(mat) .* (vertices0 .+ Ref(cell))
            mesh!(plot, mrect; color = colface, transparency = true, inspectable = false)
            wireframe!(plot, mrect; color = coledge, transparency = true, strokewidth = 1, inspectable = false)
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

matrixstring(x::Number) = numberstring(Quantica.chop(x))

function matrixstring(s::SMatrix)
    ss = repr("text/plain", s)
    pos = findfirst(isequal('\n'), ss)
    return pos === nothing ? ss : ss[pos:end]
end

matrixstring(s::Quantica.SMatrixView) = matrixstring(Quantica.unblock(s))

numberstring(x) = isreal(x) ? string(" ", real(x)) : isimag(x) ? string(" ", imag(x), "im") : string(" ", x)

isreal(x) = all(o -> imag(o) ≈ 0, x)
isimag(x) = all(o -> real(o) ≈ 0, x)

#endregion


############################################################################################
# selfenergy_plottable
#   Build plottable objects pi for each type of SelfEnergy (identified by its solver), and
#   associated qplot keywords
#region

selfenergy_plottable(::Quantica.AbstractSelfEnergySolver; kw...) = ()

function selfenergy_plottable(solver::Quantica.SelfEnergyModelSolver, ph; kw...)
    p1, k1 = ph, (; siteoutline = 7)
    return ((p1, k1),)
end

function selfenergy_plottable(s::Quantica.SelfEnergySchurSolver,
    hlead; kw...)
    p1, k1 = hlead, (; selector = siteselector())
    return ((p1, k1),)
end

function selfenergy_plottable(s::Quantica.SelfEnergyCouplingSchurSolver,
    hcoupling, hlead;  kw...)
    p1, k1 = hlead, (; selector = siteselector())
    p2, k2 = hcoupling, (; siteoutline = 7)
    return ((p1, k1), (p2, k2))
end

function selfenergy_plottable(s::Quantica.SelfEnergyGenericSolver, hcoupling; kw...)
    p1, k1 = hcoupling, (; siteoutline = 7)
    return ((p1, k1),)
end

#endregion

############################################################################################
# convert_arguments
#region

Makie.convert_arguments(::PointBased, lat::Lattice, sublat = missing) =
    (Point.(Quantica.sites(lat, sublat)),)
Makie.convert_arguments(p::PointBased, lat::Lattice{<:Any,1}, sublat = missing) =
    Makie.convert_arguments(p, Quantica.lattice(lat, dim = Val(2)), sublat)
Makie.convert_arguments(p::PointBased, h::Union{AbstractHamiltonian,GreenFunction,GreenSolution}, sublat = missing) =
    Makie.convert_arguments(p, Quantica.lattice(h), sublat)
Makie.convert_arguments(p::Type{<:LineSegments}, g::Union{GreenFunction,GreenSolution}, sublat = missing) =
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
