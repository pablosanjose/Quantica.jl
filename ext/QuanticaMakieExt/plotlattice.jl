
############################################################################################
# plotlattice recipe
#region

@recipe(PlotLattice) do scene
    Theme(
        ssao = true,
        fxaa = true,
        ambient = Vec3f(0.7),
        diffuse = Vec3f(0.5),
        backlight = 0.8f0,
        shading = false,
        force_transparency = false,
        shellopacity = 0.07,
        cellopacity = 0.03,
        cellcolor = RGBAf(0,0,1),
        sitecolor = missing,
        siteopacity = missing,
        maxsiteradius = 0.5,
        siteradius = 0.25,
        siteborder = 2,
        siteborderdarken = 0.6,
        sitedarken = 0.0,
        sitecolormap = :Spectral_9,
        hopcolor = missing,
        hopopacity = missing,
        maxhopradius = 0.1,
        hopradius = 0.03,
        hopoffset = 1.0,
        hopdarken = 0.85,
        hopcolormap = :Spectral_9,
        pixelscale = 6,
        selector = missing,
        flatsizefactor = 2√2,
        boundaries = missing,
        hide = :cell, # :hops, :sites, :bravais, :cell, :axes...
    )
end

#endregion

############################################################################################
# qplot
#region

function Quantica.qplot(h::Union{Lattice{<:Any,3},AbstractHamiltonian{<:Any,3}}; fancyaxis = true, axis = (;), figure = (;), inspector = false, plotkw...)
    fig, ax = empty_fig_axis_3D(plotlat_default_3D...; fancyaxis, axis, figure)
    fancyaxis ? plotlattice!(ax, h; plotkw...) :
                plotlattice!(ax, h; flatsizefactor = 1, plotkw...)  # Makie BUG workaround?
    inspector && DataInspector()
    return fig
end

function Quantica.qplot(h::Union{Lattice,AbstractHamiltonian}; axis = (;), figure = (;), inspector = false, plotkw...)
    fig, ax = empty_fig_axis_2D(plotlat_default_2D...; axis, figure)
    plotlattice!(ax, h; plotkw...)
    inspector && DataInspector()
    return fig
end

function Quantica.qplot(g::GreenFunction; fancyaxis = true, axis = (;), figure = (;), inspector = false, children = missing, plotkw...)
    fig, ax = empty_fig_axis(g; fancyaxis, axis, figure)
    Σkws = Iterators.cycle(parse_children(children))
    for (Σ, childkw) in zip(Quantica.selfenergies(Quantica.contacts(g)), Σkws)
        primitives = selfenergy_plottable(Quantica.solver(Σ), Quantica.plottables(Σ)...; childkw...)
        for (prim, primkw) in primitives
            plotlattice!(ax, prim; plotkw..., childkw..., primkw...)
        end
    end
    # Makie BUG: To allow inspector to show topmost tooltip, it must be transparent
    # if other layers (here the leads) are transparent
    plotlattice!(ax, parent(g); plotkw..., force_transparency = inspector)
    inspector && DataInspector()
    return fig
end

Quantica.qplot!(x::Union{Lattice,AbstractHamiltonian,GreenFunction}; kw...) =
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
    return _siteprimitives(ls, h, opts, opacityflag)
end


function _siteprimitives(ls::LatticeSlice{<:Any,E}, h, opts, opacityflag) where {E}
    sp = SitePrimitives{E}()
    mat = Quantica.matrix(first(harmonics(h)))
    lat = parent(ls)
    for sc in Quantica.subcells(ls)
        ni = Quantica.cell(sc)
        for i in Quantica.siteindices(sc)
            push_siteprimitive!(sp, opts, lat, i, ni, mat[i, i], opacityflag)
        end
    end
    return sp
end

function hoppingprimitives(ls, selector, h, radii, plot)   # function barrier
    opts = (plot[:hopcolor][], plot[:hopopacity][], plot[:shellopacity][], plot[:hopradius][])
    return _hoppingprimitives(ls, selector, h, radii, opts)
end

function _hoppingprimitives(ls::LatticeSlice{<:Any,E}, selector, h, radii, opts) where {E}
    hp = HoppingPrimitives{E}()
    hp´ = HoppingPrimitives{E}()
    lat = parent(ls)
    counter = 0
    for sc in Quantica.subcells(ls)
        nj = Quantica.cell(sc)
        for j in Quantica.siteindices(sc)
            counter += 1
            radius = isempty(radii) ? Float32(0) : radii[counter]
            for har in harmonics(h)
                dn = Quantica.dcell(har)
                ni = nj + dn
                mat = Quantica.matrix(har)
                rows = rowvals(mat)
                for ptr in nzrange(mat, j)
                    i = rows[ptr]
                    Quantica.isonsite((i, j), dn) && continue
                    ri = Quantica.site(lat, i, ni)
                    if (i, ri, ni) in selector
                        push_hopprimitive!(hp, opts, lat, (i, j), (ni, nj), radius, mat[i, j], true)
                    else
                        push_hopprimitive!(hp´, opts, lat, (i, j), (ni, nj), radius, mat[i, j], false)
                    end
                end
            end
        end
    end
    return hp, hp´
end

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
    maxsiteradius = plot[:maxsiteradius][]
    return update_radii!(p, extremarads, siteradius, maxsiteradius)
end

function update_radii!(p::HoppingPrimitives, plot, extremarads)
    hopradius = plot[:hopradius][]
    maxhopradius = plot[:maxhopradius][]
    return update_radii!(p, extremarads, hopradius, maxhopradius)
end

function update_radii!(p, extremarads, radius, maxradius)
    for (i, r) in enumerate(p.radii)
        p.radii[i] = primitive_radius(normalize_range(r, extremarads), radius, maxradius)
    end
    return p
end

primitive_radius(normr, radius::Number, maxradius) = radius
primitive_radius(normr, radius, maxradius) = maxradius * normr

## primitive_scales ##

function primitive_scales(p::HoppingPrimitives, plot)
    hopradius = plot[:hopradius][]
    maxhopradius = plot[:maxhopradius][]
    scales = Vec3f[]
    for (r, v) in zip(p.radii, p.vectors)
        push!(scales, primitive_scale(r, v, hopradius, maxhopradius))
    end
    return scales
end

primitive_scale(normr, v, hopradius::Number, maxhopradius) = Vec3f(hopradius, hopradius, norm(v)/2)
primitive_scale(normr, v, hopradius, maxhopradius) = Vec3f(normr * maxhopradius, normr * maxhopradius, norm(v)/2)

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
    pixelscale = plot[:pixelscale][]
    hopradius = plot[:hopradius][]
    linewidths = Float32[]
    for r in p.radii
        linewidth = primitive_linewidth(r, hopradius, pixelscale)
        append!(linewidths, (linewidth, linewidth))
    end
    return linewidths
end

primitive_linewidth(normr, hopradius::Number, pixelscale) = pixelscale
primitive_linewidth(normr, hopradius, pixelscale) = pixelscale * normr

#endregion

#endregion

############################################################################################
# PlotLattice for AbstractHamiltonian and Lattice
#region

Makie.plot!(plot::PlotLattice{Tuple{L}}) where {L<:Lattice} = plotlattice!(plot, hamiltonian(lat))

function Makie.plot!(plot::PlotLattice{Tuple{H}}) where {E,L,H<:AbstractHamiltonian{<:Any,E,L}}
    h = to_value(plot[1])
    lat = Quantica.lattice(h)
    sel = sanitize_selector(plot[:selector][], lat)
    asel = Quantica.apply(sel, lat)
    latslice = lat[asel]
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
        hp, hp´ = hoppingprimitives(latslice, asel, h, radii, plot)
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
        if E == 3 && plot[:shading][]
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
        if E == 3 && plot[:shading][]
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
            markerspace = :data,
            ssao = plot[:ssao][],
            ambient = plot[:ambient][],
            diffuse = plot[:diffuse][],
            backlight = plot[:backlight][],
            fxaa = plot[:fxaa][],
            transparency,
            inspector_label)
    return plot
end

function plotsites_flat!(plot::PlotLattice, sp::SitePrimitives, transparency)
    inspector_label = (self, i, r) -> sp.tooltips[i]
    factor = plot[:flatsizefactor][]
    markersize = factor ≈ 1 ? sp.radii : factor * sp.radii
    scatter!(plot, sp.centers; color = sp.colors, markersize,
            markerspace = :data,
            strokewidth = plot[:siteborder][],
            strokecolor = darken.(sp.colors, Ref(plot[:siteborderdarken][])),
            ambient = plot[:ambient][],
            diffuse = plot[:diffuse][],
            backlight = plot[:backlight][],
            fxaa = plot[:fxaa][],
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
        ssao = plot[:ssao][],
        ambient = plot[:ambient][],
        diffuse = plot[:diffuse][],
        backlight = plot[:backlight][],
        fxaa = plot[:fxaa][],
        transparency,
        inspector_label)
    return plot
end

function plothops_flat!(plot::PlotLattice, hp::HoppingPrimitives, transparency)
    inspector_label = (self, i, r) -> (hp.tooltips[(i+1) ÷ 2])
    colors = hp.colors
    segments = primitive_segments(hp, plot)
    linewidths = primitive_linewidths(hp, plot)
    linesegments!(plot, segments; color = colors, linewidth = linewidths,
        backlight = plot[:backlight][],
        fxaa = plot[:fxaa][],
        transparency,
        inspector_label)
    return plot
end

function plotbravais!(plot::PlotLattice, lat::Lattice{<:Any,E,L}, latslice) where {E,L}
    iszero(L) && return plot
    bravais = Quantica.bravais(lat)
    vs = Point{E}.(Quantica.bravais_vectors(bravais))
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

matrixstring(x::Number) = numberstring(x)

function matrixstring(s::SMatrix)
    ss = repr("text/plain", s)
    pos = findfirst(isequal('\n'), ss)
    return pos === nothing ? ss : ss[pos:end]
end

numberstring(x) = isreal(x) ? string(" ", real(x)) : isimag(x) ? string(" ", imag(x), "im") : string(" ", x)

isreal(x) = all(o -> imag(o) ≈ 0, x)
isimag(x) = all(o -> real(o) ≈ 0, x)

#endregion


############################################################################################
# selfenergy_plottable
#   Build plottable objects for each type of SelfEnergy (identified by its solver)
#region

selfenergy_plottable(::Quantica.AbstractSelfEnergySolver; kw...) = ()

function selfenergy_plottable(solver::Quantica.SelfEnergyModelSolver; kw...)
    p1, k1 = solver.ph, (;)
    return ((p1, k1),)
end

function selfenergy_plottable(s::Quantica.SelfEnergySchurSolver,
    hlead, negative; numcells = 1, kw...)
    p1, k2 = _selfenergy_plottable_hlead(hlead, negative, numcells)
    return ((p1, k2),)
end

function selfenergy_plottable(s::Quantica.SelfEnergyUnicellSchurSolver,
    hlead, hcoupling, negative; numcells = 1, kw...)
    p1 = hcoupling
    k1 = (; selector = siteselector())
    (p2, k2) = _selfenergy_plottable_hlead(hlead, negative, numcells)
    return ((p1, k1), (p2, k2))
end

function _selfenergy_plottable_hlead(hlead, negative, numcells)
    n = max(1, numcells)
    cellrng = negative ? (-n:-1) : (1:n)
    p = hlead
    k = (; selector = siteselector(; cells = cellrng))
    return (p, k)
end

#endregion

############################################################################################
# convert_arguments
#region

Makie.convert_arguments(::PointBased, lat::Lattice, sublat = missing) =
    (Point.(Quantica.sites(lat, sublat)),)
Makie.convert_arguments(p::PointBased, h::Union{AbstractHamiltonian,GreenFunction,GreenSolution}, sublat = missing) =
    Makie.convert_arguments(p, Quantica.lattice(h), sublat)

Makie.convert_arguments(p::Type{<:LineSegments}, g::Union{GreenFunction,GreenSolution}, sublat = missing) =
    Makie.convert_arguments(p, Quantica.hamiltonian(g), sublat)

function Makie.convert_arguments(::Type{<:LineSegments}, h::AbstractHamiltonian{<:Any,E}, sublat = missing) where {E}
    segments = Point{E,Float32}[]
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