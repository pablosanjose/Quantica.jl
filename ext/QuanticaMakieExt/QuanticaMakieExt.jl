module QuanticaMakieExt

using Makie
using Quantica
using Makie.GeometryBasics
using Makie.GeometryBasics: Ngon
using Quantica: Lattice, LatticeSlice, AbstractHamiltonian, Harmonic, Bravais, SVector,
      argerror, harmonics, sublats, siterange, site, norm, normalize, nsites,
      nzrange, rowvals, sanitize_SVector

import Quantica: plotlattice, plotlattice!, qplot

############################################################################################
# plotlattice recipe
#region

@recipe(PlotLattice) do scene
    Theme(
        ssao = true,
        fxaa = true,
        ambient = Vec3f(0.5),
        diffuse = Vec3f(0.5),
        backlight = 4.0f0,
        shading = false,
        shellopacity = 0.05,
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
        tooltips = true,
        digits = 3,
        axes = missing, # can be e.g. (1,2) to project onto the x,y plane
        hide = :cell, # :hops, :sites, :bravais, :cell, :axes...
    )
end

Makie.plot!(plot::PlotLattice) = plotlat!(plot, to_value(plot[1]))

function Quantica.qplot(h::Union{Lattice,AbstractHamiltonian})
    plotlattice(h)
end

#endregion

############################################################################################
# Primitives
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
        dn = Quantica.cell(sc)
        for i in Quantica.siteindices(sc)
            push_siteprimitive!(sp, opts, lat, i, dn, mat[i, i], opacityflag)
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
        dnj = Quantica.cell(sc)
        for j in Quantica.siteindices(sc)
            counter += 1
            radius = isempty(radii) ? Float32(0) : radii[counter]
            for har in harmonics(h)
                dni = dnj + Quantica.dcell(har)
                mat = Quantica.matrix(har)
                rows = rowvals(mat)
                for ptr in nzrange(mat, j)
                    i = rows[ptr]
                    ri = Quantica.site(lat, i, dni)
                    if (i, ri, dni) in selector
                        push_hopprimitive!(hp, opts, lat, (i, j), (dni, dnj), radius, mat[i, j], true)
                    else
                        push_hopprimitive!(hp´, opts, lat, (i, j), (dni, dnj), radius, mat[i, j], false)
                    end
                end
            end
        end
    end
    return hp, hp´
end

## push! ##

function push_siteprimitive!(sp, (sitecolor, siteopacity, shellopacity, siteradius), lat, i, dn, matii, opacityflag)
    r = Quantica.site(lat, i, dn)
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
push_sitehue!(sp, sitecolor::Function, i, r, s) = push!(sp.hues, sitecolor(i, r))
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

function push_hopprimitive!(hp, (hopcolor, hopopacity, shellopacity, hopradius), lat, (i, j), (dni, dnj), radius, matij, opacityflag)
    src, dst = Quantica.site(lat, j, dnj), Quantica.site(lat, i, dni)
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
push_hophue!(hp, hopcolor::Function, ij, rdr, s) = push!(hp.hues, hopcolor(ij, rdr))
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

update_colors!(p, plot) =
    update_colors!(p, plot, extrema(p.hues), extrema(p.opacities))

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
primitive_color(c, extrema, colormap, ::Missing) = RGBAf(colormap[mod1(round(Int, c), length(colormap))])
primitive_color(c, extrema, colormap, _) = RGBAf(colormap[normalize_range(c, extrema)])

# opacity == missing means reduced opacity in shell
primitite_opacity(α, extrema, ::Missing) = α
primitite_opacity(α, extrema, _) = normalize_range(α, extrema)

## update_radii! ##

update_radii!(p, plot) = update_radii!(p, plot, extrema(p.radii))

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

plotlat!(plot::PlotLattice, lat::Lattice) = plotlat!(plot, hamiltonian(lat))

using Infiltrator

function plotlat!(plot::PlotLattice, h::AbstractHamiltonian{<:Any,E,L}) where {E,L}
    lat = Quantica.lattice(h)
    sel = sanitize_selectoror(plot[:selector][])
    asel = Quantica.apply(sel, lat)
    latslice = lat[asel]
    latslice´ = Quantica.growdiff(latslice, h)

    hidesites = ishidden((:sites, :all), plot)
    hidehops = ishidden((:hops, :hoppings, :links, :all), plot)
    hidebravais = ishidden((:bravais, :all), plot)
    hideshell = ishidden((:shell, :all), plot)

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
        forcetrans = hopopacity isa Function || (hopopacity isa Real && hopopacity < 1)
        if E == 3 && plot[:shading][]
            plothops_shading!(plot, hp, forcetrans)
            hideshell || plothops_shading!(plot, hp´, true)
        else
            plothops_flat!(plot, hp, forcetrans)
            hideshell || plothops_flat!(plot, hp´, true)
        end
    end

    # plot sites
    if !hidesites
        siteopacity = plot[:siteopacity][]
        transparency = siteopacity isa Function || (siteopacity isa Real && siteopacity < 1)
        if E == 3 && plot[:shading][]
            plotsites_shading!(plot, sp, transparency)
            hideshell || plotsites_shading!(plot, sp´, true)
        else
            plotsites_flat!(plot, sp, transparency)
            hideshell || plotsites_flat!(plot, sp´, true)
        end
    end

    return plot
end

sanitize_selectoror(::Missing) = Quantica.siteselectoror()
sanitize_selectoror(s::Quantica.SiteSelector) = s
sanitize_selectoror(s) = argerror("Specify a site selector with `selector = siteselector(; kw...)`")

function joint_colors_radii_update!(p, p´, plot)
    hext, oext = jointextrema(p.hues, p´.hues), jointextrema(p.opacities, p´.opacities)
    update_colors!(p, plot, hext, oext)
    update_colors!(p´, plot, hext, oext)
    rext = jointextrema(p.radii, p´.radii)
    update_radii!(p, plot, rext)
    update_radii!(p´, plot, rext)
    return nothing
end

function plotsites_shading!(plot::PlotLattice, sp::SitePrimitives, transparency)
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
    scatter!(plot, sp.centers; color = sp.colors, markersize = 2√2 * sp.radii,
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

function plothops_shading!(plot::PlotLattice, hp::HoppingPrimitives, transparency)
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
    bravais = Quantica.bravais(lat)
    vs = Point{E}.(Quantica.bravais_vectors(bravais))
    vtot = sum(vs)
    r0 = Point{E,Float32}(Quantica.mean(Quantica.sites(lat))) - 0.5 * vtot

    if !ishidden(:axes, plot)
        for (v, color) in zip(vs, (:red, :green, :blue))
            arrows!(plot, [r0], [v]; color, inspectable = false)
        end
    end

    if !ishidden((:cell, :cells), plot)
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
# tools
#region

function darken(rgba::RGBAf, v = 0.66)
    r = max(0, min(rgba.r * (1 - v), 1))
    g = max(0, min(rgba.g * (1 - v), 1))
    b = max(0, min(rgba.b * (1 - v), 1))
    RGBAf(r,g,b,rgba.alpha)
end

darken(colors::Vector, v = 0.66) = darken.(colors, Ref(v))

transparent(rgba::RGBAf, v = 0.5) = RGBAf(rgba.r, rgba.g, rgba.b, rgba.alpha * v)

maybedim(color, dn, dimming) = iszero(dn) ? color : transparent(color, 1 - dimming)

dnshell(::Lattice{<:Any,<:Any,L}, span = -1:1) where {L} =
    sort!(vec(SVector.(Iterators.product(ntuple(_ -> span, Val(L))...))), by = norm)

ishidden(s, plot::PlotLattice) = ishidden(s, plot[:hide][])
ishidden(s, ::Nothing) = false
ishidden(s::Symbol, hide::Symbol) = s === hide
ishidden(s::Symbol, hides::Tuple) = s in hides
ishidden(ss, hides) = any(s -> ishidden(s, hides), ss)

normalize_range(c::T, (min, max)) where {T} = min ≈ max ? T(0.5) : T((c - min)/(max - min))

jointextrema(v, v´) = min(minimum(v), minimum(v´)), max(maximum(v), maximum(v´))

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

# ############################################################################################
# # convert_arguments
# #region

# function Makie.convert_arguments(::PointBased, lat::Lattice{<:Any,E}, sublat = missing) where {E}
#     lat = lattice(h)
#     sites = Point{E,Float32}[]
#     dns = dnshell(lat)
#     for dn in dns
#         append_dcell_sites!(sites, dn, lat, sublatsrc)
#     end
#     return (sites,)
# end

# function Makie.convert_arguments(::PointBased, h::AbstractHamiltonian{<:Any,E}, sublatsrc = missing) where {E}
#     lat = lattice(h)
#     sites = Point{E,Float32}[]
#     for har in harmonics(h)
#         append_harmonic_sites!(sites, har, lat, sublatsrc)
#     end
#     return (sites,)
# end

# # linesegments is also PointBased
# function Makie.convert_arguments(::Type{<:LineSegments}, h::AbstractHamiltonian{<:Any,E}, sublatsrc = missing) where {E}
#     segments = Point{E,Float32}[]
#     append_segment!(segments, h, sublatsrc)
#     return (segments,)
# end

# function Makie.convert_arguments(::Type{<:LineSegments}, har::Harmonic, lat::Lattice{<:Any,E}, sublatsrc = missing) where {E}
#     segments = Point{E,Float32}[]
#     append_segment!(segments, har, lat, sublatsrc)
#     return (segments,)
# end

# #endregion

end # module