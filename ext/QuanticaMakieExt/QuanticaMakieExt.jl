module QuanticaMakieExt

using Makie
using Quantica
using Makie.GeometryBasics
using Makie.GeometryBasics: Ngon
using Quantica: Lattice, AbstractHamiltonian, Harmonic, Bravais, SVector,
      dcell, argerror, harmonics, sublats, siterange, site, norm, normalize, nsites,
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
        shaded = false,
        inter_opacity = 0.05,
        cell_opacity = 0.03,
        sitecolor = missing,
        siteopacity = missing,
        maxsiteradius = 0.5,
        siteradius = 0.25,
        siteborder = 3,
        siteborderdarken = 0.6,
        hopcolor = missing,
        hopopacity = missing,
        maxhopradius = 0.1,
        hopradius = 0.03,
        hopoffset = 1.0,
        hopdarken = 0.85,
        pixelscale = 6,
        cells = missing,
        tooltips = true,
        digits = 3,
        axes = missing, # can be e.g. (1,2) to project onto the x,y plane
        hide = :cell, # :hops, :sites, :bravais, :cell, :axes...
        colormap = :Spectral_9,
    )
end

Makie.plot!(plot::PlotLattice) = plotlat_dispatch!(plot, to_value(plot[1]))

function Quantica.qplot(h::AbstractHamiltonian)
    plotlattice(h)
end

#endregion

############################################################################################
# site_primitives
#region

struct SitePrimitives{E}
    centers::Vector{Point{E,Float32}}
    indices::Vector{Int}
    colors::Vector{Float32}
    opacity::Vector{Float32}
    radii::Vector{Float32}
    tooltips::Vector{String}
end

SitePrimitives{E}() where {E} =
    SitePrimitives(Point{E,Float32}[], Int[], Float32[], Float32[], Float32[], String[])

function append_site_primitives!(sp::SitePrimitives, plot, cell, har::Harmonic, lat::Lattice)
    mat = Quantica.matrix(har)
    dn = dcell(har)
    dr = Quantica.bravais_matrix(lat) * (cell + dn)
    sites = Quantica.sites(lat)
    offset = length(sp.centers)

    intracell = iszero(dn)
    if intracell
        for s in sublats(lat), i in siterange(lat, s)
            r = sites[i] + dr
            push!(sp.centers, r)
            push!(sp.indices, i)
            push_sitecolor!(sp, plot[:sitecolor][], i, r, s)
            push_siteopacity!(sp, plot[:siteopacity][], plot[:inter_opacity][], i, r, intracell)
            push_siteradius!(sp, plot[:siteradius][], i, r)
            push_sitetooltip!(sp, i, r, mat[i, i])
        end
    else
        for s in sublats(lat), j in siterange(lat, s), ptr in nzrange(mat, j)
            i = rowvals(mat)[ptr]
            # avoid adding duplicate targets
            i in view(sp.indices, offset + 1:length(sp.indices)) && continue
            r = sites[i] + dr
            push!(sp.centers, r)
            push!(sp.indices, i)
            push_sitecolor!(sp, plot[:sitecolor][], i, r, s)
            push_siteopacity!(sp, plot[:siteopacity][], plot[:inter_opacity][], i, r, intracell)
            push_siteradius!(sp, plot[:siteradius][], i, r)
            push_sitetooltip!(sp, i, r, mat[i, i])
        end
    end
    return sp
end

push_sitecolor!(sp, ::Missing, i, r, s) = push!(sp.colors, s)
push_sitecolor!(sp, sitecolor::Function, i, r, s) = push!(sp.colors, sitecolor(i, r))
push_sitecolor!(sp, sitecolor, i, r, s) = argerror("Unrecognized sitecolor")

push_siteopacity!(sp, ::Missing, bop, i, r, intracell) = push!(sp.opacity, intracell ? 1.0 : bop)
push_siteopacity!(sp, siteopacity::Real, bop, i, r, intracell) = push!(sp.opacity, siteopacity)
push_siteopacity!(sp, siteopacity::Function, bop, i, r, intracell) = push!(sp.opacity, siteopacity(i, r))
push_siteopacity!(sp, siteopacity, bop, i, r, intracell) = argerror("Unrecognized siteradius")

push_siteradius!(sp, siteradius::Real, i, r) = push!(sp.radii, siteradius)
push_siteradius!(sp, siteradius::Function, i, r) = push!(sp.radii, siteradius(i, r))
push_siteradius!(sp, siteradius, i, r) = argerror("Unrecognized siteradius")

push_sitetooltip!(sp, i, r, mat) = push!(sp.tooltips, matrixstring(i, mat))
push_sitetooltip!(sp, i, r) = push!(sp.tooltips, positionstring(i, r))

#endregion

############################################################################################
# hopping_primitives
#region

struct HoppingPrimitives{E}
    centers::Vector{Point{E,Float32}}
    vectors::Vector{Vec{E,Float32}}
    indices::Vector{Tuple{Int,Int}}
    colors::Vector{Float32}
    opacity::Vector{Float32}
    radii::Vector{Float32}
    tooltips::Vector{String}
end

HoppingPrimitives{E}() where {E} =
    HoppingPrimitives(Point{E,Float32}[], Vec{E,Float32}[], Tuple{Int,Int}[], Float32[], Float32[], Float32[], String[])

function append_hopping_primitives!(hp::HoppingPrimitives, plot, cell, har::Harmonic, lat::Lattice, intracell)
    mat = Quantica.matrix(har)
    dn = dcell(har)
    r0dst = Quantica.bravais_matrix(lat) * (cell + dn)
    r0src = Quantica.bravais_matrix(lat) * cell
    sites = Quantica.sites(lat)

    for sj in sublats(lat), j in siterange(lat, sj), ptr in nzrange(mat, j)
        i = rowvals(mat)[ptr]
        src, dst = sites[j] + r0src, sites[i] + r0dst
        intracell && (dst = (src + dst)/2)
        r, dr = (src + dst)/2, (dst - src)
        push!(hp.centers, r)
        push!(hp.vectors, dr)
        push!(hp.indices, (i, j))
        push_hopcolor!(hp, plot[:hopcolor][], (i, j), (r, dr), sj)
        push_hopopacity!(hp, plot[:hopopacity][], plot[:inter_opacity][], (i, j), (r, dr), intracell)
        push_hopradius!(hp, plot[:hopradius][], (i, j), (r, dr))
        push_hoptooltip!(hp, (i, j), mat[i, j])
    end
    return hp
end

push_hopcolor!(hp, ::Missing, ij, rdr, s) = push!(hp.colors, s)
push_hopcolor!(hp, hopcolor::Function, ij, rdr, s) = push!(hp.colors, hopcolor(ij, rdr))
push_hopcolor!(hp, hopcolor, ij, rdr, s) = argerror("Unrecognized hopcolor")

push_hopopacity!(hp, ::Missing, bop, ij, rdr, intracell) = push!(hp.opacity, intracell ? 1.0 : bop)
push_hopopacity!(hp, hopopacity::Real, bop, ij, rdr, intracell) = push!(hp.opacity, hopopacity)
push_hopopacity!(hp, hopopacity::Function, bop, ij, rdr, intracell) = push!(hp.opacity, hopopacity(ij, rdr))
push_hopopacity!(hp, hopopacity, bop, ij, rdr, intracell) = argerror("Unrecognized hopradius")

push_hopradius!(hp, hopradius::Real, ij, rdr) = push!(hp.radii, hopradius)
push_hopradius!(hp, hopradius::Function, ij, rdr) = push!(hp.radii, hopradius(ij, rdr))
push_hopradius!(hp, hopradius, ij, rdr) = argerror("Unrecognized hopradius")

push_hoptooltip!(hp, (i, j), mat) = push!(hp.tooltips, matrixstring(i, j, mat))

#endregion

############################################################################################
# PlotLattice for AbstractHamiltonian and Lattice
#region

function plotlat_dispatch!(plot::PlotLattice, h::AbstractHamiltonian{<:Any,E,L}) where {E,L}
    lat = Quantica.lattice(h)

    hidesites = ishidden((:sites, :all), plot)
    hidehops = ishidden((:hops, :hoppings, :links, :all), plot)
    hidebravais = ishidden((:bravais, :all), plot)
    hideinter = ishidden((:inter, :all), plot)

    cells = sanitize_plotcells(plot[:cells][], lat)
    hars = harmonics(h)

    # plot bravais axes
    if !hidebravais
        plotbravais!(plot, lat, cells)
    end

    # plot hoppings
    if !hidehops
        hop_prims_intra = HoppingPrimitives{E}()
        hop_prims_inter = HoppingPrimitives{E}()
        hopopacity = plot[:hopopacity][]
        forcetrans = hopopacity isa Function || (hopopacity isa Real && hopopacity < 1)
        for cell in cells, har in hars
            dcell´ = dcell(har)
            intracells = cell + dcell´ in cells
            hp = ifelse(intracells, hop_prims_intra, hop_prims_inter)
            append_hopping_primitives!(hp, plot, cell, har, lat, intracells)
        end
        if E == 3 && plot[:shaded][]
            plothops_shaded!(plot, hop_prims_intra, forcetrans)
            hideinter || plothops_shaded!(plot, hop_prims_inter, true)
        else
            plothops_flat!(plot, hop_prims_intra, forcetrans)
            hideinter || plothops_flat!(plot, hop_prims_inter, true)
        end
    end

    # plot sites
    if !hidesites
        site_prims_intra = SitePrimitives{E}()
        site_prims_inter = SitePrimitives{E}()
        siteopacity = plot[:siteopacity][]
        forcetrans = siteopacity isa Function || (siteopacity isa Real && siteopacity < 1)
        for cell in cells, har in hars
            dcell´ = dcell(har)
            intracells = iszero(dcell´)
            !intracells && cell + dcell´ in cells && continue
            sp = ifelse(intracells, site_prims_intra, site_prims_inter)
            append_site_primitives!(sp, plot, cell, har, lat)
        end
        if E == 3 && plot[:shaded][]
            plotsites_shaded!(plot, site_prims_intra, forcetrans)
            hideinter || plotsites_shaded!(plot, site_prims_inter, true)
        else
            plotsites_flat!(plot, site_prims_intra, forcetrans)
            hideinter || plotsites_flat!(plot, site_prims_inter, true)
        end
    end

    return plot
end

sanitize_plotcells(::Missing, lat) = (Quantica.zerocell(lat),)
sanitize_plotcells(cells, lat) = sanitize_SVector.(cells)

plotlat_dispatch!(plot::PlotLattice, lat::Lattice) = plotlat_dispatch!(plot, hamiltonian(lat))

function plotsites_shaded!(plot::PlotLattice, sp::SitePrimitives, transparency)
    inspector_label = (self, i, r) -> sp.tooltips[i]
    centers = sp.centers
    colors = primitive_colors(sp, plot)
    radii = primitive_radii(sp, plot)
    meshscatter!(plot, centers; color = colors, markersize = radii,
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
    centers = sp.centers
    colors = primitive_colors(sp, plot)
    radii = primitive_radii(sp, plot, 2√2)
    scatter!(plot, centers; color = colors, markersize = radii,
            markerspace = :data,
            strokewidth = plot[:siteborder][],
            strokecolor = darken.(colors, Ref(plot[:siteborderdarken][])),
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
    centers = hp.centers
    vectors = hp.vectors
    colors = primitive_colors(hp, plot)
    scales = primitive_scales(hp, plot)
    cyl = Cylinder(Point3f(0., 0., -1.0), Point3f(0., 0, 1.0), Float32(1))
    meshscatter!(plot, centers; color = colors,
        rotations = vectors, markersize = scales, marker = cyl,
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
    colors = primitive_colors(hp, plot)
    segments = primitive_segments(hp, plot)
    linewidths = primitive_linewidths(hp, plot)
    linesegments!(plot, segments; color = colors, linewidth = linewidths,
        backlight = plot[:backlight][],
        fxaa = plot[:fxaa][],
        transparency,
        inspector_label)
    return plot
end

function plotbravais!(plot::PlotLattice, lat::Lattice{<:Any,E,L}, cells) where {E,L}
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
        colface = RGBAf(0, 0, 1, plot[:cell_opacity][])
        coledge = RGBAf(0, 0, 1, 5 * plot[:cell_opacity][])
        rect = Rect{L,Int}(Point{L,Int}(0), Point{L,Int}(1))
        mrect0 = GeometryBasics.mesh(rect, pointtype=Point{L,Float32}, facetype=QuadFace{Int})
        vertices0 = mrect0.position
        mat = Quantica.bravais_matrix(bravais)
        for cell in cells
            mrect = GeometryBasics.mesh(rect, pointtype=Point{E,Float32}, facetype=QuadFace{Int})
            vertices = mrect.position
            vertices .= Ref(r0) .+ Ref(mat) .* (vertices0 .+ Ref(cell))
            mesh!(plot, mrect; color = colface, transparency = true, inspectable = false)
            wireframe!(plot, mrect; color = coledge, transparency = true, strokewidth = 1, inspectable = false)
        end
    end

    return plot
end

############################################################################################
# Primitive properties
#region

primitive_colors(p::SitePrimitives, plot) =
    primitive_colors(p.colors, p.opacity, plot[:sitecolor][], plot[:siteopacity][],
                     Makie.ColorSchemes.colorschemes[plot[:colormap][]])

primitive_colors(p::HoppingPrimitives, plot) =
    primitive_colors(p.colors, p.opacity, plot[:hopcolor][], plot[:hopopacity][],
                     Makie.ColorSchemes.colorschemes[plot[:colormap][]], plot[:hopdarken][])

function primitive_colors(colors, opacity, pcolor, popacity, colormap, pdarken = 0.0)
    isempty(colors) && return RGBAf[]
    minc, maxc = extrema(colors)
    mino, maxo = extrema(opacity)
    colors = [transparent(
              darken(primitive_color(c, (minc, maxc), colormap, pcolor), pdarken),
              primitite_opacity(α, (mino, maxo), popacity))
              for (c, α) in zip(colors, opacity)]
    return colors
end

# color == missing means sublat color
primitive_color(c, extrema, colormap, ::Missing) = RGBAf(colormap[mod1(round(Int, c), length(colormap))])
primitive_color(c, extrema, colormap, _) = RGBAf(colormap[normalize_range(c, extrema)])

# opacity == missing means inter opacity
primitite_opacity(α, extrema, ::Missing) = α
primitite_opacity(α, extrema, _) = normalize_range(α, extrema)

function primitive_radii(p::SitePrimitives, plot, factor = 1.0)
    isempty(p.radii) && return Float32[]
    siteradius = plot[:siteradius][]
    maxsiteradius = plot[:maxsiteradius][]
    minr, maxr = extrema(p.radii)
    radii = [primitive_radius(factor * normalize_range(radius, (minr, maxr)), siteradius, maxsiteradius) for radius in p.radii]
    return radii
end

primitive_radius(normr, siteradius::Number, maxsiteradius) = siteradius
primitive_radius(normr, siteradius, maxsiteradius) = maxsiteradius * normr

function primitive_scales(p::HoppingPrimitives, plot)
    isempty(p.radii) && return Vec3f[]
    hopradius = plot[:hopradius][]
    maxhopradius = plot[:maxhopradius][]
    minr, maxr = extrema(p.radii)
    scales = [primitive_scale(normalize_range(r, (minr, maxr)), v, hopradius, maxhopradius)
              for (r, v) in zip(p.radii, p.vectors)]
    return scales
end

primitive_scale(normr, v, hopradius::Number, maxhopradius) = Vec3f(hopradius, hopradius, norm(v)/2)
primitive_scale(normr, v, hopradius, maxhopradius) = Vec3f(normr * maxhopradius, normr * maxhopradius, norm(v)/2)

function primitive_segments(p::HoppingPrimitives{E}, plot) where {E}
    segments = Point{E,Float32}[]
    for (r, dr) in zip(p.centers, p.vectors)
        push!(segments, r - dr/2)
        push!(segments, r + dr/2)
    end
    return segments
end

function primitive_linewidths(p::HoppingPrimitives{E}, plot) where {E}
    isempty(p.radii) && return Float32[]
    pixelscale = plot[:pixelscale][]
    hopradius = plot[:hopradius][]
    minr, maxr = extrema(p.radii)
    linewidths = Float32[]
    for r in p.radii
        linewidth = primitive_linewidth(normalize_range(r, (minr, maxr)), hopradius, pixelscale)
        append!(linewidths, (linewidth, linewidth))
    end
    return linewidths
end

primitive_linewidth(normr, hopradius::Number, pixelscale) = pixelscale
primitive_linewidth(normr, hopradius, pixelscale) = pixelscale * normr


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