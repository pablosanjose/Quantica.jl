module QuanticaMakieExt

using Makie
using Quantica
using Makie.GeometryBasics
using Makie.GeometryBasics: Ngon
using Quantica: Lattice, AbstractHamiltonian, Harmonic, Bravais, SVector,
      foreach_hop, foreach_site, dcell,
      harmonics, sublats, site, norm, normalize

import Quantica: qplot, qplot!

############################################################################################
# QPlot recipe
#region

@recipe(QPlot) do scene
    Theme(
        ssao = true,
        fxaa = true,
        ambient = Vec3f(0.5),
        diffuse = Vec3f(0.5),
        backlight = 4.0f0,
        shaded = false,
        boundary_dimming = 0.95,
        cell_dimming = 0.97,
        siteradius = 0.2,
        siteborder = 3,
        siteborderdarken = 0.6,
        hopthickness = 6,
        hopradius = 0.03,
        hopoffset = 1.0,
        hopdarken = 0.85,
        supercell = missing,
        tooltips = true,
        digits = 3,
        axes = missing, # can be e.g. (1,2) to project onto the x,y plane
        hide = nothing, # :hops, :sites, :bravais, :cell, :axes...
        cellfaces = false,
        colormap = :Spectral_9,
    )
end

Makie.plot!(plot::QPlot) = qplotdispatch!(plot, to_value(plot[1]))

#endregion

############################################################################################
# QPlot for AbstractHamiltonian and Lattice
#region

function qplotdispatch!(plot::QPlot, h::AbstractHamiltonian)
    lat = Quantica.lattice(h)
    E = Quantica.embdim(lat)
    colors = Iterators.cycle(RGBAf.(Makie.ColorSchemes.colorschemes[plot[:colormap][]]))

    hidesites = ishidden((:sites, :all), plot)
    hidehops = ishidden((:hops, :hoppings, :links, :all), plot)
    hidebravais = ishidden((:bravais, :all), plot)
    hideboundary = ishidden((:boundary, :all), plot)

    sc = plot[:supercell][]

    # plot bravais axes
    if !hidebravais
        plotbravais!(plot, lat, sc)
    end

    if sc !== missing && sc > 1
        h = supercell(h, sc)
        lat = Quantica.lattice(h)
    end

    # plot hoppings
    if !hidehops
        for har in reverse(harmonics(h))  # Draw intracell hops last
            hideboundary && !iszero(dcell(har)) && continue
            for (sublatsrc, color) in zip(sublats(lat), colors)
                color´ = darken(color, plot[:hopdarken][])
                color´´ = maybedim(color´, dcell(har), plot[:boundary_dimming][])
                plothops!(plot, har, lat, sublatsrc, color´´, hidesites)
            end
        end
    end

    # plot sites
    if !hidesites
        site_label = site_tooltip(h)
        for har in harmonics(h)
            hideboundary && !iszero(dcell(har)) && continue
            colors´ = RGBAf[]
            sites = Point{E,Float32}[]
            for (sublat, color) in zip(sublats(lat), colors)
                color´ = maybedim(color, dcell(har), plot[:boundary_dimming][])
                len = length(sites)
                append_harmonic_sites!(sites, har, lat, sublat)
                append!(colors´, Iterators.repeated(color´, length(sites) - len))
            end
            plotsites!(plot, sites, colors´, site_label)
        end
    end

    return plot
end

function qplotdispatch!(plot::QPlot, lat::Lattice)
    E = Quantica.embdim(lat)
    colors = Iterators.cycle(RGBAf.(Makie.ColorSchemes.colorschemes[plot[:colormap][]]))

    sites = Point{E,Float32}[]
    offset = 0
    colors´ = RGBAf[]
    for dn in dnshell(lat)
        plot[:onlyonecell][] && !iszero(dn) && continue
        for (sublat, color) in zip(sublats(lat), colors)
            append_dcell_sites!(sites, dn, lat, sublat)
            color´ = iszero(dn) ? color : transparent(color, 1 - plot[:boundary_dimming][])
            colors´ = append!(colors´, Iterators.repeated(color´, length(sites) - offset))
            offset = length(sites)
        end
    end

    site_label = site_tooltip(lat)
    plotsites!(plot, sites, colors´, site_label)

    return plot
end

function plotsites!(plot::QPlot, sites::Vector{<:Point{E}}, color, inspector_label) where {E}
    if E == 3 && plot[:shaded][]
        meshscatter!(plot, sites; color,
            markerspace = :data,
            markersize = plot[:siteradius][],
            ssao = plot[:ssao][],
            ambient = plot[:ambient][],
            diffuse = plot[:diffuse][],
            backlight = plot[:backlight][],
            fxaa = plot[:fxaa][],
            transparency = istransparent(color),
            inspector_label)
    else
        scatter!(plot, sites; color,
            markerspace = :data,
            markersize = sqrt(2) * 2 * plot[:siteradius][],
            strokewidth = plot[:siteborder][],
            strokecolor = darken(color, plot[:siteborderdarken][]),
            ambient = plot[:ambient][],
            diffuse = plot[:diffuse][],
            backlight = plot[:backlight][],
            fxaa = plot[:fxaa][],
            transparency = istransparent(color),
            inspector_label)
    end
    return plot
end

function plothops!(plot::QPlot, har, lat::Lattice{<:Any,E}, sublatsrc, color, hidesites) where {E}
    if E == 3 && plot[:shaded][]
        cyl = Cylinder(Point3f(0., 0., -1.0), Point3f(0., 0, 1.0), Float32(1))
        radius = plot[:hopradius][]
        offset = ifelse(hidesites, 0.0, 0.95 * plot[:hopoffset][]*plot[:siteradius][])
        centers, vectors, scales = centers_vectors_and_scales(har, lat, radius, sublatsrc; offset)
        meshscatter!(plot, centers; color,
            rotations = vectors, markersize = scales, marker = cyl,
            ssao = plot[:ssao][],
            ambient = plot[:ambient][],
            diffuse = plot[:diffuse][],
            backlight = plot[:backlight][],
            fxaa = plot[:fxaa][],
            transparency = istransparent(color))
    else
        offset = ifelse(hidesites, 0.0, plot[:hopoffset][] * plot[:siteradius][])
        segments = Point{E,Float32}[]
        append_segment!(segments, har, lat, sublatsrc; offset)
        linesegments!(plot, segments; color,
            linewidth = plot[:hopthickness][],
            backlight = plot[:backlight][],
            fxaa = plot[:fxaa][],
            transparency = istransparent(color))
    end
    return plot
end

istransparent(colors::Vector) = istransparent(first(colors))
istransparent(color::RGBAf) = color.alpha != 1.0

function plotbravais!(plot::QPlot, lat::Lattice{<:Any,E,L}, supercell) where {E,L}
    bravais = Quantica.bravais(lat)
    vs = Point{E}.(Quantica.bravais_vectors(bravais))
    vtot = sum(vs)
    r0 = Point{E,Float32}(Quantica.mean(Quantica.sites(lat))) - 0.5 * vtot

    if !ishidden(:axes, plot)
        for (v, color) in zip(vs, (:red, :green, :blue))
            arrows!(plot, [r0], [v]; color, inspectable = false)
        end
    end

    if !ishidden(:cell, plot)
        colface = RGBAf(0,0,1,1-plot[:cell_dimming][])
        coledge = RGBAf(0,0,1, 5 * (1-plot[:cell_dimming][]))
        shifts = supercell === missing ? (zero(SVector{L,Int}),) : dnshell(lat, 0:supercell-1)
        rect = Rect{L,Int}(Point{L,Int}(0), Point{L,Int}(1))
        mrect0 = GeometryBasics.mesh(rect, pointtype=Point{L,Float32}, facetype=QuadFace{Int})
        vertices0 = mrect0.position
        mat = Quantica.bravais_matrix(bravais)
        for shift in shifts
            mrect = GeometryBasics.mesh(rect, pointtype=Point{E,Float32}, facetype=QuadFace{Int})
            vertices = mrect.position
            vertices .= Ref(r0) .+ Ref(mat) .* (vertices0 .+ Ref(shift))
            plot[:cellfaces][] &&
                mesh!(plot, mrect; color = colface, transparency = true, inspectable = false)
            wireframe!(plot, mrect; color = coledge, transparency = true, strokewidth = 1, inspectable = false)
        end
    end

    return plot
end


############################################################################################
# tooltips
#region

function site_tooltip(h::AbstractHamiltonian)
    return (self, i, p) -> matrixstring(i, h[][i,i])
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
# tools
#region

toPoint32(p::SVector{1}) = Point2f(SVector{2,Float32}(first(p), zero(Float32)))
toPoint32(p::SVector{2}) = Point2f(p)
toPoint32(p::SVector{3}) = Point3f(p)
toPoint32(ps::Pair{S,S}) where {S<:SVector} = toPoint32(first(ps)) => toPoint32(last(ps))

function darken(rgba::RGBAf, v = 0.66)
    r = max(0, min(rgba.r * (1 - v), 1))
    g = max(0, min(rgba.g * (1 - v), 1))
    b = max(0, min(rgba.b * (1 - v), 1))
    RGBAf(r,g,b,rgba.alpha)
end

darken(colors::Vector, v = 0.66) = darken.(colors, Ref(v))

transparent(rgba::RGBAf, v = 0.5) = RGBAf(rgba.r, rgba.g, rgba.b, rgba.alpha * v)

maybedim(color, dn, dimming) = iszero(dn) ? color : transparent(color, 1 - dimming)

function append_dcell_sites!(sites, dn::SVector, lat::Lattice, sublatsrc)
    for i in siterange(lat, sublatsrc)
        push!(sites, site(lat, i, dn))
    end
    return sites
end

function append_harmonic_sites!(sites, har::Harmonic, lat::Lattice, sublatsrc)
    if iszero(dcell(har))
        foreach_site(lat, sublatsrc) do r
            push!(sites, toPoint32(r))
        end
    else
        foreach_hop(har, lat, sublatsrc) do pair, dn
            src, dst = pair
            push!(sites, toPoint32(src))
            push!(sites, toPoint32(dst))
        end
    end
    return sites
end

function append_segment!(segments, args...; offset = 0.0)
    foreach_hop(args...) do pair, dn
        src, dst = offset_pair(pair, offset, dn)
        push!(segments, toPoint32(src))
        push!(segments, toPoint32(dst))
    end
    return segments
end

function centers_vectors_and_scales(har::Harmonic, lat, radius, sublatsrc; offset = 0.0)
    centers = Point3f[]
    vectors = Vec3f[]
    scales  = Vec3f[]
    foreach_hop(har, lat, sublatsrc) do pair, dn
        src, dst = offset_pair(pair, offset, dn)
        center, vector = (src + dst)/2, (dst - src)
        scale = Vec3f(radius, radius, norm(vector)/2)
        push!(centers, center)
        push!(vectors, vector)
        push!(scales, scale)
    end
    return centers, vectors, scales
end

function offset_pair(pair, offset, dn)
    src, dst = pair
    iszero(dn) && (dst = (src + dst)/2)
    if !iszero(offset)
        nvec = normalize(dst - src)
        src += offset * nvec
        iszero(dn) || (dst -= offset * nvec)
    end
    return src, dst
end


dnshell(::Lattice{<:Any,<:Any,L}, span = -1:1) where {L} =
    sort!(vec(SVector.(Iterators.product(ntuple(_ -> span, Val(L))...))), by = norm)

ishidden(s, plot::QPlot) = ishidden(s, plot[:hide][])
ishidden(s, ::Nothing) = false
ishidden(s::Symbol, hide::Symbol) = s === hide
ishidden(s::Symbol, hides::Tuple) = s in hides
ishidden(ss, hides) = any(s -> ishidden(s, hides), ss)

#endregion

############################################################################################
# convert_arguments
#region

function Makie.convert_arguments(::PointBased, lat::Lattice{<:Any,E}, sublat = missing) where {E}
    lat = lattice(h)
    sites = Point{E,Float32}[]
    dns = dnshell(lat)
    for dn in dns
        append_dcell_sites!(sites, dn, lat, sublatsrc)
    end
    return (sites,)
end

function Makie.convert_arguments(::PointBased, h::AbstractHamiltonian{<:Any,E}, sublatsrc = missing) where {E}
    lat = lattice(h)
    sites = Point{E,Float32}[]
    for har in harmonics(h)
        append_harmonic_sites!(sites, har, lat, sublatsrc)
    end
    return (sites,)
end

# linesegments is also PointBased
function Makie.convert_arguments(::Type{<:LineSegments}, h::AbstractHamiltonian{<:Any,E}, sublatsrc = missing) where {E}
    segments = Point{E,Float32}[]
    append_segment!(segments, h, sublatsrc)
    return (segments,)
end

function Makie.convert_arguments(::Type{<:LineSegments}, har::Harmonic, lat::Lattice{<:Any,E}, sublatsrc = missing) where {E}
    segments = Point{E,Float32}[]
    append_segment!(segments, har, lat, sublatsrc)
    return (segments,)
end

#endregion

############################################################################################
# Old
#region

function addtooltips!(scene, h)
    sceneplot = scene[end]
    visible = Node(false)
    N = Quantica.blockdim(h)
    poprect = lift(scene.events.mouseposition) do mp
        Rectf((mp .+ 5), 1,1)
    end
    textpos = lift(scene.events.mouseposition) do mp
        Vec3f((mp .+ 5 .+ (0, -50))..., 0)
    end
    popup = poly!(campixel(scene), poprect, raw = true, color = RGBAf(1,1,1,0), visible = visible)
    translate!(popup, Vec3f(0, 0, 10000))
    text!(popup, " ", textsize = 30, position = textpos, align = (:center, :center),
        color = :black, strokewidth = 4, strokecolor = :white, raw = true, visible = visible)
    text_field = popup.plots[end]
    on(scene.events.mouseposition) do event
        subplot, idx = mouse_selection(scene)
        layer = findfirst(isequal(subplot), sceneplot.plots)
        if layer !== nothing && idx > 0
            idx´ = fix_linesegments_bug(idx, subplot)
            txt = popuptext(sceneplot, layer, idx´, h)
            text_field[1] = txt
            visible[] = true
        else
            visible[] = false
        end
        return
    end
    return scene
end

# idx returned by mouse_selection seems wrong by a factor 2 in LineSegments subplot
fix_linesegments_bug(idx, subplot::LineSegments) = Int(idx/2) # catches odd idx
fix_linesegments_bug(idx, subplot) = idx

function popuptext(sceneplot, layer, idx, h)
    cache = sceneplot[:_tooltips_rowcolhar][]
    checkbounds(Bool, cache, layer) || return string("Bug in layer: ", layer, " of ", length(cache))
    checkbounds(Bool, cache[layer], idx) || return string("Bug in idx: ", idx, " of ", length(cache[layer]))
    (row, col_or_zero, haridx) = cache[layer][idx]
    if col_or_zero == 0
        col = iszero(col_or_zero) ? row : col_or_zero
        har = h.harmonics[1]
    else
        col = col_or_zero
        har = h.harmonics[haridx]
    end
    element = round.(har.h[row, col], digits = sceneplot[:digits][])
    txt = iszero(col_or_zero) ? matrixstring(col, element) : matrixstring(row, col, element)
    return txt
end

#endregion

end # module