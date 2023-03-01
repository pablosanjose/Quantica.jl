module QuanticaMakieExt

using Makie
using Quantica
using Makie.GeometryBasics
using Quantica: Lattice, AbstractHamiltonian, Harmonic, Bravais, SVector, foreach_hop, dcell,
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
        dimming = 0.95,
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
        hide = nothing, # :hops, :sites, :bravais
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
    hidedimmed = ishidden((:dimmed, :all), plot)

    sc = plot[:supercell][]

    # plot bravais axes
    if !hidebravais
        plotbravais!(plot, lat, sc)
    end

    if sc !== missing
        h = supercell(h, sc)
        lat = Quantica.lattice(h)
    end

    # plot hoppings
    if !hidehops
        for har in reverse(harmonics(h))  # Draw intracell hops last
            hidedimmed && !iszero(dcell(har)) && continue
            for (sublatsrc, color) in zip(sublats(lat), colors)
                color´ = darken(color, plot[:hopdarken][])
                color´´ = maybedim(color´, dcell(har), plot[:dimming][])
                plothops!(plot, har, lat, sublatsrc, color´´, hidesites)
            end
        end
    end

    # plot sites
    if !hidesites
        for har in harmonics(h)
            hidedimmed && !iszero(dcell(har)) && break
            for (sublat, color) in zip(sublats(lat), colors)
                color´ = maybedim(color, dcell(har), plot[:dimming][])
                sites = append_harmonic_sites!(Point{E,Float32}[], har, lat, sublat)
                plotsites!(plot, sites, color´)
            end
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
            color´ = iszero(dn) ? color : transparent(color, 1 - plot[:dimming][])
            colors´ = append!(colors´, Iterators.repeated(color´, length(sites) - offset))
            offset = length(sites)
        end
    end

    plotsites!(plot, sites, colors´)

    return plot
end

function plotsites!(plot::QPlot, sites::Vector{<:Point{E}}, color) where {E}
    if E == 3 && plot[:shaded][]
        meshscatter!(plot, sites; color,
            markerspace = :data,
            markersize = plot[:siteradius][],
            ssao = plot[:ssao][],
            ambient = plot[:ambient][],
            diffuse = plot[:diffuse][],
            backlight = plot[:backlight][],
            fxaa = plot[:fxaa][])
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
            overdraw = istransparent(color))
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
            overdraw = istransparent(color))
    else
        offset = ifelse(hidesites, 0.0, plot[:hopoffset][] * plot[:siteradius][])
        segments = Point{E,Float32}[]
        append_segment!(segments, har, lat, sublatsrc; offset)
        linesegments!(plot, segments; color,
            linewidth = plot[:hopthickness][],
            backlight = plot[:backlight][],
            fxaa = plot[:fxaa][],
            overdraw = istransparent(color))
    end
    return plot
end

istransparent(colors::Vector) = istransparen(first(colors))
istransparent(color::RGBAf) = color.alpha != 1.0

function plotbravais!(plot::QPlot, lat::Lattice{<:Any,E,L}, supercell) where {E,L}
    overdraw = !(E == 3 && plot[:shaded][])

    bravais = Quantica.bravais(lat)
    vs = Point{E}.(Quantica.bravais_vectors(bravais))
    vtot = sum(vs)
    r0 = Point{E,Float32}(Quantica.mean(Quantica.sites(lat))) - 0.5 * vtot

    if !ishidden(:axes, plot)
        r0s = [r0 for _ in 1:L]
        arrows!(plot, r0s, vs; color = [:red, :green, :blue])
    end

    shifts = supercell === missing ? (zero(SVector{L,Int}),) : dnshell(lat, 0:supercell-1)
    if !ishidden(:cell, plot)
        ILE = SMatrix{L,E,Int}(I)
        corner = SVector{E,Int}(ntuple(i -> ifelse(i<=L, 1, 0), Val(E)))
        for shift in shifts
            rect = Rect{E,Float32}(Point{E,Int}(0), Point{E,Int}(corner))
            m = GeometryBasics.mesh(rect)
            mat = Quantica.bravais_matrix(bravais) * ILE
            m.position .= Ref(r0) .+ Ref(mat) .* (m.position .+ Ref(ILE' * shift))
            segments = [[s..., first(s)] for s in Iterators.partition(m.position, 4)]
            mesh!(plot, m; color = RGBAf(0,0,1,0.05), overdraw)
            lines!.(Ref(plot), segments; color = RGBAf(0,0,0.5,0.25), strokewidth = 1)
        end
    end

    return plot
end

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
    foreach_hop(har, lat, sublatsrc) do pair, dn
        src, dst = pair
        push!(sites, toPoint32(src))
        !iszero(dn) && push!(sites, dst)
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