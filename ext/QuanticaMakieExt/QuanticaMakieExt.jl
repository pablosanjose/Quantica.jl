module QuanticaMakieExt

using Makie
using Quantica
using Makie.GeometryBasics
using Makie.GeometryBasics: Ngon
using Quantica: Lattice, AbstractHamiltonian, Harmonic, Bravais, SVector,
      foreach_hop, foreach_site, dcell, argerror,
      harmonics, sublats, site, norm, normalize, nsites

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
        sitecolor = :sublat,
        siteopacity = 1.0,
        siteradius = 0.2,
        siteborder = 3,
        siteborderdarken = 0.6,
        hopcolor = :sublat,
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

    # # plot hoppings
    # if !hidehops
    #     for har in reverse(harmonics(h))  # Draw intracell hops last
    #         hideboundary && !iszero(dcell(har)) && continue
    #         # centers, vectors, colors, sizes, offsets
    #         hop_prims = hopping_primitives(plot, har, lat)
    #         hop_tts = hopping_tooltips(har)
    #         if E == 3 && plot[:shaded][]
    #             plothops_shaded!(plot, hop_prims, hop_tts)
    #         else
    #             plothops_flat!(plot, hop_prims, hop_tts)
    #         end
    #     end
    # end

    # plot sites
    if !hidesites
        for har in harmonics(h)
            hideboundary && !iszero(dcell(har)) && continue
            # centers, colors´, sizes, offsets
            site_prims = site_primitives(plot, har, lat)
            site_tts = site_tooltip(h)
            if E == 3 && plot[:shaded][]
                plotsites_shaded!(plot, site_prims, site_tts)
            else
                plotsites_flat!(plot, site_prims, site_tts)
            end
        end
    end

    return plot
end

function qplotdispatch!(plot::QPlot, lat::Lattice)
    E = Quantica.embdim(lat)
    hideboundary = ishidden((:boundary, :all), plot)
    hidesites = ishidden((:sites, :all), plot)
    hidebravais = ishidden((:bravais, :all), plot)

    sc = plot[:supercell][]

    # plot bravais axes
    if !hidebravais
        plotbravais!(plot, lat, sc)
    end

    if sc !== missing && sc > 1
        lat = supercell(lat, sc)
    end

    # plot sites
    if !hidesites
        for dn in dnshell(lat)
            hideboundary && !iszero(dn) && continue
            # centers, colors´, sizes, offsets
            site_prims = site_primitives(plot, dn, lat)
            site_tts = site_tooltip(lat)
            if E == 3 && plot[:shaded][]
                plotsites_shaded!(plot, site_prims, site_tts)
            else
                plotsites_flat!(plot, site_prims, site_tts)
            end
        end
    end

    return plot
end

function plotsites_shaded!(plot::QPlot, (centers, colors, radii), inspector_label)
    meshscatter!(plot, centers; color = colors, markersize = radii,
            markerspace = :data,
            ssao = plot[:ssao][],
            ambient = plot[:ambient][],
            diffuse = plot[:diffuse][],
            backlight = plot[:backlight][],
            fxaa = plot[:fxaa][],
            transparency = istransparent(colors),
            inspector_label)
    return plot
end

function plotsites_flat!(plot::QPlot, (centers, colors, radii), inspector_label)
    scatter!(plot, centers; color = colors, markersize = radii,
            markerspace = :data,
            strokewidth = plot[:siteborder][],
            strokecolor = darken.(colors, Ref(plot[:siteborderdarken][])),
            ambient = plot[:ambient][],
            diffuse = plot[:diffuse][],
            backlight = plot[:backlight][],
            fxaa = plot[:fxaa][],
            transparency = istransparent(colors),
            inspector_label)
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
# site_primitives
#   centers, colors, radii for a given lattice and dn
#region

function site_primitives(plot, dn::SVector, lat::Lattice{<:Any,E}) where {E}
    # centers
    dr = Quantica.bravais_matrix(lat) * dn
    centers = [Point{E,Float32}(r + dr) for r in Quantica.sites(lat)]

    # colors
    colormap = Makie.ColorSchemes.colorschemes[plot[:colormap][]]
    colors = RGBAf[]
    sitecolor = plot[:sitecolor][]
    if sitecolor == :sublat
        cyclic_colors = Iterators.cycle(RGBAf.(colormap))
        for (s, color) in zip(sublats(lat), cyclic_colors)
            iszero(dn) || (color = transparent(color, 1 - plot[:boundary_dimming][]))
            append!(colors, Iterators.repeated(color, nsites(lat, s)))
        end
    elseif sitecolor isa AbstractVector
        length(sitecolor) == length(centers) ||
            argerror("The sitecolor vector has $(length(sitecolor)) elements, expected $(length(centers))")
        normalized_colors = normalize_float_range(sitecolor)
        resize!(colors, length(normalized_colors))
        colors .= getindex.(Ref(colormap), normalized_colors)
    elseif sitecolor isa Function
        normalized_colors = [sitecolor(i, r) for (i, r) in enumerate(centers)]
        normalize_float_range!(normalized_colors)
        resize!(colors, length(normalized_colors))
        colors .= getindex.(Ref(colormap), normalized_colors)
    else
        argerror("Unrecognized sitecolor")
    end

    # opacity
    siteopacity = plot[:siteopacity][]
    if siteopacity isa Real
        if siteopacity < 1
            colors .= transparent.(colors, siteopacity)
        end
    elseif siteopacity isa AbstractVector
        length(siteopacity) == length(centers) ||
            argerror("The siteopacity vector has $(length(siteopacity)) elements, expected $(length(centers))")
        normalized_opacity = normalize_float_range(siteopacity)
        colors .= transparent.(colors, normalized_opacity)
    elseif sitecolor isa Function
        normalized_opacity = [siteopacity(i, r) for (i, r) in enumerate(centers)]
        normalize_float_range!(normalized_opacity)
        colors .= transparent.(colors, normalized_opacity)
    else
        argerror("Unrecognized siteopacity")
    end


    # radii
    maxradius, siteradius = sanitize_siteradius(plot[:siteradius][])
    scatter_scaling = ifelse(plot[:shaded][], 1.0, (2 * sqrt(2)))
    radii = Float32[]
    if siteradius isa AbstractVector
        length(siteradius) == length(centers) ||
            argerror("The siteradius vector has $(length(siteradius)) elements, expected $(length(centers))")
        resize!(radii, length(centers))
        normalize_float_range!(radii, siteradius)
        radii .*= scatter_scaling * maxradius
    elseif siteradius isa Function
        normalized_radii = [siteradius(i, r) for (i, r) in enumerate(centers)]
        resize!(radii, length(centers))
        normalize_float_range!(radii, normalized_radii)
        radii .*= scatter_scaling * maxradius
    else
        argerror("Unrecognized siteradius")
    end

    return centers, colors, radii
end

sanitize_siteradius(x::Real) = (x, Returns(x))
sanitize_siteradius(f::Function) = (0.2, f)
sanitize_siteradius(v::AbstractVector) = (0.2, v)
sanitize_siteradius((x, f)::Tuple{Real,Function}) = (x, f)
sanitize_siteradius((x, v)::Tuple{Real,AbstractVector}) = (x, v)
sanitize_siteradius(_) = argerror("Invalid siteradius: expected a Real, a Function, an AbstractVector a Tuple{Real,Function} or a Tuple{Real,AbstractVector}")

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

#endregion

############################################################################################
# tooltips
#region

function site_tooltip(h::AbstractHamiltonian)
    return (self, i, p) -> matrixstring(i, h[][i,i])
end

function site_tooltip(lat::Lattice)
    return (self, i, p) -> positionstring(i, SVector(p))
end

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

dnshell(::Lattice{<:Any,<:Any,L}, span = -1:1) where {L} =
    sort!(vec(SVector.(Iterators.product(ntuple(_ -> span, Val(L))...))), by = norm)

ishidden(s, plot::QPlot) = ishidden(s, plot[:hide][])
ishidden(s, ::Nothing) = false
ishidden(s::Symbol, hide::Symbol) = s === hide
ishidden(s::Symbol, hides::Tuple) = s in hides
ishidden(ss, hides) = any(s -> ishidden(s, hides), ss)


normalize_float_range(v) = normalize_float_range!(similar(v), v)

function normalize_float_range!(dst::AbstractVector{T}, v::AbstractVector{<:AbstractFloat}) where {T}
    minv, maxv = extrema(v)
    if maxv - minv ≈ 0
        fill!(dst, T(one(eltype(v))))
    else
        @. dst = T((v - minv) / (maxv - minv))
    end
    return dst
end

normalize_float_range!(_...) = argerror("Unexpected input: expected a vector of floats")

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