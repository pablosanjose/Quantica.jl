using .VegaLite

"""
    vlplot(b::Bandstructure{1}; kw...)

Plots the 1D bandstructure `b` using VegaLite.

    vlplot(h::Hamiltonian; kw...)

Plots the the Hamiltonian lattice projected along `axes` using VegaLite.

# Keyword arguments and defaults:
    - `size = 800`: the `(width, height)` of the plot (or `max(width, height)` if a single number)
    - `points = false`: whether to plot points on line plots
    - `labels`: labels for the x and y plot axes. Defaults `("φ/2π", "ε")` and `("x", "y")` respectively
    - `scaling = (1/2π, 1)`: `(scalex, scaley)` scalings for the x (Bloch phase) and y (energy) variables
    - `xlims = missing` and `ylims = missing`: `(xmin, xmax)` and `(ymin, ymax)` to constrain plot range
    - `bands = missing`: bands to plot (or all if `missing`)
    - `axes = (1,2)`: lattice axes to project onto the plot x-y plane
    - `digits = 4`: number of significant digits to show in onsite energy and hopping tooltips
    - `plotsites = true`: whether to plot sites
    - `plotlinks = true`: whether to plot links
    - `sitesize = 15`: diameter of sites in pixels. Can be a function of site index.
    - `siteopacity = 0.9`: opacity of sites. Can be a function of site index.
    - `linksize = 0.25`: thickness of hopping links as a fraction of sitesize. Can be a function of site indices.
    - `linkopacity = 1.0`: opacity of hopping links. Can be a function of site indices.
    - `sitecolor = missing`: function of site index that returns a real to be enconded into a color, using `colorscheme`
    - `linkcolor = sitecolor`: function of link indices that returns a real to be enconded into a color, using `colorscheme`
    - `colorscheme`: Color scheme from `https://vega.github.io/vega/docs/schemes/` to be used (defaults "category10" or "lightgreyred")
"""
function VegaLite.vlplot(b::Bandstructure;
    labels = ("φ", "ε"), scaling = (1, 1), size = 640, points = false, xlims = missing, ylims = missing, bands = missing,
    sitesize = 30)
    labelx, labely = labels
    table = bandtable(b, make_it_two(scaling), bands)
    sizes = make_it_two(size)
    corners = _corners(table)
    plotrange = (xlims, ylims)
    (domainx, domainy), _ = domain_size(corners, size, plotrange)
    p = table |> vltheme(sizes, points) + @vlplot(
        mark = :line,
        x = {:x, scale = {domain = domainx, nice = false}, title = labelx, sort = nothing},
        y = {:y, scale = {domain = domainy, nice = false}, title = labely, sort = nothing},
        color = "band:n",
        selection = {grid = {type = :interval, bind = :scales}})
    return p
end

function bandtable(b::Bandstructure{1}, (scalingx, scalingy), bandsiter)
    bandsiter´ = bandsiter === missing ? eachindex(bands(b)) : bandsiter
    ks = vertices(b.kmesh)
    bnds = bands(b)
    table = [(x = v[1] * scalingx, y = v[2] * scalingy, band = i, tooltip = string(v))
             for i in bandsiter´ for v in vertices(bnds[i])]
    return table
end

function VegaLite.vlplot(h::Hamiltonian{LA};
                         labels = ("x","y"), size = 800, axes::Tuple{Int,Int} = (1,2), xlims = missing, ylims = missing, digits = 4,
                         sitesize = 15, siteopacity = 0.9, linksize = 0.25, linkopacity = 1.0,
                         sitecolor = missing, linkcolor = sitecolor, colorscheme = "lightgreyred", discretecolorscheme = "category10",
                         plotsites = true, plotlinks = true) where {E,LA<:Lattice{E}}
    directives = (; axes = axes, digits = digits,
                    sitesize_func = sitefunc(sitesize), siteopacity_func = sitefunc(siteopacity),
                    linksize_func = linkfunc(linksize), linkopacity_func = linkfunc(linkopacity),
                    sitecolor_func = sitefunc(sitecolor), linkcolor_func = sitefunc(linkcolor))
    table      = linkstable(h, directives)
    maxthick   = maximum(s -> ifelse(s.islink, s.scale, zero(s.scale)), table)
    maxsize    = plotsites ? maximum(s -> ifelse(s.islink, zero(s.scale), s.scale), table) : sqrt(15*maxthick)
    maxopacity = maximum(s -> s.opacity, table)
    corners    = _corners(table)
    plotrange  = (xlims, ylims)
    (domainx, domainy), sizes = domain_size(corners, size, plotrange)
    linkcolorfield = ifelse(linkcolor === missing, :sublat, :color)
    sitecolorfield = ifelse(sitecolor === missing, :sublat, :color)
    if (plotsites && sitecolor !== missing) || (plotlinks && linkcolor !== missing)
        colorscheme´ = colorscheme
        colorrange = extrema(s -> s.color, table)
    else
        colorscheme´ = discretecolorscheme
        colorrange = nothing
    end
    p = vltheme(sizes)
    if plotlinks
        p += @vlplot(
            mark = {:rule},
            size = {:scale,
                scale = {range = [0, (maxsize)^2], domain = [0, maxthick], clamp = false},
                legend = needslegend(linksize)},
            color = {linkcolorfield,
                scale = {domain = colorrange, scheme = colorscheme´},
                legend = needslegend(sitecolor)},
            opacity = {:opacity,
                scale = {range = [0, 1], domain = [0, maxopacity]},
                legend = needslegend(linkopacity)},
            transform = [{filter = "datum.islink"}],
            selection = {grid2 = {type = :interval, bind = :scales}},
            encoding = {
                x = {:x, scale = {domain = domainx, nice = false}, axis = {grid = false}},
                y = {:y, scale = {domain = domainy, nice = false}, axis = {grid = false}},
                x2 = {:x2, scale = {domain = domainx, nice = false}, axis = {grid = false}},
                y2 = {:y2, scale = {domain = domainy, nice = false}, axis = {grid = false}}})
    end
    if plotsites
        p += @vlplot(
            mark = {:circle, stroke = :black},
            size = {:scale,
                scale = {range = [0, (maxsize)^2], domain = [0, maxsize], clamp = false},
                legend = needslegend(sitesize)},
            color = {sitecolorfield,
                scale = {domain = colorrange, scheme = colorscheme´},
                legend = needslegend(sitecolor)},
            opacity = {:opacity,
                scale = {range = [0, 1], domain = [0, maxopacity]},
                legend = needslegend(siteopacity)},
            selection = {grid1 = {type = :interval, bind = :scales}},
            transform = [{filter = "!datum.islink"}],
            encoding = {
                x = {:x, scale = {domain = domainx, nice = false}, axis = {grid = false}, title = labels[1]},
                y = {:y, scale = {domain = domainy, nice = false}, axis = {grid = false}, title = labels[2]}
            })
    end
    return table |> p
end

needslegend(x::Number) = nothing
needslegend(x) = true

function vltheme((sizex, sizey), points = false)
    p = @vlplot(
        tooltip = :tooltip,
        width = sizex, height = sizey,
        config = {
            circle = {stroke = :black, strokeWidth = 1, size = 200},
            line = {point = points},
            rule = {strokeWidth = 3},
            scale = {minOpacity = 0, maxOpacity = 1.0}})
    return p
end

sitefunc(f::Function) = f
sitefunc(r::Number) = i -> r
sitefunc(::Missing) = i -> 0.0

linkfunc(f::Function) = f
linkfunc(t) = (i, j) -> t

function linkstable(h::Hamiltonian, d)
    (a1, a2) = d.axes
    lat = h.lattice
    T = numbertype(lat)
    slats = sublats(lat)
    rs = allsitepositions(lat)
    table = NamedTuple{(:x, :y, :x2, :y2, :sublat, :tooltip, :scale, :color, :opacity, :islink),
                       Tuple{T,T,T,T,NameType,String,Float64,Float64,Float64,Bool}}[]
    h0 = h.harmonics[1].h
    rows = Int[] # list of plotted destination sites for dn != 0

    for har in h.harmonics
        resize!(rows, 0)
        ridx = 0
        for ssrc in slats
            if iszero(har.dn)
                for (i, r) in enumeratesites(lat, ssrc)
                    x = get(r, a1, zero(T)); y = get(r, a2, zero(T))
                    ridx += 1
                    push!(table, (x = x, y = y, x2 = x, y2 = y,
                                sublat = sublatname(lat, ssrc), tooltip = matrixstring_inline(i, h0[i, i], d.digits),
                                scale = d.sitesize_func(ridx), color = d.sitecolor_func(ridx),
                                opacity = d.siteopacity_func(ridx), islink = false))
                end
            end
            for sdst in slats
                itr = nonzero_indices(har, siterange(lat, sdst), siterange(lat, ssrc))
                for (row, col) in itr
                    iszero(har.dn) && col == row && continue
                    rsrc = rs[col]
                    rdst = (rs[row] + bravais(lat) * har.dn)
                    # sites in neighboring cells connected to dn=0 cell. But add only once.
                    if !iszero(har.dn) && !in(row, rows)
                        x = get(rdst, a1, zero(T)); y = get(rdst, a2, zero(T))
                        push!(table,
                            (x = x, y = y, x2 = x, y2 = y,
                            sublat = sublatname(lat, sdst), tooltip = matrixstring_inline(row, h0[row, row], d.digits),
                            scale =  d.sitesize_func(row), color = d.sitecolor_func(row),
                            opacity = 0.5 * d.siteopacity_func(row), islink = false))
                        push!(rows, row)
                    end
                    # draw half-links but only intracell
                    rdst = ifelse(iszero(har.dn), (rdst + rsrc) / 2, rdst)
                    x  = get(rsrc, a1, zero(T)); y  = get(rsrc, a2, zero(T))
                    x´ = get(rdst, a1, zero(T)); y´ = get(rdst, a2, zero(T))
                    # Exclude links perpendicular to the screen
                    rdst ≈ rsrc || push!(table,
                        (x = x, y = y, x2 = x´, y2 = y´,
                        sublat = sublatname(lat, ssrc), tooltip = matrixstring_inline(row, col, har.h[row, col], d.digits),
                        scale = d.linksize_func(row, col), color = d.linkcolor_func(col),
                        opacity = ifelse(iszero(har.dn), 1.0, 0.5) * d.linkopacity_func(row, col), islink = true))
                end
            end
        end
    end
    return table
end

function _corners(table)
    min´ = max´ = (first(table).x, first(table).y)
    for row in table
        min´ = min.(min´, (row.x, row.y))
        max´ = max.(max´, (row.x, row.y))
    end
    return min´, max´
end

function domain_size(corners, size, (rangex, rangey))
    domainx = iclamp((corners[1][1], corners[2][1]), rangex)
    domainy = iclamp((corners[1][2], corners[2][2]), rangey)
    dx = domainx[2]-domainx[1]
    dy = domainy[2]-domainy[1]
    sizex, sizey = compute_sizes(size, (dx, dy))
    return (domainx, domainy), (sizex, sizey)
end

make_it_two(x::Number) = (x, x)
make_it_two(x::Tuple{Number,Number}) = x

function compute_sizes(size::Number, (dx, dy))
    if dx > dy
        sizex = size
        sizey = size * dy/dx
    else
        sizex = size * dx/dy
        sizey = size
    end
    return sizex, sizey
end

compute_sizes(ss::Tuple{Number,Number}, d) = ss
