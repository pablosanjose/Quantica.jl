using .VegaLite

"""
    vlplot(b::Bandstructure{1}; size = 640, points = false, labels = ("φ/2π", "ε"), scaling = (1/2π, 1), range = missing)

Plots the 1D bandstructure `b` using VegaLite.

    vlplot(h::Hamiltonian; size = 800, labels = ("x", "y"), axes::Tuple{Int,Int} = (1,2), range = missing)

Plots the the Hamiltonian lattice projected along `axes` using VegaLite.

# Options:
    - `size`: the `(width, height)` of the plot (or `width == height` if a single number)
    - `points`: whether to plot points on line plots
    - `labels`: labels for the x and y plot axes
    - `scaling`: `(scalex, scaley)` scalings for the x (Bloch phase) and y (energy) variables
    - `range`: `(ymin, ymax)` or `((xmin, xmax), (ymin, ymax))` to constrain plot range
    - `axes`: lattice axes to project onto the plot x-y plane
"""
function VegaLite.vlplot(b::Bandstructure; labels = ("φ", "ε"), scaling = (1, 1), size = 640, points = false, range = missing)
    labelx, labely = labels
    table = bandtable(b, make_it_two(scaling))
    sizes = make_it_two(size)
    corners = _corners(table)
    range´ = sanitize_plotrange(range)
    (domainx, domainy), _ = domain_size(corners, size, range´)
    p = table |> vltheme(sizes, points) + @vlplot(
        mark = :line,
        x = {:x, scale = {domain = domainx, nice = false}, title = labelx},
        y = {:y, scale = {domain = domainy, nice = false}, title = labely},
        color = "band:n",
        selection = {grid = {type = :interval, bind = :scales}})
    return p
end

function bandtable(b::Bandstructure{1}, (scalingx, scalingy) = (1, 1))
    ks = vertices(b.kmesh)
    table = [(x = v[1] * scalingx, y = v[2] * scalingy, band = i, tooltip = string(v))
             for (i, bnd) in enumerate(bands(b)) for v in vertices(bnd)]
    return table
end

function VegaLite.vlplot(h::Hamiltonian{LA};
                         labels = ("x","y"), size = 800, axes::Tuple{Int,Int} = (1,2), range = missing) where {E,LA<:Lattice{E}}
    table = linkstable(h, axes)
    corners = _corners(table)
    range´ = sanitize_plotrange(range)
    (domainx, domainy), sizes = domain_size(corners, size, range´)
    p = table |> vltheme(sizes) +
        @vlplot(:rule, color = :sublat, opacity = {:opacity, legend = nothing},
            transform = [{filter = "datum.islink"}],
            selection = {grid2 = {type = :interval, bind = :scales}},
            encoding = {x = :x, y = :y, x2 = :x2, y2 = :y2}) +
        @vlplot(:circle, color = :sublat, opacity = {:opacity, legend = nothing},
            selection = {grid1 = {type = :interval, bind = :scales}},
            transform = [{filter = "!datum.islink"}],
            encoding = {
                x = {:x, scale = {domain = domainx, nice = false}, axis = {grid = false}, title = labels[1]},
                y = {:y, scale = {domain = domainy, nice = false}, axis = {grid = false}, title = labels[2]}
            })
    return p
end

function linkstable(h::Hamiltonian, (a1, a2) = (1, 2))
    lat = h.lattice
    T = numbertype(lat)
    slats = sublats(lat)
    rs = allsitepositions(lat)
    table = NamedTuple{(:x, :y, :x2, :y2, :sublat, :tooltip, :opacity, :islink),
                       Tuple{T,T,T,T,NameType,String,Float64,Bool}}[]
    h0 = h.harmonics[1].h
    rows = Int[] # list of plotted destination sites for dn != 0
    for har in h.harmonics
        resize!(rows, 0)
        for ssrc in slats
            if iszero(har.dn)
                for (i, r) in enumeratesites(lat, ssrc)
                    x = get(r, a1, zero(T)); y = get(r, a2, zero(T))
                    push!(table, (x = x, y = y, x2 = x, y2 = y,
                                sublat = sublatname(lat, ssrc), tooltip = matrixstring_inline(i, h0[i, i]),
                                opacity = 1.0, islink = false))
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
                            sublat = sublatname(lat, sdst), tooltip = matrixstring_inline(row, h0[row, row]),
                            opacity = 0.5, islink = false))
                        push!(rows, row)
                    end
                    # draw half-links but only intracell
                    rdst = iszero(har.dn) ? (rdst + rsrc) / 2 : rdst
                    x  = get(rsrc, a1, zero(T)); y  = get(rsrc, a2, zero(T))
                    x´ = get(rdst, a1, zero(T)); y´ = get(rdst, a2, zero(T))
                    # Exclude links perpendicular to the screen
                    rdst ≈ rsrc || push!(table,
                        (x = x, y = y, x2 = x´, y2 = y´,
                        sublat = sublatname(lat, ssrc), tooltip = matrixstring_inline(row, col, har.h[row, col]),
                        opacity = iszero(har.dn) ? 1.0 : 0.5, islink = true))
                end
            end
        end
    end
    return table
end

function vltheme((sizex, sizey), points = false)
    p = @vlplot(
        tooltip = :tooltip,
        width = sizex, height = sizey,
        config = {
            circle = {stroke = :black, strokeWidth = 1, size = 200},
            line = {point = points},
            rule = {strokeWidth = 3}})
    return p
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
    sizex, sizey = make_it_two(size)
    domainx = iclamp((corners[1][1], corners[2][1]), rangex)
    domainy = iclamp((corners[1][2], corners[2][2]), rangey)
    dx = domainx[2]-domainx[1]
    dy = domainy[2]-domainy[1]
    if dx > dy
        sizex = size
        sizey = size * dy/dx
    else
        sizex = size * dx/dy
        sizey = size
    end
    return (domainx, domainy), (sizex, sizey)
end

sanitize_plotrange(::Missing) = (missing, missing)
sanitize_plotrange(yrange::Tuple{Number, Number}) = (missing, yrange)
sanitize_plotrange(ranges::NTuple{2,Tuple{Number, Number}}) = ranges

make_it_two(x::Number) = (x, x)
make_it_two(x::Tuple{Number,Number}) = x