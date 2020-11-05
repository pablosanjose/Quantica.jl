using .VegaLite

export DensityShader, CurrentShader

#######################################################################
# Shaders
#######################################################################
struct CurrentShader{K,T}
    axis::Int
    kernel::K
    transform::T
end

"""
    CurrentShader(axis = 0; kernel = 1, transform = identity)

Construct a `CurrentShader` object that can be used in `vlplot(h, psi; ...)` to visualize
the current of a state `psi` along a given `axis` under Hamiltonian `h`, optionally
transformed by `transform`. An `axis = 0` represent the norm of the current.

The current along a link `i=>k` is defined as

    j_ki = (r[k][axis]-r[i][axis]) * imag(psi[k]' * h[k,i] * kernel * psi[j])

The current at a site `i` is the sum of links incident on that site

    j_i = ∑_k j_ki

The visualized current is `transform(j_ki)` and `transform(j_i)`, respectively.
"""
CurrentShader(axis = 0; kernel = 1, transform = identity) =
    CurrentShader(axis, kernel, transform)

struct DensityShader{K,T}
    kernel::K
    transform::T
end

"""
    DensityShader(kernel = 1, transform = identity)

Construct a `DensityShader` object that can be used in `vlplot(h, psi; ...)` to visualize
the density of a state `psi`, optionally transformed by `transform`.

The visualized density at a site `i` is defined as

    ρ_i = transform(psi[i]' * kernel * psi[i])

The density at a half-link is equal to the density at the originating site.
"""
DensityShader(; kernel = 1, transform = identity) = DensityShader(kernel, transform)

function site_shader(shader::CurrentShader, psi::AbstractVector{T}, h) where {T}
    pos = allsitepositions(h.lattice)
    br = bravais(h.lattice)
    current = zeros(real(eltype(T)), length(pos))
    for hh in h.harmonics, (row, col) in nonzero_indices(hh)
        dr = pos[row] + br * hh.dn - pos[col]
        dj = imag(psi[row]' * shader.kernel * hh.h[row, col] * psi[col])
        j = iszero(shader.axis) ? real(shader.transform(norm(dr * dj))) : real(shader.transform(dr[shader.axis] * dj))
        current[row] += j
    end
    return i -> current[i]
end

site_shader(shader::DensityShader, psi::AbstractVector, h) =
    i -> real(shader.transform(psi[i]' * shader.kernel * psi[i]))

site_shader(shader::Function, psi, h) = i -> shader(psi[i])
site_shader(shader::Number, psi, h) = i -> shader
site_shader(shader::Missing, psi, h) = i -> 0.0

function link_shader(shader::CurrentShader, psi::AbstractVector{T}, h) where {T}
    pos = allsitepositions(h.lattice)
    # Assuming that only links from base unit cell are plotted
    h0 = first(h.harmonics).h
    current = (row, col) -> begin
        dr = pos[row] - pos[col]
        dj = imag(psi[row]' * shader.kernel * h0[row, col] * psi[col])
        j = iszero(shader.axis) ? shader.transform(norm(dr * dj)) : shader.transform(dr[shader.axis] * dj)
        return real(j)
    end
    return current
end

link_shader(shader::DensityShader, psi::AbstractVector, h) =
    (row,col) -> real(shader.transform(psi[col]' * shader.kernel * psi[col]))

link_shader(shader::Function, psi, h) = (row, col) -> shader(psi[row], psi[col])
link_shader(shader::Number, psi, h) = (row, col) -> shader
link_shader(shader::Missing, psi, h) = (row, col) -> 0.0

#######################################################################
# vlplot
#######################################################################
"""
    vlplot(b::Bandstructure{1}; kw...)

Plot the 1D bandstructure `b` using VegaLite in 2D.

    vlplot(h::Hamiltonian; kw...)

Plot the the Hamiltonian lattice projected along `axes` using VegaLite.

    vlplot(h::Hamiltonian, psi::AbstractVector; kw...)

Plot an eigenstate `psi` on the lattice, using one of various possible visualization
shaders: `sitesize`, `siteopacity`, `sitecolor`, `linksize`, `linkopacity` or `linkcolor`
(see keywords below). If `psi` is obtained in a flattened form, it will be automatically
"unflattened" to restore the orbital structure of `h`.

# Keyword arguments and defaults

## Common
    - `size = 800`: the `(width, height)` of the plot (or `max(width, height)` if a single number)
    - `labels`: labels for the x and y plot axes. Defaults `("φ/2π", "ε")` and `("x", "y")` respectively
    - `scaling = (1/2π, 1)`: `(scalex, scaley)` scalings for the x (Bloch phase) and y (energy) variables
    - `xlims = missing` and `ylims = missing`: `(xmin, xmax)` and `(ymin, ymax)` to constrain plot range

## Bandstructure-specifc
    - `points = false`: whether to plot points on line plots
    - `bands = missing`: bands to plot (or all if `missing`)
    - `thickness = 2`: thickness of band lines

## Hamiltonian-specific
    - `axes = (1,2)`:lattice axes to project onto the plot x-y plane
    - `digits = 4`: number of significant digits to show in onsite energy and hopping tooltips
    - `plotsites = true`: whether to plot sites
    - `plotlinks = true`: whether to plot links
    - `sitestroke = :white`: the color of site outlines. If `nothing`, no outline will be plotted.
    - `colorscheme = "redyellowblue"`: Color scheme from `https://vega.github.io/vega/docs/schemes/`
    - `discretecolorscheme = "category10"`: Color scheme from `https://vega.github.io/vega/docs/schemes/`
    - `maxdiameter = 15`: maximum diameter of sites, useful when `sitesize` is not a constant.
    - `maxthickness = 0.25`: maximum thickness of links, as a fraction of `maxdiameter`, useful when `linksize` is not a constant.
    - `sitesize = maxdiameter`: diameter of sites.
    - `siteopacity = 0.9`: opacity of sites.
    - `sitecolor = missing`: color of sites. If `missing`, colors will encode the site sublattice following `discretecolorscheme`
    - `linksize = maxthickness`: thickness of hopping links.
    - `linkopacity = 1.0`: opacity of hopping links.
    - `linkcolor = sitecolor`: color of links. If `missing`, colors will encode source site sublattice, following `discretecolorscheme`
    - `colorrange = missing`: the range of values to encode using `sitecolor` and `linkcolor`. If `missing` the minimum and maximum values will be used.

Apart from a number, all site shaders (`sitesize`, `siteopacity` and `sitecolor`) can also
be a real-valued function of the wavefunction amplitude `psi` at each site, `psi -> value(psi)`.
Similarly, link shaders (`linksize`, `linkopacity` and `linkcolor`) can be a function of the
wavefunction amplitudes at end and origin of the link, `(psi´, psi) -> value(psi´, psi)`.

Any shader can also be a `CurrentShader` or a `DensityShader` object, that computes the
current or density at each site or link (see `CurrentShader` and `DensityShader`).

Note that values of sizes are relative, so there is no need to normalize shader values. The
maximum visual value of site diameter and link thickness is given by `maxdiameter` (in
pixels) and `maxthickness` (as a fraction of `maxdiameter`)
"""
function VegaLite.vlplot(b::Bandstructure;
    labels = ("φ", "ε"), scaling = (1, 1), size = 640, points = false, xlims = missing, ylims = missing, bands = missing, thickness = 2)
    labelx, labely = labels
    table = bandtable(b, make_it_two(scaling), bands)
    sizes = make_it_two(size)
    corners = _corners(table)
    plotrange = (xlims, ylims)
    (domainx, domainy), _ = domain_size(corners, size, plotrange)
    p = table |> vltheme(sizes, points) + @vlplot(
        mark = {:line, strokeWidth = thickness},
        x = {:x, scale = {domain = domainx, nice = false}, title = labelx, sort = nothing},
        y = {:y, scale = {domain = domainy, nice = false}, title = labely, sort = nothing},
        color = "band:n",
        selection = {grid = {type = :interval, bind = :scales}})
    return p
end

# We optimize 1D band plots by collecting connected simplices into line segments (because :rule is considerabley slower than :line)
function bandtable(b::Bandstructure{1,T}, (scalingx, scalingy), bandsiter) where {T}
    bandsiter´ = bandsiter === missing ? eachindex(bands(b)) : bandsiter
    NT = typeof((;x = zero(T), y = zero(T), band = 1, tooltip = 1))
    table = NT[]
    for (nb, band) in enumerate(bands(b))
        verts = vertices(band)
        sinds = band.simpinds
        isempty(sinds) && continue
        s0 = (0, 0)
        for s in sinds
            if first(s) == last(s0) || s0 == (0, 0)
                push!(table, (; x = verts[first(s)][1] * scalingx, y = verts[first(s)][2] * scalingy, band = nb, tooltip = degeneracy(band, first(s))))
            else
                push!(table, (; x = verts[last(s0)][1] * scalingx, y = verts[last(s0)][2] * scalingy, band = nb, tooltip = degeneracy(band, last(s0))))
                push!(table, (; x = T(NaN), y = T(NaN), band = nb, tooltip = 0))  # cut
                push!(table, (; x = verts[first(s)][1] * scalingx, y = verts[first(s)][2] * scalingy, band = nb, tooltip = degeneracy(band, first(s))))
            end
            s0 = s
        end
        push!(table, (; x = verts[last(s0)][1] * scalingx, y = verts[last(s0)][2] * scalingy, band = nb, tooltip = degeneracy(band, last(s0))))
    end
    return table
end

function VegaLite.vlplot(h::Hamiltonian{LA}, psi = missing;
                         labels = ("x","y"), size = 800, axes::Tuple{Int,Int} = (1,2), xlims = missing, ylims = missing, digits = 4,
                         plotsites = true, plotlinks = true,
                         maxdiameter = 20, maxthickness = 0.25,
                         sitestroke = :white, sitesize = maxdiameter, siteopacity = 0.9, sitecolor = missing,
                         linksize = maxthickness, linkopacity = 1.0, linkcolor = sitecolor,
                         colorrange = missing,
                         colorscheme = "redyellowblue", discretecolorscheme = "category10") where {E,LA<:Lattice{E}}
    psi´ = unflatten_or_reinterpret_or_missing(psi, h)
    checkdims_psi(h, psi´)

    directives     = (; axes = axes, digits = digits,
                        sitesize_shader = site_shader(sitesize, psi´, h), siteopacity_shader = site_shader(siteopacity, psi´, h),
                        linksize_shader = link_shader(linksize, psi´, h), linkopacity_shader = link_shader(linkopacity, psi´, h),
                        sitecolor_shader = site_shader(sitecolor, psi´, h), linkcolor_shader = site_shader(linkcolor, psi´, h))

    table          = linkstable(h, directives)
    maxsiteopacity = maximum(s.opacity for s in table if !s.islink)
    maxlinkopacity = maximum(s.opacity for s in table if s.islink)
    maxsitesize    = maximum(s.scale for s in table if !s.islink)
    maxlinksize    = maximum(s.scale for s in table if s.islink)

    corners    = _corners(table)
    plotrange  = (xlims, ylims)
    (domainx, domainy), sizes = domain_size(corners, size, plotrange)
    if (plotsites && sitecolor !== missing) || (plotlinks && linkcolor !== missing)
        colorfield = :color
        colorscheme´ = colorscheme
        colorrange´ = sanitize_colorrange(colorrange, table)
    else
        colorfield = :sublat
        colorscheme´ = discretecolorscheme
        colorrange´ = nothing
    end

    p = vltheme(sizes)
    if plotlinks
        p += @vlplot(
            mark = {:rule},
            strokeWidth = {:scale,
                scale = {range = (0, maxthickness * maxdiameter), domain = (0, maxlinksize)},
                legend = needslegend(linksize)},
            color = {colorfield,
                scale = {domain = colorrange´, scheme = colorscheme´},
                legend = needslegend(linkcolor)},
            strokeOpacity = {:opacity,
                scale = {range = (0, 1), domain = (0, maxlinkopacity)},
                legend = needslegend(linkopacity)},
            transform = [{filter = "datum.islink"}],
            selection = {grid2 = {type = :interval, bind = :scales}},
            encoding = {
                x = {:x, scale = {domain = domainx, nice = false}, axis = {grid = false}, title = labels[1]},
                y = {:y, scale = {domain = domainy, nice = false}, axis = {grid = false}, title = labels[2]},
                x2 = {:x2, scale = {domain = domainx, nice = false}, axis = {grid = false}},
                y2 = {:y2, scale = {domain = domainy, nice = false}, axis = {grid = false}}})
    end
    if plotsites
        p += @vlplot(
            mark = {:circle, stroke = sitestroke},
            size = {:scale,
                scale = {range = (0, maxdiameter^2), domain = (0, maxsitesize), clamp = false},
                legend = needslegend(sitesize)},
            color = {colorfield,
                scale = {domain = colorrange´, scheme = colorscheme´},
                legend = needslegend(sitecolor)},
            opacity = {:opacity,
                scale = {range = (0, 1), domain = (0, maxsiteopacity)},
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

function vltheme((sizex, sizey), points = false)
    p = @vlplot(
        tooltip = :tooltip,
        width = sizex, height = sizey,
        config = {
            circle = {strokeWidth = 1, size = 200},
            line = {point = points},
            scale = {minOpacity = 0, maxOpacity = 1.0}})
    return p
end

sanitize_colorrange(::Missing, table) = extrema(s -> s.color, table)
sanitize_colorrange(r::AbstractVector, table) = convert(Vector, r)
sanitize_colorrange(r::Tuple, table) = [first(r), last(r)]

needslegend(x::Number) = nothing
needslegend(x) = true

unflatten_or_reinterpret_or_missing(psi::Missing, h) = missing
unflatten_or_reinterpret_or_missing(psi, h) = unflatten_or_reinterpret(psi, h)

checkdims_psi(h, psi) = size(h, 2) == size(psi, 1) || throw(ArgumentError("The eigenstate length $(size(psi,1)) must match the Hamiltonian dimension $(size(h, 2))"))
checkdims_psi(h, ::Missing) = nothing

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
                                scale = d.sitesize_shader(ridx), color = d.sitecolor_shader(ridx),
                                opacity = d.siteopacity_shader(ridx), islink = false))
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
                            scale =  d.sitesize_shader(row), color = d.sitecolor_shader(row),
                            opacity = 0.5 * d.siteopacity_shader(row), islink = false))
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
                        scale = d.linksize_shader(row, col), color = d.linkcolor_shader(col),
                        opacity = ifelse(iszero(har.dn), 1.0, 0.5) * d.linkopacity_shader(row, col), islink = true))
                end
            end
        end
    end
    return table
end

function _corners(table)
    min´ = max´ = (first(table).x, first(table).y)
    for row in table
        x, y = row.x, row.y
        (isnan(x) || isnan(y)) && continue
        min´ = min.(min´, (x, y))
        max´ = max.(max´, (x, y))
        if isdefined(row, :x2)
            x2, y2 = row.x2, row.y2
            min´ = min.(min´, (x2, y2))
            max´ = max.(max´, (x2, y2))
        end
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