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

site_shader(shader, h, psi) = i -> applyshader(shader, h, psi, i)
link_shader(shader, h, psi) = (i, j, dn) -> applyshader(shader, h, psi, i, j, dn)

applyshader(shader, h, psi, i...) = applyshader_vector(shader, h, psi, i...)
applyshader(shader, h, psi::AbstractMatrix, i...) =
    sum(col -> applyshader_vector(shader, h, view(psi, :, col), i...), axes(psi, 2))

applyshader_vector(shader::Number, h, x...) = shader
applyshader_vector(shader::Missing, h, x...) = 0.0
applyshader_vector(shader::Function, h, psi, i) = shader(psi[i])
applyshader_vector(shader::Function, h, psi, i, j, dn) = shader(psi[i], psi[j])
applyshader_vector(shader::DensityShader, h, psi, i) = real(shader.transform(psi[i]' * shader.kernel * psi[i]))
applyshader_vector(shader::DensityShader, h, psi, i, j, dn) = real(shader.transform(psi[j]' * shader.kernel * psi[j]))

function applyshader_vector(shader::CurrentShader, h, psi::AbstractVector{T}, src) where {T}
    pos = allsitepositions(h.lattice)
    c = zero(real(eltype(T)))
    for har in h.harmonics
        rows = rowvals(har.h)
        for ptr in nzrange(har.h, src)
            dst = rows[ptr]
            c += applyshader_vector(shader, h, psi, dst, src, har.dn)
        end
    end
    return c
end

function applyshader_vector(shader::CurrentShader, h, psi, i, j, dn)
    pos = allsitepositions(h.lattice)
    # Assuming that only links from base unit cell are plotted
    dr = pos[i] - pos[j] + bravais(h) * dn
    dc = imag(psi[i]' * shader.kernel * h[dn][i, j] * psi[j])
    c = iszero(shader.axis) ? shader.transform(norm(dr * dc)) : shader.transform(dr[shader.axis] * dc)
    return real(c)
end

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
"unflattened" to restore the orbital structure of `h`. The vector `psi` must have length
equal to the number of sites (after unflattening if required).

If `h` is defined on an unbounded `L`-dimensional lattice, and `psi` is of the form

    psi = [cell₁ => psi₁, cell₂ => psi₂,...]

where `cellᵢ::NTuple{L,Int}` are indices of unit cells, said unit cells will all be plotted
together with their corresponding `psiᵢ`. Otherwise, a single cell at the origin will be
plotted.

    vlplot(h::Hamiltonian, psi::AbstractMatrix; kw...)

Same as above, but columns of psi will be summed over after applying shaders to each.

    vlplot(h::Hamiltonian, psi::Subspace; kw...)

Equivalent to `vlplot(h, psi.basis; kw...)`.

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
    - `onlyonecell = false`: whether to omit plotting links to neighboring unit cells in unbounded lattices
    - `sitestroke = :white`: the color of site outlines. If `nothing`, no outline will be plotted.
    - `colorscheme = "redyellowblue"`: Color scheme from `https://vega.github.io/vega/docs/schemes/`
    - `discretecolorscheme = "category10"`: Color scheme from `https://vega.github.io/vega/docs/schemes/`
    - `maxdiameter = 15`: maximum diameter of sites, useful when `sitesize` is not a constant.
    - `mindiameter = 0`: site diameters below this value will be excluded (may be used with `plotlinks = false` to increase performance)
    - `maxthickness = 0.25`: maximum thickness of links, as a fraction of `maxdiameter`, useful when `linksize` is not a constant.
    - `sitesize = maxdiameter`: diameter of sites.
    - `siteopacity = 0.9`: opacity of sites.
    - `sitecolor = missing`: color of sites. If `missing`, colors will encode the site sublattice following `discretecolorscheme`
    - `linksize = maxthickness`: thickness of hopping links.
    - `linkopacity = 1.0`: opacity of hopping links.
    - `linkcolor = missing`: color of links. If `missing`, colors will encode source site sublattice, following `discretecolorscheme`
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
function bandtable(b::Bandstructure{1,C,T}, (scalingx, scalingy), bandsiter) where {C,T}
    bandsiter´ = bandsiter === missing ? eachindex(bands(b)) : bandsiter
    NT = typeof((;x = zero(T), y = zero(T), band = 1, tooltip = 1))
    table = NT[]
    for nb in bandsiter´
        band = bands(b, nb)
        verts = vertices(band)
        simps = band.simps
        isempty(simps) && continue
        s0 = (0, 0)
        for s in simps
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
                         onlyonecell = false, plotsites = true, plotlinks = true,
                         maxdiameter = 20, mindiameter = 0, maxthickness = 0.25,
                         sitestroke = :white, sitesize = maxdiameter, siteopacity = 0.9, sitecolor = missing,
                         linksize = maxthickness, linkopacity = 1.0, linkcolor = missing,
                         colorrange = missing,
                         colorscheme = "redyellowblue", discretecolorscheme = "category10") where {E,LA<:Lattice{E}}

    table = linkstable(h, psi; axes, digits, onlyonecell, plotsites, plotlinks,
                               sitesize, siteopacity, sitecolor, linksize, linkopacity, linkcolor)
    maxsitesize    = plotsites ? maximum(s.scale for s in table if !s.islink)   : 0.0
    maxlinksize    = plotlinks ? maximum(s.scale for s in table if s.islink)    : 0.0

    mindiameter > 0 && filter!(s -> s.islink || s.scale >= mindiameter*maxsitesize/maxdiameter, table)

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
                scale = {range = (0, 1), domain = (0, 1)},
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
                scale = {range = (0, 1), domain = (0, 1)},
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

unflatten_orbitals_or_reinterpret_or_missing(psi::Missing, h) = missing
unflatten_orbitals_or_reinterpret_or_missing(psi::AbstractArray, h) = unflatten_orbitals_or_reinterpret(psi, orbitalstructure(h))
unflatten_orbitals_or_reinterpret_or_missing(s::Subspace, h) = unflatten_orbitals_or_reinterpret(s.basis, orbitalstructure(h))

checkdims_psi(h, psi) = size(h, 2) == size(psi, 1) ||
    throw(ArgumentError("The eigenstate length $(size(psi,1)) must match the Hamiltonian dimension $(size(h, 2))"))
checkdims_psi(h, ::Missing) = nothing

function linkstable(h, psi; kw...)
    T = numbertype(h.lattice)
    table = NamedTuple{(:x, :y, :x2, :y2, :sublat, :tooltip, :scale, :color, :opacity, :islink),
                       Tuple{T,T,T,T,NameType,String,Float64,Float64,Float64,Bool}}[]
    linkstable!(table, h, psi; kw...)
    return table
end

function linkstable!(table, h, cellpsi::AbstractVector{<:Pair}; kw...)
    foreach(cellpsi) do (cell, psi)
        linkstable!(table, h, psi, cell; kw..., onlyonecell = true)
    end
    return table
end

function linkstable!(table, h, psi, cell = missing;
                     axes, digits, onlyonecell, plotsites, plotlinks,
                     sitesize, siteopacity, sitecolor, linksize, linkopacity, linkcolor) where {LA,L}
    psi´ = unflatten_orbitals_or_reinterpret_or_missing(psi, h)
    checkdims_psi(h, psi´)
    d = (; axes = axes, digits = digits,
        sitesize_shader    = site_shader(sitesize, h, psi´),
        siteopacity_shader = site_shader(siteopacity, h, psi´),
        sitecolor_shader   = site_shader(sitecolor, h, psi´),
        linksize_shader    = link_shader(linksize, h, psi´),
        linkopacity_shader = link_shader(linkopacity, h, psi´),
        linkcolor_shader   = link_shader(linkcolor, h, psi´))
    (a1, a2) = d.axes
    lat = h.lattice
    br = bravais(h)
    T = numbertype(lat)
    r0 = cell === missing ? br * zero(SVector{latdim(h),T}) : br * toSVector(T,cell)
    slats = sublats(lat)
    rs = allsitepositions(lat)
    h0 = h.harmonics[1].h
    rows = Int[] # list of plotted destination sites for dn != 0

    for har in h.harmonics
        !iszero(har.dn) && onlyonecell && continue
        resize!(rows, 0)
        ridx = 0
        for ssrc in slats
            if iszero(har.dn)
                for (i, r) in enumeratesites(lat, ssrc)
                    x = get(r + r0, a1, zero(T)); y = get(r + r0, a2, zero(T))
                    ridx += 1
                    plotsites && push!(table, (x = x, y = y, x2 = x, y2 = y,
                                sublat = sublatname(lat, ssrc), tooltip = matrixstring_inline(i, h0[i, i], d.digits),
                                scale = d.sitesize_shader(ridx), color = d.sitecolor_shader(ridx),
                                opacity = d.siteopacity_shader(ridx), islink = false))
                end
            end
            for sdst in slats
                itr = nonzero_indices(har, siterange(lat, sdst), siterange(lat, ssrc))
                for (row, col) in itr
                    iszero(har.dn) && col == row && continue
                    rsrc = rs[col] + r0
                    rdst = (rs[row] + bravais(lat) * har.dn + r0)
                    # sites in neighboring cells connected to dn=0 cell. But add only once.
                    if !iszero(har.dn) && !in(row, rows)
                        x = get(rdst, a1, zero(T)); y = get(rdst, a2, zero(T))
                        plotsites && push!(table,
                            (x = x, y = y, x2 = x, y2 = y,
                            sublat = sublatname(lat, sdst), tooltip = matrixstring_inline(row, h0[row, row], d.digits),
                            scale =  d.sitesize_shader(row), color = d.sitecolor_shader(row),
                            opacity = 0.5 * d.siteopacity_shader(row), islink = false))
                        push!(rows, row)
                    end
                    plotlinks || continue
                    # draw half-links but only intracell
                    rdst = ifelse(cell !== missing || iszero(har.dn), (rdst + rsrc) / 2, rdst)
                    x  = get(rsrc, a1, zero(T)); y  = get(rsrc, a2, zero(T))
                    x´ = get(rdst, a1, zero(T)); y´ = get(rdst, a2, zero(T))
                    # Exclude links perpendicular to the screen
                    rdst ≈ rsrc || push!(table,
                        (x = x, y = y, x2 = x´, y2 = y´,
                        sublat = sublatname(lat, ssrc), tooltip = matrixstring_inline(row, col, har.h[row, col], d.digits),
                        scale = d.linksize_shader(row, col, har.dn), color = d.linkcolor_shader(row, col, har.dn),
                        opacity = ifelse(iszero(har.dn), 1.0, 0.5) * d.linkopacity_shader(row, col, har.dn), islink = true))
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