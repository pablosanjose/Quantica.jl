using .VegaLite

"""
    vlplot(b::Bandstructure{1}; size = 640, points = false, labels = ("φ/2π", "ε"), scaling = (1/2π, 1))

Plots the 1D bandstructure `b` using VegaLite.

# Options:
    - `size`: the `(width, height)` of the plot (or `width == height` if a single number)
    - `points`: whether to plot a point at each sampled energy/momentum
    - `labels`: labels for the x and y axes
    - `scaling`: `(scalex, scaley)` scalings for the x (Bloch phase) and y (energy) variables
"""
function VegaLite.vlplot(b::Bandstructure; size = 640, points = false, labels = ("φ/2π", "ε"), scaling = (1/2π, 1))
    sizex, sizey = make_it_two(size)
    labelx, labely = labels
    scalingx, scalingy = make_it_two(scaling)
    table = bandtable(b, (scalingx, scalingy))
    p = table |> @vlplot(
        mark = {
            :line,
            point = points
        },
        selection = {
            grid = {type = :interval, bind = :scales}
        },
        width = sizex,
        height = sizey,
        x = {
            :phi,
            title = labelx
        },
        y = {
            :energy,
            title = labely
        },
        color = "band:n"
        )
end

make_it_two(x::Number) = (x, x)
make_it_two(x::Tuple{Number,Number}) = x

function bandtable(b::Bandstructure{1}, (scalingx, scalingy) = (1, 1))
    ks = vertices(b.kmesh)
    table = [(phi = v[1] * scalingx, energy = v[2] * scalingy, band = i)
             for (i, bnd) in enumerate(bands(b)) for v in vertices(bnd)]
    return table
end
