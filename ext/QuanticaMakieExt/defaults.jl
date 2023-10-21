############################################################################################
# qplot defaults
#region

## qplot defaults

function qplotdefaults(; figure::NamedTuple = (;), axis::NamedTuple = (;))
    global default_figure_kwarg = figure
    global default_axis_kwarg = axis
    return (; default_figure_kwarg, default_axis_kwarg)
end

default_figure_kwarg = (;)
default_axis_kwarg = (;)

## plotlattice defaults

const plotlat_default_figure = (; resolution = (1200, 1200), fontsize = 40)

const plotlat_default_axis3D = (;
    xlabel = "x", ylabel = "y", zlabel = "z",
    xticklabelcolor = :gray, yticklabelcolor = :gray, zticklabelcolor = :gray,
    xspinewidth = 0.2, yspinewidth = 0.2, zspinewidth = 0.2,
    xlabelrotation = 0, ylabelrotation = 0, zlabelrotation = 0,
    xticklabelsize = 30, yticklabelsize = 30, zticklabelsize = 30,
    xlabelsize = 40, ylabelsize = 40, zlabelsize = 40,
    xlabelfont = :italic, ylabelfont = :italic, zlabelfont = :italic,
    perspectiveness = 0.0, aspect = :data)

const plotlat_default_axis2D = (; autolimitaspect = 1)

const plotlat_default_lscene = (;)

const plotlat_default_2D =
    (plotlat_default_figure, plotlat_default_axis2D)
const plotlat_default_3D =
    (plotlat_default_figure, plotlat_default_axis3D, plotlat_default_lscene)

## plotbands defaults

const plotbands_default_figure = (; resolution = (1200, 1200), fontsize = 40)

const plotbands_default_axis3D = (;
    xticklabelcolor = :gray, yticklabelcolor = :gray, zticklabelcolor = :gray,
    xspinewidth = 0.2, yspinewidth = 0.2, zspinewidth = 0.2,
    xlabelrotation = 0, ylabelrotation = 0, zlabelrotation = 0,
    xticklabelsize = 30, yticklabelsize = 30, zticklabelsize = 30,
    xlabelsize = 40, ylabelsize = 40, zlabelsize = 40,
    xlabelfont = :italic, ylabelfont = :italic, zlabelfont = :italic,
    perspectiveness = 0.0, aspect = :data)

const plotbands_default_axis2D = (; autolimitaspect = nothing)

const plotbands_default_lscene = (;)

const plotbands_default_2D =
    (plotbands_default_figure, plotbands_default_axis2D)
const plotbands_default_3D =
    (plotbands_default_figure, plotbands_default_axis3D, plotbands_default_lscene)

#endregion
