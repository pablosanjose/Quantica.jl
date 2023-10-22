############################################################################################
# qplot defaults
#region

## qplot defaults

function qplotdefaults(;
    figure::Union{Missing,NamedTuple} = missing,
    axis2D::Union{Missing,NamedTuple} = missing,
    axis3D::Union{Missing,NamedTuple} = missing,
    lscene::Union{Missing,NamedTuple} = missing,
    inspector::Union{Missing,NamedTuple} = missing)
    ismissing(figure) || (global user_default_figure = figure)
    ismissing(axis2D) || (global user_default_axis2D = axis2D)
    ismissing(axis3D) || (global user_default_axis3D = axis3D)
    ismissing(lscene) || (global user_default_lscene = lscene)
    ismissing(inspector) || (global user_default_inspector = inspector)
    return (; user_default_figure, user_default_axis2D, user_default_axis3D, user_default_lscene, user_default_inspector)
end

qplotdefaults(defaults::NamedTuple) = qplotdefaults(; defaults...)

user_default_figure = (;)
user_default_axis2D = (;)
user_default_axis3D = (;)
user_default_lscene = (;)
user_default_inspector = (;)

# Choose used_default_axis for each given type of plotted object
axis_defaults(::PlotArgumentType{3}, fancyaxis) = axis_defaults(fancyaxis)
axis_defaults(::PlotArgumentType, fancyaxis) = user_default_axis2D
axis_defaults(fancyaxis::Bool) = ifelse(fancyaxis, user_default_lscene, user_default_axis3D)

empty_fig_axis(b::PlotArgumentType; kw...) = (checkplotdim(b); _empty_fig_axis(b; kw...))

_empty_fig_axis(::PlotLatticeArgumentType{3}; kw...) = empty_fig_axis_3D(plotlat_default_3D...; kw...)
_empty_fig_axis(b::PlotLatticeArgumentType; kw...) = empty_fig_axis_2D(plotlat_default_2D...; kw...)
_empty_fig_axis(::PlotBandsArgumentType{3}; kw...) = empty_fig_axis_3D(plotbands_default_3D...; kw...)
_empty_fig_axis(b::PlotBandsArgumentType; kw...) = empty_fig_axis_2D(plotbands_default_2D...; kw...)

function empty_fig_axis_2D(default_figure, default_axis2D; axis = user_default_axis2D, figure = user_default_figure, kw...)
    axis´ = merge(user_default_axis2D, axis)
    fig = Figure(; default_figure..., figure...)
    ax = Axis(fig[1,1]; default_axis2D..., axis´...)
    tight_ticklabel_spacing!(ax)  # Workaround for Makie issue #3009
    return fig, ax
end

function empty_fig_axis_3D(default_figure, default_axis3D, default_lscene;
    fancyaxis = true,
    axis = axis_defaults(fancyaxis),
    figure = user_default_figure, kw...)
    fig = Figure(; default_figure..., figure...)
    axis´ = merge(axis_defaults(fancyaxis), axis)
    ax = fancyaxis ?
        LScene(fig[1,1]; default_lscene..., axis´...) :
        Axis3(fig[1,1]; default_axis3D..., axis´...)
    return fig, ax
end

checkplotdim(::PlotArgumentType{E}) where {E} =
    E > 3 && argerror("Cannot represent a mesh in an $E-dimensional embedding space")

## plotlattice defaults

const plotlat_default_figure = (; resolution = (1200, 1200), fontsize = 40)

const plotlat_default_axis3D = (;
    xlabel = "x", ylabel = "y", zlabel = "z",
    xticklabelcolor = :gray, yticklabelcolor = :gray, zticklabelcolor = :gray,
    xspinewidth = 0.2, yspinewidth = 0.2, zspinewidth = 0.2,
    xlabelrotation = 0, ylabelrotation = 0, zlabelrotation = 0,
    xticklabelsize = 30, yticklabelsize = 30, zticklabelsize = 30,
    xlabelsize = 35, ylabelsize = 35, zlabelsize = 35,
    xlabelfont = :italic, ylabelfont = :italic, zlabelfont = :italic,
    perspectiveness = 0.0, aspect = :data)

const plotlat_default_axis2D = (; xlabel = "x", ylabel = "y", autolimitaspect = 1)

const plotlat_default_lscene = (;)

const plotlat_default_2D =
    (plotlat_default_figure, plotlat_default_axis2D)
const plotlat_default_3D =
    (plotlat_default_figure, plotlat_default_axis3D, plotlat_default_lscene)

## plotbands defaults

const plotbands_default_figure = (; resolution = (1200, 1200), fontsize = 40)

const plotbands_default_axis3D = (;
    xlabel = "ϕ₁", ylabel = "ϕ₂", zlabel = "ε",
    xticklabelcolor = :gray, yticklabelcolor = :gray, zticklabelcolor = :gray,
    xspinewidth = 0.2, yspinewidth = 0.2, zspinewidth = 0.2,
    xlabelrotation = 0, ylabelrotation = 0, zlabelrotation = 0,
    xticklabelsize = 30, yticklabelsize = 30, zticklabelsize = 30,
    xlabelsize = 35, ylabelsize = 35, zlabelsize = 35,
    xlabelfont = :italic, ylabelfont = :italic, zlabelfont = :italic,
    perspectiveness = 0.4, aspect = :data)

const plotbands_default_axis2D = (; xlabel = "ε", ylabel = "ϕ", autolimitaspect = nothing)

const plotbands_default_lscene = (;)

const plotbands_default_2D =
    (plotbands_default_figure, plotbands_default_axis2D)
const plotbands_default_3D =
    (plotbands_default_figure, plotbands_default_axis3D, plotbands_default_lscene)

## inspector defaults

const default_inspector = (; fontsize = 20)

#endregion
