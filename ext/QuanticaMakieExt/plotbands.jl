############################################################################################
# plotlattice recipe
#region

@recipe(PlotBands) do scene
    Theme(
        ssao = true,
        fxaa = true,
        ambient = Vec3f(0.7),
        diffuse = Vec3f(0.5),
        backlight = 0.8f0,
        colormap = :Spectral_9,
        color = missing,
        opacity = 1.0,
        size = 3,
        maxsize = 6,
        nodedarken = 0.6,
        hide = ()
    )
end

#endregion

############################################################################################
# qplot
#region

function Quantica.qplot(b::Bands{<:Any,2}; axis = (;), figure = (;), inspector = false, plotkw...)
    fig, ax = empty_fig_axis_2D(plotbands_default_2D...; axis, figure)
    plotbands!(ax, b; plotkw...)
    inspector && DataInspector()
    return fig
end

function Quantica.qplot(b::Bands{<:Any,3}; fancyaxis = false, axis = (;), figure = (;), inspector = false, plotkw...)
    fig, ax = empty_fig_axis_3D(plotbands_default_3D...; fancyaxis, axis, figure)
    plotbands!(ax, b; plotkw...)
    inspector && DataInspector()
    return fig
end

const plotbands_default_figure = (; resolution = (1200, 1200), fontsize = 40)

const plotbands_default_axis3D = (;
    xlabel = "k₁", ylabel = "k₂", zlabel = "ϵ",
    xticklabelcolor = :gray, yticklabelcolor = :gray, zticklabelcolor = :gray,
    xspinewidth = 0.2, yspinewidth = 0.2, zspinewidth = 0.2,
    xlabelrotation = 0, ylabelrotation = 0, zlabelrotation = 0,
    xticklabelsize = 30, yticklabelsize = 30, zticklabelsize = 30,
    xlabelsize = 40, ylabelsize = 40, zlabelsize = 40,
    xlabelfont = :italic, ylabelfont = :italic, zlabelfont = :italic,
    perspectiveness = 0.0, aspect = :data)

const plotbands_default_axis2D = (; xlabel = "k", ylabel = "ϵ", autolimitaspect = nothing)

const plotbands_default_lscene = (;)

const plotbands_default_2D =
    (plotbands_default_figure, plotbands_default_axis2D)
const plotbands_default_3D =
    (plotbands_default_figure, plotbands_default_axis3D, plotbands_default_lscene)

#endregion

############################################################################################
# PlotBands for 2D Bands
#region

function Makie.plot!(plot::PlotBands{Tuple{B}}) where {B<:Bands{<:Any,2}}
    bands = to_value(plot[1])
    bp = bandprimitives(bands, plot)
    verts = bp.verts[bp.simps]
    color = bp.colors[bp.simps]
    linewidth = bp.sizes[bp.simps]
    linesegments!(plot, verts; color, linewidth, inspectable = false)
    if !ishidden((:nodes), plot)
        markersize = bp.sizes
        color´ = darken.(bp.colors, plot[:nodedarken][])
        scatter!(plot, bp.verts; color = color´, markersize)
    end
    return plot
end

function Makie.plot!(plot::PlotBands{Tuple{B}}) where {B<:Bands{<:Any,3}}
    bands = to_value(plot[1])
    bp = bandprimitives(bands, plot)
    verts = bp.verts
    color = bp.colors
    simps = simplices_matrix(bp)
    transparency = has_transparencies(plot[:opacity][])
    opts = (;
        ssao = plot[:ssao][],
        ambient = plot[:ambient][],
        diffuse = plot[:diffuse][],
        backlight = plot[:backlight][],
        fxaa = plot[:fxaa][])
    mesh!(plot, verts, simps; color, inspectable = false, transparency, opts...)
    if !ishidden((:nodes), plot)
        markersize = bp.sizes
        color´ = darken.(color, plot[:nodedarken][])
        scatter!(plot, verts; color = color´, markersize, transparency, opts...)
    end
    return plot
end

#endregion

############################################################################################
# PlotBands Primitives
#region

struct BandPrimitives{E}
    verts::Vector{Point{E,Float32}}
    simps::Vector{Int}                  # flattened neighbor matrix
    hues::Vector{Float32}
    opacities::Vector{Float32}
    sizes::Vector{Float32}
    colors::Vector{RGBAf}
end

#region ## Constructors ##

BandPrimitives{E}() where {E} =
    BandPrimitives(Point{E,Float32}[], Int[], Float32[], Float32[], Float32[], RGBAf[])

function bandprimitives(bands, plot)   # function barrier
    opts = (plot[:color][], plot[:opacity][], plot[:size][])
    bp = _bandprimitives(bands, opts)
    update_colors!(bp, plot)
    update_sizes!(bp, plot)
    return bp
end

function _bandprimitives(bands::Quantica.Bands{<:Any,E}, (hue, opacity, size)) where {E}
    bp = BandPrimitives{E}()
    for (s, subband) in enumerate(Quantica.subbands(bands))
        offset = length(bp.verts)
        append!(bp.simps, offset .+ reinterpret(Int, Quantica.simplices(subband)))
        for vert in Quantica.vertices(subband)
            ψ = Quantica.states(vert)
            kϵ = Quantica.coordinates(vert)
            k = Quantica.base_coordinates(vert)
            ϵ = Quantica.energy(vert)
            push!(bp.verts, kϵ)
            push_subbandhue!(bp, hue, ψ, ϵ, k, s)
            push_subbandopacity!(bp, opacity, ψ, ϵ, k, s)
            push_subbandsize!(bp, size, ψ, ϵ, k, s)
        end
    end
    return bp
end

#endregion

#region ## API ##

simplices_matrix(s::Quantica.Subband) = simplices_matrix(Quantica.mesh(s))

simplices_matrix(m::Quantica.Mesh) =
    reshape(reinterpret(Int, Quantica.simplices(m)), 3, length(Quantica.simplices(m)))'

simplices_matrix(bp::BandPrimitives{E}) where {E} = reshape(bp.simps, E, length(bp.simps) ÷ E)'

## push! ##

push_subbandhue!(bp, ::Missing, ψ, ϵ, k, s) = push!(bp.hues, s)
push_subbandhue!(bp, hue::Real, ψ, ϵ, k, s) = push!(bp.hues, hue)
push_subbandhue!(bp, hue::Function, ψ, ϵ, k, s) = push!(bp.hues, hue(ψ, ϵ, k))
push_subbandhue!(bp, hue, ψ, ϵ, k, s) = argerror("Unrecognized color")

push_subbandopacity!(bp, opacity::Real, ψ, ϵ, k, s) = push!(bp.opacities, opacity)
push_subbandopacity!(bp, opacity::Function, ψ, ϵ, k, s) = push!(bp.opacities, opacity(ψ, ϵ, k))
push_subbandopacity!(bp, opacity, ψ, ϵ, k, s) = argerror("Unrecognized opacity")

push_subbandsize!(bp, size::Real, ψ, ϵ, k, s) = push!(bp.sizes, size)
push_subbandsize!(bp, size::Function, ψ, ϵ, k, s) = push!(bp.sizes, size(ψ, ϵ, k))
push_subbandsize!(bp, size, ψ, ϵ, k, s) = argerror("Unrecognized size")

## update_color! ##

function update_colors!(p::BandPrimitives, plot)
    extremahues = safeextrema(p.hues)
    extremaops = safeextrema(p.opacities)
    color = plot[:color][]
    opacity = plot[:opacity][]
    colormap = Makie.ColorSchemes.colorschemes[plot[:colormap][]]
    # reuse update_colors! from plotlattice.jl
    return update_colors!(p, extremahues, extremaops, color, opacity, colormap, 0.0)
end

## update_sizes! ##

function update_sizes!(p::BandPrimitives, plot)
    extremasizes = safeextrema(p.sizes)
    size = plot[:size][]
    maxsize = plot[:maxsize][]
    return update_sizes!(p, extremasizes, size, maxsize) # function barrier
end

# almost identical to update_radii! from plotlattice.jl
function update_sizes!(p, extremasizes, size, maxsize)
    for (i, r) in enumerate(p.sizes)
        p.sizes[i] = primitive_size(normalize_range(r, extremasizes), size, maxsize)
    end
    return p
end

primitive_size(normr, size::Number, maxsize) = size
primitive_size(normr, size, maxsize) = maxsize * normr

#endregion
#endregion

############################################################################################
# convert_argments
#region

function Makie.convert_arguments(T::Type{<:Makie.Mesh}, s::Quantica.Subband{<:Any,3})
    verts = [Point3f(Quantica.coordinates(v)) for v in Quantica.vertices(s)]
    simps = simplices_matrix(s)
    return convert_arguments(T, verts, simps)
end

function Makie.convert_arguments(::Type{<:LineSegments}, s::Quantica.Subband{<:Any,2})
    verts = Quantica.vertices(s)
    simps = Quantica.simplices(s)
    segments = [Point2f(Quantica.coordinates(verts[i])) for s in simps for i in s]
    return (segments,)
end

#endregion