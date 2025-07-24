############################################################################################
# plotlattice recipe
#region

@recipe(PlotBands) do scene
    Theme(
        colormap = :Spectral_9,
        color = missing,
        opacity = 1.0,
        size = 1,
        minmaxsize = (0, 6),
        nodesizefactor = 4,
        nodedarken = 0.0,
        hide = ()   # :nodes, :bands, :wireframe
    )
end

#endregion

############################################################################################
# qplot
#   Supports plotting b::Bandstructure, but also b[...]::Vector{Subbands} and
#   slice(b, ...)::Vector{Mesh}
#region

function Quantica.qplot(b::PlotBandsArgumentType;
    fancyaxis = true, axis = axis_defaults(b, fancyaxis), figure = user_default_figure, inspector = true, plotkw...)
    fig, ax = empty_fig_axis(b; fancyaxis, axis, figure)
    plotbands!(ax, b; plotkw...)
    inspector && DataInspector(; default_inspector..., user_default_inspector...)
    return fig
end

function Quantica.qplot!(b::PlotBandsArgumentType; inspector = true, kw...)
    p = plotbands!(b; kw...)
    inspector && DataInspector(; default_inspector..., user_default_inspector...)
    return p
end

#endregion

############################################################################################
# PlotBands Primitives
#region

struct MeshPrimitives{E,S}
    verts::Vector{Point{E,Float32}}
    simps::Vector{Int}                  # flattened neighbor matrix
    hues::Vector{Float32}
    opacities::Vector{Float32}
    sizes::Vector{Float32}
    tooltips::Vector{String}
    colors::Vector{RGBAf}
end

#region ## Constructors ##

MeshPrimitives{E,S}() where {E,S} =
    MeshPrimitives{E,S}(Point{E,Float32}[], Int[], Float32[], Float32[], Float32[], String[], RGBAf[])

function meshprimitives(meshes, plot)   # function barrier
    opts = (plot[:color][], plot[:opacity][], plot[:size][])
    E, S = Quantica.dims(first(meshes))
    mp = MeshPrimitives{E,S}()
    for (s, mesh) in enumerate(meshes)
        bandprimitives!(mp, mesh, s, opts)
    end
    update_colors!(mp, plot)
    update_sizes!(mp, plot)
    return mp
end

function bandprimitives!(mp, mesh, s, (hue, opacity, size))
    offset = length(mp.verts)
    append!(mp.simps, offset .+ reinterpret(Int, Quantica.simplices(mesh)))
    for (ivert, vert) in enumerate(Quantica.vertices(mesh))
        ψ = Quantica.states(vert)
        kϵ = Quantica.coordinates(vert)
        k = Quantica.base_coordinates(vert)
        ϵ = Quantica.energy(vert)
        push!(mp.verts, kϵ)
        push_subbandhue!(mp, hue, ψ, ϵ, k, s)
        push_subbandopacity!(mp, opacity, ψ, ϵ, k, s)
        push_subbandsize!(mp, size, ψ, ϵ, k, s)
        push_subbandtooltip!(mp, ψ, ϵ, k, s, ivert)
    end
    return mp
end

#endregion

#region ## API ##

simplices_matrix(s::Quantica.Subband) = simplices_matrix(Quantica.mesh(s))

simplices_matrix(m::Quantica.Mesh) =
    reshape(reinterpret(Int, Quantica.simplices(m)), Quantica.embdim(m), length(Quantica.simplices(m)))'

simplices_matrix(mp::MeshPrimitives{<:Any,S}) where {S} = reshape(mp.simps, S, length(mp.simps) ÷ S)'

## push! ##

push_subbandhue!(mp, ::Missing, ψ, ϵ, k, s) = push!(mp.hues, s)
push_subbandhue!(mp, hue::Real, ψ, ϵ, k, s) = push!(mp.hues, hue)
push_subbandhue!(mp, hue::Function, ψ, ϵ, k, s) = push!(mp.hues, hue(ψ, ϵ, k))
push_subbandhue!(mp, ::Symbol, ψ, ϵ, k, s) = push!(mp.hues, 0f0)
push_subbandhue!(mp, ::Makie.Colorant, ψ, ϵ, k, s) = push!(mp.hues, 0f0)
push_subbandhue!(mp, hue, ψ, ϵ, k, s) = argerror("Unrecognized color")

push_subbandopacity!(mp, opacity::Real, ψ, ϵ, k, s) = push!(mp.opacities, opacity)
push_subbandopacity!(mp, opacity::Function, ψ, ϵ, k, s) = push!(mp.opacities, opacity(ψ, ϵ, k))
push_subbandopacity!(mp, opacity, ψ, ϵ, k, s) = argerror("Unrecognized opacity")

push_subbandsize!(mp, size::Real, ψ, ϵ, k, s) = push!(mp.sizes, size)
push_subbandsize!(mp, size::Function, ψ, ϵ, k, s) = push!(mp.sizes, size(ψ, ϵ, k))
push_subbandsize!(mp, size, ψ, ϵ, k, s) = argerror("Unrecognized size")

push_subbandtooltip!(mp, ψ, ϵ, k, s, iv) =
    push!(mp.tooltips, "Subband $s, vertex $iv:\n k = $k\n ϵ = $ϵ\n degeneracy = $(size(ψ, 2))")

## update_color! ##

function update_colors!(p::MeshPrimitives, plot)
    extremahues = safeextrema(p.hues)
    extremaops = safeextrema(p.opacities)
    color = plot[:color][]
    opacity = plot[:opacity][]
    colormap = get_colorscheme(plot[:colormap][])
    # reuse update_colors! from plotlattice.jl
    return update_colors!(p, extremahues, extremaops, color, opacity, colormap, 0.0)
end

## update_sizes! ##

function update_sizes!(p::MeshPrimitives, plot)
    extremasizes = safeextrema(p.sizes)
    size = plot[:size][]
    minmaxsize = plot[:minmaxsize][]
    return update_sizes!(p, extremasizes, size, minmaxsize) # function barrier
end

# almost identical to update_radii! from plotlattice.jl
function update_sizes!(p, extremasizes, size, minmaxsize)
    for (i, r) in enumerate(p.sizes)
        p.sizes[i] = primitive_size(normalize_range(r, extremasizes), size, minmaxsize)
    end
    return p
end

primitive_size(normr, size::Number, minmaxsize) = size
primitive_size(normr, size, (mins, maxs)) = mins + (maxs - mins) * normr

#endregion
#endregion

############################################################################################
# PlotBands for 1D and 2D Bandstructure (2D and 3D embedding space)
#region

function Makie.plot!(plot::PlotBands{Tuple{S}}) where {S<:PlotBandsArgumentType}
    meshes = Quantica.meshes(to_value(plot[1]))
    mp = meshprimitives(meshes, plot)
    return plotmeshes!(plot, mp)
end

plotmeshes!(plot, mp::MeshPrimitives{<:Any,E}) where {E} =
    Quantica.argerror("Can only plot meshes in 2D and 3D space, got an $(E)D mesh")

function plotmeshes!(plot, mp::MeshPrimitives{<:Any,2})
    if !ishidden((:bands, :subbands), plot)
        verts = mp.verts[mp.simps]
        color = mp.colors[mp.simps]
        linewidth = mp.sizes[mp.simps]
        linesegments!(plot, verts; color, linewidth, inspectable = false)
    end
    if !ishidden((:nodes, :points, :vertices), plot)
        inspector_label = (self, i, r) -> mp.tooltips[i]
        markersize = mp.sizes .* plot[:nodesizefactor][]
        color´ = darken.(mp.colors, plot[:nodedarken][])
        scatter!(plot, mp.verts; color = color´, markersize, inspector_label)
    end
    return plot
end

function plotmeshes!(plot, mp::MeshPrimitives{<:Any,3})
    transparency = has_transparencies(plot[:opacity][])
    if !ishidden((:bands, :subbands), plot)
        simps = simplices_matrix(mp)
        if !ishidden((:wireframe, :simplices), plot)
            color´ = darken.(mp.colors, plot[:nodedarken][])
            poly!(plot, mp.verts, simps; color = mp.colors, inspectable = false, transparency,
                 strokewidth = plot[:size][])
        else
            mesh!(plot, mp.verts, simps; color = mp.colors, inspectable = false, transparency)
        end
    end
    if !ishidden((:nodes, :points, :vertices), plot)
        inspector_label = (self, i, r) -> mp.tooltips[i]
        markersize = mp.sizes .* plot[:nodesizefactor][]
        color´ = darken.(mp.colors, plot[:nodedarken][])
        scatter!(plot, mp.verts; color = color´, markersize, transparency, inspector_label)
    end
    return plot
end

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
