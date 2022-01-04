using GeometryBasics
using GLMakie: to_value, RGBAf0, Vec3f0, FRect, @recipe, LineSegments, Theme,
    lift, campixel, SceneSpace, Node, Axis, text!, on, mouse_selection, poly!, scale!,
    translate!, linesegments!, mesh!, scatter!, meshscatter!
import GLMakie: plot!, plot
using SparseArrays: AbstractSparseMatrixCSC
using Quantica: Mesh

#######################################################################
# Tools
#######################################################################

function matrixidx(h::AbstractSparseMatrixCSC, row, col)
    for ptr in nzrange(h, col)
        rowvals(h)[ptr] == row && return ptr
    end
    return 0
end

matrixidx(h::DenseMatrix, row, col) = LinearIndices(h)[row, col]

transparent(rgba::RGBAf0, v = 0.5) = RGBAf0(rgba.r, rgba.g, rgba.b, rgba.alpha * v)

function darken(rgba::RGBAf0, v = 0.66)
    r = max(0, min(rgba.r * (1 - v), 1))
    g = max(0, min(rgba.g * (1 - v), 1))
    b = max(0, min(rgba.b * (1 - v), 1))
    RGBAf0(r,g,b,rgba.alpha)
end

function lighten(rgba, v = 0.66)
    darken(rgba, -v)
end

#######################################################################
# plot(::Mesh)
#######################################################################

function plot(m::Quantica.Mesh{Quantica.BandVertex{<:Any,1}}; kw...)
    return bandplot2d(m; kw...)
end

function plot(m::Quantica.Mesh{Quantica.BandVertex{<:Any,2}}; kw...)
    return bandplot3d(m; kw...)
end

@recipe(BandPlot2D, mesh) do scene
    Theme(
    linethickness = 3.0,
    wireframe = true,
    colors = map(t -> RGBAf0((0.8 .* t)...),
        ((0.973, 0.565, 0.576), (0.682, 0.838, 0.922), (0.742, 0.91, 0.734),
         (0.879, 0.744, 0.894), (1.0, 0.84, 0.0), (1.0, 1.0, 0.669),
         (0.898, 0.762, 0.629), (0.992, 0.843, 0.93), (0.88, 0.88, 0.88)))
    )
end

function plot!(plot::BandPlot2D)
    mesh = to_value(plot[1])
    # bands = haskey(plot, :bands) ? to_value(plot[:bands]) : eachindex(bs.bands)
    colors = Iterators.cycle(plot[:colors][])
    color = first(colors)
    # for (nb, color) in zip(bands, colors)
        # band = bs.bands[nb]
        vertices = Quantica.vertex_coordinates(mesh)
        simplices = Quantica.simplices(mesh)
        linesegments!(plot, (t -> vertices[first(t)] => vertices[last(t)]).(simplices),
                      linewidth = plot[:linethickness][], color = color)
    # end
    return plot
 end

@recipe(BandPlot3D, mesh) do scene
    Theme(
    linethickness = 1.0,
    wireframe = true,
    linedarken = 0.5,
    ssao = true, ambient = Vec3f0(0.55), diffuse = Vec3f0(0.4),
    colors = map(t -> RGBAf0(t...),
        ((0.973, 0.565, 0.576), (0.682, 0.838, 0.922), (0.742, 0.91, 0.734),
         (0.879, 0.744, 0.894), (1.0, 0.84, 0.0), (1.0, 1.0, 0.669),
         (0.898, 0.762, 0.629), (0.992, 0.843, 0.93), (0.88, 0.88, 0.88)))
    )
end

function plot!(plot::BandPlot3D)
    mesh = to_value(plot[1])
    # bandinds = haskey(plot, :bands) ? to_value(plot[:bands]) : eachindex(bs.bands)
    colors = Iterators.cycle(plot[:colors][])
    color = first(colors)
    # for (nb, color) in zip(bandinds, colors)
        # band = bs.bands[nb]
        vertices = collect(Quantica.vertex_coordinates(mesh))
        simplices = Quantica.simplices(mesh)
        connectivity = [s[j] for s in simplices, j in 1:3]
        if isempty(connectivity)
            scatter!(plot, vertices, color = color)
        else
            mesh!(plot, vertices, connectivity; color = color, transparency = false,
                ssao = plot[:ssao][], ambient = plot[:ambient][], diffuse = plot[:diffuse][])
            if plot[:wireframe][]
                edgevertices = collect(Quantica.edge_coordinates(mesh))
                linesegments!(plot, edgevertices, color = darken(color, plot[:linedarken][]), linewidth = plot[:linethickness][])
            end
        end
    # end
    return plot
 end