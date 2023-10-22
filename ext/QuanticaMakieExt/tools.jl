############################################################################################
# tools
#region

function darken(rgba::RGBAf, v = 0.66)
    r = max(0, min(rgba.r * (1 - v), 1))
    g = max(0, min(rgba.g * (1 - v), 1))
    b = max(0, min(rgba.b * (1 - v), 1))
    RGBAf(r,g,b,rgba.alpha)
end

darken(colors::Vector, v = 0.66) = darken.(colors, Ref(v))

transparent(rgba::RGBAf, v = 0.5) = RGBAf(rgba.r, rgba.g, rgba.b, rgba.alpha * v)

maybedim(color, dn, dimming) = iszero(dn) ? color : transparent(color, 1 - dimming)

dnshell(::Lattice{<:Any,<:Any,L}, span = -1:1) where {L} =
    sort!(vec(SVector.(Iterators.product(ntuple(_ -> span, Val(L))...))), by = norm)

ishidden(s, plot::Union{PlotLattice,PlotBands}) = ishidden(s, plot[:hide][])
ishidden(s, ::Nothing) = false
ishidden(s::Symbol, hide::Symbol) = s === hide
ishidden(s::Symbol, hides::Tuple) = s in hides
ishidden(ss, hides) = any(s -> ishidden(s, hides), ss)

normalize_range(c::T, (min, max)) where {T} = min ≈ max ? T(c) : T((c - min)/(max - min))

jointextrema(v, v´) = min(minimum(v; init = 0f0), minimum(v´; init = 0f0)), max(maximum(v; init = 0f0), maximum(v´; init = 0f0))

safeextrema(v::Missing) = (Float32(0), Float32(1))
safeextrema(v) = isempty(v) ? (Float32(0), Float32(1)) : extrema(v)

has_transparencies(x::Real) = !(x ≈ 1)
has_transparencies(::Missing) = false
has_transparencies(x) = true

#endregion
