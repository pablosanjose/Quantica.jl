############################################################################################
# RegionPresets
#region

module RegionPresets

using StaticArrays
using Quantica: sanitize_SVector

struct Region{E,F} <: Function
    f::F
end

Region{E}(f::F) where {E,F<:Function} = Region{E,F}(f)

(region::Region{E})(r::SVector{E}) where {E} = region.f(r)

(region::Region{E})(r) where {E} = region.f(sanitize_SVector(SVector{E,Float64}, r))
(region::Region{E})(r::Number...) where {E} = region(r)

Base.show(io::IO, ::Region{E}) where {E} =
    print(io, "Region{$E} : region in $(E)D space")

Base.:&(r1::Region{E}, r2::Region{E}) where {E} = Region{E}(r -> r1.f(r) && r2.f(r))
Base.:|(r1::Region{E}, r2::Region{E}) where {E}  = Region{E}(r -> r1.f(r) || r2.f(r))
Base.xor(r1::Region{E}, r2::Region{E}) where {E}  = Region{E}(r -> xor(r1.f(r), r2.f(r)))
Base.:!(r1::Region{E}) where {E}  = Region{E}(r -> !r1.f(r))

extended_eps(T = Float64) = sqrt(eps(T))

segment(side = 10.0, c...) = Region{1}(_region_segment(side, c...))

circle(radius = 10.0, c...) = Region{2}(_region_ellipse((radius, radius), c...))

ellipse(radii = (10.0, 15.0), c...) = Region{2}(_region_ellipse(radii, c...))

square(side = 10.0, c...) = Region{2}(_region_rectangle((side, side), c...))

rectangle(sides = (10.0, 15.0), c...) = Region{2}(_region_rectangle(sides, c...))

sphere(radius = 10.0, c...) = Region{3}(_region_ellipsoid((radius, radius, radius), c...))

spheroid(radii = (10.0, 15.0, 20.0), c...) = Region{3}(_region_ellipsoid(radii, c...))

cube(side = 10.0, c...) = Region{3}(_region_cuboid((side, side, side), c...))

cuboid(sides = (10.0, 15.0, 20.0), c...) = Region{3}(_region_cuboid(sides, c...))

function _region_segment(l, c = 0)
    return r -> abs(2*(r[1]-c)) <= l * (1 + extended_eps())
end

function _region_ellipse((rx, ry), (cx, cy) = (0, 0))
    return r -> ((r[1]-cx)/rx)^2 + ((r[2]-cy)/ry)^2 <= 1 + extended_eps(Float64)
end

function _region_rectangle((lx, ly), (cx, cy) = (0, 0))
    return r -> abs(2*(r[1]-cx)) <= lx * (1 + extended_eps()) &&
                abs(2*(r[2]-cy)) <= ly * (1 + extended_eps())
end

function _region_ellipsoid((rx, ry, rz), (cx, cy, cz) = (0, 0, 0))
    return r -> ((r[1]-cx)/rx)^2 + ((r[2]-cy)/ry)^2 + ((r[3]-cz)/rz)^2 <= 1 + eps()
end

function _region_cuboid((lx, ly, lz), (cx, cy, cz) = (0, 0, 0))
    return r -> abs(2*(r[1]-cx)) <= lx * (1 + extended_eps()) &&
                abs(2*(r[2]-cy)) <= ly * (1 + extended_eps()) &&
                abs(2*(r[3]-cy)) <= lz * (1 + extended_eps())
end

end # module

const RP = RegionPresets

#endregion