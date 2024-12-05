############################################################################################
# Integrator paths
#region

abstract type AbstractIntegrationPath end

struct PolygonPath{P} <: AbstractIntegrationPath
    pts::P              # vertices of integration path
end

struct SawtoothPath{T<:Real} <: AbstractIntegrationPath
    realpts::Vector{T}
    slope::T
    imshift::Bool
end

# straight from 0 to im*∞, but using a real variable from 0 to inf as parameter
# this is because QuadGK doesn't understand complex infinities
struct RadialPath{T<:AbstractFloat} <: AbstractIntegrationPath
    rate::T
    angle::T
    cisinf::Complex{T}
end

#region ## Constructors ##

SawtoothPath(pts::Vector{T}, slope, imshift) where {T<:Real} =
    SawtoothPath(pts, T(slope), imshift)

RadialPath(rate, angle) = RadialPath(promote(float(rate), float(angle))...)

function RadialPath(rate::T, angle::T) where {T<:AbstractFloat}
    0 <= angle < π/2 || argerror("The radial angle should be in the interval [0, π/2), got $angle")
    RadialPath(rate, angle, cis(angle))
end

#endregion

#region ## API ##

points(p::PolygonPath, args...; _...) = p.pts
points(p::PolygonPath{<:Function}, mu, kBT; params...) = p.pts(mu, kBT; params...)

points(p::RadialPath{T}, mu, kBT; _...) where {T} = iszero(kBT) ?
    [mu - T(Inf)*conj(p.cisinf), maybe_imshift(mu)] :
    [mu - T(Inf)*conj(p.cisinf), maybe_imshift(mu), mu + T(Inf)*p.cisinf]

function points(p::SawtoothPath, mu, kBT; _...)
    pts = complex(p.realpts)
    if (maximum(real, pts) > real(mu) > minimum(real, pts) && !any(≈(real(mu)), pts))
        # insert µ if it is within extrema and not already included
        sort!(push!(pts, mu), by = real)
        iszero(kBT) && cut_tail!(x -> real(x) > real(mu), pts)
    end
    p.imshift && imshift!(pts)
    triangular_sawtooth!(pts, p.slope)
    return pts
end

maybe_imshift(x::T) where {T<:Real} = float(x) + sqrt(eps(float(T)))*im
maybe_imshift(x::Complex) = float(x)

imshift!(v::AbstractArray{Complex{T}}) where {T} = (v .+= sqrt(eps(T))*im; v)

# Must be a type-stable tuple, and avoiding 0 in RadialPath
realpoints(p::RadialPath{T}, pts) where {T} = (-T(Inf), T(0), ifelse(length(pts)==2, T(0), T(Inf)))
realpoints(::Union{SawtoothPath,PolygonPath}, pts) = 1:length(pts)

# note: pts[2] == µ
point(x::Real, p::RadialPath, pts) = pts[2] + p.rate*x*ifelse(x <= 0, conj(p.cisinf), p.cisinf)

function point(x::Real, p::Union{SawtoothPath,PolygonPath}, pts)
    x0 = floor(Int, x)
    p0, p1 = pts[x0], pts[ceil(Int, x)]
    return p0 + (x-x0)*(p1 - p0)
end

# the minus sign inverts the RadialPath, so it is transformed to run from Inf to 0
jacobian(x::Real, p::RadialPath, pts) =  p.rate*ifelse(x <= 0, conj(p.cisinf), p.cisinf)
jacobian(x::Real, p::Union{SawtoothPath,PolygonPath}, pts) = pts[ceil(Int, x)] - pts[floor(Int, x)]

function triangular_sawtooth!(pts, slope)
    for i in 2:length(pts)
        push!(pts, 0.5 * (pts[i-1] + pts[i]) + im * slope * 0.5 * (pts[i] - pts[i-1]))
    end
    sort!(pts, by = real)
    return pts
end

#endregion

#endregion
#endregion

############################################################################################
# Paths module
#region

module Paths

using Quantica: Quantica, PolygonPath, SawtoothPath, RadialPath, argerror

## API

sawtooth(ω1::Real, ω2::Real, ωs::Real...; kw...) = sawtooth(float.((ω1, ω2, ωs...)); kw...)
sawtooth(ωmax::Real; kw...) = sawtooth!([-float(ωmax), float(ωmax)]; kw...)
sawtooth(ωpoints::Tuple; kw...) = sawtooth!(collect(float.(ωpoints)); kw...)
sawtooth(ωpoints::AbstractVector; kw...) = sawtooth!(float.(ωpoints); kw...)

sawtooth!(pts::AbstractVector{<:AbstractFloat}; slope = 1, imshift = true) =
    SawtoothPath(sort!(pts), slope, imshift)

sawtooth!(pts; kw...) = argerror("Path.sawtooth expects a collection of real points, got $pts")

radial(rate, angle) = RadialPath(rate, angle)

polygon(ωmax::Real) = polygon((-ωmax, ωmax))
polygon(ω1::Number, ω2::Number, ωs::Number...) = PolygonPath((ω1, ω2, ωs...))
polygon(ωs) = PolygonPath(ωs)

end # Module

#endregion

############################################################################################
# Integrator - integrates a function f along a complex path ωcomplex(ω::Real), connecting ωi
#   The path is piecewise linear in the form of a triangular sawtooth with a given ± slope
#region

struct Integrator{I,T,P,O<:NamedTuple,C,F}
    integrand::I    # call!(integrand, ω::Complex; params...)::Union{Number,Array{Number}}
    pts::P          # points in the integration interval
    result::T       # can be missing (for scalar integrand) or a mutable type (nonscalar)
    quadgk_opts::O  # kwargs for quadgk
    callback::C     # callback to call at each integration step (callback(ω, i(ω)))
    post::F         # function to apply to the result of the integration
end

#region ## Constructor ##

function Integrator(result, f, path; post = identity, callback = Returns(nothing), quadgk_opts...)
    quadgk_opts´ = NamedTuple(quadgk_opts)
    return Integrator(f, path, result, quadgk_opts´, callback, post)
end

Integrator(f, path; kw...) = Integrator(missing, f, path; kw...)

#endregion

#region ## API ##

integrand(I::Integrator) = I.integrand

points(I::Integrator) = I.pts

options(I::Integrator) = I.quadgk_opts

## call! ##
# scalar version
function call!(I::Integrator{<:Any,Missing}; params...)
    fx = x -> begin
        y = call!(I.integrand, x; params...)  # should be a scalar
        I.callback(x, y)
        return y
    end
    result, err = quadgk(fx, points(I)...; I.quadgk_opts...)
    result´ = I.post(result)
    return result´
end

# nonscalar version
function call!(I::Integrator{<:Any,T}; params...) where {T}
    fx! = (y, x) -> begin
        y .= serialize(call!(I.integrand, x; params...))
        I.callback(x, y)
        return nothing
    end
    result, err = quadgk!(fx!, serialize(I.result), points(I)...; I.quadgk_opts...)
    # note: post-processing is not element-wise & can be in-place
    result´ = I.post(unsafe_deserialize(I.result, result))
    return result´
end

(I::Integrator)(; params...) = copy(call!(I; params...))

#endregion
#endregion
