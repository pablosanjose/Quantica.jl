############################################################################################
# Paths module
#region

abstract type AbstractIntegrationPath end

module Paths

using Quantica: AbstractIntegrationPath
using IntegrationInterface

## Path.sawtooth

struct SawtoothPath{T<:AbstractFloat} <: AbstractIntegrationPath
    realpts::Vector{T}
    complexpts::Vector{Complex{T}}
    slope::T
    imshift::Bool
end

SawtoothPath(realpts::Vector{T}, slope, imshift) where {T<:AbstractFloat} =
    SawtoothPath(realpts, complex.(realpts), T(slope), imshift)

SawtoothPath(pts, args...) = argerror("Path.sawtooth expects a collection of real points, got $pts")

function (p::SawtoothPath)(mu, kBT; kw...)
    pts = resize!(p.complexpts, length(p.realpts))
    pts .= complex.(p.realpts)
    if (maximum(real, pts) > real(mu) > minimum(real, pts) && !any(≈(real(mu)), pts))
        # insert µ if it is within extrema and not already included
        sort!(push!(pts, mu), by = real)
        iszero(kBT) && cut_tail!(x -> real(x) > real(mu), pts)
    end
    imshift && imshift!(pts)
    triangular_sawtooth!(pts, p.slope)
    return Domain.Box(pts)
end

sawtooth(ω1::Real, ω2::Real, ωs::Real...; kw...) = sawtooth(float.((ω1, ω2, ωs...)); kw...)
sawtooth(ωmax::Real; kw...) = sawtooth!([-float(ωmax), float(ωmax)]; kw...)
sawtooth(ωpoints::Tuple; kw...) = sawtooth!(collect(float.(ωpoints)); kw...)
sawtooth(ωpoints::AbstractVector; kw...) = sawtooth!(float.(ωpoints); kw...)

sawtooth!(pts::AbstractVector; slope = 1, imshift = true) =
    SawtoothPath(sort!(pts), slope, imshift)

function triangular_sawtooth!(pts, slope)
    for i in 2:length(pts)
        push!(pts, 0.5 * (pts[i-1] + pts[i]) + im * slope * 0.5 * (pts[i] - pts[i-1]))
    end
    sort!(pts, by = real)
    return pts
end

## Path.radial

struct RadialPath <: AbstractIntegrationPath
    rate::Real
    angle::Real
end

function (p::RadialPath)(mu, kBT; kw...)
    cispoint = p.rate*cis(p.angle)
    p1 = Infinity(mu-conj(cispoint))
    p2 = maybe_imshift(mu)
    p3 = ifelse(iszero(kBT), p2, Infinity(mu+cispoint))
    return Domain.Box(p1, p2, p3)
end

radial(rate, angle) = RadialPath(rate, angle)

## Path.polygon

struct PolygonPath{P} <: AbstractIntegrationPath
    pts::P
end

polygon(ωmax::Real) = polygon((-ωmax, ωmax))
polygon(ω1::Number, ω2::Number, ωs::Number...) = polygon((ω1, ω2, ωs...))
polygon(ωs) = PolygonPath(ωs)

(p::PolygonPath)(mu, kBT; kw...) = Domain.interval(p.pts)

## tools

maybe_imshift(x::T) where {T<:Real} = float(x) + sqrt(eps(float(T)))*im
maybe_imshift(x::Complex) = float(x)

imshift!(v::AbstractArray{Complex{T}}) where {T} = (v .+= sqrt(eps(T))*im; v)

end # Module
