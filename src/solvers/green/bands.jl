############################################################################################
# Series
#   A generalization of dual numbers to arbitrary powers of differential ε, also negative
#   When inverting (dividing), negative powers may be produced if leading terms are zero
#   Higher terms can be lost throughout operations.
#       Series{N}(f(x), f'(x)/1!, f''(x)/2!,..., f⁽ᴺ⁾(x)/N!) = Series[f(x + ε), {ε, 0, N-1}]
#   If we need derivatives respect to x/α instead of x, we do rescale(::Series, α)
#region

struct Series{N,T}
    x::SVector{N,T}   # term coefficients
    pow::Int          # power of first term
end

Series(x::Tuple, pow = 0) = Series(SVector(x), pow)
Series(x...) = Series(SVector(x), 0)

Series{N}(x...) where {N} = Series{N}(SVector(x))
Series{N}(t::Tuple) where {N} = Series{N}(SVector(t))
Series{N}(d::Series) where {N} = Series{N}(d.x)
Series{N}(x::SVector{<:Any,T}, pow = 0) where {N,T} =
    Series(SVector(padtuple(x, zero(T), Val(N))), pow)

function rescale(d::Series{N}, α::Number) where {N}
    αp = cumprod((α ^ d.pow, ntuple(Returns(α), Val(N-1))...))
    d´ = Series(Tuple(d.x) .* αp, d.pow)
    return d´
end

chop(d::Series) = Series(chop.(d.x), d.pow)

function trim(d::Series{N}) where {N}
    nz = leading_zeros(d)
    iszero(nz) && return d
    pow = d.pow + nz
    t = ntuple(i -> d[i + pow - 1], Val(N))
    return Series(t, pow)
end

function leading_zeros(d::Series{N}) where {N}
    @inbounds for i in 0:N-1
        iszero(d[d.pow + i]) || return i
    end
    return 0
end

Base.first(d::Series) = first(d.x)

Base.getindex(d::Series{N,T}, i::Integer) where {N,T} =
    d.pow <= i < d.pow + N ? (@inbounds d.x[i-d.pow+1]) : zero(T)

Base.eltype(::Series{<:Any,T}) where {T} = T

Base.one(::Type{<:Series{N,T}}) where {N,T<:Number} = Series{N}(one(T))
Base.zero(::Type{<:Series{N,T}}) where {N,T} = Series(zero(SVector{N,T}), 0)
Base.iszero(d::Series) = iszero(d.x)
Base.transpose(d::Series) = d

function Base.:+(d::Series{N}, d´::Series{N}) where {N}
    f, f´ = trim(d), trim(d´)
    pow = min(f.pow, f´.pow)
    t = ntuple(i -> f[pow + i - 1] + f´[pow + i - 1], Val(N))
    Series(t, pow)
end

Base.:+(d::Series) = d
Base.:-(d::Series, d´::Series) = d + (-d´)
Base.:-(d::Series) = Series(.-(d.x), d.pow)
Base.:*(d::Number, d´::Series) = Series(d * d´.x, d´.pow)
Base.:*(d´::Series, d::Number) = Series(d * d´.x, d´.pow)
Base.:/(d::Series{N}, d´::Series{N}) where {N} = d * inv(d´)
Base.:/(d::Series, d´::Number) = Series(d.x / d´, d.pow)

function Base.:*(d::Series{N}, d´::Series{N}) where {N}
    iszero(d´) && return d´
    f, f´ = trim(d), trim(d´)
    pow = f.pow + f´.pow
    s = product_matrix(f.x) * f´.x
    return Series(s, pow)
end

function Base.inv(d::Series{N}) where {N}
    d´ = trim(d) # remove leading zeros
    iszero(d´) && argerror("Divide by zero")
    pow = d´.pow
    # impose d * inv(d) = 1. This is equivalent to Ud * inv(d).x = 1.x, where
    # Ud = hcat(d.x, shift(d.x, 1), ...)
    s = inv(product_matrix(d´.x))[:, 1]  # faster than \
    invd = Series(Tuple(s), -pow)
    return invd
end

# Ud = [x1 0 0 0; x2 x1 0 0; x3 x2 x1 0; x4 x3 x2 x1]
# product of two series d*d´ is Ud * d´.x
function product_matrix(s::SVector{N,T}) where {N,T}
    t = ntuple(Val(N)) do i
        shiftpad(s, i-1)
    end
    return hcat(t...)
end

# shift SVector to the right by i, padding on the left with zeros
shiftpad(s::SVector{N,T}, i) where {N,T} =
    SVector(ntuple(j -> j - i > 0 ? s[j - i] : zero(T), Val(N)))

#endregion

############################################################################################
# g0(ω) for a D-simplex, with D´ = D + 1
#region

struct Simplex{D,T,S1,S2,S3,SU<:SMatrix{D,D,T}}
    ei::S1        # eᵢ::SVector{D´,T} = energy of vertex i
    kij::S2       # kᵢ[j]::SMatrix{D´,D,T,DD´} = coordinate j of momentum for vertex i
    eij::S3       # ϵᵢʲ::SMatrix{D´,D´,T,D´D´} = e_j - e_i
    U⁻¹::SU       # inv(Uᵢⱼ) for Uᵢⱼ::SMatrix{D,D,T,DD} = kⱼ[i] - k₀[i] (edges as cols)
    U⁻¹Q⁻¹::SU    # Q cols are basis of shift vectors δrᵝ
    phi´::S2      # kij * Q
    w::SVector{D,T} # U⁻¹ * k₀
    VD::T         # D!V = |det(U)|
end

function Simplex(ei::SVector{D´}, kij::SMatrix{D´,D,T}) where {D´,D,T}
    eij = chop(ei' .- ei)
    k0 = kij[1, :]
    U = kij[SVector{D}(2:D´),:]' .- k0          # edges as columns
    U⁻¹ = inv(U)
    VD = abs(det(U))
    w = U⁻¹ * k0
    Δe = eij[1, SVector{D}(2:D´)]               # eⱼ - e₀
    v = chop(transpose(U⁻¹) * Δe)
    if iszero(v)
        Q = one(SMatrix{D,D,T})                 # special case
    else
        vext = hcat(v, zero(SMatrix{D,D-1,T}))  # pad with zero to get full Q from QR
        Q´, R´ = qr(vext)                       # full orthonormal basis with v as first vec
        Q = rotate45(Q´ * sign(R´[1,1]))        # rotate 1 & 2 by 45º -> none parallel to v
    end
    phi´ = kij * Q
    U⁻¹Q⁻¹ = U⁻¹ * Q'
    return Simplex(ei, kij, eij, U⁻¹, U⁻¹Q⁻¹, phi´, w, VD)
end

rotate45(s::SMatrix{D,D}) where {D} =
    hcat((s[:,1] + s[:,2])/√2, (s[:,1] - s[:,2])/√2, s[:,SVector{D-2}(3:D)])

function g_simplex(ω, dn, s::Simplex{D}) where {D}
    gβ = ntuple(Val(D)) do β
        ϕ´verts = s.phi´[:, β]
        g_simplex(Val(D+1), ω, dn, s, ϕ´verts)
    end
    return first(first(gβ)), last.(gβ)
end

function g_simplex(::Val{N}, ω::Number, dn::SVector{D}, s::Simplex{D,T}, ϕ´verts::SVector) where {D,N,T}
    # phases ϕverts[j+1] will be perturbed by ϕ´verts[j+1]*dϕ, for j in 0:D
    # Similartly, ϕedges[j+1,k+1] will be perturbed by ϕ´edges[j+1,k+1]*dϕ
    ϕ´edges = ϕ´verts' .- ϕ´verts
    ϕverts0 = s.kij * dn
    ϕverts = Series{N}.(ϕverts0, ϕ´verts)
    ϕedges = Series{N}.(chop.(ϕverts0' .- ϕverts0), ϕ´edges)
    Δverts = ω .- s.ei
    eedges = s.eij
    zedges = zkj_series.(ϕedges, eedges)
    eϕ = cis_series.(ϕverts)
    γα = γα_series(ϕedges, zedges, eedges)            # SMatrix{D´,D´}
    if iszero(eedges)                                 # special case, full energy degeneracy
        Δ0 = chop(first(Δverts))
        eγαJ = iszero(Δ0) ? zero(Series{N,complex(T)}) :  im * sum(γα[1,:] .* eϕ) / Δ0
    else
        J = J_series.(zedges, eedges, transpose(Δverts))  # SMatrix{D´,D´}
        eγαJ = sum(γα .* J .* transpose(eϕ))              # manual contraction is slower!
    end
    gsum = (-im)^(D+1) * s.VD * trim(chop(eγαJ))
    return gsum[0], gsum[1]
end

zkj_series(ϕ, e) = iszero(e) ? ϕ : ϕ/e

function cis_series(z::Series{N}) where {N}
    @assert iszero(z.pow)
    c = cis_series(z[0], Val(N))
    # Go from dz differential to dϕ
    return rescale(c, z[1])
end

# Series of cis(ϕ)
function cis_series(ϕ::Real, ::Val{N}) where {N}
    E₀, Eᵢ = complex(1.0), ntuple(n -> im^n/(factorial(n)), Val(N-1))
    E = cis(ϕ) * Series{N}(E₀, Eᵢ...)
    return E
end

@inline function γα_series(ϕedges::S, zedges::S, eedges::SMatrix{D´,D´}) where {N,T,D´,S<:SMatrix{D´,D´,Series{N,T}}}
    # js = ks = SVector{D´}(1:D´)
    # α⁻¹ = α⁻¹_series.(js', ks, Ref(zedges), Ref(eedges))
    # γ⁻¹ = γ⁻¹_series.(js', Ref(ϕedges), Ref(eedges))
    # γα = inv.(α⁻¹ .* γ⁻¹)
    # return γα
    ## BUG: broadcast over SArrays is currently allocations-buggy
    ## https://github.com/JuliaArrays/StaticArrays.jl/issues/1178
    js = ks = SVector{D´}(1:D´)
    jks = Tuple(tuple.(js', ks))
    α⁻¹ = SMatrix{D´,D´}(α⁻¹_series.(jks, Ref(zedges), Ref(eedges)))
    γ⁻¹ = SVector(γ⁻¹_series.(Tuple(js), Ref(ϕedges), Ref(eedges)))
    γα = inv.(α⁻¹ .* transpose(γ⁻¹))
    return γα
end

function α⁻¹_series((j, k), zedges::SMatrix{D´,D´,T}, eedges) where {D´,T<:Series}
    x = one(T)
    @inbounds j != k && !iszero(eedges[k, j]) || return x
    @inbounds for l in 1:D´
        if l != j && !iszero(eedges[l, j])
            x *= eedges[l, j]
            if l != k # ekj != 0, already constrained above
                x *= zedges[l, j] - zedges[k, j]
            end
        end
    end
    return x
end

function γ⁻¹_series(j, ϕedges::SMatrix{D´,D´,T}, eedges) where {D´,T<:Series}
    x = one(T)
    @inbounds for l in 1:D´
        if l != j && iszero(eedges[l, j])
            x *= ϕedges[l, j]
        end
    end
    return x
end

@inline function J_series(z::Series{N,T}, e, Δ) where {N,T}
    iszero(e) && return zero(Series{N,Complex{T}})
    J = J_series(z[0], Δ, Val(N))
    # Go from d(zΔ) = dz*Δ differential to dϕ
    return rescale(J, z[1] * Δ)
end

# Series of J(zΔ) = cis(zΔ) * [Ci(|z|Δ) - i Si(zΔ)] (variable zΔ for Series)
function J_series(z::T, Δ::T, ::Val{N}) where {N,T<:Number}
    C = complex(T)
    iszero(Δ) && return Series{N}(C(Inf))
    zΔ = z * Δ
    imπ = im * ifelse(Δ > 0, 0, π) # strangely enough, union splitting is faster than stable
    if iszero(zΔ)
        J₀ = log(abs(Δ)) + imπ #+ MathConstants.γ + log(|z|) # not needed, cancels out
        Jᵢ = ntuple(n -> (-im)^n/(n*factorial(n)), Val(N-1))
        J = Series{N}(J₀, Jᵢ...)
        E = cis_series(zΔ, Val(N))
        EJ = E * J
    else
        ciszΔ =  cis(zΔ)
        J₀, J₁ = cosint(abs(zΔ)) - im*sinint(zΔ) + imπ, conj(ciszΔ)/zΔ
        E₀, E₁ = ciszΔ, im * ciszΔ
        J´ = Series{2}(J₀, J₁)
        E´ = Series{2}(E₀, E₁)
        EJ = Series{N}(E´ * J´)
    end
    return EJ
end

#endregion
