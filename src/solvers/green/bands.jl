############################################################################################
# Taylor
#   A generalization of dual numbers to arbitrary order of differential ε, including negtive
#   When inverting (dividing), negative powers may be produced if leading terms are zero
#   Higher terms can be lost throughout operations.
#       Taylor{N}(f(x), f'(x), f''(x),..., f⁽ᴺ⁾(x)) = Series[f(x + ε), {ε, 0, N-1}]
#   If we need derivatives respect to x/α instead of x, we do rescale(::Taylor, α)
#region

struct Taylor{N,T}
    x::SVector{N,T}   # term coefficients
    pow::Int          # power of first term
end

Taylor(x::Tuple, pow = 0) = Taylor(SVector(x), pow)
Taylor(x...) = Taylor(SVector(x), 0)

Taylor{N}(x...) where {N} = Taylor{N}(SVector(x))
Taylor{N}(t::Tuple) where {N} = Taylor{N}(SVector(t))
Taylor{N}(d::Taylor) where {N} = Taylor{N}(d.x)
Taylor{N}(x::SVector{<:Any,T}, pow = 0) where {N,T} =
    Taylor(SVector(padtuple(x, zero(T), Val(N))), pow)

# rescale(t::Taylor{N}, α::Number) where {N} =
#     Taylor(ntuple(i -> α^(t.pow + i - 1) * t.x[i], Val(N)), t.pow)

function rescale(d::Taylor{N}, α::T) where {N,T<:Number}
    αp = cumprod((α ^ d.pow, ntuple(Returns(α), Val(N-1))...))
    d´ = Taylor(d.x .* αp, d.pow)
    return d´
end

chop(d::Taylor) = Taylor(chop.(d.x), d.pow)

function trim(d::Taylor{N}) where {N}
    nz = leading_zeros(d)
    iszero(nz) && return d
    pow = d.pow + nz
    t = ntuple(i -> d[i + pow - 1], Val(N))
    return Taylor(t, pow)
end

function leading_zeros(d::Taylor{N}) where {N}
    for i in 0:N-1
        iszero(d[d.pow + i]) || return i
    end
    return 0
end

Base.first(d::Taylor) = first(d.x)

Base.getindex(d::Taylor{N,T}, i::Integer) where {N,T} =
    d.pow <= i < d.pow + N ? d.x[i-d.pow+1] : zero(T)

Base.eltype(::Taylor{<:Any,T}) where {T} = T

Base.one(::Type{<:Taylor{N,T}}) where {N,T<:Number} = Taylor{N}(one(T))
Base.zero(::Type{<:Taylor{N,T}}) where {N,T} = Taylor(zero(SVector{N,T}), 0)
Base.iszero(d::Taylor) = iszero(d.x)

function Base.:+(d::Taylor{N}, d´::Taylor{N}) where {N}
    f, f´ = trim(d), trim(d´)
    pow = min(f.pow, f´.pow)
    t = ntuple(i -> f[pow + i - 1] + f´[pow + i - 1], Val(N))
    Taylor(t, pow)
end

Base.:-(d::Taylor, d´::Taylor) = d + (-d´)
Base.:-(d::Taylor) = Taylor(.-(d.x), d.pow)
Base.:*(d::Number, d´::Taylor) = Taylor(d * d´.x, d´.pow)
Base.:*(d´::Taylor, d::Number) = Taylor(d * d´.x, d´.pow)
Base.:/(d::Taylor{N}, d´::Taylor{N}) where {N} = d * inv(d´)
Base.:/(d::Taylor, d´::Number) = Taylor(d.x / d´, d.pow)

function Base.:*(d::Taylor{N}, d´::Taylor{N}) where {N}
    f, f´ = trim(d), trim(d´)
    pow = f.pow + f´.pow
    s = product_matrix(f.x) * f´.x
    return Taylor(s, pow)
end

function Base.inv(d::Taylor{N}) where {N}
    d´ = trim(d) # remove leading zeros
    iszero(d´) && argerror("Divide by zero")
    pow = d´.pow
    # impose d * inv(d) = 1. This is equivalent to Ud * inv(d).x = 1.x, where
    # Ud = hcat(d.x, shift(d.x, 1), ...)
    s = inv(product_matrix(d´.x))[:, 1]  # faster than \
    invd = Taylor(Tuple(s), -pow)
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
    ki::S2        # kᵢ[j]::SMatrix{D´,D,T,DD´} = coordinate j of momentum for vertex i
    eij::S3       # eᵢʲ::SMatrix{D´,D´,T,D´D´} = e_j - e_i
    kij::SVector{D,S3}  # kⱼ[β] - kᵢ[β] for β = 1...D
    Uij::SU       # Uᵢⱼ::SMatrix{D,D,T,DD} = k_ij − k_0j
    Uij⁻¹::SU     # inv(Uᵢⱼ)
    VD::T          # D!V = |det(U)|
end

function Simplex(ei::SVector{D´}, ki::SMatrix{D´,D}) where {D´,D}
    eij = chop.(ei' .- ei)
    kij = SVector(ntuple(β -> ki[:,β]' .- ki[:,β], Val(D)))
    Uij = ki[SVector{D}(2:D´),:] .- ki[1,:]'
    Uij⁻¹ = inv(Uij)
    VD = abs(det(Uij))
    return Simplex(ei, ki, eij, kij, Uij, Uij⁻¹, VD)
end

# β is the direction of the dr_β differential
function g0_simplex(ω, dn::SVector{D}, s::Simplex{D,T}, β = 1, ::Val{N} = Val(D+1)) where {D,T,N}
    ϕverts = s.ki * dn
    ϕedges = chop.(ϕverts' .- ϕverts)
    Δverts = ω .- s.ei
    eedges = s.eij
    kβverts = s.ki[:,β]
    kβedges = s.kij[β]
    g0sum = zero(Taylor{N,complex(T)})
    for j in 0:D
        ϕj = Taylor{N}(ϕverts[j+1], kβverts[j+1])
        ϕij = Taylor{N}.(ϕedges[:, j+1], kβedges[:, j+1])
        eij = eedges[:, j+1]
        Δj = Δverts[j+1]
        g0sum += g0_term(j, ϕj, ϕij, eij, Δj)
    end
    g0sum *= im^(D+1) * s.VD
    return chop(g0sum)
end

function g0_term(j, ϕ::Taylor{N,T}, ϕs, es::SVector{D´}, Δ) where {N,T,D´}
    D = D´ - 1
    Jⱼ = zero(Taylor{N,complex(T)})
    γⱼ = gamma_taylor(j, ϕs, es)
    for k in 0:D
        iszero(es[k+1]) && continue
        zkj = ϕs[k+1] / es[k+1]
        αₖⱼ = alpha_taylor(k, j, zkj, ϕs, es)
        Jₖⱼ = J_taylor(zkj, Δ)
        Jⱼ += αₖⱼ * Jₖⱼ
    end
    Jⱼ *= cis_taylor(ϕ)
    return γⱼ * Jⱼ
end

function gamma_taylor(j, ϕs::SVector{D´,T}, es) where {D´,T<:Taylor}
    D = D´ - 1
    γⱼ⁻¹ = one(T)
    for l in 0:D
        l != j && iszero(es[l+1]) && (γⱼ⁻¹ *= ϕs[l+1])
    end
    γⱼ = inv(γⱼ⁻¹)
    return γⱼ
end

function alpha_taylor(k, j, zkj::T, ϕs::SVector{D´}, es) where {D´,T<:Taylor}
    D = D´ - 1
    α⁻¹ = one(T)
    for l in 0:D
        if l != j && !iszero(es[l+1])
            α⁻¹ *= es[l+1]
            if l != k # ekj != 0 because it is constrained by caller
                zlj = ϕs[l+1]/es[l+1]
                α⁻¹ *= zkj - zlj
            end
        end
    end
    α = inv(α⁻¹)
    return α
end

function J_taylor(z::Taylor{N}, Δ) where {N}
    @assert iszero(z.pow)
    return rescale(J_taylor(z[0], Δ, Val(N)), z[1] * Δ)
end


# Taylor of J(zΔ) = cis(zΔ) * [Ci(|zΔ|) - i Si(zΔ)]
function J_taylor(z::Real, Δ::Real, ::Val{N}) where {N}
    zΔ = z * Δ
    if iszero(zΔ)  # need higher orders in case the cancellations are deep
        J₀ = complex(MathConstants.γ) + im * ifelse(Δ >= 0, 0, π) + im * ifelse(zΔ <= 0, π, 0)
        Jᵢ = ntuple(n -> (-im)^n/(n*factorial(n)), Val(N-1))
        J = Taylor{N}(J₀, Jᵢ...)
        E = cis_taylor(zΔ, Val(N))
        d = J * E
    else # no cancellations, only first order needed?
        ciszΔ =  cis(zΔ)
        J₀, J₁ = cosint(abs(zΔ)) - im*sinint(zΔ) + im * ifelse(Δ > 0, 0, π), conj(ciszΔ)/zΔ
        E₀, E₁ = ciszΔ, im * ciszΔ
        J = Taylor{2}(J₀, J₁)
        E = Taylor{2}(E₀, E₁)
        d = Taylor{N}(J * E)
    end
    return d
end

function cis_taylor(z::Taylor{N}) where {N}
    @assert iszero(z.pow)
    return rescale(cis_taylor(z[0], Val(N)), z[1])
end

# Taylor of cis(ϕ)
function cis_taylor(ϕ::Real, ::Val{N}) where {N}
    E₀, Eᵢ = complex(1.0), ntuple(n -> im^n/(factorial(n)), Val(N-1))
    E = cis(ϕ) * Taylor{N}(E₀, Eᵢ...)
    return E
end

#endregion
