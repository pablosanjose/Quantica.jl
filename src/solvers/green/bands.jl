############################################################################################
# Taylor
#   A generalization of dual numbers to arbitrary order of differential ε.
#   When inverting (dividing), all negative powers of ε are dropped (they are assumed to be
#   cancelled before the end). First term is always power zero (that can be zero).
#   Higher terms can be lost throughout operations.
#       Taylor{N}(f(x), f'(x), f''(x),..., f⁽ᴺ⁾(x)) = Series[f(x + ε), {ε, 0, N-1}]
#   If we need derivatives respect to x/α instead of x, we do rescale(::Taylor, α)
#region

struct Taylor{N,T}
    x::SVector{N,T}
end

Taylor(x::Tuple) = Taylor(SVector(x))
Taylor(x...) = Taylor(SVector(x))

Taylor{N}(x...) where {N} = Taylor{N}(SVector(x))
Taylor{N}(t::Tuple) where {N} = Taylor{N}(SVector(t))
Taylor{N}(d::Taylor) where {N} = Taylor{N}(d.x)
Taylor{N}(x::SVector{<:Any,T}) where {N,T} = Taylor{N,T}(SVector(padtuple(x, zero(T), Val(N))))

rescale(t::Taylor{N}, α) where {N} = Taylor(ntuple(i -> α^(i-1), Val(N)) .* t.x)

chop(d::Taylor) = Taylor(chop.(d.x))

leading(d::Taylor) = d[0]

Base.zero(::Type{<:Taylor{N,T}}) where {N,T} = Taylor(zero(SVector{N,T}))
Base.one(::Type{<:Taylor{N,T}}) where {N,T} = Taylor{N}(one(T))

Base.getindex(d::Taylor{N,T}, i::Integer) where {N,T} = 0 <= i < N ? d.x[i+1] : zero(T)

Base.eltype(::Taylor{<:Any,T}) where {T} = T

Base.:+(d::Taylor, d´::Taylor) = Taylor(d.x .+ d´.x)
Base.:-(d::Taylor, d´::Taylor) = Taylor(d.x .- d´.x)
Base.:-(d::Taylor) = Taylor(.-(d.x))
Base.:*(d::Number, d´::Taylor) = Taylor(d * d´.x)
Base.:*(d´::Taylor, d::Number) = Taylor(d * d´.x)
Base.:/(d::Taylor{N}, d´::Taylor{N}) where {N} = d * inv(d´)
Base.:/(d::Taylor, d´::Number) = Taylor(d.x / d´)

function Base.:*(d::Taylor{N,T1}, d´::Taylor{N,T2}) where {N,T1,T2}
    T = promote_type(T1, T2)
    t = ntuple(Val(N)) do j
        # This is simpler but slower: sum(i -> d[i] * d´[j - 1 - i], 0:j-1; init = zero(T))
        x = zero(T)
        @simd for i in 0:j-1
            x += d[i] * d´[j - 1 - i]
        end
        return x
    end
    return Taylor(t)
end

function leading_zeros(d::Taylor{N}) where {N}
    for i in 0:N-1
        iszero(d[i]) || return i
    end
    return 0
end

function Base.inv(d::Taylor{N,T}) where {N,T}
    nz = leading_zeros(d)
    nz == N && argerror("Divide by zero")
    T´ = float(T)
    s = zero(MVector{N,T´})
    # impose (1/ε^nz)*inv(d) * d = 1 and solve recursively,
    # factoring out leading order xmin * ε^nz
    xmin = d[nz]
    s[1] = inv(xmin)
    for j in 1:N-1
        x = zero(T)
        for i in 0:j-1
            x += s[i + 1] * d[nz + j - i]
        end
        s[j + 1] = - x / xmin
    end
    # shift away all inverse powers of ε, padding with zeros
    t = iszero(nz) ? Tuple(s) : ntuple(Val(N)) do i
        i´ = i + nz
        i´ <= N ? s[i´] : zero(T´)
    end
    d´ = Taylor(t)
    return d´
end

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
    V::T          # V = |det(U)|
end

function Simplex(ei::SVector{D´}, ki::SMatrix{D´,D}) where {D´,D}
    eij = chop.(ei' .- ei)
    kij = SVector(ntuple(β -> ki[:,β]' .- ki[:,β], Val(D)))
    Uij = ki[SVector{D}(2:D´),:] .- ki[1,:]'
    Uij⁻¹ = inv(Uij)
    V = abs(det(Uij))
    return Simplex(ei, ki, eij, kij, Uij, Uij⁻¹, V)
end

# β is the direction of the dr_β differential
function g0_simplex(ω, dn::SVector{D}, s::Simplex{D,T}, β = 1, ::Val{N} = Val(D)) where {D,T,N}
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
        add_g0_term!(g0sum, j, ϕj, ϕij, eij, Δj)
    end
    g0sum *= (-im)^D * factorial(D) * s.V
    # return g0sum[0], g0sum[1]
    return g0sum
end

function add_g0_term!(g0sum, j, ϕ::Taylor{N,T}, ϕs, es::SVector{D´}, Δ) where {N,T,D´}
    D = D´ - 1
    Jⱼ = zero(Taylor{N,complex(T)})
    γⱼ = gamma_taylor(j, ϕs, es)
    for k in 0:D
        iszero(es[k+1]) && continue
        zkj = ϕs[k+1] / es[k+1]
        αₖⱼ = alpha_taylor(k, j, zkj, ϕs, es)
        Jₖⱼ = rescale(J_taylor(leading(zkj) * Δ, Val(N)), Δ / es[k+1])
        Jⱼ += αₖⱼ * Jₖⱼ
    end
    Jⱼ *= cis_taylor(leading(ϕ), Val(N))
    g0sum += γⱼ * Jⱼ
    return g0sum
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
                α⁻¹ *= chop(zkj - zlj)
            end
        end
    end
    α = inv(α⁻¹)
    return α
end

# Taylor of J(zΔ) = cis(zΔ) * [Ci(|zΔ|) - i Si(zΔ)]
function J_taylor(zΔ::Real, ::Val{N}) where {N}
    if iszero(zΔ)  # need higher orders in case the cancellations are deep
        J₀, Jᵢ = complex(MathConstants.γ), ntuple(n -> (-im)^n/(n*factorial(n)), Val(N-1))
        J = Taylor{N}(J₀, Jᵢ...)
        E = cis_taylor(zΔ, Val(N))
        d = J * E
    else # no cancellations, only first order needed?
        ciszΔ =  cis(zΔ)
        J₀, J₁ = cosint(abs(zΔ)) - im * sinint(zΔ), conj(ciszΔ)/zΔ
        E₀, E₁ = ciszΔ, im * ciszΔ
        J = Taylor{2}(J₀, J₁)
        E = Taylor{2}(E₀, E₁)
        d = Taylor{N}(J * E)
    end
    return d
end

# Taylor of cis(ϕ)
function cis_taylor(ϕ::Real, ::Val{N}) where {N}
    E₀, Eᵢ = complex(1.0), ntuple(n -> im^n/(factorial(n)), Val(N-1))
    E = cis(ϕ) * Taylor{N}(E₀, Eᵢ...)
    return E
end

#endregion
