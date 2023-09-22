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

chop(d::Series) = Series(chop(d.x), d.pow)

function trim(d::Series{N}) where {N}
    nz = leading_zeros(d)
    iszero(nz) && return d
    pow = d.pow + nz
    t = ntuple(i -> d[i + pow - 1], Val(N))
    return Series(t, pow)
end

trim(x) = x

function leading_zeros(d::Series{N}) where {N}
    @inbounds for i in 0:N-1
        iszero(d[d.pow + i]) || return i
    end
    return 0
end

function trim_and_map(func, d::Series{N}, d´::Series{N}) where {N}
    f, f´ = trim(d), trim(d´)
    pow = min(f.pow, f´.pow)
    t = ntuple(i -> func(f[pow + i - 1], f´[pow + i - 1]), Val(N))
    return Series(t, pow)
end

scalar(d::Series) = d[0]
scalar(d) = d

Base.first(d::Series) = first(d.x)

function Base.getindex(d::Series{N,T}, i::Integer) where {N,T}
    i´ = i - d.pow + 1
    checkbounds(Bool, d.x, i´) ? (@inbounds d.x[i´]) : zero(T)
end

Base.eltype(::Series{<:Any,T}) where {T} = T

Base.one(::Type{<:Series{N,T}}) where {N,T} = Series{N}(one(T))
Base.one(d::S) where {S<:Series} = one(S)
Base.zero(::Type{<:Series{N,T}}) where {N,T} = Series(zero(SVector{N,T}), 0)
Base.zero(d::S) where {S<:Series} = zero(S)
Base.iszero(d::Series) = iszero(d.x)
Base.transpose(d::Series) = d  # act as a scalar

Base.:-(d::Series) = Series(-d.x, d.pow)
Base.:+(d::Series, d´::Series) = trim_and_map(+, d, d´)
Base.:-(d::Series, d´::Series) = trim_and_map(-, d, d´)
Base.:+(d::Number, d´::Series{N}) where {N} = Series{N}(d) + d´
Base.:-(d::Number, d´::Series{N}) where {N} = Series{N}(d) - d´
Base.:*(d::Number, d´::Series) = Series(d * d´.x, d´.pow)
Base.:*(d´::Series, d::Number) = Series(d * d´.x, d´.pow)
Base.:/(d::Series{N}, d´::Series{N}) where {N} = d * inv(d´)
Base.:/(d::Series, d´::Number) = Series(d.x / d´, d.pow)
Base.:^(d::Series, n::Integer) = Base.power_by_squaring(d, n)

# necessary for the case n = 0 and n = 1 in power_by_squaring
Base.copy(d::Series) = d

function Base.:*(d::Series{N}, d´::Series{N}) where {N}
    x, x´ = promote(d.x, d´.x)
    dp = Series(x, d.pow)
    dp´ = Series(x´, d´.pow)
    return dp * dp´
end

function Base.:*(d::Series{N,T}, d´::Series{N,T}) where {N,T}
    iszero(d´) && return d´
    f, f´ = trim(d), trim(d´)
    pow = f.pow + f´.pow
    s = product_matrix(f.x) * f´.x
    return Series(s, pow)
end

function Base.inv(d::Series)
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
function product_matrix(s::SVector{N}) where {N}
    t = ntuple(Val(N)) do i
        shiftpad(s, i - 1)
    end
    return hcat(t...)
end

# shift SVector to the right by i, padding on the left with zeros
shiftpad(s::SVector{N,T}, i) where {N,T} =
    SVector(ntuple(j -> j - i > 0 ? s[j - i] : zero(T), Val(N)))

#endregion

############################################################################################
# BandSimplex: encodes energy and momenta of vertices, and derived quantitities
#region

struct Expansions{D,T,DD}
    cis::Tuple{Complex{T},Vararg{Complex{T},D}}
    J0::NTuple{D,Complex{T}}
    Jmat::SMatrix{D,D,Complex{T},DD}
    log::NTuple{D,T}
end

struct BandSimplex{D,T,S1<:SVector{<:Any,<:Real},S2<:SMatrix{<:Any,<:Any,<:Real},S3<:SMatrix{<:Any,<:Any,<:Real},R<:Ref{<:Expansions{D,T}}}     # D = manifold dimension
    ei::S1        # eᵢ::SVector{D´,T} = energy of vertex i
    kij::S2       # kᵢ[j]::SMatrix{D´,D,T,DD´} = coordinate j of momentum for vertex i
    eij::S3       # ϵᵢʲ::SMatrix{D´,D´,T,D´D´} = e_j - e_i
    dual::S1      # dual::SVector{D´,T}, first hyperdual coefficient
    VD::T         # D!V = |det(kᵢʲ - kᵢ⁰)|
    refex::R      # Ref to Series expansions for necessary functions
end

# Precomputes the Series expansion coefficients for cis, J(z->0) and J(z)
function Expansions(::Val{D}, ::Type{T}) where {D,T}  # here order = D
    C = complex(T)
    # series coefs of cis(t) around t = 0
    cis = ntuple(n -> C(im)^(n-1)/(factorial(n-1)), Val(D+1))
    # series coefs of Ci(t) - im * Si(t) around t = 0, starting at order 1
    J0 = ntuple(n -> C(-im)^n/(n*factorial(n)), Val(D))
    # series coefs of Ci(t) - im * Si(t) around t = t_0 is cis(t0) * Jmat * SA[1/t, 1/t², ...]
    Jmat = ntuple(Val(D*D)) do ij
        j, i = fldmod1(ij, D)
        j > i ? zero(C) : ifelse(isodd(i), 1, -1) * C(im)^(i-j) / (i*factorial(i-j))
    end |> SMatrix{D,D,C}
    # series coefs of ln(x) around x = Δ is ln(Δ) + log .* SA[1/Δ, 1/Δ², ...]
    log = ntuple(n -> -T(-1)^n/n, Val(D))
    return Expansions(cis, J0, Jmat, log)
end

BandSimplex(ei, kij, refex...) = BandSimplex(real(ei), real(kij), refex...)

function BandSimplex(ei::SVector{D´,T}, kij::SMatrix{D´,D,T}, refex = Ref(Expansions(Val(D), T))) where {D´,D,T}
    D == D´ - 1 ||
        argerror("The dimension $D of Bloch phases in simplex should be one less than the number of vertices $(D´)")
    eij = chop(ei' .- ei)
    k0 = kij[1, :]
    U = kij[SVector{D}(2:D´),:]' .- k0          # edges as columns
    VD = abs(det(U)) / (2π)^D
    dual = generate_dual(eij)
    return BandSimplex(ei, kij, eij, dual, VD, refex)
end

function generate_dual(eij::SMatrix{D´,D´,T}) where {D´,T}
    dual = rand(SVector{D´,T})
    iszero(eij) && return dual
    while !is_valid_dual(dual, eij)
        dual = rand(SVector{D´,T})
    end
    return dual
end

# check whether iszero(eʲₖφʲₗ - φʲₖeʲₗ) for nonzero e's
function is_valid_dual(phi, es)
    phis = phi' .- phi
    for j in axes(es, 2), k in axes(es, 1), l in axes(es, 1)
        l != k != j && l != k || continue
        eʲₖ = es[k,j]
        eʲₗ = es[l,j]
        (iszero(eʲₖ) || iszero(eʲₗ)) && continue
        eʲₖ * phis[l, j] ≈ phis[k, j] * eʲₗ && return false
    end
    return true
end

function g_integrals(s::BandSimplex, ω, dn, val...)
    g₀, gi = iszero(dn) ?
        g_integrals_local(s, ω, val...) :
        g_integrals_nonlocal(s, ω, dn, val...)
    return g₀, gi
end

#endregion

############################################################################################
# g_integrals_local: zero-dn g₀(ω) and gⱼ(ω) with normal or hyperdual numbers for φ
#region

function g_integrals_local(s::BandSimplex{D,T}, ω, ::Val{N} = Val(0)) where {D,T,N}
    eⱼ = s.ei
    eₖʲ = s.eij
    g₀, gⱼ = begin
        if N > 0 || is_degenerate(eₖʲ)
            eⱼ´ = s.dual
            order = ifelse(N > 0, N, D+1)
            eⱼseries = Series{order}.(eⱼ, eⱼ´)
            g_integrals_local_e(s, ω, eⱼseries)
        else
            g_integrals_local_e(s, ω, eⱼ)
        end
    end
    return g₀, gⱼ
end

# whether any eₖ == eⱼ for j != k
function is_degenerate(eₖʲ::SMatrix{D´}) where {D´}
    for j in 2:D´, k in 1:j-1
        iszero(eₖʲ[k,j]) && return true
    end
    return false
end

function g_integrals_local_e(s::BandSimplex{D,T}, ω::Number, eⱼ) where {D,T}
    Δⱼ  = ω .- eⱼ
    eₖʲ  = map(chop, transpose(eⱼ) .- eⱼ)  # broadcast too hard for inference
    qⱼ = q_vector(eₖʲ)                              # SVector{D´,T}
    lΔⱼ = logim.(Δⱼ, s.refex)
    Eᴰⱼ = (-1)^D .* Δⱼ.^(D-1) .* lΔⱼ ./ factorial(D-1)
    Eᴰ⁺¹ⱼ = (-1)^(D+1) .* Δⱼ.^D .* lΔⱼ ./ factorial(D)
    if iszero(eₖʲ)                                  # special case, full energy degeneracy
        Δ0 = chop(first(Δⱼ))
        if iszero(Δ0)
            g₀ = zero(complex(T))
            gⱼ = SVector(ntuple(Returns(g₀), Val(D)))
        else
            g₀ = complex(scalar(inv(Δ0)))/factorial(D)
            gⱼ = SVector(ntuple(Returns(g₀/(D+1)), Val(D)))
        end
    else
        g₀ = scalar(sum(qⱼ .* Eᴰⱼ))
        gⱼ = ntuple(Val(D)) do j
            j´ = j + 1
            D´ = D + 1
            x = (Eᴰⱼ[j´] - (-Δⱼ[j´])^(D-1)/factorial(D)) * qⱼ[j´]
            for k in 1:D´
                if k != j´
                    x -= (qⱼ[j´] * Eᴰ⁺¹ⱼ[j´] + qⱼ[k] * Eᴰ⁺¹ⱼ[k]) / eₖʲ[k, j´]
                end
            end
            return scalar(x)
        end |> SVector
    end
    return g₀, gⱼ
end

function q_vector(eₖʲ::SMatrix{D´,D´,S}) where {D´,S}
    qⱼ = ntuple(Val(D´)) do j
        x = one(S)
        for k in 1:D´
            j != k && (x *= eₖʲ[k, j])
        end
        return inv(x)
    end
    return qⱼ
end

# imaginary log with branchcut in the lower plane
logim(x::Complex) = iszero(imag(x)) ? logim(real(x)) : log(-im * x)
logim(x::Real) =  log(abs(x)) - im * 0.5π * sign(x)
logim(x, ex) = logim(x)

# required for local degenerate case (expansion of logim(Δ::Series, ex))
function logim(s::Series{N}, ex) where {N}
    s₀ = scalar(s)
    l₀ = logim(s₀)
    log_coeff = tupletake(ex.log, Val(N-1))
    invzΔ = cumprod(ntuple(Returns(1/s₀), Val(N-1)))
    lⱼ = log_coeff .* invzΔ
    l = Series(l₀, lⱼ...)
    return rescale(l, s[1])
end

#endregion

############################################################################################
# g_integrals_nonlocal: finite-dn g₀(ω) and gⱼ(ω) with normal or hyperdual numbers for φ
#region

function g_integrals_nonlocal(s::BandSimplex{D,T}, ω, dn, ::Val{N} = Val(0)) where {D,T,N}
    ϕⱼ = s.kij * dn
    ϕₖʲ = chop.(transpose(ϕⱼ) .- ϕⱼ)
    eₖʲ = s.eij
    g₀, gⱼ = begin
        if N > 0 || is_degenerate(ϕₖʲ, eₖʲ)
            order = ifelse(N > 0, N, D+1)
        ## This dynamical estimate of the order is not type-stable. Not worth it
        # order = N == 0 ? simplex_degeneracy(ϕₖʲ, eₖʲ) + 1 : N
        # if order > 1
            ϕⱼ´ = s.dual
            ϕⱼseries = Series{order}.(ϕⱼ, ϕⱼ´)
            g_integrals_nonlocal_ϕ(s, ω, ϕⱼseries)
        else
            g_integrals_nonlocal_ϕ(s, ω, ϕⱼ)
        end
    end
    return g₀, gⱼ
end

# # Computes how many denominator zeros in any of the terms of g₀ and gⱼ (max of both)
# function simplex_degeneracy(ϕₖʲ::SMatrix{D´}, eₖʲ) where {D´}
#     deg = degα = degd = 0
#     for j in 1:D´
#         degγ = 0
#         for k in 1:D´
#             k == j && continue
#             e = eₖʲ[k, j]
#             ϕ = ϕₖʲ[k, j]
#             iszero(e) && (degγ += iszero(ϕ))
#             degα = degd = 0
#             for l in 1:D´
#                 e´ = eₖʲ[l,j]
#                 iszero(e) && continue
#                 ϕ´ = ϕₖʲ[l,j]
#                 tt = ϕ*e´≈e*ϕ´
#                 l != k && l != j && (degα += tt)
#                 iszero(e´) && continue
#                 !iszero(e) && tt && iszero(ϕ´) && (degd = 1)
#             end
#             deg = max(deg, degγ + degα + degd)
#         end
#         deg >= D´ && return D´
#     end
#     return deg
# end

# If any ϕₖʲ = 0, or if any tₖʲ and tₗʲ are equal
function is_degenerate(ϕₖʲ::SMatrix{D´}, eₖʲ) where {D´}
    for j in 2:D´, k in 1:j-1
        # iszero(ϕₖʲ[k,j]) && iszero(eₖʲ[k,j]) && return true # fails for g₁
        iszero(ϕₖʲ[k,j]) && return true
        if !iszero(eₖʲ[k,j])
            for l in 1:D´
                if l != k && l != j
                    ϕₖʲ[k,j]*eₖʲ[l,j] ≈ eₖʲ[k,j]*ϕₖʲ[l,j] && return true
                end
            end
        end
    end
    return false
end

function g_integrals_nonlocal_ϕ(s::BandSimplex{D,T}, ω::Number, ϕⱼ) where {D,T}
    eⱼ  = s.ei
    eₖʲ = s.eij
    Δⱼ  = ω .- eⱼ
    ϕₖʲ  = map(chop, transpose(ϕⱼ) .- ϕⱼ)  # broadcast too hard for inference
    tₖʲ = divide_if_nonzero.(ϕₖʲ, eₖʲ)
    eϕⱼ  = cis_scalar.(ϕⱼ, s.refex)
    αₖʲγⱼ  = αγ_matrix(ϕₖʲ, tₖʲ, eₖʲ)               # αₖʲγⱼ :: SMatrix{D´,D´}
    if iszero(eₖʲ)                                  # special case, full energy degeneracy
        Δ0 = chop(first(Δⱼ))
        if iszero(Δ0)
            g₀ = zero(complex(T))
            gⱼ = SVector(ntuple(Returns(g₀), Val(D)))
        else
            Δ0⁻¹ = inv(Δ0)
            γⱼ = αₖʲγⱼ[1,:]                         # if eₖʲ == 0, then αₖʲ == 1
            λⱼ = γⱼ .* eϕⱼ
            λₖʲ = divide_if_nonzero.(transpose(λⱼ), ϕₖʲ)
            q = (-im)^D * Δ0⁻¹
            g₀ = q * sum(scalar.(λⱼ))
            gⱼ = ntuple(Val(D)) do j
                q * scalar(λⱼ[j+1] + im * sum(λₖʲ[:,j+1] - transpose(λₖʲ)[:,j+1]))
            end |> SVector
        end
    else
        αₖʲγⱼeϕⱼ = αₖʲγⱼ .* transpose(eϕⱼ)          # αₖʲγⱼeϕⱼ :: SMatrix{D´,D´}
        Jₖʲ = J_scalar.(tₖʲ, eₖʲ, transpose(Δⱼ), s.refex) # Jₖʲ :: SMatrix{D´,D´}
        αₖʲγⱼeϕⱼJₖʲ = αₖʲγⱼeϕⱼ .* Jₖʲ
        Λⱼ = sum(αₖʲγⱼeϕⱼJₖʲ, dims = 1)
        Λₖʲ = Λ_matrix(eₖʲ, ϕₖʲ, Λⱼ, Δⱼ, tₖʲ, αₖʲγⱼeϕⱼ, Jₖʲ)
        q´ = (-im)^(D+1)
        g₀ = q´ * sum(scalar.(Λⱼ))
        gⱼ = ntuple(Val(D)) do j
            q´ * scalar(Λⱼ[j+1] + im * sum(Λₖʲ[:,j+1] - transpose(Λₖʲ)[:,j+1]))
        end |> SVector
    end
    return g₀, gⱼ
end

# Series of cis(ϕ)
function cis_scalar(s::Series{N}, ex) where {N}
    @assert iszero(s.pow)
    cis_series = Series(tupletake(ex.cis, Val(N)))
    c = cis(s[0]) * cis_series
    # Go from ds differential to dϕ
    return rescale(c, s[1])
end

cis_scalar(s, ex) = cis(s)

divide_if_nonzero(a, b) = iszero(b) ? a : a/b

function αγ_matrix(ϕedges::S, tedges::S, eedges::SMatrix{D´,D´}) where {D´,S<:SMatrix{D´,D´}}
    js = ks = SVector{D´}(1:D´)
    kjs = tuple.(ks, js')
    α⁻¹ = α⁻¹_scalar.(kjs, Ref(tedges), Ref(eedges))
    γ⁻¹ = γ⁻¹_scalar.(js', Ref(ϕedges), Ref(eedges))
    γα = inv.(α⁻¹ .* γ⁻¹)
    return γα
end

function α⁻¹_scalar((k, j), tedges::SMatrix{D´,D´,S}, eedges) where {D´,S}
    x = one(S)
    j != k && !iszero(eedges[k, j]) || return x
    @inbounds for l in 1:D´
        if l != j  && !iszero(eedges[l, j])
            x *= eedges[l, j]
            if l != k # ekj != 0, already constrained above
                x *= chop(tedges[l, j] - tedges[k, j])
            end
        end
    end
    return x
end

function γ⁻¹_scalar(j, ϕedges::SMatrix{D´,D´,S}, eedges) where {D´,S}
    x = one(S)
    @inbounds for l in 1:D´
        if l != j && iszero(eedges[l, j])
            x *= ϕedges[l, j]
        end
    end
    return x
end

function Λ_matrix(eₖʲ::SMatrix{D´}, ϕₖʲ, Λⱼ, Δⱼ, tₖʲ, αₖʲγⱼeϕⱼ, Jₖʲ) where {D´}
    js = ks = SVector{D´}(1:D´)
    ## Inference currently struggles with this
    # kjs = tuple.(ks, js')
    # Λₖʲ = Λ_scalar.(kjs, ϕₖʲ, transpose(Λⱼ), transpose(Δⱼ), Ref(eₖʲ), Ref(tₖʲ), Ref(αₖʲγⱼeϕⱼ), Ref(Jₖʲ))
    kjs = Tuple(tuple.(ks, js'))
    Λₖʲtup = ntuple(Val(D´*D´)) do i
        (k,j) = kjs[i]
        Λ_scalar((k,j), ϕₖʲ[k,j], Λⱼ[j], Δⱼ[j], eₖʲ, tₖʲ, αₖʲγⱼeϕⱼ, Jₖʲ)
    end
    Λₖʲ = SMatrix{D´,D´}(Λₖʲtup)
    return Λₖʲ
end

function Λ_scalar((k, j), ϕₖʲ, Λⱼ, Δⱼ, emat::SMatrix{D´,D´,T}, tmat, αγeϕmat, Jmat) where {D´,T}
    Λₖʲ = zero(typeof(Λⱼ))
    j == k && return Λₖʲ
    eₖʲ = emat[k,j]
    if iszero(eₖʲ)
        Λₖʲ = Λⱼ / ϕₖʲ
    else
        tₖʲ = tmat[k,j]
        Jₖʲ = Jmat[k,j]
        @inbounds for l in 1:D´
            if !iszero(emat[l,j])
                tₗʲ = tmat[l,j]
                Jₗʲ = Jmat[l,j]
                if tₗʲ == tₖʲ
                    Λₖʲ -= (αγeϕmat[l, j] / eₖʲ) * (inv(tₗʲ) + im * Δⱼ * Jₗʲ)
                else
                    Λₖʲ -= (αγeϕmat[l, j] / eₖʲ) * chop(Jₗʲ - Jₖʲ) * inv(chop(tₗʲ - tₖʲ))
                end
            end
        end
    end
    return Λₖʲ
end

function J_scalar(t::T, e, Δ, ex) where {T<:Real}
    iszero(e) && return zero(complex(T))
    tΔ = t * Δ
    J = iszero(tΔ) ? logim(Δ) : cis(tΔ) * J_integral(tΔ, t, Δ)
    return J
end

function J_scalar(t::Series{N,T}, e, Δ, ex) where {N,T<:Real}
    iszero(e) && return zero(Series{N,Complex{T}})
    N´ = N - 1
    C = complex(T)
    iszero(Δ) && return Series{N}(C(Inf))
    t₀ = t[0]
    tΔ = t₀ * Δ
    if iszero(tΔ)
        J₀ = logim(Δ)
        Jᵢ = tupletake(ex.J0, Val(N´)) # ntuple(n -> C(-im)^n/(n*factorial(n)), Val(N´))
        J = Series{N}(J₀, Jᵢ...)
        cis_coefs = tupletake(ex.cis, Val(N))
        E = Series(cis_coefs)   # cis(tΔ) == 1
        EJ = E * J
    else
        cistΔ =  cis(tΔ)
        J₀ = J_integral(tΔ, t₀, Δ)
        if N > 1
            invzΔ = cumprod(ntuple(Returns(1/tΔ), Val(N-1)))
            Jmat = smatrixtake(ex.Jmat, Val(N´))
            Jᵢ = Tuple(inv(cistΔ) * (Jmat * SVector(invzΔ)))
            J = Series(J₀, Jᵢ...)
        else
            J = Series(J₀)
        end
        cis_coefs = tupletake(ex.cis, Val(N))
        Eᵢ = cistΔ .* cis_coefs
        E = Series(Eᵢ)
        EJ = E * J
    end
    return rescale(EJ, t[1] * Δ)
end

function J_integral(tΔ, t, Δ)
    J = iszero(imag(tΔ)) ?
        cosint(abs(tΔ)) - im*sinint(real(tΔ)) - im*0.5π*sign(Δ) :
        -gamma(0, im*tΔ) - im*0.5π*(sign(real(Δ))+sign(real(tΔ)))
    return J
end

#endregion

############################################################################################
# AppliedBandsGreenSolver <: AppliedGreenSolver
#region

struct AppliedBandsGreenSolver{B,SB<:Subband,BS<:BandSimplex} <: AppliedGreenSolver
    subbands::Vector{SB}
    sbsimps::Vector{Vector{BS}}     # BandSimplex for each simplex in each subband
    boundary::B
end

#region ## API ##

minimal_callsafe_copy(s::AppliedBandsGreenSolver) = s   # solver is read-only

needs_omega_shift(s::AppliedBandsGreenSolver) = false

subbands(g::GreenFunction{<:Any,<:Any,<:Any,<:AppliedBandsGreenSolver}) = g.solver.subbands

#endregion

#region ## apply ##

function apply(s::GS.Bands,  h::AbstractHamiltonian{T,<:Any,L}, cs::Contacts) where {T,L}
    b = bands(h, s.bandsargs...; s.bandskw..., projectors = true)
    sbs = subbands(b)
    refex = Ref(Expansions(Val(L), T))
    sbsimps = subband_simplices.(sbs, refex)
    boundary = s.boundary
    return AppliedBandsGreenSolver(sbs, sbsimps, boundary)
end

function subband_simplices(sb::Subband, ex::Expansions)
    refex = Ref(ex)
    sbs = [BandSimplex(energies(sb, simp), transpose(base_coordinates(sb, simp)), refex)
        for simp in simplices(sb)]
    return sbs
end

#endregion

#region ## call ##

function (s::AppliedBandsGreenSolver)(ω, Σblocks, cblockstruct)
    g0slicer = BandsGreenSlicer(complex(ω), s, s.boundary)
    gslicer = TMatrixSlicer(g0slicer, Σblocks, cblockstruct)
    return gslicer
end

#endregion

#endregion

############################################################################################
# BandsGreenSlicer <: GreenSlicer
#region

struct BandsGreenSlicer{C,B,S<:AppliedBandsGreenSolver{B}} <: GreenSlicer{C}
    ω::C
    solver::S
    boundary::B
end

#region ## API ##

Base.getindex(s::BandsGreenSlicer, i::CellOrbitals, j::CellOrbitals) =
    s.boundary === missing ? inf_band_slice(s, i, j) : semi_band_slice(s, i, j)

function inf_band_slice(s::BandsGreenSlicer{C}, i::CellOrbitals, j::CellOrbitals) where {C}
    solver = s.solver
    rows, cols = orbindices(i), orbindices(j)
    dist = cell(i) - cell(j)
    g = zeros(C, length(rows), length(cols))
    for (sb, bsimps) in zip(solver.subbands, solver.sbsimps)
        projs = projectors(sb)
        for (simpind, simp) in enumerate(simplices(sb))
            bsimp = bsimps[simpind]
            v₀, vⱼs... = simp
            g₀, gⱼs = g_integrals(bsimp, s.ω, dist)
            ψ = view(states(vertices(sb, v₀)), rows, :)
            ψ´ = view(states(vertices(sb, v₀)), cols, :)'
            pind = (simpind, v₀)
            muladd_ψPψ⁺!(g, ψ, ψ´, bsimp.VD * (g₀ - sum(gⱼs)), projs, pind)
            for (j, gⱼ) in enumerate(gⱼs)
                vⱼ = vⱼs[j]
                ψ = view(states(vertices(sb, vⱼ)), rows, :)
                ψ´ = view(states(vertices(sb, vⱼ)), cols, :)'
                pind = (simpind, vⱼ)
                muladd_ψPψ⁺!(g, ψ, ψ´, bsimp.VD * gⱼ, projs, pind)
            end
        end
    end
    return g
end

function semi_band_slice(s::BandsGreenSlicer{C}, i::CellOrbitals, j::CellOrbitals) where {C}
    interalerror("semi_band_slice: work-in-progress")
end

# does g += α * ψPψ´, where P = projs[pind] is the vertex projector onto the simplex subspace
# If pind = (simpind, vind) is not in projs::Dict, no projector P is necessary
muladd_ψPψ⁺!(g, ψ, ψ´, α, projs, pind) =
    haskey(projs, pind) ? mul!(g, ψ, projs[pind] * ψ´, α, 1) : mul!(g, ψ, ψ´, α, 1)

minimal_callsafe_copy(s::BandsGreenSlicer) = s  # it is read-only

#endregion

#endregion
