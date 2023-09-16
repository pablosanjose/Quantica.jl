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

trim(x::Number) = x

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

Base.one(::Type{<:Series{N,T}}) where {N,T<:Number} = Series{N}(one(T))
Base.zero(::Type{<:Series{N,T}}) where {N,T} = Series(zero(SVector{N,T}), 0)
Base.iszero(d::Series) = iszero(d.x)
Base.transpose(d::Series) = d  # act as a scalar

Base.:+(d::Series, d´::Series) = trim_and_map(+, d, d´)
Base.:-(d::Series, d´::Series) = trim_and_map(-, d, d´)
Base.:+(d::Number, d´::Series{N}) where {N} = Series{N}(d) + d´
Base.:-(d::Number, d´::Series{N}) where {N} = Series{N}(d) - d´
Base.:*(d::Number, d´::Series) = Series(d * d´.x, d´.pow)
Base.:*(d´::Series, d::Number) = Series(d * d´.x, d´.pow)
Base.:/(d::Series{N}, d´::Series{N}) where {N} = d * inv(d´)
Base.:/(d::Series, d´::Number) = Series(d.x / d´, d.pow)

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

struct BandSimplex{D,T,S1,S2,S3,SU<:SMatrix{D,D,T}}     # D = manifold dimension
    ei::S1        # eᵢ::SVector{D´,T} = energy of vertex i
    kij::S2       # kᵢ[j]::SMatrix{D´,D,T,DD´} = coordinate j of momentum for vertex i
    eij::S3       # ϵᵢʲ::SMatrix{D´,D´,T,D´D´} = e_j - e_i
    U⁻¹::SU       # inv(Uᵢⱼ) for Uᵢⱼ::SMatrix{D,D,T,DD} = kⱼ[i] - k₀[i] (edges as cols)
    phi´::S1      # first hyperdual coefficient of φ
    w::SVector{D,T} # U⁻¹ * k₀
    VD::T         # D!V = |det(U)|
end

function BandSimplex(es::NTuple{<:Any,Number}, ks::NTuple{<:Any,SVector})
    ei = SVector(es)
    kij = transpose(reduce(hcat, ks))   # ks are rows of kij
    return BandSimplex(ei, kij)
end

function BandSimplex(ei::SVector{D´}, kij::SMatrix{D´,D,T}) where {D´,D,T}
    D == D´ - 1 ||
        argerror("The dimension $D of Bloch phases in simplex should be one less than the number of vertices $(D´)")
    eij = chop(ei' .- ei)
    k0 = kij[1, :]
    U = kij[SVector{D}(2:D´),:]' .- k0          # edges as columns
    U⁻¹ = inv(U)
    VD = abs(det(U))
    w = U⁻¹ * k0
    phi´ = generate_phi´(eij)
    return BandSimplex(ei, kij, eij, U⁻¹, phi´, w, VD)
end

function generate_phi´(eij::SMatrix{D´,D´,T}) where {D´,T}
    phi´ = rand(SVector{D´,T})
    iszero(eij) && return phi´
    while !is_valid_phi´(phi´, eij)
        phi´ = rand(SVector{D´,T})
    end
    return phi´
end

# check whether iszero(eʲₖφʲₗ - φʲₖeʲₗ) for nonzero e's
function is_valid_phi´(phi, es)
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

function g_integrals(s::BandSimplex, ω, dn)
    # g0, gi = if iszero(dn)
    #     if any(iszero, s.eij)
    #         g_integrals_local_series(s, ω, dn)
    #     else
    #         g_integrals_local(s, ω, dn)
    #     end
    # else
    #     if any(iszero, dn)
    #         g_integrals_nonlocal_series(s, ω, dn)
    #     else
    #         g_integrals_nonlocal(s, ω, dn)
    #     end
    # end
    # g0, gi = g_integrals_nonlocal_series(s, ω, dn)
    g0, gi = g_integrals_nonlocal(s, ω, dn)
    return g0, gi
end

#endregion

############################################################################################
# g_integrals_nonlocal: g₀(ω) and gⱼ(ω) with normal or hyperdual numbers for φ
#region

# struct Expansions{N,TC<:NTuple{N},TJ,SJ}
#     cis::TC
#     J0::TJ
#     Jmat::SJ
# end

# # Precomputes the Series expansion coefficients for cis, J(z->0) and J(z)
# function Expansions(::Val{N´}, ::Type{T}) where {N´,T}  # here N´ = N-1
#     C = complex(T)
#     cis = ntuple(n -> C(im)^(n-1)/(factorial(n-1)), Val(N´+1))
#     J0 = ntuple(n -> C(-im)^n/(n*factorial(n)), Val(N´))
#     Jmat = ntuple(Val(N´*N´)) do ij
#         j, i = fldmod1(ij, N´)
#         j > i ? zero(C) : ifelse(isodd(i), 1, -1) * C(im)^(i-j) / (i*factorial(i-j))
#     end |> SMatrix{N´,N´,C}
#     return Expansions(cis, J0, Jmat)
# end

function g_integrals_nonlocal(s::BandSimplex{D,T}, ω, dn, ::Val{N} = Val(0)) where {D,T,N}
    ϕⱼ = s.kij * dn
    ϕₖʲ = chop.(transpose(ϕⱼ) .- ϕⱼ)
    eₖʲ = s.eij
    g0, gj = begin
        if N > 0 || is_degenerate(ϕₖʲ, eₖʲ)
            # phases ϕⱼ[j+1] will be perturbed by ϕⱼ´[j+1]*dϕ, for j in 0:D
            # Similartly, ϕₖʲ[j+1,k+1] will be perturbed by ϕₖʲ´[j+1,k+1]*dϕ
            ϕⱼ´ = s.phi´
            order = ifelse(N > 0, N, D+1)
            ϕⱼseries = Series{order}.(ϕⱼ, ϕⱼ´)
            # ex = Expansions(Val(order-1), T)
            g_integrals_nonlocal_ϕ(s, ω, ϕⱼseries)
        else
            g_integrals_nonlocal_ϕ(s, ω, ϕⱼ)
        end
    end
    return g0, gj
end

# If any ϕₖʲ is zero, or if any tₖʲ and tₗʲ are equal
function is_degenerate(ϕₖʲ::SMatrix{D´}, eₖʲ) where {D´}
    for j in 2:D´, k in 1:j-1
        iszero(ϕₖʲ[k,j]) && return true
        for l in 1:D´
            if l != j && l != k
                ϕₖʲ[k,j]*eₖʲ[l,j] ≈ eₖʲ[k,j]*ϕₖʲ[l,j] && return true
            end
        end
    end
    return false
end

function g_integrals_nonlocal_ϕ(s::BandSimplex{D,T}, ω::Number, ϕⱼ) where {D,T}
    eⱼ  = s.ei
    eₖʲ = s.eij
    Δⱼ  = ω .- eⱼ
    ϕₖʲ  = map(x -> trim(chop(x)), transpose(ϕⱼ) .- ϕⱼ)  # broadcast too hard for inference
    tₖʲ = divide_if_nonzero.(ϕₖʲ, eₖʲ)
    eϕⱼ  = cis.(ϕⱼ)
    αₖʲγⱼ  = αγ_matrix(ϕₖʲ, tₖʲ, eₖʲ)               # αₖʲγⱼ :: SMatrix{D´,D´}
    if iszero(eₖʲ)                                  # special case, full energy degeneracy
        Δ0 = chop(first(Δⱼ))
        if iszero(Δ0)
            g0 = zero(complex(T))
            gj = SVector(ntuple(Returns(g0), Val(D)))
        else
            Δ0⁻¹ = inv(Δ0)
            γⱼ = αₖʲγⱼ[1,:]                         # if eₖʲ == 0, then αₖʲ == 1
            λⱼ = γⱼ .* eϕⱼ
            λₖʲ = λⱼ ./ ϕₖʲ
            q = (-im)^D * s.VD * Δ0⁻¹
            g0 = q * trim(chop(sum(λⱼ))) |> scalar
            gj = ntuple(Val(D)) do j
                q * scalar(trim(chop(λⱼ[j+1] + im * sum(λₖʲ[:,j+1] - transpose(λₖʲ)[:,j+1]))))
            end |> SVector
        end
    else
        αₖʲγⱼeϕⱼ = αₖʲγⱼ .* transpose(eϕⱼ)          # αₖʲγⱼeϕⱼ :: SMatrix{D´,D´}
        Jₖʲ = J_matrix.(tₖʲ, eₖʲ, transpose(Δⱼ))  # Jₖʲ :: SMatrix{D´,D´}
        αₖʲγⱼeϕⱼJₖʲ = αₖʲγⱼeϕⱼ .* Jₖʲ
        Λⱼ = sum(αₖʲγⱼeϕⱼJₖʲ, dims = 1)
        Λⱼsum = sum(Λⱼ)                             # αₖʲγⱼJʲₖ (manual contraction slower!)
        Λₖʲ = Λ_matrix(eₖʲ, ϕₖʲ, Λⱼ, Δⱼ, tₖʲ, αₖʲγⱼeϕⱼ, Jₖʲ)
        q´ = (-im)^(D+1) * s.VD
        g0 = q´ * scalar(trim(chop(Λⱼsum)))
        gj = ntuple(Val(D)) do j
            q´ * scalar(trim(chop(Λⱼ[j+1] + im * sum(Λₖʲ[:,j+1] - transpose(Λₖʲ)[:,j+1]))))
        end |> SVector
    end
    return g0, gj
end

# Series of cis(ϕ)
@inline function Base.cis(s::Series{N,T}) where {N,T}
    @assert iszero(s.pow)
    C = Complex{T}
    cis_series = Series(ntuple(n -> C(im)^(n-1)/(factorial(n-1)), Val(N)))
    c = cis(s[0]) * cis_series
    # Go from ds differential to dϕ
    return rescale(c, s[1])
end

divide_if_nonzero(a, b) = iszero(b) ? a : a/b

@inline function αγ_matrix(ϕedges::S, tedges::S, eedges::SMatrix{D´,D´}) where {D´,S<:SMatrix{D´,D´}}
    # js = ks = SVector{D´}(1:D´)
    # α⁻¹ = α⁻¹_series.(js', ks, Ref(tedges), Ref(eedges))
    # γ⁻¹ = γ⁻¹_series.(js', Ref(ϕedges), Ref(eedges))
    # γα = inv.(α⁻¹ .* γ⁻¹)
    # return γα
    ## BUG: broadcast over SArrays is currently allocations-buggy
    ## https://github.com/JuliaArrays/StaticArrays.jl/issues/1178
    js = ks = SVector{D´}(1:D´)
    jks = Tuple(tuple.(js', ks))
    α⁻¹ = SMatrix{D´,D´}(α⁻¹_scalar.(jks, Ref(tedges), Ref(eedges)))
    γ⁻¹ = SVector(γ⁻¹_scalar.(Tuple(js), Ref(ϕedges), Ref(eedges)))
    γα = inv.(α⁻¹ .* transpose(γ⁻¹))
    return γα
end

function α⁻¹_scalar((j, k), tedges::SMatrix{D´,D´,S}, eedges) where {D´,S}
    x = one(S)
    @inbounds j != k && !iszero(eedges[k, j]) || return x
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
        for l in 1:D´
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

function J_matrix(t::T, e, Δ) where {T<:Number}
    iszero(e) && return zero(complex(T))
    tΔ = t * Δ
    imπ = im * ifelse(Δ > 0, 0, π)
    J = iszero(tΔ) ? log(abs(Δ)) + imπ : cis(tΔ) * (cosint(abs(tΔ)) - im*sinint(tΔ) + imπ)
    return J
end

@inline function J_matrix(t::Series{N,T}, e, Δ) where {N,T}
    iszero(e) && return zero(Series{N,Complex{T}})
    N´ = N - 1
    C = complex(T)
    iszero(Δ) && return Series{N}(C(Inf))
    tΔ = t[0] * Δ
    imπ = im * ifelse(Δ > 0, 0, π) # strangely enough, union splitting is faster than stable
    if iszero(tΔ)
        J₀ = log(abs(Δ)) + imπ #+ MathConstants.γ + log(|t|) # not needed, cancels out
        Jᵢ = ntuple(n -> C(-im)^n/(n*factorial(n)), Val(N´))
        J = Series{N}(J₀, Jᵢ...)
        E = cis(tΔ)
        EJ = E * J
    else
        cistΔ =  cis(tΔ)
        J₀ = cosint(abs(tΔ)) - im*sinint(tΔ) + imπ
        if N > 1
            invzΔ = cumprod(ntuple(Returns(1/tΔ), Val(N-1)))
            Jmat = ntuple(Val(N´*N´)) do ij
                j, i = fldmod1(ij, N´)
                j > i ? zero(C) : ifelse(isodd(i), 1, -1) * C(im)^(i-j) / (i*factorial(i-j))
            end |> SMatrix{N´,N´,C}
            Jᵢ = Tuple(conj(cistΔ) * (Jmat * SVector(invzΔ)))
            J = Series(J₀, Jᵢ...)
        else
            J = Series(J₀)
        end
        cis_series = ntuple(n -> C(im)^(n-1)/(factorial(n-1)), Val(N))
        Eᵢ = cistΔ .* cis_series
        E = Series(Eᵢ)
        EJ = E * J
    end
    return return rescale(EJ, t[1] * Δ)
end

#endregion
