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

function trim_and_map(func, d::Series{N}, d´::Series{N}) where {N}
    f, f´ = trim(d), trim(d´)
    pow = min(f.pow, f´.pow)
    t = ntuple(i -> func(f[pow + i - 1], f´[pow + i - 1]), Val(N))
    return Series(t, pow)
end

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
    Q⁻¹::SU       # Q::SMatrix{D,D,T,DD} = cols are basis of shift vectors δrᵝ
    phi´::S2      # kij * Q
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
    Q = generate_Q(eij, kij)                    # Q is unitary
    phi´ = kij * Q
    Q⁻¹ = Q'
    return BandSimplex(ei, kij, eij, U⁻¹, Q⁻¹, phi´, w, VD)
end

function generate_Q(eij, kij::SMatrix{<:Any,D,T}) where {D,T}
    Q = one(SMatrix{D,D,T})
    iszero(eij) && return Q
    while !is_valid_Q(Q, eij, kij)
        Q = first(qr(rand(SMatrix{D,D,T})))
    end
    return Q
end

function is_valid_Q(Q, es, ks)
    for qβ in eachcol(Q)
        phi = ks * qβ
        phis = phi' .- phi
        for j in axes(es, 2), k in axes(es, 1), l in axes(es, 1)
            l != k != j && l != k || continue
            eʲₖ = es[k,j]
            eʲₗ = es[l,j]
            (iszero(eʲₖ) || iszero(eʲₗ)) && continue
            phis[k, j] * eʲₗ ≈ eʲₖ * phis[l, j] && return false
        end
    end
    return true
end

#endregion

############################################################################################
# BandSimplexSubspace: basis of subspaces at each vertex that interpolate inside simplex
#region

struct BandSimplexSubspace{D,N,S<:SMatrix{<:Any,N},M<:MatrixView,X<:BandSimplex{D}}
    simplex::X
    states::Tuple{M,Vararg{M,D}}  # basis of degenerate states at each vertex
    coords::Tuple{S,Vararg{S,D}}  # an s::Smatrix per vertex such that states * s = subspace
end

function BandSimplexSubspace(vs::NTuple{D´,BandVertex{T,D´}}) where {D´,T}
    cs = coordinates.(vs)
    es = energy.(cs)
    ks = base_coordinates.(cs)
    simplex = BandSimplex(es, ks)
    states´ = states.(vs)
    coords = simplexsubspace(states´)
    return BandSimplexSubspace(simplex, states´, coords)
end

#endregion

############################################################################################
# g0(ω) and g_j(ω) for a BandSimplex{D}, with D´ = D + 1
#region

struct Expansions{N,TC<:NTuple{N},TJ,SJ}
    cis::TC
    J0::TJ
    Jmat::SJ
end

# Precomputes the Series expansion coefficients for cis, J(z->0) and J(z)
function Expansions(::Val{N´}, ::Type{T}) where {N´,T}  # here N´ = N-1
    C = complex(T)
    cis = ntuple(n -> C(im)^(n-1)/(factorial(n-1)), Val(N´+1))
    J0 = ntuple(n -> C(-im)^n/(n*factorial(n)), Val(N´))
    Jmat = ntuple(Val(N´*N´)) do ij
        j, i = fldmod1(ij, N´)
        j > i ? zero(C) : ifelse(isodd(i), 1, -1) * C(im)^(i-j) / (i*factorial(i-j))
    end |> SMatrix{N´,N´,C}
    return Expansions(cis, J0, Jmat)
end

# OPTIM: could perhaps dispatch to Val(2) in the case with no degeneracies?
g_simplex(s::BandSimplex{D}, ω, dn) where {D} = g_simplex(s, ω, dn, Val(D+1))

function g_simplex(s::BandSimplex{D,T}, ω, dn, ::Val{N}) where {D,T,N}
    gβ = ntuple(Val(D)) do β
        ϕ´verts = s.phi´[:, β]
        ex = Expansions(Val(N-1), T)
        g_simplex(s, ω, dn, ϕ´verts, ex)
    end
    g0, g1 = first(first(gβ)), last.(gβ)
    # gk should be SVector(g1)' * s.Q⁻¹, but we return the transpose
    gk = s.Q⁻¹' * SVector(g1)
    return g0, gk
end

function g_simplex(s::BandSimplex{D,T}, ω::Number, dn::SVector{D}, ϕ´verts::SVector, ex::Expansions{N}) where {D,T,N}
    # phases ϕverts[j+1] will be perturbed by ϕ´verts[j+1]*dϕ, for j in 0:D
    # Similartly, ϕedges[j+1,k+1] will be perturbed by ϕ´edges[j+1,k+1]*dϕ
    ϕ´edges = ϕ´verts' .- ϕ´verts
    ϕverts0 = s.kij * dn
    ϕverts = Series{N}.(ϕverts0, ϕ´verts)
    ϕedges = Series{N}.(chop.(ϕverts0' .- ϕverts0), ϕ´edges)
    Δverts = ω .- s.ei
    eedges = s.eij
    zedges = zkj_series.(ϕedges, eedges)
    eϕ = cis_series.(ϕverts, Ref(ex))
    γα = γα_series(ϕedges, zedges, eedges)            # SMatrix{D´,D´}
    if iszero(eedges)                                 # special case, full energy degeneracy
        Δ0 = chop(first(Δverts))
        eγαJ = iszero(Δ0) ? zero(Series{N,complex(T)}) :  im * sum(γα[1,:] .* eϕ) / Δ0
    else
        J = J_series.(zedges, eedges, transpose(Δverts), Ref(ex))  # SMatrix{D´,D´}
        eγαJ = sum(γα .* J .* transpose(eϕ))              # manual contraction is slower!
    end
    gsum = (-im)^(D+1) * s.VD * trim(chop(eγαJ))
    return gsum[0], gsum[1]
end

zkj_series(ϕ, e) = iszero(e) ? ϕ : ϕ/e

@inline function cis_series(z::Series{N}, ex) where {N}
    @assert iszero(z.pow)
    c = cis_series(z[0], ex)
    # Go from dz differential to dϕ
    return rescale(c, z[1])
end

# Series of cis(ϕ)
cis_series(ϕ::Real, ex) = cis(ϕ) * Series(ex.cis)

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

function α⁻¹_series((j, k), zedges::SMatrix{D´,D´,S}, eedges) where {D´,S<:Series}
    x = one(S)
    @inbounds j != k && !iszero(eedges[k, j]) || return x
    @inbounds for l in 1:D´
        if l != j  && !iszero(eedges[l, j])
            x *= eedges[l, j]
            if l != k # ekj != 0, already constrained above
                x *= zedges[l, j] - zedges[k, j]
            end
        end
    end
    return x
end

function γ⁻¹_series(j, ϕedges::SMatrix{D´,D´,S}, eedges) where {D´,S<:Series}
    x = one(S)
    @inbounds for l in 1:D´
        if l != j && iszero(eedges[l, j])
            x *= ϕedges[l, j]
        end
    end
    return x
end

@inline function J_series(z::Series{N,T}, e, Δ, ex) where {N,T}
    iszero(e) && return zero(Series{N,Complex{T}})
    J = J_series(z[0], Δ, ex)
    # Go from d(zΔ) = dz*Δ differential to dϕ
    return rescale(J, z[1] * Δ)
end

# Series of J(zΔ) = cis(zΔ) * [Ci(|z|Δ) - i Si(zΔ)] (variable zΔ for Series)
function J_series(z::T, Δ::T, ex::Expansions{N}) where {N,T<:Number}
    C = complex(T)
    iszero(Δ) && return Series{N}(C(Inf))
    zΔ = z * Δ
    imπ = im * ifelse(Δ > 0, 0, π) # strangely enough, union splitting is faster than stable
    if iszero(zΔ)
        J₀ = log(abs(Δ)) + imπ #+ MathConstants.γ + log(|z|) # not needed, cancels out
        Jᵢ = ex.J0  # = ntuple(n -> (-im)^n/(n*factorial(n)), Val(N-1))
        J = Series{N}(J₀, Jᵢ...)
        E = cis_series(zΔ, ex)
        EJ = E * J
    else
        ciszΔ =  cis(zΔ)
        J₀ = cosint(abs(zΔ)) - im*sinint(zΔ) + imπ
        if N > 1
            invzΔ = cumprod(ntuple(Returns(1/zΔ), Val(N-1)))
            Jᵢ = Tuple(conj(ciszΔ) * (ex.Jmat * SVector(invzΔ)))
                # Jᵢ = conj(ciszΔ) .* ntuple(Val(N-1)) do n
                #    (-1)^(n-1) * sum(m -> im^m * zΔ^(m-n)/(n*factorial(m)), 0:n-1)
                # end
            J = Series(J₀, Jᵢ...)
        else
            J = Series(J₀)
        end
        Eᵢ = ciszΔ .* ex.cis
        E = Series(Eᵢ)
        EJ = E * J
    end
    return EJ
end

#endregion
