#######################################################################
# Kernel Polynomial Method : momenta
#######################################################################
using Base.Threads

struct MomentaKPM{T,B<:Tuple}
    mulist::Vector{T}
    bandbracket::B
end

struct KPMBuilder{A,H<:AbstractMatrix,T,K,B}
    A::A
    h::H
    bandbracket::Tuple{B,B}
    order::Int
    missingket::Bool
    mulist::Vector{T}
    ket::K
    ket0::K
    ket1::K
end

function KPMBuilder(h::AbstractMatrix{Tv}, A = _defaultA(Tv); ket = missing, order = 10, bandrange = missing) where {Tv}
    eh = eltype(eltype(h))
    aA = eltype(eltype(A))
    mulist = zeros(promote_type(eh, aA), order + 1)
    bandbracket = bandbracketKPM(h, bandrange)
    missingket = ket === missing
    ket´ = missingket ? ketundef(h) : ket
    iscompatibleket(h, ket´) || throw(ArgumentError("ket is incompatible with Hamiltonian"))
    builder = KPMBuilder(A, h, bandbracket, order, missingket, mulist, ket´, similar(ket´), similar(ket´))
    return builder
end

ketundef(h::AbstractMatrix{T}) where {T<:Number} =
    Vector{T}(undef, size(h, 2))
ketundef(h::AbstractMatrix{S}) where {N,T,S<:SMatrix{N,N,T}} =
    Vector{SVector{N,T}}(undef, size(h, 2))

iscompatibleket(h::AbstractMatrix{T1}, ket::AbstractArray{T2}) where {T1,T2} =
    _iscompatibleket(T1, T2)
_iscompatibleket(::Type{T1}, ::Type{T2}) where {T1<:Real, T2<:Real} = true
_iscompatibleket(::Type{T1}, ::Type{T2}) where {T1<:Number, T2<:Complex} = true
_iscompatibleket(::Type{S1}, ::Type{S2}) where {N, S1<:SMatrix{N,N}, S2<:SVector{N}} =
    _iscompatibleket(eltype(S1), eltype(S2))
_iscompatibleket(::Type{S1}, ::Type{S2}) where {N, S1<:SMatrix{N,N}, S2<:SMatrix{N}} =
    _iscompatibleket(eltype(S1), eltype(S2))
_iscompatibleket(t1, t2) = false

function matrixKPM(h::Hamiltonian{<:Lattice,L}, method = missing) where {L}
    iszero(L) ||
        throw(ArgumentError("Hamiltonian is defined on an infinite lattice. Convert it to a matrix first using `bloch(h, φs...)`"))
    m = similarmatrix(h, method)
    return bloch!(m, h)
end

matrixKPM(h::AbstractMatrix) = h
matrixKPM(A::UniformScaling) = A

"""
    momentaKPM(h::AbstractMatrix, A = I; ket = missing, order = 10, randomkets = 1, bandrange = missing)

Compute the Kernel Polynomial Method (KPM) momenta `μ_n = ⟨ket|T_n(h) A|ket⟩/⟨ket|ket⟩` where `T_n(x)`
is the Chebyshev polynomial of order `n`, for a given `ket::AbstractVector`, hamiltonian `h`, and
observable `A`. If `ket` is `missing`, momenta are computed by means of a stochastic trace
`μ_n = Tr[A T_n(h)] ≈ ∑ₐ⟨a|A T_n(h)|a⟩/N` over `N = randomkets` normalized random `|a⟩`.
Furthermore, the trace over a specific set of kets can also be computed; in this case
`ket::AbstractMatrix` must be a matrix where the columns are the kets involved in the calculation.

The order of the Chebyshev expansion is `order`. The `bandbrange = (ϵmin, ϵmax)` should completely encompass
the full bandwidth of `hamiltonian`. If `missing` it is computed automatically using `ArnoldiMethods` (must be loaded).

# Examples

```
julia> h = LatticePresets.cubic() |> hamiltonian(hopping(1)) |> unitcell(region = RegionPresets.sphere(10));

julia> momentaKPM(bloch(h), bandrange = (-6,6))
Quantica.MomentaKPM{Complex{Float64},Tuple{Float64,Float64}}(Complex{Float64}[0.9594929736144989 + 0.0im, 0.00651662540445511 - 1.3684099632763213e-18im, 0.4271615999695687 + 0.0im, 0.011401934070884771 - 8.805365601448575e-19im, 0.2759482493684239 + 0.0im, 0.001128522288518446 + 4.914851192831956e-19im, 0.08738420162067032 + 0.0im, 0.0007921516166325597 + 2.0605151351830466e-19im, 0.00908824008889868 + 0.0im, -5.638793856739318e-20 - 2.2295921941414733e-35im, 1.2112238859024637e-16 + 0.0im], (0.0, 6.030150753768845))
```
"""
function momentaKPM(h::Hamiltonian, A = _defaultA(eltype(h)); bandrange = missing, kw...)
    bandrange´ = bandrange === missing ? bandrangeKPM(h) : bandrange
    momenta = momentaKPM(matrixKPM(h), matrixKPM(A); bandrange = bandrange´, kw...)
    return momenta
end

function momentaKPM(h::AbstractMatrix, A = _defaultA(eltype(h)); randomkets = 1, kw...)
    b = KPMBuilder(h, A; kw...)
    if b.missingket
        pmeter = Progress(b.order * randomkets, "Averaging moments: ")
        for n in 1:randomkets
            randomize!(b.ket)
            addmomentaKPM!(b, pmeter)
        end
        b.mulist ./= randomkets
    else
        pmeter = Progress(b.order, "Computing moments: ")
        addmomentaKPM!(b, pmeter)
    end
    jackson!(b.mulist)
    return MomentaKPM(b.mulist, b.bandbracket)
end

_defaultA(::Type{T}) where {T<:Number} = one(T) * I
_defaultA(::Type{S}) where {N,T,S<:SMatrix{N,N,T}} = one(T) * I

# This iterates bras <psi_n| = <psi_0|AT_n(h) instead of kets (faster CSC multiplication)
# In practice we iterate their conjugate |psi_n> = T_n(h') A'|psi_0>, and do the projection
# onto the start ket, |psi_0>
function addmomentaKPM!(b::KPMBuilder{<:AbstractMatrix,<:AbstractSparseMatrix}, pmeter)
    mulist, ket, ket0, ket1 = b.mulist, b.ket, b.ket0, b.ket1
    h, A, bandbracket = b.h, b.A, b.bandbracket
    order = length(mulist) - 1
    mul!(ket0, A', ket)
    mulscaled!(ket1, h', ket0, bandbracket)
    mulist[1] += proj(ket0, ket)
    mulist[2] += proj(ket1, ket)
    for n in 3:(order+1)
        ProgressMeter.next!(pmeter; showvalues = ())
        iterateKPM!(ket0, h', ket1, bandbracket)
        mulist[n] += proj(ket0, ket)
        ket0, ket1 = ket1, ket0
    end
    return mulist
end

function addmomentaKPM!(b::KPMBuilder{<:UniformScaling, <:AbstractSparseMatrix,T}, pmeter) where {T}
    mulist, ket, ket0, ket1, = b.mulist, b.ket, b.ket0, b.ket1
    h, A, bandbracket = b.h, b.A, b.bandbracket
    order = length(mulist) - 1
    ket0 .= ket
    mulscaled!(ket1, h', ket0, bandbracket)
    mulist[1] += μ0 = 1.0
    mulist[2] += μ1 = proj(ket1, ket0)
    # This is not used in the currently activated codepath (BLAS mul!), but is needed in the
    # commented out @threads codepath
    thread_buffers = (zeros(T, Threads.nthreads()), zeros(T, Threads.nthreads()))
    for n in 3:2:(order+1)
        ProgressMeter.next!(pmeter; showvalues = ())
        ProgressMeter.next!(pmeter; showvalues = ()) # twice because of 2-step
        proj11, proj10 = iterateKPM!(ket0, h', ket1, bandbracket, thread_buffers)
        mulist[n] += 2 * proj11 - μ0
        n + 1 > order + 1 && break
        mulist[n + 1] += 2 * proj10 - μ1
        ket0, ket1 = ket1, ket0
    end
    A.λ ≈ 1 || (mulist .*= A.λ)
    return mulist
end

function mulscaled!(y, h´, x, (center, halfwidth))
    mul!(y, h´, x)
    invhalfwidth = 1/halfwidth
    @. y = (y - center * x) * invhalfwidth
    return y
end

function iterateKPM!(ket0, h´, ket1, (center, halfwidth), buff = ())
    α = 2/halfwidth
    β = 2center/halfwidth
    mul!(ket0, h´, ket1, α, -1)
    @. ket0 = ket0 - β * ket1
    return proj_or_nothing(buff, ket0, ket1)
end

proj_or_nothing(::Tuple{}, ket0, ket1) = nothing
proj_or_nothing(buff, ket0, ket1) = (proj(ket1, ket1), proj(ket1, ket0))

# This is equivalent to tr(ket1'*ket2) for matrices, and ket1'*ket2 for vectors
proj(ket1, ket2) = dot(vec(ket1), vec(ket2))

# function iterateKPM!(ket0, h´, ket1, (center, halfwidth), thread_buffers = ())
#     h = parent(h´)
#     nz = nonzeros(h)
#     rv = rowvals(h)
#     α = -2 * center / halfwidth
#     β = 2 / halfwidth
#     reset_buffers!(thread_buffers)
#     @threads for row in 1:size(ket0, 1)
#         ptrs = nzrange(h, row)
#         @inbounds for col in 1:size(ket0, 2)
#             k1 = ket1[row, col]
#             tmp = α * k1 - ket0[row, col]
#             for ptr in ptrs
#                 tmp += β * adjoint(nz[ptr]) * ket1[rv[ptr], col]
#             end
#             # |k0⟩ → (⟨k1|2h - ⟨k0|)' = 2h'|k1⟩ - |k0⟩
#             ket0[row, col] = tmp
#             update_buffers!(thread_buffers, k1, tmp)
#         end
#     end
#     return sum_buffers(thread_buffers)
# end

# reset_buffers!(::Tuple{}) = nothing
# function reset_buffers!((q, q´))
#     fill!(q, zero(eltype(q)))
#     fill!(q´, zero(eltype(q´)))
#     return nothing
# end

# @inline update_buffers!(::Tuple{}, k1, tmp) = nothing
# @inline function update_buffers!((q, q´), k1, tmp)
#     q[threadid()]  += dot(k1, k1)
#     q´[threadid()] += dot(tmp, k1)
#     return nothing
# end

# @inline sum_buffers(::Tuple{}) = nothing
# @inline sum_buffers((q, q´)) = (sum(q), sum(q´))

function randomize!(v::AbstractVector{T}) where {T}
    v .= _randomize.(v)
    normalize!(v)
    return v
end
@inline _randomize(v::T) where {T<:Real} = randn(R)
@inline _randomize(v::T) where {R,T<:Complex{R}} = randn(R) + im * randn(R)
@inline _randomize(v::T) where {T<:SArray} = _randomize.(v)

function jackson!(μ::AbstractVector)
    order = length(μ) - 1
    @inbounds for n in eachindex(μ)
        μ[n] *= ((order - n + 1) * cos(π * n / (order + 1)) +
                sin(π * n / (order + 1)) * cot(π / (order + 1))) / (order + 1)
    end
    return μ
end

function bandbracketKPM(h, ::Missing)
    bandbracketKPM(h, bandrangeKPM(h))
end
bandbracketKPM(h, (ϵmin, ϵmax)::Tuple{T,T}, pad = float(T)(0.01)) where {T} = ((ϵmax + ϵmin) / 2, (ϵmax - ϵmin) / (2 - pad))

bandrangeKPM(h::Hamiltonian) = bandrangeKPM(matrixKPM(h, ArnoldiMethodPackage()))

function bandrangeKPM(h::AbstractMatrix{T}) where {T}
    @warn "Computing spectrum bounds... Consider using the `bandrange` option for faster performance."
    checkloaded(:ArnoldiMethod)
    R = real(T)
    decompl, _ = Main.ArnoldiMethod.partialschur(h, nev=1, tol=1e-4, which = Main.ArnoldiMethod.LR());
    decomps, _ = Main.ArnoldiMethod.partialschur(h, nev=1, tol=1e-4, which = Main.ArnoldiMethod.SR());
    ϵmax = R(real(decompl.eigenvalues[1]))
    ϵmin = R(real(decomps.eigenvalues[1]))
    @warn  "Computed bandrange = ($ϵmin, $ϵmax)"
    return (ϵmin, ϵmax)
end

#######################################################################
# Kernel Polynomial Method : observables
#######################################################################
"""
    dosKPM(h::AbstractMatrix; resolution = 2, kw...)

Compute, using the Kernel Polynomial Method (KPM), the local density of states `ρ(ϵ) =
⟨ket|δ(ϵ-h)|ket⟩/⟨ket|ket⟩` for a given `ket::AbstractVector` and hamiltonian `h`, or the
global density of states `ρ(ϵ) = Tr[δ(ϵ-h)]` if `ket` is `missing`.

If `ket` is an `AbstractMatrix` it evaluates the trace over the set of kets in `ket` (see
`momentaKPM` and its options `kw` for further details). The result is a tuple of energy
points `xk::Vector` and real `ρ::Vector` values (any imaginary part in ρ is dropped), where
the number of energy points `xk` is `order * resolution`, rounded to the closest integer.

    dosKPM(μ::MomentaKPM; resolution = 2)

Same as above with momenta `μ` as input.

    dosKPM(h::Hamiltonian; kw...)

Equivalent to `dosKPM(bloch(h); kw...)` for finite hamiltonians (zero dimensional).
"""
dosKPM(h; resolution = 2, kw...) =
    dosKPM(momentaKPM(h; kw...), resolution = resolution)

dosKPM(μ::MomentaKPM; resolution = 2) = real.(densityKPM(μ; resolution = resolution))

"""
    densityKPM(h::AbstractMatrix, A; resolution = 2, kw...)

Compute, using the Kernel Polynomial Method (KPM), the local spectral density of an operator
`A` `ρ_A(ϵ) = ⟨ket|A δ(ϵ-h)|ket⟩/⟨ket|ket⟩` for a given `ket::AbstractVector` and
hamiltonian `h`, or the global spectral density `ρ_A(ϵ) = Tr[A δ(ϵ-h)]` if `ket` is
`missing`. If `ket` is an `AbstractMatrix` it evaluates the trace over the set of kets in
`ket` (see `momentaKPM` and its options `kw` for further details). A tuple of energy points
`xk` and `ρ_A` values is returned where the number of energy points `xk` is `order *
resolution`, rounded to the closest integer.

    densityKPM(momenta::MomentaKPM; resolution = 2)

Same as above with the KPM momenta as input (see `momentaKPM`).

    densityKPM(h::Hamiltonian, A::Hamiltonian; kw...)

Equivalent to `densityKPM(bloch(h), bloch(A); kw...)` for finite Hamiltonians (zero dimensional).
"""
densityKPM(h, A; resolution = 2, kw...) =
    densityKPM(momentaKPM(h, A; kw...); resolution = resolution)

function densityKPM(momenta::MomentaKPM{T}; resolution = 2) where {T}
    checkloaded(:FFTW)
    (center, halfwidth) = momenta.bandbracket
    numpoints = round(Int, length(momenta.mulist) * resolution)
    ρlist = zeros(T, numpoints)
    copyto!(ρlist, momenta.mulist)
    Main.FFTW.r2r!(ρlist, Main.FFTW.REDFT01, 1)  # DCT-III in FFTW
    xk = [cos(π * (k + 0.5) / numpoints) for k in 0:numpoints - 1]
    @. ρlist = center + halfwidth * ρlist / (π * sqrt(1.0 - xk^2))
    @. xk = center + halfwidth * xk
    return xk, ρlist
end

"""
    averageKPM(h::AbstractMatrix, A; kBT = 0, Ef = 0, kw...)

Compute, using the Kernel Polynomial Method (KPM), the thermal expectation value `<A> = Σ_k
f(E_k) <k|A|k> =  ∫dE f(E) Tr [A δ(E-H)] = Tr [A f(H)]` for a given hermitian operator `A`
and a hamiltonian `h` (see `momentaKPM` and its options `kw` for further details).
`f(E)` is the Fermi-Dirac distribution function, `kBT` is the temperature in energy
units and `Ef` the Fermi energy.

    averageKPM(μ::MomentaKPM, A; kBT = 0, Ef = 0)

Same as above with the KPM momenta as input (see `momentaKPM`).

    averageKPM(h::Hamiltonian, A::Hamiltonian; kw...)

Equivalent to `averageKPM(bloch(h), bloch(A); kw...)` for finite Hamiltonians (zero
dimensional).
"""
averageKPM(h, A; kBT = 0, Ef = 0, kw...) = averageKPM(momentaKPM(h, A; kw...); kBT = kBT, Ef = Ef)

function averageKPM(momenta::MomentaKPM{T}; kBT = 0.0, Ef = 0.0) where {T}
    (center, halfwidth) = momenta.bandbracket
    order = length(momenta.mulist) - 1
    # if !iszero(kBT)
    #     @warn "Finite temperature requires numerical evaluation of the integrals"
    #     checkloaded(:QuadGK)
    # end
    average = sum(n -> momenta.mulist[n + 1] * fermicheby(n, Ef, kBT, center, halfwidth), 0:order)
    return average
end

# Pending issue: Unexpected behaviour with center != 0.
function fermicheby(n, Ef, kBT, center, halfwidth)
    kBT´ = kBT / halfwidth;
    Ef´ = (Ef - center) / halfwidth;
    T = typeof(Ef´)
    if kBT´ == 0
        int = n == 0 ? 0.5+asin(Ef´)/π : -2.0*sin(n*acos(Ef´))/(n*π)
    else
        throw(error("Finite temperature not yet implemented"))
        # η = 1e-10
        # int = Main.QuadGK.quadgk(E´ -> _intfermi(n, E´, Ef´, kBT´), -1.0+η, 1.0-η, atol= 1e-10, rtol=1e-10)[1]
    end
    return T(int)
end

_intfermi(n, E´, Ef´, kBT´) = fermifun(E´, Ef´, kBT´) * 2/(π*(1-E´^2)^(1/2)) * chebypol(n, E´) / (1+(n == 0 ? 1 : 0))

fermifun(E´, Ef´, kBT´) = kBT´ == 0 ? (E´<Ef´ ? 1.0 : 0.0) : (1/(1+exp((E´-Ef´)/(kBT´))))

function chebypol(m::Int, x::T) where {T<:Number}
    cheby0 = one(T)
    cheby1 = x
    if m == 0
        chebym = cheby0
    elseif m == 1
        chebym = cheby1
    else
        for i in 2:m
            chebym = 2x * cheby1 - cheby0
            cheby0, cheby1 = cheby1, chebym
        end
    end
    return chebym
end
