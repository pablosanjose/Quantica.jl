#######################################################################
# Kernel Polynomial Method : momenta
#######################################################################
using Base.Threads

struct MomentaKPM{T,B<:Tuple}
    mulist::Vector{T}
    bandbracket::B
end

struct KPMBuilder{A,H<:AbstractMatrix,T,K,B}
    h::H
    A::A
    kets::K
    kets0::K
    kets1::K
    bandbracket::Tuple{B,B}
    order::Int
    mulist::Vector{T}
end

function KPMBuilder(h, A, kets, order, bandrange)
    eh = eltype(eltype(h))
    eA = eltype(eltype(A))
    mulist = zeros(promote_type(eh, eA), order + 1)
    bandbracket = bandbracketKPM(h, bandrange)
    h´ = matrixKPM(h)
    A´ = matrixKPM(A)
    kets´ = Matrix(kets, h)
    builder = KPMBuilder(h´, A´, kets´, similar(kets´), similar(kets´), bandbracket, order, mulist)
    return builder
end

function matrixKPM(h::Hamiltonian{<:Lattice,L}, matrixtype = missing) where {L}
    iszero(L) ||
        throw(ArgumentError("Hamiltonian is defined on an infinite lattice. Reduce it to zero-dimensions with `wrap` or `unitcell`."))
    m = similarmatrix(h, matrixtype)
    return bloch!(m, h)
end

matrixKPM(A::UniformScaling) = A

"""
    momentaKPM(h::Hamiltonian, A = I; kets = randomkets(1), order = 10, bandrange = missing)

Compute the Kernel Polynomial Method (KPM) momenta `μ_n = ∑⟨ket|T_n(h) A|ket⟩`, where the
sum is over `kets` and where `T_n(x)` is the Chebyshev polynomial of order `n`, for a given
`ket`, hamiltonian `h`, and observable `A`.

`kets` can be a `KetModel` or a tuple of `KetModel`s (see `ket` and `randomkets`). A `kets =
randomkets(R, ...)` produces a special `RepeatedKets` object that can be used to compute
momenta by means of a stochastic trace `μ_n = Tr[A T_n(h)] ≈ ∑ₐ⟨a|A T_n(h)|a⟩`, where the
`|a⟩` are the `R` random `kets` of norm 1/√R.

The order of the Chebyshev expansion is `order`. The `bandbrange = (ϵmin, ϵmax)` should
completely encompass the full bandwidth of `hamiltonian`. If `missing` it is computed
automatically using `ArnoldiMethods` (must be loaded).

# Examples

```
julia> h = LatticePresets.cubic() |> hamiltonian(hopping(1)) |> unitcell(region = RegionPresets.sphere(10));

julia> momentaKPM(h, bandrange = (-6,6)).mulist |> length
11
```
"""
function momentaKPM(h::Hamiltonian, A = I; kets = randomkets(1), order = 10, bandrange = missing)
    builder = KPMBuilder(h, A, kets, order, bandrange)
    momenta = momentaKPM(builder)
    return momenta
end

function momentaKPM(b::KPMBuilder)
    pmeter = Progress(b.order, "Computing moments: ")
    addmomentaKPM!(b, pmeter)
    jackson!(b.mulist)
    return MomentaKPM(b.mulist, b.bandbracket)
end

# This iterates bras <psi_n| = <psi_0|AT_n(h) instead of kets (faster CSC multiplication)
# In practice we iterate their conjugate |psi_n> = T_n(h') A'|psi_0>, and do the projection
# onto the start ket, |psi_0>
function addmomentaKPM!(b::KPMBuilder{<:AbstractMatrix,<:AbstractSparseMatrix}, pmeter)
    mulist, kets, kets0, kets1 = b.mulist, b.kets, b.kets0, b.kets1
    h, A, bandbracket = b.h, b.A, b.bandbracket
    order = length(mulist) - 1
    mul!(kets0, A', kets)
    mulscaled!(kets1, h', kets0, bandbracket)
    mulist[1] += proj(kets0, kets)
    mulist[2] += proj(kets1, kets)
    for n in 3:(order+1)
        ProgressMeter.next!(pmeter; showvalues = ())
        iterateKPM!(kets0, h', kets1, bandbracket)
        mulist[n] += proj(kets0, kets)
        kets0, kets1 = kets1, kets0
    end
    return mulist
end

function addmomentaKPM!(b::KPMBuilder{<:UniformScaling, <:AbstractSparseMatrix,T}, pmeter) where {T}
    mulist, kets, kets0, kets1, = b.mulist, b.kets, b.kets0, b.kets1
    h, A, bandbracket = b.h, b.A, b.bandbracket
    order = length(mulist) - 1
    kets0 .= kets
    mulscaled!(kets1, h', kets0, bandbracket)
    mulist[1] += μ0 = proj(kets0, kets0)
    mulist[2] += μ1 = proj(kets1, kets0)
    # This is not used in the currently activated codepath (BLAS mul!), but is needed in the
    # commented out @threads codepath
    thread_buffers = (zeros(T, Threads.nthreads()), zeros(T, Threads.nthreads()))
    for n in 3:2:(order+1)
        ProgressMeter.next!(pmeter; showvalues = ())
        ProgressMeter.next!(pmeter; showvalues = ()) # twice because of 2-step
        proj11, proj10 = iterateKPM!(kets0, h', kets1, bandbracket, thread_buffers)
        mulist[n] += 2 * proj11 - μ0
        n + 1 > order + 1 && break
        mulist[n + 1] += 2 * proj10 - μ1
        kets0, kets1 = kets1, kets0
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

function iterateKPM!(kets0, h´, kets1, (center, halfwidth), buff = ())
    α = 2/halfwidth
    β = 2center/halfwidth
    mul!(kets0, h´, kets1, α, -1)
    @. kets0 = kets0 - β * kets1
    return proj_or_nothing(buff, kets0, kets1)
end

proj_or_nothing(::Tuple{}, kets0, kets1) = nothing
proj_or_nothing(buff, kets0, kets1) = (proj(kets1, kets1), proj(kets1, kets0))

# This is equivalent to tr(ket1'*ket2) for matrices, and ket1'*ket2 for vectors
proj(kets1, kets2) = dot(kets1, kets2)

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

bandrangeKPM(h::Hamiltonian) = bandrangeKPM(matrixKPM(h, flatten))

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
    dosKPM(h::Hamiltonian; resolution = 2, kets = randomkets(1), kw...)

Compute, using the Kernel Polynomial Method (KPM), the density of states per site of
zero-dimensional Hamiltonian `h`, `ρ(ϵ) = ∑⟨ket|δ(ϵ-h)|ket⟩/(NR) ≈ Tr[δ(ϵ-h)]/N` (N is the
number of sites, and the sum is over `R` random `kets`). If `kets` are not `randomkets` but
one or more `KetModel`s (see `ket`), the division by `NR` is ommitted, which results in a
*local* density of states `ρ(ϵ) = ∑⟨ket|δ(ϵ-h)|ket⟩` at sites specified by `kets`.

The result is a tuple of energy points `xk::Vector` and real `ρ::Vector` values (any
residual imaginary part in ρ is dropped), where the number of energy points `xk` is `order *
resolution`, rounded to the closest integer.

    dosKPM(μ::MomentaKPM; resolution = 2)

Same as above with KPM momenta `μ` as input. Equivalent to `densityKPM(μ; kw...)` except
that imaginary parts are dropped.

# See also:
    `momentaKPM`, `densityKPM`, `averageKPM`
"""
function dosKPM(h; resolution = 2, kets = randomkets(1), kw...)
    N = dos_normalization_factor(kets, h)
    dosKPM(momentaKPM(h, I/N; kets = kets, kw...), resolution = resolution)
end

dosKPM(μ::MomentaKPM; resolution = 2) = real.(densityKPM(μ; resolution = resolution))

dos_normalization_factor(kets::StochasticTraceKets, h) = nsites(h)
dos_normalization_factor(kets, h) = 1

"""
    densityKPM(h::Hamiltonian, A; resolution = 2, kets = randomkets(1), kw...)

Compute, using the Kernel Polynomial Method (KPM), the spectral density of `A` for
zero-dimensional Hamiltonian `h`, `ρ_A(ϵ) = ∑⟨ket|A δ(ϵ-h)|ket⟩/R ≈ Tr[Aδ(ϵ-h)]` (the sum is
over `R` random `kets`). `A` can itself be a `Hamiltonian` or a `UniformScaling` `λ*I`. If
`kets` are not `randomkets` but one or more `KetModel`s (see `ket`), the division by `R` is
ommitted, which results in a *local* spectral density `ρ_A(ϵ) = ∑⟨ket|Aδ(ϵ-h)|ket⟩` at sites
specified by `kets`.

The result is a tuple of energy points `xk::Vector` and real `ρ_A::Vector` values (unlike
for `dosKPM`, all imaginary parts in `ρ_A` are preserved), where the number of energy points
`xk` is `order * resolution`, rounded to the closest integer.

    densityKPM(momenta::MomentaKPM; resolution = 2)

Same as above with the KPM momenta as input (see `momentaKPM`).

# See also:
    `dosKPM`, `momentaKPM`, `averageKPM`
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
    averageKPM(h::Hamiltonian, A; kBT = 0, Ef = 0, kw...)

Compute, using the Kernel Polynomial Method (KPM), the thermal expectation value `<A> = Σ_k
f(E_k) <k|A|k> = ∫dE f(E) Tr [A δ(E-H)] = Tr [A f(H)]` for a given hermitian operator `A`
and a zero-dimensional hamiltonian `h` (see `momentaKPM` and its options `kw` for further
details). `f(E)` is the Fermi-Dirac distribution function, `|k⟩` are `h` eigenstates with
energy `E_k`, kBT` is the temperature in energy units and `Ef` the Fermi energy.

    averageKPM(μ::MomentaKPM, A; kBT = 0, Ef = 0)

Same as above with the KPM momenta as input (see `momentaKPM`).

# See also:
    `dosKPM`, `momentaKPM`, `averageKPM`
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
