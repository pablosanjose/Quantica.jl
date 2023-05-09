############################################################################################
# KPMBuilder
#   computes μⁱʲₙ = ⟨ψⱼ|kernel T_n(h)|ψᵢ⟩
#region

struct KPMBuilder{O,M,T,K,H}
    h::H
    kernel::O
    ket::K
    ket1::K
    ket2::K
    bandCH::T   # (center, halfwidth)
    mulist::M
end

function KPMBuilder(h, kernel, ket, bandCH, order)
    mulist = empty_mulist(ket, order)
    ket1, ket2 = similar(ket), similar(ket)
    return KPMBuilder(h, kernel, ket, ket1, ket2, bandCH, mulist)
end

empty_mulist(::Vector{C}, order) where {C<:Complex} =
    zeros(C, order + 1)
empty_mulist(ket::Matrix{C}, order) where {C<:Complex} =
    [zeros(C, size(ket, 2), size(ket, 2)) for _ in 1:(order + 1)]

function momentaKPM(h, ket, bandCH; order = 10, kernel = I)
    pmeter = Progress(order, "Computing moments: ")
    builder = KPMBuilder(h, kernel, ket, bandCH, order)
    mulist = addmomentaKPM!(builder, pmeter)
    jackson!(mulist)
    return mulist
end

# Single step algorithm (non-identity Kernel)
function addmomentaKPM!(b::KPMBuilder{<:AbstractMatrix}, pmeter)
    seed_singlestep_KPM!(b)
    mulist, h, bandCH, kmat, kmat0, kmat1 = b.mulist, b.h, b.bandCH, b.ket, b.ket1, b.ket2
    order = length(mulist) - 1
    for n in 3:(order+1)
        ProgressMeter.next!(pmeter; showvalues = ())
        iterateKPM!(kmat0, h', kmat1, bandCH)
        mulat!(mulist, n, kmat0', kmat, 1, 1)           # μ[n] += kmat0'*kmat1
        kmat0, kmat1 = kmat1, kmat0
    end
    return mulist
end

# Double step algorithm (identity Kernel)
function addmomentaKPM!(b::KPMBuilder{<:UniformScaling}, pmeter)
    μ0, μ1 = seed_doublestep_KPM!(b)
    mulist, h, bandCH, A, kmat0, kmat1 = b.mulist, b.h, b.bandCH, b.kernel, b.ket1, b.ket2
    order = length(mulist) - 1
    for n in 3:2:(order+1)
        ProgressMeter.next!(pmeter; showvalues = ())
        ProgressMeter.next!(pmeter; showvalues = ())    # twice because of 2-step
        iterateKPM!(kmat0, h', kmat1, bandCH)
        mulat!(mulist, n, kmat1', kmat1, 2, 1)
        minusat!(mulist, n, μ0)                         # μ[n] += 2*kmat1'*kmat1 - μ0
        n + 1 > order + 1 && break
        mulat!(mulist, n + 1, kmat1', kmat0, 2, 1)
        minusat!(mulist, n + 1, μ1)                     # μ[n+1] += 2*kmat1'*kmat0 - μ1
        kmat0, kmat1 = kmat1, kmat0
    end
    A.λ ≈ 1 || (mulist .*= A.λ)
    return mulist
end

function seed_singlestep_KPM!(b::KPMBuilder)
    mul!(b.ket1, b.kernel', b.ket)
    mulscaled!(b.ket2, b.h', b.ket1, b.bandCH)
    mulat!(b.mulist, 1, b.ket1', b.ket, 1, 1)
    mulat!(b.mulist, 2, b.ket2', b.ket, 1, 1)
    return nothing
end

function seed_doublestep_KPM!(b::KPMBuilder{<:UniformScaling,Vector{T}}) where {T<:Matrix}
    copy!(b.ket1, b.ket)
    mulscaled!(b.ket2, b.h', b.ket1, b.bandCH)
    μ0, μ1 = copy(b.mulist[1]), copy(b.mulist[2])
    mulat!(b.mulist, 1, b.ket1', b.ket1, 1, 1)
    mulat!(b.mulist, 2, b.ket2', b.ket1, 1, 1)
    μ0 .= b.mulist[1] .- μ0
    μ1 .= b.mulist[2] .- μ1
    return μ0, μ1
end

function seed_doublestep_KPM!(b::KPMBuilder{<:UniformScaling,Vector{T}}) where {T<:Number}
    copy!(b.ket1, b.ket)
    mulscaled!(b.ket2, b.h', b.ket1, b.bandCH)
    b.mulist[1] += μ0 = b.ket1' * b.ket1
    b.mulist[2] += μ1 = b.ket2' * b.ket1
    return μ0, μ1
end

# y = rescaled(h) * x
function mulscaled!(y, h´, x, (center, halfwidth))
    mul!(y, h´, x)
    invhalfwidth = 1/halfwidth
    @. y = (y - center * x) * invhalfwidth
    return y
end

# kmat0 = 2 * rescaled(h) * kmat1 - kmat1
function iterateKPM!(kmat0, h, kmat1, (center, halfwidth))
    α = 2/halfwidth
    β = 2center/halfwidth
    mul!(kmat0, h, kmat1, α, -1)
    @. kmat0 = kmat0 - β * kmat1
    return kmat0
end

mulat!(C::Vector{<:Matrix}, n, A, B, α, β) = mul!(C[n], A, B, α, β)
mulat!(C::Vector{<:Number}, n, A, B, α, β) = (C[n] = α * A * B + β * C[n])

minusat!(A::Vector{<:Matrix}, n, x) = (A[n] .-= x)
minusat!(A::Vector{<:Number}, n, x) = (A[n] -= x)

function jackson!(μ::AbstractVector)
    order = length(μ) - 1
    @inbounds for n in eachindex(μ)
        μ[n] *= ((order - n + 1) * cos(π * n / (order + 1)) +
                sin(π * n / (order + 1)) * cot(π / (order + 1))) / (order + 1)
    end
    return μ
end

#endregion

############################################################################################
# AppliedKPMGreenSolver
#region

struct AppliedKPMGreenSolver{T<:AbstractFloat,M<:Union{Complex{T},AbstractMatrix{Complex{T}}}} <: AppliedGreenSolver
    momenta::Vector{M}
    bandCH::Tuple{T,T}
end

#region ## API ##

moments(s::AppliedKPMGreenSolver) = s.moments

#endregion

#region ## apply ##

function apply(s::GS.KPM,  h::Hamiltonian{<:Any,<:Any,0}, cs::Contacts)
    hmat = h(())
    bandCH = band_ceter_halfwifth(hmat, s.bandrange)
    ket = contact_basis(h, cs)
    momenta = momentaKPM(hmat, ket, bandCH; order = s.order)
    return AppliedKPMGreenSolver(momenta, bandCH)
end

apply(::GS.KPM, h::AbstractHamiltonian, cs::Contacts) =
    argerror("Can only use KPM with bounded non-parametric Hamiltonians")

band_ceter_halfwifth(_, (ϵmin, ϵmax)) = 0.5 * (emin + emax), 0.5 * (emax - emin)

function band_ceter_halfwifth(h, ::Missing)
    @warn "Computing spectrum bounds... Consider using the `bandrange` option for faster performance."
    # (ϵmin, ϵmax) = GS.bandrange_arnoldi(h)
    (emin, emax) = GS.bandrange_arpack(h)
    @warn  "Computed real bandrange = ($emin, $emax)"
    bandCH = 0.5 * (emin + emax), 0.5 * (emax - emin)
    return bandCH
end

#endregion

#endregion top