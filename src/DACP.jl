#######################################################################
# Implementation of the Dual applications of Chebyshev polynomials
# method (DACP) of ref: scipost_202106_00048v2
# to compute effcifiently thousands of central eigenvalues. 
#     Steps:
#         - Exponential (semi-circle) filtering:     
#         - Chebyshev Evolution
#         - Subspace diagonalization
# 
#######################################################################
struct DACPbuilder{H<:AbstractMatrix, P<:AbstractMatrix}
    h::H
    hsquared::H
    emax
    emin
    a
    ψ0::P
    ψ1::P
    ψ2::P
end

function DACPbuilder(h::Hamiltonian, a, ψ)
    (emin, emax) = bandrangeKPM(h)
    round(emin + emax, digits = 4) == 0 ? nothing : @warn(
        "spectrum is not E -> -E symmetric")
    a < emax ? nothing : @warn("a must be smaller than E_max")
    hmat = bloch!(similarmatrix(h, flatten), h) 
    #the flatten looks redundant
    return DACPbuilder(hmat, hmat * hmat, emax, emin, a, ψ, similar(ψ), 
        similar(ψ))
end

struct DACPsubspace{H<:AbstractMatrix, P<:AbstractMatrix}
    h::H
    emax
    emin
    a
    ψe::P
end

"""
given a `h::Union{ParametricHamiltonian, Hamiltonian}` returns
its projection to a lower subspace 𝕃 with eigenvalues inside the energy interval:
(-`a`, `a`). The dimension of the desired subspace must be given as an input `d`
and `a < min(Emax, |Emin|)`

REMARKS:
    - Validity is conditioned to the requirement a << emax
    - in order to accurately span 𝕃, we form a basis by Chebyshev evolution of 
        `ψe` using `n = (l*d-1)/2` states with l>=1 (set by default to l = 1.5).
    - for a given `d`, `a` must be appropriately chosen to ensure that the number of
        eigenstates in [−a, a] is a little less than the dimension of constructed basis, 
        i.e. < 2n + 1. This require prior knowledge of the spectrum (Weak point)
    - Precision can be improved using block evolution of a set of random eigenstates
        and their subsequent cross correlation to build new S and H matrices
"""
function DACP(h::Union{ParametricHamiltonian, Hamiltonian}, a, d)
    @checkloaded(:ArnoldiMethod)
    builder = semicircle_filter(h, a)
    s_mat, h_mat = proj_h_s(builder, d)
    return s_mat, h_mat
end

"""
    semicircle_filter(h::ParametricHamiltonian, a) 
Given an energy cutoff `a::Float64`, which defines the spectral
window (-a, a), a hamiltonian `H`, and a random ket `ψ`
s.t. |ψ⟩ = ∑ᵢcᵢ|ϕᵢ⟩ + ∑ⱼdⱼ|χⱼ⟩ where {|ϕᵢ⟩} and {|χj⟩} are
eigenstates in a subspace of H with energies inside (-a, a), it
returns `ψ_e` such |ψₑ⟩ ≈ ∑ᵢ c'ᵢ|ψᵢ⟩, that is some linear 
combination of eigenstates that live in the 𝕃 subspace by means
of a exponential filter implemented by means of a Chebyshev
polynomials filter.
"""
semicircle_filter(h::ParametricHamiltonian, a) = 
    semicircle_filter(h(), a)
semicircle_filter(h::Hamiltonian, a) =  
    semicircle_filter(flatten(h), a, 
        ket(first(randomkets(1)), flatten(h)))

semicircle_filter(h, a, ψ) = 
    chebyshev_filter(DACPbuilder(h, a, ψ.amplitudes))

"""
    chebyshev!(b::DACPbuilder)
computes the action of a `K`'th order Chebyshev polynomial T_nk(𝔽)
on a random ket `b.ψ0`. 𝔽 = (ℍ^2 - Ec)/E0 is the operator that maps 
the spectral window (`a`², `Emax`²) of ℍ^2 into the interval 
x ∈ (-1, 1) where the T_nk(x) is cosine like. As a result of this 
transformation, the ket components in the (0, `a`²) interval of ℍ²
will be exponentially amplified. 
    -> Iterate over kets to increase performance
"""
function chebyshev_filter(b::DACPbuilder)  
    ψ0, ψ1, ψ2, emax, emin, a, hsquared, h =  
        b.ψ0, b.ψ1, b.ψ2, b.emax, b.emin, b.a, b.hsquared, b.h
    bounds = (emax, a)
    mul_f!(ψ1, hsquared, ψ0, bounds)
    K = Int(ceil(12*emax/a))
    return DACPsubspace(h, emax, emin, a/emax, 
        iterate_chebyshev(K, ψ0, ψ1, ψ2, hsquared, mul_f!, bounds))
end

"""
    proj_h_s()
returns the projected hamiltonian, h, and overlap matrix, s, to 
the 𝕃 subspace.
"""
function proj_h_s(b, d; l = 1.5)
    h, ψe, bounds = b.h, b.ψe, (b.emax, b.emin, b.a)
    n = Int(ceil((l*d - 1)/2))
    dim = Int(2*(2n+1))
    ψh, ψ1, ψ2 = similar(ψe), similar(ψe), similar(ψe)
    s_h_arrs = iterate_chebyshev(dim, ψh, ψe, ψ1, ψ2, h, mul_g!, bounds)
    return [arr_to_mat(arr, dim) for arr in s_h_arrs]
end

"""
    `iterate_chebyshev(K, ψ0, ψ1, ψ2, hsquared, mul_operator, bounds)`

returns the action TK(𝔽)|ψ0⟩ on a random vector `ψ0` see: `semicircle_filter()`

    `iterate_chebyshev(K, ψh, ψ0, ψ1, ψ2, h, mul_operator, bounds)`

returns two arrays smat and hmat which contain the projections of ⟨ψe|Tk(G)|ψe⟩ and
⟨ψe|ℍ Tk(G)|ψe⟩ = ⟨ψh|Tk(G)|ψe⟩. Note that |ψ0⟩ = |ψe⟩ is a vector exponentially 
filtered within the 𝕃 subspace see: `semicircle_filter()`
"""
function iterate_chebyshev(K, ψ0, ψ1, ψ2, hsquared, mul_operator, bounds)
    pmeter = Progress(K, "Computing $(K) order Chebyshev pol...")
    for i in 3:K
        ProgressMeter.next!(pmeter; showvalues = ())
        mul_operator(ψ2, hsquared, ψ1, bounds)
        ψ2 *= 2
        ψ2 -= ψ0 
        ψ0 = copy(ψ1)
        ψ1 = copy(ψ2)
    end
    nm = norm(ψ1)
    return ψ1./nm #normalized
end

function iterate_chebyshev(K, ψh, ψ0, ψ1, ψ2, h, mul_operator, bounds)
    pmeter = Progress(K, "Computing S and H matrices...")
    #init
    ψe = similar(ψ0)
    copy!(ψe, ψ0)
    s_ar = zeros(ComplexF64, Int(K))
    h_ar = similar(s_ar)
    mul!(ψh, h, ψ0)
    mul_operator(ψ1, h, ψ0, bounds)
    s_ar[1], s_ar[2] = dot(ψ0, ψ0), dot(ψh, ψ0)
    h_ar[1], h_ar[2] = dot(ψ0, ψ1), dot(ψh, ψ1)
    for i in 3:Int(K)
        ProgressMeter.next!(pmeter; showvalues = ())       
        mul_operator(ψ2, h, ψ1, bounds)
        ψ2 .*= 2
        ψ2 .-= ψ0
        s_ar[i] = dot(ψe, ψ2)
        h_ar[i] = dot(ψh, ψ2)
        ψ0 = copy(ψ1)
        ψ1 = copy(ψ2)
    end
    return s_ar, h_ar
end
"""
returns the action of the operator 𝔽 on a state x
"""
function mul_f!(y, hsquared, x, (emax, a))
    mul!(y, hsquared, x)
    ec = (emax^2 + a^2)/2
    e0 = (emax^2 − a^2)/2
    @. y = (y - ec * x)/e0
    return y
end

"""
returns the action of the operator 𝔾 on a state x
"""
function mul_g!(y, h, x, (emax, emin, a))
    # emax += 1e-5
    # emin -= 1e-5
    mul!(y, h, x)
    ec = (emin + emax)/2
    e0 = (-emin + emax)/2
    @. y = (y - ec * x)/e0
end

"compose the S or H matrix out two lists of ⟨ψe|Tk(𝔾|ψe⟩ and ⟨ψe|ℍTk(𝔾|ψe⟩
with k = 1,..., 4n+2, see:
    `iterate_chebyshev(K, ψh, ψ0, ψ1, ψ2, h, mul_operator, bounds)`"
function arr_to_mat(arr::AbstractArray{T}, dim) where {T}
    println("dimension of the constructed basis is $(dim/2)")
    mat = zeros(T, Int(dim/2), Int(dim/2))
    for i in 0:Int(dim/2)-1
        for j in 0:Int(dim/2)-1
            mat[j+1, i+1] = (arr[Int(i+j+1)] + arr[Int(abs(i-j)+1)])/2
        end
    end
    # ishermitian(mat) ? nothing : @warn "mat is non hermitian"
    return Hermitian(mat) #Hermitian(mat) # caution
end
"""
Diagonaliser
"""
function DACPdiagonaliser(h::Hamiltonian, a, d) 
    smat, hmat = DACP(h, a, d)
    return DACPdiagonaliser(smat, hmat)
end

function DACPdiagonaliser(s::AbstractMatrix{T}, h::AbstractMatrix{T}; 
    threshold = 1e-12) where {T}
    F = eigen(s)
    lowe_filter = findall(x -> abs(x) > threshold, F.values)
    V = F.vectors[:, lowe_filter]
    Λsq = diagm(F.values[lowe_filter])
    U = V * Λsq
    h_red = U' * h * U
    println(size(h_red))
    println(size(h))
    eigvals = eigen(h_red).values
end