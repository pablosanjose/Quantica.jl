############################################################################################
# Implementation of the Dual applications of Chebyshev polynomials
# method (DACP) of ref: scipost_202106_00048v3
# to compute efficiently thousands of central eigenvalues. 
# Steps:
#    1. Exponential (semi_circle) filtering of n-random vectors
#   (2a) Estimation of subspace dimension using the KPM
#    2(b). Chebyshev Evolution
#    3. Subspace diagonalization
# To do:  
#    0. Use SVectors structs. Body and return type instability in the proj_h_s method ->
#        store_base = true
#    1. Subspace is smaller than expected although there is oversampling 
#          - negligible amplitudes of the filtered vectors in (-a, -a+ϵ) and (a-ϵ, a)
#            ϵ -> 0
#    2. Prior knowledge of degeneracies is required -> Adaptive DACP
############################################################################################
using NumericalIntegration

# Builders and structs

struct DACPbuilder{H<:SparseMatrixCSC{ComplexF64, Int64}, N<:Float64, 
    V<:Vector{Matrix{ComplexF64}}, V1<:Matrix{ComplexF64}}
    h::H
    hsquared::H
    emax::N
    emin::N
    a::N
    ψ0::V
    ψ1::V1
end

function DACPbuilder(h::Hamiltonian, a, ψ; eps = 1e-4)
    (emin, emax) = bandrangeKPM(h, quiet = true)
    emin *= (1 + eps)
    emax *= (1 + eps)
    round(emin + emax, digits = 4) == 0 ? nothing : @warn(
        "spectrum is not E -> -E symmetric")
    a < emax ? nothing : @warn("a must be smaller than E_max")
    hmat = bloch!(similarmatrix(h, flatten), h)
    return DACPbuilder(hmat, hmat * hmat, emax, emin, a, ψ, similar(ψ[1]))
end

struct DACPsubspace{H<:SparseMatrixCSC{ComplexF64, Int64}, N<:Float64}
    h::H
    emax::N
    emin::N
    a::N
    ψe::Vector{Matrix{ComplexF64}}
end

"""
given a `h::Union{ParametricHamiltonian, Hamiltonian}` returns
the projection of `h` and of the overlap matrix, `s` to a lower subspace 𝕃 with eigenvalues
inside the energy interval: (-`a`, `a`). 
The dimension of the desired subspace must be given as an input `d` and 
`a < min(Emax, |Emin|)`

REMARKS:
    - Validity is conditioned to the requirement `a << emax`
    - in order to accurately span 𝕃, we form a basis by Chebyshev evolution of 
        `ψe` using `n = (l*d-1)/2` states with l>=1 (set by default to l = 1.5).
    - for a given `d`, `a` must be appropriately chosen to ensure that the number of
        eigenstates in `[−a, a]` is a little less than the dimension of constructed basis, 
        i.e. < 2n + 1 (overestimation). This requires prior knowledge about the spectrum
        edges. However, the subspace dimension, `d`, can be efficiently determined using 
        the KPM method. 
"""
proj_DACP(h::Union{ParametricHamiltonian, Hamiltonian}, a;
    store_basis= true::Bool, d = missing::Union{Missing, Real}, maxdeg = 1) =
    proj_DACP(h::Union{ParametricHamiltonian, Hamiltonian}, a, Val(store_basis);
        d = missing::Union{Missing, Real}, maxdeg = maxdeg) 


proj_DACP(h::Union{ParametricHamiltonian, Hamiltonian}, a, store_basis::Val{true}; 
    d = missing::Union{Missing, Real}, maxdeg = 1) = _proj_DACP(h, a, d, store_basis; maxdeg = maxdeg)


proj_DACP(h::Union{ParametricHamiltonian, Hamiltonian}, a, store_basis::Val{false};
    d = missing::Union{Missing, Real}, maxdeg = 1) = _proj_DACP(h, a, d, store_basis; maxdeg = maxdeg)

function _proj_DACP(h, a, d, store_basis; maxdeg = 1)
    println(maxdeg)
    builder = semicircle_filter(h, a; numkets = maxdeg)
    return proj_h_s(builder, h, d, store_basis)
end


############################################################################################
#    1. Exponential (semi_circle) filtering of n-random vectors
############################################################################################
"""
    `semicircle_filter(h::ParametricHamiltonian, a)` 
Given an energy cutoff `a::Float64`, which defines the spectral window (-a, a), a 
hamiltonian `H`, and a random ket `ψ` s.t. |ψ⟩ = ∑ᵢcᵢ|ϕᵢ⟩ + ∑ⱼdⱼ|χⱼ⟩ where {|ϕᵢ⟩} and {|χj⟩}
are eigenstates in a subspace of H with energies inside (-a, a), it returns `ψ_e` s.t.
|ψₑ⟩ ≈ ∑ᵢ c'ᵢ|ψᵢ⟩, that is some linear combination of eigenstates that live in the 𝕃 
subspace by means of an exponential filter implemented by means of a Chebyshev iteration
"""
semicircle_filter(h::ParametricHamiltonian, a; kw...) = semicircle_filter(h(), a; kw...)

semicircle_filter(h::Hamiltonian, a; numkets = 1) =  
    semicircle_filter(flatten(h), a, random_ket_generator(h, numkets))

semicircle_filter(h, a, ψ) = chebyshev_filter(DACPbuilder(h, a, ψ))

random_ket_generator(h, numkets) = 
    [ket(first(randomkets(1)), flatten(h)).amplitudes for i in 1:numkets]

"""
    `chebyshev!(b::DACPbuilder)`
computes the action of a `K`'th order Chebyshev polynomial T_nk(𝔽) on a random ket `b.ψ0`. 
𝔽 = (ℍ^2 - Ec)/E0 is the operator that maps the spectral window (`a`², `Emax`²) of ℍ^2 into
the interval x ∈ (-1, 1) where the T_nk(x) is cosine like. As a result of this 
transformation, the ket components in the (0, `a`²) interval of ℍ² will be exponentially
amplified. 
    -> Iterate over kets to increase performance
eps != 0 adds performs the exponential filtering in a slighlty larger interval
(this is to avoid subsampling at the edges of the spectrum) - desabled
"""
function chebyshev_filter(b::DACPbuilder, eps = 0.)
    checkloaded(:ArnoldiMethod)
    ψ0, ψ1, emax, emin, a, h =  
        b.ψ0, b.ψ1, b.emax, b.emin, b.a, b.h
    a += (a/emax * eps)
    bounds = (maximum([abs(emax), abs(emin)]), a)
    K = Int(ceil(12*emax/a))
    return DACPsubspace(h, emax, emin, a/emax, 
        iterate_chebyshev(K, ψ0, ψ1, b.hsquared, bounds))
end

"""
    `iterate_chebyshev(K, ψ0::Vector{Matrix{T}}, ψ1, hsquared, bounds; thrs = 1e-12)`
uses a single Chebyshev filtering loop to compute a block of exponentially filtered linearly 
independent randomkets with dimension `numkets` passed as a kwarg.

    `iterate_chebyshev(K, ψ0::Matrix{ComplexF64}, ψ1, hsquared, bounds)`
returns the action TK(𝔽)|ψ0⟩ on a random vector `ψ0`, where `K` is the cutoff of the Cheby
iteration and `ψ1` an auxiliary ket.
    see: `semicircle_filter()`
"""                         

function iterate_chebyshev(K, ψ0::Matrix{ComplexF64}, ψ1, hsquared, bounds) 
    pmeter = Progress(K, "Computing $(K) order Chebyshev pol...")
    mul_f!(ψ1, hsquared, ψ0, bounds)
    for i in 3:Int(K*2)
        ProgressMeter.next!(pmeter; showvalues = ())
        iterateDACP_f!(ψ0, hsquared, ψ1, bounds)
        ψ0, ψ1 = ψ1, ψ0
    end
    return normalize!(ψ0)
end

"""

    `iterate_chebyshev(K, ψ0::Vector{Matrix{ComplexF64}}, ψ1, hsquared, bounds)`
returns the action TK(𝔽)|ψ0⟩ on a block of random vectors `ψ0` see: `semicircle_filter()`.
numkets Cheby loops are required. 
"""
iterate_chebyshev(K, ψ0::Vector{Matrix{ComplexF64}}, ψ1, hsquared, bounds) =
    [iterate_chebyshev(K, ψ0[i], ψ1, hsquared, bounds) for i in 1:length(ψ0)]

############################################################################################
#   (2a) Estimation of subspace dimension using the KPM
############################################################################################
""""
    `subspace_dimension(b)`
performs the numerical integration of the `dos` inside the interval `(-a,a)`. `dos` is 
computed using the KPM, see `dosKPM` with a number of momenta `N` enought to resolve the 
interval `(-a, a)`, i.e. `N = bandwidth/a` Arguments: `b::DACPbuilder`
 """
function subspace_dimension(h, b)
    a, emax, emin = b.a, b.emax, b.emin
    # @warning "If the subspace dimension, `d`, is known set `d = d` as a kw argument in
    #     `DACP()` or `DACPdiagonaliser()` for a speed boost"
    checkloaded(:NumericalIntegration)
    order = Int(ceil(10*(emax - emin)/a))
    es, dos = dosKPM(flatten(h), order = order, resolution = 2, bandrange = (b.emin, b.emax))
    indices = findall(x -> x <= a, abs.(es))
    subspace_dim = Int(ceil(abs(integrate(es[indices], dos[indices])*size(h,1))))
    println("subspace dimension: ", subspace_dim)
    return subspace_dim
end

############################################################################################
#    2. Chebyshev Evolution
############################################################################################
"""
computes the reduced h and s matrices
"""
proj_h_s(builder::DACPsubspace, h, d::Missing, store_basis; kw...) =
    proj_h_s(builder::DACPsubspace, subspace_dimension(h, builder), store_basis; kw...)

proj_h_s(builder::DACPsubspace, h, d::Real, store_basis; kw...) =
    proj_h_s(builder::DACPsubspace, d, store_basis; kw...)


proj_h_s(b, d, store_basis; kw...) = 
    proj_h_s(b.h, b.ψe, (b.emax, b.emin, b.a), d, store_basis; kw...)


function proj_h_s(h, ψe, bounds, d, store_basis::Val{true}; kw...)
    l = 1.5
    n = Int(ceil((l * d - 1)/2))
    Kp = n
    ar = bounds[3]/abs(bounds[1])
    indices = Int.(floor.(vcat([[m*π/ar-1, m*π/ar] for m in 1:Kp]...)))
    pushfirst!(indices, 1) # the 𝕀 in {𝕀, T_{k-1}, T_k...}

    basis = chebyshev_basis(indices, ψe, h, bounds)
    smat = zeros(ComplexF64, size(basis, 2), size(basis, 2))
    hmat = similar(smat)
    
    mul!(smat, basis', basis)
    mul!(hmat, basis', h * basis)
    return smat, hmat, basis
end
 
function chebyshev_basis(indices, ψ0::Vector{Matrix{T}}, h, bounds) where {T}
    pmeter = Progress(length(indices), 
        "Computing $(length(indices)+1) order Chebyshev pol...")
    basis = zeros(T, length(ψ0[1]), Int(ceil(length(ψ0)*length(indices))))
    for it in 1:length(ψ0)
        ψi = copy(ψ0[it])
        count = 0
        for i in 1:indices[end]
            ProgressMeter.next!(pmeter; showvalues = ())
            ψ0[it], ψi = _chebyshev_loop!(ψ0[it], ψi, h, bounds, i) # Evolution loop
            if i in indices
                count += 1
                basis[:, Int(count+(it-1)*length(indices))] = ψ0[it]./norm(ψ0[it])
            else nothing end
        end
    end
    return basis
end

"returns the projected S and H matrices without the requirement to store the basis.
Only ⟨ψE|Tₖ|ψE⟩ and ⟨ψE|ℍTₖ|ψE⟩ are stored at 4 different \"instants\" for each k_m"

function proj_h_s(h, ψe, bounds, d, store_basis::Val{false}; kw...)
    l = 1.5
    n = Int(ceil((l * d - 1)/2))
    Kp = 2n #the factor two is required
    ar = bounds[3]/abs(bounds[1])
    indices = Int.(floor.(vcat([[m*π/ar-2, m*π/ar-1, m*π/ar, m*π/ar+1] for m in 1:Kp]...)))
    pushfirst!(indices, 1) # the 𝕀 in {𝕀, T_{k-1}, T_k...}
    return chebyshev_proj(ar, indices, ψe, h, bounds)
end

function chebyshev_proj(ar, indices, ψ0::Vector{Matrix{ComplexF64}}, h, bounds)
    Kp = length(indices)
    ψh = similar(ψ0[1])
    aux_vec = zeros(ComplexF64, Kp, 2)
    pmeter = Progress(Kp, "Computing $(Kp+1) order Chebyshev pol...")
    for it in 1:1#length(ψ0)
        count = 0
        ψi = copy(ψ0[it])
        ψe = copy(ψ0[it]) #redundant?
        mul!(ψh, h, ψi)
        for i in 1:indices[end]
            ProgressMeter.next!(pmeter; showvalues = ())
            ψ0[it], ψi = _chebyshev_loop!(ψ0[it], ψi, h, bounds, i) # Evolution loop
            if i in indices #storing ⟨ψE|Tₖ|ψE⟩, ⟨ψE|ℍTₖ|ψE⟩ for those T_k with k ∈ indices
                count +=1
                aux_vec[count,1] = dot(ψe, ψ0[it])#/(norm(ψ0[it]))
                aux_vec[count,2] = dot(ψh, ψ0[it])#/(norm(ψ0[it]))
            else nothing end
        end
    end
    return build_matrices(aux_vec, indices, ar)
end
 

function build_matrices(v, indices, ar)
    dim = Int64((length(v[:,1])-1)/4 +1)
    smat = zeros(ComplexF64, dim, dim)
    hmat = similar(smat) 
    println(indices)
    for j in 1:dim
        for i in 1:dim
            println("i: ", i, "j: ", j)
            println(ijselector(ar, i, j))
            indexp, indexm = [findall(x -> x == ijselector(ar, i, j)[ite], indices)[1] for ite in 1:2]
            smat[i,j] = 1/2 * (v[indexp, 1] + v[indexm, 1])
            hmat[i,j] = 1/2 * (v[indexp, 2] + v[indexm, 2])
        end
    end
    return smat, hmat     
end
"""
aux_vec is organized s.t. {1, Tk_1-2, Tk_1-1, Tk_1, Tk_1+1, Tk_2-1}, given two indices 
`(i,j)`, `ijselector(i, j)` returns  a couple of indices `x+y, abs(x-y)` for the calculation
of s and h see eq (22) of scipost_202106_00048v3.
"""
function ijselector(ar, i, j)
    if i % 2 == 0 && j % 2 == 0
        k_p = π/ar * (i+j)÷2 - 2
        k_m = π/ar * abs((i-j)÷2)
    elseif i % 2 == 1 && j % 2 == 0 
        k_p = π/ar * (i+j-1)÷2 -1
        k_m = abs(π/ar * (i-j-1)÷2 + 1)
    elseif i % 2 == 0 && j % 2 == 1
        k_p = π/ar *(i+j-1)÷2 -1
        k_m = abs(π/ar * (i-j+1)÷2 -1)
    else
        k_p = π/ar *(i+j-2)÷2
        k_m = π/ar * abs((i-j)÷2)
    end
    k_p = ifelse(k_p <= 0.5, 1., k_p)
    k_m = ifelse(k_m <= 0.5, 1., k_m)
    return [Int(floor(k_p)), Int(floor(k_m))]
end

function _chebyshev_loop!(ψ0, ψi, h, bounds, i)
    if i == 1
        copy!(ψ0, ψi)
    elseif i == 2
        mul_g!(ψ0, h, ψi, bounds)
    else
        iterateDACP_g!(ψi, h, ψ0, bounds)
        ψ0, ψi = ψi, ψ0
    end
    return ψ0, ψi
end

"""
returns the action of the operator `𝔽` on a state `x`
"""
function mul_f!(y, mat, x, (emax, a))
    ec = (emax^2 + a^2)/2
    e0 = (emax^2 − a^2)/2
    mul!(y, mat, x)
    @. y = (y - ec * x)/e0
end

"""
returns the action of the operator `𝔾` on a state `x`
"""
function mul_g!(y, mat, x, (emax, emin, a))
    ec = (emin + emax)/2
    e0 = (-emin + emax)/2
    mul!(y, mat, x)
    @. y = (y - ec * x)/e0
end

"""
action of the chebyshev iteration of the operator `𝔽` on a state `x`
"""
function iterateDACP_f!(y, mat, x, (emax, a))
    ec = (emax^2 + a^2)/2
    e0 = (emax^2 − a^2)/2
    mul!(y, mat, x, 2/e0, -1) 
    @. y = y - x * 2ec/e0
end

"""
action of the chebyshev iteration of the operator `𝔾` on a state `x`
"""
function iterateDACP_g!(y, mat, x, (emax, emin, a))
    ec = (emin + emax)/2
    e0 = (-emin + emax)/2
    mul!(y, mat, x, 2/e0, -1) 
    @. y = y - x * 2ec/e0
end

############################################################################################
#    3. Subspace diagonalization
############################################################################################

"""
Diagonaliser, uses a Generalize Schur Decomposition (QZ) to solve the GEP
so we are taking care of possible degeneracies
"""
function DACPdiagonaliser(h::Hamiltonian, a; store_basis = true, maxdeg = 1, 
    d = missing::Union{Missing, Real})
    smat, hmat = proj_DACP(h, a, Val(store_basis), maxdeg = maxdeg, d = d)[1:2]
    return DACPdiagonaliser(hmat, smat)
end

# DACPdiagonaliser(h, s) = eigen(h, s).values

"""
    `DACPdiagonaliser(h::AbstractMatrix{T}, s::AbstractMatrix{T}; threshold = 1e-12)`
    solves the GEP problem defined by the hamiltonian matrix `h` and the overlap matrix `s`
which are built using an overcomplete basis corresponding to a number `numkets` of Chebyshev
evolutions. 
    It returns the eigendescomposition (eigenvalues and eigenvectors) of the target subspace
of a hamiltonian, `h`. Note that we throw all linear dependencies by means of an SVD of the
overlap matrix. We select the subspace corresponding to all singular values up to 
`tolerance = 1e-12`
"""
function DACPdiagonaliser(h::AbstractMatrix{T}, s::AbstractMatrix{T}; 
    tolerance = 1e-12) where {T}
    F = eigen(s)
    lowe_filter = findall(x -> abs(x) > tolerance, F.values)
    V = F.vectors[:, lowe_filter]
    Λsq = diagm(sqrt.(1 ./ Complex.(F.values[lowe_filter])))
    U = V * Λsq
    h_red = U' * h * U
    println("size reduced subspace up to tol = $(1e-12): ", size(h_red, 1))
    return eigen(h_red).values
end