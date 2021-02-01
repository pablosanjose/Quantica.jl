#######################################################################
# Green's function
#######################################################################
abstract type AbstractGreensSolver end

struct GreensFunction{S<:AbstractGreensSolver,L,B<:NTuple{L,Union{Int,Missing}},H<:Hamiltonian}
    solver::S
    h::H
    boundaries::B
end

"""
    greens(h::Hamiltonian, solveobject; boundaries::NTuple{L,Integer} = missing)

Construct the Green's function `g::GreensFunction` of `L`-dimensional Hamiltonian `h` using
the provided `solveobject`. Currently valid `solveobject`s are

- the `Bandstructure` of `h` (for an unbounded `h` or an `Hamiltonian{<:Superlattice}}`)
- the `Spectrum` of `h` (for a bounded `h`)
- `SingleShot1D(; direct = false)` (single-shot generalized [or direct if `direct = true`] eigenvalue approach for 1D Hamiltonians)

If a `boundaries = (n₁, n₂, ...)` is provided, a reflecting boundary is assumed for each
non-missing `nᵢ` perpendicular to Bravais vector `i` at a cell distance `nᵢ` from the
origin.

    h |> greens(h -> solveobject(h), args...)

Curried form equivalent to the above, giving `greens(h, solveobject(h), args...)`.

    g(ω, cells::Pair)

From a constructed `g::GreensFunction`, obtain the retarded Green's function matrix at
frequency `ω` between unit cells `src` and `dst` by calling `g(ω, src => dst)`, where `src,
dst` are `::NTuple{L,Int}` or `SVector{L,Int}`. If not provided, `cells` default to
`(1, 1, ...) => (1, 1, ...)`.

    g(ω, missing)

If allowed by the used `solveobject`, build an efficient function `cells -> g(ω, cells)`
that can produce the Greens function between different cells at fixed `ω` without repeating
cell-independent parts of the computation.

# Examples

```jldoctest
julia> g = LatticePresets.square() |> hamiltonian(hopping(-1)) |> greens(bandstructure(resolution = 17))
GreensFunction{Bandstructure}: Green's function from a 2D bandstructure
  Matrix size    : 1 × 1
  Element type   : scalar (Complex{Float64})
  Band simplices : 512

julia> g(0.2)
1×1 Array{Complex{Float64},2}:
 6.663377810046025 - 24.472789025006396im

julia> m = similarmatrix(g); g(m, 0.2)
1×1 Array{Complex{Float64},2}:
 6.663377810046025 - 24.472789025006396im
```

# See also
    `greens!`, `SingleShot1D`
"""
greens(h::Hamiltonian{<:Any,L}, solverobject; boundaries = filltuple(missing, Val(L))) where {L} =
    GreensFunction(greensolver(solverobject, h), h, boundaries)
greens(solver::Function, args...; kw...) = h -> greens(h, solver(h), args...; kw...)

# solver fallback
greensolver(s::AbstractGreensSolver) = s

# call API fallback
(g::GreensFunction)(ω, cells = default_cells(g)) = greens!(similarmatrix(g), g, ω, cells)

similarmatrix(g::GreensFunction, type = Matrix{blocktype(g.h)}) = similarmatrix(g.h, type)

greens!(matrix, g, ω, cells) = greens!(matrix, g, ω, sanitize_cells(cells, g))

default_cells(::GreensFunction{S,L}) where {S,L} = filltuple(1, Val(L)) => filltuple(1, Val(L))

sanitize_cells((cell0, cell1)::Pair{<:Integer,<:Integer}, ::GreensFunction{S,1}) where {S} =
    SA[cell0] => SA[cell1]
sanitize_cells((cell0, cell1)::Pair{<:NTuple{L,Integer},<:NTuple{L,Integer}}, ::GreensFunction{S,L}) where {S,L} =
    SVector(cell0) => SVector(cell1)
sanitize_cells(cells, g::GreensFunction{S,L}) where {S,L} =
    throw(ArgumentError("Cells should be of the form `cᵢ => cⱼ`, with each `c` an `NTuple{$L,Integer}`"))

const SVectorPair{L} = Pair{SVector{L,Int},SVector{L,Int}}

#######################################################################
# SingleShot1DGreensSolver
#######################################################################
"""
    SingleShot1D(; direct = false)

Return a Greens function solver using the generalized eigenvalue approach, whereby given the
energy `ω`, the eigenmodes of the infinite 1D Hamiltonian, and the corresponding infinite
and semi-infinite Greens function can be computed by solving the generalized eigenvalue
equation

    A⋅φχ = λ B⋅φχ
    A = [0 I; V ω-H0]
    B = [I 0; 0 V']

This is the matrix form of the problem `λ(ω-H0)φ - Vφ - λ²V'φ = 0`, where `φχ = [φ; λφ]`,
and `φ` are `ω`-energy eigenmodes, with (possibly complex) momentum `q`, and eigenvalues are
`λ = exp(-iqa₀)`. The algorithm assumes the Hamiltonian has only `dn = (0,)` and `dn = (±1,
)` Bloch harmonics (`H0`, `V` and `V'`), so its unit cell will be enlarged before applying
the solver if needed. Bound states in the spectrum will yield delta functions in the density
of states that can be resolved by adding a broadening in the form of a small positive
imaginary part to `ω`.

To avoid singular solutions `λ=0,∞`, the nullspace of `V` is projected out of the problem.
This produces a new `A´` and `B´` with reduced dimensions. `B´` can often be inverted,
turning this into a standard eigenvalue problem, which is slightly faster to solve. This is
achieved with `direct = true`. However, `B´` sometimes is still non-invertible for some
values of `ω`. In this case use `direct = false` (the default).

# Examples
```jldoctest
julia> using LinearAlgebra

julia> h = LP.honeycomb() |> hamiltonian(hopping(1)) |> unitcell((1,-1), (10,10)) |> Quantica.wrap(2);

julia> g = greens(h, SingleShot1D(), boundaries = (0,))
GreensFunction{SingleShot1DGreensSolver}: Green's function using the single-shot 1D method
  Matrix size    : 40 × 40
  Reduced size   : 20 × 20
  Element type   : scalar (ComplexF64)
  Boundaries     : (0,)

julia> tr(g(0.3))
-32.193416068730684 - 3.4399800712973074im
```

# See also
    `greens`
"""
struct SingleShot1D
    invert_B::Bool
    cutoff::Int
end

SingleShot1D(; direct = false, cutoff = 1) = SingleShot1D(direct, cutoff)

struct SingleShot1DTemporaries{T}
    b2s::Matrix{T}      # size = dim_b, 2dim_s
    φ::Matrix{T}        # size = dim_H0, 2dim_s
    χ::Matrix{T}        # size = dim_H0, 2dim_s
    ss1::Matrix{T}      # size = dim_s, dim_s
    ss2::Matrix{T}      # size = dim_s, dim_s
    Hs1::Matrix{T}      # size = dim_H0, dim_s
    Hs2::Matrix{T}      # size = dim_H0, dim_s
    HH0::Matrix{T}      # size = dim_H0, dim_H0
    HH1::Matrix{T}      # size = dim_H0, dim_H0
    HH2::Matrix{T}      # size = dim_H0, dim_H0
    HH3::Matrix{T}      # size = dim_H0, dim_H0
    vH::Vector{T}       # size = dim_H0
    v2s1::Vector{Int}   # size = 2dim_s
    v2s2::Vector{Int}   # size = 2dim_s
end

function SingleShot1DTemporaries{T}(dim_H, dim_s, dim_b) where {T}
    b2s = Matrix{T}(undef, dim_b, 2dim_s)
    φ   = Matrix{T}(undef, dim_H, 2dim_s)
    χ   = Matrix{T}(undef, dim_H, 2dim_s)
    ss1 = Matrix{T}(undef, dim_s, dim_s)
    ss2 = Matrix{T}(undef, dim_s, dim_s)
    Hs1 = Matrix{T}(undef, dim_H, dim_s)
    Hs2 = Matrix{T}(undef, dim_H, dim_s)
    HH0 = Matrix{T}(undef, dim_H, dim_H)
    HH1 = Matrix{T}(undef, dim_H, dim_H)
    HH2 = Matrix{T}(undef, dim_H, dim_H)
    HH3 = Matrix{T}(undef, dim_H, dim_H)
    vH   = Vector{T}(undef, dim_H)
    v2s1 = Vector{Int}(undef, 2dim_s)
    v2s2 = Vector{Int}(undef, 2dim_s)
    return  SingleShot1DTemporaries{T}(b2s, φ, χ, ss1, ss2, Hs1, Hs2, HH0, HH1, HH2, HH3, vH, v2s1, v2s2)
end

struct SingleShot1DGreensSolver{T<:Complex,R<:Real,H<:Hessenberg{T}} <: AbstractGreensSolver
    invert_B::Bool
    λ2min::R
    A::Matrix{T}
    B::Matrix{T}
    minusH0::SparseMatrixCSC{T,Int}
    V::SparseMatrixCSC{T,Int}
    Pb::Matrix{T}        # size = dim_b, dim_H0
    Ps::Matrix{T}        # size = dim_s, dim_H0
    Ps´::Matrix{T}       # size = dim_s´, dim_H0
    H0ss::Matrix{T}
    H0bs::Matrix{T}
    Vss::Matrix{T}
    Vbs::Matrix{T}
    hessbb::H
    velocities::Vector{R}   # size = 2dim_s = 2num_modes
    maxdn::Int
    temps::SingleShot1DTemporaries{T}
end

function Base.show(io::IO, g::GreensFunction{<:SingleShot1DGreensSolver})
    print(io, summary(g), "\n",
"  Matrix size    : $(size(g.solver.V, 1)) × $(size(g.solver.V, 2))
  Reduced size   : $(size(g.solver.Ps, 1)) × $(size(g.solver.Ps, 1))
  Element type   : $(displayelements(g.h))
  Boundaries     : $(g.boundaries)")
end

Base.summary(g::GreensFunction{<:SingleShot1DGreensSolver}) =
    "GreensFunction{SingleShot1DGreensSolver}: Green's function using the single-shot 1D method"

hasbulk(gs::SingleShot1DGreensSolver) = !iszero(size(gs.Pb, 1))

function greensolver(s::SingleShot1D, h)
    latdim(h) == 1 || throw(ArgumentError("Cannot use a SingleShot1D Green function solver with an $(latdim(h))-dimensional Hamiltonian"))
    return SingleShot1DGreensSolver(h, s.invert_B, s.cutoff)
end

## Preparation

function SingleShot1DGreensSolver(h::Hamiltonian, invert_B, cutoff)
    latdim(h) == 1 || throw(ArgumentError("Cannot use a SingleShot1D Green function solver with an $(latdim(h))-dimensional Hamiltonian"))
    maxdn = max(1, maximum(har -> abs(first(har.dn)), h.harmonics))
    H = flatten(maxdn == 1 ? h : unitcell(h, (maxdn,)))
    T = complex(blockeltype(H))
    λ2min = cutoff^2 * eps(real(T))
    H0, V, V´ = H[(0,)], H[(1,)], H[(-1,)]
    Pb, Ps, Ps´ = bulk_surface_projectors(H0, V, V´, λ2min)
    H0ss = Ps * H0 * Ps'
    H0bs = Pb * H0 * Ps'
    Vss  = Ps * V * Ps'
    Vbs  = Pb * V * Ps'
    hessbb = hessenberg!(Pb * (-H0) * Pb')
    dim_s = size(Ps, 1)
    A = zeros(T, 2dim_s, 2dim_s)
    B = zeros(T, 2dim_s, 2dim_s)
    dim_s, dim_b, dim_H = size(Ps, 1), size(Pb, 1), size(H0, 2)
    velocities = fill(zero(real(T)), 2dim_s)
    temps = SingleShot1DTemporaries{T}(dim_H, dim_s, dim_b)
    return SingleShot1DGreensSolver(invert_B, λ2min, A, B, -H0, V, Pb, Ps, Ps´, H0ss, H0bs, Vss, Vbs, hessbb, velocities, maxdn, temps)
end

function bulk_surface_projectors(H0::AbstractMatrix{T}, V, V´, cutoff2) where {T}
    SVD = svd(Matrix(V), full = true)
    W, S, U = SVD.U, SVD.S, SVD.V
    dim_b = count(s -> abs2(s) < cutoff2, S)
    dim_s = length(S) - dim_b
    if iszero(dim_b)
        Ps = Matrix{T}(I, dim_s, dim_s)
        Ps´ = copy(Ps)
        Pb = Ps[1:0, :]
    else
        Ps = U'[1:dim_s, :]
        Pb = U'[dim_s+1:end, :]
        Ps´ = W'[1:dim_s, :]
        Pb´ = W'[dim_s+1:end, :]
    end
    return Pb, Ps, Ps´
    return Pb, Ps, Ps´
end

## Solver execution

function (gs::SingleShot1DGreensSolver)(ω)
    A, B = single_shot_surface_matrices(gs, ω)
    λs, φχs = eigen_funcbarrier!(A, B, gs)
    dim_s = size(φχs, 1) ÷ 2
    φs = view(φχs, 1:dim_s, :)
    χs = view(φχs, dim_s+1:2dim_s, :)
    φ = gs.temps.φ
    χ = gs.temps.χ
    if hasbulk(gs)
        reconstruct_bulk!(φ, ω, λs, φs, gs)
        reconstruct_bulk!(χ, ω, λs, χs, gs)
    else
        copy!(φ, φs)
        copy!(χ, χs)
    end
    return classify_retarded_advanced(λs, φs, χs, φ, χ, gs)
end

function eigen_funcbarrier!(A::AbstractMatrix{T}, B, gs)::Tuple{Vector{T},Matrix{T}} where {T<:Complex}
    if gs.invert_B
        factB = lu!(B)
        B⁻¹A = ldiv!(factB, A)
        λs, φχs = eigen!(B⁻¹A; sortby = abs)
        clean_λ!(λs, gs.λ2min)
    else
        alpha, beta, _, φχ´ = LAPACK.ggev!('N', 'V', A, B)
        λ´ = clean_λ!(alpha, beta, gs.λ2min)
        λs, φχs = GeneralizedEigen(LinearAlgebra.sorteig!(λ´, φχ´, abs)...)
    end
    return λs, φχs
end

function clean_λ!(λ::AbstractVector{T}, cutoff2) where {T}
    λs .= (λ -> ifelse(isnan(λ), T(Inf),
                ifelse(abs2(λ) < λ2min, zero(T),
                ifelse(abs2(λ) > 1/λ2min, T(Inf), λ)))).(λs)
    return λs
end

function clean_λ!(αs::AbstractVector{T}, βs, cutoff2) where {T}
    λs = αs
    for i in eachindex(αs)
        α2, β2 = abs2(αs[i]), abs2(βs[i])
        if α2 < cutoff2
            λs[i] = zero(T)
        elseif β2 < cutoff2
            λs[i] = T(Inf)
        else
            λs[i] = αs[i] / βs[i]
        end
    end
    return λs
end

function single_shot_surface_matrices(gs::SingleShot1DGreensSolver{T}, ω) where {T}
    A, B = gs.A, gs.B
    dim_s = size(gs.H0bs, 2)
    fill!(A, 0)
    fill!(B, 0)
    for i in 1:dim_s
        A[i, dim_s + i] = B[i, i] = one(T)
        A[dim_s + i, dim_s + i] = ω
    end
    tmp = view(gs.temps.b2s, :, 1:dim_s)  # gs.temps.b2s has 2dim_s cols
    dim = size(A, 1) ÷ 2
    A21 = view(A, dim+1:2dim, 1:dim)
    A22 = view(A, dim+1:2dim, dim+1:2dim)
    B22 = view(B, dim+1:2dim, dim+1:2dim)

    # A22 = ωI - H₀ₛₛ - H₀ₛᵦ g₀ᵦᵦ H₀ᵦₛ - V'ₛᵦ g₀ᵦᵦ Vᵦₛ
    A22 .-= gs.H0ss  # ω already added to diagonal above
    if hasbulk(gs)
        copy!(tmp, gs.H0bs)
        ldiv!(gs.hessbb + ω*I, tmp)
        mul!(A22, gs.H0bs', tmp, -1, 1)
        copy!(tmp, gs.Vbs)
        ldiv!(gs.hessbb + ω*I, tmp)
        mul!(A22, gs.Vbs', tmp, -1, 1)
    end

    # A21 = - Vₛₛ - H₀ₛᵦ g₀ᵦᵦ Vᵦₛ
    A21 .= .- gs.Vss
    if hasbulk(gs)
        copy!(tmp, gs.Vbs)
        ldiv!(gs.hessbb + ω*I, tmp)
        mul!(A21, gs.H0bs', tmp, -1, 1)
    end

    # B22 = -A21'
    B22 .= .- A21'

    chkfinite(A)
    chkfinite(B)

    return A, B
end

function chkfinite(A::AbstractMatrix)
    for a in A
        if !isfinite(a)
            throw(ArgumentError("Matrix contains Infs or NaNs. This may happen when the energy ω exactly matches a bound state in the spectrum. Try adding a small positive imaginary part to ω."))
        end
    end
    return true
end

# φ = [Pₛ' + Pᵦ' g₀ᵦᵦ (λ⁻¹Vᵦₛ + H₀ᵦₛ)] φₛ
function reconstruct_bulk!(φ, ω, λs, φs, gs)
    tmp = gs.temps.b2s
    mul!(tmp, gs.Vbs, φs)
    tmp ./= transpose(λs)
    mul!(tmp, gs.H0bs, φs, 1, 1)
    ldiv!(gs.hessbb + ω*I, tmp)
    mul!(φ, gs.Pb', tmp)
    mul!(φ, gs.Ps', φs, 1, 1)
    return φ
end

# function classify_retarded_advanced(λs, φs, φ, χ, gs)
function classify_retarded_advanced(λs, φs, χs, φ, χ, gs)
    vs = compute_velocities_and_normalize!(φ, χ, λs, gs)
    # order for ret-evan, ret-prop, adv-prop, adv-evan
    p = sortperm!(gs.temps.v2s1, vs; rev = true)
    p´ = gs.temps.v2s2
    Base.permute!!(vs, copy!(p´, p))
    Base.permute!!(λs, copy!(p´, p))
    Base.permutecols!!(φ, copy!(p´, p))
    Base.permutecols!!(χ, copy!(p´, p))
    Base.permutecols!!(φs, copy!(p´, p))

    ret, adv = nonsingular_ret_adv(λs, vs)
    # @show vs
    # @show abs.(λs)
    # @show ret, adv, length(λs), length(vs)

    λR   = view(λs, ret)
    χR   = view(χ, :, ret)   # This resides in part of gs.temps.χ
    φR   = view(φ, :, ret)   # This resides in part of gs.temps.φ
    # overwrite output of eigen to preserve normalization of full φ
    φRs  = mul!(view(φs, :, ret), gs.Ps, φR)
    iφRs = issquare(φRs) ? rdiv!(copyto!(gs.temps.ss1, I), lu!(φRs)) : pinv(φRs)

    iλA  = view(λs, adv)
    iλA .= inv.(iλA)
    χA  = view(χ, :, adv)   # This resides in part of gs.temps.χ
    φA   = view(φ, :, adv)   # This resides in part of gs.temps.φ
    # overwrite output of eigen to preserve normalization of full χ
    χAs´ = mul!(view(χs, :, adv), gs.Ps´, χA)
    iχAs´ = issquare(χAs´) ? rdiv!(copyto!(gs.temps.ss2, I), lu!(χAs´)) : pinv(χAs´)

    return λR, χR, iφRs, iλA, φA, iχAs´
end

# The velocity is given by vᵢ = im * φᵢ'(V'λᵢ-Vλᵢ')φᵢ / φᵢ'φᵢ = 2*imag(χᵢ'Vφᵢ)/φᵢ'φᵢ
function compute_velocities_and_normalize!(φ, χ, λs, gs)
    vs = gs.velocities
    tmp = gs.temps.vH
    for (i, λ) in enumerate(λs)
        abs2λ = abs2(λ)
        if abs2λ ≈ 1
            φi = view(φ, :, i)
            χi = view(χ, :, i)
            norm2φi = dot(φi, φi)
            mul!(tmp, gs.V, φi)
            v = 2*imag(dot(χi, tmp))/norm2φi
            φi .*= inv(sqrt(norm2φi * abs(v)))
            χi .*= inv(sqrt(norm2φi * abs(v)))
            vs[i] = v
        else
            vs[i] = abs2λ < 1 ? Inf : -Inf
        end
    end # sortperm(vs) would now give the order of adv-evan, adv-prop, ret-prop, ret-evan
    return view(vs, 1:length(λs))
end

function nonsingular_ret_adv(λs::AbstractVector{T}, vs) where {T}
    rmin, rmax = 1, 0
    amin, amax = 1, 0
    for (i, v) in enumerate(vs)
        aλ2 = abs2(λs[i])
        if iszero(aλ2)
            rmin = i + 1
        elseif v > 0
            rmax = i
            amin = i + 1
        elseif v < 0 && isfinite(aλ2)
            amax = i
        end
    end
    return rmin:rmax, amin:amax
end

## Greens execution

(g::GreensFunction{<:SingleShot1DGreensSolver})(ω, cells) = g(ω, missing)(sanitize_cells(cells, g))

# Infinite: G∞_{N}  = GVᴺ G∞_{0}
function (g::GreensFunction{<:SingleShot1DGreensSolver,1,Tuple{Missing}})(ω, ::Missing)
    gs = g.solver
    factors = gs(ω)
    G∞⁻¹ = inverse_G∞!(gs.temps.HH0, ω, gs, factors) |> lu!
    return cells -> G_infinite!(gs, factors, G∞⁻¹, cells)
end

# Semiinifinite: G_{N,M} = (GVᴺ⁻ᴹ - GVᴺGV⁻ᴹ)G∞_{0}
function (g::GreensFunction{<:SingleShot1DGreensSolver,1,Tuple{Int}})(ω, ::Missing)
    gs = g.solver
    factors = gs(ω)
    G∞⁻¹ = inverse_G∞!(gs.temps.HH0, ω, gs, factors) |> lu!
    N0 = only(g.boundaries)
    return cells -> G_semiinfinite!(gs, factors, G∞⁻¹, shift_cells(cells, N0))
end

function invG(g::GreensFunction{<:SingleShot1DGreensSolver}, ω)
    gs = g.solver
    factors = gs(ω)
    onlyhalf = only(g.boundaries) === missing ? false : true
    G∞⁻¹ = inverse_G∞!(gs.temps.HH0, ω, gs, factors, onlyhalf)
    return G∞⁻¹
end

function G_infinite!(gs, factors, G∞⁻¹, (src, dst))
    src´ = div(only(src), gs.maxdn, RoundToZero)
    dst´ = div(only(dst), gs.maxdn, RoundToZero)
    N = dst´ - src´
    GVᴺ = GVᴺ!(gs.temps.HH1, N, gs, factors)
    G∞ = rdiv!(GVᴺ, G∞⁻¹)
    return G∞
end

function G_semiinfinite!(gs, factors, G∞⁻¹, (src, dst))
    M = div(only(src), gs.maxdn, RoundToZero)
    N = div(only(dst), gs.maxdn, RoundToZero)
    if sign(N) != sign(M)
        G∞ = fill!(gs.temps.HH1, 0)
    else
        GVᴺ⁻ᴹ = GVᴺ!(gs.temps.HH1, N-M, gs, factors)
        GVᴺ = GVᴺ!(gs.temps.HH2, N, gs, factors)
        GV⁻ᴹ = GVᴺ!(gs.temps.HH3, -M, gs, factors)
        mul!(GVᴺ⁻ᴹ, GVᴺ, GV⁻ᴹ, -1, 1) # (GVᴺ⁻ᴹ - GVᴺGV⁻ᴹ)
        G∞ = rdiv!(GVᴺ⁻ᴹ , G∞⁻¹)
    end
    return G∞
end

shift_cells((src, dst), N0) = (only(src) - N0, only(dst) - N0)

# G∞⁻¹ = G₀⁻¹ - V´GrV - VGlV´ with V´GrV = V'*χR*φRs⁻¹Ps and VGlV´ = iχA*φAs´⁻¹Ps´
function  inverse_G∞!(matrix, ω, gs, (λR, χR, iφRs, iλA, φA, iχAs´), onlyhalf = false)
    G0⁻¹ = inverse_G0!(matrix, ω, gs)
    V´GrV = mul!(gs.temps.Hs2, gs.V', mul!(gs.temps.Hs1, χR, iφRs))
    mul!(G0⁻¹, V´GrV, gs.Ps, -1, 1)
    if onlyhalf
        G∞⁻¹ = G0⁻¹
    else
        VGlV´ = mul!(gs.temps.Hs2, gs.V, mul!(gs.temps.Hs1, φA, iχAs´))
        G∞⁻¹ = mul!(G0⁻¹, VGlV´, gs.Ps´, -1, 1)
    end
    return G∞⁻¹
end

# G₀⁻¹ = ω - H₀
function inverse_G0!(G0⁻¹, ω, gs)
    copy!(G0⁻¹, gs.minusH0)
    for i in axes(G0⁻¹, 1)
        G0⁻¹[i, i] += ω
    end
    return G0⁻¹
end

function GVᴺ!(GVᴺ, N, gs, (λR, χR, iφRs, iλA, φA, iχAs´))
    if N == 0
        copyto!(GVᴺ, I)
    elseif N > 0
        χᴿλᴿᴺ´ = N == 1 ? χR : (gs.temps.Hs1 .= χR .* transpose(λR) .^ (N-1))
        mul!(GVᴺ, mul!(gs.temps.Hs2, χᴿλᴿᴺ´, iφRs), gs.Ps)
    else
        φᴬλᴬᴺ´ = N == -1 ? φA : (gs.temps.Hs1 .= φA .* transpose(iλA) .^ (-N-1))
        mul!(GVᴺ, mul!(gs.temps.Hs2, φᴬλᴬᴺ´, iχAs´), gs.Ps´)
    end
    return GVᴺ
end

#######################################################################
# BandGreensSolver
#######################################################################
# struct SimplexData{D,E,T,C<:SMatrix,DD,SA<:SubArray}
#     ε0::T
#     εmin::T
#     εmax::T
#     k0::SVector{D,T}
#     Δks::SMatrix{D,D,T,DD}     # k - k0 = Δks * z
#     volume::T
#     zvelocity::SVector{D,T}
#     edgecoeffs::NTuple{E,Tuple{T,C}} # s*det(Λ)/w.w and Λc for each edge
#     dωzs::NTuple{E,NTuple{2,SVector{D,T}}}
#     defaultdη::SVector{D,T}
#     φ0::SA
#     φs::NTuple{D,SA}
# end

# struct BandGreensSolver{P<:SimplexData,E,H<:Hamiltonian} <: AbstractGreensSolver
#     simplexdata::Vector{P}
#     indsedges::NTuple{E,Tuple{Int,Int}} # all distinct pairs of 1:V, where V=D+1=num verts
#     h::H
# end

# function Base.show(io::IO, g::GreensFunction{<:BandGreensSolver})
#     print(io, summary(g), "\n",
# "  Matrix size    : $(size(g.h, 1)) × $(size(g.h, 2))
#   Element type   : $(displayelements(g.h))
#   Band simplices : $(length(g.solver.simplexdata))")
# end

# Base.summary(g::GreensFunction{<:BandGreensSolver}) =
#     "GreensFunction{Bandstructure}: Green's function from a $(latdim(g.h))D bandstructure"

# function greensolver(b::Bandstructure{D}, h) where {D}
#     indsedges = tuplepairs(Val(D))
#     v = [SimplexData(simplex, band, indsedges) for band in bands(b) for simplex in band.simplices]
#     return BandGreensSolver(v,  indsedges, h)
# end

# edges_per_simplex(L) = binomial(L,2)

# function SimplexData(simplex::NTuple{V}, band, indsedges) where {V}
#     D = V - 1
#     vs = ntuple(i -> vertices(band)[simplex[i]], Val(V))
#     ks = ntuple(i -> SVector(Base.front(Tuple(vs[i]))), Val(V))
#     εs = ntuple(i -> last(vs[i]), Val(V))
#     εmin, εmax = extrema(εs)
#     ε0 = first(εs)
#     k0 = first(ks)
#     Δks = hcat(tuple_minus_first(ks)...)
#     zvelocity = SVector(tuple_minus_first(εs))
#     volume = abs(det(Δks))
#     edgecoeffs = edgecoeff.(indsedges, Ref(zvelocity))
#     dωzs = sectionpoint.(indsedges, Ref(zvelocity))
#     defaultdη = dummydη(zvelocity)
#     φ0 = vertexstate(first(simplex), band)
#     φs = vertexstate.(Base.tail(simplex), Ref(band))
#     return SimplexData(ε0, εmin, εmax, k0, Δks, volume, zvelocity, edgecoeffs, dωzs, defaultdη, φ0, φs)
# end

# function edgecoeff(indsedge, zvelocity::SVector{D}) where {D}
#     basis = edgebasis(indsedge, Val(D))
#     othervecs = Base.tail(basis)
#     edgevec = first(basis)
#     cutvecs = (v -> dot(zvelocity, edgevec) * v - dot(zvelocity, v) * edgevec).(othervecs)
#     Λc = hcat(cutvecs...)
#     Λ = hcat(zvelocity, Λc)
#     s = sign(det(hcat(basis...)))
#     coeff = s * (det(Λ)/dot(zvelocity, zvelocity))
#     return coeff, Λc
# end

# function edgebasis(indsedge, ::Val{D}) where {D}
#     inds = ntuple(identity, Val(D+1))
#     swappedinds = tupleswapfront(inds, indsedge) # places the two edge vertindices first
#     zverts = (i->unitvector(SVector{D,Int}, i-1)).(swappedinds)
#     basis = (z -> z - first(zverts)).(Base.tail(zverts)) # first of basis is edge vector
#     return basis
# end

# function sectionpoint((i, j), zvelocity::SVector{D,T}) where {D,T}
#     z0, z1 = unitvector(SVector{D,Int}, i-1), unitvector(SVector{D,Int}, j-1)
#     z10 = z1 - z0
#     # avoid numerical cancellation errors due to zvelocity perpendicular to edge
#     d = chop(dot(zvelocity, z10), maximum(abs.(zvelocity)))
#     dzdω = z10 / d
#     dz0 = z0 - z10 * dot(zvelocity, z0) / d
#     return dzdω, dz0   # The section z is dω * dzdω + dz0
# end

# # A vector, not parallel to zvelocity, and with all nonzero components and none equal
# function dummydη(zvelocity::SVector{D,T}) where {D,T}
#     (D == 1 || iszero(zvelocity)) && return SVector(ntuple(i->T(i), Val(D)))
#     rng = MersenneTwister(0)
#     while true
#         dη = rand(rng, SVector{D,T})
#         isparallel = dot(dη, zvelocity)^2 ≈ dot(zvelocity, zvelocity) * dot(dη, dη)
#         isvalid = allunique(dη) && !isparallel
#         isvalid && return dη
#     end
#     throw(error("Unexpected error finding dummy dη"))
# end

# function vertexstate(ind, band)
#     ϕind = 1 + band.dimstates*(ind - 1)
#     state = view(band.states, ϕind:(ϕind+band.dimstates-1))
#     return state
# end

# ## Call API

# function greens!(matrix, g::GreensFunction{<:BandGreensSolver,L}, ω::Number, (src, dst)::SVectorPair{L}) where {L}
#     fill!(matrix, zero(eltype(matrix)))
#     dn = dst - src
#     for simplexdata in g.solver.simplexdata
#         g0, gjs = green_simplex(ω, dn, simplexdata, g.solver.indsedges)
#         addsimplex!(matrix, g0, gjs, simplexdata)
#     end
#     return matrix
# end

# function green_simplex(ω, dn, data::SimplexData{L}, indsedges) where {L}
#     dη = data.Δks' * dn
#     phase = cis(dot(dn, data.k0))
#     dω = ω - data.ε0
#     gz = simplexterm.(dω, Ref(dη), Ref(data), data.edgecoeffs, data.dωzs, indsedges)
#     g0z, gjz = first.(gz), last.(gz)
#     g0 = im^(L-1) * phase * sum(g0z)
#     gj = -im^L * phase * sum(gjz)
#     return g0, gj
# end

# function simplexterm(dω, dη::SVector{D,T}, data, coeffs, (dzdω, dz0), (i, j)) where {D,T}
#     bailout = Complex(zero(T)), Complex.(zero(dη))
#     z = dω * dzdω + dz0
#     # Edges with divergent sections do not contribute
#     all(isfinite, z) || return bailout
#     z0 = unitvector(SVector{D,T},i-1)
#     z1 = unitvector(SVector{D,T},j-1)
#     coeff, Λc = coeffs
#     # If dη is zero (DOS) use a precomputed (non-problematic) simplex-constant vector
#     dη´ = iszero(dη) ? data.defaultdη : dη
#     d = dot(dη´, z)
#     d0 = dot(dη´, z0)
#     d1 = dot(dη´, z1)
#     # Skip if singularity in formula
#     (d ≈ d0 || d ≈ d1) && return bailout
#     s = sign(dot(dη´, dzdω))
#     coeff0 = coeff / prod(Λc' * dη´)
#     coeffj = isempty(Λc) ? zero(dη) : (Λc ./ ((dη´)' * Λc)) * sumvec(Λc)
#     params = s, d, d0, d1
#     zs = z, z0, z1
#     g0z = iszero(dη) ? g0z_asymptotic(D, coeff0, params) : g0z_general(coeff0, params)
#     gjz = iszero(dη) ? gjz_asymptotic(D, g0z, coeffj, coeff0, zs, params) :
#                        gjz_general(g0z, coeffj, coeff0, zs, params)
#     return g0z, gjz
# end

# sumvec(::SMatrix{N,M,T}) where {N,M,T} = SVector(ntuple(_->one(T),Val(M)))

# g0z_general(coeff0, (s, d, d0, d1)) =
#     coeff0 * cis(d) * ((cosint_c(-s*(d0-d)) + im*sinint(d0-d)) - (cosint_c(-s*(d1-d)) + im*sinint(d1-d)))

# gjz_general(g0z, coeffj, coeff0, (z, z0, z1), (s, d, d0, d1)) =
#     g0z * (im * z - coeffj) + coeff0 * ((z0-z) * cis(d0) / (d0-d) - (z1-z) * cis(d1) / (d1-d))

# g0z_asymptotic(D, coeff0, (s, d, d0, d1)) =
#     coeff0 * (cosint_a(-s*(d0-d)) - cosint_a(-s*(d1-d))) * (im*d)^(D-1)/factorial(D-1)

# function gjz_asymptotic(D, g0z, coeffj, coeff0, (z, z0, z1), (s, d, d0, d1))
#     g0z´ = g0z
#     for n in 1:(D-1)
#         g0z´ += coeff0 * im^n * (im*d)^(D-1-n)/factorial(D-1-n) *
#                 ((d0-d)^n - (d1-d)^n)/(n*factorial(n))
#     end
#     gjz = g0z´ * (im * z - im * coeffj * d / D) +
#         coeff0 * ((z0-z) * (im*d0)^D / (d0-d) - (z1-z) * (im*d1)^D / (d1-d)) / factorial(D)
#     return gjz
# end

# cosint_c(x::Real) = ifelse(iszero(abs(x)), zero(x), cosint(abs(x))) + im*pi*(x<=0)

# cosint_a(x::Real) = ifelse(iszero(abs(x)), zero(x), log(abs(x))) + im*pi*(x<=0)

# function addsimplex!(matrix, g0, gjs, simplexdata)
#     φ0 = simplexdata.φ0
#     φs = simplexdata.φs
#     vol = simplexdata.volume
#     for c in CartesianIndices(matrix)
#         (row, col) = Tuple(c)
#         x = g0 * (φ0[row] * φ0[col]')
#         for (φ, gj) in zip(φs, gjs)
#             x += (φ[row]*φ[col]' - φ0[row]*φ0[col]') * gj
#         end
#         matrix[row, col] += vol * x
#     end
#     return matrix
# end