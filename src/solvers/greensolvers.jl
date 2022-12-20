############################################################################################
# GreenSolvers module
#region

module GreenSolvers

using LinearAlgebra: I, lu, ldiv!, mul!
using SparseArrays
using SparseArrays: SparseMatrixCSC, AbstractSparseMatrix
using Quantica: Quantica,
      AbstractGreenSolver, AppliedDirectGreenSolver, AppliedInverseGreenSolver, AppliedLeadSolver,
      GreenBlock, GreenBlockInverse, LatticeBlock,
      Hamiltonian, ParametricHamiltonian, AbstractHamiltonian, BlockStructure,
      HybridSparseMatrixCSC, lattice, zerocell, SVector, sanitize_SVector, siteselector,
      foreach_cell, foreach_site, store_diagonal_ptrs, cell
import Quantica: call!, apply, GreenLead

############################################################################################
# BlockView  - see scattering.pdf notes for derivations
#region

struct BlockView{S<:AppliedDirectGreenSolver} <: AppliedDirectGreenSolver
    inds::Vector{Int}
    parent::S
end

function (s::BlockView)(ω; kw...)
    m = s.parent(ω; kw...)
    return view(m, s.inds)
end

#endregion

############################################################################################
# Schur and AppliedSchur
#   This solver can be used to produce a GreenBlock or a GreenLead
#region

const AbstractHamiltonian1D{T,E,B} = AbstractHamiltonian{T,E,1,B}

struct Schur{T<:AbstractFloat} <: AbstractGreenSolver
    shift::T
    onlyintracell::Bool
end

struct AppliedSchurSolver{T<:AbstractFloat,E,B,H<:AbstractHamiltonian1D{T,E,B}} <: AppliedDirectGreenSolver
    factors::SchurFactorsSolver{T,B}
    h::H
    latblock::LatticeBlock{T,E,1}
    boundary::T
    onlyintracell::Bool
end

#region ## Constructors ##

Schur(; shift = 1.0, onlyintracell = false) = Schur(shift, onlyintracell)

#endregion

#region ## apply ##

function apply(s::Schur, h::AbstractHamiltonian1D, latblock::LatticeBlock, boundaries)
    boundary = only(boundaries)
    h´ = hamiltonian(h)  # to extract the Hamiltonian if h isa ParametricHamiltonian
    factors = SchurFactorsSolver(h´, s.shift)
    return AppliedSchurSolver(factors, h, latblock, boundary, onlyintracell)
end

#region

#region ## call API ##

function copy_callsafe(s::AppliedSchurSolver)
    factors = copy_callsafe(s.factors)
    h = copy_callsafe(s.h)
    return AppliedSchurSolver(factors, h, s.latblock, s.boundary, s.onlyintracell)
end

function call!(s::AppliedSchurSolver; params...)
    h´ = call!(s.h; params...)  # this is a no-op if s.h is not a ParametricHamiltonian
    # the h wrapped into s.factors should be === h´, see apply above
    return AppliedSchurSolver(s.factors, h´, s.latblock, s.boundary, s.onlyintracell)
end

function call!(s::AppliedSchurSolver{T}, ω; params...) where {T}
    call!(s; params...)       # this is a no-op if s.h is not a ParametricHamiltonian
    factors = call!(s.factors, ω)
    latblock = s.latblock
    boundary = s.boundary

    # allocate GreenBlock matrix
    bis = blockinds(s.h, s.latblock)
    dim = sum(length, bis)
    g = fill(complex(T(NaN)), dim, dim) # NaN = sentinel for not computed (if onlyintracell)

    offseti = offsetj = 0
    itr = zip(subcells(latblock), bis, 1:length(bis))
    for (scelli, indsi, blocki) in itr, (scellj, indsj, blockj) in itr
        s.onlyintracell && blocki != blockj && continue
        grangei = offseti + 1 : offseti + length(indsi)
        grangej = offsetj + 1 : offsetj + length(indsj)
        offseti += last(grangei)
        offsetj += last(grangej)
        xi = only(cell(scelli)) - boundary
        xj = only(cell(scellj)) - boundary
        gblock = view(g, grangei, grangej)
        isfinite(boundary) ?
            green_schur_semi!(gblock, factors, (xi, indsi), (xj, indsj)) :
            green_schur_inf!(gblock, factors, xi - xj, indsi, indsj)
    end
    return g
end

# flat indices of unitcell sites in each subcell
function blockinds(h, latblock)
    flatinds = Vector{Int}[]
    for scell in subcells(latblock)
        sinds = Int[]
        push!(flatinds, sinds)
        for iunflat in siteindices(scell)
            append!(sinds, flatrange(h, iunflat))
        end
    end
    return flatinds
end

# Semiinifinite:
# Gₙₘ = (Ghⁿ⁻ᵐ - GhⁿGh⁻ᵐ)G∞₀₀ = G∞ₙₘ - GhⁿG∞₀ₘ
# Gₙₘ = G∞ₙₘ - G₁₁L(R'G₁₁L)ⁿ⁻¹ R'G∞₀ₘ       for n > 1
# Gₙₘ = G∞ₙₘ - G₋₁₋₁R(L'G₋₁₋₁R)⁻ⁿ⁻¹L'G∞₀ₘ   for n < -1
function green_schur_semi!(gblock, ((R, Z11, Z21), (L, Z11´, Z21´)), (xi, indsi), (xj, indsj))

end

#endregion
#endregion

############################################################################################
# SchurFactorsSolver - see scattering.pdf notes for derivations
#   Computes the factors R, Z21 and Z11. The retarded self-energy on the open unitcell
#   surface of a semi-infinite lead extending towards the right reads Σᵣ = R Z21 Z11⁻¹ R'.
#   Computes also the L, Z21´ and Z11´ for the lead towards the left  Σₗ = L Z21´ Z11´⁻¹ L'.
#region

struct SchurWorkspace{C}
    GL::Matrix{C}
    GR::Matrix{C}
    A::Matrix{C}
    B::Matrix{C}
    Z11::Matrix{C}
    Z21::Matrix{C}
    Z11´::Matrix{C}
    Z21´::Matrix{C}
end

struct SchurFactorsSolver{T,B}
    shift::T                                          # called Ω in the scattering.pdf notes
    hm::HybridSparseMatrixCSC{T,B}
    h0::HybridSparseMatrixCSC{T,B}
    hp::HybridSparseMatrixCSC{T,B}
    l_leq_r::Bool                                     # whether l <= r (left and right surface dims)
    iG::SparseMatrixCSC{Complex{T},Int}               # to store iG = ω - h0 - Σₐᵤₓ
    ptrs::Tuple{Vector{Int},Vector{Int},Vector{Int}}  # iG ptrs for h0 nzvals, diagonal and Σₐᵤₓ surface
    sinds::Vector{Int}                                # site indices on the smalles surface (left for l<=r, right for l>r)
    L::Matrix{Complex{T}}                             # l<=r ? PL : PL*H' === hp PR
    R::Matrix{Complex{T}}                             # l<=r ? PR*H === hm PL : PR
    R´L´::Matrix{Complex{T}}  # [R'; -L']
    tmp::SchurWorkspace{Complex{T}}
end

#region ## Constructors ##

SchurFactorsSolver(::AbstractHamiltonian, _) =
    argerror("The Schur solver requires 1D Hamiltonians with 0 and ±1 as only Bloch Harmonics.")

function SchurFactorsSolver(h::Hamiltonian{T,<:Any,1}, shift) where {T}
    hm, h0, hp = hybridharmonics(h)
    fhm, fh0, fhp = flat(hm), flat(h0), flat(hp)
    L, R, sinds, l_leq_r = left_right_projectors(fhm, fhp)
    R´L´ = [R'; -L']
    iG, (p, pd) = store_diagonal_ptrs(fh0)
    ptrs = (p, pd, pd[sinds])
    workspace = SchurWorkspace{Complex{T}}(size(L))
    return SchurFactorsSolver(shift, hm, h0, hp, l_leq_r, iG, ptrs, sinds, L, R, R´L´, workspace)
end

function SchurWorkspace{C}((n, d)) where {C}
    GL = Matrix{C}(undef, n, d)
    GR = Matrix{C}(undef, n, d)
    A = Matrix{C}(undef, 2d, 2d)
    B = Matrix{C}(undef, 2d, 2d)
    Z11 = Matrix{C}(undef, d, d)
    Z21 = Matrix{C}(undef, d, d)
    Z11´ = Matrix{C}(undef, d, d)
    Z21´ = Matrix{C}(undef, d, d)
    return SchurWorkspace(GL, GR, A, B, Z11, Z21, Z11´, Z21´)
end

function hybridharmonics(h)
    for hh in harmonics(h)
        dn = dcell(hh)
        dn == SA[0] || dn == SA[1] || dn == SA[-1] ||
            argerror("Too many harmonics, try `supercell` to reduce to strictly nearest-neighbors.")
    end
    hm, h0, hp = h[-1], h[0], h[1]
    flat(hm) == flat(hp)' ||
        argerror("The Hamiltonian should have h[1] = h[-1]' to use the Schur solver")
    return hm, h0, hp
end

# hp = L*R' = PL H' PR'. We assume hm = hp'
function left_right_projectors(hm::SparseMatrixCSC, hp::SparseMatrixCSC)
    linds = unique!(sort!(copy(rowvals(hp))))
    rinds = Int[]
    for col in axes(hp, 2)
        isempty(nzrange(hp, col)) || push!(rinds, col)
    end
    # dense projectors
    o = one(eltype(hp))*I
    allrows = 1:size(hp,1)
    l_leq_r = length(linds) <= length(rinds)
    PR = o[allrows, rinds]
    PL = o[allrows, linds]
    if l_leq_r
        sinds = linds
        R = Matrix(hm[:, linds])  # R = PR H = hm PL
        L = PL
    else
        sinds = rinds
        R = PR
        L = Matrix(hp[:, rinds])  # L = PL H' = hp PR
    end
    return L, R, sinds, l_leq_r
end

#endregion

#region ## API ##

## Call API ##

# copy_callsafe should copy anything that may mutate with call!(...; params...) and any
# object in the output of call!(::SchurFactorsSolver, ...; params...)
function copy_callsafe(s::SchurFactorsSolver)
    hm, h0, hp = copy_callsafe(s.hm), copy_callsafe(s.h0), copy_callsafe(s.hp)
    L, R = copy(s.L), copy(s.R)
    tmp = copy_callsafe(s.tmp)
    return SchurFactorsSolver(s.shift, hm, h0, hp, s.l_leq_r, s.iG, s.ptrs, s.sinds,
                              L, R, s.R´L´, tmp)
end


function copy_callsafe(w::SchurWorkspace)
    Z11, Z21, Z11´, Z21´ = copy(w.Z11), copy(w.Z21), copy(w.Z11´), copy(w.Z21´)
    return SchurWorkspace(w.GL, w.GR, w.A, w.B, Z11, Z21, Z11´, Z21´)
end

function call!(s::SchurFactorsSolver, ω)
    R, Z11, Z21, L, Z11´, Z21´ = s.R, s.tmp.Z11, s.tmp.Z21, s.L, s.tmp.Z11´, s.tmp.Z21´
    update_LR!(s)
    update_iG!(s, ω)

    A, B = pencilAB!(s)
    sch = schur!(A, B)
    whichmodes = Vector{Bool}(undef, length(sch.α))

    # Retarded modes
    retarded_modes!(whichmodes, sch)
    checkmodes(whichmodes)
    ordschur!(sch, whichmodes)
    copy!(Z11, view(sch.Z, 1:r, 1:sum(whichmodes)))
    copy!(Z21, view(sch.Z, r+1:2r, 1:sum(whichmodes)))

    # Advanced modes
    advanced_modes!(whichmodes, sch)
    checkmodes(whichmodes)
    ordschur!(sch, whichmodes)
    copy!(Z11´, view(sch.Z, 1:r, 1:sum(whichmodes)))
    copy!(Z21´, view(sch.Z, r+1:2r, 1:sum(whichmodes)))

    return (R, Z11, Z21), (L, Z11´, Z21´)
end

# need this barrier for type-stability (sch.α and sch.β are finicky)
function retarded_modes!(whichmodes, sch)
    whichmodes .= abs.(sch.α) .< abs.(sch.β)
    return whichmodes
end

function advanced_modes!(whichmodes, sch)
    whichmodes .= abs.(sch.β) .< abs.(sch.α)
    return whichmodes
end

checkmodes(whichmodes) = sum(whichmodes) == length(whichmodes) ÷ 2 ||
    argerror("Cannot differentiate retarded from advanced modes. Consider increasing imag(ω) or check that your Hamiltonian is Hermitian")

## Pencil A - λB ##

# Compute G*R and G*L where G = inv(ω - h0 - Σₐᵤₓ) for Σₐᵤₓ = -iΩL'L or -iΩR'R
# From this compute the deflated A - λB, whose eigenstates are the deflated eigenmodes
# Pencil A - λB :
#    A = [R'GL  (1-δ)iΩR'GL; -L'GL  1-(1-δ)iΩL´GL] and B = [1-δiΩR'GR  -R'GR; δiΩL'GR  L'GR]
#    where δ = l <= r ? 1 : 0
#    A = [Γₗ (1-δ)iΩΓₗ] + [0 0; 0 1] and B = [-Γᵣ -δiΩΓᵣ] + [1 0; 0 0]
#    Γₗ = [R'; -L']GL  and  Γᵣ = [R'; -L']GR
function pencilAB!(s::SchurFactorsSolver{T}) where {T}
    o, z = one(Complex{T}), zero(Complex{T})
    iGlu = lu(s.iG)
    Ω = s.shift
    GL = ldiv!(s.tmp.GL, iGlu, s.L)
    GR = ldiv!(s.tmp.GR, iGlu, s.R)
    A, B, R´L´ = s.tmp.A, s.tmp.B, s.R´L´
    fill!(A, z)
    fill!(B, z)
    mul!(view(A, :, 1:d), R´L´, GL)
    mul!(view(B, :, d+1:2d), R´L´, GR, -1, 0)
    if s.l_leq_r
        view(A, :, d+1:2d) .= view(A, :, 1:d) .* (im*Ω)
    else
        view(B, :, 1:d) .= view(B, :, d+1:2d) .* (im*Ω)
    end
    for i in 1:d
        A[d+i, d+i] += o
        B[i, i] += o
    end
    return A, B
end

# updates L and R from the present hm and hp
function update_LR!(s)
    d = size(s.L, 2)
    if s.l_leq_r
        copy!(s.R, flat(s.hm)[:, s.sinds])
        view(s.R´L´, 1:d, :) .= s.R'
    else
        copy!(s.L, flat(s.hp)[:, s.sinds])
        view(s.R´L´, d+1:2d, :) .= .- s.L'
    end
    return s
end

# updates iG = ω - h0 - Σₐᵤₓ from the present h0
function update_iG!(s::SchurFactorsSolver{T}, ω) where {T}
    Ω = s.shift
    nzs, nzsh0 = nonzeros(s.iG), nonzeros(flat(s.h0))
    ps, pds, pss = s.ptrs
    fill!(nzs, zero(Complex{T}))
    for (p, p´) in enumerate(ps)
        nzs[p´] = -nzsh0[p]
    end
    for pd in pds
        nzs[pd] += ω
    end
    for ps in pss
        nzs[ps] += im*Ω
    end
    return s
end

#endregion

#endregion

end # module

const GS = GreenSolvers

