############################################################################################
# GreenSolvers module
#region

module GreenSolvers

using LinearAlgebra: I, lu, ldiv!, mul!
using SparseArrays
using SparseArrays: SparseMatrixCSC, AbstractSparseMatrix
using Quantica: Quantica, AbstractGreenSolver, AbstractAppliedGreenSolver, Hamiltonian,
      ParametricHamiltonian, AbstractHamiltonian, BlockStructure, HybridSparseMatrixCSC,
      lattice, zerocell, SVector, sanitize_SVector, siteselector, foreach_cell, foreach_site,
      store_diagonal_ptrs
import Quantica: call!, apply

############################################################################################
# Locations where GreenSolvers should be applied
#region

struct Locations{N<:NamedTuple}
    options::N
end

Locations(; kw...) = Locations((; kw...))

struct CellSites{L}
    cell::SVector{L,Int}
    sites::Vector{Int}
end

# Encode boundaries as floats to allow Inf and distant boundaries
struct AppliedLocations{T<:AbstractFloat,L}
    boundaries::SVector{L,T}
    scells::Vector{CellSites{L}}
end

apply(l::Locations, h::AbstractHamiltonian{T,<:Any,L}) where {T,L} =
    AppliedLocations{T,L}(h; l.options...)

function AppliedLocations{T,L}(h; boundaries = missing, kw...) where {T,L}
    boundaries´ = boundaries === missing ? sanitize_SVector(T, ntuple(Returns(Inf), Val(L))) :
                                           sanitize_SVector(T, boundaries)

    asel = apply(siteselector(; kw...), lattice(h))
    scells = CellSites{L}[]
    foreach_cell(asel) do cell
        sites = Int[]
        foreach_site(asel, cell) do s, i, r
            push!(sites, i)
        end
        found = !isempty(sites)
        found && push!(scells, CellSites(cell, sites))
        return found
    end
    return AppliedLocations(boundaries´, scells)
end

#endregion

############################################################################################
# Schur  - see scattering.pdf notes for derivations
#region

const AbstractHamiltonian1D{T,E,B} = AbstractHamiltonian{T,E,1,B}

struct Schur{N<:NamedTuple} <: AbstractGreenSolver
    options::N
end

Schur(; inversefree = true, shift = 1.0, kw...) = Schur((; inversefree, shift, locations = Locations(; kw)))

Schur(h::AbstractHamiltonian; kw...) = apply(Schur(; kw...), h)

#### AppliedSchur ##########################################################################

struct SchurWorkspace{C}
    GL::Matrix{C}
    GR::Matrix{C}
    A::Matrix{C}
    B::Matrix{C}
    Z11::Matrix{C}
    Z21::Matrix{C}
    Z11bar::Matrix{C}
    Z21bar::Matrix{C}
end

struct AppliedSchur{T,B,H<:AbstractHamiltonian1D{T,<:Any,B},N<:NamedTuple} <: AbstractAppliedGreenSolver
    options::N
    h::H
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

## Constructors ##

function SchurWorkspace{C}((n, d)) where {C}
    GL = Matrix{C}(undef, n, d)
    GR = Matrix{C}(undef, n, r)
    A = Matrix{C}(undef, 2d, 2d)
    B = Matrix{C}(undef, 2d, 2d)
    Z11 = Matrix{C}(undef, d, d)
    Z21 = Matrix{C}(undef, d, d)
    Z11bar = Matrix{C}(undef, d, d)
    Z21bar = Matrix{C}(undef, d, d)
    return SchurWorkspace(GL, GR, A, B, Z11, Z21, Z11bar, Z21bar)
end

## Apply ##

apply(::Schur, ::AbstractHamiltonian) =
    argerror("The Schur method requires 1D AbstractHamiltonians with 0, ±1 as only Bloch Harmonics.")

function apply(s::Schur, h::AbstractHamiltonian1D{T}) where {T}
    options = s.options
    hm, h0, hp = hybridharmonics(h)
    fhm, fh0, fhp = flat(hm), flat(h0), flat(hp)
    L, R, sinds, l_leq_r = left_right_projectors(fhm, fhp)
    R´L´ = [R'; -L']
    iG, (p, pd) = store_diagonal_ptrs(fh0)
    ptrs = (p, pd, pd[sinds])
    workspace = SchurWorkspace{Complex{T}}(size(L))
    return AppliedSchur(options, h, hm, h0, hp, l_leq_r, iG, ptrs, sinds, L, R, R´L´, workspace)
end

function hybridharmonics(h)
    for hh in harmonics(h)
        dn = dcell(hh)
        dn == SA[0] || dn == SA[1] || dn == SA[-1] ||
            throw(ArgumentError("Too many harmonics, try `supercell` to reduce to strictly nearest-neighbors."))
    end
    return h[-1], h[0], h[1]
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

## Pencil ##

# Compute G*R and G*L where G = inv(ω - h0 - Σₐᵤₓ)
# Pencil A - λB :
#    A = [R'GL  (1-δ)iΩR'GL; -L'GL  1-(1-δ)iΩL´GL] and B = [1-δiΩR'GR  -R'GR; δiΩL'GR  L'GR]
#    where δ = l <= r ? 1 : 0
#    A = [Γₗ (1-δ)iΩΓₗ] + [0 0; 0 1] and B = [-Γᵣ -δiΩΓᵣ] + [1 0; 0 0]
#    Γₗ = [R'; -L']GL  and  Γᵣ = [R'; -L']GR
function pencilAB(s::AppliedSchur{T}, ω) where {T}
    o, z = one(Complex{T}), zero(Complex{T})
    update_iG!(s, ω)
    iGlu = lu(s.iG)
    Ω = s.options.shift
    update_LR!(s)
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

function update_iG!(s::AppliedSchur{T}, ω) where {T}
    Ω = s.options.shift
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

# ## Self-energy of unit cell ##

# function selfenergy(s::AppliedSchur, ω; kw...)
#     # Update harmonics (aliased in s.h0, s.hm, s.hp)
#     update_h!(s.h, ω; kw...)
#     Z11, Z21, Z11bar, Z21bar = s.tmp.Z11, s.tmp.Z21, s.tmp.Z11bar, s.tmp.Z21bar
#     d = size(s.L, 2)
#     if !iszero(r)
#         sch = schur!(A, B)
#         # Retarded modes
#         whichmodes = Vector{Bool}(undef, length(sch.α))
#         retarded_modes!(whichmodes, sch, imω)
#         ordschur!(sch, whichmodes)
#         copy!(ZrL, view(sch.Z, 1:r, 1:sum(whichmodes)))
#         copy!(ZrR, view(sch.Z, r+1:2r, 1:sum(whichmodes)))
#         # Advanced modes
#         advanced_modes!(whichmodes, sch, imω)
#         ordschur!(sch, whichmodes)
#         copy!(ZaL, view(sch.Z, 1:r, 1:sum(whichmodes)))
#         copy!(ZaR, view(sch.Z, r+1:2r, 1:sum(whichmodes)))
#     end
#     return ZrL, ZrR, ZaL, ZaR
# end

# update_h!(h::Hamiltonian, ω; kw...) = h
# update_h!(h::ParametricHamiltonian, ω; kw...) = call!(h; kw...)

#endregion

end # module

const GS = GreenSolvers

