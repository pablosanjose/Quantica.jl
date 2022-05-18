############################################################################################
# GreenSolvers module
#region

module GreenSolvers

using LinearAlgebra: I, lu, ldiv!, mul!
using SparseArrays
using SparseArrays: SparseMatrixCSC, AbstractSparseMatrix
using Quantica: Quantica, AbstractGreenSolver, AbstractAppliedGreenSolver, Hamiltonian,
      ParametricHamiltonian, AbstractHamiltonian, BlockStructure, HybridSparseMatrixCSC,
      lattice, zerocell, SVector, sanitize_SVector, siteselector, foreach_cell, foreach_site
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
    g0inv::SparseMatrixCSC{Complex{T},Int}            # to store ω - h0 - Σₐᵤₓ
    ptrs::Tuple{Vector{Int},Vector{Int},Vector{Int}}  # g0inv ptrs for h0 nzvals, diagonal and Σₐᵤₓ surface
    PL::Matrix{Complex{T}}                            # projector on left surface
    PR::Matrix{Complex{T}}                            # projector on right surface
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

## apply ##

apply(s::Schur, h::AbstractHamiltonian) =
    argerror("The Schur method requires 1D AbstractHamiltonians with 0, ±1 as only Bloch Harmonics.")

function apply(s::Schur, h::AbstractHamiltonian1D{T}) where {T}
    options = s.options
    hm, h0, hp = hybridharmonics(h)
    fhm, fh0, fhp = flat(hm), flat(h0), flat(hp)
    PL, PR, L, R, Σauxinds, l_leq_r = left_right_projectors(fhm, fhp)
    R´L´ = [R'; -L']
    g0inv, ptrs = g0inv_pointers(fh0, Σauxinds)
    workspace = SchurWorkspace{Complex{T}}(size(L))
    return AppliedSchur(options, h, hm, h0, hp, l_leq_r, g0inv, ptrs, PL, PR, L, R, R´L´, workspace)
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
        Σauxinds = linds
        R = Matrix(hm[:, linds])  # R = PR H = hm PL
        L = PL
    else
        Σauxinds = rinds
        R = PR
        L = Matrix(hp[:, rinds])  # L = PL H' = hp PR
    end
    return PL, PR, L, R, Σauxinds, l_leq_r
end

function g0inv_pointers(fh0, Σauxinds)
    g0inv = I - fh0
    
    for col in axes(g0inv, 2), p in nzrange(g0inv, col)

end

# # Compute G*R and G*L where G = inv(ω - h0 - im*Ω(LL' + RR'))
# # Pencil A - λB :
# #    A = [1-im*L'GL  -L'GLhp; im*R'GL  R'GLhp] and B = [L'GRhp' im*L'GR; -R'GRhp' 1-im*R'GR]
# #    A = [-im*Γₗ -Γₗhp] + [1 0; 0 0] and B = [Γᵣhp' im*Γᵣ] + [0 0; 0 1]
# #    Γₗ = [L'; -R']*G*L  and  Γᵣ = [L'; -R']*G*R
# function pencilAB(s::AppliedSchur, ω; kw...)
#     h0, hp, hm = call!(s.flathars; kw...)
#     Hp = s.tmp.Hp
#     copyto!(Hp, nonzeros(hp))
#     Ω = s.options.shift
#     iG = inverse_G!(h0, ω, Ω, s.ptrsDS)
#     iGlu = lu(iG)
#     GL = ldiv!(s.tmp.GL, iGlu, s.L)
#     GR = ldiv!(s.tmp.GR, iGlu, s.R)
#     l, r = size(GL, 2), size(GR, 2)
#     A, B = s.tmp.A, s.tmp.B
#     mul!(view(A, :, 1:l), s.LmR´, GL, -1, 0)
#     mul!(view(B, :, l+1:l+r), s.LmR´, GR, 1, 0)
#     mul!(view(A, :, l+1:l+r), view(A, :, 1:l), Hp, 1, 0)
#     mul!(view(B, :, 1:l), view(A, :, 1:l), Hp', 1, 0)
#     view(A, :, 1:l) .*= im * Ω
#     view(B, :, l+1:l+r) .*= im * Ω
#     add_diagonal!(A, 1, 1:l)
#     add_diagonal!(B, 1, l+1:l+r)
#     return A, B
# end

# # Builds ω - h0 - iΩ(LL' + RR')
# function inverse_G!(h0, ω, Ω, (ptrsD, ptrsS))
#     nz = nonzeros(h0)
#     @. nz = -nz
#     for ptr in ptrsD
#         nz[ptr] += ω
#     end
#     for ptr in ptrsS
#         nz[ptr] += Ω*im
#     end
#     return h0
# end

# # returns invariant subspaces of retarded and advanced eigenmodes
# function mode_subspaces(A, B)


# end

# # # returns invariant subspaces of retarded and advanced eigenmodes
# # function mode_subspaces(s::Schur1DGreensSolver, A::AbstractArray{T}, B::AbstractArray{T}, imω) where {T}
# #     ZrL, ZrR, ZaL, ZaR = s.tmp.rr1, s.tmp.rr2, s.tmp.rr3, s.tmp.rr4
# #     r = size(A, 1) ÷ 2
# #     if !iszero(r)
# #         sch = schur!(A, B)
# #         # Retarded modes
# #         whichmodes = Vector{Bool}(undef, length(sch.α))
# #         retarded_modes!(whichmodes, sch, imω)
# #         ordschur!(sch, whichmodes)
# #         copy!(ZrL, view(sch.Z, 1:r, 1:sum(whichmodes)))
# #         copy!(ZrR, view(sch.Z, r+1:2r, 1:sum(whichmodes)))
# #         # Advanced modes
# #         advanced_modes!(whichmodes, sch, imω)
# #         ordschur!(sch, whichmodes)
# #         copy!(ZaL, view(sch.Z, 1:r, 1:sum(whichmodes)))
# #         copy!(ZaR, view(sch.Z, r+1:2r, 1:sum(whichmodes)))
# #     end
# #     return ZrL, ZrR, ZaL, ZaR
# # end
end # module

const GS = GreenSolvers

#endregion