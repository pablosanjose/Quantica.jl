############################################################################################
# GreenSolvers module
#region

module GreenSolvers

using LinearAlgebra: I, lu, ldiv!, mul!
using SparseArrays
using SparseArrays: SparseMatrixCSC, AbstractSparseMatrix
using Quantica: Quantica, AbstractAppliedGreenSolver, Hamiltonian, Flat,
      ParametricHamiltonian, AbstractHamiltonian, OrbitalStructure,
      harmonics, dcell, SA, orbitalstructure, merged_flatten_mul!, flatten, store_onsites,
      add_diagonal!, zerocell, filltuple, siteindices
import Quantica: call!, apply

abstract type AbstractGreenSolver end

############################################################################################
# Locations for GreenSolvers
#region

# struct Locations{N<:NamedTuple}
#     options::N
# end

# Locations(; kw...) = Locations((; kw...))

# struct Subcell{L}
#     cell::SVector{L,Int}
#     sites::Vector{Int}
# end

# struct AppliedLocations{L,B<:NTuple{L,Any}}
#     boundaries::B
#     cellsites::Vector{Subcell{L}}
# end

# apply(l::Locations, h::AbstractHamiltonian{<:Any,<:Any,L}) where {L} =
#     AppliedLocations{L}(h; l.options...)

# function AppliedLocations{L}(h; boundaries = filltuple(0, Val(L)), cells = (zerocell(Val(L)),), kw...) where {L}
#     siteindices()
# end

#endregion

############################################################################################
# Schur
#region

const AbstractHamiltonian1D = AbstractHamiltonian{<:Any,<:Any,1}

struct Schur{N<:NamedTuple} <: AbstractGreenSolver
    options::N
end

# Schur(; inversefree = true, shift = 1.0, kw...) = Schur((; inversefree, shift, locations = Locations(; kw)))
Schur(; inversefree = true, shift = 1.0, kw...) = Schur((; inversefree, shift))

Schur(h::AbstractHamiltonian1D; kw...) = apply(Schur(; kw...), h)
Schur(h::AbstractHamiltonian; kw...) = throw(ArgumentError("The Schur method requires 1D AbstractHamiltonians with 0, ±1 as only Bloch Harmonics."))

#### FlatHarmonics #########################################################################

struct FlatHarmonics{C<:Number,H<:AbstractHamiltonian1D}
    h::H
    flatorbstruct::OrbitalStructure{C}
    h0::SparseMatrixCSC{C,Int}  # These will be an alias for the internal harmonics of h
    hp::SparseMatrixCSC{C,Int}  # if h is already flat
    hm::SparseMatrixCSC{C,Int}  # if h is already flat
    function FlatHarmonics{C,H}(h, flatorbstruct, h0, hp) where {C<:Number, H<:AbstractHamiltonian1D}
        for hh in harmonics(h)
            dn = dcell(hh)
            dn == SA[0] || dn == SA[1] || dn == SA[-1] ||
                throw(ArgumentError("Too many harmonics, try `supercell`."))
        end
        return new(h, flatorbstruct, h0, hp, hm)
    end
end

## Constructors ##

FlatHarmonics(h::H, flatos::OrbitalStructure{C}, h0, hp, hm) where {H,C} =
    FlatHarmonics{C,H}(h, flatos, h0, hp, hm)

function flatharmonics(h::AbstractHamiltonian1D)
    h´ = store_onsites(h)  # ensure structural diagonal to shift by ω later
    os = orbitalstructure(h´)
    flatos = flatten(os)
    h0 = flatten(h´[(0,)], os, flatos)
    hp = flatten(h´[(1,)], os, flatos)
    hm = flatten(h´[(-1,)], os, flatos)
    return FlatHarmonics(h´, flatos, h0, hp, hm)
end

flatharmonics(h::Flat) = flatharmonics(parent(h))

## API ##

function call!(f::FlatHarmonics; kw...)
    call!(f.h; kw...)  # updates internal harmonics if h::ParametricHamiltonian, or no-op
    copy_flat!(f)
    return f.h0, f.hp, f.hm
end

function copy_flat!(f::FlatHarmonics)
    os = orbitalstructure(f.h)
    flatos = f.flatorbstruct
    merged_flatten_mul!(f.h0, (os, flatos), f.h[SA[0]], 1, 1, 0)
    merged_flatten_mul!(f.hp, (os, flatos), f.h[SA[1]], 1, 1, 0)
    merged_flatten_mul!(f.hm, (os, flatos), f.h[SA[-1]], 1, 1, 0)
    return f
end

#### AppliedSchurGreenSolver ##########################################################################

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

struct AppliedSchurGreenSolver{C<:Number,F<:FlatHarmonics{C},N<:NamedTuple} <: AbstractAppliedGreenSolver
    options::N
    flathars::F
    l_leq_r::Bool                           # whether l <= r
    ptrsDS::Tuple{Vector{Int},Vector{Int}}  # ptrs of h0 on diagonal and Σₐᵤₓ surface
    L::Matrix{C}
    R::Matrix{C}
    LmR´::Matrix{C}  # [L'; -R']
    tmp::SchurWorkspace{C}
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

function apply(s::Schur, h::AbstractHamiltonian)
    options = s.options
    flathars = flatharmonics(h)
    # we only need their sparse structure; it does not change upon call!(flathars, ω; kw...)
    h0, hp, hm = flathars.h0, flathars.hp, flathars.hm
    L, R, Σinds, l_leq_r = left_right_projectors(hp, hm)
    LmR´ = [L'; -R']
    Sptrs = sizehint!(Int[], length(Σinds))
    diagonal_pointers!(Sptrs, h0, Σinds)
    Dptrs = sizehint!(Int[], size(h0, 1))
    diagonal_pointers!(Dptrs, h0)
    C = eltype(h0)
    workspace = SchurWorkspace{C}(size(L))
    return AppliedSchurGreenSolver(options, flathars, l_leq_r, (Dptrs, Sptrs), L, R, LmR´, workspace)
end

# hp = L*R' = PL H' PR'. We assume hm = hp'
function left_right_projectors(hp::SparseMatrixCSC, hm::SparseMatrixCSC)
    linds = unique!(sort!(copy(rowvals(hp))))
    rinds = Int[]
    for col in axes(hp, 2)
        isempty(nzrange(hp, col)) || push!(rinds, col)
    end
    # dense projectors
    o = one(eltype(hp))*I
    allrows = 1:size(hp,1)
    l_leq_r = length(linds) <= length(rinds)
    ## THIS IS WRONG: hm is only updated when calling the Green function.
    if l_leq_r
        Σinds = linds
        R = Matrix(hm[:, linds])  # R = PR H = hm PL
        L = o[allrows, linds]     # L = PL
    else
        Σinds = rinds
        R = o[allrows, rinds]     # R = PR
        L = Matrix(hp[:, rinds])  # L = PL H' = hp PR
    end
    return L, R, Σinds, l_leq_r
end

# Compute G*R and G*L where G = inv(ω - h0 - im*Ω(LL' + RR'))
# Pencil A - λB :
#    A = [1-im*L'GL  -L'GLhp; im*R'GL  R'GLhp] and B = [L'GRhp' im*L'GR; -R'GRhp' 1-im*R'GR]
#    A = [-im*Γₗ -Γₗhp] + [1 0; 0 0] and B = [Γᵣhp' im*Γᵣ] + [0 0; 0 1]
#    Γₗ = [L'; -R']*G*L  and  Γᵣ = [L'; -R']*G*R
function pencilAB(s::AppliedSchurGreenSolver, ω; kw...)
    h0, hp, hm = call!(s.flathars; kw...)
    Hp = s.tmp.Hp
    copyto!(Hp, nonzeros(hp))
    Ω = s.options.shift
    iG = inverse_G!(h0, ω, Ω, s.ptrsDS)
    iGlu = lu(iG)
    GL = ldiv!(s.tmp.GL, iGlu, s.L)
    GR = ldiv!(s.tmp.GR, iGlu, s.R)
    l, r = size(GL, 2), size(GR, 2)
    A, B = s.tmp.A, s.tmp.B
    mul!(view(A, :, 1:l), s.LmR´, GL, -1, 0)
    mul!(view(B, :, l+1:l+r), s.LmR´, GR, 1, 0)
    mul!(view(A, :, l+1:l+r), view(A, :, 1:l), Hp, 1, 0)
    mul!(view(B, :, 1:l), view(A, :, 1:l), Hp', 1, 0)
    view(A, :, 1:l) .*= im * Ω
    view(B, :, l+1:l+r) .*= im * Ω
    add_diagonal!(A, 1, 1:l)
    add_diagonal!(B, 1, l+1:l+r)
    return A, B
end

# Builds ω - h0 - iΩ(LL' + RR')
function inverse_G!(h0, ω, Ω, (ptrsD, ptrsS))
    nz = nonzeros(h0)
    @. nz = -nz
    for ptr in ptrsD
        nz[ptr] += ω
    end
    for ptr in ptrsS
        nz[ptr] += Ω*im
    end
    return h0
end

# returns invariant subspaces of retarded and advanced eigenmodes
function mode_subspaces(A, B)


end

# # returns invariant subspaces of retarded and advanced eigenmodes
# function mode_subspaces(s::Schur1DGreensSolver, A::AbstractArray{T}, B::AbstractArray{T}, imω) where {T}
#     ZrL, ZrR, ZaL, ZaR = s.tmp.rr1, s.tmp.rr2, s.tmp.rr3, s.tmp.rr4
#     r = size(A, 1) ÷ 2
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
end # module

const GS = GreenSolvers

#endregion