############################################################################################
# SchurFactorsSolver - see scattering.pdf notes for derivations
#   Auxiliary functions for AppliedSchurGreenSolverSolver
#   Computes dense factors PR*R*Z21, Z11 and R'*PR'. The retarded self-energy on the open
#   unitcell surface of a semi-infinite rightward lead reads Σᵣ = PR R Z21 Z11⁻¹ R' PR'
#   Computes also the leftward PL*L*Z21´, Z11´, L'*PL', with  Σₗ = PL L Z21´ Z11´⁻¹ L' PL'
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
    LD::Matrix{C}
    DL::Matrix{C}
    RD::Matrix{C}
    DR::Matrix{C}
end

struct SchurFactorsSolver{T,B}
    shift::T                                          # called Ω in the scattering.pdf notes
    hm::HybridSparseBlochMatrix{T,B}
    h0::HybridSparseBlochMatrix{T,B}
    hp::HybridSparseBlochMatrix{T,B}
    l_leq_r::Bool                                     # whether l <= r (left and right surface dims)
    iG::SparseMatrixCSC{Complex{T},Int}               # to store iG = ω - h0 - Σₐᵤₓ
    ptrs::Tuple{Vector{Int},Vector{Int},Vector{Int}}  # iG ptrs for h0 nzvals, diagonal and Σₐᵤₓ surface
    linds::Vector{Int}                                # orbital indices on left surface
    rinds::Vector{Int}                                # orbital indices on right surface
    sinds::Vector{Int}                                # orbital indices on the smallest surface (left for l<=r, right for l>r)
    L::Matrix{Complex{T}}                             # l<=r ? PL : PL*H' === hp PR  (n × min(l,r))
    R::Matrix{Complex{T}}                             # l<=r ? PR*H === hm PL : PR   (n × min(l,r))
    R´L´::Matrix{Complex{T}}                          # [R'; -L']. L and R must be dense for iG \ (L,R)
    tmp::SchurWorkspace{Complex{T}}
end

#region ## Constructors ##

SchurFactorsSolver(::AbstractHamiltonian, _) =
    argerror("The Schur solver requires 1D Hamiltonians with 0 and ±1 as only Bloch Harmonics.")

function SchurFactorsSolver(h::Hamiltonian{T,<:Any,1}, shift = one(Complex{T})) where {T}
    hm, h0, hp = nearest_cell_harmonics(h)
    fhm, fh0, fhp = flat(hm), flat(h0), flat(hp)
    # h*'s may be updated after flat but only fh* structure matters
    linds, rinds, L, R, sinds, l_leq_r = left_right_projectors(fhm, fhp)
    R´L´ = [R'; -L']
    iG, (p, pd) = store_diagonal_ptrs(fh0)
    ptrs = (p, pd, pd[sinds])
    workspace = SchurWorkspace{Complex{T}}(size(L), length(linds), length(rinds))
    return SchurFactorsSolver(shift, hm, h0, hp, l_leq_r, iG, ptrs, linds, rinds, sinds, L, R, R´L´, workspace)
end

function SchurWorkspace{C}((n, d), l, r) where {C}
    GL = Matrix{C}(undef, n, d)
    GR = Matrix{C}(undef, n, d)
    A = Matrix{C}(undef, 2d, 2d)
    B = Matrix{C}(undef, 2d, 2d)
    Z11 = Matrix{C}(undef, d, d)
    Z21 = Matrix{C}(undef, d, d)
    Z11´ = Matrix{C}(undef, d, d)
    Z21´ = Matrix{C}(undef, d, d)
    LD = Matrix{C}(undef, l, d)
    DL = Matrix{C}(undef, d, l)
    RD = Matrix{C}(undef, r, d)
    DR = Matrix{C}(undef, d, r)
    return SchurWorkspace(GL, GR, A, B, Z11, Z21, Z11´, Z21´, LD, DL, RD, DR)
end

function nearest_cell_harmonics(h)
    for hh in harmonics(h)
        dn = dcell(hh)
        dn == SA[0] || dn == SA[1] || dn == SA[-1] ||
            argerror("Too many harmonics, try `supercell` to reduce to strictly nearest-cell harmonics.")
    end
    hm, h0, hp = h[-1], h[0], h[1]
    flat(hm) == flat(hp)' ||
        argerror("The Hamiltonian should have h[1] = h[-1]' to use the Schur solver")
    return hm, h0, hp
end

# hp = L*R' = PL H' PR'. We assume hm = hp'
function left_right_projectors(hm::SparseMatrixCSC, hp::SparseMatrixCSC)
    linds = stored_cols(hm)
    rinds = stored_cols(hp)
    # dense projectors
    o = one(eltype(hp)) * I
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
    return linds, rinds, L, R, sinds, l_leq_r
end

# Build a new sparse matrix mat´ with same structure as mat plus the diagonal
# return also: (1) ptrs to mat´ for each nonzero in mat, (2) diagonal ptrs in mat´
function store_diagonal_ptrs(mat::SparseMatrixCSC{T}) where {T}
    # same structure as mat + I, but avoiding accidental cancellations
    # (note that the nonzeros of G⁻¹ = mat´ will be overwritten by update_iG! before use)
    mat´ = mat + Diagonal(iszero.(diag(mat)))
    pmat, pdiag = Int[], Int[]
    rows, rows´ = rowvals(mat), rowvals(mat´)
    for col in axes(mat´, 2)
        ptrs = nzrange(mat, col)
        ptrs´ = nzrange(mat´, col)
        p, p´ = first(ptrs), first(ptrs´)
        while p´ in ptrs´
            row´ = rows´[p´]
            row´ == col && push!(pdiag, p´)
            if p in ptrs && row´ == rows[p]
                push!(pmat, p´)
                p += 1
            end
            p´ += 1
        end
    end
    return mat´, (pmat, pdiag)
end

#endregion

#region ## API ##

## Call API ##

# # minimal_callsafe_copy should copy anything that may mutate with call!(...; params...) and
# # any object in the output of call!(::SchurFactorsSolver, ...; params...)
# function minimal_callsafe_copy(s::SchurFactorsSolver)
#     hm, h0, hp = minimal_callsafe_copy(s.hm), minimal_callsafe_copy(s.h0), minimal_callsafe_copy(s.hp)
#     L, R = copy(s.L), copy(s.R)
#     tmp = minimal_callsafe_copy(s.tmp)
#     return SchurFactorsSolver(s.shift, hm, h0, hp, s.l_leq_r, s.iG, s.ptrs, s.sinds,
#                               L, R, s.R´L´, tmp)
# end


# function minimal_callsafe_copy(w::SchurWorkspace)
#     Z11, Z21, Z11´, Z21´ = copy(w.Z11), copy(w.Z21), copy(w.Z11´), copy(w.Z21´)
#     return SchurWorkspace(w.GL, w.GR, w.A, w.B, Z11, Z21, Z11´, Z21´)
# end

call!_output(s::SchurFactorsSolver) =
    (s.tmp.RD, s.tmp.Z11, s.tmp.DR), (s.tmp.LD, s.tmp.Z11´, s.DL)

function call!(s::SchurFactorsSolver, ω)
    R, Z11, Z21, L, Z11´, Z21´ = s.R, s.tmp.Z11, s.tmp.Z21, s.L, s.tmp.Z11´, s.tmp.Z21´
    update_LR!(s)     # We must update L, R in case a parametric parent has been call!-ed
    update_iG!(s, ω)  # also iG = ω - h0 + iΩP'P

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

    RZ21, LZ21´, LD, DL, RD, DR = s.tmp.GR, s.tmp.GL, s.tmp.LD, s.tmp.DL, s.tmp.RD, s.tmp.DR
    linds, rinds = s.linds, s.rinds
    mul!(RZ21, R, Z21)
    PR_R_Z21 = copy!(RD, view(RZ21, rinds, :))
    R´_PR = copy!(DR, view(R', :, rinds))
    mul!(LZ21´, L, Z21´)
    PL_L_Z21´ = copy!(LD, view(LZ21´, linds, :))
    L´_PL = copy!(DL, view(L', :, linds))

    return (PR_R_Z21, Z11, R´_PR), (PL_L_Z21´, Z11´, L´_PL)
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

# updates L and R from the current hm and hp
function update_LR!(s)
    d = size(s.L, 2)
    if s.l_leq_r
        # slicing is faster than a view of sparse
        copy!(s.R, flat(s.hm)[:, s.sinds])
        view(s.R´L´, 1:d, :) .= s.R'
    else
        # slicing is faster than a view of sparse
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

############################################################################################
# SelfEnergySchurSolver <: ExtendedSelfEnergySolver <: AbstractSelfEnergySolver
#region

struct SelfEnergySchurSolver{T,B,M<:BlockSparseMatrix} <: ExtendedSelfEnergySolver
    fsolver::SchurFactorsSolver{T,B}
    leftside::Bool
end

#region ## Constructors ##

SelfEnergySchurSolver(fsolver::SchurFactorsSolver, side::Symbol) =
    SelfEnergySchurSolver(fsolver, isleftside(side))

function isleftside(side)
    if side == :R
        return false
    elseif side == :L
        return true
    else
        argerror("Unexpeced side = $side in SelfEnergySchurSolver. Only :L and :R are allowed.")
    end
end

#endregion

#region ## API ##

# This solver produces two solutions (L/R) for the price of one. We can opt out of calling
# it if we know it has already been called, so the solution is already in its call!_output
function call!(s::SelfEnergySchurSolver, ω;
               skipsolve_internal = false, params...)
    fsolver = s.fsolver
    Rfactors, Lfactors = skipsolve_internal ? call!_output(fsolver) : call!(fsolver, ω)
    return ifelse(leftside_internal, Lfactors, Rfactors)
end

#endregion

#endregion top

############################################################################################
# AppliedSchurGreenSolverSolver
#region

struct AppliedSchurGreenSolver{T,G<:GreenFunction{T,<:Any,0}} <: AppliedGreenSolver
    boundary::T
    gR::G           # GreenFunction for unitcell with ΣR
    gL::G           # GreenFunction for unitcell with ΣL
    g∞::G           # GreenFunction for unitcell with ΣR + ΣL
end

#region ## apply ##

function apply(s::GS.Schur, h::AbstractHamiltonian1D, contacts::Contacts)
    h´ = hamiltonian(h)
    fsolver = SchurFactorsSolver(h´, s.shift)
    ΣR_solver = SelfEnergySchurSolver(fsolver, :R)
    ΣL_solver = SelfEnergySchurSolver(fsolver, :L)
    h0 = unitcell_hamiltonian(h)
    rsites = stored_cols(unflat(h[1]))
    lsites = stored_cols(unflat(h[-1]))
    latslice_l = lattice(h0)[cellsites((), lsites)]
    latslice_r = lattice(h0)[cellsites((), rsites)]
    ΣL = SelfEnergy(ΣL_solver, latslice_l)
    ΣR = SelfEnergy(ΣR_solver, latslice_r)
    ohR = attach(h0, ΣR)
    gR = greenfunction(ohR, GS.SparseLU())
    ohL = attach(h0, ΣL)
    gL = greenfunction(ohL, GS.SparseLU())
    oh∞ = attach(ohL, ΣR)
    g∞ = greenfunction(oh∞, GS.SparseLU())
    boundary = round(only(s.boundary))
    return AppliedSchurGreenSolver(boundary, gR, gL, g∞)
end

#endregion

#region ## call API ##

minimal_callsafe_copy(s::AppliedSchurGreenSolver) = AppliedSchurGreenSolver(
    s.boundary, minimal_callsafe_copy.((s.gr, s.gl, s.g∞))...)

function (s::AppliedSchurGreenSolver)(ω, Σblocks, cblockstruct)
    gRω = call!(s.gR, ω)
    gLω = call!(s.gL, ω; skipsolve_internal = true) # already solved in gRω (aliased)
    g∞ω = call!(s.gL, ω; skipsolve_internal = true) # already solved in gRω (aliased)
    g0 = SchurGreenSlicer(gRω, gLω, g∞ω, s.boundary)
    slicer = TMatrixSlicer(g0, Σblocks, cblockstruct)
    return slicer
end

#endregion

############################################################################################
# SchurGreenSlicer
#   Slicer for a 1D lead, with or without a single boundary
#   Semiinifinite:
#       Gₙₘ = (Ghⁿ⁻ᵐ - GhⁿGh⁻ᵐ)G∞₀₀ = G∞ₙₘ - GhⁿG∞₀ₘ
#       Gₙₘ = G∞ₙₘ - G₁₁L(R'G₁₁L)ⁿ⁻¹ R'G∞₀ₘ       for n > 1
#       Gₙₘ = G∞ₙₘ - G₋₁₋₁R(L'G₋₁₋₁R)⁻ⁿ⁻¹L'G∞₀ₘ   for n < -1
#region

struct SchurGreenSlicer{T}  <: GreenSlicer
    
end

#endregion
