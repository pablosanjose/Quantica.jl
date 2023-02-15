############################################################################################
# SchurFactorsSolver - see scattering.pdf notes for derivations
#   Auxiliary functions for AppliedSchurGreenSolverSolver
#   Computes dense factors PR*R*Z21, Z11 and R'*PR'. The retarded self-energy on the open
#   unitcell surface of a semi-infinite rightward lead reads Σᵣ = PR R Z21 Z11⁻¹ R' PR'
#   Computes also the leftward PL*L*Z11´, Z21´, L'*PL', with  Σₗ = PL L Z11´ Z21´⁻¹ L' PL'
#region

struct SchurWorkspace{C}
    GL::Matrix{C}
    GR::Matrix{C}
    LG::Matrix{C}
    RG::Matrix{C}
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
    LG = Matrix{C}(undef, d, n)
    RG = Matrix{C}(undef, d, n)
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
    return SchurWorkspace(GL, GR, LG, RG, A, B, Z11, Z21, Z11´, Z21´, LD, DL, RD, DR)
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

call!_output(s::SchurFactorsSolver) =
    (s.tmp.RD, s.tmp.Z11, s.tmp.DR), (s.tmp.LD, s.tmp.Z21´, s.tmp.DL)

function call!(s::SchurFactorsSolver, ω)
    R, Z11, Z21, L, Z11´, Z21´ = s.R, s.tmp.Z11, s.tmp.Z21, s.L, s.tmp.Z11´, s.tmp.Z21´
    update_LR!(s)     # We must update L, R in case a parametric parent has been call!-ed
    update_iG!(s, ω)  # also iG = ω - h0 + iΩP'P

    A, B = pencilAB!(s)
    sch = schur!(A, B)
    whichmodes = Vector{Bool}(undef, length(sch.α))
    r = size(A, 1) ÷ 2

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

    RZ21, LZ11´, LD, DL, RD, DR = s.tmp.GR, s.tmp.GL, s.tmp.LD, s.tmp.DL, s.tmp.RD, s.tmp.DR
    linds, rinds = s.linds, s.rinds
    # compute rightward blocks: PR*R*Z21, Z11 and R'*PR'
    mul!(RZ21, R, Z21)
    PR_R_Z21 = copy!(RD, view(RZ21, rinds, :))
    R´_PR = copy!(DR, view(R', :, rinds))
    # compute leftward blocks: PL*L*Z11´, Z21´, L'*PL'
    mul!(LZ11´, L, Z11´)
    PL_L_Z11´ = copy!(LD, view(LZ11´, linds, :))
    L´_PL = copy!(DL, view(L', :, linds))

    return (PR_R_Z21, Z11, R´_PR), (PL_L_Z11´, Z21´, L´_PL)
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
    d = size(s.L, 2)
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

struct SelfEnergySchurSolver{T,B} <: ExtendedSelfEnergySolver
    fsolver::SchurFactorsSolver{T,B}
    leftside::Bool
end

#region ## Constructors ##

SelfEnergySchurSolver(fsolver::SchurFactorsSolver, side::Symbol) =
    SelfEnergySchurSolver(fsolver, isleftside(side))

function isleftside(side)
    if side == :L
        return true
    elseif side == :R
        return false
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
    return ifelse(s.leftside, Lfactors, Rfactors)
end

call!_output(s::SelfEnergySchurSolver) = call!(s, 0.0; skipsolve_internal = true)

#endregion

#endregion top

############################################################################################
# AppliedSchurGreenSolverSolver
#region

const GFUnit{T,E,H,N,S} =
    GreenFunction{T,E,0,AppliedSparseLU{Complex{T}},H,Contacts{0,N,S}}

# With Lazy we delay computing fields until first use
struct AppliedSchurGreenSolver{T,B,G<:GFUnit{T},G∞<:GFUnit{T}} <: AppliedGreenSolver
    boundary::T
    gL::Lazy{G}                    # GreenFunction for unitcell with ΣL
    gR::Lazy{G}                    # GreenFunction for unitcell with ΣR
    g∞::Lazy{G∞}                   # GreenFunction for unitcell with ΣR + ΣL
    fsolver::SchurFactorsSolver{T,B}
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

    ohL = attach(h0, ΣL)
    gL = Lazy{green_type(h0, ΣL)}(() -> greenfunction(ohL, GS.SparseLU()))
    ohR = attach(h0, ΣR)
    gR = Lazy{green_type(h0, ΣR)}(() -> greenfunction(ohR, GS.SparseLU()))
    oh∞ = attach(ohL, ΣR)
    g∞ = Lazy{green_type(h0, ΣL, ΣR)}(() -> greenfunction(oh∞, GS.SparseLU()))

    boundary = round(only(s.boundary))
    return AppliedSchurGreenSolver(boundary, gL, gR, g∞, fsolver)
end

green_type(::H,::S) where {T,E,H<:AbstractHamiltonian{T,E},S} =
    GFUnit{T,E,H,1,Tuple{S}}
green_type(::H,::S1,::S2) where {T,E,H<:AbstractHamiltonian{T,E},S1,S2} =
    GFUnit{T,E,H,2,Tuple{S1,S2}}

#endregion

#region ## call API ##

minimal_callsafe_copy(s::AppliedSchurGreenSolver) = AppliedSchurGreenSolver(
    s.boundary, minimal_callsafe_copy.((s.gr, s.gl, s.g∞))...)

function (s::AppliedSchurGreenSolver)(ω, Σblocks, cblockstruct)
    # call! fsolver once for all the g's
    call!(s.fsolver, ω)
    gLω = Lazy{slicer_type(s.gL)}(() -> slicer(call!(s.gL[], ω; skipsolve_internal = true)))
    gRω = Lazy{slicer_type(s.gR)}(() -> slicer(call!(s.gR[], ω; skipsolve_internal = true)))
    g∞ω = Lazy{slicer_type(s.g∞)}(() -> slicer(call!(s.g∞[], ω; skipsolve_internal = true)))
    g0slicer = SchurGreenSlicer(gLω, gRω, g∞ω, s.boundary, s.fsolver)
    gslicer = TMatrixSlicer(g0slicer, Σblocks, cblockstruct)
    return gslicer
end

slicer_type(::Lazy{<:GFUnit{T}}) where {T} = SparseLUSlicer{Complex{T}}

#endregion

############################################################################################
# SchurGreenSlicer
#   Slicer for a 1D lead using the LR Schur factors, with or without a single boundary
#   For n >= 1:
#       hⁿ ≡ h₊ⁿ = (LR')ⁿ
#       h⁻ⁿ ≡ h₋ⁿ = (RL')ⁿ
#   Infinite lattice:
#       G∞ₙₙ = G∞₀₀ = (ω*I - h0 - ΣR - ΣL)⁻¹
#       G∞ₙₘ = (G₁₁h₊)ⁿ⁻ᵐ G∞₀₀ = G₁₁L (R'G₁₁L)ⁿ⁻ᵐ⁻¹ R'G∞₀₀                  for n-m >= 1
#       G∞ₙₘ = (G₋₁₋₁h₋)ᵐ⁻ⁿ G∞₀₀ = G₋₁₋₁R(L'G₋₁₋₁R)ᵐ⁻ⁿ⁻¹L'G∞₀₀              for n-m <= -1
#   Semiinifinite lattice:
#       Gₙₘ = (Ghⁿ⁻ᵐ - GhⁿGh⁻ᵐ)G∞₀₀ = G∞ₙₘ - GhⁿG∞₀ₘ
#       Gₙₘ = G∞ₙₘ - G₁₁L(R'G₁₁L)ⁿ⁻¹ R'G∞₀ₘ                                 for m,n >= 1
#       Gₙₘ = G∞ₙₘ - G₋₁₋₁R(L'G₋₁₋₁R)¹⁻ⁿL'G∞₀ₘ                              for m,n <= -1
#region

# With Lazy we delay computing fields until first use
struct SchurGreenSlicer{C}  <: GreenSlicer{C}
    L::Matrix{C}
    R::Matrix{C}
    G₋₁₋₁::Lazy{SparseLUSlicer{C}}     # gLω
    G₁₁::Lazy{SparseLUSlicer{C}}       # gRω
    G∞₀₀::Lazy{SparseLUSlicer{C}}      # g∞ω
    L´G∞₀₀::Lazy{Matrix{C}}
    R´G∞₀₀::Lazy{Matrix{C}}
    G₁₁L::Lazy{Matrix{C}}
    G₋₁₋₁R::Lazy{Matrix{C}}
    R´G₁₁L::Lazy{Matrix{C}}
    L´G₋₁₋₁R::Lazy{Matrix{C}}
    boundary::C
end

#region ## Constructors ##

function SchurGreenSlicer(gLω::Lazy{S}, gRω::Lazy{S}, g∞ω::Lazy{S}, boundary, fsolver) where {C,S<:SparseLUSlicer{C}}
    L, R = fsolver.L, fsolver.R
    # temporaries
    gL, gR, L´g, R´g = fsolver.tmp.GL, fsolver.tmp.GR, fsolver.tmp.LG, fsolver.tmp.RG
    G₁₁L   = lazy_ldiv!(gRω, L, gL)
    G₋₁₋₁R = lazy_ldiv!(gLω, R, gR)
    L´G∞₀₀ = lazy_rdiv!(L, g∞ω, L´g)
    R´G∞₀₀ = lazy_rdiv!(R, g∞ω, R´g)

    d = size(L, 2)
    RGL = similar(R, d, d)
    LGR = similar(L, d, d)
    R´G₁₁L = Lazy{Matrix{C}}(() -> mul!(RGL, R', G₁₁L[]))
    L´G₋₁₋₁R = Lazy{Matrix{C}}(() -> mul!(LGR, L', G₋₁₋₁R[]))

    return SchurGreenSlicer(L, R, gLω, gRω, g∞ω, L´G∞₀₀, R´G∞₀₀, G₁₁L, G₋₁₋₁R, R´G₁₁L, L´G₋₁₋₁R, complex(boundary))
end

# note that gLω[].source and gRω[].source are taller than L, R, due to extended sites
# but size(L, 2) = size(R, 2) = min(l, r) = d (deflated surface)
# and size(gLω[].source, 2) = l, size(gRω[].source, 2) = r
function lazy_ldiv!(gω::Lazy{G}, L, gL) where {C,G<:SparseLUSlicer{C}}
    lazy = Lazy{Matrix{C}}() do
        g = gω[]
        Lext = view(g.source, :, axes(L, 2))
        fill!(Lext, zero(C))
        copyto!(Lext, CartesianIndices(L), L, CartesianIndices(L))
        copy!(gL, view(ldiv!(g.fact, Lext), axes(L)...))
        return gL
    end
    return lazy
end

function lazy_rdiv!(L, g∞ω::Lazy{G}, L´g) where {C,G<:SparseLUSlicer{C}}
    lazy = Lazy{Matrix{C}}() do
        g = g∞ω[]
        Lext = view(g.source, :, axes(L, 2))
        fill!(Lext, zero(C))
        copyto!(Lext, CartesianIndices(L), L, CartesianIndices(L))
        copy!(L´g, view(ldiv!(g.fact', Lext), axes(L)...)')
        return L´g
    end
    return lazy
end

#endregion

#region ## API ##

function Base.getindex(s::SchurGreenSlicer, i::CellOrbitals, j::CellOrbitals)
    if isinf(s.boundary)
        return inf_schur_slice(s, i, j)
    else
        return semi_schur_slice(s, i, j)
    end
end

function inf_schur_slice(s::SchurGreenSlicer, i::CellOrbitals, j::CellOrbitals)
    rows, cols = orbindices(i), orbindices(j)
    dist = only(cell(i) - cell(j))
    if dist == 0
        g = s.G∞₀₀[]
        i´, j´ = cellorbs((), rows), cellorbs((), cols)
        return g[i´, j´]
    elseif dist >= 1                                      # G∞ₙₘ = G₁₁L (R'G₁₁L)ⁿ⁻ᵐ⁻¹ R'G∞₀₀
        R´G∞₀₀ = view(s.R´G∞₀₀[], :, cols)
        R´G₁₁L = s.R´G₁₁L[]
        G₁₁L = view(s.G₁₁L[], rows, :)
        G = G₁₁L * (R´G₁₁L^(dist - 1)) * R´G∞₀₀
        # add view for type-stability
        return G
    else # dist <= -1                                 # G∞ₙₘ = G₋₁₋₁R (L'G₋₁₋₁R)ᵐ⁻ⁿ⁻¹ L'G∞₀₀
        L´G∞₀₀ = view(s.L´G∞₀₀[], :, cols)
        L´G₋₁₋₁R = s.L´G₋₁₋₁R[]
        G₋₁₋₁R = view(s.G₋₁₋₁R[], rows, :)
        G = G₋₁₋₁R * (L´G₋₁₋₁R^(- dist - 1)) * L´G∞₀₀
        # add view for type-stability
        return G
    end
end

function semi_schur_slice(s::SchurGreenSlicer{C}, i, j) where {C}
    n = only(cell(i)) - Int(s.boundary)
    m = only(cell(j)) - Int(s.boundary)
    rows, cols = orbindices(i), orbindices(j)
    if n * m <= 0 # This includes inter-boundary
        # need to add view with specific index types for type stability
        return zeros(C, norbs(i), norbs(j))
    elseif n == m == 1
        g = s.G₁₁[]
        i´, j´ = cellorbs((), rows), cellorbs((), cols)
        return g[i´, j´]
    elseif n == m == -1
        g = s.G₋₁₋₁[]
        i´, j´ = cellorbs((), rows), cellorbs((), cols)
        return g[i´, j´]
    elseif m >= 1  # also n >= 1                       # Gₙₘ = G∞ₙₘ - G₁₁L(R'G₁₁L)ⁿ⁻¹ R'G∞₀ₘ
        i´ = cellorbs(n, rows)
        j´ = cellorbs(m, cols)
        G∞ₙₘ = inf_schur_slice(s, i´, j´)
        i´ = cellorbs(0, :)
        R´G∞₀ₘ = s.R' * inf_schur_slice(s, i´, j´)
        R´G₁₁L = s.R´G₁₁L[]
        G₁₁L = view(s.G₁₁L[], rows, :)
        Gₙₘ = n == 1 ?
            mul!(G∞ₙₘ, G₁₁L, R´G∞₀ₘ, -1, 1) :
            mul!(G∞ₙₘ, G₁₁L, (R´G₁₁L^(n-1)) * R´G∞₀ₘ, -1, 1)
        return Gₙₘ
    else  # m, n <= -1                             # Gₙₘ = G∞ₙₘ - G₋₁₋₁R(L'G₋₁₋₁R)⁻ⁿ⁻¹L'G∞₀ₘ
        i´ = cellorbs(n, rows)
        j´ = cellorbs(m, cols)
        G∞ₙₘ = inf_schur_slice(s, i´, j´)
        i´ = cellorbs(0, :)
        L´G∞₀ₘ = s.L' * inf_schur_slice(s, i´, j´)
        L´G₋₁₋₁R = s.L´G₋₁₋₁R[]
        G₋₁₋₁R = view(s.G₋₁₋₁R[], rows, :)
        Gₙₘ = n == -1 ?
            mul!(G∞ₙₘ, G₋₁₋₁R, L´G∞₀ₘ, -1, 1) :
            mul!(G∞ₙₘ, G₋₁₋₁R, (L´G₋₁₋₁R^(-n-1)) * L´G∞₀ₘ, -1, 1)
        return Gₙₘ
    end
end

#endregion

#endregion
