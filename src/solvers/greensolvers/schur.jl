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
    is_nearest = length(harmonics(h)) == 3 && all(harmonics(h)) do hh
        dn = dcell(hh)
        dn == SA[0] || dn == SA[1] || dn == SA[-1]
    end
    is_nearest ||
        argerror("Too many or too few harmonics. Perhaps try `supercell` to ensure strictly nearest-cell harmonics.")

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

minimal_callsafe_copy(s::SchurFactorsSolver) =
    SchurFactorsSolver(s.shift, copy(s.hm), copy(s.h0), copy(s.hp), s.l_leq_r, copy(s.iG),
    s.ptrs, s.linds, s.rinds, s.sinds, copy(s.L), copy(s.R), copy(s.R´L´),
    minimal_callsafe_copy(s.tmp))

minimal_callsafe_copy(s::SchurWorkspace) =
    SchurWorkspace(copy.((s.GL, s.GR, s.LG, s.RG, s.A, s.B, s.Z11, s.Z21, s.Z11´, s.Z21´,
    s.LD, s.DL, s.RD, s.DR))...)

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
# AppliedSchurGreenSolver
#region

# We delay initialization of some fields until they are first needed (which may be never)
mutable struct AppliedSchurGreenSolver{T,B,O,O∞,G,G∞} <: AppliedGreenSolver
    fsolver::SchurFactorsSolver{T,B}
    boundary::T
    ohL::O                  # OpenHamiltonian for unitcell with ΣL
    ohR::O                  # OpenHamiltonian for unitcell with ΣR
    oh∞::O∞                 # OpenHamiltonian for unitcell with ΣL + ΣR
    gL::G                   # Lazy field: GreenFunction for ohL
    gR::G                   # Lazy field: GreenFunction for ohR
    g∞::G∞                  # Lazy field: GreenFunction for oh∞
    function AppliedSchurGreenSolver{T,B,O,O∞,G,G∞}(fsolver, boundary, ohL, ohR, oh∞) where {T,B,O,O∞,G,G∞}
        s = new()
        s.fsolver = fsolver
        s.boundary = boundary
        s.ohL = ohL
        s.ohR = ohR
        s.oh∞ = oh∞
        return s
    end
end

AppliedSchurGreenSolver{G,G∞}(fsolver::SchurFactorsSolver{T,B}, boundary, ohL::O, ohR::O, oh∞::O∞) where {T,B,O,O∞,G,G∞} =
    AppliedSchurGreenSolver{T,B,O,O∞,G,G∞}(fsolver, boundary, ohL, ohR, oh∞)

#region ## API ##

schurfactorsolver(s::AppliedSchurGreenSolver) = s.fsolver

#endregion

#region ## getproperty ##

function Base.getproperty(s::AppliedSchurGreenSolver, f::Symbol)
    if !isdefined(s, f)
        if f == :gL
            s.gL = greenfunction(s.ohL, GS.SparseLU())
        elseif f == :gR
            s.gR = greenfunction(s.ohR, GS.SparseLU())
        elseif f == :g∞
            s.g∞ = greenfunction(s.oh∞, GS.SparseLU())
        else
            argerror("Unknown field $f for AppliedSchurGreenSolver")
        end
    end
    return getfield(s, f)
end

#endregion

#region ## apply ##

function apply(s::GS.Schur, h::AbstractHamiltonian1D, contacts::Contacts)
    h´ = hamiltonian(h)
    fsolver = SchurFactorsSolver(h´, s.shift)
    h0 = unitcell_hamiltonian(h)
    rsites = stored_cols(unflat(h[1]))
    lsites = stored_cols(unflat(h[-1]))
    latslice_l = lattice(h0)[cellsites((), lsites)]
    latslice_r = lattice(h0)[cellsites((), rsites)]
    ΣR_solver = SelfEnergySchurSolver(fsolver, :R)
    ΣL_solver = SelfEnergySchurSolver(fsolver, :L)
    ΣL = SelfEnergy(ΣL_solver, latslice_l)
    ΣR = SelfEnergy(ΣR_solver, latslice_r)

    ohL = attach(h0, ΣL)
    ohR = attach(h0, ΣR)
    oh∞ = attach(ohL, ΣR)
    G, G∞ = green_type(h0, ΣL), green_type(h0, ΣL, ΣR)
    boundary = round(only(s.boundary))
    solver = AppliedSchurGreenSolver{G,G∞}(fsolver, boundary, ohL, ohR, oh∞)
    return solver
end

const GFUnit{T,E,H,N,S} =
    GreenFunction{T,E,0,AppliedSparseLUGreenSolver{Complex{T}},H,Contacts{0,N,S}}

green_type(::H,::S) where {T,E,H<:AbstractHamiltonian{T,E},S} =
    GFUnit{T,E,H,1,Tuple{S}}
green_type(::H,::S1,::S2) where {T,E,H<:AbstractHamiltonian{T,E},S1,S2} =
    GFUnit{T,E,H,2,Tuple{S1,S2}}

#endregion

#region ## call API ##

function minimal_callsafe_copy(s::AppliedSchurGreenSolver{<:Any,<:Any,<:Any,<:Any,G,G∞}) where {G,G∞}
    s´ = AppliedSchurGreenSolver{G,G∞}(s.fsolver, s.boundary,
        minimal_callsafe_copy(s.ohL),
        minimal_callsafe_copy(s.ohR),
        minimal_callsafe_copy(s.oh∞))
    isdefined(s, :gR) && (s´.gR = minimal_callsafe_copy(s.gR))
    isdefined(s, :gL) && (s´.gL = minimal_callsafe_copy(s.gL))
    isdefined(s, :g∞) && (s´.g∞ = minimal_callsafe_copy(s.g∞))
    return s´
end

function (s::AppliedSchurGreenSolver)(ω, Σblocks, cblockstruct)
    # call! fsolver once for all the g's
    call!(s.fsolver, ω)
    g0slicer = SchurGreenSlicer(ω, s)
    g0slicer.G∞₀₀
    gslicer = TMatrixSlicer(g0slicer, Σblocks, cblockstruct)
    return gslicer
end

#endregion

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

# We delay initialization of most fields until they are first needed (which may be never)
mutable struct SchurGreenSlicer{C,A<:AppliedSchurGreenSolver}  <: GreenSlicer{C}
    ω::C
    solver::A
    boundary::C
    L::Matrix{C}
    R::Matrix{C}
    G₋₁₋₁::SparseLUSlicer{C}
    G₁₁::SparseLUSlicer{C}
    G∞₀₀::SparseLUSlicer{C}
    L´G∞₀₀::Matrix{C}
    R´G∞₀₀::Matrix{C}
    G₁₁L::Matrix{C}
    G₋₁₋₁R::Matrix{C}
    R´G₁₁L::Matrix{C}
    L´G₋₁₋₁R::Matrix{C}
    function SchurGreenSlicer{C,A}(ω, solver) where {C,A}
        s = new()
        s.ω = ω
        s.solver = solver
        s.boundary = solver.boundary
        s.L = solver.fsolver.L
        s.R = solver.fsolver.R
        return s
    end
end

SchurGreenSlicer(ω, solver::A) where {T,A<:AppliedSchurGreenSolver{T}} =
    SchurGreenSlicer{Complex{T},A}(ω, solver)

#region ## getproperty ##

function Base.getproperty(s::SchurGreenSlicer, f::Symbol)
    if !isdefined(s, f)
        solver = s.solver
        d = size(s.L, 2)
        if f == :G₋₁₋₁
            s.G₋₁₋₁ = slicer(call!(solver.gL, s.ω; skipsolve_internal = true))
        elseif f == :G₁₁
            s.G₁₁ = slicer(call!(solver.gR, s.ω; skipsolve_internal = true))
        elseif f == :G∞₀₀
            s.G∞₀₀ = slicer(call!(solver.g∞, s.ω; skipsolve_internal = true))
        elseif f == :L´G∞₀₀
            tmp = solver.fsolver.tmp.LG
            s.L´G∞₀₀ = extended_rdiv!(tmp, s.L, s.G∞₀₀)
        elseif f == :R´G∞₀₀
            tmp = solver.fsolver.tmp.RG
            s.R´G∞₀₀ = extended_rdiv!(tmp, s.R, s.G∞₀₀)
        elseif f == :G₁₁L
            tmp = solver.fsolver.tmp.GL
            s.G₁₁L = extended_ldiv!(tmp, s.G₁₁, s.L)
        elseif f == :G₋₁₋₁R
            tmp = solver.fsolver.tmp.GR
            s.G₋₁₋₁R = extended_ldiv!(tmp, s.G₋₁₋₁, s.R)
        elseif f == :R´G₁₁L
            tmp = similar(s.R, d, d)
            s.R´G₁₁L = mul!(tmp, s.R', s.G₁₁L)
        elseif f == :L´G₋₁₋₁R
            tmp = similar(s.L, d, d)
            s.L´G₋₁₋₁R = mul!(tmp, s.L', s.G₋₁₋₁R)
        else
            argerror("Unknown field $f for SchurGreenSlicer")
        end
    end
    return getfield(s, f)
end

# note that g.source is taller than L, R, due to extended sites, but of >= witdth
# size(L, 2) = size(R, 2) = min(l, r) = d (deflated surface)
function extended_ldiv!(gL::Matrix{C}, g::SparseLUSlicer, L) where {C}
    Lext = view(g.source, :, axes(L, 2))
    fill!(Lext, zero(C))
    copyto!(Lext, CartesianIndices(L), L, CartesianIndices(L))
    copy!(gL, view(ldiv!(g.fact, Lext), axes(L)...))
    return gL
end

function extended_rdiv!(L´g::Matrix{C}, L, g::SparseLUSlicer) where {C}
    Lext = view(g.source, :, axes(L, 2))
    fill!(Lext, zero(C))
    copyto!(Lext, CartesianIndices(L), L, CartesianIndices(L))
    copy!(L´g, view(ldiv!(g.fact', Lext), axes(L)...)')
    return L´g
end

#endregion

#region ## API ##

Base.getindex(s::SchurGreenSlicer, i::CellOrbitals, j::CellOrbitals) =
    isinf(s.boundary) ? inf_schur_slice(s, i, j) : semi_schur_slice(s, i, j)

function inf_schur_slice(s::SchurGreenSlicer, i::CellOrbitals, j::CellOrbitals)
    rows, cols = orbindices(i), orbindices(j)
    dist = only(cell(i) - cell(j))
    if dist == 0
        g = s.G∞₀₀
        i´, j´ = cellorbs((), rows), cellorbs((), cols)
        return g[i´, j´]
    elseif dist >= 1                                      # G∞ₙₘ = G₁₁L (R'G₁₁L)ⁿ⁻ᵐ⁻¹ R'G∞₀₀
        R´G∞₀₀ = view(s.R´G∞₀₀, :, cols)
        R´G₁₁L = s.R´G₁₁L
        G₁₁L = view(s.G₁₁L, rows, :)
        G = G₁₁L * (R´G₁₁L^(dist - 1)) * R´G∞₀₀
        return G
    else # dist <= -1                                 # G∞ₙₘ = G₋₁₋₁R (L'G₋₁₋₁R)ᵐ⁻ⁿ⁻¹ L'G∞₀₀
        L´G∞₀₀ = view(s.L´G∞₀₀, :, cols)
        L´G₋₁₋₁R = s.L´G₋₁₋₁R
        G₋₁₋₁R = view(s.G₋₁₋₁R, rows, :)
        G = G₋₁₋₁R * (L´G₋₁₋₁R^(- dist - 1)) * L´G∞₀₀
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
        g = s.G₁₁
        i´, j´ = cellorbs((), rows), cellorbs((), cols)
        return g[i´, j´]
    elseif n == m == -1
        g = s.G₋₁₋₁
        i´, j´ = cellorbs((), rows), cellorbs((), cols)
        return g[i´, j´]
    elseif m >= 1  # also n >= 1                       # Gₙₘ = G∞ₙₘ - G₁₁L(R'G₁₁L)ⁿ⁻¹ R'G∞₀ₘ
        i´ = cellorbs(n, rows)
        j´ = cellorbs(m, cols)
        G∞ₙₘ = inf_schur_slice(s, i´, j´)
        i´ = cellorbs(0, :)
        R´G∞₀ₘ = s.R' * inf_schur_slice(s, i´, j´)
        R´G₁₁L = s.R´G₁₁L
        G₁₁L = view(s.G₁₁L, rows, :)
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
        L´G₋₁₋₁R = s.L´G₋₁₋₁R
        G₋₁₋₁R = view(s.G₋₁₋₁R, rows, :)
        Gₙₘ = n == -1 ?
            mul!(G∞ₙₘ, G₋₁₋₁R, L´G∞₀ₘ, -1, 1) :
            mul!(G∞ₙₘ, G₋₁₋₁R, (L´G₋₁₋₁R^(-n-1)) * L´G∞₀ₘ, -1, 1)
        return Gₙₘ
    end
end

# TODO: Perhaps too conservative
function minimal_callsafe_copy(s::SchurGreenSlicer)
    s´ = SchurGreenSlicer(s.ω, minimal_callsafe_copy(s.solver))
    isdefined(s, :G₋₁₋₁)    && (s´.G₋₁₋₁    = minimal_callsafe_copy(s.G₋₁₋₁))
    isdefined(s, :G₁₁)      && (s´.G₁₁      = minimal_callsafe_copy(s.G₁₁))
    isdefined(s, :G∞₀₀)     && (s´.G∞₀₀     = minimal_callsafe_copy(s.G∞₀₀))
    isdefined(s, :L´G∞₀₀)   && (s´.L´G∞₀₀   = copy(s.L´G∞₀₀))
    isdefined(s, :R´G∞₀₀)   && (s´.R´G∞₀₀   = copy(s.R´G∞₀₀))
    isdefined(s, :G₁₁L)     && (s´.G₁₁L     = copy(s.G₁₁L))
    isdefined(s, :G₋₁₋₁R)   && (s´.G₋₁₋₁R   = copy(s.G₋₁₋₁R))
    isdefined(s, :R´G₁₁L)   && (s´.R´G₁₁L   = copy(s.R´G₁₁L))
    isdefined(s, :L´G₋₁₋₁R) && (s´.L´G₋₁₋₁R = copy(s.L´G₋₁₋₁R))
    return s´
end
#endregion

#endregion

############################################################################################
# SelfEnergySchurSolver <: ExtendedSelfEnergySolver <: AbstractSelfEnergySolver
#   Extended self energy solver for deflated ΣL or ΣR Schur factors of lead unitcell
#   Implements syntax attach(h0, glead; ...)
#region

struct SelfEnergySchurSolver{T,B,V<:Union{Missing,Vector{Int}}} <: ExtendedSelfEnergySolver
    fsolver::SchurFactorsSolver{T,B}
    isleftside::Bool
    leadtoparent::V  # orbital index in parent for each orbital in open lead surface
                     # so that Σlead[leadtoparent, leadtoparent] == Σparent
end


#region ## Constructors ##

SelfEnergySchurSolver(fsolver::SchurFactorsSolver, side::Symbol, parentinds = missing) =
    SelfEnergySchurSolver(fsolver, isleftside(side), parentinds)

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

#region ## SelfEnergy (attach) API ##

# This syntax checks that the selected sites of hparent match the L/R surface of the semi-infinite lead
# and if so, builds the extended Self Energy directly, with the correct site order
function SelfEnergy(hparent::AbstractHamiltonian, glead::GreenFunction{<:Any,<:Any,1,<:AppliedSchurGreenSolver}; negative = false, transform = missing, kw...)
    isempty(contacts(glead)) || argerror("Tried to attach a lead with $(length(selfenergies(contacts(glead)))) contacts. \nCurrently, a lead with contacts cannot be contacted to another system using the simple `attach(x, glead; ...)` syntax. Use `attach(x, glead, model; ...)` instead")
    sel = siteselector(; kw...)
    lsparent = lattice(hparent)[sel]
    schursolver = solver(glead)
    fsolver = schurfactorsolver(schursolver)
    # we obtain latslice of open surface in gL/gR
    gunit = negative ? schursolver.gL : schursolver.gR
    blocksizes(blockstructure(hamiltonian(gunit))) == blocksizes(blockstructure(hparent)) ||
        argerror("The orbital structure of parent and lead Hamiltonians do not match")
    # This is a SelfEnergy for a lead unit cell with a SelfEnergySchurSolver
    Σlead = only(selfenergies(contacts(gunit)))
    lslead = latslice(Σlead)
    # find lead site index in lslead for each site in lsparent
    leadsites, displacement = lead_siteids_foreach_parent_siteids(lsparent, lslead, transform)
    # translate lead site indices to lead orbital indices using lead's ContactBlockStructure
    leadcbs = blockstructure(contacts(gunit))
    leadorbs = contact_sites_to_orbitals(leadsites, leadcbs)
    solver´ = SelfEnergySchurSolver(fsolver, negative, leadorbs)
    hlead = parent(glead)
    boundary = solver(glead).boundary
    plottables = (hlead, negative, transform, displacement, boundary)
    return SelfEnergy(solver´, lsparent, plottables)
end

# find ordering of lslead sites that match lsparent sites, modulo a displacement
function lead_siteids_foreach_parent_siteids(lsparent, lslead, transform)
    np, nl = nsites(lsparent), nsites(lslead)
    np == nl || argerror("The contact surface has $np sites, which doesn't match the $nl sites in the lead surface")
    sp = collect(sites(lsparent))
    sl = collect(sites(lslead))
    transform === missing || (sl .= transform.(sl))
    displacement = mean(sp) - mean(sl)
    sl .+= Ref(displacement)
    tree = KDTree(sl)
    indslead, dists = nn(tree, sp)
    iszero(chop(maximum(dists))) && allunique(indslead) ||
        argerror("The contact and lead surface sites have same number of sites but do not match (modulo a displacement). Perhaps an error in the `attach` site selection? Otherwise consider using the `transform` keyword to specify an attachment transformation.")
    return indslead, displacement
end

#endregion

#region ## API ##

# This solver produces two solutions (L/R) for the price of one. We can opt out of calling
# it if we know it has already been called, so the solution is already in its call!_output
function call!(s::SelfEnergySchurSolver, ω;
               skipsolve_internal = false, params...)
    fsolver = s.fsolver
    Rfactors, Lfactors = skipsolve_internal ? call!_output(fsolver) : call!(fsolver, ω)
    factors = maybe_match_parent(ifelse(s.isleftside, Lfactors, Rfactors), s.leadtoparent)
    return factors
end

call!_output(s::SelfEnergySchurSolver) = call!(s, missing; skipsolve_internal = true)

maybe_match_parent((V, ig, V´), leadtoparent) =
    (view(V, leadtoparent, :), ig, view(V´, :, leadtoparent))

maybe_match_parent(factors, ::Missing) = factors

minimal_callsafe_copy(s::SelfEnergySchurSolver) =
    SelfEnergySchurSolver(minimal_callsafe_copy(s.fsolver), s.isleftside, s.leadtoparent)

#endregion

#endregion

############################################################################################
# SelfEnergyUnitcellSchurSolver <: ExtendedSelfEnergySolver <: AbstractSelfEnergySolver
#   Extended self energy solver with g⁻¹ = h0 + ExtendedSchurΣ, plus arbitrary couplings
#   to parent Hamiltonian
#   Implements syntax attach(h0, glead, model; ...)
#region

struct SelfEnergyUnicellSchurSolver{C,G,H,S<:SparseMatrixView,S´<:SparseMatrixView} <: ExtendedSelfEnergySolver
    gunit::G
    hcoupling::H
    V´::S´
    g⁻¹::InverseGreenBlockSparse{C}
    V::S
end

#region ## Constructors ##

function SelfEnergyUnicellSchurSolver(gunit::GreenFunction{T}, hcoupling::AbstractHamiltonian{T}, nparent) where {T}
    hmatrix = call!_output(hcoupling)
    invgreen = inverse_green(solver(gunit))
    lastflatparent = last(flatrange(hcoupling, nparent))
    size(hmatrix, 1) == lastflatparent + length(orbrange(invgreen)) ||
        internalerror("SelfEnergyUnicellSchurSolver builder: $(size(hmatrix, 1)) != $lastflatparent + $(length(orbrange(invgreen)))")
    parentrng, leadrng = 1:lastflatparent, lastflatparent+1:size(hmatrix, 1)
    sizeV = size(invgreen, 1), lastflatparent
    V = SparseMatrixView(view(hmatrix, leadrng, parentrng), sizeV)
    V´ = SparseMatrixView(view(hmatrix, parentrng, leadrng), reverse(sizeV))
    return SelfEnergyUnicellSchurSolver(gunit, hcoupling, V´,invgreen,V)
end

#endregion

#region ## API ##

h0unit(s::SelfEnergyUnicellSchurSolver) = parent(s.gunit)

hcoupling(s::SelfEnergyUnicellSchurSolver) = s.hcoupling

#endregion

#region ## SelfEnergy (attach) API ##

# With this syntax we attach the surface unit cell of the lead (left or right) to hparent
# through the model coupling. We thus first apply the model to the 0D lattice of hparent's
# selected surface plus the lead unit cell (checking one unit cell is enough), and then
# build an extended self energy
function SelfEnergy(hparent::AbstractHamiltonian, glead::GreenFunction{<:Any,<:Any,1,<:AppliedSchurGreenSolver}, model::AbstractModel; negative = false, transform = missing, kw...)
    isempty(contacts(glead)) || argerror("Tried to attach a lead with $(length(selfenergies(contacts(glead)))) contacts. \nCurrently, a lead with contacts cannot be contacted to another system using the simple `attach(x, glead; ...)` syntax. Use `attach(x, glead[...], model; ...)` instead")
    schursolver = solver(glead)
    gunit = copy_lattice(negative ? schursolver.gL : schursolver.gR)
    lat0lead = lattice(parent(gunit))
    sitesunit = sites(lat0lead)
    transform === missing || (sitesunit .= transform.(sitesunit))
    sel = siteselector(; kw...)
    lsparent = lattice(hparent)[sel]
    lat0parent = lattice0D(lsparent)
    lat0 = combine(lat0parent, lat0lead)
    nparent, ntotal = nsites(lat0parent), nsites(lat0)
    interblockmodel = interblock(model, 1:nparent, nparent+1:ntotal)
    hcoupling = hamiltonian(lat0, interblockmodel)
    solver´ = SelfEnergyUnicellSchurSolver(gunit, hcoupling, nparent)
    hlead = parent(glead)
    boundary = solver(glead).boundary
    plottables = (hlead, hcoupling, negative, transform, boundary)
    return SelfEnergy(solver´, lsparent, plottables)
end

#endregion

#region ## API ##

function call!(s::SelfEnergyUnicellSchurSolver, ω; params...)
    call!(s.hcoupling, (); params...)
    call!(s.gunit, ω; params...)
    update!(s.V)
    update!(s.V´)
    return matrix(s.V´), matrix(s.g⁻¹), matrix(s.V)
end

call!_output(s::SelfEnergyUnicellSchurSolver) = matrix(s.V´), matrix(s.g⁻¹), matrix(s.V)

minimal_callsafe_copy(s::SelfEnergyUnicellSchurSolver) =
    SelfEnergyUnicellSchurSolver(
        minimal_callsafe_copy(s.gunit),
        minimal_callsafe_copy(s.hcoupling),
        minimal_callsafe_copy(s.V´),
        minimal_callsafe_copy(s.g⁻¹),
        minimal_callsafe_copy(s.V))

#endregion

#endregion


#endregion top