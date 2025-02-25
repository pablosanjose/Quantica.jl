# NOTE: this solver is probably too convoluted and could benefit from a refactor to make it
# more maintainable. The priority here was performance (minimal number of computations)
# The formalism is based on currently unpublished notes (scattering.pdf)

# Design overview: The G₋₁₋₁, G₁₁ and G∞₀₀ slicers inside SchurGreenSlicer correspond to
# boundary cells in seminf and origin cell in an infinite 1D Hamiltonian, based on a LU
# solver. These are not computed upon creating SchurGreenSlicer, since not all are necessary
# for any given slice. Same for their parent gL, dR and g∞ 0D unitcell GreenFunctions inside
# AppliedSchurGreenSolver. Their respective OpenHamiltonians contain selfenergies with solver
# type SelfEnergySchurSolver which themselves all alias the same SchurFactorsSolver devoted
# to computing objects using the deflated Schur algorithm on which the selfenergies depend.
# When building SchurGreenSlicer with a given ω and params we know for sure that we will
# need the SchurFactorsSolver objects for that ω, but not which of G₋₁₋₁, G₁₁ or G∞₀₀.
# We also don't want to compute the SchurFactorsSolver objects more than once for a given ω
# Hence, we do a call!(fsolver::SchurFactorsSolver, ω) upon constructing SchurGreenSlicer,
# but whenever building the uninitialized slicers G₁₁ etc we do a skipsolve_internal = true
# to avoid recomputing SchurFactorsSolver. Also, since the unitcell Hamiltonians harmonics
# h0, hm, hp in SchurFactorsSolver alias those of hamiltonian(g::GreenFunction), we need to
# do call!(parent(g); params...) upon constructing SchurFactorsSolver too.

############################################################################################
# SchurFactorsSolver - see scattering.pdf notes for derivations
#   Auxiliary functions for AppliedSchurGreenSolverSolver
#   Computes dense factors PR*R*Z21, Z11 and R'*PR'. The retarded self-energy on the open
#   unitcell surface of a semi-infinite rightward lead reads Σᵣ = PR R Z21 Z11⁻¹ R' PR'
#   Computes also the leftward PL*L*Z11´, Z21´, L'*PL', with  Σₗ = PL L Z11´ Z21´⁻¹ L' PL'
#region

struct SchurWorkspace{C}
    GL::Matrix{ComplexF64}
    GR::Matrix{ComplexF64}
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
    hm::HybridSparseMatrix{T,B}                       # aliases parent hamiltonian
    h0::HybridSparseMatrix{T,B}                       # aliases parent hamiltonian
    hp::HybridSparseMatrix{T,B}                       # aliases parent hamiltonian
    l_leq_r::Bool                                     # whether l <= r (left and right surface dims)
    iG::SparseMatrixCSC{Complex{T},Int}               # to store iG = ω - h0 - Σₐᵤₓ
    ptrs::Tuple{Vector{Int},Vector{Int},Vector{Int}}  # iG ptrs for h0 nzvals, diagonal and Σₐᵤₓ surface
    linds::Vector{Int}                                # orbital indices on left surface
    rinds::Vector{Int}                                # orbital indices on right surface
    sinds::Vector{Int}                                # orbital indices on the smallest surface (left for l<=r, right for l>r)
    L::Matrix{ComplexF64}                             # l<=r ? PL : PL*H' === hp PR  (n × min(l,r))
    R::Matrix{ComplexF64}                             # l<=r ? PR*H === hm PL : PR   (n × min(l,r))
    R´L´::Matrix{ComplexF64}                          # [R'; -L']. L and R must be dense for iG \ (L,R)
    tmp::SchurWorkspace{Complex{T}}                   # L, R, R´L´ need 64bit
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
    return SchurFactorsSolver(T(shift), hm, h0, hp, l_leq_r, iG, ptrs, linds, rinds, sinds, L, R, R´L´, workspace)
end

function SchurWorkspace{C}((n, d), l, r) where {C}
    GL = Matrix{ComplexF64}(undef, n, d)
    GR = Matrix{ComplexF64}(undef, n, d)
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
        argerror("Too many or too few harmonics (need 3, got $(length(harmonics(h)))). Perhaps try `supercell` to ensure strictly nearest-cell harmonics.")
    hm, h0, hp = h[hybrid(-1)], h[hybrid(0)], h[hybrid(1)]
    flat(hm) ≈ flat(hp)' ||
        argerror("The Hamiltonian should have h[1] ≈ h[-1]' to use the Schur solver")
    return hm, h0, hp
end

# hp = L*R' = PL H' PR'. We assume hm = hp'
function left_right_projectors(hm::SparseMatrixCSC, hp::SparseMatrixCSC)
    linds = stored_cols(hm)
    rinds = stored_cols(hp)
    # dense projectors
    o = one(ComplexF64) * I
    allrows = 1:size(hp,1)
    l_leq_r = length(linds) <= length(rinds)
    PR = o[allrows, rinds]
    PL = o[allrows, linds]
    if l_leq_r
        sinds = linds
        R = Matrix{ComplexF64}(hm[:, linds])  # R = PR H = hm PL
        L = PL
    else
        sinds = rinds
        R = PR
        L = Matrix{ComplexF64}(hp[:, rinds])  # L = PL H' = hp PR
    end
    return linds, rinds, L, R, sinds, l_leq_r
end

# Build a new sparse matrix mat´ with same structure as mat plus the diagonal
# Return also:
#   (1) pointers pmat´ to mat´ for each nonzero in mat
#   (2) diagonal ptrs pdiag´ in mat´
function store_diagonal_ptrs(mat::SparseMatrixCSC{T}) where {T}
    mat´ = store_diagonal(mat)
    pmat´, pdiag´ = Int[], Int[]
    rows, rows´ = rowvals(mat), rowvals(mat´)
    for col in axes(mat´, 2)
        ptrs = nzrange(mat, col)
        ptrs´ = nzrange(mat´, col)
        p, p´ = first(ptrs), first(ptrs´)
        while p´ in ptrs´
            row´ = rows´[p´]
            row´ == col && push!(pdiag´, p´)
            if p in ptrs && row´ == rows[p]
                push!(pmat´, p´)
                p += 1
            end
            p´ += 1
        end
    end
    return mat´, (pmat´, pdiag´)
end

# ensure diagonal is stored *without* dropping any structural zeros
function store_diagonal(mat::SparseMatrixCSC{T}) where {T}
    m, n = size(mat)
    d = min(m, n)
    I, J, V = findnz(mat)
    append!(I, 1:d)
    append!(J, 1:d)
    append!(V, Iterators.repeated(zero(T), d))
    return sparse(I, J, V, m, n)
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

# this does not include a parentcontacts because there are none
# (SchurFactorsSolver is not an AppliedGreenSolver, so it may have different API)
function minimal_callsafe_copy(s::SchurFactorsSolver, parentham)
    hm´, h0´, hp´ = nearest_cell_harmonics(parentham)
    s´ = SchurFactorsSolver(s.shift, hm´, h0´, hp´, s.l_leq_r, copy(s.iG),
        s.ptrs, s.linds, s.rinds, s.sinds, copy(s.L), copy(s.R), copy(s.R´L´),
        minimal_callsafe_copy(s.tmp))
    return s´
end

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
# AppliedSchurGreenSolver in 1D
#region

# Mutable: we delay initialization of some fields until they are first needed (which may be never)
mutable struct AppliedSchurGreenSolver{T,B,O,O∞,G,G∞,P} <: AppliedGreenSolver
    fsolver::SchurFactorsSolver{T,B}
    boundary::T
    ohL::O                  # OpenHamiltonian for unitcell with ΣL      (aliases parent h)
    ohR::O                  # OpenHamiltonian for unitcell with ΣR      (aliases parent h)
    oh∞::O∞                 # OpenHamiltonian for unitcell with ΣL + ΣR (aliases parent h)
    gL::G                   # Lazy field: GreenFunction for ohL
    gR::G                   # Lazy field: GreenFunction for ohR
    g∞::G∞                  # Lazy field: GreenFunction for oh∞
    integrate_opts::P       # for algorithms relying on integration (like densitymatrix)
    function AppliedSchurGreenSolver{T,B,O,O∞,G,G∞,P}(fsolver, boundary, ohL, ohR, oh∞, integrate_opts) where {T,B,O,O∞,G,G∞,P}
        s = new()
        s.fsolver = fsolver
        s.boundary = boundary
        s.ohL = ohL
        s.ohR = ohR
        s.oh∞ = oh∞
        s.integrate_opts = integrate_opts
        return s
    end
end

const GreenFunctionSchurEmptyLead1D{T,E} = GreenFunction{T,E,1,<:AppliedSchurGreenSolver,<:Any,<:EmptyContacts}
const GreenFunctionSchurLead1D{T,E} = GreenFunction{T,E,1,<:AppliedSchurGreenSolver,<:Any,<:Any}

AppliedSchurGreenSolver{G,G∞}(fsolver::SchurFactorsSolver{T,B}, boundary, ohL::O, ohR::O, oh∞::O∞, iopts::P) where {T,B,O,O∞,G,G∞,P} =
    AppliedSchurGreenSolver{T,B,O,O∞,G,G∞,P}(fsolver, boundary, ohL, ohR, oh∞, iopts)

#region ## API ##

schurfactorsolver(s::AppliedSchurGreenSolver) = s.fsolver

boundaries(s::AppliedSchurGreenSolver) = isfinite(s.boundary) ? (1 => s.boundary,) : ()

#endregion

#region ## getproperty ##

function Base.getproperty(s::AppliedSchurGreenSolver, f::Symbol)
    if !isdefined(s, f)
        if f == :gL
            s.gL = greenfunction(s.ohL, GS.SparseLU())  # oh's harmonics alias parent h[()],
        elseif f == :gR                                 # but not used until called with ω
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

function apply(s::GS.Schur, h::AbstractHamiltonian1D{T}, contacts) where {T}
    h´ = hamiltonian(h)
    fsolver = SchurFactorsSolver(h´, s.shift)  # aliasing of h´
    boundary = T(round(only(s.boundary)))
    ohL, ohR, oh∞, G, G∞ = schur_openhams_types(fsolver, h, boundary)
    solver = AppliedSchurGreenSolver{G,G∞}(fsolver, boundary, ohL, ohR, oh∞, s.integrate_opts)
    return solver
end

function schur_openhams_types(fsolver, h, boundary)
    h0 = unitcell_hamiltonian(h)   # h0 is non-parametric, but will alias h.h first harmonic
    rsites = stored_cols(hamiltonian(h)[unflat(1)])
    lsites = stored_cols(hamiltonian(h)[unflat(-1)])
    orbslice_l = sites_to_orbs(lattice(h0)[sites(lsites)], h)
    orbslice_r = sites_to_orbs(lattice(h0)[sites(rsites)], h)
    ΣR_solver = SelfEnergySchurSolver(fsolver, h, :R, boundary)
    ΣL_solver = SelfEnergySchurSolver(fsolver, h, :L, boundary)
    ΣL = SelfEnergy(ΣL_solver, orbslice_l)
    ΣR = SelfEnergy(ΣR_solver, orbslice_r)
    # ohL, ohR, oh∞ have no parameters, but will be updated by call!(h; params...)
    ohL = attach(h0, ΣL)
    ohR = attach(h0, ΣR)
    oh∞ = ohR |> attach(ΣL)
    G, G∞ = green_type(h0, ΣL), green_type(h0, ΣL, ΣR)
    return ohL, ohR, oh∞, G, G∞
end

const GFUnit{T,E,H,N,S} =
    GreenFunction{T,E,0,AppliedSparseLUGreenSolver{Complex{T}},H,Contacts{0,N,S,OrbitalSliceGrouped{T,E,0}}}

green_type(::H,::S) where {T,E,H<:AbstractHamiltonian{T,E},S} =
    GFUnit{T,E,H,1,Tuple{S}}
green_type(::H,::S1,::S2) where {T,E,H<:AbstractHamiltonian{T,E},S1,S2} =
    GFUnit{T,E,H,2,Tuple{S1,S2}}

#endregion

#region ## call API ##

function minimal_callsafe_copy(s::AppliedSchurGreenSolver, parentham, _)
    fsolver´ = minimal_callsafe_copy(s.fsolver, parentham)
    ohL´, ohR´, oh∞´, G, G∞ = schur_openhams_types(fsolver´, parentham, s.boundary)
    s´ = AppliedSchurGreenSolver{G,G∞}(fsolver´, s.boundary, ohL´, ohR´, oh∞´, s.integrate_opts)
    # we don't copy the lazy fields gL, gR, g∞, even if already materialized, since they
    # must be linked to ohL´, ohR´, oh∞´, not the old ones.
    return s´
end

function build_slicer(s::AppliedSchurGreenSolver, g, ω, Σblocks, corbitals; params...)
    # overwrites hamiltonian(g) with params whenever parent(g) isa ParametricHamiltonian
    # Necessary because its harmonics are aliased in the SchurFactorSolver inside s
    call!(parent(g); params...)
    g0slicer = SchurGreenSlicer(ω, s)
    gslicer = maybe_TMatrixSlicer(g0slicer, Σblocks, corbitals)
    return gslicer
end

integrate_opts(s::AppliedSchurGreenSolver) = s.integrate_opts

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
# should not have any contacts (we defer to TMatrixSlicer for that)
mutable struct SchurGreenSlicer{C,A<:AppliedSchurGreenSolver}  <: GreenSlicer{C}
    ω::C
    solver::A
    boundary::C
    L::Matrix{C}
    R::Matrix{C}
    G₋₁₋₁::SparseLUGreenSlicer{C}   # These are independent of parent hamiltonian
    G₁₁::SparseLUGreenSlicer{C}     # as they only rely on call!_output(solver.fsolver)
    G∞₀₀::SparseLUGreenSlicer{C}    # which is updated after call!(solver.fsolver, ω)
    L´G∞₀₀::Matrix{C}
    R´G∞₀₀::Matrix{C}
    G₁₁L::Matrix{C}
    G₋₁₋₁R::Matrix{C}
    R´G₁₁L::Matrix{C}
    L´G₋₁₋₁R::Matrix{C}
    function SchurGreenSlicer{C,A}(ω, solver) where {C,A}
        # call! the expensive fsolver only once to compute Schur factors, necessary to
        # evaluate unitcell selfenergies in OpenHamiltonians of solver
        # (see skipsolve_internal = true further below)
        call!(solver.fsolver, ω)
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
        # Issue #268: the result of the following call!'s depends on the current value of h0
        # which aliases the parent h. This is only a problem if `s` was obtained through
        # `gω = call!(g, ω; params...)`. In that case, doing call!(g, ω; params´...) before
        # gω[sites...] will be call!-ing e.g. solver.g∞ with the wrong h0 (the one from
        # params´...). However, if `gω = g(ω; params...)` a copy was made, so it is safe.
        if f == :G₋₁₋₁
            # we ran SchurFactorSolver when constructing s, so skipsolve_internal = true
            # to avoid running it again
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

# note that g.sourceC is taller than L, R, due to extended sites, but of >= witdth
# size(L, 2) = size(R, 2) = min(l, r) = d (deflated surface)
function extended_ldiv!(gL::Matrix{C}, g::SparseLUGreenSlicer, L) where {C}
    Lext = view(g.source64, :, axes(L, 2))
    fill!(Lext, zero(C))
    copyto!(Lext, CartesianIndices(L), L, CartesianIndices(L))
    copy!(gL, view(ldiv!(g.fact, Lext), axes(L)...))
    return gL
end

function extended_rdiv!(L´g::Matrix{C}, L, g::SparseLUGreenSlicer) where {C}
    Lext = view(g.source64, :, axes(L, 2))
    fill!(Lext, zero(C))
    copyto!(Lext, CartesianIndices(L), L, CartesianIndices(L))
    copy!(L´g, view(ldiv!(g.fact', Lext), axes(L)...)')
    return L´g
end

#endregion

#region ## API ##

function Base.getindex(s::SchurGreenSlicer, i::CellOrbitals, j::CellOrbitals)
    G = isinf(s.boundary) ? inf_schur_slice(s, i, j) : semi_schur_slice(s, i, j)
    # for type-stability with SVector indices
    return maybe_SMatrix(G, orbindices(i), orbindices(j))
end

function inf_schur_slice(s::SchurGreenSlicer, i::CellOrbitals, j::CellOrbitals)
    rows, cols = orbindices(i), orbindices(j)
    dist = only(cell(i) - cell(j))
    if dist == 0
        g = s.G∞₀₀
        i´, j´ = CellOrbitals((), rows), CellOrbitals((), cols)
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
        return zeros(C, norbitals(i), norbitals(j))
    elseif n == m == 1
        g = s.G₁₁
        i´, j´ = CellOrbitals((), rows), CellOrbitals((), cols)
        return g[i´, j´]
    elseif n == m == -1
        g = s.G₋₁₋₁
        i´, j´ = CellOrbitals((), rows), CellOrbitals((), cols)
        return g[i´, j´]
    elseif m >= 1  # also n >= 1                       # Gₙₘ = G∞ₙₘ - G₁₁L(R'G₁₁L)ⁿ⁻¹ R'G∞₀ₘ
        i´ = CellOrbitals(n, rows)
        j´ = CellOrbitals(m, cols)
        G∞ₙₘ = inf_schur_slice(s, i´, j´)
        i´ = CellOrbitals(0, :)
        R´G∞₀ₘ = s.R' * inf_schur_slice(s, i´, j´)
        R´G₁₁L = s.R´G₁₁L
        G₁₁L = view(s.G₁₁L, rows, :)
        Gₙₘ = n == 1 ?
            mul!(G∞ₙₘ, G₁₁L, R´G∞₀ₘ, -1, 1) :
            mul!(G∞ₙₘ, G₁₁L, (R´G₁₁L^(n-1)) * R´G∞₀ₘ, -1, 1)
        return Gₙₘ
    else  # m, n <= -1                             # Gₙₘ = G∞ₙₘ - G₋₁₋₁R(L'G₋₁₋₁R)⁻ⁿ⁻¹L'G∞₀ₘ
        i´ = CellOrbitals(n, rows)
        j´ = CellOrbitals(m, cols)
        G∞ₙₘ = inf_schur_slice(s, i´, j´)
        i´ = CellOrbitals(0, :)
        L´G∞₀ₘ = s.L' * inf_schur_slice(s, i´, j´)
        L´G₋₁₋₁R = s.L´G₋₁₋₁R
        G₋₁₋₁R = view(s.G₋₁₋₁R, rows, :)
        Gₙₘ = n == -1 ?
            mul!(G∞ₙₘ, G₋₁₋₁R, L´G∞₀ₘ, -1, 1) :
            mul!(G∞ₙₘ, G₋₁₋₁R, (L´G₋₁₋₁R^(-n-1)) * L´G∞₀ₘ, -1, 1)
        return Gₙₘ
    end
end

maybe_SMatrix(G::Matrix, rows::SVector{L}, cols::SVector{L´}) where {L,L´} = SMatrix{L,L´}(G)
maybe_SMatrix(G, rows, cols) = G

#endregion

#endregion

############################################################################################
# schur_eigvals
#   computes schur_eigenvalues of all lead modes
#region

schur_eigvals(g::GreenFunctionSchurLead1D, ω::Real; params...) =
    schur_eigvals(g, retarded_omega(ω, solver(g)); params...)

schur_eigvals(g::GreenFunctionSchurLead1D, ω::Complex; params...) =
    schur_eigvals((parent(g), g.solver), ω; params...)

function schur_eigvals((h, solver)::Tuple{AbstractHamiltonian1D,AppliedSchurGreenSolver}, ω::Complex; params...)
    call!(h; params...)             # update the (Parametric)Hamiltonian with the params
    sf = solver.fsolver             # obtain the SchurFactorSolver that computes the AB pencil
    update_LR!(sf)                  # Ensure L and R matrices are updated after updating h
    update_iG!(sf, ω)               # shift inverse G with ω
    A, B = pencilAB!(sf)            # build the pecil
    λs = eigvals!(A, B)             # extract the λs as geeraized eigenvales of the pencil
    return λs
end

retarded_eigvals(g, ω, minabs = 0; params...) =
    filter!(λ -> 1 > abs(λ) > minabs, schur_eigvals(g, ω; params...))

advanced_eigvals(g, ω, maxabs = Inf; params...) =
    filter!(λ -> 1 < abs(λ) < maxabs, schur_eigvals(g, ω; params...))

propagating_eigvals(g, ω, margin = 0; params...) =
    iszero(margin) ?
        filter!(λ -> abs(λ) ≈ 1, schur_eigvals(g, ω; params...)) :
        filter!(λ -> 1 - margin < abs(λ) < 1 + margin, schur_eigvals(g, ω; params...))

function decay_lengths(g::GreenFunctionSchurLead1D, args...; reverse = false, params...)
    λs = reverse ? advanced_eigvals(g, args...; params...) :  retarded_eigvals(g, args...; params...)
    ls = @. -1/log(abs(λs))         # compute the decay lengths in units of a0
    return ls
end

decay_lengths(g::AbstractHamiltonian1D, µ = 0, args...; params...) =
    decay_lengths(greenfunction(g, GS.Schur()), µ, args...; params...)

#endregion

############################################################################################
# AppliedSchurGreenSolver in 2D
#region

struct AppliedSchurGreenSolver2D{L,H<:AbstractHamiltonian1D,S<:AppliedSchurGreenSolver,F,P<:NamedTuple}  <: AppliedGreenSolver
    h1D::H
    solver1D::S
    axis1D::SVector{1,Int}
    wrapped_axes::SVector{L,Int}
    phase_func::F                   #  phase_func(ϕ) = (; param_name = ϕ)
    integrate_opts::P
end

const GreenFunctionSchur2D{T,L} = GreenFunction{T,<:Any,L,<:AppliedSchurGreenSolver2D}

# should not have any contacts (we defer to TMatrixSlicer for that)
struct SchurGreenSlicer2D{C,S<:AppliedSchurGreenSolver2D,F<:Function} <: GreenSlicer{C}
    ω::C
    solver::S
    slicer_generator::F
end

function apply(s::GS.Schur, h::AbstractHamiltonian, _)
    L = latdim(h)
    L == 2 ||
        argerror("GreenSolvers.Schur currently only implemented for 1D and 2D AbstractHamiltonians")
    axis1D = SVector(s.axis)
    wrapped_axes = inds_complement(Val(L), axis1D)
    h1D = @stitch(h, wrapped_axes, ϕ_internal)
    phase_func(ϕ_internal) = (; ϕ_internal)
    # we don't pass contacts to solver1D. They will be applied with T-matrix slicer
    solver1D = apply(s, h1D, missing)
    return AppliedSchurGreenSolver2D(h1D, solver1D, axis1D, wrapped_axes, phase_func, s.integrate_opts)
end

#region ## API ##

function minimal_callsafe_copy(s::AppliedSchurGreenSolver2D, args...)
    h1D´ = minimal_callsafe_copy(s.h1D)
    solver1D´ = minimal_callsafe_copy(s.solver1D, h1D´, missing)
    return AppliedSchurGreenSolver2D(h1D´, solver1D´, s.axis1D, s.wrapped_axes, s.phase_func, s.integrate_opts)
end

function build_slicer(s::AppliedSchurGreenSolver2D, g, ω, Σblocks, corbitals; params...)
    function slicer_generator(s, ϕ_internal)
        # updates h1D that is aliased into solver1D.fsolver with the apropriate phases,
        # included in params
        call!(s.h1D; params..., s.phase_func(ϕ_internal)...)
        return SchurGreenSlicer(ω, s.solver1D)
    end
    g0slicer = SchurGreenSlicer2D(ω, s, slicer_generator)
    gslicer = maybe_TMatrixSlicer(g0slicer, Σblocks, corbitals)
    return gslicer
end

function Base.getindex(s::SchurGreenSlicer2D, i::CellOrbitals, j::CellOrbitals)
    uas, was = s.solver.axis1D, s.solver.wrapped_axes
    ni, nj = cell(i), cell(j)
    i´, j´ = CellOrbitals(ni[uas], orbindices(i)), CellOrbitals(nj[uas], orbindices(j))
    dn = (ni - nj)[was]
    function integrand(x)
        ϕs = sanitize_SVector(2π * x)
        s1D = s.slicer_generator(s.solver, ϕs)
        gij = s1D[i´, j´] .* cis(-dot(dn, ϕs))
        return gij
    end
    integral, err = quadgk(integrand, -0.5, 0.5; s.solver.integrate_opts...)
    return integral
end

boundaries(s::AppliedSchurGreenSolver2D) = boundaries(s.solver1D)

integrate_opts(s::AppliedSchurGreenSolver2D) = s.integrate_opts

#endregion

#endregion


############################################################################################
# densitymatrix - dedicated Schur method (integration with Fermi points as segments)
#   1D and 2D routines. Does not support non-Nothing contacts
#region

struct DensityMatrixSchurSolver{T,L,A,G<:GreenSlice{T,<:Any,L},O<:NamedTuple}
    gs::G                           # GreenSlice
    orbaxes::A                      # axes of GreenSlice (orbrows, orbcols)
    hmat::Matrix{Complex{T}}        # it spans just the unit cell, dense Bloch matrix
    psis::Matrix{Complex{T}}        # it spans just the unit cell, Eigenstates
    integrate_opts::O               # callback [missing or a function f(ϕ, z) or f(ϕ1, ϕ2, z)] + kwargs for quadgk
end

## Constructor

function densitymatrix(s::Union{AppliedSchurGreenSolver,AppliedSchurGreenSolver2D}, gs::GreenSlice; callback = Returns(nothing), quadgk_opts...)
    check_no_boundaries_schur(s)
    check_no_contacts_schur(gs)
    return densitymatrix_schur(gs; integrate_opts(s)..., callback, quadgk_opts...)
end

check_no_boundaries_schur(s) = isempty(boundaries(s)) ||
    argerror("Boundaries not implemented for DensityMatrixSchurSolver. Consider using the generic integration solver.")

check_no_contacts_schur(gs) = has_selfenergy(gs) &&
    argerror("The Schur densitymatrix solver currently support only `nothing` contacts")

function densitymatrix_schur(gs; integrate_opts...)   # integrate_opts contains callback
    g = parent(gs)
    hmat = similar_Array(hamiltonian(g))
    psis = similar(hmat)
    orbaxes = orbrows(gs), orbcols(gs)
    solver = DensityMatrixSchurSolver(gs, orbaxes, hmat, psis, NamedTuple(integrate_opts))
    return DensityMatrix(solver, gs)
end

# API

## call

# use computed Fermi points to integrate spectral function by segments
# returns an AbstractMatrix
# we don't use Integrator, because that is meant for integrals over energy, not momentum
(s::DensityMatrixSchurSolver)(µ, kBT; params...) = integrate_rho_schur(s, µ, kBT; params...)

function integrate_rho_schur(s::DensityMatrixSchurSolver{<:Any,1}, µ, kBT; params...)
    result = call!_output(s.gs)
    data = serialize(result)
    g = parent(s.gs)
    xs = fermi_points_integration_path(g, µ; params...)
    function integrand!(x)
        ϕ = 2π * x
        z = fermi_h!(s, SA[ϕ], µ, inv(kBT); params...)
        callback(s)(ϕ, z)
        return z
    end
    fx! = (y, x) -> (y .= integrand!(x))
    quadgk!(fx!, data, xs...; quadgk_opts(s)...)
    return result
end

function integrate_rho_schur(s::DensityMatrixSchurSolver{<:Any,2}, µ, kBT; params...)
    s2D = solver(s.gs)
    result = call!_output(s.gs)
    data = serialize(result)
    h1D, s1D = (s2D.h1D, s2D.solver1D)

    axisorder = SA[only(s2D.axis1D), only(s2D.wrapped_axes)]

    function integrand_inner!(x1, x2)
        ϕ = 2π * SA[x1, x2][axisorder]
        z = fermi_h!(s, ϕ, µ, inv(kBT); params...)
        callback(s)(ϕ..., z)
        return z
    end

    function integrand_outer!(data, x2)
        xs = fermi_points_integration_path((h1D, s1D), µ; params..., s2D.phase_func(SA[2π*x2])...)
        inner! = (y, x1) -> (y .= integrand_inner!(x1, x2))
        quadgk!(inner!, data, xs...; quadgk_opts(s)...)
        return data
    end

    quadgk!(integrand_outer!, data, -0.5, 0.0, 0.5; quadgk_opts(s)...)

    return result
end

# this g can be a GreenFunction or a (h1D, s1D) pair
fermi_points_integration_path(g, µ; params...) =
    sanitize_integration_path(fermi_points(g, µ; params...))

function fermi_points(g, µ; params...)
    λs = propagating_eigvals(g, µ + 0im, 1e-2; params...)
    xs = @. real(-im*log(λs)) / (2π)
    return xs
end

function sanitize_integration_path(xs, (xmin, xmax) = (-0.5, 0.5))
    sort!(xs)
    pushfirst!(xs, xmin)
    push!(xs, xmax)
    xs´ = [mean(view(xs, rng)) for rng in approxruns(xs)]  # eliminate approximate duplicates
    return xs´
end

function fermi_h!(s, ϕ, µ, β = 0; params...)
    h = parent(parent(s.gs)) # may be a ParametricHamiltonian
    bs = blockstructure(h)
    # Similar to spectrum(h, ϕ; params...), but less work (no sort! or sanitization)
    copy!(s.hmat, call!(h, ϕ; params...))  # sparse to dense
    ϵs, psis = eigen!(Hermitian(s.hmat))
    # special-casing β = Inf with views turns out to be slower
    fs = (@. ϵs = fermi(ϵs - µ, β))
    fpsis = (s.psis .= psis .* transpose(fs))
    ρcell = EigenProduct(bs, psis, fpsis, ϕ)
    result = call!_output(s.gs)
    getindex!(result, ρcell, s.orbaxes...)
    data = serialize(result)
    return data
end

quadgk_opts(s::DensityMatrixSchurSolver) = quadgk_opts(; s.integrate_opts...)
quadgk_opts(; callback = Return(nothing), quadgk_opts...) = quadgk_opts

callback(s::DensityMatrixSchurSolver) = s.integrate_opts.callback

#endregion top
