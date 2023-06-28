
const GreenFunctionSchurEmptyLead{T,E} = GreenFunction{T,E,1,<:AppliedSchurGreenSolver,<:Any,<:EmptyContacts}
const GreenFunctionSchurLead{T,E} = GreenFunction{T,E,1,<:AppliedSchurGreenSolver,<:Any,<:Any}

############################################################################################
# SelfEnergy(h, glead::GreenFunctionSchurEmptyLead; kw...)
#   Extended self energy solver for deflated ΣL or ΣR Schur factors of lead unitcell
#region

struct SelfEnergySchurSolver{T,B,V<:Union{Missing,Vector{Int}},H<:AbstractHamiltonian} <: ExtendedSelfEnergySolver
    fsolver::SchurFactorsSolver{T,B}
    hlead::H
    isleftside::Bool
    leadtoparent::V  # orbital index in parent for each orbital in open lead surface
                     # so that Σlead[leadtoparent, leadtoparent] == Σparent
end


#region ## Constructors ##

SelfEnergySchurSolver(fsolver::SchurFactorsSolver, hlead, side::Symbol, parentinds = missing) =
    SelfEnergySchurSolver(fsolver, hlead, isleftside(side), parentinds)

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

# This syntax checks that the selected sites of hparent match the L/R surface of the
# semi-infinite lead (possibly by first transforming the lead lattice with `transform`)
# and if so, builds the extended Self Energy directly, using the same intercell coupling of
# the lead, but using the correct site order of hparent
function SelfEnergy(hparent::AbstractHamiltonian, glead::GreenFunctionSchurEmptyLead; reverse = false, transform = missing, kw...)
    sel = siteselector(; kw...)
    lsparent = lattice(hparent)[sel]
    schursolver = solver(glead)
    fsolver = schurfactorsolver(schursolver)
    isfinite(schursolver.boundary) ||
        argerror("The form attach(h, glead; sites...) assumes a semi-infinite lead, but got `boundary = Inf`")
    # we obtain latslice of open surface in gL/gR
    gunit = reverse ? schursolver.gL : schursolver.gR
    blocksizes(blockstructure(hamiltonian(gunit))) == blocksizes(blockstructure(hparent)) ||
        argerror("The orbital structure of parent and lead Hamiltonians do not match. Maybe you meant to use `attach(h, g1D, coupling; sites...)`?")
    # This is a SelfEnergy for a lead unit cell with a SelfEnergySchurSolver
    Σlead = only(selfenergies(contacts(gunit)))
    lslead = latslice(Σlead)
    # find lead site index in lslead for each site in lsparent
    leadsites, displacement = lead_siteind_foreach_parent_siteind(lsparent, lslead, transform)
    # convert lead site indices to lead orbital indices using lead's ContactBlockStructure
    leadcbs = blockstructure(contacts(gunit))
    leadorbs = contact_sites_to_orbitals(leadsites, leadcbs)
    # translate glead unitcell by displacement, so it overlaps sel sites (modulo transform)
    hlead = copy(parent(glead))
    transform === missing || Quantica.transform!(hlead, transform)
    translate!(hlead, displacement)
    solver´ = SelfEnergySchurSolver(fsolver, hlead, reverse, leadorbs)

    reverse && Base.reverse!(hlead)
    plottables = (hlead,)
    return SelfEnergy(solver´, lsparent, plottables)
end

# find ordering of lslead sites that match lsparent sites, modulo a displacement
function lead_siteind_foreach_parent_siteind(lsparent, lslead, transform)
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

# This solver produces two solutions (L/R) for the price of one. We can opt out of calling
# it if we know it has already been called, so the solution is already in its call!_output
function call!(s::SelfEnergySchurSolver, ω;
               skipsolve_internal = false, params...)
    fsolver = s.fsolver
    Rfactors, Lfactors = if skipsolve_internal
        call!_output(fsolver)
    else
        # first apply params to the lead Hamiltonian
        call!(s.hlead; params...)
        call!(fsolver, ω)
    end
    factors = maybe_match_parent(ifelse(s.isleftside, Lfactors, Rfactors), s.leadtoparent)
    return factors
end

call!_output(s::SelfEnergySchurSolver) = call!(s, missing; skipsolve_internal = true)

maybe_match_parent((V, ig, V´), leadtoparent) =
    (view(V, leadtoparent, :), ig, view(V´, :, leadtoparent))

maybe_match_parent(factors, ::Missing) = factors

minimal_callsafe_copy(s::SelfEnergySchurSolver) =
    SelfEnergySchurSolver(minimal_callsafe_copy(s.fsolver), minimal_callsafe_copy(s.hlead),
        s.isleftside, s.leadtoparent)

#endregion

#endregion

############################################################################################
# SelfEnergy(h, glead, model::AbstractModel; kw...)
#   Depending on whether glead has zero contacts or not, it yields an Extended or Generic
#       self-energy. Model gives arbitrary couplings to parent Hamiltonian.
#   For the Extended self energy, is uses g⁻¹ = h0 + ExtendedSchurΣ  (adds extended sites)
#   V and V´ are SparseMatrixView because they span only the coupled parent <-> unitcell
#       sites, but for Extended they need to be padded with zeros over the extended sites
#region

struct SelfEnergyCouplingSchurSolver{C,G,H,S<:SparseMatrixView,S´<:SparseMatrixView} <: ExtendedSelfEnergySolver
    gunit::G
    hcoupling::H
    V´::S´                              # aliases a view of hcoupling
    g⁻¹::InverseGreenBlockSparse{C}     # aliases the one in solver(gunit)::AppliedSparseLUGreenSolver
    V::S                                # aliases a view of hcoupling
end

#region ## Constructors ##

# This is similar to SelfEnergyGenericSolver in generic.jl, but <: ExtendedSelfEnergySolver
# It relies on gunit having a AppliedSparseLUGreenSolver, so a sparse g⁻¹ can be extracted
function SelfEnergyCouplingSchurSolver(gunit::GreenFunction{T}, hcoupling::AbstractHamiltonian{T}, nparent) where {T}
    hmatrix = call!_output(hcoupling)
    invgreen = inverse_green(solver(gunit))       # the solver is AppliedSparseLUGreenSolver
    lastflatparent = last(flatrange(hcoupling, nparent))
    size(hmatrix, 1) == lastflatparent + length(orbrange(invgreen)) ||
        internalerror("SelfEnergyCouplingSchurSolver builder: $(size(hmatrix, 1)) != $lastflatparent + $(length(orbrange(invgreen)))")
    parentrng, leadrng = 1:lastflatparent, lastflatparent+1:size(hmatrix, 1)
    sizeV = size(invgreen, 1), lastflatparent
    V = SparseMatrixView(view(hmatrix, leadrng, parentrng), sizeV)
    V´ = SparseMatrixView(view(hmatrix, parentrng, leadrng), reverse(sizeV))
    return SelfEnergyCouplingSchurSolver(gunit, hcoupling, V´, invgreen, V)
end

#endregion

#region ## API ##

# With this syntax we attach the surface unit cell of the lead (left or right) to hparent
# through the model coupling. The lead is transformed with `transform` to align it to
# hparent. Then we apply the model to the 0D lattice of hparent's selected surface plus the
# lead unit cell, and then build an extended self energy
function SelfEnergy(hparent::AbstractHamiltonian, glead::GreenFunctionSchurLead, model::AbstractModel;
                    reverse = false, transform = missing, kw...)
    schursolver = solver(glead)
    gunit = copy_lattice(reverse ? schursolver.gL : schursolver.gR)
    lat0lead = lattice(gunit)            # lat0lead is the zero cell of parent(glead)
    hlead = copy(parent(glead))          # hlead is used only for plottables

    # move hlead and lat0lead to the left or right of boundary (if boundary is finite)
    boundary = schursolver.boundary
    xunit = isfinite(boundary) ? boundary + ifelse(reverse, -1, 1) : zero(boundary)
    if !iszero(xunit)
        bm = bravais_matrix(hlead)
        translate!(hlead, bm * SA[xunit])
        translate!(lat0lead, bm * SA[xunit])
    end

    # apply transform
    if transform !== missing
        transform!(lat0lead, transform)
        transform!(hlead, transform)
    end

    # combine gunit and parent sites into lat0
    sel = siteselector(; kw...)
    lsparent = lattice(hparent)[sel]
    lat0parent = lattice0D(lsparent)
    lat0 = combine(lat0parent, lat0lead)
    nparent, ntotal = nsites(lat0parent), nsites(lat0)

    # apply model to lat0 to get hcoupling
    interblockmodel = interblock(model, 1:nparent, nparent+1:ntotal)
    hcoupling = hamiltonian(lat0, interblockmodel;
        orbitals = vcat(norbitals(hparent), norbitals(hlead)))

    gslice = glead[cells = SA[xunit]]
    Σs = selfenergies(contacts(glead))
    solver´ = extended_or_regular_solver(Σs, gslice, gunit, hcoupling, nparent)

    reverse && Base.reverse!(hlead)
    plottables = (hcoupling, hlead)

    return SelfEnergy(solver´, lsparent, plottables)
end

# No contacts -> Extended solver
extended_or_regular_solver(::Tuple{}, gslice, gunit, hcoupling, nparent) =
    SelfEnergyCouplingSchurSolver(gunit, hcoupling, nparent)

# With contacts -> Regular Generic solver - see generic.jl
extended_or_regular_solver(::Tuple, gslice, gunit, hcoupling, nparent) =
    SelfEnergyGenericSolver(gslice, hcoupling, nparent)

function call!(s::SelfEnergyCouplingSchurSolver, ω; params...)
    call!(s.hcoupling; params...)
    call!(s.gunit, ω; params...)
    update!(s.V)
    update!(s.V´)
    return matrix(s.V´), matrix(s.g⁻¹), matrix(s.V)
end

call!_output(s::SelfEnergyCouplingSchurSolver) = matrix(s.V´), matrix(s.g⁻¹), matrix(s.V)

minimal_callsafe_copy(s::SelfEnergyCouplingSchurSolver) =
    SelfEnergyCouplingSchurSolver(
        minimal_callsafe_copy(s.gunit),
        minimal_callsafe_copy(s.hcoupling),
        minimal_callsafe_copy(s.V´),
        minimal_callsafe_copy(s.g⁻¹),
        minimal_callsafe_copy(s.V))

#endregion

#endregion

############################################################################################
# SelfEnergy(h, glead::GreenFunctionSchurLead; kw...)
#   Regular (Generic) self energy, since Extended is not possible for lead with contacts
#   Otherwise equivalent to SelfEnergy(h, glead::GreenFunctionSchurEmptyLead; kw...)
#region

function SelfEnergy(hparent::AbstractHamiltonian, glead::GreenFunctionSchurLead; reverse = false, transform = missing, sites...)
    blocksizes(blockstructure(hamiltonian(glead))) == blocksizes(blockstructure(hparent)) ||
        argerror("The orbital structure of parent and lead Hamiltonians do not match")
    # find boundary ± 1
    schursolver = solver(glead)
    boundary = schursolver.boundary
    isfinite(boundary) ||
        argerror("The form attach(h, glead; sites...) assumes a semi-infinite lead, but got `boundary = Inf`")
    xunit = boundary + ifelse(reverse, -1, 1)
    gslice = glead[cells = SA[xunit]]
    # lattice slices for parent and lead unit cell
    lsparent = getindex(lattice(hparent); sites...)
    lslead = lattice(glead)[cells = SA[xunit]]
    # find lead site index in lslead for each site in lsparent
    leadsites, displacement = lead_siteind_foreach_parent_siteind(lsparent, lslead, transform)
    # convert lead site indices to lead orbital indices using lead's ContactBlockStructure
    leadbs = blockstructure(glead)                              # This is a BlockStructure
    leadorbs = contact_sites_to_orbitals(leadsites, leadbs)
    # build V and V´ as a leadorbs reordering of inter-cell harmonics of hlead
    hlead = copy(hamiltonian(glead))  # careful, not parent, which could be a ParametricHamiltonian
    h₊₁, h₋₁ = hlead[SA[1]], hlead[SA[-1]]
    V  = SparseMatrixView(view(h₊₁, :, leadorbs))
    V´ = SparseMatrixView(view(h₋₁, leadorbs, :))
    solver´ = SelfEnergyGenericSolver(gslice, hlead, V´, V)

    reverse && Base.reverse!(hlead)
    plottables = (hlead,)

    return SelfEnergy(solver´, lsparent, plottables)
end

#endregion
