############################################################################################
# SparseLU - for 0D AbstractHamiltonians
#   It doesn't use T-matrix for contacts. Instead it incorporates them into the LU factor-
#   ization, possibly using inverse-free self-energies (using extended sites).
#region

struct AppliedSparseLUGreenSolver{C} <: AppliedGreenSolver
    invgreen::InverseGreenBlockSparse{C}
end

mutable struct SparseLUGreenSlicer{C} <:GreenSlicer{C}
    fact::SparseArrays.UMFPACK.UmfpackLU{ComplexF64,Int}  # of full system plus extended orbs
    nonextrng::UnitRange{Int}       # range of non-extended orbital indices
    unitcinds::Vector{Vector{Int}}  # non-extended fact indices per contact
    unitcindsall::Vector{Int}       # merged and uniqued unitcinds
    source64::Matrix{ComplexF64}    # preallocation for ldiv! source @ contacts
    sourceC::Matrix{C}              # alias of source64 or C conversion
    unitg::Matrix{C}                # lazy storage of a full ldiv! solve of all (nonextrng) sites
    function SparseLUGreenSlicer{C}(fact, nonextrng, unitcinds, unitcindsall, source64) where {C}
        s = new()
        s.fact = fact   # Note that there is no SparseArrays.UMFPACK.UmfpackLU{ComplexF32}
        s.nonextrng = nonextrng
        s.unitcinds = unitcinds
        s.unitcindsall = unitcindsall
        s.source64 = source64
        s.sourceC = convert(Matrix{C}, source64)
        # note that unitg is not allocated here. It is allocated on first use.
        return s
    end
end

#region ## API ##

inverse_green(s::AppliedSparseLUGreenSolver) = s.invgreen

unitcellinds_contacts(s::SparseLUGreenSlicer) = s.unitcinds
unitcellinds_contacts(s::SparseLUGreenSlicer, i::Integer) =
    1 <= i <= length(s.unitcinds) ? s.unitcinds[i] :
        argerror("Cannot access contact $i, there are $(length(s.unitcinds)) contacts")
unitcellinds_contacts_merged(s::SparseLUGreenSlicer) = s.unitcindsall

minimal_callsafe_copy(s::AppliedSparseLUGreenSolver) =
    AppliedSparseLUGreenSolver(minimal_callsafe_copy(s.invgreen))

minimal_callsafe_copy(s::SparseLUGreenSlicer{C}) where {C} =
    SparseLUGreenSlicer{C}(s.fact, s.nonextrng, s.unitcinds, s.unitcindsall, copy(s.source64))

#endregion

#region ## apply ##

function apply(::GS.SparseLU, h::AbstractHamiltonian0D, cs::Contacts)
    invgreen = inverse_green(h, cs)
    return AppliedSparseLUGreenSolver(invgreen)
end

apply(::GS.SparseLU, h::AbstractHamiltonian, cs::Contacts) =
    argerror("Can only use GreenSolver.SparseLU with 0D AbstractHamiltonians")

#endregion

#region ## call ##

# Σblocks and contactorbitals are not used here, because they are already inside invgreen
function (s::AppliedSparseLUGreenSolver{C})(ω, Σblocks, contactorbitals) where {C}
    invgreen = s.invgreen
    nonextrng = orbrange(invgreen)
    unitcinds = invgreen.unitcinds
    unitcindsall = invgreen.unitcindsall
    source64 = convert(Matrix{ComplexF64}, s.invgreen.source)
    # the H0 and Σs inside invgreen have already been updated by the parent call!(g, ω; ...)
    update!(invgreen, ω)
    igmat = matrix(invgreen)

    fact = try
        lu(igmat)
    catch
        argerror("Encountered a singular G⁻¹(ω) at ω = $ω, cannot factorize")
    end

    so = SparseLUGreenSlicer{C}(fact, nonextrng, unitcinds, unitcindsall, source64)
    return so
end

#endregion

#endregion

############################################################################################
# SparseLUGreenSlicer indexing
#region

function Base.view(s::SparseLUGreenSlicer, i::Integer, j::Integer)
    dstinds = unitcellinds_contacts(s, i)
    srcinds = unitcellinds_contacts(s, j)
    source64 = view(s.source64, :, 1:length(srcinds))
    sourceC = view(s.sourceC, :, 1:length(srcinds))
    return compute_or_retrieve_green(s, dstinds, srcinds, source64, sourceC)
end

Base.view(s::SparseLUGreenSlicer, ::Colon, ::Colon) =
    compute_or_retrieve_green(s, s.unitcindsall, s.unitcindsall, s.source64, s.sourceC)

function Base.view(s::SparseLUGreenSlicer{C}, i::CellOrbitals, j::CellOrbitals) where {C}
    # cannot use s.source64, because it has only ncols = number of orbitals in contacts
    source64 = similar_source64(s, j)
    sourceC = convert(Matrix{C}, source64)  # will alias if C == ComplexF64
    v = compute_or_retrieve_green(s, orbindices(i), orbindices(j), source64, sourceC)
    return v
end

# Implements cache for full ldiv! solve (unitg)
# source64 and sourceC are of size (total_orbs_including_extended, length(srcinds))
function compute_or_retrieve_green(s::SparseLUGreenSlicer{C}, dstinds, srcinds, source64, sourceC) where {C}
    if isdefined(s, :unitg)
        g = view(s.unitg, dstinds, srcinds)
    else
        fact = s.fact
        allinds = 1:size(fact, 1) # axes not defined on SparseArrays.UMFPACK.UmfpackLU
        one!(source64, srcinds)
        gext = ldiv!(fact, source64)
        sourceC === gext || copy!(sourceC, gext)  # required when C != ComplexF64
        # allinds may include extended orbs -> exclude them from the view
        dstinds´ = ifelse(dstinds === allinds, s.nonextrng, dstinds)
        srcinds´ = convert(typeof(srcinds), 1:length(srcinds))
        g = view(sourceC, dstinds´, srcinds´)
        if srcinds === allinds
            s.unitg = copy(view(gext, s.nonextrng, s.nonextrng))     # exclude extended orbs
        end
    end
    return g
end

similar_source64(s::SparseLUGreenSlicer, ::CellOrbitals{<:Any,Colon}) =
    similar(s.source64, size(s.source64, 1), maximum(s.nonextrng))
similar_source64(s::SparseLUGreenSlicer, j::CellOrbitals) =
    similar(s.source64, size(s.source64, 1), norbitals(j))

# getindex must return a Matrix
Base.getindex(s::SparseLUGreenSlicer, i::CellOrbitals, j::CellOrbitals) = copy(view(s, i, j))

#endregion

#endregion
