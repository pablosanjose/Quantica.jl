############################################################################################
# SparseLU - for 0D AbstractHamiltonians
#region

struct AppliedSparseLU{C} <:AppliedGreenSolver
    invgreen::InverseGreenBlockSparse{C}
end

struct SparseLUSlicer{C} <:GreenSlicer
    fact::SparseArrays.UMFPACK.UmfpackLU{C,Int}  # of full system plus extended orbs
    unitcinds::Vector{Vector{Int}}               # non-extended fact indices per contact
    unitcindsall::Vector{Int}                    # merged and uniqued unitcinds
    source::Matrix{C}                            # preallocation for ldiv! solve
end

#region ## API ##

function apply(::GS.SparseLU, h::AbstractHamiltonian0D, cs::Contacts)
    invgreen = inverse_green(h, cs)
    return AppliedSparseLU(invgreen)
end

apply(::GS.SparseLU, ::OpenHamiltonian) =
    argerror("Can only use SparseLU with bounded Hamiltonians")

# Σblocks and contactblockstruct are not used here, because they are already inside invgreen
function (s::AppliedSparseLU)(ω, Σblocks, contactblockstruct)
    invgreen = s.invgreen
    unitcinds = invgreen.unitcinds
    unitcindsall = invgreen.unitcindsall
    source = s.invgreen.source
    # the H0 and Σs inside invgreen have already been updated by the parent call!(g, ω; ...)
    update!(invgreen, ω)
    igmat = sparse(invgreen)
    fact = try
        lu(igmat)
    catch
        argerror("Encountered a singular G⁻¹(ω) at ω = $ω, cannot factorize")
    end
    so = SparseLUSlicer(fact, unitcinds, unitcindsall, source)
    return so
end

unitcellinds_contacts(s::SparseLUSlicer) = s.unitcinds
unitcellinds_contacts_merged(s::SparseLUSlicer) = s.unitcindsall

minimal_callsafe_copy(s::SparseLUSlicer) =
    SparseLUSlicer(s.fact, s.unitcinds, s.unitcindsall, copy(s.source))

#endregion

############################################################################################
# SparseLUSlicer indexing
#region

function Base.view(s::SparseLUSlicer, i::ContactIndex, j::ContactIndex)
    dstinds = s.unitcinds[Int(i)]
    srcinds = s.unitcinds[Int(j)]
    source = view(s.source, :, 1:length(srcinds))
    return _view(s, dstinds, srcinds, source)
end

Base.view(s::SparseLUSlicer, ::Colon, ::Colon) =
    _view(s, s.unitcindsall, s.unitcindsall, s.source)

function _view(s::SparseLUSlicer{C}, dstinds::Vector{Int}, srcinds::Vector{Int}, source) where {C}
    fact = s.fact
    one!(source, srcinds)
    gext = ldiv!(fact, source)
    g = view(gext, dstinds, :)
    return g
end

function Base.getindex(s::SparseLUSlicer, i::CellOrbitals, j::CellOrbitals)
    # cannot use source, because it has only ncols = number of orbitals in contacts
    source = similar(s.source, size(s.source, 1), norbs(j))
    return _view(s, orbindices(i), orbindices(j), source)
end

#endregion

#endregion

