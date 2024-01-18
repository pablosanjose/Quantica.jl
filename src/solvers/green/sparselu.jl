############################################################################################
# SparseLU - for 0D AbstractHamiltonians
#region

struct AppliedSparseLUGreenSolver{C} <:AppliedGreenSolver
    invgreen::InverseGreenBlockSparse{C}
end

mutable struct SparseLUSlicer{C} <:GreenSlicer{C}
    fact::SparseArrays.UMFPACK.UmfpackLU{C,Int}  # of full system plus extended orbs
    nonextrng::UnitRange{Int}                    # range of non-extended orbital indices
    unitcinds::Vector{Vector{Int}}               # non-extended fact indices per contact
    unitcindsall::Vector{Int}                    # merged and uniqued unitcinds
    source::Matrix{C}                            # preallocation for ldiv! source @ contacts
    unitg::Matrix{C}                             # lazy storage of a full ldiv! solve of all (nonextrng) sites
    function SparseLUSlicer{C}(fact, nonextrng, unitcinds, unitcindsall, source) where {C}
        s = new()
        s.fact = fact
        s.nonextrng = nonextrng
        s.unitcinds = unitcinds
        s.unitcindsall = unitcindsall
        s.source = source
        return s
    end
end

SparseLUSlicer(fact::Factorization{C}, nonextrng, unitcinds, unitcindsall, source) where {C} =
    SparseLUSlicer{C}(fact, nonextrng, unitcinds, unitcindsall, source)

#region ## API ##

inverse_green(s::AppliedSparseLUGreenSolver) = s.invgreen

unitcellinds_contacts(s::SparseLUSlicer) = s.unitcinds
unitcellinds_contacts(s::SparseLUSlicer, i::Integer) =
    1 <= i <= length(s.unitcinds) ? s.unitcinds[i] :
        argerror("Cannot access contact $i, there are $(length(s.unitcinds)) contacts")
unitcellinds_contacts_merged(s::SparseLUSlicer) = s.unitcindsall

minimal_callsafe_copy(s::AppliedSparseLUGreenSolver) =
    AppliedSparseLUGreenSolver(minimal_callsafe_copy(s.invgreen))

minimal_callsafe_copy(s::SparseLUSlicer) =
    SparseLUSlicer(s.fact, s.nonextrng, s.unitcinds, s.unitcindsall, copy(s.source))

#endregion

#region ## apply ##

function apply(::GS.SparseLU, h::AbstractHamiltonian0D, cs::Contacts)
    invgreen = inverse_green(h, cs)
    return AppliedSparseLUGreenSolver(invgreen)
end

apply(::GS.SparseLU, h::AbstractHamiltonian, cs::Contacts) =
    argerror("Can only use SparseLU with bounded Hamiltonians")

#endregion

#region ## call ##

# Σblocks and contactorbitals are not used here, because they are already inside invgreen
function (s::AppliedSparseLUGreenSolver{C})(ω, Σblocks, contactorbitals) where {C}
    invgreen = s.invgreen
    nonextrng = orbrange(invgreen)
    unitcinds = invgreen.unitcinds
    unitcindsall = invgreen.unitcindsall
    source = s.invgreen.source
    # the H0 and Σs inside invgreen have already been updated by the parent call!(g, ω; ...)
    update!(invgreen, ω)
    igmat = matrix(invgreen)

    fact = try
        lu(igmat)
    catch
        argerror("Encountered a singular G⁻¹(ω) at ω = $ω, cannot factorize")
    end

    so = SparseLUSlicer{C}(fact, nonextrng, unitcinds, unitcindsall, source)
    return so
end

#endregion

#endregion

############################################################################################
# SparseLUSlicer indexing
#region

function Base.view(s::SparseLUSlicer, i::Integer, j::Integer)
    dstinds = unitcellinds_contacts(s, i)
    srcinds = unitcellinds_contacts(s, j)
    source = view(s.source, :, 1:length(srcinds))
    return compute_or_retrieve_green(s, dstinds, srcinds, source)
end

Base.view(s::SparseLUSlicer, ::Colon, ::Colon) =
    compute_or_retrieve_green(s, s.unitcindsall, s.unitcindsall, s.source)

function Base.view(s::SparseLUSlicer, i::CellOrbitals, j::CellOrbitals)
    # cannot use s.source, because it has only ncols = number of orbitals in contacts
    source = similar_source(s, j)
    v = compute_or_retrieve_green(s, orbindices(i), orbindices(j), source)
    return v
end

# Implements cache for full ldiv! solve (unitg)
function compute_or_retrieve_green(s::SparseLUSlicer{C}, dstinds, srcinds, source) where {C}
    if isdefined(s, :unitg)
        g = view(s.unitg, dstinds, srcinds)
    else
        fact = s.fact
        allinds = 1:size(fact, 1)
        one!(source, srcinds)
        gext = ldiv!(fact, source)
        dstinds´ = ifelse(dstinds === allinds, s.nonextrng, dstinds)
        g = view(gext, dstinds´, :)
        if srcinds === allinds
            s.unitg = copy(view(gext, s.nonextrng, s.nonextrng))
        end
    end
    return g
end

similar_source(s::SparseLUSlicer, ::CellOrbitals{<:Any,Colon}) =
    similar(s.source, size(s.source, 1), maximum(s.nonextrng))
similar_source(s::SparseLUSlicer, j::CellOrbitals) =
    similar(s.source, size(s.source, 1), norbitals(j))

# getindex must return a Matrix
Base.getindex(s::SparseLUSlicer, i::CellOrbitals, j::CellOrbitals) = copy(view(s, i, j))

#endregion

#endregion
