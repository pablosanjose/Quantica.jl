############################################################################################
# InverseGreenBlockSparse
#    Specialized variant of BlockSparseMatrix with ω*I and -H as first and second block
#region

struct InverseGreenBlockSparse{C}
    mat::BlockSparseMatrix{C}
    unitcinds::Vector{Vector{Int}}  # orbital indices in parent unitcell of each contact
    unitcindsall::Vector{Int}       # merged, uniqued and sorted contactinds
    source::Matrix{C}               # preallocation for ldiv! solve
end

#region ## Contructor ##

function InverseGreenBlockSparse(h::AbstractHamiltonian{T}, Σs, contacts) where {T}
    hdim = flatsize(h)
    haxis = 1:hdim
    ωblock = MatrixBlock((zero(Complex{T}) * I)(hdim), haxis, haxis)
    hblock = MatrixBlock(call!_output(h), haxis, haxis)
    extoffset = hdim
    # these are indices of contact orbitals within the merged orbital slice
    unitcinds = unit_contact_inds(contacts)
    # holds all non-extended orbital indices
    unitcindsall = unique!(sort!(reduce(vcat, unitcinds)))
    checkcontactindices(unitcindsall, hdim)
    solvers = solver.(Σs)
    blocks = selfenergyblocks!(extoffset, unitcinds, 1, (ωblock, -hblock), solvers...)
    mat = BlockSparseMatrix(blocks...)
    source = zeros(Complex{T}, size(mat, 2), length(unitcindsall))
    return InverseGreenBlockSparse(mat, unitcinds, unitcindsall, source)
end

# switch from contactinds (relative to merged contact orbslice) to unitcinds (relative
# to parent unitcell)
function unit_contact_inds(contacts)
    orbindsall = siteindices(only(subcells(orbslice(contacts))))
    unitcinds = [orbindsall[cinds] for cinds in contactinds(contacts)]
    return unitcinds
end

selfenergyblocks!(extoffset, contactinds, ci, blocks) = blocks

function selfenergyblocks!(extoffset, contactinds, ci, blocks, s::RegularSelfEnergySolver, ss...)
    c = contactinds[ci]
    Σblock = MatrixBlock(call!_output(s), c, c)
    return selfenergyblocks!(extoffset, contactinds, ci + 1, (blocks..., -Σblock), ss...)
end

function selfenergyblocks!(extoffset, contactinds, ci, blocks, s::ExtendedSelfEnergySolver, ss...)
    Σᵣᵣ, Vᵣₑ, gₑₑ⁻¹, Vₑᵣ = shiftedmatblocks(call!_output(s), contactinds[ci], extoffset)
    extoffset += size(gₑₑ⁻¹, 1)
    return selfenergyblocks!(extoffset, contactinds, ci + 1, (blocks..., -Σᵣᵣ, -Vᵣₑ, -gₑₑ⁻¹, -Vₑᵣ), ss...)
end

function shiftedmatblocks((Σᵣᵣ, Vᵣₑ, gₑₑ⁻¹, Vₑᵣ)::Tuple{AbstractArray}, cinds, shift)
    extsize = size(gₑₑ⁻¹, 1)
    Σᵣᵣ´ = MatrixBlock(Σᵣᵣ, cinds, cinds)
    Vᵣₑ´ = MatrixBlock(Vᵣₑ, cinds, shift+1:shift+extsize)
    Vₑᵣ´ = MatrixBlock(Vₑᵣ, shift+1:shift+extsize, cinds)
    gₑₑ⁻¹ = MatrixBlock(gₑₑ⁻¹, shift+1:shift+extsize, shift+1:shift+extsize)
    return Σᵣᵣ´, Vᵣₑ´, gₑₑ⁻¹´, Vₑᵣ´
end

checkcontactindices(allcontactinds, hdim) = maximum(allcontactinds) <= hdim ||
    internalerror("InverseGreenBlockSparse: unexpected contact indices beyond Hamiltonian dimension")

#endregion

#region ## API ##

SparseArrays.sparse(s::InverseGreenBlockSparse) = sparse(s.mat)

function update!(s::InverseGreenBlockSparse, ω)
    bsm = s.mat
    Imat = blockmat(first(blocks(bsm)))
    Imat.diag .= ω   # Imat should be <: Diagonal
    return update!(bsm)
end

#endregion
#endregion

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

function apply(::GS.SparseLU, oh::OpenHamiltonian{<:Any,<:Any,0}, cs::Contacts)
    invgreen = InverseGreenBlockSparse(hamiltonian(oh), selfenergies(oh), cs)
    return AppliedSparseLU(invgreen)
end

apply(::GS.SparseLU, ::OpenHamiltonian) =
    argerror("Can only use SparseLU with bounded Hamiltonians")

function (s::AppliedSparseLU{C})(ω, Σs, bs) where {C}
    invgreen = s.invgreen
    unitcinds = invgreen.unitcinds
    unitcindsall = invgreen.unitcindsall
    source = s.invgreen.source
    # the H0 and Σs wrapped in invgreen have already been updated by the parent call!
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

minimal_callsafe_copy(s::SparseLUSlicer) =
    SparseLUSlicer(s.fact, s.unitcinds, s.unitcindsall, copy(s.source))

#endregion

############################################################################################
# SparseLUSlicer indexing
#region

function Base.view(s::SparseLUSlicer{C}, i::ContactIndex, j::ContactIndex) where {C}
    fact = s.fact
    srcinds = s.unitcinds[Int(j)]
    dstinds = s.unitcinds[Int(i)]
    source = view(s.source, :, 1:length(srcinds))
    fill!(source, zero(C))
    for (col, row) in enumerate(srcinds)
        source[row, col] = one(C)
    end
    gext = ldiv!(fact, source)
    g = view(gext, dstinds, :)
    return g
end

function Base.view(s::SparseLUSlicer{C}, ::Colon, ::Colon) where {C}
    fact = s.fact
    source = s.source
    allinds = s.unitcindsall
    fill!(source, zero(C))
    for col in axes(source, 2)
        source[allinds[col], col] = one(C)
    end
    gext = ldiv!(fact, source)
    g = view(gext, allinds, :)
    return g
end

#endregion

#endregion

