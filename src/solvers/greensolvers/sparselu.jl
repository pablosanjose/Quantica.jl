############################################################################################
# InverseGreenBlockSparse
#    Specialized variant of BlockSparseMatrix with ω*I and -H as first and second block
#region

struct InverseGreenBlockSparse{C}
    mat::BlockSparseMatrix{C}
    allcontactinds::Vector{Int}    # merged, uniqued and sorted non-extended contact indices
end

#region ## Contructor ##

function InverseGreenBlockSparse(h::AbstractHamiltonian{T}, Σs, contacts) where {T}
    hdim = flatsize(h)
    haxis = 1:hdim
    ωblock = MatrixBlock((zero(Complex{T}) * I)(hdim), haxis, haxis)
    hblock = MatrixBlock(call!_output(h), haxis, haxis)
    extoffset = hdim
    cinds = contactinds(contacts)
    # holds all non-extended orbital indices
    allcontactinds = unique!(sort!(reduce(vcat, cinds)))
    checkcontactindices(allcontactinds, hdim)
    solvers = solver.(Σs)
    blocks = selfenergyblocks!(extoffset, cinds, 1, (ωblock, -hblock), solvers...)
    blocks = (ωblock, -hblock, blocks...)
    mat = BlockSparseMatrix(blocks...)
    return InverseGreenBlockSparse(mat, allcontactinds)
end

selfenergyblocks!(extoffset, contactinds, ci, blocks) = blocks

function selfenergyblocks!(extoffset, contactinds, ci, blocks, s::RegularSelfEnergySolver, ss...)
    cinds = contactinds[ci]
    Σblock = MatrixBlock(call!_output(s), cinds, cinds)
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

struct SparseLUSolution{C} <:GreenSlicer
    fact::SparseArrays.UMFPACK.UmfpackLU{C,Int}  # of full system plus extended orbs
    allcontactinds::Vector{Int}                  # all non-extended contact indices
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
    allcinds = invgreen.allcontactinds
    # the H0 and Σs wrapped in invgreen have already been updated by the parent call!
    update!(invgreen, ω)
    igmat = sparse(invgreen)
    fact = try
        lu(igmat)
    catch
        argerror("Encountered a singular G⁻¹(ω) at ω = $ω, cannot factorize")
    end
    source = zeros(C, size(igmat, 2), length(allcinds))
    so = SparseLUSolution(fact, allcinds, source)
    return GreenSolution(so, Σs, bs)
end

function (s::SparseLUSolution{C})() where {C}
    fact = s.fact
    allcinds = s.allcontactinds
    source = s.source
    fill!(source, zero(C))
    for (col, row) in enumerate(allcinds)
        source[row, col] = one(C)
    end
    gext = ldiv!(fact, source)
    g = gext[allcinds, :]
    return g
end

#endregion

#endregion

