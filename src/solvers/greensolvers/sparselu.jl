############################################################################################
# InverseGreenBlockSparse
#    Specialized variant of BlockSparseMatrix with ω*I and -H as first and second block
#region

struct InverseGreenBlockSparse{C}
    mat::BlockSparseMatrix{C}
    contactinds::Vector{Int}         # uniqued and sorted non-extended contact indices
end

#region ## Contructor ##

function InverseGreenBlockSparse(h::AbstractHamiltonian{T}, Σs) where {T}
    hdim = flatsize(h)
    haxis = 1:hdim
    ωblock = MatrixBlock((zero(Complex{T}) * I)(hdim), haxis, haxis)
    hblock = MatrixBlock(call!_output(h), haxis, haxis)
    extoffset = hdim
    contactinds = Int[]
    solvers = solver.(Σs)
    blocks = selfenergyblocks!(extoffset, contactinds, (ωblock, -hblock), solvers...)
    unique!(sort!(contactinds))
    checkcontactindices(contactinds, hdim)
    mat = BlockSparseMatrix(blocks...)
    return InverseGreenBlockSparse(mat, contactinds)
end

selfenergyblocks!(extoffset, contactinds, blocks) = blocks

function selfenergyblocks!(extoffset, contactinds, blocks, s::RegularSelfEnergySolver, ss...)
    block = call!_output(s)
    appendinds!(contactinds, block)
    return selfenergyblocks!(extoffset, contactinds, (blocks..., -block), ss...)
end

function selfenergyblocks!(extoffset, contactinds, blocks, s::ExtendedSelfEnergySolver, ss...)
    Σᵣᵣ, Vᵣₑ, gₑₑ⁻¹, Vₑᵣ = blockshift!(call!_output(s), extoffset)
    appendinds!(contactinds, Σᵣᵣ)
    extoffset += size(gₑₑ⁻¹, 1)
    return selfenergyblocks!(extoffset, contactinds, (blocks..., -Σᵣᵣ, -Vᵣₑ, -gₑₑ⁻¹, -Vₑᵣ), ss...)
end

function appendinds!(contactinds, block)
    blockrows(block) == blockcols(block) ||
        internalerror("Unexpected mismatch between rows and columns of self-energy block")
    return append!(contactinds, blockrows(block))
end

checkcontactindices(contactinds, hdim) = maximum(contactinds) <= hdim ||
        internalerror("Unexpected contact indices beyond Hamiltonian dimension")

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

struct ExecutedSparseLU{C} <:GreenMatrixSolver
    fact::SparseArrays.UMFPACK.UmfpackLU{C,Int}  # of full system plus extended orbs
    blockstruct::MultiBlockStructure{0}          # of merged contact LatticeSlice
    contactinds::Vector{Int}                     # non-extended contact indices
    source::Matrix{C}                            # preallocation for ldiv! solve
end

#region ## API ##

function apply(::GS.SparseLU, oh::OpenHamiltonian{<:Any,<:Any,0})
    invgreen = InverseGreenBlockSparse(hamiltonian(oh), selfenergies(oh))
    return AppliedSparseLU(invgreen)
end

apply(::GS.SparseLU, ::OpenHamiltonian) =
    argerror("Can only use SparseLU with bounded Hamiltonians")

call!(s::AppliedSparseLU; params...) = s

function call!(s::AppliedSparseLU{C}, h, contacts, ω; params...) where {C}
    call!(h, (); params...)       # call!_output is wrapped in invgreen's hblock - update it
    Σs = call!(contacts, ω; params...)                        # same for invgreen's Σ blocks
    invgreen = s.invgreen
    cinds = invgreen.contactinds
    bs = blockstruct(contacts)
    update!(invgreen, ω)
    igmat = sparse(invgreen)
    fact = try
        lu(igmat)
    catch
        argerror("Encountered a singular G⁻¹(ω) at ω = $ω, cannot factorize")
    end
    source = zeros(C, size(igmat, 2), length(cinds))
    so = ExecutedSparseLU(fact, bs, cinds, source)
    gc = so()
    Γs = linewidth.(Σs)
    ls = latslice(contacts)
    return GreenMatrix(so, gc, Γs, ls)
end

function (s::ExecutedSparseLU{C})() where {C}
    bs = s.blockstruct
    fact = s.fact
    cinds = s.contactinds
    source = s.source
    fill!(source, zero(C))
    for (col, row) in enumerate(cinds)
        source[row, col] = one(C)
    end
    gext = ldiv!(fact, source)
    g = HybridMatrix(gext[cinds, :], bs)
    return g
end

#endregion

#endregion

