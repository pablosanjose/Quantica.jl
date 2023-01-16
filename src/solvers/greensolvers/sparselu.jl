############################################################################################
# InverseGreenBlockSparse
#    Specialized variant of BlockSparseMatrix with ω*I and -H as first and second block
#region

struct InverseGreenBlockSparse{C}
    mat::BlockSparseMatrix{C}
    allcontactinds::Vector{Int}    # merged, uniqued and sorted non-extended contact indices
end

#region ## Contructor ##

function InverseGreenBlockSparse(h::AbstractHamiltonian{T}, Σs) where {T}
    hdim = flatsize(h)
    haxis = 1:hdim
    ωblock = MatrixBlock((zero(Complex{T}) * I)(hdim), haxis, haxis)
    hblock = MatrixBlock(call!_output(h), haxis, haxis)
    extoffset = hdim
    allcontactinds = Int[]
    solvers = solver.(Σs)
    blocks = selfenergyblocks!(extoffset, allcontactinds, (ωblock, -hblock), solvers...)
    unique!(sort!(allcontactinds))
    checkcontactindices(allcontactinds, hdim)
    mat = BlockSparseMatrix(blocks...)
    return InverseGreenBlockSparse(mat, allcontactinds)
end

selfenergyblocks!(extoffset, allcontactinds, blocks) = blocks

function selfenergyblocks!(extoffset, allcontactinds, blocks, s::RegularSelfEnergySolver, ss...)
    block = call!_output(s)
    appendinds!(allcontactinds, block)
    return selfenergyblocks!(extoffset, allcontactinds, (blocks..., -block), ss...)
end

function selfenergyblocks!(extoffset, allcontactinds, blocks, s::ExtendedSelfEnergySolver, ss...)
    Σᵣᵣ, Vᵣₑ, gₑₑ⁻¹, Vₑᵣ = blockshift!(call!_output(s), extoffset)
    appendinds!(allcontactinds, Σᵣᵣ)
    extoffset += size(gₑₑ⁻¹, 1)
    return selfenergyblocks!(extoffset, allcontactinds, (blocks..., -Σᵣᵣ, -Vᵣₑ, -gₑₑ⁻¹, -Vₑᵣ), ss...)
end

function appendinds!(allcontactinds, block)
    blockrows(block) == blockcols(block) ||
        internalerror("Unexpected mismatch between rows and columns of self-energy block")
    return append!(allcontactinds, blockrows(block))
end

checkcontactindices(allcontactinds, hdim) = maximum(allcontactinds) <= hdim ||
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

struct ExecutedSparseLU{C} <:GreenFixedSolver
    fact::SparseArrays.UMFPACK.UmfpackLU{C,Int}  # of full system plus extended orbs
    allcontactinds::Vector{Int}                  # all non-extended contact indices
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
    allcinds = invgreen.allcontactinds
    cinds = contactinds(contacts)
    update!(invgreen, ω)
    igmat = sparse(invgreen)
    fact = try
        lu(igmat)
    catch
        argerror("Encountered a singular G⁻¹(ω) at ω = $ω, cannot factorize")
    end
    source = zeros(C, size(igmat, 2), length(allcinds))
    so = ExecutedSparseLU(fact, allcinds, source)
    gc = so()
    Γs = linewidth.(Σs)
    ls = latslice(contacts)
    return GreenFixed(so, gc, Γs, cinds, ls)
end

function (s::ExecutedSparseLU{C})() where {C}
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

