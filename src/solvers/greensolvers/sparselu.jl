############################################################################################
# InverseGreenBlockSparse
#    Specialized variant of BlockSparseMatrix with ω*I and -H as first and second block
#region

struct InverseGreenBlockSparse{C}
    mat::BlockSparseMatrix{C}
    contactinds::Vector{Int}         # uniqued and sorted non-extended contact indices
end

#region ## Contructor ##

function InverseGreenBlockSparse(h::AbstractHamiltonian{T}, c::Contacts) where {T}
    hdim = flatsize(h)
    haxis = 1:hdim
    ωblock = MatrixBlock((zero(Complex{T}) * I)(hdim), haxis, haxis)
    hblock = MatrixBlock(call!_output(h), haxis, haxis)
    extoffset = hdim
    contactinds = Int[]
    solvers = solver.(selfenergies(c))
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
    ig::Ref{InverseGreenBlockSparse{C}}          # is only assigned upon first call!
    blockstruct::Ref{MultiBlockStructure{0}}     # is only assigned upon first call!
end

struct ExecutedSparseLU{C} <:GreenMatrixSolver
    fact::SparseArrays.UMFPACK.UmfpackLU{C,Int}  # of full system plus extended orbs
    blockstruct::MultiBlockStructure{0}          # of merged contact LatticeSlice
    contactinds::Vector{Int}                     # non-extended contact indices
end

#region ## API ##

AppliedSparseLU{C}() where {C} =
    AppliedSparseLU{C}(Ref{InverseGreenBlockSparse{C}}(), Ref{MultiBlockStructure{0}}())

#endregion

#region ## API ##

apply(::GS.SparseLU, ::AbstractHamiltonian{T,<:Any,0}) where {T} =
    AppliedSparseLU{Complex{T}}()
apply(::GS.SparseLU, ::AbstractHamiltonian) =
    argerror("Can only use SparseLU with bounded Hamiltonians")

reset(::AppliedSparseLU{C}) where {C} = AppliedSparseLU{C}()

call!(s::AppliedSparseLU; params...) = s  # no need to reset

function call!(s::AppliedSparseLU, h, contacts, ω)
    if !isdefined(s.ig, 1)
        s.ig[] = InverseGreenBlockSparse(h, contacts)
    end
    ig = s.ig[]

    # must happen after call!(contacts, ω; params...) to ensure mergedlattice is initialized
    if !isdefined(s.blockstruct, 1)
        s.blockstruct[] = MultiBlockStructure(latslice(contacts), h)
    end
    blockstruct = s.blockstruct[]

    contactinds = ig.contactinds

    update!(ig, ω)
    igmat = sparse(ig)
    fact = try
        lu(igmat)
    catch
        argerror("Encountered a singular G⁻¹(ω) at ω = $ω, cannot factorize")
    end

    return ExecutedSparseLU(fact, blockstruct, contactinds)
end

function (s::ExecutedSparseLU{C})() where {C}
    bs = s.blockstruct
    fact = s.fact
    cinds = s.contactinds
    g⁻¹dim = size(fact, 2)
    source = zeros(C, g⁻¹dim, length(cinds))
    o = one(C)
    for (col, row) in enumerate(cinds)
        source[row, col] = o
    end
    gext = ldiv!(fact, source)
    g = HybridMatrix(gext[cinds, :], bs)
    return g
end

#endregion

#endregion

