############################################################################################
# HybridMatrix and MultiBlockStructure constructors
#region

#region ## Constructors ##

HybridMatrix(ls::LatticeSlice, h::Union{AbstractHamiltonian,SublatBlockStructure}) =
    HybridMatrix(0I, ls, h)

function HybridMatrix(mat, ls::LatticeSlice, h::AbstractHamiltonian)
    mb = MultiBlockStructure(ls, h)
    s = flatsize(mb)
    checkblocksize(mat, (s, s))
    return HybridMatrix(mat, mb)
end

function HybridMatrix(mat::UniformScaling{T}, mb::MultiBlockStructure) where {T}
    s = flatsize(mb)
    mat´ = Matrix{T}(mat, s, s)
    return HybridMatrix(mat´, mb)
end

MultiBlockStructure(ls::LatticeSlice, h::AbstractHamiltonian, lss = ()) =
    MultiBlockStructure(ls, blockstructure(h), lss)

function MultiBlockStructure(ls::LatticeSlice{<:Any,<:Any,L},
                             bs::SublatBlockStructure, lss = ()) where {L}
    cells = SVector{L,Int}[]
    subcelloffsets = [0]
    siteoffsets = [0]
    offset = 0
    for subcell in subcells(ls)
        c = cell(subcell)
        push!(cells, c)
        for i in siteindices(subcell)
            offset += blocksize(bs, i)
            push!(siteoffsets, offset)
        end
        push!(subcelloffsets, offset)
    end
    contactinds = Vector{Int}[contact_indices(ls´, siteoffsets, ls) for ls´ in lss]
    return MultiBlockStructure(cells, subcelloffsets, siteoffsets, contactinds)
end

# find flatindices corresponding to merged_ls of sites in ls´
function contact_indices(ls´, siteoffsets, merged_ls)
    contactinds = Int[]
    for scell´ in subcells(ls´)
        so = findsubcell(cell(scell´), merged_ls)
        so === nothing && continue
        # here offset is the number of sites in merged_ls before scell
        (scell, offset) = so
        for i´ in siteindices(scell´), (n, i) in enumerate(siteindices(scell))
            n´ = offset + n
            i == i´ && append!(contactinds, siteoffsets[n´]+1:siteoffsets[n´+1])
        end
    end
    return contactinds
end

#endregion
#endregion

############################################################################################
# green
#region

green(s::AbstractGreenSolver) = h -> green(h, s)

green() = h -> green(h)

green(h::AbstractHamiltonian, s::AbstractGreenSolver = default_green_solver(h)) =
    GreenFunction(h, apply(s, h))

# default_green_solver(::AbstractHamiltonian0D) = GS.SparseLU()
# default_green_solver(::AbstractHamiltonian1D) = GS.Schur()
# default_green_solver(::AbstractHamiltonian) = GS.Bands()
default_green_solver(::AbstractHamiltonian) = GS.NoSolver()

#endregion

############################################################################################
# attach
#region

attach(g::GreenFunction, args...; kw...) = attach(g, SelfEnergy(hamiltonian(g), args...; kw...))
attach(Σ::SelfEnergy) = g -> attach(g, Σ)
attach(args...; kw...) = g -> attach(g, SelfEnergy(hamiltonian(g), args...; kw...))
attach(g::GreenFunction, Σ::SelfEnergy) =
    GreenFunction(hamiltonian(g), reset(solver(g)), attach(contacts(g), Σ))

#endregion

############################################################################################
# call API
#region

## TODO: test copy(g) for aliasing problems
(g::GreenFunction)(; params...) = copy(call!(g; params...))
(g::GreenFunction)(ω; params...) = copy(call!(g, ω; params...))

function call!(g::GreenFunction; params...)
    h´ = call!(hamiltonian(g); params...)
    solver´ = call!(solver(g); params...)
    contacts´ = call!(contacts(g); params...)
    return GreenFunction(h´, solver´, contacts´)
end

function call!(g::GreenFunction, ω; params...)
    h = hamiltonian(g)
    cs = contacts(g)
    call!(h, (); params...)
    Σs = call!(cs, ω; params...)
    so = call!(solver(g), h, cs, ω; params...)
    ls = latslice(cs)
    gc = so()
    Γs = linewidth.(Σs)
    return GreenMatrix(so, gc, Γs, ls)
end

function linewidth(Σ::MatrixBlock)
    Σmat = blockmat(Σ)
    Γ = Σmat - Σmat'
    Γ .*= im
    return Γ
end

#endregion
