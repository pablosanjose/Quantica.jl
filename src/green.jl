############################################################################################
# HybridMatrix and MultiBlockStructure constructors
#region

#region ## Constructors ##

HybridMatrix(ls::LatticeSlice, h::AbstractHamiltonian) = HybridMatrix(ls, blockstructure(h))

function HybridMatrix(ls::LatticeSlice{T}, b::SublatBlockStructure) where {T}
    mb = MultiBlockStructure(ls, b)
    s = flatsize(mb)
    mat = zeros(Complex{T}, s, s)
    return HybridMatrix(mat, mb)
end

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

# find flatindices corresponding to mergedls of sites in ls´
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

green(s::AbstractGreenSolver = default_green_solver(h)) = h -> green(h, s)

green(h::AbstractHamiltonian, s::AbstractGreenSolver = default_green_solver(h)) =
    GreenFunction(h, apply(s, h))

#endregion

############################################################################################
# attach
#region

# new contacts should create new GreenFunction with empty preallocs
attach(g::GreenFunction, Σ::SelfEnergy) =
    GreenFunction(hamiltonian(h), solver(g), attach(contacts(g), Σ), similar(preallocs(g), 0))

attach(g::GreenFunction, args...; kw...) = attach(g, SelfEnergy(hamiltonian(g), args...; kw...))

attach(Σ::SelfEnergy) = g -> attach(g, Σ)

attach(args...; kw...) = g -> attach(g, SelfEnergy(hamiltonian(g), args...; kw...))

# function attach(g::GreenFunction, Σ::SelfEnergy)

# end

#endregion

############################################################################################
# call API
#region

(g::GreenFunction)(; params...) = copy(call!(g; params...))
(g::GreenFunction)(ω; params...) = copy(call!(g, ω; params...))

function call!(g::GreenFunction; params...)
    h´ = call!(hamiltonian(g); params...)
    solver´ = call!(solver(g); params...)
    contacts´ = call!(contacts(g); params...)
    preallocs´ = preallocs(g)
    return GreenFunction(h´, solver´, contacts´, preallocs´)
end

# function call!(g::GreenFunction, ω; params...)
# end

#endregion
