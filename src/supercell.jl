
############################################################################################
# supercell
#region

struct LatticeMask
    lat
    smat
    mask
end

supercell(l::Lattice{<:Any,<:Any,L}, v::Vararg{<:Any,L´}; kw...) where {L} =
    supercell(x, sanitize_SMatrix(SMatrix{L,L´,Int}, v); kw...)

function supercell(lat::Lattice, smat::SMatrix{L,L´,Int};
                   cellseed = zero(SVector{L,Int}), kwselector...) where {L,L´}
    smatfull = prependfull(smat, true)  # added columns come first
    masklist = supercell_masklist(smatfull, cellseed, lat)
    selector = siteselector(lat; kwselector...)
    smatperp = convert(SMatrix{L, L-L´,Int}, view(smatfull, :, L´+1:L))
    seedperp = zero(SVector{L-L´,Int})
    masklist = supercell_selector_masklist(smatperp, seedperp, selector, masklist)
end

# build masklist = [(siteindex, cell)] for all sites in full supercell defined by smat
function supercell_masklist(smat::SMatrix{L,L,Int}, cellseed::SVector{L,Int}, lat) where {L}
    supercell_nsites = nsites(lat) * round(Int, abs(det(smat)))
    iszero(supercell_nsites) && throw(ArgumentError("Supercell is empty. Singular supercell matrix?"))
    masklist = Vector{Tuple{Int,SVector{L,Int}}}(undef, supercell_nsites)
    br = bravais_matrix(lat)
    projector = pinverse(br * smat)
    counter = 0
    iter = BoxIterator(cellseed)
    for c in iter, (i, site) in enumerate(sites(lat))
        cell = SVector(Tuple(c))
        r = site + br * cell
        δn = projector * r
        if all(x -> 0 <= x < 1, δn)
            counter += 1
            masklist[counter] = (i, cell)
        end
        counter == supercell_nsites && break
    end
    counter == supercell_nsites || throw(ErrorException("Internal error: failed to find all sites in supercell"))
    return masklist
end

# build masklist´ = [(siteindex, cell´)] where cell´ varies along axes smatperp not in smat
function supercell_selector_masklist(smatperp, seedperp, selector, masklist)
    masklist´ = similar(masklist, 0)
    iter = BoxIterator(seedperp)
    for c in iter, (i, cell) in masklist
        cell´ = cell + smatperp * SVector(Tuple(c))
        if (i, cell´) in selector
            push!(masklist´, (i, cell´))
            acceptcell!(iter, c)
        end
    end
    return masklist´
end


#endregion