############################################################################################
# Mesh
#region

# Marching Tetrahedra mesh
function mesh(rngs::Vararg{<:AbstractRange,L}) where {L}
    vmat   = [SVector(pt) for pt in Iterators.product(rngs...)]
    verts  = vec(vmat)
    cinds  = CartesianIndices(vmat)
    neighs = marching_neighbors(cinds)     # a Vector of Vectors of point indices (Ints)
    simps  = find_simplices(neighs, L+1)   # a Vector of Vectors of L+1 point indices (Ints)
    return Mesh(verts, neighs, simps)
end

# cind is a CartesianRange over vertices
function marching_neighbors(cinds)
    linds = LinearIndices(cinds)
    nmat = [Int[] for _ in cinds]
    for cind in cinds
        nlist = nmat[cind]
        forward = max(cind, first(cinds)):min(cind + oneunit(cind), last(cinds))
        backward = max(cind - oneunit(cind), first(cinds)):min(cind, last(cinds))
        for cind´ in Iterators.flatten((forward, backward))
            cind === cind´ && continue
            push!(nlist, linds[cind´])
        end
    end
    neighs = vec(nmat)
    return neighs
end

# groups of n all-to-first connected neighbors, ordered
function find_simplices(neighs, nverts)
    counter = nverts
    simps = [[i] for i in eachindex(neighs)]
    push_simplices!(simps, neighs, nverts, counter - 1)
    # a single final filter! is faster than one per pass
    filter!(simp -> length(simp) == nverts, simps)
    return simps
end

function push_simplices!(simps, neighs, nverts, counter)
    counter > 0 || return simps
    sinds = eachindex(simps)
    for n in sinds
        simp = simps[n]
        lastvert = last(simp)
        isfirst = true
        for neigh in neighs[lastvert]
            neigh > lastvert && first(simp) in neighs[neigh] || continue
            if !isfirst
                simp = copy(simps[n])
                simp[end] = neigh
                push!(simps, simp)
            else
                push!(simp, neigh)
            end
            isfirst = false
        end
    end
    return push_simplices!(simps, neighs, nverts, counter - 1)
end

#endregion