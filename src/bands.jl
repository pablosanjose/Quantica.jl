############################################################################################
# Mesh
#region

# Marching Tetrahedra mesh
function mesh(rngs::Vararg{<:Any,L}) where {L}
    vmat   = [SVector(pt) for pt in Iterators.product(rngs...)]
    verts  = vec(vmat)
    cinds  = CartesianIndices(vmat)
    neighs = marching_neighbors_forward(cinds)  # sorted neighbors of i, with n[i][j] > i
    simps  = build_cliques(neighs, L+1)    # a Vector of Vectors of L+1 point indices (Ints)
    return Mesh(verts, neighs, simps)
end

# forward neighbors, cind is a CartesianRange over vertices
function marching_neighbors_forward(cinds)
    linds = LinearIndices(cinds)
    nmat = [Int[] for _ in cinds]
    for cind in cinds
        nlist = nmat[cind]
        forward = max(cind, first(cinds)):min(cind + oneunit(cind), last(cinds))
        for cind´ in forward
            cind === cind´ && continue
            push!(nlist, linds[cind´])
        end
        sort!(nlist)
    end
    neighs = vec(nmat)
    return neighs
end

#endregion

############################################################################################
# Cliques
#region

# groups of n all-to-all connected neighbors, sorted
function build_cliques(neighs, nverts)
    cliques = Vector{Int}[]
    for (src, dsts) in enumerate(neighs), ids in Combinations(length(dsts), nverts - 1)
        if all_adjacent(ids, dsts, neighs)
            clique = prepend!(dsts[ids], src)
            push!(cliques, clique)
        end
    end
    return cliques
end

# Check whether dsts[ids] are all mutual neighbors. ids are a total of nverts-1 indices of dsts = neighs[src].
function all_adjacent(ids, dsts, neighs)
    nids = length(ids)
    for (n, id) in enumerate(ids), n´ in n+1:nids
        dst = dsts[ids[n´]]
        dst in neighs[dsts[id]] || return false
    end
    return true
end

#endregion

############################################################################################
# bands
#region

bands(h::AbstractHamiltonian, mesh::Mesh; solver = ES.LinearAlgebra(), kw...) =
    bands(bloch(h, solver), mesh; solver, kw...)

function bands(bloch::Bloch, mesh::Mesh{SVector{L,T}};
    mapping = missing, solver = ES.LinearAlgebra(), showprogress = true) where {T,L}
    thread_solvers = [Eigensolver{T,L}(solver, bloch, mapping) for _ in 1:Threads.nthreads()]
    # Step 1/2 - Diagonalize:
    spectra = bands_diagonalize(thread_solvers, mesh, showprogress)
    return spectra
end

function bands_diagonalize(thread_solvers::Vector{Eigensolver{T,L,S}}, mesh::Mesh{SVector{L,T}},
    showprogress) where {T,L,S}
    meter = Progress(length(vertices(mesh)), "Step 1/2 - Diagonalizing: ")
    verts = vertices(mesh)
    spectra = Vector{S}(undef, length(verts))
    Threads.@threads for i in eachindex(verts)
        vert = verts[i]
        solver = thread_solvers[Threads.threadid()]
        spectra[i] = solver(vert)
        showprogress && ProgressMeter.next!(meter)
    end
    return spectra
end

#endregion

############################################################################################
# splitbands
#region

# splitbands(b) = b

#endregion