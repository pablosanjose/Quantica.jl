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
# append_subspaces!
#region

function append_subspaces!(bandverts::Vector{V}, basevert, spectrum) where {T,V<:BandVertex{T}}
    energies´ = T[T(real(ε)) for ε in energies(spectrum)]
    states´ = states(spectrum)
    subs = collect(approxruns(energies´))
    for rng in subs
        state = orthonormalize!(view(states´, :, rng))
        energy = mean(i -> energies´[i], rng)
        push!(bandverts, BandVertex(basevert, energy, state))
    end
    return bandverts
end

# Gram-Schmidt but with column normalization only when norm^2 >= threshold (otherwise zero!)
function orthonormalize!(m::AbstractMatrix, threshold = 0)
    @inbounds for j in axes(m, 2)
        col = view(m, :, j)
        for j´ in 1:j-1
            col´ = view(m, :, j´)
            norm2´ = dot(col´, col´)
            iszero(norm2´) && continue
            r = dot(col´, col)/norm2´
            col .-= r .* col´
        end
        norm2 = real(dot(col, col))
        factor = ifelse(norm2 < threshold, zero(norm2), 1/sqrt(norm2))
        col .*= factor
    end
    return m
end

#endregion

############################################################################################
# bands
#region

bands(h::AbstractHamiltonian, mesh::Mesh; solver = ES.LinearAlgebra(), kw...) =
    bands(bloch(h, solver), mesh; solver, kw...)

function bands(bloch::Bloch, basemesh::Mesh{SVector{L,T}};
    mapping = missing, solver = ES.LinearAlgebra(), showprogress = true) where {T,L}
    thread_solvers = [Eigensolver{T,L}(solver, bloch, mapping) for _ in 1:Threads.nthreads()]
    # Step 1/2 - Diagonalize:
    spectra = bands_diagonalize(thread_solvers, basemesh, showprogress)
    # Step 2/2 - Knit bands:
    bandmesh = bands_knit(spectra, basemesh, showprogress) 
    return spectra
end

function bands_diagonalize(thread_solvers::Vector{Eigensolver{T,L,S}}, basemesh::Mesh{SVector{L,T}},
    showprogress) where {T,L,S}
    meter = Progress(length(vertices(basemesh)), "Step 1/2 - Diagonalizing: ")
    verts = vertices(basemesh)
    spectra = Vector{S}(undef, length(verts))
    Threads.@threads for i in eachindex(verts)
        vert = verts[i]
        solver = thread_solvers[Threads.threadid()]
        spectra[i] = solver(vert)
        showprogress && ProgressMeter.next!(meter)
    end
    return spectra
end

function bands_knit(spectra::Vector{S}, basemesh::Mesh{SVector{L,T}}, showprogress) where {L,T,O,S<:Spectrum{<:Any,O}}
    bandverts = BandVertex{T,L,O}[]
    baseverts = vertices(basemesh)
    for (spectrum, basevert) in zip(spectra, baseverts)
        append_subspaces!(bandverts, basevert, spectrum)
    end

    
    simps = simplices(basemesh)
    
    simpitr = marchingsimplices(basemesh) # D+1 - dimensional iterator over simplices

    S       = basis_slice_type(first(spectra))
    verts   = SVector{D+1,T}[]
    vbases  = S[]
    vptrs   = Array{UnitRange{Int}}(undef, size(vertices(basemesh)))
    simps   = NTuple{D+1,Int}[]
    sbases  = NTuple{D+1,Matrix{C}}[]
    sptrs   = Array{UnitRange{Int}}(undef, size(simpitr))

    cbase = eachindex(basemesh)
    lbase = LinearIndices(cbase)

    # Collect vertices
    for csrc in cbase
        len = length(verts)
        push_verts!((verts, vbases), spectra[csrc])
        vptrs[csrc] = len+1:length(verts)
    end

    # Store subspace projections in vertex adjacency matrix
    prog = Progress(length(cbase), "Step 2/2 - Knitting bands: ")
    S´ = basis_block_type(first(spectra))
    I, J, V = Int[], Int[], S´[]
    for csrc in cbase
        for cdst in neighbors_forward(basemesh, csrc)
            push_adjs!((I, J, V), spectra[csrc], spectra[cdst], vptrs[csrc], vptrs[cdst])
        end
        showprog && ProgressMeter.next!(prog)
    end
    n = length(verts)
    adjprojs = sparse(I, J, V, n, n)
    adjmat = sparsealiasbool(adjprojs)

    # Build simplices from stable cycles around base simplices
    buffers = NTuple{D+1,Int}[], NTuple{D+1,Int}[]
    emptybases = filltuple(fill(zero(C), 0, 0), Val(D+1))  # sentinel bases for deg == 1 simplices
    for (csimp, vs) in zip(CartesianIndices(simpitr), simpitr)  # vs isa NTuple{D+1,CartesianIndex{D}}
        len = length(simps)
        ranges = getindex.(Ref(vptrs), vs)
        push_simps!((simps, sbases, verts), buffers, ranges, adjprojs, emptybases)
        sptrs[csimp] = len+1:length(simps)
    end

    bands = [Band(basemesh, verts, vbases, vptrs, adjmat, simps, sbases, sptrs)]
    return bands
end

#endregion

############################################################################################
# splitbands
#region

# splitbands(b) = b

#endregion