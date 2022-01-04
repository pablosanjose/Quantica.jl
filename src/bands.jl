############################################################################################
# mesh
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
# build_cliques
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

function bands(bloch::Bloch, basemesh::Mesh{SVector{L,T}};
    mapping = missing, solver = ES.LinearAlgebra(), showprogress = true) where {T,L}
    thread_solvers = [Eigensolver{T,L}(solver, bloch, mapping) for _ in 1:Threads.nthreads()]
    # Step 1/2 - Diagonalize:
    spectra = bands_diagonalize(thread_solvers, basemesh, showprogress)
    # Step 2/2 - Knit bands:
    bandmesh = bands_knit(spectra, basemesh, showprogress) 
    return bandmesh
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
    # Build band vertices
    baseverts = vertices(basemesh)
    length(spectra) == length(baseverts) ||
        throw(error("Unexpected bands_knit error: spectra and base vertices not of equal length"))
    bandverts = BandVertex{T,L,O}[]
    coloffsets = [0] # offsets for intervals of bandverts corresponding to each basevertex
    for (basevert, spectrum) in zip(baseverts, spectra)
        append_band_column!(bandverts, basevert, spectrum)
        push!(coloffsets, length(bandverts))
    end

    # Build band neighbors
    meter = Progress(length(vertices(basemesh)), "Step 2/2 - Knitting: ")
    baseneighs = neighbors_forward(basemesh)
    bandneighs = [Int[] for _ in bandverts]
    for isrcbase in eachindex(baseverts)
        for idstbase in baseneighs[isrcbase]
            colsrc = coloffsets[isrcbase]+1:coloffsets[isrcbase+1]
            coldst = coloffsets[idstbase]+1:coloffsets[idstbase+1]
            colproj = states(spectra[idstbase])' * states(spectra[isrcbase])
            for isrc in colsrc
                src = bandverts[isrc]
                available_connections = degeneracy(src)
                for idst in coldst
                    dst = bandverts[idst]
                    proj = view(colproj, parentcols(dst), parentcols(src))
                    connections = connection_rank(proj)
                    iszero(connections) && continue
                    push!(bandneighs[isrc], idst)
                    available_connections -= connections
                    available_connections <= 0 && break
                end
            end
        end
        showprogress && ProgressMeter.next!(meter)
    end

    # Build band simplices
    bandsimps = build_cliques(bandneighs, L+1)
    orient_simplices!(bandsimps, bandverts)

    return Mesh(bandverts, bandneighs, bandsimps)
end

# equivalent to r = round(Int, tr(proj'proj)), but if r > 0, must compute and count singular values
function connection_rank(proj)
    r = round(Int, sum(abs2, proj))
    (iszero(r) || minimum(size(proj)) == 1) && return r
    s = svdvals(proj)
    r = count(s -> abs2(s) >= 0.5, s)
    return r
end

# collect spectrum into a band column (vector of BandVertices for equal base vertex)
function append_band_column!(bandverts, basevert, spectrum)
    T = eltype(basevert)
    energies´ = [maybereal(ε, T) for ε in energies(spectrum)]
    states´ = states(spectrum)
    subs = collect(approxruns(energies´))
    for (i, rng) in enumerate(subs)
        state = orthonormalize!(view(states´, :, rng))
        energy = mean(i -> energies´[i], rng)
        push!(bandverts, BandVertex(basevert, energy, state))
    end
    return bandverts
end

maybereal(energy, ::Type{T}) where {T<:Real} = T(real(energy))
maybereal(energy, ::Type{T}) where {T<:Complex} = T(energy)

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

function orient_simplices!(simplices, vertices::Vector{B}) where {L,B<:BandVertex{<:Any,L}}
    for simplex in simplices
        k0 = base_coordinates(vertices[simplex[1]])
        edges = ntuple(i -> base_coordinates(vertices[simplex[i+1]])-k0, Val(L))
        volume = det(hcat(edges...))
        if volume < 0 # switch last
            simplex[end], simplex[end-1] = simplex[end-1], simplex[end]
        end
    end
    return simplices
end

#endregion

############################################################################################
# splitbands
#region

# splitbands(b) = b

#endregion