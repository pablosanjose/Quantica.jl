############################################################################################
# mesh
#region

# Marching Tetrahedra mesh
function mesh(rngs::Vararg{<:Any,L}) where {L}
    vmat   = [SVector(pt) for pt in Iterators.product(rngs...)]
    verts  = vec(vmat)
    cinds  = CartesianIndices(vmat)
    neighs_forward = marching_neighbors_forward(cinds)  # sorted neighbors of i, with n[i][j] > i
    simps  = build_cliques(neighs_forward, L+1)    # a Vector of Vectors of L+1 point indices (Ints)
    return Mesh(verts, neighs_forward, simps)
end

# forward neighbors, cind is a CartesianRange over vertices
function marching_neighbors_forward(cinds)
    linds = LinearIndices(cinds)
    matrix_forward = [Int[] for _ in cinds]
    for cind in cinds
        nlist = matrix_forward[cind]
        forward  = max(cind, first(cinds)):min(cind + oneunit(cind), last(cinds))
        for cind´ in forward
            cind === cind´ && continue
            push!(nlist, linds[cind´])
        end
    end
    neighs_forward = vec(matrix_forward)
    return neighs_forward
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
    bandmesh = bands_knit(spectra, first(thread_solvers), basemesh, showprogress) 
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

# Each base vertex holds a column of subspaces. Each subspace s of degeneracy d will connect
# to up to d other subspaces s´ in columns of a neighboring base vertex. N connections are
# possible if the projector ⟨s'|s⟩ has N singular values greater than 1/2
function bands_knit(spectra::Vector{S}, solver, basemesh::Mesh{SVector{L,T}}, showprogress) where {L,T,O,S<:Spectrum{<:Any,O}}
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
    meter = Progress(length(spectra), "Step 2/2 - Knitting: ")
    baseneighs = neighbors_forward(basemesh)
    baseneighs_extra = similar(baseneighs, 0)
    len = length(baseneighs)
    bandneighs = [Int[] for _ in bandverts]
    # `enumerate` (unlike `eachindex`) allows loop to continue if `spectra` grows in loop
    for (isrcbase, srcspect) in enumerate(spectra)
        knit_refined = isrcbase > len
        srcneighs = knit_refined ? baseneighs_extra[isrcbase - len] : baseneighs[isrcbase]
        for idstbase in srcneighs
            srcrange  = coloffsets[isrcbase]+1:coloffsets[isrcbase+1]
            dstrange  = coloffsets[idstbase]+1:coloffsets[idstbase+1]
            dstspect  = spectra[idstbase]
            colproj   = states(dstspect)' * states(srcspect)
            validedge = knit_base_edge!(bandneighs, bandverts, srcrange, dstrange, colproj, knit_refined)
            if !validedge && isrcbase <= length(baseverts) # (only check non-refined edges)
                # refine base edge (non-recursive), adding a new column mid-edge
                newvertex = 0.5 * (baseverts[isrcbase] + baseverts[idstbase])
                newspectrum = solver(newvertex)
                push!(spectra, newspectrum)  # this extends the outermost loop
                append_band_column!(bandverts, newvertex, newspectrum)
                # don'f forget to add empty neighbors to added bandverts
                append!(bandneighs, [Int[] for _ in length(bandneighs)+1:length(bandverts)])
                push!(coloffsets, length(bandverts))
                newneighs = intersect(neighbors(basemesh, isrcbase), neighbors(basemesh, idstbase))
                push!(newneighs, isrcbase, idstbase)
                push!(baseneighs_extra, newneighs)
            end
        end
        showprogress && ProgressMeter.next!(meter)  # This might run over if we refine
    end

    # Build band simplices
    bandsimps = build_cliques(bandneighs, L+1)
    orient_simplices!(bandsimps, bandverts)

    return Mesh(bandverts, bandneighs, bandsimps)
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

# Take two intervals (srcrange, dstrange) of bandverts (linked by base mesh)
# and fill bandneighs with their connections, using the projector colproj
# If knit_refined, reverse src <-> dst in bandneighs (so they remain "forward" neighs)
function knit_base_edge!(bandneighs, bandverts, srcrange, dstrange, colproj, knit_refined)
    for isrc in srcrange
        src = bandverts[isrc]
        # available_connections = degeneracy(src)
        for idst in dstrange
            dst = bandverts[idst]
            proj = view(colproj, parentcols(dst), parentcols(src))
            connections = connection_rank(proj) # sentinel for invalid link: -1
            if connections > 0
                knit_refined ? push!(bandneighs[idst], isrc) : push!(bandneighs[isrc], idst)
            elseif connections < 0 && !knit_refined
                # for i in srcrange
                #     # undo all added neighbors for all sources in this base edge
                #     empty!(bandneighs[i])
                # end
                return false              # stop processing this edge, return valid = false
            end
        end
        # available_connections == 0 || @show "incomplete"
    end
    return true  # return valid = true
end

# equivalent to r = round(Int, tr(proj'proj)), but if r > 0, must compute and count singular values
# if the rank is borderline (any singular value is ≈ 1/√2), return -1 (sentinel value)
function connection_rank(proj)
    if size(proj, 1) == 1 || size(proj, 2) == 1
        rfloat = sum(abs2, proj)
        return rfloat ≈ 0.5 ? -1 : round(Int, rfloat)
    else
        sv = svdvals(proj)
        sv .= abs2.(sv)
        return any(≈(0.5), sv) ? -1 : count(>(0.5), sv)
    end
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