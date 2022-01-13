############################################################################################
# mesh methods
#region

# Marching Tetrahedra mesh
function mesh(rngs::Vararg{<:Any,L}) where {L}
    vmat   = [SVector(pt) for pt in Iterators.product(rngs...)]
    verts  = vec(vmat)
    cinds  = CartesianIndices(vmat)
    neighs = marching_neighbors(cinds)  # sorted neighbors of i, with n[i][j] > i
    simps  = build_cliques(neighs, L+1)    # a Vector of Vectors of L+1 point indices (Ints)
    return Mesh(verts, neighs, simps)
end

# forward neighbors, cind is a CartesianRange over vertices
function marching_neighbors(cinds)
    linds = LinearIndices(cinds)
    matrix = [Int[] for _ in cinds]
    for cind in cinds
        forward  = max(cind, first(cinds)):min(cind + oneunit(cind), last(cinds))
        for cind´ in forward
            cind === cind´ && continue
            push!(matrix[cind], linds[cind´])
            push!(matrix[cind´], linds[cind])
        end
    end
    neighs = vec(matrix)
    return neighs
end

# simplices are not recomputed for performance
function split_edge!(m, (i, j), k)
    i == j && return m
    if i > j
        i, j = j, i
    end
    delete_edge!(m, (i, j))
    verts = vertices(m)
    push!(verts, k)
    push!(neighbors(m), Int[])
    dst = length(verts)
    newneighs = intersect(neighbors(m, i), neighbors(m, j))
    push!(newneighs, i, j)
    sort!(newneighs)
    for src in newneighs
        push!(neighbors(m, src), dst)
        push!(neighbors(m, dst), src)
    end
    return m
end

function delete_edge!(m, (i, j))
    i == j && return m
    if i > j
        i, j = j, i
    end
    fast_setdiff!(neighbors(m, i), j)
    fast_setdiff!(neighbors(m, j), i)
    return m
end

# delete elements in c and d if the one in c is not in rng
function fast_setdiff!((c, d)::Tuple, rng)
    i = 0
    for (i´, x) in enumerate(c)
        x in rng && continue
        i += 1
        c[i] = x
        d[i] = d[i´]
    end
    resize!(c, i)
    resize!(d, i)
    return (c, d)
end

# delete elements in c and d if the one in c is not in rng
function fast_setdiff!(c::Vector, rng)
    i = 0
    for x in c
        x in rng && continue
        i += 1
        c[i] = x
    end
    resize!(c, i)
    return c
end

# groups of n all-to-all connected neighbors, sorted
build_cliques(neighs, nverts) = build_cliques!(Vector{Int}[], neighs, nverts)

function build_cliques!(cliques, neighs, nverts)
    empty!(cliques)
    for (src, dsts) in enumerate(neighs)
        dsts_f = filter(>(src), dsts)  # indexable forward neighbors
        for ids in Combinations(length(dsts_f), nverts - 1)
            if all_adjacent(ids, dsts_f, neighs)
                clique = prepend!(dsts_f[ids], src)
                push!(cliques, clique)
            end
        end
    end
    return cliques
end

# Check whether dsts_f[ids] are all mutual neighbors. ids are a total of nverts-1 indices
# of dsts_f = neighbor_forward(neighs, src)
function all_adjacent(ids, dsts_f, neighs)
    nids = length(ids)
    for (n, id) in enumerate(ids), n´ in n+1:nids
        dst = dsts_f[ids[n´]]
        dst in neighs[dsts_f[id]] || return false
    end
    return true
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

# Computes connected subsets in a mesh, in the form of a (vsinds, svinds)
# vsinds::Vector{Int} is the subset index for each band vertex
# svinds::Vector{Vector{Int}} is a list of vertex indices for each subset
subsets(m::Mesh) = subsets(neighbors(m))

function subsets(neighs::Vector{Vector{Int}})
    vsinds = zeros(Int, length(neighs))
    svinds = Vector{Int}[]
    sidx = 0
    vidx = 1
    while vidx !== nothing
        sidx += 1
        sv = [vidx]
        push!(svinds, sv)
        vsinds[vidx] = sidx
        for i in sv, j in neighs[i]
            iszero(vsinds[j]) || continue
            vsinds[j] = sidx
            push!(sv, j)
        end
        sort!(sv)
        vidx = findfirst(iszero, vsinds)
    end
    return vsinds, svinds
end

#endregion