############################################################################################
# spectrum
#region

function SpectrumSolver(h::AbstractHamiltonian{T,<:Any,L}, S = SVector{L,T};
                        solver::AbstractEigenSolver = ES.LinearAlgebra(), mapping = missing,
                        transform = missing) where {L,T}
    solver´ = apply(solver, h, S, mapping, transform)
    return SpectrumSolver(solver´)
end

spectrum(h::AbstractHamiltonian, φs, solver = ES.LinearAlgebra(); kw...) =
    SpectrumSolver(call!(h; kw...); solver)(φs)
spectrum(h::AbstractHamiltonian{<:Any,<:Any,0}, solver = ES.LinearAlgebra(); kw...) =
    SpectrumSolver(call!(h; kw...); solver)()

#endregion

############################################################################################
# Spectrum indexing
#region

Base.first(s::Spectrum) = energies(s)
Base.last(s::Spectrum) = states(s)
Base.iterate(s::Spectrum) = first(s), Val(:states)
Base.iterate(s::Spectrum, ::Val{:states}) = last(s), Val(:done)
Base.iterate(::Spectrum, ::Val{:done}) = nothing
Base.Tuple(s::Spectrum) = (first(s), last(s))

Base.getindex(s::Spectrum, i...; around = missing) = get_around(Tuple(s), around, i...)

get_around((es, ss), ε0::Missing, i) = (es[i], ss[:,i])
get_around(s, ε0::Number) = get_around(s, ε0, 1)
get_around(s, ε0::Number, i::Integer) = get_around(s, ε0, i:i)

function get_around((es, ss), ε0::Number, which)
    # Get indices of eachindex(es) such that if sorted by `by` will occupy `which` positions
    rngs = partialsort(eachindex(es), which, by = rng -> abs(es[rng] - ε0))
    return (es[rngs], ss[:, rngs])
end

#endregion

############################################################################################
# bands
#region

bands(h::AbstractHamiltonian, rngs::AbstractRange...; kw...) = bands(h, mesh(rngs...); kw...)

function bands(h::AbstractHamiltonian, basemesh::Mesh{SVector{L,T}}; solver = ES.LinearAlgebra(),
              transform = missing, mapping = missing, showprogress = true, defects = (), patches = 0,
              degtol = missing, split = true, warn = true) where {T,L}
    solvers = [SpectrumSolver(h, SVector{L,T}; solver, mapping, transform) for _ in 1:Threads.nthreads()]
    defects´ = sanitize_Vector_of_SVectors(SVector{L,T}, defects)
    degtol´ = degtol isa Number ? degtol : sqrt(eps(real(T)))
    return bands_precompilable(solvers, basemesh, showprogress, defects´, patches, degtol´, split, warn)
end

function bands_precompilable(solvers::Vector{A}, basemesh::Mesh{SVector{L,T}},
    showprogress, defects, patches, degtol, split, warn) where {T,L,B,A<:SpectrumSolver{T,L,B}}

    basemesh = copy(basemesh) # will become part of Band, possibly refined
    spectra = Vector{Spectrum{T,B}}(undef, length(vertices(basemesh)))
    bandverts = BandVertex{T,L+1}[]
    bandneighs = Vector{Int}[]
    bandneideg = similar(bandneighs)
    coloffsets = Int[]
    crossed = NTuple{6,Int}[]
    frustrated = similar(crossed)
    subbands = Subband{T,L+1}[]
    data = (; basemesh, spectra, bandverts, bandneighs, bandneideg, coloffsets, solvers, L,
              crossed, frustrated, subbands, defects, patches, showprogress, degtol, split, warn)

    # Step 1 - Diagonalize:
    # Uses multiple SpectrumSolvers (one per Julia thread) to diagonalize h at each
    # vertex of basemesh. Then, it collects each of the produced Spectrum (aka "columns")
    # into a bandverts::Vector{BandVertex}, recording the coloffsets for each column
    bands_diagonalize!(data)

    # Step 2 - Knit seams:
    # Each base vertex holds a column of subspaces. Each subspace s of degeneracy d will
    # connect to other subspaces s´ in columns of a neighboring base vertex. Connections are
    # possible if the projector ⟨s'|s⟩ has any singular value greater than 1/2
    bands_knit!(data)

    # Step 3 - Patch seams:
    # Dirac points and other topological band defects will usually produce dislocations in
    # mesh connectivity that results in missing simplices.
    if L>1
        bands_patch!(data)
    end

    # Step 4 - Split subbands
    bands_split!(data)

    return Bands(subbands, solvers)
end

#endregion

############################################################################################
# bands_diagonalize!
#region

function bands_diagonalize!(data)
    baseverts = vertices(data.basemesh)
    meter = Progress(length(baseverts), "Step 1 - Diagonalizing: ")
    push!(data.coloffsets, 0) # first element
    Threads.@threads for i in eachindex(baseverts)
        vert = baseverts[i]
        solver = data.solvers[Threads.threadid()]
        data.spectra[i] = solver(vert)
        data.showprogress && ProgressMeter.next!(meter)
    end
    # Collect bands vertices and store column offsets
    for (basevert, spectrum) in zip(baseverts, data.spectra)
        append_bands_column!(data, basevert, spectrum)
    end
    ProgressMeter.finish!(meter)
end


# collect spectrum into a band column (vector of BandVertices for equal base vertex)
function append_bands_column!(data, basevert, spectrum)
    T = eltype(basevert)
    energies´ = [maybereal(ε, T) for ε in energies(spectrum)]
    states´ = states(spectrum)
    subs = collect(approxruns(energies´, data.degtol))
    for (i, rng) in enumerate(subs)
        state = orthonormalize!(view(states´, :, rng))
        energy = mean(i -> energies´[i], rng)
        push!(data.bandverts, BandVertex(basevert, energy, state))
    end
    push!(data.coloffsets, length(data.bandverts))
    foreach(_ -> push!(data.bandneideg, Int[]), length(data.bandneideg)+1:length(data.bandverts))
    foreach(_ -> push!(data.bandneighs, Int[]), length(data.bandneighs)+1:length(data.bandverts))
    return data
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

#endregion

############################################################################################
# bands_knit!
#region

function bands_knit!(data)
    meter = Progress(length(data.spectra), "Step 2 - Knitting: ")
    for isrcbase in eachindex(data.spectra)
        for idstbase in neighbors_forward(data.basemesh, isrcbase)
            knit_seam!(data, isrcbase, idstbase)
        end
        data.showprogress && ProgressMeter.next!(meter)
    end
    ProgressMeter.finish!(meter)
    return data
end

# Take two intervals (srcrange, dstrange) of bandverts (linked by base mesh)
# and fill bandneighs with their connections, using the projector colproj
function knit_seam!(data, ib, jb)
    srcrange = column_range(data, ib)
    dstrange = column_range(data, jb)
    colproj  = states(data.spectra[jb])' * states(data.spectra[ib])
    for i in srcrange
        src = data.bandverts[i]
        for j in dstrange
            dst = data.bandverts[j]
            proj = view(colproj, parentcols(dst), parentcols(src))
            connections = connection_rank(proj)
            if connections > 0
                push!(data.bandneighs[i], j)
                push!(data.bandneighs[j], i)
                push!(data.bandneideg[i], connections)
                push!(data.bandneideg[j], connections)
                # populate crossed with all crossed links if lattice dimension > 1
                if data.L > 1
                    for i´ in first(srcrange):i-1, j´ in data.bandneighs[i´]
                        j´ in dstrange || continue
                        # if crossed, push! with ordered i´ < i, j´ > j
                        j´ > j && push!(data.crossed, (ib, jb, i´, i, j´, j))
                    end
                end
            end
        end
    end
    return data
end

# Number of singular values greater than √min_squared_overlap.
# Fast rank-1 |svd|^2 is r = tr(proj'proj). For higher ranks and r > 0 we must compute and
# count singular values. The min_squared_overlap is arbitrary, and is fixed heuristically to
# a high enough value to connect in the coarsest base lattices
function connection_rank(proj)
    min_squared_overlap = 0.5
    rankf = sum(abs2, proj)
    fastrank = ifelse(rankf > min_squared_overlap, 1, 0)  # For rank=1 proj: upon doubt, connect
    if iszero(fastrank) || size(proj, 1) == 1 || size(proj, 2) == 1
        return fastrank
    else
        sv = svdvals(proj)
        return count(s -> abs2(s) > min_squared_overlap, sv)
    end
end

column_range(data, ibase) = data.coloffsets[ibase]+1:data.coloffsets[ibase+1]

#endregion

############################################################################################
# bands_patch!
#region

function bands_patch!(data)
    insert_defects!(data)
    data.patches > 0 || return data
    queue_frustrated!(data)
    data.warn && isempty(data.defects) &&
        @warn "Trying to patch $(length(data.frustrated)) band dislocations without a list `defects` of defect positions."
    meter = data.patches < Inf ? Progress(data.patches, "Step 3 - Patching: ") :
                                 ProgressUnknown("Step 3 - Patching: ")
    newcols = 0
    done = false
    while !isempty(data.frustrated) && !done
        (ib, jb, i, i´, j, j´) = pop!(data.frustrated)
        # check edge has not been previously split
        jb in neighbors(data.basemesh, ib) || continue
        newcols += 1
        done = newcols == data.patches
        # new vertex
        k = crossing(data, (ib, jb, i´, i, j´, j))
        # insert and connect a new column into band
        insert_column!(data, (ib, jb), k)
        # From the added crossings, remove all that are non-frustrated
        queue_frustrated!(data)
        data.showprogress && ProgressMeter.next!(meter)
    end
    # If we added new columns, base edges will have split -> rebuild base simplices
    newcols > 0 && rebuild_cliques!(data.basemesh)
    data.showprogress && ProgressMeter.finish!(meter)
    ndefects = length(data.frustrated)
    data.warn && !iszero(ndefects) &&
        @warn("Warning: Band with $ndefects dislocation defects. Consider specifying topological defect locations with `defects` (or adjusting mesh) and/or increasing `patches`")
    return data
end

function insert_defects!(data)
    # insert user-provided defects as new columns in band
    foreach(k -> insert_defect_column!(data, k), data.defects)
    # detect possible defects in band and append them to data.defects
    mindeg = minimum(degeneracy, data.bandverts)
    for (v, ns) in zip(data.bandverts, data.bandneighs)
        k = base_coordinates(v)
        d = degeneracy(v)
        # exclude v if v does not increase degeneracy over minimum
        degeneracy(v) == mindeg && continue
        # only select vertices that have greater degeneracy than all its neighbors
        any(n -> degeneracy(data.bandverts[n]) >= d, ns) && continue
        # exclude v if it is already in data.defects
        any(kd -> kd ≈ k, data.defects) && continue
        push!(data.defects, base_coordinates(v))
    end
    return data
end

function insert_defect_column!(data, kdefect)
    base = data.basemesh
    for k in vertices(base)
        k ≈ kdefect && return data
    end
    # find closest edge (center) to kdefect
    (ib, jb) = argmin(((i, j) for i in eachindex(vertices(base)) for j in neighbors(base, i))) do (i, j)
        sum(abs2, 0.5*(vertices(base, i) + vertices(base, j)) - kdefect)
    end
    insert_column!(data, (ib, jb), kdefect)
    return data
end

function insert_column!(data, (ib, jb), k)
    solver = first(data.solvers)
    # remove all bandneighs in this seam
    delete_seam!(data, ib, jb)
    # create new vertex in basemesh by splitting the edge, and connect to neighbors
    split_edge!(data.basemesh, (ib, jb), k)
    # compute spectrum at new vertex
    spectrum = solver(k)
    push!(data.spectra, spectrum)
    # collect spectrum into a set of new band vertices
    append_bands_column!(data, k, spectrum)
    # knit all new seams
    newbasevertex = length(vertices(data.basemesh)) # index of new base vertex
    newbaseneighs = neighbors(data.basemesh, newbasevertex)
    jb´ = newbasevertex             # destination of all new edges
    for ib´ in newbaseneighs        # neighbors of new vertex are new edge sources
        knit_seam!(data, ib´, jb´)  # also pushes new crossings into data.crossed
    end
    return data
end

# queue frustrated crossings and sort decreasing distance between crossing and defects
function queue_frustrated!(data)
    # we must check all crossed each time, as their frustration can change at any point
    for c in data.crossed
        is_frustrated_crossing(data, c) && !in(c, data.frustrated) && push!(data.frustrated, c)
    end
    isempty(data.defects) || reverse!(sort!(data.frustrated, by = i -> distance_to_defects(data, i)))
    return data
end

function distance_to_defects(data, ic)
    kc = crossing(data, ic)
    return minimum(d -> sum(abs2, kc - d), data.defects)
end

is_frustrated_crossing(data, (ib, jb, i, i´, j, j´)) =
    is_frustrated_link(data, (ib, jb, i, j)) || is_frustrated_link(data, (ib, jb, i´, j´))

function is_frustrated_link(data, (ib, jb, i, j))
    deg = degeneracy_link(data, i, j)
    for kb in neighbors(data.basemesh, ib), kb´ in neighbors(data.basemesh, jb)
        kb == kb´ || continue
        count = 0
        for k in data.bandneighs[i], k´ in data.bandneighs[j]
            k == k´ && k in column_range(data, kb) || continue
            count += min(degeneracy_link(data, i, k), degeneracy_link(data, j, k))
        end
        count < deg && return true
    end
    return false
end

function degeneracy_link(data, i, j)
    for (k, d) in zip(data.bandneighs[i], data.bandneideg[i])
        k == j && return d
    end
    return 0
end

function crossing(data, (ib, jb, i, i´, j, j´))
    εi, εi´, εj, εj´ = energy.(getindex.(Ref(data.bandverts), (i, i´, j, j´)))
    ki, kj = vertices(data.basemesh, ib), vertices(data.basemesh, jb)
    λ = (εi - εi´) / (εj´ - εi´ - εj + εi)
    k = ki + λ * (kj - ki)
    return k
end

function delete_seam!(data, isrcbase, idstbase)
    srcrange = column_range(data, isrcbase)
    dstrange = column_range(data, idstbase)
    for isrc in srcrange
        fast_setdiff!((data.bandneighs[isrc], data.bandneideg[isrc]), dstrange)
    end
    for idst in dstrange
        fast_setdiff!((data.bandneighs[idst], data.bandneideg[idst]), srcrange)
    end
    return data
end

#endregion

############################################################################################
# bands_split!
#region

function bands_split!(data)
    if data.split
        # vsinds are the subband index of each vertex index
        # svinds is lists of band vertex indices that belong to the same subband
        vsinds, svinds = subsets(data.bandneighs)
        meter = Progress(length(svinds), "Step 4 - Splitting: ")
        new2old = sortperm(vsinds)
        old2new = invperm(new2old)
        offset = 0
        for subset in svinds
            # avoid subbands with no simplices
            if length(subset) > data.L
                sverts  = data.bandverts[subset]
                sneighs = [ [old2new[dstold] - offset for dstold in data.bandneighs[srcold]]
                        for srcold in subset]
                sband = Subband(sverts, sneighs)
                isempty(sband) || push!(data.subbands, sband)
            end
            offset += length(subset)
            data.showprogress && ProgressMeter.next!(meter)
        end
    else
        sband = Subband(data.bandverts, data.bandneighs)
        isempty(sband) || push!(data.subbands, sband)
    end
    return data
end

#endregion

############################################################################################
# Subband slicing and indexing
#   Example: in a 2D lattice, subband[kx,ky,:] is a vertical slice at fixed momentum kx, ky
#region

Base.getindex(b::Bands, xs...) = [s[xs...] for s in subbands(b)]
Base.getindex(s::Subband, xs...) = Subband(slice(s, xs, Val(true)))

slice(b::Bands, xs...) = [slice(s, xs...) for s in subbands(b)]

slice(s::Subband, xs::Union{Colon,Number}...) = slice(s, xs)
# default: slice -> mesh with same embedding dimension as subband and smaller simplex length
slice(s::Subband, xs::Tuple) = slice(s, xs::Tuple, Val(false))
# optional: slice -> mesh with reduced embedding dimension = simplex length + 1
slice(s::Subband, xs::Tuple, ::Val{true}) = slice(s, perp_axes(s, xs...), slice_axes(s, xs...))
slice(s::Subband, xs::Tuple, ::Val{false}) = slice(s, perp_axes(s, xs...), all_axes(s))

function slice(subband::Subband{<:Any,E}, paxes::NTuple{N}, saxes::Tuple) where {E,N}
    maximum(first, paxes) <= embdim(subband) && maximum(saxes) <= embdim(subband) ||
        throw(ArgumentError("Cannot slice subband along more than $(embdim(subband)) axes"))
    V = slice_vertex_type(subband, saxes)
    S = slice_skey_type(paxes)
    verts = V[]
    neighs = Vector{Int}[]
    vinds = Dict{S,Int}()
    vindstemp = Int[]
    subtemp = Int[]
    data = (; subband, paxes, saxes, verts, neighs, vinds, vindstemp, subtemp)

    foreach_simplex(subband, paxes) do sind
        simp = simplices(subband, sind)
        slice_simplex!(data, simp)
    end
    return Mesh{E-N}(verts, neighs)
end

perp_axes(::Subband{T}, xs...) where {T} = perp_axes(T, 1, xs...)
perp_axes(T::Type, dim, ::Colon, xs...) = perp_axes(T, dim + 1, xs...)
perp_axes(T::Type, dim, x::Number, xs...) = ((dim, T(x)), perp_axes(T, dim + 1, xs...)...)
perp_axes(T::Type, dim) = ()

slice_axes(::Subband{<:Any,E}, xs...) where {E} = slice_axes(1, padtuple(xs, :, Val(E))...)
slice_axes(dim::Int, ::Number, xs...) = slice_axes(dim + 1, xs...)
slice_axes(dim::Int, ::Colon, xs...) = (dim, slice_axes(dim + 1, xs...)...)
slice_axes(dim::Int) = ()

all_axes(::Subband{<:Any,E}) where {E} = ntuple(identity, Val(E))

slice_vertex_type(::Subband{T,<:Any}, ::NTuple{N}) where {T,N} = BandVertex{T,N}

slice_skey_type(::NTuple{N}) where {N} = SVector{N+1,Int}

function slice_simplex!(data, simp)
    empty!(data.vindstemp)
    perpaxes = SVector(first.(data.paxes))
    paraxes  = SVector(data.saxes)
    k = SVector(last.(data.paxes))
    sub = data.subtemp
    for sub´ in Combinations(length(simp), length(perpaxes) + 1)
        # subsimplex must have minimal degeneracy on first vertex
        copy!(sub, sub´)  # cannot modify sub´ because it also acts as state in Combinations
        sort!(sub, by = i -> degeneracy(vertices(data.subband, simp[i])))
        key = vindskey(data.paxes, simp, sub)
        if !haskey(data.vinds, key)
            kε0, edgemat = vertmat_simplex(data.paxes, vertices(data.subband), simp, sub)
            dvper = edgemat[perpaxes, :]
            λs = dvper \ (k - kε0[perpaxes])
            sum(λs) < 1 && all(>=(0), λs) || continue
            dvpar = edgemat[paraxes, :]
            kε = kε0[paraxes] + dvpar * λs
            φ = interpolate_state(λs, vertices(data.subband), simp, sub)
            push!(data.verts, BandVertex(kε, φ))
            push!(data.neighs, Int[])
            vind = length(data.verts)
            data.vinds[key] = vind
            push!(data.vindstemp, vind)
        else
            vind = data.vinds[key]
            push!(data.vindstemp, vind)
        end
    end
    # all-to-all among new vertices
    for i in data.vindstemp, j in data.vindstemp
        i == j || push!(data.neighs[i], j)
    end
    return data
end

# a sorted SVector of the N+1 parent vertex indices is = simp[sub] (N = num perp slice axes)
# identifying each distinct vertex in slice -> then, vinds[is] = new vertex index
vindskey(::NTuple{N}, simp, sub) where {N} = sort(SVector(ntuple(i -> simp[sub[i]], Val(N+1))))

function vertmat_simplex(::NTuple{N}, vs, simp, sub) where {N}
    kε0 = coordinates(vs[simp[sub[1]]])
    mat = hcat(ntuple(i -> coordinates(vs[simp[sub[i+1]]]) - kε0, Val(N))...)
    return kε0, mat
end

# we assume that sub is ordered and that simp is sorted so that the first vertex has minimal
# degeneracy within the simplex (see order_simplices!)
function interpolate_state(λs, vs, simp, sub)
    v0 = vs[simp[sub[1]]]
    φ0 = states(v0)
    deg0 = degeneracy(v0)
    φ = copy(φ0)
    φ .*= 1 - sum(λs)
    for (i, λi) in enumerate(λs)
        vi = vs[simp[sub[i+1]]]
        φi = states(vi)
        degi = degeneracy(vi)
        if  degi == deg0
            φ .+= λi .* φi
        elseif degi > deg0
            proj = φi'φ0  # size(proj) == (degi, deg0)
            Q = qr!(proj).Q * Matrix(I, degi, deg0)
            mul!(φ, φi, Q, λi, 1)
        else
            throw(ErrorException("Unexpected simplex ordering: first should be minimal degeneracy"))
        end
    end
    return φ
end

#endregion