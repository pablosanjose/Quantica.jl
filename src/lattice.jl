############################################################################################
# Sublat constructors
#region

sublat(sites...; name = :_) =
    Sublat(sanitize_Vector_of_float_SVectors(sites), Symbol(name))

sublat(sites::Vector; name = :_) =
    Sublat(sanitize_Vector_of_float_SVectors(sites), Symbol(name))

#endregion

############################################################################################
# Bravais constructors
#region

Bravais(::Type{T}, E, m) where {T} = Bravais(T, Val(E), m)
Bravais(::Type{T}, ::Val{E}, m::NTuple{E´,Number}) where {T,E,E´} =
    Bravais{T,E,1}(sanitize_Matrix(T, E, (m,)))
Bravais(::Type{T}, ::Val{E}, m::NTuple{L,Any}) where {T,E,L} =
    Bravais{T,E,L}(sanitize_Matrix(T, E, m))
Bravais(::Type{T}, ::Val{E}, m::SMatrix{E,L}) where {T,E,L} =
    Bravais{T,E,L}(sanitize_Matrix(T, E, m))
Bravais(::Type{T}, ::Val{E}, m::AbstractMatrix) where {T,E} =
    Bravais{T,E,size(m,2)}(sanitize_Matrix(T, E, m))

#endregion

############################################################################################
# Unitcell constructors
#region

function unitcell(sublats, names, ::Type{S}) where {S<:SVector}
    sites´ = S[]
    offsets´ = [0]  # length(offsets) == length(sublats) + 1
    for s in eachindex(sublats)
        for site in sites(sublats[s])
            push!(sites´, sanitize_SVector(S, site))
        end
        push!(offsets´, length(sites´))
    end
    names´ = uniquenames!(sanitize_Vector_of_Symbols(names))
    return Unitcell(sites´, names´, offsets´)
end

function unitcell(u::Unitcell, names, ::Type{S}) where {S<:SVector}
    sites´ = sanitize_SVector.(S, sites(u))
    names´ = uniquenames!(sanitize_Vector_of_Symbols(names))
    offsets´ = offsets(u)
    Unitcell(sites´, names´, offsets´)
end

function uniquenames!(names::Vector{Symbol})
    allnames = Symbol[:_]
    for (i, name) in enumerate(names)
        name in allnames && (names[i] = uniquename(allnames, name, i))
        push!(allnames, name)
    end
    return names
end

function uniquename(allnames, name, i)
    newname = Symbol(Char(64+i)) # Lexicographic, starting from Char(65) = 'A'
    return newname in allnames ? uniquename(allnames, name, i + 1) : newname
end

#endregion

############################################################################################
# Lattice constructors
#region

lattice(s::Sublat, ss::Sublat...; kw...) = _lattice(promote(s, ss...)...; kw...)

# Start with an empty list of nranges, to be filled as they are requested
Lattice(b::Bravais{T}, u::Unitcell{T}) where {T} = Lattice(b, u, Tuple{Int,T}[])

function _lattice(ss::Sublat{T,E}...;
                  bravais = (),
                  dim = Val(E),
                  type::Type{T´} = T,
                  names = sublatname.(ss)) where {T,E,T´}
    u = unitcell(ss, names, postype(dim, type))
    b = Bravais(type, dim, bravais)
    return Lattice(b, u)
end

function lattice(lat::Lattice{T,E};
                 bravais = bravais_matrix(lat),
                 dim = Val(E),
                 type::Type{T´} = T,
                 names = sublatnames(lat)) where {T,E,T´}
    u = unitcell(unitcell(lat), names, postype(dim, type))
    b = Bravais(type, dim, bravais)
    return Lattice(b, u)
end

postype(dim, type) = SVector{dim,type}
postype(::Val{E}, type) where {E} = SVector{E,type}

#endregion


############################################################################################
# neighbors
#region

function nrange(n, lat)
    for (n´, r) in nranges(lat)
        n == n´ && return r
    end
    r = compute_nrange(n, lat)
    push!(nranges(lat), (n, r))
    return r
end


function compute_nrange(n, lat::Lattice{T}) where {T}
    latsites = sites(lat)
    dns = BoxIterator(zerocell(lat))
    br = bravais_matrix(lat)
    # 128 is a heuristic cutoff for kdtree vs brute-force search
    if length(latsites) <= 128
        dists = fill(T(Inf), n)
        for dn in dns
            iszero(dn) || ispositive(dn) || continue
            for (i, ri) in enumerate(latsites), (j, rj) in enumerate(latsites)
                j <= i && iszero(dn) && continue
                r = ri - rj + br * dn
                update_dists!(dists, r'r)
            end
            isfinite(last(dists)) || acceptcell!(dns, dn)
        end
        dist = sqrt(last(dists))
    else
        tree = KDTree(latsites)
        dist = T(Inf)
        for dn in dns
            iszero(dn) || ispositive(dn) || continue
            for r0 in latsites
                r = r0 + br * dn
                dist = min(dist, compute_nrange(n, tree, r, nsites(lat)))
            end
            isfinite(dist) || acceptcell!(dns, dn)
        end
    end
    return dist
end

function update_dists!(dists, dist)
    len = length(dists)
    for (n, d) in enumerate(dists)
        isapprox(dist, d) && break
        if dist < d
            dists[n+1:len] .= dists[n:len-1]
            dists[n] = dist
            break
        end
    end
    return dists
end

function compute_nrange(n, tree, r::AbstractVector, nmax)
    for m in n:nmax
        _, dists = knn(tree, r, 1 + m, true)
        popfirst!(dists)
        unique_sorted_approx!(dists)
        length(dists) == n && return maximum(dists)
    end
    return convert(eltype(r), Inf)
end

function unique_sorted_approx!(v::AbstractVector)
    i = 1
    xprev = first(v)
    for j in 2:length(v)
        if v[j] ≈ xprev
            xprev = v[j]
        else
            i += 1
            xprev = v[i] = v[j]
        end
    end
    resize!(v, i)
    return v
end

function ispositive(ndist)
    result = false
    for i in ndist
        i == 0 || (result = i > 0; break)
    end
    return result
end

#endregion