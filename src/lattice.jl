############################################################################################
# Sublat constructors
#region

sublat(sites...; name = :_) =
    Sublat(sanitize_Vector_of_SVectors(sites), Symbol(name))

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

function _lattice(ss::Sublat...;
    bravais = (), dim::Val{E} = valdim(first(ss)), type::Type{T} = numbertype(first(ss)), names = name.(ss)) where {E,T}
    u = unitcell(ss, names, SVector{E,T})
    b = Bravais(sanitize_SVector_of_SVectors(SVector{E,T}, bravais))
    return Lattice(b, u)
end

function lattice(l::Lattice;
    bravais = bravais(lat), dim::Val{E} = valdim(l), type::Type{T} = numbertype(l), names = names(l)) where {E,T}
    u = unitcell(unitcell(lat), names, SVector{E,T})
    b = Bravais(sanitize_SVector_of_SVectors(SVector{E,T}, bravais))
    return Lattice(b, u)
end

#endregion

############################################################################################
# supercell
#region



#endregion