############################################################################################
# Sublat
#region

sublat(sites...; name = :_) =
    Sublat(sanitize_Vector_of_SVectors(sites), Symbol(name))

#endregion

############################################################################################
# Unitcell
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
# Lattice
#region

function lattice(ss::Sublat{T´,E´}...;
    bravais = (), dim::Val{E} = Val(E´), type::Type{T} = T´, names = name.(ss)) where {E,T,E´,T´}
    u = unitcell(ss, names, SVector{E,T})
    b = Bravais(sanitize_Tuple_of_SVectors(SVector{E,T}, bravais))
    return Lattice(b, u)
end

function lattice(l::Lattice{T´,E´};
    bravais = bravais(lat), dim::Val{E} = Val(E´), type::Type{T} = T´, names = names(l)) where {E,T,E´,T´}
    u = unitcell(unitcell(lat), names, SVector{E,T})
    b = Bravais(sanitize_Tuple_of_SVectors(SVector{E,T}, bravais))
    return Lattice(b, u)
end

#endregion