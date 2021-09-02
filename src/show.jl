
############################################################################################
# Lattice
#region

Base.summary(::Sublat{T,E}) where {T,E} =
    "Sublat{$E,$T} : sublattice of $T-typed sites in $(E)D space"

Base.show(io::IO, s::Sublat) = print(io, summary(s), "\n",
"  Sites    : $(nsites(s))
  Name     : $(displayname(s))")

###

Base.summary(::Lattice{T,E,L}) where {T,E,L} =
    "Lattice{$T,$E,$L} : $(L)D lattice in $(E)D space"

function Base.show(io::IO, lat::Lattice)
    i = get(io, :indent, "")
    print(io, i, summary(lat), "\n",
"$i  Bravais vectors : $(display_rounded_vectors(bravais_vectors(lat)))
$i  Sublattices     : $(nsublats(lat))
$i    Names         : $(displaynames(lat))
$i    Sites         : $(display_as_tuple(sublatlengths(lat))) --> $(nsites(lat)) total per unit cell")
end

displaynames(l::Lattice) = display_as_tuple(sublatnames(l), ":")

displayname(s::Sublat) = sublatname(s) == Symbol(:_) ? "pending" : string(":", sublatname(s))

display_as_tuple(v, prefix = "") = isempty(v) ? "()" :
    string("(", prefix, join(v, string(", ", prefix)), ")")

display_rounded_vectors(vs) = isempty(vs) ? "[]" : display_rounded_vector.(vs)
display_rounded_vector(v) = round.(v, digits = 6)

#endregion

############################################################################################
# Model
#region

function Base.show(io::IO, m::TightbindingModel)
    ioindent = IOContext(io, :indent => "  ")
    print(io, "TightbindingModel: model with $(length(terms(m))) terms", "\n")
    foreach(t -> print(ioindent, t, "\n"), m.terms)
end

function Base.show(io::IO, o::OnsiteTerm{F,<:SiteSelector}) where {F}
    i = get(io, :indent, "")
    print(io,
"$(i)OnsiteTerm{$(displayparameter(F))}:
$(i)  Sublattices      : $(o.selector.sublats === missing ? "any" : o.selector.sublats)
$(i)  Coefficient      : $(o.coefficient)")
end

function Base.show(io::IO, h::HoppingTerm{F,<:HopSelector}) where {F}
    i = get(io, :indent, "")
    print(io,
"$(i)HoppingTerm{$(displayparameter(F))}:
$(i)  Sublattice pairs : $(h.selector.sublats === missing ? "any" : h.selector.sublats)
$(i)  dn cell distance : $(h.selector.dcells === missing ? "any" : h.selector.dcells)
$(i)  Hopping range    : $(displayrange(h.selector.range))
$(i)  Coefficient      : $(h.coefficient)")
end

displayparameter(::Type{<:Function}) = "Function"
displayparameter(::Type{T}) where {T} = "$T"

displayrange(r::Real) = round(r, digits = 6)
displayrange(::Missing) = "any"
displayrange(nr::Neighbors) = "Neighbors($(Int(nr)))"
displayrange(rs::Tuple) = "($(displayrange(first(rs))), $(displayrange(last(rs))))"

#endregion

############################################################################################
# Hamiltonian
#region

function Base.show(io::IO, h::Union{Hamiltonian,ParametricHamiltonian})
    i = get(io, :indent, "")
    print(io, i, summary(h), "\n",
"$i  Bloch harmonics  : $(length(harmonics(h)))
$i  Harmonic size    : $((n -> "$n × $n")(size(h, 1)))
$i  Orbitals         : $(norbitals(orbitalstructure(h)))
$i  Element type     : $(displaytype(blocktype(h)))
$i  Onsites          : $(nonsites(h))
$i  Hoppings         : $(nhoppings(h))
$i  Coordination     : $(round(coordination(h), digits = 5))")
    showextrainfo(io, i, h)
end

function Base.show(io::IO, h::FlatHamiltonian)
    i = get(io, :indent, "")
    print(io, i, "FlatHamiltonian: orbital-flattening wrapper for\n")
    ioindent = IOContext(io, :indent => "  ")
    show(ioindent, parent(h))
end

Base.summary(h::Hamiltonian{T,E,L}) where {T,E,L} =
    "Hamiltonian{$T,$E,$L}: Hamiltonian on a $(L)D Lattice in $(E)D space"

Base.summary(h::ParametricHamiltonian{T,E,L}) where {T,E,L} =
    "ParametricHamiltonian{$T,$E,$L}: Parametric Hamiltonian on a $(L)D Lattice in $(E)D space"

displaytype(::Type{S}) where {N,T,S<:SMatrix{N,N,T}} = "$N × $N blocks ($T)"
displaytype(::Type{T}) where {T} = "scalar ($T)"

# fallback
showextrainfo(io, i, h) = nothing

showextrainfo(io, i, h::ParametricHamiltonian) = print(io, i, "\n",
"$i  Parameters       : $(parameters(h))")

#endregion

############################################################################################
# Bloch
#region

function Base.show(io::IO, b::Bloch)
    i = get(io, :indent, "")
    ioindent = IOContext(io, :indent => "  ")
    print(io, i, summary(b), " for \n")
    show(ioindent, hamiltonian(b))
end

Base.summary(h::Bloch{L,O}) where {L,O} =
    "Bloch{$L,$O}: Bloch matrix constructor with target eltype $O"

#endregion