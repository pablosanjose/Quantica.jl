############################################################################################
# show tools
#region

display_as_tuple(v, prefix = "") = isempty(v) ? "()" :
    string("(", prefix, join(v, string(", ", prefix)), ifelse(length(v) == 1, ",)", ")"))

display_rounded_vectors(vs) = isempty(vs) ? "[]" : display_rounded_vector.(vs)
display_rounded_vector(v) = round.(v, digits = 6)

pluraltext(m, sing) = ifelse(length(m) == 1, "1 $sing", "$(length(m)) $(sing)s")

displayparameter(::Type{<:Function}) = "Function"
displayparameter(::Type{T}) where {T} = "$T"

displayrange(r::Real) = round(r, digits = 6)
displayrange(::Missing) = "any"
displayrange(nr::Neighbors) = "Neighbors($(Int(nr)))"
displayrange(rs::Tuple) = "($(displayrange(first(rs))), $(displayrange(last(rs))))"

#endregion

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

#endregion

############################################################################################
# LatticeSlice
#region

Base.summary(::LatticeSlice{T,E,L}) where {T,E,L} =
    "LatticeSlice{$T,$E,$L} : collection of subcells for a $(L)D lattice in $(E)D space"

function Base.show(io::IO, ls::LatticeSlice)
    i = get(io, :indent, "")
    print(io, i, summary(ls), "\n",
"$i  Cells       : $(length(subcells(ls)))
$i  Cell range  : $(isempty(ls) ? "empty" : boundingbox(ls))
$i  Total sites : $(nsites(ls))")
end

#endregion

############################################################################################
# Selectors
#region

function Base.show(io::IO, s::Union{SiteSelector,HopSelector})
    i = get(io, :indent, "")
    ioindent = IOContext(io, :indent => i * "  ")
    print(io, i, summary(s), "\n")
    print_selector(io, s)
end

Base.summary(m::SiteSelector) =
    "SiteSelector: a rule that defines a finite collection of sites in a lattice"

Base.summary(m::HopSelector) =
    "HopSelector: a rule that defines a finite collection of hops between sites in a lattice"

function print_selector(io::IO, s::SiteSelector)
    i = get(io, :indent, "")
    print(io,
"$(i)  Region            : $(s.region === missing ? "any" : "Function")
$(i)  Sublattices       : $(s.sublats === missing ? "any" : s.sublats)
$(i)  Cells             : $(s.cells === missing ? "any" : s.cells)")
end

function print_selector(io::IO, s::HopSelector)
    i = get(io, :indent, "")
    print(io,
"$(i)  Region            : $(s.region === missing ? "any" : "Function")
$(i)  Sublattice pairs  : $(s.sublats === missing ? "any" : s.sublats)
$(i)  Cell distances    : $(s.dcells === missing ? "any" : s.dcells)
$(i)  Hopping range     : $(displayrange(s.range))
$(i)  Reverse hops      : $(s.adjoint)")
end

#endregion

############################################################################################
# Models and Modifiers
#region

function Base.show(io::IO, m::TightbindingModel)
    i = get(io, :indent, "")
    ioindent = IOContext(io, :indent => i * "  ")
    print(io, i, summary(m))
    foreach(t -> print(ioindent, "\n", t), terms(m))
end

function Base.show(io::IO, m::ParametricModel)
    i = get(io, :indent, "")
    ioindent = IOContext(io, :indent => i * "  ")
    print(io, i, summary(m))
    foreach(t -> print(ioindent, "\n", t), terms(m))
    foreach(t -> print(ioindent, "\n", t), terms(nonparametric(m)))
end

function Base.show(io::IO, t::Union{AbstractModelTerm,Modifier})
    i = get(io, :indent, "")
    ioindent = IOContext(io, :indent => i * "  ")
    print(io, i, summary(t), "\n")
    print_selector(io, t.selector)
    if !(t isa Modifier)
        print(io, "\n", "$(i)  Coefficient       : $(t.coefficient)")
    end
    if t isa AbstractParametricTerm || t isa Modifier
        print(io, "\n", "$(i)  Parameters        : $(parameters(t))")
    end
end

Base.summary(m::TightbindingModel) = "TightbindingModel: model with $(pluraltext(terms(m), "term"))"
Base.summary(m::ParametricModel) = "ParametricModel: model with $(pluraltext(allterms(m), "term"))"
Base.summary(::OnsiteTerm{F}) where {F} = "OnsiteTerm{$(displayparameter(F))}:"
Base.summary(::HoppingTerm{F}) where {F} = "HoppingTerm{$(displayparameter(F))}:"
Base.summary(::ParametricOnsiteTerm{N}) where {N} = "ParametricOnsiteTerm{ParametricFunction{$N}}"
Base.summary(::ParametricHoppingTerm{N}) where {N} = "ParametricHoppingTerm{ParametricFunction{$N}}"
Base.summary(::OnsiteModifier{N}) where {N} = "OnsiteModifier{ParametricFunction{$N}}:"
Base.summary(::HoppingModifier{N}) where {N} = "HoppingModifier{ParametricFunction{$N}}:"

#endregion

############################################################################################
# Hamiltonian
#region

function Base.show(io::IO, h::Union{Hamiltonian,ParametricHamiltonian,Operator})
    i = get(io, :indent, "")
    print(io, i, summary(h), "\n", showstring(h, i))
    showextrainfo(io, i, h)
end

showstring(h::Union{Hamiltonian,ParametricHamiltonian}, i) =
"$i  Bloch harmonics  : $(length(harmonics(h)))
$i  Harmonic size    : $((n -> "$n × $n")(size(h, 1)))
$i  Orbitals         : $(norbitals(h))
$i  Element type     : $(displaytype(blocktype(h)))
$i  Onsites          : $(nonsites(h))
$i  Hoppings         : $(nhoppings(h))
$i  Coordination     : $(round(coordination(h), digits = 5))"

showstring(o::Operator, i) = showstring(hamiltonian(o), i)

Base.summary(h::Hamiltonian{T,E,L}) where {T,E,L} =
    "Hamiltonian{$T,$E,$L}: Hamiltonian on a $(L)D Lattice in $(E)D space"

Base.summary(h::ParametricHamiltonian{T,E,L}) where {T,E,L} =
    "ParametricHamiltonian{$T,$E,$L}: Parametric Hamiltonian on a $(L)D Lattice in $(E)D space"

Base.summary(h::Operator{H}) where {T,E,L,H<:AbstractHamiltonian{T,E,L}} =
    "Operator{$T,$E,$L}: Operator on a $(L)D Lattice in $(E)D space"

displaytype(::Type{S}) where {N,T,S<:SMatrix{N,N,T}} = "$N × $N blocks ($T)"
displaytype(::Type{S}) where {N,T,S<:SMatrixView{N,N,T}} = "At most $N × $N blocks ($T)"
displaytype(::Type{T}) where {T} = "scalar ($T)"

# fallback
showextrainfo(io, i, h) = nothing

showextrainfo(io, i, h::ParametricHamiltonian) = print(io, i, "\n",
"$i  Parameters       : $(parameters(h))")

showextrainfo(io, i, o::Operator) = showextrainfo(io, i, hamiltonian(o))
#endregion

############################################################################################
# OpenHamiltonian
#region

function Base.show(io::IO, oh::OpenHamiltonian)
    i = get(io, :indent, "")
    print(io, i, summary(oh), "\n",
"$i  Number of contacts : $(length(selfenergies(oh)))
$i  Contact solvers    : $(solvernames(oh))", "\n")
    ioindent = IOContext(io, :indent => i * "  ")
    show(ioindent, hamiltonian(oh))
end

Base.summary(oh::OpenHamiltonian{T,E,L}) where {T,E,L} = "OpenHamiltonian{$T,$E,$L}: Hamiltonian with a set of open contacts"

solvernames(oh::OpenHamiltonian) = nameof.(typeof.(solver.(selfenergies(oh))))

#endregion

############################################################################################
# AbstractEigenSolver
#region

function Base.show(io::IO, s::AbstractEigenSolver)
    i = get(io, :indent, "")
    print(io, i, summary(s))
end

Base.summary(s::AbstractEigenSolver) =
    "AbstractEigenSolver ($(Base.nameof(typeof(s))))"

#endregion

############################################################################################
# AppliedEigenSolver
#region

function Base.show(io::IO, s::AppliedEigenSolver)
    i = get(io, :indent, "")
    ioindent = IOContext(io, :indent => i * "  ")
    print(io, i, summary(s), "\n")
end

Base.summary(::AppliedEigenSolver{T,L}) where {T,L} =
    "AppliedEigenSolver{$T,$L}: eigensolver over an $L-dimensional parameter manifold of type $T"

#endregion

############################################################################################
# Spectrum
#region

function Base.show(io::IO, s::Spectrum)
    i = get(io, :indent, "")
    ioindent = IOContext(io, :indent => i * "  ")
    print(io, i, summary(s))
    println(ioindent, "\nEnergies:")
    show(ioindent, MIME("text/plain"), energies(s))
    println(ioindent, "\nStates:")
    show(ioindent, MIME("text/plain"), states(s))
end

Base.summary(::Spectrum{T,B}) where {T,B} =
    "Spectrum{$T,$B} :"

#endregion

############################################################################################
# AbstractMesh
#region

Base.summary(::Mesh{V}) where {V} =
    "Mesh{$(nameof(V))}: Mesh with vertices of type $(nameof(V))"

Base.summary(::Subband{T,L}) where {T,L} =
    "Subband{$T,$L}: Subband in a $L-dimensional space (like energy-momentum)"

function Base.show(io::IO, m::AbstractMesh)
    i = get(io, :indent, "")
    print(io, i, summary(m), "\n",
"$i  Mesh dim  : $(dim(m))
$i  Space dim : $(embdim(m))
$i  Vertices  : $(length(vertices(m)))
$i  Edges     : $(isempty(neighbors(m)) ? 0 : sum(length, neighbors(m)) ÷ 2)
$i  Simplices : $(length(simplices(m)))")
end

#endregion

############################################################################################
# Bandstructure
#region

function Base.show(io::IO, b::Bandstructure)
    i = get(io, :indent, "")
    print(io, i, summary(b), "\n",
"$i  Subbands  : $(nsubbands(b))
$i  Vertices  : $(nvertices(b))
$i  Edges     : $(nedges(b))
$i  Simplices : $(nsimplices(b))")
end

Base.summary(::Bandstructure{T,E,L}) where {T,E,L} =
    "Bandstructure{$T,$E,$L}: $(E)D Bandstructure over a $L-dimensional parameter space of type $T"

#endregion

############################################################################################
# GreenFunction, GreenSolution and GreenSlice
#region

function Base.show(io::IO, g::GreenFunction)
    i = get(io, :indent, "")
    Σs = selfenergies(contacts(g))
    print(io, i, summary(g), "\n",
"$i  Solver          : $(typename(solver(g)))
$i  Contacts        : $(length(Σs))
$i  Contact solvers : $(display_as_tuple(typename.(solver.(Σs))))
$i  Contact sizes   : $(display_as_tuple(nsites.(latslice.(Σs))))", "\n")
    ioindent = IOContext(io, :indent => i * "  ")
    show(ioindent, parent(g))
end

Base.summary(g::GreenFunction{T,E,L}) where {T,E,L} =
    "GreenFunction{$T,$E,$L}: Green function of a $(typename(hamiltonian(g))){$T,$E,$L}"

function Base.show(io::IO, g::GreenSolution)
    i = get(io, :indent, "")
    print(io, i, summary(g))
end

Base.summary(g::GreenSolution{T,E,L,S}) where {T,E,L,S} =
    "GreenSolution{$T,$E,$L}: Green function at arbitrary positions, but at a fixed energy"

function Base.show(io::IO, g::GreenSlice)
    i = get(io, :indent, "")
    print(io, i, summary(g))
end

Base.summary(g::GreenSlice{T,E,L}) where {T,E,L} =
    "GreenSlice{$T,$E,$L}: Green function at arbitrary energy, but at a fixed lattice positions"

#endregion

############################################################################################
# Conductance
#region

function Base.show(io::IO, G::Conductance)
    i = get(io, :indent, "")
    print(io, i, summary(G), "\n",
"$i  Current contact  : $(currentcontact(G))
$i  Bias contact     : $(biascontact(G))")
end

Base.summary(::Conductance{T}) where {T} =
    "Conductance{$T}: Zero-temperature conductance dIᵢ/dVⱼ from contacts i,j, in units of e^2/h"

#endregion

############################################################################################
# Transmission
#region

function Base.show(io::IO, T::Transmission)
    i = get(io, :indent, "")
    print(io, i, summary(T), "\n",
"$i  From contact  : $(biascontact(parent(T)))
$i  To contact    : $(currentcontact(parent(T)))")
end

Base.summary(::Transmission) =
    "Transmission: total transmission between two different contacts"

#endregion

############################################################################################
# Integrator
#region

function Base.show(io::IO, I::Integrator)
    i = get(io, :indent, "")
    print(io, i, summary(I), "\n",
"$i  Integration path    : $(points(I))
$i  Integration options : $(display_namedtuple(options(I)))
$i  Integrand:          :\n")
    ioindent = IOContext(io, :indent => i * "  ")
    show(ioindent, integrand(I))
end

Base.summary(::Integrator) = "Integrator: Complex-plane integrator"


display_namedtuple(nt::NamedTuple) = isempty(nt) ? "()" : "$nt"

#endregion

############################################################################################
# Josephson
#region

function Base.show(io::IO, J::JosephsonDensity)
    i = get(io, :indent, "")
    print(io, i, summary(J), "\n",
"$i  kBT                     : $(temperature(J))
$i  Contact                 : $(contact(J))
$i  Number of phase shifts  : $(numphaseshifts(J))")
end

Base.summary(::JosephsonDensity{T}) where {T} =
    "JosephsonDensity{$T} : Equilibrium (dc) Josephson current observable before integration over energy"

#endregion

############################################################################################
# ldos
#region

function Base.show(io::IO, D::Union{LocalSpectralDensitySolution, LocalSpectralDensitySlice})
    i = get(io, :indent, "")
    print(io, i, summary(D), "\n",
"$i  kernel   : $(kernel(D))")
end

Base.summary(::LocalSpectralDensitySolution{T}) where {T} =
    "LocalSpectralDensitySolution{$T} : local density of states at fixed energy and arbitrary location"

Base.summary(::LocalSpectralDensitySlice{T}) where {T} =
    "LocalSpectralDensitySlice{$T} : local density of states at a fixed location and arbitrary energy"

#endregion

############################################################################################
# current
#region

function Base.show(io::IO, J::Union{CurrentDensitySolution, CurrentDensitySlice})
    i = get(io, :indent, "")
    print(io, i, summary(J), "\n",
"$i  charge      : $(charge(J))
$i  direction   : $(direction(J))")
end

Base.summary(::CurrentDensitySolution{T}) where {T} =
    "CurrentDensitySolution{$T} : current density at a fixed energy and arbitrary location"

Base.summary(::CurrentDensitySlice{T}) where {T} =
    "CurrentDensitySlice{$T} : current density at a fixed location and arbitrary energy"

#endregion
