############################################################################################
# Currying
#region

transform(f::Function) = x -> transform(x, f)

transform!(f::Function) = x -> transform!(x, f)

translate(δr) = x -> translate(x, δr)

translate!(δr) = x -> translate!(x, δr)

#endregion

############################################################################################
# Lattice transform/translate
#region

function transform!(l::Lattice, f::Function, keepranges = false)
    return keepranges ?
        Lattice(transform!(bravais(l), f), transform!(unitcell(l), f), nranges(l)) :
        Lattice(transform!(bravais(l), f), transform!(unitcell(l), f))
end

function transform!(b::Bravais{<:Any,E}, f::Function) where {E}
    m = matrix(b)
    for j in axes(m, 2)
        v = SVector(ntuple(i -> m[i, j], Val(E)))
        m[:, j] .= f(v) - f(zero(v))
    end
    return b
end

transform!(u::Unitcell, f::Function) = (map!(f, sites(u), sites(u)); u)
transform(l::Lattice, f::Function) = transform!(copy(l), f)

# translate! does not change neighbor ranges, keep whichever have already been computed
translate!(lat::Lattice{T,E}, δr::SVector{E,T}) where {T,E} = transform!(lat, r -> r + δr, true)
translate!(lat::Lattice{T,E}, δr) where {T,E} = translate!(lat, sanitize_SVector(SVector{E,T}, δr))
translate(l::Lattice, δr) = translate!(copy(l), δr)

#endregion

############################################################################################
# Hamiltonian transform/translate
#region

transform(h::AbstractHamiltonian, f::Function) = transform!(copy_lattice(h), f)
transform!(h::AbstractHamiltonian, f::Function) = (transform!(lattice(h), f); h)

translate(h::AbstractHamiltonian, δr) = translate!(copy_lattice(h), δr)
translate!(h::AbstractHamiltonian, δr) = (translate!(lattice(h), δr); h)

#endregion

############################################################################################
# combine
#   type-stable with Hamiltonians, but not with ParametricHamiltonians, as the field
#   builder.modifiers isa Vector{Any} in that case.
#region

function combine(hams::AbstractHamiltonian...; coupling::AbstractModel=TightbindingModel())
    check_unique_names(coupling, hams...)
    lat = combine(lattice.(hams)...)
    builder = IJVBuilder(lat, hams...)
    interblockmodel = interblock(coupling, hams...)
    builder´ = maybe_add_modifiers(builder, coupling)
    model´, block´ = parent(interblockmodel), block(interblockmodel)
    add!(builder´, model´, block´)
    return hamiltonian(builder´)
end

# No need to have unique names if nothing is parametric
check_unique_names(::TightbindingModel, ::Hamiltonian...) = nothing

function check_unique_names(::AbstractModel, hs::AbstractHamiltonian...)
    names = tupleflatten(sublatnames.(lattice.(hs))...)
    allunique(names) || argerror("Cannot combine ParametricHamiltonians with non-unique sublattice names, since modifiers could be tied to the original names. Assign unique names on construction.")
    return nothing
end

function check_unique_names(::AbstractModel, hs::Hamiltonian...)
    names = tupleflatten(sublatnames.(lattice.(hs))...)
    allunique(names) || argerror("Cannot combine Hamiltonians with non-unique sublattice names using a ParametricModel, since modifiers could be tied to the original names. Assign unique names on construction.")
    return nothing
end

maybe_add_modifiers(b, ::ParametricModel) = IJVBuilderWithModifiers(b)
maybe_add_modifiers(b, ::TightbindingModel) = b

#endregion

############################################################################################
# stitch(::AbstractHamiltonian, phases::Tuple)
# stitch(::AbstractHamiltonian, wrapaxes::SVector)
#region

struct StitchModifier{H<:ParametricHamiltonian,W<:Tuple,D<:Tuple} <: AppliedModifier
    ph::H
    wrapped_phases::W
    groups_dcells_uw::D
end

StitchModifier(ph, (wp, wa, ua)) =
    StitchModifier(ph, wp, stitch_groups(harmonics(parent(ph)), wa, ua))

# groups of harmonic indices with same unwrapped (i.e. non-wrapped) dcell
# dcells_u = dcell for each harmonic along unwrapped axes
# dcells_w = dcell for each harmonic along wrapped axes
function stitch_groups(hars, wa::NTuple{W}, ua::NTuple{U}) where {W,U}
    dcells_u = SVector{U,Int}[dcell(har)[SVector(ua)] for har in hars]
    dcells_w = SVector{W,Int}[dcell(har)[SVector(wa)] for har in hars]
    unique_dcells_u = unique!(sort(dcells_u, by=norm))
    groups = [findall(==(dcell), dcells_u) for dcell in unique_dcells_u]
    return groups, dcells_u, dcells_w
end

stitch_groups(m::StitchModifier) = m.groups_dcells_uw

#region ## stitch(::AbstractHamiltonian, ...)

stitch(phases) = h -> stitch(h, phases)

function stitch(h::AbstractHamiltonian, phases)
    wp, wa, ua = split_axes(h, phases)
    return _stitch(h, wp, wa, ua)
end

# wa, ua = tuples of indices of wrapped/unwrapped axes
# wp = phases along wrapped axes
function _stitch(h::Hamiltonian, wp, wa, ua)
    isempty(wa) && return minimal_callsafe_copy(h)
    lat = lattice(h)
    b´ = bravais_matrix(lat)[:, SVector(ua)]
    lat´ = lattice(lat; bravais=b´)
    bs´ = blockstructure(h)
    bloch´ = copy_matrices(bloch(h))
    hars´ = stitch_harmonics(h, wp, wa, ua)
    return Hamiltonian(lat´, bs´, hars´, bloch´)
end

function _stitch(ph::ParametricHamiltonian, wp, wa, ua)
    isempty(wa) && return minimal_callsafe_copy(ph)
    h = parent(ph)
    h´ = _stitch(h, missing, wa, ua)  # this returns a zero Hamiltonian
    ph´ = parametric(h´, StitchModifier(ph, (wp, wa, ua)))
    return ph´
end

# indices for wrapped and unwrapped axes, and wrapped phases
function split_axes(::AbstractHamiltonian{<:Any,<:Any,L}, phases::Tuple) where {L}
    length(phases) == L || argerror("Expected $L `stitch` phases, got $(length(phases))")
    return _split_axes((), (), (), 1, phases...)
end

_split_axes(wp, wa, ua, n, ::Colon, xs...) = _split_axes(wp, wa, (ua..., n), n + 1, xs...)
_split_axes(wp, wa, ua, n, x, xs...) = _split_axes((wp..., x), (wa..., n), ua, n + 1, xs...)
_split_axes(wp, wa, ua, n) = wp, wa, ua

function split_axes(::AbstractHamiltonian{<:Any,<:Any,L}, wrapaxes::SVector) where {L}
    allunique(wrapaxes) && issorted(wrapaxes) && all(i -> 1<=i<=L, wrapaxes) ||
        argerror("Wrap axes should be a sorted SVector of unique axis indices between 1 and $L")
    wa = Tuple(wrapaxes)
    wp = Tuple(zero(wrapaxes))
    ua = inds_complement(Val(L), wa)
    return wp, wa, ua
end

function stitch_harmonics(h, phases_w, wa, ua)
    groups, dcells_u, dcells_w = stitch_groups(harmonics(h), wa, ua)
    hars´ = [sum_harmonics_group(h, inds, phases_w, dcells_u, dcells_w) for inds in groups]
    return hars´
end

# similar to merge_sparse in tools.jl, but we sum everything with bloch phases
# function sum_harmonics_group(hars::Vector{<:Harmonic{<:Any,<:Any,B}}, inds, phases_w, dcells_u, dcells_w) where {B}
function sum_harmonics_group(h::AbstractHamiltonian{<:Any,<:Any,<:Any,B}, inds, phases_w, dcells_u, dcells_w) where {B}
    hars = harmonics(h)
    I, J, V = Int[], Int[], B[]
    for i in inds
        I´, J´, V´ = findnz(unflat(matrix(hars[i])))
        dn_w = dcells_w[i]
        apply_bloch_phases!(V´, phases_w, dn_w)
        append!(I, I´)
        append!(J, J´)
        append!(V, V´)
    end
    dn_u = dcells_u[first(inds)]
    bs = blockstructure(matrix(hars[first(inds)]))
    n = unflatsize(bs)
    mat = sparse(I, J, V, n, n)
    return Harmonic(dn_u, HybridSparseMatrix(bs, mat))
end

function apply_bloch_phases!(vals, phases_w, dn_w)
    e⁻ⁱᵠᵈⁿ = blochfactor(dn_w, phases_w)
    vals .*= e⁻ⁱᵠᵈⁿ
    return vals
end

# If phases are missing, we just store structural zeros (for the parametric case)
apply_bloch_phases!(vals::AbstractArray{T}, ::Missing, _) where {T} = fill!(vals, zero(T))

Base.parent(m::StitchModifier) = m.ph

# copy(StitchModifer) must dealias, since m.groups_dcells_uw can be mutated, e.g. by reverse_bravais!
Base.copy(m::StitchModifier) = StitchModifier(m.ph, m.wrapped_phases, copy.(m.groups_dcells_uw))

#endregion

#region ## applymodifier! API

function applymodifiers!(ph, m::StitchModifier; kw...)
    h_parent = call!(m.ph; kw...)
    hars_parent = harmonics(h_parent)
    h = hamiltonian(ph)
    wp = m.wrapped_phases
    groups, dcells_u, dcells_w = stitch_groups(m)
    for inds in groups
        sum_harmonics_group!(h, hars_parent, inds, wp, dcells_u, dcells_w)
    end
    return ph
end

# inds are hars_parent indices
function sum_harmonics_group!(h, hars_parent, inds, phases_w, dcells_u, dcells_w)
    dn_u = dcells_u[first(inds)]
    mat = h[dn_u]                               # flat sparse matrix
    for i in inds
        dn_w = dcells_w[i]
        e⁻ⁱᵠᵈⁿ = blochfactor(phases_w, dn_w)
        mat_parent = flat(hars_parent[i])       # flat sparse matrix
        # by construction, all structural elements in mat_parent are in mat too. See tools.jl
        merged_flat_mul!(mat, mat_parent, e⁻ⁱᵠᵈⁿ, 1, 1)
    end
    return h
end

parameter_names(m::StitchModifier) = parameter_names(m.ph)

# all parent harmonics that get summed into each stitched harmonic adds its pointers.
function _merge_pointers!(p, m::StitchModifier)
    hars = harmonics(hamiltonian(m.ph))
    groups = first(m.groups_dcells_uw)
    for (pn, group) in zip(p, groups)
        for pn´ in group
            append!(pn, eachindex(nonzeros(unflat(hars[pn´]))))
        end
        unique!(sort!(pn))
    end
    return p
end

#endregion
#endregion

############################################################################################
# @stitch(h, phases, ϕname)
#    Note that blochfactor returns 1 if the phase is Missing
#region

macro stitch(h, phases_or_axes, name)
    quote
        wp, wa, ua = split_axes($(esc(h)), $(esc(phases_or_axes)))
        was = SVector(wa)
        mod = @hopping!((t, i, j; $name = $missing) -->
            t * blochfactor((cell(i) - cell(j))[was], $(esc(name))); dcells = !iszero)
        h´ = $(esc(h)) |> mod
        isempty(wa) ? h´ : _stitch(h´, wp, wa, ua)
    end
end

#endregion


############################################################################################
# reverse - flip all Bravais vectors of a lattice, and all dn in hamiltonian harmonics
#   As a general rule, reverse does not change the Hamiltonian, only the meaning of the
#   Bloch phase ϕ -> -ϕ, so that H(k) -> H(k), but H(ϕ) -> H(-ϕ)
#   reverse_bravais!(ph::ParametricHamiltonian) is dangerous - it flips the harmonics of parent(ph)!
#   We don't export it or document it to avoid user surprises.
#region

Base.reverse(lat::Lattice) = reverse_bravais!(copy(lat))

# cannot copy only lattice, since we also transform harmonics to match
Base.reverse(h::AbstractHamiltonian) = reverse_bravais!(copy(h))

# unexported
reverse_bravais!(lat::Lattice) = (matrix(bravais(lat)) .*= -1; lat)

function reverse_bravais!(h::Hamiltonian)
    reverse_bravais!(lattice(h))
    flip_dcells!(h)
    return h
end

function reverse_bravais!(ph::ParametricHamiltonian)
    reverse_bravais!(lattice(parent(ph)))
    flip_dcells!(parent(ph))
    flip_dcells!(hamiltonian(ph))
    flip_dcells!.(modifiers(ph))
    return ph
end

function flip_dcells!(h::Hamiltonian)
    hars = harmonics(h)
    for (i, har) in enumerate(hars)
        hars[i] = Harmonic(-dcell(har), matrix(har))
    end
    return h
end

# by default, modifiers do not care about reverse
flip_dcells!(m::AbstractModifier) = m

# StitchModifier is special, in that it contains a reference to the dn of stitched
# harmonics that are a sum over subsets of parent harmonics. If the dcell of the former are
# flipped, we must flip the dcell reference to them as well.
function flip_dcells!(m::StitchModifier)
    _, dcells_u, _ = stitch_groups(m)
    dcells_u .*= -1
    return m
end

# AppliedHoppingModifiers contain CellSite's that contain nonzero dcell that must be flipped
function flip_dcells!(m::AppliedHoppingModifier)
    ptrs = pointers(m)
    for pcell in ptrs, (i, p) in enumerate(pcell)
        (ptr, r, dr, si, sj, norbs) = p
        pcell[i] = (ptr, r, dr, reverse(si), reverse(sj), norbs)
    end
    return m
end


#endregion
