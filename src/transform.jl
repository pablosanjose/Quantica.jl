############################################################################################
# Currying and non-mutating methods
#region

transform(f...; kw...) = x -> transform(x, f...; kw...)

transform!(f...; kw...) = x -> transform!(x, f...; kw...)

transform(x, f...) = transform!(copy(x), f...)

translate(δr) = x -> translate(x, δr)

translate!(δr) = x -> translate!(x, δr)

translate(x, δr) = translate!(copy(x), δr)

#endregion

############################################################################################
# Lattice transform!
#region

function transform!(l::Lattice, f::Function, keepnranges = false)
    return keepnranges ?
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

#endregion

############################################################################################
# Lattice translate
#region

translate!(lat::Lattice{T,E}, δr) where {T,E} = translate!(lat, sanitize_SVector(SVector{E,T}, δr))

# translate! does not change neighbor ranges, keep whichever have already been computed
translate!(lat::Lattice{T,E}, δr::SVector{E,T}) where {T,E} = transform!(l, r -> r + δr, true)

#endregion

############################################################################################
# Hamiltonian transform
#region

# specialized
transform(h::Hamiltonian, modifiers...; kw...) = transform!(copy_harmonics(h), modifiers...; kw...)

transform!(h::Hamiltonian, modifiers::Modifier...; kw...) = applymodifiers!(h, modifiers...; kw...)

#endregion