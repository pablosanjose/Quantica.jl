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
# Lattice reverse - flip all Bravais vectors
#region

Base.reverse(lat::Lattice) = reverse!(copy(lat))

Base.reverse!(lat::Lattice) = (matrix(bravais(lat)) .*= -1; lat)

Base.reverse(h::AbstractHamiltonian) = reverse!(copy(h))

function Base.reverse!(h::AbstractHamiltonian)
    reverse!(lattice(h))
    flip_harmonics!(h)
    return h
end

function flip_harmonics!(h::Hamiltonian)
    hars = harmonics(h)
    for (i, har) in enumerate(hars)
        hars[i] = Harmonic(-dcell(har), matrix(har))
    end
    return h
end

function flip_harmonics!(h::AbstractHamiltonian)
    flip_harmonics!(parent(h))
    flip_harmonics!(hamiltonian(h))
    return h
end

#end

############################################################################################
# Hamiltonian transform/translate
#region

transform(h::AbstractHamiltonian, f::Function) = transform!(copy_lattice(h), f)
transform!(h::AbstractHamiltonian, f::Function) = (transform!(lattice(h), f); h)

translate(h::AbstractHamiltonian, δr) = translate!(copy_lattice(h), δr)
translate!(h::AbstractHamiltonian, δr) = (translate!(lattice(h), δr); h)

#endregion
