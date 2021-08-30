############################################################################################
# Bloch constructor
#region

bloch(h::AbstractHamiltonian, φs::Tuple; kw...) = bloch(h)(φs...; kw...)

function bloch(h::Hamiltonian)
    mats = matrix.(harmonics(h))
    output = merge_structure(mats)  # see tools.jl
    return Bloch(h, output)
end

function bloch(f::FlatHamiltonian)
    os = orbitalstructure(parent(f))
    flatos = orbitalstructure(f)
    mats = flatten.(matrix.(harmonics(f)), Ref(os), Ref(flatos))
    output = merge_structure(mats)  # see tools.jl
    return Bloch(f, output)
end

#eager flattening of Hamiltonian for performance (avoids a flatten on each call later)
# bloch(f::FlatHamiltonian) = bloch(hamiltonian(f))

maybe_flatten_bloch!(output, h::FlatHamiltonian, φs) = maybe_flatten_bloch!(output, parent(h), φs)

# Adds harmonics, assuming output has same structure of merged harmonics
function maybe_flatten_bloch!(output, h::Hamiltonian{<:Any,<:Any,L}, φs::SVector{L}) where {L}
    hars = harmonics(h)
    os = orbitalstructure(h)
    flatos = flatten(os)
    fill!(nonzeros(output), zero(eltype(output)))
    for (n, har) in enumerate(hars)
        e⁻ⁱᵠᵈⁿ = cis(-dot(φs, dcell(har)))
        maybe_flatten_merged_mul!(output, (os, flatos), matrix(har), e⁻ⁱᵠᵈⁿ, 1, 1)  # see tools.jl
    end
    return output
end

#endregion