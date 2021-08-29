############################################################################################
# Bloch constructor
#region

bloch(h::AbstractHamiltonian, φs::Tuple; kw...) = bloch(h)(φs...; kw...)

function bloch(h::Hamiltonian)
    output = merge_structure(matrix.(harmonics(h)))  # see tools.jl
    return Bloch(h, output)
end

#eager flattening of Hamiltonian for performance (avoids a flatten on each call later)
bloch(f::FlatHamiltonian) = bloch(hamiltonian(f))

# Adds harmonics, assuming output has same structure of merged harmonics
function maybe_flatten_bloch!(output, h::Hamiltonian{<:Any,<:Any,L}, φs::SVector{L}) where {L}
    hars = harmonics(h)
    os = orbitalstructure(h)
    for (n, har) in enumerate(hars)
        e⁻ⁱᵠᵈⁿ = cis(-dot(φs, dcell(har)))
        add = ifelse(n == 1, false, true)
        maybe_flatten_merged_mul!(output, os, matrix(har), e⁻ⁱᵠᵈⁿ, add)  # see tools.jl
    end
    return output
end

#endregion