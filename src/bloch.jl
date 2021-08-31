############################################################################################
# Bloch constructor
#region

bloch(φs::Tuple; kw...) = h -> bloch(h, φs; kw...)
bloch(h, φs; kw...) = copy(bloch!(h, φs; kw...))
bloch(h::AbstractHamiltonian, φs::Tuple; kw...) = bloch(h)(φs...; kw...)

function bloch(h::Union{Hamiltonian,ParametricHamiltonian})
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

#endregion

############################################################################################
# Bloch call API
#region

(b::Bloch{L})(φs::Vararg{Number,L} ; kw...) where {L} = b(φs; kw...)
(b::Bloch{L})(φs::NTuple{L,Number} ; kw...) where {L} = b(SVector(φs); kw...)
(b::Bloch)(φs::SVector; kw...) = copy(matrix!(b, φs; kw...))

matrix!(b::Bloch, φs; kw...) = maybe_flatten_bloch!(matrix(b), hamiltonian(b), φs; kw...)

maybe_flatten_bloch!(output, h::FlatHamiltonian, φs; kw...) = maybe_flatten_bloch!(output, parent(h), φs; kw...)
maybe_flatten_bloch!(output, h::ParametricHamiltonian, φs; kw...) = maybe_flatten_bloch!(output, h(; kw...), φs)

# Adds harmonics, assuming output has same structure of merged harmonics
function maybe_flatten_bloch!(output, h::Hamiltonian{<:Any,<:Any,L}, φs::SVector{L}) where {L}
    hars = harmonics(h)
    os = orbitalstructure(h)
    flatos = flatten(os)
    fill!(nonzeros(output), zero(eltype(output)))
    for har in hars
        e⁻ⁱᵠᵈⁿ = cis(-dot(φs, dcell(har)))
        maybe_flatten_merged_mul!(output, (os, flatos), matrix(har), e⁻ⁱᵠᵈⁿ, 1, 1)  # see tools.jl
    end
    return output
end

#endregion