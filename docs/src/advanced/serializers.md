
# Serializers

Serializers are useful to translate between a complex data structure and a stream of plain numbers of a given type. Serialization and deserialization is a common encode/decode operation in programming language.

In Quantica, a `s::Serializer{T}` is an object that takes an `h::AbstractHamiltonian`, a selection of the sites and hoppings to be translated, and an `encoder`/`decoder` pair of functions to translate each element into a portion of the stream. This `s` can then be used to convert the specified elements of `h` into a vector of scalars of type `T` and back, possibly after applying some parameter values. Consider this example from the `serializer` docstring
```julia
julia> h1 = LP.linear() |> hopping((r, dr) -> im*dr[1]) - @onsite((r; U = 2) -> U);

julia> as = serializer(Float64, h1; encoder = s -> reim(s), decoder = v -> complex(v[1], v[2]))
AppliedSerializer : translator between a selection of of matrix elements of an AbstractHamiltonian and a collection of scalars
  Object            : ParametricHamiltonian
  Object parameters : [:U]
  Stream parameter  : :stream
  Output eltype     : Float64
  Encoder/Decoder   : Single
  Length            : 6

julia> v = serialize(as; U = 4)
6-element Vector{Float64}:
 -4.0
  0.0
 -0.0
 -1.0
  0.0
  1.0

julia> h2 = deserialize!(as, v);

julia> h2 == h1(U = 4)
true

julia> h3 = hamiltonian(as)
ParametricHamiltonian{Float64,1,1}: Parametric Hamiltonian on a 1D Lattice in 1D space
  Bloch harmonics  : 3
  Harmonic size    : 1 × 1
  Orbitals         : [1]
  Element type     : scalar (ComplexF64)
  Onsites          : 1
  Hoppings         : 2
  Coordination     : 2.0
  Parameters       : [:U, :stream]

julia> h3(stream = v, U = 5) == h1(U = 4)  # stream overwrites the U=5 onsite terms
true
```
The serializer functionality is designed with efficiency in mind. Using the in-place `serialize!`/`deserialize!` pair we can do the encode/decode round trip without allocations
```
julia> using BenchmarkTools

julia> v = Vector{Float64}(undef, length(as));

julia> deserialize!(as, serialize!(v, as)) === Quantica.call!(h1, U = 4)
true

julia> @btime deserialize!($as, serialize!($v, $as));
  149.737 ns (0 allocations: 0 bytes)
```
It also allows powerful compression into relevant degrees of freedom through appropriate use of encoders/decoders, see the `serializer` docstring.

## Serializers of OrbitalSliceArrays

Serialization of `OrbitalSliceArray`s is simpler than for `AbstractHamiltonians`, as there is no need for an intermediate `Serializer` object. To serialize an `m::OrbitalSliceArray` simply do `v = serialize(m)`, or `v = serialize(T::Type, m)` if you want a specific eltype for `v`. To deserialize, just do `m´ = deserialize(m, v)`.
