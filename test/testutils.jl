# Helpers shared by all test files. Loaded into each test sandbox by runtests.jl

"""
    cycled(xs, i)

Element `i` of `xs`, wrapping around cyclically, i.e. `xs[mod1(i, length(xs))]`.
"""
cycled(xs, i) = xs[mod1(i, length(xs))]

"""
    cyclic_product(full::Tuple, cycling::Tuple)

Return the combinations `(f..., c...)` of all elements of the Cartesian product of the
collections in `full`, each extended with one element `c` from each collection in `cycling`.
The cycling elements are not multiplied in: they are chosen Latin-square style, as
`cycled(c, i₁ + i₂ + ...)` where `iₖ` are the indices into `full`. Hence, every element
of a cycling collection appears together with every element of each `full` collection (provided
the other `full` collections are long enough). So all pairs among `full` collections and all
(full, cycling) pairs are covered, at a fraction of the cost of a full product. Pairs of two
cycling collections are not guaranteed to be fully covered.

# Example
```julia
julia> cyclic_product(((1, 2), (:a, :b, :c)), (("x", "y", "z"),))
6-element Vector{Tuple{Int64, Symbol, String}}:
 (1, :a, "y")
 (2, :a, "z")
 (1, :b, "z")
 (2, :b, "x")
 (1, :c, "x")
 (2, :c, "y")
```
All 6 combinations of `(1, 2)` and `(:a, :b, :c)` appear, instead of the 18 of a full product
with `("x", "y", "z")`. Both `1` and `2` meet all of `"x"`, `"y"`, `"z"`, but each of `:a, :b, :c`
meets only two of them, because `(1, 2)` is too short to cycle through all three.
"""
cyclic_product(full::Tuple, cycling::Tuple) =
    vec([(getindex.(full, Tuple(I))..., (cycled(c, sum(Tuple(I))) for c in cycling)...)
        for I in CartesianIndices(map(eachindex, full))])
